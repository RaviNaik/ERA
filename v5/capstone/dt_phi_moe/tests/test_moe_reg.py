import pytest
import torch

from test_official import GATE_TARGET, tiny


def _setup(tiny_dir):
    import train_sft
    from dtphi.phimoe import modeling_phimoe as mp
    from learnable_sparsity.sparse_linear import SparseLinear, convert_to_sparse_model
    m = tiny(tiny_dir)
    convert_to_sparse_model(m, [GATE_TARGET], [100, 1000], 50)
    for p in m.parameters():
        p.requires_grad = False
    for mod in m.modules():
        if isinstance(mod, SparseLinear):
            mod.diff_mask.gate.requires_grad_(True)
    m.train()
    mp.DETERMINISTIC_MOE = True
    return train_sft, m


def _steps(train_sft, m, tokens):
    from learnable_sparsity.sparse_linear import step_all_sparse_modules
    for a, b in tokens:  # two tokens per step, so some of the 4 experts of a layer get no token
        ids = torch.tensor([[a, b]])
        loss = m(input_ids=ids, labels=ids).loss - 1e-5 * train_sft.sum_sparse_weight_norms(model=m)
        loss.backward()
        for p in m.parameters():
            p.grad = None
        step_all_sparse_modules(model=m)


TOKENS = [(5, 600), (17, 900), (3, 44), (250, 777), (31, 8), (91, 402)]


def test_official_sum_fails_when_an_expert_gets_no_tokens(tiny_dir):
    from dtphi.phimoe import modeling_phimoe as mp
    train_sft, m = _setup(tiny_dir)
    try:
        with pytest.raises(RuntimeError, match="second time"):
            _steps(train_sft, m, TOKENS)
    finally:
        mp.DETERMINISTIC_MOE = False


def test_moe_reg_keeps_every_layer_in_the_regulariser_and_survives_idle_experts(tiny_dir):
    from dtphi import moe_reg
    from dtphi.phimoe import modeling_phimoe as mp
    from learnable_sparsity.sparse_linear import SparseLinear
    train_sft, m = _setup(tiny_dir)
    restore, stats = moe_reg.install(m, train_sft, torch.device("cpu"))
    try:
        _steps(train_sft, m, TOKENS)
        n_layers = sum(isinstance(x, SparseLinear) for x in m.modules())
        assert stats["sums"] == len(TOKENS) and stats["layers"] == n_layers == 8
        assert stats["idle_terms"] > 0  # the idle-expert case did occur
        # the total equals the plain sum over ALL layers of ||W * mask||^2 with the current masks
        with torch.enable_grad():  # train mode: the mask is the sigmoid one, as in the forward
            expect = sum(((x.weight.detach() * x.diff_mask(target_shape=x.weight.shape, dtype=x.weight.dtype)).pow(2).sum()
                          for x in m.modules() if isinstance(x, SparseLinear)))
        ids = torch.tensor([[1, 2]])
        m(input_ids=ids)
        got = train_sft.sum_sparse_weight_norms(model=m)
        assert torch.allclose(got, expect.float(), rtol=1e-5)
    finally:
        restore()
        mp.DETERMINISTIC_MOE = False
    assert train_sft.sum_sparse_weight_norms.__name__ == "sum_sparse_weight_norms"
