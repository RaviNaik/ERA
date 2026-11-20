import shutil

import pytest
import torch

from dtphi import official
from dtphi.modeling import load_model, make_device_map
from dtphi.train import GATE_TARGET


def test_vendored_files_are_the_upstream_files():
    assert official.verify_vendored() >= 30


def test_a_changed_vendored_file_is_detected(tmp_path, monkeypatch):
    shutil.copytree(official.OFFICIAL_DIR, tmp_path / "official")
    (tmp_path / "official" / "src" / "wanda_calibration.py").write_text("# edited\n")
    monkeypatch.setattr(official, "OFFICIAL_DIR", tmp_path / "official")
    with pytest.raises(RuntimeError, match="differs from the upstream"):
        official.verify_vendored()


def test_device_map_blocks():
    b = make_device_map(32, 2)
    assert b["model.embed_tokens"] == 0 and b["model.layers.15"] == 0 and b["model.layers.16"] == 1
    assert b["model.norm"] == 1 and b["lm_head"] == 1
    assert set(make_device_map(5, 1).values()) == {0}
    with pytest.raises(ValueError):
        make_device_map(4, 0)


def tiny(tiny_dir):
    return load_model(str(tiny_dir), 0, dtype=torch.float32)


def test_official_convert_hits_exactly_the_expert_gate_projections(tiny_dir):
    from learnable_sparsity.sparse_linear import SparseLinear, convert_to_sparse_model
    m = tiny(tiny_dir)
    convert_to_sparse_model(m, [GATE_TARGET], [100, 1000], 10)
    names = [n for n, x in m.named_modules() if isinstance(x, SparseLinear)]
    assert len(names) == 2 * 4  # 2 layers x 4 experts
    assert all(n.endswith(".block_sparse_moe.experts." + n.split(".experts.")[1]) and n.endswith(".w1") for n in names)
    for n, mod in m.named_modules():
        if n.endswith((".w2", ".w3")):
            assert isinstance(mod, torch.nn.Linear)


def test_official_hard_mask_equals_official_apply_mask(tiny_dir, data_dir):
    import apply_mask
    from learnable_sparsity.sparse_linear import convert_to_sparse_model, extract_all_masks
    m = tiny(tiny_dir)
    convert_to_sparse_model(m, [GATE_TARGET], [100, 1000], 10)
    for p in m.parameters():
        p.requires_grad = False
    for mod in m.modules():
        if hasattr(mod, "diff_mask"):
            mod.diff_mask.gate.data.normal_(0, 0.01)
    masks = extract_all_masks(m)  # leaves the model in eval mode
    ids = torch.tensor([[5, 6, 7, 8, 9, 10]])
    with torch.no_grad():
        a = m(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
    plain = tiny(tiny_dir)
    deleted = apply_mask.apply_mask(model=plain, masks=masks)
    assert deleted > 0
    with torch.no_grad():
        b = plain(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
    assert torch.equal(a, b)


def test_device_safe_regulariser_sum_equals_the_official_one_on_one_device(tiny_dir, data_dir):
    """train.py replaces official sum_sparse_weight_norms by a version that accumulates on the loss device; on a single
    device both must give the same number."""
    import train_sft
    from dtphi import data
    from dtphi.modeling import load_tokenizer
    from learnable_sparsity.sparse_linear import SparseLinear, convert_to_sparse_model
    m = tiny(tiny_dir)
    convert_to_sparse_model(m, [GATE_TARGET], [100, 1000], 10)
    for p in m.parameters():
        p.requires_grad = False
    for mod in m.modules():
        if isinstance(mod, SparseLinear):
            mod.diff_mask.gate.requires_grad_(True)
    m.train()
    tok = load_tokenizer(str(tiny_dir))
    ids = torch.tensor([data.plain_prompt_ids(tok, "What is 2 plus 2?")])
    m(input_ids=ids, attention_mask=torch.ones_like(ids))
    official_sum = train_sft.sum_sparse_weight_norms(m)
    mine = torch.zeros(())
    for module in m.modules():
        if isinstance(module, SparseLinear) and hasattr(module, "sparse_weight_norm"):
            mine = mine + module.sparse_weight_norm.float().to(mine.device)
    assert official_sum.item() > 0 and torch.equal(official_sum, mine)


def test_real_checkpoint_has_512_expert_gate_projections_under_the_target_name():
    import json
    from conftest import MODEL_DIR
    idx = MODEL_DIR / "model.safetensors.index.json"
    if not idx.exists():
        pytest.skip("real model folder not found")
    keys = json.loads(idx.read_text())["weight_map"]
    w1 = [k for k in keys if GATE_TARGET in k and k.endswith(".weight")]
    assert len(w1) == 32 * 16 and all(".block_sparse_moe.experts." in k for k in w1)
    assert not [k for k in keys if GATE_TARGET in k and ".block_sparse_moe.experts." not in k]


def test_deterministic_moe_switch_removes_training_noise(tiny_dir):
    """With jitter noise on, model.train() routes randomly unless DETERMINISTIC_MOE is set; then it equals eval mode."""
    from dtphi.phimoe import modeling_phimoe as mp
    m = tiny(tiny_dir)
    for mod in m.modules():
        if hasattr(mod, "input_jitter_noise"):
            mod.input_jitter_noise = 0.05
            mod.router_jitter_noise = 0.05
    ids = torch.randint(0, 1000, (4, 64))
    try:
        mp.DETERMINISTIC_MOE = False
        m.eval()
        ref = m(input_ids=ids).logits
        m.train()
        assert any(not torch.equal(m(input_ids=ids).logits, ref) for _ in range(3))  # original behaviour is noisy
        mp.DETERMINISTIC_MOE = True
        for _ in range(3):
            assert torch.equal(m(input_ids=ids).logits, ref)
    finally:
        mp.DETERMINISTIC_MOE = False
