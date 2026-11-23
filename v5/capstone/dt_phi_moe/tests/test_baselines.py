import json

import pytest
import torch
import torch.nn as nn

from conftest import TEAM10_DIR
from dtphi import evaluate, report, train_baseline as tb


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.ModuleDict({"w1": nn.Linear(6, 5, bias=False),
                                                    "w2": nn.Linear(5, 6, bias=False)}) for _ in range(2)])

    def forward(self, x):
        for l in self.layers:
            x = l["w2"](torch.relu(l["w1"](x)))
        return x


def test_lora_starts_at_the_base_model_and_merging_equals_the_adapter_path(tmp_path):
    torch.manual_seed(0)
    m = Tiny()
    x = torch.randn(3, 6)
    ref = m(x)
    names = tb.add_lora(m, ["w1"], r=2, alpha=4)
    assert names == ["layers.0.w1", "layers.1.w1"]
    assert torch.equal(m(x), ref)                     # B = 0: no change at the start
    trainable = [n for n, p in m.named_parameters() if n.endswith((".lora_A", ".lora_B"))]
    assert len(trainable) == 4
    for n in names:                                   # pretend training happened
        m.get_submodule(n).lora_B.data.normal_(0, 0.5)
    tuned = m(x)
    assert not torch.allclose(tuned, ref)
    tb.save_lora(m, tmp_path, names, 2, 4, ["w1"])
    torch.manual_seed(0)
    fresh = Tiny()                                    # the same weights as the model above before the adapters
    info = tb.apply_lora(fresh, tmp_path)
    assert info == {"lora_modules_merged": 2, "r": 2, "alpha": 4}
    assert torch.allclose(fresh(x), tuned, atol=1e-5)


def test_lora_target_that_matches_nothing_is_an_error():
    with pytest.raises(ValueError, match="no linear layer"):
        tb.add_lora(Tiny(), ["q_proj"], 2, 4)


def test_stochastic_rounding_keeps_the_mean_and_only_uses_the_two_neighbours():
    torch.manual_seed(0)
    x = torch.full((200000,), 1.0 + 2 ** -10)         # not a bf16 number: neighbours 1.0 and 1 + 2**-7
    r = tb.round_stochastic(x, torch.bfloat16)
    assert r.dtype == torch.bfloat16
    assert set(r.float().unique().tolist()) == {1.0, 1.0 + 2 ** -7}
    assert r.float().mean().item() == pytest.approx(1.0 + 2 ** -10, abs=3e-4)
    neg = tb.round_stochastic(-x, torch.bfloat16)
    assert neg.float().mean().item() == pytest.approx(-(1.0 + 2 ** -10), abs=3e-4)
    assert torch.equal(tb.round_stochastic(x, torch.float32), x)
    exact = torch.tensor([1.0, -2.0, 0.5, 0.0])       # representable values are never moved
    assert torch.equal(tb.round_stochastic(exact, torch.bfloat16).float(), exact)


def test_optimizer_matches_torch_adamw_in_fp32():
    torch.manual_seed(0)
    p1 = nn.Parameter(torch.randn(8, 4))
    p2 = nn.Parameter(p1.detach().clone())
    a = torch.optim.AdamW([p1], lr=1e-2, weight_decay=0.1)
    b = tb.StochasticAdamW([p2], lr=1e-2, weight_decay=0.1)
    for _ in range(6):
        g = torch.randn(8, 4)
        p1.grad, p2.grad = g.clone(), g.clone()
        a.step()
        b.step()
    assert torch.allclose(p1, p2, atol=1e-6)


def test_bf16_updates_smaller_than_the_weight_spacing_still_accumulate():
    """lr 1e-5 against weights near 1.0 (bf16 spacing 2**-7 = 0.0078): round-to-nearest would never move them."""
    torch.manual_seed(0)
    p = nn.Parameter(torch.ones(20000, dtype=torch.bfloat16))
    opt = tb.StochasticAdamW([p], lr=1e-5)
    for _ in range(1000):
        p.grad = torch.ones_like(p)                   # constant gradient: Adam steps about lr each time
        opt.step()
    moved = (1.0 - p.float().mean().item())
    assert moved == pytest.approx(1000 * 1e-5, rel=0.3)   # about 0.01 in expectation
    naive = torch.ones(20000, dtype=torch.bfloat16)
    for _ in range(1000):
        naive = (naive.float() - 1e-5).to(torch.bfloat16)
    assert naive.float().mean().item() == 1.0         # what plain rounding does: nothing


def test_ft_scope_and_save_load_roundtrip(tmp_path):
    torch.manual_seed(0)
    m = Tiny()
    params = tb.ft_params(m, "all")
    assert len(params) == 4
    with pytest.raises(ValueError):
        tb.ft_params(m, "nope")
    m.layers[0]["w1"].weight.data += 1.0
    tb.save_ft(params, tmp_path, "all")
    other = Tiny()
    info = tb.apply_ft(other, tmp_path)
    assert info["ft_tensors_loaded"] == 4
    for (n, a), (_, b) in zip(m.named_parameters(), other.named_parameters()):
        assert torch.equal(a, b), n
    bad = json.loads((tmp_path / "ft_index.json").read_text())
    bad["weight_map"]["ghost.weight"] = "ft-00001.safetensors"
    (tmp_path / "ft_index.json").write_text(json.dumps(bad))
    with pytest.raises(KeyError, match="not in the model"):
        tb.apply_ft(other, tmp_path)


def _train_args(tiny_dir, data_dir, out, method, *extra):
    return tb.parse_args(["--method", method, "--model_dir", str(tiny_dir), "--data_dir", str(data_dir),
                          "--output_dir", str(out), "--n_gpus", "0", "--train_steps", "10", "--train_batch_size", "2",
                          "--train_pool", "40", *extra])


def test_lora_then_evaluation_end_to_end_on_cpu(tiny_dir, data_dir, tmp_path):
    out = tmp_path / "run"
    s = tb.run(_train_args(tiny_dir, data_dir, out / "train_lora", "lora", "--learning_rate", "5e-3",
                           "--lora_r", "4", "--lora_alpha", "8"))
    assert s["method"] == "lora" and s["checkpoint"]["modules"] and len(s["checkpoint"]["modules"]) == 8  # 2 layers x 4 experts
    assert 0 < s["trainable_share"] < 0.5 and s["final_loss"] == s["final_loss"]
    assert (out / "train_lora" / "lora.safetensors").exists()
    common = ["--model_dir", str(tiny_dir), "--data_dir", str(data_dir), "--out_dir", str(out),
              "--team10_dir", str(TEAM10_DIR), "--n_gpus", "0", "--gsm8k_limit", "4", "--gen_batch", "2",
              "--max_new_tokens", "4", "--eval_suffix", "_t"]
    ev = evaluate.run(evaluate.parse_args(["--tag", "lora", "--ckpt_dir", str(out / "train_lora"),
                                           "--parts", "gsm8k"] + common))
    assert ev["gsm8k"]["n"] == 4 and ev["weights_changed"]["lora_modules_merged"] == 8
    assert (out / "eval_lora_t" / "gsm8k_generations.jsonl").exists()   # suffix keeps results apart
    with pytest.raises(ValueError, match="needs --ckpt_dir"):
        evaluate.run(evaluate.parse_args(["--tag", "ft", "--parts", "gsm8k"] + common))


def test_full_finetune_of_the_gate_weights_end_to_end_on_cpu(tiny_dir, data_dir, tmp_path):
    out = tmp_path / "run"
    s = tb.run(_train_args(tiny_dir, data_dir, out / "train_ft", "ft", "--ft_scope", "gate",
                           "--learning_rate", "1e-3"))
    assert s["trainable_params"] == 8 * 32 * 64      # the 8 gate projections of 32 x 64, as in the DT test
    common = ["--model_dir", str(tiny_dir), "--data_dir", str(data_dir), "--out_dir", str(out),
              "--team10_dir", str(TEAM10_DIR), "--n_gpus", "0", "--gsm8k_limit", "4", "--gen_batch", "2",
              "--max_new_tokens", "4"]
    ev = evaluate.run(evaluate.parse_args(["--tag", "ft", "--ckpt_dir", str(out / "train_ft"),
                                           "--parts", "gsm8k"] + common))
    assert ev["weights_changed"]["ft_tensors_loaded"] == 8
    evaluate.run(evaluate.parse_args(["--tag", "base", "--parts", "gsm8k"] + common))
    res, md = report.build(str(out))
    assert "Full FT" in md and "Training: Full FT" in md and res["ft"]["gsm8k"]["n"] == 4
