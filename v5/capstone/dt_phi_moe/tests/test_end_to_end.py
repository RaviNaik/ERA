import json

import pytest
import torch

from conftest import TEAM10_DIR
from dtphi import evaluate, train


@pytest.fixture
def cpu_cuda(monkeypatch):
    """The official retention_forward_kl.response_logprobs calls Tensor.cuda(); on the CPU test box it is a no-op."""
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)


def test_official_training_then_evaluation_end_to_end_on_cpu(tiny_dir, data_dir, tmp_path, cpu_cuda):
    import train_sft
    keep = list(train_sft.TARGET_MODULES)
    out = tmp_path / "run"
    try:
        tr = train.run(train.parse_args([
            "--model_dir", str(tiny_dir), "--data_dir", str(data_dir), "--output_dir", str(out / "train"),
            "--n_gpus", "0", "--train_steps", "20", "--train_batch_size", "2", "--train_pool", "40",
            "--learning_rate", "5e-3"]))
    finally:
        train_sft.TARGET_MODULES[:] = keep
    w = tr["wanda_prior"]
    assert w["sparse_layers"] == 8 and w["layers_with_wanda_scores"] + w["layers_without_scores"] == 8
    assert 0 <= w["deleted_frac_step0"] < 0.6
    assert (out / "train" / "masks" / "gsm8k" / "model.safetensors").exists()
    log = [json.loads(l) for l in (out / "train" / "train_log.jsonl").read_text().splitlines()]
    assert [r["iter"] for r in log] == [10, 20] and all(r["loss"] == r["loss"] and r["loss"] > 0 for r in log)
    assert tr["mask"]["total_params"] == 8 * 32 * 64  # 8 gate projections of 32 x 64

    common = ["--model_dir", str(tiny_dir), "--data_dir", str(data_dir), "--out_dir", str(out),
              "--team10_dir", str(TEAM10_DIR), "--n_gpus", "0", "--gsm8k_limit", "6", "--gen_batch", "3",
              "--max_new_tokens", "6", "--team10_max_new_tokens", "6"]
    base = evaluate.run(evaluate.parse_args(["--tag", "base", "--parts", "gsm8k,gsm8k_chat,team10"] + common))
    assert base["gsm8k"]["n"] == 6 and base["gsm8k_chat"]["n"] == 6
    assert base["team10"]["indic_v1"]["n"] == 44 and base["team10"]["safety_v1"]["n"] == 41
    first = json.loads((out / "eval_base" / "gsm8k_generations.jsonl").read_text().splitlines()[0])
    assert {"prompt_ids", "response_ids", "text", "correct"} <= set(first)

    dt = evaluate.run(evaluate.parse_args(["--tag", "dt", "--mask_dir", str(out / "train" / "masks" / "gsm8k"),
                                           "--parts", "gsm8k,kl"] + common))
    assert dt["weights_zeroed"] == tr["mask"]["zero_params"]
    assert dt["forward_kl"]["n_prompts"] >= 1
    assert dt["forward_kl"]["forward_kl_per_token_mean"] == dt["forward_kl"]["forward_kl_per_token_mean"]


def test_forward_kl_is_zero_for_the_same_model_and_log_ratio_otherwise():
    from dtphi.eval_gsm8k import forward_kl
    a = [torch.tensor([-1.0, -2.0]), torch.tensor([-0.5])]
    assert forward_kl(a, a)["forward_kl_per_token_mean"] == 0.0
    b = [torch.tensor([-1.5, -2.5]), torch.tensor([-1.0])]
    assert forward_kl(a, b)["forward_kl_per_token_mean"] == pytest.approx(0.5)  # mean of log p0 - log p: (-1)-(-1.5), (-2)-(-2.5), (-0.5)-(-1) are all +0.5


def test_two_shards_merge_to_the_single_process_evaluation(tiny_dir, data_dir, tmp_path):
    common = ["--model_dir", str(tiny_dir), "--data_dir", str(data_dir), "--team10_dir", str(TEAM10_DIR),
              "--n_gpus", "0", "--gsm8k_limit", "7", "--gen_batch", "3", "--max_new_tokens", "6",
              "--team10_max_new_tokens", "6", "--parts", "gsm8k,gsm8k_chat,team10"]
    one = evaluate.run(evaluate.parse_args(["--tag", "base", "--out_dir", str(tmp_path / "one")] + common))
    two_args = ["--tag", "base", "--out_dir", str(tmp_path / "two"), "--num_shards", "2"] + common
    for k in (0, 1):
        evaluate.run(evaluate.parse_args(two_args + ["--shard", str(k)]))
    two = evaluate.merge(evaluate.parse_args(two_args + ["--merge"]))
    for part in ("gsm8k", "gsm8k_chat"):
        assert {x: two[part][x] for x in ("n", "correct")} == {x: one[part][x] for x in ("n", "correct")}
        a = (tmp_path / "one" / "eval_base" / f"{part}_generations.jsonl").read_text()
        b = (tmp_path / "two" / "eval_base" / f"{part}_generations.jsonl").read_text()
        assert a == b
    assert two["team10"] == one["team10"]
