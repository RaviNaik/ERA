import json
from math import comb

import pytest

from dtphi import paired


def _write(path, correct, prompts=None):
    with open(path, "w") as f:
        for i, c in enumerate(correct):
            p = prompts[i] if prompts else [i, i + 1]
            f.write(json.dumps({"idx": i, "prompt_ids": p, "response_ids": [1], "text": "x", "correct": c}) + "\n")


def test_mcnemar_exact_known_values():
    assert paired.mcnemar_exact(0, 0) == 1.0
    assert paired.mcnemar_exact(5, 5) == 1.0
    assert paired.mcnemar_exact(10, 0) == pytest.approx(2 * 0.5 ** 10)
    # b=2, c=8: 2 * P(X <= 2), X ~ Binomial(10, 0.5)
    assert paired.mcnemar_exact(2, 8) == pytest.approx(2 * (1 + 10 + 45) / 1024)
    assert paired.mcnemar_exact(8, 2) == paired.mcnemar_exact(2, 8)
    assert 0 < paired.mcnemar_exact(400, 300) <= 1   # large counts do not overflow
    assert comb(10, 2) == 45


def test_compare_counts_difference_and_interval(tmp_path):
    a = [True] * 50 + [False] * 50
    b = [True] * 50 + [True] * 10 + [False] * 40      # B is right on 10 more questions, wrong on none
    _write(tmp_path / "a.jsonl", a)
    _write(tmp_path / "b.jsonl", b)
    r = paired.compare(tmp_path / "a.jsonl", tmp_path / "b.jsonl", resamples=2000)
    assert (r["n"], r["only_a"], r["only_b"]) == (100, 0, 10)
    assert r["acc_a"] == 0.5 and r["acc_b"] == 0.6 and r["diff"] == pytest.approx(0.1)
    assert r["mcnemar_p"] == pytest.approx(2 * 0.5 ** 10)
    lo, hi = r["ci95"]
    assert 0 < lo <= 0.1 <= hi <= 0.2          # all ten discordant pairs go one way, so the interval excludes 0


def test_identical_runs_have_no_difference(tmp_path):
    c = [True, False] * 20
    _write(tmp_path / "a.jsonl", c)
    _write(tmp_path / "b.jsonl", c)
    r = paired.compare(tmp_path / "a.jsonl", tmp_path / "b.jsonl", resamples=500)
    assert r["diff"] == 0 and r["mcnemar_p"] == 1.0 and r["ci95"] == [0.0, 0.0]


def test_different_questions_or_prompts_are_refused(tmp_path):
    _write(tmp_path / "a.jsonl", [True] * 5)
    _write(tmp_path / "b.jsonl", [True] * 4)
    with pytest.raises(ValueError, match="same questions"):
        paired.compare(tmp_path / "a.jsonl", tmp_path / "b.jsonl")
    _write(tmp_path / "c.jsonl", [True] * 5, prompts=[[9, 9]] * 5)
    with pytest.raises(ValueError, match="different prompt"):
        paired.compare(tmp_path / "a.jsonl", tmp_path / "c.jsonl")


def test_compare_run_finds_every_evaluated_variant(tmp_path):
    for tag, correct in (("base", [False] * 30), ("dt", [True] * 12 + [False] * 18), ("lora", [True] * 3 + [False] * 27)):
        (tmp_path / f"eval_{tag}_full").mkdir()
        _write(tmp_path / f"eval_{tag}_full" / "gsm8k_chat_generations.jsonl", correct)
    res, md = paired.compare_run(tmp_path, "_full", resamples=200)
    assert set(res) == {"dt/gsm8k_chat", "lora/gsm8k_chat"}
    assert res["dt/gsm8k_chat"]["only_b"] == 12 and res["lora/gsm8k_chat"]["only_b"] == 3
    assert "Deletion Tuning, chat template" in md and "LoRA, chat template" in md and "Full FT" not in md
    paired.main(["--run_dir", str(tmp_path), "--eval_suffix", "_full", "--resamples", "100"])
    assert json.loads((tmp_path / "paired_full.json").read_text()).keys() == res.keys()
    assert (tmp_path / "paired_full.md").exists()
