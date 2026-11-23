"""Paired comparison of two GSM8K evaluations of the same questions (base vs Deletion Tuning, LoRA or full FT).

Both runs answered the same questions, so the right test looks at the questions where they disagree, not at two
independent accuracies. For each pair of `gsm8k*_generations.jsonl` files this reports

  - accuracy of each, the difference and its 95% interval (bootstrap over questions, paired),
  - the discordant counts: b = right in A and wrong in B, c = wrong in A and right in B,
  - the exact two-sided McNemar p-value (binomial test of b against c, p = 0.5).

    python -m dtphi.paired --run_dir runs/<run>                        # base vs every variant that was evaluated
    python -m dtphi.paired --a eval_base/gsm8k_chat_generations.jsonl --b eval_dt/gsm8k_chat_generations.jsonl

Nothing here needs a GPU; it only reads the generation files that dtphi.evaluate already writes."""
from __future__ import annotations

import argparse
import json
import random
from math import comb
from pathlib import Path

VARIANTS = (("dt", "Deletion Tuning"), ("lora", "LoRA"), ("ft", "Full FT"))
PARTS = (("gsm8k", "training format"), ("gsm8k_chat", "chat template"))


def load_correct(path: str | Path) -> dict[int, tuple[bool, tuple]]:
    """idx -> (correct, prompt_ids) from one generations file."""
    out: dict[int, tuple[bool, tuple]] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[int(r["idx"])] = (bool(r["correct"]), tuple(r["prompt_ids"]))
    return out


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value: b + c discordant pairs, under H0 each falls on either side with p = 0.5."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1))
    return min(1.0, 2.0 * tail / 2 ** n)


def bootstrap_diff_ci(a: list[int], b: list[int], resamples: int = 10000, seed: int = 0,
                      level: float = 0.95) -> tuple[float, float]:
    """Percentile interval of mean(b) - mean(a) over questions resampled with replacement (paired)."""
    n = len(a)
    rng = random.Random(seed)
    diffs = []
    for _ in range(resamples):
        s = 0
        for _ in range(n):
            i = rng.randrange(n)
            s += b[i] - a[i]
        diffs.append(s / n)
    diffs.sort()
    lo = diffs[int(((1 - level) / 2) * resamples)]
    hi = diffs[min(resamples - 1, int((1 - (1 - level) / 2) * resamples))]
    return lo, hi


def compare(path_a: str | Path, path_b: str | Path, resamples: int = 10000, seed: int = 0) -> dict:
    """Paired comparison of two generation files; the files must hold the same questions (same idx and prompt)."""
    A, B = load_correct(path_a), load_correct(path_b)
    if A.keys() != B.keys():
        raise ValueError(f"the two files do not hold the same questions ({len(A)} vs {len(B)}, "
                         f"{len(A.keys() ^ B.keys())} indices differ)")
    ids = sorted(A)
    for i in ids:
        if A[i][1] != B[i][1]:
            raise ValueError(f"question {i} has a different prompt in the two files (different prompt format?)")
    a = [int(A[i][0]) for i in ids]
    b = [int(B[i][0]) for i in ids]
    n = len(ids)
    only_a = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)  # right in A, wrong in B
    only_b = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)  # wrong in A, right in B
    lo, hi = bootstrap_diff_ci(a, b, resamples, seed)
    return {"n": n, "acc_a": sum(a) / n, "acc_b": sum(b) / n, "diff": (sum(b) - sum(a)) / n,
            "ci95": [lo, hi], "only_a": only_a, "only_b": only_b, "mcnemar_p": mcnemar_exact(only_a, only_b),
            "resamples": resamples, "seed": seed}


def compare_run(run_dir: str | Path, suffix: str = "", resamples: int = 10000, seed: int = 0) -> tuple[dict, str]:
    """Base against every evaluated variant, for both GSM8K formats. `suffix` is the evaluate --eval_suffix."""
    d = Path(run_dir)
    res: dict = {}
    rows = []
    for tag, label in VARIANTS:
        for part, pname in PARTS:
            pa = d / f"eval_base{suffix}" / f"{part}_generations.jsonl"
            pb = d / f"eval_{tag}{suffix}" / f"{part}_generations.jsonl"
            if not (pa.exists() and pb.exists()):
                continue
            r = compare(pa, pb, resamples, seed)
            res[f"{tag}/{part}"] = r
            rows.append(f"| {label}, {pname} | {r['n']} | {100 * r['acc_a']:.1f} | {100 * r['acc_b']:.1f} | "
                        f"{100 * r['diff']:+.1f} | [{100 * r['ci95'][0]:+.1f}, {100 * r['ci95'][1]:+.1f}] | "
                        f"{r['only_a']} / {r['only_b']} | {r['mcnemar_p']:.2g} |")
    md = ["## Paired comparison with the base model (same questions)", "",
          "| variant, format | n | base % | variant % | change (points) | 95% interval | base-only / variant-only | "
          "McNemar p |", "|---|---:|---:|---:|---:|---:|---:|---:|", *rows, "",
          f"Interval: bootstrap over questions, {resamples} resamples, seed {seed}. 'base-only / variant-only' are "
          "the questions only the base model / only the variant answered correctly; McNemar p is the exact two-sided "
          "test on those two counts.", ""]
    return res, "\n".join(md)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run_dir", default="", help="compare base with every evaluated variant of this run")
    p.add_argument("--eval_suffix", default="", help="the --eval_suffix the evaluations were run with")
    p.add_argument("--a", default="", help="generations file A (the reference, e.g. base)")
    p.add_argument("--b", default="", help="generations file B")
    p.add_argument("--resamples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    if a.run_dir:
        res, md = compare_run(a.run_dir, a.eval_suffix, a.resamples, a.seed)
        if not res:
            raise SystemExit(f"no pair of generation files to compare in {a.run_dir} (suffix '{a.eval_suffix}')")
        (Path(a.run_dir) / f"paired{a.eval_suffix}.json").write_text(json.dumps(res, indent=1))
        (Path(a.run_dir) / f"paired{a.eval_suffix}.md").write_text(md)
        print(md)
    elif a.a and a.b:
        print(json.dumps(compare(a.a, a.b, a.resamples, a.seed), indent=1))
    else:
        p.error("give --run_dir, or --a and --b")


if __name__ == "__main__":
    main()
