#!/usr/bin/env python3
"""Build results/scores.csv, results/runs_row.csv (G12's runs.csv columns) and results/environment.json.

Reads, under --root/results: runs/<format>/summary.json (G9's record format, from run_mini_gguf.py),
gguf_check_<format>.json, quant_times.json, speed_<format>.json, gpus.txt, env_probe.json.
score = unweighted mean of the five eval_mini primary metrics, as G12's runs.csv defines it.
"""
from __future__ import annotations

import argparse, csv, datetime, json, statistics
from pathlib import Path

COLUMNS = ["date", "model", "revision", "format", "bpw", "size_gb", "router_type", "output_type", "eval_set", "n_items",
           "score", "score_by_kind", "quant_time_s", "eval_time_s", "gen_tok_s", "machine", "runtime", "notes"]
TASKS = ["arc_challenge", "hellaswag", "mmlu", "gsm8k", "truthfulqa_mc2"]
EVAL_SET = ("G9 eval_mini mini (arc_challenge 500, hellaswag 400, mmlu 570, gsm8k 5-shot 200, truthfulqa_mc2 200; "
            "seed 42)")


def load(p: Path, default=None):
    return json.loads(p.read_text()) if p.exists() else default


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--run-by", required=True)
    ap.add_argument("--formats", required=True)
    ap.add_argument("--llama-tag", required=True)
    ap.add_argument("--llama-cpp-python", required=True)
    ap.add_argument("--lab-sha", required=True)
    a = ap.parse_args()
    res = a.root / "results"
    formats = a.formats.split()
    times = load(res / "quant_times.json", {})
    gpus = (res / "gpus.txt").read_text().strip().replace("\n", " + ") if (res / "gpus.txt").exists() else ""
    probe = load(res / "env_probe.json", {})

    rows, scores, env_formats = [], {}, {}
    for fmt in formats:
        summ = load(res / "runs" / fmt / "summary.json")
        chk = load(res / f"gguf_check_{fmt}.json")
        if not summ or not chk:
            print(f"{fmt}: missing summary.json or gguf check, skipped")
            continue
        prim = {r["task"]: r for r in summ["records"] if r["primary"]}
        missing = [t for t in TASKS if t not in prim]
        if missing:
            print(f"{fmt}: tasks not finished: {missing}, skipped")
            continue
        by_kind = {t: round(prim[t]["value"], 4) for t in TASKS}
        score = round(statistics.fmean(by_kind.values()), 4)
        n_items = sum(prim[t]["n_items"] for t in TASKS)
        eval_s = round(sum(prim[t].get("wall_clock_s", 0) or 0 for t in TASKS))
        speed = load(res / f"speed_{fmt}.json", {})
        scores[fmt] = {**{t: round(100 * v, 1) for t, v in by_kind.items()}, "mean": round(100 * score, 2)}
        meta = summ["run_meta"]
        rows.append({
            "date": datetime.date.today().isoformat(), "model": a.model, "revision": a.revision[:12], "format": fmt,
            "bpw": chk["bits_per_weight"], "size_gb": round(chk["bytes"] / 1e9, 2), "router_type": chk["router_type"],
            "output_type": chk["output_type"], "eval_set": EVAL_SET, "n_items": n_items, "score": f"{score:.4f}",
            "score_by_kind": json.dumps(by_kind), "quant_time_s": 0.0 if fmt == "bf16" else times.get(fmt, ""),
            "eval_time_s": eval_s, "gen_tok_s": speed.get("gen_tok_s", ""), "machine": meta.get("hardware", gpus),
            "runtime": f"llama.cpp {a.llama_tag}; llama-cpp-python {a.llama_cpp_python} in-process, all layers on GPU",
            "notes": (f"run by {a.run_by}; own server (not Kaggle); G9 harness sha256 {meta.get('g9_harness_sha256')}; "
                      f"lab commit {a.lab_sha[:12]}; score = unweighted mean of the 5 eval_mini primary metrics "
                      f"(acc_norm ARC/HellaSwag, acc MMLU/TruthfulQA-mc2, exact_match GSM8K); same questions and "
                      f"runtime as bf16; GGUFs made with llama.cpp {a.llama_tag}"),
        })
        env_formats[fmt] = {"gguf_bytes": chk["bytes"], "n_params": chk["n_params"], "architecture": chk["architecture"],
                            "n_ctx": meta.get("n_ctx"), "n_gpu_layers": meta.get("n_gpu_layers"),
                            "backend": meta.get("backend"), "lm_eval": meta.get("lm_eval")}

    with open(res / "scores.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["format", *TASKS, "mean"])
        for fmt, s in scores.items():
            w.writerow([fmt, *[s[t] for t in TASKS], s["mean"]])
    with open(res / "runs_row.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    (res / "environment.json").write_text(json.dumps({
        "run_by": a.run_by, "finished": datetime.datetime.now().isoformat(timespec="seconds"), "model": a.model,
        "revision": a.revision, "llama_cpp_tag": a.llama_tag, "llama_cpp_python": a.llama_cpp_python,
        "lab_commit": a.lab_sha, "gpus": gpus, "packages": probe, "formats": env_formats}, indent=1))
    print((res / "scores.csv").read_text())
    if "bf16" in scores:
        for fmt, s in scores.items():
            if fmt != "bf16":
                print(f"{fmt}: mean {s['mean'] - scores['bf16']['mean']:+.2f} points vs bf16")
    print(f"wrote {res / 'runs_row.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
