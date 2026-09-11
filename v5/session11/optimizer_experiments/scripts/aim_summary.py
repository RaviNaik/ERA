"""Summarise every training run from results/runs.jsonl (written by trainer.train).

Aim itself tracks the full metric curves (view with `uv run aim up`); this script
just produces the flat table the README embeds.
"""
from __future__ import annotations

import json
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
JSONL = os.path.join(REPO, "results", "runs.jsonl")


def load():
    rows = []
    with open(JSONL) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    rows = load()
    try:
        import pandas as pd
    except ImportError:
        pd = None

    def val_at_200(curve):
        st, vl = curve.get("step", []), curve.get("val_loss", [])
        if len(st) < 2 or st[-1] < 190:
            return float("nan")
        for i in range(1, len(st)):
            if st[i] >= 200:
                if st[i] - st[i - 1] > 60:
                    return float("nan")
                f = (200 - st[i - 1]) / (st[i] - st[i - 1])
                return vl[i - 1] + f * (vl[i] - vl[i - 1])
        return float("nan")

    flat = []
    for r in rows:
        hp, sm = r["hparams"], r["summary"]
        flat.append({
            "experiment": r["experiment"],
            "run": r["label"],
            "aim_hash": (r.get("aim_hash") or "")[:8],
            "width": hp.get("n_embd"),
            "n_params_M": round(hp.get("n_params", 0) / 1e6, 2),
            "schedule": hp.get("schedule"),
            "base_lr": hp.get("base_lr"),
            "warmup": hp.get("warmup"),
            "steps": hp.get("total_steps"),
            "val@200": round(val_at_200(r.get("curve", {})), 4),
            "final_val": round(sm["final_val_loss"], 4),
            "final_train": round(sm["final_train_loss"], 4),
            "wall_s": round(sm["wall_clock_s"], 1),
        })

    if pd is not None:
        df = pd.DataFrame(flat).sort_values(["experiment", "run"]).reset_index(drop=True)
        print(df.to_string(index=False))
        out = os.path.join(REPO, "results", "runs_summary.csv")
        df.to_csv(out, index=False)
        print(f"\n{len(flat)} runs  ->  {os.path.relpath(out, REPO)}")
        print("\nAs markdown:\n")
        print(df.to_markdown(index=False))
    else:
        for r in flat:
            print(r)


if __name__ == "__main__":
    main()
