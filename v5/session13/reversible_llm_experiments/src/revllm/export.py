"""Exports the experiment results to the static session13 webapp (data.js).

Reads the per-run records (`results/<label>.json`, located next to
`runs.jsonl`) and, if present, the notebook-only tables
(`results/notebook_tables.json`, see scripts/extract_notebook_tables.py).
The webapp draws every chart from this file; no numbers are hardcoded in its
JavaScript.
"""

from __future__ import annotations

import json
from pathlib import Path

REVERSIBLE = {"euler", "midpoint"}
MB_PER_GB = 1024


def load_runs(runs_jsonl: str | Path = "results/runs.jsonl") -> list[dict]:
    """Reads runs.jsonl, keeping only the most recent record per label."""
    path = Path(runs_jsonl)
    if not path.exists():
        return []
    latest: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            latest[record["summary"]["label"]] = record
    return list(latest.values())


def _r(value, digits=4):
    return None if value is None else round(float(value), digits)


def _downsample(values: list, max_points: int = 160) -> list:
    if len(values) <= max_points:
        return list(values)
    idx = [round(i * (len(values) - 1) / (max_points - 1)) for i in range(max_points)]
    return [values[i] for i in idx]


def _run_entry(record: dict) -> dict:
    s = record["summary"]
    rev = s.get("reversibility") or {}
    return {
        "label": s["label"],
        "variant": s["variant"],
        "status": s["status"],
        "batch": s["batch_size"],
        "steps": s["total_steps"],
        "tokensM": _r(s["tokens_seen"] / 1e6, 2),
        "peakLr": s["peak_lr"],
        "valLoss": _r(s.get("final_val_loss")),
        "valPpl": _r(s.get("final_val_perplexity"), 2),
        "valAcc": _r(s.get("final_val_accuracy")),
        "trainLoss": _r(s.get("final_train_loss")),
        "tokensPerSec": _r(s.get("mean_tokens_per_sec"), 0),
        "stepMs": _r(s.get("mean_step_time_ms"), 1),
        "dataMs": _r(s.get("mean_data_ms"), 1),
        "forwardMs": _r(s.get("mean_forward_ms"), 1),
        "backwardMs": _r(s.get("mean_backward_ms"), 1),
        "optimizerMs": _r(s.get("mean_optimizer_ms"), 2),
        "wallMin": _r(s.get("wall_clock_s", 0) / 60, 2),
        "mfu": _r(s.get("mfu")),
        "modelTflops": _r(s.get("model_tflops_per_sec"), 2),
        "peakGB": _r((s.get("peak_step_memory_mb") or 0) / MB_PER_GB, 3),
        "savedGB": _r((s.get("activation_saved_mb") or 0) / MB_PER_GB, 3),
        "staticGB": _r((s.get("static_memory_mb") or 0) / MB_PER_GB, 3),
        "actPerSeqMB": _r(s.get("activation_saved_mb_per_sample"), 2),
        "gradNormMean": _r(s.get("mean_grad_norm"), 3),
        "gradNormMax": _r(s.get("max_grad_norm"), 1),
        "clipFrac": _r(s.get("frac_steps_clipped")),
        "reconError": rev.get("max_rel_error"),
        "gradCosine": _r(rev.get("grad_cosine"), 8),
        "aimRun": s.get("aim_run_hash"),
    }


def _curves(record: dict) -> dict:
    ev, h = record.get("evals", {}), record.get("history", {})
    return {
        "evalTokensM": [_r(t / 1e6, 3) for t in ev.get("tokens", [])],
        "evalWallMin": [_r(t / 60, 3) for t in ev.get("wall_clock_s", [])],
        "valLoss": [_r(v) for v in ev.get("val_loss", [])],
        "trainLoss": [_r(v) for v in ev.get("train_loss", [])],
        "valAcc": [_r(v) for v in ev.get("val_accuracy", [])],
        "stepTokensM": _downsample([_r(t / 1e6, 3) for t in h.get("tokens", [])]),
        "stepLoss": _downsample([_r(v) for v in h.get("loss", [])]),
        "gradNorm": _downsample([_r(v, 3) for v in h.get("grad_norm", [])]),
        "lr": _downsample([_r(v, 6) for v in h.get("lr", [])]),
        "tokensPerSec": _downsample([_r(v, 0) for v in h.get("tokens_per_sec", [])]),
    }


def build_session_data(records: list[dict], winner: str | None = None, tables: dict | None = None,
                       full_records: dict[str, dict] | None = None) -> dict:
    runs = [_run_entry(r) for r in records]
    ok = [r for r in runs if r["status"] == "ok"]
    reversible_ok = [r for r in ok if r["variant"] in REVERSIBLE]
    if winner is None and reversible_ok:
        at_x = [r for r in reversible_ok if r["batch"] == min(x["batch"] for x in reversible_ok)]
        winner = min(at_x, key=lambda r: r["valLoss"])["variant"]
    env = (records[0].get("env") if records else None) or {}
    tables = tables or {}
    max_batch = tables.get("max_batch", {})
    base = next((r for r in ok if r["variant"] == "baseline"), None)
    best = next((r for r in ok if r["variant"] == winner and r["batch"] == (base or {}).get("batch")), None)

    hero = [
        {"val": "20.1M", "lbl": "parameters (9 layers, d=256)", "cls": ""},
        {"val": "50M", "lbl": "training tokens per run", "cls": ""},
    ]
    if base and best:
        hero += [
            {"val": f"−{(1 - best['peakGB'] / base['peakGB']) * 100:.0f}%", "lbl": "peak memory, same batch", "cls": "green"},
            {"val": f"{base['actPerSeqMB'] / best['actPerSeqMB']:.0f}×", "lbl": "fewer stored activations", "cls": "indigo"},
            {"val": f"{best['valLoss'] - base['valLoss']:+.3f}", "lbl": f"val loss, {winner} vs baseline", "cls": "amber"},
        ]
    if max_batch.get("baseline") and max_batch.get(winner):
        hero.append({"val": f"{max_batch[winner] / max_batch['baseline']:.1f}×",
                     "lbl": f"max batch ({max_batch['baseline']} → {max_batch[winner]})", "cls": "rose"})

    return {
        "meta": {
            "gpu": env.get("gpu_name"),
            "gpuMemoryGB": _r((env.get("gpu_total_memory_mb") or 0) / MB_PER_GB, 1),
            "torch": env.get("torch"),
            "winner": winner,
            "tokenBudgetM": 50,
        },
        "heroStats": hero,
        "runs": runs,
        "curves": {r["summary"]["label"]: _curves((full_records or {}).get(r["summary"]["label"], r)) for r in records},
        "tables": tables,
    }


def export_webapp_data(
    runs_jsonl: str | Path = "results/runs.jsonl",
    out_path: str | Path = "../webapp/data.js",
    winner: str | None = None,
) -> dict:
    runs_jsonl = Path(runs_jsonl)
    records = load_runs(runs_jsonl)
    full = {}
    for record in records:
        path = runs_jsonl.parent / f"{record['summary']['label']}.json"
        if path.exists():
            full[record["summary"]["label"]] = json.loads(path.read_text(encoding="utf-8"))
    records = [full.get(r["summary"]["label"], r) for r in records]
    tables_path = runs_jsonl.parent / "notebook_tables.json"
    tables = json.loads(tables_path.read_text(encoding="utf-8")) if tables_path.exists() else {}
    data = build_session_data(records, winner, tables, full)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "/* Generated by revllm.export from notebooks/results/. Do not edit by hand. */\n"
        "window.SESSION_DATA = " + json.dumps(data, indent=1) + ";\n",
        encoding="utf-8",
    )
    return data
