"""Export notebook training summaries to the static session13 webapp."""

from __future__ import annotations

import json
from pathlib import Path

REVERSIBLE = {"euler", "midpoint"}


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


def _fmt(value, spec: str, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:{spec}}{suffix}"


def build_session_data(records: list[dict], winner: str | None = None) -> dict:
    keys = (
        "label", "variant", "status", "batch_size", "total_steps",
        "final_val_loss", "final_val_accuracy", "final_val_perplexity",
        "mean_tokens_per_sec", "mfu", "peak_step_memory_mb",
        "activation_saved_mb_per_sample", "wall_clock_s", "tokens_seen",
    )
    runs = [{k: r["summary"].get(k) for k in keys} for r in records]
    ok = [r for r in runs if r["status"] == "ok"]
    reversible_ok = [r for r in ok if r["variant"] in REVERSIBLE]
    if winner is None and reversible_ok:
        winner = min(reversible_ok, key=lambda r: r["final_val_loss"])["variant"]
    max_batch = max((r["batch_size"] for r in reversible_ok), default=None)

    findings = ["Run the notebook to populate measured loss, throughput, and memory results."]
    if ok:
        best = min(ok, key=lambda r: r["final_val_loss"])
        fastest = max(ok, key=lambda r: r["mean_tokens_per_sec"])
        lightest = min(ok, key=lambda r: r["activation_saved_mb_per_sample"])
        findings = [
            f"Selected reversible variant: {winner}.",
            f"Lowest validation loss: {best['label']} ({_fmt(best['final_val_loss'], '.4f')}).",
            f"Highest throughput: {fastest['label']} ({_fmt(fastest['mean_tokens_per_sec'], ',.0f')} tokens/s).",
            f"Smallest activation memory: {lightest['label']} "
            f"({_fmt(lightest['activation_saved_mb_per_sample'], '.2f')} MB per sequence).",
        ]
        failed = [r for r in runs if r["status"] != "ok"]
        findings += [f"{r['label']} ended with status '{r['status']}'." for r in failed]
    return {
        "heroStats": [
            {"value": "20M", "label": "parameters"},
            {"value": "50M", "label": "tokens per run"},
            {"value": winner or "pending", "label": "selected reversible variant"},
            {"value": str(max_batch) if max_batch else "pending", "label": "largest reversible batch trained"},
        ],
        "runs": runs,
        "findings": findings,
    }


def export_webapp_data(
    runs_jsonl: str | Path = "results/runs.jsonl",
    out_path: str | Path = "../webapp/data.js",
    winner: str | None = None,
) -> dict:
    data = build_session_data(load_runs(runs_jsonl), winner)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "window.SESSION_DATA = " + json.dumps(data, indent=2) + ";\n", encoding="utf-8"
    )
    return data
