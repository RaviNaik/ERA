"""Export notebook training summaries to the static session13 webapp."""

from __future__ import annotations

import json
from pathlib import Path


def load_runs(runs_jsonl: str | Path = "results/runs.jsonl") -> list[dict]:
    path = Path(runs_jsonl)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def build_session_data(records: list[dict]) -> dict:
    runs = []
    for record in records:
        hparams = record.get("hparams", {})
        summary = record.get("summary", {})
        runs.append(
            {
                "label": summary.get("label", hparams.get("label", "run")),
                "variant": summary.get("variant", hparams.get("variant", "unknown")),
                "batch_size": hparams.get("batch_size", 0),
                "final_val_loss": summary.get("final_val_loss", 0.0),
                "final_val_accuracy": summary.get("final_val_accuracy", 0.0),
                "mean_tokens_per_sec": summary.get("mean_tokens_per_sec", 0.0),
                "peak_memory_mb": summary.get("peak_memory_mb", 0.0),
                "tokens_seen": summary.get("tokens_seen", 0),
            }
        )
    best_reversible = "pending"
    reversible = [run for run in runs if run["variant"] in {"euler", "midpoint"}]
    if reversible:
        best_reversible = min(reversible, key=lambda run: run["final_val_loss"])[
            "variant"
        ]
    max_batch = "pending"
    if reversible:
        max_batch = str(max(run["batch_size"] for run in reversible))
    findings = [
        "Run the notebook to populate measured loss, throughput, and memory results."
    ]
    if runs:
        fastest = max(runs, key=lambda run: run["mean_tokens_per_sec"])
        lightest = min(runs, key=lambda run: run["peak_memory_mb"])
        findings = [
            f"Best reversible loss variant: {best_reversible}.",
            f"Fastest measured run: {fastest['label']} at {fastest['mean_tokens_per_sec']:.0f} tokens/s.",
            f"Lowest peak memory run: {lightest['label']} at {lightest['peak_memory_mb']:.0f} MB.",
        ]
    return {
        "heroStats": [
            {"value": "20M", "label": "target parameters"},
            {"value": "50M", "label": "tokens per full run"},
            {"value": best_reversible, "label": "best reversible variant"},
            {"value": max_batch, "label": "max stable batch"},
        ],
        "runs": runs,
        "findings": findings,
    }


def export_webapp_data(
    runs_jsonl: str | Path = "results/runs.jsonl",
    out_path: str | Path = "../webapp/data.js",
) -> dict:
    data = build_session_data(load_runs(runs_jsonl))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "window.SESSION_DATA = " + json.dumps(data, indent=2) + ";\n", encoding="utf-8"
    )
    return data
