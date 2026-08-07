"""
Webapp Data Exporter - reads the generated submission_artifacts/ bundle and
emits webapp/data.js — a single JS file that embeds the *real* run output
as a `window.TDES_DATA` object.

This is what lets the dashboard at v5/session6/webapp/ show genuine,
reproducible numbers instead of hand-typed placeholders: every field in
data.js is read back off disk from files run_demo.py itself just wrote.
Re-running run_demo.py regenerates data.js from scratch.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.mixture.mixture_scheduler import CURRICULUM_STAGES


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _count_test_functions(tests_dir: Path) -> int:
    total = 0
    for p in tests_dir.glob("test_*.py"):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("def test_"):
                total += 1
    return total


def export_webapp_data(artifacts_dir: Path, webapp_dir: Path, project_root: Path,
                        run_config: dict) -> Path:
    """Build webapp/data.js from the artifacts just written to disk."""
    manifests_dir = artifacts_dir / "manifests"
    ledgers_dir = artifacts_dir / "ledgers"
    checkpoints_dir = artifacts_dir / "checkpoints"

    evidence = _read_json(artifacts_dir / "evidence.json", {})
    performance = _read_json(artifacts_dir / "performance.json", {})
    packing_report = _read_json(ledgers_dir / "packing_report.json", {})
    tokenizer = _read_json(artifacts_dir / "tokenizer_spec.json", {})
    manifest_summary = _read_json(manifests_dir / "manifest_summary.json", {})
    mixture_actual = _read_json(ledgers_dir / "mixture_actual.json", {})
    opus_summary = _read_json(ledgers_dir / "opus_summary.json", {})
    doc_loss_summary = _read_json(ledgers_dir / "doc_loss_summary.json", {})
    replay = _read_json(ledgers_dir / "replay_hashes.json", {"steps_replayed": []})

    # Shard manifests -> flat rows for the shard grid
    manifests = []
    for p in sorted(manifests_dir.glob("shard_*.json")):
        m = _read_json(p, {})
        if not m:
            continue
        manifests.append({
            "shard_id": m.get("shard_id"),
            "lane": m.get("capability_lane"),
            "token_count": m.get("token_count"),
            "admission": m.get("admission"),
            "admission_score": m.get("admission_score"),
            "language": m.get("language"),
            "is_eval": m.get("is_eval", False),
        })

    opus_decisions = _read_jsonl(ledgers_dir / "opus_decisions.jsonl")
    consumption = (_read_jsonl(ledgers_dir / "consumption.jsonl")
                   + _read_jsonl(ledgers_dir / "consumption_resumed.jsonl"))
    learning = (_read_jsonl(ledgers_dir / "learning.jsonl")
                + _read_jsonl(ledgers_dir / "learning_resumed.jsonl"))

    # Checkpoints, with forked_from inferred by matching weights_hash back to
    # the main-branch checkpoint it was cloned from (the fork's own meta.json
    # does not carry that field — see checkpoint_manager.save()).
    checkpoints = []
    raw_ckpts = []
    for p in sorted(checkpoints_dir.glob("ckpt_*")):
        meta = _read_json(p / "meta.json")
        if meta:
            raw_ckpts.append(meta)
    for c in raw_ckpts:
        entry = dict(c)
        if c.get("branch_id") != "main":
            origin = next((o for o in raw_ckpts
                           if o["checkpoint_id"] != c["checkpoint_id"]
                           and o.get("branch_id") == "main"
                           and o.get("weights_hash") == c.get("weights_hash")), None)
            if origin:
                entry["forked_from"] = origin["checkpoint_id"]
        checkpoints.append(entry)

    replay_hashes = replay.get("steps_replayed", [])

    curriculum_stages = [{
        "id": s.stage_id,
        "name": s.name,
        "budget": f"{s.token_budget_fraction:.0%}",
        "weights": s.lane_weights,
    } for s in CURRICULUM_STAGES]

    test_count = _count_test_functions(project_root / "tests")

    meta = {
        **run_config,
        "total_docs": manifest_summary.get("total"),
        "admitted_docs": manifest_summary.get("admitted"),
        "blocked_docs": manifest_summary.get("blocked"),
        "test_count": test_count,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    data = {
        "meta": meta,
        "evidence": evidence,
        "performance": performance,
        "packingReport": packing_report,
        "manifests": manifests,
        "opusDecisions": opus_decisions,
        "consumption": consumption,
        "learning": learning,
        "checkpoints": checkpoints,
        "replayHashes": replay_hashes,
        "mixtureActual": mixture_actual,
        "opusSummary": opus_summary,
        "docLossSummary": doc_loss_summary,
        "tokenizer": tokenizer,
        "manifestSummary": manifest_summary,
        "curriculumStages": curriculum_stages,
    }

    webapp_dir.mkdir(parents=True, exist_ok=True)
    out_path = webapp_dir / "data.js"
    header = (
        "// AUTO-GENERATED by run_demo.py -> src/audit/webapp_export.py\n"
        "// Source: submission_artifacts/ (this run). Do not hand-edit — it is\n"
        "// overwritten every time the demo runs. Every field here was read back\n"
        "// off disk from files the pipeline itself produced.\n"
    )
    out_path.write_text(
        header + "window.TDES_DATA = " + json.dumps(data, indent=2, ensure_ascii=False) + ";\n",
        encoding="utf-8",
    )
    return out_path
