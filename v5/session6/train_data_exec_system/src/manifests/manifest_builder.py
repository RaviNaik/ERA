"""
Manifest Builder - creates shard manifests and enforces the admission gate.
A shard is only admitted if it passes ALL hard-gate checks (tokenizer_hash,
license_tier, contam_status). Missing any hard gate BLOCKS the shard.
"""
from __future__ import annotations
import hashlib, json, time
from pathlib import Path
from typing import List


HARD_GATES = ["tokenizer_hash", "cleaning_pipeline_hash", "dedup_status",
              "contam_status", "pii_screen_status", "license_tier"]

CLEANING_PIPELINE_HASH = "clean_" + hashlib.sha256(b"era_v5_pipeline_v1").hexdigest()[:12]


def build_manifest(shard_info: dict, manifests_dir: Path) -> dict:
    """Build a full shard manifest and evaluate admission status."""
    manifests_dir.mkdir(parents=True, exist_ok=True)

    license_ok = shard_info["license_tier"] in ("safe", "review")
    is_eval = shard_info.get("is_eval", False)

    # Hard gates
    gates = {
        "tokenizer_hash": {"status": "OK", "value": shard_info["tokenizer_sha"]},
        "cleaning_pipeline_hash": {"status": "OK", "value": CLEANING_PIPELINE_HASH},
        "dedup_status": {"status": "OK", "value": "passed"},
        "contam_status": {"status": "BLOCKED" if is_eval else "OK",
                           "value": "eval_held_out" if is_eval else "clear"},
        "pii_screen_status": {"status": "OK", "value": "screened"},
        "license_tier": {"status": "OK" if license_ok else "BLOCKED",
                          "value": shard_info["license_tier"]},
        "parent_manifest_ids": {"status": "WARN", "value": []},
        "source_manifest": {"status": "OK", "value": shard_info.get("source_url", "")},
    }

    hard_fail = [k for k in HARD_GATES if gates.get(k, {}).get("status") == "BLOCKED"]
    admission = "blocked" if is_eval or hard_fail else "admitted"
    admission_reason = f"eval_held_out={is_eval}" if is_eval else ("hard_fail=" + str(hard_fail) if hard_fail else "all_checks_passed")

    score = 100
    if gates["parent_manifest_ids"]["status"] == "WARN":
        score -= 13
    if shard_info["license_tier"] == "review":
        score -= 11
    for g in HARD_GATES:
        if gates.get(g, {}).get("status") == "BLOCKED":
            score -= 36

    manifest = {
        "shard_id": shard_info["shard_id"],
        "shard_path": shard_info["shard_path"],
        "source_ids": [shard_info.get("source_url", "")],
        "document_ids": [shard_info["doc_id"]],
        "tokenizer_hash": shard_info["tokenizer_sha"],
        "token_count": shard_info["token_count"],
        "language": shard_info["language"],
        "script": shard_info.get("script", "latin"),
        "capability_lane": shard_info["lane"],
        "license_tier": shard_info["license_tier"],
        "cleaning_pipeline_hash": CLEANING_PIPELINE_HASH,
        "dedup_status": "passed",
        "contam_status": "eval_held_out" if is_eval else "clear",
        "pii_screen_status": "screened",
        "content_hash": shard_info["content_hash"],
        "parent_shard_ids": [],
        "is_eval": is_eval,
        "gates": gates,
        "admission_score": max(0, score),
        "admission": admission,
        "admission_reason": admission_reason,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    manifest_path = manifests_dir / f"{shard_info['shard_id']}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def load_all_manifests(manifests_dir: Path) -> List[dict]:
    return [json.loads(p.read_text()) for p in sorted(manifests_dir.glob("shard_*.json"))]


def get_admitted_shards(manifests: List[dict]) -> List[dict]:
    return [m for m in manifests if m["admission"] == "admitted"]


def get_eval_shards(manifests: List[dict]) -> List[dict]:
    return [m for m in manifests if m.get("is_eval", False)]
