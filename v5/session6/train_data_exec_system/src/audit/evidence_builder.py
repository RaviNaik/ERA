"""
Evidence Builder - generates the complete submission evidence bundle.
Produces: run.log, evidence.json, evidence.md, performance.json
All evidence is derived from actual run artifacts (not hardcoded).
"""
from __future__ import annotations
import json, time, hashlib
from pathlib import Path
from typing import List, Dict, Any


class EvidenceBuilder:
    """Collects evidence from run artifacts and writes the evidence bundle."""

    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = artifacts_dir
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._log_lines: List[str] = []
        self._evidence: Dict[str, Any] = {}

    def log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self._log_lines.append(line)
        try:
            print(line)
        except UnicodeEncodeError:
            print(line.encode('ascii', errors='replace').decode('ascii'))

    def record_pass(self, key: str, description: str, evidence_path: str) -> None:
        self._evidence[key] = {"result": "PASS", "description": description,
                                "evidence": evidence_path}
        self.log(f"[PASS] {key}: {description}")

    def record_fail(self, key: str, description: str, reason: str) -> None:
        self._evidence[key] = {"result": "FAIL", "description": description,
                                "reason": reason}
        self.log(f"[FAIL] {key}: {description} — {reason}")

    def add_section(self, title: str) -> None:
        self.log(f"\n{'='*60}")
        self.log(f"  {title}")
        self.log(f"{'='*60}")

    def write_run_log(self) -> Path:
        log_path = self.artifacts_dir / "run.log"
        log_path.write_text("\n".join(self._log_lines), encoding="utf-8")
        return log_path

    def write_evidence_json(self) -> Path:
        evidence_path = self.artifacts_dir / "evidence.json"
        evidence_path.write_text(json.dumps(self._evidence, indent=2), encoding="utf-8")
        return evidence_path

    def write_evidence_md(self) -> Path:
        lines = [
            "# Training Data Execution System — Evidence Bundle",
            "",
            f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
            "",
            "## Requirement Results",
            "",
            "| Requirement | Result | Evidence |",
            "|-------------|--------|----------|",
        ]
        labels = {
            "tokenizer_integrity": "Tokenizer integrity",
            "eval_firewall": "Evaluation firewall",
            "packing_correctness": "Packing correctness",
            "mixture_compliance": "Mixture compliance",
            "opus_audit_trail": "OPUS audit trail",
            "crash_recovery": "Crash recovery",
            "replay": "Replay hash match",
            "learning_trace": "Learning trace",
            "throughput": "Throughput & utilization",
        }
        for key, label in labels.items():
            e = self._evidence.get(key, {})
            result = e.get("result", "UNKNOWN")
            badge = "✅ PASS" if result == "PASS" else "❌ FAIL"
            evidence = e.get("evidence", e.get("reason", "—"))
            lines.append(f"| {label} | {badge} | {evidence} |")

        lines += [
            "",
            "## Log Excerpt",
            "",
            "```",
        ]
        # Add key PASS markers
        for line in self._log_lines:
            if "[PASS]" in line or "[FAIL]" in line or "===" in line or "[CHECKPOINT]" in line or "[CRASH]" in line or "[RESUME]" in line or "[REPLAY]" in line or "[FORK]" in line:
                lines.append(line)
        lines.append("```")

        md_path = self.artifacts_dir / "evidence.md"
        md_path.write_text("\n".join(lines), encoding="utf-8")
        return md_path

    def write_performance_json(self, learning_entries: List[dict],
                                packing_stats: dict) -> Path:
        if not learning_entries:
            avg_loss = 0.0
            avg_tps = 0.0
        else:
            avg_loss = sum(e["loss"] for e in learning_entries) / len(learning_entries)
            avg_tps = sum(e["tokens_per_sec"] for e in learning_entries) / len(learning_entries)

        perf = {
            "total_steps": len(learning_entries),
            "average_loss": round(avg_loss, 6),
            "final_loss": round(learning_entries[-1]["loss"], 6) if learning_entries else 0.0,
            "average_tokens_per_sec": round(avg_tps, 1),
            "packing_policy": packing_stats.get("policy", ""),
            "packing_utilization_pct": packing_stats.get("utilization_pct", 0),
            "useful_tokens_total": packing_stats.get("useful_tokens", 0),
            "num_packed_sequences": packing_stats.get("num_sequences", 0),
            "context_len": packing_stats.get("context_len", 0),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        perf_path = self.artifacts_dir / "performance.json"
        perf_path.write_text(json.dumps(perf, indent=2))
        return perf_path
