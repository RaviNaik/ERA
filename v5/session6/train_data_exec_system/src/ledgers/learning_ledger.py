"""
Learning Ledger - records what the model learned from each batch.
Tracks loss per step, tokens-per-second, and per-doc loss attribution.
This is the "token-level or sample-level loss tracking" requirement.
"""
from __future__ import annotations
import json, time
from pathlib import Path
from typing import Dict, List, Optional


class LearningLedger:
    """
    Records the learning signal from each training step.
    Links loss values back to source documents for full auditability.
    """

    def __init__(self, ledger_path: Path):
        self.ledger_path = ledger_path
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: List[dict] = []
        if ledger_path.exists():
            for line in ledger_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self._entries.append(json.loads(line))

    def record(self, step: int, batch_id: str, loss: float,
               loss_bearing_tokens: int, tokens_per_sec: float,
               doc_loss_map: Dict[str, float], lane: str) -> None:
        """Record per-step learning metrics."""
        entry = {
            "step": step,
            "batch_id": batch_id,
            "loss": round(loss, 6),
            "loss_bearing_tokens": loss_bearing_tokens,
            "tokens_per_sec": round(tokens_per_sec, 1),
            "perplexity": round(2.718281828 ** loss, 4) if loss < 20 else 9999.0,
            "lane": lane,
            "doc_loss_map": {k: round(v, 6) for k, v in doc_loss_map.items()},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self._entries.append(entry)
        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def get_all(self) -> List[dict]:
        return list(self._entries)

    def average_loss(self) -> float:
        if not self._entries:
            return 0.0
        return sum(e["loss"] for e in self._entries) / len(self._entries)

    def get_doc_loss_summary(self) -> Dict[str, List[float]]:
        """Aggregate per-doc losses across all steps."""
        summary: Dict[str, List[float]] = {}
        for entry in self._entries:
            for doc_id, loss in entry.get("doc_loss_map", {}).items():
                summary.setdefault(doc_id, []).append(loss)
        return summary

    def truncate_to(self, n_entries: int) -> None:
        self._entries = self._entries[:n_entries]
        with open(self.ledger_path, "w", encoding="utf-8") as f:
            for entry in self._entries:
                f.write(json.dumps(entry) + "\n")
