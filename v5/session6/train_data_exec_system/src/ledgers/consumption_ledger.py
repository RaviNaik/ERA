"""
Consumption Ledger - records what data was fed at each training step.
Stored as JSONL (appendable). Ledger offset = current step number.
"""
from __future__ import annotations
import json, time
from pathlib import Path
from typing import Dict, List, Optional


class ConsumptionLedger:
    """
    Tracks every batch fed to the model.
    Each entry: step, batch_id, shard_ids, doc_ids, token_count, lane_breakdown.
    The ledger_offset is the single source of truth for checkpoint resumption.
    """

    def __init__(self, ledger_path: Path):
        self.ledger_path = ledger_path
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: List[dict] = []
        self._offset = 0
        # Load existing if present
        if ledger_path.exists():
            for line in ledger_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self._entries.append(json.loads(line))
            self._offset = len(self._entries)

    @property
    def ledger_offset(self) -> int:
        return self._offset

    def record(self, step: int, batch_id: str, batch, doc_lanes: Dict[str, int]) -> None:
        """Record a batch consumption event."""
        from src.packing.packer import PackedBatch
        entry = {
            "step": step,
            "batch_id": batch_id,
            "shard_ids": batch.shard_ids,
            "doc_ids": [span[0] for span in batch.doc_spans],
            "token_count": batch.context_len if hasattr(batch, "context_len") else int(batch.loss_mask.sum() + batch.pad_count),
            "useful_tokens": batch.useful_tokens,
            "loss_bearing_tokens": int(batch.loss_mask.sum()),
            "lane": batch.lane,
            "lane_breakdown": doc_lanes,
            "policy": batch.policy,
            "utilization": round(batch.utilization, 4),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self._entries.append(entry)
        self._offset = len(self._entries)
        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def get_entry(self, step: int) -> Optional[dict]:
        for e in self._entries:
            if e["step"] == step:
                return e
        return None

    def get_all(self) -> List[dict]:
        return list(self._entries)

    def truncate_to(self, offset: int) -> None:
        """Truncate ledger to a given offset (for crash recovery)."""
        self._entries = self._entries[:offset]
        self._offset = len(self._entries)
        with open(self.ledger_path, "w", encoding="utf-8") as f:
            for entry in self._entries:
                f.write(json.dumps(entry) + "\n")
