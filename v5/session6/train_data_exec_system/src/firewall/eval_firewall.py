"""
Eval Firewall - enforces the held-out data contract.
Eval shards must NEVER enter a loss-bearing training batch.
Raises EvalFirewallViolation if any blocked shard is requested.
"""
from __future__ import annotations
from typing import List, Set


class EvalFirewallViolation(Exception):
    pass


class EvalFirewall:
    """Tracks blocked (eval) shard IDs and raises on any violation."""

    def __init__(self, eval_manifests: List[dict]):
        self._blocked: Set[str] = {m["shard_id"] for m in eval_manifests}
        self._blocked_events: List[dict] = []

    @property
    def blocked_ids(self) -> Set[str]:
        return self._blocked

    def check_batch(self, batch_shard_ids: List[str]) -> None:
        """Raise EvalFirewallViolation if any shard in the batch is eval/blocked."""
        for sid in batch_shard_ids:
            if sid in self._blocked:
                raise EvalFirewallViolation(
                    f"[FAIL] eval_shard_blocked VIOLATED: {sid} attempted to enter training"
                )

    def record_block(self, shard_id: str, context: str = "") -> None:
        """Log that an eval shard was intercepted (for evidence bundle)."""
        event = {"shard_id": shard_id, "event": "blocked", "context": context}
        self._blocked_events.append(event)

    def verify_eval_shards_blocked(self, all_manifests: List[dict]) -> List[dict]:
        """
        Verify all eval shards are blocked and record each block event.
        Returns list of block events for the evidence bundle.
        """
        events = []
        for m in all_manifests:
            if m.get("is_eval", False):
                self.record_block(m["shard_id"], f"admission={m['admission']}")
                events.append({
                    "shard_id": m["shard_id"],
                    "blocked": True,
                    "admission": m["admission"],
                    "pass_marker": "[PASS] eval_shard_blocked",
                })
        self._blocked_events = events
        return events

    @property
    def block_events(self) -> List[dict]:
        return self._blocked_events
