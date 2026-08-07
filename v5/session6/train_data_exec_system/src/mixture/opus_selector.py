"""
OPUS Selector - Optimizer-induced Projected Utility Selection.
Per-iteration dynamic data selection: accept/reject/defer candidates.
Keep-fraction default 40% - selects highest-utility batches.
Protected-floor override: force-accepts best indic/agentic if below floor.
"""
from __future__ import annotations
import hashlib, json, time
from typing import Dict, List, Optional, Tuple
import numpy as np
from src.mixture.mixture_scheduler import PROTECTED_FLOORS


KEEP_FRACTION = 0.40
DEFER_FRACTION = 0.15   # borderline cases


class OPUSSelector:
    """
    Simulates OPUS selection using a proxy score (avg token entropy estimate).
    In production: scores updates by projecting onto a stable proxy direction.
    Here: scores are simulated based on lane utility and batch diversity.
    """

    def __init__(self, proxy_mix: str = "balanced", keep_fraction: float = KEEP_FRACTION):
        self.proxy_mix = proxy_mix
        self.keep_fraction = keep_fraction
        self._decisions: List[dict] = []
        self._step = 0
        # Per-lane token budget tracking for floor enforcement
        self._lane_tokens_accepted: Dict[str, int] = {}
        self._total_tokens_accepted = 0

    def _proxy_score(self, batch_info: dict, stage_weights: Optional[Dict[str, float]] = None) -> float:
        """
        Simulate OPUS utility score. Higher = more useful for the proxy direction.
        Balanced proxy: all lanes equally weighted.
        English-heavy proxy: penalizes indic/agentic.

        When `stage_weights` (the current curriculum stage's lane weights, from
        MixtureScheduler.get_lane_weights) is supplied, the score is nudged by
        how that lane is weighted *this stage* relative to an even split —
        this is what makes the curriculum schedule actually influence which
        batches get selected, instead of only being logged for display.
        """
        lane = batch_info.get("lane", "web")
        lane_scores = {
            "web": 0.55,
            "code": 0.75,
            "indic": 0.65 if self.proxy_mix == "balanced" else 0.30,
            "agentic": 0.70 if self.proxy_mix == "balanced" else 0.25,
            "reasoning": 0.80,
            "stem": 0.72,
        }
        base = lane_scores.get(lane, 0.5)
        # Add noise for realism
        rng = np.random.default_rng(hash(batch_info.get("batch_id", "x")) % 2**32)
        noise = rng.uniform(-0.05, 0.05)
        score = base + noise

        if stage_weights:
            avg_weight = 1.0 / max(1, len(stage_weights))
            lane_weight = stage_weights.get(lane, 0.0)
            # Above-average weight this stage -> multiplier > 1 (boosts score);
            # near-zero weight this stage -> multiplier towards 0.5 (suppresses it).
            multiplier = float(np.clip(lane_weight / avg_weight, 0.5, 1.5)) if avg_weight else 1.0
            score *= multiplier

        return float(np.clip(score, 0.0, 1.0))

    def _floor_override_needed(self, lane: str) -> bool:
        """Check if a lane is below protected floor and needs override."""
        floor = PROTECTED_FLOORS.get(lane, 0.0)
        if floor == 0.0 or self._total_tokens_accepted == 0:
            return False
        actual = self._lane_tokens_accepted.get(lane, 0) / self._total_tokens_accepted
        return actual < floor

    def select_batch(self, batch_info: dict, all_scores: List[float],
                      step: Optional[int] = None,
                      stage_id: Optional[str] = None,
                      stage_weights: Optional[Dict[str, float]] = None) -> Tuple[str, float, str]:
        """
        Decide: accept / reject / defer for one candidate batch.
        Returns (decision, score, reason).

        `step` should be the trainer's absolute training step (not an internal
        counter) so the persisted decision log stays correctly numbered across
        a crash/resume boundary, where a fresh OPUSSelector instance is used.
        `stage_id`/`stage_weights` come from MixtureScheduler and are recorded
        on the decision for audit, and `stage_weights` also feeds the score
        (see `_proxy_score`) so the curriculum stage has a real, observable
        effect on what gets accepted.
        """
        score = self._proxy_score(batch_info, stage_weights)
        lane = batch_info.get("lane", "web")
        token_count = batch_info.get("token_count", 0)

        # Protected floor override: force-accept if lane is below floor
        if self._floor_override_needed(lane):
            decision = "accept"
            reason = f"protected_floor_override: {lane} below {PROTECTED_FLOORS.get(lane, 0.0):.0%}"
        else:
            # Rank-based selection: keep top keep_fraction
            sorted_scores = sorted(all_scores, reverse=True)
            threshold_idx = max(0, int(len(sorted_scores) * self.keep_fraction) - 1)
            threshold = sorted_scores[threshold_idx] if sorted_scores else 0.5

            defer_threshold = threshold * (1 - DEFER_FRACTION)
            if score >= threshold:
                decision = "accept"
                reason = f"score={score:.3f} >= threshold={threshold:.3f}"
            elif score >= defer_threshold:
                decision = "defer"
                reason = f"score={score:.3f} in defer zone [{defer_threshold:.3f},{threshold:.3f})"
            else:
                decision = "reject"
                reason = f"score={score:.3f} < defer_threshold={defer_threshold:.3f}"

        # Update lane tracking if accepted
        if decision == "accept":
            self._lane_tokens_accepted[lane] = self._lane_tokens_accepted.get(lane, 0) + token_count
            self._total_tokens_accepted += token_count

        effective_step = step if step is not None else self._step
        record = {
            "step": effective_step,
            "batch_id": batch_info.get("batch_id", f"batch_{effective_step:04d}"),
            "lane": lane,
            "decision": decision,
            "score": round(score, 4),
            "reason": reason,
            "token_count": token_count,
            "stage_id": stage_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self._decisions.append(record)
        self._step += 1
        return decision, score, reason

    def get_decisions(self) -> List[dict]:
        return self._decisions

    def get_summary(self) -> dict:
        return summarize_decisions(self._decisions, self.keep_fraction)


def summarize_decisions(decisions: List[dict], keep_fraction_target: float) -> dict:
    """
    Summarize a list of OPUS decision records (as produced by `select_batch`).
    Shared by `OPUSSelector.get_summary()` and by callers that need to merge
    decisions from multiple selector instances — e.g. run_demo.py combines
    the pre-crash and post-resume selectors' decisions into one audit trail,
    since a fresh OPUSSelector is created for the resumed run.
    """
    total = len(decisions)
    if total == 0:
        return {"total": 0}
    accepted = sum(1 for d in decisions if d["decision"] == "accept")
    rejected = sum(1 for d in decisions if d["decision"] == "reject")
    deferred = sum(1 for d in decisions if d["decision"] == "defer")
    overrides = sum(1 for d in decisions if "floor_override" in d["reason"])
    return {
        "total_candidates": total,
        "accepted": accepted,
        "rejected": rejected,
        "deferred": deferred,
        "floor_overrides": overrides,
        "keep_fraction_target": keep_fraction_target,
        "keep_fraction_actual": round(accepted / total, 3) if total else 0,
        "effective_token_multiplier": round(1 / max(0.01, accepted / total), 2) if accepted else 0,
    }
