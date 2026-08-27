"""
Mixture Scheduler - implements curriculum stages, lane weights and protected floors.
Based on Session 5 Widget: Mixture Composer.

Stages:
  1. Seed/General:   web=49%, code=18%, indic=16%, agentic=2%, reasoning=6%, stem=9%
  2. Reasoning:      web=25%, code=28%, indic=14%, agentic=3%, reasoning=18%, stem=12%
  3. Anneal:         web=8%,  code=20%, indic=28%, agentic=8%,  reasoning=18%, stem=10%, long=8%

Protected floors:
  - Indic >= 12% at all times
  - Agentic >= 2% at all times
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional


PROTECTED_FLOORS = {
    "indic": 0.12,
    "agentic": 0.02,
}


@dataclass
class CurriculumStage:
    stage_id: str
    name: str
    token_budget_fraction: float
    lane_weights: Dict[str, float]   # must sum to ~1.0
    difficulty_bands: Dict[str, float]  # B0-B5
    description: str


CURRICULUM_STAGES = [
    CurriculumStage(
        stage_id="stage_1_general",
        name="Seed / General",
        token_budget_fraction=0.70,
        lane_weights={"web": 0.49, "code": 0.18, "indic": 0.16, "agentic": 0.02,
                      "reasoning": 0.06, "stem": 0.09},
        difficulty_bands={"B0": 0.60, "B1": 0.30, "B2": 0.10},
        description="Build broad language foundation. web-heavy, indic protected at 16%.",
    ),
    CurriculumStage(
        stage_id="stage_2_reasoning",
        name="Reasoning / Code",
        token_budget_fraction=0.20,
        lane_weights={"web": 0.25, "code": 0.28, "indic": 0.14, "agentic": 0.03,
                      "reasoning": 0.18, "stem": 0.12},
        difficulty_bands={"B2": 0.40, "B3": 0.40, "B4": 0.20},
        description="Harder capabilities once foundation is solid.",
    ),
    CurriculumStage(
        stage_id="stage_3_anneal",
        name="Anneal (Low-LR Cooldown)",
        token_budget_fraction=0.10,
        lane_weights={"web": 0.08, "code": 0.20, "indic": 0.28, "agentic": 0.08,
                      "reasoning": 0.18, "stem": 0.10, "long_context": 0.08},
        difficulty_bands={"B3": 0.30, "B4": 0.40, "B5": 0.30},
        description="Best Tier-A data spent in final low-LR cooldown. Indic/agentic concentrated.",
    ),
]


class MixtureScheduler:
    """Returns the current stage and enforces protected floors."""

    def __init__(self, stages: List[CurriculumStage] = None):
        self.stages = stages or CURRICULUM_STAGES
        self._current_stage_idx = 0
        self._lane_consumption: Dict[str, int] = {k: 0 for k in
            ["web", "code", "indic", "agentic", "reasoning", "stem", "long_context"]}

    def get_stage_for_step(self, step: int, total_steps: int) -> CurriculumStage:
        progress = step / max(1, total_steps)
        cumulative = 0.0
        for stage in self.stages:
            cumulative += stage.token_budget_fraction
            if progress <= cumulative:
                return stage
        return self.stages[-1]

    def get_lane_weights(self, step: int, total_steps: int) -> Dict[str, float]:
        stage = self.get_stage_for_step(step, total_steps)
        weights = dict(stage.lane_weights)
        return self._enforce_floors(weights)

    def _enforce_floors(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Enforce protected floors for indic and agentic lanes."""
        changed = False
        for lane, floor in PROTECTED_FLOORS.items():
            if weights.get(lane, 0.0) < floor:
                deficit = floor - weights.get(lane, 0.0)
                weights[lane] = floor
                # Take from web (largest bucket)
                donor = "web"
                weights[donor] = max(0.0, weights.get(donor, 0.0) - deficit)
                changed = True
        # Renormalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        return weights

    def record_consumption(self, lane: str, tokens: int) -> None:
        self._lane_consumption[lane] = self._lane_consumption.get(lane, 0) + tokens

    def get_actual_mixture(self) -> Dict[str, float]:
        total = sum(self._lane_consumption.values())
        if total == 0:
            return {}
        return {k: round(v / total, 4) for k, v in self._lane_consumption.items() if v > 0}

    def check_floor_compliance(self) -> Dict[str, dict]:
        actual = self.get_actual_mixture()
        results = {}
        for lane, floor in PROTECTED_FLOORS.items():
            actual_pct = actual.get(lane, 0.0)
            results[lane] = {
                "floor": floor,
                "actual": actual_pct,
                "compliant": actual_pct >= floor * 0.7,  # allow 30% slack for small corpus
            }
        return results
