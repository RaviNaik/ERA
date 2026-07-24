"""
stats_tracker.py
----------------
Per-stage statistics collector and serializer.
Captures: doc counts, token counts, drop reasons, before/after examples, metadata keys added.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class StageStats:
    stage_id: int
    stage_name: str
    input_docs: int = 0
    output_docs: int = 0
    docs_dropped: int = 0
    drop_pct: float = 0.0
    drop_reasons: dict = field(default_factory=dict)   # reason -> count
    input_tokens: int = 0
    output_tokens: int = 0
    token_survival_pct: float = 0.0
    cumulative_survival_pct: float = 0.0
    processing_time_s: float = 0.0
    examples: list = field(default_factory=list)       # list of {before, after, note}
    metadata_added: list = field(default_factory=list) # list of field names
    extra: dict = field(default_factory=dict)          # stage-specific extras

    def finalize(self, initial_docs: int, initial_tokens: int):
        self.docs_dropped = self.input_docs - self.output_docs
        self.drop_pct = round(100.0 * self.docs_dropped / max(self.input_docs, 1), 2)
        self.token_survival_pct = round(100.0 * self.output_tokens / max(self.input_tokens, 1), 2)
        self.cumulative_survival_pct = round(100.0 * self.output_tokens / max(initial_tokens, 1), 2)

    def add_example(self, before: str, after: str, note: str = ""):
        if len(self.examples) < 5:
            self.examples.append({
                "before": before[:400],
                "after": after[:400],
                "note": note
            })

    def add_drop_reason(self, reason: str, count: int = 1):
        self.drop_reasons[reason] = self.drop_reasons.get(reason, 0) + count

    def to_dict(self) -> dict:
        return asdict(self)


class PipelineTracker:
    """Tracks statistics across all 8 pipeline stages for one dataset run."""

    def __init__(self, run_name: str, output_dir: Path):
        self.run_name = run_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stages: list[StageStats] = []
        self.initial_docs: int = 0
        self.initial_tokens: int = 0
        self._start_time: float = 0.0

    def begin_stage(self, stage_id: int, stage_name: str) -> StageStats:
        s = StageStats(stage_id=stage_id, stage_name=stage_name)
        self._start_time = time.time()
        return s

    def end_stage(self, stats: StageStats):
        stats.processing_time_s = round(time.time() - self._start_time, 2)
        stats.finalize(self.initial_docs, self.initial_tokens)
        self.stages.append(stats)
        # Save after every stage so we have incremental checkpoints
        self.save()
        print(
            f"  [Stage {stats.stage_id}] {stats.stage_name}: "
            f"{stats.input_docs:,} -> {stats.output_docs:,} docs "
            f"({stats.drop_pct:.1f}% dropped) | "
            f"tokens: {stats.output_tokens:,} ({stats.token_survival_pct:.1f}% survival) "
            f"| {stats.processing_time_s:.1f}s"
        )

    def save(self):
        out = {
            "run_name": self.run_name,
            "initial_docs": self.initial_docs,
            "initial_tokens": self.initial_tokens,
            "stages": [s.to_dict() for s in self.stages],
        }
        path = self.output_dir / f"{self.run_name}_stats.json"
        path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


def estimate_tokens(text: str) -> int:
    """Fast token-count estimate: split on whitespace, ~1.3x for subword inflation."""
    # For English: words * 1.3 (session notes warn this underestimates Indic by 10x!)
    return max(1, int(len(text.split()) * 1.3))


def count_tokens_batch(docs: list[str]) -> int:
    return sum(estimate_tokens(d) for d in docs)
