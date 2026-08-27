"""
Trainer - runs the training loop with consumption + learning ledger recording.
Simulates crash at a specific step. Supports resume from checkpoint.
"""
from __future__ import annotations
import time, hashlib
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from src.model.tiny_gpt import TinyGPT
from src.dataloader.deterministic_dataloader import DeterministicDataLoader, batch_hash
from src.ledgers.consumption_ledger import ConsumptionLedger
from src.ledgers.learning_ledger import LearningLedger
from src.checkpoints.checkpoint_manager import CheckpointManager
from src.mixture.mixture_scheduler import MixtureScheduler
from src.mixture.opus_selector import OPUSSelector


class SimulatedCrashError(Exception):
    """Deliberately raised to simulate a training crash."""
    pass


class Trainer:
    """Manages the full training loop with ledger, checkpoint, and crash support."""

    def __init__(self, model: TinyGPT, loader: DeterministicDataLoader,
                 consumption_ledger: ConsumptionLedger, learning_ledger: LearningLedger,
                 checkpoint_manager: CheckpointManager,
                 scheduler: MixtureScheduler, opus: OPUSSelector,
                 run_id: str = "run_A", branch_id: str = "main",
                 checkpoint_every: int = 5, crash_at_step: Optional[int] = None,
                 lr: float = 1e-3, total_steps: Optional[int] = None):
        self.model = model
        self.loader = loader
        self.consumption_ledger = consumption_ledger
        self.learning_ledger = learning_ledger
        self.checkpoint_manager = checkpoint_manager
        self.scheduler = scheduler
        self.opus = opus
        self.run_id = run_id
        self.branch_id = branch_id
        self.checkpoint_every = checkpoint_every
        self.crash_at_step = crash_at_step
        self.lr = lr
        self.total_steps = total_steps or len(loader)
        self._step = 0
        self._last_checkpoint_id: Optional[str] = None

    def run(self, start_step: int = 0, max_steps: Optional[int] = None,
            logger=None) -> dict:
        """Run training from start_step. Returns summary."""
        self._step = start_step
        self.loader.seek(start_step)
        # Respect self.total_steps as a real budget, not just a display value:
        # never run past it, and never run past what the loader can supply.
        remaining_by_budget = max(0, self.total_steps - start_step)
        n_steps = max_steps if max_steps is not None else min(
            len(self.loader) - start_step, remaining_by_budget)
        results = {"steps_run": 0, "final_loss": None, "crash": False,
                   "last_checkpoint": None, "batch_ids": []}
        active_stage_id = None

        all_batch_infos = [{
            "batch_id": bid,
            "lane": batch.lane,
            "token_count": batch.useful_tokens,
        } for bid, batch in self.loader._packed_batches]
        all_scores = [self.opus._proxy_score(bi) for bi in all_batch_infos]

        for _ in range(n_steps):
            if self._step >= len(self.loader):
                break

            # Simulate crash BEFORE processing
            if self.crash_at_step is not None and self._step == self.crash_at_step:
                if logger:
                    logger(f"[CRASH] Simulated crash at step {self._step}")
                results["crash"] = True
                results["crash_at_step"] = self._step
                raise SimulatedCrashError(f"Deliberate crash at step {self._step}")

            # Curriculum stage for this step drives OPUS's score (see
            # OPUSSelector._proxy_score) so lane weights genuinely affect what
            # gets accepted, not just what gets logged.
            stage = self.scheduler.get_stage_for_step(self._step, self.total_steps)
            stage_weights = self.scheduler.get_lane_weights(self._step, self.total_steps)
            if logger and stage.stage_id != active_stage_id:
                logger(f"  [STAGE] step={self._step:03d} entering {stage.stage_id} "
                       f"({stage.name})")
                active_stage_id = stage.stage_id

            batch_id, batch = self.loader.get_batch_at(self._step)
            bi = {"batch_id": batch_id, "lane": batch.lane, "token_count": batch.useful_tokens}
            decision, score, reason = self.opus.select_batch(
                bi, all_scores, step=self._step,
                stage_id=stage.stage_id, stage_weights=stage_weights)

            # Reject: permanently skip. Defer: also skip this cycle (boundary
            # cases aren't trained on now, distinct from accept) — only
            # accepted batches enter the consumption/learning ledgers. Either
            # way the checkpoint cadence below still runs on every dataloader
            # position, not just accepted ones — otherwise a reject/defer
            # landing exactly on a checkpoint_every boundary would silently
            # drop that checkpoint instead of just skipping training.
            if decision == "accept":
                # Forward pass
                t0 = time.perf_counter()
                targets = np.concatenate([batch.input_ids[1:], [0]])
                fwd = self.model.forward(batch.input_ids, batch.loss_mask, targets)
                elapsed = time.perf_counter() - t0
                loss = fwd.get("loss", 10.0)
                tps = batch.useful_tokens / max(elapsed, 1e-6)

                # Gradient update
                self.model.step_update(loss, self.lr)

                # Build per-doc loss map
                token_losses = fwd.get("token_losses", np.ones(len(batch.input_ids)) * loss)
                doc_loss_map = {}
                for doc_id, start, end in batch.doc_spans:
                    span_loss_mask = batch.loss_mask[start:end]
                    span_token_loss = token_losses[start:end]
                    bearing = span_loss_mask.sum()
                    if bearing > 0:
                        doc_loss_map[doc_id] = float((span_token_loss * span_loss_mask).sum() / bearing)

                # Record in ledgers
                lane_breakdown = {batch.lane: batch.useful_tokens}
                self.consumption_ledger.record(self._step, batch_id, batch, lane_breakdown)
                self.learning_ledger.record(
                    self._step, batch_id, loss,
                    fwd.get("loss_bearing_tokens", int(batch.loss_mask.sum())),
                    tps, doc_loss_map, batch.lane
                )
                self.scheduler.record_consumption(batch.lane, batch.useful_tokens)

                results["batch_ids"].append(batch_id)
                results["final_loss"] = loss
                results["steps_run"] += 1

                if logger:
                    logger(f"  step={self._step:03d} loss={loss:.4f} lane={batch.lane} "
                           f"tps={tps:.0f} opus={decision}")
            elif logger:
                logger(f"  step={self._step:03d} lane={batch.lane} opus={decision} "
                       f"(skipped: {reason})")

            # Checkpoint — fixed dataloader-position cadence, independent of
            # this step's accept/reject/defer outcome.
            if (self._step + 1) % self.checkpoint_every == 0:
                ckpt_id = self.checkpoint_manager.save(
                    step=self._step + 1,
                    model=self.model,
                    optimizer_lr=self.lr,
                    ledger_offset=self._step + 1,
                    run_id=self.run_id,
                    branch_id=self.branch_id,
                )
                self._last_checkpoint_id = ckpt_id
                results["last_checkpoint"] = ckpt_id
                if logger:
                    logger(f"  [CHECKPOINT] {ckpt_id} at ledger_offset={self._step + 1}")

            self._step += 1
            self.loader.seek(self._step)

        results["last_step"] = self._step
        return results
