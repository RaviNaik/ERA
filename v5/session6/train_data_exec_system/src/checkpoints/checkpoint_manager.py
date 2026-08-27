"""
Checkpoint Manager - saves/loads full training state tied to ledger offset.
A checkpoint binds: model weights + optimizer state + RNG state + ledger_offset.
Crash recovery: load checkpoint, verify ledger_offset, resume from exactly next batch.
Fork: load earlier checkpoint, start new branch with new branch_id.
"""
from __future__ import annotations
import json, hashlib, time
from pathlib import Path
from typing import Optional
import numpy as np
from src.model.tiny_gpt import TinyGPT


class CheckpointManager:
    """Saves and loads deterministic checkpoints tied to ledger offsets."""

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._saved: list = []

    def save(self, step: int, model: TinyGPT, optimizer_lr: float,
             ledger_offset: int, run_id: str, branch_id: str = "main",
             rng_state: Optional[dict] = None) -> str:
        """Save a checkpoint. Returns checkpoint_id."""
        ckpt_id = f"ckpt_{run_id}_{branch_id}_step{step:05d}"
        ckpt_dir = self.checkpoint_dir / ckpt_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        state = model.state_dict()
        weights_hash = model.get_weights_hash()

        # Save weights
        np.savez_compressed(ckpt_dir / "weights.npz", **state)

        # Save metadata
        meta = {
            "checkpoint_id": ckpt_id,
            "step": step,
            "ledger_offset": ledger_offset,
            "run_id": run_id,
            "branch_id": branch_id,
            "optimizer_lr": optimizer_lr,
            "weights_hash": weights_hash,
            "rng_state": rng_state or {},
            "next_expected_batch_offset": ledger_offset,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (ckpt_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        self._saved.append(meta)
        return ckpt_id

    def load(self, checkpoint_id: str) -> dict:
        """Load a checkpoint. Returns (model_state, meta)."""
        ckpt_dir = self.checkpoint_dir / checkpoint_id
        meta = json.loads((ckpt_dir / "meta.json").read_text())
        weights = np.load(ckpt_dir / "weights.npz")
        state = {k: weights[k] for k in weights.files}
        state["_step"] = state.get("_step", np.array(meta["step"])).item()
        return {"state": state, "meta": meta}

    def list_checkpoints(self) -> list:
        ckpts = []
        for p in sorted(self.checkpoint_dir.iterdir()):
            meta_path = p / "meta.json"
            if meta_path.exists():
                ckpts.append(json.loads(meta_path.read_text()))
        return ckpts

    def get_latest(self) -> Optional[dict]:
        ckpts = self.list_checkpoints()
        return ckpts[-1] if ckpts else None

    def fork(self, checkpoint_id: str, new_run_id: str, new_branch_id: str) -> dict:
        """Fork from an earlier checkpoint — creates a new branch."""
        ckpt_data = self.load(checkpoint_id)
        meta = ckpt_data["meta"]
        fork_meta = {
            **meta,
            "run_id": new_run_id,
            "branch_id": new_branch_id,
            "forked_from": checkpoint_id,
            "forked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        return {"state": ckpt_data["state"], "meta": fork_meta}
