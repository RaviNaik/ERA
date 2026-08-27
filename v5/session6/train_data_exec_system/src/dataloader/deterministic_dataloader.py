"""
Deterministic Dataloader - ledger-offset based batch iterator.
Same seed + same corpus = same batch sequence every time.
Supports seek(offset), replay, and fork via re-seeding.
"""
from __future__ import annotations
import hashlib, json
from typing import List, Optional, Tuple
import numpy as np
from src.packing.packer import DocToken, PackedBatch, Packer
from src.manifests.manifest_builder import get_admitted_shards
from src.shards.shard_creator import load_shard_tokens
from src.tokenizer.frozen_tokenizer import FrozenTokenizer
from src.firewall.eval_firewall import EvalFirewall


def batch_hash(batch: PackedBatch) -> str:
    """Deterministic hash of a packed batch for replay verification."""
    return "bh_" + hashlib.sha256(batch.input_ids.tobytes()).hexdigest()[:16]


class DeterministicDataLoader:
    """
    Deterministic dataloader backed by a manifest registry.
    The ledger_offset tracks the exact position in the data stream.
    """

    def __init__(self, manifests: List[dict], tokenizer: FrozenTokenizer,
                 firewall: EvalFirewall, context_len: int = 64,
                 policy: str = "structure_preserving",
                 seed: int = 42, run_id: str = "run_A"):
        self.manifests = get_admitted_shards(manifests)   # only admitted shards
        self.tokenizer = tokenizer
        self.firewall = firewall
        self.context_len = context_len
        self.policy = policy
        self.seed = seed
        self.run_id = run_id
        self._ledger_offset = 0
        self._batch_order: List[int] = []   # shard indices in shuffled order
        self._packer = Packer(context_len=context_len, policy=policy)
        self._packed_batches: List[Tuple[str, PackedBatch]] = []
        self._prepare()

    def _prepare(self) -> None:
        """Tokenize all admitted shards and pack into batches."""
        rng = np.random.default_rng(self.seed)
        indices = list(range(len(self.manifests)))
        rng.shuffle(indices)

        doc_tokens: List[DocToken] = []
        for idx in indices:
            m = self.manifests[idx]
            tokens = load_shard_tokens(m["shard_path"])
            is_agentic = m["capability_lane"] == "agentic"
            masked_spans = []
            if is_agentic:
                from src.packing.packer import extract_agentic_masked_spans
                masked_spans = extract_agentic_masked_spans(tokens.tolist(), self.tokenizer)
            doc_tokens.append(DocToken(
                doc_id=m["document_ids"][0],
                shard_id=m["shard_id"],
                lane=m["capability_lane"],
                token_ids=tokens.tolist(),
                is_agentic=is_agentic,
                masked_spans=masked_spans,
            ))

        batches = self._packer.pack(doc_tokens)
        # Assign batch IDs deterministically
        self._packed_batches = []
        for i, b in enumerate(batches):
            batch_id = f"{self.run_id}_batch_{i:04d}"
            b.sequence_id = batch_id
            self._packed_batches.append((batch_id, b))

    def __len__(self) -> int:
        return len(self._packed_batches)

    @property
    def ledger_offset(self) -> int:
        return self._ledger_offset

    def seek(self, offset: int) -> None:
        """Seek to a specific ledger offset (for resume)."""
        assert 0 <= offset <= len(self._packed_batches), f"offset {offset} out of range"
        self._ledger_offset = offset

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[str, PackedBatch]:
        if self._ledger_offset >= len(self._packed_batches):
            raise StopIteration
        batch_id, batch = self._packed_batches[self._ledger_offset]
        # Firewall check
        self.firewall.check_batch(batch.shard_ids)
        self._ledger_offset += 1
        return batch_id, batch

    def get_batch_at(self, offset: int) -> Tuple[str, PackedBatch]:
        """Get the batch at a specific offset without advancing the cursor."""
        batch_id, batch = self._packed_batches[offset]
        self.firewall.check_batch(batch.shard_ids)
        return batch_id, batch

    def get_batch_hash(self, offset: int) -> str:
        _, b = self._packed_batches[offset]
        return batch_hash(b)

    @property
    def all_batch_ids(self) -> List[str]:
        return [bid for bid, _ in self._packed_batches]
