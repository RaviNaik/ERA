"""
Packing Engine - implements all 5 packing policies from Session 6 Widget 5.
Produces input_ids, loss_mask, attention_mask, position_ids per packed batch.

Policies:
  1. pad_each_doc          - one doc per sequence, padded  (~42% util)
  2. concat_and_chop       - concatenate + chop at boundary (~70% util, high risk)
  3. greedy_pack           - greedy bin-packing            (~84% util, medium risk)
  4. best_fit_pack         - sorted bin-packing            (~84% util, medium risk)
  5. structure_preserving  - greedy + intra-sequence masks (~84% util, LOW risk)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


PAD_TOKEN_ID = 0
EOS_TOKEN_ID = 50256  # cl100k_base EOS


@dataclass
class PackedBatch:
    """One packed training sequence with all required tensors."""
    sequence_id: str
    input_ids: np.ndarray          # [seq_len]
    loss_mask: np.ndarray          # [seq_len] float32: 1=loss, 0=no-loss
    attention_mask: np.ndarray     # [seq_len, seq_len] bool causal mask
    position_ids: np.ndarray       # [seq_len]
    doc_spans: List[Tuple[str, int, int]]   # (doc_id, start, end)
    shard_ids: List[str]
    lane: str
    policy: str
    pad_count: int
    useful_tokens: int
    utilization: float


@dataclass
class DocToken:
    """A tokenized document ready for packing."""
    doc_id: str
    shard_id: str
    lane: str
    token_ids: List[int]
    is_agentic: bool = False
    # For agentic: spans where loss should be 0 (tool observations)
    masked_spans: List[Tuple[int, int]] = field(default_factory=list)


def _build_masks(token_ids: List[int], context_len: int, doc_spans: List[Tuple[str, int, int]],
                 loss_override_zeros: List[Tuple[int, int]] = None,
                 structure_preserving: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build all required tensors for a packed sequence."""
    seq_len = context_len
    ids = np.array(token_ids[:seq_len], dtype=np.int32)
    actual_len = len(ids)

    # Pad to context_len
    pad_count = seq_len - actual_len
    if pad_count > 0:
        ids = np.concatenate([ids, np.full(pad_count, PAD_TOKEN_ID, dtype=np.int32)])

    # Loss mask: 1 for all real tokens, 0 for pad
    lm = np.ones(seq_len, dtype=np.float32)
    lm[actual_len:] = 0.0

    # Zero out loss for agentic tool observations
    if loss_override_zeros:
        for start, end in loss_override_zeros:
            end = min(end, actual_len)
            if start < end:
                lm[start:end] = 0.0

    # Attention mask: causal within each doc span, no cross-doc attention
    attn = np.zeros((seq_len, seq_len), dtype=bool)
    if structure_preserving:
        # Each doc only attends to its own tokens
        for _, dstart, dend in doc_spans:
            for i in range(dstart, min(dend, actual_len)):
                attn[i, dstart:i+1] = True
    else:
        # Standard causal mask for all real tokens
        for i in range(actual_len):
            attn[i, :i+1] = True

    # Position IDs: reset per doc in structure_preserving mode
    pos = np.zeros(seq_len, dtype=np.int32)
    if structure_preserving:
        for _, dstart, dend in doc_spans:
            for i in range(dstart, min(dend, actual_len)):
                pos[i] = i - dstart
    else:
        pos[:actual_len] = np.arange(actual_len, dtype=np.int32)

    return ids, lm, attn, pos


def _pack_docs_into_sequence(docs: List[DocToken], seq_id: str, context_len: int,
                               policy: str) -> PackedBatch:
    """Pack a list of DocTokens into a single sequence."""
    token_ids = []
    doc_spans = []
    shard_ids = []
    loss_zeros = []
    lane = docs[0].lane if docs else "web"

    for doc in docs:
        start = len(token_ids)
        doc_toks = doc.token_ids[:context_len - len(token_ids) - 1]
        token_ids.extend(doc_toks)
        token_ids.append(EOS_TOKEN_ID)  # EOS boundary
        end = len(token_ids)
        doc_spans.append((doc.doc_id, start, end))
        shard_ids.append(doc.shard_id)
        # Remap agentic masked spans
        if doc.is_agentic and doc.masked_spans:
            for ms, me in doc.masked_spans:
                loss_zeros.append((start + ms, start + me))
        if len(token_ids) >= context_len:
            break

    struct = policy == "structure_preserving"
    ids, lm, attn, pos = _build_masks(token_ids, context_len, doc_spans, loss_zeros, struct)

    actual_useful = int(lm.sum())
    pad_count = context_len - len([t for t in token_ids if t != PAD_TOKEN_ID])
    util = actual_useful / context_len

    return PackedBatch(
        sequence_id=seq_id,
        input_ids=ids,
        loss_mask=lm,
        attention_mask=attn,
        position_ids=pos,
        doc_spans=doc_spans,
        shard_ids=shard_ids,
        lane=lane,
        policy=policy,
        pad_count=max(0, context_len - len([t for t in token_ids])),
        useful_tokens=actual_useful,
        utilization=util,
    )


class Packer:
    """Implements all 5 packing policies."""

    def __init__(self, context_len: int = 64, policy: str = "structure_preserving"):
        self.context_len = context_len
        self.policy = policy

    def pack(self, docs: List[DocToken]) -> List[PackedBatch]:
        if self.policy == "pad_each_doc":
            return self._pad_each_doc(docs)
        elif self.policy == "concat_and_chop":
            return self._concat_and_chop(docs)
        elif self.policy == "greedy_pack":
            return self._greedy_pack(docs, policy="greedy_pack")
        elif self.policy == "best_fit_pack":
            return self._best_fit_pack(docs)
        elif self.policy == "structure_preserving":
            return self._greedy_pack(docs, policy="structure_preserving")
        else:
            raise ValueError(f"Unknown policy: {self.policy}")

    def _pad_each_doc(self, docs: List[DocToken]) -> List[PackedBatch]:
        batches = []
        for i, doc in enumerate(docs):
            batches.append(_pack_docs_into_sequence([doc], f"seq_{i:04d}", self.context_len, "pad_each_doc"))
        return batches

    def _concat_and_chop(self, docs: List[DocToken]) -> List[PackedBatch]:
        all_tokens = []
        all_doc_spans = []
        all_shard_ids = []
        for doc in docs:
            start = len(all_tokens)
            all_tokens.extend(doc.token_ids)
            all_tokens.append(EOS_TOKEN_ID)
            all_doc_spans.append((doc.doc_id, start, len(all_tokens), doc.shard_id))

        batches = []
        seq_idx = 0
        for chunk_start in range(0, len(all_tokens), self.context_len):
            chunk_end = chunk_start + self.context_len
            chunk = all_tokens[chunk_start:chunk_end]
            if not chunk:
                break
            spans = [(did, max(0, s - chunk_start), min(self.context_len, e - chunk_start), sid)
                     for did, s, e, sid in all_doc_spans
                     if s < chunk_end and e > chunk_start]
            doc_spans_norm = [(did, s, e) for did, s, e, _ in spans]
            shard_ids_norm = [sid for _, _, _, sid in spans]

            ids, lm, attn, pos = _build_masks(chunk, self.context_len, doc_spans_norm,
                                               structure_preserving=False)
            useful = int(lm.sum())
            batches.append(PackedBatch(
                sequence_id=f"seq_{seq_idx:04d}", input_ids=ids, loss_mask=lm,
                attention_mask=attn, position_ids=pos, doc_spans=doc_spans_norm,
                shard_ids=shard_ids_norm, lane="mixed", policy="concat_and_chop",
                pad_count=self.context_len - len(chunk), useful_tokens=useful,
                utilization=useful / self.context_len,
            ))
            seq_idx += 1
        return batches

    def _greedy_pack(self, docs: List[DocToken], policy: str) -> List[PackedBatch]:
        bins: List[List[DocToken]] = []
        bin_usage: List[int] = []
        for doc in docs:
            doc_len = len(doc.token_ids) + 1  # +1 for EOS
            placed = False
            for i, (b, used) in enumerate(zip(bins, bin_usage)):
                if used + doc_len <= self.context_len:
                    b.append(doc)
                    bin_usage[i] += doc_len
                    placed = True
                    break
            if not placed:
                bins.append([doc])
                bin_usage.append(doc_len)

        return [_pack_docs_into_sequence(b, f"seq_{i:04d}", self.context_len, policy)
                for i, b in enumerate(bins)]

    def _best_fit_pack(self, docs: List[DocToken]) -> List[PackedBatch]:
        sorted_docs = sorted(docs, key=lambda d: len(d.token_ids), reverse=True)
        bins: List[List[DocToken]] = []
        bin_usage: List[int] = []
        for doc in sorted_docs:
            doc_len = len(doc.token_ids) + 1
            best_i, best_rem = None, self.context_len + 1
            for i, (b, used) in enumerate(zip(bins, bin_usage)):
                rem = self.context_len - used
                if rem >= doc_len and rem < best_rem:
                    best_i, best_rem = i, rem
            if best_i is not None:
                bins[best_i].append(doc)
                bin_usage[best_i] += doc_len
            else:
                bins.append([doc])
                bin_usage.append(doc_len)

        return [_pack_docs_into_sequence(b, f"seq_{i:04d}", self.context_len, "best_fit_pack")
                for i, b in enumerate(bins)]


def compute_packing_stats(batches: List[PackedBatch], context_len: int) -> dict:
    total_positions = len(batches) * context_len
    useful = sum(b.useful_tokens for b in batches)
    avg_lb = float(sum(float(b.loss_mask.sum()) for b in batches)) / max(1, len(batches))
    return {
        "policy": batches[0].policy if batches else "none",
        "num_sequences": len(batches),
        "context_len": context_len,
        "total_positions": int(total_positions),
        "useful_tokens": int(useful),
        "utilization_pct": round(100.0 * useful / total_positions, 2) if total_positions else 0.0,
        "avg_loss_bearing": round(avg_lb, 2),
    }


def extract_agentic_masked_spans(token_ids: List[int], tokenizer) -> List[Tuple[int, int]]:
    """
    Find [OBSERVATION] ... [/OBSERVATION] spans and mark them as loss=0.
    This prevents the model from learning to reproduce tool results.
    """
    text = tokenizer.decode(token_ids)
    spans = []
    import re
    for m in re.finditer(r'\[OBSERVATION\].*?\[/OBSERVATION\]', text, re.DOTALL):
        # Find token positions corresponding to this text span
        pre_text = text[:m.start()]
        pre_tokens = tokenizer.encode(pre_text)
        span_tokens = tokenizer.encode(m.group())
        spans.append((len(pre_tokens), len(pre_tokens) + len(span_tokens)))
    return spans
