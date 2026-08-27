"""
Shard Creator - tokenizes documents into immutable binary shards.
Each shard is content-addressed (SHA-256 of token bytes).
Shards are stored as .npz files and are IMMUTABLE once created.
"""
from __future__ import annotations
import hashlib, json, time
from pathlib import Path
from typing import List, Tuple
import numpy as np
from src.corpus.corpus_generator import Document
from src.tokenizer.frozen_tokenizer import FrozenTokenizer


def create_shard(doc: Document, tokenizer: FrozenTokenizer, shard_dir: Path) -> dict:
    """Tokenize a document and write an immutable shard (.npz)."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    token_ids = tokenizer.encode(doc.text)
    token_arr = np.array(token_ids, dtype=np.int32)
    token_bytes = token_arr.tobytes()
    content_hash = "sha256_" + hashlib.sha256(token_bytes).hexdigest()
    shard_id = "shard_" + hashlib.sha256(token_bytes).hexdigest()[:12]
    shard_path = shard_dir / f"{shard_id}.npz"
    if not shard_path.exists():
        np.savez_compressed(shard_path, token_ids=token_arr)
    return {
        "shard_id": shard_id,
        "shard_path": str(shard_path),
        "doc_id": doc.doc_id,
        "lane": doc.lane,
        "language": doc.language,
        "script": doc.script,
        "license_tier": doc.license_tier,
        "source_url": doc.source_url,
        "capability_tags": doc.capability_tags,
        "is_eval": doc.is_eval,
        "token_count": len(token_ids),
        "content_hash": content_hash,
        "tokenizer_sha": tokenizer.tokenizer_sha,
    }


def load_shard_tokens(shard_path: str) -> np.ndarray:
    data = np.load(shard_path)
    return data["token_ids"]


def verify_shard_integrity(shard_path: str, expected_hash: str) -> bool:
    arr = load_shard_tokens(shard_path)
    actual = "sha256_" + hashlib.sha256(arr.tobytes()).hexdigest()
    return actual == expected_hash
