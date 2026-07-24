"""
stage5_dedup.py -- Stage 5: Global Deduplication
-------------------------------------------------
Implements MinHash LSH deduplication from Session 4 notes.

Algorithm:
  1. Shingling: break text into overlapping word n-grams (k=5)
  2. MinHash: N=128 hash permutations, each computes min(hash(shingle))
  3. LSH Banding: b bands x r rows, candidate if signature matches in >=1 band
  4. Jaccard threshold: 0.80 (tunable)

For Indic scripts: normalize to NFC before hashing (prevents different
Unicode representations of same text from being missed).

Memory note: For 100M docs with 256 perms -> 100GB RAM needed.
Our demo runs on ~40K docs, tractable in < 2GB.
"""

import re
import hashlib
import unicodedata
from datasketch import MinHash, MinHashLSH
from stats_tracker import StageStats, estimate_tokens

# Session 4 Widget 5 parameters
N_PERMUTATIONS = 128
SHINGLE_SIZE   = 5        # word n-gram size
JACCARD_THRESH = 0.80     # Jaccard similarity threshold for near-dups


def text_to_shingles(text: str, k: int = SHINGLE_SIZE) -> set[str]:
    """Break text into overlapping word k-grams (shingles)."""
    # NFC normalize before hashing (critical for Indic)
    text = unicodedata.normalize("NFC", text.lower())
    words = text.split()
    if len(words) < k:
        # Short doc: use character 4-grams instead
        return {text[i:i+4] for i in range(max(1, len(text)-3))}
    return {" ".join(words[i:i+k]) for i in range(len(words) - k + 1)}


def build_minhash(shingles: set[str], num_perm: int = N_PERMUTATIONS) -> MinHash:
    """Create a MinHash signature from a set of shingles."""
    m = MinHash(num_perm=num_perm)
    for s in shingles:
        m.update(s.encode("utf-8"))
    return m


def exact_hash(text: str) -> str:
    """SHA-256 hash for exact duplicate detection (stage before fuzzy)."""
    return hashlib.sha256(unicodedata.normalize("NFC", text).encode("utf-8")).hexdigest()


def run_stage5(docs: list[dict], stats: StageStats) -> list[dict]:
    """
    Two-pass deduplication:
      Pass 1: Exact deduplication via SHA-256 hash
      Pass 2: Near-duplicate detection via MinHash LSH
    """
    stats.input_docs = len(docs)
    stats.input_tokens = sum(estimate_tokens(d["text"]) for d in docs)
    stats.metadata_added = ["content_hash", "is_near_dup", "dup_of"]

    # --- Pass 1: Exact deduplication ---
    seen_hashes: dict[str, str] = {}
    after_exact = []
    exact_dups = 0

    for doc in docs:
        h = exact_hash(doc["text"])
        if h in seen_hashes:
            exact_dups += 1
            stats.add_drop_reason("exact_duplicate")
        else:
            seen_hashes[h] = doc.get("url", doc.get("id", ""))
            doc = dict(doc)
            doc["content_hash"] = h
            after_exact.append(doc)

    print(f"    Exact dedup: {len(docs):,} → {len(after_exact):,} "
          f"({exact_dups:,} exact duplicates removed)")

    # --- Pass 2: MinHash LSH near-duplicate detection ---
    lsh = MinHashLSH(threshold=JACCARD_THRESH, num_perm=N_PERMUTATIONS)
    minhashes = {}

    # Build LSH index
    for i, doc in enumerate(after_exact):
        shingles = text_to_shingles(doc["text"])
        m = build_minhash(shingles)
        minhashes[i] = m
        try:
            lsh.insert(str(i), m)
        except ValueError:
            pass  # already inserted (shouldn't happen)

    # Query for near-duplicates
    is_dup = set()
    near_dup_count = 0

    for i, doc in enumerate(after_exact):
        if i in is_dup:
            continue
        result = lsh.query(minhashes[i])
        for j_str in result:
            j = int(j_str)
            if j != i and j not in is_dup:
                is_dup.add(j)  # mark later doc as dup, keep earlier one
                near_dup_count += 1
                stats.add_drop_reason("near_duplicate_minhash")

    output = []
    examples_added = 0
    for i, doc in enumerate(after_exact):
        if i not in is_dup:
            doc = dict(doc)
            doc["is_near_dup"] = False
            output.append(doc)
        else:
            if examples_added < 3:
                stats.add_example(
                    before=doc["text"][:200],
                    after="[NEAR-DUPLICATE REMOVED]",
                    note=f"Jaccard >= {JACCARD_THRESH} with an earlier document"
                )
                examples_added += 1

    stats.output_docs = len(output)
    stats.output_tokens = sum(estimate_tokens(d["text"]) for d in output)
    stats.extra = {
        "algorithm": "MinHash LSH",
        "n_permutations": N_PERMUTATIONS,
        "shingle_size": SHINGLE_SIZE,
        "jaccard_threshold": JACCARD_THRESH,
        "exact_dups_removed": exact_dups,
        "near_dups_removed": near_dup_count,
        "memory_note": (
            f"This demo: {len(docs):,} docs. "
            "Production 100M docs with 256 perms requires >= 768GB RAM node."
        ),
        "indic_note": "NFC normalization applied before hashing to catch Unicode variant duplicates.",
    }
    return output
