"""Validation protocol item 3 (research note Sec. 10.3 / Sec. 8.2): directly
measure the order-sensitivity trade-off Design A makes.

Theorem 1 (research note Sec. 2) proves the one-hot Kronecker codec gives
*exact* orthogonality (cos = 0) to any positional rearrangement of the same
byte multiset. Theorem 2 / Sec. 4.3 proves Design A gives a *soft*, nonzero
similarity instead. This script builds a batch of anagram/transposition word
pairs and measures the actual cosine-similarity distribution under both
codecs, so the trade is quantified rather than only asserted.

Operates directly on raw byte strings via a tiny throwaway "vocabulary" (the
words under test), computing the pre-projection z-normalized code `kappa(b)`
so the numbers are exact reproductions of Theorems 1-2, not an artifact of a
random projection matrix.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from fourier_embeddings.model.codecs import ByteGridCodec
from fourier_embeddings.utils.logging_setup import setup_logging

logger = logging.getLogger("fourier_embeddings.analysis.order_sensitivity")

# The worked examples from the research note itself (Sec. 2, Sec. 4.3, Sec. 7.1).
#
# NOTE on "cat"/"tac": the research note's Sec. 2 prose claims this pair is
# "completely disjoint (value, position) cells" with cos=0 under Kronecker.
# It is not: reversing an odd-length string fixes its middle character, so
# "cat"[1]='a' and "tac"[1]='a' share position 1. Theorem 1's own formula
# (agree at 1 of 3 positions -> cos=(3-1)/3=1/3) and Sec. 4.3's own Fourier
# worked example (which explicitly treats "a@1 matches a@1" as a direct,
# same-position term) both disagree with the Sec. 2 prose. The corollary
# itself ("a *derangement* -- no fixed point -- gives exact orthogonality")
# is correct; "cat"/"tac" is just not an instance of one. Kept below because
# it's still the reference example, and this script's job is to measure and
# report what actually comes out, not to assume the note's number -- see
# `abcde`/`bcdea` for a pair that *is* a true derangement.
NAMED_PAIRS = [
    ("cat", "tac", "anagram-with-a-fixed-point (research note's example; see NOTE above)"),
    ("abcde", "bcdea", "derangement (true zero-fixed-point rearrangement)"),
    ("mistake", "mistkae", "transposition"),
    ("compute", "commute", "one-byte-substitution"),
    ("train", "trian", "transposition"),
    ("listen", "silent", "anagram-with-a-fixed-point"),
    ("read", "dear", "anagram-with-fixed-points"),
]

DEFAULT_WORDS = [
    "training", "trainer", "transformer", "embedding", "tokenizer", "attention",
    "gradient", "optimizer", "language", "sequence", "position", "collision",
    "vocabulary", "projection", "kronecker", "fourier", "spectral", "codec",
    "internationalisation", "byte", "network", "wavelet", "frequency", "kernel",
]


def transpose_pair(word: str, rng: random.Random) -> str:
    if len(word) < 2:
        return word
    i = rng.randrange(len(word) - 1)
    chars = list(word)
    chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def anagram_pair(word: str, rng: random.Random) -> str:
    chars = list(word)
    rng.shuffle(chars)
    return "".join(chars)


def build_word_codec(words: list[str], pos_dim: int, kind: str, fourier_dim: int = 32,
                      schedule: str = "standard") -> tuple[ByteGridCodec, dict[str, int]]:
    byte_lists = [w.encode("utf-8") for w in words]
    word_to_id = {w: i for i, w in enumerate(words)}
    if kind == "kronecker":
        codec = ByteGridCodec(len(words), 8, byte_lists, pos_dim=pos_dim, pos_factor="onehot")
    else:
        codec = ByteGridCodec(len(words), 8, byte_lists, pos_dim=pos_dim, pos_factor="fourier",
                               pos_factor_dim=fourier_dim, fourier_schedule=schedule)
    return codec, word_to_id


@torch.no_grad()
def codes_for_words(codec: ByteGridCodec, word_to_id: dict[str, int]) -> torch.Tensor:
    ids = torch.arange(len(word_to_id))
    byte_ids = codec.byte_ids[ids]
    valid_mask = codec.valid_mask[ids]
    length = codec.length[ids]
    return codec._grid_codes(byte_ids, valid_mask, length)  # pre-projection kappa(b)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pos-dim", type=int, default=16)
    ap.add_argument("--fourier-dim", type=int, default=16)
    ap.add_argument("--n-random-pairs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/order_sensitivity.json")
    ap.add_argument("--log-file", default="logs/order_sensitivity.log")
    args = ap.parse_args()

    setup_logging(args.log_file, name="fourier_embeddings.analysis.order_sensitivity")
    rng = random.Random(args.seed)

    # Build the full word list: named pairs + generated transposition/anagram pairs.
    words = set()
    pairs: list[tuple[str, str, str]] = list(NAMED_PAIRS)
    for w in DEFAULT_WORDS:
        pairs.append((w, transpose_pair(w, rng), "generated-transposition"))
        pairs.append((w, anagram_pair(w, rng), "generated-anagram"))
    extra_pool = DEFAULT_WORDS * ((args.n_random_pairs // len(DEFAULT_WORDS)) + 1)
    for w in extra_pool[: max(0, args.n_random_pairs - len(pairs))]:
        pairs.append((w, transpose_pair(w, rng), "generated-transposition"))

    for a, b, _ in pairs:
        words.add(a)
        words.add(b)
    words = sorted(words)

    max_len = max(len(w.encode("utf-8")) for w in words)
    pos_dim = max(args.pos_dim, max_len)

    kron_codec, w2i = build_word_codec(words, pos_dim, "kronecker")
    fourier_codec, _ = build_word_codec(words, pos_dim, "fourier", args.fourier_dim, "standard")

    kron_codes = codes_for_words(kron_codec, w2i)
    fourier_codes = codes_for_words(fourier_codec, w2i)

    rows = []
    for a, b, kind in pairs:
        ia, ib = w2i[a], w2i[b]
        row = {
            "word_a": a, "word_b": b, "pair_kind": kind,
            "kronecker_cosine": cosine(kron_codes[ia], kron_codes[ib]),
            "fourier_cosine": cosine(fourier_codes[ia], fourier_codes[ib]),
        }
        rows.append(row)

    kron_vals = torch.tensor([r["kronecker_cosine"] for r in rows])
    fourier_vals = torch.tensor([r["fourier_cosine"] for r in rows])

    summary = {
        "n_pairs": len(rows),
        "kronecker": {
            "mean": kron_vals.mean().item(), "std": kron_vals.std().item(),
            "min": kron_vals.min().item(), "max": kron_vals.max().item(),
            "frac_exactly_zero": (kron_vals.abs() < 1e-5).float().mean().item(),
        },
        "fourier": {
            "mean": fourier_vals.mean().item(), "std": fourier_vals.std().item(),
            "min": fourier_vals.min().item(), "max": fourier_vals.max().item(),
            "frac_exactly_zero": (fourier_vals.abs() < 1e-5).float().mean().item(),
        },
    }
    logger.info(f"summary: {summary}")
    logger.info("named pairs (research-note worked examples):")
    for r in rows[: len(NAMED_PAIRS)]:
        logger.info(f"  {r['word_a']!r} vs {r['word_b']!r} ({r['pair_kind']}): "
                     f"kronecker={r['kronecker_cosine']:.4f} fourier={r['fourier_cosine']:.4f}")

    report = {"config": vars(args), "summary": summary, "pairs": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
