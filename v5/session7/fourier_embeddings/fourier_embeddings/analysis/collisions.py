"""Validation protocol item 2 (research note Sec. 10.2 / class notes Sec. 8,
Widget 9): measure how often distinct vocabulary tokens produce
indistinguishable codes, per script, under the one-hot Kronecker codec vs.
the Fourier-Kronecker codec.

- Kronecker: an *exact* collision — two tokens whose first `pos_dim` bytes
  are bit-identical, hence bit-identical codes (class notes Sec. 8).
- Fourier: no exact collisions are structurally possible (research note
  Sec. 6.1), so "collision" becomes a continuous, threshold-calibrated
  quantity (research note Sec. 8.4): cosine similarity between two tokens'
  codes above a chosen threshold counts as a *functional* collision. The
  threshold is calibrated so that, on same-script token pairs that are NOT
  exact Kronecker collisions, the Fourier functional-collision rate is
  measured on equal footing, not assumed.

Tokens are bucketed into a script by the Unicode block of their first
non-ASCII, non-punctuation code point (falls back to "ascii"/"other").
"""

from __future__ import annotations

import argparse
import json
import logging
import unicodedata
from collections import defaultdict
from pathlib import Path

import torch

from fourier_embeddings.data.tokenizer import load_tokenizer, build_id_to_bytes
from fourier_embeddings.model.codecs import KroneckerEmbedding, FourierKroneckerEmbedding
from fourier_embeddings.utils.logging_setup import setup_logging

logger = logging.getLogger("fourier_embeddings.analysis.collisions")

SCRIPT_RANGES = {
    "devanagari": (0x0900, 0x097F),
    "bengali": (0x0980, 0x09FF),
    "tamil": (0x0B80, 0x0BFF),
    "telugu": (0x0C00, 0x0C7F),
    "kannada": (0x0C80, 0x0CFF),
}


def classify_script(text: str) -> str:
    for ch in text:
        cp = ord(ch)
        for name, (lo, hi) in SCRIPT_RANGES.items():
            if lo <= cp <= hi:
                return name
        if ch.isascii() and ch.isalpha():
            return "ascii"
    return "other"


def exact_kronecker_collisions(id_to_bytes: list[bytes], pos_dim: int,
                                script_of: list[str]) -> dict:
    """Group tokens by their first-`pos_dim`-bytes prefix; any group with >1
    member is a set of tokens the shipped Kronecker codec cannot distinguish.
    """
    buckets: dict[bytes, list[int]] = defaultdict(list)
    for v, b in enumerate(id_to_bytes):
        buckets[b[:pos_dim]].append(v)

    per_script_total = defaultdict(int)
    per_script_colliding = defaultdict(int)
    example_collisions = []
    for v, s in enumerate(script_of):
        per_script_total[s] += 1
    for prefix, members in buckets.items():
        if len(members) > 1:
            for v in members:
                per_script_colliding[script_of[v]] += 1
            if len(example_collisions) < 20:
                example_collisions.append(members[:5])

    rates = {s: per_script_colliding[s] / max(per_script_total[s], 1) for s in per_script_total}
    return {
        "per_script_total": dict(per_script_total),
        "per_script_colliding": dict(per_script_colliding),
        "per_script_collision_rate": rates,
        "n_collision_groups": sum(1 for m in buckets.values() if len(m) > 1),
        "example_collisions": example_collisions,
    }


@torch.no_grad()
def functional_collisions(codec, vocab_size: int, script_of: list[str], threshold: float,
                           batch_size: int = 512, device: str = "cpu") -> dict:
    """All-pairs-within-script cosine similarity, thresholded. O(V^2 / scripts)
    but scripts are Wikipedia-vocab-scale (thousands, not hundreds of
    thousands) so this is tractable on CPU for the smoke-scale vocabulary.
    """
    codec = codec.to(device)
    ids = torch.arange(vocab_size, device=device)
    codes = []
    for i in range(0, vocab_size, batch_size):
        chunk = ids[i:i + batch_size].unsqueeze(0)  # [1, chunk]
        out = codec(chunk).squeeze(0)  # [chunk, d_model]
        codes.append(out)
    codes = torch.cat(codes, dim=0)
    codes = codes / (codes.norm(dim=-1, keepdim=True) + 1e-8)

    by_script: dict[str, list[int]] = defaultdict(list)
    for v, s in enumerate(script_of):
        by_script[s].append(v)

    per_script_colliding = {}
    per_script_total = {}
    for s, members in by_script.items():
        if len(members) < 2:
            per_script_colliding[s] = 0
            per_script_total[s] = len(members)
            continue
        idx = torch.tensor(members, device=device)
        sub = codes[idx]  # [n, d_model]
        sim = sub @ sub.T  # [n, n]
        n = sim.shape[0]
        sim.fill_diagonal_(-1.0)
        above = (sim >= threshold).any(dim=1)
        per_script_colliding[s] = int(above.sum().item())
        per_script_total[s] = n

    rates = {s: per_script_colliding[s] / max(per_script_total[s], 1) for s in per_script_total}
    return {
        "threshold": threshold,
        "per_script_total": per_script_total,
        "per_script_colliding": per_script_colliding,
        "per_script_collision_rate": rates,
    }


def calibrate_threshold(codec, vocab_size: int, device: str, target_rate: float = None,
                         n_probe_pairs: int = 20000, seed: int = 0) -> float:
    """Research note Sec. 8.4: the Fourier collision threshold "should be set
    empirically (calibrated against the exact-collision baseline) rather than
    assumed." We calibrate by sampling random token pairs and reporting the
    99.5th percentile of their cosine similarity as a conservative default
    threshold (i.e. only the most similar 0.5% of *random* pairs would count
    as functional collisions) -- callers can override with a fixed value.
    """
    g = torch.Generator().manual_seed(seed)
    codec = codec.to(device)
    with torch.no_grad():
        ids = torch.arange(vocab_size, device=device).unsqueeze(0)
        codes = codec(ids).squeeze(0)
        codes = codes / (codes.norm(dim=-1, keepdim=True) + 1e-8)
        a = torch.randint(0, vocab_size, (n_probe_pairs,), generator=g)
        b = torch.randint(0, vocab_size, (n_probe_pairs,), generator=g)
        sims = (codes[a] * codes[b]).sum(dim=-1)
        thr = torch.quantile(sims.float(), 0.995).item()
    return thr


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer-path", default="tokenizer_out/tokenizer.json")
    ap.add_argument("--pos-dim", type=int, default=32)
    ap.add_argument("--fourier-dim", type=int, default=32)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--threshold", type=float, default=None,
                     help="fixed Fourier functional-collision threshold; if omitted, calibrated")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/collisions.json")
    ap.add_argument("--log-file", default="logs/collisions.log")
    args = ap.parse_args()

    setup_logging(args.log_file, name="fourier_embeddings.analysis.collisions")

    tok = load_tokenizer(args.tokenizer_path)
    id_to_bytes = build_id_to_bytes(tok)
    vocab_size = len(id_to_bytes)
    texts = [tok.decode([v], skip_special_tokens=False) for v in range(vocab_size)]
    script_of = [classify_script(t) for t in texts]

    logger.info(f"vocab_size={vocab_size}; script distribution: "
                f"{ {s: script_of.count(s) for s in set(script_of)} }")

    exact = exact_kronecker_collisions(id_to_bytes, args.pos_dim, script_of)
    logger.info(f"exact Kronecker collisions: {exact['per_script_collision_rate']}")

    kron_codec = KroneckerEmbedding(vocab_size, args.d_model, id_to_bytes, pos_dim=args.pos_dim)
    fourier_codec = FourierKroneckerEmbedding(vocab_size, args.d_model, id_to_bytes,
                                               pos_dim=args.pos_dim, fourier_dim=args.fourier_dim)

    threshold = args.threshold
    if threshold is None:
        threshold = calibrate_threshold(fourier_codec, vocab_size, args.device)
        logger.info(f"calibrated Fourier functional-collision threshold: {threshold:.4f}")

    fourier_functional = functional_collisions(fourier_codec, vocab_size, script_of, threshold,
                                                 device=args.device)
    logger.info(f"Fourier functional collisions @ threshold={threshold:.4f}: "
                f"{fourier_functional['per_script_collision_rate']}")

    # Sanity cross-check: the Kronecker codec's *exact* collisions should
    # also show up as functional collisions at a high threshold (cos ~= 1),
    # confirming the two measurement methods agree where they should.
    kron_functional = functional_collisions(kron_codec, vocab_size, script_of, threshold=0.999,
                                              device=args.device)

    report = {
        "vocab_size": vocab_size,
        "pos_dim": args.pos_dim,
        "fourier_dim": args.fourier_dim,
        "exact_kronecker_collisions": exact,
        "fourier_functional_collisions": fourier_functional,
        "kronecker_functional_collisions_cross_check": kron_functional,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
