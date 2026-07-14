#!/usr/bin/env python3
"""
Train the shared 10k BPE tokenizer for the faithful Markdown corpus.

Pipeline:
    python build_wiki_faithful_markdown.py   # fetch + convert corpus
    python train_tokenizer.py                # train tokenizer

Training choices:
  - Model:         HuggingFace BPE
  - Vocab size:    10,000
  - min_frequency: 1
  - Normalizer:    NFKC only
  - Pre-tokenizer: Metaspace (▁ as space marker)
  - Decoder:       Metaspace
  - Languages:     English, Hindi, Telugu, Kannada
  - Weights:       en×3, hi×4, te×4, kn×6

Metaspace is used instead of ByteLevel because ByteLevel spends too many
tokens on UTF-8 bytes for Indic scripts (3 bytes/char for Devanagari etc.).
Metaspace preserves punctuation, URLs, and spaces through the encode/decode
round-trip, satisfying the faithfulness requirement.
"""
from __future__ import annotations

import io
import sys

# Ensure UTF-8 output on Windows consoles
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import json
import math
import tempfile
from pathlib import Path

import regex
from tokenizers import Tokenizer
from tokenizers.decoders import Metaspace as MetaspaceDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.trainers import BpeTrainer


ROOT          = Path(__file__).resolve().parent
CORPUS        = ROOT / "corpus"
OUT_TOKENIZER = ROOT / "tokenizer.json"
OUT_METRICS   = ROOT / "metrics.json"

LANGS   = ["en", "hi", "te", "kn"]
WEIGHTS = {"en": 3, "hi": 4, "te": 4, "kn": 6}

# Faithful unit = contiguous letter/mark/number run OR one visible non-space char
FAITHFUL_UNIT_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]")

LANG_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "kn": "Kannada",
}


def faithful_units(text: str) -> int:
    return len(FAITHFUL_UNIT_RE.findall(text))


def make_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer    = NFKC()
    tokenizer.pre_tokenizer = Metaspace(replacement="▁", prepend_scheme="never")
    tokenizer.decoder       = MetaspaceDecoder(replacement="▁", prepend_scheme="never")
    return tokenizer


def train() -> tuple[Tokenizer, dict]:
    texts = {
        code: (CORPUS / f"{code}.faithful.txt").read_text(encoding="utf-8")
        for code in LANGS
    }
    units = {code: faithful_units(text) for code, text in texts.items()}

    print("Corpus stats:")
    for code, text in texts.items():
        print(f"  [{code}] {LANG_NAMES[code]}: {len(text):,} chars, {units[code]:,} faithful units")

    with tempfile.TemporaryDirectory() as tmp:
        files: list[str] = []
        tmpdir = Path(tmp)
        for code, text in texts.items():
            path = tmpdir / f"{code}.txt"
            path.write_text(text, encoding="utf-8")
            # Repeat file according to its weight (oversampling)
            files.extend([str(path)] * WEIGHTS[code])

        print(f"\nTraining with weights: {WEIGHTS}")
        print(f"Effective file count: {len(files)} (repeated by weight)")

        tokenizer = make_tokenizer()
        trainer = BpeTrainer(
            vocab_size=10000,
            min_frequency=1,
            special_tokens=["[UNK]"],
        )
        tokenizer.train(files, trainer)

    print(f"\nVocab size: {tokenizer.get_vocab_size():,}")

    # Compute token counts and fertility ratios
    token_counts = {
        code: len(tokenizer.encode(text).ids)
        for code, text in texts.items()
    }
    ratios = {code: token_counts[code] / units[code] for code in LANGS}
    spread = max(ratios.values()) - min(ratios.values())
    score  = 1000 / spread

    # Hindi penalty (from evaluator)
    hindi_penalty = math.exp(max(0.0, ratios["hi"] / 1.2 - 1.0))

    metrics = {
        "variant":  "wiki_faithful_markdown",
        "languages": LANG_NAMES,
        "weights":   WEIGHTS,
        "vocab_size": tokenizer.get_vocab_size(),
        "faithful_units": units,
        "unit_policy": (
            "Counts each contiguous Unicode letter/mark/number run as one unit "
            "and each visible non-space punctuation/symbol character as one unit."
        ),
        "token_counts": token_counts,
        "ratios":  ratios,
        "spread":  spread,
        "score":   score,
        "hindi_exp1_penalty_factor":  hindi_penalty,
        "hindi_exp1_adjusted_score":  score / hindi_penalty,
    }
    return tokenizer, metrics


def main() -> int:
    tokenizer, metrics = train()

    tokenizer.save(str(OUT_TOKENIZER))
    OUT_METRICS.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nTokenizer saved -> {OUT_TOKENIZER}")
    print(f"Metrics saved   -> {OUT_METRICS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
