#!/usr/bin/env python3
"""
Evaluate tokenizer.json on the faithful Markdown corpus.

Checks:
  1. Faithfulness: decode(encode(text)) preserves all visible non-whitespace chars
  2. Fertility ratios: tokens / faithful_units for each language
  3. Score: 1000 / (max_fertility - min_fertility)
  4. Hindi penalty: exp(max(0, hi_fertility/1.2 - 1))

Run:
    python evaluate_tokenizer.py

Languages: English (en), Hindi (hi), Telugu (te), Kannada (kn)
"""
from __future__ import annotations

import io
import sys

# Ensure UTF-8 output on Windows consoles
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import json
import math
import re
from pathlib import Path

import regex
from tokenizers import Tokenizer


ROOT      = Path(__file__).resolve().parent
CORPUS    = ROOT / "corpus"
TOKENIZER = ROOT / "tokenizer.json"
LANGS     = ["en", "hi", "te", "kn"]

LANG_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "kn": "Kannada",
}

# Faithful unit = contiguous letter/mark/number run OR one visible non-space char
FAITHFUL_UNIT_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]")

# Visible-text extractor for round-trip check (non-whitespace chars only)
_VISIBLE_RE = re.compile(r"\S")


def faithful_units(text: str) -> int:
    return len(FAITHFUL_UNIT_RE.findall(text))


def visible_chars(text: str) -> str:
    """Extract only non-whitespace characters for round-trip comparison."""
    return "".join(_VISIBLE_RE.findall(text))


def check_faithfulness(tokenizer: Tokenizer, samples: list[str]) -> list[str]:
    """
    Check decode(encode(s)) preserves visible text.
    Returns a list of failure descriptions (empty = all passed).
    """
    failures = []
    for s in samples:
        encoded  = tokenizer.encode(s)
        decoded  = tokenizer.decode(encoded.ids)
        orig_vis = visible_chars(s)
        dec_vis  = visible_chars(decoded)
        if orig_vis != dec_vis:
            failures.append(
                f"Round-trip changed visible text for sample {s!r}: "
                f"decoded {decoded!r}"
            )
    return failures


def main() -> int:
    print(f"Loading tokenizer from: {TOKENIZER}")
    tokenizer = Tokenizer.from_file(str(TOKENIZER))

    # ── Faithfulness check ──────────────────────────────────────
    print("\nFaithfulness check ...")
    samples = [
        "https://hi.wikipedia.org/wiki/भारत#cite_ref-1",
        "https://en.wikipedia.org/wiki/India_(disambiguation)",
        "India's population is 1,428,627,663.",
        "The [Union] of India (भारत गणराज्य) was established in 1950.",
        "https://kn.wikipedia.org/wiki/ಭಾರತ#cite_note-1",
    ]
    failures = check_faithfulness(tokenizer, samples)
    if failures:
        print("  ✗ FAITHFULNESS FAILURES:")
        for f in failures:
            print(f"    - {f}")
        print("\nTokenizer is NOT acceptable for faithful Markdown evaluation.")
        return 1
    else:
        print("  ✓ All round-trip checks passed.")

    # ── Fertility evaluation ────────────────────────────────────
    print("\nEvaluating fertility ratios ...")
    rows: dict[str, dict] = {}
    for code in LANGS:
        txt_path = CORPUS / f"{code}.faithful.txt"
        if not txt_path.exists():
            print(f"  [WARN] corpus/{code}.faithful.txt not found — run build_wiki_faithful_markdown.py first")
            return 1
        text  = txt_path.read_text(encoding="utf-8")
        units = faithful_units(text)
        toks  = len(tokenizer.encode(text).ids)
        rows[code] = {
            "language":      LANG_NAMES[code],
            "tokens":        toks,
            "faithful_units": units,
            "ratio":         toks / units,
        }

    ratios     = [row["ratio"] for row in rows.values()]
    spread     = max(ratios) - min(ratios)
    score      = 1000 / spread
    hi_penalty = math.exp(max(0.0, rows["hi"]["ratio"] / 1.2 - 1.0))

    result = {
        "rows":                       rows,
        "spread":                     spread,
        "score":                      score,
        "hindi_exp1_penalty_factor":  hi_penalty,
        "hindi_exp1_adjusted_score":  score / hi_penalty,
    }

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # ── Summary table ───────────────────────────────────────────
    print("\n" + "─" * 60)
    print(f"  {'Language':<12} {'Tokens':>10} {'F-Units':>10} {'Ratio':>8}  Threshold")
    print("─" * 60)
    for code, row in rows.items():
        thresh = "✓ OK" if row["ratio"] <= 1.2 else "✗ OVER 1.2"
        print(
            f"  {row['language']:<12} {row['tokens']:>10,} {row['faithful_units']:>10,} "
            f"{row['ratio']:>8.4f}  {thresh}"
        )
    print("─" * 60)
    print(f"  Spread:           {spread:.6f}")
    print(f"  Raw score:        {score:.2f}")
    print(f"  Hindi penalty:    {hi_penalty:.6f}")
    print(f"  Adjusted score:   {score / hi_penalty:.2f}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
