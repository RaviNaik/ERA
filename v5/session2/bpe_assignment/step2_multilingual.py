"""
step2_multilingual.py — Naive Multilingual BPE (No Optimizations)
==================================================================
STEP 2: Understand the problem.

Train a BPE tokenizer on ALL 4 language Wikipedia pages for India —
English, Hindi, Telugu, and Kannada — but with NO special preprocessing:

  ✗ No Unicode normalization
  ✗ No ZWJ/ZWNJ cleanup
  ✗ No oversampling of Indic languages
  ✗ No grapheme-aware pre-tokenization
  ✗ Just concatenate everything equally and run BPE

Purpose: To observe and understand WHY naive multilingual BPE produces
poor, imbalanced fertility ratios for Indic scripts vs English.

Run:
    python step2_multilingual.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
from utils import (
    DATA_DIR, MODELS_DIR,
    fetch_all_languages,
    report_word_token_details,
    report_metrics,
    count_words,
    WIKI_PAGES,
)


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
VOCAB_SIZE = 10_000
MIN_FREQ   = 2
MODEL_PATH = MODELS_DIR / "step2_naive_multilingual.json"

SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[BOS]", "[EOS]"]


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  STEP 2: Naive Multilingual BPE (No Optimizations)")
    print("=" * 60)

    # 1. Download data
    print("\n[1] Fetching Wikipedia data ...")
    texts = fetch_all_languages()

    print("\n  Corpus statistics (raw, no cleaning):")
    total_chars = 0
    for lang, text in texts.items():
        name = WIKI_PAGES[lang][2]
        w = count_words(text)
        c = len(text)
        total_chars += c
        print(f"    {name:<10}: {w:>7,} words, {c:>9,} chars")
    print(f"    {'Total':<10}: {total_chars:>9,} chars (combined)")

    # 2. Build tokenizer
    print("\n[2] Building naive BPE tokenizer ...")
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    # Just whitespace — no grapheme awareness, no cleanup
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.BPEDecoder()

    # 3. Configure trainer
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQ,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    # 4. Train on all 4 languages equally (1× each, raw text, no normalization)
    print(f"\n[3] Training on all 4 languages (1× each, vocab_size={VOCAB_SIZE}) ...")
    print("     NOTE: No preprocessing — text fed as-is from Wikipedia")

    def all_languages_iterator():
        """Yield lines from all 4 languages interleaved (equal weight)."""
        for lang, text in texts.items():
            for line in text.split("\n"):
                line = line.strip()
                if line:
                    yield line

    tokenizer.train_from_iterator(all_languages_iterator(), trainer=trainer)

    actual_vocab = tokenizer.get_vocab_size()
    print(f"  Training complete. Actual vocab size: {actual_vocab:,}")

    # 5. Save model
    tokenizer.save(str(MODEL_PATH))
    print(f"  Model saved → {MODEL_PATH}")

    # 6. Evaluate
    print("\n[4] Computing fertility ratios ...")
    print()
    ratios = report_word_token_details(tokenizer, texts)

    # 7. Report
    report_metrics(ratios, step_name="STEP 2 Results — Naive Multilingual BPE")

    # 8. Diagnostic explanation
    print("Why the score is bad — Root Cause Analysis:")
    print("-" * 60)
    print()
    print("  1. ENGLISH DOMINANCE IN VOCAB")
    print("     English uses only ~26 base characters (ASCII).")
    print("     The BPE algorithm quickly merges them into long English words.")
    print("     Result: many vocab slots consumed by common English subwords.")
    print()
    print("  2. INDIC SCRIPT FRAGMENTATION")
    print("     Devanagari (Hindi), Telugu, Kannada each have 50+ base characters")
    print("     plus hundreds of vowel-sign (matra) combinations.")
    print("     With limited vocab slots left, Indic words get split into many")
    print("     small pieces → high fertility ratio (many tokens per word).")
    print()
    print("  3. ZWJ / ZWNJ POLLUTION")
    print("     Invisible Unicode joiner chars (U+200D, U+200C) in Wikipedia text")
    print("     cause the same visible word to have multiple byte representations.")
    print("     The tokenizer treats क्ष, क्‍ष, क्‌ष as THREE different sequences,")
    print("     wasting vocab slots on duplicates of the same word.")
    print()
    print("  4. NO GRAPHEME BOUNDARY AWARENESS")
    print("     Whitespace-only pre-tokenization allows BPE to cut inside a matra")
    print("     or halant sequence — producing tokens that don't map to any")
    print("     linguistically meaningful unit in the Indic script.")
    print()
    print("  → Fix all of these in Step 3 (step3_optimized.py)")
    print()
    print(f"  → Model saved: {MODEL_PATH}")


if __name__ == "__main__":
    main()
