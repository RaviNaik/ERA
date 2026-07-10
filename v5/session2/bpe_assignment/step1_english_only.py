"""
step1_english_only.py — Vanilla BPE Tokenizer on English Only
==============================================================
STEP 1: Establish a baseline.

Train a vanilla BPE tokenizer on India's Wikipedia page in English.
- Vocab size: 10,000 tokens
- Pre-tokenizer: simple Whitespace split
- No special preprocessing

After training, compute fertility ratios for ALL 4 languages using this
English-only tokenizer, to see how badly it handles Indic scripts.

Run:
    python step1_english_only.py
"""

import sys
from pathlib import Path

# Make sure utils is importable when running from any directory
sys.path.insert(0, str(Path(__file__).parent))

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
from utils import (
    DATA_DIR, MODELS_DIR,
    fetch_all_languages,
    report_word_token_details,
    report_metrics,
    count_words,
)


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
VOCAB_SIZE   = 10_000
MIN_FREQ     = 2
MODEL_PATH   = MODELS_DIR / "step1_en_only.json"

SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[BOS]", "[EOS]"]


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  STEP 1: Vanilla BPE — English Only")
    print("=" * 55)

    # 1. Download data (cached if already present)
    print("\n[1] Fetching Wikipedia data ...")
    texts = fetch_all_languages()

    en_text = texts["en"]
    print(f"\n  English corpus: {count_words(en_text):,} words, {len(en_text):,} chars")

    # 2. Build tokenizer
    print("\n[2] Building BPE tokenizer ...")
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    # Whitespace pre-tokenizer — the simplest possible split
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # ByteLevel decoder for clean decoding
    tokenizer.decoder = decoders.BPEDecoder()

    # 3. Configure trainer
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQ,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    # 4. Train on English only
    print(f"\n[3] Training on English text (vocab_size={VOCAB_SIZE}) ...")
    def english_iterator():
        # Feed line-by-line for memory efficiency
        for line in en_text.split("\n"):
            line = line.strip()
            if line:
                yield line

    tokenizer.train_from_iterator(english_iterator(), trainer=trainer)

    actual_vocab = tokenizer.get_vocab_size()
    print(f"  Training complete. Actual vocab size: {actual_vocab:,}")

    # 5. Save model
    tokenizer.save(str(MODEL_PATH))
    print(f"  Model saved → {MODEL_PATH}")

    # 6. Evaluate on all 4 languages
    print("\n[4] Computing fertility ratios on all 4 languages ...")
    print("     (Expect: English ≈ good, Indic scripts ≈ terrible)")
    print()
    ratios = report_word_token_details(tokenizer, texts)

    # 7. Print report & score
    report_metrics(ratios, step_name="STEP 1 Results — English-Only BPE")

    print("\nObservations:")
    en_ratio = ratios.get("en", 0)
    print(f"  • English fertility ratio:  {en_ratio:.4f}")
    print(f"    (< 1.5 means BPE is efficient for English)")
    print()
    print("  • Indic language ratios will be very high because:")
    print("    - The English-only tokenizer has never seen Hindi/Telugu/Kannada chars")
    print("    - It will fall back to character-level encoding (or [UNK] tokens)")
    print("    - Each Indic word becomes many fragmented byte-pieces")
    print()
    print(f"  → Model saved: {MODEL_PATH}")


if __name__ == "__main__":
    main()
