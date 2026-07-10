"""
step3c_bytelevel.py — Optimized Multilingual BPE with ByteLevel
=============================================================
STEP 3C: Combines our mathematical sweet spot (2x oversampling) 
with HuggingFace's ByteLevel tokenization.

ByteLevel maps all 256 bytes to unique Unicode characters before training.
This guarantees that there will NEVER be an [UNK] token. For Indic scripts,
this means a single complex character (often 3 bytes in UTF-8) will be 
initially split into 3 base byte-tokens before BPE merges them.

Let's see how this affects the fertility ratios and the final score!

Run:
    python step3c_bytelevel.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
from utils import (
    DATA_DIR, MODELS_DIR,
    fetch_all_languages,
    report_word_token_details,
    report_metrics,
    normalize_text,
    WIKI_PAGES,
)

VOCAB_SIZE = 10_000
MIN_FREQ   = 2

# We use the mathematical sweet spot discovered in Step 3A
INDIC_OVERSAMPLE_FACTOR = 2

# No [UNK] token needed for ByteLevel BPE!
SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]"]

MODEL_PATH = MODELS_DIR / "step3c_bytelevel.json"


def preprocess(texts: dict[str, str]) -> dict[str, str]:
    print("\n[1] Preprocessing — NFKC normalization + ZWJ/ZWNJ removal ...")
    cleaned = {}
    for lang, text in texts.items():
        original_len = len(text)
        text = normalize_text(text)
        cleaned[lang] = text
        name = WIKI_PAGES[lang][2]
        print(f"    {name:<10}: {original_len:>9,} → {len(text):>9,} chars ")
    return cleaned


def lines_from_text(text: str):
    """Yield non-empty lines from text."""
    for line in text.split("\n"):
        line = line.strip()
        if line:
            yield line


def main():
    print("=" * 60)
    print("  STEP 3C: ByteLevel BPE + Oversampling (2x)")
    print("=" * 60)

    print("\n[0] Fetching Wikipedia data ...")
    texts = fetch_all_languages()
    cleaned_texts = preprocess(texts)

    print("\n[2] Building ByteLevel BPE Tokenizer ...")
    
    # Initialize BPE (notice no unk_token is specified, because it's not needed)
    tokenizer = Tokenizer(models.BPE())

    # ByteLevel pre-tokenizer maps bytes to characters
    # add_prefix_space=True ensures words are treated consistently whether they
    # are at the start of a sentence or follow a space.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    
    # ByteLevel decoder maps the characters back to bytes for the final text
    tokenizer.decoder = decoders.ByteLevel()

    print("\n[3] Training ByteLevel BPE ...")
    
    # The initial_alphabet must be set to the 256 mapped byte characters
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQ,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True
    )

    def oversampled_iterator():
        # English: 1x
        yield from lines_from_text(cleaned_texts["en"])

        # Indic languages: 2x
        for lang in ["hi", "te", "kn"]:
            for _ in range(INDIC_OVERSAMPLE_FACTOR):
                yield from lines_from_text(cleaned_texts[lang])

    tokenizer.train_from_iterator(oversampled_iterator(), trainer=trainer)
    print(f"  Vocab size after training: {tokenizer.get_vocab_size():,}")

    tokenizer.save(str(MODEL_PATH))
    print(f"  Saved → {MODEL_PATH}")

    print("\n[4] Evaluating Fertility Ratios ...")
    ratios = report_word_token_details(tokenizer, cleaned_texts)
    
    report_metrics(ratios, step_name="Step 3C — ByteLevel BPE")


if __name__ == "__main__":
    main()
