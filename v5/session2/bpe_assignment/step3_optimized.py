"""
step3_optimized.py — Optimized Multilingual BPE (Best Score)
=============================================================
STEP 3: Maximum score through surgical optimizations.

Applies ALL recommended strategies from problem_tobe_solved.md:

  ✓ Unicode NFKC normalization
  ✓ ZWJ / ZWNJ removal (invisible joiners that fragment Indic tokens)
  ✓ Two vocabulary allocation strategies, both run and compared:

      Strategy A — Oversampling
      ─────────────────────────
      Feed English once, Indic languages N× to force the BPE
      algorithm to allocate proportionally more vocab slots to
      scripts with larger character inventories.

      Strategy B — Merged Vocabulary (Recommended)
      ────────────────────────────────────────────
      Train 4 independent BPE tokenizers (one per language),
      then merge their vocabularies into a single 10,000-token dict.
      This guarantees controlled allocation with no language dominating.

The best-scoring model is saved as step3_optimized.json.

Run:
    python step3_optimized.py
"""

import sys
import json
import unicodedata
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent))

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders, normalizers
from utils import (
    DATA_DIR, MODELS_DIR,
    fetch_all_languages,
    report_word_token_details,
    report_metrics,
    count_words,
    normalize_text,
    WIKI_PAGES,
)


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
VOCAB_SIZE = 10_000
MIN_FREQ   = 2

# Oversampling repetitions for Indic languages in Strategy A
INDIC_OVERSAMPLE_FACTOR = 10

SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[BOS]", "[EOS]"]

MODEL_PATH_A = MODELS_DIR / "step3_strategy_a_oversample.json"
MODEL_PATH_B = MODELS_DIR / "step3_strategy_b_merged.json"
MODEL_PATH_BEST = MODELS_DIR / "step3_optimized.json"


# ──────────────────────────────────────────────────────────────
# Shared Preprocessing
# ──────────────────────────────────────────────────────────────

def preprocess(texts: dict[str, str]) -> dict[str, str]:
    """
    Apply NFKC normalization + ZWJ/ZWNJ removal to all language texts.
    This is the foundation that BOTH strategies depend on.
    """
    print("\n[2] Preprocessing — NFKC normalization + ZWJ/ZWNJ removal ...")
    cleaned = {}
    for lang, text in texts.items():
        original_len = len(text)
        text = normalize_text(text)
        cleaned[lang] = text
        name = WIKI_PAGES[lang][2]
        print(f"    {name:<10}: {original_len:>9,} → {len(text):>9,} chars "
              f"(removed {original_len - len(text):,} control chars)")
    return cleaned


def build_base_tokenizer() -> Tokenizer:
    """
    Create a fresh BPE tokenizer with:
    - Whitespace + Punctuation pre-tokenizer (respects word boundaries better)
    - NFKC normalizer built into the tokenizer itself
    """
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    # Built-in normalizer: NFKC + lowercase (optional; we keep case for Indic scripts)
    tokenizer.normalizer = normalizers.NFKC()

    # WhitespaceSplit is cleaner than Whitespace for multilingual:
    # it splits on whitespace but doesn't touch punctuation aggressively
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

    tokenizer.decoder = decoders.BPEDecoder()
    return tokenizer


def make_trainer(**kwargs) -> trainers.BpeTrainer:
    return trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQ,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        **kwargs,
    )


def lines_from_text(text: str):
    """Yield non-empty lines from text."""
    for line in text.split("\n"):
        line = line.strip()
        if line:
            yield line


# ──────────────────────────────────────────────────────────────
# Strategy A — Oversampling
# ──────────────────────────────────────────────────────────────

def train_strategy_a(cleaned_texts: dict[str, str]) -> Tokenizer:
    """
    Strategy A: Oversample Indic languages.

    Feed 1× English + N× Hindi/Telugu/Kannada into a single BPE trainer.
    The BPE frequency counts will be proportionally higher for Indic scripts,
    forcing the algorithm to allocate more merges (and vocab slots) to them.

    Why this works:
        BPE is a greedy frequency algorithm. If we repeat the Indic texts
        10×, the pair frequencies for Indic character sequences become 10×
        larger, so the algorithm merges them earlier and assigns more tokens.
    """
    print("\n" + "─" * 60)
    print(f"  Strategy A: Oversampling (Indic ×{INDIC_OVERSAMPLE_FACTOR})")
    print("─" * 60)

    tokenizer = build_base_tokenizer()
    trainer = make_trainer()

    def oversampled_iterator():
        # English: 1×
        yield from lines_from_text(cleaned_texts["en"])

        # Indic languages: N×
        for lang in ["hi", "te", "kn"]:
            for _ in range(INDIC_OVERSAMPLE_FACTOR):
                yield from lines_from_text(cleaned_texts[lang])

    total_en_lines = sum(1 for _ in lines_from_text(cleaned_texts["en"]))
    total_indic_lines = sum(
        sum(1 for _ in lines_from_text(cleaned_texts[lang])) for lang in ["hi", "te", "kn"]
    )
    print(f"  English lines (1×):      {total_en_lines:,}")
    print(f"  Indic lines (×{INDIC_OVERSAMPLE_FACTOR}):     {total_indic_lines * INDIC_OVERSAMPLE_FACTOR:,}")

    tokenizer.train_from_iterator(oversampled_iterator(), trainer=trainer)
    print(f"  Vocab size after training: {tokenizer.get_vocab_size():,}")

    tokenizer.save(str(MODEL_PATH_A))
    print(f"  Saved → {MODEL_PATH_A}")
    return tokenizer


# ──────────────────────────────────────────────────────────────
# Strategy B — Merged Vocabulary
# ──────────────────────────────────────────────────────────────

def train_single_language(text: str, lang: str, name: str) -> dict:
    """
    Train a BPE tokenizer on a single language.
    No fixed cap per language — the trainer will use up to VOCAB_SIZE tokens,
    but in practice each language's natural text limits this organically.

    Returns the tokenizer's vocab and merges as a dict for merging.
    """
    # Determine per-language vocab budget
    # We split 10,000 evenly: 2,500 per language
    # (no hard constraint per problem statement, but equal split is fair baseline)
    per_lang_vocab = VOCAB_SIZE // len(WIKI_PAGES)

    tokenizer = build_base_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=per_lang_vocab,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        show_progress=False,
    )

    tokenizer.train_from_iterator(lines_from_text(text), trainer=trainer)
    actual = tokenizer.get_vocab_size()
    print(f"    {name:<10}: trained {actual:,} tokens (budget: {per_lang_vocab:,})")

    # Save intermediate
    path = MODELS_DIR / f"step3_lang_{lang}.json"
    tokenizer.save(str(path))
    return path


def merge_tokenizers(lang_model_paths: dict[str, Path]) -> Tokenizer:
    """
    Merge multiple language-specific BPE vocabularies into one tokenizer.

    Approach:
      1. Load each language's vocab + merges from its JSON file
      2. Assign new contiguous IDs (handle collisions by keeping first occurrence)
      3. Combine all merges in frequency order (special tokens first)
      4. Build a new Tokenizer from the merged vocab

    This guarantees each language gets proportional representation.
    """
    merged_vocab = {}   # token_str → id
    merged_merges = []  # list of (a, b) merge tuples
    seen_tokens = set()

    # Always add special tokens first
    for i, tok in enumerate(SPECIAL_TOKENS):
        merged_vocab[tok] = i
        seen_tokens.add(tok)

    next_id = len(SPECIAL_TOKENS)

    for lang, path in lang_model_paths.items():
        name = WIKI_PAGES[lang][2]
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        lang_vocab = data["model"]["vocab"]
        lang_merges = data["model"]["merges"]

        added = 0
        for token, _ in sorted(lang_vocab.items(), key=lambda x: x[1]):
            if token not in seen_tokens:
                merged_vocab[token] = next_id
                seen_tokens.add(token)
                next_id += 1
                added += 1

        # Add merges (string pairs like "Ĥ e" format used by HF tokenizers)
        for merge in lang_merges:
            if merge not in merged_merges:
                merged_merges.append(merge)

        print(f"    {name:<10}: +{added:>5,} new tokens → merged vocab now {next_id:,}")

    print(f"  Total merged vocab size: {len(merged_vocab):,} tokens")

    # Build the merged BPE model
    # HuggingFace BPE needs vocab as {str: int} and merges as list of (str, str) tuples
    merge_tuples = []
    for merge in merged_merges:
        if isinstance(merge, str):
            parts = merge.split(" ", 1)
            if len(parts) == 2:
                merge_tuples.append((parts[0], parts[1]))
        elif isinstance(merge, (list, tuple)) and len(merge) == 2:
            merge_tuples.append((merge[0], merge[1]))

    bpe_model = models.BPE(
        vocab=merged_vocab,
        merges=merge_tuples,
        unk_token="[UNK]",
    )

    merged_tokenizer = Tokenizer(bpe_model)
    merged_tokenizer.normalizer = normalizers.NFKC()
    merged_tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    merged_tokenizer.decoder = decoders.BPEDecoder()

    return merged_tokenizer


def train_strategy_b(cleaned_texts: dict[str, str]) -> Tokenizer:
    """
    Strategy B: Merged Vocabulary.

    Train 4 independent language-specific BPE tokenizers,
    then merge all 4 vocabularies into a single 10,000-token tokenizer.

    Why this is better than oversampling:
        - Deterministic: each language is guaranteed vocab representation
        - No frequency bias: BPE independently learns the best merges for each script
        - Grapheme clusters in Indic scripts are merged without competing against
          high-frequency English pairs
    """
    print("\n" + "─" * 60)
    print("  Strategy B: Merged Vocabulary (4 independent tokenizers)")
    print("─" * 60)

    lang_model_paths = {}
    for lang, text in cleaned_texts.items():
        name = WIKI_PAGES[lang][2]
        path = train_single_language(text, lang, name)
        lang_model_paths[lang] = path

    print("\n  Merging vocabularies ...")
    merged_tokenizer = merge_tokenizers(lang_model_paths)

    merged_tokenizer.save(str(MODEL_PATH_B))
    print(f"  Saved → {MODEL_PATH_B}")
    return merged_tokenizer


# ──────────────────────────────────────────────────────────────
# Score Comparison
# ──────────────────────────────────────────────────────────────

def compute_score(ratios: dict[str, float]) -> float:
    """Compute 1000 / (max_ratio - min_ratio)."""
    vals = list(ratios.values())
    spread = max(vals) - min(vals)
    return 1000.0 / spread if spread > 0 else float("inf")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  STEP 3: Optimized Multilingual BPE")
    print("  Goal: Minimize spread across fertility ratios → maximize score")
    print("=" * 60)

    # 1. Download data
    print("\n[1] Fetching Wikipedia data ...")
    texts = fetch_all_languages()

    # 2. Preprocess (NFKC + ZWJ/ZWNJ removal)
    cleaned_texts = preprocess(texts)

    # ── Strategy A ──────────────────────────────────────────
    print("\n[3] Training Strategy A — Oversampling ...")
    tokenizer_a = train_strategy_a(cleaned_texts)

    print("\n  Evaluating Strategy A ...")
    ratios_a = report_word_token_details(tokenizer_a, cleaned_texts)
    score_a = compute_score(ratios_a)
    report_metrics(ratios_a, step_name="Strategy A — Oversampling")

    # ── Strategy B ──────────────────────────────────────────
    print("\n[4] Training Strategy B — Merged Vocabulary ...")
    tokenizer_b = train_strategy_b(cleaned_texts)

    print("\n  Evaluating Strategy B ...")
    ratios_b = report_word_token_details(tokenizer_b, cleaned_texts)
    score_b = compute_score(ratios_b)
    report_metrics(ratios_b, step_name="Strategy B — Merged Vocabulary")

    # ── Pick Winner ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  Strategy A (Oversampling) score: {score_a:.2f}")
    print(f"  Strategy B (Merged Vocab)  score: {score_b:.2f}")

    if score_b >= score_a:
        best_tokenizer = tokenizer_b
        best_name = "Strategy B (Merged Vocabulary)"
        best_ratios = ratios_b
    else:
        best_tokenizer = tokenizer_a
        best_name = "Strategy A (Oversampling)"
        best_ratios = ratios_a

    best_score = max(score_a, score_b)
    print(f"\n  ✓ Winner: {best_name}")
    print(f"  ✓ Best Score: {best_score:.2f}")

    # Save best model
    best_tokenizer.save(str(MODEL_PATH_BEST))
    print(f"\n  Best model saved → {MODEL_PATH_BEST}")

    # Final summary
    report_metrics(best_ratios, step_name=f"STEP 3 FINAL — {best_name}")

    print("\nOptimizations Applied:")
    print("─" * 60)
    print("  ✓ NFKC Unicode normalization")
    print("     Collapses compatibility characters (same glyph, different bytes)")
    print()
    print("  ✓ ZWJ / ZWNJ removal (U+200D, U+200C)")
    print("     Prevents duplicate tokens for visually identical Indic conjuncts")
    print()
    print("  ✓ Strategy A: Indic oversampling ×" + str(INDIC_OVERSAMPLE_FACTOR))
    print("     Biases BPE frequency counts toward Indic scripts")
    print()
    print("  ✓ Strategy B: Per-language independent training + merge")
    print("     Guarantees each language gets proportional vocab allocation")
    print("     No frequency competition between English and Indic scripts")
    print()
    print(f"  → Best model: {MODEL_PATH_BEST}")


if __name__ == "__main__":
    main()
