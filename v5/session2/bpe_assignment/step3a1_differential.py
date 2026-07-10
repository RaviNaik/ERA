"""
step3a1_differential.py — Experiment 3A1: Differential Per-Language Oversampling
==================================================================================
Key insight from Experiment 3A (uniform ×2 Indic, Score: 2396):
  - Hindi ratio = 1.1293 — too LOW (over-boosted, Hindi is 80% of English size)
  - Kannada ratio = 1.5465 — spread-driver (still needs more boosting)
  - English ratio = 1.5138 — fine
  - Telugu ratio  = 1.4941 — nearly ideal

Root cause: a uniform ×2 factor treats Hindi the same as Kannada, which is wrong.
Hindi (8K words) barely needs any boost. Kannada (979 words) needs ~10× more
data to compensate for its tiny corpus.

Strategy:
  - en × 1  (10,027 words — dominant, no boost)
  - hi × 1  (8,022 words  — nearly equal to English, minimal boost)
  - te × 4  (2,453 words  — 0.24× English, proportional boost ≈ 4.1×)
  - kn × 10 (979 words    — 0.10× English, proportional boost ≈ 10.2×)

Grid Search: sweep plausible ranges around these proportional estimates.
"""

import sys
import time
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path(__file__).parent))

from step3_optimized import build_base_tokenizer, make_trainer, lines_from_text, preprocess
from utils import fetch_all_languages, compute_fertility, MODELS_DIR, WIKI_PAGES
from tokenizers import trainers

VOCAB_SIZE = 10_000
MIN_FREQ   = 2
SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[BOS]", "[EOS]"]

def make_silent_trainer():
    """BpeTrainer with progress display suppressed."""
    return trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQ,
        special_tokens=SPECIAL_TOKENS,
        show_progress=False,
    )

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
BASELINE_3A = {"en": 1, "hi": 2, "te": 2, "kn": 2}   # Experiment 3A for comparison
BASELINE_3A_SCORE = 2396.89

# Grid to search — 1×2×3×3 = 18 combinations
EN_OPTIONS = [1]           # Always 1 — English is the reference
HI_OPTIONS = [1, 2]        # 1 = no boost (recommended), 2 = 3A baseline
TE_OPTIONS = [3, 4, 5]     # Inverse ratio ≈ 4.1×
KN_OPTIONS = [8, 10, 12]   # Inverse ratio ≈ 10.2×


def compute_score(ratios: dict) -> float:
    vals = list(ratios.values())
    spread = max(vals) - min(vals)
    return 1000.0 / spread if spread > 0 else float("inf")


def train_and_evaluate(cleaned_texts: dict, factors: dict, silent: bool = True) -> dict:
    """Train a BPE tokenizer with per-language oversampling and return metrics."""
    tokenizer = build_base_tokenizer()
    trainer   = make_silent_trainer() if silent else make_trainer()

    def iterator():
        for lang, factor in factors.items():
            for _ in range(factor):
                yield from lines_from_text(cleaned_texts[lang])

    tokenizer.train_from_iterator(iterator(), trainer=trainer)

    ratios = {lang: compute_fertility(tokenizer, text) for lang, text in cleaned_texts.items()}
    score  = compute_score(ratios)
    spread = max(ratios.values()) - min(ratios.values())

    return {
        "ratios":    ratios,
        "score":     score,
        "spread":    spread,
        "vocab":     tokenizer.get_vocab_size(),
        "tokenizer": tokenizer,
        "factors":   factors,
    }


def print_row(factors, result, tag=""):
    r = result["ratios"]
    en_f, hi_f, te_f, kn_f = factors["en"], factors["hi"], factors["te"], factors["kn"]
    tag_str = f"  ← {tag}" if tag else ""
    print(
        f"  en×{en_f} hi×{hi_f:>2} te×{te_f:>2} kn×{kn_f:>3}  |"
        f"  {result['score']:>9.2f}  |  {result['spread']:.4f}  |"
        f"  {r['en']:.4f}  {r['hi']:.4f}  {r['te']:.4f}  {r['kn']:.4f}{tag_str}"
    )


def grid_search(cleaned_texts: dict) -> tuple:
    combos = list(product(EN_OPTIONS, HI_OPTIONS, TE_OPTIONS, KN_OPTIONS))
    print(f"\n  Sweeping {len(combos)} combinations...")
    print(f"  {'Factors':<26}| {'Score':>10}  | {'Spread'}  | X_EN    X_HI    X_TE    X_KN")
    print("  " + "─" * 90)

    best_score  = 0.0
    best_result = None
    all_results = []

    for en_f, hi_f, te_f, kn_f in combos:
        factors = {"en": en_f, "hi": hi_f, "te": te_f, "kn": kn_f}
        result  = train_and_evaluate(cleaned_texts, factors, silent=True)
        all_results.append(result)

        tag = "★ NEW BEST" if result["score"] > best_score else ""
        print_row(factors, result, tag)

        if result["score"] > best_score:
            best_score  = result["score"]
            best_result = result

    return best_result, all_results


def main():
    print("=" * 65)
    print("  EXPERIMENT 3A1 — Differential Per-Language Oversampling")
    print("  Each language gets an oversampling factor proportional to")
    print("  its inverse corpus-size ratio relative to English.")
    print("=" * 65)

    # Load + preprocess
    print("\n[1] Loading and preprocessing data ...")
    texts = fetch_all_languages()
    cleaned_texts = preprocess(texts)

    # Baseline sanity check — re-run Experiment 3A
    print("\n[2] Baseline — Experiment 3A (uniform Indic ×2) ...")
    baseline = train_and_evaluate(cleaned_texts, BASELINE_3A, silent=True)
    b = baseline["ratios"]
    print(f"     Score:  {baseline['score']:.2f}  (expected ~{BASELINE_3A_SCORE:.0f})")
    print(f"     Spread: {baseline['spread']:.4f}")
    print(f"     X_EN={b['en']:.4f}  X_HI={b['hi']:.4f}  X_TE={b['te']:.4f}  X_KN={b['kn']:.4f}")

    # Grid search
    print("\n[3] Grid search — differential factors ...")
    t0 = time.time()
    best, all_results = grid_search(cleaned_texts)
    elapsed = time.time() - t0

    # Summary
    f = best["factors"]
    r = best["ratios"]
    improvement = best["score"] - baseline["score"]

    print("\n" + "=" * 65)
    print("  BEST RESULT — EXPERIMENT 3A1")
    print("=" * 65)
    print(f"  Factors:     en×{f['en']}  hi×{f['hi']}  te×{f['te']}  kn×{f['kn']}")
    print(f"  Vocab size:  {best['vocab']:,}")
    print(f"  X_EN:        {r['en']:.4f}")
    print(f"  X_HI:        {r['hi']:.4f}")
    print(f"  X_TE:        {r['te']:.4f}")
    print(f"  X_KN:        {r['kn']:.4f}")
    print(f"  Spread:      {best['spread']:.4f}")
    print(f"  Score:       {best['score']:.2f}")
    print(f"  vs 3A:       {improvement:+.2f} pts  ({'improvement' if improvement > 0 else 'regression'})")
    print(f"  Time:        {elapsed:.0f}s")
    print("=" * 65)

    # All results sorted by score (top 5)
    print("\n  Top 5 combinations:")
    sorted_results = sorted(all_results, key=lambda x: x["score"], reverse=True)
    print(f"  {'Factors':<26}| {'Score':>10}  | {'Spread'}  | X_EN    X_HI    X_TE    X_KN")
    print("  " + "─" * 90)
    for res in sorted_results[:5]:
        print_row(res["factors"], res)

    # Save best model
    model_path = MODELS_DIR / "step3a1_differential.json"
    best["tokenizer"].save(str(model_path))
    print(f"\n  ✓ Best model saved → {model_path}")

    # Return key numbers for updating the webapp
    print("\n  ── Copy these values into data.js ──")
    print(f"  en: {{ words: 10027, tokens: {round(r['en'] * 10027)}, ratio: {r['en']:.4f} }},")
    print(f"  hi: {{ words: 8022,  tokens: {round(r['hi'] * 8022)},  ratio: {r['hi']:.4f} }},")
    print(f"  te: {{ words: 2453,  tokens: {round(r['te'] * 2453)},  ratio: {r['te']:.4f} }},")
    print(f"  kn: {{ words: 979,   tokens: {round(r['kn'] * 979)},   ratio: {r['kn']:.4f} }},")
    x_min_lang = min(r, key=r.get)
    x_max_lang = max(r, key=r.get)
    print(f"  xMin: {r[x_min_lang]:.4f}, xMax: {r[x_max_lang]:.4f},")
    print(f"  spread: {best['spread']:.4f}, score: {best['score']:.2f},")
    print(f"  Training Data: 'en×{f['en']} · hi×{f['hi']} · te×{f['te']} · kn×{f['kn']}'")


if __name__ == "__main__":
    main()
