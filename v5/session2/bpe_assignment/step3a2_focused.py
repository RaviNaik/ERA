"""
step3a2_focused.py — Experiment 3A2: Focused Sweet-Spot Search
================================================================
Hypothesis: The spread in Experiment 3A (score 2397) is driven by Hindi being
over-boosted at ×2 (X_HI=1.13, which is X_min). From Experiment 3A1 we know
that hi×1 brings X_HI up to 1.38 — right in the cluster with EN/TE/KN.

If we fix hi×1, keep te×2 (which gave X_TE=1.49 in 3A), and sweep kn from
×2 to ×6 to find where X_KN meets the cluster:

  en×1, hi×1, te×2, kn×[2, 3, 4, 5, 6]

Predicted ratios (estimated):
  X_EN ≈ 1.51 (stable — not being oversampled)
  X_HI ≈ 1.38 (from 3A1 observation at hi×1)
  X_TE ≈ 1.49 (from 3A observation at te×2)
  X_KN ≈ decreasing as kn increases from 1.55 → ?

If X_KN drops to ~1.43, spread = 1.51-1.38 = 0.13 → Score ≈ 7,700.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from step3_optimized import build_base_tokenizer, lines_from_text, preprocess
from utils import fetch_all_languages, compute_fertility, MODELS_DIR
from tokenizers import trainers

VOCAB_SIZE     = 10_000
MIN_FREQ       = 2
SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[BOS]", "[EOS]"]

# Reference baselines
REF_3A  = {"factors": {"en":1,"hi":2,"te":2,"kn":2},  "score": 2396.89}
REF_3A1 = {"factors": {"en":1,"hi":1,"te":3,"kn":12}, "score": 1910.27}


def make_trainer(silent=True):
    return trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQ,
        special_tokens=SPECIAL_TOKENS,
        show_progress=not silent,
    )


def compute_score(ratios):
    vals = list(ratios.values())
    spread = max(vals) - min(vals)
    return 1000.0 / spread if spread > 0 else float("inf")


def train_and_evaluate(cleaned_texts, factors, silent=True):
    tokenizer = build_base_tokenizer()
    trainer   = make_trainer(silent=silent)

    def iterator():
        for lang, n in factors.items():
            for _ in range(n):
                yield from lines_from_text(cleaned_texts[lang])

    tokenizer.train_from_iterator(iterator(), trainer=trainer)

    ratios = {lang: compute_fertility(tokenizer, text)
              for lang, text in cleaned_texts.items()}
    score  = compute_score(ratios)
    spread = max(ratios.values()) - min(ratios.values())

    return {"ratios": ratios, "score": score, "spread": spread,
            "vocab": tokenizer.get_vocab_size(), "tokenizer": tokenizer,
            "factors": factors}


def print_header():
    print(f"\n  {'Factors':<30} | {'Score':>9}  | {'Spread':>7} |"
          f"  {'X_EN':>7}  {'X_HI':>7}  {'X_TE':>7}  {'X_KN':>7}")
    print("  " + "─" * 88)


def print_row(result, tag=""):
    f = result["factors"]
    r = result["ratios"]
    label = f"en×{f['en']} hi×{f['hi']} te×{f['te']} kn×{f['kn']}"
    tag_s = f"  ← {tag}" if tag else ""
    print(f"  {label:<30} | {result['score']:>9.2f}  | {result['spread']:>7.4f} |"
          f"  {r['en']:>7.4f}  {r['hi']:>7.4f}  {r['te']:>7.4f}  {r['kn']:>7.4f}{tag_s}")


def main():
    print("=" * 65)
    print("  EXPERIMENT 3A2 — Focused Sweet-Spot Search")
    print("  Hypothesis: hi×1 + te×2 with kn sweep beats 3A's 2397")
    print("=" * 65)

    print("\n[1] Loading and preprocessing ...")
    texts         = fetch_all_languages()
    cleaned_texts = preprocess(texts)

    # ── Primary sweep: fix en×1, hi×1, te×2, vary kn ────────────
    # kn×2 = 3A without the Hindi boost
    # kn×3, ×4 = progressively reduce X_KN toward the cluster
    sweep_configs = [
        {"en": 1, "hi": 1, "te": 2, "kn": 2},
        {"en": 1, "hi": 1, "te": 2, "kn": 3},
        {"en": 1, "hi": 1, "te": 2, "kn": 4},
        {"en": 1, "hi": 1, "te": 2, "kn": 5},
        {"en": 1, "hi": 1, "te": 2, "kn": 6},
        # Also test te×3 with moderate kn to see Telugu's effect
        {"en": 1, "hi": 1, "te": 3, "kn": 2},
        {"en": 1, "hi": 1, "te": 3, "kn": 3},
        {"en": 1, "hi": 1, "te": 3, "kn": 4},
    ]

    print(f"\n[2] Running {len(sweep_configs)} sweep configurations ...")
    print_header()

    best_score  = 0.0
    best_result = None
    all_results = []

    t0 = time.time()
    for factors in sweep_configs:
        result = train_and_evaluate(cleaned_texts, factors, silent=True)
        all_results.append(result)

        vs_3a = result["score"] - REF_3A["score"]
        tag = f"{'▲ +' if vs_3a >= 0 else '▼ '}{abs(vs_3a):.0f} vs 3A"
        if result["score"] > best_score:
            tag = "★ NEW BEST  " + tag
            best_score  = result["score"]
            best_result = result

        print_row(result, tag)

    elapsed = time.time() - t0

    # ── Summary ────────────────────────────────────────────────
    f = best_result["factors"]
    r = best_result["ratios"]
    delta = best_result["score"] - REF_3A["score"]

    print("\n" + "=" * 65)
    print("  BEST RESULT — EXPERIMENT 3A2")
    print("=" * 65)
    print(f"  Factors  :  en×{f['en']} · hi×{f['hi']} · te×{f['te']} · kn×{f['kn']}")
    print(f"  Vocab    :  {best_result['vocab']:,}")
    print(f"  X_EN     :  {r['en']:.4f}")
    print(f"  X_HI     :  {r['hi']:.4f}")
    print(f"  X_TE     :  {r['te']:.4f}")
    print(f"  X_KN     :  {r['kn']:.4f}")
    print(f"  Spread   :  {best_result['spread']:.4f}")
    print(f"  Score    :  {best_result['score']:.2f}")
    print(f"  vs 3A    :  {delta:+.2f} pts  "
          f"({'IMPROVEMENT ✓' if delta > 0 else 'regression ✗'})")
    print(f"  Time     :  {elapsed:.0f}s")
    print("=" * 65)

    # ── Ranked table ───────────────────────────────────────────
    print("\n  All results ranked by score:")
    print_header()
    for res in sorted(all_results, key=lambda x: x["score"], reverse=True):
        print_row(res)

    # ── Save best model ────────────────────────────────────────
    model_path = MODELS_DIR / "step3a2_focused.json"
    best_result["tokenizer"].save(str(model_path))
    print(f"\n  ✓ Best model saved → {model_path}")

    # ── data.js copy-paste output ──────────────────────────────
    x_min_lang = min(r, key=r.get)
    x_max_lang = max(r, key=r.get)
    print("\n  ── Paste into data.js ──")
    print(f"  en: {{ words: 10027, tokens: {round(r['en']*10027)}, ratio: {r['en']:.4f} }},")
    print(f"  hi: {{ words: 8022,  tokens: {round(r['hi']*8022)},  ratio: {r['hi']:.4f} }},")
    print(f"  te: {{ words: 2453,  tokens: {round(r['te']*2453)},  ratio: {r['te']:.4f} }},")
    print(f"  kn: {{ words: 979,   tokens: {round(r['kn']*979)},   ratio: {r['kn']:.4f} }},")
    print(f"  xMin: {r[x_min_lang]:.4f}, xMax: {r[x_max_lang]:.4f},")
    print(f"  spread: {best_result['spread']:.4f}, score: {best_result['score']:.2f},")
    print(f"  Training Data: 'en×{f['en']} · hi×{f['hi']} · te×{f['te']} · kn×{f['kn']}'")


if __name__ == "__main__":
    main()
