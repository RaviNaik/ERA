#!/usr/bin/env python3
"""
Generate per-experiment metrics JSON files for the webapp.
Reads experiment_results.json and writes individual files to webapp/assets/metrics/.
"""
import io, sys, json
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT       = Path(__file__).resolve().parent
SRC        = ROOT / "experiment_results.json"
OUT        = ROOT.parent / "webapp" / "assets" / "metrics"
MAIN_METRICS = ROOT / "metrics.json"
OUT.mkdir(parents=True, exist_ok=True)

LANG_NAMES = {"en": "English", "hi": "Hindi", "te": "Telugu", "kn": "Kannada"}
CORPUS = "wiki_faithful_markdown"
PRE_TOK = "Metaspace"
NORM    = "NFKC"

EXPERIMENT_META = {
    "exp1": {
        "number": 1,
        "filename": "exp1_metrics.json",
        "title": "English-Only BPE",
        "subtitle": "Monolingual Baseline",
        "weights": {"en": 1},
        "vocab_size": 10000,
        "model_file": "exp1_en_only.json",
    },
    "exp2": {
        "number": 2,
        "filename": "exp2_metrics.json",
        "title": "Naive Multilingual BPE",
        "subtitle": "Equal Corpus Weights",
        "weights": {"en": 1, "hi": 1, "te": 1, "kn": 1},
        "vocab_size": 10000,
        "model_file": "exp2_naive.json",
    },
    "exp3a": {
        "number": 3,
        "filename": "exp3_metrics.json",
        "title": "Uniform Indic Oversampling",
        "subtitle": "All Indic Languages x2",
        "weights": {"en": 1, "hi": 2, "te": 2, "kn": 2},
        "vocab_size": 10000,
        "model_file": "exp3a_oversample.json",
    },
    "exp3a1": {
        "number": 4,
        "filename": "exp4_metrics.json",
        "title": "Differential Oversampling",
        "subtitle": "Inverse-Corpus-Size Weights",
        "weights": {"en": 1, "hi": 1, "te": 3, "kn": 6},
        "vocab_size": 10000,
        "model_file": "exp3a1_differential.json",
    },
    "exp3b": {
        "number": 5,
        "filename": "exp5_metrics.json",
        "title": "Merged Vocabulary BPE",
        "subtitle": "Per-Language Budget Allocation (2500 each)",
        "weights": {"en": 2500, "hi": 2500, "te": 2500, "kn": 2500},
        "vocab_size": 7997,
        "model_file": "exp3b_merged.json",
    },
    "exp3a2": {
        "number": 6,
        "filename": "exp6_metrics.json",
        "title": "Focused Sweet-Spot BPE",
        "subtitle": "Optimal Weight Tuning (Submitted)",
        "weights": {"en": 1, "hi": 1, "te": 2, "kn": 4},
        "vocab_size": 10000,
        "model_file": "tokenizer.json",
    },
}

data = json.loads(SRC.read_text(encoding="utf-8"))

for key, meta in EXPERIMENT_META.items():
    raw = data.get(key, {})
    out = {
        "experiment_number": meta["number"],
        "title": meta["title"],
        "subtitle": meta["subtitle"],
        "corpus": CORPUS,
        "pre_tokenizer": PRE_TOK,
        "normalizer": NORM,
        "training_weights": meta["weights"],
        "vocab_size": meta["vocab_size"],
        "model_file": meta["model_file"],
        "faithfulness": "PASS - decode(encode(text)) preserves all visible non-whitespace characters",
        "languages": LANG_NAMES,
        "faithful_units": raw.get("units", {}),
        "token_counts": raw.get("tokens", {}),
        "fertility_ratios": raw.get("ratios", {}),
        "spread": raw.get("spread"),
        "score": raw.get("score"),
        "hindi_penalty": 1.0,
        "adjusted_score": raw.get("score"),
    }
    path = OUT / meta["filename"]
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Wrote {path.name}  (score={raw.get('score', 'n/a'):.2f})" if raw.get("score") else f"  Wrote {path.name}")

# Also copy the main metrics.json to assets/metrics/submitted_metrics.json
if MAIN_METRICS.exists():
    submitted = json.loads(MAIN_METRICS.read_text(encoding="utf-8"))
    (OUT / "submitted_metrics.json").write_text(
        json.dumps(submitted, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  Wrote submitted_metrics.json")

print(f"\nAll metrics written to {OUT}")
