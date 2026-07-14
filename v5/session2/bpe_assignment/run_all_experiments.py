#!/usr/bin/env python3
"""
run_all_experiments.py
======================
Re-runs all experiments on the faithful Markdown corpus with Metaspace
tokenizer (satisfying the evaluator's faithfulness requirement).

Experiments:
  1  English-only BPE baseline
  2  Naive multilingual (equal weights)
  3A Oversampling en×1, hi×2, te×2, kn×2  (mild boost)
  3A1 Differential: en×1, hi×1, te×3, kn×6
  3A2 Focused sweet-spot: en×1, hi×1, te×2, kn×4  (best historical)
  3B  Merged vocabulary (4 independent tokenizers merged)

Each experiment:
  - Reads corpus from bpe_assignment/corpus/*.faithful.txt
  - Uses Metaspace pre-tokenizer + decoder (lossless round-trip)
  - Uses NFKC normalizer
  - Evaluates with faithful-unit denominator
  - Saves tokenizer to webapp/assets/models/

Run:
    python run_all_experiments.py
"""
from __future__ import annotations

import io
import sys
import json
import tempfile
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import regex
from tokenizers import Tokenizer
from tokenizers.decoders import Metaspace as MetaspaceDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.trainers import BpeTrainer

ROOT        = Path(__file__).resolve().parent
CORPUS      = ROOT / "corpus"
MODELS_OUT  = ROOT.parent / "webapp" / "assets" / "models"
MODELS_OUT.mkdir(parents=True, exist_ok=True)

LANGS = ["en", "hi", "te", "kn"]
LANG_NAMES = {"en": "English", "hi": "Hindi", "te": "Telugu", "kn": "Kannada"}

FAITHFUL_UNIT_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]")


def faithful_units(text: str) -> int:
    return len(FAITHFUL_UNIT_RE.findall(text))


def load_corpus() -> dict[str, str]:
    texts = {}
    for code in LANGS:
        path = CORPUS / f"{code}.faithful.txt"
        texts[code] = path.read_text(encoding="utf-8")
    return texts


def make_tokenizer() -> Tokenizer:
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.normalizer    = NFKC()
    tok.pre_tokenizer = Metaspace(replacement="\u2581", prepend_scheme="never")
    tok.decoder       = MetaspaceDecoder(replacement="\u2581", prepend_scheme="never")
    return tok


def train_from_weighted_files(texts: dict[str, str], weights: dict[str, int],
                               vocab_size: int = 10_000, min_freq: int = 1) -> Tokenizer:
    with tempfile.TemporaryDirectory() as tmp:
        files: list[str] = []
        tmpdir = Path(tmp)
        for code, text in texts.items():
            p = tmpdir / f"{code}.txt"
            p.write_text(text, encoding="utf-8")
            files.extend([str(p)] * weights.get(code, 1))

        tok = make_tokenizer()
        trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=min_freq,
                             special_tokens=["[UNK]"])
        tok.train(files, trainer)
    return tok


def evaluate(tok: Tokenizer, texts: dict[str, str]) -> dict:
    units  = {code: faithful_units(text) for code, text in texts.items()}
    toks   = {code: len(tok.encode(text).ids) for code, text in texts.items()}
    ratios = {code: toks[code] / units[code] for code in LANGS}
    spread = max(ratios.values()) - min(ratios.values())
    score  = 1000 / spread
    return {"units": units, "tokens": toks, "ratios": ratios,
            "spread": spread, "score": score, "vocab": tok.get_vocab_size()}


def print_result(name: str, res: dict) -> None:
    print(f"\n  [{name}]  score={res['score']:.2f}  spread={res['spread']:.4f}  vocab={res['vocab']:,}")
    for code in LANGS:
        print(f"    {LANG_NAMES[code]:<10}: {res['tokens'][code]:>8,} toks / {res['units'][code]:>8,} units = {res['ratios'][code]:.4f}")


def save(tok: Tokenizer, name: str) -> Path:
    p = MODELS_OUT / f"{name}.json"
    tok.save(str(p))
    return p


# ── Experiment 3B helper: merge N independent tokenizers ──────────────────────

def train_lang_tokenizer(text: str, vocab_budget: int) -> Tokenizer:
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "lang.txt"
        p.write_text(text, encoding="utf-8")
        tok = make_tokenizer()
        trainer = BpeTrainer(vocab_size=vocab_budget, min_frequency=1,
                             special_tokens=["[UNK]"])
        tok.train([str(p)], trainer)
    return tok


def merge_tokenizers(lang_tokenizers: dict[str, Tokenizer], total_vocab: int = 10_000) -> Tokenizer:
    merged_vocab: dict[str, int] = {}
    merged_merges: list[tuple[str, str]] = []
    seen: set[str] = set()

    # Special tokens first
    for i, tok_str in enumerate(["[UNK]"]):
        merged_vocab[tok_str] = i
        seen.add(tok_str)
    next_id = 1

    for code, tok in lang_tokenizers.items():
        path = Path(tempfile.mktemp(suffix=".json"))
        tok.save(str(path))
        data = json.loads(path.read_text(encoding="utf-8"))
        path.unlink()

        lang_vocab = data["model"]["vocab"]
        lang_merges = data["model"]["merges"]

        added = 0
        for token, _ in sorted(lang_vocab.items(), key=lambda x: x[1]):
            if token not in seen and next_id < total_vocab:
                merged_vocab[token] = next_id
                seen.add(token)
                next_id += 1
                added += 1

        for merge in lang_merges:
            if isinstance(merge, str):
                parts = merge.split(" ", 1)
                if len(parts) == 2:
                    pair = (parts[0], parts[1])
                    if pair not in merged_merges:
                        merged_merges.append(pair)
            elif isinstance(merge, (list, tuple)) and len(merge) == 2:
                pair = (merge[0], merge[1])
                if pair not in merged_merges:
                    merged_merges.append(pair)

        print(f"    {LANG_NAMES[code]:<10}: +{added:>4} tokens  (vocab now {next_id:,})")

    bpe = BPE(vocab=merged_vocab, merges=merged_merges, unk_token="[UNK]")
    merged_tok = Tokenizer(bpe)
    merged_tok.normalizer    = NFKC()
    merged_tok.pre_tokenizer = Metaspace(replacement="\u2581", prepend_scheme="never")
    merged_tok.decoder       = MetaspaceDecoder(replacement="\u2581", prepend_scheme="never")
    return merged_tok


# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 65)
    print("  Re-running all experiments on faithful Markdown corpus")
    print("  Using: Metaspace pre-tokenizer/decoder + NFKC normalizer")
    print("=" * 65)

    texts = load_corpus()
    units = {code: faithful_units(text) for code, text in texts.items()}
    print("\nCorpus faithful units:")
    for code in LANGS:
        print(f"  {LANG_NAMES[code]:<10}: {units[code]:>8,}")

    all_results: dict[str, dict] = {}

    # ── Experiment 1: English-only ───────────────────────────────
    print("\n[1] Experiment 1: English-only BPE ...")
    tok1 = train_from_weighted_files({"en": texts["en"]}, {"en": 1}, vocab_size=10_000)
    res1 = evaluate(tok1, texts)
    print_result("Exp1 EN-only", res1)
    save(tok1, "exp1_en_only")
    all_results["exp1"] = res1

    # ── Experiment 2: Naive multilingual (equal weights) ─────────
    print("\n[2] Experiment 2: Naive multilingual (en×1, hi×1, te×1, kn×1) ...")
    weights2 = {"en": 1, "hi": 1, "te": 1, "kn": 1}
    tok2 = train_from_weighted_files(texts, weights2)
    res2 = evaluate(tok2, texts)
    print_result("Exp2 Naive", res2)
    save(tok2, "exp2_naive")
    all_results["exp2"] = res2

    # ── Experiment 3A: Oversampling (en×1, hi×2, te×2, kn×2) ────
    print("\n[3] Experiment 3A: Oversampling (en×1, hi×2, te×2, kn×2) ...")
    weights3a = {"en": 1, "hi": 2, "te": 2, "kn": 2}
    tok3a = train_from_weighted_files(texts, weights3a)
    res3a = evaluate(tok3a, texts)
    print_result("Exp3A Oversample", res3a)
    save(tok3a, "exp3a_oversample")
    all_results["exp3a"] = res3a

    # ── Experiment 3A1: Differential (en×1, hi×1, te×3, kn×6) ───
    print("\n[4] Experiment 3A1: Differential (en×1, hi×1, te×3, kn×6) ...")
    weights3a1 = {"en": 1, "hi": 1, "te": 3, "kn": 6}
    tok3a1 = train_from_weighted_files(texts, weights3a1)
    res3a1 = evaluate(tok3a1, texts)
    print_result("Exp3A1 Differential", res3a1)
    save(tok3a1, "exp3a1_differential")
    all_results["exp3a1"] = res3a1

    # ── Experiment 3A2: Focused (en×1, hi×1, te×2, kn×4) ────────
    print("\n[5] Experiment 3A2: Focused sweet-spot (en×1, hi×1, te×2, kn×4) ...")
    weights3a2 = {"en": 1, "hi": 1, "te": 2, "kn": 4}
    tok3a2 = train_from_weighted_files(texts, weights3a2)
    res3a2 = evaluate(tok3a2, texts)
    print_result("Exp3A2 Focused", res3a2)
    save(tok3a2, "exp3a2_focused")
    all_results["exp3a2"] = res3a2

    # ── Experiment 3B: Merged vocabulary ─────────────────────────
    print("\n[6] Experiment 3B: Merged vocabulary (2500 per language) ...")
    per_lang_vocab = 2500
    lang_toks = {}
    for code in LANGS:
        print(f"    Training {LANG_NAMES[code]} ({per_lang_vocab} budget) ...")
        lang_toks[code] = train_lang_tokenizer(texts[code], per_lang_vocab)
    print("  Merging ...")
    tok3b = merge_tokenizers(lang_toks)
    res3b = evaluate(tok3b, texts)
    print_result("Exp3B Merged", res3b)
    save(tok3b, "exp3b_merged")
    all_results["exp3b"] = res3b

    # ── Copy final tokenizer.json to assets/models too ───────────
    import shutil
    final_tok = ROOT / "tokenizer.json"
    if final_tok.exists():
        shutil.copy(final_tok, MODELS_OUT / "tokenizer.json")
        print(f"\n  Copied tokenizer.json -> {MODELS_OUT/'tokenizer.json'}")

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  SUMMARY — All Experiments (faithful-unit fertility)")
    print("=" * 65)
    print(f"  {'Experiment':<20} {'Score':>9}  {'Spread':>7}  {'X_EN':>7}  {'X_HI':>7}  {'X_TE':>7}  {'X_KN':>7}")
    print("  " + "-" * 63)
    names = {
        "exp1": "Exp1 EN-only",
        "exp2": "Exp2 Naive",
        "exp3a": "Exp3A Oversample",
        "exp3a1": "Exp3A1 Differential",
        "exp3a2": "Exp3A2 Focused",
        "exp3b": "Exp3B Merged",
    }
    for key, name in names.items():
        r = all_results[key]
        print(f"  {name:<20} {r['score']:>9.2f}  {r['spread']:>7.4f}"
              f"  {r['ratios']['en']:>7.4f}  {r['ratios']['hi']:>7.4f}"
              f"  {r['ratios']['te']:>7.4f}  {r['ratios']['kn']:>7.4f}")
    print("=" * 65)

    # ── Output JSON for copy into data.js ─────────────────────────
    out_path = ROOT / "experiment_results.json"
    out_path.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n  Results saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
