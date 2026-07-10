"""
utils.py — Shared utilities for BPE Tokenizer Assignment
=========================================================
Contains helpers for:
  - Fetching India's Wikipedia page in multiple languages
  - Computing the fertility ratio (total tokens / total whitespace words)
  - Printing a formatted report of X1..X4 and the assignment score
"""

import os
import re
import unicodedata
import requests
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Wikipedia language codes → (lang_code, page_title, short_name)
WIKI_PAGES = {
    "en": ("en", "India", "English"),
    "hi": ("hi", "भारत", "Hindi"),
    "te": ("te", "భారతదేశం", "Telugu"),
    "kn": ("kn", "ಭಾರತ", "Kannada"),
}


# ──────────────────────────────────────────────────────────────
# Wikipedia Fetching
# ──────────────────────────────────────────────────────────────

def fetch_wikipedia(lang_code: str, title: str, out_path: Path, force: bool = False) -> str:
    """
    Download the plain-text extract of a Wikipedia article via the API.
    Saves the text to `out_path`. Returns the text.

    Uses the `extracts` API which returns clean plain text (no markup).
    """
    if out_path.exists() and not force:
        print(f"  [cache] {out_path.name} already exists, skipping download.")
        return out_path.read_text(encoding="utf-8")

    print(f"  [fetch] Downloading '{title}' from {lang_code}.wikipedia.org ...")
    url = f"https://{lang_code}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,          # plain text, no HTML
        "exsectionformat": "plain",
        "titles": title,
        "format": "json",
        "redirects": 1,
    }
    headers = {
        "User-Agent": "BPE-Tokenizer-Assignment/1.0 (ERA-Session2; educational project) python-requests"
    }
    resp = requests.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    pages = data["query"]["pages"]
    page = next(iter(pages.values()))
    text = page.get("extract", "")
    if not text:
        raise ValueError(f"Empty extract for '{title}' on {lang_code}.wikipedia.org")

    out_path.write_text(text, encoding="utf-8")
    print(f"  [fetch] Saved {len(text):,} chars → {out_path}")
    return text


def fetch_all_languages(force: bool = False) -> dict[str, str]:
    """Download all 4 language Wikipedia pages for India. Returns {lang: text}."""
    texts = {}
    for lang, (lang_code, title, name) in WIKI_PAGES.items():
        out_path = DATA_DIR / f"india_{lang}.txt"
        text = fetch_wikipedia(lang_code, title, out_path, force=force)
        texts[lang] = text
        print(f"  [{name}] {len(text):,} chars")
    return texts


def load_text(lang: str) -> str:
    """Load a previously downloaded Wikipedia text from disk."""
    path = DATA_DIR / f"india_{lang}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"Text file not found: {path}\n"
            "Run fetch_all_languages() first."
        )
    return path.read_text(encoding="utf-8")


# ──────────────────────────────────────────────────────────────
# Text helpers
# ──────────────────────────────────────────────────────────────

def count_words(text: str) -> int:
    """
    Count total words by whitespace splitting.
    This is the denominator for the fertility ratio.
    """
    return len(text.split())


def compute_fertility(tokenizer, text: str) -> float:
    """
    Fertility ratio = (number of subword tokens produced) / (number of whitespace words).
    Lower is better — it means the tokenizer represents each word with fewer pieces.

    Args:
        tokenizer: A trained HuggingFace `tokenizers.Tokenizer` object.
        text:      The raw input text.

    Returns:
        float: fertility ratio X_i
    """
    # HuggingFace tokenizer may limit batch size; encode in chunks to be safe
    words = text.split()
    total_tokens = 0

    # Encode in batches of 1000 words to avoid memory issues
    chunk_size = 1000
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        encoded = tokenizer.encode(chunk)
        total_tokens += len(encoded.ids)

    total_words = len(words)
    if total_words == 0:
        return float("inf")

    return total_tokens / total_words


# ──────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────

def report_metrics(ratios: dict[str, float], step_name: str = "") -> None:
    """
    Print a formatted table of fertility ratios and the assignment score.

    Args:
        ratios: dict like {"en": 1.15, "hi": 1.40, "te": 1.55, "kn": 1.60}
        step_name: label for the report header
    """
    lang_names = {
        "en": "English (X1)",
        "hi": "Hindi   (X2)",
        "te": "Telugu  (X3)",
        "kn": "Kannada (X4)",
    }

    print()
    print("=" * 55)
    if step_name:
        print(f"  {step_name}")
        print("=" * 55)

    print(f"  {'Language':<18} {'Ratio':>10}  {'Words':>10}  {'Tokens':>10}")
    print("-" * 55)

    sorted_items = sorted(ratios.items(), key=lambda x: x[1])
    for lang, ratio in ratios.items():
        print(f"  {lang_names.get(lang, lang):<18} {ratio:>10.4f}")

    print("-" * 55)
    sorted_ratios = sorted(ratios.values())
    x_min = sorted_ratios[0]
    x_max = sorted_ratios[-1]
    spread = x_max - x_min

    if spread > 0:
        score = 1000.0 / spread
    else:
        score = float("inf")

    print(f"  Best  (min X): {x_min:.4f}")
    print(f"  Worst (max X): {x_max:.4f}")
    print(f"  Spread (X_max - X_min): {spread:.4f}")
    print(f"  *** SCORE = 1000 / {spread:.4f} = {score:.2f} ***")
    print("=" * 55)
    print()


def report_word_token_details(tokenizer, texts: dict[str, str]) -> dict[str, float]:
    """
    Compute and report fertility ratios for all provided language texts.
    Returns the ratios dict.
    """
    ratios = {}
    for lang, text in texts.items():
        words = count_words(text)
        ratio = compute_fertility(tokenizer, text)
        tokens = int(ratio * words)
        lang_name = WIKI_PAGES.get(lang, (lang, lang, lang))[2]
        print(f"  {lang_name:<10}: {words:>7,} words → {tokens:>8,} tokens  (ratio = {ratio:.4f})")
        ratios[lang] = ratio
    return ratios


# ──────────────────────────────────────────────────────────────
# Text Cleaning (used in Step 3)
# ──────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """
    Apply NFKC Unicode normalization and strip invisible joiners.
    Used in the optimized tokenizer (Step 3) to reduce token fragmentation.

    Steps:
      1. NFKC normalization — collapses compatibility equivalents
      2. Remove ZWJ (U+200D) and ZWNJ (U+200C) — invisible but alter tokenization
      3. Remove other control/format characters (category C*)
      4. Collapse repeated whitespace
    """
    # 1. NFKC normalize
    text = unicodedata.normalize("NFKC", text)

    # 2. Remove zero-width joiners/non-joiners
    text = text.replace("\u200d", "").replace("\u200c", "")

    # 3. Remove other Unicode control/format chars (keep newlines for word splitting)
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch)[0] != "C" or ch in ("\n", "\t", " ")
    )

    # 4. Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
