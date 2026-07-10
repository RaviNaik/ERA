"""
utils.py — Shared utilities for BPE Tokenizer Assignment
=========================================================
Contains helpers for:
  - Fetching India's Wikipedia page in exactly 4 languages
    (only the 4 specified pages, nothing more)
  - Computing the fertility ratio (total tokens / total whitespace words)
  - Printing a formatted report of X1..X4 and the assignment score

Data sources (fixed — no other pages allowed):
  English : https://en.wikipedia.org/wiki/India
  Hindi   : https://hi.wikipedia.org/wiki/भारत
  Telugu  : https://te.wikipedia.org/wiki/భారతదేశం
  Kannada : https://kn.wikipedia.org/wiki/ಭಾರತ
"""

import re
import time
import unicodedata
import requests
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
DATA_DIR  = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# The ONLY 4 pages allowed for this assignment.
# Format: lang_key → (lang_code, wikipedia_title, display_name)
WIKI_PAGES = {
    "en": ("en", "India",        "English"),
    "hi": ("hi", "भारत",         "Hindi"),
    "te": ("te", "భారతదేశం",     "Telugu"),
    "kn": ("kn", "ಭಾರತ",         "Kannada"),
}

HEADERS = {
    "User-Agent": "BPE-Tokenizer-Assignment/1.0 (ERA-Session2; educational) python-requests"
}


# ──────────────────────────────────────────────────────────────
# Wikipedia Fetching
# ──────────────────────────────────────────────────────────────

def fetch_wikipedia(lang_code: str, title: str) -> str:
    """
    Download the plain-text extract of one Wikipedia article via the
    MediaWiki Extracts API. Retries on 429 rate-limit with exponential backoff.
    Returns the full plain-text content of the article.
    """
    url = f"https://{lang_code}.wikipedia.org/w/api.php"
    params = {
        "action":          "query",
        "prop":            "extracts",
        "explaintext":     True,         # plain text, no HTML markup
        "exsectionformat": "plain",      # section headers as plain text
        "titles":          title,
        "format":          "json",
        "redirects":       1,
    }
    for attempt in range(5):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if resp.status_code == 429:
                wait = 8 * (2 ** attempt)
                print(f"  [rate-limit] 429 for '{title}' — waiting {wait}s before retry...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            pages = resp.json()["query"]["pages"]
            page  = next(iter(pages.values()))
            if "missing" in page:
                raise ValueError(f"Article '{title}' not found on {lang_code}.wikipedia.org")
            return page.get("extract", "")
        except requests.exceptions.RequestException as exc:
            if attempt < 4:
                time.sleep(4 * (attempt + 1))
            else:
                raise RuntimeError(f"Failed to fetch '{title}' after 5 attempts: {exc}") from exc
    return ""


def fetch_all_languages(force: bool = False) -> dict[str, str]:
    """
    Download all 4 Wikipedia pages (India in EN/HI/TE/KN).
    Results are cached to data/india_{lang}.txt.
    Set force=True to re-download even if cache exists.

    NOTE: The data imbalance (EN ≈ 65K chars, KN ≈ 7.6K chars) is REAL —
    it reflects how developed each Wikipedia edition is, not an API issue.
    The assignment's scoring formula and the Step 3 optimizations (oversampling,
    merged vocab) are designed to handle this constraint.
    """
    texts = {}
    for lang, (lang_code, title, name) in WIKI_PAGES.items():
        out_path = DATA_DIR / f"india_{lang}.txt"
        if out_path.exists() and not force:
            print(f"  [cache] {out_path.name} already exists, skipping download.")
            text = out_path.read_text(encoding="utf-8")
        else:
            print(f"  [fetch] Downloading '{title}' from {lang_code}.wikipedia.org ...")
            text = fetch_wikipedia(lang_code, title)
            out_path.write_text(text, encoding="utf-8")
            print(f"  [fetch] Saved {len(text):,} chars → {out_path.name}")
        texts[lang] = text
        words = len(text.split())
        print(f"  [{name}] {len(text):,} chars | {words:,} words")
    return texts


def load_text(lang: str) -> str:
    """Load a cached Wikipedia text from disk."""
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
    Count total whitespace-split words.
    This is the denominator for the fertility ratio.
    """
    return len(text.split())


def compute_fertility(tokenizer, text: str) -> float:
    """
    Fertility ratio = (subword tokens produced) / (whitespace words).
    Lower is better: the tokenizer represents each word with fewer pieces.
    Encodes in chunks of 1000 words to avoid memory issues.
    """
    words = text.split()
    total_tokens = 0
    chunk_size = 1000
    for i in range(0, len(words), chunk_size):
        chunk   = " ".join(words[i : i + chunk_size])
        encoded = tokenizer.encode(chunk)
        total_tokens += len(encoded.ids)
    total_words = len(words)
    return (total_tokens / total_words) if total_words > 0 else float("inf")


# ──────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────

def report_metrics(ratios: dict[str, float], step_name: str = "") -> None:
    """Print a formatted table of fertility ratios and the assignment score."""
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
    print(f"  {'Language':<18} {'Ratio':>10}")
    print("-" * 55)
    for lang, ratio in ratios.items():
        print(f"  {lang_names.get(lang, lang):<18} {ratio:>10.4f}")
    print("-" * 55)
    sorted_ratios = sorted(ratios.values())
    x_min  = sorted_ratios[0]
    x_max  = sorted_ratios[-1]
    spread = x_max - x_min
    score  = (1000.0 / spread) if spread > 0 else float("inf")
    print(f"  Best  (min X): {x_min:.4f}")
    print(f"  Worst (max X): {x_max:.4f}")
    print(f"  Spread (X_max - X_min): {spread:.4f}")
    print(f"  *** SCORE = 1000 / {spread:.4f} = {score:.2f} ***")
    print("=" * 55)
    print()


def report_word_token_details(tokenizer, texts: dict[str, str]) -> dict[str, float]:
    """
    Compute and print fertility ratios for all provided language texts.
    Returns the ratios dict {lang: ratio}.
    """
    ratios = {}
    for lang, text in texts.items():
        words  = count_words(text)
        ratio  = compute_fertility(tokenizer, text)
        tokens = int(ratio * words)
        name   = WIKI_PAGES.get(lang, (lang, lang, lang))[2]
        print(f"  {name:<10}: {words:>7,} words → {tokens:>8,} tokens  (ratio = {ratio:.4f})")
        ratios[lang] = ratio
    return ratios


# ──────────────────────────────────────────────────────────────
# Text Cleaning (used in Step 3)
# ──────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """
    Apply NFKC Unicode normalization and strip invisible joiners.
    Used in the optimized tokenizer (Step 3).

    Steps:
      1. NFKC normalization — collapses compatibility equivalents
      2. Remove ZWJ (U+200D) and ZWNJ (U+200C) — invisible chars that cause
         the same visible Indic word to have multiple byte representations,
         wasting vocabulary slots on duplicates.
      3. Strip other Unicode control/format chars (keep whitespace)
      4. Collapse repeated whitespace
    """
    # 1. NFKC
    text = unicodedata.normalize("NFKC", text)
    # 2. Zero-width joiners / non-joiners
    text = text.replace("\u200d", "").replace("\u200c", "")
    # 3. Other control/format chars
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch)[0] != "C" or ch in ("\n", "\t", " ")
    )
    # 4. Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
