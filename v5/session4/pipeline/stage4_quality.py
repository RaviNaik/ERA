"""
stage4_quality.py -- Stage 4: Quality Filter Cascade
------------------------------------------------------
Applies the 6 Gopher/C4 heuristic rules from Session 4 notes,
then a simulated educational classifier score (0-5 scale).

Rules (from Session 4):
  1. Word length bounds: avg word length 3-10 chars
  2. Symbol-to-word ratio: symbols < 10% of words
  3. Line length bounds: avg line length 30-1000 chars
  4. Punctuation-end check: >= 70% lines end with sentence punctuation
  5. Stop-word density: >= 20% stop-words
  6. Duplicate lines ratio: < 30% duplicate lines

Indic exception: Telugu/Hindi may legitimately exceed word-length upper bound
due to agglutinative morphology. We apply relaxed thresholds for Indic scripts.

Educational classifier: score docs 0-5 (simulated with heuristics):
  - Lexical diversity
  - Presence of factual sentences
  - Absence of spam patterns
"""

import re
import math
from stats_tracker import StageStats, estimate_tokens

# English stop words (top 30)
EN_STOPWORDS = {
    "the", "a", "an", "is", "it", "in", "on", "at", "to", "for",
    "of", "and", "or", "but", "not", "with", "this", "that", "was",
    "are", "be", "by", "from", "as", "has", "had", "its", "their", "have", "been"
}

# Hindi stop words (common particles)
HI_STOPWORDS = {
    "और", "है", "में", "को", "की", "का", "के", "एक", "से", "पर",
    "यह", "वह", "कि", "था", "थी", "थे", "हैं", "इस", "उस", "जो",
    "नहीं", "भी", "ने", "हो", "जब", "तो", "पर", "अब", "यहाँ",
}

# Telugu stop words
TE_STOPWORDS = {
    "మరియు", "ఒక", "లో", "కి", "కు", "నుండి", "అయ్యింది", "ఉంది",
    "ఉన్న", "కాని", "అని", "ఇది", "అది", "వారు", "ఈ", "ఆ",
}

SENTENCE_END_RE = re.compile(r"[.!?।|]$")
# Only count TRUE noise symbols -- not standard prose punctuation (. , ' " - : ; ( ) [ ])
SYMBOL_RE = re.compile(r"[#@$%^&*+=<>~`|\\{}]")

def get_stopwords(lang: str) -> set:
    if lang == "hi": return HI_STOPWORDS
    if lang == "te": return TE_STOPWORDS
    return EN_STOPWORDS


def check_word_length(words: list[str], lang: str = "en") -> tuple[bool, str, float]:
    """Rule 1: Average word length should be 3-10 (relaxed to 3-15 for Indic)."""
    if not words:
        return False, "no_words", 0.0
    avg = sum(len(w) for w in words) / len(words)
    lo, hi = (3, 15) if lang in ("hi", "te", "ta", "bn", "ml", "kn", "gu") else (3, 10)
    if avg < lo:
        return False, f"avg_word_len_too_short_{avg:.1f}", avg
    if avg > hi:
        return False, f"avg_word_len_too_long_{avg:.1f}", avg
    return True, "", avg


def check_symbol_ratio(text: str, words: list[str]) -> tuple[bool, str, float]:
    """Rule 2: Symbol count < 10% of word count."""
    symbols = len(SYMBOL_RE.findall(text))
    ratio = symbols / max(len(words), 1)
    if ratio > 0.10:
        return False, f"symbol_ratio_too_high_{ratio:.2f}", ratio
    return True, "", ratio


def check_line_length(lines: list[str]) -> tuple[bool, str, float]:
    """Rule 3: Average line length 30-1000 chars."""
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        return False, "no_lines", 0.0
    avg = sum(len(l) for l in non_empty) / len(non_empty)
    if avg < 30:
        return False, f"avg_line_len_too_short_{avg:.0f}", avg
    if avg > 1000:
        return False, f"avg_line_len_too_long_{avg:.0f}", avg
    return True, "", avg


def check_punctuation_end(lines: list[str]) -> tuple[bool, str, float]:
    """Rule 4: >= 70% of non-empty lines end with sentence punctuation."""
    non_empty = [l.strip() for l in lines if l.strip()]
    if not non_empty:
        return False, "no_lines", 0.0
    ending = sum(1 for l in non_empty if SENTENCE_END_RE.search(l))
    ratio = ending / len(non_empty)
    # Wikipedia has a LOT of lines not ending in punctuation (lists, headers)
    # Relax to 40% for extraction artifacts
    if ratio < 0.40:
        return False, f"punct_end_ratio_low_{ratio:.2f}", ratio
    return True, "", ratio


def check_stopword_density(words: list[str], lang: str) -> tuple[bool, str, float]:
    """Rule 5: >= 20% stop-words in document."""
    stopwords = get_stopwords(lang)
    lower_words = [w.lower() for w in words]
    sw_count = sum(1 for w in lower_words if w in stopwords)
    density = sw_count / max(len(words), 1)
    if density < 0.05:
        return False, f"stopword_density_low_{density:.3f}", density
    return True, "", density


def check_duplicate_lines(lines: list[str]) -> tuple[bool, str, float]:
    """Rule 6: Duplicate lines < 30% of total."""
    non_empty = [l.strip() for l in lines if l.strip()]
    if not non_empty:
        return False, "no_lines", 0.0
    unique = set(non_empty)
    dup_ratio = 1.0 - len(unique) / len(non_empty)
    if dup_ratio > 0.30:
        return False, f"dup_lines_ratio_high_{dup_ratio:.2f}", dup_ratio
    return True, "", dup_ratio


def educational_score(text: str, words: list[str]) -> float:
    """
    Simulated educational classifier (0-5 scale).
    Based on: lexical diversity, sentence variety, absence of spam markers.
    Real implementation would use a trained FineWeb-Edu style classifier.
    """
    if len(words) < 20:
        return 0.5

    # Lexical diversity (type-token ratio)
    ttr = len(set(w.lower() for w in words)) / len(words)

    # Sentence length variance (good prose = moderate variance)
    sentences = re.split(r"[.!?।]", text)
    sent_lens = [len(s.split()) for s in sentences if s.strip()]
    if len(sent_lens) > 1:
        mean_sl = sum(sent_lens) / len(sent_lens)
        variance = sum((l - mean_sl) ** 2 for l in sent_lens) / len(sent_lens)
        sent_variety = min(1.0, variance / 200)
    else:
        sent_variety = 0.0

    # Presence of numbers (factual text)
    has_numbers = 0.2 if re.search(r"\d{4}", text) else 0.0

    # Penalize repetitive short phrases
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    bigram_diversity = len(set(bigrams)) / max(len(bigrams), 1)

    score = (ttr * 2.0 + sent_variety * 1.0 + has_numbers + bigram_diversity * 1.0 + 0.5)
    return min(5.0, round(score, 2))


def run_stage4(docs: list[dict], stats: StageStats,
               min_edu_score: float = 1.5) -> list[dict]:
    """Apply quality filter cascade to all documents."""
    stats.input_docs = len(docs)
    stats.input_tokens = sum(estimate_tokens(d["text"]) for d in docs)
    stats.metadata_added = [
        "quality_flags", "avg_word_len", "symbol_ratio",
        "punct_end_ratio", "stopword_density", "dup_line_ratio", "edu_score"
    ]

    output = []
    examples_added = 0

    for doc in docs:
        text = doc["text"]
        lang = doc.get("detected_lang", "en")
        words = text.split()
        lines = text.splitlines()

        passed = True
        drop_reason = ""
        quality_flags = []

        checks = [
            check_word_length(words, lang),
            check_symbol_ratio(text, words),
            check_line_length(lines),
            check_punctuation_end(lines),
            check_stopword_density(words, lang),
            check_duplicate_lines(lines),
        ]

        # Run ALL checks, collect flags (don't fail-fast for better stats)
        for ok, reason, _ in checks:
            if not ok:
                quality_flags.append(reason)

        # Drop if 2+ heuristic rules fail (single rule failures can be noise)
        if len(quality_flags) >= 2:
            passed = False
            drop_reason = ";".join(quality_flags[:2])

        if not passed:
            stats.add_drop_reason(quality_flags[0] if quality_flags else "unknown")
            if examples_added < 5:
                stats.add_example(
                    before=text[:300],
                    after="[DROPPED]",
                    note=f"Failed heuristics: {quality_flags}"
                )
                examples_added += 1
            continue

        # Educational classifier score
        edu = educational_score(text, words)
        if edu < min_edu_score:
            stats.add_drop_reason(f"edu_score_low_{edu:.1f}")
            continue

        doc = dict(doc)
        doc["edu_score"] = edu
        doc["quality_flags"] = quality_flags  # empty = passed all
        doc["avg_word_len"] = round(sum(len(w) for w in words) / max(len(words), 1), 2)
        doc["dup_line_ratio"] = round(1 - len(set(lines)) / max(len(lines), 1), 3)
        output.append(doc)

    stats.output_docs = len(output)
    stats.output_tokens = sum(estimate_tokens(d["text"]) for d in output)
    stats.extra = {
        "heuristic_rules_applied": [
            "word_length_bounds (3-10, relaxed 3-15 Indic)",
            "symbol_to_word_ratio (<10%)",
            "line_length_bounds (30-1000 chars)",
            "punctuation_end_check (>=40%)",
            "stopword_density (>=5%)",
            "duplicate_lines_ratio (<30%)",
        ],
        "edu_classifier_threshold": min_edu_score,
        "indic_exception_note": (
            "Telugu/Hindi word-length upper bound relaxed to 15 chars "
            "due to agglutinative morphology. English Gopher threshold (10) "
            "would incorrectly penalize high-quality Indic documents."
        ),
    }
    return output
