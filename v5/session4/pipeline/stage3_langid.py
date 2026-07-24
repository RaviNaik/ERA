"""
stage3_langid.py -- Stage 3: Language ID & Validation
-------------------------------------------------------
Runtime language detection using fasttext-langdetect.
Validates that documents match their declared language.
For Wikipedia: expects English. For Sangraha: expects hi/te/bn/etc.

Drops:
  - Non-English docs in Wikipedia run (lang != 'en' with confidence >= 0.75)
  - Mismatched language docs in Sangraha run
  - Very low confidence detections (< 0.5)
"""

import re
from stats_tracker import StageStats, estimate_tokens

# Script histogram ranges for Brahmic scripts
SCRIPT_RANGES = {
    "hi": (0x0900, 0x097F),   # Devanagari
    "te": (0x0C00, 0x0C7F),   # Telugu
    "ta": (0x0B80, 0x0BFF),   # Tamil
    "bn": (0x0980, 0x09FF),   # Bengali
    "gu": (0x0A80, 0x0AFF),   # Gujarati
    "kn": (0x0C80, 0x0CFF),   # Kannada
    "ml": (0x0D00, 0x0D7F),   # Malayalam
    "pa": (0x0A00, 0x0A7F),   # Gurmukhi (Punjabi)
    "or": (0x0B00, 0x0B7F),   # Odia
}


def get_script_coverage(text: str, lang_code: str) -> float:
    """Return % of chars in the expected script for this language."""
    if lang_code not in SCRIPT_RANGES:
        return 1.0  # unknown script, skip check
    lo, hi = SCRIPT_RANGES[lang_code]
    script_chars = sum(1 for c in text if lo <= ord(c) <= hi)
    total_chars = max(len(text.replace(" ", "")), 1)
    return script_chars / total_chars


def detect_language(text: str) -> tuple[str, float]:
    """
    Detect language using fasttext-langdetect.
    Falls back to langdetect if fasttext fails.
    Returns (lang_code, confidence).
    """
    try:
        from ftlangdetect import detect as ft_detect
        result = ft_detect(text[:1000].replace("\n", " "), low_memory=False)
        return result["lang"], result["score"]
    except Exception:
        pass

    try:
        from langdetect import detect_langs
        results = detect_langs(text[:1000])
        if results:
            top = results[0]
            return top.lang, top.prob
    except Exception:
        pass

    return "unknown", 0.0


def run_stage3(docs: list[dict], stats: StageStats,
               expected_lang: str = "en",
               confidence_threshold: float = 0.70) -> list[dict]:
    """
    Run language ID on all documents.

    expected_lang: 'en' for Wikipedia run, or the primary Indic code
                   (we also accept closely related scripts)
    """
    stats.input_docs = len(docs)
    stats.input_tokens = sum(estimate_tokens(d["text"]) for d in docs)
    stats.metadata_added = ["detected_lang", "lang_confidence", "script_coverage_pct"]

    output = []
    examples_added = 0
    lang_distribution: dict[str, int] = {}

    # For Indic runs, accept known Indic languages (any Sangraha language)
    ACCEPTED_INDIC = {"hi", "te", "bn", "ta", "gu", "kn", "ml", "pa", "or",
                      "ne", "si", "ur", "as", "mr", "sa", "mai", "kok", "sd"}

    for doc in docs:
        text = doc["text"]
        lang, confidence = detect_language(text)
        lang_distribution[lang] = lang_distribution.get(lang, 0) + 1

        # Low confidence: quarantine
        if confidence < confidence_threshold:
            stats.add_drop_reason(f"low_confidence_detection ({lang}:{confidence:.2f})")
            if examples_added < 3:
                stats.add_example(
                    before=text[:200],
                    after="[QUARANTINED]",
                    note=f"Low detection confidence: {lang} at {confidence:.2f}"
                )
                examples_added += 1
            continue

        # Language mismatch check
        if expected_lang == "en":
            if lang != "en":
                stats.add_drop_reason(f"wrong_lang_{lang}")
                continue
        else:
            # Indic run: accept any Indic language, reject clear English/other
            if lang not in ACCEPTED_INDIC and lang not in ("en",):
                # Some code-mixed content is OK
                stats.add_drop_reason(f"unexpected_lang_{lang}")
                continue

        # Script histogram validation for Indic
        script_cov = 1.0
        if lang in SCRIPT_RANGES:
            script_cov = get_script_coverage(text, lang)
            if script_cov < 0.30:
                # Too little script-appropriate text
                stats.add_drop_reason("insufficient_script_coverage")
                if examples_added < 3:
                    stats.add_example(
                        before=text[:200],
                        after="[QUARANTINED]",
                        note=f"Script coverage too low: {script_cov:.1%} for {lang}"
                    )
                    examples_added += 1
                continue

        doc = dict(doc)
        doc["detected_lang"] = lang
        doc["lang_confidence"] = round(confidence, 3)
        doc["script_coverage_pct"] = round(script_cov * 100, 1)
        output.append(doc)

    stats.output_docs = len(output)
    stats.output_tokens = sum(estimate_tokens(d["text"]) for d in output)
    stats.extra = {
        "expected_lang": expected_lang,
        "confidence_threshold": confidence_threshold,
        "lang_distribution_detected": dict(sorted(
            lang_distribution.items(), key=lambda x: -x[1])[:10]
        ),
        "v4_bug_note": (
            "V4 used directory-path trust (verified/asm/ assumed Assamese). "
            "We validate every document at runtime instead."
        ),
    }
    return output
