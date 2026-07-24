"""
stage6_pii.py -- Stage 6: PII Scrubbing
-----------------------------------------
Applies regex-based pattern matching for structured PII,
following the exact patterns specified in Session 4 notes.

Patterns scraped:
  - Email addresses
  - IPv4 addresses
  - Phone numbers (international + Indian mobile)
  - Indian AADHAAR (12-digit space-separated)
  - Indian PAN card (ABCDE1234F)

Each match is replaced with a typed placeholder (not empty string).

Note on precision-recall tradeoff: We do NOT run ML NER (personal name
detection) in this demo as it requires a large NER model. In production,
Microsoft Presidio or a fine-tuned model would handle personal names.
"""

import re
from stats_tracker import StageStats, estimate_tokens

# --- Regex patterns from Session 4 notes -----------------------------------
PII_PATTERNS = [
    # (name, pattern, placeholder)
    (
        "email",
        re.compile(
            r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
        ),
        "[EMAIL_REDACTED]",
    ),
    (
        "ipv4",
        re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
        ),
        "[IP_REDACTED]",
    ),
    (
        "phone_intl",
        re.compile(
            r"(?:\+\d{1,3}[\s\-]?)?(?:\d[\s\-]?){8,12}\d"
        ),
        "[PHONE_REDACTED]",
    ),
    (
        "aadhaar",
        re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b"),
        "[AADHAAR_REDACTED]",
    ),
    (
        "pan_card",
        re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
        "[PAN_REDACTED]",
    ),
]


def scrub_pii(text: str) -> tuple[str, dict]:
    """Apply all PII patterns. Returns (scrubbed_text, redaction_counts)."""
    counts = {}
    for name, pattern, placeholder in PII_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            counts[name] = len(matches)
            text = pattern.sub(placeholder, text)
    return text, counts


def run_stage6(docs: list[dict], stats: StageStats) -> list[dict]:
    """
    Scrub PII from all documents.
    Documents are NOT dropped unless they become too short after scrubbing.
    """
    stats.input_docs = len(docs)
    stats.input_tokens = sum(estimate_tokens(d["text"]) for d in docs)
    stats.metadata_added = ["pii_redactions", "pii_types_found"]

    output = []
    total_redactions = 0
    docs_with_pii = 0
    examples_added = 0

    for doc in docs:
        original = doc["text"]
        scrubbed, counts = scrub_pii(original)

        if counts:
            docs_with_pii += 1
            total_redactions += sum(counts.values())
            if examples_added < 5:
                stats.add_example(
                    before=original[:300],
                    after=scrubbed[:300],
                    note=f"PII found: {counts}"
                )
                examples_added += 1

        if len(scrubbed.strip()) < 50:
            stats.add_drop_reason("too_short_after_pii_scrub")
            continue

        doc = dict(doc)
        doc["text"] = scrubbed
        doc["pii_redactions"] = sum(counts.values())
        doc["pii_types_found"] = list(counts.keys())
        output.append(doc)

    stats.output_docs = len(output)
    stats.output_tokens = sum(estimate_tokens(d["text"]) for d in output)
    stats.extra = {
        "total_pii_redactions": total_redactions,
        "docs_with_pii": docs_with_pii,
        "patterns_applied": [p[0] for p in PII_PATTERNS],
        "placeholder_strategy": "typed_placeholder_not_empty_string",
        "ner_note": (
            "ML NER (personal name detection via Presidio) not run in this demo. "
            "In production, threshold >= 0.6 recommended to avoid masking "
            "Indic place names (e.g., Mysuru/Mumbai) as personal names."
        ),
        "precision_recall_note": (
            "High aggressiveness (threshold < 0.4) increases recall but hurts precision. "
            "Indic NER models confuse common nouns and place names as personal names."
        ),
    }
    return output
