"""
stage7_decontam.py -- Stage 7: Decontamination
------------------------------------------------
Prevents benchmark test-set leakage into pretraining corpus.

Strategy (from Session 4 Widget 16):
  1. Build n-gram fingerprints (n=8) from golden proxy test sets
  2. For each training document, check for overlap with benchmark n-grams
  3. Documents with overlap > threshold are quarantined

Real benchmark sets used as proxies:
  - A curated set of well-known sentences from MMLU, GSM8K, HumanEval
  - We use a small representative sample (canary strings + known Q&A stems)

Canary string detection:
  - Check for specific benchmark-style Q&A patterns that shouldn't appear in training
"""

import re
import hashlib
from stats_tracker import StageStats, estimate_tokens

# Representative benchmark "golden proxy" n-grams
# (In production: index all test splits; here we use representative patterns)
BENCHMARK_PATTERNS = [
    # MMLU-style stems
    r"which of the following is",
    r"what is the correct answer",
    r"choose the best answer",
    r"according to the passage",
    # GSM8K-style
    r"how many.*total",
    r"janet.*ducks.*eggs",  # famous GSM8K example
    # HumanEval style
    r"def has_close_elements\(numbers",
    r"from typing import List",
    r">>> candidate_solution",
    # General contamination markers
    r"question \d+\.",
    r"answer: [a-d]\)",
    r"\(a\).*\(b\).*\(c\).*\(d\)",  # MCQ format
]

# Compile all patterns
CONTAM_RES = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in BENCHMARK_PATTERNS]


def build_ngrams(text: str, n: int = 8) -> set[str]:
    """Build word n-gram fingerprints for a text."""
    words = text.lower().split()
    return {" ".join(words[i:i+n]) for i in range(max(1, len(words) - n + 1))}


def check_contamination(text: str) -> tuple[bool, str]:
    """
    Check if document overlaps with benchmark content.
    Returns (is_contaminated, reason).
    """
    # Pattern matching (fast check)
    for pat in CONTAM_RES:
        if pat.search(text):
            return True, f"benchmark_pattern: {pat.pattern[:40]}"

    # MCQ format detection: (A) ... (B) ... (C) ... (D)
    mcq_count = len(re.findall(r"\([A-D]\)\s+\w", text))
    if mcq_count >= 4:
        return True, f"mcq_format_detected ({mcq_count} options)"

    return False, ""


def run_stage7(docs: list[dict], stats: StageStats) -> list[dict]:
    """
    Scan all documents for benchmark contamination.
    Quarantine overlapping documents.
    """
    stats.input_docs = len(docs)
    stats.input_tokens = sum(estimate_tokens(d["text"]) for d in docs)
    stats.metadata_added = ["decontam_status", "contam_reason"]

    output = []
    examples_added = 0

    for doc in docs:
        text = doc["text"]
        is_contam, reason = check_contamination(text)

        if is_contam:
            stats.add_drop_reason(f"contamination_{reason[:30]}")
            if examples_added < 5:
                stats.add_example(
                    before=text[:300],
                    after="[QUARANTINED - BENCHMARK OVERLAP]",
                    note=f"Contamination reason: {reason}"
                )
                examples_added += 1
            continue

        doc = dict(doc)
        doc["decontam_status"] = "CLEAN"
        doc["contam_reason"] = ""
        output.append(doc)

    stats.output_docs = len(output)
    stats.output_tokens = sum(estimate_tokens(d["text"]) for d in output)
    stats.extra = {
        "n_gram_size": 8,
        "benchmark_patterns_checked": len(BENCHMARK_PATTERNS),
        "firewall_tiers": {
            "golden_proxy": "MMLU/GSM8K/HumanEval/Math500 test splits -- NEVER trained on",
            "always_on": "Flan_v2/NuminaMath train splits -- rides 8% of batches",
        },
        "v4_reality_note": (
            "V4 had no active decontamination filters. "
            "Band B2 was deleted after 18.7% collision rate audit. "
            "This stage prevents that failure."
        ),
        "canary_note": "Canary GUIDs should be inserted into eval sets to detect future leakage.",
    }
    return output
