"""
stage1_extract.py — Stage 1: Extract
--------------------------------------
For Wikipedia (HuggingFace): text is already prose, but still contains:
  - Wiki markup remnants (== Section ==, [[links]], {{templates}})
  - Reference markers: [1], [2], [note 1]
  - Lone punctuation lines
  - Excessive blank sections
  - Very short stub articles (< 150 chars after cleaning)

For Sangraha (already cleaned text, some HTML snippets may remain).

Droppers: empty after cleaning, stub articles < 150 chars.
"""

import re
from stats_tracker import StageStats, estimate_tokens

# Wiki markup patterns to strip
WIKI_SECTION_RE = re.compile(r"={2,}[^=]+={2,}")          # == Heading ==
WIKI_LINK_RE    = re.compile(r"\[\[([^\|\]]+\|)?([^\]]+)\]\]")  # [[link|text]] -> text
WIKI_TEMPLATE_RE = re.compile(r"\{\{[^}]*\}\}")            # {{template}}
WIKI_REF_RE     = re.compile(r"\[\d+\]|\[note\s*\d+\]")   # [1], [note 1]
WIKI_EXT_LINK_RE = re.compile(r"\[https?://[^\s\]]+[^\]]*\]")  # [http://...]
HTML_TAG_RE     = re.compile(r"<[^>]+>")                   # residual HTML tags
CONTROL_LINE_RE = re.compile(r"^[^a-zA-Z\u0900-\u0D7F]+$") # lines with no letters


def extract_document(text: str) -> tuple[str, str]:
    """
    Clean wiki/HTML markup from text.
    Returns (cleaned_text, drop_reason_or_empty).
    """
    if not text or not text.strip():
        return "", "empty_input"

    orig_len = len(text)

    # Remove wiki templates (nested ones too, greedily)
    text = WIKI_TEMPLATE_RE.sub("", text)
    # Expand wiki links to just the display text
    text = WIKI_LINK_RE.sub(r"\2", text)
    # Remove section headers
    text = WIKI_SECTION_RE.sub("", text)
    # Remove external link markup
    text = WIKI_EXT_LINK_RE.sub("", text)
    # Remove reference markers
    text = WIKI_REF_RE.sub("", text)
    # Remove residual HTML tags
    text = HTML_TAG_RE.sub("", text)

    # Drop lines that are purely punctuation/numbers with no letters
    lines = text.splitlines()
    lines = [l for l in lines if not CONTROL_LINE_RE.match(l.strip()) or not l.strip()]
    text = "\n".join(lines)

    # Collapse excessive blank lines (>2 in a row → 1)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    if not text:
        return "", "empty_after_extraction"

    if len(text) < 150:
        return "", "stub_too_short"

    removed_pct = round(100 * (orig_len - len(text)) / max(orig_len, 1), 1)
    return text, ""


def run_stage1(docs: list[dict], stats: StageStats) -> list[dict]:
    """
    Process a list of document dicts (must have 'text' key).
    Returns filtered list with 'text' replaced by extracted text.
    Adds 'orig_len', 'extracted_len', 'extraction_removed_pct' to each doc.
    """
    stats.input_docs = len(docs)
    stats.input_tokens = sum(estimate_tokens(d["text"]) for d in docs)
    stats.metadata_added = ["orig_len", "extracted_len", "extraction_removed_pct"]

    output = []
    examples_added = 0

    for doc in docs:
        raw_text = doc.get("text", "")
        cleaned, reason = extract_document(raw_text)

        if reason:
            stats.add_drop_reason(reason)
            continue

        doc = dict(doc)
        doc["orig_len"] = len(raw_text)
        doc["extracted_len"] = len(cleaned)
        doc["extraction_removed_pct"] = round(
            100.0 * (doc["orig_len"] - doc["extracted_len"]) / max(doc["orig_len"], 1), 2
        )
        doc["text"] = cleaned
        output.append(doc)

        if examples_added < 5 and doc["extraction_removed_pct"] > 5:
            stats.add_example(
                before=raw_text[:300],
                after=cleaned[:300],
                note=f"Removed {doc['extraction_removed_pct']:.1f}% markup/boilerplate"
            )
            examples_added += 1

    stats.output_docs = len(output)
    stats.output_tokens = sum(estimate_tokens(d["text"]) for d in output)
    return output
