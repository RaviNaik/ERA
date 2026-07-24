"""
stage2_normalize.py -- Stage 2: Normalize
------------------------------------------
Implements the exact clean_text() function from Session 4 lecture notes.

Operations (in order):
  1. Unicode NFC normalization (canonical composition)
  2. HTML entity unescaping
  3. Strip noise/invisible chars: BOM, ZWSP, BIDI overrides, C0/C1 control codes
     ***  PRESERVE ZWNJ (U+200C) and ZWJ (U+200D) for Brahmic scripts ***
  4. Collapse whitespace runs

Ghost-Tag bonus pass:
  Rewrites chat-format ghost markers ([USER], <human>, ### Instruction:) to
  canonical placeholders -- prevents subword collision during SFT.

Drop condition: text < 100 chars after normalization.
"""

import re
import unicodedata
import html as html_mod
from stats_tracker import StageStats, estimate_tokens

# --- Noise character regex --------------------------------------------------
# Strips: BOM U+FEFF, ZWSP U+200B, BIDI overrides U+202E/U+202D,
#         U+FFFD, C0 codes (except 0x09/0x0A/0x0D), C1 codes
# PRESERVES: U+200C (ZWNJ) and U+200D (ZWJ) -- vital for Brahmic scripts
NOISE_RE = re.compile(
    r"[\uFEFF\u200B\u202E\u202D\uFFFD"
    r"\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]"
)

# --- Ghost-tag patterns (Wikipedia wont have these, but Sangraha SFT data might)
GHOST_TAGS = [
    re.compile(r"\[USER\]\s*:?",      re.IGNORECASE),
    re.compile(r"\[ASSISTANT\]\s*:?", re.IGNORECASE),
    re.compile(r"<human>\s*:?",       re.IGNORECASE),
    re.compile(r"<bot>\s*:?",         re.IGNORECASE),
    re.compile(r"### Instruction:\s*", re.IGNORECASE),
    re.compile(r"### Response:\s*",    re.IGNORECASE),
    re.compile(r"Human:\s*",           re.IGNORECASE),
    re.compile(r"Assistant:\s*",       re.IGNORECASE),
]


def count_noise_chars(text: str) -> dict:
    """Count each category of noise character found."""
    counts = {
        "bom": text.count("\uFEFF"),
        "zwsp": text.count("\u200B"),
        "bidi_rtl": text.count("\u202E"),
        "bidi_ltr": text.count("\u202D"),
        "replacement_char": text.count("\uFFFD"),
        "zwnj_preserved": text.count("\u200C"),
        "zwj_preserved": text.count("\u200D"),
    }
    return {k: v for k, v in counts.items() if v > 0}


def count_html_entities(text: str) -> int:
    """Count HTML entities like &amp; &lt; &gt; &nbsp;"""
    return len(re.findall(r"&[a-zA-Z]+;|&#\d+;", text))


def count_ghost_tags(text: str) -> int:
    return sum(len(p.findall(text)) for p in GHOST_TAGS)


def clean_text(s: str) -> tuple[str, dict]:
    """
    Main normalization function from Session 4 notes.
    Returns (cleaned_text, metadata_dict).
    """
    meta = {}

    # 1. NFC Unicode normalization
    s = unicodedata.normalize("NFC", s)

    # 2. Unescape HTML entities
    entity_count = count_html_entities(s)
    if entity_count:
        meta["html_entities_removed"] = entity_count
    s = html_mod.unescape(s)

    # 3. Count noise chars before stripping
    noise = count_noise_chars(s)
    if noise:
        meta["noise_chars_removed"] = noise

    # 4. Strip replacement chars explicitly first
    s = s.replace("\uFFFD", "")

    # 5. Strip noise (regex -- preserves ZWNJ U+200C and ZWJ U+200D)
    s = re.sub(NOISE_RE, "", s)

    # 6. Collapse whitespace
    s = re.sub(r"\r\n|\r", "\n", s)          # normalize line endings
    s = re.sub(r"[^\S\n]+", " ", s)           # collapse horizontal whitespace
    s = re.sub(r"\n{3,}", "\n\n", s)          # max 2 blank lines
    s = s.strip()

    return s, meta


def run_ghost_tag_pass(text: str) -> tuple[str, int]:
    """Strip ghost chat-format markers. Returns (cleaned, count_removed)."""
    count = 0
    for pat in GHOST_TAGS:
        matches = pat.findall(text)
        count += len(matches)
        text = pat.sub("", text)
    return text.strip(), count


def run_stage2(docs: list[dict], stats: StageStats) -> list[dict]:
    """
    Normalize all documents. Drops docs shorter than 100 chars after cleaning.
    """
    stats.input_docs = len(docs)
    stats.input_tokens = sum(estimate_tokens(d["text"]) for d in docs)
    stats.metadata_added = [
        "html_entities_removed", "noise_chars_removed",
        "ghost_tags_removed", "norm_len"
    ]

    output = []
    total_entities = 0
    total_noise = 0
    total_ghost = 0
    examples_added = 0

    for doc in docs:
        original = doc["text"]
        cleaned, meta = clean_text(original)

        # Ghost-tag bonus pass
        cleaned, ghost_count = run_ghost_tag_pass(cleaned)
        if ghost_count:
            meta["ghost_tags_removed"] = ghost_count
            total_ghost += ghost_count

        if meta.get("html_entities_removed"):
            total_entities += meta["html_entities_removed"]
        if meta.get("noise_chars_removed"):
            total_noise += sum(meta["noise_chars_removed"].values())

        if len(cleaned) < 100:
            stats.add_drop_reason("too_short_after_norm")
            continue

        doc = dict(doc)
        doc["text"] = cleaned
        doc["norm_len"] = len(cleaned)
        doc.update(meta)
        output.append(doc)

        if examples_added < 5 and (total_entities > 0 or total_noise > 0 or ghost_count > 0):
            stats.add_example(
                before=original[:300],
                after=cleaned[:300],
                note=f"Entities: {meta.get('html_entities_removed', 0)}, "
                     f"Noise chars: {sum(meta.get('noise_chars_removed', {}).values())}, "
                     f"Ghost tags: {meta.get('ghost_tags_removed', 0)}"
            )
            examples_added += 1

    stats.output_docs = len(output)
    stats.output_tokens = sum(estimate_tokens(d["text"]) for d in output)
    stats.extra = {
        "total_html_entities_removed": total_entities,
        "total_noise_chars_removed": total_noise,
        "total_ghost_tags_removed": total_ghost,
        "zwnj_preserved_note": "ZWNJ (U+200C) and ZWJ (U+200D) preserved for Brahmic scripts",
    }
    return output
