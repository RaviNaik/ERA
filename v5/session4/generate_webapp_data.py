"""
generate_webapp_data.py
------------------------
Builds webapp/data.js from the real pipeline output in data/*_stats.json and
data/manifests/*/sample_manifests.json. This replaces hand-typed numbers in
data.js (which drift from reality every time the pipeline is re-run) with a
single generated source of truth.

Run after run_pipeline.py:
    .venv\\Scripts\\python.exe generate_webapp_data.py
"""

import json
import re
import sys
import pathlib

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = pathlib.Path(__file__).parent
DATA_DIR = ROOT / "data"
WEBAPP_DIR = ROOT / "webapp"

STAGE_ICON = {
    1: "🔍", 2: "✨", 3: "🌐", 4: "🧹",
    5: "🔁", 6: "🔒", 7: "🛡", 8: "📋",
}
STAGE_COLOR = {
    1: "#6366f1", 2: "#8b5cf6", 3: "#06b6d4", 4: "#10b981",
    5: "#f59e0b", 6: "#ef4444", 7: "#f97316", 8: "#64748b",
}

DATASET_META = {
    "wikipedia": {
        "source": "wikimedia/wikipedia",
        "config": "20231101.en",
        "license": "CC-BY-SA 4.0",
        "description": "English Wikipedia streamed from Hugging Face",
    },
    "sangraha": {
        "source": "ai4bharat/sangraha",
        "config": "verified / hin + tel",
        "license": "CC-BY 4.0",
        "description": "Indic dataset (Hindi + Telugu) — real web + PDF-extracted text demonstrating the Sovereign Indic Exception",
    },
    "c4_crawl": {
        "source": "allenai/c4",
        "config": "en.noclean",
        "license": "ODC-BY",
        "description": "allenai/c4 en.noclean — unfiltered CommonCrawl pages. No quality filtering by source. Contains nav boilerplate, SEO spam, multilingual noise, short stubs.",
    },
}

# Generic "what was removed" copy per stage id, reused across runs since the
# mechanism doesn't change -- only the specific counts do (those come from
# drop_summary, generated per-run below).
WHAT_REMOVED = {
    1: "Markup/boilerplate remnants (templates, links, nav fragments), reference markers, residual HTML tags",
    2: "HTML entities, BOM/ZWSP/BIDI overrides and other invisible noise characters, ghost chat-format tags. ZWNJ/ZWJ preserved for Brahmic scripts.",
    3: "Wrong-language and low language-detection-confidence documents (validated at runtime, not trusted from directory paths)",
    4: "Documents failing 2+ Gopher/C4 heuristic rules (word length, symbol ratio, line length, punctuation-end, stopword density, duplicate lines) or low educational score",
    5: "Exact duplicates (SHA-256) and near-duplicates (MinHash LSH, Jaccard >= 0.80)",
    6: "Emails, IPv4 addresses, phone numbers, AADHAAR and PAN numbers — replaced with typed placeholders, not empty strings",
    7: "Documents overlapping benchmark n-grams or matching MCQ/Q&A patterns (decontamination firewall)",
    8: "Nothing dropped — stamps SHA-256 shard_id, license, token count and provenance on every admitted document",
}


def humanize_reason(reason: str) -> str:
    """Collapse a specific drop_reason key into a readable category label."""
    reason = re.sub(r"\s*\([^)]*\)", "", reason)          # strip "(en:0.58)" etc.
    reason = re.sub(r"_\d+(\.\d+)?$", "", reason)           # strip trailing _22 / _0.44
    return reason.replace("_", " ")


def summarize_drops(stage: dict) -> str:
    """Aggregate drop_reasons into a short human summary of top categories."""
    dropped = stage["docs_dropped"]
    if dropped == 0:
        return "No docs dropped"
    agg: dict[str, int] = {}
    for reason, count in stage["drop_reasons"].items():
        cat = humanize_reason(reason)
        agg[cat] = agg.get(cat, 0) + count
    top = sorted(agg.items(), key=lambda x: -x[1])[:3]
    parts = [f"{count:,} {cat}" for cat, count in top]
    return f"{dropped:,} dropped — " + ", ".join(parts)


def load_stats(run_name: str) -> dict:
    path = DATA_DIR / f"{run_name}_stats.json"
    return json.loads(path.read_text(encoding="utf-8"))


def build_run_block(run_name: str, data_key: str) -> dict:
    stats = load_stats(run_name)
    stages_out = []
    for st in stats["stages"]:
        sid = st["stage_id"]
        examples = st.get("examples", [])
        ex = examples[0] if examples else {"before": "(no example captured)", "after": "", "note": ""}
        stages_out.append({
            "id": sid, "name": st["stage_name"],
            "icon": STAGE_ICON[sid], "color": STAGE_COLOR[sid],
            "input": st["input_docs"], "output": st["output_docs"],
            "dropped": st["docs_dropped"], "drop_pct": st["drop_pct"],
            "survival": round(100 - st["drop_pct"], 1),
            "input_tok": st["input_tokens"], "output_tok": st["output_tokens"],
            "drop_summary": summarize_drops(st),
            "what_removed": WHAT_REMOVED[sid],
            "metadata_added": st.get("metadata_added", []),
            "processing_time_s": st.get("processing_time_s", 0),
            "example": {"before": ex["before"], "after": ex["after"], "note": ex.get("note", "")},
        })

    final = stats["stages"][-1]
    initial_docs = stats["initial_docs"]
    admitted_docs = final["output_docs"]
    total_time = round(sum(s["processing_time_s"] for s in stats["stages"]), 1)

    meta = dict(DATASET_META[data_key])
    meta.update({
        "initial_docs": initial_docs,
        "initial_tokens": stats["initial_tokens"],
        "admitted_docs": admitted_docs,
        "admitted_tokens": final["output_tokens"],
        "survival_pct": round(100 * admitted_docs / max(initial_docs, 1), 1),
        "total_time_s": total_time,
    })

    if data_key == "sangraha":
        dedup_stage = next(s for s in stats["stages"] if s["stage_name"] == "Deduplicate")
        meta["dedup_note"] = (
            f"Global dedup removed {dedup_stage['docs_dropped']:,} duplicates "
            f"({dedup_stage['drop_pct']}%) from real ai4bharat/sangraha web + PDF-extracted text. "
            "Real Indic web crawls have heavy news syndication — the same article appears "
            "across many regional sites, which is why global dedup is load-bearing."
        )

    return {"meta": meta, "stages": stages_out}


STRATEGIES = [
    {"id": 1, "name": "Extraction", "icon": "🔍", "color": "#6366f1", "gap": "Earlier approach kept nav links and cookie banners as document body content", "fix": "trafilatura on raw WARC; strip {{templates}}, [[links]], ==Sections==, [refs]", "drop": "Stubs <150 chars, empty after markup strip"},
    {"id": 2, "name": "Normalization", "icon": "✨", "color": "#8b5cf6", "gap": "No shared clean_text() normalization function — garbage tokens baked into tokenizer", "fix": "NFC Unicode → HTML unescape → strip BOM/ZWSP/BIDI → PRESERVE ZWNJ/ZWJ → collapse whitespace", "drop": "Docs <100 chars after normalization"},
    {"id": 3, "name": "Language ID", "icon": "🌐", "color": "#06b6d4", "gap": "Trusted directory naming (verified/asm/) without runtime validation; Python dict misrouting bug", "fix": "fastText detection on every doc + script-histogram validation (Unicode codepoint block coverage)", "drop": "Confidence <0.70, wrong-language docs, <30% expected script coverage"},
    {"id": 4, "name": "Quality Filter", "icon": "🧹", "color": "#10b981", "gap": "English-heavy proxy classifier penalised Indic scripts; bad noise admitted via always-on bypass", "fix": "6 Gopher/C4 heuristic rules (word-len, symbol-ratio, line-len, punct-end, stopword, dup-lines) + FineWeb-Edu edu score; Indic-aware thresholds", "drop": "2+ heuristic failures or low edu_score"},
    {"id": 5, "name": "Deduplication", "icon": "🔁", "color": "#f59e0b", "gap": "Only local per-source dedup — cross-shard duplicates leaked, wasting compute", "fix": "Pass 1: SHA-256 exact hash. Pass 2: MinHash LSH (128 perms, shingle k=5, Jaccard ≥0.80)", "drop": "Exact SHA-256 matches, near-dups with Jaccard ≥0.80"},
    {"id": 6, "name": "PII Scrub", "icon": "🔒", "color": "#ef4444", "gap": "Lacked PII scrubbing for Indic pipelines — personal identifiers exposed in training set", "fix": "Regex: email, IPv4, phone, AADHAAR, PAN. Typed placeholder [EMAIL_REDACTED] not empty string", "drop": "Docs <50 chars after scrubbing (rare)"},
    {"id": 7, "name": "Decontamination", "icon": "🛡", "color": "#f97316", "gap": "No active decontamination firewall — 18.7% benchmark collision rate discovered post-training", "fix": "n-gram fingerprint (n=8) + MCQ pattern scan against MMLU/GSM8K/HumanEval. 3-tier firewall.", "drop": "Benchmark overlap or MCQ format detected"},
    {"id": 8, "name": "Manifest", "icon": "📋", "color": "#64748b", "gap": "Non-deterministic row counters for shard IDs; words×1.3 token estimate → 10× undercount for Indic", "fix": "shard_id = sha256[:12] (content-addressed). Token counting made consistent end-to-end.", "drop": "None — stamps provenance on all admitted docs"},
    {"id": 9, "name": "Ghost-Tag Trap", "icon": "👻", "color": "#ec4899", "gap": "Chat markers [USER]/[ASSISTANT]/### Instruction: baked as ghost subwords during pretraining → SFT collision", "fix": "Rewrite ALL chat-format markers to canonical special tokens at pretraining ingestion", "drop": "N/A — rewrite stage"},
]


def load_sample_manifest() -> dict:
    path = DATA_DIR / "manifests" / "wikipedia" / "sample_manifests.json"
    manifests = json.loads(path.read_text(encoding="utf-8"))
    return manifests[0]


def to_js_literal(value, indent=0) -> str:
    pad = "  " * indent
    if isinstance(value, dict):
        items = ",\n".join(f'{pad}  "{k}": {to_js_literal(v, indent + 1)}' for k, v in value.items())
        return "{\n" + items + f"\n{pad}}}"
    if isinstance(value, list):
        # Compact stage objects on one line each for readability
        items = ",\n".join(f"{pad}  {to_js_literal(v, indent + 1)}" for v in value)
        return "[\n" + items + f"\n{pad}]"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return json.dumps(value)


def main():
    wikipedia = build_run_block("wikipedia", "wikipedia")
    sangraha = build_run_block("sangraha", "sangraha")
    c4_crawl = build_run_block("c4_crawl", "c4_crawl")
    sample_manifest = load_sample_manifest()

    payload = {
        "wikipedia": wikipedia,
        "sangraha": sangraha,
        "c4_crawl": c4_crawl,
        "strategies": STRATEGIES,
        "sampleManifest": sample_manifest,
    }

    header = (
        "// data.js -- Generated by generate_webapp_data.py from real pipeline output\n"
        "// (data/wikipedia_stats.json, data/sangraha_stats.json, data/c4_crawl_stats.json)\n"
        f"// Wikipedia: {wikipedia['meta']['initial_docs']:,} articles | "
        f"Sangraha: {sangraha['meta']['initial_docs']:,} Indic docs | "
        f"C4 en.noclean: {c4_crawl['meta']['initial_docs']:,} web pages\n"
        "// Do not hand-edit -- re-run run_pipeline.py then this script instead.\n\n"
    )
    body = "window.PIPELINE_DATA = " + to_js_literal(payload) + ";\n"

    out_path = WEBAPP_DIR / "data.js"
    out_path.write_text(header + body, encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"  Wikipedia: {wikipedia['meta']['initial_docs']:,} -> {wikipedia['meta']['admitted_docs']:,} ({wikipedia['meta']['survival_pct']}%)")
    print(f"  Sangraha:  {sangraha['meta']['initial_docs']:,} -> {sangraha['meta']['admitted_docs']:,} ({sangraha['meta']['survival_pct']}%)")
    print(f"  C4 crawl:  {c4_crawl['meta']['initial_docs']:,} -> {c4_crawl['meta']['admitted_docs']:,} ({c4_crawl['meta']['survival_pct']}%)")


if __name__ == "__main__":
    main()
