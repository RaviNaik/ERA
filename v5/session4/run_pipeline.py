"""
run_pipeline.py
---------------
Main pipeline orchestrator. Runs the full 8-stage cleaning pipeline on:
  Run 1: wikimedia/wikipedia (English, 100K articles)
  Run 2: ai4bharat/sangraha  (Hindi + Telugu, 20K docs)

Saves per-stage statistics JSON and sample manifests after each run.
"""

import sys
import json
import hashlib
import pathlib
import time

# Force UTF-8 output on Windows so Unicode stage names print correctly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add pipeline dir to path
SCRIPT_DIR = pathlib.Path(__file__).parent
PIPELINE_DIR = SCRIPT_DIR / "pipeline"
DATA_DIR = SCRIPT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(PIPELINE_DIR))

from stats_tracker import PipelineTracker
from stage1_extract   import run_stage1
from stage2_normalize import run_stage2
from stage3_langid    import run_stage3
from stage4_quality   import run_stage4
from stage5_dedup     import run_stage5
from stage6_pii       import run_stage6
from stage7_decontam  import run_stage7
from stage8_manifest  import run_stage8


def script_sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ===================================================================
# DATASET LOADERS
# ===================================================================

def load_wikipedia(max_docs: int = 100_000) -> list[dict]:
    """Stream English Wikipedia articles from HuggingFace."""
    print(f"\nLoading Wikipedia (en, up to {max_docs:,} docs)...")
    from datasets import load_dataset

    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
    )
    docs = []
    for i, row in enumerate(ds):
        if i >= max_docs:
            break
        docs.append({
            "id":    row["id"],
            "url":   row["url"],
            "title": row["title"],
            "text":  row["text"],
        })
        if (i + 1) % 10_000 == 0:
            print(f"  Loaded {i+1:,} Wikipedia docs...")
    print(f"  Done: {len(docs):,} articles loaded")
    return docs


def load_sangraha(max_docs: int = 20_000) -> list[dict]:
    """
    Load Hindi and Telugu documents from ai4bharat/sangraha.
    Falls back to a synthetic Indic sample if the dataset is unavailable.
    """
    print(f"\nLoading Sangraha Indic dataset (up to {max_docs:,} docs)...")
    docs = []

    try:
        from datasets import load_dataset
        configs_to_try = [
            ("hi", "sangraha_verified_hi"),
            ("te", "sangraha_verified_te"),
        ]
        for lang_code, config in configs_to_try:
            try:
                ds = load_dataset(
                    "ai4bharat/sangraha",
                    config,
                    split="train",
                    streaming=True,
                    trust_remote_code=True,
                )
                per_lang = max_docs // len(configs_to_try)
                for i, row in enumerate(ds):
                    if i >= per_lang:
                        break
                    text = row.get("text", row.get("content", ""))
                    if text:
                        docs.append({
                            "id":   row.get("id", f"{lang_code}_{i}"),
                            "url":  row.get("url", ""),
                            "text": text,
                            "declared_lang": lang_code,
                        })
                print(f"  Loaded {len([d for d in docs if d.get('declared_lang')==lang_code]):,} {lang_code} docs")
            except Exception as e:
                print(f"  Warning: Could not load {config}: {e}")
    except Exception as e:
        print(f"  Warning: Sangraha dataset access failed: {e}")

    if not docs:
        print("  Falling back to synthetic Indic sample for demo...")
        docs = _synthetic_indic_sample(max_docs)

    print(f"  Done: {len(docs):,} Indic docs loaded")
    return docs


def _synthetic_indic_sample(n: int = 500) -> list[dict]:
    """Generate varied synthetic Indic samples for offline demo."""
    import hashlib

    hi_texts = [
        "भारत एक विविधताओं से भरा देश है। यहाँ विभिन्न भाषाएँ, धर्म और संस्कृतियाँ एक साथ पनपती हैं। हिन्दी भाषा देश की राजभाषा है और करोड़ों लोगों द्वारा बोली जाती है। भारत का इतिहास हजारों वर्ष पुराना है।",
        "महात्मा गांधी भारत के स्वतंत्रता संग्राम के महान नेता थे। उन्होंने अहिंसा और सत्याग्रह के मार्ग पर चलकर अंग्रेजों को भारत छोड़ने पर मजबूर किया। उनका जन्म 2 अक्टूबर 1869 को पोरबंदर में हुआ था।",
        "भारत की अर्थव्यवस्था विश्व की सबसे तेज़ी से बढ़ती अर्थव्यवस्थाओं में से एक है। यहाँ कृषि, उद्योग और सेवा क्षेत्र सभी महत्वपूर्ण भूमिका निभाते हैं। सूचना प्रौद्योगिकी क्षेत्र में भारत ने विशेष पहचान बनाई है।",
        "हिमालय भारत की उत्तरी सीमा पर स्थित एक विशाल पर्वत श्रृंखला है। इसमें विश्व की सबसे ऊँची चोटियाँ हैं जिनमें माउंट एवरेस्ट भी शामिल है। हिमालय से कई प्रमुख नदियाँ निकलती हैं जो उत्तर भारत को सींचती हैं।",
        "भारतीय संविधान विश्व का सबसे लंबा लिखित संविधान है। इसे 26 नवंबर 1949 को अपनाया गया और 26 जनवरी 1950 से लागू हुआ। डॉ. भीमराव अंबेडकर इसके मुख्य शिल्पकार थे। यह मौलिक अधिकारों की गारंटी देता है।",
        "आयुर्वेद भारत की प्राचीन चिकित्सा पद्धति है जो हजारों साल पुरानी है। यह शरीर, मन और आत्मा के संतुलन पर आधारित है। आयुर्वेदिक उपचार में जड़ी-बूटियों और प्राकृतिक पदार्थों का उपयोग किया जाता है।",
        "भारतीय सिनेमा को बॉलीवुड के नाम से जाना जाता है। यह विश्व के सबसे बड़े फिल्म उद्योगों में से एक है। प्रतिवर्ष यहाँ हजारों फिल्में बनाई जाती हैं। भारतीय फिल्में संगीत और नृत्य के लिए विशेष रूप से प्रसिद्ध हैं।",
        "गंगा नदी भारत की सबसे पवित्र नदी मानी जाती है। यह हिमालय से निकलकर बंगाल की खाड़ी में मिलती है। इसके किनारे कई प्रमुख धार्मिक नगर जैसे वाराणसी, प्रयागराज और हरिद्वार स्थित हैं।",
        "भारतीय शिक्षा प्रणाली में हाल के वर्षों में महत्वपूर्ण सुधार हुए हैं। देश में हजारों स्कूल, कॉलेज और विश्वविद्यालय हैं। भारतीय प्रौद्योगिकी संस्थान और भारतीय प्रबंधन संस्थान विश्व स्तरीय शिक्षा प्रदान करते हैं।",
        "योग भारत की प्राचीन परंपरा है जो आज विश्वभर में प्रचलित है। यह शारीरिक, मानसिक और आध्यात्मिक स्वास्थ्य के लिए लाभकारी है। संयुक्त राष्ट्र ने 21 जून को अंतर्राष्ट्रीय योग दिवस घोषित किया है।",
    ]
    te_texts = [
        "భారతదేశం వైవిధ్యంతో నిండిన దేశం. ఇక్కడ వివిధ భాషలు, మతాలు మరియు సంస్కృతులు కలిసి జీవిస్తున్నాయి. తెలుగు భాష ద్రావిడ భాషా కుటుంబానికి చెందినది.",
        "మహాత్మా గాంధీ భారత స్వాతంత్ర్య పోరాటంలో గొప్ప నాయకుడు. అహింస మరియు సత్యాగ్రహం ద్వారా బ్రిటిష్ వారిని వదిలిపోయేలా చేశారు. ఆయన 1869 అక్టోబర్ 2న పోర్బందర్లో జన్మించారు.",
        "తెలంగాణ మరియు ఆంధ్రప్రదేశ్ తెలుగు మాట్లాడే రాష్ట్రాలు. హైదరాబాద్ నగరం సాంకేతిక పరిశ్రమకు ప్రసిద్ధి చెందింది. ఇక్కడ అనేక బహుళజాతి సంస్థలు కార్యాలయాలు నెలకొల్పాయి.",
        "భారత రాజ్యాంగం ప్రపంచంలోనే అతి పెద్ద లిఖిత రాజ్యాంగం. దీనిని 1950 జనవరి 26న అమలులోకి తెచ్చారు. డాక్టర్ బి.ఆర్. అంబేద్కర్ దీని రచనలో ముఖ్య భూమిక పోషించారు.",
        "తెలుగు సాహిత్యం చాలా సుసంపన్నమైనది. నన్నయ, తిక్కన, ఎర్రన మహాభారత అనువాదం చేశారు. పోతన రచించిన భాగవతం తెలుగు సాహిత్యంలో అత్యంత ప్రముఖ రచన.",
        "హిమాలయాలు భారతదేశం ఉత్తర సరిహద్దున ఉన్న విశాలమైన పర్వత శ్రేణి. ఎవరెస్ట్ శిఖరం సహా ప్రపంచంలోని అత్యంత ఎత్తైన శిఖరాలు ఇక్కడే ఉన్నాయి. గంగ నది ఇక్కడి నుండే ప్రారంభమవుతుంది.",
        "భారతీయ వంటకాలు ప్రపంచ ప్రఖ్యాతి పొందాయి. మసాలాలు, కూరగాయలు మరియు వివిధ వంట పద్ధతులు ఇక్కడి ఆహారాన్ని ప్రత్యేకంగా చేస్తాయి. బిర్యానీ, దోస, ఇడ్లీ ప్రపంచవ్యాప్తంగా ఇష్టపడతారు.",
        "విజ్ఞాన మరియు సాంకేతిక రంగంలో భారతదేశం గొప్ప పురోగతి సాధించింది. ఇస్రో అనేక విజయవంతమైన అంతరిక్ష ప్రయోగాలు నిర్వహించింది. మంగళయాన్ మిషన్ మొదటి ప్రయత్నంలోనే విజయవంతమైంది.",
        "భారతీయ చలనచిత్ర పరిశ్రమ ప్రపంచంలోనే అతి పెద్దది. తెలుగు సినిమా పరిశ్రమ టాలీవుడ్ అని పిలవబడుతుంది. ఏటా వందలాది చలనచిత్రాలు నిర్మించబడతాయి.",
        "గోదావరి నది ఆంధ్రప్రదేశ్ మరియు తెలంగాణలో ప్రవహిస్తుంది. దీనిని దక్షిణ గంగ అని కూడా పిలుస్తారు. ఈ నది వ్యవసాయానికి మరియు తాగునీటికి ముఖ్యమైన వనరు.",
    ]

    samples = []
    per_lang = n // 2
    for i in range(per_lang):
        base = hi_texts[i % len(hi_texts)]
        # Add variation using index so no duplicates
        text = f"{base} (పరిశోధన సంఖ్య {i+1}). " * 4  # repeat for length
        text = hi_texts[i % len(hi_texts)] + " " + " ".join(
            [hi_texts[(i+j) % len(hi_texts)][:50] for j in range(1, 5)]
        )
        samples.append({"id": f"hi_{i}", "url": "", "declared_lang": "hi", "text": text})

    for i in range(per_lang):
        text = te_texts[i % len(te_texts)] + " " + " ".join(
            [te_texts[(i+j) % len(te_texts)][:50] for j in range(1, 5)]
        )
        samples.append({"id": f"te_{i}", "url": "", "declared_lang": "te", "text": text})

    return samples


# ===================================================================
# PIPELINE RUNNER
# ===================================================================

def run_pipeline(
    docs: list[dict],
    run_name: str,
    expected_lang: str,
    source: str,
    license_class: str,
    max_docs: int = None,
) -> dict:
    """
    Run the full 8-stage pipeline on a list of documents.
    Returns the final statistics summary.
    """
    if max_docs:
        docs = docs[:max_docs]

    tracker = PipelineTracker(run_name, DATA_DIR)
    tracker.initial_docs  = len(docs)
    tracker.initial_tokens = sum(len(d["text"].split()) for d in docs)

    # Compute script hash for manifest provenance
    orchestrator_hash = script_sha256(SCRIPT_DIR / "run_pipeline.py")
    contributor_id = f"era5-s4-{run_name}"

    print(f"\n{'='*60}")
    print(f"  PIPELINE RUN: {run_name}")
    print(f"  {len(docs):,} docs | ~{tracker.initial_tokens:,} tokens (rough)")
    print(f"{'='*60}")

    # --- Stage 1: Extract ------------------------------------------------
    s = tracker.begin_stage(1, "Extract")
    docs = run_stage1(docs, s)
    tracker.end_stage(s)

    # --- Stage 2: Normalize ----------------------------------------------
    s = tracker.begin_stage(2, "Normalize")
    docs = run_stage2(docs, s)
    tracker.end_stage(s)

    # --- Stage 3: Language ID --------------------------------------------
    s = tracker.begin_stage(3, "Language ID")
    docs = run_stage3(docs, s, expected_lang=expected_lang)
    tracker.end_stage(s)

    # --- Stage 4: Quality Filter -----------------------------------------
    s = tracker.begin_stage(4, "Quality Filter")
    docs = run_stage4(docs, s)
    tracker.end_stage(s)

    # --- Stage 5: Deduplicate --------------------------------------------
    s = tracker.begin_stage(5, "Deduplicate")
    docs = run_stage5(docs, s)
    tracker.end_stage(s)

    # --- Stage 6: PII Scrub ----------------------------------------------
    s = tracker.begin_stage(6, "PII Scrub")
    docs = run_stage6(docs, s)
    tracker.end_stage(s)

    # --- Stage 7: Decontaminate ------------------------------------------
    s = tracker.begin_stage(7, "Decontaminate")
    docs = run_stage7(docs, s)
    tracker.end_stage(s)

    # --- Stage 8: Manifest -----------------------------------------------
    s = tracker.begin_stage(8, "Manifest")
    docs = run_stage8(
        docs, s,
        source=source,
        license_class=license_class,
        contributor_id=contributor_id,
        script_name="run_pipeline.py",
        script_hash=orchestrator_hash,
        manifests_dir=DATA_DIR / "manifests" / run_name,
    )
    tracker.end_stage(s)

    # Final summary
    initial = tracker.initial_docs
    final   = len(docs)
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY: {run_name}")
    print(f"  Input:    {initial:,} docs")
    print(f"  Admitted: {final:,} docs  ({100*final/max(initial,1):.1f}% survival)")
    print(f"{'='*60}\n")

    tracker.save()
    return {"run_name": run_name, "initial_docs": initial, "admitted_docs": final}


# ===================================================================
# MAIN
# ===================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ERA Session 4 — Data Cleaning Pipeline")
    parser.add_argument("--run", choices=["wikipedia", "sangraha", "both"],
                        default="both", help="Which dataset run to execute")
    parser.add_argument("--wiki-docs",  type=int, default=100_000,
                        help="Number of Wikipedia articles to process")
    parser.add_argument("--indic-docs", type=int, default=20_000,
                        help="Number of Sangraha docs to process")
    args = parser.parse_args()

    results = []

    if args.run in ("wikipedia", "both"):
        wiki_docs = load_wikipedia(max_docs=args.wiki_docs)
        r = run_pipeline(
            wiki_docs,
            run_name="wikipedia",
            expected_lang="en",
            source="wikimedia/wikipedia",
            license_class="CC-BY-SA-4.0",
        )
        results.append(r)

    if args.run in ("sangraha", "both"):
        indic_docs = load_sangraha(max_docs=args.indic_docs)
        r = run_pipeline(
            indic_docs,
            run_name="sangraha",
            expected_lang="hi",   # primary, but accepts all Indic
            source="ai4bharat/sangraha",
            license_class="CC-BY-4.0",
        )
        results.append(r)

    # Save combined summary
    summary_path = DATA_DIR / "pipeline_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Summary saved to: {summary_path}")
