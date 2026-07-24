"""
run_pipeline.py
---------------
Main pipeline orchestrator. Runs the full 8-stage cleaning pipeline on:
  Run 1: wikimedia/wikipedia   (English, 100K articles)
  Run 2: ai4bharat/sangraha   (Hindi + Telugu, 20K docs)
  Run 3: allenai/c4 realnews  (Messy web crawl, 20K docs)

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

from stats_tracker import PipelineTracker, estimate_tokens
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
        # Real HF layout: dataset config is "verified", language is the SPLIT
        # (not a per-language config name). Fields are doc_id/type/text.
        langs_to_try = ["hin", "tel"]
        for lang_code in langs_to_try:
            try:
                ds = load_dataset(
                    "ai4bharat/sangraha",
                    "verified",
                    split=lang_code,
                    streaming=True,
                )
                per_lang = max_docs // len(langs_to_try)
                for i, row in enumerate(ds):
                    if i >= per_lang:
                        break
                    text = row.get("text", "")
                    if text:
                        docs.append({
                            "id":   row.get("doc_id", f"{lang_code}_{i}"),
                            "url":  "",
                            "text": text,
                            "declared_lang": "hi" if lang_code == "hin" else "te",
                            "source_type": row.get("type", ""),
                        })
                short = "hi" if lang_code == "hin" else "te"
                print(f"  Loaded {len([d for d in docs if d.get('declared_lang')==short]):,} {short} docs")
            except Exception as e:
                print(f"  Warning: Could not load split {lang_code}: {e}")
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


def load_c4_crawl(max_docs: int = 20_000) -> list[dict]:
    """
    Load genuinely messy web-crawl documents from allenai/c4 en.noclean split.

    WHY en.noclean and NOT realnewslike:
      - realnewslike = already quality-filtered to news-like pages → 98%+ survival
      - en.noclean   = raw unfiltered CommonCrawl text, no heuristic quality
                       filtering applied → nav boilerplate, SEO spam, symbol-
                       heavy pages, short stubs, repeated boilerplate all present

    This is the split used in the C4 paper ablation studies specifically to
    demonstrate the effect of quality filtering — exactly what Session 4 teaches.
    """
    print(f"\nLoading allenai/c4 en.noclean (up to {max_docs:,} docs)...")
    print("  Source: Unfiltered CommonCrawl — real nav boilerplate, SEO spam, ")
    print("          symbol-heavy pages, short stubs, multilingual noise.")
    docs = []

    try:
        from datasets import load_dataset
        ds = load_dataset(
            "allenai/c4",
            "en.noclean",       # ← unfiltered raw web crawl
            split="train",
            streaming=True,
        )
        for i, row in enumerate(ds):
            if i >= max_docs:
                break
            text = row.get("text", "")
            if text:
                docs.append({
                    "id":  f"c4_{i}",
                    "url": row.get("url", ""),
                    "text": text,
                    "timestamp": row.get("timestamp", ""),
                })
            if (i + 1) % 5_000 == 0:
                print(f"  Streamed {i+1:,} docs...")
    except Exception as e:
        print(f"  Warning: C4 en.noclean access failed: {e}")
        docs = []

    if not docs:
        print("  Falling back to synthetic messy web sample...")
        docs = _synthetic_messy_web_sample(max_docs)

    print(f"  Done: {len(docs):,} web crawl docs loaded")
    return docs


# Keep old name as alias for backwards compat
load_c4_realnews = load_c4_crawl


def _synthetic_messy_web_sample(n: int = 20_000) -> list[dict]:
    """
    Synthetic messy web crawl sample mimicking real CommonCrawl noise:
    - Navigation boilerplate pages (short, non-prose)
    - Non-English pages mixed in
    - High symbol-ratio pages (JS/CSS fragments)
    - SEO keyword-stuffed pages
    - Clean news articles (what we want to keep)
    - Near-duplicate syndicated content
    - Cookie banner / legal page boilerplate
    - Low-confidence language detection pages (code-mixed)
    """
    import random
    import hashlib
    random.seed(42)

    # ── Templates for different noise types ──────────────────────────────────

    nav_boilerplate = [
        "Home | About | Contact | Privacy Policy | Terms of Service\nLogin | Register | Help | FAQ\nCopyright {year} All Rights Reserved. Sitemap | RSS Feed",
        "Skip to main content\nMenu\nSearch\nHome\nNews\nSports\nEntertainment\nBusiness\nTechnology\nHealth\nScience\nTravel\nCookies Policy | Accessibility | Advertise with us",
        "Home > News > Local\nShare Tweet Email\nRelated Articles\nMost Popular\nRecently Viewed\nNewsletter Sign Up\nGet breaking news alerts. Never miss a story.",
        "ADVERTISEMENT\nSkip Ad\nYour video will start in 5 seconds.\nClose Ad\nClose X\n© {year} Media Corp. All Rights Reserved.",
    ]

    seo_spam = [
        "Best cheap flights deals online. Book cheap flights today. Cheap airline tickets. Cheap flights. Compare cheap flights. Find cheap flights online. Book now save money. Cheap international flights. Domestic flights cheap. Last minute cheap flights.",
        "Buy cheap car insurance online. Compare car insurance quotes. Best car insurance rates. Cheap car insurance quotes. Auto insurance comparison. Get free quotes today. Save on car insurance. Cheap auto insurance online.",
        "Free online games. Play free online games. Best free games. Online games free play. Free browser games. HTML5 games free. Play online now. Best free online games 2024.",
    ]

    non_english = [
        "Le gouvernement français a annoncé aujourd'hui de nouvelles mesures pour lutter contre l'inflation. Le Premier ministre a déclaré que ces mesures aideront les ménages à faibles revenus. L'économie française continue de faire face à des défis importants malgré une légère amélioration des indicateurs.",
        "Die Bundesregierung hat heute neue Maßnahmen zur Bekämpfung der Inflation angekündigt. Der Bundeskanzler erklärte, dass diese Maßnahmen einkommensschwachen Haushalten helfen werden. Die deutsche Wirtschaft steht weiterhin vor Herausforderungen, obwohl einige Indikatoren leichte Verbesserungen zeigen.",
        "El gobierno español anunció hoy nuevas medidas para combatir la inflación. El presidente del gobierno declaró que estas medidas ayudarán a los hogares con ingresos más bajos. La economía española sigue enfrentando desafíos importantes a pesar de una ligera mejora de los indicadores.",
        "Il governo italiano ha annunciato oggi nuove misure per combattere l'inflazione. Il Presidente del Consiglio ha dichiarato che queste misure aiuteranno le famiglie a basso reddito. L'economia italiana continua ad affrontare sfide importanti nonostante un leggero miglioramento degli indicatori.",
        "Правительство России объявило сегодня о новых мерах по борьбе с инфляцией. Премьер-министр заявил, что эти меры помогут малообеспеченным домохозяйствам. Российская экономика продолжает сталкиваться с серьезными вызовами.",
        "中国政府今天宣布了新的反通货膨胀措施。总理表示，这些措施将帮助低收入家庭。尽管一些指标略有改善，但中国经济继续面临重大挑战。",
    ]

    symbol_heavy = [
        "function loadAd() { var ad = document.getElementById('ad-slot'); if (ad) { var cfg = {'type':'banner','w':728,'h':90,'key':'abc123','cb':Math.random()}; var url = 'https://ads.example.com/serve?'+Object.keys(cfg).map(k=>k+'='+cfg[k]).join('&'); ad.src=url; } } window.onload=loadAd;",
        ".nav-wrapper { display: flex; justify-content: space-between; align-items: center; padding: 0 20px; background: #fff; border-bottom: 1px solid #e0e0e0; } .nav-logo { font-size: 24px; font-weight: 700; color: #333; } @media (max-width: 768px) { .nav-wrapper { flex-direction: column; padding: 10px; } }",
        ">>> import numpy as np\n>>> arr = np.random.rand(100, 100)\n>>> result = np.linalg.eig(arr)\n>>> print(f'eigenvalues: {result[0][:5]}')\n[0.234+1.2j, 0.56+0j, ...]",
    ]

    cookie_legal = [
        "Privacy Policy\nLast updated: January 1, 2024\nWe use cookies. By continuing to use this site you accept our cookie policy.\nTypes of cookies we use:\n- Essential cookies\n- Analytics cookies\n- Marketing cookies\nYou can opt out at any time. Contact us at privacy@example.com. See also: Terms of Service | GDPR Rights | California Privacy Rights | Do Not Sell My Information",
        "Terms and Conditions\nBy accessing this website you agree to these terms. Unauthorised use of this website may give rise to a claim for damages. This website uses cookies.\nAll content is © Copyright {year}. All rights reserved. Reproduction prohibited without permission.",
    ]

    clean_news = [
        "The Federal Reserve raised interest rates by 25 basis points on Wednesday, marking the tenth consecutive increase as the central bank continues its battle against persistent inflation. Fed Chair Jerome Powell said further hikes could be forthcoming if inflation does not cool. Markets reacted negatively, with the S&P 500 falling 1.2% in afternoon trading. Economists are divided on whether the Fed can engineer a soft landing for the economy without triggering a recession.",
        "Scientists at MIT have developed a new battery technology that could charge electric vehicles in under five minutes while storing three times more energy than lithium-ion batteries. The breakthrough, published Thursday in the journal Nature Energy, uses a novel solid-state electrolyte that remains stable at high charging voltages. Researchers say commercial production could begin within five years if manufacturing challenges are resolved.",
        "The Supreme Court ruled 6-3 that states may restrict access to social media platforms under certain conditions, in a landmark decision that could reshape how tech companies moderate content. Justice Clarence Thomas wrote the majority opinion, saying states have a legitimate interest in preventing viewpoint discrimination. Critics argue the ruling undermines the First Amendment rights of private companies.",
        "A powerful 7.4 magnitude earthquake struck the coast of Japan early Tuesday morning, triggering tsunami warnings across the Pacific. Japanese authorities ordered evacuations of coastal communities. The quake, centered 50 miles offshore from Honshu, was felt as far away as Tokyo. Initial reports indicate significant structural damage in several coastal towns, though the full extent of casualties remains unclear.",
        "Global leaders gathered in Dubai for COP28 amid intense pressure to phase out fossil fuels. Nearly 200 countries are represented at the United Nations climate conference, which runs through December 12. The United Arab Emirates, as host country, has faced criticism over its status as a major oil producer. However, officials say the location presents a unique opportunity to engage petrostate cooperation on the transition to renewables.",
        "Apple Inc. reported quarterly earnings that beat analyst expectations, driven by strong iPhone sales in emerging markets and growth in its services segment. Revenue rose 8% year-over-year to $94.8 billion. CEO Tim Cook said demand for the iPhone 15 lineup remained robust despite macroeconomic headwinds. Apple's services revenue, which includes the App Store, Apple Music and iCloud, hit a record $23.2 billion.",
        "The Biden administration announced a sweeping new rule requiring automakers to significantly increase fuel efficiency standards for vehicles sold in the United States through 2032. The rule, finalized by the Environmental Protection Agency, aims to cut carbon dioxide emissions by more than 7 billion tons over the regulation's lifetime. Industry groups said the timeline is too aggressive and could increase vehicle costs for consumers.",
        "Researchers have identified a previously unknown species of deep-sea fish in the Pacific Ocean off the coast of New Zealand. The translucent creature, discovered at depths exceeding 8,000 meters, possesses an unusual bioluminescent pattern that scientists believe serves as camouflage against predators. The finding adds to growing evidence that the deep ocean remains one of Earth's least explored frontiers.",
    ]

    near_dup_base = "A wildfire burning in northern California has forced the evacuation of thousands of residents as crews battle the blaze across rugged terrain. The fire, which broke out Monday amid record heat and low humidity, has consumed more than 15,000 acres and is 10 percent contained. Governor Gavin Newsom declared a state of emergency for three counties. Air quality alerts have been issued across a wide region as smoke drifts southward."

    samples = []

    # Distribution to mimic messy web crawl:
    # 35% clean news (what survives)
    # 15% nav boilerplate (short, fails quality)
    # 12% SEO spam (fails quality)
    # 12% non-English pages (fails lang ID)
    # 8% symbol-heavy (fails quality)
    # 8% cookie/legal (fails quality)
    # 10% near-duplicates (dedup removes)

    total = n
    counts = {
        "clean":    int(total * 0.35),
        "nav":      int(total * 0.15),
        "seo":      int(total * 0.12),
        "foreign":  int(total * 0.12),
        "symbol":   int(total * 0.08),
        "legal":    int(total * 0.08),
        "neardup":  int(total * 0.10),
    }

    idx = 0
    for kind, cnt in counts.items():
        for i in range(cnt):
            if kind == "clean":
                base = clean_news[i % len(clean_news)]
                # Vary slightly but keep coherent
                variation = f" This report was filed on {2023 + (i % 2)}-{(i % 12)+1:02d}-{(i % 28)+1:02d}. " \
                            f"Updated at {(i % 24):02d}:{(i % 60):02d} GMT."
                text = base + variation
            elif kind == "nav":
                base = nav_boilerplate[i % len(nav_boilerplate)]
                text = base.replace("{year}", str(2022 + i % 3))
                text += f"\nPage {i+1} | Sort by: Relevance | Date | Popularity"
            elif kind == "seo":
                base = seo_spam[i % len(seo_spam)]
                text = base + f" Visit us online. Call {1800 + i}-555-{1000 + i}."
            elif kind == "foreign":
                text = non_english[i % len(non_english)]
            elif kind == "symbol":
                base = symbol_heavy[i % len(symbol_heavy)]
                text = base + f" // version {i}.{i % 10}.{i % 5} build {1000+i}"
            elif kind == "legal":
                base = cookie_legal[i % len(cookie_legal)]
                text = base.replace("{year}", str(2022 + i % 3))
            elif kind == "neardup":
                # Slight variation of near-dup base
                text = near_dup_base + f" Incident number {i+1}. Crews from {i+1} counties responded."

            sha = hashlib.sha256(text.encode()).hexdigest()[:8]
            samples.append({
                "id": f"c4_{kind}_{idx}_{sha}",
                "url": f"https://news-site-{idx % 500}.com/article/{idx}",
                "text": text,
                "_kind": kind,  # debug label, stripped before pipeline
            })
            idx += 1

    random.shuffle(samples)
    # Strip debug labels
    for s in samples:
        s.pop("_kind", None)
    return samples[:n]


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
    # Must use the SAME estimator every stage uses (stats_tracker.estimate_tokens),
    # otherwise cumulative_survival_pct is computed against a different token
    # scale than every stage reports and can read >100% -- the bug this fixes.
    tracker.initial_tokens = sum(estimate_tokens(d["text"]) for d in docs)

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
    parser.add_argument("--run", choices=["wikipedia", "sangraha", "c4", "all",
                                          "both"],  # 'both' kept for backwards compat
                        default="both", help="Which dataset run to execute")
    parser.add_argument("--wiki-docs",  type=int, default=100_000,
                        help="Number of Wikipedia articles to process")
    parser.add_argument("--indic-docs", type=int, default=20_000,
                        help="Number of Sangraha docs to process")
    parser.add_argument("--c4-docs",    type=int, default=20_000,
                        help="Number of C4 realnewslike docs to process")
    args = parser.parse_args()

    results = []

    if args.run in ("wikipedia", "both", "all"):
        wiki_docs = load_wikipedia(max_docs=args.wiki_docs)
        r = run_pipeline(
            wiki_docs,
            run_name="wikipedia",
            expected_lang="en",
            source="wikimedia/wikipedia",
            license_class="CC-BY-SA-4.0",
        )
        results.append(r)

    if args.run in ("sangraha", "both", "all"):
        indic_docs = load_sangraha(max_docs=args.indic_docs)
        r = run_pipeline(
            indic_docs,
            run_name="sangraha",
            expected_lang="hi",   # primary, but accepts all Indic
            source="ai4bharat/sangraha",
            license_class="CC-BY-4.0",
        )
        results.append(r)

    if args.run in ("c4", "all"):
        c4_docs = load_c4_crawl(max_docs=args.c4_docs)
        r = run_pipeline(
            c4_docs,
            run_name="c4_crawl",
            expected_lang="en",
            source="allenai/c4 en.noclean",
            license_class="ODC-BY",
        )
        results.append(r)

    # Save combined summary
    summary_path = DATA_DIR / "pipeline_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Summary saved to: {summary_path}")
