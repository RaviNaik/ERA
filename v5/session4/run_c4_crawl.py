"""
run_c4_crawl.py
---------------
Runs the full 8-stage cleaning pipeline on a synthetic messy web crawl dataset
that mimics real CommonCrawl / realnewslike noise:
  - Nav boilerplate pages  (15%) → fails quality filter
  - SEO keyword spam       (12%) → fails quality filter
  - Non-English pages      (12%) → fails language ID
  - Symbol-heavy JS/CSS    ( 8%) → fails quality filter
  - Cookie/legal pages     ( 8%) → fails quality filter
  - Near-duplicate content (10%) → removed by dedup
  - Clean news articles    (35%) → admitted

Expected survival: ~35-45% (matches trainer's typical web crawl benchmarks)
Produces: data/c4_crawl_stats.json
"""

import sys
import json
import hashlib
import pathlib
import time
import random

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR   = pathlib.Path(__file__).parent
PIPELINE_DIR = SCRIPT_DIR / "pipeline"
DATA_DIR     = SCRIPT_DIR / "data"
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


# ── Noise distributions matching real web crawl ──────────────────────────────

NAV_BOILERPLATE = [
    "Home | About | Contact | Privacy Policy | Terms of Service\n"
    "Login | Register | Help | FAQ\n"
    "Copyright 2024 All Rights Reserved. Sitemap | RSS Feed\n"
    "Follow us: Facebook | Twitter | Instagram | LinkedIn",

    "Skip to main content\nMenu\nSearch\nHome\nNews\nSports\nEntertainment\n"
    "Business\nTechnology\nHealth\nScience\nTravel\n"
    "Cookies Policy | Accessibility | Advertise with us",

    "Home > News > Local\nShare Tweet Email Print Save\nRelated Articles\n"
    "Most Popular\nRecently Viewed\nNewsletter Sign Up\n"
    "Get breaking news alerts. Never miss a story.\nSubscribe now.",

    "ADVERTISEMENT\nSkip Ad\nYour video will start in 5 seconds.\n"
    "Close Ad   Close X\n© 2024 Media Corp. All Rights Reserved.\n"
    "About | Privacy | Terms | Contact | Advertise | Careers",

    "Menu\nHome  Sports  Business  Entertainment  World  Science  Health\n"
    "Search  Login  Subscribe\nToday's Paper  E-Edition  Archives\n"
    "© 2024 Publishing Group Inc. All rights reserved.",
]

SEO_SPAM = [
    "Best cheap flights deals online. Book cheap flights today. Cheap airline tickets. "
    "Cheap flights. Compare cheap flights. Find cheap flights online. Book now save money. "
    "Cheap international flights. Domestic flights cheap. Last minute cheap flights. "
    "Discount airfare deals. Budget airline tickets. Cheap first class tickets. Flight deals today.",

    "Buy cheap car insurance online. Compare car insurance quotes. Best car insurance rates. "
    "Cheap car insurance quotes. Auto insurance comparison. Get free quotes today. "
    "Save on car insurance. Cheap auto insurance online. Low cost car insurance. "
    "Affordable auto insurance. Car insurance deals. Discount car insurance rates.",

    "Free online games. Play free online games. Best free games. Online games free play. "
    "Free browser games. HTML5 games free. Play online now. Best free online games 2024. "
    "No download games. Instant play games. Multiplayer online games free. Flash games alternatives.",

    "Lose weight fast. Best diet pills 2024. Weight loss supplements. Burn fat fast. "
    "Best weight loss program. How to lose belly fat. Diet tips to lose weight fast. "
    "Keto diet plan. Intermittent fasting results. Best exercises for weight loss fast at home.",

    "Make money online fast. Work from home jobs. Passive income ideas 2024. "
    "Best side hustles. How to make $1000 a day online. Legitimate work from home opportunities. "
    "Online income without investment. Freelance jobs online for beginners. Make money fast today.",
]

NON_ENGLISH = [
    "Le gouvernement français a annoncé aujourd'hui de nouvelles mesures pour lutter contre "
    "l'inflation croissante. Le Premier ministre a déclaré lors d'une conférence de presse que "
    "ces mesures cibleront spécifiquement les ménages à faibles revenus les plus touchés par "
    "la hausse des prix de l'énergie et des denrées alimentaires. L'opposition a critiqué ces "
    "annonces, les qualifiant d'insuffisantes face à l'ampleur de la crise économique actuelle.",

    "Die Bundesregierung hat heute ein umfassendes Paket neuer Maßnahmen zur Bekämpfung der "
    "Inflation angekündigt. Der Bundeskanzler erklärte auf einer Pressekonferenz, dass diese "
    "Maßnahmen gezielt einkommensschwache Haushalte unterstützen sollen, die am stärksten von "
    "steigenden Energie- und Lebensmittelpreisen betroffen sind. Die Opposition bezeichnete "
    "die Ankündigungen als unzureichend angesichts der wirtschaftlichen Herausforderungen.",

    "El gobierno español anunció hoy un amplio paquete de nuevas medidas para combatir la "
    "inflación creciente. El presidente del gobierno declaró en rueda de prensa que estas "
    "medidas se centrarán específicamente en los hogares de bajos ingresos más afectados por "
    "el aumento de los precios de la energía y los alimentos. La oposición criticó los anuncios "
    "calificándolos de insuficientes ante la magnitud de la crisis económica actual.",

    "Il governo italiano ha annunciato oggi un ampio pacchetto di nuove misure per combattere "
    "l'inflazione crescente. Il Presidente del Consiglio ha dichiarato in conferenza stampa che "
    "queste misure si concentreranno specificamente sulle famiglie a basso reddito più colpite "
    "dall'aumento dei prezzi dell'energia e dei prodotti alimentari. L'opposizione ha criticato "
    "gli annunci definendoli insufficienti di fronte alla portata dell'attuale crisi economica.",

    "Правительство России объявило сегодня о масштабном пакете новых мер по борьбе с "
    "растущей инфляцией. Премьер-министр заявил на пресс-конференции, что эти меры будут "
    "направлены именно на малоимущие домохозяйства, наиболее пострадавшие от роста цен на "
    "энергоносители и продукты питания. Оппозиция подвергла критике заявления, назвав их "
    "недостаточными с учётом масштабов текущего экономического кризиса.",

    "中国政府今天宣布了一套全面的新措施，以应对日益严峻的通货膨胀问题。总理在新闻发布会上表示，"
    "这些措施将专门针对受能源和食品价格上涨影响最严重的低收入家庭。反对派对这些声明进行了批评，"
    "称其在当前经济危机规模面前显得远远不够。中国经济继续面临重大挑战，需要采取更为有力的政策措施。",

    "Japonya hükümeti bugün artan enflasyonla mücadele için kapsamlı bir önlemler paketi "
    "açıkladı. Başbakan, bir basın toplantısında bu önlemlerin özellikle enerji ve gıda "
    "fiyatlarındaki artıştan en çok etkilenen düşük gelirli haneleri hedef alacağını söyledi. "
    "Muhalefet açıklamaları, mevcut ekonomik krizin boyutu karşısında yetersiz bularak eleştirdi.",
]

SYMBOL_HEAVY = [
    "function initTracking(){var _paq=window._paq=window._paq||[];_paq.push(['trackPageView']);"
    "_paq.push(['enableLinkTracking']);(function(){var u='//analytics.example.com/';_paq.push"
    "(['setTrackerUrl',u+'matomo.php']);_paq.push(['setSiteId','42']);var d=document,g=d.createElement"
    "('script'),s=d.getElementsByTagName('script')[0];g.async=true;g.src=u+'matomo.js';"
    "s.parentNode.insertBefore(g,s)})();}",

    ".wrapper{max-width:1200px;margin:0 auto;padding:0 20px}.header{background:#1a1a2e;"
    "color:#eee;padding:15px 0}.nav ul{list-style:none;display:flex;gap:20px}.nav a{"
    "color:#fff;text-decoration:none;font-weight:500;transition:color .3s}.nav a:hover{"
    "color:#e94560}@media(max-width:768px){.nav ul{flex-direction:column}.header{padding:10px 0}}",

    "SELECT u.id, u.name, u.email, COUNT(o.id) as order_count, SUM(o.total) as revenue "
    "FROM users u LEFT JOIN orders o ON u.id=o.user_id WHERE u.created_at >= '2024-01-01' "
    "AND u.status='active' GROUP BY u.id, u.name, u.email HAVING revenue > 1000 "
    "ORDER BY revenue DESC LIMIT 100; -- index: idx_users_status_created",
]

COOKIE_LEGAL = [
    "Privacy Policy | Last updated: January 1, 2024\n"
    "We use cookies and similar tracking technologies to track activity on our service.\n"
    "Types of Data Collected: Personal Data (email address, first name, last name, phone number)\n"
    "Usage Data (IP address, browser type, pages visited, time spent, referring URL)\n"
    "Cookies: Session Cookies | Persistent Cookies | Preference Cookies | Security Cookies\n"
    "You have the right to: access, rectify, object, restrict, portability, and erasure.\n"
    "Contact: privacy@example.com | DPO: dpo@example.com\n"
    "See also: Terms of Service | GDPR Rights | CCPA | Do Not Sell My Information",

    "Terms and Conditions of Use\n"
    "1. Acceptance: By accessing this site you agree to be bound by these Terms.\n"
    "2. Intellectual Property: All content © 2024 Example Corp. All rights reserved.\n"
    "3. Limitation of Liability: In no event shall Example Corp be liable for any indirect, "
    "incidental, punitive or consequential damages.\n"
    "4. Governing Law: These Terms shall be governed by the laws of the State of California.\n"
    "5. Severability: If any provision is found invalid, remaining terms stay in force.\n"
    "Contact us: legal@example.com | +1-800-555-0100",

    "Cookie Notice\nThis website uses cookies to enhance user experience.\n"
    "Essential cookies: required for the website to function properly.\n"
    "Analytics cookies: help us understand how visitors interact with the website.\n"
    "Marketing cookies: used to deliver relevant advertisements.\n"
    "Preferences cookies: remember your preferences and settings.\n"
    "Accept All | Reject Non-Essential | Manage Preferences\n"
    "Learn more in our Privacy Policy. Last updated: March 2024.",
]

CLEAN_NEWS = [
    "The Federal Reserve raised interest rates by 25 basis points on Wednesday, marking the "
    "eleventh consecutive increase as the central bank continues its battle against persistent "
    "inflation. Fed Chair Jerome Powell said further hikes could be forthcoming if inflation "
    "does not cool meaningfully over the coming months. Markets reacted negatively to the "
    "announcement, with the S&P 500 falling 1.4% in afternoon trading and Treasury yields "
    "rising sharply. Economists remain divided on whether the Fed can engineer a soft landing "
    "for the economy without triggering a broad recession. Consumer spending has held up "
    "remarkably well despite higher borrowing costs, complicating the central bank's outlook.",

    "Scientists at MIT have developed a new solid-state battery technology that could charge "
    "electric vehicles in under five minutes while storing three times more energy than "
    "conventional lithium-ion batteries. The breakthrough, published Thursday in the journal "
    "Nature Energy, uses a novel sulfide-based electrolyte that remains stable at high charging "
    "voltages and temperatures. Lead researcher Dr. Sarah Chen said the team overcame the "
    "main challenge of dendrite formation that has historically plagued solid-state designs. "
    "Industry analysts say commercialization could begin within five years if manufacturing "
    "challenges at scale are resolved. Several major automakers have already expressed interest.",

    "Global leaders gathered in Dubai for COP28 amid intense pressure to agree on a clear "
    "phase-out of fossil fuels for the first time in the history of UN climate negotiations. "
    "Nearly 200 countries are represented at the conference, which runs through December 12. "
    "The United Arab Emirates, as host country, has faced significant criticism over its dual "
    "role as a major oil producer and climate conference host. However, UAE officials said the "
    "location presents a unique opportunity to bring petrostate cooperation on clean energy "
    "investments. A contentious debate over 'phasing out' versus 'phasing down' fossil fuel "
    "language dominated the first days of negotiations.",

    "Apple Inc. reported quarterly earnings that beat analyst expectations by a wide margin, "
    "driven by unexpectedly strong iPhone sales in India and Southeast Asia and accelerating "
    "growth in its high-margin services segment. Total revenue rose 8% year-over-year to "
    "$94.8 billion, topping the Wall Street consensus estimate of $91.2 billion. CEO Tim Cook "
    "said demand for the iPhone 15 Pro lineup remained exceptionally robust despite broader "
    "macroeconomic headwinds and a slowing smartphone market globally. Apple's services revenue, "
    "which includes the App Store, Apple Music, iCloud and Apple TV+, hit a record $23.2 "
    "billion for the quarter, underscoring the company's strategic pivot toward recurring revenues.",

    "A powerful 7.4 magnitude earthquake struck the coast of Japan early Tuesday morning, "
    "triggering tsunami warnings across much of the Pacific basin. Japanese authorities ordered "
    "the immediate evacuation of tens of thousands of coastal community residents across four "
    "prefectures. The quake, centered approximately 50 miles offshore from the Noto Peninsula "
    "on Honshu island, was felt as far away as Tokyo, 300 miles to the southwest. Major "
    "infrastructure damage was reported across several coastal towns, with roads buckled, "
    "bridges collapsed, and port facilities submerged. The Japan Meteorological Agency said "
    "aftershocks exceeding magnitude 5.0 were expected to continue for several days.",

    "The Supreme Court ruled 6-3 that states may restrict how large social media platforms "
    "moderate user content under certain conditions, in a landmark decision that could "
    "fundamentally reshape the internet's content moderation landscape. Justice Clarence Thomas "
    "wrote the majority opinion, holding that states have a legitimate interest in preventing "
    "viewpoint-based discrimination by dominant online platforms. The decision arose from "
    "challenges to laws passed by Florida and Texas requiring major platforms to host content "
    "they would otherwise remove. First Amendment scholars said the ruling raises profound "
    "questions about the constitutional status of editorial discretion exercised by private companies.",

    "Researchers have identified a previously unknown species of deep-sea fish in the Pacific "
    "Ocean at depths exceeding 8,200 meters off the coast of New Zealand's North Island. "
    "The translucent creature, temporarily named Pseudoliparis swirei 2, was captured on "
    "video by a remotely operated deep-sea submersible during a six-week research expedition. "
    "It possesses an unusual multi-spectral bioluminescent pattern that scientists believe may "
    "serve simultaneously as prey luring and predator camouflage. The finding adds to growing "
    "scientific evidence that the hadal zone — ocean depths below 6,000 meters — remains one "
    "of Earth's most biodiverse and least understood ecosystems.",

    "The Biden administration unveiled a sweeping new federal rule requiring automakers to "
    "dramatically increase average fuel efficiency across their vehicle fleets through 2032, "
    "the most aggressive emissions standard in American history. The rule, finalized by the "
    "Environmental Protection Agency after two years of development and public comment, aims "
    "to cut carbon dioxide emissions by more than 7.3 billion tons over the regulation's "
    "projected lifetime. Major automakers said the timeline is technically feasible but "
    "financially challenging, warning that compliance costs could push average new vehicle "
    "prices significantly higher for American consumers already stretched by elevated inflation.",
]

NEAR_DUP_BASE = (
    "A rapidly spreading wildfire burning in northern California has forced the mandatory "
    "evacuation of more than 25,000 residents as firefighting crews battle the blaze across "
    "steep, rugged terrain made inaccessible by the fire's erratic spread. The fire, which "
    "ignited Monday during a record-breaking heat wave with relative humidity below 5%, has "
    "consumed more than 42,000 acres and is approximately 8 percent contained as of Thursday "
    "morning. Governor Gavin Newsom declared a state of emergency for three northern California "
    "counties, activating National Guard resources. Air quality warnings have been issued across "
    "a 300-mile corridor as thick smoke drifts southward toward the Bay Area."
)


def generate_messy_web_sample(n: int = 20_000) -> list[dict]:
    """
    Generate a realistic messy web crawl sample of n documents.
    Distribution mirrors typical CommonCrawl realnews content:
      35% clean news articles   → most survive the pipeline
      15% nav boilerplate       → fail quality (short lines, no punct)
      12% SEO keyword spam      → fail quality (low stopword density, dup bigrams)
      12% non-English pages     → fail language ID
       8% symbol-heavy JS/CSS   → fail quality (high symbol ratio)
       8% cookie/legal pages    → fail quality (list structure, no prose)
      10% near-duplicate news   → removed by MinHash dedup
    """
    random.seed(42)

    counts = {
        "clean":   int(n * 0.35),
        "nav":     int(n * 0.15),
        "seo":     int(n * 0.12),
        "foreign": int(n * 0.12),
        "symbol":  int(n * 0.08),
        "legal":   int(n * 0.08),
        "neardup": int(n * 0.10),
    }

    samples = []
    idx = 0

    for kind, cnt in counts.items():
        for i in range(cnt):
            if kind == "clean":
                base = CLEAN_NEWS[i % len(CLEAN_NEWS)]
                variation = (
                    f" Filed: {2023 + (i % 2)}-{(i % 12)+1:02d}-{(i % 28)+1:02d}."
                    f" By staff reporter. Updated {(i % 24):02d}:{(i % 60):02d} GMT."
                )
                text = base + variation

            elif kind == "nav":
                base = NAV_BOILERPLATE[i % len(NAV_BOILERPLATE)]
                text = base + f"\nPage {i+1} of results | Items per page: 10 | 25 | 50"

            elif kind == "seo":
                base = SEO_SPAM[i % len(SEO_SPAM)]
                text = base + (
                    f" Best {kind} deals {2022 + i % 3}. Top rated. Guaranteed."
                    f" Call now: {1800+i}-555-{1000+i}. Limited time offer."
                )

            elif kind == "foreign":
                text = NON_ENGLISH[i % len(NON_ENGLISH)]

            elif kind == "symbol":
                base = SYMBOL_HEAVY[i % len(SYMBOL_HEAVY)]
                text = base + f"\n// build {1200+i} | minified | {(i%100)/10:.1f}kB"

            elif kind == "legal":
                base = COOKIE_LEGAL[i % len(COOKIE_LEGAL)]
                text = base + f"\nVersion {1 + i//100}.{i % 100} | Effective {2022 + i%3}-01-01"

            elif kind == "neardup":
                # Near-duplicate with minor variation — will be caught by MinHash LSH
                text = (
                    NEAR_DUP_BASE
                    + f" Update {i+1}: Incident command confirmed {3000+i} structures threatened."
                    f" Evacuation center at site {i % 10 + 1} serving {500 + i*3} evacuees."
                )

            sha = hashlib.sha256(text.encode()).hexdigest()[:8]
            samples.append({
                "id":  f"webcrawl_{kind}_{idx}_{sha}",
                "url": f"https://news-source-{idx % 800}.com/article-{idx}",
                "text": text,
            })
            idx += 1

    random.shuffle(samples)
    samples = samples[:n]
    print(f"  Generated {len(samples):,} docs")
    print(f"  Distribution: {counts}")
    return samples


def script_sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    print("\n" + "="*60)
    print("  MESSY WEB CRAWL RUN — Synthetic CommonCrawl Sample")
    print("  Mimics: nav boilerplate + SEO spam + non-English +")
    print("          symbol-heavy + legal pages + near-duplicates")
    print("="*60)

    # ── Load docs ─────────────────────────────────────────────────
    print("\nGenerating messy web crawl sample (20,000 docs)...")
    docs = generate_messy_web_sample(20_000)

    tracker = PipelineTracker("c4_crawl", DATA_DIR)
    tracker.initial_docs   = len(docs)
    tracker.initial_tokens = sum(len(d["text"].split()) for d in docs)
    script_hash = script_sha256(SCRIPT_DIR / "run_pipeline.py")

    print(f"\n  {len(docs):,} docs | ~{tracker.initial_tokens:,} words loaded")
    print("="*60)

    # ── Stage 1: Extract ──────────────────────────────────────────
    s = tracker.begin_stage(1, "Extract")
    docs = run_stage1(docs, s)
    tracker.end_stage(s)

    # ── Stage 2: Normalize ────────────────────────────────────────
    s = tracker.begin_stage(2, "Normalize")
    docs = run_stage2(docs, s)
    tracker.end_stage(s)

    # ── Stage 3: Language ID ──────────────────────────────────────
    s = tracker.begin_stage(3, "Language ID")
    docs = run_stage3(docs, s, expected_lang="en")
    tracker.end_stage(s)

    # ── Stage 4: Quality Filter ───────────────────────────────────
    s = tracker.begin_stage(4, "Quality Filter")
    docs = run_stage4(docs, s)
    tracker.end_stage(s)

    # ── Stage 5: Deduplicate ──────────────────────────────────────
    s = tracker.begin_stage(5, "Deduplicate")
    docs = run_stage5(docs, s)
    tracker.end_stage(s)

    # ── Stage 6: PII Scrub ────────────────────────────────────────
    s = tracker.begin_stage(6, "PII Scrub")
    docs = run_stage6(docs, s)
    tracker.end_stage(s)

    # ── Stage 7: Decontaminate ────────────────────────────────────
    s = tracker.begin_stage(7, "Decontaminate")
    docs = run_stage7(docs, s)
    tracker.end_stage(s)

    # ── Stage 8: Manifest ─────────────────────────────────────────
    s = tracker.begin_stage(8, "Manifest")
    docs = run_stage8(
        docs, s,
        source="synthetic/web-crawl-sample",
        license_class="ODC-BY",
        contributor_id="era5-s4-c4_crawl",
        script_name="run_c4_crawl.py",
        script_hash=script_hash,
        manifests_dir=DATA_DIR / "manifests" / "c4_crawl",
    )
    tracker.end_stage(s)

    # ── Summary ───────────────────────────────────────────────────
    initial  = tracker.initial_docs
    admitted = len(docs)
    survival = 100 * admitted / max(initial, 1)

    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY: Web Crawl (Messy Synthetic)")
    print(f"  Input:    {initial:,} docs")
    print(f"  Admitted: {admitted:,} docs  ({survival:.1f}% survival)")
    print(f"{'='*60}\n")

    tracker.save()
    out_path = DATA_DIR / "c4_crawl_stats.json"
    print(f"Stats saved to: {out_path}")


if __name__ == "__main__":
    main()
