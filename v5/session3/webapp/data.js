// ── Benchmark data ──────────────────────────────────────────────
const BENCH = {
  general:[
  {name:'MMLU',gemma:85.2,bharat:85,gap:'≈',note:'Match via general web'},
    {name:'MMLU-Pro',gemma:82,bharat:78,gap:'close',note:'Professional reasoning'},
    {name:'GPQA Diamond',gemma:84.3,bharat:72,gap:'close',note:'PhD-level science'},
    {name:'IFEval',gemma:88,bharat:87,gap:'≈',note:'Instruction following'},
    {name:'WildBench',gemma:75,bharat:76,gap:'↑',note:'Open-ended chat'},
  ],
  indic:[
    {name:'MILU (AI4Bharat)',gemma:58,bharat:75,gap:'↑',note:'+17 pts via always-on Indic'},
    {name:'BharatBench',gemma:54,bharat:70,gap:'↑',note:'+16 pts via India classifier'},
    {name:'FLORES-200 (chrF)',gemma:68,bharat:76,gap:'↑',note:'+8 via IndicTrans2 synthetic'},
    {name:'IndicGenBench',gemma:61,bharat:72,gap:'↑',note:'MT, summarisation, QA'},
    {name:'IndicXTREME',gemma:72,bharat:82,gap:'↑',note:'Classification, QA, NLI'},
  ],
  coding:[
    {name:'LiveCodeBench v6',gemma:80,bharat:82,gap:'↑',note:'Fresh competitive problems'},
    {name:'HumanEval+',gemma:90,bharat:88,gap:'≈',note:'Python function gen'},
    {name:'SWE-Bench Verified',gemma:42,bharat:40,gap:'close',note:'GitHub issue resolution'},
    {name:'DS-1000',gemma:72,bharat:76,gap:'↑',note:'Data science tasks'},
    {name:'IndicEval-XL',gemma:38,bharat:62,gap:'↑',note:'NEW: Indic-prompted code'},
  ],
  math:[
    {name:'AIME 2024+2025',gemma:89.2,bharat:75,gap:'close',note:'Competition math'},
    {name:'MATH-500',gemma:92,bharat:86,gap:'≈',note:'Olympiad math'},
    {name:'GSM8K',gemma:97,bharat:97,gap:'≈',note:'Grade school word problems'},
    {name:'IIT-JEE Bench',gemma:64,bharat:72,gap:'↑',note:'NEW: Custom Indian entrance'},
    {name:'NEET-Bench',gemma:68,bharat:76,gap:'↑',note:'NEW: Medical entrance'},
  ],
  agentic:[
    {name:'τ²-bench',gemma:86.4,bharat:80,gap:'close',note:'Multi-turn tool use'},
    {name:'Berkeley BFCL v3',gemma:78,bharat:80,gap:'↑',note:'Function calling'},
    {name:'AgentBench',gemma:71,bharat:68,gap:'close',note:'Multi-step agent tasks'},
    {name:'ToolEval',gemma:72,bharat:76,gap:'↑',note:'Tool selection + execution'},
  ]
};

const BENCH_LABELS = {
  general:'General & Reasoning',indic:'Indic Language',
  coding:'Coding',math:'Math & Science',agentic:'Agentic'
};

const BENCH_INSIGHTS = {
  general:'General benchmarks: our model roughly matches Gemma 4 31B. We trade slight headroom on GPQA for massive Indic gains — the correct tradeoff for an India-first model.',
  indic:'This is the primary differentiation. MILU +17 pts and BharatBench +16 pts come from the always-on Indic Tier 2 channel and the India-perspective classifier. No other open model targets this gap.',
  coding:'LiveCodeBench +2 pts via code-heavy pre-training (15% share). IndicEval-XL is our custom benchmark for Indic-prompted coding — a market gap no existing model covers.',
  math:'IIT-JEE and NEET custom benchmarks cover the biggest exam prep market in the world. India has 2M+ JEE aspirants annually. This is a unique differentiator over any existing model.',
  agentic:'Agentic performance slightly below Gemma 4 on τ²-bench but ahead on BFCL and ToolEval due to India-specific API training (DigiLocker, IRCTC, UPI schemas).'
};

// ── Language / Tokenizer data ────────────────────────────────────
const LANGS = [
  {lang:'Hindi',     script:'Devanagari',speakers:'600M+',prio:1,curr:3.1,target:2.0,vocab:15000},
  {lang:'Bengali',   script:'Bengali',   speakers:'230M+',prio:1,curr:3.5,target:2.2,vocab:10000},
  {lang:'Marathi',   script:'Devanagari',speakers:'95M+', prio:1,curr:3.0,target:2.0,vocab:5000},
  {lang:'Telugu',    script:'Telugu',    speakers:'85M+', prio:1,curr:4.2,target:2.5,vocab:10000},
  {lang:'Tamil',     script:'Tamil',     speakers:'80M+', prio:1,curr:3.7,target:2.3,vocab:10000},
  {lang:'Gujarati',  script:'Gujarati',  speakers:'60M+', prio:1,curr:3.5,target:2.2,vocab:8000},
  {lang:'Urdu',      script:'Perso-Arabic',speakers:'70M+',prio:1,curr:2.5,target:1.8,vocab:6000},
  {lang:'Kannada',   script:'Kannada',  speakers:'60M+', prio:2,curr:4.2,target:2.5,vocab:8000},
  {lang:'Malayalam', script:'Malayalam',speakers:'35M+', prio:2,curr:5.0,target:2.3,vocab:8000},
  {lang:'Punjabi',   script:'Gurmukhi', speakers:'35M+', prio:2,curr:3.5,target:1.8,vocab:4000},
  {lang:'Odia',      script:'Odia',     speakers:'40M+', prio:2,curr:4.5,target:2.2,vocab:4000},
  {lang:'Sanskrit',  script:'Devanagari',speakers:'—',   prio:2,curr:2.5,target:1.8,vocab:3000},
  {lang:'Assamese',  script:'Bengali',  speakers:'15M+', prio:3,curr:4.0,target:2.3,vocab:0},
  {lang:'Maithili',  script:'Devanagari',speakers:'14M+',prio:3,curr:3.2,target:2.1,vocab:0},
  {lang:'Manipuri',  script:'Meitei',   speakers:'2M+',  prio:3,curr:5.5,target:2.8,vocab:0},
];

const VOCAB_ALLOC = [
  {cat:'English (general BPE)',   val:35000,color:'#6baed6'},
  {cat:'Hindi / Devanagari',       val:15000,color:'#9b82d0'},
  {cat:'Bengali',                   val:10000,color:'#52b788'},
  {cat:'Telugu',                    val:10000,color:'#e07b6a'},
  {cat:'Tamil',                     val:10000,color:'#f4a76f'},
  {cat:'Gujarati',                  val:8000, color:'#d48fbc'},
  {cat:'Kannada',                   val:8000, color:'#7ec8c8'},
  {cat:'Malayalam',                 val:8000, color:'#b5c34e'},
  {cat:'Marathi (overlap)',         val:5000, color:'#c09060'},
  {cat:'Urdu (Perso-Arabic)',       val:6000, color:'#a0c080'},
  {cat:'Punjabi/Odia/Others',       val:8000, color:'#e0b0d0'},
  {cat:'Code (Python/JS/Bash/SQL)', val:15000,color:'#4a90c4'},
  {cat:'Math / LaTeX / Science',    val:8000, color:'#3a9e72'},
  {cat:'Special / Control / Fallback',val:14000,color:'#b0a0c8'},
];

// ── Domain mix data ──────────────────────────────────────────────
const DOMAIN_MIX = [
  {name:'General Web (EN)', pct:40,color:'#6baed6',tokens:'~6T'},
  {name:'Indic Monolingual',pct:12,color:'#52b788',tokens:'~1.8T'},
  {name:'Code',             pct:15,color:'#9b82d0',tokens:'~2.25T'},
  {name:'Math & Science',   pct:8, color:'#f4a76f',tokens:'~1.2T'},
  {name:'Books & Long-form',pct:10,color:'#d48fbc',tokens:'~1.5T'},
  {name:'Parallel Indic-EN',pct:5, color:'#7ec8c8',tokens:'~750B'},
  {name:'India-specific',   pct:5, color:'#e07b6a',tokens:'~750B'},
  {name:'Agentic/Structured',pct:3,color:'#b5c34e',tokens:'~450B'},
  {name:'Conversational',   pct:2, color:'#c09060',tokens:'~300B'},
];

const ANNEAL_DOMAINS = [
  {name:'General Web',     pre:40,ann:5},
  {name:'Indic (Tier 2)',  pre:12,ann:30},
  {name:'Code',            pre:15,ann:20},
  {name:'Math & Science',  pre:8, ann:25},
  {name:'India-specific',  pre:5, ann:12},
  {name:'Agentic/Struct.', pre:3, ann:8},
];

const KEY_SOURCES = [
  {tier:'T1',name:'FineWeb-Edu',scale:'1.3T tokens',why:'Highest educational signal/token ratio; ODC-By license'},
  {tier:'T1',name:'FineWeb',scale:'4T tokens (sample)',why:'Broad web; India-perspective classifier applied at ingestion'},
  {tier:'T1',name:'The Stack v2',scale:'2T tokens',why:'600+ PL; use permissive subset; Python/JS/TS primary'},
  {tier:'T1',name:'arXiv + OpenWebMath',scale:'195B tokens',why:'Physics, Math, CS with LaTeX preserved for symbol tokens'},
  {tier:'T1',name:'NCERT + IIT OCW',scale:'5B tokens',why:'India curriculum grounding; Class 6–12 Hindi + English'},
  {tier:'T2',name:'Sangraha Verified',scale:'50B tokens',why:'AI4Bharat gold-standard; 22 Indic langs; CC-BY 4.0'},
  {tier:'T2',name:'IndicCorpV2',scale:'300B tokens',why:'Largest open Indic corpus; 24 languages; CC-BY 4.0'},
  {tier:'T2',name:'BPCC Parallel',scale:'10B tokens',why:'Indic-English pairs; cross-lingual transfer signal'},
  {tier:'T2',name:'Custom Indic Crawl',scale:'200B tokens (target)',why:'.gov.in, .in news, Sahitya Akademi, Indian Kanoon'},
];

// ── Pipeline steps ───────────────────────────────────────────────
const PIPELINE = [
  {name:'Text Extraction',  tag:'All sources',
   desc:'Trafilatura on raw WARC files (not WET). Brahmic script normalizer (indic-nlp-library): NFC Unicode, standardize Devanagari matras, strip zero-width joiners. Tesseract OCR + IndicOCR for PDF government docs (NCERT, RBI reports).',
   removes:['HTML navbars, cookie banners, JS','Malformed Unicode & ISCII fragments','Invalid OCR artifacts'],
   stats:[{v:'Trafilatura',l:'Tool'},{v:'IndicOCR',l:'Indic PDFs'},{v:'NFC',l:'Normalisation'}]},
  {name:'Language ID',      tag:'GlotLID',
   desc:'GlotLID (500+ languages) routes documents. Indic ≥0.85 conf → Tier 2. English ≥0.80 → Tier 1. Code → D3. Code-mixed (>30% Latin + >30% Devanagari) tagged as Hinglish and sent to Tier 2 always-on channel.',
   removes:['Unidentified / low-confidence docs','Mislabelled English in Indic fields'],
   stats:[{v:'0.85',l:'Indic threshold'},{v:'0.80',l:'EN threshold'},{v:'>30%',l:'Code-mix cutoff'}]},
  {name:'Exact Dedup',      tag:'SHA-256',
   desc:'SHA-256 of normalised text body. Global hash table; dedup within and across crawl snapshots. URL-level: if same URL appears in multiple snapshots, retain most recent clean version.',
   removes:['Verbatim duplicates across snapshots','Same-URL redundant crawls'],
   stats:[{v:'SHA-256',l:'Hash fn'},{v:'URL-level',l:'Also deduped'},{v:'Global',l:'Scope'}]},
  {name:'Fuzzy Dedup',      tag:'MinHash LSH',
   desc:'MinHash LSH (same as FineWeb). Per-snapshot scope — global fuzzy dedup hurts more than it helps (FineWeb ablation). Indic: hash normalised Unicode (NFC), not raw bytes, to catch same text in different encodings.',
   removes:['Near-duplicates (Jaccard ≥0.80)','Different-encoding same-content docs'],
   stats:[{v:'13-gram',l:'N-gram size'},{v:'128',l:'Hash fns'},{v:'0.80',l:'Jaccard threshold'}]},
  {name:'Heuristic Filters',tag:'Rule-based',
   desc:'Universal: min 50 words, max 100K words, bullet fraction ≤90%, repeated n-gram ≤20%, symbol fraction ≤10%, stop-word ≥20%, terminal punctuation ≥70%. Indic-specific: script purity ≥60%, no invalid Brahmic conjuncts (catches OCR errors), reject >50% ASCII in declared Devanagari doc.',
   removes:['Navigation dumps, TOC-only pages','Boilerplate / template-generated','Random token dumps'],
   stats:[{v:'≥50',l:'Min words'},{v:'≤20%',l:'Repeat n-gram'},{v:'≥70%',l:'Terminal punct'}]},
  {name:'Quality Classifier',tag:'DeBERTa-v3 / MuRIL',
   desc:'English: FineWeb-Edu style. Label 200K docs with Gemma 4, train DeBERTa-v3-base, retain score ≥3.0 for D1, ≥4.0 for anneal. Indic: separate MuRIL-based classifier trained on Sangraha Verified (positive) vs noisy web (negative). CANNOT use English classifier on Indic — this is the V4 bug.',
   removes:['Low educational value English','Noisy / low-quality Indic web content'],
   stats:[{v:'DeBERTa-v3',l:'EN model'},{v:'MuRIL',l:'Indic model'},{v:'≥3.0',l:'Quality bar'}]},
  {name:'India-Perspective Classifier',tag:'Tier 1 only',
   desc:'DeBERTa-v3-large fine-tuned on 100K labelled samples. Score 0–10: 0–2 discard (colonial / India-negative framing), 3–5 retain, 6–10 upsample 2×. Quarterly retraining. Metadata score stored per doc for threshold adjustment.',
   removes:['Colonial-era degrading texts','Western-biased geopolitical framing','Systematic anti-India content without basis'],
   stats:[{v:'0–2',l:'Discard'},{v:'3–5',l:'Retain'},{v:'6–10',l:'Upsample 2×'}]},
  {name:'PII / Safety Filter',tag:'Llama Guard 3',
   desc:'Redact AADHAAR (12-digit), PAN (ABCDE1234F), Indian mobile numbers in personal context. Llama Guard 3 fine-tuned on Indian legal definitions (IT Act 2000, POCSO). Religious / caste-based hate speech → full document discard.',
   removes:['AADHAAR / PAN numbers','Indian mobile numbers (personal)','Hate speech (religious, caste-based)','CSAM (hash-matched)'],
   stats:[{v:'AADHAAR',l:'Redacted'},{v:'LlamaGuard 3',l:'Safety model'},{v:'IT Act',l:'Legal basis'}]},
  {name:'Contamination Check',tag:'Golden Proxy',
   desc:'BM25 index all Tier 3 test items on Day 1. For each training doc, check top-3 BM25 matches. Exact 8-gram match OR Jaccard ≥0.7 → remove. Monthly manual audit of 500 random Tier 3 items: "Have you seen this before?" memorisation probe.',
   removes:['Verbatim benchmark questions','Paraphrased benchmark content','Source URLs from benchmark hosts'],
   stats:[{v:'8-gram',l:'Exact match'},{v:'0.7',l:'Jaccard threshold'},{v:'Monthly',l:'Audit cadence'}]},
];

// Retention funnel
const RETENTION = {
  stages:['Raw','After Lang ID','After Exact Dedup','After Fuzzy Dedup','After Heuristics','After Quality','After India Classifier','After PII/Safety','Final'],
  english: [100,85,70,55,45,35,30,29,28],
  indic:   [100,90,65,50,42,38, 38,37,37],
  code:    [100,95,80,70,65,60, 60,59,59],
};

// SFT data
const SFT_TASKS = [
  {title:'Coding',icon:'💻',vol:2000000,total:7500000,note:'Python, JS, India-context problems (GST, UPI, IRCTC logic)'},
  {title:'Agentic',icon:'🤖',vol:1000000,total:7500000,note:'Tool-use, multi-step, India API schemas (DigiLocker, NSE, IRCTC)'},
  {title:'Indic SFT',icon:'🇮🇳',vol:1500000,total:7500000,note:'12 languages; IndicTrans2 pipeline + native generation'},
  {title:'Math & Science',icon:'🔢',vol:1000000,total:7500000,note:'JEE/NEET problems; OpenMathInstruct-2 + synthetic verification'},
  {title:'General Chat',icon:'💬',vol:1000000,total:7500000,note:'OASST2, UltraChat-200k, WizardLM Evol-Instruct'},
  {title:'India-specific',icon:'🏛️',vol:500000,total:7500000,note:'Constitution, history, ISRO, Ayurveda, state governance Q&A'},
];

const RL_TRACKS = [
  {title:'Math (Verifiable)',icon:'🔢',
   sources:'MATH 12.5K · DeepMath-103K · AIME 1983–2024 · IIT-JEE 3K · NCERT Exemplar 10K',
   reward:'+1.0 correct | 0.0 wrong | +0.3 right method\n-0.1 missing <think> block'},
  {title:'Code (Execution)',icon:'💻',
   sources:'HumanEval · MBPP · LeetCode 3K · CodeContests 13K · SWE-Bench train split',
   reward:'+1.0 all tests pass | k/n partial pass\n-0.1 syntax error | -0.05 timeout'},
  {title:'Agentic (Partial)',icon:'🤖',
   sources:'τ²-bench train · AgentTraj 50K · Tool call format synthetic dataset',
   reward:'+1.0 goal complete | +0.5 partial | +0.1 valid format\n-0.2 invalid JSON'},
  {title:'Indic Quality (Hybrid)',icon:'🇮🇳',
   sources:'MuRIL reward model · IndicNLP grammar tools · IndicTrans2 round-trip BLEU',
   reward:'0.3×grammar + 0.3×roundtrip_BLEU + 0.4×reward_model'},
];

const EVAL_CARDS = [
  {title:'General',items:[
    {bench:'MMLU',target:'≥85%'},{bench:'MMLU-Pro',target:'≥75%'},
    {bench:'GPQA Diamond',target:'≥70%'},{bench:'IFEval',target:'≥85%'},
    {bench:'ARC-Challenge',target:'≥90%'},
  ]},
  {title:'Coding',items:[
    {bench:'HumanEval+',target:'≥88%'},{bench:'LiveCodeBench v6',target:'≥80%'},
    {bench:'SWE-Bench Verified',target:'≥40%'},{bench:'DS-1000',target:'≥75%'},
    {bench:'IndicEval-XL',target:'≥60%'},
  ]},
  {title:'Math & Science',items:[
    {bench:'GSM8K',target:'≥97%'},{bench:'MATH-500',target:'≥85%'},
    {bench:'AIME 2024+2025',target:'≥70%'},{bench:'IIT-JEE Bench ★',target:'≥72%'},
    {bench:'NEET-Bench ★',target:'≥76%'},
  ]},
  {title:'Indic',items:[
    {bench:'MILU',target:'≥75%'},{bench:'BharatBench',target:'≥70%'},
    {bench:'FLORES-200 (chrF)',target:'≥75'},{bench:'IndicXTREME',target:'≥80%'},
    {bench:'IndicQA',target:'≥75%'},
  ]},
];
