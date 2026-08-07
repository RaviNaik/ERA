"""
Corpus Generator — produces 30 synthetic documents across 4 capability lanes.
Each document carries full provenance metadata as required by Session 3/6 contracts.
"""

from __future__ import annotations
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Document:
    doc_id: str
    lane: str          # web | code | indic | agentic
    source_url: str
    license_tier: str  # safe | review | unsafe
    text: str
    is_eval: bool = False          # held-out eval data — NEVER trains
    capability_tags: List[str] = field(default_factory=list)
    language: str = "en"
    script: str = "latin"

    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


# ── Synthetic Documents ──────────────────────────────────────────────────────

_WEB_DOCS = [
    ("Climate policy changed after the monsoon report because farmers needed "
     "early warning systems. The government announced a new framework for "
     "agricultural resilience including drought-resistant crop subsidies and "
     "rural weather station networks. Scientists at IIT Bombay confirmed that "
     "rainfall patterns have shifted significantly over the past decade."),

    ("The history of Indian classical music spans over two thousand years. "
     "Hindustani and Carnatic traditions diverged around the twelfth century "
     "when Persian influences reached northern courts. Ragas encode not just "
     "melodic material but the time of day and season in which they should be "
     "performed. Modern fusion explores these boundaries while preserving the "
     "fundamental structure of raga grammar."),

    ("Plate tectonics explains how Earth's outer shell breaks into moving "
     "plates that carry the continents across the globe. The Deccan Traps "
     "formed when India drifted over a mantle hotspot roughly 65 million years "
     "ago. This volcanic episode may have contributed to the mass extinction "
     "event. Modern GPS measurements track plate velocities to millimetre "
     "precision, confirming drift rates of several centimetres per year."),

    ("Open source software has transformed how technology spreads globally. "
     "The Linux kernel, released in 1991, now runs most internet servers, "
     "Android phones and supercomputers. Permissive licences like MIT and "
     "Apache allow commercial use while copyleft licences like GPL require "
     "derivative works to remain open. Choosing the right licence is a "
     "fundamental engineering and legal decision for any project."),

    ("Economic research shows that infrastructure investment in rural areas "
     "creates sustainable employment and reduces migration to cities. A study "
     "of road construction programs in Bihar and Odisha found that villages "
     "within five kilometres of new roads showed 23 percent higher household "
     "income after five years. Digital infrastructure including mobile networks "
     "and broadband amplifies these effects."),

    ("Quantum computing differs fundamentally from classical computing. A qubit "
     "can exist in a superposition of zero and one until measured, unlike a "
     "classical bit which is always one or the other. Grover's algorithm "
     "provides a quadratic speedup for unstructured search. Shor's algorithm "
     "solves integer factorisation in polynomial time, threatening current "
     "public-key cryptography systems."),

    ("Biodiversity loss is accelerating at rates not seen since the last mass "
     "extinction. The IPBES report estimates one million species face extinction "
     "within decades. Protected area networks cover only 17 percent of land "
     "surface. Corridor conservation connects fragmented habitats allowing "
     "species to migrate as climate zones shift. Indigenous land stewardship "
     "has often proved more effective than centrally managed reserves."),

    ("The Mumbai-Ahmedabad High Speed Rail corridor will reduce travel time "
     "from over six hours to under three. Engineers face the challenge of "
     "building a largely elevated track across the Thane creek and through "
     "coastal zones. Japanese Shinkansen technology adapted for Indian "
     "conditions requires extensive localisation of maintenance practices, "
     "worker training and spare part supply chains."),
]

_CODE_DOCS = [
    ("def train(model, loader, optimizer, loss_fn, device):\n"
     "    model.train()\n"
     "    total_loss = 0.0\n"
     "    for batch_idx, (input_ids, targets, loss_mask) in enumerate(loader):\n"
     "        input_ids = input_ids.to(device)\n"
     "        targets = targets.to(device)\n"
     "        loss_mask = loss_mask.to(device)\n"
     "        optimizer.zero_grad()\n"
     "        logits = model(input_ids)\n"
     "        loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))\n"
     "        loss = (loss * loss_mask.view(-1)).sum() / loss_mask.sum()\n"
     "        loss.backward()\n"
     "        optimizer.step()\n"
     "        total_loss += loss.item()\n"
     "    return total_loss / len(loader)\n"),

    ("import hashlib\nfrom pathlib import Path\n\n"
     "def compute_content_hash(file_path: str) -> str:\n"
     "    h = hashlib.sha256()\n"
     "    with open(file_path, 'rb') as f:\n"
     "        for chunk in iter(lambda: f.read(65536), b''):\n"
     "            h.update(chunk)\n"
     "    return h.hexdigest()\n\n"
     "def verify_shard_integrity(shard_dir: Path) -> bool:\n"
     "    manifest = json.loads((shard_dir / 'manifest.json').read_text())\n"
     "    actual = compute_content_hash(shard_dir / manifest['shard_id'])\n"
     "    return actual == manifest['content_hash']\n"),

    ("class BestFitPacker:\n"
     "    def __init__(self, context_len: int):\n"
     "        self.context_len = context_len\n"
     "        self.bins = []\n\n"
     "    def pack(self, docs):\n"
     "        sorted_docs = sorted(docs, key=lambda d: len(d), reverse=True)\n"
     "        for doc in sorted_docs:\n"
     "            placed = False\n"
     "            best_bin, best_remaining = None, self.context_len + 1\n"
     "            for b in self.bins:\n"
     "                remaining = self.context_len - sum(len(d) for d in b)\n"
     "                if remaining >= len(doc) and remaining < best_remaining:\n"
     "                    best_bin, best_remaining = b, remaining\n"
     "            if best_bin is not None:\n"
     "                best_bin.append(doc)\n"
     "            else:\n"
     "                self.bins.append([doc])\n"
     "        return self.bins\n"),

    ("def greedy_pack(token_seqs, context_len):\n"
     "    bins, current_bin, current_len = [], [], 0\n"
     "    for seq in token_seqs:\n"
     "        if current_len + len(seq) <= context_len:\n"
     "            current_bin.append(seq)\n"
     "            current_len += len(seq)\n"
     "        else:\n"
     "            if current_bin:\n"
     "                bins.append(current_bin)\n"
     "            current_bin = [seq]\n"
     "            current_len = len(seq)\n"
     "    if current_bin:\n"
     "        bins.append(current_bin)\n"
     "    return bins\n"),

    ("import numpy as np\n\n"
     "def build_causal_mask(seq_len: int, doc_spans):\n"
     "    mask = np.zeros((seq_len, seq_len), dtype=bool)\n"
     "    for start, end in doc_spans:\n"
     "        for i in range(start, end):\n"
     "            mask[i, start:i+1] = True\n"
     "    return mask\n\n"
     "def build_loss_mask(seq_len: int, loss_spans, pad_positions=None):\n"
     "    lm = np.ones(seq_len, dtype=np.float32)\n"
     "    if pad_positions:\n"
     "        for pos in pad_positions:\n"
     "            lm[pos] = 0.0\n"
     "    for start, end in (loss_spans or []):\n"
     "        lm[start:end] = 0.0\n"
     "    return lm\n"),

    ("class MinHashLSH:\n"
     "    def __init__(self, num_perm=128, threshold=0.8):\n"
     "        self.num_perm = num_perm\n"
     "        self.threshold = threshold\n"
     "        import hashlib, random\n"
     "        r = random.Random(42)\n"
     "        self.hash_fns = [(r.randint(1, 2**31), r.randint(0, 2**31))\n"
     "                          for _ in range(num_perm)]\n\n"
     "    def minhash(self, shingles):\n"
     "        sig = []\n"
     "        for a, b in self.hash_fns:\n"
     "            mn = min((a * s + b) % (2**31) for s in shingles)\n"
     "            sig.append(mn)\n"
     "        return sig\n\n"
     "    def jaccard(self, sig1, sig2):\n"
     "        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)\n"
     "        return matches / len(sig1)\n"),
]

_INDIC_DOCS = [
    ("भारत में भाषा और तकनीक का मिलन तेजी से बढ़ रहा है। कृत्रिम बुद्धिमत्ता "
     "के क्षेत्र में हिंदी और अन्य भारतीय भाषाओं के लिए नए मॉडल विकसित हो रहे हैं। "
     "संगठन जैसे AI4Bharat ने संग्रह जैसे डेटासेट बनाए हैं जो भारतीय भाषाओं के "
     "लिए प्रशिक्षण डेटा प्रदान करते हैं। यह काम भारतीय उपयोगकर्ताओं के लिए "
     "सुलभ और प्रभावी AI समाधान बनाने में मदद करता है।"),

    ("తెలుగు వార్తలు: ఆంధ్రప్రదేశ్ మరియు తెలంగాణ రాష్ట్రాలలో వ్యవసాయ రంగం "
     "అభివృద్ధి చెందుతోంది. కొత్త సాంకేతికతలు రైతులకు సహాయపడుతున్నాయి. "
     "డ్రోన్ సర్వే మరియు మట్టి పరీక్ష కార్యక్రమాలు పంట దిగుబడిని పెంచడంలో "
     "కీలక పాత్ర పోషిస్తున్నాయి. ప్రభుత్వం కొత్త నీటి పారుదల పద్ధతులను "
     "ప్రోత్సహిస్తోంది."),

    ("ਪੰਜਾਬ ਵਿੱਚ ਖੇਤੀਬਾੜੀ ਦੀ ਤਬਦੀਲੀ: ਨਵੀਂ ਤਕਨੀਕਾਂ ਅਤੇ ਡਿਜੀਟਲ ਭੁਗਤਾਨ "
     "ਪ੍ਰਣਾਲੀਆਂ ਕਿਸਾਨਾਂ ਦੀ ਮਦਦ ਕਰ ਰਹੀਆਂ ਹਨ। ਸਰਕਾਰੀ ਅਤੇ ਪ੍ਰਾਈਵੇਟ ਭਾਈਵਾਲੀ "
     "ਨੇ ਸਿੱਧੇ ਲਾਭ ਟ੍ਰਾਂਸਫਰ ਪ੍ਰੋਗਰਾਮਾਂ ਨੂੰ ਲਾਗੂ ਕੀਤਾ ਹੈ। ਮੋਬਾਈਲ ਐਪਲੀਕੇਸ਼ਨਾਂ "
     "ਕਿਸਾਨਾਂ ਨੂੰ ਮੌਸਮ ਦੀ ਜਾਣਕਾਰੀ ਅਤੇ ਬਾਜ਼ਾਰ ਭਾਅ ਉਪਲਬਧ ਕਰਾਉਂਦੀਆਂ ਹਨ।"),

    ("কৃত্রিম বুদ্ধিমত্তা বাংলা ভাষার জন্য নতুন সুযোগ তৈরি করছে। "
     "ভারত এবং বাংলাদেশের গবেষকরা মিলে বাংলা এনএলপি মডেল তৈরি করছেন। "
     "প্রাকৃতিক ভাষা প্রক্রিয়াকরণ প্রযুক্তি বাংলা টেক্সট বিশ্লেষণ, "
     "অনুবাদ এবং প্রশ্নোত্তর সিস্টেমে ব্যবহার হচ্ছে।"),

    ("ಕನ್ನಡ ಭಾಷೆಯ ಡಿಜಿಟಲ್ ಯುಗ: ಕನ್ನಡದಲ್ಲಿ ಕೃತಕ ಬುದ್ಧಿಮತ್ತೆ ಮತ್ತು ಯಂತ್ರ ಕಲಿಕೆ "
     "ಕ್ಷೇತ್ರಗಳಲ್ಲಿ ಸಂಶೋಧನೆ ಹೆಚ್ಚುತ್ತಿದೆ. ಬೆಂಗಳೂರಿನ ತಂತ್ರಜ್ಞಾನ ಕಂಪನಿಗಳು "
     "ಕನ್ನಡ ಧ್ವನಿ ಗುರುತಿಸುವ ವ್ಯವಸ್ಥೆಗಳನ್ನು ಅಭಿವೃದ್ಧಿಪಡಿಸುತ್ತಿವೆ. "
     "ಮೊಬೈಲ್ ಅಪ್ಲಿಕೇಶನ್‌ಗಳು ಕನ್ನಡ ಬಳಕೆದಾರರಿಗೆ ಹೆಚ್ಚು ಸುಲಭವಾಗುತ್ತಿವೆ."),

    ("मराठी भाषेतील तंत्रज्ञानाची प्रगती: महाराष्ट्रात डिजिटल क्रांती "
     "वेगाने होत आहे. मराठी भाषेसाठी ऑटोमॅटिक स्पीच रेकग्निशन आणि "
     "मशीन ट्रान्सलेशन सिस्टम विकसित केल्या जात आहेत. शेती, शिक्षण आणि "
     "आरोग्य या क्षेत्रात AI चा वापर वाढत आहे. मराठी डेटासेट्स तयार करणे "
     "हे या कामाचा महत्त्वाचा भाग आहे।"),
]

_AGENTIC_DOCS = [
    # Agentic docs have a special structure: user/tool_call/observation/assistant
    # The MASKED tokens are tool observations — no loss applied to them
    ("[USER] Find all NSF grants about cryo-EM motion correction and identify "
     "labs that could buy new microscopes. [/USER]\n"
     "[ASSISTANT] I will search for relevant NSF grants first. [/ASSISTANT]\n"
     "[TOOL_CALL] search_grants(query='cryo-EM motion correction', agency='NSF') [/TOOL_CALL]\n"
     "[OBSERVATION] Found 12 grants totalling $18.5M awarded 2022-2026. [/OBSERVATION]\n"
     "[ASSISTANT] I found 12 grants. Let me look up the PI information for each. [/ASSISTANT]\n"
     "[TOOL_CALL] lookup_awardee(grant_id='NSF-2041338') [/TOOL_CALL]\n"
     "[OBSERVATION] PI: Dr. Sarah Chen, Ridgeline Structural Biology Lab, Award: $1.85M [/OBSERVATION]\n"
     "[ASSISTANT] The lab received $1.85M — sufficient for a new microscope system. "
     "I will now check their equipment inventory. [/ASSISTANT]\n"
     "[TOOL_CALL] fetch_org_profile(org='Ridgeline Structural Biology Lab') [/TOOL_CALL]\n"
     "[OBSERVATION] ERROR 429: rate_limited [/OBSERVATION]\n"
     "[ASSISTANT] Rate limited — switching to open web search for this lab. [/ASSISTANT]\n"
     "[TOOL_CALL] web_search(query='Ridgeline Structural Biology Lab cryo-EM equipment') [/TOOL_CALL]\n"
     "[OBSERVATION] Lab recently acquired $1.85M NSF award; current facility uses 2019 equipment. [/OBSERVATION]\n"
     "[ASSISTANT] Based on my research: Ridgeline Lab (PI: Dr. Sarah Chen) has the NSF award "
     "and aging equipment — they are a strong candidate for a new cryo-EM microscope purchase. [/ASSISTANT]"),

    ("[USER] Cancel my flight PNR X7Q2 and rebook to Tuesday under fare rules. [/USER]\n"
     "[ASSISTANT] Let me check the fare rules on your ticket first. [/ASSISTANT]\n"
     "[TOOL_CALL] get_fare_rules(pnr='X7Q2') [/TOOL_CALL]\n"
     "[OBSERVATION] {\"changeable\": true, \"fee\": 0, \"advance_days\": 1} [/OBSERVATION]\n"
     "[ASSISTANT] Your ticket is freely changeable with no fee and at least 1 day advance. "
     "Tuesday is 3 days away — within policy. Proceeding to rebook. [/ASSISTANT]\n"
     "[TOOL_CALL] rebook(pnr='X7Q2', date='2026-07-29', flight='AI-202') [/TOOL_CALL]\n"
     "[OBSERVATION] Booking confirmed. New PNR: X7Q9. Departure: 2026-07-29 08:30 [/OBSERVATION]\n"
     "[ASSISTANT] Your flight has been rebooked to Tuesday July 29 at 08:30. "
     "New reference: X7Q9. No fee was charged per your fare rules. [/ASSISTANT]"),

    ("[USER] Reply to the latest forum post by user pradeepta_k saying: Thanks, fixed. [/USER]\n"
     "[ASSISTANT] I will find the latest post by pradeepta_k and reply to it. [/ASSISTANT]\n"
     "[TOOL_CALL] search_forum_posts(user='pradeepta_k', sort='newest', limit=1) [/TOOL_CALL]\n"
     "[OBSERVATION] Post #4127: 'Fixed the authentication bug' by pradeepta_k, 2 hours ago [/OBSERVATION]\n"
     "[ASSISTANT] Found the post. Navigating to reply. [/ASSISTANT]\n"
     "[TOOL_CALL] open_post(post_id=4127) [/TOOL_CALL]\n"
     "[OBSERVATION] Page loaded: post #4127 with reply form available [/OBSERVATION]\n"
     "[ASSISTANT] Page is open. Submitting the reply now. [/ASSISTANT]\n"
     "[TOOL_CALL] post_reply(post_id=4127, text='Thanks, fixed.') [/TOOL_CALL]\n"
     "[OBSERVATION] Reply posted successfully. Reply ID: 8832 [/OBSERVATION]\n"
     "[ASSISTANT] Done. I have replied 'Thanks, fixed.' to pradeepta_k's post #4127. [/ASSISTANT]"),

    ("[USER] What is the revenue of company Apex Dynamics in its top region for fiscal 2023? "
     "Use the attached CSV. [/USER]\n"
     "[ASSISTANT] I will read the CSV to find Apex Dynamics' revenue by region. [/ASSISTANT]\n"
     "[TOOL_CALL] read_csv(file='company_revenue_2023.csv', filter='Apex Dynamics') [/TOOL_CALL]\n"
     "[OBSERVATION] Rows: APAC: $342M, EMEA: $518M, Americas: $289M [/OBSERVATION]\n"
     "[ASSISTANT] The top region for Apex Dynamics is EMEA with revenue of $518 million "
     "in fiscal 2023. [/ASSISTANT]"),
]

# Three eval (held-out) documents — these must NEVER enter training
_EVAL_DOCS = [
    ("EVAL: What is the capital of India? A) Mumbai B) New Delhi C) Kolkata D) Chennai "
     "Answer: B) New Delhi. India's capital was moved from Calcutta to New Delhi in 1911 "
     "during the British Raj."),

    ("EVAL: Solve: if 3x + 6 = 21, find x. Solution: subtract 6 from both sides: 3x = 15. "
     "Divide by 3: x = 5. Answer: x = 5."),

    ("EVAL: Count integers between 1 and 1000 divisible by neither 3 nor 7. "
     "By inclusion-exclusion: 1000 - floor(1000/3) - floor(1000/7) + floor(1000/21) "
     "= 1000 - 333 - 142 + 47 = 572. Answer: 572."),
]


def build_corpus() -> List[Document]:
    """Build the full 30-document toy corpus with provenance metadata."""
    docs = []
    idx = 0

    for text in _WEB_DOCS:
        docs.append(Document(
            doc_id=f"doc_{idx:04d}",
            lane="web",
            source_url=f"https://example-web.corpus/{idx:04d}",
            license_tier="safe",
            text=text,
            capability_tags=["general_web", "knowledge"],
            language="en",
            script="latin",
        ))
        idx += 1

    for text in _CODE_DOCS:
        docs.append(Document(
            doc_id=f"doc_{idx:04d}",
            lane="code",
            source_url=f"https://example-code.corpus/{idx:04d}",
            license_tier="safe",
            text=text,
            capability_tags=["code", "python"],
            language="en",
            script="latin",
        ))
        idx += 1

    # Indic docs with correct language/script metadata
    indic_meta = [
        ("hi", "devanagari"), ("te", "telugu"), ("pa", "gurmukhi"),
        ("bn", "bengali"), ("kn", "kannada"), ("mr", "devanagari"),
    ]
    for text, (lang, script) in zip(_INDIC_DOCS, indic_meta):
        docs.append(Document(
            doc_id=f"doc_{idx:04d}",
            lane="indic",
            source_url=f"https://example-indic.corpus/{idx:04d}",
            license_tier="safe",
            text=text,
            capability_tags=["indic", "multilingual", lang],
            language=lang,
            script=script,
        ))
        idx += 1

    for text in _AGENTIC_DOCS:
        docs.append(Document(
            doc_id=f"doc_{idx:04d}",
            lane="agentic",
            source_url=f"https://example-agentic.corpus/{idx:04d}",
            license_tier="safe",
            text=text,
            capability_tags=["agentic", "tool_use", "multi_turn"],
            language="en",
            script="latin",
        ))
        idx += 1

    # Add eval docs (HELD OUT — never train)
    for text in _EVAL_DOCS:
        docs.append(Document(
            doc_id=f"eval_{idx:04d}",
            lane="web",
            source_url=f"https://eval.held-out/{idx:04d}",
            license_tier="safe",
            text=text,
            is_eval=True,
            capability_tags=["eval", "held_out"],
            language="en",
            script="latin",
        ))
        idx += 1

    return docs


def save_corpus(docs: List[Document], output_dir: Path) -> Path:
    """Save corpus to JSONL for reference."""
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "corpus.jsonl"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for doc in docs:
            record = asdict(doc)
            record["content_hash"] = doc.content_hash()
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return corpus_path
