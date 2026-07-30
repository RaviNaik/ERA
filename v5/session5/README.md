# V5 Mixture-and-Curriculum Plan
## A Defended Specification for the 40B India-First Language Model

> **Author:** Ravi Naik  
> **Session:** ERA V5 — Session 5 (Data Mixtures and Curriculum)  
> **Interactive Dashboard:** [Launch Webapp →](./webapp/index.html)

---

## 0. Preamble: How This Spec Is Structured

A mixture plan that cannot be falsified is not a plan — it is a wish list. Every number in this document is therefore written as a **testable hypothesis** tied to a concrete benchmark and a real supply figure from the dataset inventory. Where a target cannot be met with real data, this document says so plainly and states what must be synthesized instead.

The plan proceeds in this order:

1. Start from the **benchmarks** the model must pass.
2. Map benchmarks to **data formats** that produce them.
3. Size each lane's **demand** against the **real supply** from the inventory.
4. Set a **protected floor** the OPUS selector may never breach.
5. Hold back an **anneal reserve** of the best data for the final cooldown.
6. Lay out the **difficulty and reasoning-length bands** with examples.
7. Commit to **proxy runs at 1B and 3B scale** before trusting any number at full scale.

---

## 1. Target Benchmarks — Composing Backward

The model has three primary capability mandates: **agentic/coding excellence**, **controllable reasoning**, and **native Indic fluency**. Working backward from those mandates produces this benchmark set:

| Capability | Primary Benchmarks | Secondary |
|---|---|---|
| Coding | LiveCodeBench, Aider Polyglot | Codeforces ELO |
| Agentic / Tool-use | SWE-bench Verified, tau-bench, BFCL v3, GAIA | Terminal-Bench, BrowseComp |
| Reasoning + STEM | AIME 2024/25, GPQA Diamond, HLE | FrontierMath |
| Long-context | HELMET long-eval | SCROLLS |
| Indic understanding | MILU (AI4Bharat) | IndicGenBench |
| General knowledge | MMLU | HellaSwag |

Each benchmark maps to a **training data format** (loss map):
- **SWE-bench** → code-editing trajectories where loss is on the generated patch only
- **BFCL** → structured function-call JSON where loss is on the correct argument values
- **AIME** → chain-of-thought + integer final answer; problem statement is masked
- **MILU** → native-language MCQ; reasoning and chosen option in the loss

These formats are the shopping list. The mixture sections below are the budget that fills it.

---

## 2. The Pretrain Mixture — Main Run (2T token budget)

The total pretrain budget is **2 trillion tokens**. The anneal budget (§5) is held separately and not counted here.

### 2.1 Lane Allocations

| Lane | Share | Demand at 2T | Real Supply | Status |
|---|---|---|---|---|
| General Web | **32%** | 640B | 4.5T (DCLM + FineWeb-Edu + V4 D1/D2) | ✅ Covered |
| Code | **24%** | 480B | 1.1T (Stack v2 + D3 + CommitPack) | ✅ Covered |
| Indic | **18%** | 360B | 276B (all tiers) | ⚠️ Needs repetition + synthesis |
| STEM / Math | **12%** | 240B | 250B (proof-pile-2 + D4 + peS2o + Reasoning) | ✅ Barely covered |
| Reasoning traces | **6%** | 120B | 85B (AON + OpenThoughts2 + NuminaMath) | ⚠️ 1.4× repetition |
| Long-context | **5%** | 100B | 100B (repo-packed 60B + book corpora 40B) | ✅ Exactly covered |
| Agentic / Tool-use | **3%** | 60B | 0.63B (all 9 agentic datasets combined) | 🔴 Must synthesize (95× gap) |
| **Total** | **100%** | **2T** | | |

### 2.2 Rationale for Each Lane

**General Web (32%)** is the largest lane because it is the most abundant data. At 32% of a 2T run, demand is 640B against a 4.5T supply — clean room to grow. It buys MMLU, HellaSwag, and general world knowledge. Dropping it below 30% risks degrading breadth of factual knowledge. The 32% figure (down from a naive 60%+ web-heavy baseline) is the cost of buying the other capabilities.

**Code (24%)** is the second largest because coding is the primary commercial differentiation. 480B tokens against a 1.1T supply means no repetition in the main run. Code also improves non-code reasoning benchmarks by teaching structure, decomposition, and long dependency chains (the cross-domain benefit documented in DeepSeek-Coder ablations). Going above 28% would squeeze General Web too hard and degrade MMLU.

**Indic (18%)** sits 6 percentage points above the protected floor (§4). The floor is 12%; 18% is a deliberate purchase of headroom for MILU and IndicGenBench. At 360B demand vs 276B real supply, the lane needs repetition and synthesis — detailed in §3.

**STEM / Math (12%)** feeds AIME, GPQA, and HLE. At 240B demand versus 250B real supply, this lane is exactly at parity with no repetition needed in the main run. Raising it further (e.g., to 15%) would require either synthesis or repetition immediately and yields diminishing returns at pretrain — AIME gains primarily come from the later reasoning and RLVR stages.

**Reasoning traces (6%)** deposits the structural pattern of careful multi-step reasoning into the base weights before the dedicated reasoning training stages (Sessions 17–18). 120B demand vs 85B real supply requires 1.4× repetition — acceptable, since repetition below 2× carries very low marginal cost. All five datasets (AON, OpenThoughts2, OpenMathReasoning, OpenR1-Math, NuminaMath) are used. AON (78B) provides the bulk.

**Long-context (5%)** is sized at the exact real supply (100B tokens: 60B repo-packed code at 32K+ context, 40B book-length corpora). Going beyond 5% would require synthesis and is not warranted in pretraining; the long-eval benchmark gains primarily from the model's ability to track dependencies, not from sheer token volume.

**Agentic (3%)** is the most constrained lane. Real supply across all 9 datasets (SWE-Gym, SWE-smith, OpenHands, ToolBench, ToolACE, Glaive, xLAM, NexusRaven, Hermes) totals only 627M tokens. At 3% of 2T, demand is 60B — a **95× gap**. This lane cannot be filled; it must be built. The 3% allocation is set at 1 point above the protected floor (§4) to signal intent while acknowledging that pretraining cannot deliver agentic capability alone — the anneal and SFT stages are where agentic ability concentrates (§5). The synthesis strategy is described in §6.

### 2.3 What This Mixture Buys vs. What It Starves

**Funded in pretrain:** LiveCodeBench, Aider, MMLU, MILU, IndicGenBench, long-eval (partial), AIME (structural foundation)

**Intentionally starved in pretrain:** SWE-bench Verified, tau-bench, BFCL (concentrated in anneal and SFT by design — these benchmarks require trajectory-quality data that is reserved for the anneal)

---

## 3. The Indic Slot — Tier-by-Tier Breakdown

The 18% Indic slot demands **360B tokens** at a 2T run. Real supply is **276B tokens** across 6 datasets. The 84B gap must be closed by repetition and synthesis. This section does not hide behind a single headline number.

### 3.1 Tier Definitions

| Tier | Description | Quality | Use Priority |
|---|---|---|---|
| **A — Verified native** | Human-written, language-verified, manually curated | Highest | Pretrain + Anneal |
| **B — Unverified crawl** | Web-crawled, language-tagged but not human-verified | Medium | Pretrain only |
| **C — Translated** | Parallel text, human or machine translated | Medium-low | Pretrain only |
| **D — Synthetic** | Model-generated from verified seeds | Variable | Pretrain only |

### 3.2 Split Across the 360B Demand

| Tier | Share of Indic | Demand (360B base) | Real Supply | Repetition / Synthesis Required |
|---|---|---|---|---|
| **A: Verified native** | **38%** | **137B** | Sangraha verified: 64B | **2.1× repeat** of verified shards |
| **B: Unverified crawl** | **22%** | **79B** | Sangraha unverified 24B + IndicCorpV2 20.9B = **45B** | **1.8× repeat** |
| **C: Translated** | **20%** | **72B** | BPCC 3B + Samanantar 2B = **5B** | **14.4× gap → synthesis** |
| **D: Synthetic** | **20%** | **72B** | None (purpose-built) | **100% synthesis** |
| **Total** | 100% | 360B | 114B real | 246B synthesized or repeated |

### 3.3 Honest Accounting Notes

**Tier A** at 2.1× repetition is defensible. The V4 run showed that high-quality data can be repeated up to ~4× before marginal value degrades significantly. 2.1× is well inside that window.

**Tier C is the problem.**  BPCC and Samanantar together provide only ~5B tokens of translated parallel text. Reaching 72B requires either 14× repetition (unacceptable — this would cause memorization artifacts in parallel sentence patterns) or **synthetic parallel generation**. The proxy experiment (§7) will test whether 14× Tier C repetition or Tier C synthesis produces better IndicGenBench chrF scores.

**Tier D synthesis pipeline:** Use Sangraha verified documents as seeds. Prompt a teacher model (e.g., Gemini Flash) to generate: (1) paraphrases of verified documents in the same language, (2) domain-diverse continuations, (3) style-varied rewrites. All synthetic documents are scored by an Indic quality classifier (MuRIL-based, trained on Sangraha verified as positives) and only examples scoring ≥ 3.5/5 are admitted.

### 3.4 Language Distribution Within the Indic Slot

Priority languages (by verified native supply and benchmark coverage in MILU):

| Language | Priority | Target Indic % | Key Datasets |
|---|---|---|---|
| Hindi | 🔴 High | 25% | Sangraha-hi, IndicCorpV2-hi |
| Tamil | 🔴 High | 15% | Sangraha-ta, BPCC-ta |
| Telugu | 🔴 High | 12% | Sangraha-te, IndicCorpV2-te |
| Bengali | 🟡 Medium | 10% | Sangraha-bn, Samanantar-bn |
| Kannada | 🟡 Medium | 8% | Sangraha-kn |
| Malayalam | 🟡 Medium | 8% | Sangraha-ml |
| Marathi | 🟡 Medium | 7% | Sangraha-mr |
| Gujarati | 🟢 Covered | 5% | Sangraha-gu |
| Other 14 Indic languages | 🟢 Covered | 10% | IndicCorpV2, Samanantar |

---

## 4. Protected Floors — What the Selector Cannot Cross

The OPUS dynamic selector (used in V4 production) optimizes for token utility against a proxy direction. When the proxy is English-heavy (as in V4), the selector naturally starves Indic and Agentic lanes. Two floors are set in the OPUS always-on channel:

| Lane | Floor | Enforcement |
|---|---|---|
| **Indic** | **≥ 12% of every batch** | OPUS always-on channel forces best Indic batches regardless of proxy utility score until 12% is reached |
| **Agentic** | **≥ 2% of every batch** | Same mechanism; ensures agentic trajectory data is never completely absent |

**Why 12% for Indic?** This is the minimum at which MILU accuracy in Tier-1 languages (Hindi, Tamil, Telugu) remains competitive at 3B parameter scale based on V4 ablation data. Below 12%, the V4 proxy showed Hindi comprehension degrading faster than English on identical parameter counts.

**Why 2% for Agentic?** Below 2%, agentic trajectory data contributes so few tokens that representation in the embedding space is negligible. 2% ensures at least some function-call and tool-use pattern is present in the base weights before post-training.

**The selector is also given a balanced proxy direction** (not English-heavy), so the always-on floor is a backstop, not the primary mechanism. With a balanced proxy, OPUS naturally raises Indic and Agentic selection above the floor.

---

## 5. The Anneal Reserve

The anneal is a short, low-learning-rate cooldown phase at the end of pretraining. It concentrates the best, scarcest data that was intentionally held back from the main run.

### 5.1 Reserve Size

**2% of the 2T total run = 40 billion tokens** are held back for the anneal. This reserve is separate from the main pretrain budget in §2; the main run trains on 1.96T tokens total.

The 2% figure matches the V4 reality (stage 2/6 in the training lifecycle). Too small and the anneal cannot shift the model's final quality distribution meaningfully; too large and the main run loses coverage.

### 5.2 Anneal Mixture

| Lane | Anneal % | Anneal Tokens (40B × %) | Datasets Reserved |
|---|---|---|---|
| Indic | **28%** | 11.2B | Sangraha verified **top-quartile shards** (educational score ≥ 4.0/5.0) |
| Reasoning | **18%** | 7.2B | AON best shards + OpenThoughts2 (full) |
| Code | **20%** | 8.0B | CommitPackFT + Stack v2 top-quality subset |
| Agentic | **8%** | 3.2B | **SWE-Gym (150M)** + **SWE-smith (120M)** + **OpenHands rollouts (90M)** — all reserved for anneal |
| Long-context | **8%** | 3.2B | Book-length corpora top subset |
| STEM / Math | **10%** | 4.0B | proof-pile-2 + peS2o (highest educational scores) |
| General Web | **8%** | 3.2B | FineWeb-Edu educational score ≥ 4.5/5.0 |
| **Total** | **100%** | **40B** | |

### 5.3 What Gets Reserved Starting Now

The following datasets are **not used in the main pretrain run** and are held exclusively for anneal:
- SWE-Gym (150M tokens, 2.4K samples) — the highest-signal agentic dataset
- SWE-smith top 5K samples (out of 26K)
- OpenHands rollouts (90M tokens)
- Sangraha verified shards with educational classifier score ≥ 4.0
- OpenThoughts2 (3B tokens, full dataset)

Spending these in the main run would waste them on a model that cannot yet exploit them. The anneal concentrates them into the final low-LR phase where they land cleanest.

### 5.4 Warmup Band

At the pretrain→anneal boundary, a **3B-token 60/40 warmup band** (60% main-run mix, 40% anneal mix) is inserted to prevent gradient norm explosion. This follows the V4 fix for the Indic Hindi embedding gradient spike (which reached 150× baseline without it).

---

## 6. Agentic Slot — Synthesis Strategy

Real supply: **627M tokens** across 9 datasets. Demand at 3%: **60B tokens**. Synthesis gap: **~59.4B tokens (95×)**.

### 6.1 Synthesis Tiers for Agentic Data

**Tier 1 — One-shot function calls (cheap, high volume):**  
Synthesize using API schemas from public sources (OpenAPI specs, GitHub function definitions). Prompt a teacher model with the schema and a user intent; the teacher generates the matching function call. Score by AST validity and schema conformance. Target: 40B tokens of Tier 1.

**Tier 2 — Multi-turn tool-use conversations:**  
Take Tier 1 calls and build multi-turn dialogues. Insert tool return observations (masked in the loss) and model responses. Target: 15B tokens.

**Tier 3 — Long trajectory generation (expensive, reserved for anneal):**  
Use SWE-Gym and SWE-smith scaffolding to generate new trajectories via rollout. Only the top-quality (verified-passing) trajectories are admitted. These ~4.4B tokens are anneal-only (already reserved in §5.2).

### 6.2 Loss Map Rule (Critical)

Tool observations and environment returns are **always masked** in agentic training data. Only the model's own reasoning, tool parameters, and final answers carry gradient. Applying loss to tool returns would teach the model to hallucinate environment state — the cardinal failure mode in agentic training.

---

## 7. Difficulty and Reasoning-Length Bands

The model must be trained to produce different reasoning depths so the reasoning-effort control (low/medium/high/ultra) can be learned. This requires the pretrain corpus to contain traces **binned by length** across all capability lanes.

### 7.1 Band Definitions

| Band | Level | Reasoning Tokens (approx) | Example Domain |
|---|---|---|---|
| B0 | Nursery | 0–20 tokens | Simple factual recall |
| B1 | Medium | 21–80 tokens | Multi-step arithmetic |
| B2 | Hard (High-school) | 81–200 tokens | Algebraic proof |
| B3 | Undergraduate | 201–500 tokens | Combinatorics with inclusion-exclusion |
| B4 | Graduate | 501–1500 tokens | Research-level math derivation |
| B5 | PhD / Frontier | 1500+ tokens | Expert cross-domain reasoning |

### 7.2 Concrete Examples Per Band

**B0 — Nursery (reasoning: ~15 tokens)**
> Problem: "What is 7 × 8?"  
> Reasoning: "7 multiplied by 8 equals 56."  
> Answer: **56**  
> *Training format: answer token in the loss, problem masked.*

**B1 — Medium (reasoning: ~55 tokens)**
> Problem: "A train travels 240 km in 3 hours. How far in 5 hours at the same speed?"  
> Reasoning: "Speed = 240/3 = 80 km/h. Distance in 5 hours = 80 × 5 = 400 km."  
> Answer: **400 km**

**B2 — Hard (reasoning: ~140 tokens)**
> Problem: "How many integers between 1 and 1000 are divisible by 3 or 5?"  
> Reasoning: "Divisible by 3: ⌊1000/3⌋ = 333. Divisible by 5: ⌊1000/5⌋ = 200. Divisible by 15: ⌊1000/15⌋ = 66. Inclusion-exclusion: 333 + 200 − 66 = 467."  
> Answer: **467**

**B3 — Undergraduate (reasoning: ~350 tokens)**
> Problem: AIME 2024 — "Count n ≤ 1000 divisible by neither 3 nor 7."  
> Reasoning: "Total = 1000. Div by 3: 333. Div by 7: 142. Div by 21: 47. Neither = 1000 − 333 − 142 + 47 = 572. Verify: 1000 × (1−1/3) × (1−1/7) = 1000 × 2/3 × 6/7 ≈ 571.4 ✓"  
> Answer: **572**

**B4 — Graduate (reasoning: ~900 tokens)**
> Problem: GPQA Diamond physics MCQ on quantum entanglement measurement correlation.  
> Reasoning: "Rule out A and D by dimensional analysis. B ignores the Bell inequality coupling term. C correctly applies the cos²θ correlation. Angular separation Δθ = 30°, correlation = cos²(30°) = 0.75. Matches C."  
> Answer: **C**

**B5 — PhD / Frontier (reasoning: 2000+ tokens)**
> Problem: HLE-style expert question requiring cross-domain synthesis (e.g., information-theoretic bound on learning from a specific data distribution).  
> Reasoning: Multi-step derivation involving measure theory, PAC learning bounds, and empirical verification via Rademacher complexity — with multiple self-correction steps and dead-end recovery.  
> Answer: Exact symbolic bound.

### 7.3 Band Distribution in the Mixture

| Stage | B0 | B1 | B2 | B3 | B4 | B5 |
|---|---|---|---|---|---|---|
| Seed / General (0–15%) | 60% | 25% | 10% | 5% | 0% | 0% |
| Reasoning phase (15–60%) | 10% | 20% | 30% | 30% | 10% | 0% |
| Long-context / Anneal (60–100%) | 5% | 10% | 20% | 30% | 25% | 10% |

B5 examples are anneal-only. Introducing PhD-level reasoning too early (before the B2–B3 foundation is solid) wastes exposure — the model cannot yet use the gradient from examples it cannot process.

---

## 8. Curriculum Staging Order

| Stage | Token Range | Key Mixture Properties | Difficulty Band |
|---|---|---|---|
| **Seed** | 0–30B | 50% General Web, 18% Code, 12% STEM, 15% Indic, 5% Reasoning | B0–B1 only |
| **General** | 30B–600B | 34% General Web, 24% Code, 18% Indic, 12% STEM, 6% Reasoning, 5% Long-ctx, 3% Agentic | B0–B2 |
| **Reasoning** | 600B–1.4T | Reasoning rises to 8%, Code stays 24%, General Web falls to 28% | B2–B3 |
| **Long-context** | 1.4T–1.7T | Long-context rises to 8%, Reasoning 6%, Code 22% | B3–B4 |
| **Warmup Band** | 1.7T–1.72T | 60% main-run / 40% anneal blend — gradient stabilizer | B2–B3 |
| **Anneal** | 1.72T–1.96T | See §5 anneal mixture | B3–B5 |

At each seam between stages, a **~3B-token 60/40 warmup band** (60% old mix / 40% new mix) prevents gradient norm explosion. Embeddings are unfrozen at every seam to avoid the V4 Hindi 150× gradient spike.

---

## 9. Proxy Experiment Specification

> **The mixture is a hypothesis. This section defines the experiment that tests it.**

### 9.1 Why Proxies

Running the full 2T experiment to validate a mixture choice costs ~$100K and 60+ days. Running it at 1B parameters on 30B tokens costs ~$200 and takes a few hours. The proxy does not need to reproduce final scores — it needs to **rank competing recipes reliably.** If variant A beats B at 1B scale, the ordering should hold at 40B scale.

### 9.2 Proxy Run Specifications

**Scale:** 1B parameter dense transformer, 30B training tokens (1.5% of the full 2T budget)

**Three ablation variants:**

| Variant | Description | Key Difference |
|---|---|---|
| **A — Proposed Mix** | 32% Web, 24% Code, 18% Indic, 12% STEM, 6% Reasoning, 5% Long-ctx, 3% Agentic | This spec |
| **B — Web-Heavy Baseline** | 60% Web, 18% Code, 6% Indic, 8% STEM, 4% Reasoning, 2% Long-ctx, 2% Agentic | Naive default |
| **C — No Indic Floor** | Same as A but Indic floor removed; OPUS English-heavy proxy allowed to starve Indic | Tests floor necessity |

**Each variant trains 3 seeds** (different random shuffles). Results are averaged.

### 9.3 Metrics That Confirm or Refute the Mixture

| Metric | Benchmark | Decision Rule to Endorse Variant A |
|---|---|---|
| MILU accuracy (Hindi + Telugu) | MILU 5-shot | A > B by **≥ 3pp** and A > C by **≥ 5pp** |
| AIME correctness rate | AIME 2024 (15 problems) | A ≥ B (reasoning traces don't hurt) |
| Function-call accuracy | BFCL v3 single-turn | A ≥ B − 2pp (agentic floor doesn't cost code quality) |
| Code perplexity | Held-out The Stack v2 test set | A perplexity ≤ B + 0.5 |
| General knowledge | MMLU 5-shot | A ≥ B − 1pp (acceptable small trade-off for Indic) |

**If A fails any of these conditions**, the mixture is revised. Specifically:
- If MILU A < B + 3pp → raise Indic share to 20%, run 3B proxy
- If general knowledge degrades by > 2pp → lower Indic to 16%, raise Web to 34%

### 9.4 3B Follow-up Run

If the 1B proxy endorses variant A, a **3B parameter, 100B token** run confirms the mix scales before going to full 40B. The 3B run also tests the anneal reserve composition (§5) using a 2% anneal of 2B tokens.

### 9.5 Data Gating Threshold

The plan is only trustworthy once the corpus behind it meets minimum cleaning standards. The gating threshold for entering any proxy run:

- **General Web**: ≥ 100B cleaned, deduplicated, manifested tokens
- **Indic Tier A**: ≥ 30B verified tokens with educational score ≥ 2.0
- **Code**: ≥ 50B Stack v2 tokens through the 8-stage pipeline
- **Reasoning**: ≥ 20B AON tokens with pipeline stages 1–7 applied
- All shards must carry a valid provenance manifest (§8, Session 4)

---

## 10. Summary — Decisions at a Glance

| Decision Point | Value | Reasoning |
|---|---|---|
| Total pretrain budget | 2T tokens | Compute-optimal + repeat margin for scarce lanes |
| Anneal reserve | 40B (2%) | Matches V4 reality; concentrates best data at low LR |
| Indic floor (OPUS) | 12% always-on | Below this, MILU degrades non-linearly at 3B scale |
| Agentic floor (OPUS) | 2% always-on | Minimum for embedding-space representation |
| Indic: Tier A share | 38% of 18% | Max quality; 2.1× repeat is safe |
| Indic: Tier C honest gap | 14.4× synthesis needed | States the problem, not a solution by fiat |
| Anneal agentic datasets | SWE-Gym + SWE-smith + OpenHands | Reserved; highest token-per-sample density |
| Warmup band at seams | 3B tokens, 60/40 blend | Prevents gradient explosion at mixture transitions |
| Proxy experiment scale | 1B / 30B tokens | Sufficient for mix ranking; costs ~$200 |
| Decision metric for Indic | MILU: A > B by ≥ 3pp | Concrete, falsifiable, benchmark-anchored |

---

## 11. Appendix — Full Dataset Inventory by Slot

### Code (1.1T tokens total)
| Dataset | Tokens | Tier | License |
|---|---|---|---|
| The Stack v2 | ~900B | B | Permissive + opt-out |
| D3 Code (V4 corpus) | ~199B | B | V4 lineage |
| CommitPack / CommitPackFT | ~4B | B | Mixed permissive |

### Agentic & Tool-use (627M tokens total — real supply)
| Dataset | Tokens | Tier | Reserved for Anneal? |
|---|---|---|---|
| SWE-Gym | 150M | A | **Yes — anneal only** |
| SWE-smith | 120M | A | **Partial (top 5K)** |
| OpenHands rollouts | 90M | A | **Yes — anneal only** |
| ToolBench | 80M | D | Main run |
| ToolACE | 60M | A/D | Main run |
| Glaive FC v2 | 50M | D | Main run |
| NexusRaven | 30M | A | Main run |
| xLAM / APIGen | 25M | A/D | Main run |
| Hermes FC | 22M | A/D | Main run |

**Main run real supply: ~337M tokens.** The remaining ~59.7B must be synthesized (§6).

### Reasoning & Math (85.1B tokens total)
| Dataset | Tokens | Tier | Use |
|---|---|---|---|
| AON (V4 corpus) | 78B | A | Main run + anneal top shards |
| OpenThoughts2 | ~3B | A/D | **Anneal only** |
| OpenMathReasoning | ~2B | A/D | Main run |
| OpenR1-Math | ~1.6B | D | Main run |
| NuminaMath | ~500M | A | Main run |

### Long-context (100B tokens)
| Dataset | Tokens | Tier |
|---|---|---|
| Repo-packed code (32K+ ctx) | ~60B | B |
| Book-length corpora (packed) | ~40B | B |

### Indic (276B total; see §3 for tier breakdown)
| Dataset | Tokens | Tier |
|---|---|---|
| Sangraha synthetic | 162B | C/D |
| Sangraha verified | 64B | A |
| IndicCorpV2 | ~20.9B | B |
| Sangraha unverified | ~24B | B |
| BPCC parallel | ~3B | B |
| Samanantar | ~2B | C |

### General Web & STEM (4.8T tokens)
| Dataset | Tokens | Tier |
|---|---|---|
| DCLM-Baseline | ~2.6T | B |
| FineWeb-Edu | ~1.3T | B |
| D2 Web-Diverse (V4) | ~627B | B |
| D1 Web-Foundation (V4) | ~164B | B |
| proof-pile-2 | ~55B | A |
| D4 STEM (V4) | ~49B | B |
| peS2o | ~42B | A |

---

*Specification complete. All numbers are tied to real supply figures from the dataset inventory. Where the numbers cannot be met with real data, this document says so plainly and describes what must be synthesized instead.*
