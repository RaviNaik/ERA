// ============================================================
// data.js — All data constants for the V5 Mixture & Curriculum
// Dashboard (Session 5 Assignment)
// ============================================================

const DATA = {

  // ── 1. PRETRAIN MIXTURE ──────────────────────────────────
  pretrain: [
    {
      lane: "General Web",
      share: 32,
      demand2T: 640,
      supply: 4500,
      status: "covered",
      color: "#6366f1",
      benchmarks: ["MMLU", "HellaSwag"],
      datasets: ["DCLM-Baseline (2.6T)", "FineWeb-Edu (1.3T)", "V4 D1/D2 (791B)"],
    },
    {
      lane: "Code",
      share: 24,
      demand2T: 480,
      supply: 1100,
      status: "covered",
      color: "#22d3ee",
      benchmarks: ["LiveCodeBench", "Aider Polyglot"],
      datasets: ["The Stack v2 (900B)", "D3 Code (199B)", "CommitPack (4B)"],
    },
    {
      lane: "Indic",
      share: 18,
      demand2T: 360,
      supply: 276,
      status: "repeat+synth",
      color: "#f59e0b",
      benchmarks: ["MILU", "IndicGenBench"],
      datasets: ["Sangraha verified (64B)", "Sangraha synthetic (162B)", "IndicCorpV2 (21B)", "Sangraha unverified (24B)", "BPCC (3B)", "Samanantar (2B)"],
    },
    {
      lane: "STEM / Math",
      share: 12,
      demand2T: 240,
      supply: 146,
      status: "repeat",
      color: "#10b981",
      benchmarks: ["AIME 2024/25", "GPQA Diamond"],
      datasets: ["proof-pile-2 (55B)", "D4 STEM (49B)", "peS2o (42B)"],
    },
    {
      lane: "Reasoning",
      share: 6,
      demand2T: 120,
      supply: 85,
      status: "repeat",
      color: "#8b5cf6",
      benchmarks: ["AIME (structural)", "HLE"],
      datasets: ["AON V4 (78B)", "OpenThoughts2 (3B)", "OpenMathReasoning (2B)", "NuminaMath (0.5B)"],
    },
    {
      lane: "Long-context",
      share: 5,
      demand2T: 100,
      supply: 100,
      status: "covered",
      color: "#06b6d4",
      benchmarks: ["HELMET long-eval"],
      datasets: ["Repo-packed 32K+ (60B)", "Book corpora (40B)"],
    },
    {
      lane: "Agentic",
      share: 3,
      demand2T: 60,
      supply: 0.363,
      status: "must-synth",
      color: "#f43f5e",
      benchmarks: ["SWE-bench Verified", "tau-bench", "BFCL v3"],
      datasets: ["ToolBench (80M)", "ToolACE (60M)", "Glaive FC (50M)", "NexusRaven (30M)", "xLAM (25M)", "Hermes (22M)", "SWE-smith main-run share (96M)"],
      note: "627M total inventory; 264M (SWE-Gym 150M + OpenHands 90M + SWE-smith top-5K ~24M) reserved for anneal — see anneal panel.",
    },
  ],

  // ── 2. ANNEAL MIXTURE ────────────────────────────────────
  anneal: [
    { lane: "Indic",       share: 28, color: "#f59e0b" },
    { lane: "Code",        share: 20, color: "#22d3ee" },
    { lane: "Reasoning",   share: 18, color: "#8b5cf6" },
    { lane: "STEM / Math", share: 10, color: "#10b981" },
    { lane: "Agentic",     share: 8,  color: "#f43f5e" },
    { lane: "Long-context",share: 8,  color: "#06b6d4" },
    { lane: "General Web", share: 8,  color: "#6366f1" },
  ],

  // ── 3. INDIC TIER BREAKDOWN ──────────────────────────────
  indicTiers: [
    {
      tier: "A — Verified Native",
      share: 38,
      demandB: 136.8,
      realSupplyB: 64,
      note: "Sangraha verified (64B) → 2.1× repeat",
      color: "#f59e0b",
      qual: "Highest",
    },
    {
      tier: "B — Unverified Crawl",
      share: 22,
      demandB: 79.2,
      realSupplyB: 44.9,
      note: "Sangraha unverified 24B + IndicCorpV2 20.9B → 1.8× repeat",
      color: "#fbbf24",
      qual: "Medium",
    },
    {
      tier: "C — Translated",
      share: 20,
      demandB: 72,
      realSupplyB: 5,
      note: "BPCC 3B + Samanantar 2B → 14.4× gap → ~67B synthesized (real 5B used as seed, not repeated)",
      color: "#fcd34d",
      qual: "Medium-low",
      warning: true,
    },
    {
      tier: "D — Synthetic",
      share: 20,
      demandB: 72,
      realSupplyB: 162,
      note: "Sangraha synthetic (162B) already covers demand — 90B surplus; select top 72B by quality score",
      color: "#fde68a",
      qual: "Variable",
      surplus: true,
    },
  ],

  // ── 4. DATASET INVENTORY (AGENTIC) ──────────────────────
  agenticDatasets: [
    { name: "SWE-Gym",       samples: 2400,  tokens: 150, tier: "A", anneal: true },
    { name: "SWE-smith",     samples: 26000, tokens: 120, tier: "A", anneal: "partial" },
    { name: "OpenHands",     samples: 10000, tokens: 90,  tier: "A", anneal: true },
    { name: "ToolBench",     samples: 120000,tokens: 80,  tier: "D", anneal: false },
    { name: "ToolACE",       samples: 110000,tokens: 60,  tier: "A/D",anneal: false },
    { name: "Glaive FC v2",  samples: 113000,tokens: 50,  tier: "D", anneal: false },
    { name: "NexusRaven",    samples: 40000, tokens: 30,  tier: "A", anneal: false },
    { name: "xLAM / APIGen", samples: 60000, tokens: 25,  tier: "A/D",anneal: false },
    { name: "Hermes FC",     samples: 15000, tokens: 22,  tier: "A/D",anneal: false },
  ],

  // ── 5. DIFFICULTY BANDS ──────────────────────────────────
  difficultyBands: [
    {
      band: "B0",
      level: "Nursery",
      reasoningTokens: "0–20",
      example: {
        problem: "What is 7 × 8?",
        reasoning: "7 multiplied by 8 equals 56.",
        answer: "56",
      },
      color: "#4ade80",
      stageUse: "Seed / General (0–15% of run)",
    },
    {
      band: "B1",
      level: "Medium",
      reasoningTokens: "21–80",
      example: {
        problem: "A train travels 240 km in 3 hours. How far in 5 hours at the same speed?",
        reasoning: "Speed = 240/3 = 80 km/h. Distance in 5 hours = 80 × 5 = 400 km.",
        answer: "400 km",
      },
      color: "#a3e635",
      stageUse: "Seed / General (0–15% of run)",
    },
    {
      band: "B2",
      level: "Hard (High-school)",
      reasoningTokens: "81–200",
      example: {
        problem: "How many integers between 1 and 1000 are divisible by 3 or 5?",
        reasoning: "Div by 3: ⌊1000/3⌋ = 333. Div by 5: ⌊1000/5⌋ = 200. Div by 15: ⌊1000/15⌋ = 66. Inclusion-exclusion: 333 + 200 − 66 = 467.",
        answer: "467",
      },
      color: "#facc15",
      stageUse: "Reasoning phase (15–60% of run)",
    },
    {
      band: "B3",
      level: "Undergraduate",
      reasoningTokens: "201–500",
      example: {
        problem: "AIME 2024 — Count n ≤ 1000 divisible by neither 3 nor 7.",
        reasoning: "Total = 1000. Div by 3: 333. Div by 7: 142. Div by 21: 47. Neither = 1000 − 333 − 142 + 47 = 572. Density check: 1000 × (2/3) × (6/7) ≈ 571.4 ✓",
        answer: "572",
      },
      color: "#fb923c",
      stageUse: "Reasoning phase (15–60%) + Long-ctx (60–80%)",
    },
    {
      band: "B4",
      level: "Graduate",
      reasoningTokens: "501–1500",
      example: {
        problem: "GPQA Diamond — Physics MCQ on quantum entanglement measurement correlation.",
        reasoning: "Rule out A and D by dimensional analysis. B ignores the Bell inequality coupling term. C correctly applies cos²θ correlation. With Δθ = 30°, correlation = cos²(30°) = 0.75. Matches C.",
        answer: "C",
      },
      color: "#f87171",
      stageUse: "Anneal (final 2% of run)",
    },
    {
      band: "B5",
      level: "PhD / Frontier",
      reasoningTokens: "1500+",
      example: {
        problem: "HLE — Expert cross-domain reasoning on information-theoretic learning bounds.",
        reasoning: "Multi-step derivation: measure theory → PAC learning bounds → Rademacher complexity → empirical verification with self-correction. Explores two dead ends before finding the tight bound.",
        answer: "Exact symbolic bound (2000+ reasoning tokens)",
      },
      color: "#e879f9",
      stageUse: "Anneal only (reserved; PhD-level before B3 foundation = wasted gradient)",
    },
  ],

  // ── 6. CURRICULUM STAGES ─────────────────────────────────
  curriculumStages: [
    {
      name: "Seed",
      tokenRange: "0–30B",
      position: 0,
      web: 52.3, code: 17.6, indic: 14.5, stem: 11.7, reasoning: 3.9, longctx: 0, agentic: 0,
      bands: "B0–B1",
      note: "Warm start. Language and factual structure.",
    },
    {
      name: "General",
      tokenRange: "30B–600B",
      position: 15,
      web: 36.6, code: 24.2, indic: 18, stem: 12.1, reasoning: 4.8, longctx: 2.3, agentic: 2,
      bands: "B0–B2",
      note: "Main pretrain proportions. All lanes active.",
    },
    {
      name: "Reasoning",
      tokenRange: "600B–1.4T",
      position: 50,
      web: 30.8, code: 24.7, indic: 18.4, stem: 12.3, reasoning: 6.6, longctx: 3.8, agentic: 3.4,
      bands: "B2–B3",
      note: "Reasoning and agentic shares grow. Web falls.",
    },
    {
      name: "Long-context",
      tokenRange: "1.4T–2T",
      position: 80,
      web: 28.1, code: 23.1, indic: 17.7, stem: 11.5, reasoning: 6.6, longctx: 9.4, agentic: 3.6,
      bands: "B3–B4",
      note: "Long-context surges to 9.4%. Sequences stretch. Main run (Seed→here) totals exactly 2T.",
    },
    {
      name: "Anneal",
      tokenRange: "2.003T–2.043T",
      position: 92,
      web: 8, code: 20, indic: 28, stem: 10, reasoning: 18, longctx: 8, agentic: 8,
      bands: "B3–B5",
      note: "Reserved best data. Low LR. Concentrated quality. 40B, additional to the 2T main run (not carved out of it).",
    },
  ],

  // ── 7. PROXY EXPERIMENT ──────────────────────────────────
  proxyExperiment: {
    scale1B: {
      params: "1B",
      tokens: "30B",
      cost: "~$200",
      duration: "~4 hours",
      variants: [
        { id: "A", label: "Proposed Mix", description: "32% Web, 24% Code, 18% Indic, 12% STEM, 6% Reasoning, 5% Long-ctx, 3% Agentic" },
        { id: "B", label: "Web-Heavy Baseline", description: "60% Web, 18% Code, 6% Indic, 8% STEM, 4% Reasoning, 2% Long-ctx, 2% Agentic" },
        { id: "C", label: "No Indic Floor", description: "Same as A but OPUS English-heavy proxy allowed to starve Indic" },
      ],
    },
    scale3B: {
      params: "3B",
      tokens: "100B",
      cost: "~$2,000",
      duration: "~2 days",
    },
    metrics: [
      { metric: "MILU accuracy (Hindi + Telugu)", benchmark: "MILU 5-shot", decision: "A > B by ≥ 3pp AND A > C by ≥ 5pp" },
      { metric: "AIME correctness", benchmark: "AIME 2024", decision: "A ≥ B (no regression from Indic investment)" },
      { metric: "Function-call accuracy", benchmark: "BFCL v3 single-turn", decision: "A ≥ B − 2pp (acceptable small cost)" },
      { metric: "Code perplexity", benchmark: "Stack v2 held-out test", decision: "A perplexity ≤ B + 0.5 nats" },
      { metric: "General knowledge", benchmark: "MMLU 5-shot", decision: "A ≥ B − 1pp (acceptable trade-off)" },
    ],
  },

  // ── 8. PROTECTED FLOORS ──────────────────────────────────
  protectedFloors: [
    {
      lane: "Indic",
      floor: 12,
      reason: "Below 12%, MILU accuracy in Tier-1 languages degrades non-linearly at 3B scale",
      mechanism: "OPUS always-on channel forces best Indic batches regardless of proxy utility",
    },
    {
      lane: "Agentic",
      floor: 2,
      reason: "Below 2%, agentic trajectories have negligible embedding-space representation",
      mechanism: "OPUS always-on channel forces best Agentic batches regardless of proxy utility, enforced independently of the Indic floor",
    },
  ],
};
