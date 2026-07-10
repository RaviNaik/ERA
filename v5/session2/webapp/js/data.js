// js/data.js — All experiment constants and metadata
// Assets live at ../bpe_assignment/ relative to webapp/index.html,
// which resolves to /bpe_assignment/ on the Netlify-served session2 root.

export const ASSET_BASE = '../bpe_assignment';

export const LANG_META = {
  en: { name: 'English',  title: 'India',       chars: 64971, words: 10027, avgLen: 5.48, color: '#6cb8e0', light: '#d4eef8', soft: '#a8d8f0', file: 'india_en.txt' },
  hi: { name: 'Hindi',   title: 'भारत',          chars: 43596, words: 8022,  avgLen: 4.43, color: '#9d8fe0', light: '#e4dff8', soft: '#c5bdf0', file: 'india_hi.txt' },
  te: { name: 'Telugu',  title: 'భారతదేశం',      chars: 19822, words: 2453,  avgLen: 6.58, color: '#5ebc8e', light: '#d0f0e0', soft: '#9dd9be', file: 'india_te.txt' },
  kn: { name: 'Kannada', title: 'ಭಾರತ',          chars: 7600,  words: 979,   avgLen: 6.43, color: '#e08a5e', light: '#fce4d4', soft: '#f0b898', file: 'india_kn.txt' },
};

export const EXPERIMENTS = [
  {
    id: 'step1',
    step: 'Experiment 1',
    name: 'English-Only BPE',
    desc: 'Vanilla BPE trained on English Wikipedia only. Establishes a baseline — no Indic awareness whatsoever.',
    accent: '#e07c8c', accentLight: '#fde0e6',
    config: {
      'Target Vocab':    '10,000',
      'Actual Vocab':    '2,922',
      'Pre-tokenizer':   'Whitespace',
      'Normalization':   'None',
      'Training Data':   'English only (1×)',
      'Min Frequency':   '2',
    },
    results: {
      en: { words: 10027, tokens: 15245, ratio: 1.5204 },
      hi: { words: 8022,  tokens: 35026, ratio: 4.3662 },
      te: { words: 2453,  tokens: 17056, ratio: 6.9531 },
      kn: { words: 979,   tokens: 6434,  ratio: 6.5720 },
    },
    xMin: 1.5204, xMax: 6.9531, spread: 5.4327, score: 184.07,
    modelFile: 'step1_en_only.json',
    rank: 7,
    insight: 'Indic scripts have no coverage — each word explodes into individual byte tokens. Telugu reaches 6.95 tokens/word.',
    findings: [
      'Indic characters (Devanagari, Telugu, Kannada) are completely absent from the trained vocabulary.',
      'Indic words are heavily fragmented, falling back to individual bytes (e.g., Telugu splits into 6.95 tokens/word).',
      'The vocabulary size is limited to 2,922 tokens (instead of the target 10,000) due to the small size of the English corpus.',
      'The difference between the minimum (English: 1.52) and maximum (Telugu: 6.95) fertility ratios is extremely large.'
    ],
    conclusions: [
      'An English-only vocabulary has zero Indic script awareness, yielding highly inefficient tokenization.',
      'This establishes our baseline with a very low score of 184.'
    ]
  },
  {
    id: 'step2',
    step: 'Experiment 2',
    name: 'Naive Multilingual BPE',
    desc: 'All 4 languages concatenated equally (1× each) with no preprocessing. Exposes the imbalance issues.',
    accent: '#5eba80', accentLight: '#d0f5e4',
    config: {
      'Target Vocab':    '10,000',
      'Actual Vocab':    '7,118',
      'Pre-tokenizer':   'Whitespace',
      'Normalization':   'None',
      'Training Data':   'All 4 languages (1× each)',
      'Min Frequency':   '2',
    },
    results: {
      en: { words: 10027, tokens: 15197, ratio: 1.5156 },
      hi: { words: 8022,  tokens: 11078, ratio: 1.3810 },
      te: { words: 2453,  tokens: 5137,  ratio: 2.0946 },
      kn: { words: 979,   tokens: 2099,  ratio: 2.1440 },
    },
    xMin: 1.3810, xMax: 2.1440, spread: 0.7631, score: 1310.49,
    modelFile: 'step2_naive_multilingual.json',
    rank: 4,
    insight: 'Surprisingly competitive — joint training naturally distributes vocab. Hindi benefits most (8K words, large corpus). Telugu/Kannada lag due to less training data.',
    findings: [
      'Joint training on all four concatenated corpora achieves a vocabulary size of 7,118 tokens.',
      'Vocabulary is distributed naturally by raw language frequency: Hindi and English dominate the vocabulary slots.',
      'Fertility ratios drop drastically for Indic languages (e.g., Telugu drops from 6.95 to 2.09).',
      'Telugu and Kannada still lag behind English and Hindi due to their smaller training corpus sizes.'
    ],
    conclusions: [
      'Joint training is highly effective at distributing subwords naturally across languages, raising the score to 1,310.',
      'Performance remains bottlenecked by corpus size imbalances.'
    ]
  },
  {
    id: 'step3a2',
    step: 'Experiment 3A2 ★★',
    name: 'Focused Sweet-Spot — en×1 hi×1 te×2 kn×2',
    desc: 'Insight from 3A1: Hindi was over-boosted at ×2 in Experiment 3A (X_HI=1.13, too low). Removing Hindi's boost (hi×1) while keeping te×2 and kn×2 tightly clusters all four ratios — achieving a spread of just 0.2481 and a score of 4,031.',
    accent: '#2b9e6a', accentLight: '#d0f5e4',
    config: {
      'Target Vocab':    '10,000',
      'Actual Vocab':    '10,000',
      'Pre-tokenizer':   'WhitespaceSplit',
      'Normalization':   'NFKC + ZWJ/ZWNJ',
      'Training Data':   'en×1 · hi×1 · te×2 · kn×2',
      'Min Frequency':   '2',
    },
    results: {
      en: { words: 10027, tokens: 14407, ratio: 1.4368 },
      hi: { words: 8022,  tokens: 10599, ratio: 1.3212 },
      te: { words: 2453,  tokens: 2916,  ratio: 1.1887 },
      kn: { words: 979,   tokens: 1201,  ratio: 1.2268 },
    },
    xMin: 1.1887, xMax: 1.4368, spread: 0.2481, score: 4031.09,
    modelFile: 'step3a2_focused.json',
    rank: 1,
    insight: 'The breakthrough: Hindi (8K words, 80% of English) never needed ×2 oversampling. By keeping hi×1 and te×2, kn×2, all four ratios converge tightly between 1.19 and 1.44 — a spread of 0.2481, more than halving Experiment 3A\\'s spread of 0.4172.',
    findings: [
      'All four fertility ratios converge tightly: X_EN=1.44, X_HI=1.32, X_TE=1.19, X_KN=1.23.',
      'Removing Hindi\\'s ×2 oversampling lets English rebalance and raises X_HI from 1.13 to 1.32.',
      'Keeping te×2 and kn×2 prevents the word-memorisation trap seen in Experiment 3A1.',
      'A sweep of kn×3-6 showed kn×3+ causes Kannada to memorise whole words (X_KN=1.00), confirming kn×2 is optimal.',
      'Spread drops from 0.4172 (3A) to 0.2481 — a 40.5% reduction.'
    ],
    conclusions: [
      'The root cause of 3A\\'s spread was Hindi over-boosting, not Indic under-representation.',
      'en×1 · hi×1 · te×2 · kn×2 achieves a score of 4,031 — a +68% improvement over Experiment 3A.'
    ]
  },
  {
    id: 'step3a',
    step: 'Experiment 3A',
    name: 'Optimized — Oversampling (2× Indic)',
    desc: 'NFKC normalization + ZWJ/ZWNJ removal + all Indic languages repeated ×2. Was the best result until Experiment 3A2 revealed Hindi was being over-boosted.',
    accent: '#d4902a', accentLight: '#fdefd4',
    config: {
      'Target Vocab':    '10,000',
      'Actual Vocab':    '10,000',
      'Pre-tokenizer':   'WhitespaceSplit',
      'Normalization':   'NFKC + ZWJ/ZWNJ',
      'Training Data':   'English 1× · Indic 2×',
      'Min Frequency':   '2',
    },
    results: {
      en: { words: 10027, tokens: 15179, ratio: 1.5138 },
      hi: { words: 8022,  tokens: 9059,  ratio: 1.1293 },
      te: { words: 2453,  tokens: 3665,  ratio: 1.4941 },
      kn: { words: 979,   tokens: 1514,  ratio: 1.5465 },
    },
    xMin: 1.1293, xMax: 1.5465, spread: 0.4172, score: 2396.89,
    modelFile: 'step3_strategy_a_oversample.json',
    rank: 2,
    insight: 'Previously the top scorer. Hindi over-boosted at ×2 (X_HI=1.13) was later identified as the spread driver. Experiment 3A2 fixed this by dropping hi to ×1, raising the score to 4,031.',
    findings: [
      'Removing ZWJ/ZWNJ characters and applying NFKC normalization prevents duplicate slots for identical glyph variants.',
      'Oversampling all Indic corpora by ×2 reduces their fertility ratios significantly.',
      'Hindi (8K words) is large enough to produce a good vocabulary at ×1, but ×2 over-corrects its ratio to 1.13.',
      'Achieves the highest actual vocabulary size of exactly 10,000 tokens.'
    ],
    conclusions: [
      'Uniform ×2 for all Indic languages was a reasonable first approach, yielding a score of 2,397.',
      'Superseded by Experiment 3A2, which revealed that Hindi did not need the ×2 boost.'
    ]
  },
  {
    id: 'step3a1',
    step: 'Experiment 3A1',
    name: 'Differential Per-Language Oversampling',
    desc: 'Per-language oversampling factors proportional to inverse corpus size: en×1, hi×1, te×3, kn×12. Tests whether targeting small languages more aggressively improves on the uniform ×2 of 3A.',
    accent: '#c46e3a', accentLight: '#fde8d4',
    config: {
      'Target Vocab':    '10,000',
      'Actual Vocab':    '10,000',
      'Pre-tokenizer':   'WhitespaceSplit',
      'Normalization':   'NFKC + ZWJ/ZWNJ',
      'Training Data':   'en×1 · hi×1 · te×3 · kn×12',
      'Min Frequency':   '2',
    },
    results: {
      en: { words: 10027, tokens: 15276, ratio: 1.5235 },
      hi: { words: 8022,  tokens: 11105, ratio: 1.3843 },
      te: { words: 2453,  tokens: 2453,  ratio: 1.0000 },
      kn: { words: 979,   tokens: 979,   ratio: 1.0000 },
    },
    xMin: 1.0000, xMax: 1.5235, spread: 0.5235, score: 1910.27,
    modelFile: 'step3a1_differential.json',
    rank: 3,
    insight: 'Over-sampling Telugu and Kannada so heavily causes the tokenizer to memorise entire words as single vocabulary entries (ratio = 1.00 = 1 token per word). This prevents the learning of composable subword merges, and the spread is actually WIDER than Experiment 3A because English and Hindi are now far above 1.00.',
    findings: [
      'A grid search of 18 factor combinations (en×1, hi×1–2, te×3–5, kn×8–12) was performed.',
      'Best found combination: en×1 · hi×1 · te×3 · kn×12, Score: 1,910.',
      'Telugu and Kannada reach a fertility ratio of exactly 1.00 — meaning entire words are stored as single tokens.',
      'Whole-word memorisation prevents composable subword learning and widens the spread to 0.5235 from 3A\'s 0.4172.',
      'Adding ×2 for Hindi (3A approach) actually helped by pulling Hindi\'s ratio down into the cluster.'
    ],
    conclusions: [
      'Differential oversampling is WORSE than uniform ×2, scoring 1,910 vs 3A\'s 2,397.',
      'Heavy per-language oversampling causes Indic word memorisation, not subword learning — confirming 3A is the global optimum for this corpus.'
    ]
  },
  {
    id: 'step3b',
    step: 'Experiment 3B',
    name: 'Optimized — Merged Vocabulary',
    desc: 'Four independent BPE tokenizers (2,500 tokens each) merged into one. NFKC + ZWJ/ZWNJ preprocessing applied.',
    accent: '#5080d0', accentLight: '#d4e4f8',
    config: {
      'Target Vocab':    '10,000',
      'Actual Vocab':    '6,848',
      'Pre-tokenizer':   'WhitespaceSplit',
      'Normalization':   'NFKC + ZWJ/ZWNJ',
      'Training Data':   'Per-language (2,500 each)',
      'Min Frequency':   '2',
    },
    results: {
      en: { words: 10027, tokens: 15484, ratio: 1.5442 },
      hi: { words: 8022,  tokens: 10468, ratio: 1.3049 },
      te: { words: 2453,  tokens: 4863,  ratio: 1.9825 },
      kn: { words: 979,   tokens: 2029,  ratio: 2.0725 },
    },
    perLangTrained: { en: 2500, hi: 2294, te: 1564, kn: 768 },
    xMin: 1.3049, xMax: 2.0725, spread: 0.7676, score: 1302.74,
    modelFile: 'step3_optimized.json',
    rank: 5,
    insight: 'Guaranteed equal allocation per language. Kannada only trained 768/2500 tokens due to small corpus (979 words) — data bottleneck limits gains. Beaten by the mathematically perfect oversampling factor of Experiment 3A.',
    findings: [
      'Merging four separate tokenizers (2,500 budget each) guarantees vocabulary space per language.',
      'Due to small data volume, Kannada trains only 768 tokens, leaving 1,732 slots unused.',
      'The final merged vocabulary size is only 6,848 unique tokens.',
      'The resulting fertility ratio spread (0.7676) leads to a score of 1,302.'
    ],
    conclusions: [
      'Guaranteed allocation is severely limited by a data bottleneck in small languages, resulting in wasted vocabulary slots.',
      'This strategy is outperformed by the joint training with oversampling (3A).'
    ]
  },
  {
    id: 'step3c',
    step: 'Experiment 3C',
    name: 'Experiment — ByteLevel BPE',
    desc: 'Uses GPT-2 style ByteLevel tokenization, mapping 256 bytes to characters. Eliminated [UNK] tokens but failed spectacularly on Indic scripts.',
    accent: '#8e7cc3', accentLight: '#e4dcf5',
    config: {
      'Target Vocab':    '10,000',
      'Actual Vocab':    '5,907',
      'Pre-tokenizer':   'ByteLevel',
      'Normalization':   'NFKC + ZWJ/ZWNJ',
      'Training Data':   'English 1× · Indic 2×',
      'Min Frequency':   '2',
    },
    results: {
      en: { words: 10027, tokens: 15506, ratio: 1.5464 },
      hi: { words: 8022,  tokens: 28005, ratio: 3.4910 },
      te: { words: 2453,  tokens: 14082, ratio: 5.7407 },
      kn: { words: 979,   tokens: 5072,  ratio: 5.1808 },
    },
    xMin: 1.5464, xMax: 5.7407, spread: 4.1943, score: 238.42,
    modelFile: 'step3c_bytelevel.json',
    rank: 6,
    insight: 'Indic characters take 3 bytes in UTF-8. A 5-letter Telugu word starts as 15 byte-tokens. The small corpus lacks the frequency data to learn how to re-assemble them, leaving the text highly fragmented.',
    findings: [
      'GPT-2 style ByteLevel tokenization maps 256 bytes to characters, successfully eliminating [UNK] tokens.',
      'Indic characters require 3 bytes per character in UTF-8, tripling the initial length of sequences.',
      'Due to data scarcity, the tokenizer fails to learn merges to re-assemble bytes, resulting in extreme fragmentation.',
      'Fertility ratios explode for Hindi (3.49), Telugu (5.74), and Kannada (5.18).'
    ],
    conclusions: [
      'Byte-level tokenization is highly detrimental for Indic scripts under low-resource constraints.',
      'The heavy byte-level fragmentation results in a low score of 238.'
    ]
  },
];

// Ordered by score descending for leaderboard
export const LEADERBOARD = [...EXPERIMENTS].sort((a, b) => b.score - a.score);

export const PREPROCESSING = [
  { transform: 'NFKC Normalization', op: 'unicodedata.normalize("NFKC", text)', effect: 'Collapses compatibility equivalents — different byte sequences for the same visible glyph', removed: { en: 49, hi: 69, te: 64, kn: 45 } },
  { transform: 'Remove ZWJ', op: 'Strip U+200D', effect: 'Prevents क्ष vs क्‍ष being treated as different tokens' },
  { transform: 'Remove ZWNJ', op: 'Strip U+200C', effect: 'Prevents क्‌ष variant from creating a 3rd duplicate token' },
  { transform: 'Collapse whitespace', op: 're.sub(r"[ \\t]+", " ", text)', effect: 'Normalizes inconsistent spacing in Wikipedia extracts' },
];
