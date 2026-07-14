// js/data.js — All experiment constants and metadata
// All experiments use the faithful Markdown corpus (HTML->MD conversion)
// and Metaspace pre-tokenizer/decoder (satisfies evaluator faithfulness check).
// Fertility = tokens / faithful_units  (NOT tokens / whitespace words)

export const ASSET_BASE = './assets';

// Corpus sizes from faithful Markdown (HTML -> Markdown, links/URLs preserved)
export const LANG_META = {
  en: { name: 'English',  title: 'India',     chars: 601843, faithfulUnits: 186367, color: '#6cb8e0', light: '#d4eef8', soft: '#a8d8f0', file: 'india_en.txt' },
  hi: { name: 'Hindi',   title: '\u092d\u093e\u0930\u0924',        chars: 463156, faithfulUnits: 88359,  color: '#9d8fe0', light: '#e4dff8', soft: '#c5bdf0', file: 'india_hi.txt' },
  te: { name: 'Telugu',  title: '\u0c2d\u0c3e\u0c30\u0c24\u0c26\u0c47\u0c36\u0c02',    chars: 199575, faithfulUnits: 36292,  color: '#5ebc8e', light: '#d0f0e0', soft: '#9dd9be', file: 'india_te.txt' },
  kn: { name: 'Kannada', title: '\u0cad\u0cbe\u0cb0\u0ca4',        chars: 64057,  faithfulUnits: 12293,  color: '#e08a5e', light: '#fce4d4', soft: '#f0b898', file: 'india_kn.txt' },
};

// Experiments are listed best-first (highest score first) for the page display.
// Numbered 1-6 in chronological order of discovery.

export const EXPERIMENTS = [

  // ── Experiment 6 — Best score, submitted tokenizer ─────────────────────
  {
    id: 'exp6',
    step: 'Experiment 6',
    name: 'Focused Sweet-Spot BPE',
    subtitle: 'Optimal Weight Tuning \u2014 en\u00d71 hi\u00d71 te\u00d72 kn\u00d74',
    desc: 'Insight from Experiment 4: English and Hindi have enough corpus data at \u00d71; ' +
          'only Telugu (\u00d72) and Kannada (\u00d74) need a targeted boost. ' +
          'This pulls all four languages into a tight 0.03 spread cluster, ' +
          'achieving a score of 33,207. This is the submitted tokenizer.',
    accent: '#1db954', accentLight: '#d0f8e4',
    config: {
      'Target Vocab':   '10,000',
      'Actual Vocab':   '10,000',
      'Pre-tokenizer':  'Metaspace',
      'Normalization':  'NFKC',
      'Training Weights': 'en\u00d71 \u00b7 hi\u00d71 \u00b7 te\u00d72 \u00b7 kn\u00d74',
      'Min Frequency':  '1',
      'Faithfulness':   'PASS \u2014 decode(encode(x)) == x',
    },
    results: {
      en: { faithfulUnits: 186367, tokens: 118098, ratio: 0.6337 },
      hi: { faithfulUnits: 88359,  tokens: 57834,  ratio: 0.6545 },
      te: { faithfulUnits: 36292,  tokens: 23115,  ratio: 0.6369 },
      kn: { faithfulUnits: 12293,  tokens: 7676,   ratio: 0.6244 },
    },
    xMin: 0.6244, xMax: 0.6545, spread: 0.0301, score: 33207.21,
    modelFile: 'tokenizer.json',
    metricsFile: 'exp6_metrics.json',
    rank: 1,
    insight: 'All four ratios lie within 0.62\u20130.65 (spread 0.030). ' +
             'English and Hindi are large enough to self-regulate at \u00d71. ' +
             'Telugu \u00d72 and Kannada \u00d74 pull the smaller-corpus languages into the same band.',
    findings: [
      'All four ratios lie within 0.6244\u20130.6545 \u2014 a spread of just 0.0301.',
      'English (186K units) and Hindi (88K units) need no oversampling at \u00d71.',
      'Telugu \u00d72 and Kannada \u00d74 bring smaller-corpus languages into the cluster.',
      'Hindi penalty factor = 1.0; ratio 0.6545 is far below the 1.2 threshold.',
      'decode(encode(text)) passes for all samples including URLs, #, /, (, ), [, ].',
    ],
    conclusions: [
      'Score 33,207 is 42\u00d7 better than the English-only baseline (779) and 8.8\u00d7 better than naive multilingual (3,788).',
      'Targeted per-language oversampling \u2014 boosting only where corpus is small \u2014 is the key insight.',
    ],
  },

  // ── Experiment 5 — Merged vocabulary ──────────────────────────────────────
  {
    id: 'exp5',
    step: 'Experiment 5',
    name: 'Merged Vocabulary BPE',
    subtitle: 'Per-Language Budget Allocation \u2014 2,500 tokens each',
    desc: 'Four independent BPE tokenizers trained per language (2,500 vocab budget each), ' +
          'then merged into a single shared vocabulary. Guarantees every language gets a ' +
          'dedicated vocabulary block. Score 13,877.',
    accent: '#5080d0', accentLight: '#d4e4f8',
    config: {
      'Target Vocab':   '10,000 (2,500 per language)',
      'Actual Vocab':   '7,997',
      'Pre-tokenizer':  'Metaspace',
      'Normalization':  'NFKC',
      'Training Weights': 'Independent per language',
      'Min Frequency':  '1',
      'Faithfulness':   'PASS \u2014 decode(encode(x)) == x',
    },
    results: {
      en: { faithfulUnits: 186367, tokens: 139934, ratio: 0.7509 },
      hi: { faithfulUnits: 88359,  tokens: 65811,  ratio: 0.7448 },
      te: { faithfulUnits: 36292,  tokens: 29646,  ratio: 0.8169 },
      kn: { faithfulUnits: 12293,  tokens: 9755,   ratio: 0.7935 },
    },
    perLangTrained: { en: 2499, hi: 1871, te: 1843, kn: 1783 },
    xMin: 0.7448, xMax: 0.8169, spread: 0.0721, score: 13877.23,
    modelFile: 'exp3b_merged.json',
    metricsFile: 'exp5_metrics.json',
    rank: 2,
    insight: 'Score 13,877. Each language gets a guaranteed vocabulary budget. ' +
             'Cross-language merge conflicts slightly inflate ratios (0.74\u20130.82) ' +
             'compared to joint training, but the cluster is still very tight.',
    findings: [
      'All four ratios lie between 0.74 and 0.82 \u2014 a tight cluster.',
      'Hindi filled only 1,871/2,500 budget slots; its large corpus is efficient.',
      'Telugu and Kannada filled ~1,800 slots each despite smaller corpora.',
      'Score 13,877 is 2.4\u00d7 lower than Experiment 6 due to merge-compatibility overhead.',
    ],
    conclusions: [
      'Merged vocabulary is robust but slightly sub-optimal vs joint training with per-language weights.',
      'This approach guarantees minimum Indic coverage regardless of corpus balance.',
    ],
  },

  // ── Experiment 4 — Differential oversampling ──────────────────────────────
  {
    id: 'exp4',
    step: 'Experiment 4',
    name: 'Differential Oversampling BPE',
    subtitle: 'Inverse-Corpus-Size Weights \u2014 en\u00d71 hi\u00d71 te\u00d73 kn\u00d76',
    desc: 'Oversampling factors set proportional to the inverse of each corpus size. ' +
          'Telugu (smallest Indic, \u00d73) and Kannada (smallest, \u00d76) are boosted aggressively. ' +
          'Discovers the "overshoot" problem: Kannada drops too low, making Hindi the new outlier. Score 7,247.',
    accent: '#c46e3a', accentLight: '#fde8d4',
    config: {
      'Target Vocab':   '10,000',
      'Actual Vocab':   '10,000',
      'Pre-tokenizer':  'Metaspace',
      'Normalization':  'NFKC',
      'Training Weights': 'en\u00d71 \u00b7 hi\u00d71 \u00b7 te\u00d73 \u00b7 kn\u00d76',
      'Min Frequency':  '1',
      'Faithfulness':   'PASS \u2014 decode(encode(x)) == x',
    },
    results: {
      en: { faithfulUnits: 186367, tokens: 121722, ratio: 0.6531 },
      hi: { faithfulUnits: 88359,  tokens: 60631,  ratio: 0.6862 },
      te: { faithfulUnits: 36292,  tokens: 21905,  ratio: 0.6036 },
      kn: { faithfulUnits: 12293,  tokens: 6739,   ratio: 0.5482 },
    },
    xMin: 0.5482, xMax: 0.6862, spread: 0.1380, score: 7246.84,
    modelFile: 'exp3a1_differential.json',
    metricsFile: 'exp4_metrics.json',
    rank: 3,
    insight: 'Score 7,247. Kannada (0.548) and Telugu (0.604) are over-compressed \u2014 ' +
             'they are now too low compared to English (0.653) and Hindi (0.686), ' +
             'which become the new spread drivers.',
    findings: [
      'Kannada (0.548) and Telugu (0.604) are over-boosted at \u00d76 and \u00d73 respectively.',
      'English (0.653) and Hindi (0.686) are now the outliers \u2014 the spread flips.',
      'Spread 0.138 is larger than Experiment 5 because both ends of the range are now pulled apart.',
    ],
    conclusions: [
      'Over-boosting small languages creates a new imbalance; the fix is moderate weights.',
      'Key insight for Experiment 6: reduce te to \u00d72 and kn to \u00d74 to hit the sweet spot.',
    ],
  },

  // ── Experiment 3 — Uniform oversampling ───────────────────────────────────
  {
    id: 'exp3',
    step: 'Experiment 3',
    name: 'Uniform Indic Oversampling BPE',
    subtitle: 'All Indic Languages \u00d72 \u2014 en\u00d71 hi\u00d72 te\u00d72 kn\u00d72',
    desc: 'All three Indic languages oversampled \u00d72 to counter their smaller corpus sizes. ' +
          'First significant improvement over the naive baseline. Reveals that Hindi is ' +
          'over-boosted at \u00d72 (its corpus is large enough at \u00d71). Score 4,438.',
    accent: '#d4902a', accentLight: '#fdefd4',
    config: {
      'Target Vocab':   '10,000',
      'Actual Vocab':   '10,000',
      'Pre-tokenizer':  'Metaspace',
      'Normalization':  'NFKC',
      'Training Weights': 'en\u00d71 \u00b7 hi\u00d72 \u00b7 te\u00d72 \u00b7 kn\u00d72',
      'Min Frequency':  '1',
      'Faithfulness':   'PASS \u2014 decode(encode(x)) == x',
    },
    results: {
      en: { faithfulUnits: 186367, tokens: 120872, ratio: 0.6486 },
      hi: { faithfulUnits: 88359,  tokens: 50610,  ratio: 0.5728 },
      te: { faithfulUnits: 36292,  tokens: 23674,  ratio: 0.6523 },
      kn: { faithfulUnits: 12293,  tokens: 9811,   ratio: 0.7981 },
    },
    xMin: 0.5728, xMax: 0.7981, spread: 0.2253, score: 4438.14,
    modelFile: 'exp3a_oversample.json',
    metricsFile: 'exp3_metrics.json',
    rank: 4,
    insight: 'Score 4,438. Hindi drops to 0.573 (over-boosted) while Kannada remains highest (0.798). ' +
             'The Hindi\u2013Kannada gap drives the 0.225 spread.',
    findings: [
      'Hindi drops to 0.573 \u2014 over-boosted at \u00d72 (Hindi has 88K units, large enough at \u00d71).',
      'Kannada (smallest corpus, 12K units) still lags at 0.798 despite \u00d72 boost.',
      'Key discovery: per-language weights are needed, not uniform oversampling.',
    ],
    conclusions: [
      'Uniform oversampling improves over naive (3,788 \u2192 4,438) but leaves a 0.225 spread.',
      'Insight: drop Hindi to \u00d71 and boost Kannada further in the next experiment.',
    ],
  },

  // ── Experiment 2 — Naive multilingual ─────────────────────────────────────
  {
    id: 'exp2',
    step: 'Experiment 2',
    name: 'Naive Multilingual BPE',
    subtitle: 'Equal Corpus Weights \u2014 All Languages \u00d71',
    desc: 'All four languages concatenated with equal weight (1\u00d7 each), no oversampling or tuning. ' +
          'Establishes a strong multilingual baseline. Kannada \u2014 with the smallest corpus \u2014 ' +
          'drives the spread at 0.864. Score 3,788.',
    accent: '#5eba80', accentLight: '#d0f5e4',
    config: {
      'Target Vocab':   '10,000',
      'Actual Vocab':   '10,000',
      'Pre-tokenizer':  'Metaspace',
      'Normalization':  'NFKC',
      'Training Weights': 'en\u00d71 \u00b7 hi\u00d71 \u00b7 te\u00d71 \u00b7 kn\u00d71',
      'Min Frequency':  '1',
      'Faithfulness':   'PASS \u2014 decode(encode(x)) == x',
    },
    results: {
      en: { faithfulUnits: 186367, tokens: 111759, ratio: 0.5997 },
      hi: { faithfulUnits: 88359,  tokens: 54454,  ratio: 0.6163 },
      te: { faithfulUnits: 36292,  tokens: 25561,  ratio: 0.7043 },
      kn: { faithfulUnits: 12293,  tokens: 10617,  ratio: 0.8637 },
    },
    xMin: 0.5997, xMax: 0.8637, spread: 0.2640, score: 3788.01,
    modelFile: 'exp2_naive.json',
    metricsFile: 'exp2_metrics.json',
    rank: 5,
    insight: 'Score 3,788. Equal weights produce a 0.264 spread driven entirely by Kannada ' +
             '(smallest corpus, 12K faithful units vs English 186K).',
    findings: [
      'English (0.600) and Hindi (0.616) self-balance \u2014 their large corpora self-regulate.',
      'Telugu (0.704) and Kannada (0.864) lag due to 5\u00d7 and 15\u00d7 smaller corpus vs English.',
      'Kannada is the sole bottleneck: 12K units vs English 186K.',
    ],
    conclusions: [
      'Naive multilingual training already achieves 3,788 \u2014 a strong starting point.',
      'The corpus size gap between Kannada and English identifies oversampling as the next lever.',
    ],
  },

  // ── Experiment 1 — English-only baseline ──────────────────────────────────
  {
    id: 'exp1',
    step: 'Experiment 1',
    name: 'English-Only BPE',
    subtitle: 'Monolingual Baseline \u2014 English Wikipedia Only',
    desc: 'Vanilla BPE trained exclusively on the English Wikipedia faithful Markdown corpus. ' +
          'Proves that an English-only vocabulary completely fails Indic scripts \u2014 ' +
          'every Indic character is an unknown and maps to a single token. Score 779.',
    accent: '#e07c8c', accentLight: '#fde0e6',
    config: {
      'Target Vocab':   '10,000',
      'Actual Vocab':   '10,000',
      'Pre-tokenizer':  'Metaspace',
      'Normalization':  'NFKC',
      'Training Weights': 'English only',
      'Min Frequency':  '1',
      'Faithfulness':   'PASS \u2014 decode(encode(x)) == x',
    },
    results: {
      en: { faithfulUnits: 186367, tokens: 95249,  ratio: 0.5111 },
      hi: { faithfulUnits: 88359,  tokens: 153208, ratio: 1.7339 },
      te: { faithfulUnits: 36292,  tokens: 65149,  ratio: 1.7951 },
      kn: { faithfulUnits: 12293,  tokens: 21272,  ratio: 1.7304 },
    },
    xMin: 0.5111, xMax: 1.7951, spread: 1.2841, score: 778.79,
    modelFile: 'exp1_en_only.json',
    metricsFile: 'exp1_metrics.json',
    rank: 6,
    insight: 'All three Indic languages cluster near 1.73 \u2014 each word maps to ~3.4 tokens ' +
             'because Devanagari, Telugu, and Kannada characters have no BPE merges and ' +
             'fall back to individual Unicode code points.',
    findings: [
      'English ratio 0.511 is excellent (the tokenizer was trained solely on English).',
      'All three Indic languages cluster near ratio 1.73 \u2014 character-level fragmentation.',
      'Score 779 vs Experiment 6\u2019s 33,207 = 42\u00d7 improvement from multilingual training.',
    ],
    conclusions: [
      'English-only BPE has zero Indic awareness. Multilingual training is mandatory.',
      'This establishes the floor: even perfect monolingual tokenization fails multilingually.',
    ],
  },
];

// Ordered by score descending for leaderboard
export const LEADERBOARD = [...EXPERIMENTS].sort((a, b) => b.score - a.score);
