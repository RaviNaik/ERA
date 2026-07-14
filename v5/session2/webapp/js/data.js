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

export const EXPERIMENTS = [

  // ── SUBMITTED TOKENIZER — Experiment 3A2 (BEST) ───────────────────────────
  {
    id: 'step3a2',
    step: 'Submitted \u2605\u2605\u2605',
    name: 'Focused Sweet-Spot \u2014 en\u00d71 hi\u00d71 te\u00d72 kn\u00d74',
    desc: 'Final submitted tokenizer (Experiment 3A2). Hypothesis: English and Hindi have ' +
          'enough data at \u00d71; only Telugu (\u00d72) and Kannada (\u00d74) need boosting. ' +
          'Result: all four ratios compress into a 0.03 spread \u2014 score 33,207. ' +
          'Passes decode(encode(text)) faithfulness check for all samples including URLs, brackets, and punctuation.',
    accent: '#1db954', accentLight: '#d0f8e4',
    config: {
      'Target Vocab':  '10,000',
      'Actual Vocab':  '10,000',
      'Pre-tokenizer': 'Metaspace (\u25b1)',
      'Normalization': 'NFKC',
      'Training Data': 'en\u00d71 \u00b7 hi\u00d71 \u00b7 te\u00d72 \u00b7 kn\u00d74 (faithful Markdown)',
      'Min Frequency': '1',
      'Faithfulness':  'PASS \u2014 decode(encode(x)) == x',
    },
    results: {
      en: { faithfulUnits: 186367, tokens: 118098, ratio: 0.6337 },
      hi: { faithfulUnits: 88359,  tokens: 57834,  ratio: 0.6545 },
      te: { faithfulUnits: 36292,  tokens: 23115,  ratio: 0.6369 },
      kn: { faithfulUnits: 12293,  tokens: 7676,   ratio: 0.6244 },
    },
    xMin: 0.6244, xMax: 0.6545, spread: 0.0301, score: 33207.21,
    modelFile: 'tokenizer.json',
    rank: 1,
    insight: 'Score 33,207 with all four ratios in a 0.62\u20130.65 band (spread only 0.030). ' +
             'English and Hindi are large enough to need no oversampling. ' +
             'Telugu \u00d72 and Kannada \u00d74 pull the smaller-corpus languages into the same tight cluster.',
    findings: [
      'All four ratios lie within 0.6244\u20130.6545 \u2014 a spread of just 0.0301.',
      'English (186K units) and Hindi (88K units) need no boosting at \u00d71.',
      'Telugu \u00d72 and Kannada \u00d74 bring smaller-corpus languages into the same band.',
      'Hindi penalty factor = 1.0 (ratio 0.6545 is far below the 1.2 threshold).',
      'decode(encode(text)) passes for all samples including URLs, #, /, (, ), [, ].',
    ],
    conclusions: [
      'Score 33,207 is 5.3\u00d7 better than the naive multilingual baseline (6,309).',
      'Targeted per-language oversampling is the key insight: boost only where the corpus is small.',
      'This is the submitted tokenizer.json for the ERA Session 2 assignment.',
    ],
  },

  // ── Experiment 3B: Merged vocabulary ─────────────────────────────────────
  {
    id: 'step3b',
    step: 'Experiment 3B',
    name: 'Merged Vocabulary \u2014 4 independent tokenizers',
    desc: 'Four independent BPE tokenizers (2,500 vocab each) trained per language, then merged. ' +
          'Guarantees vocabulary space allocation but produces slightly higher ratios than joint training.',
    accent: '#5080d0', accentLight: '#d4e4f8',
    config: {
      'Target Vocab':  '10,000 (2,500 per lang)',
      'Actual Vocab':  '7,997',
      'Pre-tokenizer': 'Metaspace (\u25b1)',
      'Normalization': 'NFKC',
      'Training Data': 'Per-language independent (faithful Markdown)',
      'Min Frequency': '1',
      'Faithfulness':  'PASS \u2014 decode(encode(x)) == x',
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
    rank: 2,
    insight: 'Score 13,877. Each language gets guaranteed vocab budget, but inter-language merge conflicts ' +
             'produce slightly higher ratios (0.74\u20130.82) than the joint-training experiments.',
    findings: [
      'All four ratios are tightly clustered between 0.74 and 0.82.',
      'Hindi trained only 1,871/2,500 budget tokens \u2014 its large corpus fills the budget efficiently.',
      'Telugu and Kannada each trained ~1,800 tokens despite their smaller corpora.',
      'Score 13,877 is strong but 2.4\u00d7 lower than the submitted 3A2.',
    ],
    conclusions: [
      'Merged vocabulary is robust but slightly sub-optimal vs joint training with targeted oversampling.',
      'The merge process introduces cross-language token conflicts that inflate ratios slightly.',
    ],
  },

  // ── Experiment 3A1: Differential oversampling ─────────────────────────────
  {
    id: 'step3a1',
    step: 'Experiment 3A1',
    name: 'Differential Oversampling \u2014 en\u00d71 hi\u00d71 te\u00d73 kn\u00d76',
    desc: 'Aggressive per-language oversampling proportional to inverse corpus size. ' +
          'Boosts small languages but overshoots \u2014 Telugu and Kannada become too low, creating a new spread.',
    accent: '#c46e3a', accentLight: '#fde8d4',
    config: {
      'Target Vocab':  '10,000',
      'Actual Vocab':  '10,000',
      'Pre-tokenizer': 'Metaspace (\u25b1)',
      'Normalization': 'NFKC',
      'Training Data': 'en\u00d71 \u00b7 hi\u00d71 \u00b7 te\u00d73 \u00b7 kn\u00d76 (faithful Markdown)',
      'Min Frequency': '1',
      'Faithfulness':  'PASS \u2014 decode(encode(x)) == x',
    },
    results: {
      en: { faithfulUnits: 186367, tokens: 121722, ratio: 0.6531 },
      hi: { faithfulUnits: 88359,  tokens: 60631,  ratio: 0.6862 },
      te: { faithfulUnits: 36292,  tokens: 21905,  ratio: 0.6036 },
      kn: { faithfulUnits: 12293,  tokens: 6739,   ratio: 0.5482 },
    },
    xMin: 0.5482, xMax: 0.6862, spread: 0.1380, score: 7246.84,
    modelFile: 'exp3a1_differential.json',
    rank: 3,
    insight: 'Score 7,247. Over-boosting Telugu (\u00d73) and Kannada (\u00d76) pushes them too low (0.548\u20130.604), ' +
             'making English (0.653) and Hindi (0.686) the new spread drivers. ' +
             'The fix in 3A2: reduce te to \u00d72 (not \u00d73) to pull it back up toward the cluster.',
    findings: [
      'Telugu (0.604) and Kannada (0.548) benefit from oversampling but overshoot.',
      'English (0.653) and Hindi (0.686) are now the outliers that drive the spread.',
      'Spread 0.138 is larger than 3A2 (0.030) because of over-compression of small languages.',
    ],
    conclusions: [
      'Insight for 3A2: \u00d73 on Telugu is too aggressive \u2014 \u00d72 brings it into the cluster without overshooting.',
      '\u00d76 on Kannada is also too aggressive; \u00d74 achieves the same cluster alignment.',
    ],
  },

  // ── Experiment 3A: Uniform oversampling ──────────────────────────────────
  {
    id: 'step3a',
    step: 'Experiment 3A',
    name: 'Uniform Oversampling \u2014 en\u00d71 hi\u00d72 te\u00d72 kn\u00d72',
    desc: 'All Indic languages oversampled \u00d72. First major improvement over naive equal-weight training. ' +
          'Boosts all Indic languages uniformly, but Hindi is over-boosted (large corpus at \u00d72 \u2192 too low).',
    accent: '#d4902a', accentLight: '#fdefd4',
    config: {
      'Target Vocab':  '10,000',
      'Actual Vocab':  '10,000',
      'Pre-tokenizer': 'Metaspace (\u25b1)',
      'Normalization': 'NFKC',
      'Training Data': 'en\u00d71 \u00b7 hi\u00d72 \u00b7 te\u00d72 \u00b7 kn\u00d72 (faithful Markdown)',
      'Min Frequency': '1',
      'Faithfulness':  'PASS \u2014 decode(encode(x)) == x',
    },
    results: {
      en: { faithfulUnits: 186367, tokens: 120872, ratio: 0.6486 },
      hi: { faithfulUnits: 88359,  tokens: 50610,  ratio: 0.5728 },
      te: { faithfulUnits: 36292,  tokens: 23674,  ratio: 0.6523 },
      kn: { faithfulUnits: 12293,  tokens: 9811,   ratio: 0.7981 },
    },
    xMin: 0.5728, xMax: 0.7981, spread: 0.2253, score: 4438.14,
    modelFile: 'exp3a_oversample.json',
    rank: 4,
    insight: 'Score 4,438. Uniform \u00d72 over-boosts Hindi (drops to 0.573 \u2014 now lowest), ' +
             'while Kannada remains highest (0.798). The Hindi\u2013Kannada gap drives the spread.',
    findings: [
      'Hindi drops to 0.573 from 0.616 (over-boosted at \u00d72 \u2014 already large enough at \u00d71).',
      'Kannada (smallest corpus) still lags at 0.798 despite \u00d72 boost \u2014 needs more.',
      'The key insight: Hindi \u00d72 is wasteful; dropping it to \u00d71 frees budget for kn/te.',
    ],
    conclusions: [
      'Uniform oversampling improves on naive (3,788 \u2192 4,438) but leaves a 0.225 spread.',
      'Lesson: per-language tuning is necessary \u2014 Hindi has enough data at \u00d71.',
    ],
  },

  // ── Experiment 2: Naive multilingual ─────────────────────────────────────
  {
    id: 'step2',
    step: 'Experiment 2',
    name: 'Naive Multilingual \u2014 equal weights',
    desc: 'All four languages concatenated equally (1\u00d7 each). No oversampling. ' +
          'Surprisingly strong baseline on the faithful Markdown corpus.',
    accent: '#5eba80', accentLight: '#d0f5e4',
    config: {
      'Target Vocab':  '10,000',
      'Actual Vocab':  '10,000',
      'Pre-tokenizer': 'Metaspace (\u25b1)',
      'Normalization': 'NFKC',
      'Training Data': 'en\u00d71 \u00b7 hi\u00d71 \u00b7 te\u00d71 \u00b7 kn\u00d71 (faithful Markdown)',
      'Min Frequency': '1',
      'Faithfulness':  'PASS \u2014 decode(encode(x)) == x',
    },
    results: {
      en: { faithfulUnits: 186367, tokens: 111759, ratio: 0.5997 },
      hi: { faithfulUnits: 88359,  tokens: 54454,  ratio: 0.6163 },
      te: { faithfulUnits: 36292,  tokens: 25561,  ratio: 0.7043 },
      kn: { faithfulUnits: 12293,  tokens: 10617,  ratio: 0.8637 },
    },
    xMin: 0.5997, xMax: 0.8637, spread: 0.2640, score: 3788.01,
    modelFile: 'exp2_naive.json',
    rank: 5,
    insight: 'Score 3,788. Even equal weights produce good ratios (0.60\u20130.86). ' +
             'Kannada (smallest corpus, 12K units) drives the spread at 0.864.',
    findings: [
      'English (0.600) and Hindi (0.616) nearly balance \u2014 their large corpora self-regulate.',
      'Telugu (0.704) and Kannada (0.864) lag due to smaller corpus sizes.',
      'Kannada is the bottleneck: only 12K faithful units vs English 186K.',
    ],
    conclusions: [
      'Naive equal-weight training on faithful corpus is already score 3,788 \u2014 a strong baseline.',
      'The gap between Kannada and the others identifies oversampling as the next step.',
    ],
  },

  // ── Experiment 1: English-only ────────────────────────────────────────────
  {
    id: 'step1',
    step: 'Experiment 1',
    name: 'English-Only BPE \u2014 Baseline',
    desc: 'Vanilla BPE trained on English Wikipedia only. Proves that an English-only vocabulary ' +
          'completely fails Indic scripts \u2014 all three fragment into single Unicode code points.',
    accent: '#e07c8c', accentLight: '#fde0e6',
    config: {
      'Target Vocab':  '10,000',
      'Actual Vocab':  '10,000',
      'Pre-tokenizer': 'Metaspace (\u25b1)',
      'Normalization': 'NFKC',
      'Training Data': 'English only (faithful Markdown)',
      'Min Frequency': '1',
      'Faithfulness':  'PASS \u2014 decode(encode(x)) == x',
    },
    results: {
      en: { faithfulUnits: 186367, tokens: 95249,  ratio: 0.5111 },
      hi: { faithfulUnits: 88359,  tokens: 153208, ratio: 1.7339 },
      te: { faithfulUnits: 36292,  tokens: 65149,  ratio: 1.7951 },
      kn: { faithfulUnits: 12293,  tokens: 21272,  ratio: 1.7304 },
    },
    xMin: 0.5111, xMax: 1.7951, spread: 1.2841, score: 778.79,
    modelFile: 'exp1_en_only.json',
    rank: 6,
    insight: 'Score 779. All three Indic languages cluster near 1.73 \u2014 each Indic word maps to ~3.4 tokens ' +
             'because Devanagari, Telugu, and Kannada characters have no BPE merges.',
    findings: [
      'English ratio 0.511 is excellent (trained on English).',
      'All three Indic languages cluster near 1.73 \u2014 character-level fragmentation.',
      'Score 779 vs submitted 33,207 = 42\u00d7 improvement from multilingual training.',
    ],
    conclusions: [
      'English-only BPE has zero Indic awareness. Multilingual training is mandatory.',
      'This establishes the floor; even a perfect single-language tokenizer fails multilingually.',
    ],
  },
];

// Ordered by score descending for leaderboard
export const LEADERBOARD = [...EXPERIMENTS].sort((a, b) => b.score - a.score);
