// js/data.js — All experiment constants and metadata
// All experiments now use the faithful Markdown corpus (HTML->MD conversion)
// and Metaspace pre-tokenizer/decoder (satisfies evaluator faithfulness check).
// Fertility = tokens / faithful_units  (NOT tokens / whitespace words)

export const ASSET_BASE = './assets';

// Corpus sizes from faithful Markdown (HTML -> Markdown, links/URLs preserved)
// These are much larger than the plain-text API extract
export const LANG_META = {
  en: { name: 'English',  title: 'India',     chars: 601843, faithfulUnits: 186367, color: '#6cb8e0', light: '#d4eef8', soft: '#a8d8f0', file: 'india_en.txt' },
  hi: { name: 'Hindi',   title: '\u092d\u093e\u0930\u0924',        chars: 463156, faithfulUnits: 88359,  color: '#9d8fe0', light: '#e4dff8', soft: '#c5bdf0', file: 'india_hi.txt' },
  te: { name: 'Telugu',  title: '\u0c2d\u0c3e\u0c30\u0c24\u0c26\u0c47\u0c36\u0c02',    chars: 199575, faithfulUnits: 36292,  color: '#5ebc8e', light: '#d0f0e0', soft: '#9dd9be', file: 'india_te.txt' },
  kn: { name: 'Kannada', title: '\u0cad\u0cbe\u0cb0\u0ca4',        chars: 64057,  faithfulUnits: 12293,  color: '#e08a5e', light: '#fce4d4', soft: '#f0b898', file: 'india_kn.txt' },
};

// Helper: compute score from results object
function score(results) {
  const ratios = Object.values(results).map(r => r.ratio);
  const spread = Math.max(...ratios) - Math.min(...ratios);
  return 1000 / spread;
}

export const EXPERIMENTS = [
  // ── SUBMITTED TOKENIZER (Best Model) ─────────────────────────────────────
  {
    id: 'faithful',
    step: 'Submitted \u2605\u2605\u2605',
    name: 'Faithful Markdown BPE — en\u00d73 hi\u00d74 te\u00d74 kn\u00d76',
    desc: 'Final submitted tokenizer. Trained on HTML\u2192Markdown faithful corpus. ' +
          'Weights en\u00d73\u00b7hi\u00d74\u00b7te\u00d74\u00b7kn\u00d76 were chosen to balance all four languages under 0.8. ' +
          'Passes decode(encode(text)) faithfulness check \u2014 URLs, punctuation and brackets are perfectly preserved.',
    accent: '#1db954', accentLight: '#d0f8e4',
    config: {
      'Target Vocab':  '10,000',
      'Actual Vocab':  '10,000',
      'Pre-tokenizer': 'Metaspace (\u25b1)',
      'Normalization': 'NFKC',
      'Training Data': 'en\u00d73 \u00b7 hi\u00d74 \u00b7 te\u00d74 \u00b7 kn\u00d76 (faithful Markdown)',
      'Min Frequency': '1',
      'Faithfulness':  'PASS \u2014 decode(encode(x)) == x',
    },
    results: {
      en: { faithfulUnits: 186367, tokens: 115768, ratio: 0.6212 },
      hi: { faithfulUnits: 88359,  tokens: 53036,  ratio: 0.6002 },
      te: { faithfulUnits: 36292,  tokens: 24793,  ratio: 0.6832 },
      kn: { faithfulUnits: 12293,  tokens: 9327,   ratio: 0.7587 },
    },
    xMin: 0.6002, xMax: 0.7587, spread: 0.1585, score: 6309.49,
    modelFile: 'tokenizer.json',
    rank: 4,
    insight: 'Submitted tokenizer. Score 6309 with all ratios < 0.8. ' +
             'The faithful Markdown corpus (5\u20138\u00d7 larger than plain-text) + Metaspace decoder are the key fixes from the original rejected submission.',
    findings: [
      'All four fertility ratios are between 0.60 and 0.76 \u2014 well under the 1.2 threshold.',
      'English 0.621 and Hindi 0.600 are nearly identical, showing excellent cross-lingual balance.',
      'Kannada 0.759 benefits from kn\u00d76 oversampling without reaching word-memorisation.',
      'Hindi penalty factor = 1.0 (no penalty): ratio 0.600 is far below the 1.2 threshold.',
      'decode(encode(text)) passes for all sample strings including URLs with #, /, (, ), [, ].',
    ],
    conclusions: [
      'Faithful Markdown corpus + Metaspace tokenizer satisfies the evaluator faithfulness requirement.',
      'This tokenizer is the valid, reproducible submission for the assignment.',
    ],
  },

  // ── Experiment 3A2: Focused sweet-spot ────────────────────────────────────
  {
    id: 'step3a2',
    step: 'Experiment 3A2 \u2605\u2605',
    name: 'Focused Sweet-Spot \u2014 en\u00d71 hi\u00d71 te\u00d72 kn\u00d74',
    desc: 'Best scoring experiment. Hypothesis: en and hi have enough data at \u00d71; ' +
          'only te and kn need boosting. Result: all four ratios compress into a 0.03 spread.',
    accent: '#2b9e6a', accentLight: '#d0f5e4',
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
    modelFile: 'exp3a2_focused.json',
    rank: 1,
    insight: 'Extraordinary score of 33,207 \u2014 all four languages cluster within a 0.03 band (0.62\u20130.65). ' +
             'en\u00d71 and hi\u00d71 are sufficient as they have large corpora; te\u00d72 and kn\u00d74 correct the smaller-corpus languages.',
    findings: [
      'All four ratios lie within 0.6244\u20130.6545 \u2014 a spread of just 0.0301.',
      'English (186K units) and Hindi (88K units) need no boosting; their large corpora self-regulate.',
      'Telugu \u00d72 and Kannada \u00d74 bring smaller languages into the same band without over-fitting.',
      'Score 33,207 is 5.3\u00d7 better than the submitted tokenizer (6,309).',
    ],
    conclusions: [
      'Targeted per-language oversampling is far superior to uniform oversampling.',
      'This configuration achieves near-optimal ratio compression on the faithful Markdown corpus.',
    ],
  },

  // ── Experiment 3B: Merged vocabulary ─────────────────────────────────────
  {
    id: 'step3b',
    step: 'Experiment 3B',
    name: 'Merged Vocabulary \u2014 4 independent tokenizers',
    desc: 'Four independent BPE tokenizers (2,500 vocab each) trained per language, then merged. ' +
          'Guarantees vocabulary space allocation but suffers from merge compatibility issues.',
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
    insight: 'Merged vocab achieves score 13,877 with ratios in 0.74\u20130.82 range. ' +
             'Each language gets guaranteed allocation, but cross-language merge conflicts produce slightly higher ratios than the joint-training experiments.',
    findings: [
      'All four ratios are tightly clustered between 0.74 and 0.82.',
      'Hindi trained only 1,871/2,500 budget tokens \u2014 enough data to fill nearly the full budget.',
      'Telugu and Kannada each trained ~1,800 tokens despite smaller corpora.',
      'Score 13,877 is 2.2\u00d7 better than the submitted tokenizer.',
    ],
    conclusions: [
      'Merged vocabulary is a robust strategy but slightly sub-optimal vs joint training with targeted oversampling.',
      'Score 13,877 validates the approach; further gains require inter-language merge alignment.',
    ],
  },

  // ── Experiment 3A1: Differential oversampling ─────────────────────────────
  {
    id: 'step3a1',
    step: 'Experiment 3A1',
    name: 'Differential Oversampling \u2014 en\u00d71 hi\u00d71 te\u00d73 kn\u00d76',
    desc: 'Aggressive per-language oversampling proportional to inverse corpus size. ' +
          'Dramatically boosts small languages but still leaves a 0.138 spread due to English having too much data.',
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
    insight: 'Score 7,247. Kannada drops to 0.548 and Telugu to 0.604, but this creates a new spread ' +
             'because English (0.653) and Hindi (0.686) are now the outliers. ' +
             'The fix in 3A2 was to also reduce te and kn slightly, pulling everything into a tighter band.',
    findings: [
      'Telugu (0.604) and Kannada (0.548) benefit most from aggressive oversampling.',
      'But English (0.653) and Hindi (0.686) become the new spread drivers.',
      'Spread 0.138 is larger than 3A2 (0.030) because the large languages are not compressed enough.',
    ],
    conclusions: [
      'Aggressive small-language oversampling alone is insufficient \u2014 must also moderate large-language ratios.',
      'The insight that led to 3A2: reduce te to \u00d72 (instead of \u00d73) to bring it back up toward en/hi.',
    ],
  },

  // ── Experiment 3A: Uniform oversampling ──────────────────────────────────
  {
    id: 'step3a',
    step: 'Experiment 3A',
    name: 'Uniform Oversampling \u2014 en\u00d71 hi\u00d72 te\u00d72 kn\u00d72',
    desc: 'All Indic languages oversampled \u00d72. First major improvement over naive equal-weight training.',
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
    rank: 5,
    insight: 'Score 4,438. Uniform \u00d72 boosts Hindi too aggressively (0.573 \u2014 lowest), ' +
             'while Kannada at 0.798 is still the highest. ' +
             'The spread is driven by the Hindi\u2013Kannada gap.',
    findings: [
      'Hindi benefits most from \u00d72: ratio drops from 0.616 (Exp2) to 0.573.',
      'Kannada (smallest corpus) still lags at 0.798 despite \u00d72 boost.',
      'English rises slightly (0.600 \u2192 0.649) because vocab allocation shifts to Indic scripts.',
      'Uniform oversampling is better than naive but leaves a 0.225 spread.',
    ],
    conclusions: [
      'Uniform oversampling is a good starting point but needs per-language tuning.',
      'Hindi\u00d72 is unnecessary \u2014 hi has enough data at \u00d71 (discovered in 3A1/3A2).',
    ],
  },

  // ── Experiment 2: Naive multilingual ─────────────────────────────────────
  {
    id: 'step2',
    step: 'Experiment 2',
    name: 'Naive Multilingual \u2014 equal weights',
    desc: 'All four languages concatenated equally (1\u00d7 each). No oversampling, no special tuning. ' +
          'Surprising baseline: score 3,788 already competitive because the faithful corpus is well-balanced.',
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
    rank: 6,
    insight: 'Even with equal weights, the faithful Markdown corpus produces good ratios (0.60\u20130.86). ' +
             'The 0.264 spread is driven by Kannada (smallest corpus, highest ratio 0.864). ' +
             'Oversampling Kannada is the clear next step.',
    findings: [
      'English (0.600) and Hindi (0.616) are nearly balanced \u2014 their large corpora self-regulate.',
      'Telugu (0.704) lags because its corpus is 5\u00d7 smaller than English.',
      'Kannada (0.864) is the spread driver \u2014 only 12K faithful units vs English 186K.',
    ],
    conclusions: [
      'Naive equal-weight training on faithful corpus already achieves score 3,788 \u2014 a strong baseline.',
      'The bottleneck is Kannada\u2019s small corpus; per-language oversampling is the fix.',
    ],
  },

  // ── Experiment 1: English-only ────────────────────────────────────────────
  {
    id: 'step1',
    step: 'Experiment 1',
    name: 'English-Only BPE \u2014 Baseline',
    desc: 'Vanilla BPE trained on English Wikipedia only. Establishes that an English-only vocabulary ' +
          'has no Indic coverage \u2014 all three Indic languages fragment badly.',
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
    rank: 7,
    insight: 'English-only vocabulary fragments all Indic text. Hindi, Telugu, and Kannada all cluster near 1.73 \u2014 ' +
             'each Indic word becomes ~3.4 tokens on average (because Devanagari/Telugu/Kannada characters ' +
             'have no BPE merges and fall back to individual Unicode code points).',
    findings: [
      'English ratio 0.511 is excellent (the tokenizer was trained on English).',
      'All three Indic languages cluster near 1.73 \u2014 character-level fragmentation.',
      'Score 778 is low because the spread (1.284) is dominated by the English vs Indic gap.',
      'This establishes the floor: even a perfect single-language tokenizer fails multilingually.',
    ],
    conclusions: [
      'English-only BPE has zero Indic awareness. Multilingual training is necessary.',
      'Baseline score 778 vs submitted 6,309 = 8\u00d7 improvement from adding all four languages.',
    ],
  },
];

// Ordered by score descending for leaderboard
export const LEADERBOARD = [...EXPERIMENTS].sort((a, b) => b.score - a.score);
