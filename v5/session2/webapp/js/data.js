// js/data.js — All experiment constants and metadata
// Assets live at ../bpe_assignment/ relative to webapp/index.html,
// which resolves to /bpe_assignment/ on the Netlify-served session2 root.

export const ASSET_BASE = '../bpe_assignment';

export const LANG_META = {
  en: { name: 'English',  title: 'India',       flag: '🇬🇧', chars: 64971, words: 10027, avgLen: 5.48, color: '#6cb8e0', light: '#d4eef8', soft: '#a8d8f0', file: 'india_en.txt' },
  hi: { name: 'Hindi',   title: 'भारत',          flag: '🇮🇳', chars: 43596, words: 8022,  avgLen: 4.43, color: '#9d8fe0', light: '#e4dff8', soft: '#c5bdf0', file: 'india_hi.txt' },
  te: { name: 'Telugu',  title: 'భారతదేశం',      flag: '🏛️', chars: 19822, words: 2453,  avgLen: 6.58, color: '#5ebc8e', light: '#d0f0e0', soft: '#9dd9be', file: 'india_te.txt' },
  kn: { name: 'Kannada', title: 'ಭಾರತ',          flag: '🌟', chars: 7600,  words: 979,   avgLen: 6.43, color: '#e08a5e', light: '#fce4d4', soft: '#f0b898', file: 'india_kn.txt' },
};

export const EXPERIMENTS = [
  {
    id: 'step1',
    step: 'Step 1',
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
    rank: 4,
    insight: 'Indic scripts have no coverage — each word explodes into individual byte tokens. Telugu reaches 6.95 tokens/word.',
  },
  {
    id: 'step2',
    step: 'Step 2',
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
    rank: 2,
    insight: 'Surprisingly competitive — joint training naturally distributes vocab. Hindi benefits most (8K words, large corpus). Telugu/Kannada lag due to less training data.',
  },
  {
    id: 'step3a',
    step: 'Step 3A ★',
    name: 'Optimized — Oversampling (2×)',
    desc: 'NFKC normalization + ZWJ/ZWNJ removal + Indic languages repeated ×2 to perfectly balance vocab allocation. This is the optimal mathematical sweet-spot.',
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
    rank: 1,
    insight: 'A sweep of oversampling factors revealed that exactly 2× is the mathematical sweet spot! It perfectly balances English token efficiency with Indic scripts, compressing the spread to just 0.41.',
  },
  {
    id: 'step3b',
    step: 'Step 3B',
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
    rank: 3,
    insight: 'Guaranteed equal allocation per language. Kannada only trained 768/2500 tokens due to small corpus (979 words) — data bottleneck limits gains. Beaten by the mathematically perfect oversampling factor of Step 3A.',
  },
  {
    id: 'step3c',
    step: 'Step 3C',
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
    rank: 5,
    insight: 'Indic characters take 3 bytes in UTF-8. A 5-letter Telugu word starts as 15 byte-tokens. The small corpus lacks the frequency data to learn how to re-assemble them, leaving the text highly fragmented.',
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
