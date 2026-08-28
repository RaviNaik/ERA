/* ============================================================
   SESSION 9 — The Loss Harness — DATA
   Every number here comes from the executed notebook
   (loss_harness/session9_loss_harness.ipynb) and its
   assets/results.json, run on an NVIDIA RTX A6000.
   ============================================================ */

window.SESSION_DATA = {

  meta: {
    session: 9,
    title: "The Loss Harness",
    subtitle: "Making next-token cross-entropy correct, observable — then adding a t+2 head",
    month: "August 2026",
  },

  setup: {
    hardware: "NVIDIA RTX A6000 · 47.5 GiB · CUDA 12.4 · PyTorch 2.6.0",
    dataset: "WikiText-103-raw",
    docs: 1153246,
    tokens: 115689651,
    tokenizer: "GPT-2 BPE (tiktoken)",
    vocab: 50257,
    dModel: 512,
    nLayers: 8,
    nHeads: 8,
    dFF: 1408,
    maxSeqLen: 512,
    norm: "RMSNorm, pre-norm residual stream",
    ffn: "SwiGLU — down( silu(gate(x)) ⊙ up(x) )",
    trunkParams: 51692544,
    tokEmbParams: 25731584,
    seqLen: 512,
    batch: 12,
    tokensPerStep: 6144,
    steps: 4000,
    optimizer: "AdamW, lr 3e-4",
    baselineWallSec: 683,
    mtpWallSec: 958,
  },

  heroStats: [
    { val: "7 / 7", lbl: "Part-1 checks passing", cls: "indigo" },
    { val: "1.00%", lbl: "Untrained ppl gap from ln V", cls: "" },
    { val: "3.3×", lbl: "Memory: ordinary ÷ chunked CE", cls: "amber" },
    { val: "+1.27", lbl: "Final t+2 − t+1 loss gap (nats)", cls: "rose" },
  ],

  /* ---- Part 1: the shift check (decoded strings, not ids) ---- */
  shiftRows: [
    { pos: 0, in: "Sen", tgt: "j" },
    { pos: 1, in: "j", tgt: "ō" },
    { pos: 2, in: "ō", tgt: " no" },
    { pos: 3, in: " no", tgt: " V" },
    { pos: 4, in: " V", tgt: "alky" },
    { pos: 5, in: "alky", tgt: "ria" },
    { pos: 6, in: "ria", tgt: " 3" },
    { pos: 7, in: " 3", tgt: " :" },
    { pos: 8, in: " :", tgt: " Un" },
    { pos: 9, in: " Un", tgt: "recorded" },
    { pos: 10, in: "recorded", tgt: " Chronicles" },
    { pos: 11, in: " Chronicles", tgt: " (" },
  ],

  /* ---- Part 1: the seven numbers ---- */
  sevenNumbers: [
    {
      n: 1, tag: "shapes", title: "Every tensor shape, every dimension named",
      headline: "tokens [4, 32] → hidden [4, 32, 512] → logits [4, 32, 50257]",
      detail: "[batch, position] gains the hidden width D, then one raw score per vocabulary token V. The logits tensor is 98× larger than the hidden state that produced it — the reason Part 1 ends with a memory measurement.",
      verdict: "named",
    },
    {
      n: 2, tag: "shift", title: "Verify the shift — print strings, not ids",
      headline: "0 / 12 mismatches — target[i] == input[i+1]",
      detail: "Read as decoded sub-word strings. This is the check that catches a target shifted the wrong way — the bug that produces a beautiful loss curve while the model just learns to copy its input.",
      verdict: "correct",
    },
    {
      n: 3, tag: "padding", title: "Mask padding — watch the contributing count move",
      headline: "234 → 167 contributing tokens · loss 10.6834 → 10.8603",
      detail: "The loss went UP after masking. Padding tokens are trivially predictable, so counting them was flattering the mean. The contributing-token count changing from 234 to 167 is the proof the mask took effect.",
      verdict: "−67 tokens",
    },
    {
      n: 4, tag: "boundary", title: "Pack two documents, mask the join",
      headline: "loss 10.7940 → 10.7902 over 47 → 46 positions",
      detail: "The last token of doc A ( ' where' ) has no relationship to the first token of doc B ( 'As' ). Its own loss is 10.9666 — above the mean. On an untrained model the delta from masking is tiny; once trained, that unmasked pair is scored as a permanent failure and drags both the loss and the gradient toward a relationship that doesn't exist.",
      verdict: "join isolated",
    },
    {
      n: 5, tag: "perplexity", title: "Perplexity sanity anchor",
      headline: "untrained ppl 55,984 vs V 50,257 — gap 1.00%",
      detail: "Untrained loss 10.9328 nats vs ln V = 10.8249. A freshly-initialised head scores the vocabulary near-uniformly. If this were far from ln V, the target alignment is broken and nothing downstream can be trusted.",
      verdict: "PASS",
    },
    {
      n: 6, tag: "tying", title: "Tied vs. untied head parameter count",
      headline: "51,692,544 vs 77,424,128 params — ratio 1.498×",
      detail: "Untying gives the head its own [V, D] matrix: 50257 × 512 = 25,731,584 params — the single largest block in the model, as big as the whole token-embedding table.",
      verdict: "+25.7M",
    },
    {
      n: 7, tag: "memory", title: "Peak memory — ordinary vs. hand-written chunked CE",
      headline: "5.99 GiB vs 1.83 GiB — ratio 3.3×",
      detail: "Same loss to 9.5e-7, same gradients to 2.6e-8. Ordinary materialises the full [16384, 50257] logits tensor and its gradient; chunked (size 1,024) never holds more than one slice and back-propagates each chunk before computing the next.",
      verdict: "3.3× less",
    },
  ],

  memory: {
    nTokens: 16384, chunk: 1024, dtype: "bfloat16",
    theoreticalFull: "1.53 GiB", theoreticalChunk: "98 MiB",
    ordinaryGiB: 5.99, chunkedGiB: 1.83, ratio: 3.3,
    lossDiff: "9.5e-7", gradDiff: "2.6e-8",
  },

  /* ---- Part 2: MTP two heads ---- */
  mtp: {
    head1: { predicts: "t+1", lossStart: 10.8917, lossEnd: 4.7166, pplStart: 53730, pplEnd: 111.8 },
    head2: { predicts: "t+2", lossStart: 10.9536, lossEnd: 5.9851, pplStart: 57157, pplEnd: 397.5 },
    sum:   { lossStart: 21.8453, lossEnd: 10.7017 },
    baseline: { lossStart: 10.9386, lossEnd: 4.6513, pplEnd: 104.7 },
    gapStart: 0.0618, gapEnd: 1.2685, gapMean: 1.0645, gapWarmup: 1.1254, gapMin: 0.7736, gapMax: 1.3420,
    fracL2Above: 1.0,
  },

  /* downsampled training curves (every 25 steps) straight from the notebook logs */
  steps: [0,25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500,525,550,575,600,625,650,675,700,725,750,775,800,825,850,875,900,925,950,975,1000,1025,1050,1075,1100,1125,1150,1175,1200,1225,1250,1275,1300,1325,1350,1375,1400,1425,1450,1475,1500,1525,1550,1575,1600,1625,1650,1675,1700,1725,1750,1775,1800,1825,1850,1875,1900,1925,1950,1975,2000,2025,2050,2075,2100,2125,2150,2175,2200,2225,2250,2275,2300,2325,2350,2375,2400,2425,2450,2475,2500,2525,2550,2575,2600,2625,2650,2675,2700,2725,2750,2775,2800,2825,2850,2875,2900,2925,2950,2975,3000,3025,3050,3075,3100,3125,3150,3175,3200,3225,3250,3275,3300,3325,3350,3375,3400,3425,3450,3475,3500,3525,3550,3575,3600,3625,3650,3675,3700,3725,3750,3775,3800,3825,3850,3875,3900,3925,3950,3975,3999],
  curveH1: [10.892,7.826,7.607,7.239,7.058,6.923,6.825,6.576,6.625,6.604,6.473,6.445,6.34,6.38,6.333,6.168,6.25,6.231,6.276,6.242,6.021,6.167,6.161,6.024,6.09,6.112,5.87,5.83,5.768,5.811,5.802,5.83,5.885,5.995,5.732,5.661,5.647,5.717,5.647,5.709,5.558,5.521,5.646,5.487,5.476,5.464,5.405,5.643,5.607,5.456,5.503,5.46,5.433,5.332,5.206,5.326,5.286,5.234,5.348,5.367,5.22,5.337,5.199,5.302,5.149,5.084,5.249,5.036,5.197,4.989,5.202,5.301,5.188,4.987,4.925,5.065,5.125,5.053,5.093,4.916,5.156,5.013,5.032,4.942,5.127,5.025,4.955,4.976,5.107,4.998,4.778,4.836,4.912,4.621,4.856,5.028,4.775,4.8,4.926,4.919,4.928,4.941,4.917,4.981,4.84,4.944,4.761,4.912,4.619,4.819,4.935,4.782,4.886,4.742,4.715,4.662,4.746,4.706,4.712,4.709,4.565,4.727,4.854,4.698,4.803,4.631,4.756,4.641,4.688,4.675,4.582,4.706,4.657,4.721,4.731,4.731,4.497,4.788,4.456,4.585,4.649,4.756,4.556,4.576,4.615,4.732,4.609,4.595,4.733,4.526,4.614,4.585,4.74,4.575,4.51,4.634,4.681,4.61,4.423,4.668,4.717],
  curveH2: [10.954,7.864,7.613,7.444,7.425,7.364,7.313,7.146,7.208,7.253,7.144,7.171,7.054,7.131,7.111,6.971,7.067,7.052,7.109,7.074,6.895,7.009,6.998,6.924,7.004,6.97,6.769,6.734,6.699,6.737,6.75,6.776,6.799,6.916,6.699,6.608,6.635,6.669,6.631,6.678,6.558,6.515,6.607,6.451,6.461,6.465,6.427,6.672,6.664,6.477,6.529,6.524,6.494,6.397,6.275,6.377,6.322,6.306,6.448,6.414,6.301,6.434,6.271,6.416,6.262,6.234,6.349,6.153,6.285,6.088,6.347,6.38,6.292,6.107,6.065,6.22,6.252,6.19,6.252,6.105,6.302,6.164,6.14,6.085,6.26,6.187,6.103,6.075,6.277,6.185,5.963,6.022,6.092,5.765,6.028,6.195,5.94,6.018,6.105,6.171,6.114,6.135,6.096,6.176,6.008,6.159,6.019,6.118,5.857,5.975,6.139,6.028,6.11,5.931,5.932,5.87,5.952,5.95,5.941,5.927,5.794,5.969,6.053,5.921,5.994,5.858,5.998,5.886,5.907,5.888,5.818,5.995,5.896,5.924,5.986,5.966,5.719,5.971,5.675,5.826,5.89,6.035,5.75,5.79,5.833,5.963,5.843,5.853,5.964,5.868,5.877,5.851,6.017,5.835,5.798,5.911,5.924,5.875,5.658,5.922,5.985],
  curveSum: [21.845,15.689,15.221,14.684,14.483,14.287,14.138,13.722,13.833,13.857,13.617,13.617,13.395,13.511,13.444,13.139,13.316,13.283,13.384,13.317,12.916,13.176,13.159,12.948,13.093,13.082,12.639,12.563,12.467,12.548,12.551,12.606,12.684,12.911,12.431,12.269,12.282,12.386,12.278,12.386,12.117,12.035,12.253,11.938,11.937,11.929,11.832,12.315,12.271,11.933,12.032,11.984,11.928,11.729,11.482,11.703,11.607,11.54,11.796,11.781,11.521,11.772,11.47,11.718,11.411,11.318,11.598,11.189,11.482,11.076,11.549,11.681,11.48,11.095,10.99,11.285,11.377,11.243,11.345,11.022,11.458,11.177,11.172,11.027,11.386,11.212,11.058,11.05,11.383,11.183,10.741,10.858,11.005,10.386,10.885,11.223,10.714,10.819,11.031,11.09,11.042,11.076,11.013,11.157,10.848,11.102,10.78,11.03,10.476,10.795,11.075,10.809,10.995,10.673,10.647,10.532,10.697,10.656,10.653,10.636,10.359,10.696,10.907,10.619,10.797,10.489,10.754,10.527,10.595,10.563,10.4,10.701,10.553,10.645,10.717,10.697,10.216,10.759,10.131,10.412,10.539,10.791,10.306,10.367,10.449,10.694,10.452,10.448,10.697,10.394,10.491,10.436,10.756,10.409,10.308,10.545,10.605,10.484,10.081,10.59,10.702],
  curveBase: [10.939,7.76,7.531,7.099,7.013,6.918,6.832,6.587,6.623,6.598,6.475,6.442,6.345,6.402,6.335,6.176,6.254,6.242,6.272,6.256,6.027,6.17,6.15,6.0,6.077,6.098,5.851,5.818,5.759,5.792,5.788,5.819,5.875,5.975,5.728,5.636,5.627,5.697,5.625,5.671,5.542,5.5,5.598,5.465,5.44,5.431,5.368,5.606,5.56,5.432,5.465,5.432,5.396,5.282,5.172,5.286,5.274,5.201,5.287,5.328,5.19,5.302,5.135,5.259,5.121,5.043,5.214,4.994,5.15,4.961,5.153,5.263,5.162,4.959,4.88,5.019,5.072,5.025,5.035,4.867,5.102,4.972,4.974,4.896,5.085,4.989,4.918,4.937,5.065,4.965,4.741,4.808,4.862,4.603,4.817,4.996,4.717,4.744,4.873,4.879,4.892,4.907,4.881,4.949,4.786,4.899,4.725,4.878,4.568,4.75,4.881,4.744,4.825,4.694,4.693,4.612,4.691,4.657,4.659,4.654,4.511,4.672,4.807,4.654,4.761,4.59,4.704,4.607,4.616,4.623,4.527,4.653,4.602,4.657,4.68,4.68,4.436,4.739,4.39,4.511,4.575,4.685,4.488,4.522,4.555,4.669,4.534,4.548,4.647,4.468,4.541,4.528,4.674,4.493,4.438,4.582,4.621,4.529,4.354,4.592,4.651],

  figures: [
    { src: "../loss_harness/assets/mtp_losses.png", cap: "t+1 vs t+2 head losses over 4,000 steps (matplotlib, full per-step)." },
    { src: "../loss_harness/assets/baseline_loss.png", cap: "Baseline single tied t+1 head — the reference curve." },
    { src: "../loss_harness/assets/mtp_loss_head1_aim.png", cap: "Aim · Head 1 (t+1) loss" },
    { src: "../loss_harness/assets/mtp_loss_head2_aim.png", cap: "Aim · Head 2 (t+2) loss" },
    { src: "../loss_harness/assets/mtp_loss_sum_aim.png", cap: "Aim · L1 + L2 (the optimised objective)" },
    { src: "../loss_harness/assets/perplexity_baseline_aim.png", cap: "Aim · baseline perplexity — 56,307 → ~105" },
  ],

  commentary: [
    {
      h: "The gap is the finding",
      p: "At step 0 both heads are random, so L2 − L1 ≈ 0 — noise. As training shapes the shared trunk the gap opens to a stable ~1.13 nats and never closes. Predicting t+2 from the same hidden state carries one extra step of genuine, irreducible uncertainty about what the text does next. Its entropy floor is simply higher.",
    },
    {
      h: "In perplexity terms",
      p: "By the end the model is effectively choosing among ~112 options for the very next token, but ~400 for the token after that. Same context, same representation — the further-out prediction is a measurably harder question.",
    },
    {
      h: "Both curves fall together, not apart",
      p: "They share one trunk, so anything that improves the hidden state helps both objectives at once. The t+2 head is extra supervision on the same representation, not a competing task fighting for capacity.",
    },
    {
      h: "The second head is nearly free on the primary objective",
      p: "The standalone baseline (single tied t+1 head) ends at 4.6513; the t+1 head inside the two-head model ends at 4.7166 — within noise. Adding a t+2 head did not measurably hurt next-token quality at this scale and step budget.",
    },
    {
      h: "The sum is what's optimised",
      p: "L_total = L1 + L2 — added, not averaged — so Head 2 pulls on the shared trunk exactly as hard as Head 1. To let t+1 dominate you would weight it (L1 + λ·L2); here λ = 1.",
    },
    {
      h: "The honest cost of MTP",
      p: "Each extra head is another V × D matrix — 25.7M params in this config, as large as the embedding table. Four dense heads would more than double the model. This is the argument for a factored output head.",
    },
  ],
};
