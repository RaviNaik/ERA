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
    baselineWallSec: 684,
    mtpWallSec: 958,
  },

  heroStats: [
    { val: "7 / 7", lbl: "Part-1 checks passing", cls: "indigo" },
    { val: "1.0%", lbl: "Untrained loss gap from ln V (nats)", cls: "" },
    { val: "4.7×", lbl: "Memory: ordinary ÷ chunked CE", cls: "amber" },
    { val: "+1.26", lbl: "Settled t+2 − t+1 loss gap (nats)", cls: "rose" },
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
      headline: "read the table by eye · 0/12 target-slice mismatches",
      detail: "The printed inputs/targets table is the deliverable — a human reads it and confirms each target is the next word. The 0/12 counter only proves targets_out is the right slice (true by construction); that logits[:, :-1] is aligned to it in the loss call is a functional fact, checked by the perplexity anchor (check 5).",
      verdict: "slice ok",
    },
    {
      n: 3, tag: "padding", title: "Mask padding — watch the contributing count move",
      headline: "234 → 167 contributing tokens · loss 10.6834 → 10.8603",
      detail: "The loss went UP after masking. Padding tokens are trivially predictable, so counting them was flattering the mean. The contributing-token count changing from 234 to 167 is the proof the mask took effect.",
      verdict: "−67 tokens",
    },
    {
      n: 4, tag: "boundary", title: "Pack two documents, mask the join",
      headline: "trained head: masking one position drops mean loss 0.178 nats",
      detail: "The last token of doc A ( ' where' ) has no relationship to the first token of doc B ( 'As' ). Untrained: its loss (10.9666) is right at the mean (10.7940 → 10.7902 masked, a noise-level delta). Trained (§6.2 cell, after Part 2): the model is good at real continuations (mean 5.65), so that position scores 13.84 — 2.53× the rest of the sequence — and masking it alone lowers the whole-sequence mean loss to 5.47. Note: the packer used for training does insert an <|endoftext|> at joins, it just adds no loss mask there.",
      verdict: "2.53× the mean",
    },
    {
      n: 5, tag: "perplexity", title: "Perplexity anchor — read it in nats",
      headline: "untrained loss 10.9328 vs ln V 10.8249 — 1.0% (nats)",
      detail: "A freshly-initialised head scores the vocabulary near-uniformly: the loss is ~1% from ln V in nats. That 1% becomes ~11% at the perplexity level (55,984 vs V = 50,257), which is why 'sits near the vocabulary size' is doing a little work — the nats gap is the honest number. If it were far off, the target alignment is broken and nothing downstream can be trusted.",
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
      headline: "10.6 GiB vs 2.25 GiB — ratio 4.7× (N = 32,768)",
      detail: "Same loss to 9.5e-7, same gradients to 2.3e-9 / 2.6e-8. The naive [N, V] figure (3.07 GiB bf16) is NOT the cost: ordinary holds logits + a float32 log-softmax up-cast + the gradient (~3×, predicted 12.4 GiB / measured 10.6), and chunked can't reach its one-chunk ideal (predicted 0.9 GiB) because a ~1.3 GiB CUDA context and the N-independent hidden/weight tensors don't go away. The asymmetry: ordinary grows as 3·N·V, every chunked term is flat in N.",
      verdict: "4.7× less",
    },
  ],

  memory: {
    nTokens: 32768, chunk: 2048, dtype: "bfloat16", nPasses: 16,
    theoreticalFull: "3.07 GiB",
    ordinaryGiB: 10.61, chunkedGiB: 2.25, ratio: 4.7,
    predictedOrdinaryGiB: 12.43, predictedChunkedGiB: 0.93,
    reservedOrdinaryGiB: 10.68, reservedChunkedGiB: 2.44,
    lossDiff: "9.5e-7", gradDiff: "2.3e-9 / 2.6e-8",
    breakdown: [
      { path: "ordinary", item: "logits [N,V] x2 (bf16)", gib: 3.07, note: "materialised by the matmul" },
      { path: "ordinary", item: "log-softmax [N,V] x4 (float32)", gib: 6.13, note: "F.cross_entropy up-casts — the biggest hidden term" },
      { path: "ordinary", item: "grad of logits [N,V] x2 (bf16)", gib: 3.07, note: "the backward pass" },
      { path: "ordinary", item: "hidden + weight + grads", gib: 0.16, note: "small next to [N,V]" },
      { path: "chunked", item: "one chunk: logits + fp32 log-softmax + grad", gib: 0.77, note: "transient, freed each of 16 passes" },
      { path: "chunked", item: "hidden [N,D] + weight [V,D] + grads", gib: 0.16, note: "does NOT shrink with chunk" },
      { path: "chunked", item: "CUDA context + cuBLAS workspace", gib: 1.3, note: "fixed cost of touching the GPU" },
    ],
  },

  /* ---- Part 2: MTP two heads (RTX A6000, 4,000 steps) ---- */
  mtp: {
    head1: { predicts: "t+1", lossStart: 10.8917, lossEnd: 4.7250, pplStart: 53730 },
    head2: { predicts: "t+2", lossStart: 10.9536, lossEnd: 5.9942, pplStart: 57157 },
    sum:   { lossStart: 21.8453, lossEnd: 10.7192 },
    baseline: { lossStart: 10.9386, lossEnd: 4.6533 },
    gapStart: 0.0618, gapEnd: 1.2692, gapMean: 1.0644, gapWarmup: 1.1253,
    fracL2Above: 1.0,
    // trailing means over the last 400 steps; per-step noise sd 0.109, SE of the mean 0.0055
    head1Settled: 4.602, head2Settled: 5.865, sumSettled: 10.467, baselineSettled: 4.555,
    gapSettled: 1.263, gapSettledSE: 231, head1MinusBaseSettled: 0.047, head1MinusBaseSE: 8.6,
    perStepNoise: 0.109, semNats: 0.0055,
  },

  /* ---- Part 1 check 4: boundary re-checked on the TRAINED t+1 head (§6.2) ---- */
  boundaryTrained: {
    untrainedMean: 10.7940, untrainedMasked: 10.7902, untrainedBoundary: 10.9666, untrainedRatio: 1.02,
    trainedMean: 5.6512, trainedMasked: 5.4732, trainedBoundary: 13.8385, trainedRatio: 2.53,
    trainedDelta: -0.1780,
  },

  /* downsampled training curves (every 25 steps) straight from the notebook logs */
  steps: [0,25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500,525,550,575,600,625,650,675,700,725,750,775,800,825,850,875,900,925,950,975,1000,1025,1050,1075,1100,1125,1150,1175,1200,1225,1250,1275,1300,1325,1350,1375,1400,1425,1450,1475,1500,1525,1550,1575,1600,1625,1650,1675,1700,1725,1750,1775,1800,1825,1850,1875,1900,1925,1950,1975,2000,2025,2050,2075,2100,2125,2150,2175,2200,2225,2250,2275,2300,2325,2350,2375,2400,2425,2450,2475,2500,2525,2550,2575,2600,2625,2650,2675,2700,2725,2750,2775,2800,2825,2850,2875,2900,2925,2950,2975,3000,3025,3050,3075,3100,3125,3150,3175,3200,3225,3250,3275,3300,3325,3350,3375,3400,3425,3450,3475,3500,3525,3550,3575,3600,3625,3650,3675,3700,3725,3750,3775,3800,3825,3850,3875,3900,3925,3950,3975,3999],
  curveH1: [10.892,7.826,7.607,7.239,7.058,6.923,6.825,6.576,6.625,6.604,6.473,6.445,6.34,6.38,6.333,6.168,6.248,6.231,6.279,6.244,6.023,6.169,6.162,6.023,6.09,6.11,5.873,5.829,5.776,5.811,5.808,5.834,5.884,6.002,5.743,5.665,5.653,5.712,5.645,5.714,5.565,5.511,5.631,5.495,5.484,5.474,5.399,5.651,5.601,5.467,5.515,5.465,5.446,5.325,5.199,5.332,5.296,5.242,5.349,5.366,5.221,5.343,5.2,5.304,5.154,5.089,5.25,5.039,5.192,4.996,5.202,5.299,5.193,4.988,4.932,5.064,5.12,5.05,5.091,4.92,5.157,5.01,5.029,4.945,5.125,5.027,4.957,4.965,5.101,5.002,4.776,4.841,4.908,4.616,4.847,5.022,4.772,4.793,4.922,4.936,4.931,4.939,4.908,4.983,4.834,4.939,4.755,4.92,4.612,4.823,4.942,4.777,4.879,4.737,4.717,4.657,4.742,4.718,4.713,4.697,4.549,4.711,4.858,4.698,4.804,4.642,4.757,4.644,4.688,4.673,4.585,4.714,4.659,4.715,4.735,4.721,4.502,4.792,4.458,4.575,4.643,4.756,4.556,4.573,4.623,4.731,4.6,4.595,4.725,4.516,4.625,4.588,4.742,4.579,4.51,4.638,4.669,4.609,4.419,4.672,4.725],
  curveH2: [10.954,7.864,7.613,7.444,7.425,7.364,7.313,7.146,7.208,7.253,7.144,7.171,7.054,7.131,7.111,6.971,7.065,7.054,7.112,7.075,6.896,7.01,6.996,6.923,7.011,6.975,6.775,6.735,6.704,6.739,6.752,6.778,6.803,6.921,6.707,6.611,6.643,6.676,6.637,6.683,6.563,6.512,6.601,6.459,6.47,6.47,6.424,6.673,6.655,6.494,6.536,6.525,6.498,6.395,6.276,6.38,6.335,6.303,6.45,6.413,6.305,6.437,6.282,6.416,6.266,6.237,6.352,6.156,6.293,6.093,6.339,6.375,6.3,6.107,6.075,6.221,6.256,6.195,6.258,6.106,6.3,6.157,6.147,6.089,6.268,6.188,6.105,6.072,6.273,6.184,5.968,6.027,6.088,5.768,6.028,6.189,5.941,6.008,6.103,6.173,6.118,6.136,6.095,6.182,6.011,6.158,6.021,6.123,5.857,5.98,6.149,6.023,6.104,5.932,5.936,5.859,5.947,5.955,5.94,5.925,5.785,5.96,6.064,5.919,5.994,5.867,6.003,5.886,5.911,5.882,5.817,5.998,5.899,5.918,5.981,5.956,5.726,5.974,5.676,5.813,5.891,6.03,5.754,5.792,5.84,5.972,5.839,5.852,5.96,5.859,5.89,5.842,6.021,5.842,5.803,5.913,5.922,5.876,5.663,5.926,5.994],
  curveSum: [21.845,15.689,15.221,14.684,14.483,14.287,14.138,13.722,13.833,13.857,13.617,13.617,13.395,13.511,13.444,13.139,13.313,13.285,13.39,13.319,12.919,13.179,13.158,12.946,13.101,13.085,12.648,12.564,12.48,12.55,12.56,12.612,12.687,12.923,12.45,12.276,12.296,12.388,12.283,12.397,12.128,12.023,12.233,11.954,11.954,11.943,11.824,12.324,12.256,11.962,12.051,11.99,11.944,11.72,11.475,11.712,11.632,11.545,11.799,11.779,11.527,11.78,11.482,11.72,11.42,11.326,11.602,11.195,11.485,11.09,11.541,11.674,11.493,11.095,11.007,11.285,11.377,11.245,11.349,11.026,11.457,11.167,11.177,11.034,11.393,11.214,11.062,11.037,11.374,11.186,10.744,10.867,10.995,10.384,10.875,11.211,10.713,10.8,11.025,11.108,11.05,11.074,11.004,11.165,10.845,11.098,10.776,11.043,10.469,10.803,11.091,10.8,10.983,10.669,10.654,10.516,10.689,10.672,10.653,10.622,10.334,10.671,10.922,10.617,10.798,10.509,10.759,10.53,10.599,10.555,10.402,10.712,10.558,10.633,10.716,10.677,10.227,10.766,10.134,10.388,10.533,10.786,10.31,10.365,10.463,10.703,10.439,10.447,10.684,10.375,10.515,10.43,10.763,10.421,10.313,10.551,10.591,10.485,10.082,10.599,10.719],
  curveBase: [10.939,7.76,7.531,7.099,7.013,6.918,6.832,6.587,6.623,6.598,6.475,6.442,6.345,6.402,6.335,6.176,6.254,6.242,6.272,6.256,6.028,6.167,6.154,5.999,6.082,6.096,5.852,5.822,5.761,5.797,5.779,5.815,5.869,5.975,5.732,5.632,5.63,5.699,5.625,5.682,5.545,5.499,5.601,5.469,5.438,5.445,5.378,5.611,5.562,5.432,5.471,5.428,5.393,5.285,5.173,5.291,5.269,5.202,5.296,5.327,5.19,5.296,5.138,5.266,5.121,5.045,5.215,5.001,5.149,4.97,5.16,5.259,5.161,4.954,4.884,5.027,5.084,5.032,5.041,4.874,5.107,4.958,4.98,4.901,5.09,4.985,4.916,4.941,5.063,4.966,4.737,4.803,4.866,4.605,4.811,4.991,4.71,4.749,4.873,4.869,4.886,4.888,4.87,4.946,4.785,4.905,4.715,4.867,4.569,4.749,4.874,4.744,4.832,4.697,4.685,4.61,4.694,4.673,4.662,4.662,4.503,4.675,4.797,4.647,4.76,4.584,4.71,4.6,4.621,4.63,4.528,4.654,4.597,4.656,4.668,4.685,4.43,4.732,4.389,4.506,4.57,4.686,4.493,4.511,4.548,4.678,4.531,4.655,4.693,4.492,4.551,4.542,4.68,4.511,4.433,4.587,4.633,4.529,4.361,4.612,4.653],

  figures: [
    { src: "../loss_harness/assets/mtp_losses.png", cap: "t+1 vs t+2 head losses over 4,000 steps (matplotlib, full per-step)." },
    { src: "../loss_harness/assets/baseline_loss.png", cap: "Baseline single tied t+1 head — the reference curve." },
    { src: "../loss_harness/assets/mtp_loss_head1_aim.png", cap: "Aim · Head 1 (t+1) loss" },
    { src: "../loss_harness/assets/mtp_loss_head2_aim.png", cap: "Aim · Head 2 (t+2) loss" },
    { src: "../loss_harness/assets/mtp_loss_sum_aim.png", cap: "Aim · L1 + L2 (the optimised objective)" },
    { src: "../loss_harness/assets/perplexity_aim.png", cap: "Aim · baseline perplexity — ~56,000 → ~100" },
  ],

  commentary: [
    {
      h: "The gap is the finding",
      p: "At step 0 both heads are random, so L2 − L1 ≈ 0 — noise. As training shapes the shared trunk the gap opens to a stable ~1.26 nats (trailing mean, 231 standard errors above zero) and never closes. Predicting t+2 from the same hidden state carries one extra step of genuine, irreducible uncertainty about what the text does next. Its entropy floor is simply higher.",
    },
    {
      h: "In perplexity terms",
      p: "The model settles at effectively ~100 options for the very next token, but ~350 for the token after that. Same context, same representation — the further-out prediction is a measurably harder question.",
    },
    {
      h: "Both curves fall together, not apart",
      p: "They share one trunk, so anything that improves the hidden state helps both objectives at once. The t+2 head is extra supervision on the same representation, not a competing task fighting for capacity.",
    },
    {
      h: "The second head is cheap — but not literally free",
      p: "Trailing means over the last 400 steps: standalone baseline 4.555, MTP t+1 head 4.602 — a +0.047-nat cost. Well inside a single step's noise (sd 0.109), but 8.6 standard errors of the trailing mean, so cleanly resolved rather than zero. And it understates the two-head model: the baseline head is tied (helps a little), the MTP head is untied and shares its trunk gradient with t+2.",
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
