/* ============================================================
   SESSION 10 — The Training Loop — DATA
   Every number here comes from the executed notebook
   (training_loop/session10_training_loop.ipynb) and its
   assets/results.json, run on an NVIDIA RTX A6000.
   ============================================================ */

window.SESSION_DATA = {

  meta: {
    session: 10,
    title: "The Training Loop",
    subtitle: "Take a small model and a real loop, and make it tell you the truth about itself",
    month: "September 2026",
  },

  setup: {
    hardware: "NVIDIA RTX A6000 · CUDA 12.4 · PyTorch 2.6.0",
    dataset: "char-level Tiny Shakespeare",
    trainChars: 1003854,
    valChars: 111540,
    vocab: 65,
    lnVocab: 4.174,
    blockSize: 256,
    nLayer: 6,
    nHead: 6,
    nEmbd: 384,
    headDim: 64,
    mlpHidden: 1536,
    paramsTotal: 10770816,
    paramsNonEmbedding: 10672512,
    optimizer: "AdamW, lr 3e-3",
  },

  heroStats: [
    { val: "~9 s.f.", lbl: "analytic vs numeric gradient match", cls: "green" },
    { val: "15.38%", lbl: "avg-of-averages accumulation error", cls: "rose" },
    { val: "1 step", lbl: "grad norm leads the loss", cls: "indigo" },
    { val: "4.75%", lbl: "measured MFU on an A6000 (noisy)", cls: "amber" },
  ],

  /* ---- the six tasks ---- */
  tasks: [
    {
      n: 1, tag: "shapes", title: "Print every tensor shape in the step",
      headline: "by-hand walk matches the fused forward() to 7.15e-7",
      detail: "One batch (B=4, T=16), walked one sub-module at a time — embeddings, attention un-rolled (q@kᵀ, causal mask, softmax, @v), MLP, head. Every activation shape printed and named, then every parameter, then after backward() every .grad. grads with the wrong shape: 0. grads that are None: 0.",
      verdict: "all named",
    },
    {
      n: 2, tag: "grad", title: "Verify one gradient by hand",
      headline: "analytic +0.008106809185 vs numeric +0.008106809166",
      detail: "transformer.h.0.mlp.c_fc.weight[0,0], on an fp64 deepcopy with dropout off. analytic from backward(); numeric from the central difference (L(w+h) − L(w−h)) / 2h with h = 1e-4. The two match to the last printed digit — relative difference 2.3e-9, ~9 significant figures on an 8.1e-3 value. (Central-difference truncation is O(h²·L'''), so this also says L''' is modest for this particular weight.)",
      verdict: "~9 sig figs",
    },
    {
      n: 3, tag: "accum", title: "Break gradient accumulation on purpose",
      headline: "static 15.38% error · +7.70% val-loss gap in a real 120-step run",
      detail: "'Average of the averages' gives a short micro-batch the same vote as a long one. Static case (token counts 4,4,2 / mean losses 2,2,5): correct 2.60, wrong 3.00. Real loop, two copies of one model fed the same (48,48,12)-token micro-batches, scored on one fixed 2,048-token batch: held-out loss 3.1303 correct vs 3.3714 wrong — a one-batch estimate, but the same batch for both, so the comparison is exact.",
      verdict: "+7.7% worse",
    },
    {
      n: 4, tag: "gradnorm", title: "Log the grad norm, catch it moving first",
      headline: "poisoned batch at step 70 · norm 0.809 → 3.834 · loss reacts at 71",
      detail: "Global grad norm sqrt(Σ gᵢ²) logged every step, measured after backward() and before opt.step(). At this size / 140-step horizon nothing naturally diverges, so the event is induced: one poisoned batch (random targets) at step 70 spikes the norm 4.7×; the EMA-smoothed loss only rises >5% at step 71. Lead = 1 step here; on a real divergence the lead is thousands of steps.",
      verdict: "leads by 1",
    },
    {
      n: 5, tag: "mfu", title: "Compute your own MFU, honestly",
      headline: "7.36 achieved ÷ 155 peak TFLOP/s = 4.75% MFU",
      detail: "flops/token = 6N + attn = 64,035,072 + 7,077,888 = 71,112,960. Timed loop (bf16 autocast, batch 16 × seq 256, 30 steps): 39.60 ms/step, 103,425 tokens/sec. Distance to 40%: 35.25 points. One short window on a shared GPU is noisy (an earlier run measured ~9.7%) — paid for by a tiny model, small batch, no torch.compile, fp32 norms/optimizer, sync points, and a T² attention term outside 6N.",
      verdict: "4.75%",
    },
    {
      n: 6, tag: "floats", title: "0.1 in fp32, bf16 and fp8 E4M3",
      headline: "rel error 1.5e-8 → 9.8e-4 → 1.6e-2 as the mantissa shrinks",
      detail: "0.1 = 1.6 × 2⁻⁴, unbiased exponent −4 everywhere. Mantissa by hand: 0.6 × 2^bits, round to nearest. fp32 gets 23 bits, bf16 7, fp8 E4M3 only 3. Train in bf16 with an fp32 master copy of the weights — bf16 keeps all 8 exponent bits so nothing underflows, and the fp32 master accumulates the tiny updates bf16 would round away.",
      verdict: "bf16 + fp32 master",
    },
  ],

  /* ---- Task 1: the forward-pass shape walk (B=4, T=16, C=384, H=6, Dh=64, F=1536, V=65) ---- */
  dims: { B: 4, T: 16, C: 384, H: 6, Dh: 64, F: 1536, V: 65 },
  shapeWalk: [
    { k: "idx (x)", s: "[4, 16]", c: "sky", m: "[B, T] — B sequences of T token ids" },
    { k: "token emb", s: "[4, 16, 384]", c: "indigo", m: "[B, T, C] — each id → a C-vector (table [V, C])" },
    { k: "pos emb", s: "[16, 384]", c: "indigo", m: "[T, C] — one vector per slot, broadcast over B" },
    { k: "c_attn", s: "[4, 16, 1152]", c: "purple", m: "[B, T, 3C] — one matmul → q|k|v packed" },
    { k: "q,k,v", s: "[4, 6, 16, 64]", c: "purple", m: "[B, H, T, Dh] — C folded into H heads" },
    { k: "scores", s: "[4, 6, 16, 16]", c: "amber", m: "[B, H, T, T] — every position vs every earlier one" },
    { k: "ctx", s: "[4, 6, 16, 64]", c: "amber", m: "[B, H, T, Dh] — values mixed by attention" },
    { k: "c_proj", s: "[4, 16, 384]", c: "purple", m: "[B, T, C] — heads merged, back into the stream" },
    { k: "mlp c_fc", s: "[4, 16, 1536]", c: "indigo", m: "[B, T, F] — widen to 4C, then GELU" },
    { k: "logits", s: "[4, 16, 65]", c: "rose", m: "[B, T, V] — one score per symbol per position" },
    { k: "loss", s: "[]", c: "green", m: "scalar — mean over the B·T counted tokens" },
  ],
  byHandDiff: "7.15e-7",

  params: [
    { name: "wte.weight / lm_head.weight (tied)", shape: "[65, 384]", n: 24960, maps: "token id ↔ embedding / hidden → vocab logits" },
    { name: "wpe.weight", shape: "[256, 384]", n: 98304, maps: "position → embedding" },
    { name: "attn.c_attn.weight (×6)", shape: "[1152, 384]", n: 442368, maps: "stream → q, k, v" },
    { name: "attn.c_proj.weight (×6)", shape: "[384, 384]", n: 147456, maps: "attention output → stream" },
    { name: "mlp.c_fc.weight (×6)", shape: "[1536, 384]", n: 589824, maps: "stream → MLP hidden" },
    { name: "mlp.c_proj.weight (×6)", shape: "[384, 1536]", n: 589824, maps: "MLP hidden → stream" },
    { name: "ln_1 / ln_2 / ln_f .weight, .bias", shape: "[384]", n: 384, maps: "LayerNorm gain and shift" },
  ],
  gradCheck: {
    param: "transformer.h.0.mlp.c_fc.weight[0, 0]",
    w0: 0.015489,
    h: "1e-4",
    lPlus: 4.321718141721,
    lMinus: 4.321716520359,
    analytic: 0.008106809185,
    numeric: 0.008106809166,
    absDiff: "1.842e-11",
    relDiff: "2.272e-9",
    decimals: 10,
  },

  /* ---- Task 3: accumulation ---- */
  accum: {
    tokenCounts: [4, 4, 2],
    meanLosses: [2.0, 2.0, 5.0],
    correct: 2.6,
    wrong: 3.0,
    relErrPct: 15.38,
    steps: 120,
    realLens: [48, 48, 12],
    microBatch: 8,
    finalValCorrect: 3.1303,
    finalValWrong: 3.3714,
    gapPct: 7.70,
  },

  /* ---- Task 4: grad norm leads the loss ---- */
  gradnorm: {
    nSteps: 140,
    injectStep: 70,
    emaBeta: 0.25,
    normBefore: 0.809,
    normAfter: 3.834,
    normJump: 4.7,
    spikeStep: 70,
    lossRoseStep: 71,
    leadSteps: 1,
  },

  /* ---- Task 5: MFU (RTX A6000) ---- */
  mfu: {
    N: 10672512,
    sixN: 64035072,
    attnTerm: 7077888,
    flopsPerToken: 71112960,
    dtype: "bf16 autocast",
    batch: 16,
    seq: 256,
    measureSteps: 30,
    stepMs: 39.60,
    tokensPerSec: 103425,
    achievedTflops: 7.36,
    peakTflops: 155,
    peakSource: "RTX A6000, bf16 tensor",
    mfuPct: 4.75,
    mfuEarlierPct: 9.71,
    distanceToFortyPts: 35.25,
    noise: "A single 30-step timed window on a shared GPU is noisy — an earlier run of the same cell measured ~9.7%. The structural reasons below are what actually cap this configuration, not the exact percentage.",
    costs: [
      { h: "Tiny model", p: "At 10.7M params 6N is small — each token is cheap in FLOPs but still pays full kernel-launch and Python-loop overhead. MFU climbs with model size; this proxy sits well below where an A6000 saturates." },
      { h: "Small batch / short sequences", p: "batch 16 × seq 256. The matmuls are memory-bandwidth bound, not compute bound — larger B and T amortise the weight reads." },
      { h: "No torch.compile / no fused kernels", p: "Eager mode launches hundreds of small kernels per step; only attention (SDPA) is fused." },
      { h: "bf16 autocast, not a full low-precision path", p: "LayerNorm, the loss reduction and the optimizer still run in fp32 — the peak column assumes pure bf16 tensor-core throughput." },
      { h: "Optimizer + sync points", p: "AdamW is elementwise and bandwidth-bound; every loss.item() forces a device sync that stalls the pipeline." },
      { h: "The attention term grows as T²", p: "It is ~10% of flops/token at T = 256 and is not in 6N — at long context it eats real time the 6N estimate ignores, depressing measured MFU." },
    ],
  },

  /* ---- Task 6: 0.1 in three formats ---- */
  floats: {
    target: 0.1,
    formats: [
      { name: "fp32", layout: "1-8-23", sign: "0", exp: "01111011", mant: "10011001100110011001101", value: 0.100000001490, relErr: "1.49e-8" },
      { name: "bf16", layout: "1-8-7", sign: "0", exp: "01111011", mant: "1001101", value: 0.100097656250, relErr: "9.77e-4" },
      { name: "fp8 E4M3", layout: "1-4-3", sign: "0", exp: "0011", mant: "101", value: 0.101562500000, relErr: "1.56e-2" },
    ],
    nanNote: "E4M3 is not IEEE: no infinity, and the single pattern S.1111.111 is reserved for NaN — which is why the largest finite magnitude is S.1111.110 = 448, not 480. The by-hand encoder saturates to 448 both on overflow and when rounding would land in that NaN slot.",
    extras: [
      { in: "−0.1", bits: "1 0011 101", value: "0.1015625 (sign = 1)", note: "only the sign bit flips vs +0.1" },
      { in: "448.0", bits: "0 1111 110", value: "448.0", note: "the largest finite E4M3 value" },
      { in: "470.0", bits: "0 1111 110", value: "448.0", note: "rounds toward S.1111.111 (NaN) → saturates to 448" },
      { in: "1000.0", bits: "0 1111 110", value: "448.0", note: "overflow → clamped to 448" },
      { in: "0.001", bits: "0 0000 001", value: "0.001953125", note: "subnormal (exp field 0), ~95% error" },
    ],
    answer: "bf16 for the model math, with an fp32 master copy of the weights. bf16 keeps all 8 exponent bits so its range equals fp32's and nothing in training underflows; it pays in precision (2–3 digits), which is fine because the fp32 master weights and fp32 optimizer accumulate the small updates bf16 alone would round away. fp16 keeps only 5 exponent bits and needs loss scaling. fp8 E4M3 (1.6% error on 0.1, clamps past 448) only converges blockwise.",
  },

  /* ---- sanity run ---- */
  sanity: {
    steps: 400,
    batch: 32,
    seq: 128,
    clip: 1.0,
    startLoss: 4.214,
    endLoss: 2.691,
    lnVocab: 4.174,
    wallSec: 14.7,
    sample: "Cotheferey cowi,fake le, bth\n\nHirere obe ale.\nS:BOSED:\nBdelatauss:\nWarthare wecrl t.\nWar dthasomee ar ce myathandanoum orour\nYowhe\nMUUf itir ble mil ndile,",
  },

  figures: [
    { src: "../training_loop/assets/accum_gap.png", cap: "Task 3 — token-weighted vs average-of-averages accumulation. Val loss (left) and train loss (right) over 120 steps; the curves start together and pull apart." },
    { src: "../training_loop/assets/gradnorm_lead.png", cap: "Task 4 — grad norm (red, right axis) vs raw & EMA loss (purple). The poisoned batch at step 70 spikes the norm; the smoothed loss only reacts at 71." },
    { src: "../training_loop/assets/train_curve.png", cap: "Sanity — 400-step run. Train loss falls from ≈ ln V to 2.69 (left); pre-clip grad norm (right) settles after the first few steps." },
  ],

  commentary: [
    {
      h: "Every serious training bug is silent",
      p: "Not one of the six checks would have raised an exception. A wrong accumulation rule (+7.7% worse and falling), a misreported gradient, a run sliding toward divergence, a machine at <5% MFU — all produce a plausible-looking loss curve. The only defense is to print things and check things.",
    },
    {
      h: "The gradient check is a unit test for autograd",
      p: "Ten-decimal agreement between backward() and a finite difference means the chain rule on that weight is exactly what you would compute by hand. It costs two extra forward passes and rules out an entire class of silent bugs — a wrong sign, a missing term, an in-place op that corrupted the graph.",
    },
    {
      h: "Normalise the loss by token, not by micro-batch",
      p: "'Average of the averages' is identical to the correct rule only when every micro-batch holds the same number of tokens — which is why it survived in every major framework until 2024. Feed unequal lengths and it optimises a subtly different objective: 15.4% off on the static case, +7.7% worse val loss here.",
    },
    {
      h: "The grad norm is the earliest warning you get",
      p: "A poisoned batch shows up in sqrt(Σ gᵢ²) on the step it happens; the loss only bends one step later here, and thousands of steps later on a real divergence. Clipping uses the same number — cap the norm, keep the direction — and belongs on from step one.",
    },
    {
      h: "Low MFU is honest at this scale",
      p: "A few percent on an A6000 is what a 10M-param model in eager mode with a 4k-token batch should get — and a single 30-step window swings (9.7% → 4.7% across two runs). The distance to 40% is a to-do list, not a bug: bigger model, bigger global batch, torch.compile, activation checkpointing, a lower-precision format on purpose.",
    },
    {
      h: "Pick the float format for its exponent, not its mantissa",
      p: "fp16 and bf16 are both 16 bits; bf16 trades 3 mantissa bits for 3 exponent bits and that is the right trade for training — range matters more than precision when an fp32 master copy is accumulating the updates. fp8 E4M3 rounds 0.1 to 1.6% and clamps anything past 448; it is a production technique, not a starting point.",
    },
  ],
};
