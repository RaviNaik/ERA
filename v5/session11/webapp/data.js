/* ============================================================
   SESSION 11 — Optimizers by Hand — DATA
   Every number here comes from optimizer_experiments/README.md,
   generated from the executed notebooks + the Aim repo (57 runs)
   from the final run on an NVIDIA RTX A6000.
   ============================================================ */

window.SESSION_DATA = {

  meta: {
    session: 11,
    title: "Optimizers by Hand",
    subtitle: "Adam, bias correction, warmup, schedules & LR transfer",
    month: "September 2026",
    preliminary: false,
  },

  setup: {
    hardware: "NVIDIA RTX A6000 · CUDA 12.4 · PyTorch 2.6.0",
    dataset: "char-level Tiny Shakespeare",
    totalChars: 1115394,
    vocab: 65,
    trainSplit: "90 / 10",
    blockSize: 256,
    nLayer: 6,
    nHead: 6,
    nEmbd: 384,
    paramsTotal: 10770816,
    optimizer: "hand-rolled AdamW (m, v, m̂, v̂, decoupled decay)",
    tracking: "Aim — every run logged: loss, lr, grad_norm, per-layer update/weight ratio, full hyper-parameters",
    aimRuns: 57,
  },

  heroStats: [
    { val: "&lt;6e-17", lbl: "hand-Adam vs torch.optim (float64)", cls: "green" },
    { val: "232 steps", lbl: "bias correction stops mattering (β2=0.99)", cls: "indigo" },
    { val: "≈1e-3", lbl: "update / weight ratio at convergence", cls: "amber" },
    { val: "+0.076", lbl: "WSD beats cosine, step 300 (lower loss)", cls: "green" },
    { val: "≈1.9e-4", lbl: "predicted LR at width 4096 (low confidence)", cls: "rose" },
  ],

  paramTable: [
    { name: "attention (6 blocks)", n: 3538944, pct: "32.9%" },
    { name: "MLP (6 blocks)", n: 7077888, pct: "65.7%" },
    { name: "token embedding / lm-head (tied)", n: 24960, pct: "0.2%" },
    { name: "position embedding", n: 98304, pct: "0.9%" },
    { name: "LayerNorm (13×, with bias)", n: 9984, pct: "0.1%" },
  ],

  /* ---- the five tasks (overview cards) ---- */
  tasks: [
    {
      n: 1, tag: "adam", title: "Reproduce Adam by hand",
      headline: "hand-computed m, v, m̂, v̂, step match torch.optim to < 6e-17",
      detail: "One weight w₀=0.7213, five gradients [0.15, −0.30, 0.05, 0.22, −0.11]. Every intermediate quantity computed by hand and diffed against torch.optim.Adam / AdamW in float64. Step 1 is almost exactly a unit step α·sign(g₁) — the bias-correction factors cancel on the first step. Adam+L2 and AdamW (decoupled decay) diverge by step 2–3. Purely analytic — identical on every machine it's been run on.",
      verdict: "<6e-17",
    },
    {
      n: 2, tag: "bias", title: "Bias correction, on vs off",
      headline: "correction factor c(t) is non-monotonic — dips below 1 before climbing back to 1",
      detail: "c(t) = √(1−β₂ᵗ) / (1−β₁ᵗ). Stays within 1% of 1 only after 76 steps (β2=0.95), 390 steps (β2=0.99), or 3,916 steps (β2=0.999) — purely analytic. Replayed on a real weight's first 20 true gradients from the 10.77M model — the two trajectories differ by 73% of the weight's total movement after 20 steps.",
      verdict: "232 steps",
    },
    {
      n: 3, tag: "warmup", title: "Update/weight ratio × warmup",
      headline: "argmax(ρ) lands on the warmup length to within a handful of steps",
      detail: "ρ = ‖ΔW‖_RMS / ‖W‖_RMS, logged per layer every step across 300-step runs at warmup ∈ {0, 50, 200}. ρ settles at ≈1.0–1.4e-3 for every attention/MLP block — right on the folklore 1e-3 target. LayerNorm gains run ~25× colder. Skipping warmup cost ~0.47 nats of final loss on this budget.",
      verdict: "ρ≈1e-3",
    },
    {
      n: 4, tag: "schedule", title: "Cosine vs WSD, judged at step 200",
      headline: "WSD ties cosine at step 200, wins by 0.076 nats at the full 300-step horizon",
      detail: "Same model, warmup 50, base_lr 1.5e-3. Frozen mid-plateau at step 200, cosine edges WSD by 0.013 (WSD hasn't started decaying yet). Re-planned to a 200-step budget, WSD beats cosine by 0.032. At the full 300 steps, WSD's longer high-LR plateau wins outright.",
      verdict: "keep WSD",
    },
    {
      n: 5, tag: "lrsweep", title: "LR sweep across width → predict width 4096",
      headline: "optimum flat 256→512 (~1.4–1.6e-3), drops sharply at 1024 (~5e-4)",
      detail: "Coarse 7-point grid × 3 widths, then a 2-seed 5-point refinement per basin (57 runs total, ~14 min on an RTX A6000). Power-law fit η*∝width^b gives b=−0.81 with a large residual (σ≈0.54) — the slope is carried almost entirely by the width-1024 point. Extrapolation: η*(4096) ≈ 1.9e-4, confidence LOW (factor of ~3).",
      verdict: "≈1.9e-4, LOW",
    },
  ],

  /* ==================== TASK 1 — Adam by hand ==================== */
  adam: {
    w0: 0.7213,
    grads: [0.15, -0.30, 0.05, 0.22, -0.11],
    lr: 0.001, beta1: 0.9, beta2: 0.999, eps: "1e-8",
    trace: [
      { t: 1, m: 0.015000, v: "2.2500e-05", mhat: 0.150000, vhat: 0.022500, step: "1.0000e-03", w: 0.72030000 },
      { t: 2, m: -0.016500, v: "1.1248e-04", mhat: -0.086842, vhat: 0.056267, step: "-3.6610e-04", w: 0.72066610 },
      { t: 3, m: -0.009850, v: "1.1486e-04", mhat: -0.036347, vhat: 0.038327, step: "-1.8566e-04", w: 0.72085176 },
      { t: 4, m: 0.013135, v: "1.6315e-04", mhat: 0.038194, vhat: 0.040849, step: "1.8898e-04", w: 0.72066279 },
      { t: 5, m: 0.000822, v: "1.7509e-04", mhat: 0.002006, vhat: 0.035088, step: "1.0709e-05", w: 0.72065208 },
    ],
    agreement: [
      { variant: "torch.optim.Adam, no decay", diff: "5.4e-17" },
      { variant: "torch.optim.Adam + L2 decay (λ=0.1)", diff: "6.9e-18" },
      { variant: "torch.optim.AdamW, decoupled decay (λ=0.1)", diff: "3.5e-18" },
    ],
    decayCompare: [
      { t: 1, adam: 0.72030000, l2: 0.72030000, adamw: 0.72022787 },
      { t: 2, adam: 0.72066610, l2: 0.72036560, adamw: 0.72052195 },
      { t: 3, adam: 0.72085176, l2: 0.72018410, adamw: 0.72063556 },
      { t: 4, adam: 0.72066279, l2: 0.71969294, adamw: 0.72037452 },
      { t: 5, adam: 0.72065208, l2: 0.71932538, adamw: 0.72029177 },
    ],
    findings: [
      "Step 1 is a unit step: the bias-correction factors cancel (m̂₁ = g₁, √v̂₁ = |g₁|), so Δ₁ ≈ α·sign(g₁) regardless of gradient magnitude.",
      "The corrected moments do the work — for the first steps v̂ ≫ v (dividing by 1−β₂ᵗ ≈ 1e-3), which is exactly what bias correction is for.",
      "Adam+L2 ≠ AdamW: with λ=0.1 the trajectories separate by step 2–3. L2-Adam feeds λw through the moment estimates (high-gradient weights decay less); AdamW multiplies every weight by (1−αλ) uniformly.",
      "Myth check: bias correction is NOT what makes AdamW ‘AdamW’. Bias correction (÷ 1−βᵗ) fixes the zero-init EMA bias and is part of plain Adam; AdamW's distinctive feature is decoupled weight decay. They're independent mechanisms.",
    ],
  },

  /* ==================== TASK 2 — bias correction ==================== */
  bias: {
    horizons: [
      { beta2: "0.95", within5: 42, within1: 76, ctx: "nanoGPT / LLM default" },
      { beta2: "0.99", within5: 232, within1: 390, ctx: "this repo's training default" },
      { beta2: "0.999", within5: 2327, within1: 3916, ctx: "Adam-paper default" },
    ],
    realWeight: {
      name: "transformer.h.0.mlp.c_fc.weight[0,0]",
      beta2: 0.99,
      rows: [
        { t: 1, c: 1.000, ratio: 1.000, gap: "~0" },
        { t: 5, c: 0.541, ratio: 0.541, gap: "1.6e-3" },
        { t: 10, c: 0.475, ratio: 0.475, gap: "3.4e-3" },
        { t: 20, c: 0.486, ratio: 0.486, gap: "5.1e-3" },
      ],
      gapPctOfMovement: 73,
    },
    step1Multiplier: [
      { beta2: "0.999", mult: 0.32, note: "smaller step" },
      { beta2: "0.99", mult: 1.00, note: "exactly a unit step" },
      { beta2: "0.95", mult: 2.24, note: "larger step" },
    ],
    findings: [
      "‘Bias correction off’ is not simply a warmup — the step-1 multiplier (1−β₁)/√(1−β₂) can be smaller OR larger than 1 depending on β₂.",
      "After ~10–15 steps c(t) dips below 1 (the m-EMA has caught up, the v-EMA hasn't → √v̂ too small → uncorrected step too big), then recovers over hundreds–thousands of steps.",
      "Net practical impact on a multi-thousand-step run: negligible — it self-heals well inside a normal warmup — but it's free and removes a confound, so leave it on.",
    ],
  },

  /* ==================== TASK 3 — update/weight ratio + warmup ==================== */
  ratio: {
    layers: [
      { layer: "L0.attn", r5: "3.9e-3", r50: "1.2e-2", r150: "8.2e-3", r299: "1.0e-3" },
      { layer: "L0.mlp", r5: "5.3e-3", r50: "8.4e-3", r150: "5.9e-3", r299: "1.3e-3" },
      { layer: "L3.mlp", r5: "4.7e-3", r50: "6.5e-3", r150: "4.5e-3", r299: "1.3e-3" },
      { layer: "L5.mlp", r5: "4.8e-3", r50: "6.2e-3", r150: "4.4e-3", r299: "1.4e-3" },
      { layer: "token embedding", r5: "4.4e-3", r50: "9.3e-3", r150: "4.8e-3", r299: "7.5e-4" },
      { layer: "LayerNorm", r5: "8.8e-5", r50: "1.7e-4", r150: "1.5e-4", r299: "3.1e-5", cold: true },
      { layer: "position embedding", r5: "2.8e-3", r50: "1.0e-2", r150: "1.0e-2", r299: "1.2e-3" },
    ],
    globalFinal: "7.7e-4",
    warmupRuns: [
      { warmup: 0, argmax: 0, peak: "3.7e-2", finalVal: 2.4671 },
      { warmup: 50, argmax: 49, peak: "4.6e-3", finalVal: 1.9922 },
      { warmup: 200, argmax: 195, peak: "5.5e-3", finalVal: 1.9999 },
    ],
    findings: [
      "Reported answer: the step at which warmup stops changing ρ IS the warmup length itself — warmup ∈ {0, 50, 200} stops changing ρ at step ≈ {0, 50, ~195–200}. argmax(ρ) lands there to within a handful of steps.",
      "Per-layer ρ converges to 1.0–1.4e-3 for every attention/MLP block; depth barely matters (spread under 1.3×).",
      "Skipping warmup hurt a lot on this budget: warmup=0 finished at val 2.47 vs 1.99–2.00 for warmup∈{50,200} — grad-clip shock on steps 0–3 plus a lower average LR (cosine starts decaying immediately). Past 'enough' warmup, the exact value barely matters (50 and 200 land within 0.008 of each other).",
    ],
  },

  /* ==================== TASK 4 — cosine vs WSD ==================== */
  schedules: {
    baseLr: 0.0015, warmup: 50, totalSteps: 300,
    checkpoints: [
      { name: "cosine, stop at 200", loss: 2.073, lr: "6.0e-4", tag: "" },
      { name: "WSD (300-plan), stop at 200", loss: 2.086, lr: "1.5e-3 (plateau)", tag: "" },
      { name: "WSD re-planned to a 200 budget", loss: 2.041, lr: "decayed 160→200", tag: "best" },
      { name: "cosine, full 300", loss: 1.920, lr: "1.5e-4", tag: "" },
      { name: "WSD, full 300", loss: 1.845, lr: "2.5e-5", tag: "best" },
    ],
    /* computed exactly from src/s11/optim.py lr_cosine / lr_wsd — deterministic */
    lrCurve: {
      steps: [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 299],
      cosine: [0.00003, 0.00078, 0.0015, 0.001467, 0.001371, 0.001222, 0.001034, 0.000825, 0.000616, 0.000428, 0.000279, 0.000183, 0.00015],
      wsd:    [0.00003, 0.00078, 0.0015, 0.0015, 0.0015, 0.0015, 0.0015, 0.0015, 0.0015, 0.0015, 0.00125, 0.000625, 0.000025],
    },
    findings: [
      "Frozen mid-plan at step 200, cosine (2.073) beats plateau-WSD (2.086) by a thin 0.013 — the 'you stopped me mid-plateau' penalty.",
      "Told up front to stop at 200, WSD decays properly and its checkpoint (2.041) beats cosine-at-200 by 0.032 — higher average LR, still got its cooldown.",
      "At the full 300-step horizon WSD wins outright (1.845 vs 1.920, −0.076).",
      "Decision: keep the WSD full-300 checkpoint. WSD gives up essentially nothing versus cosine and removes the 'commit to the horizon before you start' constraint. Only if forced to stop at exactly step 200 with no re-plan would cosine win, and only by 0.013.",
    ],
  },

  /* ==================== TASK 5 — LR sweep vs width ==================== */
  lrsweep: {
    widths: [
      { width: 256, params: "4.82M", fitted: "1.4e-3", gridBest: "1.4e-3", loss: "2.40", spread: "±0.002–0.006" },
      { width: 512, params: "19.1M", fitted: "1.6e-3", gridBest: "2.0e-3", loss: "2.18–2.22", spread: "±0.03" },
      { width: 1024, params: "75.9M", fitted: "5.0e-4", gridBest: "4.3e-4", loss: "2.17–2.18", spread: "±0.004" },
    ],
    /* coarse loss-vs-lr curves for the SVG chart, log10(lr) on x */
    coarseGrid: [0.0003, 0.0006, 0.001, 0.002, 0.004, 0.008, 0.015],
    coarseLoss: {
      256:  [2.4595, 2.4323, 2.4115, 2.4148, 2.4339, 2.4794, 2.5758],
      512:  [2.4068, 2.3744, 2.3474, 2.2076, 2.4949, 2.5899, 2.8030],
      1024: [2.2585, 2.1922, 2.4772, 2.5115, 2.6058, 2.7613, 3.1072],
    },
    powerLaw: { b: -0.81, sigma: 0.54, pred4096: "1.9e-4", band: ["6.4e-5", "5.5e-4"], endpointOnly: "1.5e-4" },
    recommendation: { value: "2e-4", confirmSweep: ["1e-4", "2e-4", "4e-4"], confidence: "LOW" },
    findings: [
      "The optimum is flat from width 256→512 (~1.4–1.6e-3), then drops sharply at 1024 (~5e-4).",
      "Power-law fit η*∝width^b gives b=−0.81, with a large residual σ≈0.54 (log-LR) — the slope is carried almost entirely by the 1024 point. SP theory predicts b=−1 for matmul params; non-scaling embedding/LayerNorm params pull it toward 0.",
      "Extrapolation to width 4096: η*(4096) ≈ 1.9e-4 (endpoint-only bracket 1.5e-4; ±2σ band 6.4e-5–5.5e-4). The moving segment alone (512→1024) points as low as 5e-5.",
      "Confidence: LOW — factor of ~3. Only 3 width points with the slope set by one; 220-step runs favour a higher LR than a long run would; extrapolating 2 octaves from a 2-octave measurement; batch size/warmup/weight decay/β2 all held fixed. Under muP the width-1024 optimum would transfer to 4096 directly.",
    ],
  },

  figures: [
    { src: "../optimizer_experiments/assets/t1_adam_by_hand.png", cap: "Task 1 — hand-computed Adam trace: m, v, m̂, v̂, the update, bias-correction denominators, and hand-vs-torch weight trajectory overlay." },
    { src: "../optimizer_experiments/assets/t2_bias_correction.png", cap: "Task 2 — bias correction on vs off over the first 20 steps: update size, weight trajectory, cumulative gap, and the correction factor c(t) at three β₂ values." },
    { src: "../optimizer_experiments/assets/t3_update_weight_ratio.png", cap: "Task 3 — global and per-layer update/weight ratio ρ across warmup ∈ {0,50,200}, LR schedules, ρ vs depth, and the loss curves." },
    { src: "../optimizer_experiments/assets/t4_cosine_vs_wsd.png", cap: "Task 4 — cosine vs WSD LR schedules and val-loss curves, zoomed on steps 150–300 where the WSD decay kicks in." },
    { src: "../optimizer_experiments/assets/t5_lr_sweep_width.png", cap: "Task 5 — loss vs LR by width (coarse + 2-seed refinement, fitted minima) and the LR-transfer power-law extrapolation to width 4096." },
    { src: "../optimizer_experiments/assets/aim_runs_explorer.png", cap: "Aim Runs Explorer — all 57 runs from the final RTX A6000 run, with per-layer update_weight_ratio columns pulled in live." },
    { src: "../optimizer_experiments/assets/aim_metrics_task4.png", cap: "Aim Metrics Explorer — the three Task 4 val-loss curves; the WSD curve pulls below cosine only after step ~240 when its decay begins." },
  ],

  commentary: [
    {
      h: "Adam is sign(g) early, EMA-smoothed later",
      p: "The unit first step is why warmup and grad-clip matter most in the first handful of steps — before the moment estimates have anything real to smooth.",
    },
    {
      h: "Bias correction is a β₂-scoped early-training correction",
      p: "Real but short-lived: it self-heals within hundreds of steps for typical β₂ values. Keep it on — it's free — but don't confuse it with AdamW's actual contribution (decoupled weight decay).",
    },
    {
      h: "Update-to-weight ratio ≈1e-3 is a genuine, layer-stable operating point",
      p: "Watching ρ per layer is a cheap divergence early-warning — and the step where warmup stops driving it is, almost by definition, the warmup length itself.",
    },
    {
      h: "WSD ≥ cosine whenever the horizon isn't nailed down",
      p: "At no measured cost when it is. The flat plateau is re-plannable; cosine forces you to commit to a horizon before you start.",
    },
    {
      h: "LR does not transfer for free under standard parametrization",
      p: "It drifts with width, and extrapolating it two octaves is a low-confidence move. muP is the fix if width is going to keep changing.",
    },
  ],
};
