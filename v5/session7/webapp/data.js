/* ============================================================
   SESSION 7 — Fourier Embeddings — DATA
   All figures below are taken verbatim from:
     - fourier_embeddings_research.md   (Part I: theory)
     - empirical_validation_report.md   (Part II: empirical run)
     - fourier_embeddings_complete_paper.md (combined, canonical)
   Nothing here is invented for display purposes.
   ============================================================ */

window.SESSION_DATA = {

  /* ── Kernel table from research note §3.2 (toy d_p=4, m=2) ──── */
  kernelToyTable: [
    { delta: 0,  cos1: 1.000,  cos2: 1.0000, K: 2.000 },
    { delta: 1,  cos1: 0.540,  cos2: 1.0000, K: 1.540 },
    { delta: 2,  cos1: -0.416, cos2: 1.0000, K: 0.584 },
    { delta: 3,  cos1: -0.990, cos2: 0.9996, K: 0.010 },
    { delta: 5,  cos1: 0.284,  cos2: 0.9988, K: 1.282 },
    { delta: 10, cos1: -0.839, cos2: 0.9950, K: 0.156 },
  ],

  /* ── Parameter accounting — theory (research note §4.5), V5 scale ── */
  paramTheoryV5: [
    { codec: "Kronecker (shipped, one-hot grid)", D: "256 × 32 = 8,192", params: "66,322,432", wall: "Yes — hard, at 32 bytes", decoupled: "No — D is the coverage budget" },
    { codec: "Design A (Fourier position)", D: "256 × d_p, d_p free (e.g. 16–32 bands)", params: "comparable or smaller if d_p shrinks", wall: "No", decoupled: "Yes — d_p sets bandwidth, not a cutoff" },
    { codec: "Design B (HRR binding)", D: "free, e.g. 512–2048", params: "far smaller if D ≈ 1,024 (~8.3M at d_model=8,096)", wall: "No", decoupled: "Yes — fully decoupled from char_dim and byte count" },
  ],

  /* ── Complementarity verdict — theory only (research note §9) ── */
  verdictTheory: {
    solved: [
      { title: "Hard truncation wall at pos_dim", desc: "φ(p) is defined for every p ∈ ℕ — no analogue of “no basis vector past position 32” exists for a smooth kernel.", ref: "§6.1" },
      { title: "D rigidly coupled to coverage, blocking weight tying", desc: "Reopened as an option, not automatically fixed — D becomes a free bandwidth choice, so D = d_model is now a valid, zero-coverage-cost configuration.", ref: "§6.2" },
      { title: "gpu_table / gpu_dynamic duplication", desc: "φ(p) is closed-form (a handful of sin/cos calls) — every Design-A implementation is, by construction, the dynamic-style path. Collapses two variants into one.", ref: "§6.3" },
      { title: "Fixed-length grid forces cropping of long tokens", desc: "Output dimensionality never depends on L — arbitrarily long tokens sum into the same fixed-size code without cropping.", ref: "§6.4" },
    ],
    unsolved: [
      { title: "Byte-similar, semantically-distant clustering", desc: "“compute” / “commute” remain highly similar under any position kernel — this is correct spelling-based behavior, not a codec defect. Needs downstream/semantic machinery, not a kernel swap.", ref: "§7.1" },
      { title: "Suffix-only morphological alignment", desc: "Both codecs are position-aligned; a shared suffix at a different absolute offset still needs the kernel to bridge a mismatch it wasn't designed to target specifically.", ref: "§7.2" },
      { title: "Compressed input path as a weak adapter", desc: "Design A/B are still fully fixed by construction (only the projection learns). Only Learnable Fourier Features offer a partial lever, at the cost of the “zero learned parameters” property.", ref: "§7.3" },
    ],
    newIssues: [
      { title: "Far-apart positions may alias under a poor schedule", desc: "Failure mode changes from “exactly zero” to “far-apart position codes may accidentally resemble each other” under a bad frequency schedule.", ref: "§7.4 / §8.1" },
      { title: "Spectral bias in the downstream projection", desc: "Networks fit low-frequency components first (Rahaman et al. 2019) — too-narrow or too-high-frequency schedules both risk a harder-to-fit representation.", ref: "§8.1" },
      { title: "Order-sensitivity guarantee is softened", desc: "The one-hot codec's exact-orthogonality guarantee for derangements is traded for a soft, kernel-dependent partial similarity.", ref: "§8.2" },
      { title: "Superposition crosstalk (HRR / Design B only)", desc: "Unbinding recovers the true value plus O(√(n/D)) interference from other bound pairs — controllable via D, but never zero.", ref: "§8.3" },
      { title: "Collision becomes threshold-dependent, not exact", desc: "“Collision” turns into a continuous cosine-similarity quantity, requiring a calibrated threshold the one-hot codec never needed.", ref: "§8.4" },
      { title: "Unvalidated at transformer-training scale (at time of Part I)", desc: "No published result trained Design A/B as a byte-level codec at controlled transformer scale — this is exactly what Part II (this run) closes.", ref: "§8.5" },
    ],
  },

  /* ── Experimental setup table (empirical report §1 / paper §11) ── */
  setupRows: [
    { k: "Model", v: "GPT-2-style decoder-only, 12 layers, 12 heads, d_model=768, block_size=1024, no bias, no dropout" },
    { k: "Arms", v: "dense (control) · kronecker (shipped) · fourier (Design A, standard schedule) · fourier_narrow (Design A, narrow-band ablation)" },
    { k: "Codec config", v: "char_dim=256, pos_dim=32, fourier_dim=32 → D = 256×32 = 8,192 for every byte-level arm" },
    { k: "Vocabulary", v: "16,384-token byte-level BPE, trained on the run's own corpus" },
    { k: "Corpus", v: "~150MB Wikipedia text — English, Hindi, Telugu, Tamil, Bengali" },
    { k: "Tokens", v: "92,083,360 train · 1,879,252 validation (2% tail split)" },
    { k: "Training", v: "5,000 steps, batch 32 × grad-accum 2 = 65,536 tok/step → 327,680,000 tokens (≈3.56 epochs)" },
    { k: "Optimizer", v: "AdamW, lr 3e-4 → 3e-5 cosine decay, 500-step warmup, weight decay 0.1, grad clip 1.0" },
    { k: "Precision", v: "bfloat16 mixed precision, --codec-mode dynamic" },
    { k: "Hardware", v: "NVIDIA RTX A6000 (48GB)" },
  ],

  /* ── Parameter efficiency, measured (§13) ────────────────────── */
  paramMeasured: [
    { arm: "Dense",            total: "110,906,112", codec: "12,582,912", share: "11.3%", reduction: "—" },
    { arm: "Kronecker",        total: "104,614,656", codec: "6,291,456",  share: "6.0%",  reduction: "50.0% (codec) / 5.7% (total)" },
    { arm: "Fourier",          total: "104,614,656", codec: "6,291,456",  share: "6.0%",  reduction: "50.0% (codec) / 5.7% (total)" },
    { arm: "Fourier (narrow)", total: "104,614,656", codec: "6,291,456",  share: "6.0%",  reduction: "50.0% (codec) / 5.7% (total)" },
  ],

  /* ── Final performance comparison (§15) ──────────────────────── */
  finalPerf: [
    { arm: "Dense",            finalLoss: 0.2083, bestLoss: 0.2089, ppl: 1.2316, wall: "59.3 min", best: true },
    { arm: "Fourier",          finalLoss: 0.2181, bestLoss: 0.2118, ppl: 1.2437, wall: "62.9 min" },
    { arm: "Kronecker",        finalLoss: 0.2197, bestLoss: 0.2136, ppl: 1.2457, wall: "63.0 min" },
    { arm: "Fourier (narrow)", finalLoss: 0.2239, bestLoss: 0.2171, ppl: 1.2509, wall: "63.0 min", worst: true },
  ],

  /* ── Collision analysis (§16) ────────────────────────────────── */
  collisionRows: [
    { script: "ASCII",      kronecker: "0.0%", fourier: "86.0%" },
    { script: "Bengali",    kronecker: "0.0%", fourier: "99.2%" },
    { script: "Devanagari", kronecker: "0.0%", fourier: "98.5%" },
    { script: "Tamil",      kronecker: "0.0%", fourier: "99.8%" },
    { script: "Telugu",     kronecker: "0.0%", fourier: "98.3%" },
    { script: "(other)",    kronecker: "15.8%", fourier: "not directly comparable" },
  ],

  /* ── Order-sensitivity probe (§17) ───────────────────────────── */
  orderSensitivitySummary: [
    { metric: "Mean cosine (rearranged pairs)", kronecker: "0.719", fourier: "0.985", best: "fourier" },
    { metric: "Std. dev.",                      kronecker: "0.154", fourier: "0.026" },
    { metric: "Min",                             kronecker: "-0.001", fourier: "0.813" },
    { metric: "Max",                             kronecker: "1.000",  fourier: "1.000" },
    { metric: "n pairs",                         kronecker: "500",    fourier: "500" },
  ],

  orderSensitivityWorked: [
    { pair: "cat / tac", kind: "anagram with a fixed point", kronecker: 0.333, fourier: 0.905 },
    { pair: "abcde / bcdea", kind: "true derangement (no fixed point)", kronecker: -0.001, fourier: 0.911 },
    { pair: "mistake / mistkae", kind: "transposition", kronecker: 0.714, fourier: 0.988 },
    { pair: "compute / commute", kind: "one-byte substitution", kronecker: 0.857, fourier: 0.879 },
    { pair: "listen / silent", kind: "anagram with a fixed point", kronecker: 0.166, fourier: 0.914 },
  ],

  /* ── HRR crosstalk probe (§18) ───────────────────────────────── */
  crosstalkTable: {
    header: ["D", "n=1", "n=2", "n=4", "n=8", "n=16", "n=32"],
    rows: [
      ["256",  "100.0%", "100.0%", "100.0%", "99.7%",  "87.8%",  "51.0%"],
      ["512",  "100.0%", "100.0%", "100.0%", "100.0%", "99.7%",  "87.2%"],
      ["1024", "100.0%", "100.0%", "100.0%", "100.0%", "100.0%", "99.6%"],
      ["2048", "100.0%", "100.0%", "100.0%", "100.0%", "100.0%", "100.0%"],
    ],
  },

  /* ── Updated verdict — theory meets practice (§19) ───────────── */
  updatedVerdict: [
    { claim: "Fourier removes the hard truncation wall", ref: "§6.1", status: "not-exercised", note: "Tokens never approached pos_dim=32 — BPE subwords are short by construction (§16)." },
    { claim: "Fourier trains successfully at transformer scale", ref: "§8.5", status: "confirmed", note: "All three byte-grid arms trained stably; Fourier reached lower val loss than Kronecker." },
    { claim: "Fourier's smooth kernel measurably helps the downstream model", ref: "§5 / §15", status: "confirmed", note: "Fourier < Kronecker on final and best val loss, consistently from step 1,500 on." },
    { claim: "A too-narrow frequency schedule underperforms a proper log-linear one", ref: "§8.1", status: "confirmed", note: "fourier_narrow worst of the three byte-grid arms on every final metric." },
    { claim: "Order-sensitivity guarantee is exactly traded, not just “reduced”", ref: "§8.2", status: "confirmed", note: "Derangement pair: Kronecker ≈ 0.0, Fourier ≈ 0.91." },
    { claim: "compute/commute-style spelling proximity is unaffected by kernel choice", ref: "§7.1", status: "confirmed", note: "Both codecs report high, close similarity (0.857 vs. 0.879)." },
    { claim: "Collision becomes threshold-dependent and needs empirical calibration", ref: "§8.4", status: "confirmed-hard", note: "A naive global-threshold calibration proved unusable for a same-script comparison." },
    { claim: "Crosstalk is controllable by D, not eliminated (Design B)", ref: "§8.3", status: "confirmed", note: "Top-1 accuracy tracks the predicted (D, n) shape; raw noise-ratio magnitude does not (instrumentation issue)." },
    { claim: "No V-dependence in codec parameter count", ref: "§1.1 / §4.5", status: "confirmed", note: "D × d_model = 6,291,456 for every byte-grid arm, independent of vocab_size=16,384." },
  ],

  /* ── References (paper §22 / §12) ────────────────────────────── */
  references: [
    { text: "Shravan, R. (2026). Kronecker Embeddings: Byte-Level Structured Token Representations for Parameter-Efficient Language Models. arXiv:2605.29459.", href: "https://arxiv.org/abs/2605.29459" },
    { text: "Rahimi, A. & Recht, B. (2007). Random Features for Large-Scale Kernel Machines. NeurIPS 2007." },
    { text: "Tancik, M. et al. (2020). Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains. NeurIPS 2020 / arXiv:2006.10739.", href: "https://arxiv.org/pdf/2006.10739" },
    { text: "Li, Y. et al. (2021). Learnable Fourier Features for Multi-Dimensional Spatial Positional Encoding. NeurIPS 2021 / arXiv:2106.02795.", href: "https://arxiv.org/abs/2106.02795" },
    { text: "Lee-Thorp, J. et al. (2021). FNet: Mixing Tokens with Fourier Transforms. NAACL 2022 / arXiv:2105.03824.", href: "https://arxiv.org/abs/2105.03824" },
    { text: "Li, Z. et al. (2020). Fourier Neural Operator for Parametric Partial Differential Equations." },
    { text: "Plate, T. (1995). Holographic Reduced Representations. IEEE Transactions on Neural Networks." },
    { text: "Wang, B. et al. (2019). Encoding word order in complex embeddings. ICLR 2020 / arXiv:1912.12333.", href: "https://arxiv.org/pdf/1912.12333" },
    { text: "Rahaman, N. et al. (2019). On the Spectral Bias of Neural Networks. ICML 2019 / arXiv:1806.08734.", href: "https://arxiv.org/pdf/1806.08734" },
    { text: "Foundation, W. (2023). Wikimedia/Wikipedia dataset, 20231101 dump. Hugging Face Datasets.", href: "https://huggingface.co/datasets/wikimedia/wikipedia" },
  ],

  /* ── Kernel formulas — exact code paths, for the interactive widget ──
     sinusoidal_position_table() in fourier_embeddings/model/codecs.py
     standard: omega_i = 1 / base^(2i/d_p),           i = 0..m-1
     narrow:   omega_i = omega_low * (1 + i/(m-1)),   omega_low = 1/base
     base = 10000, d_p = 2m
  ─────────────────────────────────────────────────────────────────── */
  kernelBase: 10000,
};
