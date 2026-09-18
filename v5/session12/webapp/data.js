/* ============================================================
   SESSION 12 — 32 Virtual GPUs, ZeRO-0..3 — DATA
   Every number here traces back to the real 32-process CPU run
   in ../parallelization_experiments (see its README + notebook).
   ============================================================ */

window.SESSION_DATA = {

  heroStats: [
    { val: '32', lbl: 'real CPU processes / rank', cls: 'indigo' },
    { val: '4', lbl: 'ZeRO stages simulated', cls: '' },
    { val: '32x', lbl: 'measured memory reduction', cls: 'green' },
    { val: '3e-8', lbl: 'max diff vs reference', cls: 'amber' },
  ],

  /* ═══════════════════ 1. Data parallelism, the baseline ═══════════════════ */
  dp: {
    steps: [
      { n: '01', t: 'Replicate', d: 'Every one of the N ranks gets an identical copy of the full model and the full optimizer state.' },
      { n: '02', t: 'Split the batch', d: 'The global batch is cut into N microbatches, one per rank — every rank sees different data.' },
      { n: '03', t: 'Forward + backward, locally', d: 'Each rank computes its own loss and its own local gradient from its own microbatch.' },
      { n: '04', t: 'All-reduce the gradients', d: 'A collective sums (then averages) the N local gradients so every rank ends up with the identical, global-batch gradient.' },
      { n: '05', t: 'Step, identically, everywhere', d: 'Every rank applies the same optimizer update to its own (identical) replica — no model drift, no extra communication needed.' },
    ],
    equivalence: {
      worldSize: 32, samplesPerRank: 2,
      referenceLoss: 2.303687810897827,
      meanRankLoss: 2.303688077488914,
      maxGradDiff: 4.470348358154297e-8,
      maxParamDiff: 1.4901161193847656e-8,
    },
    findings: [
      'Averaging 32 independently-computed local gradients reproduces the single global-batch gradient to within 4.5e-8 — float32 noise, not an approximation.',
      'Plain data parallelism (what this project calls "ZeRO-0") does not reduce per-GPU memory at all: every rank still holds a full replica of parameters, gradients, and optimizer state.',
      'The only thing data parallelism buys you is throughput (N GPUs process N microbatches per step) — the memory problem is exactly what ZeRO was invented to solve.',
    ],
  },

  /* ═══════════════════ 2. The parallelism landscape ═══════════════════ */
  landscape: [
    {
      name: 'Data Parallelism (DP)', tag: 'replicate the model, split the data',
      shards: 'nothing (model); the dataset only',
      good: 'Model already fits on one GPU; you just want more throughput.',
      bad: 'Memory per GPU never improves — a 70B model still needs ~70B×16 bytes on every single GPU.',
    },
    {
      name: 'Tensor / Model Parallelism (TP)', tag: 'split individual weight matrices',
      shards: 'the matrix multiplications inside a layer (row/column splits)',
      good: 'A single layer is too large for one GPU; needs very fast interconnect (NVLink) because every layer synchronizes.',
      bad: 'High-frequency communication (every layer, every step) — usually confined to GPUs inside one node.',
    },
    {
      name: 'Pipeline Parallelism (PP)', tag: 'split the model by layer, across devices',
      shards: 'contiguous groups of layers, one group per device',
      good: 'Very deep models; communication only at stage boundaries, so it tolerates slower interconnect than TP.',
      bad: '"Pipeline bubbles" (idle time) unless microbatching is tuned carefully; adds implementation complexity.',
    },
    {
      name: 'ZeRO (this project)', tag: 'keep DP\'s simplicity, shard the *redundant state*',
      shards: 'optimizer state (1), + gradients (2), + parameters (3) — never the compute graph itself',
      good: 'Same simple DP training loop and math; memory scales down roughly linearly with world size.',
      bad: 'More sharding = more communication (see the 1.0x/1.5x/1.0x/1.5x results below); ZeRO-3 still needs a full parameter gather to compute at all.',
    },
  ],
  landscapeNote: 'These are not mutually exclusive. Real large-scale training usually combines them: ZeRO (or FSDP) for data-parallel memory sharding, tensor parallelism inside a node for the biggest layers, and pipeline parallelism across nodes — often called "3D parallelism".',

  /* ═══════════════════ 3. ZeRO stages, deep dive ═══════════════════ */
  zeroStages: [
    {
      id: 0, name: 'ZeRO-0', subtitle: 'plain data parallelism (the baseline)',
      replicated: ['parameters', 'gradients', 'optimizer state (m, v)'],
      sharded: [],
      whatHappens: 'Gradients are all-reduced (averaged) across all N ranks. Because every rank then has the identical full gradient and the identical full optimizer state, every rank computes the exact same update — so parameters never need to be re-synchronized afterwards.',
      collectives: ['all_reduce(gradient)  — 2 ring phases'],
      formula: 'M₀ = 16P bytes/rank (fully replicated)',
      commRatio: 1.0,
      configJson: `{
  "train_batch_size": 256,
  "bf16": { "enabled": true },
  "zero_optimization": {
    "stage": 0
  }
}`,
      configNote: '"stage": 0 in a DeepSpeed-style config just means "ordinary data parallelism" — no ZeRO sharding at all. This is the config you start from.',
      pros: [
        'Simplest possible distributed training loop — no sharding logic anywhere.',
        'Lowest communication volume of the four stages (just one gradient all-reduce).',
        'Every rank has the full model at all times — no gather needed before using it (e.g. for generation/eval mid-training).',
      ],
      cons: [
        'Memory per GPU does not improve at all as you add more GPUs.',
        'A model that does not fit on one GPU cannot be trained this way, no matter how many GPUs you add.',
      ],
    },
    {
      id: 1, name: 'ZeRO-1', subtitle: 'shard the optimizer state',
      replicated: ['parameters', 'gradients'],
      sharded: ['optimizer state (m, v) — 1/N per rank'],
      whatHappens: 'Gradients are still fully all-reduced (replicated) exactly as in ZeRO-0. But each rank now only owns Adam\'s momentum/variance for 1/N of the parameters, so it can only correctly step that 1/N shard. After the local step, the freshly-updated shards are all-gathered so every rank\'s (still-replicated) parameter copy stays correct.',
      collectives: ['all_reduce(gradient) — 2 ring phases', 'all_gather(updated parameter shard) — 1 ring phase  ← NEW'],
      formula: 'M₁ = 4P + 12P/N bytes/rank',
      commRatio: 1.5,
      configJson: `{
  "train_batch_size": 256,
  "bf16": { "enabled": true },
  "zero_optimization": {
    "stage": 1,
    "reduce_bucket_size": 5e8
  }
}`,
      configNote: 'Only the optimizer_state category shrinks. "reduce_bucket_size" groups small gradient tensors into one all-reduce call, which matters much more once you have thousands of parameter tensors than it does for our 433-parameter demo model.',
      pros: [
        'Optimizer state is usually the single biggest of the three categories (12 of the 16 bytes/param in the mixed-precision Adam count) — sharding just this one already frees most of the memory ZeRO-3 eventually frees.',
        'A relatively small code/behavior change from plain DP.',
      ],
      cons: [
        'Counter-intuitive result this project measured directly: ZeRO-1 costs MORE communication than plain DP (1.5x), not less — the extra parameter all-gather is not free. Caveat: this is what you get from the direct all-reduce implementation built here (and what DeepSpeed originally shipped) — a "partition-aware" ZeRO-1 that reduce-scatters gradients like ZeRO-2 does can match ZeRO-2\'s 1.0x instead.',
        'Parameters and gradients are still fully replicated, so the memory win is smaller than ZeRO-2/3.',
      ],
    },
    {
      id: 2, name: 'ZeRO-2', subtitle: 'shard gradients too',
      replicated: ['parameters'],
      sharded: ['gradients — 1/N per rank', 'optimizer state (m, v) — 1/N per rank'],
      whatHappens: 'The gradient is never fully materialized on any one rank. Instead of all-reduce, the local gradient is reduce-scattered: every rank ends up owning only the summed gradient shard matching the optimizer-state shard it already owns. The optimizer step and the post-step all-gather work exactly as in ZeRO-1.',
      collectives: ['reduce_scatter(gradient) — 1 ring phase', 'all_gather(updated parameter shard) — 1 ring phase'],
      formula: 'M₂ = 2P + 14P/N bytes/rank',
      commRatio: 1.0,
      configJson: `{
  "train_batch_size": 256,
  "bf16": { "enabled": true },
  "zero_optimization": {
    "stage": 2,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 5e8
  }
}`,
      configNote: '"overlap_comm" hides gradient communication behind the still-running backward pass — a ZeRO-2/3-specific optimization that matters more once there is more communication to hide.',
      pros: [
        'Sharding the gradient exactly pays for the extra traffic ZeRO-1 introduced — measured communication volume returns to the same 1.0x as plain DP.',
        'Substantially better memory than ZeRO-1 for the same communication cost as ZeRO-0 — usually the best default starting point.',
      ],
      cons: [
        'Parameters are still fully replicated — the largest remaining redundancy for very large models.',
        'Gradient accumulation and mixed precision interact with reduce-scatter bucketing in ways that need care (handled by frameworks like DeepSpeed, but worth knowing it exists).',
      ],
    },
    {
      id: 3, name: 'ZeRO-3', subtitle: 'shard parameters too — full sharding',
      replicated: [],
      sharded: ['parameters — 1/N per rank', 'gradients — 1/N per rank', 'optimizer state (m, v) — 1/N per rank'],
      whatHappens: 'Parameters now live sharded at rest as well. Because the model still needs its *whole* weight tensor to run forward and backward, every rank must all-gather everyone else\'s shard to reconstruct the full parameter transiently, right before compute — then discard the borrowed 31/32 again. Since the reconstructed vector cannot stay resident at 1/N memory, this gather-then-discard happens twice: once for forward, again for backward.',
      collectives: ['all_gather(parameters, before forward) — 1 ring phase  ← NEW', 'all_gather(parameters, before backward) — 1 ring phase  ← NEW', 'reduce_scatter(gradient) — 1 ring phase'],
      formula: 'M₃ = 16P/N bytes/rank (+ a transient full-parameter gather)',
      commRatio: 1.5,
      configJson: `{
  "train_batch_size": 256,
  "bf16": { "enabled": true },
  "zero_optimization": {
    "stage": 3,
    "stage3_prefetch_bucket_size": 5e7,
    "stage3_param_persistence_threshold": 1e5,
    "stage3_max_live_parameters": 1e9,
    "offload_optimizer": { "device": "cpu", "pin_memory": true },
    "offload_param": { "device": "cpu", "pin_memory": true }
  }
}`,
      configNote: '"offload_optimizer"/"offload_param" are ZeRO-3-only extensions (sometimes called ZeRO-Infinity) that push the sharded state further, onto CPU RAM or even NVMe, trading more memory headroom for more data movement.',
      pros: [
        'The ideal 1/N reduction for every persistent state category — the only stage that lets the *parameters themselves* scale down with more GPUs.',
        'Combined with CPU/NVMe offload, this is how 100B+ parameter models get trained on GPU memory that is orders of magnitude smaller than the model.',
      ],
      cons: [
        'Same 1.5x communication volume as ZeRO-1 (in ZeRO-1\'s original, direct implementation), for a different reason here: two full-parameter gathers per step (forward + backward) instead of one extra post-step gather.',
        'The transient full-parameter gather is a real, measured extra cost — this project\'s own 433-parameter demo shows it as the single largest number in the whole memory table.',
        'If a model already fits under ZeRO-0/1/2, ZeRO-3 usually makes things slower for no memory benefit — it is not a free "always better" setting.',
      ],
    },
  ],

  /* ═══════════════════ 4. Memory & comms formula (for the calculator) ═══════════════════ */
  formulaBytes: {
    mixedPrecision: { param: 2, grad: 2, master: 4, moment: 8, label: 'bf16/fp16 params+grads, fp32 master+moments (typical production mixed precision)' },
    fullFp32: { param: 4, grad: 4, master: 0, moment: 8, label: 'plain fp32 everywhere (what this project actually ran on CPU)' },
  },

  /* ═══════════════════ 5. Real 32-process CPU experiment ═══════════════════ */
  experiment: {
    setup: [
      ['Backend', 'torch.distributed, backend="gloo" (CPU collectives) — no GPU anywhere'],
      ['Ranks', '32 genuine OS processes (confirmed via 32 distinct PIDs), spawned with torch.multiprocessing.spawn'],
      ['Model', 'DemoMLP — 3-layer regression MLP, 433 parameters, hidden_dim=16'],
      ['Data', '2 unique synthetic samples per rank (64 total), deterministic seed'],
      ['Precision', 'plain float32 throughout — 16 bytes/parameter when fully replicated (4 param + 4 grad + 4 m + 4 v)'],
      ['Padding', '433 parameters padded to 448 so they divide evenly across 32 ranks (14 elements/rank)'],
      ['reduce_scatter', 'Gloo does not implement it (confirmed: RuntimeError); reconstructed from N × dist.reduce, one call per shard owner'],
    ],
    memoryTable: [
      { stage: 0, label: 'ZeRO-0', param: 1792, grad: 1792, opt: 3584, transient: 0, total: 7168 },
      { stage: 1, label: 'ZeRO-1', param: 1792, grad: 1792, opt: 112, transient: 0, total: 3696 },
      { stage: 2, label: 'ZeRO-2', param: 1792, grad: 56, opt: 112, transient: 0, total: 1960 },
      { stage: 3, label: 'ZeRO-3', param: 56, grad: 56, opt: 112, transient: 1736, total: 224 },
    ],
    correctness: [
      { stage: 'ZeRO-0', diff: '2.980e-08' },
      { stage: 'ZeRO-1', diff: '2.980e-08' },
      { stage: 'ZeRO-2', diff: '4.470e-08' },
      { stage: 'ZeRO-3', diff: '4.470e-08' },
    ],
    formulaCheck: [
      { stage: 'ZeRO-0', measured: 7168, predicted: 7168 },
      { stage: 'ZeRO-1', measured: 3696, predicted: 3696 },
      { stage: 'ZeRO-2', measured: 1960, predicted: 1960 },
      { stage: 'ZeRO-3', measured: 224, predicted: 224 },
    ],
    commRatios: [
      { stage: 'ZeRO-0', ratio: 1.0 },
      { stage: 'ZeRO-1', ratio: 1.5 },
      { stage: 'ZeRO-2', ratio: 1.0 },
      { stage: 'ZeRO-3', ratio: 1.5 },
    ],
    helloWorld: { world: 8, expectedSum: 28, distinctPids: 8 },
    reduction: '7168 / 224 = 32.0x — an exact 32-fold reduction, measured, on 32 real processes.',
  },

  /* ═══════════════════ 6. Decision guide: worked scenarios ═══════════════════ */
  scenarios: [
    {
      title: 'A 125M model (GPT-2 small) on one 16GB GPU',
      numbers: 'P=0.125B, N=1 · ZeRO-0 memory = 16×0.125B = 2.0 GB',
      verdict: 'ZeRO-0 (plain DP / DDP)',
      reasoning: 'The model comfortably fits replicated. Adding ZeRO sharding here only adds communication for zero memory benefit — skip it.',
    },
    {
      title: '1.3B model on 8× 16GB GPUs (V100-class)',
      numbers: 'P=1.3B, N=8 · ZeRO-0 = 20.8GB (does not fit) · ZeRO-1 = 4×1.3+12×1.3/8 = 7.15GB',
      verdict: 'ZeRO-1 (or ZeRO-2)',
      reasoning: 'Plain DP does not fit per-GPU no matter how many GPUs you add — DP never shards state. Sharding just the optimizer state (ZeRO-1) already brings it comfortably under 16GB.',
    },
    {
      title: '7B model on 8× 24GB GPUs (RTX 4090 / A10-class)',
      numbers: 'ZeRO-1 = 4×7+12×7/8 = 38.5GB · ZeRO-2 = 2×7+14×7/8 = 26.25GB · ZeRO-3 = 16×7/8 = 14GB',
      verdict: 'ZeRO-3',
      reasoning: 'Both ZeRO-1 and ZeRO-2 exceed 24GB before activations are even counted. ZeRO-3 leaves ~10GB of headroom for activations and the transient parameter gather.',
    },
    {
      title: '70B model on 8× 80GB A100 (one node)',
      numbers: 'ZeRO-3 = 16×70/8 = 140GB per GPU — exceeds 80GB even at the best stage',
      verdict: 'Scale world size, or add offload / tensor parallelism',
      reasoning: 'ZeRO-3 alone is not enough at N=8. Either scale to more ranks (e.g. N=64 across 8 nodes ⇒ 16×70/64 ≈ 17.5GB, fits), or add CPU/NVMe offload (ZeRO-Infinity), or combine with tensor parallelism to reduce the effective per-GPU parameter count directly.',
    },
    {
      title: '175B model on a 512-GPU cluster',
      numbers: 'ZeRO-3 = 16×175/512 ≈ 5.5GB per GPU — memory is no longer the bottleneck',
      verdict: 'ZeRO-3 + tensor/pipeline parallelism (3D parallelism)',
      reasoning: 'At this scale, a 512-way all-gather/reduce-scatter is bandwidth- and latency-bound, not memory-bound. Real systems combine ZeRO/FSDP (sharded data parallelism) with tensor parallelism inside a node (fast NVLink) and pipeline parallelism across nodes, keeping any one collective\'s group size manageable.',
    },
    {
      title: 'A model that already fits, "just in case" ZeRO-3',
      numbers: 'Any P, N where ZeRO-0 already fits with headroom',
      verdict: 'Do not use ZeRO-3',
      reasoning: 'This project measured ZeRO-3 at 1.5x the communication volume of plain DP, plus a real transient full-parameter gather. If the model already fits, that overhead buys nothing.',
    },
  ],

  decisionChecklist: [
    'Does the model fit fully replicated (ZeRO-0) on one GPU, with room for activations? → Use ZeRO-0 / plain DDP. Simpler and cheapest to communicate.',
    'Does it fit once only the optimizer state is sharded (ZeRO-1)? → Use ZeRO-1 if you specifically need to keep gradients replicated (e.g. certain gradient-processing hooks); otherwise prefer ZeRO-2.',
    'Does it fit once gradients are sharded too (ZeRO-2)? → Use ZeRO-2. Same communication volume as plain DP, meaningfully less memory than ZeRO-1.',
    'Still doesn\'t fit? → Use ZeRO-3. Accept the 1.5x communication and the transient parameter gather in exchange for the full 1/N memory reduction.',
    'Still doesn\'t fit even at ZeRO-3 on your current GPU count? → Add CPU/NVMe offload (ZeRO-Infinity), scale world size, or combine with tensor/pipeline parallelism.',
    'Is the *collective group* itself huge (100s of GPUs)? → Communication, not memory, is now the bottleneck. Combine ZeRO with tensor parallelism (intra-node) and pipeline parallelism (inter-node) — "3D parallelism".',
  ],
};
