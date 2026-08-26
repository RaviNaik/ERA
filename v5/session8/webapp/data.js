/* ============================================================
   SESSION 8 — "How Does Attention Work Now?"
   All content data: verified chronology, sources, timeline nodes.
   Every date below was checked against the primary source's own
   arXiv submission-history page (v1 date) on 2026-08-26 — not
   accepted from model memory. See README.md for the full method.
   ============================================================ */

window.SESSION_DATA = (function () {

  /* ── Reference serving config, used by the KV-cache calculator ──
     Matches the class notes' own worked example so the numbers on
     this page reconcile with the lesson (48L, 8 KV heads, d_h=128, bf16). */
  const REF_CONFIG = {
    layers: 48,
    kvHeads: 8,
    headDim: 128,
    bytesPerNumber: 2, // bf16
  };

  /* ── Threads: the "problem lanes" a mechanism can belong to.
     A node can belong to more than one — that overlap is the point:
     the field didn't solve one problem at a time in a clean queue. */
  const THREADS = [
    { id: "compute",   label: "Quadratic Compute",  color: "#22d3ee" },
    { id: "memory",    label: "KV-Cache Memory",     color: "#f59e0b" },
    { id: "position",  label: "Position Handling",   color: "#a78bfa" },
    { id: "extend",    label: "Context Extension",   color: "#f43f5e" },
    { id: "recurrent", label: "Recurrent State",     color: "#10b981" },
    { id: "sparse",    label: "Sparsity",            color: "#6366f1" },
    { id: "compress",  label: "Compression",         color: "#06b6d4" },
    { id: "systems",   label: "Systems / I-O",       color: "#94a3b8" },
  ];

  /* ── Full bibliography, one entry per primary source used in the
     chronology. Every "date" is the arXiv v1 submission date (or,
     where noted, the best-available primary date for a non-arXiv
     source), fetched directly from the source on 2026-08-26. */
  const SOURCES = {
    vaswani2017: {
      authors: "Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin",
      title: "Attention Is All You Need",
      venue: "NeurIPS 2017",
      date: "2017-06-12",
      url: "https://arxiv.org/abs/1706.03762",
      id: "arXiv:1706.03762",
    },
    gpt1_2018: {
      authors: "Radford, Narasimhan, Salimans, Sutskever",
      title: "Improving Language Understanding by Generative Pre-Training",
      venue: "OpenAI technical report",
      date: "2018-06",
      url: "https://openai.com/index/language-unsupervised/",
      id: "OpenAI, June 2018",
    },
    bert2018: {
      authors: "Devlin, Chang, Lee, Toutanova",
      title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
      venue: "NAACL 2019",
      date: "2018-10-11",
      url: "https://arxiv.org/abs/1810.04805",
      id: "arXiv:1810.04805",
    },
    sparseTransformer2019: {
      authors: "Child, Gray, Radford, Sutskever",
      title: "Generating Long Sequences with Sparse Transformers",
      venue: "OpenAI, arXiv preprint",
      date: "2019-04-23",
      url: "https://arxiv.org/abs/1904.10509",
      id: "arXiv:1904.10509",
    },
    mqa2019: {
      authors: "Shazeer",
      title: "Fast Transformer Decoding: One Write-Head is All You Need",
      venue: "arXiv preprint",
      date: "2019-11-06",
      url: "https://arxiv.org/abs/1911.02150",
      id: "arXiv:1911.02150",
    },
    longformer2020: {
      authors: "Beltagy, Peters, Cohan",
      title: "Longformer: The Long-Document Transformer",
      venue: "Allen Institute for AI, arXiv preprint",
      date: "2020-04-10",
      url: "https://arxiv.org/abs/2004.05150",
      id: "arXiv:2004.05150",
    },
    bigbird2020: {
      authors: "Zaheer, Guruganesh, Dubey, Ainslie, Alberti, Ontanon, Pham, Ravula, Wang, Yang, Ahmed",
      title: "Big Bird: Transformers for Longer Sequences",
      venue: "NeurIPS 2020",
      date: "2020-07-28",
      url: "https://arxiv.org/abs/2007.14062",
      id: "arXiv:2007.14062",
    },
    linearAttention2020: {
      authors: "Katharopoulos, Vyas, Pappas, Fleuret",
      title: "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention",
      venue: "ICML 2020",
      date: "2020-06-29",
      url: "https://arxiv.org/abs/2006.16236",
      id: "arXiv:2006.16236",
    },
    rope2021: {
      authors: "Su, Lu, Pan, Wen, Liu",
      title: "RoFormer: Enhanced Transformer with Rotary Position Embedding",
      venue: "arXiv preprint (later Neurocomputing 2024)",
      date: "2021-04-20",
      url: "https://arxiv.org/abs/2104.09864",
      id: "arXiv:2104.09864",
    },
    alibi2021: {
      authors: "Press, Smith, Lewis",
      title: "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation",
      venue: "ICLR 2022",
      date: "2021-08-27",
      url: "https://arxiv.org/abs/2108.12409",
      id: "arXiv:2108.12409",
    },
    flashAttention2022: {
      authors: "Dao, Fu, Ermon, Rudra, Ré",
      title: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
      venue: "NeurIPS 2022",
      date: "2022-05-27",
      url: "https://arxiv.org/abs/2205.14135",
      id: "arXiv:2205.14135",
    },
    gqa2023: {
      authors: "Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón, Sanghai",
      title: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints",
      venue: "EMNLP 2023",
      date: "2023-05-22",
      url: "https://arxiv.org/abs/2305.13245",
      id: "arXiv:2305.13245",
    },
    ntkAware2023: {
      authors: "u/bloc97 (Reddit, r/LocalLLaMA)",
      title: "NTK-Aware Scaled RoPE Allows LLaMA Models to Have Extended (8k+) Context Size Without Any Fine-tuning",
      venue: "Community forum post — no peer-reviewed paper",
      date: "2023-06",
      url: "https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/",
      id: "r/LocalLLaMA, thread 14lz7j5",
    },
    yarn2023: {
      authors: "Peng, Quesnelle, Fan, Shippole",
      title: "YaRN: Efficient Context Window Extension of Large Language Models",
      venue: "ICLR 2024",
      date: "2023-08-31",
      url: "https://arxiv.org/abs/2309.00071",
      id: "arXiv:2309.00071",
    },
    streamingLLM2023: {
      authors: "Xiao, Tian, Chen, Han, Lewis",
      title: "Efficient Streaming Language Models with Attention Sinks",
      venue: "ICLR 2024",
      date: "2023-09-29",
      url: "https://arxiv.org/abs/2309.17453",
      id: "arXiv:2309.17453",
    },
    mistral2023: {
      authors: "Jiang, Sablayrolles, Mensch, Bamford, Chaplot, de las Casas, Bressand, Lengyel, Lample, Saulnier, et al.",
      title: "Mistral 7B",
      venue: "Mistral AI, arXiv preprint",
      date: "2023-10-10",
      url: "https://arxiv.org/abs/2310.06825",
      id: "arXiv:2310.06825",
    },
    mla2024: {
      authors: "DeepSeek-AI",
      title: "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model",
      venue: "arXiv preprint",
      date: "2024-05-07",
      url: "https://arxiv.org/abs/2405.04434",
      id: "arXiv:2405.04434",
    },
    deltaNet2024: {
      authors: "Yang, Wang, Zhang, Shen, Kim",
      title: "Parallelizing Linear Transformers with the Delta Rule over Sequence Length",
      venue: "NeurIPS 2024",
      date: "2024-06-10",
      url: "https://arxiv.org/abs/2406.06484",
      id: "arXiv:2406.06484",
    },
    gatedDeltaNet2024: {
      authors: "Yang, Kautz, Hatamizadeh",
      title: "Gated Delta Networks: Improving Mamba2 with Delta Rule",
      venue: "ICLR 2025 (NVIDIA)",
      date: "2024-12-09",
      url: "https://arxiv.org/abs/2412.06464",
      id: "arXiv:2412.06464",
    },
    nsa2025: {
      authors: "DeepSeek-AI (Yuan, Gao, Zhang, Xie, et al.)",
      title: "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention",
      venue: "ACL 2025",
      date: "2025-02-16",
      url: "https://arxiv.org/abs/2502.11089",
      id: "arXiv:2502.11089",
    },
    drope_unverified: {
      authors: "— (not externally published)",
      title: "“DroPE” — cited only inside this course's own Session 7/8 record",
      venue: "ERA course material (“LightningLM V4 cookbook”), not a public paper",
      date: "unverified",
      url: null,
      id: "no external primary source found",
    },
  };

  /* ── The timeline itself — the part the assignment cares about most.
     Sorted strictly by primary-source date, NOT by taxonomy, NOT by
     the order the class covered them in. */
  const TIMELINE = [
    {
      id: "baseline",
      isBaseline: true,
      date: "2017-06-12",
      dateDisplay: "12 Jun 2017",
      threads: ["compute", "memory"],
      title: "Scaled Dot-Product Attention",
      tagline: "The starting point — not a response to anything yet.",
      sourceId: "vaswani2017",
      problem:
        "RNN/LSTM sequence models process tokens one at a time. That recurrence is a training-speed bottleneck (no parallelism across time) and a modeling bottleneck (information from far away has to survive many sequential steps, so long-range dependencies decay). Attention-as-an-add-on to RNNs (Bahdanau 2014, Luong 2015) already existed, but the recurrence itself was still the bottleneck.",
      mechanism:
        "Project every token into a query, a key and a value. Compare every query with every key by dot product to get raw scores, divide by √d_k to keep the numbers in a sane range for softmax, optionally mask out illegal (future) positions, take softmax across each row so weights are positive and sum to one, then use those weights to combine the value vectors. Q×K → scores → scale → mask → softmax → weighted sum of V.",
      buys:
        "Every token can reach every other token in exactly one hop — no vanishing signal over distance. The whole layer is one big matrix multiply, so training is fully parallel across the sequence (unlike an RNN).",
      costs:
        "All-pairs comparison costs O(T²) compute and produces a O(T²)-sized score matrix. During autoregressive generation, every layer must keep every earlier token's key and value around to avoid recomputing them — a KV cache that grows with every token generated. And the mechanism itself has no built-in sense of order: identical content at two different positions produces identical Q/K/V.",
      chooseWhen:
        "Sequences are short-to-moderate (a few thousand tokens), training throughput matters more than the quadratic compute bill, and you want the most flexible, unrestricted token-to-token routing available.",
      widget: "core-mechanism",
    },

    {
      id: "sinusoidal",
      date: "2017-06-12",
      dateDisplay: "12 Jun 2017 · same paper",
      threads: ["position"],
      title: "Sinusoidal Position Encoding",
      tagline: "Response to: attention has no idea what order the tokens came in.",
      sourceId: "vaswani2017",
      problem:
        "Dot-product attention treats the input as an unordered set. If token 0 and token 4 hold the same content, they produce identical queries, keys and values — the model has no way to tell “near” from “far,” or earlier from later, unless something adds that information in.",
      mechanism:
        "Add a fixed, non-learned vector of sines and cosines at geometrically spaced frequencies to every token's embedding before the first layer: PE(pos,2i)=sin(pos/10000^(2i/d)), PE(pos,2i+1)=cos(pos/10000^(2i/d)). It is a deterministic function of absolute position — no parameters to train.",
      buys:
        "Zero learned parameters. A well-defined value exists for any position the formula is evaluated at. Different frequency bands act like clocks running at different speeds, which in principle lets the network reconstruct relative offsets from combinations of absolute signals.",
      costs:
        "The position signal is only injected once, at the input — turning it into genuinely relative-distance-aware attention scores has to be learned indirectly by later layers, it isn't built into the score itself. The original paper's own ablation found it performed “nearly identical” to a learned table at trained lengths, so it bought little beyond parameter count.",
      chooseWhen:
        "You want a parameter-free position scheme and don't need explicit relative-position structure in the attention score itself. Mostly of historical interest today — almost nothing new is trained with plain sinusoidal encoding anymore.",
      footnote:
        "The same 2017 paper also tried a learned position table as an ablation and found it performed about the same — that alternative just didn't become the field's default until GPT-1 and BERT the following year (next card).",
    },

    {
      id: "absolute-learned",
      date: "2018-06",
      dateDisplay: "Jun–Oct 2018",
      threads: ["position"],
      title: "Absolute Learned Position Embeddings",
      tagline: "Response to: hand-designing the position function is unnecessary — just learn it.",
      sourceId: "gpt1_2018",
      secondarySourceIds: ["bert2018"],
      problem:
        "Sinusoidal encoding works, but it's a hand-picked formula. Once large pretrained language models needed a default, letting position become just another trainable embedding table — the same recipe already used for tokens — was simpler to implement and reason about.",
      mechanism:
        "A trainable embedding matrix of shape [max_position, d_model], indexed by each token's absolute position and added to its token embedding, trained end-to-end with everything else.",
      buys:
        "Architectural simplicity — one more lookup table, no special-cased math. Empirically matches or beats sinusoidal encoding up to the length it was trained on, and can in principle learn any absolute pattern useful for the task.",
      costs:
        "A hard length wall: position N+1 simply has no row in the table. There is no notion of relative distance at all — position 5 and position 300 are two unrelated, arbitrary vectors with no learned relationship between them.",
      chooseWhen:
        "Maximum sequence length is fixed and known ahead of time and you never need to run past it — e.g., encoder-only classification over short, bounded inputs.",
      fixedBy: ["rope", "alibi"],
    },

    {
      id: "sparse-topk",
      date: "2019-04-23",
      dateDisplay: "23 Apr 2019",
      threads: ["compute", "sparse"],
      title: "Sparse & Strided (Top-k-style) Attention",
      tagline: "Response to: the quadratic compute bill becomes the wall as sequences get long.",
      sourceId: "sparseTransformer2019",
      problem:
        "Dense attention's O(T²) score matrix becomes the practical limit as people try feeding in much longer sequences (image pixels, long text, audio). Doubling the context quadruples the work.",
      mechanism:
        "Factorize full attention into a small union of fixed sparse patterns (e.g., strided + local) so information can still reach any position within a couple of hops, cutting per-layer cost to roughly O(T√T). This is the ancestor of the general idea later formalized as top-k attention: let each query use only its most relevant few keys instead of all of them.",
      buys:
        "Sequences tens of thousands of tokens long become trainable. Large compute and memory reduction relative to dense attention at long lengths.",
      costs:
        "Sparsity patterns are hand-designed and fixed by the architecture, not learned or content-dependent. A naive top-k variant still has to score every candidate key before it can discard most of them — if scoring everything was the expensive part, this alone hasn't removed that cost, only the cost of using the low-ranked ones afterward.",
      chooseWhen:
        "Context is very long and structured enough (grid data, strided/local patterns) that a fixed sparse mask still reaches what matters. Less useful when the relevant context is unpredictable and scattered across the sequence.",
      footnote:
        "Extended later the same year and into 2020 by Longformer (sliding-window + task-aware global tokens) and Big Bird (arXiv:2007.14062, 28 Jul 2020 — random + window + global, with a universal-approximation proof).",
      fixedBy: ["deepseek-nsa"],
    },

    {
      id: "mqa",
      date: "2019-11-06",
      dateDisplay: "6 Nov 2019",
      threads: ["memory"],
      title: "Multi-Query Attention (MQA)",
      tagline: "Response to: the KV cache dominates decoding cost once generation is one token at a time.",
      sourceId: "mqa2019",
      problem:
        "Once decoder-only models generate autoregressively, every layer must keep every earlier token's key and value around to avoid recomputing them. With one K/V head per query head (ordinary multi-head attention), that cache is large, and reading it back dominates decoding latency — especially at larger batch sizes.",
      mechanism:
        "Keep many query heads, but let all of them share a single, shared key/value head. Only one K/V projection is computed and cached per layer, instead of one per head.",
      buys:
        "KV cache shrinks by roughly the number of heads (e.g., 8× for an 8-head model). Decoding becomes far more memory-bandwidth friendly and noticeably faster, especially at scale.",
      costs:
        "Every query head now searches through the exact same keys and values — the representational diversity multi-head attention was designed to provide is reduced, and this shows up as measurable quality degradation versus full multi-head attention.",
      chooseWhen:
        "Serving cost and latency at high concurrency matter more than squeezing out the last bit of quality, and you're willing to train (or cheaply ‘uptrain’) for it.",
      fixedBy: ["gqa"],
    },

    {
      id: "sliding-window",
      date: "2020-04-10",
      dateDisplay: "10 Apr 2020",
      threads: ["compute", "sparse"],
      title: "Sliding-Window Attention",
      tagline: "Response to: most useful context in long documents is local, so a generic sparse stride is more structure than needed.",
      sourceId: "longformer2020",
      problem:
        "Sparse Transformer's fixed strided patterns help with long sequences generally, but many document tasks mostly need local context plus a handful of specific globally-important tokens (like a [CLS] token or a question) — a task-shaped pattern can beat a generic stride.",
      mechanism:
        "Each token attends only to a fixed-size window of nearby tokens (±w positions), dilated in deeper layers to widen the receptive field, plus a small set of tokens given full “global” attention. Compute and memory scale linearly — O(T·w), not O(T²).",
      buys:
        "Linear scaling lets documents with thousands of tokens run on ordinary hardware. The receptive field still grows with depth, so distant information can still propagate through several layers.",
      costs:
        "In any single layer, a token genuinely cannot see anything outside its window — information from far away has to relay through multiple layers or the global tokens. If the window is too narrow for the task, the model is architecturally blind to context that actually matters.",
      chooseWhen:
        "The task's useful context is mostly local (documents, code, audio) with a few known anchor tokens. Less suited when arbitrary long-range pairs matter equally throughout the sequence.",
      footnote:
        "Sliding-window attention resurfaced as a production choice in Mistral 7B (arXiv:2310.06825, 10 Oct 2023), paired with GQA and a rolling-buffer KV cache — shipped without attention sinks (next-but-one card), even though that fix already existed by then.",
      fixedBy: ["attention-sinks"],
    },

    {
      id: "linear-attention",
      date: "2020-06-29",
      dateDisplay: "29 Jun 2020",
      threads: ["compute", "memory", "recurrent"],
      title: "Linear Attention",
      tagline: "Response to: what if attention carried a fixed-size state instead of a growing history, the way an RNN does?",
      sourceId: "linearAttention2020",
      problem:
        "Sparse and windowed attention both still keep exact softmax over a selected or nearby set of keys — the underlying object is still a growing, indexed history. Neither addresses whether attention could instead be computed with a constant-size running state.",
      mechanism:
        "Replace softmax(QKᵀ)V with a kernel feature map φ so the sum can be regrouped: because there is no shared softmax denominator tying scores together, S = Σ φ(k)vᵀ can be accumulated as one fixed-size matrix as tokens arrive, and each new query just reads S. This is exactly the “factor the query out” trick — without softmax, every key-value term is independent, so old contributions can be pre-combined before the query even shows up. Per-token cost drops from growing with history to O(1).",
      buys:
        "A constant-size state, independent of sequence length, for both compute and memory. Decoding cost per token becomes O(1) instead of growing with how much history exists — literally an RNN with attention's parallel-trainable form.",
      costs:
        "The shared-normalization competition that softmax provided is gone — the state is a lossy running summary, not an exact per-key memory. A plain additive write can only accumulate; it has no way to correct or overwrite an earlier association once it's baked into the state.",
      chooseWhen:
        "You need genuinely unbounded-length streaming or decoding at fixed memory and can tolerate a compressed, non-exact memory of everything that happened earlier.",
      fixedBy: ["delta-rule"],
    },

    {
      id: "rope",
      date: "2021-04-20",
      dateDisplay: "20 Apr 2021",
      threads: ["position", "extend"],
      title: "RoPE (Rotary Position Embedding)",
      tagline: "Response to: absolute position schemes only indirectly express relative distance, and learned-absolute has a hard length wall.",
      sourceId: "rope2021",
      problem:
        "Both sinusoidal-added and learned-absolute position schemes bake position into the input vector once, so relative distance has to be re-derived indirectly by later layers rather than showing up directly in the attention score — and learned-absolute additionally has a hard length ceiling.",
      mechanism:
        "Instead of adding a position vector, rotate each query and key by an angle proportional to its position (pairing up dimensions and rotating each 2D pair, at several different frequencies). A dot product only depends on the angle between two vectors, so the two absolute rotations cancel and only the relative offset (i−j) survives inside the score.",
      buys:
        "Relative position is now built directly into the attention score itself. The rotation formula is defined at any position — no fixed-size lookup table, no architectural length ceiling in the formula. Shifting a whole sequence forward leaves nearby-token relationships unchanged.",
      costs:
        "Being calculable at a position is not the same as the model having learned to use that position well — evaluating the rotation formula far past the trained length doesn't guarantee good behavior there (this exact gap is what every extension method below has to address). Adds a small, fixed rotation cost per layer.",
      chooseWhen:
        "You want relative-position-aware attention with no architectural length ceiling baked into the formula, with rotation frequencies tuned to the intended training length — today's default for almost every open decoder-only LLM.",
      fixedBy: ["alibi", "ntk-aware", "yarn"],
    },

    {
      id: "alibi",
      date: "2021-08-27",
      dateDisplay: "27 Aug 2021",
      threads: ["position", "extend"],
      title: "ALiBi (Attention with Linear Biases)",
      tagline: "Response to: models trained short still need to run reliably long, without any rotation machinery.",
      sourceId: "alibi2021",
      problem:
        "Sinusoidal and learned-absolute embeddings extrapolate badly past the trained length. RoPE is at least defined at any length, but how well the model actually behaves that far out is still an open question, and it requires per-head rotation machinery.",
      mechanism:
        "Add no positional embedding at all. Instead, subtract a fixed, non-learned penalty m·(i−j) from the raw attention score, proportional to the distance between query and key, with a different fixed slope m per head — baked directly into the score before softmax.",
      buys:
        "Trains at a short context and reliably extrapolates to much longer test-time sequences with no fine-tuning. Zero extra learned parameters and one of the simplest mechanisms on this whole timeline.",
      costs:
        "The distance penalty is a fixed, hand-set inductive bias (recency preference) that the model cannot freely override — tasks that genuinely need to weight a distant token above a nearby one are fighting the bias by construction. Largely superseded in adoption by RoPE-plus-extension in most current frontier LLMs.",
      chooseWhen:
        "Extrapolation robustness at inference time matters more than architectural flexibility, or you want the fewest possible moving parts.",
    },

    {
      id: "flash-attention",
      isBonus: true,
      date: "2022-05-27",
      dateDisplay: "27 May 2022",
      threads: ["systems", "compute"],
      title: "FlashAttention",
      tagline: "Bonus — not on the assignment's minimum list. Response to: exact attention was slower than its own FLOP count, because it was I/O-bound.",
      sourceId: "flashAttention2022",
      problem:
        "Sparse and linear attention approximate their way around the O(T²) compute bill, but exact softmax attention itself was slower and more memory-hungry than its FLOP count alone suggested — naive implementations materialize the full T×T score matrix in slow GPU HBM memory, so attention was bottlenecked on memory I/O, not arithmetic.",
      mechanism:
        "Tile Q, K and V into blocks that fit in fast on-chip SRAM, compute softmax incrementally block-by-block (“onfline”/online softmax) so the full T×T matrix is never written to slow memory, and recompute instead of store during the backward pass.",
      buys:
        "The exact same mathematical output as standard attention — no approximation, no quality trade-off — at dramatically lower wall-clock time and memory. Pushes exact attention's practical context-length ceiling up simply by fixing a systems bottleneck, not the algorithm.",
      costs:
        "The asymptotic O(T²) compute scaling itself is unchanged — this is a (very large) constant-factor systems win, not a new algorithm. The kernel implementation is hardware-specific and non-trivial to port to new accelerators.",
      chooseWhen:
        "Essentially always, when running exact attention — this is now the default execution kernel underneath nearly every mechanism on this timeline, not a competing “instead of” choice.",
    },

    {
      id: "gqa",
      date: "2023-05-22",
      dateDisplay: "22 May 2023",
      threads: ["memory"],
      title: "GQA (Grouped-Query Attention)",
      tagline: "Response to: MQA saves the most cache but costs the most quality — nothing sat in between.",
      sourceId: "gqa2023",
      problem:
        "MQA's single shared K/V head saves the most cache but costs the most quality. Ordinary multi-head attention keeps full quality but the full cache. Nothing offered a tunable middle ground.",
      mechanism:
        "Split query heads into g groups; each group shares one K/V head (g=1 recovers MQA, g=num_heads recovers ordinary MHA). Existing MHA checkpoints can be cheaply “uptrained” into GQA — about 5% of the original pretraining compute — by mean-pooling existing K/V heads within each group.",
      buys:
        "A tunable knob between MQA's cache savings and MHA's quality — e.g. 8 groups instead of 32 heads gives a 4× cache reduction with much smaller quality loss than full MQA. Became the practical default across LLaMA 2/3, Mistral, Gemma, Qwen and DeepSeek.",
      costs:
        "Still linear in context length: cache size ∝ kv_heads × T. GQA lowers the slope of that line; it does not stop the line from growing. At very long context, the cache still eventually dominates memory.",
      chooseWhen:
        "Almost always, for any modern decoder LLM balancing serving cost against quality — the live question is how many groups, not whether to group at all.",
      fixedBy: ["mla"],
    },

    {
      id: "ntk-aware",
      date: "2023-06",
      dateDisplay: "Jun 2023",
      threads: ["extend", "position"],
      title: "NTK-Aware Scaled RoPE",
      tagline: "Response to: naively stretching every RoPE angle by one factor crushes local resolution.",
      sourceId: "ntkAware2023",
      isCommunitySource: true,
      problem:
        "Naively stretching all RoPE rotation angles by a fixed factor (“position interpolation”) to reach a longer context crushes the fine, high-frequency dimensions RoPE uses to distinguish nearby tokens — local attention resolution degrades even though the model can now nominally “reach” farther.",
      mechanism:
        "Change RoPE's base frequency instead of linearly rescaling every position. This stretches the low-frequency (long-range) dimensions much more than the high-frequency (local) ones — an uneven, “neural-tangent-kernel-inspired” rescaling that treats different frequency bands differently.",
      buys:
        "Extends usable context with noticeably less damage to local/nearby-token resolution than naive linear interpolation, and needs no fine-tuning to get a rough win.",
      costs:
        "Still a single global rescaling formula — it treats every frequency band with one rule rather than tailoring each band individually, so quality still degrades noticeably at more aggressive extension factors.",
      chooseWhen:
        "You need a fast, no-fine-tuning context stretch and can accept a partial quality hit. Mostly superseded by YaRN for anything where quality matters.",
      footnote:
        "No peer-reviewed paper exists for this one — the primary source is a community forum post. It's included because the YaRN paper itself credits and formalizes it, and because a chronology that skips community-origin ideas would misrepresent how this particular thread of the field actually moved.",
      fixedBy: ["yarn"],
    },

    {
      id: "yarn",
      date: "2023-08-31",
      dateDisplay: "31 Aug 2023",
      threads: ["extend", "position"],
      title: "YaRN",
      tagline: "Response to: one global rescaling formula is still too blunt an instrument.",
      sourceId: "yarn2023",
      problem:
        "NTK-aware scaling improves on naive linear interpolation but still applies one formula uniformly across the whole frequency spectrum. Some frequency bands need interpolation, some don't, and the ones in between need a smooth transition — and stretching positions also drifts softmax's effective temperature.",
      mechanism:
        "“NTK-by-parts”: split RoPE's frequency spectrum into three bands — high-frequency (local) dimensions left un-interpolated, low-frequency (global) dimensions linearly interpolated, and a ramp function smoothly blending the middle band — combined with a small attention-temperature correction.",
      buys:
        "State-of-the-art context extension using only a tiny fraction of the original pretraining tokens' worth of fine-tuning. The current standard method for stretching a RoPE model's context after the fact.",
      costs:
        "More implementation complexity than NTK-aware or linear interpolation (three bands plus a temperature term to tune). Still an extension, not native long-context training — the model was never actually trained at the extended length, so competence there is evidence, not a guarantee (the same gap the DroPE card below flags explicitly).",
      chooseWhen:
        "You have a model trained at a shorter context and need to responsibly stretch it further with a small fine-tuning budget, rather than paying for native long-context pretraining.",
    },

    {
      id: "attention-sinks",
      date: "2023-09-29",
      dateDisplay: "29 Sep 2023",
      threads: ["memory", "extend"],
      title: "Attention Sinks (StreamingLLM)",
      tagline: "Response to: fixed-size sliding-window caches catastrophically break once the earliest tokens are evicted.",
      sourceId: "streamingLLM2023",
      problem:
        "For genuinely unbounded streaming generation, a fixed-size sliding-window cache has to evict the oldest tokens to stay within budget — and once the very first few tokens are evicted, perplexity catastrophically explodes, even though those first tokens rarely carry meaningful content themselves.",
      mechanism:
        "Softmax always needs somewhere to dump attention mass it doesn't want to use elsewhere, and models learn to dump it on the first few tokens regardless of their content — these become “attention sinks.” Fix: always keep a small, fixed number of initial tokens in the cache permanently, alongside the sliding recent window, even after they would otherwise be evicted.",
      buys:
        "Stable perplexity for effectively unbounded-length streaming generation at a small, fixed KV-cache size — often with no fine-tuning required, since sink slots can sometimes be added post-hoc.",
      costs:
        "The kept sink tokens act purely as a pressure-release valve for attention, not a genuine long-term memory — content from the middle of a long-evicted history is still gone. Models trained without expecting this eviction pattern only partially benefit unless retrained with a dedicated sink token.",
      chooseWhen:
        "The deployment is a long-running or effectively infinite chat/streaming session with a hard memory budget, and stability over unbounded length matters more than perfect recall of everything said.",
    },

    {
      id: "mla",
      date: "2024-05-07",
      dateDisplay: "7 May 2024",
      threads: ["memory", "compress"],
      title: "MLA (Multi-Head Latent Attention)",
      tagline: "Response to: GQA lowers the KV-cache slope, but it's still a line that grows forever.",
      sourceId: "mla2024",
      problem:
        "GQA lowers the KV-cache slope by sharing heads, but cache size is still linear in context length, and it still stores full per-head-dimension keys/values for every retained group. At very long context or very large models, that's still a lot of memory per token.",
      mechanism:
        "Instead of caching full-width K/V per head (or per group), project K and V down into one small, shared, low-rank latent vector per token, and cache only that compressed latent. Reconstruct the full-size keys/values from it on the fly, with RoPE applied to a small decoupled slice to preserve position information.",
      buys:
        "A KV cache dramatically smaller than even GQA's at matched or better quality — DeepSeek-V2 reports performance exceeding standard MHA while needing a far smaller cache. Per-token storage is now a fixed, small latent width rather than something that scales with head count.",
      costs:
        "Meaningfully more architectural complexity — the low-rank projection/reconstruction machinery and the decoupled-RoPE trick don't exist in a plain MHA/GQA layer. Still linear in T, just with a much smaller constant in front of it.",
      chooseWhen:
        "You're building or serving a very large model where per-token KV-cache footprint is the dominant serving cost, and the extra architectural machinery to shrink it is worth paying for.",
    },

    {
      id: "delta-rule",
      date: "2024-06-10",
      dateDisplay: "10 Jun 2024",
      threads: ["recurrent"],
      title: "The Delta Rule / DeltaNet",
      tagline: "Response to: a fixed-size linear-attention state can only accumulate — it can never correct an old association.",
      sourceId: "deltaNet2024",
      problem:
        "Linear attention's fixed-size state (2020) can only accumulate — writing a new key-value association just adds to whatever is already there, so it can never revise or overwrite an outdated one. If a key currently returns 40 but should now return 55, a plain add-only write computes 40+55=95, which is simply wrong.",
      mechanism:
        "Before writing, read what the state currently returns for this key, compute the delta (the gap between that and the desired new value), and write only the correction: new_state = old_state + key ⊗ (value − old_state·key). A hardware-efficient parallel algorithm, built on products of Householder matrices, makes this practical to train at scale despite the update looking inherently sequential.",
      buys:
        "A fixed-size recurrent state that can genuinely be edited, not just grown — meaningfully better perplexity and downstream performance than plain linear attention or contemporaneous state-space baselines (Mamba) at matched scale.",
      costs:
        "More computation per step than a plain additive linear-attention write (a state-dependent correction, not a simple accumulation). Still a compressed, lossy summary of the past compared to exact softmax attention with a full KV cache.",
      chooseWhen:
        "You want RNN-like O(1) decoding memory but need the state to be genuinely revisable — e.g. as a fixed-state layer type mixed into a broader depth schedule alongside occasional exact-attention layers.",
      fixedBy: ["gated-delta-net"],
    },

    {
      id: "gated-delta-net",
      date: "2024-12-09",
      dateDisplay: "9 Dec 2024",
      threads: ["recurrent"],
      title: "Gated DeltaNet",
      tagline: "Response to: the delta rule can correct old state, but it never forgets any of it.",
      sourceId: "gatedDeltaNet2024",
      problem:
        "The plain delta rule can correct old associations, but has no notion of forgetting — irrelevant old state sticks around indefinitely, with no way to say “this part of memory is unimportant now, decay it,” the way state-space models like Mamba2's gating already could.",
      mechanism:
        "Add a learned, per-step gating term that scales how much of the existing state is retained before the delta-rule correction is applied — combining Mamba2-style adaptive forgetting with DeltaNet-style targeted correction in one update rule.",
      buys:
        "Consistently surpasses both plain Mamba2 (which forgets but can't precisely correct) and plain DeltaNet (which corrects but can't decay) on language modeling, in-context retrieval and length-extrapolation benchmarks.",
      costs:
        "One more learned gating mechanism to tune per layer/head. Still a compressed fixed-size state — no amount of gating recovers information about a specific old token once the state has actually discarded it.",
      chooseWhen:
        "Choosing among modern fixed-state (“linear”/recurrent) layer types for a hybrid depth schedule — currently one of the strongest available options in that family.",
    },

    {
      id: "deepseek-nsa",
      date: "2025-02-16",
      dateDisplay: "16 Feb 2025",
      threads: ["sparse", "compress", "systems"],
      title: "DeepSeek's Compressed Sparse Attention (NSA)",
      tagline: "Response to: sparse attention and KV-cache compression each cut one cost, but rarely both at once and rarely at real hardware speed.",
      sourceId: "nsa2025",
      problem:
        "Sparse attention (2019) and MLA-style compression (2024) each attack one cost, but most sparse-attention systems were either retrofit onto an already-pretrained dense model (not trained sparse from scratch) or not aligned with how GPUs actually execute — leaving real wall-clock speedups on the table. Separately, top-k selection still needs a cheap way to find good candidate blocks without first scoring everything densely (the exact gap flagged back at the 2019 sparse-attention card).",
      mechanism:
        "Combine three attention branches per query: (1) compressed attention over block summaries of the whole history — coarse, cheap, always on; (2) selected attention, which uses a small, cheap low-rank indexer to pick the top-k most relevant compressed blocks and re-reads their real tokens; and (3) a local sliding window for nearby precision. Trained natively from scratch — not bolted on after pretraining — with kernels designed around real GPU arithmetic intensity.",
      buys:
        "Long-context training and inference that is both algorithmically sparse and genuinely hardware-efficient — measured wall-clock speedups, not just fewer FLOPs on paper — while matching or beating full dense attention on downstream benchmarks. The low-rank indexer makes candidate proposal itself cheap, closing the “naive top-k still scores everything” gap.",
      costs:
        "Compression trades away token-level detail — several tokens now share one block summary. Top-k selection is still approximate; a genuinely useful key living in a low-ranked block can be missed. Three attention branches is meaningfully more complex to implement and train than one dense or one sparse call.",
      chooseWhen:
        "Training a large model natively for very long context from scratch, where both training-time compute and serving-time memory need to be sparse-cheap simultaneously, and the extra system complexity is worth the throughput.",
    },

    {
      id: "drope",
      date: null,
      dateDisplay: "unverified — course-internal only",
      threads: ["extend"],
      title: "DroPE",
      tagline: "Flagged, not asserted. Response to (as recorded): stretching a short-trained model's context 32× without native long-context pretraining.",
      sourceId: "drope_unverified",
      isUnverified: true,
      problem:
        "As recorded in this course's own material: a model trained at a short context (8K tokens) needs to serve a much longer one (256K) cheaply, without paying for native long-context pretraining.",
      mechanism:
        "As recorded, and explicitly not fully published: “positional recalibration, applied before annealing” — described only as a training-time step, not as a fully specified algorithm. No independently reviewable formula, code, or paper is available to check what it actually changes.",
      buys:
        "As recorded: a reported 32× context extension (8K → 256K) for one specific model and training run.",
      costs:
        "This is the one card on this page with no checkable primary source. The class notes are explicit that this is evidence for one model and procedure, not a general method — the exact algorithm and which rotary dimensions it touches were never disclosed. There is also a real, unrelated, published paper with an almost-identical name — “DRoPE: Directional Rotary Position Embedding” (arXiv:2503.15029, Mar 2025) — for agent-trajectory modeling, a completely different problem domain. They are not the same technique; conflating them would be exactly the kind of confident-but-wrong claim this assignment warns about.",
      chooseWhen:
        "Not applicable outside its own reported run. Treat any claim about “DroPE” as unverified until an external primary source (paper, released code, or model card) actually exists.",
      footnote:
        "This card is included because the assignment's minimum list names it — dropping it silently would look like an oversight rather than a deliberate, disclosed gap. The honest move is to keep the slot and say plainly what is and isn't known, which is what the professor's own warning about Session 7 was asking for.",
    },
  ];

  return { REF_CONFIG, THREADS, SOURCES, TIMELINE };
})();
