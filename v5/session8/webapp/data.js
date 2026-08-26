/* ============================================================
   HOW DOES ATTENTION WORK NOW?
   All content: a chronological account of every major attention
   mechanism, sorted by verified primary-source publication date.
   Every date was checked against the source's own submission
   record directly — not taken on trust from a model's memory.
   ============================================================ */

window.SESSION_DATA = (function () {

  /* Reference serving config used by the cost calculator further down.
     A fairly ordinary mid-size dense model: 48 layers, 8 KV heads,
     128-wide heads, stored in bf16. Nothing here is tuned to make any
     particular technique look better or worse. */
  const REF_CONFIG = {
    layers: 48,
    kvHeads: 8,
    headDim: 128,
    bytesPerNumber: 2, // bf16
  };

  const THREADS = [
    { id: "compute",   label: "Compute Cost",       color: "#22d3ee" },
    { id: "memory",    label: "Cache Memory",        color: "#f59e0b" },
    { id: "position",  label: "Position Handling",   color: "#a78bfa" },
    { id: "extend",    label: "Context Extension",   color: "#f43f5e" },
    { id: "recurrent", label: "Recurrent State",     color: "#10b981" },
    { id: "sparse",    label: "Sparsity",            color: "#6366f1" },
    { id: "compress",  label: "Compression",         color: "#06b6d4" },
    { id: "systems",   label: "Systems / I-O",       color: "#94a3b8" },
  ];

  /* Bibliography — one entry per primary source. Dates are the source's
     own v1 submission date (fetched directly from its own record), or
     the best available primary date where no preprint exists. */
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
    drope2025: {
      authors: "Gelberg, Eguchi, Akiba, Cetin",
      title: "Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings",
      venue: "Sakana AI, arXiv preprint",
      date: "2025-12-13",
      url: "https://arxiv.org/abs/2512.12167",
      id: "arXiv:2512.12167",
    },
  };

  /* The timeline: sorted strictly by primary-source date. Each entry
     optionally carries an `era` marker rendered as a short interlude
     before it, to give the sequence some breathing room and pace. */
  const TIMELINE = [
    {
      id: "baseline",
      isBaseline: true,
      date: "2017-06-12",
      dateDisplay: "12 Jun 2017",
      threads: ["compute", "memory"],
      title: "Scaled Dot-Product Attention",
      tagline: "The starting point. Nothing to respond to yet — everything else on this page responds to this.",
      sourceId: "vaswani2017",
      era: {
        title: "Where the story starts",
        blurb: "Before this, sequence models read one token at a time and passed a summary forward — useful, but slow to train and forgetful over distance. This is the idea that replaced that summary with something much more direct: let every token look at every other token, all at once.",
      },
      problem:
        "Recurrent networks process a sentence one word at a time, carrying a single running summary forward. That summary is a bottleneck twice over: training can't parallelize across the sequence (step 50 needs step 49's result), and anything that happened many steps back has to survive being repeatedly compressed into that one summary vector, which in practice means it often doesn't.",
      mechanism:
        "Give every token three separate roles: a query (what it's looking for), a key (what it advertises about itself), and a value (what it actually hands over). Compare each query against every key with a dot product to get a compatibility score, divide by √d_k to keep the numbers from blowing up, optionally block off illegal future positions, turn the row of scores into a probability distribution with softmax, and use those probabilities to blend the values together. Q×K → scores → scale → mask → softmax → weighted sum of V.",
      buys:
        "Any two tokens, however far apart, are one dot product away from each other — no information has to survive a long relay. And because there's no step-by-step recurrence, the whole layer is a matrix multiplication that trains in parallel across the full sequence.",
      costs:
        "Comparing everything with everything means the score matrix is T×T — compute and memory both grow quadratically with sequence length. And once a model starts generating text one token at a time, it has to keep every previous token's key and value on hand to avoid recomputing them, which is its own, separately growing cost. Also: nothing here encodes order. Swap two tokens' positions and their queries, keys and values don't change at all.",
      chooseWhen:
        "Sequences are short enough that quadratic cost isn't yet painful, and training speed and unrestricted token-to-token access matter more than anything else.",
      widget: "core-mechanism",
    },

    {
      id: "sinusoidal",
      date: "2017-06-12",
      dateDisplay: "12 Jun 2017 · same paper",
      threads: ["position"],
      title: "Sinusoidal Position Signal",
      tagline: "A fix for the very last line above: attention has no idea what order anything came in.",
      sourceId: "vaswani2017",
      problem:
        "A dot product compares content, not position. If the word at position 3 and the word at position 30 happen to be identical, they produce identical queries, keys and values — attention genuinely cannot tell them apart unless something external tells it where each token sits.",
      mechanism:
        "Before the first layer even runs, add a fixed pattern of sine and cosine waves — one pair of waves per pair of embedding dimensions, each pair oscillating at a different frequency — directly onto every token's embedding, keyed off its position in the sequence. No parameters, no training: the position 5 vector is always the position 5 vector.",
      buys:
        "A position signal that costs nothing to learn and is defined for any position you plug into the formula. Different frequencies act like clock hands moving at different speeds, which in principle gives the network enough structure to work out relative offsets on its own.",
      costs:
        "The signal only enters once, at the bottom of the network — turning it into something the attention score actually reacts to has to be learned indirectly, layer by layer. And the original authors' own side-by-side comparison found a plain trainable position table did just as well, which undercuts the case for the hand-designed formula.",
      chooseWhen:
        "You want a zero-parameter position scheme and don't need the attention score itself to explicitly reflect relative distance. Largely a historical footnote today.",
      footnote:
        "That side-by-side comparison — sinusoidal vs. a trainable table — was run in this very paper. The trainable version just wasn't what shipped as the headline design; it became the default a year later (next).",
    },

    {
      id: "absolute-learned",
      date: "2018-06",
      dateDisplay: "Jun–Oct 2018",
      threads: ["position"],
      title: "A Position Table You Just Train",
      tagline: "Why hand-design the position signal at all, when you can let gradient descent find one?",
      sourceId: "gpt1_2018",
      secondarySourceIds: ["bert2018"],
      problem:
        "The sine-wave recipe works, but it's exactly that — a recipe someone picked. Once large pretrained models needed a default choice, treating position the same way tokens themselves are treated — as a row in a trainable lookup table — was the simpler engineering decision, and the earlier comparison had already shown it wasn't a worse one.",
      mechanism:
        "Allocate one trainable vector per position, up to some maximum sequence length, and add it to the token embedding at that position. It trains end to end with everything else, no special formula involved.",
      buys:
        "One less piece of hand-designed math in the architecture. It matches or slightly beats the sine-wave version up to whatever length it was trained on, and can shape itself to whatever positional pattern the data actually rewards.",
      costs:
        "The table has a last row. Ask the model about position N+1 and there's nothing there — a hard wall, not a graceful decline. And there's still no built-in sense of distance: position 5 and position 300 are two unrelated rows that happen to sit near each other in memory, nothing more.",
      chooseWhen:
        "The maximum sequence length is fixed and known in advance and you'll never need to go past it — short, bounded-length inputs are the comfortable case.",
      fixedBy: ["rope", "alibi"],
    },

    {
      id: "sparse-topk",
      date: "2019-04-23",
      dateDisplay: "23 Apr 2019",
      threads: ["compute", "sparse"],
      title: "Stop Looking at Everything",
      tagline: "The first serious answer to the quadratic bill, once people tried feeding in truly long sequences.",
      sourceId: "sparseTransformer2019",
      era: {
        title: "2019 — two different teams, two different walls",
        blurb: "Within the same seven months, one group ran into the cost of comparing every token to every other token, and a different group ran into the cost of remembering every token during generation. Neither problem had been urgent at short context; both became unavoidable as soon as sequences got long.",
      },
      problem:
        "Feed in tens of thousands of tokens — raw image pixels, long documents, audio — and the T×T score matrix stops being a rounding error and starts being the whole budget. Double the sequence, quadruple the work.",
      mechanism:
        "Instead of letting every query see every key, restrict each query to a small, fixed pattern of keys — say, nearby positions plus a handful spaced at regular intervals — chosen so that information can still cross the whole sequence in a couple of hops through the pattern, even though no single layer sees everything. This is the ancestor of the broader idea of only ever scoring a handful of candidate keys per query, rather than all of them.",
      buys:
        "Sequences tens of thousands of tokens long become something you can actually train on, at a fraction of the compute dense attention would need.",
      costs:
        "The pattern of who-looks-at-whom is fixed by the architecture, not learned or adapted to content. And a naive version of \"only look at the top few keys\" still has to score every candidate before it can throw most of them away — if scoring everything was the expensive part to begin with, that cost hasn't actually gone anywhere.",
      chooseWhen:
        "Context is long and has enough regular structure — a grid, a raster, a predictable stride — that a fixed sparse pattern still reaches what matters. Less useful when the tokens that matter could be anywhere.",
      fixedBy: ["deepseek-nsa"],
    },

    {
      id: "mqa",
      date: "2019-11-06",
      dateDisplay: "6 Nov 2019",
      threads: ["memory"],
      title: "One Set of Keys and Values for Everyone",
      tagline: "A different 2019 team hit a different wall: remembering, not comparing.",
      sourceId: "mqa2019",
      problem:
        "Generate text one token at a time and every layer has to hang onto every earlier token's key and value — recomputing them from scratch each step would be wasteful. With a separate key/value pair per attention head, that running cache gets large fast, and simply reading it back off memory starts to dominate how long each generated token takes.",
      mechanism:
        "Keep all the query heads — they can still ask different kinds of questions — but collapse the keys and values down to a single shared pair, used by every head. Only one key/value projection needs to be computed and stored per layer, not one per head.",
      buys:
        "The cache shrinks by roughly the number of heads you'd otherwise have kept separately — an 8-head model's cache drops close to 8×. Generation becomes far less bottlenecked on shuffling memory around, especially with many requests running at once.",
      costs:
        "Every head is now searching the same shared keys and values, so whatever benefit came from heads specializing on different things is largely gone — and that shows up as a measurable quality gap against keeping separate heads.",
      chooseWhen:
        "Serving many requests cheaply and quickly matters more than squeezing out the last bit of quality, and retraining (or briefly adapting an existing model) to this scheme is an option.",
      fixedBy: ["gqa"],
    },

    {
      id: "sliding-window",
      date: "2020-04-10",
      dateDisplay: "10 Apr 2020",
      threads: ["compute", "sparse"],
      title: "Attend Locally, Escalate Rarely",
      tagline: "A more deliberate shape for the sparse pattern above, once the target became long documents specifically.",
      sourceId: "longformer2020",
      era: {
        title: "2020 — same target, two different bets",
        blurb: "One idea bets that most of what a token needs is nearby, plus a few designated important tokens. The other bets that the past doesn't need to be kept at all — just summarized into a fixed-size running state. Neither is obviously right; both get taken seriously.",
      },
      problem:
        "The strided patterns above cut cost generally, but a lot of document-scale tasks have a more specific shape: most of what a token needs is close by, plus a small number of genuinely global anchor tokens (a summary token, a question). A generic stride doesn't know that.",
      mechanism:
        "Give every token a narrow window of nearby positions to attend to — widened at deeper layers so the effective reach still grows with depth — plus a small, task-chosen set of tokens that get to see, and be seen by, everyone. Cost now scales with sequence length times window size, not sequence length squared.",
      buys:
        "Documents thousands of tokens long become tractable on ordinary hardware, while depth still lets information eventually travel further than any single window.",
      costs:
        "Inside one layer, a token genuinely cannot see past its window — anything further away has to relay through several layers or the designated global tokens. Pick too narrow a window for the task and the model is architecturally blind to context it actually needed.",
      chooseWhen:
        "The task's useful context is mostly local, with a few known anchor points — documents, source code, audio. Weaker fit when relevant information could be anywhere in the sequence with equal likelihood.",
      footnote:
        "This exact idea resurfaced in production years later, paired with the head-sharing trick two cards up — see the note on the attention-sinks card below for what that combination ran into.",
      fixedBy: ["attention-sinks"],
    },

    {
      id: "linear-attention",
      date: "2020-06-29",
      dateDisplay: "29 Jun 2020",
      threads: ["compute", "memory", "recurrent"],
      title: "What If the Past Were Just a Running Total?",
      tagline: "A more radical bet than a smaller pattern: don't keep the past around at all — compress it as it arrives.",
      sourceId: "linearAttention2020",
      problem:
        "Every version so far still keeps an exact, indexed record of some subset of past tokens. None of them ask whether attention could instead carry a single fixed-size running state forward, the way an old-fashioned recurrent network does — trading exactness for a memory footprint that never grows.",
      mechanism:
        "Swap the softmax step for a simpler feature transform, and a piece of arithmetic falls out for free: because nothing ties the scores together into a shared denominator anymore, each key/value pair's contribution can be pre-combined into one running matrix as tokens arrive — before any query even shows up — and a new query just reads that matrix. Per-token cost during generation drops to a constant, independent of how much history exists.",
      buys:
        "A state whose size never grows with sequence length, for both compute and memory. Generating each new token becomes O(1) work regardless of how long the conversation already is — an RNN's cheap decoding, with attention's fully parallel training.",
      costs:
        "Softmax's built-in competition between keys — one gaining weight necessarily costs another — is gone; what's left is a lossy running summary rather than an exact per-key memory. And a plain running total can only add new information in, never revise something it already committed to.",
      chooseWhen:
        "Streaming or generation needs to run for an effectively unbounded length at fixed memory, and an approximate, compressed memory of everything earlier is an acceptable trade.",
      fixedBy: ["delta-rule"],
    },

    {
      id: "rope",
      date: "2021-04-20",
      dateDisplay: "20 Apr 2021",
      threads: ["position", "extend"],
      title: "Encode Position as a Rotation",
      tagline: "A cleaner fix for the position problem than either the sine waves or the trainable table managed.",
      sourceId: "rope2021",
      era: {
        title: "2021 — position gets rebuilt from scratch",
        blurb: "Both existing position schemes bake a position into a vector once, at the bottom of the network, and hope later layers can recover relative distance from it. Two very different ideas this year both try to make the score itself carry that distance directly.",
      },
      problem:
        "Both earlier schemes bake a position into the vector once, at the very bottom of the network, and leave it to later layers to indirectly reconstruct anything relative from that. The trainable version also can't be asked about a position it never allocated a row for.",
      mechanism:
        "Instead of adding a position vector, rotate the query and key vectors themselves — pair up their dimensions, treat each pair as a 2D point, and spin it by an angle proportional to the token's position, at several different speeds across the different pairs. A dot product only cares about the angle between two vectors, so when both are rotated, their individual, absolute rotations cancel out and only the gap between their positions survives inside the score.",
      buys:
        "Relative position becomes a property of the score itself, not something later layers have to infer. The rotation formula is defined at any position you plug in — there's no table to run out of rows. And shifting an entire sequence forward changes nothing about how nearby tokens relate to each other.",
      costs:
        "Being computable at a distant position isn't the same as the network having actually learned to behave well there — a rotation angle far outside anything seen in training is still new territory, whatever the formula says. This exact gap is what every extension method further down this page exists to address. There's also a small, constant rotation cost added to every layer.",
      chooseWhen:
        "You want the score to explicitly reflect relative distance with no hard length ceiling built into the mechanism, tuned around your intended training length — the default starting point for most current models.",
      fixedBy: ["alibi", "ntk-aware", "yarn", "drope"],
    },

    {
      id: "alibi",
      date: "2021-08-27",
      dateDisplay: "27 Aug 2021",
      threads: ["position", "extend"],
      title: "Skip Position Entirely, Just Penalize Distance",
      tagline: "The same year's second answer: what if the simplest possible fix generalizes better than the elegant one?",
      sourceId: "alibi2021",
      problem:
        "Both earlier vector-based position schemes extrapolate badly once you ask a model to run longer than it trained. Rotation is at least mathematically defined out there, but whether the model actually behaves sensibly at those distances is still an open question, and it comes with real machinery to implement.",
      mechanism:
        "Don't encode position anywhere near the vectors at all. Instead, before softmax, directly subtract a penalty from each raw attention score, proportional to how far apart the query and key are — with a different fixed steepness per head, never learned, never adjusted.",
      buys:
        "A model trained on short sequences reliably keeps working on much longer ones with no extra fine-tuning at all. No new learned parameters, and about as few moving parts as a position scheme can have.",
      costs:
        "That distance penalty is a fixed opinion, not something the network can override — a task that genuinely needs to weight something far away above something nearby is fighting the mechanism itself. In practice, most current large models chose the rotation-based approach plus a separate extension method instead.",
      chooseWhen:
        "Reliable behavior far past the training length matters more than architectural flexibility, or you want the smallest possible number of moving parts.",
    },

    {
      id: "flash-attention",
      isBonus: true,
      date: "2022-05-27",
      dateDisplay: "27 May 2022",
      threads: ["systems", "compute"],
      title: "Make the Exact Version Faster Instead",
      tagline: "Not on any required list — but the story has a gap without it. A detour through hardware, not algorithms.",
      sourceId: "flashAttention2022",
      era: {
        title: "2022 — a year that isn't about the algorithm at all",
        blurb: "Everything before this either approximated the math or changed it outright. This one changes neither — it just refuses to accept that exact attention has to be slow.",
      },
      problem:
        "Every approximate method above works around the quadratic cost by changing what gets computed. But naive exact attention was slower than its own arithmetic should require, because it was writing the entire T×T score matrix out to slow accelerator memory and reading it straight back — the bottleneck was data movement, not the multiplications themselves.",
      mechanism:
        "Break Q, K and V into blocks small enough to fit in an accelerator's fast on-chip memory, run softmax incrementally as those blocks stream through, and never materialize the full T×T matrix in slow memory at all. Recompute what's needed during the backward pass instead of storing it.",
      buys:
        "The identical output exact attention always produced — no approximation anywhere — at a fraction of the wall-clock time and memory, simply by fixing where the bottleneck actually was.",
      costs:
        "The number of arithmetic operations doesn't change — this is a large constant-factor win, not a new complexity class. And the specific implementation is tied closely to how a given accelerator's memory hierarchy works, which means porting it isn't free.",
      chooseWhen:
        "Essentially always, whenever exact attention is being run at all — by now this sits underneath most of the other mechanisms on this page rather than competing with them.",
    },

    {
      id: "gqa",
      date: "2023-05-22",
      dateDisplay: "22 May 2023",
      threads: ["memory"],
      title: "A Dial Between the Two Extremes",
      tagline: "Four years after the memory problem first appeared, someone put a knob on the trade-off instead of picking one end of it.",
      sourceId: "gqa2023",
      era: {
        title: "2023 — the year everything happens at once",
        blurb: "Four separate ideas land within five months of each other, each attacking a different weak point that had been quietly accumulating: cache size, extrapolation past training length, and what happens when a fixed window runs forever.",
      },
      problem:
        "Sharing keys and values down to a single pair saves the most cache but costs the most quality; keeping a full separate pair per head keeps all the quality but none of the savings. Nothing sat in between.",
      mechanism:
        "Split the query heads into a handful of groups, and let each group share one key/value pair — one group recovers full sharing, as many groups as heads recovers no sharing at all. An existing full-head model can be cheaply converted by averaging its existing key/value heads within each new group, rather than starting over.",
      buys:
        "A dial, not a binary choice — a modest number of groups gives most of the cache savings with much less quality lost than collapsing to one pair. This became the practical default across essentially every open large model shipped afterward.",
      costs:
        "The cache still grows with sequence length; grouping only changes the slope of that line, not the fact that it keeps climbing. At sufficiently long context, it still eventually dominates memory.",
      chooseWhen:
        "Almost always, for any model balancing serving cost against quality — the live question by this point is how many groups, not whether to group.",
      fixedBy: ["mla"],
    },

    {
      id: "ntk-aware",
      date: "2023-06",
      dateDisplay: "Jun 2023",
      threads: ["extend", "position"],
      title: "Stretch the Rotation, Unevenly",
      tagline: "Not a paper — a forum post that changed how everyone thought about pushing rotation past its training length.",
      sourceId: "ntkAware2023",
      isCommunitySource: true,
      problem:
        "The obvious way to reach a longer context with rotation-based position is to squeeze every position index by a fixed factor before rotating. It works, but it squeezes the fine-grained, high-frequency rotations the model actually relies on to tell adjacent tokens apart — local precision gets damaged in exchange for reach.",
      mechanism:
        "Instead of rescaling every position by the same factor, change the base of the rotation formula itself, so the slow-moving (long-range) components get stretched much more than the fast-moving (local) ones — an uneven adjustment rather than a uniform squeeze.",
      buys:
        "Meaningfully less damage to nearby-token precision than the uniform squeeze, and it needs no retraining at all to get a real improvement.",
      costs:
        "It's still one global formula applied to every frequency at once — coarser than treating each frequency band on its own terms, which shows up as real quality loss once the extension factor gets aggressive.",
      chooseWhen:
        "A fast, no-retraining context stretch is worth a partial quality hit. Mostly displaced a few months later by the more careful version below.",
      footnote:
        "There is no peer-reviewed paper behind this one — the original source is a public forum post. It earns a place here anyway because the more formal method that followed explicitly builds on and credits it, and leaving out ideas that started outside a journal would misrepresent how this particular thread of the story actually moved.",
      fixedBy: ["yarn"],
    },

    {
      id: "yarn",
      date: "2023-08-31",
      dateDisplay: "31 Aug 2023",
      threads: ["extend", "position"],
      title: "Three Bands Instead of One Formula",
      tagline: "The forum idea, formalized and sharpened two months later.",
      sourceId: "yarn2023",
      problem:
        "Stretching the rotation base unevenly beats a uniform squeeze, but it's still one rule applied everywhere. Some frequency bands need real stretching, some barely need any, and the ones in between need a smooth handoff rather than a hard edge — and stretching positions also quietly shifts how sharp or diffuse softmax's output ends up being.",
      mechanism:
        "Split the rotation's frequency spectrum into three regions: leave the fastest, most local frequencies alone entirely, stretch the slowest, most global ones the way the uneven method above does, and blend smoothly through a middle band — plus a small correction to softmax's effective sharpness.",
      buys:
        "State-of-the-art context extension for a small fraction of what the original training run cost — this became, and largely still is, the standard way to stretch a rotation-based model's context after the fact.",
      costs:
        "More moving parts than either earlier method — three bands and a temperature correction to get right. And it's still an extension of a shorter training run, not the genuine article: the model was never actually trained at the longer length, so how well it performs there is evidence, not a guarantee.",
      chooseWhen:
        "A model already trained at a shorter context needs to responsibly reach further, and a small fine-tuning budget is available rather than the cost of training long from the start.",
    },

    {
      id: "attention-sinks",
      date: "2023-09-29",
      dateDisplay: "29 Sep 2023",
      threads: ["memory", "extend"],
      title: "Keep the First Few Tokens Forever",
      tagline: "A fix for a specific, ugly failure mode: what a fixed local window does when a conversation runs forever.",
      sourceId: "streamingLLM2023",
      problem:
        "A model serving a genuinely endless stream — a long-running chat, a live transcript — using a fixed local window has to evict the oldest tokens to stay within budget. The moment the very first few tokens get evicted, output quality falls off a cliff, even though those first tokens rarely carried any content worth remembering.",
      mechanism:
        "Softmax always has to put its attention mass somewhere, even when nothing in the current window is truly relevant — and it turns out models learn to dump that unwanted mass onto the first few tokens by default, regardless of what they contain. So: keep a small, fixed number of the very first tokens permanently in the cache, in addition to the sliding window of recent ones, and never evict them.",
      buys:
        "Stable output over an effectively unlimited stream at a small, fixed memory budget — and it often works without retraining at all, since it's really just changing what gets evicted.",
      costs:
        "Those pinned tokens function as a release valve for attention, not a real memory — anything from the middle of a long-since-evicted stretch is genuinely gone. Models trained with no expectation of this eviction pattern benefit only partially unless they're retrained with it in mind.",
      chooseWhen:
        "The deployment is a long-running or effectively endless session under a hard memory budget, where staying stable matters more than recalling everything that was ever said.",
      footnote:
        "Two weeks after this shipped, a widely-used open model shipped its own local-window attention paired with the head-grouping trick from earlier — without adopting this fix. A good reminder that a solved problem and a widely-deployed solution aren't the same event.",
    },

    {
      id: "mla",
      date: "2024-05-07",
      dateDisplay: "7 May 2024",
      threads: ["memory", "compress"],
      title: "Cache a Compressed Summary, Not the Real Thing",
      tagline: "The head-grouping dial lowered the slope of the cache-growth line. This attacks the line's height instead.",
      sourceId: "mla2024",
      era: {
        title: "2024 — memory gets more literal, and starts learning to forget",
        blurb: "One idea this year asks whether the cache itself needs to store real keys and values, or just enough to reconstruct them. A different idea, a month later, asks whether a running state needs to only ever accumulate, or whether it can actually correct itself.",
      },
      problem:
        "Grouping query heads onto shared keys and values lowers how fast the cache grows with sequence length, but it's still storing full-width keys and values for whatever's left after grouping — and cache size is still, fundamentally, linear in context length.",
      mechanism:
        "Rather than caching full-size keys and values at all, compress each token down into one small, shared low-rank vector and cache only that. Reconstruct the full-size keys and values from the compressed vector on the fly when they're actually needed, keeping a small separate slice for position information.",
      buys:
        "A cache meaningfully smaller than even the grouped version, at matched or better quality — the model that introduced this reported beating full, ungrouped attention on benchmarks while needing a far smaller cache to do it.",
      costs:
        "Real added complexity — the compress-then-reconstruct machinery, plus a separate way of handling position, don't exist in a simpler design. And it's still linear in sequence length, just with a much smaller constant multiplying it.",
      chooseWhen:
        "Serving cost per token is the dominant expense at the scale you're operating at, and the extra architectural machinery to shrink it is worth the engineering cost.",
    },

    {
      id: "delta-rule",
      date: "2024-06-10",
      dateDisplay: "10 Jun 2024",
      threads: ["recurrent"],
      title: "Teach the Running State to Correct Itself",
      tagline: "The fixed-size running state from 2020 could grow. It still couldn't ever change its mind.",
      sourceId: "deltaNet2024",
      problem:
        "A running-total state can only ever add new contributions in — it has no way to revise something it already committed to. Concretely: if a stored association currently reports 30 for some key, but the right answer is now 70, adding the new value on top gives 100, which is simply wrong. What's needed is a way to overwrite, not just accumulate.",
      mechanism:
        "Before writing anything, read what the state currently returns for a given key, work out the gap between that and the value you actually want it to return, and write only that gap. If the state returns 30 and should return 70, the correction is 40, and 30 + 40 lands exactly on 70 — not 30 + 70. A hardware-friendly algorithm, built around products of simple reflection matrices, makes this practical to train at real scale despite looking sequential on paper.",
      buys:
        "A fixed-size state that can genuinely be edited, not just grown — a meaningful jump in quality over a plain running-total state and over other constant-memory baselines from around the same time, at matched scale.",
      costs:
        "More arithmetic per step than a plain accumulation, since every write now requires a read-and-compare first. And it's still a compressed summary of everything that happened — nowhere near as exact as keeping the real keys and values around.",
      chooseWhen:
        "Constant-memory, RNN-style decoding is the goal, but the state genuinely needs to be revisable rather than purely additive — often as one layer type mixed into a design that also keeps some exact-attention layers.",
      fixedBy: ["gated-delta-net"],
    },

    {
      id: "gated-delta-net",
      date: "2024-12-09",
      dateDisplay: "9 Dec 2024",
      threads: ["recurrent"],
      title: "Now Teach It to Forget, Too",
      tagline: "Correcting old information is one skill. Deciding it no longer matters at all is a different one.",
      sourceId: "gatedDeltaNet2024",
      problem:
        "The correction mechanism above can fix a wrong association, but it has no notion of an association simply becoming irrelevant — there's no way to say \"discount this part of the state,\" the way other constant-memory designs built around explicit forgetting already could.",
      mechanism:
        "Add a learned, per-step gate that scales down how much of the existing state survives before the correction is applied — combining an adaptive-forgetting mechanism with the targeted correction from the previous idea, in one update.",
      buys:
        "Consistently outperforms both a purely-forgetting design and a purely-correcting one on language modeling, in-context retrieval, and extrapolation to longer sequences than trained on.",
      costs:
        "One more learned mechanism to get right per layer or head. And no amount of gating brings back information the state has already actually discarded — forgetting is still forgetting.",
      chooseWhen:
        "Choosing among constant-memory layer designs for a mixed architecture — currently one of the strongest options in that family.",
    },

    {
      id: "deepseek-nsa",
      date: "2025-02-16",
      dateDisplay: "16 Feb 2025",
      threads: ["sparse", "compress", "systems"],
      title: "Sparsity Comes Back, Built for the Hardware",
      tagline: "The 2019 idea — look at fewer keys — returns, but this time trained in from the start and built around how accelerators actually move data.",
      sourceId: "nsa2025",
      era: {
        title: "2025 — sparsity, taken seriously again",
        blurb: "Skipping most of the sequence was the very first idea in this story, back in 2019. It never went away, but it also never fully solved its own founding problem: finding the few keys worth reading is supposed to be cheap, and for years it mostly wasn't.",
      },
      problem:
        "Looking at fewer keys and compressing the cache each solve one cost, but most earlier sparse-attention designs were bolted onto an already-trained dense model rather than trained sparse from the start, and weren't built with real accelerator memory movement in mind — leaving real speed on the table even where the arithmetic looked cheaper. And the founding problem from 2019 was still open: proposing a short list of good candidate keys is supposed to be cheap, but naive proposal methods still have to look at everything first.",
      mechanism:
        "Give every query three parallel ways to read the past: a cheap, always-on pass over compressed summaries of blocks of history; a more expensive pass that re-reads the real tokens inside only the top few blocks, chosen by a small, cheap scoring network rather than by scoring everything in full; and a local window for immediate neighbors. Trained sparse from the very first step, with the low-level kernels built around the accelerator's actual memory bandwidth.",
      buys:
        "Long-context training and serving that's sparse in a way that translates into measured wall-clock speed, not just a smaller number on paper — while matching or beating dense attention on downstream benchmarks. The cheap scoring network genuinely closes the old \"proposing candidates still costs as much as scoring everything\" gap.",
      costs:
        "Summarizing blocks of tokens loses token-level detail by construction. The candidate selection is still approximate — a genuinely useful key sitting in a block that didn't make the shortlist can be missed entirely. And three parallel read paths are a meaningfully bigger system to implement and train correctly than one dense or one sparse call.",
      chooseWhen:
        "Training a large model natively for very long context, where both training compute and serving memory need to be cheap at the same time, and the added system complexity is worth the throughput it buys.",
    },

    {
      id: "drope",
      date: "2025-12-13",
      dateDisplay: "13 Dec 2025",
      threads: ["extend", "position"],
      title: "Train With Rotation, Then Take It Away",
      tagline: "Four years after rotation-based position first appeared, someone asked whether it needs to stick around forever.",
      sourceId: "drope2025",
      era: {
        title: "2025, ten months later — the thread closes",
        blurb: "Every rotation-stretching method in this story shares one unavoidable piece of math: to keep the rotation's phase inside familiar territory at a longer length, the slow, long-range frequencies have to be compressed hardest — exactly the frequencies that content-based, \"what does this token mean\" attention relies on. This is the first idea that doesn't try to stretch the rotation more carefully. It removes it.",
      },
      problem:
        "Every rescaling method above — the uneven stretch, the three-band version — runs into the same wall, and it isn't a tuning mistake, it's arithmetic: to keep the rotation's phase inside the range the model actually trained on, the slow-moving frequencies have to be compressed by roughly the same factor as the context extension itself. Those are exactly the frequencies long-range, content-based attention leans on, so every rescaling method, however carefully tuned, ends up quietly distorting the attention patterns it's trying to extend.",
      mechanism:
        "Train the model with rotation-based position as usual for most of pretraining — rotation genuinely helps a model learn positional structure quickly, and skipping it from the start makes early training measurably slower and worse, since attention heads have no shortcut for developing directional bias. Then, once training is mostly finished, remove the positional rotation from the architecture entirely, converting the model to one with no explicit position mechanism at all, and run a short recalibration at the original, short training length — no long-sequence data required. With no rotation left to keep in-distribution, there's no phase to compress and nothing left to warp.",
      buys:
        "Zero-shot context extension that beats the rescaling methods above on long-context retrieval benchmarks, for a recalibration budget as small as roughly half a percent to a few percent of the original training cost, depending on model size — far cheaper than training long from scratch.",
      costs:
        "This isn't a replacement for training with rotation — a model trained with no position mechanism from the very first step converges far more slowly and performs worse throughout training, since it has nothing to lean on early. So this only works as a scheduled hand-off: rotation first, then remove it. It also needs a real, if short, recalibration pass, and at larger recalibration budgets needed extra normalization to stay numerically stable.",
      chooseWhen:
        "A model already trained with rotation-based position needs to reliably serve far beyond its original context, and a small recalibration bill is preferable to either a rescaling method's quality loss or the cost of training long from the very beginning.",
      footnote:
        "Worth flagging by name, since the resemblance is easy to trip over: there is a separate, unrelated paper with an almost identical name for rotary embeddings in autonomous-agent trajectory modeling — different authors, different field, no connection to context extension at all. Getting the two confused would be exactly the kind of mistake that's easy to make with quiet confidence and never notice.",
    },
  ];

  return { REF_CONFIG, THREADS, SOURCES, TIMELINE };
})();
