/* ============================================================
   HOW DOES ATTENTION WORK NOW?
   A chronological account of every major attention mechanism,
   sorted by verified primary-source publication date.

   Every date was checked against the source's own submission
   record directly — not taken on trust from a model's memory.

   Each timeline entry carries, on top of the required
   what-it-buys / what-it-costs / when-to-choose triad:
     · intuition — the mechanism in one plain-language sentence
     · diagram   — id of the visual explainer rendered on the card
   ============================================================ */

window.SESSION_DATA = (function () {

  /* Reference serving config used by the cost visuals further down.
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
    { id: "compute",   label: "Compute Cost",       color: "#22d3ee", meter: "the T² compute bill" },
    { id: "memory",    label: "Cache Memory",        color: "#f59e0b", meter: "the KV-cache memory bill" },
    { id: "position",  label: "Position Handling",   color: "#a78bfa", meter: "how the model senses word order" },
    { id: "extend",    label: "Context Extension",   color: "#f43f5e", meter: "running past the trained length" },
    { id: "recurrent", label: "Recurrent State",     color: "#10b981", meter: "a fixed-size memory instead of a list" },
    { id: "sparse",    label: "Sparsity",            color: "#6366f1", meter: "reading fewer tokens per query" },
    { id: "compress",  label: "Compression",         color: "#06b6d4", meter: "storing a summary, not the real thing" },
    { id: "systems",   label: "Systems / I-O",       color: "#94a3b8", meter: "the same math, laid out for the hardware" },
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
      diagram: "pipeline",
      intuition:
        "Every word writes a little question, every word wears a little name-tag, and every word carries a little parcel. Each word reads all the name-tags, sees which ones answer its question, and walks away holding a blend of the matching parcels.",
      era: {
        title: "Where the story starts",
        blurb: "Before this, sequence models read one word at a time and passed a single summary forward — useful, but slow to train and forgetful over distance. This is the idea that replaced that summary with something far more direct: let every word look at every other word, all at once.",
      },
      problem:
        "Recurrent networks read a sentence one word at a time, carrying a single running summary forward. That summary is a bottleneck twice over: training can't run in parallel (step 50 needs step 49's result first), and anything from far back has to survive being squeezed into that one vector over and over — which, in practice, often means it doesn't.",
      mechanism:
        "Give every word three separate roles: a query (what it's looking for), a key (what it advertises about itself), and a value (what it hands over if picked). Score each query against every key with a dot product, divide by √d_k so the numbers stay in a sane range, optionally block off future positions, run softmax on each row to turn scores into percentages, and use those percentages to blend the values. That's the whole pipeline: Q×K → scores → scale → mask → softmax → weighted sum of V.",
      buys:
        "Any two words, however far apart, are one dot product away from each other — no signal has to survive a long relay. And with no step-by-step recurrence, the whole layer is a matrix multiply that trains in parallel across the entire sequence.",
      costs:
        "Comparing everything with everything makes the score grid T×T — compute and memory both grow with the square of the sequence length. And once the model generates text one word at a time, it has to keep every earlier word's key and value on hand to avoid recomputing them — a second, separately growing cost. Also: nothing here encodes order. Swap two words' positions and their queries, keys and values don't change at all.",
      chooseWhen:
        "Sequences are short enough that quadratic cost isn't painful yet, and training speed plus unrestricted word-to-word access matter more than anything else.",
      bridge:
        "That last line in the costs isn't a footnote — it's an open wound. This mechanism has no idea what order its own words came in, and the very same paper patches that immediately.",
    },

    {
      id: "sinusoidal",
      date: "2017-06-12",
      dateDisplay: "12 Jun 2017 · same paper",
      threads: ["position"],
      title: "Sinusoidal Position Signal",
      tagline: "A fix for the very last line above: attention has no idea what order anything came in.",
      sourceId: "vaswani2017",
      diagram: "sinusoidal",
      intuition:
        "Stamp every position with a row of clock hands — some sweeping fast, some crawling slowly. Read all the hands at once and you get a fingerprint that's unique to that position, and almost identical for the position next door.",
      problem:
        "A dot product compares content, not position. If the word at position 3 and the word at position 30 happen to be the same word, they produce the same query, key and value — attention genuinely can't tell them apart unless something outside the dot product says where each one sits.",
      mechanism:
        "Before the first layer runs, add a fixed pattern of sine and cosine waves onto every word's embedding — one wave pair per pair of embedding dimensions, each pair oscillating at its own frequency, all keyed off the word's position. No parameters, no training: position 5's vector is always exactly position 5's vector.",
      buys:
        "A position signal that costs nothing to learn and is defined for any position you can plug into the formula. Different frequencies act like clock hands at different speeds, giving the network enough structure to work out relative offsets on its own.",
      costs:
        "The signal only enters once, at the very bottom of the network — turning it into something the attention score actually reacts to has to be learned indirectly, layer by layer. And the paper's own side-by-side test found a plain trainable table did just as well, which undercuts the case for the hand-designed waves.",
      chooseWhen:
        "You want a zero-parameter position scheme and don't need the attention score itself to explicitly reflect relative distance. Mostly a historical footnote today.",
      footnote:
        "That side-by-side comparison — sinusoidal vs. a trainable table — was run in this very paper. The trainable version just wasn't what shipped as the headline design; it became the default a year later (next).",
      bridge:
        "It would take barely a year for that footnote to become the headline.",
    },

    {
      id: "absolute-learned",
      date: "2018-06",
      dateDisplay: "Jun–Oct 2018",
      threads: ["position"],
      title: "A Position Table You Just Train",
      tagline: "Why hand-design the position signal at all, when gradient descent can find one?",
      sourceId: "gpt1_2018",
      secondarySourceIds: ["bert2018"],
      diagram: "learned-table",
      intuition:
        "Give every position its own numbered pigeonhole and let training stuff whatever pattern works into each one. Simple — until you reach a position that never got a pigeonhole.",
      problem:
        "The sine-wave recipe works, but it's exactly that — a recipe someone picked. Once large pretrained models needed a default, treating position the way words themselves are treated — a row in a trainable lookup table — was the simpler engineering call, and the earlier test had shown it wasn't a worse one.",
      mechanism:
        "Allocate one trainable vector per position, up to some maximum length, and add it to the word embedding at that position. It trains end-to-end with everything else — no special formula.",
      buys:
        "One less piece of hand-designed math in the architecture. It matches or slightly beats the sine-wave version up to the length it was trained on, and can shape itself to whatever positional pattern the data actually rewards.",
      costs:
        "The table has a last row. Ask about position N+1 and there's nothing there — a hard wall, not a graceful decline. And there's still no built-in sense of distance: position 5 and position 300 are two unrelated rows that happen to sit near each other in memory, nothing more.",
      chooseWhen:
        "The maximum length is fixed and known in advance and you'll never need to go past it — short, bounded inputs are the comfortable case.",
      bridge:
        "Position was, for the moment, settled. Underneath it, a much bigger problem had been quietly building: comparing every word to every other word was becoming unaffordable at real scale.",
      fixedBy: ["rope", "alibi"],
    },

    {
      id: "sparse-topk",
      date: "2019-04-23",
      dateDisplay: "23 Apr 2019",
      threads: ["compute", "sparse"],
      title: "Stop Looking at Everything",
      tagline: "The first serious answer to the quadratic bill, once people fed in truly long sequences.",
      sourceId: "sparseTransformer2019",
      diagram: "sparse-grid",
      intuition:
        "Don't let every word read every other word. Let it read its close neighbours plus every Nth word — a fixed skeleton that still lets a message cross the whole sequence in a hop or two.",
      era: {
        title: "2019 — two different teams, two different walls",
        blurb: "Within the same seven months, one group hit the cost of comparing every token to every other token, and a different group hit the cost of remembering every token during generation. Neither problem mattered at short context; both became unavoidable the moment sequences got long.",
      },
      problem:
        "Feed in tens of thousands of tokens — raw pixels, long documents, audio — and the T×T score grid stops being a rounding error and becomes the whole budget. Double the sequence, quadruple the work.",
      mechanism:
        "Instead of every query seeing every key, restrict each query to a small fixed pattern of keys — nearby positions plus a handful at regular strides — chosen so information can still cross the whole sequence in a couple of hops, even though no single layer sees everything. This is the ancestor of the broader idea: only ever score a few candidate keys per query, not all of them.",
      buys:
        "Sequences tens of thousands of tokens long become trainable, at a fraction of the compute dense attention would need.",
      costs:
        "The pattern of who-sees-whom is fixed by the architecture, not learned or adapted to content. And a naive 'just keep the top few keys' still has to score every candidate before throwing most away — if scoring everything was the expensive part, that cost hasn't moved.",
      chooseWhen:
        "Context is long and regular enough — a grid, a raster, a predictable stride — that a fixed sparse pattern still reaches what matters. Weak when the important tokens could be anywhere.",
      bridge:
        "That was one wall. Seven months later a different team, chasing a completely different bottleneck, hit the other one.",
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
      diagram: "head-share",
      diagramConfig: { preset: 1, lockLabel: "MQA — 1 shared K/V" },
      intuition:
        "Keep every attention head's own question, but make the whole set of heads share a single set of name-tags and parcels instead of each carrying its own copy.",
      problem:
        "Generating one word at a time, every layer has to hold on to every earlier word's key and value — recomputing them each step would be wasteful. With a separate key/value pair per head, that cache gets big fast, and simply reading it back from memory starts to dominate how long each generated word takes.",
      mechanism:
        "Keep all the query heads — they can still ask different questions — but collapse the keys and values to a single shared pair used by every head. Only one K/V projection is computed and stored per layer, not one per head.",
      buys:
        "The cache shrinks by roughly the number of heads — an 8-head model's cache drops close to 8×. Generation becomes far less bottlenecked on shuffling memory, especially with many requests at once.",
      costs:
        "Every head now searches the same shared keys and values, so most of the benefit of heads specializing on different things is gone — and that shows up as a measurable quality gap against keeping heads separate.",
      chooseWhen:
        "Serving many requests cheaply and quickly matters more than the last bit of quality, and retraining (or briefly adapting) to this scheme is an option.",
      bridge:
        "Sharing keys and values fixed how much had to be remembered. It said nothing about which words get compared to which — that question kept evolving on its own.",
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
      diagram: "sliding-window",
      intuition:
        "Each word only talks to the few words on either side of it. To reach something far away, the message gets passed hand-to-hand up through the layers — plus a few 'megaphone' tokens everyone can hear.",
      era: {
        title: "2020 — same target, two different bets",
        blurb: "One idea bets most of what a token needs is nearby, plus a few designated important tokens. The other bets the past doesn't need keeping at all — just summarizing into a fixed-size running state. Neither is obviously right; both get taken seriously.",
      },
      problem:
        "The strided patterns above cut cost in general, but a lot of document-scale work has a more specific shape: most of what a word needs is close by, plus a small number of genuinely global anchors (a summary token, a question). A generic stride doesn't know that.",
      mechanism:
        "Give every word a narrow window of nearby positions to attend to — widened at deeper layers so the effective reach still grows with depth — plus a small, task-chosen set of tokens that see, and are seen by, everyone. Cost now scales with sequence length times window size, not sequence length squared.",
      buys:
        "Documents thousands of tokens long become tractable on ordinary hardware, while depth still lets information eventually travel further than any single window.",
      costs:
        "Inside one layer, a word genuinely cannot see past its window — anything further has to relay through several layers or the global tokens. Pick too narrow a window and the model is architecturally blind to context it needed.",
      chooseWhen:
        "The task's useful context is mostly local, with a few known anchor points — documents, source code, audio. Weaker when relevant information could be anywhere with equal odds.",
      footnote:
        "This exact idea resurfaced in production years later, paired with the head-sharing trick two cards up — see the attention-sinks card below for what that combination ran into.",
      bridge:
        "A narrower window was one way to stop looking at everything. Two months later someone asked a more radical version: what if the past didn't need keeping as a list at all?",
      fixedBy: ["attention-sinks"],
    },

    {
      id: "linear-attention",
      date: "2020-06-29",
      dateDisplay: "29 Jun 2020",
      threads: ["compute", "memory", "recurrent"],
      title: "What If the Past Were Just a Running Total?",
      tagline: "A more radical bet than a smaller pattern: don't keep the past at all — compress it as it arrives.",
      sourceId: "linearAttention2020",
      diagram: "running-state",
      intuition:
        "Instead of keeping every note anyone ever passed you, keep one running summary sheet and update it after every word. The sheet is the same size after 10 words or 10 million.",
      problem:
        "Every version so far still keeps an exact, indexed record of some subset of past words. None ask whether attention could instead carry one fixed-size running state forward, the way an old recurrent network does — trading exactness for a memory footprint that never grows.",
      mechanism:
        "Swap the softmax step for a simpler feature map, and a piece of arithmetic falls out for free: with nothing tying the scores into a shared denominator, each key/value pair can be folded into one running matrix as words arrive — before any query shows up — and a new query just reads that matrix. Per-word cost during generation drops to a constant, whatever the history length.",
      buys:
        "A state whose size never grows with sequence length, for both compute and memory. Generating each new word is O(1) work regardless of how long the conversation already is — an RNN's cheap decoding, with attention's parallel training.",
      costs:
        "Softmax's built-in competition between keys — one gaining weight necessarily costs another — is gone; what's left is a lossy running summary, not an exact per-key memory. And a plain running total can only add new information, never revise something it already wrote.",
      chooseWhen:
        "Streaming or generation needs to run for an effectively unbounded length at fixed memory, and an approximate, compressed memory of everything earlier is an acceptable trade.",
      bridge:
        "Trading exactness for constant-size memory solved one kind of problem. It did nothing for the oldest unsolved one on this page — attention still had no real sense of where anything was.",
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
      diagram: "rope",
      intuition:
        "Don't hand a word its seat number. Spin its query and key arrows by an angle equal to its seat number. When two arrows meet in a dot product, their absolute spins cancel and only the gap between their seats is left.",
      era: {
        title: "2021 — position gets rebuilt from scratch",
        blurb: "Both existing schemes bake a position into a vector once, at the bottom of the network, and hope later layers recover relative distance. Two very different ideas this year both try to make the score itself carry that distance directly.",
      },
      problem:
        "Both earlier schemes bake position into the vector once, at the very bottom, and leave later layers to reconstruct anything relative from it. The trainable version also can't be asked about a position it never allocated a row for.",
      mechanism:
        "Instead of adding a position vector, rotate the query and key vectors themselves — pair up their dimensions, treat each pair as a 2D point, and spin it by an angle proportional to the word's position, at several speeds across the different pairs. A dot product only cares about the angle between two vectors, so when both are rotated, their individual absolute rotations cancel and only the gap between their positions survives inside the score.",
      buys:
        "Relative position becomes a property of the score itself, not something later layers infer. The rotation formula is defined at any position — no table to run out of. And sliding a whole sequence forward changes nothing about how nearby words relate.",
      costs:
        "Being computable at a distant position isn't the same as the network having learned to behave well there — a rotation angle far outside anything seen in training is still new territory, whatever the formula says. This exact gap is what every extension method further down exists to address. There's also a small constant rotation cost per layer.",
      chooseWhen:
        "You want the score to explicitly reflect relative distance with no hard length ceiling in the mechanism, tuned around your intended training length — the default starting point for most current models.",
      bridge:
        "Rotation gave the score a genuine sense of relative distance. Four months later a very different team asked whether you needed a rotation — or any position vector at all — to get there.",
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
      diagram: "alibi",
      intuition:
        "Don't encode position anywhere. Just quietly subtract points from a word's score for every step of distance between it and the word asking — the further away, the bigger the penalty.",
      problem:
        "Both earlier vector-based schemes extrapolate badly once you ask a model to run longer than it trained. Rotation is at least mathematically defined out there, but whether the model actually behaves sensibly at those distances is still open, and it comes with real machinery to implement.",
      mechanism:
        "Don't encode position anywhere near the vectors. Instead, just before softmax, directly subtract a penalty from each raw attention score, proportional to how far apart the query and key sit — with a different fixed steepness per head, never learned, never adjusted.",
      buys:
        "A model trained on short sequences reliably keeps working on much longer ones with no extra fine-tuning. No new learned parameters, and about as few moving parts as a position scheme can have.",
      costs:
        "That distance penalty is a fixed opinion the network can't override — a task that genuinely needs to weight something far away above something nearby is fighting the mechanism itself. In practice, most current large models chose rotation plus a separate extension method instead.",
      chooseWhen:
        "Reliable behavior far past the training length matters more than architectural flexibility, or you want the smallest possible number of moving parts.",
      bridge:
        "Position, in one form or another, was mostly settled for the next couple of years. What hadn't been settled was whether the exact version of attention could ever be made to run fast.",
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
      diagram: "flash",
      intuition:
        "Same arithmetic, better logistics. Never write the giant T×T scratchpad down to slow memory at all — stream Q, K and V through the chip's tiny fast drawer in tiles and keep a running softmax as you go.",
      era: {
        title: "2022 — a year that isn't about the algorithm at all",
        blurb: "Everything before this either approximated the math or changed it outright. This one changes neither — it just refuses to accept that exact attention has to be slow.",
      },
      problem:
        "Every approximate method above works around the quadratic cost by changing what gets computed. But naive exact attention was slower than its own arithmetic should require, because it wrote the entire T×T score matrix out to slow accelerator memory and read it straight back — the bottleneck was data movement, not the multiplications.",
      mechanism:
        "Break Q, K and V into blocks small enough to fit in the accelerator's fast on-chip memory, run softmax incrementally as those blocks stream through, and never materialize the full T×T matrix in slow memory at all. Recompute what's needed during the backward pass instead of storing it.",
      buys:
        "The identical output exact attention always produced — no approximation anywhere — at a fraction of the wall-clock time and memory, simply by fixing where the bottleneck actually was.",
      costs:
        "The number of arithmetic operations doesn't change — this is a large constant-factor win, not a new complexity class. And the implementation is tied closely to a given accelerator's memory hierarchy, so porting it isn't free.",
      chooseWhen:
        "Essentially always, whenever exact attention is being run at all — by now this sits underneath most of the other mechanisms on this page rather than competing with them.",
      bridge:
        "Making the exact math fast didn't touch either of the two structural costs from Chapter Two. A year later, almost the whole field turned back to the memory one at once.",
    },

    {
      id: "gqa",
      date: "2023-05-22",
      dateDisplay: "22 May 2023",
      threads: ["memory"],
      title: "A Dial Between the Two Extremes",
      tagline: "Four years after the memory problem first appeared, someone put a knob on the trade-off instead of picking an end.",
      sourceId: "gqa2023",
      diagram: "head-share",
      diagramConfig: { preset: 2 },
      intuition:
        "Between 'everyone shares one textbook' (MQA) and 'everyone carries their own' (MHA), just form a few small study groups — each group shares one set of keys and values.",
      era: {
        title: "2023 — the year everything happens at once",
        blurb: "Four separate ideas land within five months of each other, each attacking a different weak point that had been quietly accumulating: cache size, extrapolation past training length, and what happens when a fixed window runs forever.",
      },
      problem:
        "Sharing keys and values down to a single pair saves the most cache but costs the most quality; keeping a full separate pair per head keeps all the quality but none of the savings. Nothing sat in between.",
      mechanism:
        "Split the query heads into a handful of groups, and let each group share one key/value pair — one group recovers full sharing, as many groups as heads recovers none. An existing full-head model can be converted cheaply by averaging its key/value heads within each new group, rather than starting over.",
      buys:
        "A dial, not a binary choice — a modest number of groups gives most of the cache savings with far less quality lost than collapsing to one pair. This became the practical default across essentially every open large model shipped afterward.",
      costs:
        "The cache still grows with sequence length; grouping only changes the slope of that line, not the fact that it keeps climbing. At long enough context it still eventually dominates memory.",
      chooseWhen:
        "Almost always, for any model balancing serving cost against quality — the live question by this point is how many groups, not whether to group.",
      bridge:
        "Head-sharing put a dial on cache size. It said nothing about a completely different kind of length problem — what happens when a trained model is asked to read further than it ever practiced.",
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
      diagram: "freq-stretch",
      intuition:
        "To make a ruler measure a longer room, don't shrink every mark by the same amount. Stretch the far-apart mile-markers a lot and leave the fine millimetre marks — the ones you tell nearby words apart with — almost untouched.",
      problem:
        "The obvious way to reach a longer context with rotary position is to squeeze every position index by a fixed factor before rotating. It works, but it squeezes the fine-grained, high-frequency rotations the model relies on to tell adjacent words apart — local precision gets damaged in exchange for reach.",
      mechanism:
        "Instead of rescaling every position by the same factor, change the base of the rotation formula itself, so the slow-moving (long-range) components get stretched much more than the fast-moving (local) ones — an uneven adjustment rather than a uniform squeeze.",
      buys:
        "Meaningfully less damage to nearby-word precision than the uniform squeeze, and it needs no retraining at all to get a real improvement.",
      costs:
        "It's still one global formula applied to every frequency at once — coarser than treating each frequency band on its own terms, which shows up as real quality loss once the extension factor gets aggressive.",
      chooseWhen:
        "A fast, no-retraining context stretch is worth a partial quality hit. Mostly displaced a few months later by the more careful version below.",
      footnote:
        "There is no peer-reviewed paper behind this one — the original source is a public forum post. It earns a place here because the more formal method that followed explicitly builds on and credits it, and leaving out ideas that started outside a journal would misrepresent how this thread actually moved.",
      bridge:
        "One forum post's uneven stretch was a real improvement. It took about two months for someone to formalize exactly why it worked, and do it more carefully.",
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
      diagram: "yarn-bands",
      intuition:
        "Split the rotation's frequencies into three zones: leave the fast local ones exactly alone, stretch the slow global ones, and feather smoothly through the middle — then nudge softmax's sharpness back to where it was.",
      problem:
        "Stretching the rotation base unevenly beats a uniform squeeze, but it's still one rule applied everywhere. Some frequency bands need real stretching, some barely any, and the ones between need a smooth handoff rather than a hard edge — and stretching positions also quietly shifts how sharp or diffuse softmax's output ends up.",
      mechanism:
        "Split the rotation's frequency spectrum into three regions: leave the fastest, most local frequencies alone entirely, stretch the slowest, most global ones the way the uneven method above does, and blend smoothly through a middle band — plus a small correction to softmax's effective sharpness.",
      buys:
        "State-of-the-art context extension for a small fraction of what the original training run cost — this became, and largely still is, the standard way to stretch a rotary model's context after the fact.",
      costs:
        "More moving parts than either earlier method — three bands and a temperature correction to get right. And it's still an extension of a shorter training run, not the genuine article: the model was never actually trained at the longer length, so performance there is evidence, not a guarantee.",
      chooseWhen:
        "A model already trained at a shorter context needs to responsibly reach further, and a small fine-tuning budget is available rather than the cost of training long from the start.",
      bridge:
        "Stretching a trained model's reach was one axis of the long-context problem. A completely different failure mode showed up the moment a model had to run forever, not just further.",
    },

    {
      id: "attention-sinks",
      date: "2023-09-29",
      dateDisplay: "29 Sep 2023",
      threads: ["memory", "extend"],
      title: "Keep the First Few Tokens Forever",
      tagline: "A fix for a specific, ugly failure mode: what a fixed local window does when a conversation runs forever.",
      sourceId: "streamingLLM2023",
      diagram: "sinks",
      intuition:
        "Softmax has to put its attention somewhere even when nothing nearby is relevant. Models learn to dump that leftover attention onto the first few words — so if you ever evict those, the model has nowhere to park it and output falls apart. Fix: pin them permanently.",
      problem:
        "A model serving a genuinely endless stream — a long chat, a live transcript — using a fixed local window has to evict the oldest tokens to stay in budget. The moment the very first few tokens get evicted, output quality falls off a cliff, even though those first tokens rarely carried content worth remembering.",
      mechanism:
        "Softmax always has to put its attention mass somewhere, even when nothing in the current window is truly relevant — and models learn to dump that unwanted mass onto the first few tokens by default, regardless of what they contain. So: keep a small, fixed number of the very first tokens permanently in the cache, alongside the sliding window of recent ones, and never evict them.",
      buys:
        "Stable output over an effectively unlimited stream at a small, fixed memory budget — and it often works without retraining, since it's really just changing what gets evicted.",
      costs:
        "Those pinned tokens work as a release valve for attention, not a real memory — anything from the middle of a long-since-evicted stretch is genuinely gone. Models trained with no expectation of this eviction pattern benefit only partially unless retrained with it in mind.",
      chooseWhen:
        "The deployment is a long-running or effectively endless session under a hard memory budget, where staying stable matters more than recalling everything ever said.",
      footnote:
        "Two weeks after this shipped, a widely-used open model shipped its own local-window attention paired with the head-grouping trick from earlier — without adopting this fix. A good reminder that a solved problem and a widely-deployed solution aren't the same event.",
      bridge:
        "Pinning a few tokens fixed one specific failure mode of a bounded cache. It said nothing about the cache that wasn't bounded at all — the one every ordinary transformer still carried, growing in a straight line with no ceiling.",
    },

    {
      id: "mla",
      date: "2024-05-07",
      dateDisplay: "7 May 2024",
      threads: ["memory", "compress"],
      title: "Cache a Compressed Summary, Not the Real Thing",
      tagline: "The head-grouping dial lowered the slope of the cache-growth line. This attacks the line's height instead.",
      sourceId: "mla2024",
      diagram: "mla",
      intuition:
        "Don't file the full-size keys and values for every word. File a short code, and expand it back into keys and values only at the moment you actually need them.",
      era: {
        title: "2024 — memory gets more literal, and starts learning to forget",
        blurb: "One idea this year asks whether the cache needs to store real keys and values, or just enough to reconstruct them. A different idea, a month later, asks whether a running state needs to only ever accumulate, or whether it can correct itself.",
      },
      problem:
        "Grouping query heads onto shared keys and values lowers how fast the cache grows with sequence length, but it still stores full-width keys and values for whatever's left after grouping — and cache size is still, fundamentally, linear in context length.",
      mechanism:
        "Rather than caching full-size keys and values at all, compress each word into one small shared low-rank vector and cache only that. Reconstruct the full-size keys and values from the compressed vector on the fly when they're needed, keeping a small separate slice for position information.",
      buys:
        "A cache meaningfully smaller than even the grouped version, at matched or better quality — the model that introduced this reported beating full, ungrouped attention on benchmarks while needing a far smaller cache to do it.",
      costs:
        "Real added complexity — the compress-then-reconstruct machinery, plus a separate way of handling position, don't exist in a simpler design. And it's still linear in sequence length, just with a much smaller constant.",
      chooseWhen:
        "Serving cost per token is the dominant expense at your scale, and the extra architectural machinery to shrink it is worth the engineering cost.",
      bridge:
        "Compressing the cache attacked how much had to be stored. A different idea, one month later, went back to a much older question this story had left half-answered since 2020: could a fixed-size memory ever learn to change its mind?",
    },

    {
      id: "delta-rule",
      date: "2024-06-10",
      dateDisplay: "10 Jun 2024",
      threads: ["recurrent"],
      title: "Teach the Running State to Correct Itself",
      tagline: "The fixed-size running state from 2020 could grow. It still couldn't ever change its mind.",
      sourceId: "deltaNet2024",
      diagram: "delta",
      intuition:
        "Before writing a new answer into memory, first read what memory currently says, work out the gap, and write only the gap. Memory says 30, should say 70 → write 40, not 70.",
      problem:
        "A running-total state can only ever add new contributions — it has no way to revise something it already wrote. Concretely: if a stored association currently reports 30 for some key, but the right answer is now 70, adding the new value on top gives 100, which is simply wrong. What's needed is a way to overwrite, not just accumulate.",
      mechanism:
        "Before writing anything, read what the state currently returns for a key, work out the gap between that and the value you want it to return, and write only that gap. If the state returns 30 and should return 70, the correction is 40, and 30 + 40 lands exactly on 70 — not 30 + 70. A hardware-friendly algorithm, built around products of simple reflection matrices, makes this practical to train at scale despite looking sequential on paper.",
      buys:
        "A fixed-size state that can genuinely be edited, not just grown — a meaningful quality jump over a plain running-total state and over other constant-memory baselines from around the same time, at matched scale.",
      costs:
        "More arithmetic per step than a plain accumulation, since every write now needs a read-and-compare first. And it's still a compressed summary of everything that happened — nowhere near as exact as keeping the real keys and values around.",
      chooseWhen:
        "Constant-memory, RNN-style decoding is the goal, but the state genuinely needs to be revisable rather than purely additive — often as one layer type mixed into a design that also keeps some exact-attention layers.",
      bridge:
        "Teaching a running state to correct itself was one skill. It said nothing about when a piece of that state should simply stop mattering.",
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
      diagram: "gated-delta",
      intuition:
        "Same self-correcting memory as the delta rule, plus a per-step dial that first fades down whatever's already stored before the correction is applied — so stale associations can decay away, not just be overwritten.",
      problem:
        "The correction mechanism above can fix a wrong association, but it has no notion of an association simply becoming irrelevant — there's no way to say 'discount this part of the state,' the way other constant-memory designs built around explicit forgetting already could.",
      mechanism:
        "Add a learned, per-step gate that scales down how much of the existing state survives before the correction is applied — combining an adaptive-forgetting mechanism with the targeted correction from the previous idea, in one update.",
      buys:
        "Consistently outperforms both a purely-forgetting design and a purely-correcting one on language modeling, in-context retrieval, and extrapolation to longer sequences than trained on.",
      costs:
        "One more learned mechanism to get right per layer or head. And no amount of gating brings back information the state has already actually discarded — forgetting is still forgetting.",
      chooseWhen:
        "Choosing among constant-memory layer designs for a mixed architecture — currently one of the strongest options in that family.",
      bridge:
        "Recurrent state had just relearned how to correct itself and forget. Meanwhile, the oldest idea in this entire story — just look at fewer tokens — had never gone away; it had been waiting for hardware-aware engineering to catch up to it.",
    },

    {
      id: "deepseek-nsa",
      date: "2025-02-16",
      dateDisplay: "16 Feb 2025",
      threads: ["sparse", "compress", "systems"],
      title: "Sparsity Comes Back, Built for the Hardware",
      tagline: "The 2019 idea — look at fewer keys — returns, but trained in from the start and built around how accelerators actually move data.",
      sourceId: "nsa2025",
      diagram: "nsa",
      intuition:
        "Give every query three ways to read the past at once: skim cheap summaries of every block, re-read in full only the few blocks that looked promising, and always keep a small local window. Train the whole thing sparse from step one.",
      era: {
        title: "2025 — sparsity, taken seriously again",
        blurb: "Skipping most of the sequence was the very first idea in this story, back in 2019. It never went away, but it also never fully solved its own founding problem: finding the few keys worth reading is supposed to be cheap, and for years it mostly wasn't.",
      },
      problem:
        "Looking at fewer keys and compressing the cache each solve one cost, but most earlier sparse designs were bolted onto an already-trained dense model rather than trained sparse from the start, and weren't built with real accelerator memory movement in mind — leaving real speed on the table even where the arithmetic looked cheaper. And the founding problem from 2019 was still open: proposing a short list of good candidate keys is supposed to be cheap, but naive proposal methods still have to look at everything first.",
      mechanism:
        "Give every query three parallel ways to read the past: a cheap always-on pass over compressed summaries of blocks of history; a more expensive pass that re-reads the real tokens inside only the top few blocks, chosen by a small cheap scoring network rather than by scoring everything in full; and a local window for immediate neighbors. Trained sparse from the very first step, with the low-level kernels built around the accelerator's actual memory bandwidth.",
      buys:
        "Long-context training and serving that's sparse in a way that translates into measured wall-clock speed, not just a smaller number on paper — while matching or beating dense attention on downstream benchmarks. The cheap scoring network genuinely closes the old 'proposing candidates still costs as much as scoring everything' gap.",
      costs:
        "Summarizing blocks of tokens loses token-level detail by construction. The candidate selection is still approximate — a genuinely useful key in a block that didn't make the shortlist can be missed entirely. And three parallel read paths are a meaningfully bigger system to implement and train correctly than one dense or one sparse call.",
      chooseWhen:
        "Training a large model natively for very long context, where both training compute and serving memory need to be cheap at the same time, and the added system complexity is worth the throughput it buys.",
      bridge:
        "Sparsity came back sharper, and compression came back more aggressive. But one thread from 2021 had only ever been patched, four separate times, never actually resolved — and nobody had yet asked the most direct question of all.",
    },

    {
      id: "drope",
      date: "2025-12-13",
      dateDisplay: "13 Dec 2025",
      threads: ["extend", "position"],
      title: "Train With Rotation, Then Take It Away",
      tagline: "Four years after rotary position first appeared, someone asked whether it needs to stick around forever.",
      sourceId: "drope2025",
      diagram: "drope-schedule",
      intuition:
        "Rotation helps a model learn word order fast, so keep it for almost all of training. Then remove it, run a short cheap re-settling pass at the original short length, and serve far longer — because with no rotation left, there's no phase to compress and nothing to warp.",
      era: {
        title: "2025, ten months later — the thread closes",
        blurb: "Every rotation-stretching method in this story shares one unavoidable piece of math: to keep the rotation's phase inside familiar territory at a longer length, the slow, long-range frequencies have to be compressed hardest — exactly the frequencies that content-based, 'what does this word mean' attention relies on. This is the first idea that doesn't try to stretch the rotation more carefully. It removes it.",
      },
      problem:
        "Every rescaling method above — the uneven stretch, the three-band version — runs into the same wall, and it isn't a tuning mistake, it's arithmetic: to keep the rotation's phase inside the range the model trained on, the slow-moving frequencies have to be compressed by roughly the same factor as the context extension itself. Those are exactly the frequencies long-range, content-based attention leans on, so every rescaling method, however carefully tuned, ends up quietly distorting the attention patterns it's trying to extend.",
      mechanism:
        "Train the model with rotary position as usual for most of pretraining — rotation genuinely helps a model learn positional structure quickly, and skipping it from the start makes early training measurably slower and worse, since attention heads have no shortcut for developing directional bias. Then, once training is mostly finished, remove the positional rotation from the architecture entirely, converting the model to one with no explicit position mechanism at all, and run a short recalibration at the original short training length — no long-sequence data required. With no rotation left to keep in-distribution, there's no phase to compress and nothing left to warp.",
      buys:
        "Zero-shot context extension that beats the rescaling methods above on long-context retrieval benchmarks, for a recalibration budget as small as roughly half a percent to a few percent of the original training cost, depending on model size — far cheaper than training long from scratch.",
      costs:
        "This isn't a replacement for training with rotation — a model trained with no position mechanism from the very first step converges far more slowly and performs worse throughout training, since it has nothing to lean on early. So this only works as a scheduled hand-off: rotation first, then remove it. It also needs a real, if short, recalibration pass, and at larger recalibration budgets needed extra normalization to stay numerically stable.",
      chooseWhen:
        "A model already trained with rotary position needs to reliably serve far beyond its original context, and a small recalibration bill is preferable to either a rescaling method's quality loss or the cost of training long from the very beginning.",
      footnote:
        "Worth flagging by name, since the resemblance is easy to trip over: there is a separate, unrelated paper with an almost identical name for rotary embeddings in autonomous-agent trajectory modeling — different authors, different field, no connection to context extension at all. Getting the two confused would be exactly the kind of mistake that's easy to make with quiet confidence and never notice.",
      bridge:
        "Which is roughly where this story stands right now — not finished, just current. The two costs from Chapter Two never went away; every idea on this page bought against one of them and spent on something else. That trade is the whole field, not a phase it's passing through.",
      isEpilogue: true,
    },
  ];

  return { REF_CONFIG, THREADS, SOURCES, TIMELINE };
})();
