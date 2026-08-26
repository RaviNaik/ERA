# How Does Attention Work Now?

**ERA V5 — Session 8 Assignment · Modern Attention Variants**


**🌐 Live app:** `[add your deployed Netlify/Vercel/Pages link here]`
**📂 Webapp source:** [`webapp/`](./webapp/) — `index.html`, `style.css`, `data.js`, `app.js`

---

## What this is

The app makes three deliberate choices:

1. **Starts from the mechanism, not a technique list.** The Core Mechanism
   section builds `Q×K → scores → scale → mask → softmax → weighted sum of V`
   live, from an editable 4-token example, before anything else is explained.
2. **One strict chronological timeline, not a taxonomy.** Every mechanism is
   sorted purely by *primary-source publication date* — not the order taught
   in class, not grouped into "position methods" / "sparsity methods" first.
   Related ideas are colour-tagged with **problem threads** (compute, KV-cache
   memory, position, context extension, recurrent state, sparsity,
   compression, systems) so you can see them interleave and resurface across
   eight years, without the vertical order ever bending to group them.
3. **Every date independently verified, gaps disclosed rather than papered
   over.** See [Chronology & Sources](#chronology--sources) below — this is
   the section the assignment explicitly asks for, because *"dates are
   exactly where an AI agent can sound completely confident while being
   wrong."*

## What's inside the app

| Section | What it does |
|---|---|
| **Core Mechanism** | Live, steppable widget: Q/K/V projection → raw scores → scaling → causal mask (toggle it off and watch attention leak onto future tokens) → softmax → weighted sum. Computed in-browser from a real 4-token, 4-dim example — nothing is a static screenshot. |
| **The Two Bills** | Interactive compute-vs-KV-cache-growth chart. Slide context length from 128 to 2M tokens and watch the quadratic compute curve and the linear cache curve diverge; live KV-cache calculator reconciles exactly with the class notes' own worked example (48 layers, 8 KV heads, head_dim 128, bf16 → 6.44 GB at 32,768 tokens for one user). |
| **The Timeline** | 19 mechanisms (18 required/optional + 1 bonus), one card each, sorted strictly by date. Every card answers *what does it buy / what does it give up / when would I actually choose it*, cites its primary source with a link, and — where relevant — carries a small embedded interactive: a RoPE angle-vs-distance demo, a delta-rule correction calculator, a sliding-window-eviction-and-attention-sinks simulator, an MHA→GQA→MQA head-sharing diagram, an MLA cache-width comparison, and a compress-then-select-top-k visual for DeepSeek's NSA. |
| **Cheat Sheet** | The whole timeline compressed to one row per mechanism, for a second-pass skim. |
| **Sources & Chronology** | The full bibliography in timeline order, plus an explicit correction note about the one entry (DroPE) that has no independently checkable external source. |


## Chronology & Sources

**Method:** every date below is the primary source's own **arXiv v1
submission date** (fetched directly from the paper's arXiv abstract /
submission-history page), or — for the two entries with no arXiv record — the
best available primary date, clearly marked. Nothing here was accepted
because a language model stated it confidently; every single row was checked
against the source itself on **26 Aug 2026**, specifically because the
assignment brief flags this as the exact place an agent can be "completely
confident while being wrong."

| # | Date | Mechanism | Primary source |
|---|---|---|---|
| 1 | **2017-06-12** | Scaled dot-product attention *(baseline)* | Vaswani et al., *"Attention Is All You Need,"* NeurIPS 2017 — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) |
| 2 | **2017-06-12** | Sinusoidal position encoding | Same paper, §3.5 |
| 3 | **2018-06** | Absolute learned position embeddings | Radford et al., *"Improving Language Understanding by Generative Pre-Training,"* OpenAI (June 2018); reinforced by Devlin et al., *"BERT,"* [arXiv:1810.04805](https://arxiv.org/abs/1810.04805) (11 Oct 2018) |
| 4 | **2019-04-23** | Sparse & strided (top-k-style) attention | Child, Gray, Radford, Sutskever, *"Generating Long Sequences with Sparse Transformers,"* [arXiv:1904.10509](https://arxiv.org/abs/1904.10509) |
| 5 | **2019-11-06** | Multi-Query Attention (MQA) | Shazeer, *"Fast Transformer Decoding: One Write-Head is All You Need,"* [arXiv:1911.02150](https://arxiv.org/abs/1911.02150) |
| 6 | **2020-04-10** | Sliding-window attention | Beltagy, Peters, Cohan, *"Longformer: The Long-Document Transformer,"* [arXiv:2004.05150](https://arxiv.org/abs/2004.05150) |
| 7 | **2020-06-29** | Linear attention | Katharopoulos, Vyas, Pappas, Fleuret, *"Transformers are RNNs,"* ICML 2020 — [arXiv:2006.16236](https://arxiv.org/abs/2006.16236) |
| 8 | **2021-04-20** | RoPE (Rotary Position Embedding) | Su, Lu, Pan, Wen, Liu, *"RoFormer,"* [arXiv:2104.09864](https://arxiv.org/abs/2104.09864) |
| 9 | **2021-08-27** | ALiBi | Press, Smith, Lewis, *"Train Short, Test Long,"* ICLR 2022 — [arXiv:2108.12409](https://arxiv.org/abs/2108.12409) |
| 10 | **2022-05-27** | FlashAttention *(bonus, beyond the minimum list)* | Dao, Fu, Ermon, Rudra, Ré, NeurIPS 2022 — [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) |
| 11 | **2023-05-22** | GQA (Grouped-Query Attention) | Ainslie et al., EMNLP 2023 — [arXiv:2305.13245](https://arxiv.org/abs/2305.13245) |
| 12 | **2023-06** | NTK-aware scaled RoPE | u/bloc97, [r/LocalLLaMA, thread 14lz7j5](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/) — community post, no peer-reviewed paper |
| 13 | **2023-08-31** | YaRN | Peng, Quesnelle, Fan, Shippole, ICLR 2024 — [arXiv:2309.00071](https://arxiv.org/abs/2309.00071) |
| 14 | **2023-09-29** | Attention Sinks (StreamingLLM) | Xiao, Tian, Chen, Han, Lewis, ICLR 2024 — [arXiv:2309.17453](https://arxiv.org/abs/2309.17453) |
| 15 | **2024-05-07** | MLA (Multi-Head Latent Attention) | DeepSeek-AI, *"DeepSeek-V2,"* [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) |
| 16 | **2024-06-10** | Delta Rule / DeltaNet | Yang, Wang, Zhang, Shen, Kim, NeurIPS 2024 — [arXiv:2406.06484](https://arxiv.org/abs/2406.06484) |
| 17 | **2024-12-09** | Gated DeltaNet | Yang, Kautz, Hatamizadeh (NVIDIA), ICLR 2025 — [arXiv:2412.06464](https://arxiv.org/abs/2412.06464) |
| 18 | **2025-02-16** | DeepSeek's compressed sparse attention (Native Sparse Attention) | DeepSeek-AI, ACL 2025 — [arXiv:2502.11089](https://arxiv.org/abs/2502.11089) |
| 19 | **unverified** | DroPE | No external primary source found — see correction note below |

Two techniques are cited as **context inside a related card** rather than
given their own numbered row, since the app treats them the same way — as a
footnote on the mechanism they extend, not a fully separate timeline entry:
**Big Bird** (Zaheer et al., NeurIPS 2020 — [arXiv:2007.14062](https://arxiv.org/abs/2007.14062),
28 Jul 2020, noted on card #4, sparse attention) and **Mistral 7B**
(Jiang et al. — [arXiv:2310.06825](https://arxiv.org/abs/2310.06825), 10 Oct 2023,
noted on cards #6 and #14, sliding window / attention sinks — including the
detail that Mistral shipped its sliding window *without* attention sinks,
even though the sinks paper predates it by about two weeks).
