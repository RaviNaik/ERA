# How Does Attention Work Now?

A chronological account of attention mechanisms — from scaled dot-product
attention to Sakana AI's DroPE — told strictly in the order each idea was
actually published, so that each one reads as a reply to a limitation the
previous one created, rather than as an alphabetized list of techniques.

**🌐 Live app:** https://ravinaik.github.io/ERA/v5/session8/webapp/
**📂 Source:** [`webapp/`](./webapp/) — `index.html`, `style.css`, `data.js`, `app.js`

---

## What this is

A friend asks: *"how does attention actually work now?"* Not the 2017
version — the current one, with everything that's happened to it since.
This is the answer that would actually help them: one continuous story,
not eighteen unrelated flash cards.

Three things this write-up insists on:

- **It starts from the bare mechanism, not a list of names.** `Q×K → scores
  → scale → mask → softmax → weighted sum of V` is built live, one step at a
  time, on a small worked sentence (*the bird fed its chicks* — deliberately
  not the class-notes example), before anything downstream is explained.
- **It is sorted by publication date, full stop.** Not by family, not by
  how it was taught, not by which idea "won." Related ideas are tagged with
  colour-coded threads (compute cost, cache memory, position, extension,
  recurrence, sparsity, compression, systems) so you can watch the same
  concern resurface years apart under a different name — but the order
  itself never bends to group them.
- **Every date was checked against its source, not assumed.** Dates are
  exactly where confident-sounding storytelling can quietly be wrong — see
  the correction below, which is a real example of that happening during
  the making of this page.

## What's inside

| Section | What it does |
|---|---|
| **The Mechanism Itself** | A live, step-by-step walk through Q/K/V projection → raw scores → scaling → causal masking (turn it off and watch attention leak onto words that haven't happened yet) → softmax → weighted sum — computed in-browser on the sentence *the bird fed its chicks*, with a plain-language "room full of people" analogy up front. Not a static diagram. |
| **The Two Bills** | Both costs made concrete: a comparison grid that grows as T² next to a KV-cache stack that grows as T, a real GB calculator for a mid-size model, and an explicit callout framing the rest of the page as "which of these two meters does this idea push down." |
| **The Story** | 19 ideas, one card each, sorted strictly by date, broken into short era interludes. Every card carries a **plain-words** one-liner, a **dedicated visual explainer of the mechanism itself** (sine-wave fingerprints, a position-table wall, a sparse/causal grid, head-sharing wiring, a widening sliding window, a growing-list-vs-fixed-state comparison, a RoPE rotation dial, an ALiBi distance ramp, a FlashAttention memory hierarchy, uneven frequency stretching, YaRN's three bands, sink eviction, MLA cache widths, the delta rule, a forget-gate decay chart, NSA compress-then-select, and the DroPE training schedule), the *buys / gives up / when to pick it* triad, a "moves the needle on" tag row, its primary source, and the hand-off to the next idea. |
| **Quick Reference** | The whole story compressed to one row per idea, for a second pass. |
| **Sources** | The full bibliography in story order, plus a note on the one entry that's easy to confuse with an unrelated paper of almost the same name. |

## Design notes

Fully static, dependency-free, no build step, no chart library — every
number on the page is computed live from `data.js` by `app.js`, including
the hand-drawn canvases (the score-matrix heatmaps, the cost-growth chart,
the rotation demo). `data.js` is the single source of truth for the whole
story; editing the chronology means editing one file.

---

## Sources, and how the dates were checked

Every date below is the primary source's own **v1 submission date**
(fetched directly from its own record — an arXiv submission-history page,
or the closest available primary record where no preprint exists), checked
individually rather than accepted from memory.

| # | Date | Idea | Primary source |
|---|---|---|---|
| 1 | **2017-06-12** | Scaled dot-product attention *(where the story starts)* | Vaswani et al., *"Attention Is All You Need,"* NeurIPS 2017 — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) |
| 2 | **2017-06-12** | Sinusoidal position signal | Same paper, §3.5 |
| 3 | **2018-06** | A trainable position table | Radford et al., *"Improving Language Understanding by Generative Pre-Training,"* OpenAI (June 2018); reinforced by Devlin et al., *"BERT,"* [arXiv:1810.04805](https://arxiv.org/abs/1810.04805) (11 Oct 2018) |
| 4 | **2019-04-23** | Sparse & strided attention | Child, Gray, Radford, Sutskever, *"Generating Long Sequences with Sparse Transformers,"* [arXiv:1904.10509](https://arxiv.org/abs/1904.10509) |
| 5 | **2019-11-06** | Multi-Query Attention | Shazeer, *"Fast Transformer Decoding: One Write-Head is All You Need,"* [arXiv:1911.02150](https://arxiv.org/abs/1911.02150) |
| 6 | **2020-04-10** | Sliding-window attention | Beltagy, Peters, Cohan, *"Longformer: The Long-Document Transformer,"* [arXiv:2004.05150](https://arxiv.org/abs/2004.05150) |
| 7 | **2020-06-29** | Linear attention | Katharopoulos, Vyas, Pappas, Fleuret, *"Transformers are RNNs,"* ICML 2020 — [arXiv:2006.16236](https://arxiv.org/abs/2006.16236) |
| 8 | **2021-04-20** | RoPE (rotary position embedding) | Su, Lu, Pan, Wen, Liu, *"RoFormer,"* [arXiv:2104.09864](https://arxiv.org/abs/2104.09864) |
| 9 | **2021-08-27** | ALiBi | Press, Smith, Lewis, *"Train Short, Test Long,"* ICLR 2022 — [arXiv:2108.12409](https://arxiv.org/abs/2108.12409) |
| 10 | **2022-05-27** | FlashAttention *(not on the required list — included anyway)* | Dao, Fu, Ermon, Rudra, Ré, NeurIPS 2022 — [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) |
| 11 | **2023-05-22** | GQA (grouped-query attention) | Ainslie et al., EMNLP 2023 — [arXiv:2305.13245](https://arxiv.org/abs/2305.13245) |
| 12 | **2023-06** | NTK-aware scaled RoPE | u/bloc97, [r/LocalLLaMA, thread 14lz7j5](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/) — a forum post, no peer-reviewed paper |
| 13 | **2023-08-31** | YaRN | Peng, Quesnelle, Fan, Shippole, ICLR 2024 — [arXiv:2309.00071](https://arxiv.org/abs/2309.00071) |
| 14 | **2023-09-29** | Attention Sinks (StreamingLLM) | Xiao, Tian, Chen, Han, Lewis, ICLR 2024 — [arXiv:2309.17453](https://arxiv.org/abs/2309.17453) |
| 15 | **2024-05-07** | MLA (multi-head latent attention) | DeepSeek-AI, *"DeepSeek-V2,"* [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) |
| 16 | **2024-06-10** | The delta rule / DeltaNet | Yang, Wang, Zhang, Shen, Kim, NeurIPS 2024 — [arXiv:2406.06484](https://arxiv.org/abs/2406.06484) |
| 17 | **2024-12-09** | Gated DeltaNet | Yang, Kautz, Hatamizadeh (NVIDIA), ICLR 2025 — [arXiv:2412.06464](https://arxiv.org/abs/2412.06464) |
| 18 | **2025-02-16** | Native Sparse Attention | DeepSeek-AI, ACL 2025 — [arXiv:2502.11089](https://arxiv.org/abs/2502.11089) |
| 19 | **2025-12-13** | DroPE — dropping positional embeddings after training | Gelberg, Eguchi, Akiba, Cetin (Sakana AI), *"Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings,"* [arXiv:2512.12167](https://arxiv.org/abs/2512.12167) · [project page](https://pub.sakana.ai/DroPE/) · [code](https://github.com/SakanaAI/DroPE) |

Two further techniques are cited as context on a related card rather than
given a separate row of their own, since the page treats them the same
way — as a footnote on the idea they extend: **Big Bird** (Zaheer et al.,
NeurIPS 2020 — [arXiv:2007.14062](https://arxiv.org/abs/2007.14062), 28 Jul
2020, noted alongside sparse attention) and **Mistral 7B** (Jiang et al. —
[arXiv:2310.06825](https://arxiv.org/abs/2310.06825), 10 Oct 2023, noted
alongside sliding-window attention and attention sinks — including the
detail that Mistral shipped sliding-window attention *without* attention
sinks, two weeks after the sinks paper had already shipped).

### A correction, made honestly

Early research for this page turned up a name — *DroPE* — attached to a
reported context-extension result, with no independently checkable source
behind it: no paper, no code, nothing to verify the mechanism or the
numbers against. The draft treated it that way — present in the story, but
explicitly flagged as unconfirmed, rather than inventing a citation to
paper over the gap.

That flag turned out to be premature, not permanent. DroPE is real, public,
and dated: **"Extending the Context of Pretrained LLMs by Dropping Their
Positional Embeddings"** (Gelberg, Eguchi, Akiba, Cetin — Sakana AI,
[arXiv:2512.12167](https://arxiv.org/abs/2512.12167), submitted 13 Dec
2025), with a project page and released code. Once the actual source
surfaced, the entry above was corrected to cite it properly — including its
real date, which moves it to the very end of this timeline rather than
wherever it had been provisionally placed.

It is worth being precise about one more thing: DroPE is **not** the same
technique as a different, unrelated paper with an almost identical name —
*"DRoPE: Directional Rotary Position Embedding"*
([arXiv:2503.15029](https://arxiv.org/abs/2503.15029), March 2025) — which
addresses rotary embeddings for autonomous-agent trajectory modelling, a
different field entirely. Both are cited separately above specifically so
that resemblance can't cause a mix-up.

The lesson generalizes past this one entry: a source not turning up on a
first pass is evidence of a gap in the search, not proof the source doesn't
exist. The honest response to "I can't verify this" is to say so plainly
and keep looking — not to quietly assert a date anyway, and not to give up
and drop the claim either.

One more entry is worth a small caveat for the same reason: **NTK-aware
scaled RoPE** has no peer-reviewed paper behind it at all — its primary
source really is a forum post. It's kept in the story anyway because the
formal method that followed a few months later explicitly credits and
builds on it, and dropping community-origin ideas from the record would
misrepresent how this particular thread actually moved.

---

## The shape of the story, once it's sorted by date

```
2017  exact, all-to-all attention, quadratic by construction
2017  ↳ needs some notion of order                      → sine waves
2018  ↳ order becomes just another trainable table       → learned positions
2019  the compute bill bites  → look at fewer tokens      → sparse attention
2019  the memory bill bites   → share keys/values         → multi-query attention
2020  the sparse pattern gets shaped to the task          → sliding windows
2020  what if the past were a running total, not a list?  → linear attention
2021  position becomes a rotation, not an addition        → RoPE
2021  ↳ or no position mechanism at all                   → ALiBi
2022  make the exact version fast instead of approximate  → FlashAttention
2023  a dial between full sharing and none                → GQA
2023  stretch the rotation, unevenly                      → NTK-aware scaling
2023  ↳ then more carefully, in three bands                → YaRN
2023  fixed windows get a safety valve                    → Attention Sinks
2024  compress the cache instead of just sharing it       → MLA
2024  a running total learns to correct itself            → the delta rule
2024  ↳ and then to forget, too                            → Gated DeltaNet
2025  sparsity returns, built around real hardware         → Native Sparse Attention
2025  and, at the very end: drop the rotation entirely     → DroPE
```

Read in order, the sequence isn't a list — it's an argument that keeps
replying to itself: exact and unbounded, then cheaper to remember, then
aware of distance, then longer-reaching, then recurrent again, then sparse
again, and finally, willing to question whether the position mechanism
needed to be permanent at all.

---

## Running it locally

No build step, no dependencies. Open `webapp/index.html` directly in a
browser, or serve the folder with anything static:

```bash
cd webapp
python -m http.server 8000
# then open http://localhost:8000
```
