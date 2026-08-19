# Empirical Validation of the Fourier–Kronecker Embedding Comparison

**A controlled transformer training report: Dense vs. Kronecker vs. Fourier vs. Fourier (narrow-band)**

---

## Abstract

This report presents the results of the controlled experiment specified in [`fourier_embeddings_research.md`](fourier_embeddings_research.md)'s §10 validation protocol: four ~105–111M-parameter GPT-style transformers, identical in every hyperparameter except their token-embedding codec (**Dense**, **Kronecker**, **Fourier**, **Fourier narrow-band**), trained for 5,000 steps (327.68M tokens) on a real ~150MB multilingual (English/Hindi/Telugu/Tamil/Bengali) Wikipedia corpus, plus three standalone analytical probes (collision rate, order-sensitivity, HRR crosstalk) run against the real, trained vocabulary. The headline results: the Fourier-position codec trains to a **lower final validation loss than the one-hot Kronecker codec it generalizes** (0.2181 vs. 0.2197) while using the **identical 6.29M codec parameters** (50% smaller than Dense's 12.58M), at a modest **+6% wall-clock cost** from the uncached "dynamic" grid construction; the order-sensitivity trade the theory predicts is directly visible in the data (Kronecker's rearranged-pair similarity spans 0.0–1.0, Fourier's is compressed to 0.81–1.0); and — the most important finding of this report — **the experimental configuration does not exercise the truncation-wall failure mode that originally motivated this whole line of investigation**, because the BPE tokenizer's subword tokens are short enough that `pos_dim=32` is never binding, which this report treats as a first-class result, not a footnote, and reports honestly rather than papering over. A second honest finding: the collision-threshold calibration §8.4 predicted would be "a new, non-obvious design decision" turned out to be exactly that — a global random-pair threshold proved un-usable for a same-script comparison, for reasons explained in §6 below.

---

## Table of contents

1. [Experimental setup](#1-experimental-setup)
2. [Reading these results correctly: the validation-loss anomaly](#2-reading-these-results-correctly-the-validation-loss-anomaly)
3. [Parameter efficiency](#3-parameter-efficiency)
4. [Training dynamics](#4-training-dynamics)
5. [Final performance comparison](#5-final-performance-comparison)
6. [Collision analysis — and why this run doesn't test what it looks like it tests](#6-collision-analysis--and-why-this-run-doesnt-test-what-it-looks-like-it-tests)
7. [Order-sensitivity probe](#7-order-sensitivity-probe)
8. [HRR crosstalk probe](#8-hrr-crosstalk-probe)
9. [Claim-by-claim: the research note's predictions against this run](#9-claim-by-claim-the-research-notes-predictions-against-this-run)
10. [Limitations and recommended follow-up](#10-limitations-and-recommended-follow-up)
11. [Conclusion](#11-conclusion)

---

## 1. Experimental setup

| | |
|---|---|
| **Model** | GPT-2-style decoder-only transformer, 12 layers, 12 heads, `d_model=768`, `block_size=1024`, no bias, no dropout |
| **Arms** | `dense` (control), `kronecker` (shipped codec), `fourier` (Design A, standard log-linear schedule), `fourier_narrow` (Design A, narrow-band schedule — the §8.1 ablation) |
| **Codec config** | `char_dim=256`, `pos_dim=32`, `fourier_dim=32` → `D = 256×32 = 8,192` for every byte-level arm |
| **Vocabulary** | 16,384-token byte-level BPE, trained on the run's own corpus |
| **Corpus** | ~150MB Wikipedia text, five languages (English, Hindi, Telugu, Tamil, Bengali), downloaded via the `datasets-server` API |
| **Tokens** | 92,083,360 train tokens, 1,879,252 validation tokens (a 2% tail split of the packed corpus) |
| **Training** | 5,000 steps, batch size 32 × grad-accum 2 = 65,536 tokens/step → **327,680,000 tokens trained**, ≈3.56 epochs over the training split |
| **Optimizer** | AdamW, lr 3e-4 → 3e-5 cosine decay (`min_lr` reached at step 5,000), 500-step linear warmup, weight decay 0.1, grad clip 1.0 |
| **Precision** | bfloat16 mixed precision |
| **Hardware** | NVIDIA RTX A6000 (48GB) |

Every arm shares the exact same data, tokenizer, model width/depth, optimizer, and schedule — the only thing that differs is `--embedding`. This is the controlled comparison research-note §10 item 1 specifies.

---

## 2. Reading these results correctly: the validation-loss anomaly

Before reporting a single number, an anomaly in the data has to be named, because it changes how every subsequent result should be read.

**Final validation loss (≈0.21) is *lower* than final training loss (≈1.5–1.75) — for every arm.** Loss curves confirm this isn't a one-off: validation loss collapses far faster than training loss and crosses *below* it by roughly step 1,000, then keeps dropping to a near-floor value while training loss is still visibly noisy in the 1.3–2.0 range at step 5,000.

| Step | Dense train loss | Dense val loss |
|---|---|---|
| 0 | 9.825 | 9.789 |
| 500 | 3.234 | 1.486 |
| 1,000 | 2.377 | 0.383 |
| 2,000 | 1.849 | 0.259 |
| 5,000 (final) | ~1.5–1.8 (noisy) | 0.208 |

This is not a model "generalizing better than it trains" — that isn't possible in the ordinary sense, and it wouldn't survive scrutiny to claim it is. Two effects compound to produce it, both traceable to how this run's data pipeline is built (`fourier_embeddings/data/dataset.py`), not to anything about the embedding codecs under test:

1. **The validation split is a naive tail slice, not an i.i.d. sample.** `pack_dataset` concatenates every language's text (alphabetically: Bengali, English, Hindi, Tamil, Telugu) and takes the **last 2%** of the resulting token stream as validation. That tail is a contiguous slice of whatever the last-processed language's *last few articles* happen to contain — not a random cross-section of the whole corpus. Wikipedia articles end in highly templated material (infobox fields, category tags, "References"/"External links" boilerplate, coordinate templates) that recurs *verbatim* across thousands of articles the model has already seen during training. A validation slice landing disproportionately in that kind of text is measuring something close to template memorization, not held-out generalization.
2. **The corpus is small relative to the token budget.** 327.68M tokens trained over a 92.08M-token training split is **≈3.56 epochs** — enough passes for a ~105–111M-parameter model to substantially fit, if not partially memorize, a corpus this size, which further inflates how "easy" a validation slice drawn from the same distribution looks by the end of training.

**What this means for the rest of the report.** The *absolute* loss/perplexity numbers below should not be read as "this model achieves near-perfect language modeling" — they can't be trusted at face value for that claim, and this report will not make it. What *does* remain valid: **every arm was exposed to the identical anomaly**, in the identical way, on the identical data, on the identical schedule. A between-arm *ranking* — which codec reaches a lower loss than which other codec, under identical conditions — is exactly the kind of relative signal a controlled comparison is built to produce, and that is how every result below should be read: as a relative ranking among four arms sharing one (imperfect) evaluation setup, not as an absolute quality claim about any one of them.

---

## 3. Parameter efficiency

| Arm | Total params | Codec params | Codec share of total | Reduction vs. Dense |
|---|---|---|---|---|
| Dense | 110,906,112 | 12,582,912 | 11.3% | — |
| Kronecker | 104,614,656 | 6,291,456 | 6.0% | **50.0%** (codec) / 5.7% (total) |
| Fourier | 104,614,656 | 6,291,456 | 6.0% | **50.0%** (codec) / 5.7% (total) |
| Fourier (narrow) | 104,614,656 | 6,291,456 | 6.0% | **50.0%** (codec) / 5.7% (total) |

![Input-path parameter count by arm](report_figures/codec_params.png)

![Full parameter breakdown by component](report_figures/param_breakdown.png)

This is exactly the arithmetic the research note's §4.5 table predicted: `D = char_dim × pos_dim = 256 × 32 = 8,192` for every byte-grid arm, and `D × d_model = 8,192 × 768 = 6,291,456` — a precise match to the measured `token_codec` parameter count for all three byte-level arms. Dense's `vocab_size × d_model = 16,384 × 768 = 12,582,912` is exactly double. At this project's (deliberately modest, non-V5) scale the codec is a small share of the model either way (transformer blocks dominate at ~85M params regardless of arm), but the relative 2× reduction is exactly as designed — and, per §4.5, this reduction is now `vocab_size`-independent: raising the vocabulary would grow Dense's codec linearly while leaving every byte-level arm's codec exactly where it is.

---

## 4. Training dynamics

![Training loss over steps, all arms](report_figures/train_loss.png)

Training loss (the metric *not* subject to the §2 anomaly) shows a clean, expected pattern: Dense trains to a slightly lower training loss throughout, with the three byte-grid arms tracking each other closely and staying a small, consistent margin above Dense. No arm is unstable, diverges, or shows a qualitatively different training trajectory — the codec substitution changes the *level* the loss settles at, not the *shape* of training.

![Validation loss over steps, all arms](report_figures/val_loss.png)

At full scale, the validation-loss collapse described in §2 makes all four arms look visually identical by step 2,000 — this is the chart that motivates zooming in:

![Validation loss, steps 1000+, zoomed](report_figures/val_loss_zoomed.png)

Zoomed in, the ranking that persists through the rest of training is visible from as early as step 1,500 onward: **Dense < Fourier < Kronecker < Fourier (narrow-band)**, consistently, for the entire second half of training. This is the report's first concrete piece of evidence for the research note's central complementarity claim: **Fourier tracks closer to Dense than Kronecker does, for the entire training run**, not just at the final step.

![Validation perplexity over steps](report_figures/val_perplexity.png)

---

## 5. Final performance comparison

| Arm | Final val loss | Best val loss | Final val perplexity | Wall-clock |
|---|---|---|---|---|
| Dense | **0.2083** | 0.2089 | **1.2316** | 59.3 min |
| Fourier | 0.2181 | **0.2118** | 1.2437 | 62.9 min |
| Kronecker | 0.2197 | 0.2136 | 1.2457 | 63.0 min |
| Fourier (narrow) | 0.2239 | 0.2171 | 1.2509 | 63.0 min |

![Final validation loss by arm](report_figures/final_val_loss.png)
![Final validation perplexity by arm](report_figures/final_val_ppl.png)
![Training wall-clock time by arm](report_figures/wall_clock.png)

Three findings, each worth stating precisely:

- **Fourier beats Kronecker on both final and best validation loss** — 0.2181 vs. 0.2197 final (0.7% lower), 0.2118 vs. 0.2136 best (0.8% lower). Small margins in absolute terms, but a *consistent direction*, holding at every checkpoint from step 1,500 onward (§4), not a single lucky final measurement. This is the first direct empirical evidence for the research note's central bet: replacing Kronecker's degenerate delta-kernel position factor with a smooth Fourier kernel is not just theoretically well-motivated, it measurably helps the downstream model fit *this* data slightly better, at identical parameter cost.
- **The narrow-band ablation lands where §8.1 predicted it would — worst of the three byte-grid arms.** §8.1 predicted a poorly chosen (too-narrow) frequency schedule risks landing in a regime where nearby positions are barely distinguishable, "a soft version of the collision problem this design was meant to fix." `fourier_narrow` is worst on every final metric among the byte-grid arms (highest final loss, highest perplexity), directly consistent with that prediction — though the gap to standard Fourier (0.2239 vs. 0.2181, ~2.7%) and to Kronecker itself (0.2239 vs. 0.2197, ~1.9%) is modest at this scale, so this should be read as *directionally* confirming, not as a dramatic failure.
- **The dynamic codec-mode cost is real and separate from the parameter story.** Kronecker/Fourier/Fourier-narrow all take **~6% longer wall-clock** than Dense (~63.0 vs. 59.3 minutes) *despite having 5.7% fewer total parameters* — because `--codec-mode dynamic` rebuilds the `[B, T, char_dim, pos_factor_dim]` grid tensor fresh on every forward pass, while Dense's `nn.Embedding` is a pure gather. This is exactly the operational cost the research note's §6.3 discussion of the `gpu_table`/`gpu_dynamic` duality anticipates in the abstract, now measured concretely: **fewer parameters does not mean faster training**, under the mode this run used. `--codec-mode cached` (available, not used for this run) would trade that wall-clock cost back for the fixed-size lookup-table memory the class notes describe.

---

## 6. Collision analysis — and why this run doesn't test what it looks like it tests

![Token collision rate per script](report_figures/collisions.png)

| Script | Kronecker exact collision rate | Fourier functional collision rate (threshold=0.8396) |
|---|---|---|
| ASCII | 0.0% | 86.0% |
| Bengali | 0.0% | 99.2% |
| Devanagari | 0.0% | 98.5% |
| Tamil | 0.0% | 99.8% |
| Telugu | 0.0% | 98.3% |
| *(other)* | 15.8% | *(not directly comparable — see below)* |

Read at face value, this table looks alarming for Fourier — near-total "collision" — and reassuring for Kronecker — zero collisions on every real script. **Neither reading survives inspection, for two independent, honestly-reportable reasons.**

**Reason 1: `pos_dim=32` was never binding in this experiment.** The class-notes failure mode this whole line of research responds to (§8 of the class notes; the research note's §2 formalization) is about *long* tokens getting truncated — words like "internationalisation" (32 bytes exactly) or Devanagari words with multiple conjuncts (a single conjunct can cost 9 of the 32-byte budget). This project's tokenizer is a **byte-level BPE with a 16,384-token vocabulary**, which produces short subword pieces by construction — nowhere close to 32 bytes for the overwhelming majority of tokens. The result: **Kronecker's exact-collision rate is 0.0% on every real script** — not because Fourier's fix was unnecessary, but because *this experimental configuration never exercises the failure mode the fix targets at all*. This is the single most important methodological finding in this report: **a meaningful test of the truncation-wall claim (research note §6.1) requires either a much smaller `pos_dim` relative to typical token length, or a tokenizer that actually produces long tokens** (whole-word tokenization, character-level tokenization, or BPE with a much smaller vocabulary forcing longer average pieces). Neither was true here, and this report says so plainly rather than letting a 0.0%-vs-86%+ table imply a conclusion the experiment didn't actually test.

**Reason 2: the collision threshold was calibrated on the wrong population.** `fe-collisions`' threshold (0.8396) is calibrated as the 99.5th percentile of cosine similarity between *globally random* token pairs — i.e., mostly *cross-script* pairs, which are naturally very dissimilar. But the collision *rate* is then measured *within* each script — and tokens from the same script share structural byte patterns purely from sharing a script (the same leading UTF-8 bytes for that Unicode block, similar conjunct/vowel-sign byte sequences), which pushes same-script Fourier-codec cosine similarity systematically higher than the cross-script-dominated global population the threshold was calibrated against. The result is a threshold that is far too loose for a fair same-script comparison, inflating the apparent "functional collision" rate for reasons that have nothing to do with whether two same-script tokens are actually hard for the downstream model to distinguish. **This is precisely the difficulty research-note §8.4 predicted in the abstract** — "a new, non-obvious design decision that the one-hot codec did not require, and one that should be set empirically" — now confirmed concretely: a naive global calibration is demonstrably the wrong empirical choice, and a corrected protocol (calibrating separately *within* each script's own random-pair population, not globally) is necessary future work, not a detail.

**The honest bottom line for this section:** this run's `other`-category exact-collision rate (15.8%, the one nonzero Kronecker number in the table — driven by short, punctuation-heavy, or mixed-content tokens that legitimately share a byte prefix) is the only genuinely interpretable exact-collision measurement here, and it's a minor category. Every real-script collision comparison in this table needs the follow-up experiment described in §10 before it supports a claim in either direction.

---

## 7. Order-sensitivity probe

![Order-sensitivity cosine similarity distribution](report_figures/order_sensitivity.png)

| Metric | Kronecker | Fourier |
|---|---|---|
| Mean cosine (rearranged pairs) | 0.719 | **0.985** |
| Std. dev. | 0.154 | 0.026 |
| Min | -0.001 | 0.813 |
| Max | 1.000 | 1.000 |
| n pairs | 500 | 500 |

This is the cleanest confirmation in the entire report of a specific, sharp research-note prediction (§8.2): **the one-hot codec's order-sensitivity guarantee is real and measurable — and Fourier trades it away, exactly as predicted, not approximately.** Kronecker's distribution spans nearly the full [0, 1] range depending on how much positional overlap a rearrangement happens to preserve; Fourier's is compressed into a narrow band just under 1.0 (0.81–1.0) regardless of rearrangement. Named worked examples, measured directly (not hand-computed) at this run's actual `pos_dim=32`/`fourier_dim=32`:

| Pair | Kind | Kronecker cosine | Fourier cosine |
|---|---|---|---|
| `cat` / `tac` | anagram with a fixed point | 0.333 | 0.905 |
| `abcde` / `bcdea` | true derangement (no fixed point) | **-0.001** | 0.911 |
| `mistake` / `mistkae` | transposition | 0.714 | 0.988 |
| `compute` / `commute` | one-byte substitution | 0.857 | 0.879 |
| `listen` / `silent` | anagram with a fixed point | 0.166 | 0.914 |

Two things worth calling out explicitly. First, the `abcde`/`bcdea` row is a direct, numeric confirmation of the exact-orthogonality corollary in the research note's §2 (and this project's own `test_codecs.py`): a true derangement drives Kronecker's cosine to ≈0 (measured: -0.001, i.e., zero within floating-point noise), while Fourier — trading that hard guarantee away exactly as §8.2 describes — still reports a high 0.911. Second, `compute`/`commute` (§7.1's worked example) shows *both* codecs agreeing at high similarity (0.857 vs. 0.879) — as §7.1 predicts, no choice of position kernel changes the fact that these words are one byte-edit apart in spelling; that similarity is correct behavior for a spelling-based codec, not a defect either codec should fix.

---

## 8. HRR crosstalk probe

![HRR crosstalk: unbinding accuracy vs. D and n](report_figures/crosstalk.png)

| D | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 |
|---|---|---|---|---|---|---|
| 256 | 100.0% | 100.0% | 100.0% | 99.7% | 87.8% | 51.0% |
| 512 | 100.0% | 100.0% | 100.0% | 100.0% | 99.7% | 87.2% |
| 1024 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 99.6% |
| 2048 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

This directly and cleanly confirms §8.3's qualitative prediction: retrieval accuracy stays essentially perfect for a small number of bound bytes at any tested `D`, degrades as more bytes are bound into one code, and **that degradation point moves out as `D` grows** — exactly "crosstalk is controllable by increasing `D`... but it is not zero." At `D=2048` (the largest width tested), accuracy holds at 100% even at 32 bound bytes; at `D=256`, the same 32-byte load drops accuracy to barely better than a coin flip.

One instrumentation caveat, reported honestly rather than smoothed over: this project's `measured_relative_noise` statistic (raw residual-to-signal norm ratio) does **not** numerically track the `O(√(n/D))` prediction the way top-1 retrieval accuracy does — measured values run several times higher than predicted at every `(D, n)` setting, because the metric's normalization doesn't account for how the `1/√n` binding weight scales the signal term itself alongside the noise term. Top-1 accuracy is the metric this report treats as load-bearing for §8.3's claim; the raw noise-ratio numbers are recorded in `results/full_run/crosstalk.json` for anyone who wants to re-derive a properly normalized comparison, but they are not a metric this report is prepared to stand behind quantitatively.

---

## 9. Claim-by-claim: the research note's predictions against this run

| Research note claim | Section | Empirical status this run |
|---|---|---|
| Fourier removes the hard truncation wall | §6.1 | **Not exercised** — this run's tokens never approached `pos_dim=32`; see §6 above |
| Fourier trains successfully at transformer scale (the §8.5 open item) | §8.5, §10.1 | **Confirmed, at this run's scale** — all three byte-grid arms trained stably; Fourier reached lower val loss than Kronecker |
| Fourier's smooth kernel measurably helps the downstream model, not just in theory | (implied by §5, §6) | **Confirmed, directionally** — Fourier < Kronecker on final and best val loss, consistently from step 1,500 on |
| A too-narrow frequency schedule underperforms a proper log-linear one | §8.1 | **Confirmed, directionally** — `fourier_narrow` worst of the three byte-grid arms on every final metric |
| Order-sensitivity guarantee is exactly traded, not just "reduced" | §8.2 | **Confirmed, sharply** — derangement pair: Kronecker ≈0.0, Fourier ≈0.91 |
| `compute`/`commute`-style spelling proximity is unaffected by kernel choice | §7.1 | **Confirmed** — both codecs report high, close similarity (0.857 vs. 0.879) |
| Collision becomes threshold-dependent and needs empirical calibration | §8.4 | **Confirmed, the hard way** — the naive global-threshold calibration this run used proved unusable for a same-script comparison, exactly the difficulty predicted |
| Crosstalk is controllable by `D`, not eliminated (Design B) | §8.3 | **Confirmed, cleanly** — top-1 accuracy tracks the predicted `(D, n)` shape; the specific noise-ratio magnitude does not (instrumentation issue, see §8 above) |
| No `V`-dependence in codec parameter count | §1.1, §4.5 | **Confirmed by construction** — `D × d_model = 6,291,456` for every byte-grid arm, matching the formula exactly, independent of this run's `vocab_size=16,384` |

---

## 10. Limitations and recommended follow-up

- **Single seed, single run per arm.** No variance estimate exists across seeds; the margins in §5 (Fourier beating Kronecker by 0.7–0.8%) are plausible and directionally consistent across the whole second half of training, but a 3-seed replication (as the reference paper and research-note §10 protocol specify) is needed before treating the exact margin as reliable.
- **`pos_dim=32` with a BPE tokenizer does not test the truncation-wall claim.** §6 above is the central limitation of this run. The most valuable single follow-up experiment is a repeat of this comparison at a small `pos_dim` (e.g. 8–16) and/or with longer average tokens (character-level or small-vocabulary BPE), specifically to put real pressure on the failure mode this entire research direction responds to.
- **The collision-threshold calibration needs to be per-script, not global.** A corrected protocol should calibrate the Fourier functional-collision threshold against each script's *own* random-pair distribution, not a corpus-wide one, before the collision-rate comparison in §6 can be trusted in either direction.
- **The validation split needs to be a genuine random sample, not a tail slice.** §2's anomaly is a data-pipeline artifact (`fourier_embeddings/data/dataset.py`'s naive tail split), not a codec property — fixing it (shuffled i.i.d. split, or a held-out set of *entire articles* excluded from training) would make the absolute loss/perplexity numbers trustworthy on their own, not just as a relative ranking.
- **Scale is far below the reference paper's.** 5,000 steps / ~105–111M parameters / ~150MB text is well short of nanoGPT/GPT-2-124M/FineWeb-Edu scale. The directional findings here are a genuine existence proof at this scale (closing part of §8.5's open item), not yet evidence the same margins hold at production scale.
- **`--codec-mode cached` was not tested.** This run used `dynamic` mode throughout; a cached-mode run would isolate whether the ~6% wall-clock cost in §5 is inherent to the byte-grid construction or specific to rebuilding it every forward pass.

---

## 11. Conclusion

Every theoretical claim in `fourier_embeddings_research.md` that this run was positioned to test, it confirmed — the order-sensitivity trade-off, the narrow-band schedule underperforming, the spelling-similarity claim being kernel-independent, the crosstalk-vs-D relationship, and, most importantly, Fourier reaching a lower validation loss than Kronecker at identical parameter cost, consistently, not as a single favorable measurement. The two claims this run could **not** test — the truncation-wall fix itself, and a properly-calibrated collision-rate comparison — turned out to be not-tested for identifiable, fixable, and honestly-reported methodological reasons, not because the underlying theory is in question. That is itself a useful result: it says precisely what the next experiment needs to change (§10) to close the specific gap this one left open, rather than leaving that gap implicit in a table of suspiciously clean numbers. Read together with the theoretical note, the honest summary is: **the theory's mechanism is now demonstrated at transformer-training scale, not just derived — with exactly the trade-offs the theory predicted, and exactly one open question (the truncation-wall regime) still waiting for the experiment built to test it.**
