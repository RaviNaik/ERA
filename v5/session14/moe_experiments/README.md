# Session 14 — Dense nanoGPT → Mixture-of-Experts

**Assignment.** Train a traditional linear-layer (MLP) nanoGPT, track everything, then train an MoE
version, and finally *train a linear model and convert it into an MoE* — showing the converted model
keeps training and keeps reducing loss.

| # | Notebook | What it does |
|---|---|---|
| 1 | [`01_dense_baseline`](notebooks/01_dense_baseline.ipynb) | data, parameter/FLOP accounting, sanity checks, dense training, curves, samples |
| 2 | [`02_moe_from_scratch`](notebooks/02_moe_from_scratch.ipynb) | MoE layer from first principles (router, top-k, load-balance loss, correctness check, toy experiment), MoE trained from scratch with the dense budget |
| 3 | [`03_dense_to_moe_upcycling`](notebooks/03_dense_to_moe_upcycling.ipynb) | **convert the trained dense checkpoint into an MoE and continue training**, vs a dense-continue control; expert divergence; routing; ablations |
| 4 | [`04_analysis_and_verdict`](notebooks/04_analysis_and_verdict.ipynb) | all runs side by side, what each expert specialised in, pass/fail checklist |

> **Status:** code is smoke-tested only (`scripts/smoke_test.py`, tiny model, a few steps). The real runs are
> done on a GPU server; no result numbers are claimed here until the executed notebooks are committed.

## Setup

```bash
uv sync                                   # creates .venv, installs torch (CUDA 12.4 wheels), aim, jupyter, ...
uv run python scripts/smoke_test.py       # ~1 min: core checks
uv run python scripts/smoke_test.py --notebooks   # + executes all 4 notebooks on a tiny model in a scratch dir
```

## Running the real thing

```bash
uv run python scripts/run_all.py          # builds + executes notebooks 1-4 in order (MOE_PRESET=base)
uv run python scripts/run_all.py 3 4      # only some
MOE_PRESET=small uv run python scripts/run_all.py   # laptop-sized
MOE_ABLATIONS=1 uv run python scripts/run_all.py 3  # also run the optional ablations in notebook 3
uv run aim up                             # browse all experiments (run from this directory)
uv run python scripts/aim_summary.py      # one-line-per-run table
```

Or open the notebooks in Jupyter (`uv run jupyter lab`) and run them top to bottom. Runs are
**resumable and idempotent**: a finished run is reloaded from `checkpoints/<label>.pt`, an interrupted one
resumes from its last checkpoint. Delete `checkpoints/<label>.pt` to retrain a run.

## Experiment design

**Model** — session-11 nanoGPT (pre-LN, learned positions, tied head, GELU MLP), GPT-2 BPE vocabulary (50 257, padded to 50 304).

| preset | layers × heads × width | context | dense params | MoE params (total / active) |
|---|---|---|---|---|
| `base` (default) | 8 × 8 × 512 | 512 | 51.2 M | 110.0 M / 59.7 M |
| `small` | 6 × 6 × 384 | 256 | — | — |
| `smoke` | 4 × 2 × 64 | 32 | 0.23 M | 0.43 M / 0.30 M |

**Data** — WikiText-103 (raw), GPT-2 BPE, ≈120 M train tokens, cached as `uint16`. Evaluation always uses the same fixed
batches, so every run is scored on identical validation tokens.

**MoE layer** — 8 experts, top-2, in every 2nd block (blocks 1, 3, 5, 7); the other blocks stay dense. Dropless token dispatch,
fp32 router, softmax scores (sigmoid available), gates renormalised over the selected experts.
**Balancing** is loss-free: a per-expert bias is added to the scores *only to choose the top-k* and is nudged after every
step (`bias_update = 1e-3`) towards even load, so no balancing gradient touches the router. A tiny Switch-style auxiliary loss (1e-4)
and a router z-loss (1e-3) act as backstops. Imbalance is reported as **MaxVio** (0 = perfectly even) alongside load entropy and dead-expert counts.

**Runs**

| label | init | schedule | purpose |
|---|---|---|---|
| `dense` | random | 6000 steps, lr 6e-4, cosine | part 1 baseline |
| `moe_scratch` | random | same as `dense` | part 2: MoE with the identical budget |
| `moe_upcycled` | `dense` checkpoint, MLPs → 8 copies each | +3000 steps, lr 3e-4 re-warm, cosine; probabilistic expert selection for the first 300 steps | **the trainer's task** |
| `dense_continue` | `dense` checkpoint | +3000 steps, same schedule & same batches as above | control: is the MoE better than just training longer? |

### Why the conversion is lossless (sparse upcycling)

Each selected MLP becomes `E` exact copies; the router is new and small. With gates renormalised over the top-k,
`Σ_e g_e f_e(x) = f(x) · Σ_e g_e = f(x)`, independent of the router. So the converted model's loss at step 0 equals the dense
model's — training "continues" rather than restarts — and the notebooks verify this numerically (logit diff at fp32 round-off).
Afterwards the experts diverge because the router sends them different tokens; notebook 3 measures that
(expert cosine similarity, drift from the dense MLP) together with load balance and dead experts.

Clones are hard for a hard-top-k router to tell apart, which can funnel each clone family onto a few members. To protect against that,
experts are chosen by Gumbel-top-k sampling for the first `explore_steps` steps after conversion (still lossless, since the clones are identical),
then by hard top-k. The conversion budget (+50 % of the dense run) is deliberately modest: upcycling is the cheap route when the extra budget
is small relative to the dense cost; with a large budget, training the MoE from scratch is the alternative (`moe_scratch` is the reference for that).

### Ablations (optional, notebook 3, `MOE_ABLATIONS=1`)
`no_explore`, `no_balancing`, `aux_loss_only`, `sigmoid_router`, `noise0.02` (noisy copies), `drop_upcycle0.5` (re-draw half of each expert's neurons — faster divergence, not lossless),
`top1`, `experts16` — each a full continuation from the same dense checkpoint on the same batches (experiment `session14_ablations` in Aim).

## Tracking

Everything is logged to **Aim** (`.aim/`, git-ignored): per-step loss / lr / grad-norm / tokens-per-second, train & val loss and
perplexity at each eval, and for MoE runs per-layer expert load, load entropy, max-load, dead-expert count, load-balance and z losses.
Phase-2 runs continue the global step axis of the dense run, so the curves line up in the Aim UI.
Experiment names: `session14` (main runs) and `session14_ablations`. A JSON copy of each run's history is written to
`results/<label>.json`; figures go to `assets/`.

## Layout

```
src/moe/        config.py  data.py  model.py (GPT + MoEMLP)  upcycle.py  trainer.py  runs.py  analysis.py  viz.py  utils.py
scripts/        build_notebooks.py (source of the notebooks)  run_all.py  smoke_test.py  aim_summary.py
notebooks/      01 … 04
checkpoints/    dense.pt, moe_scratch.pt, moe_upcycled.pt, dense_continue.pt   (git-ignored)
results/ assets/  per-run histories, figures
```
