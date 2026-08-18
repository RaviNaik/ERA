# Fourier vs. Kronecker Byte Embeddings — Practical Verification

This project is the empirical companion to
[`fourier_embeddings_research.md`](../fourier_embeddings_research.md): a
controlled, from-scratch transformer LM setup that trains and compares
**five token-embedding codecs** and runs the three analytical probes the
research note's [§10 minimal validation protocol](../fourier_embeddings_research.md#10-a-minimal-validation-protocol)
asks for.

It answers, empirically, the question the research note poses on paper:
*does replacing the Kronecker codec's one-hot position factor with a Fourier
(spectral) position factor fix the truncation-wall / rigid-`D` problems, and
at what cost?*

## The five codecs (`fourier_embeddings/model/codecs.py`)

| `--embedding` | What it is | Research-note ref |
|---|---|---|
| `dense` | Plain `nn.Embedding` — the "Normal Embedding" baseline | — |
| `kronecker` | Shipped one-hot byte-value ⊗ one-hot byte-position codec | class notes §7, research note §2 |
| `fourier` | **Design A** — one-hot byte-value ⊗ sinusoidal (log-linear, multi-octave) position codec | research note §4.2 |
| `fourier_narrow` | Design A with a deliberately narrow-band frequency schedule | research note §8.1 ablation |
| `hrr` | **Design B** — holographic reduced-representation (circular-convolution) binding | research note §4.4 |

`kronecker` and `fourier`/`fourier_narrow` share one implementation
(`ByteGridCodec`) — the position-factor table is the *only* thing that
differs, which is a direct, literal instantiation of the research note's
§5 claim that Kronecker is Fourier's zero-bandwidth degenerate case, not a
competing mechanism. `tests/test_codecs.py` checks this and the codecs'
other headline claims (Theorem 1, Theorem 2, the truncation-wall removal,
"no `V` in the codec parameter count") directly against closed-form math.

## What gets measured

1. **Controlled LM comparison** (`fe-run-experiment`) — all five arms
   trained with an identical model size, optimizer, schedule, and corpus;
   only `--embedding` differs. Produces per-arm `metrics.json` (loss curves,
   val perplexity, parameter breakdown, wall-clock/tokens-per-sec) plus an
   aggregated `comparison.json` / `comparison.md`.
2. **Collision measurement** (`fe-collisions`) — exact Kronecker collisions
   (identical first-`pos_dim`-byte prefixes) vs. calibrated-threshold
   functional collisions under Fourier, both broken down **per script**
   (Devanagari / Bengali / Tamil / Telugu / Kannada / ASCII), matching class
   notes §8 / Widget 9's own protocol.
3. **Order-sensitivity probe** (`fe-order-sensitivity`) — cosine similarity
   of anagram/transposition word pairs under both codecs, reproducing
   research-note §4.3's `"cat"`/`"tac"` worked example at scale and
   quantifying the §8.2 trade (a hard 0-similarity guarantee under
   Kronecker, traded for a soft nonzero similarity under Fourier).
4. **Crosstalk probe** (`fe-crosstalk`, HRR only) — unbinding accuracy and
   residual-noise scaling vs. `D` and bytes-bound-per-token, checked against
   Plate's `O(sqrt(n/D))` prediction (research note §8.3).

Every entrypoint: writes a plain-text log to `logs/`, tracks scalars to
[Aim](https://aimstack.io/) (`aim_repo/`, browse with `aim up --repo
aim_repo`), and writes its numeric results as JSON under `results/`.

## Project layout

```
fourier_embeddings/
  model/
    codecs.py        # DenseEmbedding, KroneckerEmbedding, FourierKroneckerEmbedding, HRRBindingEmbedding
    gpt.py            # nanoGPT-style decoder-only transformer, pluggable token codec
    config.py         # ModelConfig / TrainConfig dataclasses
  data/
    download.py        # pulls a multilingual (en/hi/te/ta/bn) Wikipedia sample via HF datasets-server
    tokenizer.py        # trains/loads a byte-level BPE tokenizer; builds the id->bytes map every byte codec needs
    dataset.py           # packs text into a binary token stream; random-block batch loader
  training/
    train.py             # single-arm training entrypoint (the CLI in full)
    run_experiment.py     # runs every arm back-to-back, aggregates comparison.json/.md
  analysis/
    collisions.py, order_sensitivity.py, crosstalk.py
tests/
  test_codecs.py       # codec math vs. the research note's theorems
scripts/
  smoke_test.sh         # tiny end-to-end run (this WSL box)
  run_full_experiment.sh # the real run (A6000)
```

## Setup

```bash
uv sync
```

**GPU driver note (read before syncing on a new machine):** `pyproject.toml`
pins `torch` to the `cu121` wheel index. PyPI's default `torch` wheels now
ship a CUDA-13 build, which needs a newer driver than either dev machine
this project has been run on has (a WSL box at driver-reported CUDA 12.8, an
A6000 box at driver-reported CUDA 12.2) — a driver only runs wheels built
against its *own or older* CUDA version, so the pin has to satisfy the
lowest CUDA Version across every machine you `uv sync` this on. cu121 clears
both with room to spare. **Before syncing on a new machine, run
`nvidia-smi` and check its "CUDA Version" field** (top-right of the header
line): if it's below 12.1, lower the pin further; if every machine reads
comfortably above 12.1, you can raise it for a newer torch release. After
changing the pin, `uv sync` again and verify with:
```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
This should print `True`. If it silently prints `False` instead of erroring
(this is normal torch behavior for a version mismatch — it degrades instead
of crashing), training falls back to CPU with no other symptom until you
notice it's using nothing but CPU time. `fe-train` now checks this itself
and logs a loud `*** CUDA IS NOT AVAILABLE ***` warning if it happens; pass
`--require-cuda` (set by default in `run_full_experiment.sh`) to make it a
hard error instead of a warning, so a misconfigured GPU box fails in the
first second rather than after hours of CPU training.

**WSL note:** if your project directory lives on a Windows-drive mount
(`/mnt/c`, `/mnt/d`, ...), point the virtualenv at a native Linux path
before syncing — installing ~1000 small `torch` files onto a 9p-mounted
drive is extremely slow (this is what happened building this project):

```bash
export UV_PROJECT_ENVIRONMENT=~/.venvs/fourier_embeddings
uv sync
```

## Running the smoke test (this machine)

```bash
bash scripts/smoke_test.sh
```

Downloads ~1.5MB of text total, trains a 4096-token tokenizer, packs a
tiny dataset, trains all five arms for **40 steps** each on a toy
(2-layer, 64-dim) model, runs all three analysis scripts, and runs the unit
tests — end to end in a few minutes on CPU or a small GPU. This only proves
the pipeline is correct; it is not a claim about which codec is "better" (40
steps on a few hundred KB of text is far below the noise floor).

## Running the real experiment (A6000)

```bash
bash scripts/run_full_experiment.sh
```

Environment variables override the defaults (see the script header):
`TARGET_MB_PER_LANG`, `VOCAB_SIZE`, `N_LAYER`, `N_HEAD`, `N_EMBD`,
`BLOCK_SIZE`, `MAX_STEPS`, `BATCH_SIZE`, `GRAD_ACCUM`, `ARMS`, `GPU_IDS`,
`EXPERIMENT_NAME`. Defaults land at roughly a ~50M-parameter GPT-2-small-ish
model trained on ~150MB of multilingual text (en/hi/te/ta/bn Wikipedia) for
20k steps — comfortably within an A6000's 48GB, and closer (though still far
short of) the reference paper's nanoGPT/GPT-2-124M/FineWeb-Edu scale that
the research note's §10 protocol describes. Raise `VOCAB_SIZE`,
`TARGET_MB_PER_LANG`, model width/depth, and `MAX_STEPS` from there as your
time/compute budget allows.

Track a run live with:

```bash
uv run aim up --repo aim_repo
```

### Multi-GPU

`fe-run-experiment` (and therefore `run_full_experiment.sh`) can run
multiple arms **concurrently, one per physical GPU**, instead of one arm at
a time on a single card:

```bash
GPU_IDS="0 1" bash scripts/run_full_experiment.sh
# or directly:
uv run fe-run-experiment --gpu-ids 0 1 ...
```

This is arm-level parallelism, not single-model multi-GPU (DDP) training —
each arm still trains on exactly one GPU (via `CUDA_VISIBLE_DEVICES`), but
up to `len(gpu-ids)` arms run at once, so a 4-arm comparison on 2 GPUs
finishes in roughly the time of 2 sequential arms instead of 4. This fits
this project's actual bottleneck (many independent short-ish runs to
compare) with far less complexity/risk than distributed training would add
for a single ~50M-parameter model, which isn't large enough to need DDP on
its own merits. A GPU-id queue gates scheduling (a worker only starts once
an actual GPU is free, regardless of how unevenly long individual arms
run), so this is safe with `--resume`/skip-completed-arms and with an
uneven number of arms vs. GPU ids. Leave `GPU_IDS`/`--gpu-ids` unset for the
original one-arm-at-a-time behavior (e.g. on a single-GPU box).

If you'd rather use a second idle GPU to make one arm itself faster instead
of running two arms at once, that's `--batch-size`/`--grad-accum-steps`
tuning on a single GPU (see below), not multi-GPU — this project doesn't
implement DDP for a single model.

**Batch size vs. free GPU memory:** if `nvidia-smi` shows a GPU well under
its memory ceiling during a run (e.g. 30GB/49GB used), you can often raise
`BATCH_SIZE` to use the headroom and get more tokens trained per step in
the same wall-clock — but note this changes the effective batch size
(tokens/step = batch_size × block_size × grad_accum_steps), which is a real
change to the training regimen, not a free win. Increase it *before*
starting a run (ideally before any arm in a comparison has trained a single
step), not partway through one — `fe-run-experiment` forwards the same
hyperparameters to every arm, so as long as it's set once for the whole
comparison, every arm still gets an identical regimen, which is what the
controlled-comparison protocol (research note §10) requires. The
Fourier/Kronecker "dynamic" codec mode builds an explicit per-token grid
tensor (roughly `batch_size × block_size × char_dim × pos_factor_dim × 4`
bytes) that dense embeddings don't have, so those arms use somewhat more
memory than dense at the same batch size — leave some extra margin (a few
GB) rather than sizing the batch right up to what dense alone can fit.

## Resuming an interrupted run

Everything above is safe to just re-run after a crash, an OOM, a preemptible
instance being reclaimed (SIGTERM), or you stopping it with Ctrl-C:

- **`fe-download-data` / `fe-train-tokenizer` / `fe-pack-dataset`** each skip
  their work by default if the output already exists (a language's `.txt`
  file that already reached its target size, an existing `tokenizer.json`,
  an existing packed `data_bin/`). Pass `--overwrite` to force a redo — you
  need to if the corpus languages, target size, vocab size, or tokenizer
  changed, since a stale skip would otherwise silently reuse the old one.
- **`fe-train`** checkpoints to `results/<run-name>/last.pt` every
  `--save-interval` steps (model, optimizer, step, history, best-val-loss,
  and — if Aim is on — the Aim run's hash, so metrics continue into the same
  run instead of starting a new one). A SIGTERM or Ctrl-C is caught and
  triggers one extra "emergency" checkpoint at the last completed step
  before the process exits, so a graceful stop never loses more than a
  fraction of a `--save-interval`. Continue an interrupted run with:
  ```bash
  uv run fe-train --run-name <same-run-name> --resume <same other args>
  ```
  `--resume` refuses to proceed if `results/<run-name>/last.pt` doesn't
  exist, or if its saved model config doesn't match the current args
  (mismatched `--embedding`/model-size/`--pos-dim`/... flags) — it will not
  silently load a checkpoint into the wrong-shaped model. A crash that
  happens between periodic saves (a hard kill, an actual GPU fault) loses
  only the steps since the last checkpoint, not the whole run.
- **`fe-run-experiment`** (and therefore `run_full_experiment.sh`, which
  calls it) checks each arm's `results/<experiment-name>_<arm>/` before
  running it: an arm with a `metrics.json` is treated as done and skipped;
  an arm with a checkpoint but no `metrics.json` is automatically re-invoked
  with `--resume`; an arm with neither starts fresh. So re-running
  `bash scripts/run_full_experiment.sh` after any interruption — mid-arm or
  between arms — does the right thing with no extra flags. Pass `--force` to
  `fe-run-experiment` to retrain every requested arm regardless (this is
  what `scripts/smoke_test.sh` does, since its job is to re-exercise the
  whole pipeline every time, not skip a previous smoke run's results).
- **Log files** (`logs/*.log`) are appended to, never overwritten, so
  nothing is lost across separate invocations — each (re)start writes a
  `==== run start (resume=...) ====` line so you can tell separate attempts
  apart when reading one back.

## Individual commands

Every step above is also its own CLI (`uv run fe-<name> --help`):

```
fe-download-data        # data_raw/{en,hi,te,ta,bn}.txt + manifest.json
fe-train-tokenizer       # tokenizer_out/tokenizer.json + byte_stats.json
fe-pack-dataset           # data_bin/{train,val}.bin + meta.json
fe-train                   # one arm: results/<run-name>/{metrics.json,*.pt}
fe-run-experiment            # all arms: results/<experiment-name>/{comparison.json,comparison.md}
fe-collisions                 # results/collisions.json
fe-order-sensitivity            # results/order_sensitivity.json
fe-crosstalk                     # results/crosstalk.json (HRR only)
```

## What "proving the claim" means here

The research note's §9 complementarity table makes falsifiable predictions.
This project is built so each one has a direct empirical readout:

- *"Hard truncation wall at `pos_dim` — solved."* → `fe-collisions`'s
  `exact_kronecker_collisions` (nonzero, script-skewed) vs.
  `fourier_functional_collisions` (threshold-calibrated, structurally
  incapable of exact collision) at equal `D`.
- *"Order-sensitivity guarantee softened."* → `fe-order-sensitivity`'s
  `frac_exactly_zero` and mean cosine for Kronecker (should sit at/near 0
  for anagrams) vs. Fourier (should be reliably > 0).
- *"Byte-similar, semantically-distant clustering — not solved by either
  codec."* → both codecs' cosine similarity for the `compute`/`commute` pair
  in `fe-order-sensitivity`'s output should be high and close to each other.
- *"No existence proof at transformer-training scale."* → `fe-run-experiment`'s
  `comparison.md` val-loss/perplexity table is exactly that proof, at
  whatever scale you run it.
- *"Superposition crosstalk (Design B)."* → `fe-crosstalk`'s
  `measured_relative_noise` vs. `predicted_relative_noise_O(sqrt((n-1)/D))`.

None of this is meant to declare a universal winner — per the research
note's conclusion, Fourier fixes the codec's *geometry*, not its semantics,
so the honest reading of any run here is "did the geometry issues move the
way the theorems predict," not "which arm wins."
