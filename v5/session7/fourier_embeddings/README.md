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
pins `torch` to the `cu124` wheel index because this WSL box's driver
reports CUDA 12.8 support and PyPI's default `torch` wheels now ship a
CUDA-13 build that a CUDA-12.8 driver can't initialize. On the A6000 box,
check `nvidia-smi`'s "CUDA Version" line — if it's 12.8+ the pinned index is
still correct; if the driver is newer you can drop the `[tool.uv.sources]` /
`[[tool.uv.index]]` block in `pyproject.toml` and just `uv sync` against the
default index.

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
`BLOCK_SIZE`, `MAX_STEPS`, `BATCH_SIZE`, `GRAD_ACCUM`, `ARMS`,
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
