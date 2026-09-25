# Session 13: Reversible LLM Experiments

Assignment: train a ~20M-parameter GPT on 50M tokens at batch `x`, repeat with the two reversible architectures from
[Reversing Large Language Models for Efficient Training and Fine-Tuning](https://arxiv.org/html/2512.02056v2)
(midpoint and Euler), pick the better one, and scale it to `2x`, `4x`, `8x` on a ~40 GB GPU.

| # | Experiment | Batch |
|---|---|---|
| 1 | Baseline | x = 128 |
| 2 | Reversible midpoint (h = 0.25) | x |
| 3 | Reversible Euler (symplectic / Hamiltonian) | x |
| 4-6 | Best of 2 and 3 | 2x, 4x, 8x (256, 512, 1024) |

## Project shape

- `src/revllm/model.py`: GPT (9 layers, d=256, 8 heads, ctx 512, tied 50,257-vocab embeddings = 20.1M params) and FLOPs accounting.
- `src/revllm/reversible.py`: O(1)-memory reversible stacks: one autograd Function over all layers, one sub-layer recompute per layer in backward, autocast-consistent backward, and bit-exact reversal with float64 fixed-point streams.
- `src/revllm/loss.py`: fused chunked lm_head + cross-entropy. It never materializes the (tokens x vocab) logits and computes the lm_head gradient in the same pass.
- `src/revllm/trainer.py`: instrumented training loop (loss, accuracy, perplexity, LR, grad norm, per-phase timing, tokens/s, TFLOP/s, MFU, memory breakdown, reversibility diagnostics).
- `src/revllm/probe.py`: real-training-step memory probes and a max-batch search.
- `scripts/build_notebook.py`: source of truth for `notebooks/reversible_llm.ipynb` (regenerate after editing).
- `tests/`: gradient-exactness, inverse, fused-loss, bf16 bit-exact reversal and O(1)-memory tests.

## Run (GPU machine)

```bash
uv sync
uv run pytest -q
cd notebooks
uv run jupyter nbconvert --to notebook --execute --inplace reversible_llm.ipynb --ExecutePreprocessor.timeout=-1
uv run aim up          # from notebooks/
```

Options (environment variables): `REVLLM_DEVICE` (default `cuda:0`), `REVLLM_BATCH_X` (default 128),
`REVLLM_REUSE=1` (default: skip runs already finished in `results/`, so an interrupted notebook resumes),
and `REVLLM_SMOKE=1` (a tiny WikiText-2 end-to-end check for small GPUs; writes to `*_smoke/` directories).

## Outputs (under `notebooks/`)

- `results/<run>.json`: config, environment, summary, per-step history, eval curve, diagnostics.
- `results/runs.jsonl`: one summary line per run. `results/final_comparison.csv`: the six-run table.
- `logs/<run>.log` (text) and `logs/<run>.metrics.jsonl` (per-step metrics, streamed while training).
- `.aim/`: all per-step and per-eval metrics, hparams and summary. The notebook reindexes the repo at the end.
  If runs look empty when queried from Python, run `uv run aim storage --repo . reindex --yes`.
- `../../webapp/data.js`: exported summary for the static webapp.
