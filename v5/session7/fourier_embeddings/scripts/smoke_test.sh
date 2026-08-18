#!/usr/bin/env bash
# End-to-end pipeline smoke test: tiny data, tiny model, <1 epoch, all four
# main arms + the analysis scripts. Meant to prove the whole pipeline works
# on modest hardware (a 4GB laptop GPU / CPU) before the real run on the
# A6000. Not a claim of any result -- see run_full_experiment.sh for that.
set -euo pipefail
cd "$(dirname "$0")/.."

# If the project directory lives on a slow Windows-drive mount under WSL
# (/mnt/c, /mnt/d, ...), point the venv at a native Linux path instead (see
# README.md's WSL note). Harmless / no-op on a real Linux box or a machine
# where you already `uv sync`'d in place.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/fourier_embeddings}"

echo "=== [1/6] download a small multilingual sample ==="
uv run fe-download-data --out-dir data_raw_smoke --target-mb-per-lang 0.3 \
    --langs en hi te ta bn --log-file logs/smoke_download.log

echo "=== [2/6] train a small tokenizer ==="
uv run fe-train-tokenizer --input-dir data_raw_smoke --out-dir tokenizer_out_smoke \
    --vocab-size 4096 --log-file logs/smoke_tokenizer.log

echo "=== [3/6] pack the dataset ==="
uv run fe-pack-dataset --input-dir data_raw_smoke --tokenizer-path tokenizer_out_smoke/tokenizer.json \
    --out-dir data_bin_smoke --val-fraction 0.05 --log-file logs/smoke_pack.log

echo "=== [4/6] run all embedding arms for a handful of steps ==="
# --force: this script is meant to genuinely re-exercise the pipeline every
# time it's run (e.g. after a code change), not silently skip already-
# completed arms from a previous smoke-test invocation. See README's
# "Resuming an interrupted run" section for what --force turns off.
uv run fe-run-experiment --experiment-name smoke --force \
    --data-dir data_bin_smoke --tokenizer-path tokenizer_out_smoke/tokenizer.json \
    --arms dense kronecker fourier fourier_narrow hrr \
    --block-size 64 --n-layer 2 --n-head 2 --n-embd 64 \
    --pos-dim 16 --fourier-dim 16 --hrr-dim 256 \
    --max-steps 40 --batch-size 8 --eval-interval 20 --eval-iters 5 \
    --log-interval 5 --save-interval 1000 --dtype float32 --no-aim

echo "=== [5/6] analysis scripts (collisions / order-sensitivity / crosstalk) ==="
uv run fe-collisions --tokenizer-path tokenizer_out_smoke/tokenizer.json \
    --pos-dim 16 --fourier-dim 16 --d-model 32 --out results/smoke_collisions.json \
    --log-file logs/smoke_collisions.log
uv run fe-order-sensitivity --pos-dim 12 --fourier-dim 12 --n-random-pairs 40 \
    --out results/smoke_order_sensitivity.json --log-file logs/smoke_order_sensitivity.log
uv run fe-crosstalk --D-values 64 128 --n-bound-values 1 4 8 --pos-dim 16 --n-trials 30 \
    --out results/smoke_crosstalk.json --log-file logs/smoke_crosstalk.log

echo "=== [6/6] unit tests ==="
uv run pytest tests/ -q

echo
echo "SMOKE TEST PASSED. See results/smoke/comparison.md for the tiny-run numbers."
cat results/smoke/comparison.md
