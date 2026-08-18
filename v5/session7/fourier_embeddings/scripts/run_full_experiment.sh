#!/usr/bin/env bash
# The real experiment, sized for an A6000 (48GB). Mirrors the research
# note's minimal validation protocol (Sec. 10, item 1): a controlled
# comparison of dense / kronecker / fourier / fourier_narrow (+ hrr) at a
# shared model size and a shared multilingual corpus.
#
# Tune --target-mb-per-lang, --vocab-size, --n-layer/--n-head/--n-embd,
# --block-size and --max-steps upward once you've confirmed this runs; the
# defaults here are already well past "smoke test" (a few hundred MB of
# text, a ~50M-param model) but still modest enough to finish in well under
# a day on one A6000.
#
# Safe to re-run after any interruption (crash, OOM, SIGTERM/preemption,
# Ctrl-C, or you just stopping it): every step below skips work that already
# finished. Concretely -- download/tokenizer/pack-dataset skip a language,
# tokenizer, or packed dataset that's already on disk (pass --overwrite to
# force a redo if the corpus or tokenizer config actually changed); the
# experiment runner skips any arm whose metrics.json shows it completed, and
# automatically resumes (from the last --save-interval checkpoint) any arm
# that has a checkpoint but no metrics.json. Just re-run this same command.
set -euo pipefail
cd "$(dirname "$0")/.."

# See README.md's WSL note; no-op if you already `uv sync`'d in place.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/fourier_embeddings}"

TARGET_MB_PER_LANG="${TARGET_MB_PER_LANG:-30}"
VOCAB_SIZE="${VOCAB_SIZE:-16384}"
N_LAYER="${N_LAYER:-12}"
N_HEAD="${N_HEAD:-12}"
N_EMBD="${N_EMBD:-768}"
BLOCK_SIZE="${BLOCK_SIZE:-1024}"
POS_DIM="${POS_DIM:-32}"
FOURIER_DIM="${FOURIER_DIM:-32}"
HRR_DIM="${HRR_DIM:-1024}"
MAX_STEPS="${MAX_STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-full_run}"
ARMS="${ARMS:-dense kronecker fourier fourier_narrow}"

echo "=== [1/4] download multilingual corpus (~${TARGET_MB_PER_LANG}MB/lang) ==="
uv run fe-download-data --out-dir data_raw --target-mb-per-lang "$TARGET_MB_PER_LANG" \
    --langs en hi te ta bn --log-file logs/download.log

echo "=== [2/4] train tokenizer (vocab=${VOCAB_SIZE}) ==="
uv run fe-train-tokenizer --input-dir data_raw --out-dir tokenizer_out \
    --vocab-size "$VOCAB_SIZE" --log-file logs/tokenizer.log

echo "=== [3/4] pack dataset ==="
uv run fe-pack-dataset --input-dir data_raw --tokenizer-path tokenizer_out/tokenizer.json \
    --out-dir data_bin --val-fraction 0.02 --log-file logs/pack_dataset.log

echo "=== [4/4] run experiment: ${ARMS} ==="
# shellcheck disable=SC2086
uv run fe-run-experiment --experiment-name "$EXPERIMENT_NAME" \
    --data-dir data_bin --tokenizer-path tokenizer_out/tokenizer.json \
    --arms $ARMS \
    --block-size "$BLOCK_SIZE" --n-layer "$N_LAYER" --n-head "$N_HEAD" --n-embd "$N_EMBD" \
    --pos-dim "$POS_DIM" --fourier-dim "$FOURIER_DIM" --hrr-dim "$HRR_DIM" \
    --max-steps "$MAX_STEPS" --batch-size "$BATCH_SIZE" --grad-accum-steps "$GRAD_ACCUM" \
    --learning-rate 3e-4 --warmup-steps 500 \
    --eval-interval 500 --eval-iters 100 --log-interval 50 --save-interval 1000 \
    --dtype bfloat16

echo
echo "=== analysis scripts (full vocabulary) ==="
uv run fe-collisions --tokenizer-path tokenizer_out/tokenizer.json \
    --pos-dim "$POS_DIM" --fourier-dim "$FOURIER_DIM" --d-model "$N_EMBD" --device cuda \
    --out "results/${EXPERIMENT_NAME}/collisions.json" --log-file logs/collisions.log
uv run fe-order-sensitivity --pos-dim "$POS_DIM" --fourier-dim "$FOURIER_DIM" \
    --n-random-pairs 500 --out "results/${EXPERIMENT_NAME}/order_sensitivity.json" \
    --log-file logs/order_sensitivity.log
uv run fe-crosstalk --D-values 256 512 1024 2048 --n-bound-values 1 2 4 8 16 32 \
    --pos-dim "$POS_DIM" --n-trials 500 --out "results/${EXPERIMENT_NAME}/crosstalk.json" \
    --log-file logs/crosstalk.log

echo
echo "Done. See results/${EXPERIMENT_NAME}/comparison.md"
cat "results/${EXPERIMENT_NAME}/comparison.md"
