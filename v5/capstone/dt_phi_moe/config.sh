#!/usr/bin/env bash
# Experiment control: every setting of the Deletion Tuning experiment lives here. Edit this file (or export a
# variable before calling run.sh) and then start the experiment with:   bash run.sh
# Nothing in this file runs anything; run.sh sources it.
#
# Experiment: Deletion Tuning (arXiv 2611.10866) of Phi-mini-MoE-instruct by SFT on GSM8K, with the paper's own
# hyperparameters, then GSM8K accuracy, prior-task retention and forward KL against the untouched base model.

# ---------------------------------------------------------------------------------------------------------------
# What to run. PROFILE=smoke is a quick end-to-end check of the whole chain on the real model (about 20-30 minutes);
# PROFILE=full is the experiment.
PROFILE="${PROFILE:-full}"
# Stages, in order: test (unit tests on a tiny model, CPU), base_eval, train, dt_eval, report.
# base_eval must run before dt_eval (the forward KL scores the base model's own GSM8K replies).
STAGES="${STAGES:-test base_eval train dt_eval report}"
RUN_NAME="${RUN_NAME:-dt_phi_mini_sft_gsm8k_seed0}"
FORCE="${FORCE:-0}"            # 1 = rerun stages even when their .done marker exists

# ---------------------------------------------------------------------------------------------------------------
# Hardware
GPUS="${GPUS:-0,1}"            # CUDA_VISIBLE_DEVICES; the model is split in two blocks of layers over these GPUs
N_GPUS="${N_GPUS:-2}"          # number of GPUs used by the model (len of GPUS)

# ---------------------------------------------------------------------------------------------------------------
# Paths. ASSETS_DIR holds what was transferred: phi-mini-moe-instruct/ and data/ (gsm8k, team10_sft_tests).
ASSETS_DIR="${ASSETS_DIR:-$ROOT/assets}"
MODEL_DIR="${MODEL_DIR:-$ASSETS_DIR/phi-mini-moe-instruct}"
GSM8K_DIR="${GSM8K_DIR:-$ASSETS_DIR/data/gsm8k}"
TEAM10_DIR="${TEAM10_DIR:-$ASSETS_DIR/data/team10_sft_tests}"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"

# ---------------------------------------------------------------------------------------------------------------
# Deletion Tuning, SFT regime: the paper's values (section 5.1) = the flags of the official scripts/train_sft.sh.
# The training itself is the OFFICIAL code (official/src/train_sft.py, unmodified); do not tune these.
STEPS="${STEPS:-400}"                  # --train_steps: optimizer steps
BATCH_SIZE="${BATCH_SIZE:-4}"          # --train_batch_size
GRAD_ACCUM="${GRAD_ACCUM:-1}"          # --grad_accum_steps
LR="${LR:-5e-4}"                       # --learning_rate: AdamW, constant
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
ALPHA_START="${ALPHA_START:-100}"      # --sharpness_start / --sharpness_end: linear ramp over the run
ALPHA_END="${ALPHA_END:-1000}"
LAMBDA="${LAMBDA:-1e-5}"               # --weight_reg: decays linearly to 0 (official)
ETA="${ETA:-0.00048828125}"            # --init_sparsity = 1/2048
PRIOR_STRENGTH="${PRIOR_STRENGTH:-3}"  # --prior_strength: +-3 std of the logit init
SEED="${SEED:-0}"
# Ours (not official flags): the GSM8K rows the official code trains AND runs its Wanda calibration on (it uses the
# whole training set for both). 2048 = the paper's calibration size; 400 steps x 4 = 1,600 samples are drawn from it.
TRAIN_POOL="${TRAIN_POOL:-2048}"       # 0 = all 7,473 GSM8K training rows (calibration then runs on all of them)
MAX_LEN="${MAX_LEN:-512}"              # longer GSM8K sequences are dropped

# ---------------------------------------------------------------------------------------------------------------
# Evaluation (the same for the base model and the masked model)
GSM8K_LIMIT="${GSM8K_LIMIT:-0}"        # 0 = all 1,319 test rows
GEN_BATCH="${GEN_BATCH:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-400}"
KL_LIMIT="${KL_LIMIT:-0}"              # 0 = every base reply
TEAM10_MAX_NEW_TOKENS="${TEAM10_MAX_NEW_TOKENS:-200}"
PRIOR_TASKS="${PRIOR_TASKS:-hellaswag,truthfulqa_mc2,mmlu,ifeval,winogrande,humaneval_instruct}"
PRIOR_LIMIT="${PRIOR_LIMIT:-0}"        # 0 = full benchmarks (lm-eval --limit applies per task / per MMLU subject)
PRIOR_BATCH="${PRIOR_BATCH:-8}"
EVAL_PARTS_BASE="${EVAL_PARTS_BASE:-gsm8k,gsm8k_chat,team10,prior}"
EVAL_PARTS_DT="${EVAL_PARTS_DT:-gsm8k,gsm8k_chat,kl,team10,prior}"

# ---------------------------------------------------------------------------------------------------------------
# Profile overrides (applied last)
if [[ "$PROFILE" == "smoke" ]]; then
  RUN_NAME="${RUN_NAME}_smoke"
  STEPS=5; TRAIN_POOL=64
  GSM8K_LIMIT=16; KL_LIMIT=8; TEAM10_MAX_NEW_TOKENS=32; MAX_NEW_TOKENS=200
  PRIOR_TASKS="hellaswag,winogrande,humaneval_instruct"; PRIOR_LIMIT=4
  STAGES="${SMOKE_STAGES:-base_eval train dt_eval report}"
elif [[ "$PROFILE" != "full" ]]; then
  echo "PROFILE must be full or smoke, not '$PROFILE'" >&2; return 1 2>/dev/null || exit 1
fi
OUT_DIR="${OUT_DIR:-$OUT_ROOT/$RUN_NAME}"
