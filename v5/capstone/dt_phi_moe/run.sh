#!/usr/bin/env bash
# Start the experiment. All settings are in config.sh.
#   bash run.sh                  run the STAGES of config.sh (default: test base_eval train dt_eval report)
#   bash run.sh train dt_eval    run only these stages (in the order given)
#   PROFILE=smoke bash run.sh    quick end-to-end check of the chain (5 training steps, small evaluations)
#   PROFILE=extended RUN_NAME=<run> bash run.sh   larger evaluation of a trained run (eval_*_full folders)
#   bash run.sh train_lora lora_eval train_ft ft_eval paired report   the LoRA / full FT baselines and the paired test
#   bash run.sh --dry-run        print what would run, change nothing
# A finished stage leaves <run>/.done_<stage>; it is skipped next time unless FORCE=1.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY=0; ARGS=()
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) ARGS+=("$a") ;;
  esac
done
# shellcheck source=config.sh
source "$ROOT/config.sh"
if [[ ${#ARGS[@]} -gt 0 ]]; then STAGES="${ARGS[*]}"; fi
for s in $STAGES; do
  case "$s" in test|base_eval|train|dt_eval|train_lora|lora_eval|train_ft|ft_eval|paired|report) ;; *) echo "unknown stage '$s'" >&2; exit 2 ;; esac
done

die() { echo "ERROR: $*" >&2; exit 1; }

# ---- offline environment: models, datasets, metrics and nltk data come from the transferred folders ----------------
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$ROOT/cache/hf_home"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_ALLOW_CODE_EVAL=1 TOKENIZERS_PARALLELISM=false
# Asynchronous mode hit a timing-dependent device-side assert on two GPUs (all synchronous runs passed), so sync is the default.
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
# expandable_segments is off on purpose: with two GPUs it is a known source of corrupted cross-device copies (random out-of-range indices).
# Set PYTORCH_CUDA_ALLOC_CONF yourself if you want it.
# nltk refuses data folders that others can write, so use a private copy
NLTK_PRIV="${NLTK_PRIV:-$HOME/.cache/dt_phi_moe_nltk}"
if [[ -d "$ROOT/cache/nltk_data" && $DRY == 0 ]]; then
  mkdir -p "$NLTK_PRIV" && cp -r "$ROOT/cache/nltk_data/." "$NLTK_PRIV/" && chmod -R go-w "$NLTK_PRIV"
fi
export NLTK_DATA="$NLTK_PRIV"

# ---- checks ---------------------------------------------------------------------------------------------------------
need_gpu=0; for s in $STAGES; do [[ "$s" == base_eval || "$s" == train || "$s" == dt_eval || "$s" == train_lora || "$s" == lora_eval || "$s" == train_ft || "$s" == ft_eval ]] && need_gpu=1; done
[[ -x "$PYTHON" ]] || die "python not found at $PYTHON (run: bash setup_env.sh)"
[[ -f "$MODEL_DIR/config.json" ]] || die "model not found in $MODEL_DIR"
if [[ -f "$MODEL_DIR/model.safetensors.index.json" ]]; then
  "$PYTHON" - "$MODEL_DIR" <<'PY' || die "model shards missing or truncated in $MODEL_DIR"
import json, os, sys
d = sys.argv[1]
for f in set(json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"].values()):
    if not os.path.exists(os.path.join(d, f)): sys.exit(f"missing {f}")
PY
else
  [[ -f "$MODEL_DIR/model.safetensors" ]] || die "no safetensors weights in $MODEL_DIR"
fi
"$PYTHON" -c "from dtphi import official; print('official files verified:', official.verify_vendored())" || die "the vendored official code differs from upstream"
[[ -d "$GSM8K_DIR/main" ]] || die "GSM8K not found in $GSM8K_DIR/main"
[[ -f "$TEAM10_DIR/indic_v1.jsonl" && -f "$TEAM10_DIR/safety_v1.jsonl" ]] || die "team10 sets missing in $TEAM10_DIR"
[[ -d "$HF_HOME/datasets" ]] || die "dataset cache $HF_HOME/datasets missing (needed for the prior tasks)"
if [[ "$N_GPUS" == 0 ]]; then export CUDA_VISIBLE_DEVICES=""; fi   # N_GPUS=0: everything on the CPU (tests, tiny models)
if [[ $need_gpu == 1 && $DRY == 0 && "$N_GPUS" != 0 ]]; then
  export CUDA_VISIBLE_DEVICES="$GPUS"
  "$PYTHON" - <<PY || die "CUDA check failed"
import torch, sys
n = torch.cuda.device_count()
if n < $N_GPUS: sys.exit(f"need $N_GPUS GPUs, CUDA sees {n} (GPUS=$GPUS)")
for i in range(n):
    p = torch.cuda.get_device_properties(i); print(f"GPU {i}: {p.name}, {p.total_memory / 2**30:.0f} GiB")
PY
fi
if [[ $DRY == 0 ]]; then
  mkdir -p "$OUT_DIR/logs"
  avail_gb=$(df -Pk "$OUT_DIR" | awk 'NR==2 {print int($4/1048576)}')
  need_gb=5; [[ " $STAGES " == *" train_ft "* && "$FT_SCOPE" == all ]] && need_gb=20
  [[ $avail_gb -ge $need_gb ]] || die "less than $need_gb GiB free in $OUT_DIR"
fi

cmd_for() {
  case "$1" in
    test)      echo "DT_MODEL_DIR=$MODEL_DIR DT_DATA_DIR=$GSM8K_DIR DT_TEAM10_DIR=$TEAM10_DIR $PYTHON -m pytest -q -p no:cacheprovider $ROOT/tests" ;;
    base_eval) echo "$PYTHON -m dtphi.evaluate --tag base --model_dir $MODEL_DIR --data_dir $GSM8K_DIR --team10_dir $TEAM10_DIR --out_dir $OUT_DIR --n_gpus $N_GPUS --parts $EVAL_PARTS_BASE --gsm8k_limit $GSM8K_LIMIT --gen_batch $GEN_BATCH --max_new_tokens $MAX_NEW_TOKENS --team10_max_new_tokens $TEAM10_MAX_NEW_TOKENS --prior_tasks $PRIOR_TASKS --prior_limit $PRIOR_LIMIT --prior_batch $PRIOR_BATCH --seed $SEED --eval_suffix '$EVAL_SUFFIX'" ;;
    train)     echo "$PYTHON -m dtphi.train --model_dir $MODEL_DIR --data_dir $GSM8K_DIR --output_dir $OUT_DIR/train --n_gpus $N_GPUS --train_steps $STEPS --train_batch_size $BATCH_SIZE --grad_accum_steps $GRAD_ACCUM --learning_rate $LR --weight_decay $WEIGHT_DECAY --max_grad_norm $MAX_GRAD_NORM --init_sparsity $ETA --prior_strength $PRIOR_STRENGTH --sharpness_start $ALPHA_START --sharpness_end $ALPHA_END --weight_reg $LAMBDA --seed $SEED --train_pool $TRAIN_POOL --max_len $MAX_LEN" ;;
    dt_eval)   echo "$PYTHON -m dtphi.evaluate --tag dt --model_dir $MODEL_DIR --data_dir $GSM8K_DIR --team10_dir $TEAM10_DIR --out_dir $OUT_DIR --mask_dir $OUT_DIR/train/masks/gsm8k --n_gpus $N_GPUS --parts $EVAL_PARTS_DT --gsm8k_limit $GSM8K_LIMIT --kl_limit $KL_LIMIT --gen_batch $GEN_BATCH --max_new_tokens $MAX_NEW_TOKENS --team10_max_new_tokens $TEAM10_MAX_NEW_TOKENS --prior_tasks $PRIOR_TASKS --prior_limit $PRIOR_LIMIT --prior_batch $PRIOR_BATCH --seed $SEED --eval_suffix '$EVAL_SUFFIX'" ;;
    train_lora) echo "$PYTHON -m dtphi.train_baseline --method lora --model_dir $MODEL_DIR --data_dir $GSM8K_DIR --output_dir $OUT_DIR/train_lora --n_gpus $N_GPUS --train_steps $STEPS --train_batch_size $BATCH_SIZE --grad_accum_steps $GRAD_ACCUM --learning_rate $LORA_LR --weight_decay $WEIGHT_DECAY --max_grad_norm $MAX_GRAD_NORM --seed $SEED --train_pool $TRAIN_POOL --max_len $MAX_LEN --lora_r $LORA_R --lora_alpha $LORA_ALPHA --lora_targets $LORA_TARGETS" ;;
    lora_eval) echo "$PYTHON -m dtphi.evaluate --tag lora --model_dir $MODEL_DIR --data_dir $GSM8K_DIR --team10_dir $TEAM10_DIR --out_dir $OUT_DIR --ckpt_dir $OUT_DIR/train_lora --n_gpus $N_GPUS --parts $EVAL_PARTS_LORA --gsm8k_limit $GSM8K_LIMIT --kl_limit $KL_LIMIT --gen_batch $GEN_BATCH --max_new_tokens $MAX_NEW_TOKENS --team10_max_new_tokens $TEAM10_MAX_NEW_TOKENS --prior_tasks $PRIOR_TASKS --prior_limit $PRIOR_LIMIT --prior_batch $PRIOR_BATCH --seed $SEED --eval_suffix '$EVAL_SUFFIX'" ;;
    train_ft)  echo "$PYTHON -m dtphi.train_baseline --method ft --model_dir $MODEL_DIR --data_dir $GSM8K_DIR --output_dir $OUT_DIR/train_ft --n_gpus $N_GPUS --train_steps $STEPS --train_batch_size $BATCH_SIZE --grad_accum_steps $GRAD_ACCUM --learning_rate $FT_LR --weight_decay $WEIGHT_DECAY --max_grad_norm $MAX_GRAD_NORM --seed $SEED --train_pool $TRAIN_POOL --max_len $MAX_LEN --ft_scope $FT_SCOPE" ;;
    ft_eval)   echo "$PYTHON -m dtphi.evaluate --tag ft --model_dir $MODEL_DIR --data_dir $GSM8K_DIR --team10_dir $TEAM10_DIR --out_dir $OUT_DIR --ckpt_dir $OUT_DIR/train_ft --n_gpus $N_GPUS --parts $EVAL_PARTS_FT --gsm8k_limit $GSM8K_LIMIT --kl_limit $KL_LIMIT --gen_batch $GEN_BATCH --max_new_tokens $MAX_NEW_TOKENS --team10_max_new_tokens $TEAM10_MAX_NEW_TOKENS --prior_tasks $PRIOR_TASKS --prior_limit $PRIOR_LIMIT --prior_batch $PRIOR_BATCH --seed $SEED --eval_suffix '$EVAL_SUFFIX'" ;;
    paired)    echo "$PYTHON -m dtphi.paired --run_dir $OUT_DIR --eval_suffix '$EVAL_SUFFIX'" ;;
    report)    echo "$PYTHON -m dtphi.report --run_dir $OUT_DIR --eval_suffix '$EVAL_SUFFIX'" ;;
  esac
}

echo "run: $RUN_NAME  profile: $PROFILE  stages: $STAGES"
echo "CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING"
echo "out: $OUT_DIR"
if [[ $DRY == 0 ]]; then { echo "# config used $(date -Iseconds)"; (set -o posix; set) | grep -E '^(LORA_LR|LORA_R|LORA_ALPHA|LORA_TARGETS|FT_LR|FT_SCOPE|EVAL_SUFFIX|PROFILE|RUN_NAME|STAGES|GPUS|N_GPUS|STEPS|BATCH_SIZE|GRAD_ACCUM|LR|WEIGHT_DECAY|MAX_GRAD_NORM|ALPHA_START|ALPHA_END|LAMBDA|ETA|PRIOR_STRENGTH|SEED|TRAIN_POOL|MAX_LEN|GSM8K_LIMIT|GEN_BATCH|MAX_NEW_TOKENS|KL_LIMIT|PRIOR_TASKS|PRIOR_LIMIT|PRIOR_BATCH)=' ; } > "$OUT_DIR/config_used.txt" 2>/dev/null || true; fi

for s in $STAGES; do
  sfx=""; case "$s" in base_eval|dt_eval|lora_eval|ft_eval|paired|report) sfx="$EVAL_SUFFIX" ;; esac
  marker="$OUT_DIR/.done_$s$sfx"
  if [[ -f "$marker" && "$FORCE" != 1 ]]; then echo "[$s] already done (FORCE=1 to rerun)"; continue; fi
  if [[ $DRY == 0 ]]; then
    for v in dt lora ft; do
      parts_var="EVAL_PARTS_${v^^}"
      if [[ "$s" == ${v}_eval && ! -f "$OUT_DIR/eval_base$EVAL_SUFFIX/gsm8k_generations.jsonl" && "${!parts_var}" == *kl* ]]; then
        die "${v}_eval needs the base evaluation first (eval_base$EVAL_SUFFIX/gsm8k_generations.jsonl missing): run base_eval"
      fi
    done
    [[ "$s" == lora_eval && ! -f "$OUT_DIR/train_lora/lora.safetensors" ]] && die "lora_eval needs the adapters: run train_lora"
    [[ "$s" == ft_eval && ! -f "$OUT_DIR/train_ft/ft_index.json" ]] && die "ft_eval needs the trained weights: run train_ft"
  fi
  if [[ $DRY == 0 && "$s" == dt_eval && ! -f "$OUT_DIR/train/masks/gsm8k/model.safetensors" ]]; then die "dt_eval needs the trained mask: run train"; fi
  c="$(cmd_for "$s")"
  echo "[$s] $c"
  if [[ $DRY == 1 ]]; then continue; fi
  echo "[$s] $c" >> "$OUT_DIR/commands.txt"
  t0=$(date +%s)
  if [[ "$s" == test ]]; then ( export CUDA_VISIBLE_DEVICES=""; eval "$c" ) 2>&1 | tee "$OUT_DIR/logs/$s.log"
  else eval "$c" 2>&1 | tee "$OUT_DIR/logs/$s.log"; fi
  rc=${PIPESTATUS[0]}
  [[ $rc == 0 ]] || die "stage $s failed with exit code $rc (log: $OUT_DIR/logs/$s.log)"
  echo "[$s] finished in $(( $(date +%s) - t0 )) s" | tee -a "$OUT_DIR/logs/$s.log"
  touch "$marker"
done
echo "all requested stages finished. results: $OUT_DIR/results.md (after the report stage)"
