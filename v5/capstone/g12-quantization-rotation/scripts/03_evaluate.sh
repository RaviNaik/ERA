#!/usr/bin/env bash
# Step 3: G12's guard test, then G9's eval_mini profile on every format, then generation speed.
source "$(dirname "$0")/common.sh"
TASKS="arc_challenge hellaswag mmlu gsm8k truthfulqa_mc2"
mkdir -p "$RUNS"

for fmt in $FORMATS; do
  gguf="$BF16"; [ "$fmt" = "bf16" ] || gguf="$WORK_DIR/$NAME-$fmt.gguf"
  [ -f "$gguf" ] || die "$gguf is missing: run scripts/02_convert_quantize.sh first"
  say "$fmt: shifted-logprob guard (instructor's test of 28 Oct)"
  (cd "$CODE" && EVAL_PY test_llamacpp_lm.py "$gguf") 2>&1 | tee "$LOGS/test_llamacpp_lm_$fmt.log"
  say "$fmt: eval_mini (ARC-C 500, HellaSwag 400, MMLU 570, GSM8K 200, TruthfulQA-mc2 200; resumable)"
  (cd "$CODE" && EVAL_PY run_mini_gguf.py --gguf "$gguf" --format "$fmt" --out "$RUNS" \
      --g9-harness "$CODE/run_eval_mini.py" --n-gpu-layers -1 --model "$MODEL" --revision "$REVISION") 2>&1 | tee -a "$LOGS/run_$fmt.log"
  [ -f "$ROOT/results/speed_$fmt.json" ] || EVAL_PY "$ROOT/scripts/speed.py" "$gguf" --out "$ROOT/results/speed_$fmt.json"
done

if [ -f "$RUNS/bf16/summary.json" ] && [ "$(echo $FORMATS | wc -w)" -gt 1 ]; then
  say "paired comparison with bf16 (same questions; McNemar)"
  EVAL_PY "$CODE/paired.py" "$RUNS" --base bf16 --out "$RUNS/paired.md" 2>&1 | tee "$LOGS/paired.log"
fi
say "evaluation done"
