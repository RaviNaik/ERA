#!/usr/bin/env bash
# Step 4: environment record, scores table and the runs.csv row(s). Everything lands in results/.
source "$(dirname "$0")/common.sh"
# eval firewall (TSAI, 4 Nov; G12 lead: no target or response text in committed jsonl): the committed copy has the
# per-question scores and ids, never the benchmark text or the model's answers
python3 "$ROOT/scripts/strip_item_text.py" "$RUNS" "$ROOT/results/runs"
python3 "$ROOT/scripts/make_runs_row.py" --root "$ROOT" --model "$MODEL" --revision "$REVISION" --run-by "$RUN_BY" \
   --formats "$FORMATS" --llama-tag "$LLAMA_TAG" --llama-cpp-python "$LLAMA_CPP_PYTHON" --lab-sha "$LAB_SHA"
cp "$LOGS/gpus.txt" "$LOGS/env_probe.json" "$ROOT/results/" 2>/dev/null || true
say "results/ holds: scores.csv, runs_row.csv, environment.json, runs/<format>/{summary.json,*.samples.jsonl}, logs/"
