#!/usr/bin/env bash
# Step 2: download the pinned checkpoint, convert it to a bf16 GGUF, quantize, and check the tensor types.
source "$(dirname "$0")/common.sh"
export PATH="$HOME/.local/bin:$PATH"
QUANT="$LLAMA_DIR/build/bin/llama-quantize"
TIMES="$ROOT/results/quant_times.json"; [ -f "$TIMES" ] || echo '{}' > "$TIMES"

if [ ! -f "$BF16" ]; then
  if hf_complete "$WORK_DIR/hf"; then
    say "model files already in $WORK_DIR/hf (copied or downloaded earlier): no download"
  else
    [ "$OFFLINE" = "1" ] && die "OFFLINE=1 but $WORK_DIR/hf is incomplete: copy the model folder from a machine with Hugging Face access (README, 'No Hugging Face on the server')"
    say "download $MODEL @ $REVISION (about 129 GB of fp32 safetensors)"
    # only data files: no *.py, so no code from the model repo is downloaded or run
    env -u PYTHONPATH "$CONV_VENV/bin/python" - <<PY
from huggingface_hub import snapshot_download
snapshot_download("$MODEL", revision="$REVISION", local_dir="$WORK_DIR/hf",
                  allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"])
PY
  fi
  say "convert to bf16 GGUF (about 64 GB)"
  env -u PYTHONPATH "$CONV_VENV/bin/python" "$LLAMA_DIR/convert_hf_to_gguf.py" "$WORK_DIR/hf" \
      --outtype bf16 --outfile "$BF16" 2>&1 | tee "$LOGS/convert.log"
  [ "$KEEP_HF" = "1" ] || rm -rf "$WORK_DIR/hf"
fi
env -u PYTHONPATH "$CONV_VENV/bin/python" "$ROOT/scripts/check_gguf.py" "$BF16" --format bf16 --out "$ROOT/results/gguf_check_bf16.json"

for fmt in $FORMATS; do
  [ "$fmt" = "bf16" ] && continue
  gguf="$WORK_DIR/$NAME-$fmt.gguf"
  if [ ! -f "$gguf" ]; then
    say "quantize $fmt (output layer Q8_0 below 8 bits, router untouched: G12's rule)"
    extra=(); [ "$fmt" = "Q8_0" ] || extra=(--output-tensor-type q8_0)
    t0=$(date +%s)
    "$QUANT" "${extra[@]}" "$BF16" "$gguf" "$fmt" 2>&1 | tee "$LOGS/quantize_$fmt.log"
    secs=$(( $(date +%s) - t0 ))
    python3 - "$TIMES" "$fmt" "$secs" <<'PY'
import json, sys
p, fmt, s = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = json.load(open(p)); d[fmt] = s; json.dump(d, open(p, "w"), indent=1)
PY
  fi
  env -u PYTHONPATH "$CONV_VENV/bin/python" "$ROOT/scripts/check_gguf.py" "$gguf" --format "$fmt" --out "$ROOT/results/gguf_check_$fmt.json"
done
say "convert and quantize done"
