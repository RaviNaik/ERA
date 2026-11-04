#!/usr/bin/env bash
# Run this on a machine WITH Hugging Face access, when the GPU server has none. It downloads
#   1. the pinned model files (129 GB of fp32 safetensors, no *.py), and
#   2. the five eval_mini datasets (ARC-Challenge, HellaSwag, MMLU, GSM8K, TruthfulQA) into an HF cache,
# then proves the datasets load with the network switched off. Copy the two folders to the server (README).
#
#   OUT=/path/with/140GB/free ./scripts/prefetch_hf.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/config.sh"
OUT="${OUT:?set OUT to a folder with about 140 GB free}"
mkdir -p "$OUT/hf" "$OUT/home/.cache/huggingface"
command -v uv >/dev/null || python3 -m pip install --user "uv==$UV_VERSION"
export PATH="$HOME/.local/bin:$PATH"
[ -d "$OUT/.venv" ] || { uv venv -q --python 3.12 "$OUT/.venv" && uv pip install -q --python "$OUT/.venv/bin/python" huggingface_hub "lm_eval==$LM_EVAL" "datasets==$DATASETS"; }
PY="$OUT/.venv/bin/python"
export HF_HOME="$OUT/home/.cache/huggingface"

echo "=== datasets"
"$PY" - <<'PYC'
from lm_eval.tasks import TaskManager, get_task_dict
get_task_dict(["arc_challenge", "hellaswag", "mmlu", "gsm8k", "truthfulqa_mc2"], TaskManager())
print("datasets downloaded")
PYC
echo "=== datasets again with the network off"
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 "$PY" - <<'PYC'
from lm_eval.tasks import TaskManager, get_task_dict
get_task_dict(["arc_challenge", "hellaswag", "mmlu", "gsm8k", "truthfulqa_mc2"], TaskManager())
print("OFFLINE OK: all five tasks load from the cache")
PYC

echo "=== model $MODEL @ $REVISION"
"$PY" - <<PYC
from huggingface_hub import snapshot_download
snapshot_download("$MODEL", revision="$REVISION", local_dir="$OUT/hf",
                  allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"])
PYC
du -sh "$OUT/hf" "$OUT/home/.cache/huggingface"
cat <<MSG

Copy to the server (HOST_WORK_DIR there), keeping these names:
  rsync -ah --progress $OUT/hf/                      server:HOST_WORK_DIR/hf/
  rsync -ah --progress $OUT/home/.cache/huggingface/ server:HOST_WORK_DIR/home/.cache/huggingface/
then on the server:  OFFLINE=1 ./run_docker.sh
MSG
