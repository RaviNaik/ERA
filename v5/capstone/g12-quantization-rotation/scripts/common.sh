# sourced by the other scripts
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/config.sh"
mkdir -p "$WORK_DIR" "$ROOT/results/logs"
CODE="$WORK_DIR/eval_mini_code"
LLAMA_DIR="$WORK_DIR/llama.cpp"
EVAL_VENV="$WORK_DIR/eval-venv"
CONV_VENV="$WORK_DIR/convert-venv"
NAME="${MODEL##*/}"
BF16="$WORK_DIR/$NAME-bf16.gguf"
RUNS="$ROOT/results/runs"
LOGS="$ROOT/results/logs"
say() { printf '\n=== %s\n' "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }
for f in $FORMATS; do
  case " $ALLOWED_FORMATS " in *" $f "*) ;; *) die "format $f is not one of: $ALLOWED_FORMATS";; esac
done
EVAL_PY() { env -u PYTHONPATH "$EVAL_VENV/bin/python" "$@"; }
if [ "$OFFLINE" = "1" ]; then export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1; fi
hf_complete() {   # true when every shard named in the index is present
  python3 - "$1" <<'PY'
import json, sys
from pathlib import Path
d = Path(sys.argv[1]); idx = d / "model.safetensors.index.json"
if not idx.exists(): sys.exit(1)
files = set(json.load(open(idx))["weight_map"].values())
sys.exit(0 if files and all((d / f).exists() for f in files) and (d / "config.json").exists() and (d / "tokenizer.json").exists() else 1)
PY
}
