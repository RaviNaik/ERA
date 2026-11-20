#!/usr/bin/env bash
# Create .venv with uv: official SFT environment (torch 2.7.1) + requirements.txt. Needs network once, for the packages.
# Models, datasets and metrics come from the bundle.
#   bash setup_env.sh                                                        # GPU server, CUDA 12.6 wheels (pytorch.org index)
#   TORCH_INDEX=pypi bash setup_env.sh      # torch 2.7.1 from plain PyPI (also CUDA 12.6); use it when download.pytorch.org
#                                           # or pypi.nvidia.com time out. Re-running resumes: uv caches finished wheels.
#   TORCH_INDEX=https://download.pytorch.org/whl/cu124 bash setup_env.sh    # other CUDA version
#   TORCH_INDEX=https://download.pytorch.org/whl/cpu bash setup_env.sh      # CPU only (tests)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
command -v uv >/dev/null 2>&1 || { echo "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; exit 1; }
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"   # the nvidia wheels are large (hundreds of MB each)
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-10}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-1}"   # one big wheel at a time: fewer dropped connections
ATTEMPTS="${ATTEMPTS:-10}"
# uv keeps every wheel it finished, so each retry only fetches what is still missing
retry() { local n=1; until "$@"; do [[ $n -ge $ATTEMPTS ]] && { echo "failed after $n attempts: $*" >&2; return 1; }; echo "download failed, retry $n of $ATTEMPTS ..." >&2; n=$((n+1)); sleep 5; done; }
TORCH_VERSION="${TORCH_VERSION:-2.7.1}"
PY="${PYTHON_VERSION:-3.11}"
[[ -x "$ROOT/.venv/bin/python" ]] || uv venv "$ROOT/.venv" --python "$PY"
if [[ "$TORCH_INDEX" == "pypi" ]]; then
  retry uv pip install --python "$ROOT/.venv/bin/python" "torch==$TORCH_VERSION"
else
  retry uv pip install --python "$ROOT/.venv/bin/python" "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"
fi
retry uv pip install --python "$ROOT/.venv/bin/python" -r "$ROOT/requirements.txt"
"$ROOT/.venv/bin/python" - <<'PY'
import torch, transformers, accelerate, datasets, lm_eval
print("torch", torch.__version__, "| transformers", transformers.__version__, "| accelerate", accelerate.__version__,
      "| datasets", datasets.__version__, "| cuda available:", torch.cuda.is_available(), "| GPUs:", torch.cuda.device_count())
PY
