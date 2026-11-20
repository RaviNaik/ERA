#!/usr/bin/env bash
# Create .venv with uv: official SFT environment (torch 2.7.1) + requirements.txt. Needs network once, for the packages.
# Models, datasets and metrics come from the bundle.
#   bash setup_env.sh                                                        # GPU server, CUDA 12.6 wheels
#   TORCH_INDEX=https://download.pytorch.org/whl/cu124 bash setup_env.sh    # other CUDA version
#   TORCH_INDEX=https://download.pytorch.org/whl/cpu bash setup_env.sh      # CPU only (tests)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
command -v uv >/dev/null 2>&1 || { echo "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; exit 1; }
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
TORCH_VERSION="${TORCH_VERSION:-2.7.1}"
PY="${PYTHON_VERSION:-3.11}"
uv venv "$ROOT/.venv" --python "$PY"
uv pip install --python "$ROOT/.venv/bin/python" "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"
uv pip install --python "$ROOT/.venv/bin/python" -r "$ROOT/requirements.txt"
"$ROOT/.venv/bin/python" - <<'PY'
import torch, transformers, accelerate, datasets, lm_eval
print("torch", torch.__version__, "| transformers", transformers.__version__, "| accelerate", accelerate.__version__,
      "| datasets", datasets.__version__, "| cuda available:", torch.cuda.is_available(), "| GPUs:", torch.cuda.device_count())
PY
