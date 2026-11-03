#!/usr/bin/env bash
# Step 1: tools and pinned code. Safe to rerun; finished parts are skipped.
source "$(dirname "$0")/common.sh"

say "checks"
for t in git cmake curl gcc g++ nvidia-smi nvcc; do command -v "$t" >/dev/null || die "$t not found (nvcc = CUDA toolkit, needed to build llama-cpp-python with GPU support)"; done
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader | tee "$LOGS/gpus.txt"
FREE_GB=$(df -BG --output=avail "$WORK_DIR" | tail -1 | tr -dc '0-9')
echo "free disk at $WORK_DIR: ${FREE_GB} GB"
[ "$FREE_GB" -ge 200 ] || die "need at least 200 GB free (330 GB is comfortable); set WORK_DIR to a bigger disk"
command -v uv >/dev/null || pip install --user "uv==$UV_VERSION"
export PATH="$HOME/.local/bin:$PATH"

say "G12/G9 evaluation code at lab commit $LAB_SHA"
mkdir -p "$CODE"
base="https://raw.githubusercontent.com/$LAB_REPO/$LAB_SHA/teams"
for f in llamacpp_lm.py run_mini_gguf.py paired.py combine.py test_llamacpp_lm.py; do
  curl -fsSL "$base/12-quantization/eval_mini/$f" -o "$CODE/$f"
done
curl -fsSL "$base/09-evaluation/harness/run_eval_mini.py" -o "$CODE/run_eval_mini.py"
(cd "$CODE" && sha256sum -c "$ROOT/SHA256SUMS")   # stops here if any file differs from what we reviewed
if [ -n "$TEST_TOL" ]; then
  sed -i "s/^TOL = 0.05/TOL = $TEST_TOL/" "$CODE/test_llamacpp_lm.py"
  echo "WARNING: test tolerance changed to $TEST_TOL (G12 default 0.05); say so in the PR" | tee -a "$LOGS/notes.txt"
fi

say "llama.cpp $LLAMA_TAG (converter + llama-quantize, CPU build as G12 does)"
[ -d "$LLAMA_DIR" ] || git clone -q --depth 1 --branch "$LLAMA_TAG" https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
grep -rq "SarvamMoEForCausalLM" "$LLAMA_DIR/conversion" || die "this llama.cpp build cannot convert $MODEL (architecture not registered)"
if [ ! -x "$LLAMA_DIR/build/bin/llama-quantize" ]; then
  cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DLLAMA_CURL=OFF -DGGML_CUDA=OFF > "$LOGS/build_llama_quantize.log" 2>&1
  cmake --build "$LLAMA_DIR/build" -j --target llama-quantize >> "$LOGS/build_llama_quantize.log" 2>&1
fi
test -x "$LLAMA_DIR/build/bin/llama-quantize"

say "converter venv (its own pinned requirements)"
if [ ! -f "$CONV_VENV/.ready" ]; then
  rm -rf "$CONV_VENV"
  uv venv -q --python python3 "$CONV_VENV"
  uv pip install -q --python "$CONV_VENV/bin/python" --index-strategy unsafe-best-match \
     -r "$LLAMA_DIR/requirements/requirements-convert_hf_to_gguf.txt" "$LLAMA_DIR/gguf-py" huggingface_hub > "$LOGS/convert_venv.log" 2>&1
  touch "$CONV_VENV/.ready"
fi

say "evaluation venv: llama-cpp-python $LLAMA_CPP_PYTHON with CUDA, lm_eval $LM_EVAL, datasets $DATASETS (no torch, no transformers)"
if [ ! -f "$EVAL_VENV/.ready" ]; then
  rm -rf "$EVAL_VENV"
  uv venv -q --python "$EVAL_PYTHON" "$EVAL_VENV"
  ARCHS=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d '.' | sort -u | paste -sd';')
  echo "CUDA architectures: $ARCHS (build takes 15 to 30 minutes)"
  CMAKE_ARGS="-DGGML_CUDA=on -DGGML_CUDA_NO_VMM=ON -DCMAKE_CUDA_ARCHITECTURES=$ARCHS" \
    uv pip install --python "$EVAL_VENV/bin/python" --no-cache "llama-cpp-python==$LLAMA_CPP_PYTHON" \
       "lm_eval==$LM_EVAL" "datasets==$DATASETS" > "$LOGS/eval_venv.log" 2>&1 || { tail -40 "$LOGS/eval_venv.log"; die "eval venv build failed"; }
  touch "$EVAL_VENV/.ready"
fi
EVAL_PY - <<'PY' | tee "$LOGS/env_probe.json"
import importlib.util, json, llama_cpp, lm_eval, datasets, platform
d = {"python": platform.python_version(), "llama_cpp_python": llama_cpp.__version__, "lm_eval": lm_eval.__version__,
     "datasets": datasets.__version__, "gpu_offload": bool(llama_cpp.llama_supports_gpu_offload()),
     "torch_visible": importlib.util.find_spec("torch") is not None,
     "transformers_visible": importlib.util.find_spec("transformers") is not None}
print(json.dumps(d))
assert d["gpu_offload"], "llama-cpp-python was built without GPU offload"
assert not d["torch_visible"] and not d["transformers_visible"], "torch/transformers must not be in this venv"
PY
say "setup done"
