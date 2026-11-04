# Settings for the G12 rotation task. Edit this file, nothing else. Every value below is a pin from G12's own
# pipeline (teams/12-quantization/eval_mini in the lab repo) unless the comment says it is ours.

# --- our choices -------------------------------------------------------------------------------------------
RUN_BY="RaviNaik"
MODEL="sarvamai/sarvam-30b"                              # Apache 2.0; MoE, 128 experts, ~32.2B parameters
REVISION="071ae95e933605ca1104a6b4524a36a98488efa4"      # HF commit of 23 Mar 2026, pinned on 3 Nov 2026
# bf16 is the baseline, Q4_K_M is the task (4-bit). Add Q8_0 / Q5_K_M / Q3_K_M for a fuller curve (more time).
FORMATS="bf16 Q4_K_M"
WORK_DIR="${WORK_DIR:-$HOME/era-v5-g12-work}"            # needs ~330 GB free at the peak (see README)
KEEP_HF="${KEEP_HF:-0}"                                  # 1 keeps the 128 GB fp32 download after conversion
OFFLINE="${OFFLINE:-0}"                                  # 1 = no Hugging Face access: use files copied from another machine (see README)
TEST_TOL=""                                              # empty = G12's 0.05; see README before changing it

# --- G12 / G9 pins (do not change; combine.py refuses runs made with other versions) -------------------------
LLAMA_TAG="b11669"                                       # llama.cpp for conversion and quantization
LLAMA_CPP_PYTHON="0.3.36"                                # in-process log-probabilities checked with this version
LM_EVAL="0.4.13"
DATASETS="5.1.0"
UV_VERSION="0.13.6"
EVAL_PYTHON="3.13"
LAB_REPO="theschoolofai/era-v5-lab"
LAB_SHA="da3f8df8c98ac3c373c83c0bb40a014f4751c0b4"       # lab main on 3 Nov 2026; the eval files below come from it
ALLOWED_FORMATS="bf16 Q8_0 Q6_K Q5_K_M Q4_K_M Q3_K_M Q2_K"
