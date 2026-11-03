# G12 rotation task: 4-bit quantization of Sarvam-30B

**Task (G12 brief):** quantize one small open model to 4-bit, measure the drop on the eval-mini subset, add a row to
`runs/runs.csv`. Due Sat 14 Nov. This folder runs that task end to end on one machine and follows G12's own pipeline
(`teams/12-quantization/eval_mini` in the lab repo), with the same pins, the same scoring backend and the same rules.

**Why this model.** Sarvam-30B (`sarvamai/sarvam-30b`, Apache 2.0) is a mixture-of-experts model (128 experts,
about 32.2B parameters, 2.4B active) trained for Indian languages. Our target model is an MoE for Indic, code, math
and science, so this adds a data point the dense 1.5B to 4B rows in `runs.csv` cannot give. Checked before writing
these scripts: llama.cpp **b11669 registers `SarvamMoEForCausalLM`** (`conversion/bailingmoe.py`, GGUF architecture
`bailingmoe2`), so G12's pinned build can convert it. The weights are fp32 on the Hub (128.6 GB), so the bf16 GGUF
(about 64 GB) is a conversion from fp32, as it is for any model released in fp32.

## G12 rules this follows

| Rule | Where it is enforced |
|---|---|
| llama.cpp b11669 converts and quantizes | `01_setup.sh` clones the tag, `02_convert_quantize.sh` uses it |
| bf16 baseline and 4-bit scored by the **same runtime on the same questions** | `03_evaluate.sh`; `paired.py` compares item by item (McNemar) |
| Scoring in-process with `llamacpp_lm.py` (llama-server and llama-cpp-python's server give wrong log-probabilities) | `run_mini_gguf.py` from G12, unchanged |
| Instructor's guard `test_llamacpp_lm.py` passes for every file before it is scored | `03_evaluate.sh` |
| Router (`ffn_gate_inp`) F32 or BF16; output layer Q8_0 in every quantized format | `scripts/check_gguf.py` stops the run if broken |
| G9's `run_eval_mini.py` defines the task set (ARC-C 500, HellaSwag 400, MMLU 570, GSM8K 5-shot 200, TruthfulQA-mc2 200), seed 42 | fetched at lab commit `da3f8df`; every file is checked against `SHA256SUMS` |
| llama-cpp-python 0.3.36, lm_eval 0.4.13, datasets 5.1.0, no torch or transformers in the eval venv | `01_setup.sh` |
| Hardware may differ; it is recorded | `results/environment.json`, `machine` column |
| `score` = unweighted mean of the five primary metrics | `make_runs_row.py` (reproduces G12's own OLMoE `runs.csv` numbers: Q4_K_M 0.4258) |
| No code from the model repo is downloaded or run | `02_convert_quantize.sh` downloads data files only (no `*.py`) |

## Run it (Docker, recommended)

The whole task runs in a container built from `nvidia/cuda:12.4.1-devel-ubuntu22.04` (compiler, cmake, git, uv; nothing
to install on the host except Docker and the NVIDIA Container Toolkit). Everything that matters stays on the host:

| Host folder | In the container | Holds |
|---|---|---|
| `HOST_WORK_DIR` (default `./work`) | `/work` | the 129 GB download, bf16 and quantized GGUFs, llama.cpp build, venvs, caches (about 330 GB at the peak) |
| this folder | `/app` | `config.sh` and `results/` (scores, `runs_row.csv`, logs, per-question rows) |

The container runs as your own user, so nothing on the host is owned by root. Deleting the container loses nothing;
the next run continues from the files in `work/` and `results/`.

```bash
git clone https://github.com/RaviNaik/ERA && cd ERA/v5/capstone/g12-quantization-rotation
# edit config.sh if you want other formats (default: bf16 and Q4_K_M)
export HOST_WORK_DIR=/data/g12          # a disk with 330 GB free; default is ./work
./run_docker.sh gpu-test                # both A6000s should be listed
./run_docker.sh 2>&1 | tee run_all.log  # build the image, then steps 1 to 4 (resumable: run it again after a stop)
```

Other commands: `./run_docker.sh step 3` (one step), `./run_docker.sh shell` (a shell with the same mounts),
`GPUS='"device=0,1"' ./run_docker.sh` (pick cards), `BASE_IMAGE=nvidia/cuda:12.1.1-devel-ubuntu22.04
./run_docker.sh` (older driver: CUDA 12.4 needs driver 550 or newer). `gpu-test` failing means the NVIDIA Container
Toolkit is missing: `sudo apt install nvidia-container-toolkit && sudo nvidia-ctk runtime configure --runtime=docker
&& sudo systemctl restart docker`.

## Run it without Docker

Needs a Linux machine with NVIDIA GPUs, the CUDA toolkit (`nvcc`), `git`, `cmake` (installed for you with pip if
missing), `gcc/g++`, `curl`, Python 3 and internet.

```bash
WORK_DIR=/data/g12 ./run_all.sh 2>&1 | tee run_all.log
```

Every step is resumable: if it stops, run it again. The steps (the same four run inside the container):

| Step | Script | What it does | Rough size |
|---|---|---|---|
| 1 | `scripts/01_setup.sh` | checks, pinned code, llama.cpp b11669, two venvs, CUDA build of llama-cpp-python | 15 to 30 min build |
| 2 | `scripts/02_convert_quantize.sh` | downloads the pinned revision, converts to bf16 GGUF, quantizes each format, checks tensor types | download 129 GB, bf16 64 GB, Q4_K_M about 20 GB |
| 3 | `scripts/03_evaluate.sh` | guard test, eval_mini on every format (saved after each task), generation speed, paired comparison | hours; 1,870 questions per format |
| 4 | `scripts/04_report.sh` | `results/scores.csv`, `results/runs_row.csv`, `results/environment.json` | seconds |

Disk peak is the fp32 download plus the bf16 GGUF (about 193 GB); the download is deleted after conversion unless
`KEEP_HF=1`. After that about 64 GB plus the quantized files remain.

To try a quick end-to-end check first, run step 3 for one format with a few items by hand
(`run_mini_gguf.py ... --limit 3`). A run made with `--limit` must not be reported (G12's `combine.py` refuses it).

## What comes out (`results/`)

| File | Content |
|---|---|
| `scores.csv` | five task scores and the mean, per format |
| `runs_row.csv` | one row per format in the columns of `teams/12-quantization/runs/runs.csv` |
| `runs/<format>/summary.json`, `*.samples.jsonl` | G9 record format and one row per question (what G12's `combine.py` reads) |
| `runs/paired.md` | each format against bf16 on the same questions, with 95% CI and McNemar p |
| `gguf_check_<format>.json` | tensor types, bits per weight, router and output types |
| `speed_<format>.json`, `quant_times.json` | generation tokens per second, quantize time |
| `environment.json`, `gpus.txt`, `env_probe.json`, `logs/` | versions, hardware and every log |

## Handing it in to G12

1. Before the run, ask Sreekanth in the G12 group: is a 30B MoE acceptable as the rotation model, and should the row
   go into `runs/runs.csv` or a file under `runs/` (Shubham asked on 3 Nov for the second, with `combine.py` building
   the CSV)? Post the model pick there so nobody duplicates it.
2. Open a PR to `theschoolofai/era-v5-lab` that puts the results in a topic folder under `teams/12-quantization/`
   (G12's README forbids personal folders; the lead names the topic) and adds the row. `pr-guard` must pass.
3. Post the daily note in G12: done (paths and commits), blocked, next.

## Known risks and what to do

- **Guard test fails by a small margin.** Phi-3-mini needed a looser tolerance for check 3 (0.10 instead of 0.05, see
  lab PR #259). Do not change it silently: set `TEST_TOL` in `config.sh`, rerun step 1, and say so in the PR.
- **Converter error.** Send me the end of `results/logs/convert.log`.
- **GPU out of memory in bf16.** Unlikely on 96 GB; lower `--n-ctx` is not allowed (it changes the scores). Use fewer
  GPU layers only for bf16 and record it.
- **`nvcc` missing (without Docker).** Install the CUDA toolkit; `01_setup.sh` stops at the first check. The Docker image has it.
- **Not tested here.** This machine has a 4 GB GPU, so the 30B run could not be tested. What was tested: shell
  syntax, `check_gguf.py` on small synthetic GGUFs (passes a good file, rejects an F16 router), and
  `make_runs_row.py` on G12's real OLMoE results (same scores and `runs.csv` columns). Steps 1 to 3 run for the first
  time on your server. The Docker image builds, runs as your user and writes to the mounted folders (checked here
  with a plain Ubuntu base image and stubbed GPU tools).

## Phase 2 (optional, recommended for Sarvam): code and Indic eval

G12 also has `eval_code_indic` (G9's eval-mini code + Indic half: MMLU-Pro, GSM8K, Belebele and MILU in Hindi, Telugu,
Tamil, Bengali, HumanEval+ and MBPP+ with evalplus), run on OLMoE at n_ctx 4096 (rows dated 2026-11-01 in `runs.csv`).
It would show whether 4-bit hurts Indic more than English on an Indic model. It needs a Hugging Face token with
access to the gated `ai4bharat/MILU` dataset, evalplus installed without torch, and G9 commit `fa5d87e`. It is not
built here: finish the main task first, then reuse the same bf16 and Q4_K_M files.
