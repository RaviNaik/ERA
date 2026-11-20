# Deletion Tuning of Phi-mini-MoE-instruct on GSM8K (SFT)

Deletion Tuning (DT, arXiv 2611.10866) learns a binary mask over the pretrained weights instead of changing them.
Here: the **expert gate projections (`w1`) of Phi-mini-MoE-instruct** (7.65B total, 2.4B active, 32 layers x 16
experts = 512 matrices of 960 x 4096 = 2.01B logits), trained by SFT on GSM8K with **the official DT code and the
paper's SFT settings**. Afterwards GSM8K accuracy, forgetting (six prior-task benchmarks, forward KL) and the Team 10
Indic/safety sets are measured on the base model and on the masked model.

    bash setup_env.sh                       # once: venv with the official SFT environment (uv venv + uv pip install; needs network once)
    PROFILE=smoke bash run.sh               # 5 training steps and small evaluations: checks the whole chain on the server
    bash run.sh                             # the experiment (all settings in config.sh)
    bash run.sh train dt_eval report        # only some stages
    bash run.sh --dry-run                   # print the commands

`config.sh` controls the experiment (hyperparameters, GPUs, paths, which stages, evaluation sizes); `run.sh` starts it.
Stages: `test` (unit tests, tiny random model, CPU, ~1 min), `base_eval`, `train`, `dt_eval`, `report`. Each stage logs to
`runs/<RUN_NAME>/logs/`, writes its exact command to `commands.txt` and leaves a `.done_<stage>` marker.
Results: `runs/<RUN_NAME>/results.md` and `results.json`.

## What is official code and what is ours

`official/` holds the files of `agentic-learning-ai-lab/deleting-weights` at commit `94f0340` (MIT licence),
**byte for byte**; `official/VENDORED_FILES.txt` lists the git blob hash of each, and `run.sh`, `dtphi.train` and
`dtphi.evaluate` re-check every file before running (`dtphi.official.verify_vendored`). The verl and sdft folders and the
figure are left out. Nothing in `official/` was edited.

Official code that runs the experiment:

| Official file | Used for |
|---|---|
| `src/train_sft.py` | `train_all_tasks` / `train_task`: Wanda calibration, prior, training loop (AdamW, constant lr, clip 1.0, regulariser with linear decay, sharpness schedule), mask saving |
| `src/learnable_sparsity/{sparse_linear,differentiable_mask}.py` | `SparseLinear`, `DifferentiableMask`, `convert_to_sparse_model`, `save_mask` |
| `src/wanda_calibration.py` | Wanda scores and the +-3 std prior at init_sparsity 1/2048 |
| `src/sft_data.py` | `encode` (loss on the answer only) and the left-padded `collate_train` |
| `src/apply_mask.py` | `load_mask`, `apply_mask` (applies a saved mask to the base model) |
| `src/retention_forward_kl.py` | `response_logprobs` (forward KL) |
| `src/retention_prior_tasks.py` | the six benchmarks and their metrics (`TASK_TO_METRIC`) |

Our code, in `src/dtphi/`:

| File | What it does |
|---|---|
| `phimoe/` | the Phi-mini-MoE model code, three small patches (items 7 and 8) |
| `modeling.py` | loads the model in bf16 over the GPUs (two blocks of layers) |
| `data.py` | GSM8K in the official `{question}\n{answer}` SFT format; answer extraction |
| `train.py` | builds the inputs and calls the official `train_all_tasks`, plus the wrappers below |
| `evaluate.py`, `eval_gsm8k.py`, `eval_prior.py`, `eval_team10.py`, `report.py` | evaluation of the base and the masked model, results table |
| `team10/` | the Team 10 scorers (`scorers.py` unchanged) and a plain-text stand-in for `chat-format/render.py` |

## Every difference from running the official scripts

1. **Model and target.** The paper does not use Phi-mini-MoE. The official target `gate_proj` becomes `.w1`, the gate of
   the expert MLP `act(w1 x) * w3 x -> w2`. Set through the official module-level list `train_sft.TARGET_MODULES`.
   Routers, attention, `w2`, `w3`, embeddings and the LM head are untouched. Experts are plain `nn.Linear` modules in
   this model's code, so the official converter works as it is.
2. **Task.** The paper trains DT with SFT on MMLU-Pro and BBEH, and with GRPO on GSM8K. SFT on GSM8K is our combination.
   Samples are `{question}\n{answer}` with the loss on the answer (official `encode`), the `<<calculator>>` notes removed,
   no end-of-text token appended (the official encoder appends none), no chat template, no BOS (this model uses none).
3. **Training pool.** The official code calibrates Wanda on the whole training set of the task. To get the paper's 2,048
   calibration samples the pool is 2,048 rows of the seeded-shuffled GSM8K train split (`TRAIN_POOL`; 0 = all 7,473).
   400 steps x batch 4 = 1,600 samples are drawn from it.
4. **Two GPUs.** The official code is single-GPU. Layers are split in two blocks over the two GPUs, and three runtime
   wrappers in `train.py` (official files untouched) make that work: (a) after `convert_to_sparse_model` each logit
   tensor is moved to its weight's device (the official function puts all of them on `model.device`, and the official
   Wanda prior then adds a weight-device tensor to the logits); (b) `sum_sparse_weight_norms` is replaced by the same sum
   accumulated on the loss device; (c) `init_masks_from_wanda` is wrapped to log the deleted share at step 0.
   `accelerator.prepare(model)` is not called on the split model.
5. **Prior tasks.** The official script saves the masked model to a folder and lets lm-eval load it through its `hf`
   model class; that cannot load this model's custom code. Here lm-eval's `HFLM` wraps the model object that already has
   the official mask applied. Tasks, metrics, zero-shot, chat template: as official.
6. **Forward KL.** `response_logprobs` is official. The base model's greedy replies come from Hugging Face `generate`
   (official: vLLM) on the training-format prompts, cut after the first complete `#### n`; official uses chat-template
   prompts. The per-prompt mean and average are the official formula, repeated in `eval_gsm8k.forward_kl`.
7. **Deterministic routing while training.** The official script calls `model.train()`. In Phi-mini-MoE's code that turns on
   Gumbel sampling of the experts, a random router-gradient estimator and 1% input jitter, none of which exist at
   evaluation. The mask would be trained under different routing than it is used with, so `train.py` sets the switch
   `DETERMINISTIC_MOE` (our patch in `phimoe/modeling_phimoe.py`) and the blocks route as in eval mode.
8. **Model code.** `src/dtphi/phimoe/` is Microsoft's `modeling_slimmoe.py` / `configuration_slimmoe.py` with: the unused
   `einops` and `flash_attn` imports removed (the original cannot load without a compiled flash_attn), and
   `GenerationMixin` added as an explicit base class. Attention is SDPA. Loading uses these classes directly, no
   `trust_remote_code`.
9. **GSM8K accuracy** is our own code: greedy decoding up to 400 tokens, stop after the first `#### n`, the number after
   the first `####` counts (else a `\boxed{}` number, else the last number). Two prompts: `gsm8k` (training format) and
   `gsm8k_chat` (chat template and the instruction "end your answer with '#### <number>'").
10. **Team 10 sets.** Only Indic and safety rows are scored: tool-call replies are chat-format tool calls with Team 10
   special tokens that this tokenizer lacks. `team10/chat-format/render.py` is a plain-text stand-in (the real one needs
   the Team 3 tokenizer files); `scorers.py` is the team's file unchanged.
11. **Versions.** The official SFT pins (`transformers 4.51.3`, `accelerate 1.13.0`, `datasets 4.8.2`, `numpy 1.26.4`,
    `zstandard 0.25.0`, torch 2.7.1 with CUDA 12.6) plus `lm-eval[ifeval]==0.4.11` from the official GRPO requirements.
    The official environment is Python 3.10. Environment and packages are managed with uv (`uv venv`, `uv pip install`). We tested with Python 3.11 and torch 2.7.1 (CPU wheel).

Idle experts: an expert that no calibration token reaches gets no Wanda prior in the official code (its logits stay
N(0, 0.01), about half deleted at step 0). `train_summary.json` and the log report `layers_without_scores`.

## Paper values (config.sh)

SFT: lr 5e-4, AdamW constant, no weight decay, 400 steps, batch 4, sharpness 100 -> 1000, weight regulariser 1e-5
decaying linearly to 0 (the loss subtracts it: `loss - lambda * sum ||W * mask||^2`), init sparsity 1/2048 (about 0.18%
of the logits start negative), prior +-3 std, clip 1.0. These are the flags of `official/scripts/train_sft.sh`.

## What is bundled, what is needed

`assets/phi-mini-moe-instruct` (15.3 GB), `assets/data/gsm8k`, `assets/data/team10_sft_tests`, `assets/data/prior_tasks`
(the raw prior-task datasets, kept for reference; the harness reads `cache/`), `cache/hf_home` (datasets and the
`code_eval` metric for the offline lm-eval run), `cache/nltk_data` (punkt for IFEval). `run.sh` sets the offline
variables; no model or dataset is downloaded at run time. `setup_env.sh` (uv) needs network once.

## Expected resources (estimates, not measured on the target machine)

- GPU: weights 15.3 GB over two cards, mask logits + gradients + Adam states 16 bytes x 2.01B = 32 GB over the two
  cards, plus activations (gradient checkpointing is on, as in official `main()`): roughly 25 to 30 GB per card.
- Host RAM: the official `train_all_tasks` copies the whole state dict to CPU once (about 23 GB).
- Time: unknown until the smoke run; every step runs the forward twice (checkpointing) and the backward over 512 masks.
- Evaluation: the six prior benchmarks are the long part (MMLU 14k, HellaSwag 10k, IFEval and HumanEval generate);
  `PRIOR_LIMIT` and `PRIOR_TASKS` in config.sh shorten it.

## Not verified

The unit and end-to-end tests run on a tiny random Phi-MoE on the CPU (`bash run.sh test`). Not run anywhere yet: the
real 7.65B model, anything on a GPU, the two-GPU split (the wrappers in item 4 exist for it), the memory and speed
estimates, and the official `retention_forward_kl.response_logprobs`, which calls `.cuda()` and so needs a GPU
(tested with `.cuda()` patched to a no-op). Run `PROFILE=smoke bash run.sh` on the server first.
