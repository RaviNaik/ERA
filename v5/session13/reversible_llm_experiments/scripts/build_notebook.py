"""Regenerates notebooks/reversible_llm.ipynb from scratch via nbformat.

This script is the reviewable source of truth for the notebook's code and
prose (`git diff` on it is readable, unlike diffs of .ipynb JSON). Run:

    uv run python scripts/build_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "reversible_llm.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}
cells: list[tuple[str, str]] = []


def md(src: str) -> None:
    cells.append(("markdown", src.strip("\n")))


def code(src: str) -> None:
    cells.append(("code", src.strip("\n")))


# =============================================================================
# 0. Title
# =============================================================================
md(r"""
# Session 13 — Reversible LLMs: memory, speed and batch size

**Assignment**

1. Train a ~20M-parameter GPT-style LLM on **50M tokens** at a fixed batch size `x`.
2. Train the *same* model, on the *same* data, at the *same* batch size with a **reversible** architecture, once with the **midpoint** scheme and once with the **Euler** scheme. Pick the variant that works best on loss, speed, peak memory and other findings.
3. Train the selected variant at increasing batch sizes (`2x`, `4x`, `8x`) up to the maximum that fits on a ~40 GB GPU.

| # | Experiment | Architecture | Batch size |
|---|---|---|---|
| 1 | Baseline | standard pre-LN transformer | `x` |
| 2 | Reversible, midpoint | explicit-midpoint recurrence | `x` |
| 3 | Reversible, Euler | symplectic-Euler (Hamiltonian) two-stream | `x` |
| 4 | Reversible, best of 2 and 3 | winner | `2x` |
| 5 | Reversible, best of 2 and 3 | winner | `4x` |
| 6 | Reversible, best of 2 and 3 | winner | `8x` |

**Reference:** Gal, Eliasof, Turek, Ascher, Haber (2025), *Reversing Large Language Models for Efficient Training and Fine-Tuning*, [arXiv:2512.02056v2](https://arxiv.org/html/2512.02056v2).

**Tracked for every run**, both per step and per evaluation:

* loss and next-token accuracy (train and validation), and perplexity
* learning rate, gradient norm, and fraction of steps clipped
* throughput (tokens/s), step time split into data, forward, backward and optimizer, model and hardware TFLOP/s, and MFU
* memory: static (parameters and AdamW state), activations saved for backward (total and per sequence), per-step peak, and reserved
* reversible runs only: reconstruction drift and gradient agreement with plain autograd

Each metric is written to four places: **Aim** (`aim up`), a text log (`logs/<run>.log`), a streaming per-step JSONL file (`logs/<run>.metrics.jsonl`), and a full per-run JSON record (`results/<run>.json`, with a one-line summary appended to `results/runs.jsonl`).

**How to run**

```bash
uv sync
uv run pytest -q                                   # correctness tests (gradients, inverses, fused loss)
cd notebooks
uv run jupyter nbconvert --to notebook --execute --inplace reversible_llm.ipynb --ExecutePreprocessor.timeout=-1
uv run aim up                                      # browse all metrics (run from notebooks/)
```

Environment knobs: `REVLLM_DEVICE` (default `cuda:0`), `REVLLM_BATCH_X` (default 128), `REVLLM_REUSE=1` (default; skips runs whose `results/<run>.json` already finished, so an interrupted notebook resumes where it stopped), and `REVLLM_SMOKE=1` (a tiny WikiText-2 end-to-end sanity pass that fits a 4 GB laptop GPU; its numbers are **not** results).
""")

# =============================================================================
# 1. Setup
# =============================================================================
md(r"""
## 1. Setup and environment
""")

code(r'''
import inspect
import json
import math
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import Markdown, display

from revllm import reversible as rev
from revllm.data import load_token_dataset, prepare_wikitext
from revllm.export import export_webapp_data
from revllm.model import GPT, Block, GPTConfig, build_model, flops_per_token, param_count_report
from revllm.probe import find_max_batch, probe_step
from revllm.trainer import TrainConfig, environment_info, gpu_peak_tflops, load_record, train

pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 220)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})

ROOT = Path.cwd()  # notebooks/
SMOKE = os.environ.get("REVLLM_SMOKE", "0") == "1"
REUSE = os.environ.get("REVLLM_REUSE", "1") == "1"
IS_CUDA = torch.cuda.is_available()
DEVICE = os.environ.get("REVLLM_DEVICE", "cuda:0" if IS_CUDA else "cpu")
if IS_CUDA:
    torch.cuda.set_device(DEVICE)

# ---- experiment constants ---------------------------------------------------
BLOCK_SIZE = 512
MODEL = dict(n_layer=9, n_head=8, n_embd=256)
if SMOKE:
    DATA_DIR, DATA_CONFIG = ROOT / "data_smoke", "wikitext-2-raw-v1"
    TOKEN_BUDGET, DEFAULT_X = 4 * 512 * 40, 4       # 40 steps at x
    EVAL = dict(eval_interval_tokens=4 * 512 * 10, eval_windows=16, eval_batch_size=8, final_eval_train_windows=32)
    LOSS_CHUNK = 2048
    RESULTS_DIR, LOG_DIR = ROOT / "results_smoke", ROOT / "logs_smoke"
else:
    DATA_DIR, DATA_CONFIG = ROOT / "data", "wikitext-103-raw-v1"
    TOKEN_BUDGET, DEFAULT_X = 50_000_000, 128
    EVAL = dict(eval_interval_tokens=2_500_000, eval_windows=128, eval_batch_size=64, final_eval_train_windows=487)
    LOSS_CHUNK = 8192
    RESULTS_DIR, LOG_DIR = ROOT / "results", ROOT / "logs"
BATCH_X = int(os.environ.get("REVLLM_BATCH_X", DEFAULT_X))
AIM_REPO = str(ROOT)
EXPERIMENT = "session13_reversible_llm_smoke" if SMOKE else "session13_reversible_llm"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

ENV = environment_info(DEVICE)
GPU_TOTAL_MB = ENV.get("gpu_total_memory_mb")
PEAK_TFLOPS = gpu_peak_tflops(DEVICE)
display(pd.Series(ENV | {"smoke_mode": SMOKE, "reuse_results": REUSE, "batch_x": BATCH_X,
                         "token_budget": TOKEN_BUDGET, "gpu_peak_bf16_tflops": PEAK_TFLOPS},
                  name="value").to_frame())
if SMOKE:
    display(Markdown("> **SMOKE MODE**: tiny budget and batch sizes on WikiText-2. This checks that the pipeline "
                     "works end to end. None of the numbers below are experimental results."))
''')

# =============================================================================
# 2. Data
# =============================================================================
md(r"""
## 2. Data

**WikiText-103** (raw) is tokenized with the **GPT-2 BPE** tokenizer (`tiktoken`, vocab 50,257) and cached as `uint16` memmaps. Each document gets an end-of-text token appended.

* **Training batches** are `batch_size` windows of 512 tokens drawn uniformly at random, with replacement, from the ~119M-token train split. The seed is fixed, so every run sees the same stream of batch indices. The 50M-token budget is ~0.42 epochs, so there is essentially no repetition and train and validation loss should stay close.
* **Periodic evaluation** runs every 2.5M training tokens on a *fixed* set of 128 train and 128 validation windows (65k tokens each). All runs are scored on exactly the same tokens, and because evaluation is spaced in tokens rather than steps, curves from different batch sizes line up on the x-axis.
* **Final evaluation** covers the entire validation split (487 non-overlapping windows, ~249k tokens) plus an equally sized fixed train subset. These are the headline numbers in every table.
""")

code(r'''
meta = prepare_wikitext(DATA_DIR, dataset_config=DATA_CONFIG)
data = load_token_dataset(DATA_DIR)

import tiktoken
enc = tiktoken.get_encoding("gpt2")
sample_x, _ = data.windows("val", np.array([1000]), 64, "cpu")
display(pd.Series({
    **meta,
    "train_tokens (M)": round(meta["train_tokens"] / 1e6, 2),
    "val_tokens (k)": round(meta["val_tokens"] / 1e3, 1),
    "token_budget (M)": TOKEN_BUDGET / 1e6,
    "epochs over train split": round(TOKEN_BUDGET / meta["train_tokens"], 3),
    "val windows (full pass)": len(data.eval_starts("val", BLOCK_SIZE, None)),
}, name="value").to_frame())
print("sample validation text:\n", repr(enc.decode(sample_x[0].tolist())))
''')

# =============================================================================
# 3. Model
# =============================================================================
md(r"""
## 3. The model (~20M parameters)

A GPT-2-style decoder: `n_layer=9`, `n_head=8`, `n_embd=256`, context 512, GELU MLP with 4x expansion, learned positional embeddings, and **tied** input and output embeddings. The baseline block is the standard pre-LN residual block:
""")

code(r'''
print(inspect.getsource(Block.forward))
''')

md(r"""
All three variants use the same attention and MLP sub-layers and differ only in **how they are wired across depth**, so their parameter counts are identical (asserted below). Note that **~64% of the 20.1M parameters are the token embedding** (50,257 x 256), which is shared with the output projection. The transformer backbone itself is ~7.1M parameters. This matters for compute too: the vocab projection is more than half of the forward FLOPs (next table), and its cost is identical in all variants.
""")

code(r'''
param_rows, flops_rows = {}, {}
for variant in ["baseline", "midpoint", "euler"]:
    cfg = GPTConfig(vocab_size=data.vocab_size, block_size=BLOCK_SIZE, variant=variant, **MODEL)
    report = param_count_report(cfg)
    param_rows[variant] = {k: v for k, v in report.items() if k != "config"}
    f = flops_per_token(cfg)
    flops_rows[variant] = {
        "forward MFLOP/token": f["forward"] / 1e6,
        "model (6N-style) MFLOP/token": f["model"] / 1e6,
        "hardware MFLOP/token (incl. recompute)": f["hardware"] / 1e6,
        "recompute overhead": f["hardware"] / f["model"] - 1,
        "lm_head share of forward": f["lm_head_fraction"],
    }
param_df = pd.DataFrame(param_rows)
assert param_df.loc["total"].nunique() == 1, "variants must have identical parameter counts"
N_PARAMS = int(param_df.loc["total", "baseline"])
display(Markdown(f"**Total parameters: {N_PARAMS:,}** (identical for all variants)"))
display(param_df)
display(pd.DataFrame(flops_rows).style.format("{:.3f}"))
''')

md(r"""
**Reading the FLOPs table.** *Model FLOPs* (~3 x forward) are the useful work and are the same for every variant, so they are what **MFU** is computed from. *Hardware FLOPs* add what the GPU really executes. A reversible model re-runs every backbone sub-layer once more during backward to reconstruct its inputs, which is the same price as full activation checkpointing. The lm_head is **not** recomputed (the fused loss in `loss.py` computes its gradient in the same pass as the loss), so the extra compute is only ~14% of the model FLOPs rather than the ~33% a full extra forward would cost.
""")

# =============================================================================
# 4. Reversible architectures
# =============================================================================
md(r"""
## 4. Reversible architectures: theory, implementation, verification

### 4.1 Why reversibility saves memory
Backprop through a standard residual network needs every layer's input (plus the internal activations of each sub-layer), so autograd **stores activations for all L layers**: memory grows as $O(L \cdot B \cdot T \cdot d)$. If each layer is an *invertible* map, the inputs of layer $\ell$ can be **recomputed from its outputs** during backward. Only the final hidden state needs storing, and activation memory becomes **O(1) in depth**. The price is recomputing each layer once in backward.

### 4.2 The two schemes (paper eqs. 4-5 and 8-9)

**Midpoint** (a leapfrog-style, two-step recurrence over depth). With step size $h$:
$$p^{(\ell+1)} = p^{(\ell-1)} + 2h\, f_\ell\big(p^{(\ell)}\big), \qquad f_\ell(p) = \mathrm{Attn}_\ell(\mathrm{LN}_1 p) + \mathrm{MLP}_\ell\big(\mathrm{LN}_2(p + \mathrm{Attn}_\ell(\mathrm{LN}_1 p))\big)$$
Inverse: $p^{(\ell-1)} = p^{(\ell+1)} - 2h\, f_\ell(p^{(\ell)})$, with no need to invert $f$ itself. We use $p^{(-1)} = p^{(0)} = x$ and read out $p^{(L)}$.
$f_\ell$ is *exactly* the residual branch of a standard pre-LN block, so with $h = 0.5$ the first layer **is** a standard block and later layers are $p^{(\ell+1)} = p^{(\ell-1)} + \mathrm{Block}^{\text{res}}_\ell(p^{(\ell)})$. We use **$h = 0.25$** (see the step-size sweep in Section 6).

**Euler** (symplectic Euler, the paper's Hamiltonian variant with $a=b=1$): two streams, with attention updating one and the MLP the other:
$$q^{(\ell)} = q^{(\ell-1)} + \mathrm{Attn}_\ell\big(\mathrm{LN}_1 p^{(\ell-1)}\big), \qquad p^{(\ell)} = p^{(\ell-1)} + \mathrm{MLP}_\ell\big(\mathrm{LN}_2\, q^{(\ell)}\big)$$
Inverse: $p^{(\ell-1)} = p^{(\ell)} - \mathrm{MLP}_\ell(\mathrm{LN}_2 q^{(\ell)})$, then $q^{(\ell-1)} = q^{(\ell)} - \mathrm{Attn}_\ell(\mathrm{LN}_1 p^{(\ell-1)})$. We use $q^{(0)} = p^{(0)} = x/\sqrt2$ and read out $(q^{(L)} + p^{(L)})/\sqrt2$.

**Stability.** The paper's analysis (Sec. 3) shows the midpoint scheme is only *marginally* stable: the recurrence has a parasitic, sign-alternating mode, and the backward reconstruction amplifies rounding error unless the eigenvalues $\lambda$ of the layer Jacobian satisfy $|\lambda h|$ small and nearly imaginary. The symplectic Euler scheme is volume-preserving and far better behaved. This is exactly what the reconstruction-drift diagnostics below measure.

### 4.3 The implementation
""")

code(r'''
print(inspect.getsource(rev.MidpointCell.step))
print(inspect.getsource(rev.MidpointCell.backward_step))
print(inspect.getsource(rev.EulerCell.step))
print(inspect.getsource(rev.EulerCell.backward_step))
''')

md(r"""
Four implementation details make this both **correct** and **actually memory-efficient**:

1. **One `autograd.Function` for the entire stack** (`_ReversibleSequenceFunction`). A per-layer Function that calls `save_for_backward` on each layer keeps every layer's output alive until `.backward()`, which is the same memory as the baseline. The first implementation had exactly this bug.
2. **Each sub-layer runs once in backward.** `backward_step` uses a single evaluation of $f$ (or $g$) both to reconstruct the previous state *and* to get the vector-Jacobian product. The previous implementation ran `inverse()` and then re-ran `step()` under autograd: two extra forward passes per layer, which made reversible runs ~2x slower than baseline.
3. **Backward runs under the forward's autocast state** (`torch.amp.custom_fwd/custom_bwd`). The autograd engine otherwise runs backward *outside* autocast, so the reconstruction recomputed every sub-layer in slow fp32 while the forward had used bf16. Reconstruction was then inexact (0.3-0.8% gradient error) and slow.
4. **Bit-exact reversal with fixed-point streams** (`reversible_exact=True`). Floating-point residual streams are *not* exactly invertible: $(x + u) - u \neq x$ in general. Under bf16 autocast, a 1e-8 fp32 rounding difference in a reconstructed state can flip the bf16 rounding of the next sub-layer's input, which turns into a ~1e-3 error in that layer's output and compounds over depth. With trained (sharper) weights, local tests measured 10-25% reconstruction drift and gradients that disagree with autograd (cosine as low as 0.3 with sharpened weights). We therefore keep both streams in **float64** and round every sub-layer update onto a fixed $2^{-30}$ grid before adding it (the fixed-point idea of MacKay et al., 2018). Additions of grid values are exact in float64, so the backward pass recovers every state **bit-for-bit** and recomputes each sub-layer on exactly the input the forward used. The rounding is straight-through for gradients. The cost is 8 bytes instead of 4 per stream element: still O(1) in depth.

### 4.4 Numerical verification

First, in float64 on CPU, the reversible gradients must equal plain-autograd gradients to machine precision (`naive_forward` runs the *same* cells with ordinary autograd).
""")

code(r'''
torch.manual_seed(0)
check_cfg = GPTConfig(vocab_size=64, block_size=32, n_layer=6, n_head=4, n_embd=32)
rows = []
for kind in ["midpoint", "euler"]:
    stack = rev.ReversibleStack(check_cfg, kind).double()
    x = torch.randn(3, 32, 32, dtype=torch.float64)
    diag = rev.reversibility_diagnostics(stack, x, autocast_dtype=None)
    rows.append({"variant": kind, "precision": "float64 / CPU",
                 "max reconstruction rel. error": diag["max_rel_error"],
                 "grad cosine vs autograd": diag["grad_cosine"],
                 "grad rel. error vs autograd": diag["grad_rel_error"]})
display(pd.DataFrame(rows))
''')

md(r"""
Next, the precision that training actually uses: **bf16 autocast** on the GPU, with the real model width. We compare exact fixed-point streams (used for training) with plain fp32 streams, at initialisation and with the weight matrices scaled 8x to mimic the sharper layers of a trained model. Exact mode should show a reconstruction error of **exactly 0**. Its remaining gradient difference against plain autograd is ordinary bf16 backward noise (different accumulation order), not a reversal error.
""")

code(r'''
full_cfgs = {kind: GPTConfig(vocab_size=data.vocab_size, block_size=BLOCK_SIZE, variant=kind, **MODEL)
             for kind in ["midpoint", "euler"]}
init_diag = {}
if IS_CUDA:
    x_ids, _ = data.windows("val", data.eval_starts("val", BLOCK_SIZE, 4, seed=4), BLOCK_SIZE, DEVICE)
    for kind in ["midpoint", "euler"]:
        for exact in [True, False]:
            for weight_scale in [1.0, 8.0]:
                torch.manual_seed(0)
                cfg = GPTConfig(vocab_size=data.vocab_size, block_size=BLOCK_SIZE, variant=kind,
                                reversible_exact=exact, **MODEL)
                model = GPT(cfg).to(DEVICE)
                with torch.no_grad():
                    for p in model.transformer.h.parameters():
                        if p.dim() == 2:
                            p.mul_(weight_scale)
                    emb = model.transformer.wte(x_ids) + model.transformer.wpe(torch.arange(BLOCK_SIZE, device=DEVICE))
                d = rev.reversibility_diagnostics(model.transformer.h, emb, torch.bfloat16)
                init_diag[(kind, "exact fp64 fixed-point" if exact else "plain fp32", weight_scale)] = {
                    k: v for k, v in d.items() if k != "per_layer_rel_error"}
                del model
                torch.cuda.empty_cache()
    diag_df = pd.DataFrame(init_diag).T
    diag_df.index.names = ["variant", "streams", "weight scale"]
    display(diag_df)

    # What does exactness cost in speed? One training step (forward, backward, AdamW) at batch x.
    speed_rows = []
    for kind in ["midpoint", "euler"]:
        for exact in [True, False]:
            cfg = GPTConfig(vocab_size=data.vocab_size, block_size=BLOCK_SIZE, variant=kind,
                            reversible_exact=exact, loss_chunk_size=LOSS_CHUNK, **MODEL)
            r = probe_step(cfg, BATCH_X, DEVICE, steps=4)
            speed_rows.append({"variant": kind, "streams": "exact fp64" if exact else "plain fp32",
                               "step ms": r.get("step_time_ms"), "tokens/s": r.get("tokens_per_sec"),
                               "peak MB": r.get("peak_memory_mb")})
    base_r = probe_step(GPTConfig(vocab_size=data.vocab_size, block_size=BLOCK_SIZE, variant="baseline",
                                  loss_chunk_size=LOSS_CHUNK, **MODEL), BATCH_X, DEVICE, steps=4)
    speed_rows.append({"variant": "baseline", "streams": "-", "step ms": base_r.get("step_time_ms"),
                       "tokens/s": base_r.get("tokens_per_sec"), "peak MB": base_r.get("peak_memory_mb")})
    display(Markdown(f"**Cost of exact reversal: one training step at batch {BATCH_X}**"))
    display(pd.DataFrame(speed_rows).round(1))
else:
    print("CUDA not available: skipping the bf16 check")
''')

md(r"""
### 4.5 Activation memory vs. depth

This measures the memory autograd keeps alive **between forward and backward** (the activations saved for backward) for 3 to 24 layers at a fixed batch of 8 x 512 tokens. The baseline should grow linearly with depth and the reversible stacks should stay flat. A small constant remains for all variants: the embeddings and the fused-loss gradient buffers.
""")

code(r'''
def saved_activation_mb(variant, n_layer, batch=8):
    cfg = GPTConfig(vocab_size=data.vocab_size, block_size=BLOCK_SIZE, variant=variant,
                    n_layer=n_layer, n_head=MODEL["n_head"], n_embd=MODEL["n_embd"], loss_chunk_size=LOSS_CHUNK)
    model = build_model(cfg, device=DEVICE)
    x = torch.randint(0, cfg.vocab_size, (batch, BLOCK_SIZE), device=DEVICE)
    torch.cuda.synchronize(DEVICE)
    before = torch.cuda.memory_allocated(DEVICE)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, loss, _ = model(x, x)
    saved = (torch.cuda.memory_allocated(DEVICE) - before) / 2**20
    loss.backward()
    del model, x, loss
    torch.cuda.empty_cache()
    return saved

if IS_CUDA:
    depths = [3, 6, 9, 12, 18, 24]
    depth_df = pd.DataFrame({v: [saved_activation_mb(v, L) for L in depths] for v in ["baseline", "midpoint", "euler"]},
                            index=pd.Index(depths, name="n_layer"))
    display(depth_df.round(1))
    ax = depth_df.plot(marker="o", figsize=(6.5, 3.6), title="Activation memory saved for backward (batch 8 x 512)")
    ax.set_ylabel("MB"); plt.show()
    growth = {v: (depth_df[v].iloc[-1] - depth_df[v].iloc[0]) / (depths[-1] - depths[0]) for v in depth_df}
    display(Markdown("**Marginal activation memory per extra layer** (batch 8): " +
                     ", ".join(f"{v}: **{g:.1f} MB/layer**" for v, g in growth.items())))
''')

# =============================================================================
# 5. Batch size
# =============================================================================
md(r"""
## 5. Choosing the batch size `x`

`x` has to satisfy three constraints on a ~40 GB GPU:

1. **All three variants fit at `x`**, which the baseline needs for experiments 1-3.
2. **Both reversible variants fit at `8x`.** The winner is not known yet and it must fit at experiment 6.
3. **Enough optimizer steps remain at `8x`.** The token budget is fixed at 50M, so `8x` gets 1/8 of the steps. At `x = 128` (65,536 tokens per step) that is 763 steps at `x` and 96 at `8x`. At `x = 256` it would be only 48 steps at `8x`, too few for a meaningful LR schedule.

Local measurements put the baseline at ~42 MB of saved activations per 512-token sequence and the reversible variants at ~2 MB. On 40 GB the baseline should therefore top out around batch 800-900, while `x = 128` puts `8x = 1024` beyond what the baseline can fit but well within the reversible budget. The probe below verifies this on *this* GPU with real training steps (forward, backward, AdamW). An `OOM` in the baseline column at `8x` is the expected outcome, and it is the point of the exercise.
""")

code(r'''
probe_batches = sorted({max(1, BATCH_X // 2), BATCH_X, 2 * BATCH_X, 4 * BATCH_X, 8 * BATCH_X})
probe_rows = []
if IS_CUDA:
    for variant in ["baseline", "midpoint", "euler"]:
        cfg = GPTConfig(vocab_size=data.vocab_size, block_size=BLOCK_SIZE, variant=variant,
                        loss_chunk_size=LOSS_CHUNK, **MODEL)
        for b in probe_batches:
            probe_rows.append(probe_step(cfg, b, DEVICE, steps=2))
probe_df = pd.DataFrame(probe_rows)
if len(probe_df):
    mem_table = probe_df.pivot(index="batch_size", columns="variant", values="peak_memory_mb")[["baseline", "midpoint", "euler"]]
    act_table = probe_df.pivot(index="batch_size", columns="variant", values="activation_saved_mb_per_sample")[["baseline", "midpoint", "euler"]]
    tps_table = probe_df.pivot(index="batch_size", columns="variant", values="tokens_per_sec")[["baseline", "midpoint", "euler"]]
    display(Markdown("**Peak memory per training step (MB)**. Blank means OOM."))
    display(mem_table.round(0))
    display(Markdown("**Activation memory saved for backward, per sequence (MB)**"))
    display(act_table.round(2))
    display(Markdown("**Probe throughput (tokens/s, median of 2 steps, rough)**"))
    display(tps_table.round(0))
''')

code(r'''
def fits(variant, batch):
    row = probe_df[(probe_df.variant == variant) & (probe_df.batch_size == batch)]
    return bool(len(row)) and bool(row.fits.iloc[0])

if IS_CUDA:
    checks = {
        f"baseline fits x={BATCH_X}": fits("baseline", BATCH_X),
        f"midpoint fits 8x={8 * BATCH_X}": fits("midpoint", 8 * BATCH_X),
        f"euler fits 8x={8 * BATCH_X}": fits("euler", 8 * BATCH_X),
        f"baseline fits 8x={8 * BATCH_X} (expected False on 40 GB)": fits("baseline", 8 * BATCH_X),
    }
    display(pd.Series(checks, name="result").to_frame())
    assert checks[f"baseline fits x={BATCH_X}"], "x too large for the baseline on this GPU: set REVLLM_BATCH_X lower"
    assert checks[f"midpoint fits 8x={8 * BATCH_X}"] and checks[f"euler fits 8x={8 * BATCH_X}"], \
        "8x does not fit a reversible variant: set REVLLM_BATCH_X lower"

schedule = pd.DataFrame([
    {"experiment": name, "batch": b, "tokens/step": b * BLOCK_SIZE,
     "optimizer steps": math.ceil(TOKEN_BUDGET / (b * BLOCK_SIZE))}
    for name, b in [("1-3 (x)", BATCH_X), ("4 (2x)", 2 * BATCH_X), ("5 (4x)", 4 * BATCH_X), ("6 (8x)", 8 * BATCH_X)]
])
display(schedule)
''')

# =============================================================================
# 6. Training protocol
# =============================================================================
md(r"""
## 6. Training protocol

| setting | value | notes |
|---|---|---|
| optimizer | AdamW (fused), betas (0.9, 0.95), wd 0.1 on matrices only | |
| peak LR at `x` | 1e-3 | a small-model value; a local sweep beat 3e-4 by a wide margin at this step count |
| schedule | linear warmup (5% of steps, min 10), cosine decay to 10% of peak | warmup and decay scale with each run's step count |
| LR at `2x`/`4x`/`8x` | **sqrt scaling**: `lr = 1e-3 * sqrt(batch / x)`, capped at 3e-3 | the standard Adam heuristic; see note below |
| grad clipping | global norm 1.0 | |
| precision | bf16 autocast, fp32 master weights and residual streams | |
| loss | fused chunked lm_head + cross-entropy (no full logits tensor) | same for every variant |
| dropout | 0 | reversible cells forbid dropout (the recompute would draw different masks) |
| midpoint step size | h = 0.25 (2h = 0.5) | best of h in {1/9, 0.25, 0.5} in a local sweep (below) |
| reversible streams | float64 fixed-point, exact reversal | Section 4.3, item 4 |
| seed | 1337 | same data order for every run |

**Why scale the LR for experiments 4-6?** At a fixed token budget, `8x` batch means 8x fewer optimizer steps. Keeping the LR fixed would confound "bigger batch" with "much less optimization". sqrt(k) scaling keeps the per-step noise-to-signal ratio roughly constant for Adam, so the comparison stays fair to the large-batch runs. Experiments 1-3 all use exactly the same LR, schedule and data order, which is what isolates the architecture.

**Preliminary local sweep behind the LR and h choices.** Measured on the development machine: an RTX 500 Ada 4 GB laptop GPU, WikiText-2, batch 16, 2M tokens, one seed. These numbers only justify the settings. They are *not* results of this assignment.

| run | streams | final val loss | tokens/s | recon drift at input | grad rel. error vs autograd |
|---|---|---|---|---|---|
| baseline, lr 3e-4 | - | 6.573 | 32.1k | - | - |
| **baseline, lr 1e-3** | - | **6.266** | 32.0k | - | - |
| midpoint h = 1/9 (previous default) | fp32 | 6.289 | 29.0k | 0.023 | 0.16% |
| midpoint h = 0.25 | fp32 | 6.260 | 28.8k | 0.084 | 0.25% |
| midpoint h = 0.5 | fp32 | 6.271 | 29.1k | 0.253 | 0.41% |
| Euler | fp32 | 6.300 | 29.4k | 0.112 | 1.98% |
| **midpoint h = 0.25** | **exact fp64** | **6.261** | 24.8k | **0** | 0.00% |
| midpoint h = 0.5 | exact fp64 | 6.270 | 24.8k | 0 | 0.04% |
| Euler | exact fp64 | 6.304 | 21.8k | 0 | 0.06% |

Takeaways: lr 1e-3 is far better than 3e-4 at this step count. Among midpoint step sizes, larger h drifts more (the paper's stability analysis), and h = 0.25 has the lowest loss. Exact streams remove drift completely at the same loss. On this laptop GPU (fp64 at 1/64 rate) exact mode costs 15-25% throughput; the Section 4.4 timing cell measures the cost on the real GPU.
""")

code(r'''
BASE_LR = 1e-3
LR_SCALING_FOR_SCALE_UP = "sqrt"

def make_config(variant, batch_size, label, lr_scaling="none"):
    return TrainConfig(
        variant=variant, batch_size=batch_size, block_size=BLOCK_SIZE, token_budget=TOKEN_BUDGET,
        learning_rate=BASE_LR, lr_scaling=lr_scaling, lr_ref_batch=BATCH_X, max_lr=3e-3,
        loss_chunk_size=LOSS_CHUNK, device=DEVICE, aim_repo=AIM_REPO, experiment=EXPERIMENT,
        label=label, tags=["smoke"] if SMOKE else [], **EVAL, **MODEL,
    )

preview = pd.DataFrame([
    {"label": lbl, "batch": c.batch_size, "steps": c.total_steps, "warmup": c.warmup_steps,
     "peak_lr": c.peak_lr, "eval every (steps)": c.eval_every_steps}
    for lbl, c in [(l, make_config("midpoint", b, l, s)) for l, b, s in [
        ("x", BATCH_X, "none"), ("2x", 2 * BATCH_X, LR_SCALING_FOR_SCALE_UP),
        ("4x", 4 * BATCH_X, LR_SCALING_FOR_SCALE_UP), ("8x", 8 * BATCH_X, LR_SCALING_FOR_SCALE_UP)]]
])
display(preview)
''')

# =============================================================================
# 7. Analysis helpers
# =============================================================================
md(r"""
## 7. Analysis helpers

These are defined here rather than in `src/` because they are the notebook's reporting layer. Every experiment below runs the trainer, then:

* `metrics_table`: a one-column table of every summary metric, grouped as quality, speed, compute, memory and optimization.
* `plot_run`: an 8-panel dashboard (loss vs tokens, accuracy, LR, grad norm, throughput, step-time breakdown, memory timeline, per-step peak).
* `findings`: data-driven observations, including deltas against a reference run.

`run_or_load` skips training when a finished result already exists and `REVLLM_REUSE=1`, so an interrupted notebook can be re-executed without redoing completed runs.
""")

code(r'''
RECORDS: dict[str, dict] = {}

def run_or_load(variant, batch_size, label, lr_scaling="none"):
    cfg = make_config(variant, batch_size, label, lr_scaling)
    cached = load_record(label, RESULTS_DIR) if REUSE else None
    if cached and cached["summary"]["status"] == "ok" and cached["config"]["token_budget"] == TOKEN_BUDGET:
        print(f"[reuse] {label}: loaded {RESULTS_DIR / (label + '.json')}")
        record = cached
    else:
        print(f"[train] {label}: variant={variant} batch={batch_size} steps={cfg.total_steps} peak_lr={cfg.peak_lr:.2e}")
        record = train(data, cfg, results_dir=RESULTS_DIR, log_dir=LOG_DIR)
    RECORDS[label] = record
    s = record["summary"]
    print(f"status={s['status']}  log={LOG_DIR / (label + '.log')}  aim_run={s.get('aim_run_hash')}")
    return record


SECTIONS = {
    "quality": ["final_val_loss", "final_val_perplexity", "final_val_accuracy", "final_train_loss",
                "final_train_accuracy", "generalization_gap", "best_periodic_val_loss",
                "tokens_to_val_loss_6", "tokens_to_val_loss_5", "tokens_to_val_loss_4.5"],
    "speed": ["mean_tokens_per_sec", "median_tokens_per_sec", "p10_tokens_per_sec", "p90_tokens_per_sec",
              "mean_step_time_ms", "mean_data_ms", "mean_forward_ms", "mean_backward_ms", "mean_optimizer_ms",
              "train_time_s", "eval_time_s", "wall_clock_s"],
    "compute": ["model_tflops_per_sec", "hardware_tflops_per_sec", "mfu", "hfu", "total_model_pflops",
                "model_flops_per_token", "hardware_flops_per_token"],
    "memory": ["peak_step_memory_mb", "peak_memory_reserved_mb", "static_memory_mb", "param_memory_mb",
               "optimizer_state_mb", "activation_saved_mb", "activation_saved_mb_per_sample", "peak_memory_frac_of_gpu"],
    "optimization": ["batch_size", "total_steps", "tokens_seen", "peak_lr", "warmup_steps",
                     "mean_grad_norm", "max_grad_norm", "frac_steps_clipped"],
}

def metrics_table(*labels):
    rows = []
    for section, keys in SECTIONS.items():
        for key in keys:
            rows.append({"section": section, "metric": key,
                         **{lbl: RECORDS[lbl]["summary"].get(key) for lbl in labels}})
    return pd.DataFrame(rows).set_index(["section", "metric"])


def plot_run(label, ref_label=None):
    rec = RECORDS[label]
    h, e = rec["history"], rec["evals"]
    ref = RECORDS.get(ref_label) if ref_label else None
    fig, ax = plt.subplots(2, 4, figsize=(19, 7.2))
    tok_m = np.array(h["tokens"]) / 1e6
    ax[0, 0].plot(tok_m, h["loss"], alpha=0.35, lw=0.8, label="train (per step)")
    ax[0, 0].plot(np.array(e["tokens"]) / 1e6, e["train_loss"], "o-", label="train (eval)")
    ax[0, 0].plot(np.array(e["tokens"]) / 1e6, e["val_loss"], "s-", label="val (eval)")
    if ref:
        ax[0, 0].plot(np.array(ref["evals"]["tokens"]) / 1e6, ref["evals"]["val_loss"], "k--", lw=1, label=f"val: {ref_label}")
    ax[0, 0].set(title="loss", xlabel="tokens (M)", ylim=(min(e["val_loss"]) - 0.3, min(8.0, max(e["val_loss"][1:] or [8]) + 0.3)))
    ax[0, 0].legend(fontsize=7)
    ax[0, 1].plot(np.array(e["tokens"]) / 1e6, e["train_accuracy"], "o-", label="train")
    ax[0, 1].plot(np.array(e["tokens"]) / 1e6, e["val_accuracy"], "s-", label="val")
    if ref:
        ax[0, 1].plot(np.array(ref["evals"]["tokens"]) / 1e6, ref["evals"]["val_accuracy"], "k--", lw=1, label=f"val: {ref_label}")
    ax[0, 1].set(title="next-token accuracy", xlabel="tokens (M)"); ax[0, 1].legend(fontsize=7)
    ax[0, 2].plot(h["step"], h["lr"]); ax[0, 2].set(title="learning rate", xlabel="step")
    ax[0, 3].plot(h["step"], h["grad_norm"], lw=0.8); ax[0, 3].axhline(1.0, color="r", ls=":", lw=1)
    ax[0, 3].set(title="grad norm (clip = 1.0)", xlabel="step", yscale="log")
    ax[1, 0].plot(h["step"], h["tokens_per_sec"], lw=0.8, label=label)
    if ref:
        ax[1, 0].axhline(ref["summary"]["mean_tokens_per_sec"], color="k", ls="--", lw=1, label=f"{ref_label} mean")
    ax[1, 0].set(title="throughput (tokens/s)", xlabel="step"); ax[1, 0].legend(fontsize=7)
    parts = ["data_ms", "forward_ms", "backward_ms", "optimizer_ms"]
    means = [rec["summary"].get(f"mean_{p}") or 0 for p in ["data_ms", "forward_ms", "backward_ms", "optimizer_ms"]]
    bars = {label: means}
    if ref:
        bars[ref_label] = [ref["summary"].get(f"mean_{p}") or 0 for p in ["data_ms", "forward_ms", "backward_ms", "optimizer_ms"]]
    bottom = np.zeros(len(bars))
    for i, p in enumerate(parts):
        vals = np.array([v[i] for v in bars.values()])
        ax[1, 1].bar(list(bars), vals, bottom=bottom, label=p.replace("_ms", ""))
        bottom += vals
    ax[1, 1].set(title="mean step-time breakdown (ms)"); ax[1, 1].legend(fontsize=7)
    ax[1, 2].plot(h["step"], h["memory_allocated_mb"], label="allocated after step")
    ax[1, 2].plot(h["step"], h["memory_reserved_mb"], label="reserved")
    ax[1, 2].plot(h["step"], h["activation_saved_mb"], label="saved activations")
    ax[1, 2].set(title="memory (MB)", xlabel="step"); ax[1, 2].legend(fontsize=7)
    ax[1, 3].plot(h["step"], h["step_peak_memory_mb"], label=label)
    if ref:
        ax[1, 3].axhline(ref["summary"]["peak_step_memory_mb"], color="k", ls="--", lw=1, label=f"{ref_label}")
    ax[1, 3].set(title="per-step peak memory (MB)", xlabel="step"); ax[1, 3].legend(fontsize=7)
    fig.suptitle(f"{label}: {rec['summary']['variant']}, batch {rec['summary']['batch_size']}", fontsize=13)
    fig.tight_layout(); plt.show()


def _pct(a, b):
    return (a / b - 1) * 100 if a is not None and b else float("nan")

def findings(label, ref_label=None):
    s = RECORDS[label]["summary"]
    e = RECORDS[label]["evals"]
    if s["status"] != "ok":
        display(Markdown(f"### Findings: `{label}`\n\n**Run did not finish: status `{s['status']}`** ({s['error']})."))
        return
    out = [f"### Findings: `{label}`", ""]
    out.append(f"- **Quality:** final val loss **{s['final_val_loss']:.4f}** (perplexity {s['final_val_perplexity']:.1f}), "
               f"val accuracy **{s['final_val_accuracy'] * 100:.2f}%**; train loss {s['final_train_loss']:.4f}, "
               f"generalization gap {s['generalization_gap']:+.4f}.")
    if len(e["val_loss"]) >= 3:
        last_drop = e["val_loss"][-2] - e["val_loss"][-1]
        span = (e["tokens"][-1] - e["tokens"][-2]) / 1e6
        if last_drop > 0.01:
            trend = f"fell {last_drop:.4f} over the last {span:.2f}M tokens and was still improving, so the model is budget-limited"
        elif last_drop >= 0:
            trend = f"moved only {last_drop:.4f} over the last {span:.2f}M tokens and had flattened"
        else:
            trend = f"**rose** {-last_drop:.4f} over the last {span:.2f}M tokens, a sign of an unstable or too-aggressive LR schedule"
        out.append(f"- **Convergence:** val loss {trend}.")
    out.append(f"- **Speed:** {s['mean_tokens_per_sec']:,.0f} tokens/s (p10-p90: {s['p10_tokens_per_sec']:,.0f} to {s['p90_tokens_per_sec']:,.0f}); "
               f"step {s['mean_step_time_ms']:.1f} ms = data {s['mean_data_ms']:.1f} + forward {s['mean_forward_ms'] or 0:.1f} "
               f"+ backward {s['mean_backward_ms'] or 0:.1f} + optimizer {s['mean_optimizer_ms'] or 0:.1f} ms; wall clock {s['wall_clock_s'] / 60:.1f} min. "
               f"(Forward includes the fused lm_head gradient, see Section 3.)")
    mfu = f"MFU **{s['mfu'] * 100:.1f}%**, HFU {s['hfu'] * 100:.1f}%" if s.get("mfu") else "MFU n/a (unknown GPU peak)"
    out.append(f"- **Compute:** {s['model_tflops_per_sec']:.1f} model TFLOP/s ({s['hardware_tflops_per_sec']:.1f} hardware TFLOP/s incl. recompute); {mfu}; "
               f"{s['total_model_pflops']:.3g} PFLOP total.")
    out.append(f"- **Memory:** per-step peak **{s['peak_step_memory_mb']:,.0f} MB**"
               + (f" ({s['peak_memory_frac_of_gpu'] * 100:.1f}% of the GPU)" if s.get("peak_memory_frac_of_gpu") else "")
               + f"; static {s['static_memory_mb']:,.0f} MB (params {s['param_memory_mb']:,.0f} + AdamW {s['optimizer_state_mb']:,.0f}); "
               f"activations saved for backward **{s['activation_saved_mb']:,.0f} MB** (of which {s['loss_buffer_mb']:.0f} MB is the batch-independent fused-loss "
               f"weight-gradient buffer) = **{s['activation_saved_mb_per_sample']:.2f} MB per sequence**.")
    out.append(f"- **Optimization:** {s['total_steps']} steps, peak LR {s['peak_lr']:.2e}, mean grad norm {s['mean_grad_norm']:.3f} "
               f"(max {s['max_grad_norm']:.2f}), clipped on {s['frac_steps_clipped'] * 100:.1f}% of steps.")
    if s.get("reversibility"):
        r = s["reversibility"]
        out.append(f"- **Reversibility (trained weights, bf16):** max reconstruction drift {r['max_rel_error']:.2e} "
                   f"(at the input: {r['input_rel_error']:.2e}); gradient vs plain autograd: cosine {r['grad_cosine']:.6f}, "
                   f"relative error {r['grad_rel_error']:.2e}.")
    if ref_label and ref_label in RECORDS and RECORDS[ref_label]["summary"]["status"] == "ok":
        r = RECORDS[ref_label]["summary"]
        out.append("")
        out.append(f"**Compared to `{ref_label}`:** val loss {s['final_val_loss'] - r['final_val_loss']:+.4f} "
                   f"({_pct(s['final_val_loss'], r['final_val_loss']):+.2f}%), val accuracy "
                   f"{(s['final_val_accuracy'] - r['final_val_accuracy']) * 100:+.2f} pts, throughput "
                   f"**{_pct(s['mean_tokens_per_sec'], r['mean_tokens_per_sec']):+.1f}%**, wall clock "
                   f"{_pct(s['wall_clock_s'], r['wall_clock_s']):+.1f}%, per-step peak memory "
                   f"**{_pct(s['peak_step_memory_mb'], r['peak_step_memory_mb']):+.1f}%**, activation memory per sequence "
                   f"**{r['activation_saved_mb_per_sample'] / s['activation_saved_mb_per_sample']:.1f}x smaller**"
                   if s["activation_saved_mb_per_sample"] else "")
    display(Markdown("\n".join(out)))


def report(label, ref_label=None):
    display(metrics_table(*[l for l in [label, ref_label] if l]).style.format(
        lambda v: f"{v:,.4f}" if isinstance(v, float) and abs(v) < 100 else (f"{v:,.0f}" if isinstance(v, (int, float)) else v), na_rep="-"))
    if RECORDS[label]["summary"]["status"] == "ok":
        plot_run(label, ref_label)
    findings(label, ref_label)
''')

# =============================================================================
# Experiments 1-3
# =============================================================================
md(r"""
---
## Experiment 1: Baseline, batch size `x`

The standard pre-LN transformer, trained on 50M tokens at batch `x`. It is the reference point for everything else.

**Expectations.** Activation memory grows with depth and batch, ~42 MB per 512-token sequence (Section 5). Throughput is the highest of the three variants because nothing is recomputed. Val loss should still be falling at the end of the budget: a 20M model is nowhere near converged after 50M tokens.
""")

code(r'''
LBL_BASE = "exp1_baseline_x"
run_or_load("baseline", BATCH_X, LBL_BASE)
report(LBL_BASE)
''')

md(r"""
---
## Experiment 2: Reversible midpoint, batch size `x`

Same data order, batch size, LR schedule and token budget. Only the depth recurrence changes, to $p^{(\ell+1)} = p^{(\ell-1)} + 2h f_\ell(p^{(\ell)})$ with $h=0.25$, backpropagated with the O(1)-memory reversible backward.

**Expectations.** Much smaller saved activations (~2 MB per sequence rather than ~42), at the cost of recomputing the backbone in backward: throughput a few percent to ~15% lower than the baseline, with the loss-dominated step time masking part of it. Loss should be close to the baseline (the paper reports midpoint within ~0.01 of baseline for GPT-2). Because reversal is exact, the diagnostics at the end of the run should show zero reconstruction drift even though the midpoint scheme is only marginally stable.
""")

code(r'''
LBL_MID = "exp2_midpoint_x"
run_or_load("midpoint", BATCH_X, LBL_MID)
report(LBL_MID, LBL_BASE)
''')

md(r"""
---
## Experiment 3: Reversible Euler, batch size `x`

The two-stream symplectic-Euler (Hamiltonian) update: attention updates $q$ from $p$, then the MLP updates $p$ from the new $q$.

**Expectations.** Memory is the same as midpoint: both store two float64 streams (midpoint keeps the current and previous state, Euler keeps q and p), and both are O(1) in depth. Compute and recompute are also the same. The architecture is less like a standard transformer (attention and MLP see different streams, and every layer's attention reads a state that has not yet seen that layer's MLP), so loss may be slightly worse than the baseline.
""")

code(r'''
LBL_EUL = "exp3_euler_x"
run_or_load("euler", BATCH_X, LBL_EUL)
report(LBL_EUL, LBL_BASE)
''')

# =============================================================================
# Selection
# =============================================================================
md(r"""
---
## Choosing the reversible variant

**Decision rule (fixed before looking at the results):**

1. Discard a variant that did not finish (OOM or divergence), or whose reversible gradient disagrees with autograd (cosine < 0.999). Its memory savings would come at the cost of wrong gradients.
2. **Primary criterion: final validation loss** on the full validation split.
3. If the two are within **1%** in val loss (a practical tie at this scale), prefer higher **throughput**, then lower **peak memory**, then smaller **reconstruction drift**.
""")

code(r'''
def comparison_frame(labels):
    keys = ["variant", "batch_size", "total_steps", "final_val_loss", "final_val_perplexity", "final_val_accuracy",
            "final_train_loss", "mean_tokens_per_sec", "mean_step_time_ms", "mean_forward_ms", "mean_backward_ms",
            "mfu", "wall_clock_s", "peak_step_memory_mb", "activation_saved_mb", "activation_saved_mb_per_sample",
            "mean_grad_norm", "frac_steps_clipped"]
    df = pd.DataFrame({lbl: {k: RECORDS[lbl]["summary"].get(k) for k in keys} for lbl in labels}).T
    for lbl in labels:
        r = RECORDS[lbl]["summary"].get("reversibility") or {}
        df.loc[lbl, "recon_drift_max"] = r.get("max_rel_error")
        df.loc[lbl, "grad_cosine_vs_autograd"] = r.get("grad_cosine")
    return df

cmp3 = comparison_frame([LBL_BASE, LBL_MID, LBL_EUL])
display(cmp3)

def eligible(lbl):
    s = RECORDS[lbl]["summary"]
    r = s.get("reversibility") or {}
    return s["status"] == "ok" and (r.get("grad_cosine") or 0) >= 0.999

cands = [l for l in [LBL_MID, LBL_EUL] if eligible(l)]
assert cands, "neither reversible variant is usable, see the diagnostics above"
if len(cands) == 1:
    winner_lbl, reason = cands[0], "the only variant passing the correctness and finished-run checks"
else:
    m, e = (RECORDS[l]["summary"] for l in cands)
    gap = abs(m["final_val_loss"] - e["final_val_loss"]) / max(m["final_val_loss"], e["final_val_loss"])
    if gap >= 0.01:
        winner_lbl = min(cands, key=lambda l: RECORDS[l]["summary"]["final_val_loss"])
        reason = f"lower final val loss (gap {gap:.2%} >= 1%)"
    else:
        winner_lbl = sorted(cands, key=lambda l: (-RECORDS[l]["summary"]["mean_tokens_per_sec"],
                                                  RECORDS[l]["summary"]["peak_step_memory_mb"]))[0]
        reason = f"val losses tied (gap {gap:.2%} < 1%), so the higher-throughput variant wins"
WINNER = RECORDS[winner_lbl]["summary"]["variant"]
display(Markdown(f"## Selected variant: **`{WINNER}`** ({reason})"))
''')

code(r'''
fig, ax = plt.subplots(1, 4, figsize=(19, 3.8))
colors = {LBL_BASE: "k", LBL_MID: "tab:blue", LBL_EUL: "tab:orange"}
for lbl, c in colors.items():
    ev = RECORDS[lbl]["evals"]
    ax[0].plot(np.array(ev["tokens"]) / 1e6, ev["val_loss"], "o-", color=c, label=lbl, ms=3)
    ax[1].plot(np.array(ev["wall_clock_s"]) / 60, ev["val_loss"], "o-", color=c, label=lbl, ms=3)
ax[0].set(title="val loss vs tokens", xlabel="tokens (M)", ylim=(min(cmp3.final_val_loss) - 0.2, min(cmp3.final_val_loss) + 2.5))
ax[1].set(title="val loss vs wall clock", xlabel="minutes", ylim=ax[0].get_ylim())
ax[0].legend(fontsize=8)
ax[2].bar(cmp3.index, cmp3.peak_step_memory_mb, color=list(colors.values()))
ax[2].set(title="per-step peak memory (MB)"); ax[2].tick_params(axis="x", rotation=15)
ax[3].bar(cmp3.index, cmp3.mean_tokens_per_sec, color=list(colors.values()))
ax[3].set(title="throughput (tokens/s)"); ax[3].tick_params(axis="x", rotation=15)
fig.tight_layout(); plt.show()

b, m, e = (RECORDS[l]["summary"] for l in [LBL_BASE, LBL_MID, LBL_EUL])
display(Markdown(f"""
**Architecture comparison at the same batch size ({BATCH_X}):**

| | baseline | midpoint | Euler |
|---|---|---|---|
| final val loss | {b['final_val_loss']:.4f} | {m['final_val_loss']:.4f} ({m['final_val_loss'] - b['final_val_loss']:+.4f}) | {e['final_val_loss']:.4f} ({e['final_val_loss'] - b['final_val_loss']:+.4f}) |
| val accuracy | {b['final_val_accuracy'] * 100:.2f}% | {m['final_val_accuracy'] * 100:.2f}% | {e['final_val_accuracy'] * 100:.2f}% |
| tokens/s | {b['mean_tokens_per_sec']:,.0f} | {m['mean_tokens_per_sec']:,.0f} ({_pct(m['mean_tokens_per_sec'], b['mean_tokens_per_sec']):+.1f}%) | {e['mean_tokens_per_sec']:,.0f} ({_pct(e['mean_tokens_per_sec'], b['mean_tokens_per_sec']):+.1f}%) |
| backward ms | {b['mean_backward_ms'] or 0:.1f} | {m['mean_backward_ms'] or 0:.1f} | {e['mean_backward_ms'] or 0:.1f} |
| per-step peak MB | {b['peak_step_memory_mb']:,.0f} | {m['peak_step_memory_mb']:,.0f} ({_pct(m['peak_step_memory_mb'], b['peak_step_memory_mb']):+.1f}%) | {e['peak_step_memory_mb']:,.0f} ({_pct(e['peak_step_memory_mb'], b['peak_step_memory_mb']):+.1f}%) |
| saved activations MB/seq | {b['activation_saved_mb_per_sample']:.2f} | {m['activation_saved_mb_per_sample']:.2f} | {e['activation_saved_mb_per_sample']:.2f} |
| reconstruction drift | - | {(m.get('reversibility') or {}).get('max_rel_error', float('nan')):.2e} | {(e.get('reversibility') or {}).get('max_rel_error', float('nan')):.2e} |

The per-step peak falls less than saved activations do because the peak also contains
the transient (chunk x vocab) fused-loss buffers and one layer's worth of recomputation
during backward. Those are constant or O(1) in depth, but they do not disappear.
"""))
''')

# =============================================================================
# Scaling
# =============================================================================
md(r"""
---
## Experiments 4-6: scaling the selected reversible variant to `2x`, `4x`, `8x`

The same 50M tokens each time, so each doubling of the batch halves the optimizer steps. The LR is sqrt-scaled (Section 6). Things to look for:

* **Memory** should grow roughly linearly with batch but with a ~20x smaller slope than the baseline. `8x` should fit where the baseline cannot (Section 5 probe).
* **Throughput** should rise as larger matmuls use the GPU better, then saturate.
* **Loss** at a fixed token budget typically *worsens* past the critical batch size, since fewer, larger steps are less token-efficient. The reversible model gives the *option* of large batches; whether they help is a separate question, and this is where it gets answered.
""")

md(r"""
### Experiment 4: selected reversible variant, batch size `2x`
""")

code(r'''
LBL_2X = f"exp4_{WINNER}_2x"
run_or_load(WINNER, 2 * BATCH_X, LBL_2X, LR_SCALING_FOR_SCALE_UP)
report(LBL_2X, winner_lbl)
''')

md(r"""
### Experiment 5: selected reversible variant, batch size `4x`
""")

code(r'''
LBL_4X = f"exp5_{WINNER}_4x"
run_or_load(WINNER, 4 * BATCH_X, LBL_4X, LR_SCALING_FOR_SCALE_UP)
report(LBL_4X, winner_lbl)
''')

md(r"""
### Experiment 6: selected reversible variant, batch size `8x`
""")

code(r'''
LBL_8X = f"exp6_{WINNER}_8x"
run_or_load(WINNER, 8 * BATCH_X, LBL_8X, LR_SCALING_FOR_SCALE_UP)
report(LBL_8X, winner_lbl)
''')

code(r'''
scale_lbls = [winner_lbl, LBL_2X, LBL_4X, LBL_8X]
scale = comparison_frame(scale_lbls)
scale["peak_lr"] = [RECORDS[l]["summary"]["peak_lr"] for l in scale_lbls]
display(scale)

ok = scale[[RECORDS[l]["summary"]["status"] == "ok" for l in scale_lbls]]
fig, ax = plt.subplots(1, 4, figsize=(19, 3.8))
for lbl in ok.index:
    ev = RECORDS[lbl]["evals"]
    ax[0].plot(np.array(ev["tokens"]) / 1e6, ev["val_loss"], "o-", ms=3, label=f"batch {RECORDS[lbl]['summary']['batch_size']}")
ev = RECORDS[LBL_BASE]["evals"]
ax[0].plot(np.array(ev["tokens"]) / 1e6, ev["val_loss"], "k--", lw=1, label=f"baseline {BATCH_X}")
ax[0].set(title="val loss vs tokens", xlabel="tokens (M)", ylim=(ok.final_val_loss.min() - 0.2, ok.final_val_loss.min() + 2.5))
ax[0].legend(fontsize=8)
ax[1].plot(ok.batch_size, ok.final_val_loss, "o-"); ax[1].set(title="final val loss vs batch", xlabel="batch", xscale="log", xticks=list(ok.batch_size), xticklabels=list(ok.batch_size))
ax[2].plot(ok.batch_size, ok.mean_tokens_per_sec, "o-", color="tab:green"); ax[2].set(title="throughput vs batch", xlabel="batch", xscale="log")
ax[3].plot(ok.batch_size, ok.peak_step_memory_mb, "o-", color="tab:red", label=f"{WINNER}")
if len(probe_df):
    bp = probe_df[(probe_df.variant == "baseline") & probe_df.fits]
    ax[3].plot(bp.batch_size, bp.peak_memory_mb, "s--", color="k", label="baseline (probe)")
if GPU_TOTAL_MB:
    ax[3].axhline(GPU_TOTAL_MB, color="gray", ls=":", label="GPU memory")
ax[3].set(title="per-step peak memory vs batch (MB)", xlabel="batch", xscale="log"); ax[3].legend(fontsize=8)
fig.tight_layout(); plt.show()

rows = []
for lbl in scale_lbls:
    s = RECORDS[lbl]["summary"]
    if s["status"] != "ok":
        rows.append(f"| {s['batch_size']} | {s['total_steps']} | **{s['status']}** | | | | |")
        continue
    rows.append(f"| {s['batch_size']} | {s['total_steps']} | {s['final_val_loss']:.4f} | {s['final_val_accuracy'] * 100:.2f}% | "
                f"{s['mean_tokens_per_sec']:,.0f} | {s['peak_step_memory_mb']:,.0f} | {s['wall_clock_s'] / 60:.1f} |")
x0, x8 = RECORDS[winner_lbl]["summary"], RECORDS[LBL_8X]["summary"]
text = ["| batch | steps | val loss | val acc | tokens/s | peak MB | wall clock (min) |", "|---|---|---|---|---|---|---|", *rows, ""]
if x8["status"] == "ok":
    text.append(f"From `x` to `8x`: throughput **{_pct(x8['mean_tokens_per_sec'], x0['mean_tokens_per_sec']):+.1f}%**, "
                f"peak memory **{_pct(x8['peak_step_memory_mb'], x0['peak_step_memory_mb']):+.1f}%** "
                f"({x8['peak_step_memory_mb'] / 1024:.1f} GB), val loss **{x8['final_val_loss'] - x0['final_val_loss']:+.4f}** "
                f"with {x0['total_steps'] // max(1, x8['total_steps'])}x fewer optimizer steps. "
                f"Relative to the **baseline at x**, `8x` uses {_pct(x8['peak_step_memory_mb'], b['peak_step_memory_mb']):+.1f}% "
                f"peak memory for 8x the batch.")
display(Markdown("\n".join(text)))
''')

# =============================================================================
# Max batch
# =============================================================================
md(r"""
---
## Where is the maximum? Largest batch that fits: baseline vs selected reversible variant

The equivalent of the paper's Table 3 on *this* GPU: the largest batch (to the nearest 32) for which a real training step (forward, backward, AdamW) fits. The search doubles until OOM, then bisects. This establishes that `8x` really is within the reversible model's range and shows how much headroom remains beyond it.
""")

code(r'''
max_batch = {}
max_rows = []
if IS_CUDA:
    for variant in ["baseline", WINNER]:
        cfg = GPTConfig(vocab_size=data.vocab_size, block_size=BLOCK_SIZE, variant=variant,
                        loss_chunk_size=LOSS_CHUNK, **MODEL)
        start = BATCH_X if not SMOKE else 2
        limit = 16384 if not SMOKE else 256
        max_batch[variant], rows = find_max_batch(cfg, DEVICE, start=start, limit=limit,
                                                  granularity=32 if not SMOKE else 2)
        max_rows += rows
    max_df = pd.DataFrame(max_rows)
    display(max_df[["variant", "batch_size", "fits", "peak_memory_mb", "activation_saved_mb_per_sample", "tokens_per_sec"]])
    ratio = max_batch[WINNER] / max(1, max_batch["baseline"])
    display(Markdown(
        f"**Max batch on {ENV.get('gpu_name')} ({(GPU_TOTAL_MB or 0) / 1024:.0f} GB):** baseline **{max_batch['baseline']}**, "
        f"{WINNER} **{max_batch[WINNER]}**, which is **{ratio:.1f}x larger**. "
        f"Experiment 6 used {8 * BATCH_X} = {8 * BATCH_X / max(1, max_batch[WINNER]) * 100:.0f}% of the reversible maximum, "
        f"and {8 * BATCH_X / max(1, max_batch['baseline']):.1f}x the baseline's maximum. "
        f"(For reference, the paper reports ~10x on A100 and H100 for GPT-2 sized models; our model's 50k-vocab head is "
        f"a larger share of total memory, which caps the ratio.)"))
''')

# =============================================================================
# Final comparison
# =============================================================================
md(r"""
---
## Final comparison: all six experiments
""")

code(r'''
ALL = [LBL_BASE, LBL_MID, LBL_EUL, LBL_2X, LBL_4X, LBL_8X]
final_df = comparison_frame(ALL)
final_df.insert(0, "experiment", [1, 2, 3, 4, 5, 6])
final_df["status"] = [RECORDS[l]["summary"]["status"] for l in ALL]
display(final_df)
final_df.to_csv(RESULTS_DIR / "final_comparison.csv")

fig, ax = plt.subplots(1, 3, figsize=(18, 4))
for lbl in ALL:
    if RECORDS[lbl]["summary"]["status"] != "ok":
        continue
    ev = RECORDS[lbl]["evals"]
    ax[0].plot(np.array(ev["tokens"]) / 1e6, ev["val_loss"], "o-", ms=3, label=lbl)
    ax[1].plot(np.array(ev["wall_clock_s"]) / 60, ev["val_loss"], "o-", ms=3, label=lbl)
ok_all = final_df[final_df.status == "ok"]
for a, t, xl in [(ax[0], "val loss vs tokens", "tokens (M)"), (ax[1], "val loss vs wall clock", "minutes")]:
    a.set(title=t, xlabel=xl, ylim=(ok_all.final_val_loss.min() - 0.2, ok_all.final_val_loss.min() + 2.5))
ax[0].legend(fontsize=7)
sc = ax[2].scatter(ok_all.peak_step_memory_mb / 1024, ok_all.mean_tokens_per_sec, s=80,
                   c=ok_all.final_val_loss.astype(float), cmap="viridis_r")
for lbl, row in ok_all.iterrows():
    ax[2].annotate(lbl.split("_", 1)[1], (row.peak_step_memory_mb / 1024, row.mean_tokens_per_sec), fontsize=7,
                   xytext=(4, 4), textcoords="offset points")
plt.colorbar(sc, ax=ax[2], label="final val loss")
ax[2].set(title="throughput vs peak memory", xlabel="peak memory (GB)", ylabel="tokens/s")
fig.tight_layout(); plt.show()
''')

md(r"""
## Conclusions

The summary below is generated from the measured numbers above, so it stays in sync with whatever this notebook actually ran.
""")

code(r'''
S = {l: RECORDS[l]["summary"] for l in ALL}
b, w = S[LBL_BASE], S[winner_lbl]
loser_lbl = LBL_EUL if winner_lbl == LBL_MID else LBL_MID
lo = S[loser_lbl]
ok_runs = [l for l in ALL if S[l]["status"] == "ok"]
best = min(ok_runs, key=lambda l: S[l]["final_val_loss"])
fastest = max(ok_runs, key=lambda l: S[l]["mean_tokens_per_sec"])
lines = [
    "### 1. Architecture (same batch, same data)",
    f"- **Memory:** reversible `{WINNER}` saved **{b['activation_saved_mb_per_sample'] / w['activation_saved_mb_per_sample']:.0f}x less** activation memory "
    f"per sequence than the baseline ({w['activation_saved_mb_per_sample']:.2f} vs {b['activation_saved_mb_per_sample']:.2f} MB), and per-step peak memory changed by "
    f"**{_pct(w['peak_step_memory_mb'], b['peak_step_memory_mb']):+.1f}%**.",
    f"- **Speed:** throughput changed by **{_pct(w['mean_tokens_per_sec'], b['mean_tokens_per_sec']):+.1f}%** "
    f"({w['mean_tokens_per_sec']:,.0f} vs {b['mean_tokens_per_sec']:,.0f} tokens/s). This is the cost of reconstructing activations in backward, "
    f"compared with the ~{(flops_rows['midpoint']['recompute overhead']) * 100:.0f}% extra FLOPs predicted analytically.",
    f"- **Quality:** val loss {w['final_val_loss']:.4f} vs baseline {b['final_val_loss']:.4f} (**{w['final_val_loss'] - b['final_val_loss']:+.4f}**); "
    f"the other reversible variant reached {lo['final_val_loss']:.4f}.",
    f"- **Selection:** `{WINNER}`, because {reason}.",
    "",
    "### 2. Batch-size scaling of the selected variant",
]
for l in [winner_lbl, LBL_2X, LBL_4X, LBL_8X]:
    s = S[l]
    if s["status"] == "ok":
        lines.append(f"- batch **{s['batch_size']}**: val loss {s['final_val_loss']:.4f}, {s['mean_tokens_per_sec']:,.0f} tokens/s, "
                     f"peak {s['peak_step_memory_mb'] / 1024:.1f} GB, {s['wall_clock_s'] / 60:.1f} min")
    else:
        lines.append(f"- batch **{s['batch_size']}**: **{s['status']}**")
if max_batch:
    lines.append(f"- Largest trainable batch on this GPU: baseline **{max_batch['baseline']}** vs {WINNER} **{max_batch[WINNER]}** "
                 f"(**{max_batch[WINNER] / max(1, max_batch['baseline']):.1f}x**).")
lines += ["", "### 3. Overall",
          f"- Lowest val loss: `{best}` ({S[best]['final_val_loss']:.4f}); highest throughput: `{fastest}` ({S[fastest]['mean_tokens_per_sec']:,.0f} tokens/s)."]
display(Markdown("\n".join(lines)))
''')

md(r"""
**How to read these results**

* **Reversibility is a memory tool.** It trades ~14% extra compute (one extra backbone forward) for activation memory that is constant in depth. For a 9-layer, 20M-parameter model that already fits comfortably, that trade only pays off if the freed memory buys something, such as a larger batch (which can raise throughput by using the GPU better) or, at real scale, a deeper or longer-context model that otherwise would not fit at all. The paper's throughput *gains* come from exactly this: larger batches on memory-bound deep models (96 layers).
* **At a fixed token budget, bigger batches mean fewer optimizer steps.** If loss worsens from `x` to `8x`, the runs have passed the critical batch size for this model and budget. That is an optimization effect, not a flaw in the reversible model. The memory headroom lets you choose the batch, but it does not make large batches token-efficient.
* **Exact reversal matters.** With plain floating-point streams, reconstruction error grows as training sharpens the layers (Section 4.4), and the model is trained on subtly wrong gradients. Fixed-point streams remove the issue entirely for 2x the (small) stream memory. The paper does not discuss this; it is the implementation detail most worth keeping.
""")

# =============================================================================
# Artifacts
# =============================================================================
md(r"""
## Artifacts and reproducibility
""")

code(r'''
# Smoke runs must never overwrite the real webapp data.
webapp_js = (RESULTS_DIR / "webapp_data.js") if SMOKE else (ROOT.parent.parent / "webapp" / "data.js")
session_data = export_webapp_data(RESULTS_DIR / "runs.jsonl", webapp_js, winner=WINNER)
print("webapp data written to", webapp_js)
files = sorted(p.relative_to(ROOT) for d in [RESULTS_DIR, LOG_DIR] for p in d.glob("*") if p.is_file())
display(pd.DataFrame({"file": [str(f) for f in files],
                      "size_kb": [round((ROOT / f).stat().st_size / 1024, 1) for f in files]}))
# Aim writes every metric while training, but finished runs only show up in queries
# (and immediately in the UI) once the repo index is rebuilt. `aim up` also does this.
import subprocess, sys
aim_bin = Path(sys.executable).parent / "aim"
res = subprocess.run([str(aim_bin), "storage", "--repo", AIM_REPO, "reindex", "--yes"], capture_output=True, text=True)
print("aim reindex:", "ok" if res.returncode == 0 else res.stderr[-500:])
print("Aim: run `uv run aim up` from", ROOT, "| experiment:", EXPERIMENT)
print("Aim run hashes:", {l: RECORDS[l]["summary"].get("aim_run_hash") for l in ALL})
''')

md(r"""
* `results/<run>.json`: config, environment, summary, per-step history, eval curve and diagnostics for one run.
* `results/runs.jsonl`: one summary line per run (appended; `export.py` keeps the latest per label).
* `results/final_comparison.csv`: the six-experiment table above.
* `logs/<run>.log` and `logs/<run>.metrics.jsonl`: the text log and the streaming per-step metrics.
* Aim: every per-step and per-eval metric, grouped by `subset` context (`train`, `eval_train`, `eval_val`), plus hparams, summary and Aim's GPU system metrics.
* `uv run pytest -q`: gradient-exactness tests for both reversible schemes, fused-loss exactness, a bf16 autocast gradient check, and the O(1)-memory-in-depth check.
""")

for cell_type, source in cells:
    if cell_type == "markdown":
        nb["cells"].append(nbf.v4.new_markdown_cell(source))
    else:
        nb["cells"].append(nbf.v4.new_code_cell(source))

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(NOTEBOOK_PATH))
print(f"wrote {len(nb['cells'])} cells to {NOTEBOOK_PATH}")
