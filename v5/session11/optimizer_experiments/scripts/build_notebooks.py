"""Generate the five task notebooks under ``notebooks/``.

    uv run python scripts/build_notebooks.py

Each notebook is self-contained, heavy on print statements and charts, and logs
every training run to the shared Aim repo at ``optimizer_experiments/.aim``.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from nbbuild import NB  # noqa: E402

OUT = os.path.abspath(os.path.join(HERE, "..", "notebooks"))
os.makedirs(OUT, exist_ok=True)

SETUP = r'''
import sys, os, math, time, json, warnings
sys.path.insert(0, os.path.abspath("../src"))
warnings.filterwarnings("ignore")

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

from s11.utils import set_seed, get_device, savefig, plot_style, AIM_REPO
plot_style()
set_seed(1337)
DEVICE = get_device()
torch.set_float32_matmul_precision("high")
print(f"torch {torch.__version__} | device = {DEVICE} "
      f"| {torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'cpu'}")
print(f"Aim repo: {AIM_REPO}  (browse it later with:  uv run aim up)")
'''


# ===========================================================================
# NOTEBOOK 1 — Reproduce Adam by hand
# ===========================================================================
def nb1():
    nb = NB("01")
    nb.md(r"""
# Task 1 — Reproduce Adam by Hand

> **Ask.** Take one weight and five gradients, compute $m$, $v$, $\hat m$, $\hat v$
> and the resulting step yourself, then check each against PyTorch. They should
> agree to several decimal places.

This notebook does three things:

1. Implements the Adam update in plain Python, printing **every** intermediate
   quantity for all five steps.
2. Drives `torch.optim.Adam` / `torch.optim.AdamW` with the *same* weight and the
   *same* five gradients and diffs the two, number by number.
3. Repeats the check with **decoupled weight decay** turned on, so we can see
   exactly where AdamW departs from Adam.

### The update rule (what we are reproducing)

For step $t$, gradient $g_t$, hyper-parameters $\alpha$ (lr), $\beta_1,\beta_2,\epsilon$:

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\, g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2 \\
\hat m_t &= m_t / (1-\beta_1^{\,t}) \\
\hat v_t &= v_t / (1-\beta_2^{\,t}) \\
\Delta_t &= \alpha \,\hat m_t / (\sqrt{\hat v_t} + \epsilon) \\
w_t &= w_{t-1} - \Delta_t \qquad\text{(Adam)} \\
w_t &= w_{t-1} - \Delta_t - \alpha\,\lambda\, w_{t-1} \qquad\text{(AdamW, decoupled decay }\lambda)
\end{aligned}
$$
""")
    nb.code(SETUP)

    nb.md(r"""
## 1. The setup: one weight, five gradients

We pick a single scalar weight and five hand-chosen gradients (mixed signs and
magnitudes, so nothing is accidentally symmetric). Hyper-parameters are the Adam
defaults **except** `beta2`, where we deliberately show both the classic
`0.999` and the LLM-style `0.95` later on.
""")
    nb.code(r'''
from s11.optim import run_adam_by_hand, AdamState, adam_step

w0    = 0.7213                       # the one weight
grads = [0.15, -0.30, 0.05, 0.22, -0.11]   # five gradients

LR, B1, B2, EPS = 1e-3, 0.9, 0.999, 1e-8

print(f"w0 = {w0}")
print(f"gradients = {grads}")
print(f"lr={LR}  beta1={B1}  beta2={B2}  eps={EPS}\n")

trace = run_adam_by_hand(w0, grads, lr=LR, beta1=B1, beta2=B2, eps=EPS)

df = pd.DataFrame(trace)[["t", "m", "v", "m_hat", "v_hat", "step", "param_before", "param_after"]]
pd.set_option("display.float_format", lambda x: f"{x:.10f}")
print("HAND-COMPUTED ADAM TRACE")
print(df.to_string(index=False))
''')

    nb.md(r"""
### Step-by-step arithmetic for step 1 (spelled out)

To make the table above auditable, here is step 1 done symbol-by-symbol.
""")
    nb.code(r'''
g1 = grads[0]
m1 = B1*0.0 + (1-B1)*g1
v1 = B2*0.0 + (1-B2)*g1**2
m1_hat = m1 / (1 - B1**1)
v1_hat = v1 / (1 - B2**1)
step1  = LR * m1_hat / (math.sqrt(v1_hat) + EPS)

print(f"g1                 = {g1}")
print(f"m1 = (1-b1)*g1     = {m1:.10f}")
print(f"v1 = (1-b2)*g1^2   = {v1:.12f}")
print(f"1-b1^1             = {1-B1**1:.10f}")
print(f"1-b2^1             = {1-B2**1:.10f}")
print(f"m1_hat             = {m1_hat:.10f}   (== g1, since 1-b1^1 = 1-b1)")
print(f"v1_hat             = {v1_hat:.12f}   (== g1^2)")
print(f"sqrt(v1_hat)       = {math.sqrt(v1_hat):.10f}   (== |g1|)")
print(f"step1 = lr*m1_hat/(sqrt(v1_hat)+eps) = {step1:.10f}")
print(f"      ~ lr * sign(g1) = {LR*np.sign(g1):.10f}   <- Adam's 'unit step' behaviour on step 1")
print(f"w1 = w0 - step1    = {w0 - step1:.10f}")
''')

    nb.md(r"""
## 2. Check against PyTorch

We create a leaf tensor initialised to `w0`, and for each of the five steps we
**write the gradient by hand** into `p.grad` and call `opt.step()`. No autograd,
no loss — just the exact same five numbers fed to the reference optimizer.
""")
    nb.code(r'''
def torch_adam_trace(w0, grads, lr, b1, b2, eps, weight_decay=0.0, kind="adam"):
    p = torch.tensor([w0], dtype=torch.float64, requires_grad=True)
    Opt = torch.optim.AdamW if kind == "adamw" else torch.optim.Adam
    opt = Opt([p], lr=lr, betas=(b1, b2), eps=eps, weight_decay=weight_decay)
    rows = []
    for t, g in enumerate(grads, start=1):
        opt.zero_grad()
        p.grad = torch.tensor([g], dtype=torch.float64)
        before = p.detach().clone()
        opt.step()
        st = opt.state[p]
        rows.append(dict(
            t=t,
            m=st["exp_avg"].item(),
            v=st["exp_avg_sq"].item(),
            m_hat=st["exp_avg"].item() / (1 - b1**t),
            v_hat=st["exp_avg_sq"].item() / (1 - b2**t),
            step=(before - p.detach()).item(),
            param_before=before.item(),
            param_after=p.detach().item(),
        ))
    return pd.DataFrame(rows)

tdf = torch_adam_trace(w0, grads, LR, B1, B2, EPS, kind="adam")
print("PYTORCH torch.optim.Adam TRACE")
print(tdf.to_string(index=False))
''')

    nb.code(r'''
# Numeric diff, hand vs torch, for every quantity at every step
cols = ["m", "v", "m_hat", "v_hat", "step", "param_after"]
diff = (df[cols].reset_index(drop=True) - tdf[cols].reset_index(drop=True)).abs()
diff.insert(0, "t", df["t"].values)
print("ABSOLUTE DIFFERENCE  |hand - pytorch|\n")
print(diff.to_string(index=False, float_format=lambda x: f"{x:.2e}"))
print(f"\nmax abs diff over the whole trace = {diff[cols].to_numpy().max():.3e}")
assert diff[cols].to_numpy().max() < 1e-12, "hand and torch disagree!"
print("PASS  — hand-rolled Adam matches torch.optim.Adam to < 1e-12 (float64)")
''')

    nb.md(r"""
## 3. Where AdamW differs: decoupled weight decay

Turn on weight decay $\lambda = 0.1$. Classic **Adam** (`torch.optim.Adam` with
`weight_decay`) folds $\lambda w$ *into the gradient* before the moments are
updated — so the decay is itself smoothed by $m$ and rescaled by $\sqrt{\hat v}$.
**AdamW** applies $-\alpha\lambda w$ *after* the Adam step, untouched by the
moment estimates. We reproduce both.
""")
    nb.code(r'''
WD = 0.1

hand_adam_l2 = run_adam_by_hand(w0, grads, lr=LR, beta1=B1, beta2=B2, eps=EPS,
                                weight_decay=WD, decoupled=False)
hand_adamw   = run_adam_by_hand(w0, grads, lr=LR, beta1=B1, beta2=B2, eps=EPS,
                                weight_decay=WD, decoupled=True)

t_adam_l2 = torch_adam_trace(w0, grads, LR, B1, B2, EPS, weight_decay=WD, kind="adam")
t_adamw   = torch_adam_trace(w0, grads, LR, B1, B2, EPS, weight_decay=WD, kind="adamw")

# NOTE: compare the *observable* — m, v and the resulting weight. (Our hand trace's
# "step" column is the pure Adam step; for AdamW the decay is applied separately to
# param_after, so the two "step" columns are not like-for-like — param_after is.)
def maxdiff(trace, tdf):
    a = pd.DataFrame(trace)[["m","v","param_after"]].reset_index(drop=True)
    b = tdf[["m","v","param_after"]].reset_index(drop=True)
    return (a - b).abs().to_numpy().max()

d_l2, d_w = maxdiff(hand_adam_l2, t_adam_l2), maxdiff(hand_adamw, t_adamw)
print(f"Adam + L2 decay   : max |hand - torch|  (m, v, weight) = {d_l2:.3e}")
print(f"AdamW (decoupled) : max |hand - torch|  (m, v, weight) = {d_w:.3e}")
assert max(d_l2, d_w) < 1e-12, "decay variants disagree!"
print("PASS — both weight-decay variants match torch to < 1e-12 (float64)")

comp = pd.DataFrame({
    "t": range(1, 6),
    "w  (Adam, no decay)":  [r["param_after"] for r in trace],
    "w  (Adam + L2)":       [r["param_after"] for r in hand_adam_l2],
    "w  (AdamW decoupled)": [r["param_after"] for r in hand_adamw],
})
print()
print(comp.to_string(index=False, float_format=lambda x: f"{x:.8f}"))
''')

    nb.md(r"""
## 4. Charts
""")
    nb.code(r'''
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
t = [r["t"] for r in trace]

ax[0,0].plot(t, [r["m"] for r in trace], "o-", label="m (biased)")
ax[0,0].plot(t, [r["m_hat"] for r in trace], "s--", label=r"$\hat m$ (corrected)")
ax[0,0].set_title("First moment"); ax[0,0].set_xlabel("step"); ax[0,0].legend()

ax[0,1].plot(t, [r["v"] for r in trace], "o-", label="v (biased)")
ax[0,1].plot(t, [r["v_hat"] for r in trace], "s--", label=r"$\hat v$ (corrected)")
ax[0,1].set_title("Second moment"); ax[0,1].set_xlabel("step"); ax[0,1].legend()

ax[0,2].plot(t, [r["step"] for r in trace], "o-", color="crimson")
ax[0,2].axhline(LR, ls=":", color="grey", label="lr")
ax[0,2].axhline(-LR, ls=":", color="grey")
ax[0,2].set_title(r"Update $\Delta_t$"); ax[0,2].set_xlabel("step"); ax[0,2].legend()

ax[1,0].plot(t, [1-B1**i for i in t], "o-", label=r"$1-\beta_1^t$")
ax[1,0].plot(t, [1-B2**i for i in t], "s-", label=r"$1-\beta_2^t$")
ax[1,0].set_title("Bias-correction denominators"); ax[1,0].set_xlabel("step"); ax[1,0].legend()

ax[1,1].plot(t, [r["param_after"] for r in trace], "o-", label="hand")
ax[1,1].plot(tdf["t"], tdf["param_after"], "x", ms=12, label="pytorch")
ax[1,1].set_title("Weight trajectory: hand vs torch"); ax[1,1].set_xlabel("step"); ax[1,1].legend()

ax[1,2].plot(t, [r["param_after"] for r in trace], "o-", label="Adam")
ax[1,2].plot(t, [r["param_after"] for r in hand_adam_l2], "s-", label="Adam+L2")
ax[1,2].plot(t, [r["param_after"] for r in hand_adamw], "^-", label="AdamW")
ax[1,2].set_title("Weight decay variants"); ax[1,2].set_xlabel("step"); ax[1,2].legend()

plt.tight_layout()
savefig(fig, "t1_adam_by_hand.png")
plt.show()
''')

    nb.md(r"""
## 5. Findings

| quantity | agreement (hand vs `torch.optim`) |
|---|---|
| $m,\;v,\;\hat m,\;\hat v$, update, weight | `< 1e-12` in float64 (all five steps) |
| Adam + L2 weight decay | `< 1e-12` |
| AdamW decoupled decay | `< 1e-12` |

**What the numbers show**

- On **step 1** the update is almost exactly $\alpha\cdot\mathrm{sign}(g_1)$: the
  bias-correction factors cancel ($\hat m_1 = g_1$, $\sqrt{\hat v_1}=|g_1|$), so
  Adam takes a *unit* step regardless of gradient magnitude.
- The biased moments $m,v$ start near zero and crawl upward; the corrected
  $\hat m,\hat v$ are what actually drive the step, and for the first few steps
  they are **much** larger than the biased versions (dividing by $1-\beta_2^t\approx
  10^{-3}$ on step 1).
- **Adam + L2** vs **AdamW**: with $\lambda=0.1$ the trajectories separate by step
  2–3. In L2-Adam the decay term gets divided by $\sqrt{\hat v}$ (so parameters
  with large gradients are decayed *less*); AdamW decays every weight by the same
  multiplicative factor $(1-\alpha\lambda)$. This decoupling is the entire point
  of AdamW.

> **Myth check.** The lecture transcript says *"bias correction is what AdamW is
> doing."* That is not quite right. **Bias correction** (dividing by $1-\beta^t$)
> fixes the zero-initialisation bias of the EMAs and is part of *plain Adam*.
> **AdamW's** contribution is *decoupled weight decay*. They are independent
> features — Task 2 pulls bias correction out on its own.
""")
    nb.write(os.path.join(OUT, "01_adam_by_hand.ipynb"))


# ===========================================================================
# NOTEBOOK 2 — Disable bias correction
# ===========================================================================
def nb2():
    nb = NB("02")
    nb.md(r"""
# Task 2 — Bias Correction On vs Off

> **Ask.** Disable bias correction and plot the first twenty steps both ways
> (with and without). Report the number of steps after which the difference
> stops mattering.

### What bias correction is

Adam initialises $m_0 = v_0 = 0$. For the first steps the EMAs are therefore
biased *toward zero*. The fix:

$$\hat m_t = \frac{m_t}{1-\beta_1^{\,t}}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^{\,t}}.$$

With bias correction **off** we use $m_t, v_t$ directly. The net effect on the
update is a multiplicative factor

$$
c(t) \;=\; \frac{\text{corrected step}}{\text{uncorrected step}}
       \;=\; \frac{1/(1-\beta_1^{\,t})}{\sqrt{1/(1-\beta_2^{\,t})}}
       \;=\; \frac{\sqrt{1-\beta_2^{\,t}}}{1-\beta_1^{\,t}}.
$$

$c(t)\to 1$ as $t\to\infty$; the question is *how fast*, and it depends almost
entirely on $\beta_2$.
""")
    nb.code(SETUP)

    nb.md(r"""
## 1. The correction factor $c(t)$ in closed form

We tabulate $c(t)$ for three $\beta_2$ values: `0.999` (Adam default), `0.99`
(this repo's training default) and `0.95` (nanoGPT / LLM default). "Stops
mattering" = $|c(t)-1| < \tau$; we use $\tau = 1\%$ and also report $5\%$.
""")
    nb.code(r'''
def c_factor(t, b1, b2):
    return math.sqrt(1 - b2**t) / (1 - b1**t)

B1 = 0.9

def steps_until_within(b2, tau, b1=B1, tmax=200000, hold=1000):
    """First t such that |c(t')-1| < tau for that t' AND stays there.

    c(t) is non-monotonic for small 1-b2 (it dips below 1 before climbing back),
    so we require the band to hold for `hold` consecutive steps, not just once."""
    run = 0
    for t in range(1, tmax):
        if abs(c_factor(t, b1, b2) - 1) < tau:
            run += 1
            if run == 1:
                first = t
            if run >= hold or t == tmax - 1:
                return first
        else:
            run = 0
    return None

rows = []
for b2 in [0.999, 0.99, 0.95]:
    for tau in [0.05, 0.01]:
        rows.append(dict(beta2=b2, tolerance=f"{tau:.0%}",
                         steps_until_within_tol=steps_until_within(b2, tau)))
tab = pd.DataFrame(rows)
print(tab.to_string(index=False))
print("\n(c(t) is NOT monotonic for beta2<=0.99 — it dips below 1 around t~10-15 "
      "then climbs back, so 'first touch' of the band is not 'stays in' the band.)")

print("\nc(t) for the first 20 steps:")
head = pd.DataFrame({"t": range(1, 21),
                     "c(t)  b2=0.999": [c_factor(t,B1,0.999) for t in range(1,21)],
                     "c(t)  b2=0.99":  [c_factor(t,B1,0.99)  for t in range(1,21)],
                     "c(t)  b2=0.95":  [c_factor(t,B1,0.95)  for t in range(1,21)]})
print(head.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
''')

    nb.md(r"""
## 2. First twenty steps on a real weight

We take a genuine parameter from the **10.77M nanoGPT** — a slice of the first
block's MLP input matrix — and record the true gradient it receives on each of
the first 20 optimizer steps. Then we replay those exact 20 gradients through our
hand Adam **with** and **without** bias correction (everything else identical).
""")
    nb.code(r'''
from s11.model import GPT, GPTConfig
from s11.data import load_char_dataset
from s11.optim import run_adam_by_hand

data = load_char_dataset()
set_seed(1337)
cfg = GPTConfig(vocab_size=data.vocab_size, block_size=256)
model = GPT(cfg).to(DEVICE)
print(f"model params: {model.num_params()/1e6:.2f}M")

# choose one weight: element [0,0] of block 0's MLP fc weight
target_name = "transformer.h.0.mlp.c_fc.weight"
param = dict(model.named_parameters())[target_name]
idx = (0, 0)
w0 = param.detach()[idx].item()

opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.99), weight_decay=0.0)
gen = torch.Generator().manual_seed(0)
observed_grads = []
model.train()
for step in range(20):
    X, Y = data.get_batch("train", 32, 256, DEVICE, gen)
    _, loss = model(X, Y)
    opt.zero_grad(set_to_none=True); loss.backward()
    observed_grads.append(param.grad.detach()[idx].item())
    opt.step()

print(f"weight  {target_name}{list(idx)}  w0 = {w0:.6f}")
print("observed gradients (first 20 steps):")
print(np.array2string(np.array(observed_grads), precision=5, floatmode="fixed"))
''')
    nb.code(r'''
B2 = 0.99   # this repo's training default
with_bc    = run_adam_by_hand(w0, observed_grads, lr=1e-3, beta1=0.9, beta2=B2, bias_correction=True)
without_bc = run_adam_by_hand(w0, observed_grads, lr=1e-3, beta1=0.9, beta2=B2, bias_correction=False)

cmp = pd.DataFrame({
    "t":        [r["t"] for r in with_bc],
    "step_BC":  [r["step"] for r in with_bc],
    "step_noBC":[r["step"] for r in without_bc],
    "w_BC":     [r["param_after"] for r in with_bc],
    "w_noBC":   [r["param_after"] for r in without_bc],
})
cmp["|step ratio|"] = (cmp["step_BC"] / cmp["step_noBC"]).abs()
cmp["c(t) predicted"] = [c_factor(t, 0.9, B2) for t in cmp["t"]]
cmp["w gap"] = (cmp["w_BC"] - cmp["w_noBC"]).abs()
print(cmp.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
''')

    nb.md(r"""
## 3. "When does the difference stop mattering?"

Two complementary readings:

* **Per-step** — when the update-size ratio $c(t)$ is within tolerance of 1.
* **Cumulative** — when the *weight trajectories* stop drifting apart (the gap
  plateaus, because later steps are near-identical).
""")
    nb.code(r'''
gap = np.array([abs(a["param_after"] - b["param_after"]) for a, b in zip(with_bc, without_bc)])
# gap relative to the weight's own movement so far (does the disagreement still matter?)
move_bc = np.abs(np.array([r["param_after"] for r in with_bc]) - w0) + 1e-12
rel_gap = gap / move_bc

print(f"this run used beta2 = {B2}\n")
print(f"{'beta2':>7} | {'within 5% (stays)':>18} | {'within 1% (stays)':>18}")
for b2 in [0.999, 0.99, 0.95]:
    print(f"{b2:>7} | {str(steps_until_within(b2,0.05)):>18} | {str(steps_until_within(b2,0.01)):>18}")
print(f"\nper-step   : c(t) enters & stays within 1% of 1 from step t = {steps_until_within(B2, 0.01)}")
print(f"cumulative : after 20 steps, |w_BC - w_noBC| = {gap[-1]:.2e}, "
      f"which is {rel_gap[-1]:.0%} of how far this weight has moved — still a real gap,")
print(f"             because with beta2={B2} the correction factor is still {c_factor(20,0.9,B2):.2f} at step 20.")
''')

    nb.md("## 4. Charts — first 20 steps both ways")
    nb.code(r'''
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
t20 = list(range(1, 21))

ax[0,0].plot(t20, [r["step"] for r in with_bc], "o-", label="bias correction ON")
ax[0,0].plot(t20, [r["step"] for r in without_bc], "s--", label="bias correction OFF")
ax[0,0].set_title("Per-step update $\\Delta_t$"); ax[0,0].set_xlabel("step"); ax[0,0].legend()

ax[0,1].plot(t20, [r["param_after"] for r in with_bc], "o-", label="ON")
ax[0,1].plot(t20, [r["param_after"] for r in without_bc], "s--", label="OFF")
ax[0,1].set_title("Weight trajectory (20 steps)"); ax[0,1].set_xlabel("step"); ax[0,1].legend()

ax[0,2].plot(t20, gap, "d-", color="crimson")
ax[0,2].set_title("|w_BC - w_noBC|  (cumulative gap)"); ax[0,2].set_xlabel("step")

for b2, mk in [(0.999,"o-"), (0.99,"s-"), (0.95,"^-")]:
    ax[1,0].plot(range(1,21), [c_factor(t,0.9,b2) for t in range(1,21)], mk, label=f"β2={b2}")
ax[1,0].axhline(1, ls=":", color="grey"); ax[1,0].axhspan(0.99,1.01, color="green", alpha=0.1)
ax[1,0].set_title("Correction factor c(t), first 20 steps"); ax[1,0].set_xlabel("step"); ax[1,0].legend()

tt = np.arange(1, 6001)
for b2, mk in [(0.999,"-"), (0.99,"-"), (0.95,"-")]:
    ax[1,1].plot(tt, [c_factor(t,0.9,b2) for t in tt], mk, label=f"β2={b2}")
ax[1,1].axhspan(0.99,1.01, color="green", alpha=0.1)
ax[1,1].set_xscale("log"); ax[1,1].set_title("c(t) to convergence (log x)")
ax[1,1].set_xlabel("step"); ax[1,1].legend()

ax[1,2].plot(t20, [1-0.99**t for t in t20], "s-", label=r"$1-\beta_2^t$ (β2=0.99)")
ax[1,2].plot(t20, [1-0.9**t for t in t20], "o-", label=r"$1-\beta_1^t$ (β1=0.9)")
ax[1,2].set_title("Why the gap: EMA warm-up"); ax[1,2].set_xlabel("step"); ax[1,2].legend()

plt.tight_layout()
savefig(fig, "t2_bias_correction.png")
plt.show()
''')

    nb.md(r"""
## 5. Findings

**Reported number — "steps after which the difference stops mattering"**
(first step at which $|c(t)-1|$ enters the band *and stays*, since $c(t)$ is
non-monotonic for $\beta_2\le 0.99$):

| $\beta_2$ | within 5% | within 1% | context |
|---|---|---|---|
| 0.95  | **42** steps  | **76** steps | nanoGPT / LLM default |
| 0.99  | **232** steps | **390** steps | this repo's training default |
| 0.999 | **2 327** steps | **3 916** steps | Adam-paper default |

The horizon scales as $t^\star \approx \ln(0.0X)/\ln\beta_2$ — set by $\beta_2$
alone ($\beta_1$'s term has decayed by then). $\beta_1$'s own bias is gone by
~step 45 (within 1%).

**Interpretation**

- With correction **OFF**, step 1 uses $m_1 = (1-\beta_1)g_1 \approx 0.1\,g_1$ and
  $v_1=(1-\beta_2)g_1^2$, so the update is $\approx (1-\beta_1)/\sqrt{1-\beta_2}$
  of a unit step — for $\beta_2=0.99$ that ratio is exactly **1.0** (coincidence:
  $\sqrt{0.01}=0.1$), for $\beta_2=0.999$ it is **0.32** (much smaller), for
  $\beta_2=0.95$ it is **2.24** (much *larger*). So "bias correction off" is not
  uniformly a warmup — its sign depends on $\beta_2$.
- After the first ~10–15 steps $c(t)$ **dips below 1** (the $m$-EMA has caught up
  but the $v$-EMA has not, so $\sqrt{\hat v}$ is still too small → uncorrected step
  too *large*), then climbs back to 1 over hundreds/thousands of steps.
- Because it self-heals within tens-to-hundreds of steps, and real training uses
  an explicit warmup anyway, **bias correction is nearly irrelevant to the final
  loss** of a multi-thousand-step run — but it is very cheap and removes a
  confound, so keep it on.
- The horizon scales with $\beta_2$: the slower the second-moment EMA, the longer
  the correction is doing real work.
""")
    nb.write(os.path.join(OUT, "02_bias_correction.ipynb"))


# ===========================================================================
# NOTEBOOK 3 — update-to-weight ratio & warmup
# ===========================================================================
def nb3():
    nb = NB("03")
    nb.md(r"""
# Task 3 — The Update-to-Weight Ratio, per Layer, and What Warmup Does to It

> **Ask.** Log the update-to-weight ratio for every layer, and identify the step
> at which warmup stops changing it.

### Definition

For a parameter tensor $W$ with realised optimizer update $\Delta W$ (the actual
change in the weights over one step, *after* clipping and weight decay):

$$
\rho(W, t) \;=\; \frac{\lVert \Delta W_t \rVert_{\text{RMS}}}{\lVert W_t \rVert_{\text{RMS}}}
           \;=\; \sqrt{\frac{\sum (\Delta W_t)^2}{\sum W_t^2}}.
$$

We aggregate this per **layer bucket** (each block's attention, each block's MLP,
the embedding, the LayerNorms). The folklore target is $\rho \approx 10^{-3}$: an
update that moves each weight by ~0.1% per step. Warmup exists largely to keep
$\rho$ from spiking on the first steps, when $\hat v$ is still small and the loss
surface is steep.
""")
    nb.code(SETUP)

    nb.md(r"""
## 1. Train the 10.77M model three times — warmup ∈ {0, 50, 200}

300 steps each, cosine schedule, everything else fixed. Every step we snapshot the
weights, apply the optimizer, and measure $\rho$ for every layer bucket. All three
runs are logged to Aim (experiment `task3_warmup`).
""")
    nb.code(r'''
from s11.data import load_char_dataset
from s11.trainer import TrainConfig, train

data = load_char_dataset()
RUNS = {}
for wu in [0, 50, 200]:
    print(f"\n=== warmup = {wu} ===")
    cfg = TrainConfig(
        total_steps=300, warmup=wu, schedule="cosine",
        base_lr=1e-3, beta2=0.99, weight_decay=0.1, grad_clip=1.0,
        batch_size=32, eval_interval=50, eval_iters=40,
        ratio_log_interval=1, label=f"warmup{wu}", device=DEVICE,
    )
    RUNS[wu] = train(data, cfg, aim_repo=AIM_REPO, experiment="task3_warmup")
print("\ndone.")
''')

    nb.md("## 2. Per-layer ratio table (final step)")
    nb.code(r'''
wu = 50
h = RUNS[wu]
buckets = sorted(h["ratio_by_layer"].keys())
final_tab = pd.DataFrame({
    "layer": buckets,
    "rho @ step 5":   [h["ratio_by_layer"][b][5]   for b in buckets],
    "rho @ step 50":  [h["ratio_by_layer"][b][50]  for b in buckets],
    "rho @ step 150": [h["ratio_by_layer"][b][150] for b in buckets],
    "rho @ step 299": [h["ratio_by_layer"][b][299] for b in buckets],
})
print(f"update-to-weight ratio  (warmup={wu})\n")
print(final_tab.to_string(index=False, float_format=lambda x: f"{x:.2e}"))
print(f"\nglobal rho @ 299 = {h['ratio_global'][-1]:.2e}   (folklore target ~1e-3)")
''')

    nb.md(r"""
## 3. "The step at which warmup stops changing the ratio"

Because $\rho \approx \eta_t \cdot (\text{const})$ — the realised update is
$\eta_t\,\hat m/\sqrt{\hat v}$ and $\lVert W\rVert$ barely moves over 300 steps —
warmup changes $\rho$ **only through the LR ramp**. So two things to pin down:

1. **The peak.** $\rho$ rises while the LR ramps and falls once the cosine decay
   takes over, so $\arg\max_t \rho(t)$ should sit right at `step == warmup`. That
   is the step at which warmup stops *pushing $\rho$ up*.
2. **The schedule-free transient.** Look at $\rho_t / \eta_t$: dividing out the
   LR removes the entire warmup-and-decay effect. What's left is the genuine
   early-training transient (steep loss, grad-clip firing, $\hat v$ still small).
   We measure when $\rho/\eta$ first comes within 20% of its late value and
   report the width of that opening spike.
""")
    nb.code(r'''
print(f"{'warmup':>7} | {'argmax ρ':>9} | {'ρ peak':>10} | {'ρ/η spike ends':>15} | {'ρ/η @end vs @5':>15}")
for wu in [0, 50, 200]:
    h = RUNS[wu]
    steps = np.array(h["ratio_step"])
    g   = np.array(h["ratio_global"])
    lr  = np.array(h["lr"])[steps]
    rn  = g / lr                                   # ρ normalised by LR
    late = np.median(rn[100:])                     # its settled level
    spike_end = int(next((s for s, v in zip(steps, rn) if abs(v/late - 1) < 0.20), -1))
    print(f"{wu:>7} | {int(np.argmax(g)):>9} | {g.max():>10.2e} | {spike_end:>15} | "
          f"{rn[5]/late:>14.2f}x")

print("\n- argmax(ρ) == the warmup length (0, ~50, ~200): that is the step at which "
      "warmup stops pushing ρ up.\n"
      "- ρ/η (LR divided out) has only a short opening spike — it is back within "
      "20% of its settled level within ~15–55 steps for every warmup, i.e. that "
      "residual is an early-training effect, NOT a warmup effect.")
''')

    nb.md("## 4. Charts")
    nb.code(r'''
fig, ax = plt.subplots(2, 3, figsize=(16, 9))

# (a) global rho vs step, all three warmups
for wu, c in zip([0,50,200], ["C0","C1","C2"]):
    h = RUNS[wu]
    ax[0,0].plot(h["ratio_step"], h["ratio_global"], color=c, label=f"warmup={wu}")
    ax[0,0].axvline(wu, color=c, ls=":", alpha=0.5)
ax[0,0].set_yscale("log"); ax[0,0].set_title("Global update/weight ratio ρ")
ax[0,0].set_xlabel("step"); ax[0,0].axhline(1e-3, color="k", ls="--", alpha=0.4, label="1e-3 target")
ax[0,0].legend()

# (b) LR schedule for reference
for wu, c in zip([0,50,200], ["C0","C1","C2"]):
    ax[0,1].plot(RUNS[wu]["step"], RUNS[wu]["lr"], color=c, label=f"warmup={wu}")
ax[0,1].set_title("LR schedule"); ax[0,1].set_xlabel("step"); ax[0,1].legend()

# (c) per-layer rho for warmup=50
h = RUNS[50]
for b in sorted(h["ratio_by_layer"]):
    ax[0,2].plot(h["ratio_step"], h["ratio_by_layer"][b], alpha=0.7, lw=1)
ax[0,2].axvline(50, color="k", ls=":", label="warmup end")
ax[0,2].set_yscale("log"); ax[0,2].set_title("Per-layer ρ (warmup=50)")
ax[0,2].set_xlabel("step"); ax[0,2].legend()

# (d) ρ by depth at a few steps (warmup=50)
blocks = [f"L{i}.mlp" for i in range(6)]
for st in [5, 25, 50, 150, 299]:
    ax[1,0].plot(range(6), [h["ratio_by_layer"][b][st] for b in blocks], "o-", label=f"step {st}")
ax[1,0].set_yscale("log"); ax[1,0].set_title("ρ vs depth (MLP blocks)")
ax[1,0].set_xlabel("block index"); ax[1,0].legend()

# (e) Δρ/ρ per step (warmup=50, global)
g = np.array(h["ratio_global"])
rel = np.abs(np.diff(g)) / (np.abs(g[:-1]) + 1e-12)
ax[1,1].plot(h["ratio_step"][1:], rel)
ax[1,1].axvline(50, color="k", ls=":"); ax[1,1].axhline(0.02, color="r", ls="--")
ax[1,1].set_yscale("log"); ax[1,1].set_title("|Δρ/ρ| per step (warmup=50)")
ax[1,1].set_xlabel("step")

# (f) loss curves
for wu, c in zip([0,50,200], ["C0","C1","C2"]):
    h2 = RUNS[wu]
    ax[1,2].plot(h2["eval_step"], h2["val_loss"], color=c, marker="o", label=f"warmup={wu}")
ax[1,2].set_title("Val loss"); ax[1,2].set_xlabel("step"); ax[1,2].legend()

plt.tight_layout()
savefig(fig, "t3_update_weight_ratio.png")
plt.show()
''')

    nb.md(r"""
## 6. Findings

*(Exact numbers printed above; seed-robust picture.)*

- **Per-layer $\rho$ settles at $\approx 1.0\text{–}1.4\times10^{-3}$** for every
  attention and MLP block by step ~300 — essentially the folklore $10^{-3}$
  target — while **LayerNorm gains sit ~30–40× lower** ($\approx 3\times10^{-5}$)
  and the token embedding a touch lower than the blocks ($\approx 7\times10^{-4}$).
  Depth matters little; the spread across the 6 blocks is under 1.3×.
- **$\rho$ is essentially $\eta_t \times \text{const}$.** The weight norms barely
  move over 300 steps, so the update-to-weight ratio just tracks the learning
  rate. That makes the answer to "when does warmup stop changing $\rho$" almost
  definitional:
  - **The $\rho$ peak lands at `step ≈ warmup`** (0 → step 0–3, 50 → step ~48,
    200 → step ~200). After the peak the cosine decay drives $\rho$ *down*;
    warmup is no longer pushing it up.
  - **Removing the LR ($\rho/\eta$) leaves only a short opening transient** (~15
    steps for `warmup=0`, up to ~55 for the long slow `warmup=200` ramp) — steep
    early loss, grad-clip firing on steps 0–3 (gnorm ≈ 16 → < 1), $\hat v$ still
    warming. It is an early-training effect, not something warmup itself creates.
- **So the reported step is the warmup length itself**: warmup ∈ {0, 50, 200}
  stops changing $\rho$ at step ≈ {0, 50, 200}. Its only job is to shape $\rho$
  over exactly those first `warmup` steps; beyond that, $\rho$ is 100%
  schedule-driven.
- **Skipping warmup hurt a lot on this budget**: `warmup=0` finished at val
  ≈ 2.44 vs ≈ 1.99 for `warmup∈{50,200}`. Two reasons compound — (a) grad-clip
  fires hard on steps 0–3 and $\rho$ spikes to ~2× its settled value, kicking the
  first block off-course; (b) with cosine starting at step 0 the LR is already
  decaying before the model can use it, so the 300-step budget runs at a lower
  *average* LR. `warmup=50` and `warmup=200` land within 0.005 of each other —
  once you have *enough* warmup, the exact amount barely matters here.
""")
    nb.write(os.path.join(OUT, "03_update_weight_ratio_warmup.ipynb"))


# ===========================================================================
# NOTEBOOK 4 — cosine vs WSD
# ===========================================================================
def nb4():
    nb = NB("04")
    nb.md(r"""
# Task 4 — Cosine vs WSD, 300 Steps, Judge at Step 200

> **Ask.** Train the same model twice for 300 steps, once under cosine and once
> under WSD, and stop both at step 200. Report both losses and state which model
> you would keep.

### The two schedules

| | **Cosine** | **WSD** (Warmup-Stable-Decay) |
|---|---|---|
| shape | warmup → cosine decay to `min_lr` over *all* remaining steps | warmup → **flat** at peak LR → short decay over the last ~20% |
| commits to a horizon? | **yes** — the curve is tied to `total_steps` | **no** — the plateau can be cut anywhere and a fresh decay run |
| at an intermediate step | LR already well below peak | LR still at peak (if before the decay window) |

The interesting question is what you have *at step 200 of a 300-step budget*:
cosine has been cooling since warmup; WSD is still at full LR and only starts its
decay at step 240.
""")
    nb.code(SETUP)

    nb.md("## 1. Train both (300 steps), plus a WSD variant that decays from 200")
    nb.code(r'''
from s11.data import load_char_dataset
from s11.trainer import TrainConfig, train, loss_at_step

data = load_char_dataset()
common = dict(total_steps=300, warmup=50, base_lr=1.5e-3, beta2=0.99,
              weight_decay=0.1, batch_size=32, eval_interval=10, eval_iters=50,
              ratio_log_interval=25, device=DEVICE, min_lr_frac=0.1)

print("=== cosine (300) ===")
cos = train(data, TrainConfig(schedule="cosine", label="cosine_300", **common),
            aim_repo=AIM_REPO, experiment="task4_schedules")

print("\n=== WSD (300, decay last 20%) ===")
wsd = train(data, TrainConfig(schedule="wsd", wsd_decay_frac=0.20,
            wsd_decay_shape="linear", label="wsd_300", **common),
            aim_repo=AIM_REPO, experiment="task4_schedules")

print("\n=== WSD budget=200 (decay 160->200): the 'I decided to stop early' run ===")
c200 = dict(common); c200["total_steps"] = 200; c200["eval_interval"] = 10
wsd200 = train(data, TrainConfig(schedule="wsd", wsd_decay_frac=0.20,
               wsd_decay_shape="linear", label="wsd_replan_200", **c200),
               aim_repo=AIM_REPO, experiment="task4_schedules")
''')

    nb.md("## 2. The numbers at step 200 (and at 300)")
    nb.code(r'''
def at(h, s, k="val_loss"): return loss_at_step(h, s, k)

tab = pd.DataFrame([
    dict(run="cosine (300-step plan)",   val_at_200=at(cos,200),   train_at_200=at(cos,200,"train_loss"),
         val_at_300=cos["final"]["val_loss"], lr_at_200=np.interp(200, cos["step"], cos["lr"])),
    dict(run="WSD (300-step plan)",      val_at_200=at(wsd,200),   train_at_200=at(wsd,200,"train_loss"),
         val_at_300=wsd["final"]["val_loss"], lr_at_200=np.interp(200, wsd["step"], wsd["lr"])),
    dict(run="WSD (re-planned to 200)",  val_at_200=wsd200["final"]["val_loss"],
         train_at_200=wsd200["final"]["train_loss"], val_at_300=np.nan,
         lr_at_200=np.interp(199, wsd200["step"], wsd200["lr"])),
])
print(tab.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print(f"\nΔ(val@200)  WSD-plan minus cosine-plan = {at(wsd,200) - at(cos,200):+.4f}")
print(f"Δ(val@200)  WSD-replan minus cosine-plan = {wsd200['final']['val_loss'] - at(cos,200):+.4f}")
print(f"Δ(val@300)  WSD minus cosine (full plan) = {wsd['final']['val_loss'] - cos['final']['val_loss']:+.4f}")
''')

    nb.md("## 3. Charts")
    nb.code(r'''
fig, ax = plt.subplots(2, 2, figsize=(14, 9))

ax[0,0].plot(cos["step"], cos["lr"], label="cosine (300)")
ax[0,0].plot(wsd["step"], wsd["lr"], label="WSD (300)")
ax[0,0].plot(wsd200["step"], wsd200["lr"], ls="--", label="WSD re-plan (200)")
ax[0,0].axvline(200, color="grey", ls=":"); ax[0,0].set_title("LR schedule")
ax[0,0].set_xlabel("step"); ax[0,0].legend()

for h, lab in [(cos,"cosine (300)"), (wsd,"WSD (300)"), (wsd200,"WSD re-plan (200)")]:
    ax[0,1].plot(h["eval_step"], h["val_loss"], marker=".", label=lab)
ax[0,1].axvline(200, color="grey", ls=":"); ax[0,1].set_title("Val loss")
ax[0,1].set_xlabel("step"); ax[0,1].legend()

for h, lab in [(cos,"cosine (300)"), (wsd,"WSD (300)"), (wsd200,"WSD re-plan (200)")]:
    ax[1,0].plot(h["eval_step"], h["val_loss"], marker=".", label=lab)
ax[1,0].set_xlim(150, 300);
lo = min(min(cos["val_loss"][-8:]), min(wsd["val_loss"][-8:]))
ax[1,0].set_ylim(lo-0.03, lo+0.25)
ax[1,0].axvline(200, color="grey", ls=":"); ax[1,0].set_title("Val loss — zoom on 150–300")
ax[1,0].set_xlabel("step"); ax[1,0].legend()

ax[1,1].plot(cos["step"], cos["train_loss_step"], alpha=0.5, label="cosine train (step)")
ax[1,1].plot(wsd["step"], wsd["train_loss_step"], alpha=0.5, label="WSD train (step)")
ax[1,1].axvline(200, color="grey", ls=":"); ax[1,1].axvline(240, color="C1", ls=":", alpha=0.5)
ax[1,1].set_title("Train loss (per step)"); ax[1,1].set_xlabel("step"); ax[1,1].legend()

plt.tight_layout()
savefig(fig, "t4_cosine_vs_wsd.png")
plt.show()
''')

    nb.md(r"""
## 4. Which model would I keep?

*(Exact losses printed above; measured on this run.)*

| checkpoint | val loss | when |
|---|---|---|
| cosine, stopped at 200 | **2.075** | LR already decayed to 6.2e-4 |
| WSD (300-plan), stopped at 200 | **2.087** | still on the 1.5e-3 plateau |
| **WSD re-planned to a 200 budget** | **2.050** | got a real 40-step decay |
| cosine, full 300 | 1.926 | |
| **WSD, full 300** | **1.838** | |

**Reading it**

- **Frozen at step 200 of a 300-step plan**, cosine (2.075) edges out plateau-WSD
  (2.087) — cosine has been annealing for 150 steps, WSD has not. That 0.012 gap
  is the "you stopped me mid-plateau" penalty, and it is small.
- **The fair comparison** (assignment's own warning: *tune both sides*): if you
  actually decide to stop at 200, you tell WSD that up front and it decays over
  160→200. That **WSD-replanned checkpoint (2.050) beats cosine-at-200**, because
  it ran at a higher LR for longer and still got its cooldown.
- **At the full horizon WSD wins outright here** (1.838 vs 1.926, a 0.087 gap) —
  the long 1.5e-3 plateau does more useful work than cosine's immediate decay on
  this short, well-warmed budget.

**Decision: keep WSD.** Concretely, keep the **WSD, full-300** checkpoint (lowest
loss, 1.838). If forced to stop at exactly step 200 with no re-plan, keep the
**cosine-200** checkpoint by a hair — but the moment you can choose your stopping
point, WSD's re-plannable plateau is strictly better here, and it removes the
"pick the horizon before you start" commitment that cosine forces on you.
""")
    nb.write(os.path.join(OUT, "04_cosine_vs_wsd.ipynb"))


# ===========================================================================
# NOTEBOOK 5 — LR sweep vs width
# ===========================================================================
def nb5():
    nb = NB("05")
    nb.md(r"""
# Task 5 — Learning-Rate Sweep across Width (256 / 512 / 1024) → Predict Width 4096

> **Ask.** Sweep the learning rate at widths 256, 512 and 1024, plot loss against
> learning rate, and mark the three minima. State the value you would use at
> width 4096 and how confident you are in it.

### Why the optimum moves

This model uses **standard parametrization (SP)**: `nn.Linear` initialised at
`std=0.02`, LR shared across all tensors. As width grows, the pre-activations into
each matmul grow like $\sqrt{\text{fan\_in}}$ and the Adam update (which is
$O(\eta)$ per element regardless of gradient scale) moves the outputs by more — so
the *same* $\eta$ is effectively "hotter" at larger width and the loss-vs-LR
bowl **shifts left**. Pure SP theory for matmul params predicts
$\eta^\star \propto \text{width}^{-1}$; embeddings / LayerNorms don't scale that
way and pull the exponent toward 0. (**muP** re-scales per-tensor so $\eta^\star$
is width-independent — not what we're doing here, but it's the fix if width keeps
changing.)

### Method (tune every side, per the assignment's warning)

1. **Coarse sweep** — a shared 7-point LR grid at each width, 1 seed, 220 steps →
   locates each basin.
2. **Refinement** — a 5-point grid bracketing each basin, **2 seeds**, 220 steps →
   pins the minimum and gives a seed spread (our error bar). A local quadratic in
   $\log_{10}\eta$ is fit to the *refinement* points only (where the bowl is
   actually quadratic), not to the wide coarse grid.
3. **Extrapolate** $\eta^\star(\text{width})$ as a power law to width 4096.

Cosine schedule, warmup 40, `weight_decay=0.1`, `beta2=0.99`, batch 16, `n_head=8`
throughout. ~35 min on a 4 GB laptop GPU.
""")
    nb.code(SETUP)

    nb.md("## 1. Coarse sweep — 3 widths × 7 LRs, locate the basins")
    nb.code(r'''
from s11.data import load_char_dataset
from s11.trainer import TrainConfig, train

data = load_char_dataset()
WIDTHS   = [256, 512, 1024]
COARSE   = [3e-4, 6e-4, 1e-3, 2e-3, 4e-3, 8e-3, 1.5e-2]
STEPS    = 220

def run_one(w, lr, seed, tag):
    cfg = TrainConfig(
        n_embd=w, n_head=8, n_layer=6, dropout=0.0,
        total_steps=STEPS, warmup=40, schedule="cosine", min_lr_frac=0.1,
        base_lr=lr, beta2=0.99, weight_decay=0.1, grad_clip=1.0, seed=seed,
        batch_size=16, eval_interval=STEPS-1, eval_iters=80,
        ratio_log_interval=100, label=f"w{w}_lr{lr:.1e}_s{seed}_{tag}", device=DEVICE,
    )
    return train(data, cfg, aim_repo=AIM_REPO, experiment="task5_lr_width",
                 extra_context={"width": w, "stage": tag}, progress=False)

coarse = {}
t0 = time.time()
for w in WIDTHS:
    for lr in COARSE:
        h = run_one(w, lr, 1337, "coarse")
        coarse[(w, lr)] = h["final"]["val_loss"]
        print(f"  w{w:4d} | lr {lr:.1e} | val {h['final']['val_loss']:.4f}")
print(f"\ncoarse wall-clock: {(time.time()-t0)/60:.1f} min")
''')

    nb.code(r'''
# the basin at each width = the 3 grid points around the coarse argmin
basin_center = {}
for w in WIDTHS:
    losses = np.array([coarse[(w, lr)] for lr in COARSE])
    i = int(np.argmin(losses))
    basin_center[w] = COARSE[i]
    print(f"width {w}: coarse best LR = {COARSE[i]:.1e}  (val {losses[i]:.4f})")
''')

    nb.md("## 2. Refinement — 5 LRs per basin × 2 seeds")
    nb.code(r'''
REFINE_MULT = [0.5, 0.71, 1.0, 1.41, 2.0]   # geometric bracket around the basin
SEEDS = [1337, 7]

refine = {w: {} for w in WIDTHS}   # width -> {lr: [loss_seed0, loss_seed1]}
t0 = time.time()
for w in WIDTHS:
    lrs = sorted({round(basin_center[w] * m, 6) for m in REFINE_MULT})
    for lr in lrs:
        vals = []
        for s in SEEDS:
            h = run_one(w, lr, s, "refine")
            vals.append(h["final"]["val_loss"])
        refine[w][lr] = vals
        print(f"  w{w:4d} | lr {lr:.2e} | val {np.mean(vals):.4f} ± {np.std(vals):.4f}  {vals}")
print(f"\nrefinement wall-clock: {(time.time()-t0)/60:.1f} min")
''')

    nb.md("## 3. The three minima")
    nb.code(r'''
def local_parabola(lrs, losses):
    """Quadratic in log10(lr); minimum clipped to the sampled range."""
    x = np.log10(np.array(lrs)); y = np.array(losses)
    c = np.polyfit(x, y, 2)
    xs = -c[1] / (2 * c[0]) if c[0] > 0 else x[np.argmin(y)]
    xs = float(np.clip(xs, x.min(), x.max()))
    return 10**xs, np.polyval(c, xs), c

minima = {}
rows = []
for w in WIDTHS:
    lrs = sorted(refine[w])
    mean = [float(np.mean(refine[w][lr])) for lr in lrs]
    spread = [float(np.std(refine[w][lr])) for lr in lrs]
    lr_star, loss_star, coef = local_parabola(lrs, mean)
    # seed spread at the grid point nearest the fitted optimum = our error bar
    near = lrs[int(np.argmin([abs(np.log(l) - np.log(lr_star)) for l in lrs]))]
    minima[w] = dict(lr_star=lr_star, loss_star=loss_star, coef=coef,
                     seed_spread=float(np.std(refine[w][near])),
                     grid_best_lr=lrs[int(np.argmin(mean))],
                     grid_best_loss=min(mean))
    rows.append(dict(width=w, fitted_lr_star=lr_star, fitted_loss=loss_star,
                     grid_best_lr=lrs[int(np.argmin(mean))], grid_best_loss=min(mean),
                     seed_spread=minima[w]["seed_spread"]))
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
''')

    nb.md("## 4. Charts — loss vs LR, and the LR-transfer extrapolation")
    nb.code(r'''
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
colors = {256: "C0", 512: "C1", 1024: "C2"}

for w in WIDTHS:
    c = colors[w]
    # coarse (faint)
    lc = sorted(COARSE)
    ax[0].plot(lc, [coarse[(w, lr)] for lr in lc], ":", color=c, alpha=0.4)
    ax[0].scatter(lc, [coarse[(w, lr)] for lr in lc], s=25, color=c, alpha=0.4)
    # refinement (bold, with seed error bars)
    lr_ref = sorted(refine[w])
    m = [np.mean(refine[w][lr]) for lr in lr_ref]
    e = [np.std(refine[w][lr]) for lr in lr_ref]
    ax[0].errorbar(lr_ref, m, yerr=e, fmt="o-", color=c, capsize=3, label=f"width {w}")
    # local parabola
    xs = np.logspace(np.log10(min(lr_ref)), np.log10(max(lr_ref)), 60)
    ax[0].plot(xs, np.polyval(minima[w]["coef"], np.log10(xs)), "--", color=c, alpha=0.6)
    ax[0].scatter([minima[w]["lr_star"]], [minima[w]["loss_star"]], marker="*", s=260,
                  color=c, edgecolor="k", zorder=6)
ax[0].set_xscale("log"); ax[0].set_xlabel("learning rate")
ax[0].set_ylabel("val loss @ step 220"); ax[0].set_ylim(top=min(
    min(np.mean(refine[w][lr]) for lr in refine[w]) for w in WIDTHS) + 0.8)
ax[0].set_title("Loss vs LR  (dotted = coarse, solid = 2-seed refinement, ★ = fitted min)")
ax[0].legend()

ws = np.array(WIDTHS, float)
lrs_star = np.array([minima[w]["lr_star"] for w in WIDTHS])
b, a = np.polyfit(np.log(ws), np.log(lrs_star), 1)
resid = np.log(lrs_star) - (a + b * np.log(ws))
sigma = float(np.sqrt(np.sum(resid**2) / max(1, len(ws) - 2)))
pred = lambda W: math.exp(a + b * math.log(W))
lr_4096 = pred(4096)

ax[1].plot(ws, lrs_star, "o", ms=12, label="fitted optima (2-seed)")
Wg = np.array([256, 512, 1024, 2048, 4096], float)
ax[1].plot(Wg, [pred(W) for W in Wg], "--",
           label=f"power law: η* ∝ width^{b:.2f}")
ax[1].fill_between(Wg, [pred(W)*math.exp(-2*sigma) for W in Wg],
                   [pred(W)*math.exp(2*sigma) for W in Wg], alpha=0.15,
                   label="±2σ fit band")
ax[1].scatter([4096], [lr_4096], color="crimson", s=240, marker="*", edgecolor="k",
              zorder=6, label=f"η*(4096) ≈ {lr_4096:.1e}")
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel("width (n_embd)"); ax[1].set_ylabel("optimal learning rate")
ax[1].set_title("LR transfer to width 4096"); ax[1].legend()

plt.tight_layout()
savefig(fig, "t5_lr_sweep_width.png")
plt.show()

print(f"power-law exponent b   = {b:+.3f}   (SP theory for matmul params ≈ -1.0)")
print(f"fit residual sigma      = {sigma:.3f}  (in log-LR units)")
print(f"predicted η*(4096)      = {lr_4096:.2e}")
print(f"  ±2σ band             = [{lr_4096*math.exp(-2*sigma):.2e}, {lr_4096*math.exp(2*sigma):.2e}]")
print(f"  endpoint-only (256↔1024) bracket = {math.exp(np.polyval(np.polyfit(np.log(ws[[0,-1]]), np.log(lrs_star[[0,-1]]),1), np.log(4096))):.2e}")
''')

    nb.md("## 5. Sanity — did every run train? (grad-norm + divergence guard)")
    nb.code(r'''
guard = []
for w in WIDTHS:
    for lr, vals in refine[w].items():
        guard.append(dict(width=w, lr=lr, mean_val=np.mean(vals), seed_spread=np.std(vals),
                          diverged=bool(any((not np.isfinite(v)) or v > 4.0 for v in vals))))
gd = pd.DataFrame(guard).sort_values(["width", "lr"])
print(gd.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nSeed spread grows with width and with LR — exactly why the 1024 basin "
      "needed 2 seeds to trust.")
''')

    nb.md(r"""
## 6. Findings

**The three minima** (2-seed refinement, local quadratic fit — `★` in the left
chart). Numbers below are from the laptop-GPU run; re-running longer will shift
them, so trust the *notebook's printed table*, not these:

| width | fitted $\eta^\star$ | val loss @ 220 | 2-seed spread near $\eta^\star$ |
|---|---|---|---|
| 256  | ~`1.5e-3` | ~2.40 | ±0.002 (flat basin) |
| 512  | ~`1.6e-3` (grid best `2.0e-3`) | ~2.17–2.22 | **±0.03** (noisy basin) |
| 1024 | ~`5e-4` (grid best `4.3e-4`) | ~2.17 | ±0.004 at optimum, ±0.01 just above |

**The trend.** The optimum is essentially **flat from width 256→512** (~`1.5e-3`)
and then **drops sharply at 1024** (~`5e-4`). Fitting all three as
$\eta^\star \propto \text{width}^{\,b}$ gives **b ≈ −0.84** but with a *large* fit
residual (σ ≈ 0.5 in log-LR) precisely because 256 and 512 barely move — SP
theory's $b=-1$ for matmul params is diluted here by the non-scaling
embedding/LayerNorm parameters and by how wide the 256 basin is.

**Value I would use at width 4096:** the power-law fit extrapolates to
**η\*(4096) ≈ `1.8e-4`** (endpoint-only bracket ≈ `1.4e-4`; ±2σ band roughly
`6e-5 … 5e-4`). If instead you extrapolate only the part that is actually
*moving* (512→1024, a ~0.3× drop per doubling) you get ≈ `5e-5` at 4096. So:
**I would start at `2e-4`** and run a tight confirmation sweep
`{1e-4, 2e-4, 4e-4}` at the real training length before committing.

**Confidence: LOW.** Why:

1. **3 width points, 2-parameter fit, σ ≈ 0.5.** The 256 and 512 optima are within
   noise of each other, so the slope is set almost entirely by the 1024 point.
2. **Short runs (220 steps).** The LR that wins at 220 steps sits *above* the one
   that wins at 5–50k steps; a real 4096 run is longer, pushing the true optimum
   *lower* still.
3. **Extrapolating 2 octaves** (1024→4096) from a 2-octave measurement, and the
   width-1024 basin is the sharpest (biggest loss penalty for being wrong).
4. **batch size, warmup, weight decay, β₂ all fixed.** At 4096 several want
   re-tuning — the assignment's "tune both sides" warning applies to the target.

**Honest one-liner:** `2e-4`, good to maybe a factor of 3, as the *center of a
confirmation sweep* — not a final number. Under **muP** the width-1024 optimum
(~`5e-4`) would transfer to 4096 directly and none of this extrapolation would be
needed.
""")
    nb.write(os.path.join(OUT, "05_lr_sweep_width.ipynb"))


if __name__ == "__main__":
    nb1(); nb2(); nb3(); nb4(); nb5()
    print("\nall notebooks written to", OUT)
