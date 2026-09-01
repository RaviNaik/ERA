"""Break gradient accumulation on purpose, then watch the gap.

Two ways to combine K micro-batches into one optimizer step:

  correct : sum all the per-token losses, divide by the total number of tokens.
            Every token gets one equal vote.

  wrong   : average each micro-batch's mean loss.  ("average of the averages")
            A micro-batch with half as many tokens still gets a full vote.

They are identical when every micro-batch holds the same number of tokens, which
is why this bug survived in every major framework until 2024. Feed micro-batches
of *different* lengths and they diverge -- by 15.4% on the class-notes example.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch

from .data import CharDataset, get_var_micro_batch, get_batch


# ---------------------------------------------------------------------------
# 1. the static, arithmetic-only demonstration (class-notes numbers)
# ---------------------------------------------------------------------------
def average_of_averages_demo(
    token_counts=(4, 4, 2),
    mean_losses=(2.0, 2.0, 5.0),
) -> dict:
    total_loss = sum(n * m for n, m in zip(token_counts, mean_losses))
    total_tokens = sum(token_counts)
    correct = total_loss / total_tokens
    wrong = sum(mean_losses) / len(mean_losses)
    return {
        "token_counts": token_counts,
        "mean_losses": mean_losses,
        "correct (sum loss / sum tokens)": correct,
        "wrong (mean of means)": wrong,
        "relative error %": 100 * (wrong - correct) / correct,
    }


# ---------------------------------------------------------------------------
# 2. the real training comparison
# ---------------------------------------------------------------------------
@dataclass
class AccumConfig:
    micro_batch_size: int = 8
    real_lens: tuple[int, ...] = (48, 48, 12)  # tokens per micro-batch (unequal!)
    pad_to: int = 64
    n_steps: int = 60
    lr: float = 3e-3
    eval_every: int = 5
    eval_seq_len: int = 64
    eval_batch: int = 32


@dataclass
class AccumRun:
    label: str
    steps: list[int] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)


@torch.no_grad()
def _eval_val_loss(model, ds, cfg: AccumConfig, device) -> float:
    model.eval()
    g = torch.Generator().manual_seed(0)  # same val batch every time
    x, y = get_batch(ds, "val", cfg.eval_batch, cfg.eval_seq_len, device, generator=g)
    _, loss = model(x, targets=y)
    model.train()
    return float(loss.item())


def _accumulate_correct(model, micro_batches) -> float:
    """sum of per-token losses / total tokens, then one step."""
    model.zero_grad(set_to_none=True)
    total_tokens = 0.0
    total_loss = 0.0
    for x, y, mask in micro_batches:
        _, loss_sum = model(x, targets=y, loss_mask=mask, reduction="sum")
        loss_sum.backward()
        n = float(mask.sum().item())
        total_tokens += n
        total_loss += float(loss_sum.item())
    for p in model.parameters():
        if p.grad is not None:
            p.grad.mul_(1.0 / total_tokens)
    return total_loss / total_tokens


def _accumulate_wrong(model, micro_batches) -> float:
    """average of each micro-batch's mean loss, then one step."""
    model.zero_grad(set_to_none=True)
    k = len(micro_batches)
    running = 0.0
    for x, y, mask in micro_batches:
        _, loss_mean = model(x, targets=y, loss_mask=mask, reduction="mean")
        (loss_mean / k).backward()
        running += float(loss_mean.item()) / k
    return running


def compare_accumulation(model, ds: CharDataset, cfg: AccumConfig, device) -> dict:
    """Train two copies of the same initial model, one per accumulation rule."""
    model_correct = copy.deepcopy(model).to(device).train()
    model_wrong = copy.deepcopy(model).to(device).train()
    opt_correct = torch.optim.AdamW(model_correct.parameters(), lr=cfg.lr)
    opt_wrong = torch.optim.AdamW(model_wrong.parameters(), lr=cfg.lr)

    run_correct = AccumRun("correct: sum loss / sum tokens")
    run_wrong = AccumRun("wrong: average of the averages")

    gen = torch.Generator().manual_seed(4242)
    for step in range(cfg.n_steps):
        # the SAME micro-batches feed both models this step
        micro = [
            get_var_micro_batch(
                ds, "train", cfg.micro_batch_size, rl, cfg.pad_to, device, generator=gen
            )
            for rl in cfg.real_lens
        ]
        tl_c = _accumulate_correct(model_correct, micro)
        opt_correct.step()
        tl_w = _accumulate_wrong(model_wrong, micro)
        opt_wrong.step()

        if step % cfg.eval_every == 0 or step == cfg.n_steps - 1:
            for run, m, tl in (
                (run_correct, model_correct, tl_c),
                (run_wrong, model_wrong, tl_w),
            ):
                run.steps.append(step)
                run.train_loss.append(tl)
                run.val_loss.append(_eval_val_loss(m, ds, cfg, device))

    return {"correct": run_correct, "wrong": run_wrong, "cfg": cfg}


def plot_accumulation(result: dict, path: str) -> str:
    import matplotlib.pyplot as plt

    rc, rw = result["correct"], result["wrong"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, key in zip(ax, ("val_loss", "train_loss")):
        a.plot(rc.steps, getattr(rc, key), "o-", label=rc.label, color="#1a7f37")
        a.plot(rw.steps, getattr(rw, key), "s--", label=rw.label, color="#cf222e")
        a.set_xlabel("optimizer step")
        a.set_ylabel(key.replace("_", " "))
        a.set_title(key.replace("_", " "))
        a.legend()
        a.grid(alpha=0.3)
    fig.suptitle("gradient accumulation: token-weighted vs. average-of-averages "
                 f"(micro-batch token counts {result['cfg'].real_lens})")
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path
