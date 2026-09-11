"""Training loop with Aim tracking and per-layer update-to-weight ratio logging.

Every experiment in the notebooks goes through :func:`train` so that runs are
comparable and *all of them* land in the same Aim repo (``session11/.aim``).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict

import numpy as np
import torch

from .data import CharDataset
from .model import GPT, GPTConfig, build_model
from .optim import HandAdamW, make_schedule


@dataclass
class TrainConfig:
    # data / model
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = True
    # optim
    base_lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.99
    eps: float = 1e-8
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    bias_correction: bool = True
    # schedule
    schedule: str = "cosine"          # cosine | wsd | constant
    warmup: int = 100
    total_steps: int = 300
    min_lr_frac: float = 0.1
    wsd_decay_frac: float = 0.2
    wsd_decay_shape: str = "linear"
    # loop
    batch_size: int = 32
    eval_interval: int = 25
    eval_iters: int = 40
    seed: int = 1337
    device: str = "cuda"
    ratio_log_interval: int = 1       # log update/weight ratio every N steps
    keep_model: bool = False          # return the trained model in the history dict
    label: str = "run"


def _configure_optimizer(model: GPT, cfg: TrainConfig):
    """2D params (matmuls, embeddings) get weight decay; 1D params (LayerNorm,
    biases) do not — the nanoGPT / AdamW convention."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return HandAdamW(groups, lr=cfg.base_lr, betas=(cfg.beta1, cfg.beta2),
                     eps=cfg.eps, bias_correction=cfg.bias_correction)


@torch.no_grad()
def estimate_loss(model, data: CharDataset, cfg: TrainConfig, gen):
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            X, Y = data.get_batch(split, cfg.batch_size, cfg.block_size, cfg.device, gen)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def _layer_bucket(name: str) -> str:
    if "wte" in name or "lm_head" in name:
        return "embedding"
    if "wpe" in name:
        return "pos_embedding"
    if ".attn." in name:
        parts = name.split(".")
        return f"L{parts[2]}.attn"
    if ".mlp." in name:
        parts = name.split(".")
        return f"L{parts[2]}.mlp"
    if ".ln_" in name or "ln_f" in name:
        return "layernorm"
    return name


def train(data: CharDataset, cfg: TrainConfig, aim_repo: str | None = None,
          experiment: str = "session11", extra_context: dict | None = None,
          progress: bool = True):
    """Run training; return a dict of per-step histories + final metrics."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)

    device = cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"

    mcfg = GPTConfig(vocab_size=data.vocab_size, block_size=cfg.block_size,
                     n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd,
                     dropout=cfg.dropout, bias=cfg.bias)
    model = build_model(mcfg, device=device)
    n_params = model.num_params()

    opt = _configure_optimizer(model, cfg)
    sched = make_schedule(
        cfg.schedule, base_lr=cfg.base_lr, warmup=cfg.warmup,
        total=cfg.total_steps, min_lr_frac=cfg.min_lr_frac,
        **({"decay_frac": cfg.wsd_decay_frac, "decay_shape": cfg.wsd_decay_shape}
           if cfg.schedule == "wsd" else {}),
    )

    hparams = {k: v for k, v in asdict(cfg).items()}
    hparams["n_params"] = n_params
    if extra_context:
        hparams.update(extra_context)

    run = None
    if aim_repo is not None:
        from aim import Run
        run = Run(repo=aim_repo, experiment=experiment)
        run["hparams"] = hparams
        run.name = cfg.label
        # On some filesystems aim's run-attribute store does not survive close();
        # metrics always do — so also log the scalar hparams as single-point
        # metrics under hp/* and mirror everything to results/runs.jsonl.
        for k, v in hparams.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                run.track(float(v), name=f"hp/{k}", step=0)

    hist = {
        "step": [], "lr": [], "train_loss_step": [],
        "grad_norm": [], "eval_step": [], "train_loss": [], "val_loss": [],
        "ratio_step": [], "ratio_by_layer": {}, "ratio_global": [],
    }

    model.train()
    t0 = time.time()
    running = None
    for step in range(cfg.total_steps):
        lr = sched(step)
        for g in opt.param_groups:
            g["lr"] = lr

        X, Y = data.get_batch("train", cfg.batch_size, cfg.block_size, device, gen)
        _, loss = model(X, Y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()

        # snapshot weights to measure the realised update
        if step % cfg.ratio_log_interval == 0:
            prev = {n: p.detach().clone() for n, p in model.named_parameters()}

        opt.step()

        lval = loss.item()
        running = lval if running is None else 0.9 * running + 0.1 * lval
        hist["step"].append(step)
        hist["lr"].append(lr)
        hist["train_loss_step"].append(lval)
        hist["grad_norm"].append(gnorm)
        if run is not None:
            run.track(lval, name="loss", step=step, context={"subset": "train_step"})
            run.track(lr, name="lr", step=step)
            run.track(gnorm, name="grad_norm", step=step)

        # update-to-weight ratio (RMS of update / RMS of weight), per layer bucket
        if step % cfg.ratio_log_interval == 0:
            bucket_num, bucket_den = {}, {}
            g_num = g_den = 0.0
            for n, p in model.named_parameters():
                d = (p.detach() - prev[n])
                num = float(d.pow(2).sum())
                den = float(prev[n].pow(2).sum()) + 1e-12
                b = _layer_bucket(n)
                bucket_num[b] = bucket_num.get(b, 0.0) + num
                bucket_den[b] = bucket_den.get(b, 0.0) + den
                g_num += num
                g_den += den
            hist["ratio_step"].append(step)
            gr = math.sqrt(g_num / g_den)
            hist["ratio_global"].append(gr)
            if run is not None:
                run.track(gr, name="update_weight_ratio", step=step, context={"layer": "global"})
            for b in bucket_num:
                r = math.sqrt(bucket_num[b] / bucket_den[b])
                hist["ratio_by_layer"].setdefault(b, []).append(r)
                if run is not None:
                    run.track(r, name="update_weight_ratio", step=step, context={"layer": b})

        if step % cfg.eval_interval == 0 or step == cfg.total_steps - 1:
            m = estimate_loss(model, data, cfg, gen)
            hist["eval_step"].append(step)
            hist["train_loss"].append(m["train"])
            hist["val_loss"].append(m["val"])
            if run is not None:
                run.track(m["train"], name="loss", step=step, context={"subset": "train"})
                run.track(m["val"], name="loss", step=step, context={"subset": "val"})
            if progress:
                print(f"  step {step:4d} | lr {lr:.2e} | train {m['train']:.4f} "
                      f"| val {m['val']:.4f} | gnorm {gnorm:.2f}")

    hist["wall_clock_s"] = time.time() - t0
    hist["n_params"] = n_params
    hist["final"] = {"train_loss": hist["train_loss"][-1], "val_loss": hist["val_loss"][-1]}
    hist["loss_at_200"] = {"val": loss_at_step(hist, 200, "val_loss"),
                           "train": loss_at_step(hist, 200, "train_loss")}
    if cfg.keep_model:
        hist["model"] = model
    else:
        del model, opt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {"final_val_loss": hist["final"]["val_loss"],
               "final_train_loss": hist["final"]["train_loss"],
               "val_loss_at_200": hist["loss_at_200"]["val"],
               "wall_clock_s": hist["wall_clock_s"]}
    if run is not None:
        run["summary"] = summary
        for k, v in summary.items():
            run.track(float(v), name=f"summary/{k}", step=0)
        run.close()

    # durable, filesystem-independent record for the README / aim_summary.py
    _append_run_json(aim_repo, experiment, cfg, hparams, summary,
                     hist["eval_step"], hist["val_loss"], hist["train_loss"],
                     run.hash if run is not None else None)
    return hist


def _append_run_json(aim_repo, experiment, cfg, hparams, summary,
                     eval_step, val_loss, train_loss, run_hash):
    import os as _os
    root = _os.path.abspath(aim_repo) if aim_repo else _os.getcwd()
    d = _os.path.join(root, "results")
    _os.makedirs(d, exist_ok=True)
    rec = {"experiment": experiment, "label": cfg.label, "aim_hash": run_hash,
           "hparams": hparams, "summary": summary,
           "curve": {"step": eval_step, "val_loss": val_loss, "train_loss": train_loss}}
    with open(_os.path.join(d, "runs.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


def loss_at_step(hist: dict, target_step: int, key: str = "val_loss") -> float:
    """Eval loss at ``target_step``, linearly interpolated between the two
    bracketing eval points (for the 'stop at 200' comparison in task 4).

    Returns NaN when ``target_step`` is not actually inside the evaluated range
    with a nearby anchor (e.g. sweeps that only evaluate at the final step)."""
    steps = list(hist["eval_step"])
    vals = list(hist[key])
    if not steps:
        return float("nan")
    if target_step <= steps[0]:
        return vals[0] if steps[0] - target_step <= 25 else float("nan")
    if target_step >= steps[-1]:
        return vals[-1] if target_step - steps[-1] <= 25 else float("nan")
    for i in range(1, len(steps)):
        if steps[i] >= target_step:
            s0, s1, v0, v1 = steps[i - 1], steps[i], vals[i - 1], vals[i]
            if s1 - s0 > 60:            # anchors too far apart to trust
                return float("nan")
            frac = (target_step - s0) / (s1 - s0)
            return v0 + frac * (v1 - v0)
    return float("nan")
