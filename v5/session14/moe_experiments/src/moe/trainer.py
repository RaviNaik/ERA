"""One training loop for every run (dense, MoE-from-scratch, upcycled MoE, dense control).

* All metrics go to Aim (``aim up`` to browse) and to ``results/<label>.json``.
* Global step numbering (``step_offset``) lets phase-2 runs continue the x-axis of phase 1.
* Evaluation always uses the *same* fixed batches, so every run is scored on identical tokens.
* Checkpoints make runs resumable; a finished run is simply reloaded (notebooks are re-runnable).
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch

from .data import TokenDataset
from .model import GPT, GPTConfig, param_report
from .utils import ckpt_dir, results_dir


@dataclass
class TrainConfig:
    label: str = "run"
    phase: str = "dense"              # free-form tag stored as an Aim context
    total_steps: int = 1000
    step_offset: int = 0              # global step of this run's first step
    batch_size: int = 32
    block_size: int = 512
    lr: float = 6e-4
    min_lr_frac: float = 0.1
    warmup: int = 100
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    lb_coef: float = 1e-4
    z_coef: float = 0.001
    bias_update: float = 1e-3         # loss-free balancing step size (0 disables)
    explore_steps: int = 0            # first N steps: probabilistic (Gumbel) expert selection
    explore_noise: float = 1.0
    eval_interval: int = 250
    eval_iters: int = 50
    log_interval: int = 10
    ckpt_interval: int = 1000
    seed: int = 1337
    data_seed: int | None = None      # defaults to seed + step_offset (same batches across sibling runs)
    device: str = "cuda"
    amp: bool = True                  # bf16 autocast on CUDA
    resume: bool = True


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    prog = (step - cfg.warmup) / max(1, cfg.total_steps - cfg.warmup)
    floor = cfg.lr * cfg.min_lr_frac
    return floor + 0.5 * (cfg.lr - floor) * (1 + math.cos(math.pi * min(1.0, prog)))


def make_optimizer(model: GPT, cfg: TrainConfig):
    """AdamW; decay on matrices/expert stacks, none on norms, biases and the router."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if (p.dim() < 2 or n.endswith("router.weight") or n.endswith((".b1", ".b2")))
         else decay).append(p)
    groups = [{"params": decay, "weight_decay": cfg.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    fused = cfg.device.startswith("cuda") and torch.cuda.is_available()
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=cfg.betas, fused=fused)


def routing_summary(load: torch.Tensor) -> dict:
    """load: [L_moe, E] fraction of routed slots per expert (rows sum to 1)."""
    E = load.shape[1]
    p = load.clamp_min(1e-12)
    ent = -(p * p.log()).sum(1) / math.log(E)                 # 1.0 = perfectly uniform
    return {
        "load_entropy": ent.tolist(),                          # per layer, normalised
        "max_load": load.max(1).values.tolist(),               # ideal = 1/E
        "max_vio": (load.max(1).values * E - 1).tolist(),      # (max load - mean) / mean ; 0 = perfectly even
        "load_cv": (load.std(1, unbiased=False) / load.mean(1)).tolist(),
        "dead_experts": int((load < 0.1 / E).sum()),           # < 10% of fair share
    }


@torch.no_grad()
def evaluate(model: GPT, data: TokenDataset, cfg: TrainConfig, device: str, amp_ctx) -> dict:
    model.eval()
    out, loads = {}, []
    for split in ("train", "val"):
        gen = torch.Generator().manual_seed(4242)              # identical eval tokens for every run
        losses = []
        for _ in range(cfg.eval_iters):
            x, y = data.get_batch(split, cfg.batch_size, cfg.block_size, device, gen)
            with amp_ctx():
                _, loss, aux = model(x, y)
            losses.append(loss.item())
            if aux is not None and split == "val":
                loads.append(aux["load"].float().cpu())
        out[split] = float(np.mean(losses))
    if loads:
        out["load"] = torch.stack(loads).mean(0)
    model.train()
    return out


def _results_path(label: str) -> str:
    return os.path.join(results_dir(), f"{label}.json")


def load_history(label: str) -> dict:
    with open(_results_path(label)) as f:
        return json.load(f)


def save_checkpoint(path: str, model: GPT, opt, step: int, hist: dict, done: bool, run_hash=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": asdict(model.cfg),
                "opt": None if opt is None else opt.state_dict(),
                "step": step, "hist": hist, "done": done, "aim_hash": run_hash}, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_model(path: str, device: str = "cpu") -> GPT:
    ck = torch.load(path, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ck["cfg"])).to(device)
    model.load_state_dict(ck["model"])
    return model


def train(model: GPT, data: TokenDataset, cfg: TrainConfig, experiment: str = "session14",
          aim_repo: str | None = None, extra_hparams: dict | None = None,
          progress: bool = True) -> tuple[GPT, dict]:
    """Train ``model`` in place; returns ``(model, history)``."""
    device = cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu"
    cfg.device = device
    use_amp = cfg.amp and device.startswith("cuda") and torch.cuda.is_bf16_supported()

    def amp_ctx():
        return torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else torch.autocast("cpu", enabled=False)

    ckpt_path = os.path.join(ckpt_dir(), f"{cfg.label}.pt")
    model.to(device).train()
    moe_blocks = model.moe_blocks()
    opt = make_optimizer(model, cfg)
    pr = param_report(model)

    hist = {"label": cfg.label, "phase": cfg.phase, "step_offset": cfg.step_offset,
            "n_params": pr["total"], "n_active_params": pr["active"],
            "step": [], "loss": [], "lr": [], "grad_norm": [], "lb": [], "z": [], "tok_s": [],
            "eval_step": [], "train_loss": [], "val_loss": [],
            "load": [], "load_entropy": [], "max_load": [], "max_vio": [], "load_cv": [], "dead_experts": [], "bias_absmean": []}
    start, run_hash = 0, None

    if cfg.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        hist, start, run_hash = ck["hist"], ck["step"], ck.get("aim_hash")
        if ck["done"]:
            print(f"[{cfg.label}] already finished ({start} steps) — loaded from {ckpt_path}")
            return model, hist
        opt.load_state_dict(ck["opt"])
        print(f"[{cfg.label}] resuming from step {start}")

    run = None
    hp = {**asdict(cfg), **(extra_hparams or {}), "n_params": pr["total"], "n_active_params": pr["active"],
          "model": asdict(model.cfg)}
    if aim_repo:
        from aim import Run
        run = Run(run_hash=run_hash, repo=aim_repo, experiment=experiment) if run_hash else \
            Run(repo=aim_repo, experiment=experiment)
        run.name = cfg.label
        run["hparams"] = hp
        run_hash = run.hash

    base_seed = cfg.data_seed if cfg.data_seed is not None else cfg.seed + cfg.step_offset
    data_gen = torch.Generator().manual_seed(base_seed + start)   # a resumed run draws fresh batches
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    ctx = {"phase": cfg.phase}
    tok_per_step = cfg.batch_size * cfg.block_size
    t_last, n_since = time.time(), 0

    print(f"[{cfg.label}] {pr['total']/1e6:.2f}M params ({pr['active']/1e6:.2f}M active/token) | "
          f"{cfg.total_steps} steps x {tok_per_step:,} tok = {cfg.total_steps*tok_per_step/1e6:.1f}M tokens | "
          f"lr {cfg.lr:g} | amp={'bf16' if use_amp else 'off'} | device={device}")

    def do_eval(step_local: int, final: bool = False):
        m = evaluate(model, data, cfg, device, amp_ctx)
        g = cfg.step_offset + step_local
        hist["eval_step"].append(g); hist["train_loss"].append(m["train"]); hist["val_loss"].append(m["val"])
        msg = f"  step {g:5d} | train {m['train']:.4f} | val {m['val']:.4f} (ppl {math.exp(m['val']):.1f})"
        if run:
            run.track(m["train"], name="loss", step=g, context={**ctx, "subset": "train"})
            run.track(m["val"], name="loss", step=g, context={**ctx, "subset": "val"})
            run.track(math.exp(min(m["val"], 20)), name="perplexity", step=g, context={**ctx, "subset": "val"})
        if "load" in m:
            s = routing_summary(m["load"])
            hist["load"].append(m["load"].tolist())
            for k in ("load_entropy", "max_load", "max_vio", "load_cv"):
                hist[k].append(s[k])
            hist["dead_experts"].append(s["dead_experts"])
            msg += (f" | load-entropy {np.mean(s['load_entropy']):.3f} | MaxVio {np.mean(s['max_vio']):.2f}"
                    f" | dead {s['dead_experts']}")
            hist["bias_absmean"].append([b.mlp.route_bias.abs().mean().item() for b in model.moe_blocks()])
            if run:
                for li in range(m["load"].shape[0]):
                    run.track(s["load_entropy"][li], name="router/load_entropy", step=g, context={**ctx, "layer": li})
                    run.track(s["max_load"][li], name="router/max_load", step=g, context={**ctx, "layer": li})
                    run.track(s["max_vio"][li], name="router/max_vio", step=g, context={**ctx, "layer": li})
                    for e in range(m["load"].shape[1]):
                        run.track(float(m["load"][li, e]), name="router/expert_load", step=g,
                                  context={**ctx, "layer": li, "expert": e})
                run.track(s["dead_experts"], name="router/dead_experts", step=g, context=ctx)
        if progress:
            print(msg, flush=True)
        _dump(hist, cfg.label)

    if start == 0:
        do_eval(0)                                              # loss *before* any update in this run
    t0 = time.time()
    for step in range(start, cfg.total_steps):
        g = cfg.step_offset + step
        lr = lr_at(step, cfg)
        for grp in opt.param_groups:
            grp["lr"] = lr
        explore = cfg.explore_noise if step < cfg.explore_steps else 0.0
        for b in moe_blocks:
            b.mlp.explore_noise = explore
        x, y = data.get_batch("train", cfg.batch_size, cfg.block_size, device, data_gen)
        with amp_ctx():
            _, ce, aux = model(x, y)
            loss = ce
            if aux is not None:
                loss = ce + cfg.lb_coef * aux["lb"] + cfg.z_coef * aux["z"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        if aux is not None and cfg.bias_update > 0:             # loss-free balancing: nudge selection biases
            with torch.no_grad():
                for b, ld in zip(moe_blocks, aux["load"]):
                    b.mlp.route_bias += cfg.bias_update * torch.sign(1.0 / ld.numel() - ld)
        n_since += 1

        if (step + 1) % cfg.log_interval == 0 or step == 0:
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            now = time.time()
            tok_s = n_since * tok_per_step / max(now - t_last, 1e-9)
            t_last, n_since = now, 0
            ce_v, gn_v = ce.item(), gn.item()
            lb_v = aux["lb"].item() if aux is not None else 0.0
            z_v = aux["z"].item() if aux is not None else 0.0
            for k, v in (("step", g), ("loss", ce_v), ("lr", lr), ("grad_norm", gn_v),
                         ("lb", lb_v), ("z", z_v), ("tok_s", tok_s)):
                hist[k].append(v)
            if run:
                run.track(ce_v, name="loss", step=g, context={**ctx, "subset": "train_step"})
                run.track(lr, name="lr", step=g, context=ctx)
                run.track(gn_v, name="grad_norm", step=g, context=ctx)
                run.track(tok_s, name="tokens_per_sec", step=g, context=ctx)
                if aux is not None:
                    run.track(lb_v, name="router/lb_loss", step=g, context=ctx)
                    run.track(z_v, name="router/z_loss", step=g, context=ctx)
        if (step + 1) % cfg.eval_interval == 0 or step == cfg.total_steps - 1:
            do_eval(step + 1)
        if (step + 1) % cfg.ckpt_interval == 0 and step + 1 < cfg.total_steps:
            save_checkpoint(ckpt_path, model, opt, step + 1, hist, False, run_hash)

    for b in moe_blocks:
        b.mlp.explore_noise = 0.0
    hist["wall_clock_s"] = hist.get("wall_clock_s", 0.0) + time.time() - t0
    hist["final_val_loss"] = hist["val_loss"][-1]
    hist["final_train_loss"] = hist["train_loss"][-1]
    hist["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9 if device.startswith("cuda") else None
    if run:
        run["summary"] = {k: hist[k] for k in ("final_val_loss", "final_train_loss", "wall_clock_s")}
        run.close()
    save_checkpoint(ckpt_path, model, None, cfg.total_steps, hist, True, run_hash)
    _dump(hist, cfg.label)
    print(f"[{cfg.label}] done in {hist['wall_clock_s']:.0f}s | final val loss {hist['final_val_loss']:.4f}")
    return model, hist


def _dump(hist: dict, label: str):
    os.makedirs(results_dir(), exist_ok=True)
    with open(_results_path(label), "w") as f:
        json.dump(hist, f)
