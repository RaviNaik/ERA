"""LoRA and full fine-tuning on the same GSM8K SFT task as the Deletion Tuning run, for the comparison table.

What is the same as in the DT run (train.py): the model and its split over the two GPUs, the training samples
(data.Gsm8kSplits: same pool, same `{question}\\n{answer}` format, loss on the answer tokens only), the number of steps
(400), the batch size (4), AdamW with a constant learning rate, gradient clipping at 1.0, gradient checkpointing,
deterministic MoE routing, bf16 autocast, and the loop itself (a line-for-line copy of the official train_task loop,
without the mask machinery). What differs on purpose: the learning rate (each method gets its own, see below) and the
order in which the shuffled batches arrive (the DT run draws random numbers for the Wanda prior first).

  --method lora   trainable: low-rank A, B matrices (fp32) next to frozen bf16 linear layers. Default target: the expert
                  gate projections `w1`, the same 512 matrices Deletion Tuning masks. `--lora_targets w1,q_proj,...`
                  adds more. Default r 16, alpha 32, lr 2e-4.
  --method ft     trainable: weights themselves, kept in bf16 (fp32 master weights and states for 7.65B parameters do
                  not fit on two 48 GB cards). The optimizer is AdamW with the moments stored in bf16 and every write to
                  a bf16 tensor rounded stochastically, because a plain bf16 update of 1e-5 is smaller than the bf16
                  spacing of most weights and would vanish. `--ft_scope all` trains every weight, `--ft_scope gate`
                  only the expert gate projections (the weights DT masks). Default lr 1e-5.

The learning rates are common defaults for these methods, not tuned on this task. Output folder: lora.safetensors +
lora_config.json (lora), or ft-*.safetensors + ft_index.json (ft), plus train_summary.json and train_log.jsonl."""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import official  # noqa: F401
from .data import Gsm8kSplits
from .modeling import load_model, load_tokenizer

LOG_EVERY = 10
SHARD_BYTES = 4 * 2**30
DEFAULTS = {"lora": {"learning_rate": 2e-4}, "ft": {"learning_rate": 1e-5}}


# ---------------------------------------------------------------------------------------------------------------------
# LoRA

class LoRALinear(nn.Module):
    """y = base(x) + (alpha / r) * B(A(x)); base frozen, A and B in fp32. B starts at zero, so the model starts
    exactly where the base model is."""

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base, self.r, self.scale = base, r, alpha / r
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32, device=dev))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32, device=dev))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        y = self.base(x)
        z = F.linear(F.linear(x, self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype))
        return y + z.to(y.dtype) * self.scale


def lora_target_names(model: nn.Module, targets: list[str]) -> list[str]:
    """Names of the nn.Linear modules whose last name part is one of `targets` (e.g. 'w1', 'q_proj')."""
    names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and n.split(".")[-1] in targets]
    if not names:
        raise ValueError(f"no linear layer is named any of {targets}")
    return names


def add_lora(model: nn.Module, targets: list[str], r: int, alpha: float) -> list[str]:
    names = lora_target_names(model, targets)
    for n in names:
        parent_name, _, child = n.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, LoRALinear(getattr(parent, child), r, alpha))
    return names


def save_lora(model: nn.Module, out: Path, names: list[str], r: int, alpha: float, targets: list[str]) -> dict:
    from safetensors.torch import save_file
    tensors = {}
    for n in names:
        m = model.get_submodule(n)
        tensors[f"{n}.lora_A"] = m.lora_A.detach().cpu().contiguous()
        tensors[f"{n}.lora_B"] = m.lora_B.detach().cpu().contiguous()
    save_file(tensors, str(out / "lora.safetensors"))
    cfg = {"r": r, "alpha": alpha, "targets": targets, "modules": names}
    (out / "lora_config.json").write_text(json.dumps(cfg, indent=1))
    return cfg


@torch.no_grad()
def apply_lora(model: nn.Module, ckpt_dir: str | Path) -> dict:
    """Merge saved adapters into the (plain, unwrapped) model: W <- round_to_weight_dtype(W + alpha/r * B A), the sum
    taken in fp32. This is how a LoRA model is deployed; it rounds the update once to the weight's dtype."""
    from safetensors.torch import load_file
    d = Path(ckpt_dir)
    cfg = json.loads((d / "lora_config.json").read_text())
    tensors = load_file(str(d / "lora.safetensors"))
    scale = cfg["alpha"] / cfg["r"]
    for n in cfg["modules"]:
        lin = model.get_submodule(n)
        delta = (tensors[f"{n}.lora_B"].float() @ tensors[f"{n}.lora_A"].float()) * scale
        w = lin.weight
        w.copy_((w.float() + delta.to(w.device)).to(w.dtype))
    return {"lora_modules_merged": len(cfg["modules"]), "r": cfg["r"], "alpha": cfg["alpha"]}


# ---------------------------------------------------------------------------------------------------------------------
# Full fine-tuning in bf16

def round_stochastic(x32: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Round an fp32 tensor to bf16 (or leave fp32): round up with probability equal to the discarded fraction, so
    the expected value is unchanged. (bf16 is the top 16 bits of an fp32.)"""
    if dtype == torch.float32:
        return x32
    if dtype != torch.bfloat16:
        return x32.to(dtype)
    bits = x32.contiguous().view(torch.int32)
    noise = torch.randint(0, 1 << 16, bits.shape, dtype=torch.int32, device=bits.device)
    return ((bits + noise) & -65536).view(torch.float32).to(torch.bfloat16)


class StochasticAdamW(torch.optim.Optimizer):
    """AdamW whose arithmetic is fp32 and whose state and weights may be bf16 (writes rounded stochastically). For fp32
    parameters it is plain AdamW. Same update rule as torch.optim.AdamW (decoupled weight decay)."""

    def __init__(self, params, lr=1e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["m"] = torch.zeros_like(p)
                    st["v"] = torch.zeros_like(p)
                st["step"] += 1
                t = st["step"]
                grad = p.grad.float()
                m = st["m"].float().mul_(b1).add_(grad, alpha=1 - b1)
                v = st["v"].float().mul_(b2).addcmul_(grad, grad, value=1 - b2)
                st["m"].copy_(round_stochastic(m, p.dtype))
                st["v"].copy_(round_stochastic(v, p.dtype))
                upd = (m / (1 - b1 ** t)) / ((v / (1 - b2 ** t)).sqrt() + g["eps"])
                w = p.float()
                if g["weight_decay"]:
                    w.mul_(1 - g["lr"] * g["weight_decay"])
                w.add_(upd, alpha=-g["lr"])
                p.copy_(round_stochastic(w, p.dtype))


def ft_params(model: nn.Module, scope: str) -> dict[str, nn.Parameter]:
    if scope == "all":
        return dict(model.named_parameters())
    if scope == "gate":
        sel = {n: p for n, p in model.named_parameters() if n.endswith(".w1.weight")}
        if not sel:
            raise ValueError("no expert gate projection (w1) found")
        return sel
    raise ValueError(f"unknown --ft_scope {scope!r}")


def save_ft(params: dict[str, nn.Parameter], out: Path, scope: str) -> dict:
    """Save the trained weights in shards of about 4 GiB (host memory: one shard at a time)."""
    from safetensors.torch import save_file
    index: dict[str, str] = {}
    shard: dict[str, torch.Tensor] = {}
    size, k = 0, 0

    def flush():
        nonlocal shard, size, k
        if shard:
            k += 1
            fn = f"ft-{k:05d}.safetensors"
            save_file(shard, str(out / fn))
            index.update({n: fn for n in shard})
            shard, size = {}, 0
    for n, p in params.items():
        t = p.detach().cpu().contiguous()
        if shard and size + t.numel() * t.element_size() > SHARD_BYTES:
            flush()
        shard[n] = t
        size += t.numel() * t.element_size()
    flush()
    info = {"scope": scope, "tensors": len(index), "weight_map": index}
    (out / "ft_index.json").write_text(json.dumps(info, indent=1))
    return info


@torch.no_grad()
def apply_ft(model: nn.Module, ckpt_dir: str | Path) -> dict:
    """Overwrite the trained tensors of `model` with the saved ones."""
    from safetensors.torch import load_file
    d = Path(ckpt_dir)
    info = json.loads((d / "ft_index.json").read_text())
    params = dict(model.named_parameters())
    missing = [n for n in info["weight_map"] if n not in params]
    if missing:
        raise KeyError(f"{len(missing)} saved tensors are not in the model, e.g. {missing[:3]}")
    n_el = 0
    for fn in sorted(set(info["weight_map"].values())):
        for n, t in load_file(str(d / fn)).items():
            params[n].copy_(t)
            n_el += t.numel()
    return {"ft_tensors_loaded": len(info["weight_map"]), "ft_elements": n_el, "scope": info["scope"]}


# ---------------------------------------------------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--method", required=True, choices=["lora", "ft"])
    p.add_argument("--model_dir", required=True)
    p.add_argument("--data_dir", required=True, help="folder holding main/train-*.parquet")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_gpus", type=int, default=2)
    p.add_argument("--train_steps", type=int, default=400)
    p.add_argument("--train_batch_size", type=int, default=4)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=None, help="default: 2e-4 (lora), 1e-5 (ft)")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train_pool", type=int, default=2048)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=32.0)
    p.add_argument("--lora_targets", default="w1", help="comma separated last names of the linear layers to adapt")
    p.add_argument("--ft_scope", default="all", choices=["all", "gate"])
    a = p.parse_args(argv)
    if a.learning_rate is None:
        a.learning_rate = DEFAULTS[a.method]["learning_rate"]
    return a


def run(args: argparse.Namespace) -> dict:
    official.verify_vendored()
    import sft_data  # official
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from torch.utils.data import DataLoader

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
    log = logging.getLogger("baseline")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    use_cuda = args.n_gpus > 0
    accelerator = Accelerator(mixed_precision="bf16" if use_cuda else "no", cpu=not use_cuda)
    set_seed(args.seed)
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True

    t_all = time.time()
    tok = load_tokenizer(args.model_dir)
    splits = Gsm8kSplits(tok, args.data_dir, args.seed, args.train_pool, args.max_len)
    train_set = splits.train_set("gsm8k")
    if len(train_set) < args.train_batch_size:
        raise ValueError("fewer training samples than the batch size")
    print(f"GSM8K training pool: {len(train_set)} samples ({splits.dropped} longer than {args.max_len} tokens dropped)",
          flush=True)

    from .phimoe import modeling_phimoe
    modeling_phimoe.DETERMINISTIC_MOE = True   # as in train.py: the evaluated model routes deterministically
    model = load_model(args.model_dir, args.n_gpus)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    n_total = sum(p.numel() for p in model.parameters())
    for p in model.parameters():
        p.requires_grad = False

    targets = [t for t in args.lora_targets.split(",") if t]
    lora_names: list[str] = []
    ft_set: dict[str, nn.Parameter] = {}
    if args.method == "lora":
        lora_names = add_lora(model, targets, args.lora_r, args.lora_alpha)
        trainable = [p for n, p in model.named_parameters() if n.endswith((".lora_A", ".lora_B"))]
        optim = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        ft_set = ft_params(model, args.ft_scope)
        trainable = list(ft_set.values())
        optim = StochasticAdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    for p in trainable:
        p.requires_grad = True
    n_train = sum(p.numel() for p in trainable)
    print(f"{args.method}: {n_train:,d} trainable of {n_total:,d} parameters ({n_train / n_total:.4%})", flush=True)

    collate_fn = partial(sft_data.collate_train, pad_token_id=tok.pad_token_id)
    train_loader = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn,
                              drop_last=True, num_workers=0 if len(train_set) < 256 else 4)
    iters = args.train_steps * args.grad_accum_steps
    log.info(f"[{args.method}] training: {len(train_set)} samples, batch_size={args.train_batch_size}, {iters} iters, "
             f"{n_train:,d} trainable params")
    losses: list[dict] = []
    model.train()
    optim.zero_grad()
    t_train = time.time()
    curr = 0
    with torch.enable_grad():
        while curr < iters:
            for batch in train_loader:
                input_ids = batch["input_ids"].to(accelerator.device)
                attention_mask = batch["attention_mask"].to(accelerator.device)
                label_ids = batch["label_ids"].to(accelerator.device)
                with accelerator.autocast():
                    seq_len = input_ids.shape[1]
                    seq_range = torch.arange(seq_len, device=input_ids.device, dtype=torch.int64)
                    offsets = (seq_len - attention_mask.sum(dim=1)).unsqueeze(1)
                    position_ids = (seq_range.unsqueeze(0) - offsets).clamp(min=0)   # left padding, as official
                    loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=label_ids,
                                 position_ids=position_ids).loss
                accelerator.backward(loss / args.grad_accum_steps)
                if (curr + 1) % args.grad_accum_steps == 0:
                    accelerator.clip_grad_norm_(trainable, args.max_grad_norm)
                    optim.step()
                    optim.zero_grad()
                if (curr + 1) % LOG_EVERY == 0:
                    log.info(f"[{args.method}] iter {curr + 1}/{iters} | loss={loss.item():.4f}")
                    losses.append({"iter": curr + 1, "of": iters, "loss": float(loss.item()),
                                   "elapsed_s": round(time.time() - t_all, 1)})
                curr += 1
                if curr >= iters:
                    break
    t_train = time.time() - t_train

    if args.method == "lora":
        ckpt = save_lora(model, out, lora_names, args.lora_r, args.lora_alpha, targets)
    else:
        ckpt = {k: v for k, v in save_ft(ft_set, out, args.ft_scope).items() if k != "weight_map"}
    peak = {f"cuda:{i}": round(torch.cuda.max_memory_allocated(i) / 2**30, 1)
            for i in range(torch.cuda.device_count())} if use_cuda else {}
    (out / "train_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in losses))
    summary = {"method": args.method, "args": vars(args), "train_samples": len(train_set),
               "dropped_too_long": splits.dropped, "total_params": n_total, "trainable_params": n_train,
               "trainable_share": n_train / n_total, "checkpoint": ckpt, "train_seconds": round(t_train, 1),
               "total_seconds": round(time.time() - t_all, 1),
               "seconds_per_step": round(t_train / args.train_steps, 2), "peak_gpu_gib": peak,
               "final_loss": losses[-1]["loss"] if losses else None, "command": " ".join(sys.argv)}
    (out / "train_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"done: {args.method}, {n_train:,d} trainable parameters, {t_train:.0f} s", flush=True)
    return summary


def main(argv=None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
