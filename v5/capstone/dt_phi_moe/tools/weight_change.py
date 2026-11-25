"""Did the full fine-tuning move the weights, and does a bf16 update of 1e-5 survive the optimizer?

1) Weights: compare the saved fine-tuned tensors (train_ft/ft_index.json + shards) with the base model, by weight group:
   share of elements that differ, mean |change|, mean |weight|, and the change in units of the bf16 spacing at that weight.
       .venv/bin/python tools/weight_change.py --model_dir assets/phi-mini-moe-instruct --ckpt_dir runs/<run>/train_ft [--every 8]
2) Optimizer probe (CPU, seconds): the same StochasticAdamW as the training, on a bf16 tensor, constant gradient, lr 1e-5, 400 steps,
   against plain bf16 AdamW and fp32. Reports how much of the intended movement (steps x lr) is realised on average, and the spread.
       .venv/bin/python tools/weight_change.py --probe
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402


def group_of(name: str) -> str:
    if "embed_tokens" in name:
        return "embeddings"
    if "lm_head" in name:
        return "lm_head"
    if ".experts." in name:
        return "experts " + name.rsplit(".", 2)[-2]
    if name.endswith(".gate.weight") or ".gate." in name:
        return "router"
    if "self_attn" in name:
        return "attention"
    return "norms and other"


def weight_map(d: Path) -> dict[str, str]:
    if (d / "ft_index.json").exists():
        return json.loads((d / "ft_index.json").read_text())["weight_map"]
    if (d / "model.safetensors.index.json").exists():
        return json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
    from safetensors import safe_open
    with safe_open(str(d / "model.safetensors"), framework="pt") as f:
        return {k: "model.safetensors" for k in f.keys()}


def bf16_spacing(x: torch.Tensor) -> torch.Tensor:
    """Distance to the next bf16 number at |x| (8 bits of significand): 2^(floor(log2|x|) - 7); 0 where x == 0."""
    m, e = torch.frexp(x.float())
    ulp = torch.ldexp(torch.ones_like(m), e - 8)
    return torch.where(x == 0, torch.zeros_like(ulp), ulp)


@torch.no_grad()
def compare_weights(model_dir: str, ckpt_dir: str, every: int = 1, max_tensors: int = 0) -> dict:
    from safetensors import safe_open
    base_dir, ft_dir = Path(model_dir), Path(ckpt_dir)
    bmap, fmap = weight_map(base_dir), weight_map(ft_dir)
    names = sorted(fmap)[::max(every, 1)]
    if max_tensors:
        names = names[:max_tensors]
    stats: dict[str, dict] = {}
    missing = [n for n in names if n not in bmap]
    handles: dict[Path, object] = {}

    def get(d: Path, fn: str, name: str) -> torch.Tensor:
        h = handles.get(d / fn)
        if h is None:
            h = handles[d / fn] = safe_open(str(d / fn), framework="pt")
        return h.get_tensor(name)

    for n in names:
        if n in missing:
            continue
        b, f = get(base_dir, bmap[n], n).float(), get(ft_dir, fmap[n], n).float()
        d = (f - b).abs()
        s = stats.setdefault(group_of(n), dict(tensors=0, elements=0, changed=0, sum_delta=0.0, sum_w=0.0, sum_ulps=0.0))
        s["tensors"] += 1
        s["elements"] += b.numel()
        s["changed"] += int((f != b).sum())
        s["sum_delta"] += float(d.sum())
        s["sum_w"] += float(b.abs().sum())
        ulp = bf16_spacing(b)
        s["sum_ulps"] += float((d / ulp.clamp_min(1e-30) * (ulp > 0)).sum())
    out = {}
    for g, s in sorted(stats.items()):
        out[g] = {"tensors": s["tensors"], "elements": s["elements"], "share_changed": s["changed"] / s["elements"],
                  "mean_abs_change": s["sum_delta"] / s["elements"], "mean_abs_weight": s["sum_w"] / s["elements"],
                  "mean_change_in_bf16_spacings": s["sum_ulps"] / s["elements"]}
    return {"groups": out, "tensors_compared": len(names) - len(missing), "missing_in_base": len(missing)}


@torch.no_grad()
def optimizer_probe(n: int = 200_000, steps: int = 400, lr: float = 1e-5, wstd: float = 0.02, seed: int = 0) -> dict:
    """Constant gradient: every Adam step is -lr, so the intended movement is steps x lr. Noisy gradient (random sign, same size): the
    intended movement is 0 and what is measured is the noise that the rounding adds."""
    from dtphi.train_baseline import StochasticAdamW
    res = {}
    for mode in ("constant_gradient", "random_sign_gradient"):
        for name in ("stochastic_bf16 (training)", "plain_adamw_bf16", "fp32_reference"):
            torch.manual_seed(seed)
            w0 = (torch.randn(n) * wstd).to(torch.bfloat16)
            p = torch.nn.Parameter(w0.clone() if "bf16" in name else w0.float())
            opt = StochasticAdamW([p], lr=lr) if name.startswith("stochastic") else torch.optim.AdamW([p], lr=lr, weight_decay=0.0)
            gen = torch.Generator().manual_seed(seed + 1)
            for _ in range(steps):
                g = torch.ones(n) if mode == "constant_gradient" else (torch.randint(0, 2, (n,), generator=gen) * 2 - 1).float()
                p.grad = g.to(p.dtype)
                with torch.enable_grad():
                    pass
                opt.step()
            moved = (w0.float() - p.detach().float())          # positive = moved down, the direction of a +1 gradient
            intended = steps * lr
            res[f"{mode} / {name}"] = {
                "share_of_elements_changed": float((moved != 0).float().mean()),
                "mean_movement_over_intended": float(moved.mean() / intended),
                "spread_(std)_over_intended": float(moved.std() / intended)}
    return res


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model_dir", default="")
    p.add_argument("--ckpt_dir", default="")
    p.add_argument("--every", type=int, default=1, help="use every N-th tensor (speed); 1 = all")
    p.add_argument("--max_tensors", type=int, default=0)
    p.add_argument("--probe", action="store_true", help="run the optimizer probe (no model needed)")
    p.add_argument("--out", default="")
    a = p.parse_args(argv)
    out = {}
    if a.probe:
        out["optimizer_probe"] = optimizer_probe()
        for k, v in out["optimizer_probe"].items():
            print(f"{k:60s} changed {100 * v['share_of_elements_changed']:5.1f}%  mean movement {v['mean_movement_over_intended']:6.3f} x intended  spread {v['spread_(std)_over_intended']:6.3f}")
    if a.model_dir and a.ckpt_dir:
        out["weights"] = compare_weights(a.model_dir, a.ckpt_dir, a.every, a.max_tensors)
        print(f"{out['weights']['tensors_compared']} tensors compared ({out['weights']['missing_in_base']} not found in the base model)")
        print(f"{'group':20s} {'tensors':>8s} {'changed':>9s} {'mean|chg|':>11s} {'mean|w|':>10s} {'chg/spacing':>12s}")
        for g, s in out["weights"]["groups"].items():
            print(f"{g:20s} {s['tensors']:8d} {100 * s['share_changed']:8.2f}% {s['mean_abs_change']:11.3e} {s['mean_abs_weight']:10.3e} {s['mean_change_in_bf16_spacings']:12.3f}")
    if not out:
        p.error("give --probe and/or --model_dir and --ckpt_dir")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
