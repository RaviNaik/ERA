"""Summarise a run folder: base model vs Deletion Tuning mask, as results.json and results.md."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def build(run_dir: str) -> tuple[dict, str]:
    d = Path(run_dir)
    tr = _load(d / "train" / "train_summary.json")
    base = _load(d / "eval_base" / "summary.json")
    dt = _load(d / "eval_dt" / "summary.json")
    res: dict = {"run_dir": str(d), "train": None, "base": None, "dt": None}
    L = [f"# Deletion Tuning of Phi-mini-MoE-instruct on GSM8K (SFT): {d.name}", ""]
    if tr:
        a, w, mk = tr["args"], tr["wanda_prior"], tr["mask"]
        res["train"] = {k: tr[k] for k in ("train_samples", "seconds_per_step", "train_seconds", "peak_gpu_gib",
                                           "mask", "wanda_prior")}
        L += ["## Training (official Deletion Tuning code, commit " + tr["official_commit"][:7] + ")", "",
              f"- {a['train_steps']} steps, batch {a['train_batch_size']} x accum {a['grad_accum_steps']}, lr "
              f"{a['learning_rate']}, sharpness {a['sharpness_start']} -> {a['sharpness_end']}, weight_reg "
              f"{a['weight_reg']}, init_sparsity {a['init_sparsity']:.6g}, seed {a['seed']}",
              f"- {tr['train_samples']} GSM8K training samples (pool {a['train_pool']}), {w.get('sparse_layers')} "
              f"expert gate projections, {w.get('layers_without_scores')} without a Wanda score",
              f"- deleted at step 0: {w.get('deleted_frac_step0', float('nan')):.4%} of gate weights; at the end: "
              f"{mk['sparsity_ratio']:.4%} ({mk['zero_params']:,d} of {mk['total_params']:,d})",
              f"- {tr['train_seconds']:.0f} s training ({tr['seconds_per_step']} s per step); peak GPU memory GiB "
              f"{tr['peak_gpu_gib']}",
              f"- saved mask: {mk['compressed_size_mb']:.1f} MB (official packbits + zstd)", ""]
    rows = []

    def get(x, *path):
        for k in path:
            if not isinstance(x, dict) or k not in x:
                return None
            x = x[k]
        return x
    for name, s in (("base", base), ("dt", dt)):
        if s:
            res[name] = {k: s.get(k) for k in ("gsm8k", "gsm8k_chat", "forward_kl", "prior", "weights_zeroed")}
            res[name]["team10"] = s.get("team10")
    L += ["## Results", "", "| | base | Deletion Tuning | change |", "|---|---:|---:|---:|"]

    def row(label, b, t, pct=True):
        if b is None and t is None:
            return
        f = (lambda v: "n/a" if v is None else (f"{100 * v:.2f}" if pct else f"{v:.4f}"))
        ch = "" if b is None or t is None else (f"{100 * (t - b):+.2f}" if pct else f"{t - b:+.4f}")
        L.append(f"| {label} | {f(b)} | {f(t)} | {ch} |")
    row("GSM8K accuracy, training format (%)", get(base, "gsm8k", "accuracy"), get(dt, "gsm8k", "accuracy"))
    row("GSM8K accuracy, chat template (%)", get(base, "gsm8k_chat", "accuracy"), get(dt, "gsm8k_chat", "accuracy"))
    row("replies with '####', training format (%)", get(base, "gsm8k", "share_with_####"),
        get(dt, "gsm8k", "share_with_####"))
    for task in sorted((get(base, "prior", "scores") or get(dt, "prior", "scores") or {})):
        row(f"{task} (%)", get(base, "prior", "scores", task), get(dt, "prior", "scores", task))
    row("prior-task mean (%)", get(base, "prior", "prior_task_mean"), get(dt, "prior", "prior_task_mean"))
    for ds in ("indic_v1", "safety_v1"):
        row(f"team10 {ds} exact (%)", get(base, "team10", ds, "exact"), get(dt, "team10", ds, "exact"))
    kl = get(dt, "forward_kl", "forward_kl_per_token_mean")
    L.append(f"| forward KL per token (base -> masked, GSM8K) | 0 | {'n/a' if kl is None else f'{kl:.4f}'} | |")
    L += ["", "Forward KL is measured on the base model's own greedy replies to the evaluated GSM8K test questions.",
          "Prior-task numbers: lm-evaluation-harness, zero-shot, chat template, bf16; limit "
          f"{get(base, 'prior', 'limit') if get(base, 'prior', 'limit') is not None else get(dt, 'prior', 'limit')}"
          " (0 = full benchmarks).", ""]
    return res, "\n".join(L)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run_dir", required=True)
    a = p.parse_args(argv)
    res, md = build(a.run_dir)
    (Path(a.run_dir) / "results.json").write_text(json.dumps(res, indent=1))
    (Path(a.run_dir) / "results.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
