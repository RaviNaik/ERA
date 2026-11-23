"""Summarise a run folder: base model vs Deletion Tuning, LoRA and full FT (those that were run), as results.md."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


VARIANTS = (("dt", "Deletion Tuning"), ("lora", "LoRA"), ("ft", "Full FT"))


def build(run_dir: str, eval_suffix: str = "") -> tuple[dict, str]:
    """Base against every evaluated variant (Deletion Tuning, LoRA, full FT); only the ones that exist get a column."""
    d = Path(run_dir)
    tr = _load(d / "train" / "train_summary.json")
    base = _load(d / f"eval_base{eval_suffix}" / "summary.json")
    var = {t: _load(d / f"eval_{t}{eval_suffix}" / "summary.json") for t, _ in VARIANTS}
    var = {t: s for t, s in var.items() if s}
    names = dict(VARIANTS)
    res: dict = {"run_dir": str(d), "train": None, "base": None, **{t: None for t, _ in VARIANTS}}
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
    for t in ("lora", "ft"):
        b = _load(d / f"train_{t}" / "train_summary.json")
        if b and t in var:
            a = b["args"]
            res[f"train_{t}"] = {k: b[k] for k in ("trainable_params", "total_params", "train_seconds",
                                                   "seconds_per_step", "peak_gpu_gib", "final_loss")}
            extra = (f"r {a['lora_r']}, alpha {a['lora_alpha']}, targets {a['lora_targets']}" if t == "lora"
                     else f"scope {a['ft_scope']}, bf16 weights, stochastic rounding")
            L += [f"## Training: {names[t]}", "",
                  f"- {a['train_steps']} steps, batch {a['train_batch_size']} x accum {a['grad_accum_steps']}, lr "
                  f"{a['learning_rate']}, seed {a['seed']}, {extra}",
                  f"- {b['trainable_params']:,d} trainable parameters ({b['trainable_share']:.4%} of "
                  f"{b['total_params']:,d}); {b['train_seconds']:.0f} s ({b['seconds_per_step']} s per step); peak GPU "
                  f"memory GiB {b['peak_gpu_gib']}", ""]

    def get(x, *path):
        for k in path:
            if not isinstance(x, dict) or k not in x:
                return None
            x = x[k]
        return x
    if base:
        res["base"] = {k: base.get(k) for k in ("gsm8k", "gsm8k_chat", "prior")} | {"team10": base.get("team10")}
    for t, s in var.items():
        res[t] = {k: s.get(k) for k in ("gsm8k", "gsm8k_chat", "forward_kl", "prior", "weights_zeroed",
                                        "weights_changed")} | {"team10": s.get("team10")}
    cols = [t for t, _ in VARIANTS if t in var]
    head = ["", "base"] + [names[t] for t in cols] + [f"change ({names[t]})" if len(cols) > 1 else "change"
                                                      for t in cols]
    L += ["## Results", "", "| " + " | ".join(head) + " |", "|" + "---|" + "---:|" * (len(head) - 1)]

    def row(label, path, pct=True):
        b = get(base, *path)
        vs = [get(var[t], *path) for t in cols]
        if b is None and all(v is None for v in vs):
            return
        f = (lambda v: "n/a" if v is None else (f"{100 * v:.2f}" if pct else f"{v:.4f}"))
        ch = ["" if b is None or v is None else (f"{100 * (v - b):+.2f}" if pct else f"{v - b:+.4f}") for v in vs]
        L.append("| " + " | ".join([label, f(b), *map(f, vs), *ch]) + " |")
    row("GSM8K accuracy, training format (%)", ("gsm8k", "accuracy"))
    row("GSM8K accuracy, chat template (%)", ("gsm8k_chat", "accuracy"))
    row("replies with '####', training format (%)", ("gsm8k", "share_with_####"))
    scores = get(base, "prior", "scores") or next((get(var[t], "prior", "scores") for t in cols
                                                   if get(var[t], "prior", "scores")), {}) or {}
    for task in sorted(scores):
        row(f"{task} (%)", ("prior", "scores", task))
    row("prior-task mean (%)", ("prior", "prior_task_mean"))
    for ds in ("indic_v1", "safety_v1"):
        row(f"team10 {ds} exact (%)", ("team10", ds, "exact"))
    kls = [get(var[t], "forward_kl", "forward_kl_per_token_mean") for t in cols]
    L.append("| " + " | ".join(["forward KL per token (base -> tuned, GSM8K)", "0",
                                *["n/a" if k is None else f"{k:.4f}" for k in kls], *[""] * len(cols)]) + " |")
    limit = get(base, "prior", "limit")
    if limit is None:
        limit = next((get(var[t], "prior", "limit") for t in cols if get(var[t], "prior", "limit") is not None), None)
    L += ["", "Forward KL is measured on the base model's own greedy replies to the evaluated GSM8K test questions.",
          f"Prior-task numbers: lm-evaluation-harness, zero-shot, chat template, bf16; limit {limit}"
          " (0 = full benchmarks).", ""]
    pj = d / f"paired{eval_suffix}.md"
    if pj.exists():
        L += [pj.read_text()]
    return res, "\n".join(L)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run_dir", required=True)
    p.add_argument("--eval_suffix", default="", help="the --eval_suffix the evaluations were run with")
    a = p.parse_args(argv)
    res, md = build(a.run_dir, a.eval_suffix)
    (Path(a.run_dir) / f"results{a.eval_suffix}.json").write_text(json.dumps(res, indent=1))
    (Path(a.run_dir) / f"results{a.eval_suffix}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
