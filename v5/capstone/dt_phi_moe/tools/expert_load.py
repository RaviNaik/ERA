"""Which experts does each arm use? Forward-only router statistics on a fixed set of token sequences.

For one arm (base | dt | lora | ft) the model reads the base model's own GSM8K replies (prompt + reply of eval_base*/gsm8k_generations.jsonl,
the same texts as the KL) and the router's top-2 choice is recorded at every layer and token. Reported per arm:
  - per-layer load entropy (in bits, maximum log2 E) and the largest expert share,
  - mean over layers of the share of tokens whose top-2 set is the same as the base model's (needs --ref, the base run's --save file),
  - the total variation distance of the load distribution to the base model's, mean over layers,
  - experts used by less than 1/(10 E) of the tokens (near-dead experts).

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/expert_load.py --tag base --model_dir ... --gens runs/<run>/eval_base_fast/gsm8k_generations.jsonl --save el_base.npz
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/expert_load.py --tag ft --ckpt_dir runs/<run>/train_ft ... --ref el_base.npz --save el_ft.npz --out el_ft.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from dtphi import official  # noqa: E402,F401


@torch.no_grad()
def route(model, seqs: list[list[int]], pad_id: int, batch: int = 8) -> np.ndarray:
    """Top-2 expert ids at every layer and (non-pad) token, in input order: int16 array [layers, tokens, 2]."""
    blocks = [m for m in model.modules() if m.__class__.__name__ == "PhiMoESparseMoeBlock"]
    per_layer: list[list[torch.Tensor]] = [[] for _ in blocks]
    cur: dict = {}
    hooks = []
    for li, blk in enumerate(blocks):
        def hook(_m, _inp, out, li=li):
            logits = out[1].float()                                  # (batch * seq, experts)
            top = logits.topk(2, dim=-1).indices.cpu()               # same set as the router's two choices
            per_layer[li].append(top[cur["keep"]])
        hooks.append(blk.register_forward_hook(hook))
    dev = next(model.parameters()).device
    try:
        for i in range(0, len(seqs), batch):
            chunk = seqs[i:i + batch]
            L = max(len(s) for s in chunk)
            ids = torch.tensor([s + [pad_id] * (L - len(s)) for s in chunk])
            att = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in chunk])
            cur["keep"] = att.reshape(-1).bool()
            model(input_ids=ids.to(dev), attention_mask=att.to(dev))
    finally:
        for h in hooks:
            h.remove()
    return torch.stack([torch.cat(p).to(torch.int16) for p in per_layer]).numpy()


def summarize(sel: np.ndarray, n_experts: int, ref: np.ndarray | None = None) -> dict:
    layers, tokens, _ = sel.shape
    load = np.stack([np.bincount(sel[l].reshape(-1), minlength=n_experts) for l in range(layers)]).astype(float)
    p = load / load.sum(axis=1, keepdims=True)
    ent = -(np.where(p > 0, p * np.log2(np.where(p > 0, p, 1)), 0)).sum(axis=1)
    out = {"layers": layers, "tokens": tokens, "experts": n_experts, "max_entropy_bits": math.log2(n_experts),
           "entropy_bits_mean": float(ent.mean()), "entropy_bits_min": float(ent.min()),
           "largest_share_mean": float(p.max(axis=1).mean()), "largest_share_max": float(p.max()),
           "near_dead_experts_total": int((p < 1 / (10 * n_experts)).sum())}
    if ref is not None:
        if ref.shape != sel.shape:
            raise ValueError(f"reference has shape {ref.shape}, this run {sel.shape}: not the same texts")
        a, b = np.sort(ref, axis=-1), np.sort(sel, axis=-1)
        out["same_top2_set_as_base"] = float((a == b).all(axis=-1).mean())
        out["same_top2_set_as_base_by_layer_min"] = float((a == b).all(axis=-1).mean(axis=1).min())
        pr = np.stack([np.bincount(ref[l].reshape(-1), minlength=n_experts) for l in range(layers)]).astype(float)
        pr /= pr.sum(axis=1, keepdims=True)
        out["load_total_variation_to_base_mean"] = float((0.5 * np.abs(p - pr).sum(axis=1)).mean())
    return out


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tag", required=True, choices=["base", "ft", "lora", "dt"])
    p.add_argument("--model_dir", required=True)
    p.add_argument("--ckpt_dir", default="")
    p.add_argument("--mask_dir", default="")
    p.add_argument("--gens", required=True, help="eval_base*/gsm8k_generations.jsonl (prompt_ids + response_ids)")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--n_gpus", type=int, default=1, help="0 = CPU (tests)")
    p.add_argument("--save", default="", help="npz with the top-2 choices (the base run's file is the --ref of the others)")
    p.add_argument("--ref", default="")
    p.add_argument("--out", default="")
    a = p.parse_args(argv)
    from dtphi.modeling import load_model, load_tokenizer
    tok = load_tokenizer(a.model_dir)
    recs = [json.loads(l) for l in Path(a.gens).read_text().splitlines() if l.strip()]
    recs = [r for r in recs if r["response_ids"]][:a.limit]
    seqs = [list(r["prompt_ids"]) + list(r["response_ids"]) for r in recs]
    model = load_model(a.model_dir, a.n_gpus)
    if a.tag in ("ft", "lora"):
        from dtphi.train_baseline import apply_ft, apply_lora
        print("weights:", (apply_ft if a.tag == "ft" else apply_lora)(model, a.ckpt_dir), flush=True)
    elif a.tag == "dt":
        import apply_mask  # official
        print("mask:", apply_mask.apply_mask(model=model, masks=apply_mask.load_mask(mask_dir=a.mask_dir)), flush=True)
    sel = route(model, seqs, tok.pad_token_id if tok.pad_token_id is not None else 0, a.batch)
    ref = np.load(a.ref)["sel"] if a.ref else None
    res = {"tag": a.tag, "sequences": len(seqs), **summarize(sel, model.config.num_local_experts, ref)}
    print(json.dumps(res, indent=1))
    if a.save:
        np.savez_compressed(a.save, sel=sel)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main()
