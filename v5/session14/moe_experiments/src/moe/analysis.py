"""Post-hoc analysis helpers: routing collection, expert similarity, text samples."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .model import GPT


@torch.no_grad()
def collect_routing(model: GPT, data, n_batches: int, batch_size: int, block_size: int, device: str,
                    seed: int = 7) -> dict:
    """Run ``model`` on validation text and record each token's top-1 expert per MoE layer.

    Returns tokens [N], top1 [L, N], and the mean router probability [L, E].
    """
    model.eval()
    moe_ids = [i for i, b in enumerate(model.transformer.h) if b.is_moe]
    store = {i: [] for i in moe_ids}
    hooks = []
    for i in moe_ids:
        r = model.transformer.h[i].mlp.router
        hooks.append(r.register_forward_hook(lambda m, inp, out, i=i: store[i].append(out.float().cpu())))
    gen = torch.Generator().manual_seed(seed)
    toks = []
    for _ in range(n_batches):
        x, y = data.get_batch("val", batch_size, block_size, device, gen)
        model(x, y)
        toks.append(x.reshape(-1).cpu())
    for h in hooks:
        h.remove()
    logits = [torch.cat(store[i]) for i in moe_ids]                      # each [N, E]
    return {"tokens": torch.cat(toks).numpy(),
            "top1": torch.stack([l.argmax(-1) for l in logits]).numpy(),
            "mean_prob": torch.stack([l.softmax(-1).mean(0) for l in logits]).numpy(),
            "layers": moe_ids}


@torch.no_grad()
def expert_similarity(model: GPT) -> list[np.ndarray]:
    """Per MoE layer: [E, E] cosine similarity between experts' flattened (w1, w2).

    Right after upcycling this is all ones; it drops as the experts specialise."""
    out = []
    for b in model.moe_blocks():
        m = b.mlp
        v = torch.cat([m.w1.flatten(1), m.w2.flatten(1)], 1).float()
        v = F.normalize(v, dim=1)
        out.append((v @ v.T).cpu().numpy())
    return out


@torch.no_grad()
def expert_drift(moe: GPT, dense: GPT) -> list[np.ndarray]:
    """Per MoE layer: ||W_e - W_dense|| / ||W_dense|| for each expert (how far each copy moved)."""
    res = []
    for i, b in enumerate(moe.transformer.h):
        if not b.is_moe:
            continue
        d = dense.transformer.h[i].mlp
        ref = torch.cat([d.c_fc.weight.flatten(), d.c_proj.weight.flatten()]).float()
        m = b.mlp
        cur = torch.cat([m.w1.flatten(1), m.w2.flatten(1)], 1).float()
        res.append(((cur - ref.to(cur.device)).norm(dim=1) / ref.norm()).cpu().numpy())
    return res


def make_decoder(dataset: str):
    if dataset == "synthetic":
        return lambda ids: " ".join(f"<{int(i)}>" for i in ids)
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    return lambda ids: enc.decode([int(i) for i in ids if int(i) < enc.n_vocab])


def make_encoder(dataset: str):
    if dataset == "synthetic":
        return lambda s: [1, 2, 3]
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    return lambda s: enc.encode_ordinary(s)
