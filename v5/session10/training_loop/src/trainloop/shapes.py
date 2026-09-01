"""One training step with every tensor shape printed and named.

`instrument_step` runs forward + backward once and prints three tables:

  1. activations  - the shape of every intermediate tensor in the step
  2. parameters   - the shape of every weight tensor, and what it maps
  3. gradients    - the shape of every .grad (must match its parameter)

plus a legend for the dimension letters (B, T, C, ...).
"""

from __future__ import annotations

import torch
import torch.nn as nn

LEGEND = {
    "B": "batch: independent sequences processed together in this step",
    "T": "time / sequence length: token positions, left-to-right",
    "C": "channels / n_embd: width of the residual stream",
    "H": "heads: parallel attention subspaces",
    "Dh": "head_dim = C / H: width of one attention head",
    "F": "4*C: hidden width of the MLP",
    "V": "vocab_size: number of distinct output symbols",
}


def _fmt(shape: torch.Size) -> str:
    return "[" + ", ".join(str(s) for s in shape) + "]"


def instrument_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> dict:
    cfg = model.cfg
    B, T = x.shape
    C, H = cfg.n_embd, cfg.n_head
    Dh, F, V = C // H, 4 * C, cfg.vocab_size

    records: list[tuple[str, str, str]] = []
    handles = []

    def hook(name: str, meaning: str):
        def _fn(_module, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if isinstance(t, torch.Tensor):
                records.append((name, _fmt(t.shape), meaning))
        return _fn

    tr = model.transformer
    handles.append(tr.wte.register_forward_hook(
        hook("wte(idx)  token embedding", "[B, T, C]  each token id -> a C-vector")))
    handles.append(tr.wpe.register_forward_hook(
        hook("wpe(pos)  position embedding", "[T, C]  each position -> a C-vector (broadcast over B)")))
    b0 = tr.h[0]
    handles.append(b0.ln_1.register_forward_hook(
        hook("block0.ln_1", "[B, T, C]  pre-attention LayerNorm, same shape")))
    handles.append(b0.attn.c_attn.register_forward_hook(
        hook("block0.attn.c_attn (q,k,v packed)", "[B, T, 3C]  one matmul produces q|k|v stacked")))
    handles.append(b0.attn.c_proj.register_forward_hook(
        hook("block0.attn.c_proj", "[B, T, C]  attention output projected back to the stream")))
    handles.append(b0.mlp.c_fc.register_forward_hook(
        hook("block0.mlp.c_fc", "[B, T, F]  MLP up-projection to 4*C")))
    handles.append(b0.mlp.c_proj.register_forward_hook(
        hook("block0.mlp.c_proj", "[B, T, C]  MLP down-projection back to C")))
    handles.append(tr.ln_f.register_forward_hook(
        hook("ln_f  final LayerNorm", "[B, T, C]")))
    handles.append(model.lm_head.register_forward_hook(
        hook("lm_head  output head", "[B, T, V]  a score for every vocab symbol at every position")))

    model.zero_grad(set_to_none=True)
    logits, loss = model(x, targets=y)
    for h in handles:
        h.remove()

    print("=" * 92)
    print("DIMENSION LEGEND")
    print("=" * 92)
    print(f"  B = {B:<6}  T = {T:<6}  C = {C:<6}  H = {H:<4}  Dh = {Dh:<4}  F = {F:<6}  V = {V}")
    for k, v in LEGEND.items():
        print(f"  {k:<3} {v}")

    print("\n" + "=" * 92)
    print("ACTIVATIONS  (shape of every intermediate tensor in the forward pass)")
    print("=" * 92)
    print(f"  {'tensor':<40} {'shape':<16} meaning")
    print(f"  {'-'*40} {'-'*16} {'-'*30}")
    print(f"  {'input  idx (x)':<40} {_fmt(x.shape):<16} [B, T]  the token ids fed in")
    print(f"  {'target y':<40} {_fmt(y.shape):<16} [B, T]  the next token at each position")
    for name, shape, meaning in records:
        print(f"  {name:<40} {shape:<16} {meaning}")
    # attention internals (not their own modules, so described explicitly)
    print(f"  {'  q,k,v  (each, after reshape)':<40} {_fmt(torch.Size([B, H, T, Dh])):<16} "
          f"[B, H, T, Dh]  c_attn output split 3 ways and folded into heads")
    print(f"  {'  attn scores q@k^T':<40} {_fmt(torch.Size([B, H, T, T])):<16} "
          f"[B, H, T, T]  how much each position attends to each earlier position")
    print(f"  {'logits':<40} {_fmt(logits.shape):<16} [B, T, V]")
    print(f"  {'per-token loss':<40} {_fmt(torch.Size([B, T])):<16} [B, T]  cross-entropy at each position")
    print(f"  {'loss (scalar)':<40} {_fmt(loss.shape):<16} []  one number: the mean over counted tokens")

    loss.backward()

    print("\n" + "=" * 92)
    print("PARAMETERS  (every trainable weight tensor: shape, count, what it maps)")
    print("=" * 92)
    print(f"  {'parameter':<34} {'shape':<16} {'#params':<12} meaning")
    print(f"  {'-'*34} {'-'*16} {'-'*12} {'-'*24}")
    total = 0
    param_meaning = {
        "transformer.wte.weight": "[V, C]  token embedding table (tied to lm_head)",
        "transformer.wpe.weight": "[T_max, C]  learned position table",
        "ln_1.weight": "[C]  LayerNorm gain",
        "ln_1.bias": "[C]  LayerNorm shift",
        "ln_2.weight": "[C]  LayerNorm gain",
        "ln_2.bias": "[C]  LayerNorm shift",
        "attn.c_attn.weight": "[3C, C]  projects the stream to q,k,v",
        "attn.c_attn.bias": "[3C]",
        "attn.c_proj.weight": "[C, C]  projects attention output back to the stream",
        "attn.c_proj.bias": "[C]",
        "mlp.c_fc.weight": "[F, C]  MLP up-projection",
        "mlp.c_fc.bias": "[F]",
        "mlp.c_proj.weight": "[C, F]  MLP down-projection",
        "mlp.c_proj.bias": "[C]",
        "ln_f.weight": "[C]  final LayerNorm gain",
        "ln_f.bias": "[C]  final LayerNorm shift",
        "lm_head.weight": "[V, C]  output head (tied to wte)",
    }
    for name, p in model.named_parameters():
        total += p.numel()
        meaning = next((v for k, v in param_meaning.items() if name.endswith(k)), "")
        print(f"  {name:<34} {_fmt(p.shape):<16} {p.numel():<12,} {meaning}")
    print(f"  {'-'*34}")
    print(f"  total trainable params: {total:,}")
    print(f"  non-embedding params (6*N term): {model.num_params(non_embedding=True):,}")

    print("\n" + "=" * 92)
    print("GRADIENTS  (every .grad after backward: shape must equal its parameter)")
    print("=" * 92)
    mismatches = 0
    none_grads = 0
    for name, p in model.named_parameters():
        if p.grad is None:
            none_grads += 1
            print(f"  {name:<34} grad is None  (did not participate in the loss)")
            continue
        ok = p.grad.shape == p.shape
        mismatches += int(not ok)
        flag = "OK" if ok else "SHAPE MISMATCH"
        print(f"  {name:<34} {_fmt(p.grad.shape):<16} {flag}")
    print(f"  {'-'*34}")
    print(f"  grads with wrong shape: {mismatches}   |   grads that are None: {none_grads}")

    return {"logits": logits, "loss": loss, "B": B, "T": T, "C": C, "V": V}
