"""nanoGPT (session 11 architecture) with an optional Mixture-of-Experts MLP.

* ``n_experts == 1``  -> every block uses the classic dense ``MLP`` ("linear layers").
* ``n_experts  > 1``  -> every ``moe_every``-th block swaps its MLP for ``MoEMLP``.

The MoE layer is *dropless* top-k routing (no capacity factor, no token dropping)
with gate weights renormalised over the selected experts.  That last detail is
what makes sparse upcycling exact: if all experts are copies of the dense MLP the
mixture output equals the dense MLP output, whatever the router says.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50304
    block_size: int = 512
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0
    bias: bool = True
    # MoE
    n_experts: int = 1
    top_k: int = 2
    moe_every: int = 2
    score_fn: str = "softmax"      # router score function: softmax | sigmoid


def is_moe_layer(cfg: GPTConfig, i: int) -> bool:
    return cfg.n_experts > 1 and (i % cfg.moe_every) == (cfg.moe_every - 1)


class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.n_head, self.n_embd, self.dropout = cfg.n_head, cfg.n_embd, cfg.dropout

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0,
                                           is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """The dense feed-forward block: Linear(d, 4d) -> GELU -> Linear(4d, d)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x)))), None


class MoEMLP(nn.Module):
    """Top-k routed mixture of ``n_experts`` MLPs (each identical in shape to the dense MLP).

    Expert weights are stacked so that ``w1[e]`` has the exact layout of the dense
    ``c_fc.weight`` ([4d, d]) and ``w2[e]`` that of ``c_proj.weight`` ([d, 4d]) —
    upcycling is then a plain copy.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        E, d, h = cfg.n_experts, cfg.n_embd, 4 * cfg.n_embd
        self.n_experts, self.top_k, self.score_fn = E, cfg.top_k, cfg.score_fn
        # loss-free balancing: added to the scores *only when choosing* the top-k (never to the gate weights)
        self.register_buffer("route_bias", torch.zeros(E))
        self.explore_noise = 0.0        # >0 (training only): sample experts ~ Gumbel-top-k instead of hard top-k
        self.router = nn.Linear(d, E, bias=False)
        self.w1 = nn.Parameter(torch.empty(E, h, d))
        self.w2 = nn.Parameter(torch.empty(E, d, h))
        self.b1 = nn.Parameter(torch.zeros(E, h)) if cfg.bias else None
        self.b2 = nn.Parameter(torch.zeros(E, d)) if cfg.bias else None
        self.dropout = nn.Dropout(cfg.dropout)
        for w in (self.router.weight, self.w1, self.w2):     # usable standalone, not only inside GPT
            nn.init.normal_(w, mean=0.0, std=0.02)

    def forward(self, x):
        B, T, C = x.shape
        E, k = self.n_experts, self.top_k
        xf = x.reshape(-1, C)
        N = xf.shape[0]

        logits = self.router(xf).float()                       # [N, E] router in fp32
        score = logits.softmax(-1) if self.score_fn == "softmax" else logits.sigmoid()
        sel = score + self.route_bias                          # selection score (bias steers choice only)
        if self.training and self.explore_noise > 0:           # probabilistic selection (Gumbel-top-k)
            u = torch.rand_like(sel).clamp(1e-9, 1 - 1e-9)
            sel = sel.clamp_min(1e-9).log() + self.explore_noise * -torch.log(-torch.log(u))
        topi = sel.topk(k, dim=-1).indices                     # [N, k]
        topv = score.gather(1, topi)
        gates = topv / topv.sum(-1, keepdim=True)              # weights from the *unbiased* scores, renormalised
        probs = score / score.sum(-1, keepdim=True)            # normalised scores, for the balance statistics

        # ---- dispatch: sort the N*k (token, expert) pairs by expert --------------------
        flat_e = topi.reshape(-1)
        order = flat_e.argsort(stable=True)
        tok_idx = order // k
        counts = torch.bincount(flat_e, minlength=E)
        sorted_x = xf[tok_idx]
        pieces, start = [], 0
        for e, n in enumerate(counts.tolist()):
            if n:
                xe = sorted_x[start:start + n]
                h = F.gelu(F.linear(xe, self.w1[e], None if self.b1 is None else self.b1[e]))
                pieces.append(F.linear(h, self.w2[e], None if self.b2 is None else self.b2[e]))
                start += n
        out_sorted = torch.cat(pieces, 0)
        w = gates.reshape(-1)[order].unsqueeze(-1).to(out_sorted.dtype)
        y = torch.zeros(N, C, dtype=out_sorted.dtype, device=x.device)
        y = y.index_add(0, tok_idx, out_sorted * w)

        # ---- auxiliary signals --------------------------------------------------------
        load = counts.float() / (N * k)                        # fraction of routed slots per expert
        importance = probs.mean(0)                             # mean router prob per expert
        aux = {
            "lb": E * (load.detach() * importance).sum(),      # Switch loss; 1.0 when perfectly balanced
            "z": torch.logsumexp(logits, -1).pow(2).mean(),
            "load": load.detach(),
            "entropy": -(probs * probs.clamp_min(1e-9).log()).sum(-1).mean().detach(),
        }
        return self.dropout(y.view(B, T, C)), aux


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig, moe: bool):
        super().__init__()
        self.ln_1 = LayerNorm(cfg.n_embd, cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = LayerNorm(cfg.n_embd, cfg.bias)
        self.mlp = MoEMLP(cfg) if moe else MLP(cfg)
        self.is_moe = moe

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        m, aux = self.mlp(self.ln_2(x))
        return x + m, aux


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
            drop=nn.Dropout(cfg.dropout),
            h=nn.ModuleList([Block(cfg, is_moe_layer(cfg, i)) for i in range(cfg.n_layer)]),
            ln_f=LayerNorm(cfg.n_embd, cfg.bias),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight      # weight tying

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():                  # GPT-2 scaled residual init
            if pn.endswith("c_proj.weight") or pn.endswith("mlp.w2"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, MoEMLP):
            nn.init.normal_(m.w1, mean=0.0, std=0.02)
            nn.init.normal_(m.w2, mean=0.0, std=0.02)

    def moe_blocks(self):
        return [b for b in self.transformer.h if b.is_moe]

    def forward(self, idx, targets=None):
        """Returns ``(logits, ce_loss, aux)``; ``aux`` is None for a fully dense model."""
        B, T = idx.size()
        assert T <= self.cfg.block_size
        pos = torch.arange(T, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        auxes = []
        for block in self.transformer.h:
            x, a = block(x)
            if a is not None:
                auxes.append(a)
        logits = self.lm_head(self.transformer.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        aux = None
        if auxes:
            aux = {
                "lb": torch.stack([a["lb"] for a in auxes]).mean(),
                "z": torch.stack([a["z"] for a in auxes]).mean(),
                "load": torch.stack([a["load"] for a in auxes]),          # [L_moe, E]
                "entropy": torch.stack([a["entropy"] for a in auxes]),    # [L_moe]
            }
        return logits, loss, aux

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=50):
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.block_size:]
            logits = self(ctx)[0][:, -1, :] / max(temperature, 1e-6)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], dim=1)
        self.train(was_training)
        return idx


def param_report(model: GPT) -> dict:
    """Total vs *active-per-token* parameters, plus a component breakdown."""
    cfg = model.cfg
    groups: dict[str, int] = {}
    expert_params = 0
    for n, p in model.named_parameters():
        if n.startswith("transformer.wte") or n == "lm_head.weight":
            key = "token_embedding (tied)"
        elif n.startswith("transformer.wpe"):
            key = "position_embedding"
        elif ".attn." in n:
            key = "attention"
        elif n.endswith("router.weight"):
            key = "router"
        elif ".mlp." in n:
            key = "moe_experts" if (".mlp.w" in n or ".mlp.b" in n) else "dense_mlp"
            if key == "moe_experts":
                expert_params += p.numel()
        else:
            key = "layernorm"
        groups[key] = groups.get(key, 0) + p.numel()
    total = sum(groups.values())
    active = total - expert_params * (1 - cfg.top_k / max(cfg.n_experts, 1)) if expert_params else total
    return {"total": total, "active": int(active), "groups": groups}
