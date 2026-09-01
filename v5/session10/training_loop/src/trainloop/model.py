"""A compact nanoGPT-style decoder-only transformer.

Deliberately ordinary: learned token + position embeddings, pre-norm LayerNorm,
causal multi-head self-attention, a 4x GELU MLP, and a weight-tied output head.
Nothing here is novel -- the point of the session is the *loop* around it, so the
model is kept close to Karpathy's nanoGPT so the 6*N FLOP rule and the parameter
count stay easy to reason about.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # [B, T, C] -> [B, n_head, T, head_dim]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(cfg.vocab_size, cfg.n_embd),
                "wpe": nn.Embedding(cfg.block_size, cfg.n_embd),
                "drop": nn.Dropout(cfg.dropout),
                "h": nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
                "ln_f": nn.LayerNorm(cfg.n_embd, bias=cfg.bias),
            }
        )
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # weight tying
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        """idx, targets: [B, T] int64. loss_mask: [B, T] in {0,1}, 1 = counts.

        reduction:
          "mean" - average cross-entropy over counted tokens (a scalar)
          "sum"  - summed cross-entropy over counted tokens (a scalar)
          "none" - per-token cross-entropy [B, T] (masked positions zeroed)
        """
        B, T = idx.shape
        assert T <= self.cfg.block_size, (
            f"sequence length {T} > block_size {self.cfg.block_size}"
        )
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        tok_emb = self.transformer.wte(idx)  # [B, T, n_embd]
        pos_emb = self.transformer.wpe(pos)  # [T, n_embd]
        x = self.transformer.drop(tok_emb + pos_emb)  # [B, T, n_embd]
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)  # [B, T, n_embd]
        logits = self.lm_head(x)  # [B, T, vocab_size]

        loss = None
        if targets is not None:
            per_tok = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction="none",
            ).view(B, T)  # [B, T]
            if loss_mask is not None:
                per_tok = per_tok * loss_mask
                denom = loss_mask.sum().clamp(min=1.0)
            else:
                denom = torch.tensor(float(B * T), device=logits.device)
            if reduction == "mean":
                loss = per_tok.sum() / denom
            elif reduction == "sum":
                loss = per_tok.sum()
            elif reduction == "none":
                loss = per_tok
            else:
                raise ValueError(reduction)
        return logits, loss

    # ------------------------------------------------------------------
    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # wte is tied to lm_head so it is only counted once above; the
            # position table is the only pure-embedding parameter to remove.
            n -= self.transformer.wpe.weight.numel()
        return n

    def flops_per_token(self) -> float:
        """Chinchilla/PaLM style estimate: 6*N for the matmuls plus the
        attention term 6 * n_layer * 2 * block_size * head_dim * n_head."""
        cfg = self.cfg
        n = self.num_params(non_embedding=True)
        head_dim = cfg.n_embd // cfg.n_head
        attn = 6 * cfg.n_layer * (2 * cfg.block_size * head_dim * cfg.n_head)
        return 6 * n + attn

    @torch.no_grad()
    def generate(
        self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0
    ):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
