"""A small decoder-only transformer.

Deliberately kept simple: RMSNorm, pre-norm residual stream, a SwiGLU FFN,
and ordinary multi-head causal self-attention. The model returns the
*hidden state* h [B, T, D] only. It does NOT compute logits — the output
head is a separate module you attach afterwards, exactly as in the
assignment's harness:

    hidden = model(tokens)
    logits = output_head(hidden)

Keeping the head separate is what lets us compare tied vs. untied
parameter counts, and swap in a second head for the t+2 prediction in
Part 2, without touching the trunk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int | None = None  # if None, derived SwiGLU-style from d_model
    max_seq_len: int = 512
    dropout: float = 0.0

    def __post_init__(self):
        if self.d_ff is None:
            # SwiGLU has 3 matrices instead of 2, so d_ff shrinks to keep
            # the parameter budget close to a plain 4x two-matrix FFN.
            # 2 * 4D == 3 * d_ff  =>  d_ff = 8D/3, rounded to a multiple of 64.
            raw = int(8 * self.d_model / 3)
            self.d_ff = (raw + 63) // 64 * 64


class RMSNorm(nn.Module):
    """RMSNorm: divide by root-mean-square, scale. No mean-centering, no bias."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class SwiGLU(nn.Module):
    """SwiGLU FFN: down(silu(gate(h)) * up(h)). Three matrices, one gate."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class CausalSelfAttention(nn.Module):
    """Ordinary multi-head causal self-attention (no KV-cache, no tricks)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)  # each [B, T, H, Dh]
        q = q.transpose(1, 2)  # [B, H, T, Dh]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Build the causal mask by hand: SDPA rejects passing both
        # is_causal=True and an explicit attn_mask, and we need to combine
        # causality with the (optional) key-padding mask into one bias.
        causal = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()  # [T, T], True = allowed
        allowed = causal.view(1, 1, T, T)
        if key_padding_mask is not None:
            # key_padding_mask: [B, T] True where token is real, False where padding.
            allowed = allowed & key_padding_mask[:, None, None, :]
        bias = torch.zeros(B, 1, T, T, device=x.device, dtype=q.dtype)
        bias.masked_fill_(~allowed, float("-inf"))

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=bias,
            is_causal=False,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), key_padding_mask=key_padding_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TinyGPT(nn.Module):
    """Decoder-only trunk. Returns hidden states, not logits."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """tokens: [B, T] int64 token ids -> hidden: [B, T, D]"""
        B, T = tokens.shape
        assert T <= self.cfg.max_seq_len, f"sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}"
        pos = torch.arange(T, device=tokens.device).unsqueeze(0)
        x = self.tok_emb(tokens) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        x = self.final_norm(x)
        return x


class OutputHead(nn.Module):
    """The unembedding / LM head: h [B,T,D] -> logits [B,T,V].

    Can be tied to an existing embedding weight (weight sharing) or own its
    own independent [V, D] matrix (untied). This mirrors the "tied vs
    untied" comparison the assignment asks for.
    """

    def __init__(self, vocab_size: int, d_model: int, tied_weight: nn.Parameter | None = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        if tied_weight is not None:
            self.weight = tied_weight  # shared nn.Parameter, no new memory
            self.tied = True
        else:
            w = torch.empty(vocab_size, d_model)
            nn.init.normal_(w, mean=0.0, std=0.02)
            self.weight = nn.Parameter(w)
            self.tied = False

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: [B, T, D] -> logits: [B, T, V]"""
        return hidden @ self.weight.t()

    def num_parameters(self) -> int:
        # Tied heads add zero *new* parameters (the matrix is counted once,
        # under the embedding). We report both views explicitly at the call site.
        return self.weight.numel()


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
