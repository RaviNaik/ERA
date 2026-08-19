"""A compact GPT-2-style decoder-only transformer (nanoGPT lineage) whose
*input* token embedding is pluggable — the only thing that changes between
experiment arms. Position encoding is the standard learned-absolute table
(class notes Sec. 11); this project's subject is the token codec, not
position policy (deferred to "Session 8" per the class notes), so position
handling is deliberately kept identical and boring across every arm.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .codecs import CodecSpec, build_codec, codec_param_count
from .config import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout

        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.flash = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
            mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            att = att.masked_fill(~mask, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig, id_to_bytes: list[bytes]):
        super().__init__()
        self.cfg = cfg

        spec = CodecSpec(
            kind=cfg.embedding, char_dim=cfg.char_dim, pos_dim=cfg.pos_dim,
            fourier_dim=cfg.fourier_dim, hrr_dim=cfg.hrr_dim, mode=cfg.codec_mode,
            byte_capacity=cfg.byte_capacity,
        )
        self.token_codec = build_codec(spec, cfg.vocab_size, cfg.n_embd, id_to_bytes)
        self.position_embedding = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            if isinstance(self.token_codec, nn.Module) and hasattr(self.token_codec, "emb"):
                # Only the dense codec has a [V, n_embd] weight matrix that
                # can be tied to the output head; the byte codecs' learned
                # weight is a [D, n_embd] projection, not a [V, n_embd] table,
                # so tying does not apply the same way for them (research
                # note Sec. 6.2 discusses when this option reopens).
                self.lm_head.weight = self.token_codec.emb.weight
            else:
                raise ValueError("tie_weights=True is only supported for embedding='dense'")

        self.apply(self._init_weights)
        # GPT-2-style scaled init for residual projections
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = input_ids.shape
        assert T <= self.cfg.block_size, f"sequence length {T} exceeds block_size {self.cfg.block_size}"

        tok_emb = self.token_codec(input_ids)  # [B, T, n_embd]
        pos = torch.arange(T, device=input_ids.device)
        pos_emb = self.position_embedding(pos)  # [T, n_embd]
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
            return logits, loss
        else:
            logits = self.lm_head(x[:, [-1], :])
            return logits, None

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= sum(p.numel() for p in self.position_embedding.parameters())
        return n

    def param_breakdown(self) -> dict:
        return {
            "token_codec": codec_param_count(self.token_codec),
            "position_embedding": sum(p.numel() for p in self.position_embedding.parameters()),
            "transformer_blocks": sum(p.numel() for p in self.blocks.parameters()),
            "ln_f": sum(p.numel() for p in self.ln_f.parameters()),
            "lm_head": 0 if self.cfg.tie_weights else sum(p.numel() for p in self.lm_head.parameters()),
            "total": self.num_params(),
        }

    def configure_optimizer(self, weight_decay: float, learning_rate: float,
                             betas: tuple[float, float]) -> torch.optim.Optimizer:
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() >= 2:
                decay.append(p)
            else:
                no_decay.append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
        return input_ids
