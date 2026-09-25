"""Baseline GPT and reversible GPT variants for the session 13 assignment."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .loss import chunked_cross_entropy
from .reversible import ReversibleStack


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 512
    n_layer: int = 9
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = True
    variant: str = "baseline"
    # Explicit-midpoint step size h (update is 2h * f). h = 0.5 makes each
    # update the size of a standard residual block. Smaller h is more stable
    # (paper Sec. 3). A local sweep (h = 1/9, 0.25, 0.5) gave the lowest loss
    # at 0.25.
    midpoint_h: float = 0.25
    # Bit-exact reversal via float64 fixed-point residual streams (see
    # reversible.py, note 5). False = plain fp32 streams (drifts under bf16).
    reversible_exact: bool = True
    # Rows-per-chunk for the fused lm_head+cross-entropy (see loss.py). Keeps
    # the (chunk, vocab_size) logits tensor from dominating peak memory.
    loss_chunk_size: int = 8192


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.dropout = cfg.dropout
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, channels = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(batch, steps, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, steps, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, steps, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch, steps, channels)
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
        if cfg.variant == "baseline":
            h: nn.Module = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        elif cfg.variant in ("euler", "midpoint"):
            # A single depth-wide reversible stack, not one block per layer:
            # this is what keeps activation memory O(1) in depth (see
            # reversible.py). Treated as one opaque module by the forward pass.
            h = ReversibleStack(cfg, kind=cfg.variant)
        else:
            raise ValueError(f"unknown variant: {cfg.variant}")

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(cfg.vocab_size, cfg.n_embd),
                "wpe": nn.Embedding(cfg.block_size, cfg.n_embd),
                "drop": nn.Dropout(cfg.dropout),
                "h": h,
                "ln_f": nn.LayerNorm(cfg.n_embd, bias=cfg.bias),
            }
        )
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        batch, steps = idx.shape
        if steps > self.cfg.block_size:
            raise ValueError(
                f"sequence length {steps} exceeds block_size {self.cfg.block_size}"
            )
        pos = torch.arange(0, steps, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        if self.cfg.variant == "baseline":
            for block in self.transformer.h:
                x = block(x)
        else:
            x = self.transformer.h(x)
        x = self.transformer.ln_f(x)
        if targets is None:
            logits = self.lm_head(x)
            return logits, None, None
        # Chunked path: never materialize the full (batch*steps, vocab_size)
        # logits tensor, so peak memory reflects the transformer backbone
        # (where baseline vs. reversible actually differ) rather than being
        # dominated by the vocab projection, which is identical across
        # variants. `logits` is intentionally not returned here.
        loss, accuracy = chunked_cross_entropy(
            x, targets, self.lm_head.weight, self.cfg.loss_chunk_size
        )
        return None, loss, accuracy

    def num_params(self, non_embedding: bool = False) -> int:
        total = sum(param.numel() for param in self.parameters())
        if non_embedding:
            total -= self.transformer.wpe.weight.numel()
        return total


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def param_count_report(cfg: GPTConfig) -> dict:
    model = GPT(cfg)
    buckets: dict[str, int] = {
        "embedding": 0,
        "attention": 0,
        "mlp": 0,
        "norm": 0,
        "other": 0,
    }
    for name, param in model.named_parameters():
        if "wte" in name or "wpe" in name or "lm_head" in name:
            buckets["embedding"] += param.numel()
        elif ".attn." in name:
            buckets["attention"] += param.numel()
        elif ".mlp." in name or ".f." in name or ".g." in name:
            buckets["mlp"] += param.numel()
        elif ".ln" in name or "ln_f" in name:
            buckets["norm"] += param.numel()
        else:
            buckets["other"] += param.numel()
    buckets["total"] = sum(param.numel() for param in model.parameters())
    buckets["config"] = asdict(cfg)
    del model
    return buckets


def flops_per_token(cfg: GPTConfig) -> dict[str, float]:
    """Analytic matmul FLOPs per trained token (PaLM / Chinchilla convention).

    * forward = 2 * (matmul params) + attention score/value matmuls
    * model FLOPs (used for MFU) = 3 * forward. This is identical across
      variants because they all do the same useful maths.
    * hardware FLOPs = what the GPU actually executes. Reversible variants
      re-run every backbone sub-layer once more in backward (the lm_head is
      never recomputed, see loss.py), just like activation checkpointing.
    """
    d, n_layer, steps = cfg.n_embd, cfg.n_layer, cfg.block_size
    per_layer_matmul = 3 * d * d + d * d + 4 * d * d + 4 * d * d  # qkv, proj, fc, proj
    backbone_fwd = 2 * n_layer * per_layer_matmul
    # QK^T and AV: 2 * (2 * T * d) per token per layer (causal mask not discounted).
    attention_fwd = n_layer * 4 * steps * d
    lm_head_fwd = 2 * d * cfg.vocab_size
    forward = backbone_fwd + attention_fwd + lm_head_fwd
    model = 3.0 * forward
    recompute = backbone_fwd + attention_fwd if cfg.variant in ("euler", "midpoint") else 0
    return {
        "forward": float(forward),
        "model": model,
        "hardware": model + recompute,
        "lm_head_fraction": lm_head_fwd / forward,
    }


def build_model(
    cfg: GPTConfig, device: str | torch.device = "cuda", compile_model: bool = False
) -> GPT:
    model = GPT(cfg).to(device)
    if compile_model:
        model = torch.compile(model)
    return model
