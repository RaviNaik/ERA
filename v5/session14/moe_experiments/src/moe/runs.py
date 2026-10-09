"""Glue shared by the notebooks: build models / TrainConfigs from an ``ExpConfig``."""
from __future__ import annotations

from .config import ExpConfig
from .model import GPT, GPTConfig, param_report
from .trainer import TrainConfig


def dense_config(X: ExpConfig, vocab_size: int | None = None) -> GPTConfig:
    return GPTConfig(vocab_size=vocab_size or X.vocab_size, block_size=X.block_size, n_layer=X.n_layer,
                     n_head=X.n_head, n_embd=X.n_embd, dropout=X.dropout, bias=X.bias)


def moe_config(X: ExpConfig, vocab_size: int | None = None) -> GPTConfig:
    c = dense_config(X, vocab_size)
    c.n_experts, c.top_k, c.moe_every, c.score_fn = X.n_experts, X.top_k, X.moe_every, X.score_fn
    return c


def train_config(X: ExpConfig, label: str, phase: str, steps: int, lr: float, warmup: int,
                 step_offset: int = 0, device: str = "cuda", **kw) -> TrainConfig:
    base = dict(label=label, phase=phase, total_steps=steps, step_offset=step_offset, lr=lr, warmup=warmup,
                batch_size=X.batch_size, block_size=X.block_size, min_lr_frac=X.min_lr_frac,
                weight_decay=X.weight_decay, lb_coef=X.lb_coef, z_coef=X.z_coef, bias_update=X.bias_update,
                eval_interval=X.eval_interval, eval_iters=X.eval_iters, log_interval=X.log_interval,
                ckpt_interval=max(X.eval_interval, 1000) if X.name == "base" else X.eval_interval * 2,
                seed=X.seed, device=device)
    base.update(kw)
    return TrainConfig(**base)


def flops_per_token(model: GPT) -> float:
    """Approximate training FLOPs/token = 6 * (matmul params touched per token) + attention scores."""
    c = model.cfg
    rp = param_report(model)
    matmul = rp["active"] - model.transformer.wpe.weight.numel()   # embeddings gather is free; tied head counts once
    return 6.0 * matmul + 12.0 * c.n_layer * c.n_embd * c.block_size
