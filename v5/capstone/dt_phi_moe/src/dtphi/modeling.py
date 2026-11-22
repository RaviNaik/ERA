"""Load Phi-mini-MoE-instruct (bf16) over one or two GPUs, or on the CPU for tests."""
from __future__ import annotations

import torch
from transformers import AutoTokenizer

from .phimoe import PhiMoEConfig, PhiMoEForCausalLM


def load_tokenizer(model_dir: str):
    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def make_device_map(num_layers: int, n_gpus: int) -> dict[str, int]:
    """Layers in n_gpus contiguous blocks; the embeddings go with the first block, the final norm and LM head with
    the last. Naive model parallelism: one GPU works at a time, which is the price of fitting 2 x 48 GB."""
    if n_gpus < 1:
        raise ValueError("n_gpus must be >= 1")
    dm = {"model.embed_tokens": 0}
    per = -(-num_layers // n_gpus)
    for i in range(num_layers):
        dm[f"model.layers.{i}"] = min(i // per, n_gpus - 1)
    dm["model.norm"] = n_gpus - 1
    dm["lm_head"] = n_gpus - 1
    return dm


def load_model(model_dir: str, n_gpus: int = 2, dtype=torch.bfloat16):
    """n_gpus = 0 loads on the CPU. The expert weights stay plain nn.Linear modules (w1, w2, w3)."""
    cfg = PhiMoEConfig.from_pretrained(model_dir)
    kw = dict(config=cfg, torch_dtype=dtype, attn_implementation="sdpa", low_cpu_mem_usage=True)
    if n_gpus > 0:
        if torch.cuda.device_count() < n_gpus:
            raise RuntimeError(f"asked for {n_gpus} GPUs, {torch.cuda.device_count()} visible")
        kw["device_map"] = make_device_map(cfg.num_hidden_layers, n_gpus)
        if n_gpus > 1:
            from . import devcopy
            devcopy.install_global()   # direct GPU-to-GPU copies return zeros/garbage on the lab server
    model = PhiMoEForCausalLM.from_pretrained(model_dir, **kw)
    model.eval()
    return model


def input_device(model) -> torch.device:
    return model.model.embed_tokens.weight.device
