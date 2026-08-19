"""Config dataclasses for the model and training run."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal


@dataclass
class ModelConfig:
    vocab_size: int = 16384
    block_size: int = 256          # max sequence length (context)
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = False             # bias in Linear/LayerNorm, GPT-2-style default off
    tie_weights: bool = False      # only meaningful/allowed when embedding output dim == n_embd

    # --- embedding codec selection ---
    embedding: Literal["dense", "kronecker", "fourier", "fourier_narrow", "hrr"] = "dense"
    char_dim: int = 256
    pos_dim: int = 32
    fourier_dim: int = 32
    hrr_dim: int = 1024
    codec_mode: Literal["dynamic", "cached"] = "dynamic"
    # fourier/hrr only: the actual byte-buffer width (see codecs.ByteGridCodec's
    # docstring). None = auto-detect from the longest real token in the
    # tokenizer's vocabulary, so no real token is ever cropped -- this is what
    # makes "no truncation wall" (research note Sec. 6.1) actually true of the
    # trained model, not just of phi(p) in isolation. Kronecker/onehot always
    # uses pos_dim itself; this knob doesn't apply to it (by design).
    byte_capacity: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainConfig:
    # data
    data_dir: str = "data_bin"
    tokenizer_path: str = "tokenizer_out/tokenizer.json"

    # optimization
    max_steps: int = 2000
    max_epochs: float | None = None   # if set, overrides max_steps via dataset size / batch
    batch_size: int = 32
    grad_accum_steps: int = 1
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95

    # schedule / logging
    eval_interval: int = 200
    eval_iters: int = 50
    log_interval: int = 20
    save_interval: int = 500

    # infra
    device: str = "cuda"
    dtype: str = "bfloat16"        # "float32" | "bfloat16" | "float16"
    compile: bool = False
    seed: int = 1337

    # run identity
    run_name: str = "run"
    out_dir: str = "results"
    aim_repo: str = "aim_repo"
    log_dir: str = "logs"

    def to_dict(self) -> dict:
        return asdict(self)
