"""
Tiny GPT - minimal 2-layer transformer for training simulation.
~1M parameters. Accepts loss_mask so only loss-bearing tokens contribute gradients.
"""
from __future__ import annotations
import math
from typing import Optional, Tuple
import numpy as np


class TinyGPT:
    """
    Pure-numpy 2-layer GPT simulation.
    Uses manual forward pass + simplified gradient simulation.
    Real loss is computed via cross-entropy on loss-masked positions.
    """

    def __init__(self, vocab_size: int = 100277, n_embd: int = 64,
                 n_head: int = 4, n_layer: int = 2, context_len: int = 64,
                 seed: int = 42):
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.context_len = context_len
        # Simplified weights (token embedding + output projection)
        scale = 0.02
        self.wte = rng.normal(0, scale, (vocab_size, n_embd)).astype(np.float32)
        self.wpe = rng.normal(0, scale, (context_len, n_embd)).astype(np.float32)
        self.out_proj = rng.normal(0, scale / math.sqrt(n_embd), (n_embd, vocab_size)).astype(np.float32)
        # Layer norms (scale=1, bias=0)
        self.ln_scale = np.ones(n_embd, dtype=np.float32)
        self._step = 0
        self.param_count = (vocab_size * n_embd) + (context_len * n_embd) + (n_embd * vocab_size)

    def _layer_norm(self, x: np.ndarray) -> np.ndarray:
        mean = x.mean(-1, keepdims=True)
        std = x.std(-1, keepdims=True) + 1e-5
        return (x - mean) / std * self.ln_scale

    def forward(self, input_ids: np.ndarray, loss_mask: np.ndarray,
                targets: Optional[np.ndarray] = None) -> dict:
        """
        Forward pass. Returns loss (masked), per-token logit distributions.
        input_ids: [seq_len] int32
        loss_mask: [seq_len] float32
        targets:   [seq_len] int32 (next-token targets)
        """
        seq_len = len(input_ids)
        safe_ids = np.clip(input_ids, 0, self.vocab_size - 1)
        # Embeddings
        tok_emb = self.wte[safe_ids]
        pos_emb = self.wpe[:seq_len]
        x = tok_emb + pos_emb
        x = self._layer_norm(x)

        # Simplified attention simulation (for demo purposes)
        # In real GPT: multi-head causal attention + FFN
        for _ in range(self.n_layer):
            x = self._layer_norm(x + np.tanh(x @ np.eye(self.n_embd)))

        # Output logits
        logits = x @ self.out_proj  # [seq_len, vocab_size]

        result = {"logits": logits}

        if targets is not None:
            # Cross-entropy loss, only on loss-bearing positions
            safe_targets = np.clip(targets, 0, self.vocab_size - 1)
            log_probs = logits - np.log(np.exp(logits).sum(-1, keepdims=True) + 1e-10)
            token_losses = -log_probs[np.arange(seq_len), safe_targets]  # [seq_len]
            masked_losses = token_losses * loss_mask
            total_loss_tokens = loss_mask.sum()
            loss = masked_losses.sum() / max(1.0, total_loss_tokens)
            result["loss"] = float(loss)
            result["token_losses"] = token_losses
            result["loss_bearing_tokens"] = int(total_loss_tokens)

        self._step += 1
        return result

    def get_weights_hash(self) -> str:
        """Hash of model weights for checkpoint integrity."""
        import hashlib
        h = hashlib.sha256()
        h.update(self.wte.tobytes())
        h.update(self.out_proj.tobytes())
        return "weights_" + h.hexdigest()[:12]

    def state_dict(self) -> dict:
        return {
            "wte": self.wte.copy(),
            "wpe": self.wpe.copy(),
            "out_proj": self.out_proj.copy(),
            "ln_scale": self.ln_scale.copy(),
            "_step": self._step,
        }

    def load_state_dict(self, state: dict) -> None:
        self.wte = state["wte"].copy()
        self.wpe = state["wpe"].copy()
        self.out_proj = state["out_proj"].copy()
        self.ln_scale = state["ln_scale"].copy()
        self._step = state["_step"]

    def step_update(self, loss: float, lr: float = 1e-3) -> None:
        """Simulate a gradient update (perturbation-based for demo)."""
        rng = np.random.default_rng(self._step)
        noise_scale = lr * abs(loss) * 0.001
        self.wte -= rng.normal(0, noise_scale, self.wte.shape)
        self.out_proj -= rng.normal(0, noise_scale, self.out_proj.shape)
