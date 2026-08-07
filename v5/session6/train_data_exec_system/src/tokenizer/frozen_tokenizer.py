"""
Frozen Tokenizer - wraps tiktoken tokenizer.
Computes a deterministic tokenizer_sha. All shards must be tagged with this SHA.
Changing the tokenizer invalidates every pre-tokenized shard.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import List
import tiktoken

EOS_TOKEN_STRING = "EOT"   # end-of-document marker string


class FrozenTokenizer:
    """
    Wraps tiktoken cl100k_base (GPT-4 family tokenizer).
    The tokenizer_sha ties every shard to this exact vocabulary snapshot.
    """
    MODEL_NAME = "cl100k_base"

    def __init__(self):
        self._enc = tiktoken.get_encoding(self.MODEL_NAME)
        self._sha = self._compute_sha()

    def _compute_sha(self) -> str:
        sample = []
        for i in range(min(50, self._enc.n_vocab)):
            try:
                sample.append(self._enc.decode_single_token_bytes(i).hex())
            except Exception:
                pass
        fingerprint = json.dumps({"model": self.MODEL_NAME, "n_vocab": self._enc.n_vocab, "sample": sample[:20]}, sort_keys=True)
        return "tok_" + hashlib.sha256(fingerprint.encode()).hexdigest()[:16]

    @property
    def tokenizer_sha(self) -> str:
        return self._sha

    @property
    def vocab_size(self) -> int:
        return self._enc.n_vocab

    def encode(self, text: str) -> List[int]:
        return self._enc.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        return self._enc.decode(token_ids)

    def save_frozen_spec(self, output_path: Path) -> None:
        spec = {
            "model_name": self.MODEL_NAME,
            "tokenizer_sha": self._sha,
            "vocab_size": self._enc.n_vocab,
            "eos_token_string": EOS_TOKEN_STRING,
            "agentic_mask_spans": ["[OBSERVATION]", "[/OBSERVATION]"],
            "normalizer_id": "unicode_nfc",
            "special_tokens": {"pad": "<pad>", "eos": "<eos>", "bos": "<bos>"},
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(spec, indent=2))
