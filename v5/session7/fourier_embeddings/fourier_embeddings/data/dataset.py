"""Pack raw text into a flat token-id binary stream (train/val split), and
serve random fixed-length blocks from it. Mirrors nanoGPT's `prepare.py` /
`get_batch` pattern: a `np.memmap` over a flat `uint16`/`uint32` array, so
arbitrarily large corpora never need to fit in RAM at once.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from .tokenizer import load_tokenizer

logger = logging.getLogger("fourier_embeddings.dataset")


def dtype_for_vocab(vocab_size: int):
    return np.uint16 if vocab_size < 65536 else np.uint32


def pack_dataset(input_dir: str, tokenizer_path: str, out_dir: str,
                  val_fraction: float = 0.02, chunk_chars: int = 2_000_000) -> dict:
    tok = load_tokenizer(tokenizer_path)
    vocab_size = tok.get_vocab_size()
    dtype = dtype_for_vocab(vocab_size)

    input_files = sorted(str(p) for p in Path(input_dir).glob("*.txt"))
    if not input_files:
        raise SystemExit(f"no .txt files found in {input_dir}")

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    all_ids: list[int] = []
    for fp in input_files:
        text = Path(fp).read_text(encoding="utf-8")
        # Encode in chunks so the tokenizer never has to swallow a whole
        # multi-MB file as one string.
        for start in range(0, len(text), chunk_chars):
            chunk = text[start:start + chunk_chars]
            enc = tok.encode(chunk)
            all_ids.extend(enc.ids)
        logger.info(f"encoded {fp}: running total {len(all_ids):,} tokens")

    ids = np.array(all_ids, dtype=dtype)
    n_val = int(len(ids) * val_fraction)
    train_ids, val_ids = ids[:-n_val] if n_val > 0 else ids, ids[-n_val:] if n_val > 0 else ids[:0]

    train_path = out_dir_p / "train.bin"
    val_path = out_dir_p / "val.bin"
    train_ids.tofile(train_path)
    val_ids.tofile(val_path)

    meta = {
        "vocab_size": vocab_size,
        "dtype": str(dtype),
        "train_tokens": int(len(train_ids)),
        "val_tokens": int(len(val_ids)),
        "tokenizer_path": tokenizer_path,
        "input_files": input_files,
    }
    (out_dir_p / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info(f"wrote {train_path} ({len(train_ids):,} tokens), {val_path} ({len(val_ids):,} tokens)")
    return meta


class BinDataset:
    """A memmap'd flat token stream with random fixed-length block sampling."""

    def __init__(self, bin_path: str, dtype):
        self.bin_path = bin_path
        self.dtype = dtype
        self._len = None

    def __len__(self) -> int:
        if self._len is None:
            self._len = len(np.memmap(self.bin_path, dtype=self.dtype, mode="r"))
        return self._len

    def get_batch(self, block_size: int, batch_size: int, device: str,
                   generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        # Reopen the memmap every call (nanoGPT's trick to avoid a slow
        # memory leak from repeatedly indexing one long-lived memmap object).
        data = np.memmap(self.bin_path, dtype=self.dtype, mode="r")
        n = len(data) - block_size - 1
        if n <= 0:
            raise ValueError(f"dataset too small for block_size={block_size}: {len(data)} tokens")
        ix = torch.randint(0, n, (batch_size,), generator=generator)
        x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
        if device.startswith("cuda"):
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    def num_batches_per_epoch(self, block_size: int, batch_size: int) -> int:
        n = max(len(self) - block_size - 1, 1)
        return max(n // (block_size * batch_size), 1)


def load_meta(data_dir: str) -> dict:
    return json.loads((Path(data_dir) / "meta.json").read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", default="data_raw")
    ap.add_argument("--tokenizer-path", default="tokenizer_out/tokenizer.json")
    ap.add_argument("--out-dir", default="data_bin")
    ap.add_argument("--val-fraction", type=float, default=0.02)
    ap.add_argument("--overwrite", action="store_true",
                     help="repack even if <out-dir>/meta.json already exists (default: skip, "
                          "since re-running the full pipeline after an interruption shouldn't "
                          "redo an already-finished pack).")
    ap.add_argument("--log-file", default="logs/pack_dataset.log")
    args = ap.parse_args()

    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(args.log_file), logging.StreamHandler()],
    )

    existing = Path(args.out_dir) / "meta.json"
    if not args.overwrite and existing.exists():
        logger.info(f"{existing} already exists; skipping packing (--overwrite to redo). "
                    f"NOTE: if the tokenizer or input corpus changed, you must pass --overwrite "
                    f"or the old (now-mismatched) binary dataset will be reused silently.")
        return

    meta = pack_dataset(args.input_dir, args.tokenizer_path, args.out_dir, args.val_fraction)
    logger.info(f"done: {meta}")


if __name__ == "__main__":
    main()
