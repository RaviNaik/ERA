"""Train (or load) a byte-level BPE tokenizer, and expose the token->bytes
map every byte-level codec (`KroneckerEmbedding`, `FourierKroneckerEmbedding`,
`HRRBindingEmbedding`) needs.

Byte-level BPE (GPT-2 style) is used deliberately: its base alphabet already
covers all 256 byte values, so it never needs an `<unk>` fallback and every
vocabulary id maps to a well-defined UTF-8 byte string — exactly the
"integers in, but the layer reaches back for the token's own bytes across the
seam" contract the class notes describe (Sec. 7) for the Kronecker codec.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.implementations import ByteLevelBPETokenizer

logger = logging.getLogger("fourier_embeddings.tokenizer")


def train_tokenizer(input_files: list[str], vocab_size: int, out_dir: str,
                     min_frequency: int = 2) -> str:
    tok = ByteLevelBPETokenizer()
    tok.train(
        files=input_files,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=True,
    )
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    out_path = out_dir_p / "tokenizer.json"
    tok.save(str(out_path))
    logger.info(f"trained tokenizer: vocab_size={tok.get_vocab_size()} -> {out_path}")
    return str(out_path)


def load_tokenizer(path: str) -> Tokenizer:
    return Tokenizer.from_file(path)


def build_id_to_bytes(tok: Tokenizer) -> list[bytes]:
    """id -> raw UTF-8 bytes of the text that id decodes to. This is the
    literal "read the bytes" step of the Kronecker codec (class notes Sec. 7,
    Step 1): every id's bytes are computed once, here, and cached as a
    fixed buffer inside each byte-level codec module.
    """
    vocab_size = tok.get_vocab_size()
    id_to_bytes: list[bytes] = [b""] * vocab_size
    for v in range(vocab_size):
        text = tok.decode([v], skip_special_tokens=False)
        id_to_bytes[v] = text.encode("utf-8", errors="replace")
    return id_to_bytes


def id_to_bytes_stats(id_to_bytes: list[bytes]) -> dict:
    lengths = [len(b) for b in id_to_bytes]
    return {
        "vocab_size": len(id_to_bytes),
        "min_bytes": min(lengths) if lengths else 0,
        "max_bytes": max(lengths) if lengths else 0,
        "mean_bytes": sum(lengths) / max(len(lengths), 1),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", default="data_raw")
    ap.add_argument("--out-dir", default="tokenizer_out")
    ap.add_argument("--vocab-size", type=int, default=16384)
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--overwrite", action="store_true",
                     help="retrain even if <out-dir>/tokenizer.json already exists (default: "
                          "skip training, since re-running the full pipeline after an "
                          "interruption shouldn't redo an already-finished tokenizer).")
    ap.add_argument("--log-file", default="logs/tokenizer.log")
    args = ap.parse_args()

    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(args.log_file), logging.StreamHandler()],
    )

    existing = Path(args.out_dir) / "tokenizer.json"
    if not args.overwrite and existing.exists():
        logger.info(f"{existing} already exists; skipping training (--overwrite to retrain). "
                    f"NOTE: if the vocab size or input corpus changed, you must pass --overwrite "
                    f"or the old tokenizer (and its byte map) will be reused silently.")
        return

    input_files = sorted(str(p) for p in Path(args.input_dir).glob("*.txt"))
    if not input_files:
        raise SystemExit(f"no .txt files found in {args.input_dir}; run fe-download-data first")
    logger.info(f"training on {len(input_files)} files: {input_files}")

    out_path = train_tokenizer(input_files, args.vocab_size, args.out_dir, args.min_frequency)

    tok = load_tokenizer(out_path)
    id_to_bytes = build_id_to_bytes(tok)
    stats = id_to_bytes_stats(id_to_bytes)
    logger.info(f"token byte-length stats: {stats}")

    stats_path = Path(args.out_dir) / "byte_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info(f"wrote {stats_path}")


if __name__ == "__main__":
    main()
