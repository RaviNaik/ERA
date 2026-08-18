"""Fetch a moderate-size multilingual text corpus for the embedding
comparison experiments.

Rationale (research note Sec. 8.4/§10.2): the whole point of this project is
to measure Kronecker-vs-Fourier behaviour on the multi-script setting the
class notes and research note care about (Hindi/Telugu/Tamil/Bengali byte
budgets), so the corpus should actually contain those scripts, not just
English. We pull real Wikipedia articles for five languages via the HF
`datasets-server` rows API (JSON over HTTP, no `datasets`/`pyarrow`
dependency needed, and no need to download full multi-hundred-MB parquet
shards) and write one flat UTF-8 text file per language.

This is a small, auditable HTTP client — stdlib `urllib` only.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger("fourier_embeddings.download")

ROWS_URL = "https://datasets-server.huggingface.co/rows"
DATASET = "wikimedia/wikipedia"
CONFIG_TEMPLATE = "20231101.{lang}"

DEFAULT_LANGS = ["en", "hi", "te", "ta", "bn"]
PAGE_SIZE = 100


def fetch_rows(config: str, offset: int, length: int, retries: int = 4) -> list[dict]:
    params = f"dataset={DATASET.replace('/', '%2F')}&config={config}&split=train&offset={offset}&length={length}"
    url = f"{ROWS_URL}?{params}"
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = json.loads(r.read())
            return data.get("rows", [])
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last_err = e
            wait = 2 ** attempt
            logger.warning(f"fetch failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"failed to fetch {url} after {retries} retries: {last_err}")


def download_language(lang: str, out_path: Path, target_chars: int) -> dict:
    config = CONFIG_TEMPLATE.format(lang=lang)
    total_chars = 0
    n_articles = 0
    offset = 0
    with open(out_path, "w", encoding="utf-8") as f:
        while total_chars < target_chars:
            rows = fetch_rows(config, offset, PAGE_SIZE)
            if not rows:
                logger.warning(f"[{lang}] ran out of rows at offset={offset} "
                                f"({total_chars}/{target_chars} chars collected)")
                break
            for row in rows:
                text = row["row"]["text"]
                if not text:
                    continue
                f.write(text)
                f.write("\n\n")
                total_chars += len(text)
                n_articles += 1
            offset += PAGE_SIZE
            logger.info(f"[{lang}] {total_chars}/{target_chars} chars, {n_articles} articles "
                        f"(offset={offset})")
    return {"lang": lang, "chars": total_chars, "articles": n_articles, "path": str(out_path)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="data_raw")
    ap.add_argument("--langs", nargs="+", default=DEFAULT_LANGS)
    ap.add_argument("--target-mb-per-lang", type=float, default=3.0,
                     help="Approximate target size (MB) per language, in UTF-8 characters.")
    ap.add_argument("--log-file", default="logs/download.log")
    args = ap.parse_args()

    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(args.log_file), logging.StreamHandler()],
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_chars = int(args.target_mb_per_lang * 1_000_000)

    manifest = []
    for lang in args.langs:
        out_path = out_dir / f"{lang}.txt"
        logger.info(f"downloading {lang} -> {out_path} (target ~{args.target_mb_per_lang}MB)")
        stats = download_language(lang, out_path, target_chars)
        manifest.append(stats)
        logger.info(f"[{lang}] done: {stats}")

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(f"wrote manifest -> {manifest_path}")
    total = sum(m["chars"] for m in manifest)
    logger.info(f"total corpus: {total:,} chars across {len(manifest)} languages")


if __name__ == "__main__":
    main()
