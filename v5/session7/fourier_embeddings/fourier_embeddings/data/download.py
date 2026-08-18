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
# Pause between successful page requests -- the datasets-server API rate-
# limits (HTTP 429) once a run has made a few hundred requests in a row
# (observed fetching a full --target-mb-per-lang=30 corpus), so this trades a
# little wall-clock time for not tripping the limit in the first place.
REQUEST_DELAY_SEC = 0.5


def fetch_rows(config: str, offset: int, length: int, retries: int = 8) -> list[dict]:
    params = f"dataset={DATASET.replace('/', '%2F')}&config={config}&split=train&offset={offset}&length={length}"
    url = f"{ROWS_URL}?{params}"
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = json.loads(r.read())
            return data.get("rows", [])
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code == 429:
                # Respect Retry-After if the server sends one; otherwise back
                # off much more aggressively than a transient network error
                # deserves, since 429 means "you're already being throttled."
                retry_after = e.headers.get("Retry-After") if e.headers else None
                wait = float(retry_after) if retry_after else min(60, 5 * (2 ** attempt))
            else:
                wait = min(30, 2 ** attempt)
            logger.warning(f"fetch failed ({e}); retrying in {wait:.0f}s "
                            f"(attempt {attempt + 1}/{retries})")
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
            wait = min(30, 2 ** attempt)
            logger.warning(f"fetch failed ({e}); retrying in {wait:.0f}s "
                            f"(attempt {attempt + 1}/{retries})")
            time.sleep(wait)
    raise RuntimeError(f"failed to fetch {url} after {retries} retries: {last_err}")


def download_language(lang: str, out_path: Path, target_chars: int) -> dict:
    """Best-effort: on a persistent failure (retries exhausted), logs the
    error and returns whatever was collected so far instead of raising, so
    one language's outage doesn't lose progress on the others (`main` still
    keeps going and writes a manifest that records the partial/failed status).
    """
    config = CONFIG_TEMPLATE.format(lang=lang)
    total_chars = 0
    n_articles = 0
    offset = 0
    error = None
    with open(out_path, "w", encoding="utf-8") as f:
        while total_chars < target_chars:
            try:
                rows = fetch_rows(config, offset, PAGE_SIZE)
            except RuntimeError as e:
                logger.error(f"[{lang}] giving up at offset={offset} after repeated failures: {e}")
                error = str(e)
                break
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
            time.sleep(REQUEST_DELAY_SEC)
    return {"lang": lang, "chars": total_chars, "articles": n_articles, "path": str(out_path),
            "complete": total_chars >= target_chars, "error": error}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="data_raw")
    ap.add_argument("--langs", nargs="+", default=DEFAULT_LANGS)
    ap.add_argument("--target-mb-per-lang", type=float, default=3.0,
                     help="Approximate target size (MB) per language, in UTF-8 characters.")
    ap.add_argument("--overwrite", action="store_true",
                     help="re-download a language even if its .txt file already reached the "
                          "target size (default: skip it, so re-running after an interruption "
                          "or a rate-limit failure doesn't redo already-finished languages).")
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

        if not args.overwrite and out_path.exists():
            existing_chars = out_path.stat().st_size  # byte count is a cheap lower bound on char count
            if existing_chars >= target_chars:
                text = out_path.read_text(encoding="utf-8")
                logger.info(f"[{lang}] {out_path} already has {len(text):,} chars "
                            f"(target {target_chars:,}); skipping (--overwrite to redo)")
                manifest.append({"lang": lang, "chars": len(text), "articles": None,
                                  "path": str(out_path), "complete": True, "error": None,
                                  "skipped": True})
                continue

        logger.info(f"downloading {lang} -> {out_path} (target ~{args.target_mb_per_lang}MB)")
        try:
            stats = download_language(lang, out_path, target_chars)
        except Exception as e:  # noqa: BLE001 -- one language's outage must not kill the whole run
            logger.error(f"[{lang}] unrecoverable failure, skipping: {e}")
            stats = {"lang": lang, "chars": 0, "articles": 0, "path": str(out_path),
                      "complete": False, "error": str(e)}
        manifest.append(stats)
        logger.info(f"[{lang}] done: {stats}")

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(f"wrote manifest -> {manifest_path}")
    total = sum(m["chars"] for m in manifest)
    logger.info(f"total corpus: {total:,} chars across {len(manifest)} languages")

    incomplete = [m["lang"] for m in manifest if not m.get("complete", True)]
    if incomplete:
        logger.warning(f"languages that did not reach their target size: {incomplete} -- "
                        f"re-run with --langs {' '.join(incomplete)} to retry just those "
                        f"(existing files for other languages are untouched)")


if __name__ == "__main__":
    main()
