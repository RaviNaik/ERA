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

Languages download concurrently by default (one thread per language, each
writing its own file, so there's no shared mutable state to coordinate).
This does raise the aggregate request rate against the API, which is what
originally triggered HTTP 429s on a sequential run too (fetch_rows' own
retry/backoff, including respecting a server-sent Retry-After header,
already absorbs that) -- if 429 warnings show up repeatedly in the log,
pass --max-parallel 1 to fall back to the old one-at-a-time behavior.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _fetch_one(lang: str, args, target_chars: int) -> dict:
    out_dir = Path(args.out_dir)
    out_path = out_dir / f"{lang}.txt"

    if not args.overwrite and out_path.exists():
        existing_chars = out_path.stat().st_size  # byte count is a cheap lower bound on char count
        if existing_chars >= target_chars:
            text = out_path.read_text(encoding="utf-8")
            logger.info(f"[{lang}] {out_path} already has {len(text):,} chars "
                        f"(target {target_chars:,}); skipping (--overwrite to redo)")
            return {"lang": lang, "chars": len(text), "articles": None,
                    "path": str(out_path), "complete": True, "error": None, "skipped": True}

    logger.info(f"downloading {lang} -> {out_path} (target ~{args.target_mb_per_lang}MB)")
    try:
        return download_language(lang, out_path, target_chars)
    except Exception as e:  # noqa: BLE001 -- one language's outage must not kill the others
        logger.error(f"[{lang}] unrecoverable failure, skipping: {e}")
        return {"lang": lang, "chars": 0, "articles": 0, "path": str(out_path),
                "complete": False, "error": str(e)}


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
    ap.add_argument("--max-parallel", type=int, default=None,
                     help="download this many languages concurrently (default: all of --langs "
                          "at once, since each writes to its own file and the per-request retry/"
                          "backoff below already absorbs any extra HTTP 429s the higher aggregate "
                          "request rate causes). Pass --max-parallel 1 to restore the old fully "
                          "sequential behavior if you see repeated 429 warnings in the log.")
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

    max_parallel = args.max_parallel or len(args.langs)
    logger.info(f"downloading {len(args.langs)} language(s) with up to {max_parallel} concurrent")
    manifest_by_lang: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max_parallel) as ex:
        futures = {ex.submit(_fetch_one, lang, args, target_chars): lang for lang in args.langs}
        for fut in as_completed(futures):
            lang = futures[fut]
            try:
                stats = fut.result()
            except Exception as e:  # pragma: no cover -- _fetch_one already catches its own errors
                logger.error(f"[{lang}] worker raised unexpectedly: {e}")
                stats = {"lang": lang, "chars": 0, "articles": 0, "path": None,
                          "complete": False, "error": str(e)}
            manifest_by_lang[lang] = stats
            logger.info(f"[{lang}] done: {stats}")

    # Preserve --langs order in the manifest regardless of completion order.
    manifest = [manifest_by_lang[lang] for lang in args.langs]

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
