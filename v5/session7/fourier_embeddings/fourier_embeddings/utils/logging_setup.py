"""Shared logging setup: every entrypoint logs to both stdout and a file."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(log_file: str, name: str = "fourier_embeddings", level=logging.INFO) -> logging.Logger:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        return logger  # already configured (e.g. re-entrant call)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger
