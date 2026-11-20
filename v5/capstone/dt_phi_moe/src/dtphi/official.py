"""Access to the vendored official Deletion Tuning code (official/, unmodified, commit 94f0340).

The official modules import each other as top-level modules (`import sft_data`, `from learnable_sparsity...`), so
official/src is put on sys.path. `verify_vendored()` recomputes the git blob hash of every vendored file and compares
it with official/VENDORED_FILES.txt, so a changed file is caught before a run."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_DIR = ROOT / "official"
OFFICIAL_SRC = OFFICIAL_DIR / "src"
COMMIT = "94f0340ebb574f95bb41d844165253a6785e215e"

if str(OFFICIAL_SRC) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_SRC))


def verify_vendored() -> int:
    """Number of files checked; raises if a file is missing or differs from the upstream blob hash."""
    n = 0
    for line in (OFFICIAL_DIR / "VENDORED_FILES.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        sha, path = line.split(None, 1)
        data = (OFFICIAL_DIR / path).read_bytes()
        if hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() != sha:
            raise RuntimeError(f"official/{path} differs from the upstream file (commit {COMMIT})")
        n += 1
    return n
