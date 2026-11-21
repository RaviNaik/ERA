"""Check GPU-to-GPU copies on the server: `.venv/bin/python tools/p2p_check.py`, also with CUDA_LAUNCH_BLOCKING=1."""
import os

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dtphi import devcopy

print("CUDA_LAUNCH_BLOCKING =", os.environ.get("CUDA_LAUNCH_BLOCKING"), "| devices:", torch.cuda.device_count(),
      "| peer access 0->1:", torch.cuda.can_device_access_peer(0, 1))


def direct(t):
    return t.to("cuda:1")


def direct_after_sync(t):
    torch.cuda.synchronize(0)
    return t.to("cuda:1")


def via_cpu(t):
    return t.to("cpu").to("cuda:1")


def patched(t):   # Tensor.to / copy_ / cuda as patched by devcopy.install_global
    devcopy.install_global()
    out = t.to("cuda:1")
    dst = torch.empty_like(out); dst.copy_(t)
    return out if torch.equal(dst.cpu(), out.cpu()) else out + 1


totals = {}
for name, fn in (("direct", direct), ("direct after sync", direct_after_sync), ("via CPU", via_cpu), ("patched to/copy_", patched)):
    bad = 0
    for n in (1, 32, 4096, 1 << 20, 1 << 24):
        for kind in ("whole", "view"):
            wrong = 0
            for _ in range(200):
                src = torch.randint(0, 1000, (32, max(n // 32, 2)), device="cuda:0")
                t = src[:, -1:] if kind == "view" else src
                ref = t.cpu()
                if not torch.equal(fn(t).cpu(), ref):
                    wrong += 1
            bad += wrong
            print(f"{name:18} size {n:>9} {kind:5}: {wrong}/200 wrong")
    totals[name] = bad
print("RESULT:", {k: ("WRONG" if v else "ok") for k, v in totals.items()})
