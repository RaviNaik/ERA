"""Check direct GPU-to-GPU copies: run on the server, `.venv/bin/python tools/p2p_check.py`."""
import torch

print("devices:", torch.cuda.device_count(), "peer access 0->1:", torch.cuda.can_device_access_peer(0, 1))
bad = 0
for n in (1, 32, 4096, 1 << 20, 1 << 24):
    for kind in ("direct", "view"):
        wrong = 0
        for i in range(300):
            src = torch.randint(0, 1000, (32, max(n // 32, 2)), device="cuda:0")
            t = src[:, -1:] if kind == "view" else src
            ref = t.cpu()
            got = t.to("cuda:1")
            if not torch.equal(got.cpu(), ref):
                wrong += 1
        bad += wrong
        print(f"size {n:>9} {kind:6}: {wrong}/300 wrong")
print("RESULT:", "direct copies are CORRUPTED" if bad else "direct copies fine")
