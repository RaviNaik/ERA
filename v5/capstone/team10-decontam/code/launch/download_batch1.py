"""Resumable, verified download of Hugging Face dataset files (no conversion, no decontam).

Files keep their Hub layout under <root>/<repo>/<path>. Each file is checked against the sha256 that the Hub
lists for it; a partial file is resumed with a Range request. Progress: <root>/status.txt and <root>/download.log.
"""
import hashlib, json, os, shutil, sys, threading, time, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path("/mnt/Dev/ERA_Submissions/g10-hub-downloads")
MIN_FREE = 15e9
SETS = {  # set id -> (repo, predicate on path)
    "sft-math-nemotron-math-v2": ("nvidia/Nemotron-Math-v2", lambda p: p.startswith("data/") and p.endswith(".parquet")),
    "sft-math-nemotron-proofs": ("nvidia/Nemotron-Math-Proofs-v1", lambda p: p == "data/lean.jsonl"),
    "sft-rl-math-openmathreasoning": ("nvidia/OpenMathReasoning", lambda p: p.startswith("data/cot-") and p.endswith(".parquet")),
}
log_f = open(ROOT / "download.log", "a", buffering=1)
lock = threading.Lock()
state = {}  # (set, path) -> dict(size, done, status)
t0 = time.time()


def log(msg):
    with lock:
        log_f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")


def tree(repo):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{repo}/tree/main?recursive=1", timeout=60) as r:
                return json.load(r)
        except OSError:
            time.sleep(5 * (attempt + 1))
    raise SystemExit(f"cannot list {repo}")


def fetch(set_id, repo, entry):
    path, size, want = entry["path"], entry["size"], (entry.get("lfs") or {}).get("oid")
    key = (set_id, path)
    dest = ROOT / repo / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{urllib.parse.quote(path)}"
    for attempt in range(12):
        try:
            h = hashlib.sha256()
            have = dest.stat().st_size if dest.exists() else 0
            if have > size:
                dest.unlink(); have = 0
            if have:  # resume: hash what is already there
                with open(dest, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 22), b""):
                        h.update(chunk)
            state[key].update(done=have, status="downloading")
            if have < size:
                req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"})
                with urllib.request.urlopen(req, timeout=120) as r, open(dest, "ab") as out:
                    while True:
                        while shutil.disk_usage(ROOT).free < MIN_FREE:
                            state[key]["status"] = "waiting for disk"; time.sleep(30)
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        out.write(chunk); h.update(chunk)
                        state[key]["done"] += len(chunk)
            if dest.stat().st_size != size:
                raise OSError(f"size {dest.stat().st_size} != {size}")
            if want and h.hexdigest() != want:
                dest.unlink()
                raise OSError("sha256 mismatch, file removed")
            state[key].update(done=size, status="done", sha256=h.hexdigest())
            log(f"done {set_id} {path} {size / 1e9:.2f} GB sha256 {h.hexdigest()[:12]}")
            return
        except (OSError, Exception) as e:  # connection reset, short read, mismatch: retry
            state[key]["status"] = f"retry {attempt + 1}: {str(e)[:60]}"
            log(f"retry {set_id} {path}: {e}")
            time.sleep(min(60, 5 * (attempt + 1)))
    state[key]["status"] = "FAILED"
    log(f"FAILED {set_id} {path}")


def status():
    while True:
        L = [time.strftime("%Y-%m-%d %H:%M:%S") + f"  elapsed {(time.time() - t0) / 3600:.2f} h  disk free {shutil.disk_usage(ROOT).free / 1e9:.0f} GB", ""]
        tot = sum(v["size"] for v in state.values()); got = sum(v["done"] for v in state.values())
        L.append(f"{'set':34s} {'files':>9s} {'GB done':>17s}")
        for s in SETS:
            vs = [v for (k, _), v in state.items() if k == s]
            L.append(f"{s:34s} {sum(v['status'] == 'done' for v in vs):>4d}/{len(vs):<4d} {sum(v['done'] for v in vs) / 1e9:>8.2f}/{sum(v['size'] for v in vs) / 1e9:<7.2f}")
        rate = got / max(time.time() - t0, 1)
        L += ["", f"total {got / 1e9:.1f} of {tot / 1e9:.1f} GB ({100 * got / max(tot, 1):.0f}%)   avg {rate / 1e6:.1f} MB/s   ETA {(tot - got) / max(rate, 1) / 3600:.1f} h"]
        L += [f"  {k[0]} {k[1]}: {v['done'] / 1e9:.2f}/{v['size'] / 1e9:.2f} GB {v['status']}" for k, v in state.items() if v["status"] not in ("done", "pending")]
        (ROOT / "status.txt.tmp").write_text("\n".join(L) + "\n"); os.replace(ROOT / "status.txt.tmp", ROOT / "status.txt")
        if all(v["status"] in ("done", "FAILED") for v in state.values()) and state:
            return
        time.sleep(5)


def main():
    jobs = []
    for s, (repo, ok) in SETS.items():
        for e in tree(repo):
            if e.get("size") and ok(e["path"]):
                state[(s, e["path"])] = {"size": e["size"], "done": 0, "status": "pending"}
                jobs.append((e["size"], s, repo, e))
    jobs.sort(key=lambda j: -j[0])  # big files first
    log(f"start: {len(jobs)} files, {sum(j[0] for j in jobs) / 1e9:.1f} GB, disk free {shutil.disk_usage(ROOT).free / 1e9:.0f} GB")
    th = threading.Thread(target=status); th.start()
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(lambda j: fetch(j[1], j[2], j[3]), jobs))
    th.join()
    man = {f"{s}/{p}": {"bytes": v["size"], "sha256": v.get("sha256")} for (s, p), v in state.items()}
    (ROOT / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    bad = [k for k, v in state.items() if v["status"] != "done"]
    log("finished: all files verified" if not bad else f"finished with failures: {bad[:5]}")
    sys.exit(1 if bad else 0)


main()
