"""Rebuild flagged_rows/<set>.csv and summary.csv from the record files (ids and positions only, no text).

    python3 scripts/export_flagged.py <folder with *.record.json or *.record.json.gz> <folder with the run progress.log files> <out folder>

The id of a flagged row is `<set>/<n>`; n = unit * 100_000_000 + row number in that unit's source file (0-based, counting
rows the converter later skipped). The unit -> file mapping comes from the `unit start` lines in the run logs.
"""
import csv, glob, gzip, json, re, sys
from pathlib import Path

rec_dir, log_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
STRIDE = 100_000_000
units = {}
for lg in glob.glob(str(log_dir / "*progress.log")) + glob.glob(str(log_dir / "*/progress.log")):
    for ln in open(lg, encoding="utf-8"):
        m = re.search(r"unit start (\S+) u(\d+) (\S.*?) \(", ln)
        if m:
            units[(m[1], int(m[2]))] = m[3]
(out / "flagged_rows").mkdir(parents=True, exist_ok=True)
summary = []
for p in sorted(glob.glob(str(rec_dir / "*.record.json*"))):
    r = json.load(gzip.open(p, "rt", encoding="utf-8") if p.endswith(".gz") else open(p, encoding="utf-8"))
    s, rows = r["set"], []
    for did, v in r["flagged"].items():
        u, row = divmod(int(did.rsplit("/", 1)[1]), STRIDE)
        f = units.get((s, u)) or ("(single source file or files in order)" if u == 0 else "?")
        rows.append((did, f, row, ";".join(v["benchmarks"]), v["n_items"], v.get("max_coverage")))
    rows.sort(key=lambda x: (x[1], x[2]))
    with open(out / "flagged_rows" / f"{s}.csv", "w", newline="", encoding="utf-8") as o:
        w = csv.writer(o)
        w.writerow(["id", "source_file", "row_in_file_0based", "benchmarks", "n_items", "max_coverage"])
        w.writerows(rows)
    summary.append((s, r["rows_in"], r["rows_flagged"], r["rows_canary"], r["rows_removed"], r["rows_kept"], r["run_at"], r["ok"]))
with open(out / "summary.csv", "w", newline="", encoding="utf-8") as o:
    w = csv.writer(o)
    w.writerow(["set", "rows_in", "rows_flagged", "rows_canary", "rows_removed", "rows_kept", "run_at", "ok"])
    w.writerows(summary)
print(f"{len(summary)} sets")
