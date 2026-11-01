#!/usr/bin/env python3
"""Local server for the ERA V5 capstone task board.

Serves index.html and keeps all task data in one JSON file (tasks.json, next to this script).
Python standard library only. Listens on 127.0.0.1, so only this computer can reach it.

    python3 server.py              start on port 8765 and open the browser
    python3 server.py --no-open    do not open a browser
    python3 server.py --port 9000 --data /path/to/tasks.json

Every save writes tasks.json atomically (temp file, then rename). A copy of the previous file goes to
backups/ (at most one every 5 minutes, newest 40 kept; always one before an import or restore).
"""
import argparse
import json
import os
import re
import sys
import tempfile
import threading
import time
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATUSES = {"todo", "progress", "review", "blocked", "done"}
PRIORITIES = {"high", "medium", "low"}
TYPES = {"task", "recurring", "milestone"}
REF_KINDS = {"pr", "commit", "issue", "comment", "link"}
MAX_BODY = 5 * 1024 * 1024
BACKUP_EVERY_S = 300
BACKUP_KEEP = 40
BACKUP_NAME = re.compile(r"^tasks-\d{8}-\d{6}(-[a-z]+)?\.json$")
LOCK = threading.Lock()


class Store:
    """The JSON file and its backups."""

    def __init__(self, path):
        self.path = Path(path).resolve()
        self.dir = self.path.parent / "backups"

    def read(self):
        with open(self.path, encoding="utf-8") as f:
            return json.load(f)

    def write(self, data):
        text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".tasks-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def backup(self, tag="", force=False):
        if not self.path.exists():
            return None
        self.dir.mkdir(exist_ok=True)
        names = sorted(p.name for p in self.dir.glob("tasks-*.json"))
        if not force and names:
            newest = self.dir / names[-1]
            if time.time() - newest.stat().st_mtime < BACKUP_EVERY_S:
                return None
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = "tasks-%s%s.json" % (stamp, "-" + tag if tag else "")
        dest = self.dir / name
        dest.write_bytes(self.path.read_bytes())
        for old in sorted(p.name for p in self.dir.glob("tasks-*.json"))[:-BACKUP_KEEP]:
            (self.dir / old).unlink()
        return name

    def backups(self):
        if not self.dir.exists():
            return []
        out = []
        for p in sorted(self.dir.glob("tasks-*.json"), reverse=True):
            try:
                n = len(json.loads(p.read_text(encoding="utf-8")).get("tasks", []))
            except (ValueError, OSError):
                n = None
            out.append({"name": p.name, "size": p.stat().st_size, "tasks": n,
                        "time": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")})
        return out


def validate(data):
    """Return an error message, or None when the document is usable."""
    if not isinstance(data, dict) or not isinstance(data.get("tasks"), list):
        return "document must be an object with a tasks list"
    seen = set()
    for i, t in enumerate(data["tasks"]):
        where = "task %d" % (i + 1)
        if not isinstance(t, dict):
            return where + " is not an object"
        tid = t.get("id")
        if not isinstance(tid, str) or not tid:
            return where + " has no id"
        if tid in seen:
            return "duplicate id " + tid
        seen.add(tid)
        if not isinstance(t.get("title"), str) or not t["title"].strip():
            return tid + " has no title"
        if t.get("status") not in STATUSES:
            return tid + " has an unknown status"
        if t.get("priority") not in PRIORITIES:
            return tid + " has an unknown priority"
        if t.get("type", "task") not in TYPES:
            return tid + " has an unknown type"
        if not isinstance(t.get("assignees", []), list) or not all(isinstance(a, str) for a in t.get("assignees", [])):
            return tid + " has bad assignees"
        for r in t.get("refs", []):
            if not isinstance(r, dict) or r.get("kind") not in REF_KINDS or not isinstance(r.get("ref"), str):
                return tid + " has a bad PR / commit link"
        for c in t.get("comments", []):
            if not isinstance(c, dict) or not isinstance(c.get("text"), str):
                return tid + " has a bad comment"
    if not isinstance(data.get("repos", {}), dict):
        return "repos must be an object"
    return None


def make_handler(store, port):
    allowed_hosts = {"127.0.0.1:%d" % port, "localhost:%d" % port}
    allowed_origins = {"http://" + h for h in allowed_hosts}
    index = HERE / "index.html"

    class Handler(BaseHTTPRequestHandler):
        server_version = "TaskBoard/1"

        def log_message(self, fmt, *args):
            sys.stderr.write("%s %s\n" % (self.log_date_time_string(), fmt % args))

        def send_json(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def guard(self, write):
            """Refuse requests that did not come from our own page (DNS rebinding and cross-site writes)."""
            if self.headers.get("Host", "") not in allowed_hosts:
                self.send_json(403, {"error": "bad host"})
                return False
            if write:
                origin = self.headers.get("Origin")
                if (origin and origin not in allowed_origins) or self.headers.get("X-Taskboard") != "1":
                    self.send_json(403, {"error": "forbidden"})
                    return False
            return True

        def read_body(self):
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > MAX_BODY:
                self.send_json(413, {"error": "body is empty or too large"})
                return None
            try:
                return json.loads(self.rfile.read(n).decode("utf-8"))
            except ValueError:
                self.send_json(400, {"error": "body is not valid JSON"})
                return None

        def do_GET(self):
            if not self.guard(False):
                return
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                body = index.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif path == "/api/data":
                with LOCK:
                    try:
                        self.send_json(200, store.read())
                    except (OSError, ValueError) as e:
                        self.send_json(500, {"error": "cannot read %s: %s" % (store.path.name, e)})
            elif path == "/api/rev":
                with LOCK:
                    try:
                        self.send_json(200, {"rev": store.read().get("rev", 0)})
                    except (OSError, ValueError) as e:
                        self.send_json(500, {"error": str(e)})
            elif path == "/api/backups":
                with LOCK:
                    self.send_json(200, {"backups": store.backups()})
            elif path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
            else:
                self.send_json(404, {"error": "not found"})

        def do_PUT(self):
            if self.path.split("?", 1)[0] != "/api/data":
                return self.send_json(404, {"error": "not found"})
            if not self.guard(True):
                return
            body = self.read_body()
            if body is None:
                return
            data, base, force = body.get("data"), body.get("baseRev"), bool(body.get("force"))
            err = validate(data)
            if err:
                return self.send_json(400, {"error": err})
            with LOCK:
                try:
                    current = store.read()
                except (OSError, ValueError) as e:
                    return self.send_json(500, {"error": "cannot read %s: %s" % (store.path.name, e)})
                if not force and current.get("rev", 0) != base:
                    return self.send_json(409, {"error": "tasks.json changed on disk", "rev": current.get("rev", 0)})
                tag = "import" if force else ""
                store.backup(tag, force=force)
                data["rev"] = current.get("rev", 0) + 1
                data["updated"] = datetime.now().strftime("%Y-%m-%d")
                store.write(data)
                self.send_json(200, {"rev": data["rev"], "updated": data["updated"]})

        def do_POST(self):
            if self.path.split("?", 1)[0] != "/api/restore":
                return self.send_json(404, {"error": "not found"})
            if not self.guard(True):
                return
            body = self.read_body()
            if body is None:
                return
            name = body.get("name", "")
            if not isinstance(name, str) or not BACKUP_NAME.match(name) or not (store.dir / name).is_file():
                return self.send_json(400, {"error": "unknown backup"})
            try:
                data = json.loads((store.dir / name).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return self.send_json(400, {"error": "backup is not readable"})
            err = validate(data)
            if err:
                return self.send_json(400, {"error": "backup is invalid: " + err})
            with LOCK:
                current = store.read()
                store.backup("restore", force=True)
                data["rev"] = current.get("rev", 0) + 1
                data["updated"] = datetime.now().strftime("%Y-%m-%d")
                store.write(data)
            self.send_json(200, {"rev": data["rev"]})

    return Handler


def main():
    ap = argparse.ArgumentParser(description="ERA V5 task board server")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--data", default=str(HERE / "tasks.json"), help="path of the JSON data file")
    ap.add_argument("--no-open", action="store_true", help="do not open a browser")
    a = ap.parse_args()
    store = Store(a.data)
    if not store.path.exists():
        sys.exit("Data file not found: %s" % store.path)
    try:
        err = validate(store.read())
    except (OSError, ValueError) as e:
        sys.exit("Cannot read %s: %s" % (store.path, e))
    if err:
        sys.exit("%s is not a valid task file: %s" % (store.path, err))
    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(store, a.port))
    except OSError as e:
        sys.exit("Cannot listen on 127.0.0.1:%d (%s). Is the board already running? Try --port." % (a.port, e))
    url = "http://127.0.0.1:%d/" % a.port
    print("Task board: %s\nData file:  %s\nPress Ctrl+C to stop." % (url, store.path))
    if not a.no_open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
