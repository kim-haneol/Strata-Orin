"""Recording HTTP proxy for the Strata OpenAI-compatible server.

Listens on LISTEN (default 127.0.0.1:18093), forwards every request to UPSTREAM (default 127.0.0.1:18092) and
streams the response back unchanged (SSE included). Every POST body is saved whole as OUT_DIR/NNNN-<path>.json
with its arrival time, so the exact requests an agent sent can be replayed later.

Usage: strata-request-recorder.py OUT_DIR [LISTEN_HOST:PORT] [UPSTREAM_HOST:PORT]
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import http.client
import itertools
import json
from pathlib import Path
import sys
import threading
import time

OUT = Path(sys.argv[1])
LISTEN = sys.argv[2] if len(sys.argv) > 2 else "127.0.0.1:18093"
UPSTREAM = sys.argv[3] if len(sys.argv) > 3 else "127.0.0.1:18092"
OUT.mkdir(parents=True, exist_ok=True)
counter = itertools.count(len(list(OUT.glob("*.json"))) + 1)
lock = threading.Lock()
HOP = {"connection", "keep-alive", "transfer-encoding", "content-length", "proxy-connection", "upgrade", "te"}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"   # close-delimited responses: streams pass through without re-chunking

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (time.strftime("%H:%M:%S"), fmt % args))

    def _forward(self, method):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        if method == "POST":
            with lock:
                n = next(counter)
            name = self.path.strip("/").replace("/", "_") or "root"
            try:
                parsed = json.loads(body)
            except ValueError:
                parsed = body.decode("utf-8", "replace")
            (OUT / f"{n:04d}-{name}.json").write_text(json.dumps(
                {"received_unix": time.time(), "path": self.path, "body": parsed}, ensure_ascii=False) + "\n")
        host, port = UPSTREAM.rsplit(":", 1)
        conn = http.client.HTTPConnection(host, int(port), timeout=3600)
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        if body:
            headers["Content-Length"] = str(len(body))
        conn.request(method, self.path, body=body or None, headers=headers)
        resp = conn.getresponse()
        self.send_response(resp.status, resp.reason)
        for k, v in resp.getheaders():
            if k.lower() not in HOP:
                self.send_header(k, v)
        self.send_header("Connection", "close")
        self.end_headers()
        while True:
            chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536)
            if not chunk:
                break
            self.wfile.write(chunk)
            self.wfile.flush()
        conn.close()

    def do_GET(self):
        self._forward("GET")

    def do_POST(self):
        self._forward("POST")


if __name__ == "__main__":
    lh, lp = LISTEN.rsplit(":", 1)
    print(f"recording {LISTEN} -> {UPSTREAM} into {OUT}", flush=True)
    ThreadingHTTPServer((lh, int(lp)), Handler).serve_forever()
