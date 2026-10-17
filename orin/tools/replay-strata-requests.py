"""Replay recorded agent requests against a Strata server, greedy, in their original order.

Each request body recorded by strata-request-recorder.py is sent again with temperature 0 and without streaming,
so every server configuration produces the same outputs and only the speed differs. The prompt cache sees the
same sequence of prefixes as in the real session. Per request it records the server's own timings and the
output (text and tool calls) for the identity check between configurations.

Usage: replay-strata-requests.py REQUESTS_DIR BASE_URL OUT_JSON [FIRST LAST]
"""
import json
from pathlib import Path
import sys
import time
import urllib.request


def main():
    req_dir, base, out = Path(sys.argv[1]), sys.argv[2].rstrip("/"), Path(sys.argv[3])
    files = sorted(req_dir.glob("*-v1_chat_completions.json"))
    if len(sys.argv) > 5:
        first, last = int(sys.argv[4]), int(sys.argv[5])
        files = [f for f in files if first <= int(f.name.split("-")[0]) <= last]
    results = []
    for f in files:
        body = json.loads(f.read_text())["body"]
        body = {k: v for k, v in body.items() if k not in ("stream", "stream_options")}
        body["temperature"] = 0
        t0 = time.monotonic()
        req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            resp = json.load(r)
        wall = time.monotonic() - t0
        msg = resp["choices"][0]["message"]
        results.append({"request": f.name, "wall_seconds": wall, "usage": resp.get("usage"),
                        "timings": resp.get("timings"), "finish_reason": resp["choices"][0].get("finish_reason"),
                        "content": msg.get("content"), "tool_calls": msg.get("tool_calls")})
        t = resp.get("timings") or {}
        print(f"{f.name}: prompt {t.get('prompt_n')} ({t.get('cache_n')} cached) {t.get('prompt_ms')} ms, "
              f"generated {t.get('predicted_n')} in {t.get('predicted_ms')} ms "
              f"({t.get('predicted_per_second')} tok/s), wall {wall:.1f}s", flush=True)
        out.write_text(json.dumps(results, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
