"""Short natural-EOS correctness and streaming check for a candidate API."""
import argparse
import importlib.util
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, required=True)
p.add_argument("--model", required=True)
p.add_argument("--url", default="http://127.0.0.1:18094")
a = p.parse_args()
spec = importlib.util.spec_from_file_location("cache_client", Path(__file__).with_name("bench-gyro-cache-branch.py"))
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)
result = client.request(a.url, {"model": a.model, "temperature": 0, "reasoning_effort": "none",
    "max_tokens": 128, "messages": [{"role": "user", "content": "한국어로 짧게 답해줘. 17 더하기 25는 얼마야?"}]})
result["passed"] = "42" in result["assistant"]["content"]
client.save(a.out, result)
print("PROBE_COMPLETE " + json.dumps({k: result[k] for k in ("passed", "usage", "timings", "ttft_seconds", "wall_seconds")}), flush=True)
raise SystemExit(0 if result["passed"] else 1)
