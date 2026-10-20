"""Frozen HTTP context, recall and API/cache checks for the deployed Gyro-M engine.

Run with the deployed uv/venv on Jetson. Raw synthetic requests stay outside git.
The server uses one 262144 capacity throughout; prompt token counts use its tokenizer/template.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def http(base, endpoint, body=None, timeout=10800):
    request = urllib.request.Request(base + endpoint,
        data=None if body is None else json.dumps(body, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def tokenizer(base):
    sys.path[:0] = [str(base / "src/strata/tools"), str(base / "src/strata")]
    from strata_tokenizer import Tokenizer
    from serve.frontend import ChatTemplate, openai_to_messages
    p = base / "packs/gyro-m/tokenizer"
    vocab = json.loads((p / "vocab.json").read_text())
    tokens = [None] * len(vocab)
    for token, index in vocab.items():
        tokens[index] = token
    tok = Tokenizer(tokens, (p / "merges.txt").read_text().split("\n"),
                    json.loads((p / "token_type.json").read_text()))
    template = ChatTemplate(p / "chat_template.jinja")
    def encode(body):
        messages, tools, kwargs = openai_to_messages(body)
        return tok.encode(template.render(messages, tools=tools, **kwargs), parse_special=True)
    return tok, encode


def freeze(base, out):
    out.mkdir(parents=True, exist_ok=True)
    tok, encode = tokenizer(base)
    manifest = {"capacity": 262144, "lengths": [64000, 128000, 253952],
                "max_tokens": 512, "reserved_tokens_at_max": 8192,
                "seed": "gyro-context-20261009-v1", "requests": [],
                "recall_scope": "three synthetic dispersed identifiers; not general long-context quality",
                "script_sha256": sha(Path(__file__)), "config_sha256": sha(base / "server-262k-validation.json"),
                "binary_sha256": sha(base / "build-sm87-f16cache/strata.f16cache")}
    for target in manifest["lengths"]:
        expected = {"BEGIN": "Cedar-4827", "MIDDLE": "Quartz-6193", "END": "Maple-7351"}
        head = f"Validation dataset {target}. Recover only the three TARGET identifiers.\nTARGET_BEGIN={expected['BEGIN']}\n"
        middle = f"\nTARGET_MIDDLE={expected['MIDDLE']}\n"
        tail = (f"\nTARGET_END={expected['END']}\nEnd of dataset.\n"
                "Return the exact values of TARGET_BEGIN, TARGET_MIDDLE and TARGET_END in a JSON object "
                "with keys BEGIN, MIDDLE, END. Ignore ordinary row identifiers. Output only JSON.")
        # Independent, deterministic filler for each length; no prefix reuse between long tests.
        rows = "".join(f"row {i:06d}; id {hashlib.sha256(f'{target}/{i}'.encode()).hexdigest()[:12]}; "
                       f"count {(i * 97) % 10007}; state {'ready' if i % 3 else 'pending'}; note archival entry.\n"
                       for i in range(14000))
        filler_ids = tok.encode(rows)
        body = {"model": "qwen38-flash-next-gyro-m", "temperature": 0,
                "reasoning_effort": "none", "max_tokens": 512,
                "messages": [{"role": "system", "content": "You recover identifiers accurately from supplied text."},
                             {"role": "user", "content": ""}]}
        count = target - 300
        for _ in range(10):
            left = tok.decode(filler_ids[:count // 2])
            right = tok.decode(filler_ids[count // 2:count])
            body["messages"][1]["content"] = head + left + middle + right + tail
            delta = target - len(encode(body))
            if delta >= 0:
                break
            count += delta - 8
        padding = ""
        original = body["messages"][1]["content"]
        for _ in range(32):
            current = len(encode(body))
            if current == target:
                break
            difference = target - current
            assert difference > 0, (target, current)
            padding += " item" * difference
            body["messages"][1]["content"] = original + padding
        ids = encode(body)
        assert len(ids) == target, (target, len(ids))
        path = out / f"request-{target}.json"
        write(path, body)
        positions = {}
        for key, value in expected.items():
            text = body["messages"][1]["content"]
            positions[key] = len(tok.encode(text[:text.index(f"TARGET_{key}=")]))
        manifest["requests"].append({"target_tokens": target, "request_file": path.name,
                                     "request_sha256": sha(path), "ids_sha256": hashlib.sha256(
                                         json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
                                     "expected": expected, "marker_user_token_offsets": positions})
        print(f"FROZEN tokens={target} sha256={sha(path)}", flush=True)
    write(out / "manifest.json", manifest)


def run(base, out, url, model_id="qwen38-flash-next-gyro-m", check_parking=True):
    result = {"status": "running", "manifest_sha256": sha(out / "manifest.json"), "tests": []}
    path = out / "result.json"
    manifest = json.loads((out / "manifest.json").read_text())
    step = 0
    def request(name, body, check, expected_prompt=None):
        nonlocal step
        print(f"CTX_BEGIN name={name}", flush=True)
        start = time.monotonic()
        response = http(url, "/v1/chat/completions", body)
        elapsed = time.monotonic() - start
        record = {"name": name, "elapsed_seconds": elapsed, "response": response,
                  "passed": bool(check(response))}
        if expected_prompt is not None:
            record["expected_prompt_tokens"] = expected_prompt
            record["actual_prompt_tokens"] = response.get("usage", {}).get("prompt_tokens")
            record["token_count_matches"] = record["actual_prompt_tokens"] == expected_prompt
            record["passed"] &= record["token_count_matches"]
        result["tests"].append(record)
        write(path, result)
        step += 1
        print(f"CTX_STAGE step={step} name={name} passed={record['passed']} elapsed_s={elapsed:.2f}", flush=True)
        if not record["passed"]:
            raise RuntimeError(f"validation failed: {name}; see result.json")
        return response
    try:
        deadline = time.monotonic() + 300
        while True:
            try:
                result["health"] = http(url, "/health", timeout=5)
                break
            except (urllib.error.URLError, TimeoutError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(1)
        assert result["health"]["max_context"] == 262144
        result["models"] = http(url, "/v1/models")
        common = {"model": model_id, "temperature": 0, "reasoning_effort": "none", "max_tokens": 256}
        request("short-korean", {**common, "messages": [{"role": "user", "content": "한국어로 짧게 답해줘. 17 더하기 25는 얼마야?"}]},
                lambda r: "42" in r["choices"][0]["message"].get("content", ""))
        for entry in manifest["requests"]:
            p = out / entry["request_file"]
            assert sha(p) == entry["request_sha256"]
            expected = entry["expected"]
            frozen_body = json.loads(p.read_text())
            frozen_body["model"] = model_id
            request(f"recall-{entry['target_tokens']}", frozen_body,
                    lambda r, expected=expected: all(v in r["choices"][0]["message"].get("content", "") for v in expected.values()),
                    entry["target_tokens"])
        # Explicit API tool call and continuation; no filesystem access in the model server.
        tool = {"type": "function", "function": {"name": "lookup_test_value", "description": "Look up the requested test identifier.",
                "parameters": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}}}
        messages = [{"role": "user", "content": "Call lookup_test_value with key ORIN_CHECK. Do not guess its value."}]
        response = request("tool-call", {**common, "messages": messages, "tools": [tool]},
                           lambda r: any(t["function"]["name"] == "lookup_test_value" and
                               json.loads(t["function"]["arguments"]).get("key") == "ORIN_CHECK"
                               for t in r["choices"][0]["message"].get("tool_calls", [])))
        assistant = response["choices"][0]["message"]
        messages += [assistant, {"role": "tool", "tool_call_id": assistant["tool_calls"][0]["id"], "content": "ORIN_CHECK=5391"},
                     {"role": "user", "content": "What exact value did the tool return? Answer in one sentence."}]
        request("tool-result", {**common, "messages": messages, "tools": [tool]},
                lambda r: "5391" in r["choices"][0]["message"].get("content", ""))
        if check_parking:
            # A/B/A tests parked conversation restore; prefixes exceed the root checkpoint threshold.
            histories = []
            for label, value in [("A", "7319"), ("B", "8642")]:
                messages = [{"role": "system", "content": f"Conversation {label}. " + "Keep the given code for later. " * 700},
                            {"role": "user", "content": f"My code is {value}. Repeat it once."}]
                response = request(f"conversation-{label}", {**common, "messages": messages},
                                   lambda r, value=value: value in r["choices"][0]["message"].get("content", ""))
                histories.append((label, value, messages + [response["choices"][0]["message"]]))
            for label, value, messages in histories:
                response = request(f"conversation-{label}-resume", {**common, "messages": messages + [
                    {"role": "user", "content": "What was my code? Reply only with its digits."}]},
                    lambda r, value=value: value in r["choices"][0]["message"].get("content", ""))
                assert response.get("timings", {}).get("cache_n", 0) > 0, "no cache reuse"
        else:
            result.setdefault("not_run", []).append({"name": "parked-conversation-restore",
                "reason": "whole-conversation parking disabled by the approved memory plan"})
        # Real SSE stream, including usage and clean terminal marker.
        body = {**common, "messages": [{"role": "user", "content": "In English, state briefly why a prefix cache speeds up repeated prompts."}],
                "stream": True, "stream_options": {"include_usage": True}}
        req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        chunks, done = [], False
        with urllib.request.urlopen(req, timeout=300) as stream:
            for line in stream:
                if line.startswith(b"data: "):
                    data = line[6:].strip()
                    if data == b"[DONE]":
                        done = True
                    else:
                        chunks.append(json.loads(data))
        content = "".join(c.get("choices", [{}])[0].get("delta", {}).get("content", "") for c in chunks if c.get("choices"))
        assert done and content and any(c.get("usage") for c in chunks)
        result["tests"].append({"name": "stream-english", "passed": True, "content": content, "done": done,
                                "usage": next(c["usage"] for c in chunks if c.get("usage"))})
        result["status"] = "completed"
        print("CTX_COMPLETE", flush=True)
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write(path, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["freeze", "run"])
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:18092")
    parser.add_argument("--model-id", default="qwen38-flash-next-gyro-m")
    parser.add_argument("--no-parking-check", action="store_true")
    args = parser.parse_args()
    if args.mode == "freeze":
        freeze(args.base, args.out)
    else:
        run(args.base, args.out, args.url, args.model_id, not args.no_parking_check)
