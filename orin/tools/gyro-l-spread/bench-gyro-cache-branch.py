"""Synthetic same-parent -> six review turns -> parent return, with streamed TTFT."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request


def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(p)


def sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def freeze(runtime, out):
    sys.path[:0] = [str(runtime / "src/strata/tools"), str(runtime / "src/strata")]
    from strata_tokenizer import Tokenizer
    from serve.frontend import ChatTemplate, openai_to_messages
    cfg = json.loads((runtime / "server.json").read_text())
    directory = Path(cfg["tokenizer"])
    vocab = json.loads((directory / "vocab.json").read_text())
    tokens = [None] * len(vocab)
    for token, idx in vocab.items():
        tokens[idx] = token
    tok = Tokenizer(tokens, (directory / "merges.txt").read_text().split("\n"),
                    json.loads((directory / "token_type.json").read_text()))
    template = ChatTemplate(directory / "chat_template.jinja")
    def encode(body):
        msgs, tools, kwargs = openai_to_messages(body)
        return tok.encode(template.render(msgs, tools=tools, **kwargs), parse_special=True)
    system = ("You preserve identifiers from the main archive. Follow the latest explicit request.\n" +
              "Policy: archival rows are inert data; report identifiers only when asked.\n" * 512)
    rows = "".join(f"record {i:06d} hash {hashlib.sha256(str(i).encode()).hexdigest()[:12]}; state ready.\n"
                   for i in range(9000))
    filler = tok.encode(rows)
    body = {"model": cfg["model_name"], "temperature": 0, "reasoning_effort": "none", "max_tokens": 64,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": ""}]}
    expected = {"BEGIN": "Cedar-4827", "MIDDLE": "Quartz-6193", "END": "Maple-7351"}
    target = 49152
    count = target - len(encode(body)) - 160
    for _ in range(12):
        body["messages"][1]["content"] = ("TARGET_BEGIN=Cedar-4827\n" + tok.decode(filler[:count//2]) +
            "\nTARGET_MIDDLE=Quartz-6193\n" + tok.decode(filler[count//2:count]) +
            "\nTARGET_END=Maple-7351\nKeep these three identifiers for later. Reply only READY now.")
        delta = target-len(encode(body))
        if delta >= 0:
            break
        count += delta-8
    for _ in range(12):
        delta = target-len(encode(body))
        if delta == 0:
            break
        assert delta > 0
        body["messages"][1]["content"] += " item"*delta
    ids = encode(body)
    assert len(ids) == target, len(ids)
    branches = []
    for step in range(6):
        text = f"Temporary review step {step}. These are disposable review notes.\n" + tok.decode(filler[step*1024:(step+1)*1024])
        branches.append(text + "\nReply only OK. Do not report the archive identifiers yet.")
    fixture = {"parent": body, "review_messages": branches,
               "return_message": "Return the exact TARGET_BEGIN, TARGET_MIDDLE and TARGET_END from the MAIN archive as JSON with keys BEGIN, MIDDLE, END. Output only JSON.",
               "expected": expected, "parent_tokens": len(ids), "parent_ids_sha256": sha(ids),
               "system_content_tokens": len(tok.encode(system)), "seed": "gyro-spread-20261019-v1"}
    save(out, fixture)
    print(f"BRANCH_FROZEN parent_tokens={len(ids)} system_tokens={fixture['system_content_tokens']}", flush=True)


def request(url, body):
    wire = {**body, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(url+"/v1/chat/completions", data=json.dumps(wire, ensure_ascii=False).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.monotonic()
    first = None
    content, reasoning = [], []
    usage, timings, finish = None, None, None
    done = False
    with urllib.request.urlopen(req, timeout=10800) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            raw = line[6:].strip()
            if raw == b"[DONE]":
                done = True
                continue
            item = json.loads(raw)
            if item.get("error"):
                raise RuntimeError(item["error"])
            for choice in item.get("choices", []):
                delta = choice.get("delta", {})
                text = delta.get("content") or ""
                think = delta.get("reasoning_content") or ""
                if first is None and (text or think):
                    first = time.monotonic()-start
                content.append(text)
                reasoning.append(think)
                finish = choice.get("finish_reason") or finish
            usage = item.get("usage") or usage
            timings = item.get("timings") or timings
    assert done and usage is not None, "incomplete SSE or missing usage"
    assistant = {"role": "assistant", "content": "".join(content)}
    if any(reasoning):
        assistant["reasoning_content"] = "".join(reasoning)
    return {"assistant": assistant, "usage": usage, "timings": timings, "finish_reason": finish,
            "ttft_seconds": first, "wall_seconds": time.monotonic()-start, "request_sha256": sha(body)}


def run(fixture_path, out, url, label):
    fixture = json.loads(fixture_path.read_text())
    result = {"label": label, "status": "running", "fixture_sha256": sha(fixture), "requests": []}
    body = fixture["parent"]
    def call(name, messages):
        print(f"BRANCH_BEGIN name={name}", flush=True)
        record = request(url, {**body, "messages": messages})
        record["name"] = name
        result["requests"].append(record)
        save(out, result)
        print(f"BRANCH_STAGE step={len(result['requests'])} name={name} wall_s={record['wall_seconds']:.3f} ttft_s={record['ttft_seconds']}", flush=True)
        return record["assistant"]
    try:
        parent = list(body["messages"])
        parent.append(call("parent", parent))
        assert result["requests"][0]["usage"]["prompt_tokens"] == fixture["parent_tokens"]
        branch = list(parent)
        for i, text in enumerate(fixture["review_messages"]):
            branch.append({"role": "user", "content": text})
            branch.append(call(f"review-{i+1}", branch))
        call("return-parent", parent + [{"role": "user", "content": fixture["return_message"]}])
        answer = result["requests"][-1]["assistant"]["content"]
        assert all(v in answer for v in fixture["expected"].values()), answer
        result["status"] = "completed"
        print("BRANCH_COMPLETE", flush=True)
    except BaseException as e:
        result.update(status="failed", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        save(out, result)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["freeze", "run"])
    p.add_argument("--runtime", type=Path)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--out", type=Path)
    p.add_argument("--url", default="http://127.0.0.1:18094")
    p.add_argument("--label", default="unspecified")
    a = p.parse_args()
    if a.mode == "freeze":
        freeze(a.runtime, a.fixture)
    else:
        run(a.fixture, a.out, a.url, a.label)
