"""Matched resident IPC A/B of Gyro-M engine modes, with exact prompt/output contracts and raw receipts."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def chunk_for(protocol, mode):
    chunk = protocol["prefill_chunk"]
    return chunk[mode] if isinstance(chunk, dict) else chunk


def build_for(protocol, mode):
    """The build directory whose `strata` a mode runs (the pinned build-sm87 unless the protocol names another)."""
    return protocol.get("mode_build", {}).get(mode, "build-sm87")


def main():
    root, out = map(Path, sys.argv[1:3])
    protocol = json.loads((out / "protocol.json").read_text())
    source = root / "src/strata"
    sys.path[:0] = [str(source / "tools"), str(source)]
    from strata_tokenizer import Tokenizer
    from serve.server import StrataEngine

    settings = dict(line.split("=", 1) for line in (root / "runtime.env").read_text().splitlines()
                    if line and not line.startswith("#"))
    model = protocol.get("tokenizer_model", settings["GYRO_MODEL_PATH"])
    tokenizer = Tokenizer.from_gguf(model)
    prompts = out / "prompts"
    prompts.mkdir()
    head = ("<|im_start|>system\nYou are a helpful assistant. Give detailed, accurate answers. "
            "Use the user's language. Do not just copy the reference material.<|im_end|>\n"
            "<|im_start|>user\nReference material follows.\n")
    body = "".join(
        f"측정 메모 {i}: 요청 길이 {128 + (i * 97) % 8192}, 출력 {32 + i % 481}. "
        "첫 토큰 지연에는 대기, 프롬프트 계산, 첫 생성과 전송이 포함될 수 있다. "
        "캐시가 적중하면 이미 계산한 접두부를 재사용하지만 새 입력은 계산해야 한다. "
        f"관측 온도 {40 + i % 31}, 동시 요청 {1 + i % 7}. 평균만으로 꼬리 지연을 설명하지 않는다.\n"
        for i in range(protocol.get("body_rows", 400)))
    tail = ("\n\nTask:\n이 자료를 토대로 추론 시스템의 지연을 처음 배우는 개발자에게 설명해줘. "
            "프리필, TTFT, 디코딩, 캐시 적중의 관계를 구분하고, 개선 우선순위를 정하는 예와 "
            "잘못된 결론을 피하는 방법을 한국어로 자세히 정리해줘.\n"
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    body_ids = tokenizer.encode(body)
    fixed = len(tokenizer.encode(head, parse_special=True)) + len(tokenizer.encode(tail, parse_special=True))
    inputs, manifest = {}, {}
    for length in protocol["input_lengths"]:
        prefix = head + tokenizer.decode(body_ids[:length - fixed - 32]) + "\n"
        padding = 0
        for _ in range(64):
            text = prefix + " item" * padding + tail
            ids = tokenizer.encode(text, parse_special=True)
            if len(ids) == length:
                break
            padding += length - len(ids)
            assert padding >= 0, "Prompt prefix exceeds token budget"
        else:
            raise ValueError(f"Cannot construct {length}-token input")
        assert len(ids) == length and tokenizer.decode(ids) == text
        assert tokenizer.encode(text, parse_special=True) == ids
        text_path, ids_path = prompts / f"explanation-{length}.txt", prompts / f"explanation-{length}.json"
        text_path.write_text(text)
        write_json(ids_path, ids)
        inputs[length] = ids
        manifest[str(length)] = {"input_tokens": length, "text_sha256": digest(text_path.read_bytes()),
                                 "ids_sha256": digest(ids_path.read_bytes()), "roundtrip_verified": True}
    write_json(prompts / "manifest.json", manifest)

    class RecordedEngine(StrataEngine):
        def __init__(self, *args, **kwargs):
            self.receipts = []
            self.raw_done = None
            self.ipc_log = (Path(kwargs["log"]).with_suffix(".ipc.ndjson")).open("w")
            super().__init__(*args, **kwargs)

        def _pump(self):
            proc, lines = self.proc, self.lines
            for line in proc.stdout:
                entry = {"received_monotonic": time.monotonic(), "line": line.rstrip("\n")}
                self.receipts.append(entry)
                self.ipc_log.write(json.dumps(entry) + "\n")
                self.ipc_log.flush()
                lines.put(line)
            if self.proc is proc:
                self.ended = True
            lines.put(None)

        def _parse_done(self, line):
            self.raw_done = line.strip()
            super()._parse_done(line)

    result = {"status": "starting", "protocol_sha256": digest((out / "protocol.json").read_bytes()),
              "script_sha256": digest(Path(__file__).read_bytes()), "runtime": settings,
              "binary_sha256": digest((root / build_for(protocol, protocol["mode_order"][0]) / "strata").read_bytes()),
              "mode_binaries": {mode: {"build": build_for(protocol, mode),
                                       "sha256": digest((root / build_for(protocol, mode) / "strata").read_bytes())}
                                for mode in protocol["mode_order"]},
              "started_utc": datetime.now(timezone.utc).isoformat(), "inputs": manifest, "modes": [], "runs": []}
    path = out / "result.json"
    write_json(path, result)
    step = 0
    try:
        for mode in protocol["mode_order"]:
            folder = out / mode
            folder.mkdir()
            mode_model = protocol.get("mode_native", {}).get(mode, model)
            mode_pack = protocol.get("mode_pack", {}).get(mode, str(root / "packs/gyro-m"))
            mode_mtp = protocol.get("mtp_runtime", str(root / "mtp/rt"))
            args = ["--pack", mode_pack, "--native", mode_model, "--ple-gguf", mode_model,
                    "--mmap-experts", "--expert-profile", str(source / "data/expert-profile.bin"),
                    "--expert-cache", str(protocol["expert_cache"]), "--pool-workers", str(protocol["pool_workers"]),
                    "--prefill", str(chunk_for(protocol, mode)), "--max-context", str(protocol["max_context"]),
                    "--kv", protocol["kv"], "--prompt-cache", "0", "--prompt-cache-every", "0",
                    "--conversation-cache-mib", "0", "--short-read", "0", "--suffix-draft", "0",
                    "--spec", str(protocol["verifier_capacity"]), "--spec-min-p", str(protocol["spec_min_p"]),
                    "--mtp", mode_mtp, "--mtp-max-t", str(protocol["mtp_max_t"][mode]),
                    "--eos-ids", ",".join(map(str, protocol["eos_ids"])),
                    *protocol.get("mode_args", {}).get(mode, [])]
            mode_env = protocol.get("mode_env", {}).get(mode, {})
            entry = {"mode": mode, "args": args, "env": mode_env, "engine_exit_code": None}
            result["modes"].append(entry)
            result.update(status="loading", current_mode=mode)
            write_json(path, result)
            print(f"BENCH_LOADING mode={mode}", flush=True)
            engine = None
            try:
                env = dict(os.environ, STRATA_APR="1", STRATA_DECODE_TIMING="1", **mode_env)
                loaded_at = time.monotonic()
                engine = RecordedEngine(str(root / build_for(protocol, mode) / "strata"), args, cwd=str(source),
                                        log=str(folder / "engine.log"), env=env)
                entry.update(engine_pid=engine.proc.pid, engine_info=engine.info,
                             startup_seconds=time.monotonic() - loaded_at)
                write_json(path, result)
                for index in range(1 + protocol["measured_requests_per_length_and_mode"]):
                    for length in protocol.get("mode_input_lengths", {}).get(mode, protocol["input_lengths"]):
                        ids = inputs[length]
                        label = f"{mode}-{length}-{index}"
                        print(f"REQUEST_BEGIN mode={mode} input={length} run={index} warmup={int(index == 0)}", flush=True)
                        result.update(status="running", current_request=label)
                        write_json(path, result)
                        start_wall, start = time.time(), time.monotonic()
                        receipt_start = len(engine.receipts)
                        first = last = None
                        generated, received_at = [], []
                        last_pp = -1
                        for token in engine.generate(ids, protocol["output_tokens"], protocol["sampling"], threading.Event()):
                            if token is None:
                                if engine.progress and engine.progress[0] != last_pp:
                                    last_pp = engine.progress[0]
                                    print(f"PREFILL_PROGRESS mode={mode} input={length} run={index} read={last_pp} "
                                          f"tps={engine.prefill_tok_s_mean}", flush=True)
                                continue
                            last = time.monotonic()
                            first = last if first is None else first
                            generated.append(token)
                            received_at.append(last - start)
                        end, end_wall = time.monotonic(), time.time()
                        internal = dict(engine.last)
                        raw = engine.receipts[receipt_start:]
                        pp = [entry["line"].split() for entry in raw if entry["line"].startswith("PP ")]
                        assert len(generated) == protocol["output_tokens"] == internal["generated"]
                        assert internal["finish"] == "length"
                        assert internal["reused"] == 0 and internal["prompt_read"] == length
                        assert internal["prompt_tokens"] == length
                        assert pp and int(pp[-1][1]) == length - 1 and int(pp[-1][2]) == length
                        assert internal["prompt_ms"] > 0 and internal["decode_ms"] > 0
                        # The prompt path can borrow expert-cache slots and refill
                        # them from mmap files. Only decode offload must be zero.
                        if protocol.get("require_zero_decode_offload", True):
                            assert internal["offloaded"] == 0
                        if protocol["mtp_max_t"][mode] == 1:
                            assert internal["drafts_offered"] == internal["drafts_accepted"] == 0
                        else:
                            assert 0 <= internal["drafts_accepted"] <= internal["drafts_offered"] and internal["drafts_offered"] > 0
                        run = {"mode": mode, "input_tokens": length, "index": index, "warmup": index == 0,
                               "started_unix_seconds": start_wall, "ended_unix_seconds": end_wall,
                               "input_ids_sha256": manifest[str(length)]["ids_sha256"], "output_tokens": len(generated),
                               "generated_ids": generated, "token_received_seconds": received_at,
                               "output_ids_sha256": digest(json.dumps(generated, separators=(",", ":")).encode()),
                               "text": tokenizer.decode(generated), "natural_eos_positions": [i + 1 for i, t in enumerate(generated) if t in (248044, 248046)],
                               "ttft_seconds": first - start, "elapsed_seconds": end - start,
                               "stream_decode_tps": (len(generated) - 1) / (last - first),
                               "internal_prefill_tokens": length - 1,
                               "internal_prefill_tps": (length - 1) * 1000 / internal["prompt_ms"],
                               "internal_decode_tps": len(generated) * 1000 / internal["decode_ms"],
                               "last_pp_ms": float(pp[-1][3]), "internal": internal, "raw_done": engine.raw_done}
                        result["runs"].append(run)
                        step += 1
                        write_json(path, result)
                        print(f"BENCH_STEP step={step} mode={mode} input={length} run={index} "
                              f"prefill_tps={run['internal_prefill_tps']:.2f} ttft={run['ttft_seconds']:.3f} "
                              f"decode_tps={run['stream_decode_tps']:.2f} elapsed={run['elapsed_seconds']:.3f} "
                              f"accepted={internal['drafts_accepted']} offered={internal['drafts_offered']}", flush=True)
            finally:
                if engine is not None:
                    native_process = engine.proc
                    engine.close()
                    engine.pump.join(timeout=5)
                    engine.ipc_log.close()
                    entry["engine_exit_code"] = native_process.returncode
                    write_json(path, result)
            assert entry["engine_exit_code"] == 0, f"Engine exit was {entry['engine_exit_code']}"
        expected = sum(len(protocol.get("mode_input_lengths", {}).get(mode, protocol["input_lengths"]))
                       for mode in protocol["mode_order"]) * (1 + protocol["measured_requests_per_length_and_mode"])
        assert len(result["runs"]) == expected
        result.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(path, result)


if __name__ == "__main__":
    main()
