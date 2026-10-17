"""Nsight Systems captures of Gyro-M decode and prefill on the adopted resident serving setup.

The engine starts under `nsys launch` (CUDA graph nodes traced individually). Loading and a short warmup are
not captured. Each capture is one request bracketed by `nsys start` / `nsys stop`, with its wall-clock
request, first-token and end times recorded for splitting the trace into prefill and decode.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import threading
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def main():
    root, out = map(Path, sys.argv[1:3])
    protocol = json.loads((out / "protocol.json").read_text())
    source = root / "src/strata"
    sys.path[:0] = [str(source)]
    from serve.server import StrataEngine

    settings = dict(line.split("=", 1) for line in (root / "runtime.env").read_text().splitlines()
                    if line and not line.startswith("#"))
    model = settings["GYRO_MODEL_PATH"]
    prompts = out / "prompts"
    prompts.mkdir()
    inputs = {}
    for length in sorted({c["input_tokens"] for c in [protocol["warmup"], *protocol["captures"]]}):
        src = Path(protocol["prompt_source"]) / f"explanation-{length}.json"
        dst = prompts / src.name
        shutil.copyfile(src, dst)
        inputs[length] = json.loads(dst.read_text())
        assert len(inputs[length]) == length

    build = protocol.get("build", "build-sm87")   # the pinned build unless the protocol names an experiment
    tool = protocol.get("tool", "nsys")
    if tool == "nsys":
        nsys = protocol["nsys"]
        session = protocol["session"]
        command = [nsys["binary"], "launch", f"--session-new={session}", f"--trace={nsys['trace']}",
                   f"--cuda-graph-trace={nsys['cuda_graph_trace']}"]
    else:
        # Nsight Compute: kernels matching the filter are profiled as they run (graph nodes individually);
        # nothing is bracketed, so the filter and launch count decide what is captured.
        ncu = protocol["ncu"]
        command = [ncu["binary"], "--graph-profiling", "node", *ncu["filter_args"],
                   "--set", ncu.get("set", "full"), "--import-source", "no", "-f", "-o", str(out / ncu["report"])]
    wrapper = out / f"{tool}-strata.sh"
    wrapper.write_text("#!/bin/bash\nexec " + " ".join(map(shlex.quote, [*command, str(root / build / "strata")]))
                       + ' "$@"\n')
    wrapper.chmod(0o755)

    args = ["--pack", str(root / "packs/gyro-m"), "--native", model, "--ple-gguf", model,
            "--mmap-experts", "--expert-profile", str(source / "data/expert-profile.bin"),
            "--expert-cache", str(protocol["expert_cache"]), "--pool-workers", str(protocol["pool_workers"]),
            "--prefill", str(protocol["prefill_chunk"]), "--max-context", str(protocol["max_context"]),
            "--kv", protocol["kv"], "--prompt-cache", "0", "--prompt-cache-every", "0",
            "--conversation-cache-mib", "0", "--short-read", "0", "--suffix-draft", "0",
            "--spec", str(protocol["verifier_capacity"]), "--spec-min-p", str(protocol["spec_min_p"]),
            "--mtp", str(root / "mtp/rt"), "--mtp-max-t", str(protocol["mtp_max_t"]),
            "--eos-ids", ",".join(map(str, protocol["eos_ids"])), *protocol.get("extra_args", [])]
    result = {"status": "starting", "protocol_sha256": digest(out / "protocol.json"),
              "script_sha256": digest(__file__), "runtime": settings,
              "build": build, "binary_sha256": digest(root / build / "strata"), "args": args, "wrapper": wrapper.read_text(),
              "started_utc": datetime.now(timezone.utc).isoformat(), "captures": []}
    path = out / "result.json"
    write_json(path, result)

    def run_request(ids, max_new):
        start_wall, start = time.time(), time.monotonic()
        first = None
        count = 0
        for token in engine.generate(ids, max_new, protocol["sampling"], threading.Event()):
            if token is None:
                continue
            count += 1
            if first is None:
                first, first_wall = time.monotonic(), time.time()
        end, end_wall = time.monotonic(), time.time()
        assert count == max_new and engine.last["generated"] == max_new and engine.last["reused"] == 0
        return {"input_tokens": len(ids), "max_new": max_new, "started_unix_seconds": start_wall,
                "first_token_unix_seconds": first_wall, "ended_unix_seconds": end_wall,
                "ttft_seconds": first - start, "elapsed_seconds": end - start,
                "stream_decode_tps": (count - 1) / (end - first) if count > 1 else None,
                "internal": dict(engine.last)}

    engine = None
    try:
        env = dict(os.environ, STRATA_APR="1", STRATA_DECODE_TIMING="1", **protocol.get("env", {}))
        loaded_at = time.monotonic()
        print("PROFILE_LOADING", flush=True)
        engine = StrataEngine(str(wrapper), args, cwd=str(source), log=str(out / "engine.log"), env=env)
        result.update(status="warmup", engine_info=engine.info, startup_seconds=time.monotonic() - loaded_at)
        write_json(path, result)
        warm = protocol["warmup"]
        result["warmup"] = run_request(inputs[warm["input_tokens"]], warm["max_new"])
        print(f"PROFILE_WARMUP done ttft={result['warmup']['ttft_seconds']:.3f}", flush=True)
        for capture in protocol["captures"]:
            name = capture["name"]
            if tool != "nsys":
                record = run_request(inputs[capture["input_tokens"]], capture["max_new"])
                record.update(name=name)
                result["captures"].append(record)
                write_json(path, result)
                print(f"PROFILE_CAPTURE name={name} ttft={record['ttft_seconds']:.3f}", flush=True)
                continue
            report = out / name
            # Sampling switches belong to `start`, not `launch` (nsys rejects --cpuctxsw on launch).
            subprocess.run([nsys["binary"], "start", f"--session={session}", f"--output={report}",
                            "--sample=none", "--cpuctxsw=none", "--force-overwrite=true"], check=True)
            started = time.time()
            try:
                record = run_request(inputs[capture["input_tokens"]], capture["max_new"])
            finally:
                subprocess.run([nsys["binary"], "stop", f"--session={session}"], check=True)
            record.update(name=name, nsys_start_called_unix_seconds=started,
                          nsys_stop_returned_unix_seconds=time.time(), report=f"{name}.nsys-rep")
            result["captures"].append(record)
            write_json(path, result)
            print(f"PROFILE_CAPTURE name={name} ttft={record['ttft_seconds']:.3f} "
                  f"decode_tps={record['stream_decode_tps']} elapsed={record['elapsed_seconds']:.3f}", flush=True)
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if engine is not None:
            native = engine.proc
            engine.close()
            result["engine_exit_code"] = native.returncode
        write_json(path, result)
    assert result["engine_exit_code"] == 0, f"Engine exit was {result['engine_exit_code']}"
    for capture in result["captures"] if tool == "nsys" else []:
        report = out / capture["report"]
        assert report.exists(), f"Missing {report}"
        subprocess.run([nsys["binary"], "export", "--type=sqlite", "--force-overwrite=true",
                        f"--output={report.with_suffix('.sqlite')}", str(report)], check=True)
        stats = out / f"{capture['name']}-stats"
        subprocess.run([nsys["binary"], "stats", "--force-export=false", "--format=csv",
                        "--report=cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum",
                        f"--output={stats}", str(report.with_suffix(".sqlite"))], check=True)
    result.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(path, result)


if __name__ == "__main__":
    main()
