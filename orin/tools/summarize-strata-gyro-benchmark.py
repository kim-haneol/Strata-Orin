"""Validate matched Gyro-M IPC results and summarize request-scoped ASTOP and tegrastats evidence."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys


def stats(values):
    values = [v for v in values if v is not None]
    return {"samples": len(values), "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None,
            "stdev": statistics.stdev(values) if len(values) > 1 else None}


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def main():
    base = Path(sys.argv[1])
    result = json.loads((base / "result.json").read_text())
    protocol = json.loads((base / "protocol.json").read_text())
    assert result["status"] == "completed"
    assert int((base / "exit-code.txt").read_text()) == 0
    assert all(mode["engine_exit_code"] == 0 for mode in result["modes"])
    assert result["protocol_sha256"] == digest(base / "protocol.json")
    assert result["script_sha256"] == digest(base / "strata-gyro-benchmark.py")
    baseline, *candidates = protocol.get("comparison", ["none", "mtp"])
    # One candidate keeps the v1 keys (input length); several are keyed "candidate:length".
    pair_key = (lambda mode, length: str(length)) if len(candidates) == 1 else (lambda mode, length: f"{mode}:{length}")
    chunk = protocol["prefill_chunk"]
    mode_args = protocol.get("mode_args", {})
    comparable_args = []
    for mode in result["modes"]:
        args = list(mode["args"])
        assert args[args.index("--mtp-max-t") + 1] == str(protocol["mtp_max_t"][mode["mode"]])
        args[args.index("--mtp-max-t") + 1] = "<mode>"
        expected_chunk = chunk[mode["mode"]] if isinstance(chunk, dict) else chunk
        assert args[args.index("--prefill") + 1] == str(expected_chunk)
        args[args.index("--prefill") + 1] = "<mode>"
        extra = mode_args.get(mode["mode"], [])
        assert args[len(args) - len(extra):] == extra
        comparable_args.append(args[:len(args) - len(extra)])
    assert all(args == comparable_args[0] for args in comparable_args), \
        "Engine conditions differ beyond the MTP window cap, prefill chunk and declared mode arguments"
    tier_reads = {}
    for mode in result["modes"]:
        # Cumulative "GGUF in place" bytes after each request, in request order.
        log = (base / mode["mode"] / "engine.log").read_text()
        totals = [float(v) for v in re.findall(r"expert tiers: .* files \d+ blobs ([0-9.]+) MB read", log)]
        runs = [r for r in result["runs"] if r["mode"] == mode["mode"]]
        assert len(totals) == len(runs), "One expert-tier line per request expected"
        for i, run in enumerate(runs):
            tier_reads[(run["mode"], run["input_tokens"], run["index"])] = totals[i] - totals[i - 1] if i else None
    metrics = ("internal_prefill_tps", "internal_decode_tps", "ttft_seconds", "stream_decode_tps",
               "elapsed_seconds", "last_pp_ms")
    summary = {"protocol": protocol["protocol"], "binary_sha256": result["binary_sha256"],
               "comparison": {"baseline": baseline, "candidate": candidates[0] if len(candidates) == 1 else candidates},
               "engines": {m["mode"]: {k: m["engine_info"].get(k) for k in ("expert_slots", "vram_free_mib", "mtp_max")}
                           for m in result["modes"]},
               "groups": {}, "speedups": {}, "output_comparisons": {}, "telemetry": {}, "vendor_phases": {}}
    grouped = {}
    for mode in protocol["mode_order"]:
        for length in protocol["input_lengths"]:
            key = f"{mode}-{length}"
            runs = [r for r in result["runs"] if r["mode"] == mode and r["input_tokens"] == length]
            assert len(runs) == 4 and len([r for r in runs if r["warmup"]]) == 1
            runs = [r for r in runs if not r["warmup"]]
            assert len(runs) == 3 and sorted(r["index"] for r in runs) == [1, 2, 3]
            assert all(r["output_tokens"] == 512 and len(r["generated_ids"]) == 512
                       and r["internal"]["generated"] == 512 and r["internal"]["finish"] == "length"
                       and r["internal"]["reused"] == 0 and r["internal"]["prompt_read"] == length
                       and r["internal"]["offloaded"] == 0 for r in runs)
            ids_path = base / "prompts" / f"explanation-{length}.json"
            assert digest(ids_path) == result["inputs"][str(length)]["ids_sha256"]
            assert all(r["input_ids_sha256"] == digest(ids_path) for r in runs)
            grouped[key] = runs
            group = {metric: stats([r[metric] for r in runs]) for metric in metrics}
            for metric in ("prompt_ms", "decode_ms", "drafts_accepted", "drafts_offered", "file_blobs", "file_mb", "ram_blobs"):
                group[metric] = stats([r["internal"][metric] for r in runs])
            offered = sum(r["internal"]["drafts_offered"] for r in runs)
            accepted = sum(r["internal"]["drafts_accepted"] for r in runs)
            group["acceptance_fraction"] = accepted / offered if offered else None
            group["gguf_in_place_mb_delta"] = stats([tier_reads[(mode, length, r["index"])] for r in runs])
            group["natural_eos_positions"] = [r["natural_eos_positions"] for r in runs]
            group["output_variants"] = len({r["output_ids_sha256"] for r in runs})
            summary["groups"][key] = group
    for candidate, length in ((c, n) for c in candidates for n in protocol["input_lengths"]):
        off, on = (summary["groups"][f"{mode}-{length}"] for mode in (baseline, candidate))
        summary["speedups"][pair_key(candidate, length)] = {
            "stream_decode": on["stream_decode_tps"]["mean"] / off["stream_decode_tps"]["mean"],
            "internal_decode": on["internal_decode_tps"]["mean"] / off["internal_decode_tps"]["mean"],
            "end_to_end": off["elapsed_seconds"]["mean"] / on["elapsed_seconds"]["mean"],
            "internal_prefill": on["internal_prefill_tps"]["mean"] / off["internal_prefill_tps"]["mean"],
            "ttft": off["ttft_seconds"]["mean"] / on["ttft_seconds"]["mean"]}
        comparisons = []
        for left, right in zip(grouped[f"{baseline}-{length}"], grouped[f"{candidate}-{length}"]):
            mismatch = next((i for i, pair in enumerate(zip(left["generated_ids"], right["generated_ids"]))
                             if pair[0] != pair[1]), None)
            comparisons.append({"index": left["index"], "exact_match": mismatch is None,
                                "first_mismatch_zero_based": mismatch})
        summary["output_comparisons"][pair_key(candidate, length)] = comparisons

    buckets = {key: [] for key in grouped}
    for line in (base / "telemetry.ndjson").open():
        sample = json.loads(line)
        epoch = datetime.fromisoformat(sample["timestamp_utc"].replace("Z", "+00:00")).timestamp()
        for key, runs in grouped.items():
            if any(r["started_unix_seconds"] <= epoch <= r["ended_unix_seconds"] for r in runs):
                buckets[key].append(sample)
    for key, samples in buckets.items():
        telemetry = {"samples": len(samples), "pressure_states": sorted({s["memory"]["pressure"] for s in samples})}
        for field in ("device_utilization_percent", "frequency_mhz", "temperature_c", "power_watts"):
            telemetry[f"gpu_{field}"] = stats([s["gpu"].get(field) for s in samples])
        for field in ("free_bytes", "swap_used_bytes", "used_bytes"):
            telemetry[f"memory_{field}"] = stats([s["memory"].get(field) for s in samples])
        telemetry["cpu_percent"] = stats([s["cpu"].get("utilization_percent") for s in samples])
        per_core = [s["cpu"].get("per_core_percent", []) for s in samples]
        telemetry["cpu_per_core_peak_percent"] = [max(row[i] for row in per_core) for i in range(len(per_core[0]))] if per_core else []
        engine_pid = next(m["engine_pid"] for m in result["modes"] if m["mode"] == key.split("-")[0])
        processes = [p for s in samples for p in s.get("processes", []) if p["pid"] == engine_pid]
        telemetry["engine_cpu_percent"] = stats([p.get("cpu_percent") for p in processes])
        telemetry["engine_gpu_percent"] = stats([p.get("gpu_percent") for p in processes])
        telemetry["engine_gpu_memory_bytes"] = stats([p.get("gpu_mem_bytes") for p in processes])
        telemetry["engine_resident_memory_bytes"] = stats([p.get("mem_resident_bytes") for p in processes])
        summary["telemetry"][key] = telemetry
    vendor = base / "vendor-tegrastats.log"
    if vendor.exists():
        readings = {}
        for name, value in re.findall(r"\b([A-Za-z0-9_]+)@([0-9.]+)C", vendor.read_text()):
            readings.setdefault(name, []).append(float(value))
        summary["vendor_whole_run_temperature_c"] = {name: stats(values) for name, values in readings.items()}
        # tegrastats is UTC wall time at 1 s resolution. Split each measured request into
        # prefill (start..first token) and decode (first token..end), trimming one second at
        # each edge so that boundary samples are not attributed to the wrong phase.
        samples = []
        for line in vendor.read_text().splitlines():
            m = re.match(r"(\d\d-\d\d-\d{4} \d\d:\d\d:\d\d) .*?CPU \[(.*?)\] GR3D_FREQ (\d+)%.*?VDD_GPU_SOC (\d+)mW", line)
            if m:
                epoch = datetime.strptime(m[1] + "+0000", "%m-%d-%Y %H:%M:%S%z").timestamp()
                cores = [int(c.split("%")[0]) for c in m[2].split(",") if "%" in c]
                samples.append((epoch, int(m[3]), int(m[4]) / 1000, max(cores, default=None)))
        for key, runs in grouped.items():
            phases = {"prefill": [], "decode": []}
            for r in runs:
                start, first = r["started_unix_seconds"], r["started_unix_seconds"] + r["ttft_seconds"]
                phases["prefill"] += [s for s in samples if start + 1 <= s[0] <= first - 1]
                phases["decode"] += [s for s in samples if first + 1 <= s[0] <= r["ended_unix_seconds"] - 1]
            summary["vendor_phases"][key] = {phase: {
                "samples": len(rows), "gr3d_percent": stats([s[1] for s in rows]),
                "gr3d_below_50_fraction": sum(s[1] < 50 for s in rows) / len(rows) if rows else None,
                "vdd_gpu_soc_watts": stats([s[2] for s in rows]),
                "busiest_cpu_core_percent": stats([s[3] for s in rows])} for phase, rows in phases.items()}
    (base / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print("| Input | Mode | Prefill tok/s | Prompt s | TTFT s | Stream decode tok/s | Total s |")
    print("|---:|:---:|---:|---:|---:|---:|---:|")
    for length in protocol["input_lengths"]:
        for mode in protocol["mode_order"]:
            g = summary["groups"][f"{mode}-{length}"]
            print(f"| {length} | {mode} | {g['internal_prefill_tps']['mean']:.2f} | "
                  f"{g['prompt_ms']['mean']/1000:.3f} | {g['ttft_seconds']['mean']:.3f} | "
                  f"{g['stream_decode_tps']['mean']:.2f} | {g['elapsed_seconds']['mean']:.3f} |")
    for length, speedup in summary["speedups"].items():
        print(f"{length}: prefill {speedup['internal_prefill']:.3f}x; TTFT {speedup['ttft']:.3f}x; "
              f"decode {speedup['stream_decode']:.3f}x; end-to-end {speedup['end_to_end']:.3f}x")


if __name__ == "__main__":
    main()
