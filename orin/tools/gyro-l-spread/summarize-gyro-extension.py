"""Summarize a bounded extension campaign; missing evidence stays missing."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
import sys


def stats(values):
    values = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return {"samples": len(values), "min": min(values), "max": max(values),
            "mean": statistics.mean(values), "median": statistics.median(values),
            "stdev": statistics.stdev(values) if len(values)>1 else None,
            "first": values[0], "last": values[-1]} if values else None


def telemetry(path, intervals=None):
    fields = {"cpu_percent": ("cpu", "utilization_percent"),
              "used_bytes": ("memory", "used_bytes"), "free_bytes": ("memory", "free_bytes"),
              "swap_used_bytes": ("memory", "swap_used_bytes"),
              "gpu_percent": ("gpu", "device_utilization_percent"),
              "gpu_mhz": ("gpu", "frequency_mhz"), "gpu_temperature_c": ("gpu", "temperature_c"),
              "gpu_watts": ("gpu", "power_watts")}
    values = {k: [] for k in fields}
    pressure, first, last, incomplete = set(), None, None, 0
    cores, native = [], {k: [] for k in ("cpu_percent", "mem_resident_bytes", "gpu_percent", "gpu_mem_bytes")}
    total = None
    with path.open() as stream:
        for line in stream:
            try:
                s = json.loads(line)
            except json.JSONDecodeError:
                incomplete += 1
                continue
            if intervals is not None:
                epoch = datetime.fromisoformat(s["timestamp_utc"].replace("Z", "+00:00")).timestamp()
                if not any(a <= epoch <= b for a, b in intervals):
                    continue
            first = first or s["timestamp_utc"]
            last = s["timestamp_utc"]
            total = s["memory"]["total_bytes"]
            pressure.add(s["memory"]["pressure"])
            for k, (section, field) in fields.items():
                values[k].append(s.get(section, {}).get(field))
            for i, value in enumerate(s.get("cpu", {}).get("per_core_percent") or []):
                while len(cores) <= i:cores.append([])
                cores[i].append(value)
            for process in s.get("processes", []):
                command = process.get("command") or ""
                if "/gyro-engine-extension-20261019-v1/build-" in command and "/strata " in command:
                    for k in native:native[k].append(process.get(k))
    summary = {"first_utc": first, "last_utc": last, "total_bytes": total,
               "samples": len(values["used_bytes"]), "incomplete_lines": incomplete,
               "pressure_states": sorted(pressure), **{k: stats(v) for k, v in values.items()}}
    summary["per_core_cpu_percent"] = [stats(v) for v in cores]
    summary["candidate_native_process"] = {k: stats(v) for k, v in native.items()}
    if total and summary["used_bytes"]:
        summary["minimum_total_minus_used_bytes"] = total-summary["used_bytes"]["max"]
    return summary


def benchmark(path):
    result = json.loads((path / "result.json").read_text())
    protocol = json.loads((path / "protocol.json").read_text())
    assert result["protocol_sha256"] == hashlib.sha256((path / "protocol.json").read_bytes()).hexdigest()
    assert result["script_sha256"] == hashlib.sha256((path / "strata-gyro-benchmark.py").read_bytes()).hexdigest()
    exit_path = path / "exit-code.txt"
    code = int(exit_path.read_text()) if exit_path.exists() else None
    summary = {"status": result["status"], "exit_code": code,
               "mode_binaries": result.get("mode_binaries"), "modes": result["modes"], "groups": {}}
    if result["status"] == "completed":
        assert code == 0 and all(m["engine_exit_code"] == 0 for m in result["modes"])
    for mode in protocol["mode_order"]:
        lengths = protocol.get("mode_input_lengths", {}).get(mode, protocol["input_lengths"])
        for length in lengths:
            runs = [r for r in result["runs"] if r["mode"] == mode and r["input_tokens"] == length]
            if result["status"] == "completed":
                assert len(runs) == 4 and sum(r["warmup"] for r in runs) == 1
            measured = [r for r in runs if not r["warmup"]]
            if not measured:
                continue
            for r in measured:
                assert r["output_tokens"] == len(r["generated_ids"]) == 512
                assert r["internal"]["reused"] == 0 and r["internal"]["prompt_read"] == length
                assert r["internal"]["finish"] == "length"
                assert r["input_ids_sha256"] == result["inputs"][str(length)]["ids_sha256"]
            group = {"runs": len(measured), "input_ids_sha256": measured[0]["input_ids_sha256"]}
            assert hashlib.sha256((path / "prompts" / f"explanation-{length}.json").read_bytes()).hexdigest() == group["input_ids_sha256"]
            for k in ("internal_prefill_tps", "internal_decode_tps", "ttft_seconds", "elapsed_seconds", "stream_decode_tps"):
                group[k] = stats([r[k] for r in measured])
            for k in ("drafts_offered", "drafts_accepted", "offloaded", "hits", "lookups", "prompt_ms", "decode_ms"):
                group[k] = stats([r["internal"].get(k) for r in measured])
            # Fixed-profile token graph: no cache admissions during a token. Refused entries run in the CPU pool.
            # DONE.offloaded counts GPU work from outside its cache (PCIe/peer), not CPU fallback.
            assert all(0 <= r["internal"]["hits"] <= r["internal"]["lookups"] for r in measured)
            cpu_entries = [r["internal"]["lookups"]-r["internal"]["hits"] for r in measured]
            group["cpu_routed_entries"] = stats(cpu_entries)
            routed = sum(r["internal"]["lookups"]+r["internal"]["offloaded"] for r in measured)
            group["cpu_routed_fraction"] = sum(cpu_entries)/routed if routed else None
            offered = sum(r["internal"]["drafts_offered"] for r in measured)
            group["acceptance_fraction"] = sum(r["internal"]["drafts_accepted"] for r in measured)/offered if offered else None
            group["output_sha256"] = [r["output_ids_sha256"] for r in measured]
            group["natural_eos_positions"] = [r["natural_eos_positions"] for r in measured]
            if (path / "telemetry.ndjson").exists():
                group["telemetry"] = telemetry(path / "telemetry.ndjson", intervals=[
                    (r["started_unix_seconds"], r["ended_unix_seconds"]) for r in measured])
            summary["groups"][f"{mode}:{length}"] = group
    if {"m-baseline", "m-unified"}.issubset(protocol["mode_order"]):
        left = [r for r in result["runs"] if r["mode"] == "m-baseline" and r["input_tokens"] == 2048]
        right = [r for r in result["runs"] if r["mode"] == "m-unified" and r["input_tokens"] == 2048]
        summary["m_output_comparisons"] = []
        for a, b in zip(left, right):
            mismatch = next((i for i, (x, y) in enumerate(zip(a["generated_ids"], b["generated_ids"])) if x != y), None)
            summary["m_output_comparisons"].append({"index": a["index"], "warmup": a["warmup"],
                "same_input": a["input_ids_sha256"] == b["input_ids_sha256"],
                "same_output": a["generated_ids"] == b["generated_ids"], "first_mismatch_zero_based": mismatch})
        summary["m_output_comparison_scope"] = ("First 2K warmup has identical fresh-process request history. "
            "Measured M-unified runs additionally follow 25K/80K requests; learned draft policy may differ.")
    return summary


def main():
    campaign, out = map(Path, sys.argv[1:3])
    out.mkdir(parents=True, exist_ok=True)
    summary = {"campaign": str(campaign), "direct_exits": {}, "cases": {}, "telemetry": {}, "benchmarks": {}, "unreadable_json": {}}
    for p in sorted((campaign / "results").glob("*-exit-code.txt")):
        summary["direct_exits"][p.stem.removesuffix("-exit-code")] = int(p.read_text())
    for p in sorted((campaign / "results").glob("*.json")):
        try:
            summary["cases"][p.stem] = json.loads(p.read_text())
        except json.JSONDecodeError as error:
            summary["unreadable_json"][p.name] = {"bytes": p.stat().st_size,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "error": str(error)}
    context = campaign / "fixtures/context/result.json"
    if context.exists():
        summary["cases"]["l-context"] = json.loads(context.read_text())
    for p in sorted(campaign.glob("*-telemetry.ndjson")):
        summary["telemetry"][p.name.removesuffix("-telemetry.ndjson")] = telemetry(p)
    for name in ("perf", "perf-2048", "perf-v2", "perf-2048-v2"):
        p = campaign / name
        if (p / "result.json").exists():
            summary["benchmarks"][name] = benchmark(p)
            if (p / "telemetry.ndjson").exists():
                summary["telemetry"][name] = telemetry(p / "telemetry.ndjson")
    a, b = (summary["cases"].get(k) for k in ("cache-lru", "cache-spread"))
    if a and b and a["status"] == b["status"] == "completed":
        assert a["fixture_sha256"] == b["fixture_sha256"]
        comparisons = []
        for left, right in zip(a["requests"], b["requests"], strict=True):
            comparisons.append({"name": left["name"], "same_request": left["request_sha256"] == right["request_sha256"],
                                "same_assistant": left["assistant"] == right["assistant"]})
        summary["cache_comparisons"] = comparisons
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"direct_exits": summary["direct_exits"],
                      "benchmarks": {k: v["status"] for k, v in summary["benchmarks"].items()}}, indent=2))


if __name__ == "__main__":
    main()
