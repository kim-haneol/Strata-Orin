"""Compare greedy replays of the same recorded requests under different server configurations.

Usage: compare-strata-replays.py BASE_REPLAY.json OTHER_REPLAY.json [...]
Checks that every request produced the same content and tool calls, then compares decode (server timings:
predicted_n / predicted_ms), prompt work and wall time, overall and for requests generating >= 300 tokens.
"""
import json
import sys


def load(path):
    return {r["request"]: r for r in json.load(open(path))}


def output(r):
    """The generated output: content and tool calls without their per-response random ids."""
    calls = [{k: v for k, v in c.items() if k != "id"} for c in (r["tool_calls"] or [])]
    return r["content"], calls


def totals(rows):
    gen = sum(r["timings"]["predicted_n"] for r in rows)
    dms = sum(r["timings"]["predicted_ms"] for r in rows)
    pms = sum(r["timings"]["prompt_ms"] for r in rows)
    wall = sum(r["wall_seconds"] for r in rows)
    return gen, dms, pms, wall


def main():
    paths = sys.argv[1:]
    runs = [load(p) for p in paths]
    names = sorted(set.intersection(*(set(r) for r in runs)))
    base = runs[0]
    print(f"{len(names)} requests in common")
    for path, run in zip(paths[1:], runs[1:]):
        diff = [n for n in names if output(run[n]) != output(base[n])]
        print(f"{path}: {len(diff)} requests differ from {paths[0]} in output" + (f" (first: {diff[:3]})" if diff else ""))
    print("replay | decode tok/s (all) | decode tok/s (>=300-token requests) | prompt s | wall s")
    long_names = [n for n in names if base[n]["timings"]["predicted_n"] >= 300]
    for path, run in zip(paths, runs):
        g, d, p, w = totals([run[n] for n in names])
        gl, dl, _, _ = totals([run[n] for n in long_names])
        print(f"{path} | {g / d * 1000:.2f} ({g} tok) | {gl / dl * 1000:.2f} ({gl} tok, {len(long_names)} req) | "
              f"{p / 1000:.1f} | {w:.1f}")


if __name__ == "__main__":
    main()
