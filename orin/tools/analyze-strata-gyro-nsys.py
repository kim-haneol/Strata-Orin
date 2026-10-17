"""Split Gyro-M Nsight Systems captures into prefill/decode phases and attribute kernel time by kernel and category."""
import json
from pathlib import Path
import sqlite3
import sys

# Category by kernel short name prefix. The MTP draft head is the small Q8_0 MMVQ over the 106,299-token draft
# vocabulary (grid X = rows), told apart from other small MMVQ calls by its grid.
CATEGORIES = [
    ("routed_experts", ("native_gu", "native_down", "mul_mat_q", "quantize_mmq", "moe_combine", "swiglu")),
    ("dense_projections", ("native_mmvq", "native_small_mmvq", "bf16_f32_mmvf", "native_quantize_q8_1",
                           "ampere_", "Kernel2", "dequant_kernel", "cutlass", "gemm")),
    ("hyper_connection", ("gr_",)),
    ("attention", ("attn_", "prompt_attn", "block_topk", "block_scores", "qsa", "rope")),
    ("gdn", ("gdn_",)),
    ("routing", ("route", "router", "topk")),
]
DRAFT_HEAD_ROWS = 106299


def category(name, grid_x):
    if name.startswith("native_small_mmvq") and grid_x == DRAFT_HEAD_ROWS:
        return "draft_head"
    for cat, prefixes in CATEGORIES:
        if name.startswith(prefixes):
            return cat
    return "other"


def union_ns(intervals):
    busy, cs, ce = 0, None, None
    for s, e in sorted(intervals):
        if ce is None or s > ce:
            if ce is not None:
                busy += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return busy + (ce - cs if ce is not None else 0)


def phase_summary(rows, lo, hi, top):
    sel = [r for r in rows if r[0] >= lo and r[1] <= hi]
    total = sum(e - s for s, e, *_ in sel)
    by_kernel, by_cat = {}, {}
    for s, e, name, grid_x in sel:
        k = by_kernel.setdefault(name, [0, 0])
        k[0] += e - s
        k[1] += 1
        c = category(name, grid_x)
        by_cat[c] = by_cat.get(c, 0) + e - s
    return {"wall_seconds": (hi - lo) / 1e9, "kernels": len(sel), "kernel_sum_seconds": total / 1e9,
            "gpu_busy_union_seconds": union_ns([(s, e) for s, e, *_ in sel]) / 1e9,
            "categories": {c: {"seconds": t / 1e9, "share": t / total} for c, t in
                           sorted(by_cat.items(), key=lambda x: -x[1])},
            "top_kernels": [{"name": n, "seconds": t / 1e9, "share": t / total, "launches": k,
                             "avg_us": t / k / 1e3} for n, (t, k) in
                            sorted(by_kernel.items(), key=lambda x: -x[1][0])[:top]]}


def main():
    base = Path(sys.argv[1])
    result = json.loads((base / "result.json").read_text())
    summary = {}
    for capture in result["captures"]:
        db = sqlite3.connect(base / f"{capture['name']}.sqlite")
        origin = db.execute("select utcEpochNs from TARGET_INFO_SESSION_START_TIME").fetchone()[0]
        rows = db.execute("select k.start, k.end, s.value, k.gridX from CUPTI_ACTIVITY_KIND_KERNEL k "
                          "join StringIds s on k.shortName = s.id").fetchall()
        at = lambda key: int(capture[key] * 1e9) - origin
        start, first, end = at("started_unix_seconds"), at("first_token_unix_seconds"), at("ended_unix_seconds")
        summary[capture["name"]] = {
            "request": {k: capture[k] for k in ("input_tokens", "max_new", "ttft_seconds", "elapsed_seconds",
                                                 "stream_decode_tps")},
            "prefill": phase_summary(rows, start, first, 25),
            "decode": phase_summary(rows, first, end, 25) if capture["max_new"] > 8 else None}
    (base / "kernel-phases.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name, s in summary.items():
        for phase in ("prefill", "decode"):
            p = s[phase]
            if p is None:
                continue
            print(f"{name} {phase}: wall {p['wall_seconds']:.2f}s, GPU busy {p['gpu_busy_union_seconds']:.2f}s, "
                  + ", ".join(f"{c} {v['share']:.1%}" for c, v in p["categories"].items()))


if __name__ == "__main__":
    main()
