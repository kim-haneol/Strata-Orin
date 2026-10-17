"""Compare decode-phase kernel durations of two Gyro-M Nsight Systems captures, per kernel and launch grid.

Usage: compare-strata-gyro-nsys-kernels.py BASE_RUN_DIR CANDIDATE_RUN_DIR [CAPTURE] [KERNEL_PREFIX ...]
Each run directory holds result.json and <capture>.sqlite from scripts/strata-gyro-profile.py.  The routed-expert
kernels launch with grid Y = 10 x T (T = verify-window tokens), so grouping by grid shows the cost per window size.
"""
import json
from pathlib import Path
import sqlite3
import sys


def decode_kernels(run, capture):
    result = json.loads((run / "result.json").read_text())
    cap = next(c for c in result["captures"] if c["name"] == capture)
    db = sqlite3.connect(run / f"{capture}.sqlite")
    origin = db.execute("select utcEpochNs from TARGET_INFO_SESSION_START_TIME").fetchone()[0]
    first = int(cap["first_token_unix_seconds"] * 1e9) - origin
    end = int(cap["ended_unix_seconds"] * 1e9) - origin
    rows = db.execute("select s.value, k.gridX, k.gridY, k.end - k.start from CUPTI_ACTIVITY_KIND_KERNEL k "
                      "join StringIds s on k.shortName = s.id where k.start >= ? and k.end <= ?", (first, end))
    out = {}
    for name, gx, gy, dur in rows:
        x = out.setdefault((name, gx, gy), [0, 0])
        x[0] += dur
        x[1] += 1
    return out, cap


def main():
    base, cand = Path(sys.argv[1]), Path(sys.argv[2])
    capture = sys.argv[3] if len(sys.argv) > 3 else "decode2k"
    prefixes = tuple(sys.argv[4:]) or ("native_gu_multi", "native_down_multi")
    a, cap_a = decode_kernels(base, capture)
    b, cap_b = decode_kernels(cand, capture)
    print(f"stream decode tok/s (traced): base {cap_a['stream_decode_tps']:.3f}, candidate {cap_b['stream_decode_tps']:.3f}")
    total_a = sum(v[0] for k, v in a.items() if k[0].startswith(prefixes))
    total_b = sum(v[0] for k, v in b.items() if k[0].startswith(prefixes))
    print(f"selected kernels, decode phase: base {total_a / 1e6:.1f} ms, candidate {total_b / 1e6:.1f} ms "
          f"({total_b / total_a:.3f}x)")
    print("kernel | grid | base n, avg us | candidate n, avg us | ratio")
    for key in sorted(set(a) | set(b), key=lambda k: (k[0], k[2], k[1])):
        if not key[0].startswith(prefixes):
            continue
        na, ta = a.get(key, (0, 0))[::-1]
        nb, tb = b.get(key, (0, 0))[::-1]
        avg_a = ta / na / 1e3 if na else float("nan")
        avg_b = tb / nb / 1e3 if nb else float("nan")
        print(f"{key[0][:24]} | ({key[1]},{key[2]}) | {na}, {avg_a:.1f} | {nb}, {avg_b:.1f} | {avg_b / avg_a:.3f}")


if __name__ == "__main__":
    main()
