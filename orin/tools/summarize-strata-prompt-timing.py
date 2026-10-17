import re, sys, collections, statistics
for path in sys.argv[1:]:
    tot = collections.Counter(); rb = []
    for line in open(path):
        for name, n, ms in re.findall(r" (\w+)(?:\((\d+)\))?=([\d.]+)", line.split("ms:")[1].split(" (batched")[0]):
            tot[name] += float(ms)
            if name == "read_batched": rb.append((int(n), float(ms)))
    small = [(n, ms) for n, ms in rb if n < 2048]
    mx = statistics.mean(n for n, _ in small); my = statistics.mean(m for _, m in small)
    s = sum((n - mx) * (m - my) for n, m in small) / sum((n - mx) ** 2 for n, _ in small)
    print(path.split("/")[-1], " ".join(f"{k}={v/1000:.1f}s" for k, v in sorted(tot.items(), key=lambda x: -x[1]) if v > 100),
          f"| batched<2048 fit {my - s*mx:.0f} ms + {s:.2f} ms/tok", "| ~100 tok:", [round(m) for n, m in sorted(small)[:4]])
