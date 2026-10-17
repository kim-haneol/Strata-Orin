# Decode cost table of the sub8 serving build (graph-level Nsight Systems)

Captured on 2026-10-08 with `config/strata-gyro-profile-sub8-graph-v1.json`: the same 2K / 256-token decode
capture as the node-level profile (`evidence/strata-gyro-profile-sub8-20261008-v1`), but with CUDA graphs traced
as whole graphs. Tracing overhead is small: 23.96 tok/s traced, 25.8 untraced.

## Measured

- GPU busy (union of graph executions) is 10.46 s of the 10.64 s decode phase (98.3%). There are no
  significant host gaps between graphs. This does not show that the time inside the graphs is all useful
  compute: a graph-level range also contains its internal kernel boundaries, dependencies and memory waits.
- Tracing overhead, compared like for like: the first 256 tokens of the earlier untraced 512-token runs decode
  at 24.33-24.41 tok/s, against 23.96 traced (about 1.5-1.9%; different runs).
- 117 verify windows for 256 tokens (2.19 tokens per window in this capture). Verify graphs by window size:

| Window tokens T | Verify graph | Count |
|---:|---:|---:|
| 1 | 57.5 ms | 27 |
| 2 | 71.2 ms | 29 |
| 3 | 84.6 ms | 21 |
| 4 | 97.7 ms | 40 |

  A straight line fits: **verify ≈ 44.1 ms + 13.4 ms × T**.
- Per window there is also a 2.9 ms graph on the verify stream, plus MTP draft graphs on stream 14 (3.5-5.4 ms
  each, 1.13 s in total, about 9.7 ms per window).

## Cost table (scenario estimate, not a measured non-overlapping decomposition)

The rows come from different sources: standalone benches scaled to 48 layers, node-level shares (classified by
kernel name, so MTP draft-layer kernels can also land in the dense and HC rows) and graph times. Overlap between
streams is not removed, and the rows can double-count.

Untraced window at 2K: 92.4 ms for 2.38 tokens (25.8 tok/s). Reaching 30 tok/s needs about 79 ms (−13 ms per
window).

| Part | Per window (estimate) | Basis |
|---|---:|---|
| Routed experts (T ≈ 2.8) | ~31 ms | grouped bench: 247 / 464 / 902 µs per layer at T = 1/2/4, × 48 layers |
| Dense Q8_0 projections + LM head | ~30 ms | node-level share (39%); already 120-150 GB/s |
| Hyper-connection read | ~14 ms | 101 calls × 109-151 µs (gr bench) |
| MTP draft (layer + head) | ~10 ms | draft graphs |
| Attention, GDN, routing, other | ~8 ms | node-level shares |

- The T-dependent part (13.4 ms per token) is mostly routed experts (about 11-12 ms per token) plus
  per-token hyper-connection work (about 2.4 ms per token).
- Under the scenario that nothing else changes, the 13 ms would have to come from the experts (31 → 18 ms).
  That is one scenario, not the only path the measurements allow. The expert kernels run at 53-58 GB/s.
- A rough ceiling from codebook gathers alone (8 random 2-byte shared reads per 16 weights, about 3.4 wavefronts
  per warp gather) puts the experts at about 100 GB/s. That leaves up to ~1.8× of room, which instruction
  overheads between the gathers currently consume. This ceiling is an estimate, not a measured bound. It holds only for the current lookup count, 2-byte access
  and work split. Designs that reuse lookups or change the representation or schedule are not bounded by it.

**Summary:** with the current implementation and this workload, further micro-optimization gives diminishing
returns, and no validated design yet delivers the ~14% time cut that 30 tok/s needs. Stopping speed work is a
cost decision. Neither an impossibility of 30 tok/s nor a 26-27 tok/s ceiling has been shown.
