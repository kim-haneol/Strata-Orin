# TQ2_T expert kernels, second round: A-B-B-A decode benchmark on Orin

Run on 2026-10-08 with `config/strata-gyro-sub8-ab-v1.json`. Engine starts alternate a1 → b1 → b2 → a2:

- **a** is the serving build `build-sm87-mmq-lut` (strata `e4566270…`).
- **b** is `build-sm87-sub8` (strata `2690172711be…`, kept read-only as `strata.sub8`), built from
  `patches/strata-orin-tq2t-rpw4-sub8.patch` plus the MMQ fork patch.

The patch adds two changes on top of the staged row and codebook:

- **gate/up:** each warp takes 4 rows (32 per block). A group of one entry (most decode groups) reads its q8_1
  activations once per lane step and reuses them for all 4 rows.
- **down:** the K = 40 rows (n_ff 640) use the existing 8-lane sub-warp path `row_dot_40_sub8`, now with the
  staged codebook. Before, the 32-lane loop ran its second step on only 8 lanes.

Both keep row_dot_multi's summation order and tree.

## Kernel-level evidence (decode-shaped grouped bench, locked 1.30 GHz)

`scripts/bench-strata-gyro-grouped.sh` runs two repetitions per build, per call (gate/up + SwiGLU/quantize +
down):

| Build | T=1 | T=2 | T=4 |
|---|---:|---:|---:|
| mmq-lut (serving) | 352.6 µs | 684.8 µs | 1,348 µs |
| rpw4 (gate/up 4 rows/warp only) | 329.1 µs | 628.7 µs | 1,233 µs |
| **sub8 (rpw4 + sub-warp down)** | **247.2 µs** | **464.6 µs** | **902.6 µs** |

`native_expert_parity` passes on Gyro-M layers 10 and 30 for every build, with identical error statistics.
Decode-once vs per-entry kernels are bitwise equal, and width invariance (1 vs 3 tokens) shows 0 differing
rows.

A rejected attempt is recorded here as well. Staging the gate/up activations in shared memory per block (patch
`strata-orin-tq2t-stage-lut-x.patch`, build `build-sm87-xstage`) was 6.5% *slower* (352.6 → 375.7 µs at T=1):
two block barriers per pass and the copy outweighed the L1-hit reads it replaced.

## Engine results

All 32 requests completed. Exits are 0 and systemd reports success. **Every b1, b2 and a2 output is
token-identical to a1.**

| Input | a1 decode | b1 | b2 | a2 | b vs a | Total s a → b |
|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 22.94 | 25.73 | 25.80 | 22.95 | **+12.1 to +12.5%** | 27.20 → 24.77 |
| 25,000 | 20.71 | 23.13 | 23.20 | 20.71 | **+11.7 to +12.0%** | 69.80 → 67.14 |

- Verify per window fell from 93.5 to 82.2 ms (2K) and from 94.1 to 82.8 ms (25K). Draft time is unchanged.
- Prefill (419-420 / 555 tok/s) and acceptance are unchanged.
- 69 NVMe polled completions all fell during engine loads, none inside a request.

## Decision

Serve with `build-sm87-sub8/strata.sub8` (`config/strata-gyro-orin-server.json`). 2K explanation decode is now
25.8 tok/s with identical output (23.0 before this round, 21.3 before the first expert round).

Remote read-only directory: `${TEST_HOME}/.local/share/jetson-orin/strata-gyro-sub8-ab-20261008-v1`
(`files.sha256`: 39 files, manifest `6fb3b7c8…`). The ignored local `run/` verifies against it.
