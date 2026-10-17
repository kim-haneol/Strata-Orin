# TQ2_T staged-codebook expert kernels: A-B-B-A decode benchmark on Orin

Run on 2026-10-07 with `config/strata-gyro-tq2t-stage-lut-ab-v1.json`, launch commit `9c581a5`. Engine starts
alternate a1 → b1 → b2 → a2:

- **a** is the pinned `build-sm87/strata` (`8ce7f7cb…`).
- **b** is `build-sm87-tq2t-prefetch/strata` (`1d62562c…`, also kept as `strata.tq2t-stage-lut`): the pinned
  revision plus `patches/strata-orin-tq2t-stage-lut.patch`.

The patch does three things:

1. Each warp stages its TQ2_T expert row in shared memory with `cp.async` before the unchanged lane loop.
2. The grouped gate/up and down kernels copy the 4 KB int8 trellis codebook (`tq_lut_i8`) into shared memory
   and decode from that copy.
3. It adds the decode-shaped timing mode in `native_expert_parity`.

The weights, codebook values, decode and summation order are unchanged.

All modes use the serving setup: owned 8,192-token prefill chunks, MTP cap 4 and confidence 0.5, INT8 KV, 32K
context and the full 24,576-slot cache. The 2,048- and 25,000-token prompts are the same as before, with
exactly 512 greedy output tokens, one warmup and three measurements per load.

## Why this change

- Nsight Compute on a decode-shaped standalone call (`STRATA_BENCH_GROUPED=1:30`: 10 experts × 1 token,
  rotating 64 weight copies, so the gate/up grid of 1,600 blocks matches the engine's T=1 launch) showed the
  gate/up kernel L1-bound. L1/TEX was at 70% of peak and L2 at 17%. Each warp-level global load touched about
  21 L1 sectors (972,800 requests, 20.6 M sectors), because every lane reads random entries of the 4 KB
  codebook. Stall samples were long scoreboard 22%, LG throttle 20% and MIO throttle 19%.
  Reports: remote `strata-gyro-bench-grouped-20261007-v1/ncu-T1-*.ncu-rep`.
- Decode-shaped timing at a locked 1.30 GHz, two repetitions each (`bench-lut.txt` remotely), per call
  (gate/up + SwiGLU/quantize + down):

| Build | T=1 (10 groups) | T=2 (20) | T=4 (40) |
|---|---:|---:|---:|
| pinned + bench mode | 440.2 µs | 857.6 µs | 1,691 µs |
| + row staging | 412.0 µs | 801.7 µs | 1,582 µs |
| + staged codebook (this build) | **353.0 µs** | **684.7 µs** | **1,348 µs** |

  The pinned T=1 call (440 µs) matches the engine's Nsight Systems gate/up + down time at T=1 (442 µs).
- `native_expert_parity` on Gyro-M layers 10 and 30: 0 failures, with the same error statistics as the pinned
  build. Width invariance (1 vs 3 tokens): 0 rows differ. The decode-once vs per-entry kernels are bitwise
  equal.

## Results

All 32 requests completed. Wrapper exit is 0, all four engines exited 0, systemd reports
`Result=success`/`ExecMainStatus=0`/`MainPID=0`, and no native process remains. **Every output of b1, b2 and
a2 is token-identical to a1 at both lengths.** The a1 and a2 decode rates are identical (21.28 / 19.24 tok/s),
so load-to-load drift is not visible in this run.

| Input | a1 | b1 | b2 | a2 | b vs a |
|---:|---:|---:|---:|---:|---:|
| 2,048 stream decode tok/s | 21.28 | 23.00 | 22.97 | 21.28 | **+7.9 to +8.1%** |
| 25,000 stream decode tok/s | 19.24 | 20.76 | 20.72 | 19.24 | **+7.7 to +7.9%** |
| 2,048 total s | 29.214 | 27.403 | 27.449 | 29.244 | 1.064-1.066x |
| 25,000 total s | 73.750 | 71.779 | 71.905 | 73.974 | 1.026-1.027x |
| 25,000 TTFT s | 47.191 | 47.161 | 47.245 | 47.419 | unchanged |

- Decode-timing lines: verify per window fell from 101.7 to 93.3 ms (2K, T 2.83) and from 102.3 to 93.9 ms
  (25K, T 2.82), about −8.3%. Draft time (10.2 / 12.9 ms) is unchanged because the MTP draft layer does not
  use these TQ2_T kernels. Tokens per window and acceptance (298/393, 281/420) are unchanged.
- Prefill is unchanged (530 tok/s at 25K); the prompt path uses the MMQ kernels.
- Engine starts took 62-81 s. Of 17 NVMe polled-completion timeouts, none fell inside a request window.

## Decision

Adopt the b binary for serving: same outputs, about 8% faster decode. Keep the immutable copy
`build-sm87-tq2t-prefetch/strata.tq2t-stage-lut` as the serving executable, because the experiment build
directory will be rebuilt for later patches. The pinned `build-sm87` stays as the reference.

## Artifacts

Remote read-only `${TEST_HOME}/.local/share/jetson-orin/strata-gyro-tq2t-stage-lut-ab-20261007-v1`
(`files.sha256`: 39 files, manifest `d8437737…`). The ignored local `run/` verifies against it, except the raw
`telemetry.ndjson`, which stays remote.
