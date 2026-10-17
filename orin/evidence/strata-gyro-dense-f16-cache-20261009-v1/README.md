# Dense FP16 weight cache for prompt reads (2026-10-09)

Follow-up to `evidence/strata-gyro-group-gather-20261008-v1` (serving `grpres`). Every batched prompt read
dequantized each dense Q8_0 projection to FP16 again before its cuBLAS product. That cost about 110 ms of a
~100-token read even after the coalesced kernel (`evidence/strata-gyro-prompt-fixed-cost-20261008-v1`).

## Change (`patches/strata-orin-sub8-f16cache.patch`, source commit `9acaf8e`)

`Gemm::native` (`src/prefill/gemm.cu`) keeps each weight's FP16 copy after the first dequantization, keyed by the
weight's address (with its shape and type checked), and later calls only run the product.

- The values are those the scratch would hold, and the products are the same calls: the same row slices when a
  weight exceeds the scratch size. So results are bit for bit those of the scratch path.
- Guards: total at most `STRATA_DENSE_F16_CACHE_MIB` (default 6144). A copy is made only if
  `STRATA_DENSE_F16_CACHE_FLOOR_MIB` (default 8192) stays free after it. Once one copy does not fit, no more are
  tried and the rest use the scratch path. `STRATA_DENSE_F16_CACHE_MIB=0` turns the cache off.

## Memory (`cache-memory-log.txt`)

- 16,306 MiB free after loading.
- The first prompt read filled the cache with about 300 weights: **5,597 MiB at 300 weights**, 10,567 MiB left
  free. That is more than the 4.3 GB first estimated. The cap was never reached (no "stops" line).
- Later the conversation cache parked up to ~3.5 GB in host RAM. Orin's GPU and CPU share the same 64 GB.

## Results

Outputs were identical in every run: the four crossed 1-12 replays (`compare-1-12.txt`), and the 61-request
replay against `grpres` and `enko` (`compare-61.txt`).

Requests 1-12, crossed (on, off, on, off; `prompt-timing-run1..4.txt`):

| Read (tokens) | on − off, mean of the two pairs |
|---:|---:|
| 4,346 (first read of a fresh server: the copies are made here) | +120 ms |
| 1,713 | −93 ms |
| 99-477 (10 reads) | −88 to −101 ms each |

All 61 (`prompt-timing-run5.txt`):

| Build | Prompt s | Short-read fit | ~100-token reads | Decode tok/s | Wall s |
|---|---:|---:|---:|---:|---:|
| grpres | 217.9 | 736 ms + 2.39 ms/tok | 741-800 ms | 26.96 | 1033.2 |
| **f16cache** | **211.1** | **634 ms + 2.38 ms/tok** | **633-699 ms** | 27.02 | **1026.9** |

- **Measured:** about −100 ms per batched read after the first, −6.8 s (−3.1%) prompt time over the 61 requests,
  and bit-identical outputs. Decode is unchanged; the difference is within run-to-run noise.
- **Cost:** about 5.6 GB of memory held for the server's lifetime, and +120 ms on the first read after a start.

**Adopted:** `config/strata-gyro-orin-server.json` now runs `build-sm87-f16cache/strata.f16cache` (sha256
`996f228f…`). The previous Jetson config is `server.json.grpres-20261009`.

## The day's chain on the same 61 requests (all outputs identical)

| Serving build | Decode tok/s | Prompt s | Wall s |
|---|---:|---:|---:|
| sub8 (2026-10-08 morning) | 26.16 | 271.9 | 1110.0 |
| dqpark | 26.31 | 222.9 | 1056.3 |
| recycle + enko | 27.09 | 220.0 | 1030.6 |
| grpres | 26.96 | 217.9 | 1033.2 |
| **f16cache** | **27.02** | **211.1** | **1026.9** |
