# Shared-memory TQ codebook in the fork's MMQ: A-B-B-A prefill benchmark on Orin

Run on 2026-10-07 with `config/strata-gyro-mmq-lut-ab-v1.json` (launch commit `dcb2166`). Engine starts
alternate a1 → b1 → b2 → a2:

- **a** is the serving build from `evidence/strata-gyro-tq2t-stage-lut-ab-20261007-v1`:
  `build-sm87-tq2t-prefetch/strata`, `1d62562c…`.
- **b** is `build-sm87-mmq-lut/strata` (`e4566270…`, kept read-only as `strata.mmq-lut`). It is the same
  Strata patch plus `patches/agention-llama-tq-mmq-shared-lut.patch` on the pinned agention llama.cpp fork
  (`93a24e5`), built from separate worktrees (`src-exp/mmq-lut`, `src-exp/mmq-lut-ggml`).

The fork patch fills the existing shared codebook copy (`tq_lut_i8_s`, which the fork's mat-vec kernels
already use) at the start of `mul_mat_q` for trellis types. `tq_lane_to_q8`, the MMQ tile loader's TQ decode,
then reads it there instead of from `tq_lut_i8` in global memory. Same entries, so the same int8 tiles.

Everything else matches the earlier A-B-B-A: the serving setup, the 2,048- and 25,000-token prompts, 512 greedy
tokens, and one warmup plus three measurements per load.

## Results

All 32 requests completed. Wrapper exit is 0, all engines exited 0, systemd reports success, and no native
process remains. **Every b1, b2 and a2 output is token-identical to a1.**

| Input | a1 prefill tok/s | b1 | b2 | a2 | b vs a | TTFT a → b |
|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 397.61 | 420.95 | 419.88 | 398.48 | **+5.6 to +5.9%** | 5.21 → 4.93 s |
| 25,000 | 529.07 | 555.34 | 554.91 | 530.41 | **+4.9 to +5.0%** | 47.34 → 45.10 s |

- Decode is unchanged (22.93-22.97 / 20.68-20.73 tok/s), as expected: the decode kernels are the same.
- The 25K request total goes from 72.05 to 69.79 s (1.032x). The two a loads agree within 0.3%.
- 41 NVMe polled completions all fell during engine loads, none inside a request.
- The gain is smaller than the decode kernels' 20%. **Inference:** MMQ decodes each weight tile once and
  reuses it across many token columns, so codebook reads are a smaller share of `mul_mat_q` time.

## Decision

Serve with `build-sm87-mmq-lut/strata.mmq-lut` (`config/strata-gyro-orin-server.json`). It keeps the decode
kernel gain and adds about 5% prefill, with identical outputs.

Remote read-only directory: `${TEST_HOME}/.local/share/jetson-orin/strata-gyro-mmq-lut-ab-20261007-v1`
(`files.sha256`: 39 files, manifest `73841985…`). The ignored local `run/` verifies against it, except the raw
`telemetry.ndjson`.
