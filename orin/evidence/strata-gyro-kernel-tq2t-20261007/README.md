# TQ2_T routed-expert kernel experiments on Orin

Kernel work on 2026-10-07, following `evidence/strata-gyro-ncu-20261007-v1`. The source is the port
repository's branch `orin-tq-prefetch` (`work/gyro-strata-port/strata`, not tracked here). Experiments are
built from the pinned revision `bb98d09` plus one patch in a separate worktree
(`scripts/run-strata-gyro-exp-build.sh`; remote `src-exp/tq2t-prefetch`, build `build-sm87-tq2t-prefetch`).
The pinned `build-sm87` (strata `8ce7f7cb…`) is unchanged. The experiment build directory kept its first name
after the second patch replaced the first.

## 1. Register prefetch of a row's lane steps (rejected)

Patch `a4f8bd9` (`patches/strata-orin-tq2t-prefetch.patch`), reverted in `37b2854`. It fetched each lane's
TQ2_T step words for all of a row's steps (5 for gate/up, 2 for down) into registers before decoding.

- `native_expert_parity` on Gyro-M layers 10 and 30 passed with the same error statistics as the pinned build
  (`prefetch-registers/parity-exp-layer*.log`).
- Registers rose from 64 to 128 (gate/up) and 159 (down). The switch over 1-5 steps and 1/2/4 tokens is
  inlined into each kernel. Warps active fell from 55% to 30% (gate/up) and from 38% to 15% (down). Kernel
  time at 1.29 GHz rose from 52.5 to 59.0 µs (gate/up) and from 54.2 to 82.6 µs (down). Long-scoreboard
  stall latency per warp still halved (13.4K to 6.5K), so serial load round trips were real.

## 2. Shared-memory row staging with cp.async (kept for A/B)

Patch `patches/strata-orin-tq2t-stage.patch` (port commit after the revert; strata `3e69b1e7…`). Each warp
copies its whole TQ2_T row into a per-warp shared buffer with 4-byte `cp.async`, then runs the unchanged
lane loop on the copy: the same bytes, decoded and summed in the same order. Rows over 32 blocks or blocks over
8 warps use the original loop.

- **Parity:** both layers pass with the same statistics, and width invariance (1 vs 3 tokens) reports 0
  differing rows (`stage-shared/parity-exp-layer*.log`).
- **Standalone ncu** (the parity harness's single-group, many-token call): no gain (gate/up 52.3 → 53.3 µs,
  down 53.8 → 55.2 µs). Registers stay at 64 (down drops to 48). Static shared memory is 26 KB per block,
  because one buffer is instantiated per token-count variant. **This harness is not representative of
  decode:** one call is one expert with many tokens, so rows are re-decoded from cache. The earlier "17 GB/s"
  figure in `evidence/strata-gyro-ncu-20261007-v1` comes from this harness and should not be read as the
  engine's bandwidth.
- **In-engine Nsight Systems** (`evidence/strata-gyro-profile-tq2t-stage-20261007-v1`, compared with
  `scripts/compare-strata-gyro-nsys-kernels.py`; `stage-shared/nsys-decode-compare.txt`). Same prompt and
  launch counts per window size, and the same drafts accepted (138/191):

| Kernel | T=1 | T=2 | T=3 | T=4 |
|---|---:|---:|---:|---:|
| gate/up µs (pinned → staged) | 239.8 → 225.2 | 387.2 → 366.4 | 530.0 → 502.4 | 675.6 → 642.5 |
| down µs (pinned → staged) | 201.9 → 185.7 | 327.3 → 307.1 | 452.4 → 419.6 | 578.4 → 536.4 |

  Routed-expert kernel time in decode fell 6.1% (4.965 → 4.663 s), and total GPU busy time 2.8%
  (12.87 → 12.51 s). Traced stream decode went from 17.26 to 18.78 tok/s, but most of that difference is
  tracing gaps (wall −1.2 s against busy −0.36 s), so it is not the expected untraced gain. **Expected
  untraced decode gain: about 2-3% (inference)**, still to be measured with the A-B-B-A protocol
  `config/strata-gyro-tq2t-prefetch-ab-v1.json`.

## Reading

Removing the per-step global round trips helps only a little in the engine: with 10 × T groups the kernels
already have many warps in flight. The remaining L1TEX waits are likely the 8 codebook lookups per lane step
(`tq_lut_i8`, 4 KB in global memory) and the activation reads. Both are candidates for further work, which
needs representative in-engine ncu metrics.
