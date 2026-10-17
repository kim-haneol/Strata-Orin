# Kernel round 3 and MTP re-screen on Orin (2026-10-08)

Follow-up to `evidence/strata-gyro-sub8-ab-20261008-v1`. Serving was `build-sm87-sub8` at 25.8 tok/s (2K
explanation decode). Patches live in `patches/strata-orin-*.patch`; the port branches are `orin-tq-prefetch`,
`exp-lb6`, `exp-stage4` and `exp-combo` in `work/gyro-strata-port/strata`. Every candidate passed its parity
check (`native_expert_parity` bitwise decode-once/per-entry; `gr_parity --selftest` including the T = 8
multi read), and every engine output below is token-identical to its baseline.

## Hyper-connection read (`gr_up_multi` / `gr_down_staged`)

- `gr_parity` gained a decode-shaped timing mode (`STRATA_BENCH_GR=T:iters`: real 2560/4/320 geometry, four
  rotating 13 MB weight copies). Script: `scripts/bench-strata-gyro-gr.sh`. At a locked 1.30 GHz, down + up per
  call took 109 / 129 / 151 / 182 µs for T = 1-4 (121 → 73 GB/s). The read costs about 24 µs per extra
  window token.
- Nsight Compute, T = 3 vs T = 1: `gr_up_multi` goes from 53.5 to 78.8 µs, with short-scoreboard stalls at
  37-39%. Half its shared-memory wavefronts are bank conflicts (308K of 617K), because each lane reads its
  8-float `lo` slice 32 bytes apart.
- **Row prefetch** (`strata-orin-sub8-grup.patch`) was tried before this diagnosis: engine A-B-B-A gave −0.6%
  decode, so it was rejected (`evidence/strata-gyro-grup-ab-20261008-v1`).
- **Conflict-free `lo` layout** (`strata-orin-sub8-grlo.patch`: two float4 per lane chunk, same fmaf order) made
  the read 3.5-5% faster per call (T = 3: 149.3 → 144.1 µs; T = 4: 182 → 173.5 µs).

## Routed experts (decode-shaped grouped bench, T = 1 / 2 / 4)

Nsight Compute on `sub8`: the gate/up kernel (4 rows per warp) has 61% long-scoreboard stalls on its small weight
loads. 70% of its shared wavefronts are codebook bank conflicts, from random gathers into the 4 KB table. Down
(the sub-warp path) is L1-bound at 84%.

| Build | T=1 | T=2 | T=4 |
|---|---:|---:|---:|
| sub8 (serving) | 247.3 µs | 464.6 µs | 902.4 µs |
| lb6: `__launch_bounds__(256, 6)` on gate/up (+ grlo) | 235.5 | 446.1 | 866.2 |
| stage4: the warp's 4 rows staged with cp.async (+ grlo) | 242.8 | 460.6 | 893.9 |
| combo: lb6 + stage4 | 255.4 | 483.9 | 937.1 |

The combination is slower than either change alone. The staged rows take 35 KB of shared memory per block,
which works against the 6-blocks-per-SM bound.

## Engine A-B-B-A: lb6 (`config/strata-gyro-lb6-ab-v1.json`; a = sub8, b = lb6)

| Input | a1 | b1 | b2 | a2 |
|---:|---:|---:|---:|---:|
| 2,048 decode tok/s | 25.77 | 25.79 | 25.85 | 25.68 |
| 25,000 decode tok/s | 23.19 | 23.19 | 23.24 | 23.10 |

The difference (≤ 0.4%) is within the a1/a2 spread. The bench gains (grouped call −4.5%, HC read −4%) do not
show up in the engine, which runs these kernels next to other streams (shared expert, MTP). **Not adopted**;
serving stays on `strata.sub8`.

## MTP policy re-screen on lb6 (`config/strata-gyro-mtp-rescreen-lb6-v1.json`, 2K, outputs identical)

| Mode | decode tok/s | accepted/offered |
|---|---:|---:|
| **base (cap 4, confidence 0.5)** | **25.82** | 298/393 |
| cap 5 | 25.62 | 310/432 |
| confidence 0.4 | 25.15 | 303/448 |
| cap 6, confidence 0.6 | 25.07 | 304/421 |
| cap 6 | 25.04 | 316/468 |
| confidence 0.3 | 25.03 | 310/485 |
| cap 5, confidence 0.4 | 25.02 | 317/489 |
| cap 6, confidence 0.4 | 23.94 | 321/539 |

Cheaper expert kernels did not move the optimum: cap 4 / 0.5 is still best on this prompt.

## Reading

- Routed experts now run at 53-58 GB/s in the decode-shaped bench. The limit is the TQ2_T codebook gathers:
  random 2-byte lookups give about 3.4 shared wavefronts per warp gather. That bound comes from the format and
  the Orin SM.
- Dense Q8_0 projections and both heads already run at 120-150 GB/s, except the small shared-expert matrices
  (3.4% of decode). Hyper-connection gains of a few percent per call are hidden by stream overlap in the engine.
- Further decode gains on the 2K explanation prompt therefore need either structural changes (for example a
  different expert dispatch or fused kernels across stages) or higher acceptance (draft quality, or suffix
  drafting for repetitive agent output), not more single-kernel tuning.

Remote frozen directories: `strata-gyro-lb6-ab-20261008-v1` (manifest `8a6f8664…`) and
`strata-gyro-mtp-rescreen-lb6-20261008-v1` (manifest `9f0dad12…`); bench outputs are in the
`strata-gyro-bench-*` directories copied here.
