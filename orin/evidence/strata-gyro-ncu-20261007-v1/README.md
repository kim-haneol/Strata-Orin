# Nsight Compute: Gyro-M TQ2_T expert kernels on Orin

Run on 2026-10-07 with Nsight Compute 2026.1.1, as root because GPU performance counters require it on this
board. The target was the routed-expert kernels `native_gu_multi_kernel<144>` (gate/up) and
`native_down_multi_kernel<144>`. In the Nsight Systems profile
(`evidence/strata-gyro-profile-20261007-v1`) they take 38% of decode GPU time.

## How it was captured, and what failed

- **In-engine capture failed with OOM.** The engine ran under `ncu --graph-profiling node` with the full
  30 GiB expert cache (`config/strata-gyro-ncu-*-v1.json`, series unit
  `orin-strata-gyro-ncu-series-20261007-v1`). Kernel replay pushed the engine to 61.5 GB anonymous RSS. NvMap
  allocations failed (`error 12`), and the kernel OOM killer stopped it during load
  (`in-engine-attempt-oom.txt`). None of the four series runs produced a report. An earlier single-run
  attempt failed first because `--kernel-id` accepts only a numeric invocation, not a regex. Even 3 metrics
  need 8 replay passes on Orin, so replay cannot be avoided.
- **Standalone capture.** `build-sm87/native_expert_parity <Gyro-M GGUF> <layer>` runs the same kernels on
  real Gyro-M expert weights (layers 10 and 30; the APR/Hadamard pack type 144). Its launch is
  `(160, 1, 1) x 256` for gate/up and `(320, 1, 1) x 256` for down: one block row striding over groups. In
  the engine, grid Y is 10 × T groups. The inner per-row code (`row_dot_multi`) is the same, and the
  gate/up occupancy here (65%) is already at the register limit. Absolute times are per test call, not per
  engine window.
- **Clocks.** By default ncu locks clocks to base, which is 306 MHz on this board. The first report
  (`native-expert-layer10.ncu-rep`) and the `--clock-control none` reports both ran at about 306 MHz,
  because the devfreq governor did not ramp up for a short test. The final reports (`*-maxclk`) raised the
  GPU devfreq `min_freq` to `max_freq` (1,300.5 MHz) for the run and restored it to 306 MHz afterwards
  (`gpu-devfreq-before.txt`). Only the `*-maxclk` reports are used below.

## Results at 1.29 GHz (layers 10 and 30 agree within 1%)

| Kernel | Duration | SM throughput | Memory throughput | Max bandwidth | L2 hit | Achieved occupancy | Warp cycles / issued instr. |
|---|---:|---:|---:|---:|---:|---:|---:|
| gate/up (`native_gu_multi`, 64 regs) | 52.4 µs | 32.1% | 46.6% | 24.1% | 25.5% | 65.1% | 21.4 |
| down (`native_down_multi`) | 54.1 µs | 49.4% | 49.7% | 32.8% | 77.9% | 96.3% | 13.2 |

- The gate/up call reads 1.08 MB through L2, 0.88 MB of it missing to DRAM, in 52.4 µs: about **17 GB/s**
  against a DRAM peak of about 204 GB/s. Down reads 0.54 MB through L2 (0.44 MB missing) in 54.1 µs.
- For gate/up, **63% of the stall cycles are long-scoreboard waits on L1TEX (global) loads**: 21.4 cycles
  between issues, 13.5 of them on loads. Neither compute nor memory throughput approaches its limit. The
  kernel is latency-bound: each warp's load → trellis decode → dot sequence has too few loads in flight to
  hide DRAM latency. Occupancy is capped at 4 blocks per SM by 64 registers per thread.
- Down is closer to balanced (about 50% compute and memory), with high occupancy and an L2-resident
  working set.

## Implications (inference)

- In decode, a window's routed experts run with 10 × T groups in parallel. The Nsight Systems timing (about
  24 µs per gate/up expert, about 36 GB/s) is better than the single-group test, but still a small fraction
  of DRAM bandwidth.
- Candidate kernel changes for gate/up: more memory-level parallelism per warp (issue the next row-chunk loads
  before decoding the current one, or `cp.async` double-buffering through shared memory), and fewer
  registers or a launch bound for higher occupancy. Both must keep the TQ2_T decode and accumulation order
  (`native_expert_parity` checks bitwise equality). If expert kernel time halved, decode would gain about
  20% (experts are 38% of decode GPU time). That is an upper-bound estimate, not a measurement.
- Dense Q8_0 MMVQ (33% of decode) has no standalone test here and was not profiled with ncu.

## Artifacts

The ignored local `run/` holds the `.ncu-rep` reports and logs. `*-maxclk-details.txt` are text exports of
every section for the two final reports. The remote copy is
`${TEST_HOME}/.local/share/jetson-orin/strata-gyro-ncu-20261007-v1`. The failed in-engine directories are
`strata-gyro-ncu-{decode-early,decode-late,prefill-mmq,prefill-other}-20261007-v1` and
`strata-gyro-ncu-decode-20261007-v1` (ownership restored to `jetson`).
