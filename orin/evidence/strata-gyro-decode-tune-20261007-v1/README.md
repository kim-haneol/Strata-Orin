# Gyro-M decode tuning screen on Orin

Screening run on 2026-10-07 of `config/strata-gyro-decode-tune-v1.json` (launch commit `f44dfaf`; systemd
user unit `orin-strata-gyro-decode-tune-20261007-v1`). Every mode starts from the adopted serving setup
(`--no-prefill-borrow --prefill 8192`, MTP, INT8 KV, 32K context, full 24,576-slot cache) and changes one
thing. Each mode is a separate engine load with one warmup and three measured requests: the same 2,048-token
Korean explanation prompt and exactly 512 greedy output tokens. NVMe APST was off and the 2 s I/O timeout was
active.

## Results

All 44 requests completed. Wrapper exit is 0, all 11 native engines exited 0, systemd reports
`Result=success`, `ExecMainStatus=0`, `MainPID=0`, and no native process remains. **All outputs are
token-identical to `base` in every mode.** Repetitions within each load are identical (stream decode
standard deviation ≤ 0.003 tok/s).

| Mode | Change | Stream decode tok/s | vs base | Acceptance (acc/offered) | Tokens/window | ms/window (verify + draft) |
|---|---|---:|---:|---:|---:|---:|
| base | MTP cap 4, confidence 0.5 | 21.309 | 1.000 | 75.8% (298/393) | 2.38 | 111.8 (101.6 + 10.2) |
| p03 | confidence 0.3 | 20.450 | 0.960 | 63.9% (310/485) | | |
| p07 | confidence 0.7 | 20.847 | 0.978 | 82.7% (272/329) | 2.12 | 102.0 (92.9 + 9.0) |
| cap3 | MTP cap 3 | 20.712 | 0.972 | 79.5% (271/341) | 2.12 | 102.2 (94.1 + 8.1) |
| cap6 | verifier 6, cap 6 | 20.436 | 0.959 | 67.5% (316/468) | 2.60 | 127.3 (114.2 + 13.0) |
| cap6p07 | verifier 6, cap 6, conf. 0.7 | 20.719 | 0.972 | 78.4% (286/365) | 2.26 | 109.0 (98.3 + 10.6) |
| cap8 | verifier 8, cap 8 | 19.859 | 0.932 | 62.1% (321/517) | 2.67 | 134.4 (119.8 + 14.5) |
| sh0 | `STRATA_SH_STREAM=0` | 21.175 | 0.994 | same as base | | |
| fuse | `STRATA_FUSE_HEAD_GR=1` | 21.318 | 1.000 | 75.8% (298/397) | 2.38 | 111.8 (101.6 + 10.1) |
| max4 | `STRATA_GR_DOWN_MAX4=1` | 21.360 | 1.002 | same as base | 2.38 | 111.6 (101.3 + 10.2) |
| stage | `STRATA_IQ_STAGE_GRID_MMVQ=1` | 21.281 | 0.999 | same as base | | |

Empty cells: the decode-timing line was not tabulated for that mode. It is in `run/<mode>/engine.log`.

## Interpretation

- **Measured:** none of these knobs beats the current default (cap 4, confidence 0.5, shared-expert stream
  on) on this prompt by more than 0.2%. That is below the 0.3-0.4% load-to-load decode variation seen in
  earlier runs, so `max4` and `fuse` are not distinguishable from base. Turning the shared-expert stream off
  costs 0.6%. Every other MTP policy is slower.
- Larger windows fit the earlier cost model. Each added position costs about 18 ms of verify plus draft
  time, while the current cost per output token is about 47 ms. Positions offered with lower confidence pay
  verify cost without enough acceptance: at cap 8, tokens per window rise only from 2.38 to 2.67, while the
  window time rises from 112 to 134 ms. Stricter gating (0.7, cap 3) wastes less, but gives up accepted tokens.
- **Inference:** config-level decode tuning is exhausted on this workload. The verify forward pass itself
  dominates: about 68 ms at T=1 and about 101 ms at T≈2.8. Further decode gains would need kernel-level work,
  identified by profiling the normal graph path, or a better draft (higher acceptance at the same cost).
- Limitations: one explanation prompt. Acceptance, and with it the best MTP policy, depend on content: code
  showed 89% acceptance in the earlier short smoke. Re-check the policy on chat, code and tool/JSON prompts
  before treating it as general. Single mode order; one engine load per mode.

## Engine starts and NVMe

Engine starts took 72-96 s. The kernel logged 83 NVMe `timeout, completion polled` events (about 7.5 per
load, each now a 2 s wait). All of them fell during engine loads; none fell inside a request window
(`kernel-nvme-during-run.log`).

## Artifacts

Remote read-only directory: `${TEST_HOME}/.local/share/jetson-orin/strata-gyro-decode-tune-20261007-v1`.
`files.sha256` covers 51 files (manifest `effd739e…`). The ignored local `run/` copy verifies 50 of them;
only the raw `telemetry.ndjson` (`8a8e198e…`) stays remote. The ASTOP watch event was recorded, then
acknowledged.
