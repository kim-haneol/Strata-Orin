# gr_up_multi row prefetch: A-B-B-A decode benchmark (rejected)

Run on 2026-10-08 with `config/strata-gyro-grup-ab-v1.json`:

- **a** is the serving `build-sm87-sub8`.
- **b** is `build-sm87-grup`, which adds `patches/strata-orin-sub8-grup.patch`: in the hyper-connection up kernel
  (`gr_up_multi_kernel`, 9.8% of decode GPU time at about 72 GB/s), each warp's eight `w_up` rows are loaded four
  rows ahead.

`gr_parity --selftest` passes on both builds, including the "fused GR multi max-T=8" check.

| Input | a1 decode | b1 | b2 | a2 | b vs a |
|---:|---:|---:|---:|---:|---:|
| 2,048 | 25.72 | 25.58 | 25.55 | 25.71 | −0.6% |
| 25,000 | 23.13 | 23.01 | 22.97 | 23.13 | −0.6% |

All outputs are token-identical. Verify per window rose from 83.0 to 83.4 ms.

**Rejected.** Serial row loads were not this kernel's limit. The next step is a Nsight Compute look at
`gr_up_multi` (it runs standalone in `gr_parity`) instead of another guess.

Remote frozen directory: `${TEST_HOME}/.local/share/jetson-orin/strata-gyro-grup-ab-20261008-v1` (manifest
`c4597d37…`).
