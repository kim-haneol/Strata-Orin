# Gyro-M Nsight Systems profile on Orin (serving setup)

Captured on 2026-10-07 with `config/strata-gyro-profile-v1.json` and `scripts/strata-gyro-profile.py`
(launch commit `83fe2b9`; systemd user unit `orin-strata-gyro-profile-20261007-v1b`). The engine ran the
adopted serving setup: owned 8,192-token prefill chunks, MTP cap 4 and confidence 0.5, INT8 KV, 32K context
and the full 24,576-slot cache. It started under `nsys launch` (Nsight Systems 2025.6.3) with CUDA graph nodes
traced individually. Loading and a 2K/32-token warmup were not captured. Two requests were captured with
`nsys start/stop`. They reuse the token IDs of the chunk sweep's prompts:

| Capture | Request | Traced result | Untraced reference |
|---|---|---|---|
| `decode2k` | 2,048 in, 256 out | TTFT 5.30 s, stream decode 17.26 tok/s | 5.2 s, 21.3 tok/s |
| `prefill25k` | 25,000 in, 8 out | prefill 524.5 tok/s (TTFT 47.80 s) | 528.9 tok/s |

Node-level graph tracing slows decode by about 19% and prefill by about 1%. Use these captures for kernel
identity and proportions, not for throughput. The first launch attempt failed because `nsys launch` rejects
`--cpuctxsw`. That attempt is kept remotely as `…-launch-attempt-1`, and the switches moved to `nsys start`.
Run exit, native exit and systemd state are 0/0/success, and no native process remains.

`scripts/analyze-strata-gyro-nsys.py` splits each capture at the recorded wall-clock first-token time. Kernel
timestamps are relative to the session start in UTC. It sums kernel time by name and by category
(`kernel-phases.json`). The `*-stats_*.csv` files are the stock `nsys stats` reports for whole captures.

## Where the GPU time goes

Kernel time by category (MTP draft kernels are counted with the target kernels of the same type, except the
draft head):

| Category | Decode (2K context, 14.78 s wall) | 25K prefill (47.80 s wall) |
|---|---:|---:|
| Routed experts (TQ2_T `native_gu/down_multi`; prefill `mul_mat_q`) | 4.996 s, 37.8% | 21.539 s, 45.4% |
| Dense projections (Q8_0 MMVQ; prefill dequant + cuBLAS) | 4.310 s, 32.6% | 9.341 s, 19.7% |
| Hyper-connection (`gr_*`) | 1.867 s, 14.1% | 3.569 s, 7.5% |
| MTP draft head (106,299-row MMVQ) | 0.566 s, 4.3% | |
| Attention (QSA, `prompt_attn_i8`) | 0.510 s, 3.9% | 5.275 s, 11.1% |
| GDN | 0.478 s, 3.6% | 4.770 s, 10.1% |
| Other / routing | 0.49 s, 3.7% | 2.95 s, 6.2% |

GPU busy (union of kernel intervals) is 87.1% of the decode wall time and 99.3% of the prefill wall time. The
decode gaps are partly tracing overhead.

### Decode observations (measured)

- About 117 verify windows for 256 tokens (2.19 tokens per window). The routed-expert kernels launch with grid
  Y = 10 × T (T = window tokens): T=1/2/3/4 take 240/387/530/676 µs (gate/up) and 202/327/452/578 µs (down)
  per layer. Cost rises by about 145 + 125 µs per extra window token per layer, about 13 ms per token over
  48 layers. That is most of the measured marginal verify cost of about 18 ms per window token.
- The kernels already group a window's entries by expert and decode each weight part once for up to 4 tokens
  (`native_gu_multi_kernel`, GRP_NC = 4). The linear growth therefore means consecutive tokens share few
  routed experts. That is an inference from the scaling, not a measured overlap.
- At T=1, ten experts' gate/up rows (about 0.87 MB each) take 240 µs, about 36 GB/s, and the down rows take
  about 22 GB/s. Orin's DRAM peak is about 204 GB/s. **Inference:** the TQ2_T expert kernels are limited by
  something other than DRAM bandwidth, such as trellis decode arithmetic, latency or occupancy. Nsight
  Compute metrics are needed to say which.
- Dense Q8_0 projections average about 37 ms per window. Assuming about 3.6 GB of dense, head and projection
  weights read per window, that is roughly 95 GB/s; this byte count is an estimate.
- The MTP draft head (106,299 rows) costs about 2.1 ms per call and 4.3% of decode.

### Prefill observations (measured)

- `mul_mat_q` (the expert MMQ path) is the largest single kernel at 39.9%. With MoE combine, SwiGLU and
  activation quantization, routed experts take 45%.
- Dense projections run as `dequant_kernel` (3.8%) followed by cuBLAS FP16/BF16 GEMMs (about 11.4%) and a
  split-K reduction (`Kernel2`, 3.9%). The weights are dequantized again for every chunk.
- 25K attention (`prompt_attn_i8_kernel`, 48 calls, about 84 ms each) is 8.5%, and the GDN recurrence is
  `gdn_rec_cols_pipe_kernel` at 5.8%.

## Next step

Profile the hot kernels with Nsight Compute: `native_gu_multi_kernel` and `native_down_multi_kernel` at
T=1 and T=4, `native_mmvq_multi_kernel`, `gr_up_multi_kernel` and prefill `mul_mat_q`. Check compute versus
memory throughput, achieved occupancy and stall reasons before choosing kernel work.

## Artifacts

The ignored local `run/` holds the reports and SQLite exports, and they also stay in the remote directory
`${TEST_HOME}/.local/share/jetson-orin/strata-gyro-profile-20261007-v1`. `large-artifacts.sha256` lists
their hashes. The raw ASTOP `telemetry.ndjson` stays remote.
