# One-launch group gather for short reads with resident experts (2026-10-08)

A limited follow-up to `evidence/strata-gyro-mmq-direct-20261008-v1`, proposed in review. Keep MMQ, its expert
order, 16-expert groups, channels, tiles and the zero tail. Change only how a short read copies its routed experts
into the group buffers: one `gather_native_group` launch per group instead of one `gather_native` per expert.

## Change (`patches/strata-orin-sub8-grpres.patch`, source commit `de173bd` + test `5d1beaa`)

`group_gather` was gated on `stream_all`, so reads under 1,024 tokens (the walk without streaming) gathered expert
by expert. It now also applies when the walk does not stream **and every routed expert of the layer is resident**.
That is checked explicitly per layer; ring slots, copy waits and slot events are then not involved. The streamed
walk's conditions are unchanged. `STRATA_PREFILL_GROUP_RESIDENT=0` restores the per-expert gathers.

## Checks

1. **Bytes** (`mmq_gather_group_test.cpp`, new; `run-grpres.log`): random Gyro-M-sized blobs (1,305,600 B) at
   scattered 256-byte aligned addresses. The group call matched the per-expert gathers byte for byte, over the
   whole group buffers including bytes it must not touch, for a full group (16), a partial last group (5), one
   expert, and a flush starting inside a group ([6, 13)).
   - Timing of 16 experts: 295.3 µs per expert (142 GB/s moved) vs 233.7 µs grouped (179 GB/s). The bytes are the
     same, so the gain is launch and scheduling overhead, about 20% of the copy.
2. **Requests 1-12, crossed** (on, off, on, off; `compare-1-12.txt`, `prompt-timing-1-12-run*.txt`): outputs
   identical. Ten of the eleven non-streamed batched reads were 42-80 ms faster in both pairs:
   - ~100 tokens: −43 ms; 322 tokens: −62 to −69; 477 tokens: −77.
   - One 101-token read was a 989 ms outlier in the second "on" run.
   - Their sum fell 3.6% (15.58 → 15.02 s per run).
3. **All 61** (`compare-61.txt`): outputs identical to enko.

| Run (same config apart from the binary) | Prompt s | Short-read fit | ~100-token reads | Decode tok/s | Wall s |
|---|---:|---:|---:|---:|---:|
| enko / pol-base-2 / dir0 (per-expert gathers) | 220.0 / 219.1 / 219.4 | 802 ms + 2.36 ms/tok | 772-842 ms | 27.03-27.09 | 1029.8-1031.0 |
| grpres, on | 217.9 | 736 ms + 2.39 ms/tok | 741-800 ms | 26.96 | 1033.2 |

## Reading

- **Measured:** about 40-80 ms less per short or mid read, bit-identical outputs, about 1.5 s less prompt time
  over the 61 requests.
- At the wall-time level this is within the replay's decode noise. Decode varies about ±0.5% between runs of
  identical code, which is ±5 s here, and this run's decode was at the low end. It is a prompt-latency improvement
  of ~0.15-0.25% of the workload, not a decode change.
- **Adopted** as a free, bit-identical improvement: `config/strata-gyro-orin-server.json` now runs
  `build-sm87-grpres/strata.grpres` (sha256 `084de42f…`). The previous Jetson config is
  `server.json.recycle-20261008`. This binary also carries the native short-chunk path and the direct-MMQ path,
  both off by default (`STRATA_PF_NATIVE_MAX`, `STRATA_PF_MMQ_DIRECT`).
