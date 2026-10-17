# Per-turn prompt cost on Hermes agent traffic: breakdown and two fixes (2026-10-08)

Follow-up to `evidence/hermes-replay-ab-20261008-v1`. There, every agent turn paid about 1.5-1.9 s before decoding,
even for reads of ~100 tokens on top of a cached prefix. Prompt work was a quarter of the replay's wall time.

All runs replay the same 61 recorded Hermes requests (`work/hermes-capture-20261008/requests`, greedy, in order)
with `scripts/replay-strata-requests.py`. The server config is the serving one (`config/strata-gyro-replay-*.json`
differ only in `exe` and log path). **All 61 outputs are identical across every build below** (`compare.txt`), and
the prompt cache reused the same 1,055,326 tokens in each.

## Instrumentation

- `patches/strata-orin-sub8-ptime.patch` (serving `sub8` source + `STRATA_PROMPT_TIMING=1`): one line per request
  with the host time between the steps of the prompt path. The steps are match, park, conv_restore,
  ckpt_restore/session_zero, lend, read_batched(n), ckpt_save, read_windows(n) and refill. Inside batched reads it
  also reports the draft-layer and mid-prompt checkpoint time. No device syncs are added. With it off, the replay
  matched the serving binary (26.22 vs 26.16 tok/s decode, 266.0 vs 271.9 s prompt).
- `STRATA_PREFILL_TIMING=1` (existing): the GPU time of each batched read, by phase.
- Nsight Systems: requests 1-3 replayed untraced, then requests 4-5 (99- and 153-token reads on a 6.2K cached
  prefix) traced (`run-replay-nsys.sh`, kernels in `nsys-kernels-requests-4-5.json`).

## Breakdown (serving source, `prompt-timing-p.txt`, 61 requests)

| Step | Total | Per turn |
|---|---:|---:|
| Batched prompt read | 228.6 s | short reads: 1,160 ms + 2.34 ms per token (fit over reads under 2,048) |
| Parking the outgoing conversation | 23.0 s | 0.38 s on average (0.15-0.46 s) |
| Assistant header through the verify windows (4 tokens) | 6.4 s | 0.10 s |
| Checkpoint save at the turn boundary | 6.1 s | 0.10 s |
| Checkpoint restore | 1.4 s | 0.02 s |

- The batched read time is GPU work. Wall time and GPU timeline agree within 5 ms. Host waits are 10 ms per read.
- GPU time of a 99-token read, by phase (STRATA_PREFILL_TIMING): gdn projections 271 ms, expert gather
  ("dequant" phase) 198 ms, gate/up 259 ms, down 157 ms, qsa projections 105 ms, gdn out projection 98 ms.
- Nsight (two reads together, 2.6 s of kernels):

| Kernel | Time | Launches |
|---|---:|---:|
| `dequant_kernel` (dense Q8_0 → FP16 before cuBLAS) | 937 ms | 602 |
| `mul_mat_q` (expert MMQ) | 927 ms | 2,846 |
| `copy16_kernel` (expert blob gather into MMQ group buffers) | 402 ms | 21,936 |
| cuBLAS FP16/BF16 GEMMs | ~110 ms | ~650 |

## Fix 1: coalesced Q8_0 → FP16 dequant (`strata-orin-sub8-dqpairs.patch`)

The generic dequant kernel gives each thread one 32-value group. It writes 32 scalar halves, 64 bytes apart
between neighbouring threads. The new Q8_0 kernel gives each thread a pair of values, so a warp stores 128
contiguous bytes. Each element is still computed as `(float) q * d` with round-to-nearest, bit for bit.

- `dequant_bf16_test` was extended to check the FP16 output bit for bit against the device FP32 value rounded,
  for slices starting at row 0 and at row 1. Both paths pass on the model's Q8_0 tensor (`run-dq.log`).
  `STRATA_DQ_PAIRS=0` selects the old kernel.
- Nsight: 937 → 221 ms for the same 602 launches.
- Replay: short-read fixed cost 1,160 → 812 ms. Prompt total 266.0 → 247.8 s.

## Fix 2: no park when a turn continues from its own last checkpoint (`strata-orin-sub8-dqpark.patch`)

Hermes re-renders the previous reply, so each turn resumes from the checkpoint at the last turn boundary rather
than the live session. The engine then parked the outgoing conversation (0.15-0.46 s, growing with context). The
next park dropped that copy as superseded (#342). The only thing that copy held beyond the kept checkpoint was the
reply as generated.

The change: when a request resumes from the **deepest** checkpoint of the live chain (no incoming parked
conversation, no batch slot), it is not parked. Superseded copies are still dropped. Any other resume point still
parks, for example an auxiliary request that only shares the system-prompt root. `STRATA_PARK_CONTINUE=1`
restores the old behaviour.

- Replay: 53 turns not parked, 7 parked, 3 restores from the conversation cache (the same restores as before).
  Park time 25.8 → 2.1 s. Prompt total 247.8 → 222.9 s.
- What is given up: a client that goes back to the exact reply as generated must re-read that reply's tail from
  the checkpoint. A re-rendering client never does this.

## Result (`compare.txt`)

| Build | Decode tok/s | Prompt s | Wall s |
|---|---:|---:|---:|
| sub8 (serving until now) | 26.16 | 271.9 | 1110.0 |
| sub8 + timing patch | 26.22 | 266.0 | 1101.6 |
| + coalesced dequant | 26.18 | 247.8 | 1085.2 |
| + no park on continue (**dqpark**) | 26.31 | 222.9 | 1056.3 |

Each build ran once. Wall time for the same agent work is −4.1% against the timing build, from prompt work
alone. Decode differences are within run-to-run noise.

**Adopted:** `config/strata-gyro-orin-server.json` now runs `build-sm87-dqpark/strata.dqpark`
(sha256 `53c28eed…`). The previous binary is `build-sm87-sub8/strata.sub8` (`2690172711…`). The previous remote
config is saved as `server.json.sub8-20261008`.

## What is left in a short read (dqpark, Nsight, two reads)

- `mul_mat_q` 933 ms, about 467 ms per read. There are ~228 experts per layer with ~4 rows each, and the
  experts' ~14 GB are read at about 31 GB/s. The decode kernels read the same TQ2_T format at 53-58 GB/s.
- `copy16_kernel` 403 ms, about 200 ms per read. This copies each touched expert from its cache slot into an MMQ
  group buffer. Within one layer, slots have a uniform stride. So MMQ could read the slots in place if rows were
  ordered by slot and each slot were followed by zeroed padding (MMQ reads past a matrix's last row). Not done.
- `dequant_q8_0_f16_pairs` 221 ms, about 110 ms per read. Keeping FP16 copies of the dense weights would remove
  it, at about 4.3 GB of memory. Not done.
- Checkpoint save, about 100 ms per turn: 118 MB goes into freshly allocated pageable vectors. Reusing buffers
  would remove most of this. Not done.

## Reproduce

`run-replay-ptime.sh`, `run-dq.sh`, `run-dqpark.sh` (build, then `run-replay-env.sh TAG MODE ENV FIRST LAST`) and
`run-replay-nsys.sh TAG PRE LAST MODE`. Builds use `scripts/run-strata-gyro-exp-build.sh` with the patch and
`agention-llama-tq-mmq-shared-lut.patch` (`EXTRA_TARGETS=dequant_bf16_test` for the test).
`scripts/summarize-strata-prompt-timing.py prompt-timing-*.txt` sums the timing lines.
