# Owned-buffer prefill chunk sweep on Orin

Second attempt of `config/strata-gyro-prefill-chunk-own-v1.json`, run on 2026-10-07. The first attempt
(`evidence/strata-gyro-prefill-chunk-own-20261007-v1`) was stopped because engine starts took 22 minutes.
That cause, NVMe missed completions, is documented in `evidence/jetson-nvme-timeouts-20261007`. Launch
commit `0004d91`. Systemd user unit `orin-strata-gyro-prefill-chunk-own-20261007-v2`.

Every mode uses `--no-prefill-borrow` with an explicit chunk (1,024 / 2,048 / 4,096 / 8,192). Everything else
matches the earlier owned-buffer A/B: the pinned SM87 binary, Gyro-M, MTP cap 4 and confidence 0.5, 32K
context, INT8 KV, the 24,576-slot cache, the same 2,048- and 25,000-token prompts, exactly 512 greedy output
tokens, and one warmup plus three measurements per length. Modes ran in order c1024 → c8192.

**System state changed during the run.** APST was disabled before launch. NVMe `queue/io_timeout` was
lowered from 30 s to 2 s at 11:32:25 UTC, during the c1024 25K warmup (11:31:38-11:33:33). Every measured
request ran afterwards, so all measurements share the same NVMe settings. All 27 NVMe timeouts in the run
fell inside engine loads and none inside a request window (`kernel-nvme-during-run.log`). The engine start
times do mix settings, page-cache state and thermal history.
The repeated prompts warm the PLE rows in the page cache. These requests are therefore not I/O-cold even
though prefix reuse is zero; PLE misses on new inputs are untested.

## Results

All 32 requests completed. Wrapper exit is 0, all four native engines exited 0, systemd reports
`Result=success`, `ExecMainStatus=0`, `MainPID=0`, and no native process remains. Repetitions are identical
within every group. GGUF-in-place reads are zero in every request. Every engine kept 24,576 expert slots.

Three-measurement means:

| Input | Chunk | Prefill tok/s | IPC TTFT s | Stream decode tok/s | IPC total s | VRAM free after load |
|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 1,024 | 287.21 | 7.196 | 20.34 | 32.323 | 20,174 MiB |
| 2,048 | 2,048 | 397.29 | 5.222 | 21.26 | 29.263 | 19,819 MiB |
| 2,048 | 4,096 | 399.31 | 5.196 | 21.31 | 29.178 | 18,965 MiB |
| 2,048 | 8,192 | 396.38 | 5.233 | 21.25 | 29.284 | 17,123 MiB |
| 25,000 | 1,024 | 281.45 | 88.916 | 19.94 | 114.548 | |
| 25,000 | 2,048 | 381.60 | 65.599 | 18.73 | 92.886 | |
| 25,000 | 4,096 | 475.14 | 52.699 | 19.47 | 78.944 | |
| 25,000 | **8,192** | **528.87** | **47.350** | 19.22 | **73.941** | |

At 25K, c8192 gives a 1.879x prefill/TTFT gain and 1.549x end-to-end over owned c1024. Against the original
borrowing c1024 baseline (120.0 s TTFT, 145.6 s total), it is 2.54x TTFT and 1.97x end-to-end. These are
observed times for the same input and the same output length; the outputs differ, so they are not yet
"equal-quality" speedups. The gain per
doubling shrinks: 1.36x, then 1.25x, then 1.11x. The 2K prompt fits into one chunk from c2048 upward, so
c2048-c8192 match within 1% there (TTFT 7.2 to 5.2 s).

**Decode.** No large change in engine decode cost with the chunk size was observed. The window-size
distributions, routed experts and generated histories differ, so this is not proof of identical cost. In the
`STRATA_DECODE_TIMING` lines, verify time per window at similar average window sizes stays close: 101.6-102.2 ms at T 2.83 (2K) and
99.2-102.5 ms at T 2.68-2.82 (25K). The decode tok/s differences come from different outputs, which change
how many tokens are accepted per window. At 25K, acceptance is 73.5% (c1024), 65.0% (c2048), 69.5%
(c4096) and 66.9% (c8192). One prompt's acceptance is not a property of the chunk.

**Outputs.** Greedy outputs differ from c1024. A changed accumulation and reduction order at chunk
boundaries and shapes is the likely explanation. That is not yet established: dispatch or tile choices can
change with the shape, small differences can flip MoE routing or QSA selection, and a boundary-handling bug
is not excluded. Teacher-forced logit comparison on a common continuation (drafting off) and lengths around
chunk boundaries (for example 8,191/8,192/8,193) would test this. The first difference is at token 113 for the 2K prompt (all larger chunks) and at token 10 (c2048,
c4096) or 33 (c8192) for the 25K prompt. The c1024 and c8192 25K answers are coherent and nearly identical in
content and structure. This is not a quality evaluation. Weight files, KV format and the kernel set are
unchanged and only the prefill batching differs. Numerical and quality equivalence is unverified.

Telemetry: prefill GR3D about 98.7% in every group (no samples below 50%). VDD_GPU_SOC during 25K prefill
rises from 30.0 W (c1024) to 33.7 W (c8192). Maximum available GPU temperature was 62.75 C. Memory pressure
stayed `normal`. Peak swap was 622-733 MiB. Two groups saw single low GPU-clock samples (306 and 816 MHz),
and all other samples were at 1,300.5 MHz.

## Engine start with the NVMe mitigation

| Mode | NVMe state during load | Start s | Missed-completion timeouts during load |
|---|---|---:|---:|
| c1024 | APST off, 30 s timeout | 352 | 7 |
| c2048 | APST off, 2 s timeout | 90 | 14 |
| c4096 | APST off, 2 s timeout | 79 | 6 |
| c8192 | APST off, 2 s timeout | 66 | 0 |

The earlier starts with APST on and the 30 s timeout took 1,349-1,460 s.

Each mode's three measurements share one engine load. A small difference between modes (such as the
decode differences) would need alternating engine starts (A-B-B-A) before being attributed.

## Decision and follow-ups

- Adopt `--no-prefill-borrow --prefill 8192` for serving as an experimental default: weight and KV formats
  are unchanged, and numerical and quality equivalence is still to be verified on varied inputs. The serving launcher uses 16K context, so its
  free VRAM will be higher than the 32K figure here.
- Untested: 16,384 or larger chunks (upstream keeps them opt-in). c8192 leaves about 17 GiB free at 32K
  context, so c16384 likely fits. Expect a smaller gain than the last doubling.
- Artifacts: remote read-only `${TEST_HOME}/.local/share/jetson-orin/strata-gyro-prefill-chunk-own-20261007-v2`.
  `files.sha256` covers 40 files (manifest SHA256 `36cca1de…221e`). The ignored local `run/` copy verifies
  39 of them, and only the raw `telemetry.ndjson` (`b9c440cd…c582`) stays remote. The ASTOP watch event
  (`disappeared`, unknown exit) was recorded and then acknowledged. Completion is established by the explicit
  exits and systemd state.
