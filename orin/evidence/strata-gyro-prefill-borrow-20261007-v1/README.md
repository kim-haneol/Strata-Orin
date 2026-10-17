# Gyro-M prefill: borrowed cache slots versus owned buffers on Orin

Prepared on 2026-10-07 to test a hypothesis from the v1 telemetry
(`evidence/strata-gyro-mtp-20261007-v1`). Protocol:
`config/strata-gyro-prefill-borrow-v1.json`. Harness: `scripts/strata-gyro-benchmark.py`,
`scripts/run-strata-gyro-benchmark.sh`, `scripts/summarize-strata-gyro-benchmark.py` and
`scripts/freeze-strata-gyro-benchmark.py`, generalized at `9ff2b7f` for mode-specific engine arguments
(the generalized summarizer reproduces the frozen v1 summary exactly).

## Hypothesis

The v1 25K prefill was not GPU-bound. Re-reading the frozen v1 tegrastats against request timestamps showed:

- 20-29% of 1 s samples during 25K prefill below 50% GR3D, versus none during decode;
- about one GPU-idle dip per 1,024-token chunk (17-25 detected for 25 chunks at 1 s sampling), with one CPU
  core near 100%;
- the engine's cumulative `files ... MB read (the GGUF in place)` counter rising by 17,693.5 MB per 25K
  request and 1,835.7 MB per 2K request, identically with MTP on and off (about 690 MB per chunk, close to
  the 528 borrowed slots, 0.64 GiB).

The Strata source states that each lent slot's expert "is streamed during the prompt and refilled after it"
(`src/program/generate.cpp`, around line 4438 at `bb98d09`). With `--mmap-experts` that streaming is a host
copy from the mapped GGUF. `--no-prefill-borrow` gives the prompt path its own buffers instead. With about
20.7 GB of VRAM free after loading, the full 24,576-expert cache should still fit.

## Conditions

Both modes match v1 MTP-on except for the declared mode argument: same pinned SM87 binary
(`8ce7f7cb…5354`), Gyro-M pack, MTP cap 4, confidence 0.5, verifier capacity 4, 32,768 context, INT8 KV,
**explicit 1,024-token chunk**, 24,576-slot expert cache and zero prompt, conversation and suffix reuse.
`own` adds `--no-prefill-borrow`. A bare `--prefill auto` would switch to 2,048 with that flag, so it is
not used. The inputs are the same 2,048- and 25,000-token Korean explanation prompts (same ID hashes as v1).
Each mode has one warmup and three measured requests per length. Exactly 512 greedy tokens are generated.
The mode order is borrow, then own. The run used MAXN with the existing dynamic governors (no
`jetson_clocks`).

## Results

All 16 requests completed. Wrapper exit is 0, both native engines exited 0, systemd reports
`Result=success`, `ExecMainStatus=0`, `MainPID=0`, and no native process remains. **Every measured output is
token-identical between modes at both lengths.** The three repetitions within each group are identical.
Acceptance is unchanged (290/411 and 283/385 per request).

Three-measurement arithmetic means:

| Input | Mode | Internal prefill tok/s | Internal prompt s | IPC TTFT s | IPC decode tok/s | IPC total s |
|---:|:---:|---:|---:|---:|---:|---:|
| 2,048 | borrow | 235.19 | 8.704 | 8.774 | 20.43 | 33.788 |
| 2,048 | own | **286.57** | 7.143 | **7.212** | 20.35 | 32.324 |
| 25,000 | borrow | 208.44 | 119.934 | 120.033 | 20.01 | 145.576 |
| 25,000 | own | **280.81** | 89.023 | **89.111** | 19.94 | 114.742 |

| Input | Prefill speedup | TTFT speedup | Decode ratio | End-to-end speedup |
|---:|---:|---:|---:|---:|
| 2,048 | 1.218x | 1.216x | 0.996x | 1.045x |
| 25,000 | 1.347x | 1.347x | 0.997x | 1.269x |

The 25K TTFT drops by 30.9 s, about 1.24 s per chunk. TTFT standard deviations are below 0.16 s
(borrow 25K) and 0.005 s (other groups).

Mechanism checks, all measured:

| Group | GGUF-in-place MB per request | Prefill GR3D mean | Prefill samples <50% GR3D | Prefill VDD_GPU_SOC | Busiest CPU core in prefill |
|---|---:|---:|---:|---:|---:|
| borrow 2K | 1,835.7 | 86.5% | 10.0% | 25.9 W | 56.1% |
| own 2K | **0.0** | 98.8% | 0.0% | 29.6 W | 6.5% |
| borrow 25K | 17,693.5 | 73.2% | 26.4% | 23.8 W | 85.7% |
| own 25K | **0.0** | 98.7% | 0.0% | 30.0 W | 4.5% |

With owned buffers the expert stream disappears entirely. The cumulative counter stays at the 32,087.7 MB
initial cache fill. The counter (`file_read_bytes_`, `src/core/expert_source.cpp`) counts bytes copied out of
the mapped GGUF, whether the pages came from the page cache or from the NVMe. It therefore shows that the
repeated file-backed expert supply is gone. It does not show how many of those bytes were NVMe reads. The prompt-completion refill after the last chunk (borrow: prompt minus last PP,
0.176 s at 25K) is also gone. GPU activity during prefill matches decode, and the prompt path no longer keeps a
CPU core busy. Both modes report 24,576 expert slots. VRAM free after loading is 20,798 MiB with borrow and
20,209 MiB with own buffers (589 MiB reserved).

ASTOP saw a 306 MHz minimum GPU clock sample within the borrow 25K requests, consistent with DVFS during
idle gaps. Every other group stayed at 1,300.5 MHz. Maximum available GPU temperature was 58.1-58.3 C with
borrow and 59.6-60.0 C with own buffers. Memory pressure was `normal` throughout. Peak swap was 627 MiB
(borrow) and 719 MiB (own), not zero. tegrastats `VDD_GPU_SOC` is a board rail that includes SoC load; it is
not GPU-only power.

**Decode.** Internal decode is 0.34-0.39% lower with own buffers (20.415 to 20.336 at 2K and 19.993 to
19.925 at 25K). Within-group standard deviations are about 0.002 tok/s, so this is a real difference between
the two engine loads. It is not attributed to the owned buffers, however: v1's MTP-on engine with borrowed
slots measured 20.335 and 19.919, matching the own mode. The borrow load in this run was the faster one.
The decode timing lines agree (verify 101.75 vs 102.16 ms per window at 2K). Load-to-load decode variation
of this size should be expected in later A/Bs.

## Interpretation and next steps

- **Measured:** on this Orin with Gyro-M and the full resident cache, `--no-prefill-borrow` is a strict win
  for these prompts: 1.35x prefill at 25K, 1.22x at 2K, token-identical output, no decode loss beyond
  load-to-load variation, and 0.6 GiB of VRAM.
- **Inference:** the remaining prefill is GPU-busy (about 99% GR3D), so further prefill gains need more
  efficient GPU work. Candidates are larger owned chunks (2,048/4,096; each larger buffer must still fit next
  to the full cache) and kernel work identified by profiling. Prefill throughput is now nearly flat between
  2K and 25K (286.6 vs 280.8 tok/s).
- The serving launcher and its config do not yet pass `--no-prefill-borrow`. Its 16K context differs from
  this 32K benchmark, so check the reserved buffer size and free VRAM when enabling it there.
- Limitations: one prompt per length, single mode order, no workload-wide generalization.

## Artifacts and reproduction

- Remote frozen directory (read-only):
  `${TEST_HOME}/.local/share/jetson-orin/strata-gyro-prefill-borrow-20261007-v1`. Durable systemd user unit:
  `orin-strata-gyro-prefill-borrow-20261007-v1.service`.
- ASTOP watch `strata-gyro-prefill-borrow-20261007-v1` on wrapper PID 182554 reported `disappeared` with
  unknown exit status (`terminal-run-watch.json`). The event was recorded first, then acknowledged and the
  watch removed. Success is established by the explicit exits, systemd state and validated contracts, not by
  the ASTOP event.
- `files.sha256` hashes 33 remote files. Manifest SHA256:
  `8db228bdcb543a69b58946ae6475ffd4d8387078da7f6092575918ad73789f56`. The ignored local `run/` copy verifies
  all 32 copied members. Only the raw `telemetry.ndjson` (SHA256 `0dd93c57…45cd7`) stays remote.
- `result.json` contains every request, output IDs, text, token arrival times, raw DONE lines and engine
  metadata. `summary.json` adds per-request GGUF read deltas and tegrastats phase summaries.
- Launch commit: `9ff2b7f` (`launch-commit.txt`). The freeze script was generalized afterwards to accept a
  single watch file. The version used is in the frozen directory.
- Summarize a writable copy with
  `uv run --offline --no-project --python /usr/bin/python3 --no-python-downloads python summarize-strata-gyro-benchmark.py DIR`.
- During recording, a first `pgrep -f build-sm87/strata` matched its own command line. The final
  `native-processes-final.json` uses an exact process-name match and is empty.
