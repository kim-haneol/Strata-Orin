# Gyro-M matched resident MTP A/B on Orin

Prepared on 2026-10-07 to replace the earlier unmatched CLI/resident smoke.
Protocol: `config/strata-gyro-mtp-benchmark-v1.json`.
Harness: `scripts/strata-gyro-benchmark.py` and `scripts/run-strata-gyro-benchmark.sh`.

The same pinned ARM64/SM87 Strata binary and Gyro-M pack are used for both
conditions, with MTP weights loaded in both. The off condition caps the MTP
window at one, offers no drafts and launches no MTP draft compute graph.
Both retain the same draft-KV prefill work. Context is 32,768, INT8 KV,
prefill chunk 1,024, verifier capacity four, GPU expert cache 24,576,
prompt/conversation cache zero and suffix drafting zero. Greedy generation
produces exactly 512 tokens with natural EOS stopping suppressed.

One Korean explanation prompt at each exact model-token length (2,048 and
25,000) is used unchanged across conditions and repetitions. Each condition
warms up once per length, then measures three times per length. Model startup
is excluded. The 25K input is a size proxy for the earlier gpchan audit,
not a replay of its tools, persona, memory or real provider serialization.

Internal prompt time includes sequence reset, target and draft prefill,
and any borrowed expert-cache slot refill; its last input token is processed
by the first verify window. Report `(N-1)/prompt_seconds` for that phase,
512/internal_decode_seconds for the internal decode interval, and separate
IPC TTFT, stream decode `(512-1)/(last-first)` and end-to-end-to-DONE time.
TTFT is not pure prefill. PP timers are cumulative, not additive chunk times.

Remote result directory:
`${TEST_HOME}/.local/share/jetson-orin/strata-gyro-mtp-20261007-v1`.
Durable systemd user unit: `orin-strata-gyro-mtp-bench-20261007-v3.service`.
ASTOP exact-process watches: `strata-gyro-mtp-bench-20261007-v3` (completed
requests) and `strata-gyro-mtp-prefill-20261007-v3` (prefill-token progress).

ASTOP records CPU/per-core activity, GPU activity/clocks, memory pressure
and swap. GPU temperature was null at initial idle inspection but became
available during inference (for example 56.875 C at 1,300.5 MHz); the frozen
protocol's blanket temperature-unavailable assumption is corrected here.
Retain null observations and summarize the available measurements. GPU power
remains unavailable. A tegrastats companion records independent board evidence. Direct wrapper
exit, native child exit and complete result contracts establish completion;
an ASTOP unknown exit does not establish success.

The first launch exited 127 before inference because the systemd PATH did not
include `nvpmodel` in `/usr/sbin`. The second exited 1 before engine load because
Strata's EOS parser rejects negative IDs. Both were corrected before the third
launch, using a complete PATH and unreachable positive EOS ID 2,147,483,647.
Their separate remote `-launch-attempt-1` and `-launch-attempt-2` directories
are retained; no measured requests from those attempts enter this comparison.

## Results

All 16 requests completed: four warmups and twelve measurements. Both native
engines exited 0, the wrapper's explicit exit file is 0, and systemd reports
`Result=success`, `ExecMainStatus=0`, `MainPID=0`. Every request generated
exactly 512 tokens, read its entire input with zero reused tokens and reported
zero decode offload. No natural EOS occurred in the 512-token responses.
All three repetitions produced identical tokens within each condition, and
the full outputs matched between MTP off/on at both lengths.

Three-measurement arithmetic means; decode is the common IPC stream metric.

| Input tokens | MTP drafting | Internal prefill tok/s | Internal prompt s | IPC TTFT s | IPC decode tok/s | IPC total s |
|---:|:---:|---:|---:|---:|---:|---:|
| 2,048 | off | 233.64 | 8.761 | 8.831 | 14.60 | 43.839 |
| 2,048 | on | 235.71 | 8.684 | 8.754 | 20.35 | 33.867 |
| 25,000 | off | 207.81 | 120.299 | 120.396 | 14.51 | 155.622 |
| 25,000 | on | 209.03 | 119.596 | 119.684 | 19.93 | 145.322 |

| Input tokens | Decode speedup | End-to-end speedup | Total time saved |
|---:|---:|---:|---:|
| 2,048 | 1.394x | 1.294x | 9.972 s (22.75%) |
| 25,000 | 1.374x | 1.071x | 10.300 s (6.62%) |

Prefill differs by less than 1% between these conditions; there is no evidence
here that enabling MTP materially accelerates prefill. Explanation decoding
does benefit in this workload. Long fresh inputs remain dominated by prefill:
the 25K request still waits about two minutes for its first token. These are
cold-prefix requests to a loaded model, not cold model-start requests.

Sample decode standard deviations were 0.00168 / 0.00369 tok/s at 2K (off/on)
and 0.00268 / 0.00225 tok/s at 25K. This demonstrates repeatability of these
two prompts, not a confidence interval across explanation tasks.

The internal decode means (`512 / DONE.decode_seconds`) were 14.598 / 20.335
tok/s at 2K and 14.508 / 19.919 tok/s at 25K. The small differences from IPC
decode follow the documented first-token and completion timer boundaries.
Final PP timers were 8.584 / 8.516 s at 2K and 120.126 / 119.421 s at 25K;
they exclude the final prompt-completion/refill portion and are cumulative.

| Input tokens | Draft accepted / offered per request | Acceptance | Three-run totals |
|---:|---:|---:|---:|
| 2,048 | 290 / 411 | 70.56% | 870 / 1,233 |
| 25,000 | 283 / 385 | 73.51% | 849 / 1,155 |

Off-condition draft counters were zero. Both retain the same draft-KV prefill
and MTP memory allocations; this measures the gain from speculative decoding,
not the prefill/memory difference of a completely MTP-free runner.

## Telemetry and completion

There were 1,106 ASTOP samples in the measured request windows. Every available
GPU-frequency sample was 1,300.5 MHz in all four groups. GPU activity reached
99.9-100%; mean request activity was 94.0% / 97.3% at 2K and 78.6% / 75.5% at
25K (off/on). These means include prefill, gaps and decode and do not isolate
the reason for idle portions. Available GPU temperatures ranged from 51.562
to 58.968 C. Memory pressure was `normal` in every measured sample. Peak swap
was 684.5 MiB off and 718.75 MiB on; it was not zero. GPU power and native
process GPU utilization/memory remained unavailable, preserved as null/zero
available-sample counts rather than fabricated zero usage.

Model loading took 184.976 s off and 467.957 s on. It includes filesystem/cache
conditions that were not matched and is excluded from the performance table;
do not interpret their difference as MTP startup overhead. During the slower
second load, the native process waited on file-backed pages while physical
read counters advanced. No storage-cause tuning or new engine profiling was
performed in this benchmark.

ASTOP's two exact-PID terminal events report `disappeared` with unknown exit
status on this platform. They are preserved unchanged in
`terminal-request-watch.json` and `terminal-prefill-watch.json`; success is
established by the direct exits, systemd state and validated artifacts above.
After recording the events, only these two watches were acknowledged and
removed. Final native process list is empty, MAXN mode 0 remains selected,
GPU activity is zero and ASTOP reports about 52.0 GiB available, jobs zero.

## Artifacts and reproduction

- Executed protocol and harness: remote directory above, frozen read-only.
- Launch commit: `85e6638` (full ID in `launch-commit.txt`).
- Binary SHA256: `8ce7f7cb68919ed026c54ebf0a413056c653b9fbee1688295ce7c3c5ab465354`.
- Result: `result.json` contains every warmup/measurement, all output IDs,
  text, token arrival times, timers, raw DONE lines and engine metadata.
- Derived analysis: `summary.json`, including means/medians/ranges/standard
  deviations, acceptance, output equality and request-scoped telemetry.
- Raw engine/IPC logs, input text and token IDs: local ignored `run/` copy and
  the read-only remote directory. Complete raw `telemetry.ndjson` is retained
  remotely; the local copy omits only this large file.
- `files.sha256` hashes 35 remote files. Manifest SHA256:
  `49af5e1ce1221a572914a7eae97f9262d7576bc7602d304fb09b841311be0705`.
  `local-copy-verification.json` verifies all 34 copied members and records
  the omitted telemetry hash. The manifest itself was also verified.
- `freeze-receipt.json` records the independent completion checks.
- `launch-attempt-1/` and `launch-attempt-2/` retain compact failed-launch
  diagnostics. The complete failed directories remain remotely.

Analysis command on the remote evidence before freezing was:
`uv run --offline --no-project --python /usr/bin/python3 --no-python-downloads python summarize-strata-gyro-benchmark.py RESULT_DIRECTORY`.
It validates the request counts, exact sizes, zero prefix reuse, identical
launch arguments except MTP window cap, script/protocol hashes and
known native exits before producing the summary. Source pins and the binary
identity are recorded in the result; the existing source/build validation is
in `evidence/strata-gyro-orin-20261007/`. To analyze a frozen copy,
copy its complete directory to a new writable directory first.
