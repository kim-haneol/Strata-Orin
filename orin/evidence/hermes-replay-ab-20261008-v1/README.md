# Hermes agent workload: capture and greedy replay A/B of suffix drafting (2026-10-08)

Goal (plan B): find out whether the suffix drafter raises tokens per verify window on real Hermes agent
traffic, where outputs repeat tool arguments and file content. This follows `evidence/hermes-strata-orin-20261008`
and uses the same serving binary `build-sm87-sub8/strata.sub8`.

## Capture

- **Recorder:** `scripts/strata-request-recorder.py`, running as the host user unit
  `strata-request-recorder-18093` (127.0.0.1:18093 → tunnel 18092). It saves each POST body whole and streams
  the response back. The `strataorin` profile's `base_url` was pointed at 18093 for the capture.
- **Tasks** (`run-tasks.sh` in the capture dir; `hermes -p strataorin -t file,todo --yolo --in DIR -z PROMPT`):
  - t1: write `stats.py` (mean/median/percentile + CLI) and pytest tests. 14 API calls, 373 s.
  - t2: fix a bug in `inventory.py` and add methods. 30 API calls, 639 s.
  - t3: aggregate `results.csv` by region into `summary.json` and summarize in Korean. 14 API calls, 262 s.
  - t4 (document restructuring) was stopped after the incident below.
- In total, 61 requests were recorded (58 agent calls plus auxiliary requests). The raw requests are in
  `work/hermes-capture-20261008/requests/` (gitignored).

### Incident and how representative the tasks are

- `--in DIR` did not reach the file tools. Hermes resolved "this folder" to `${TEST_HOME}`. So:
  - t1 wrote `${TEST_HOME}/stats.py` and `${TEST_HOME}/test_stats.py`. Both were moved to the trash with
    `gio trash` and can be recovered. No backup existed, so we do not know whether same-named files were there
    before.
  - t2 and t3 did not find their input files. They spent their calls on `search_files` / `read_file` across
    the home directory, then asked for the right path. t3 read other CSV files in home. This was read-only and
    the content stayed on the local model.
  - The terminal tool was not in the toolset, so t1 could not run pytest. The model said so.
- The workload is therefore real Hermes traffic: system prompt, tool schemas, tool results, code writing and
  tool-call arguments. However, two of the three tasks are "search and give up", not completed edits. Treat
  the code-writing share (t1) as smaller than it would be in a successful session.

## Replay protocol

- `scripts/replay-strata-requests.py`: sends the 61 recorded bodies in their original order, with
  `temperature 0`, without streaming. The prompt cache sees the same prefixes as the live session. Per
  request it saves the server timings and the output.
- `scripts/compare-strata-replays.py`: checks output identity (content plus tool calls, ignoring the random
  tool-call `id`) and sums `predicted_n / predicted_ms`.
- Server configs (copied here; each is the serving config plus one change):

| Mode | Change | Verify window |
|---|---|---:|
| a | none: serving config, suffix draft on (engine default), MTP cap 4 | 6 |
| b | `--spec 8 --mtp-max-t 4`: suffix may fill up to 8, MTP still capped at 4 | 8 |
| c | `--suffix-draft 0` | 4 |

- Order: b, then c, then a (a2). The first a run (a1) failed to start because the GPU lock was still held
  after the previous server stopped (`replay-a1-failed-lock.log`). The orchestrator (`run-replay*.sh`) now waits
  for the lock. **Each mode ran once (no A-B-B-A).** Run-to-run spread on the 2K benchmark has been about
  0.4%, so differences under about 1% are not resolved here.

## Results

All 61 outputs are identical across a, b and c (`compare.txt`; per-request output hashes in
`replay-*-timings.json`).

| Mode | Decode tok/s (all, 21,644 tok) | Requests ≥ 300 tok (20 req) | Decode s | Prompt s | Wall s |
|---|---:|---:|---:|---:|---:|
| c: suffix off | 26.25 | 25.99 | 824.6 | 268.7 | 1104.2 |
| a: serving (window 6) | 26.16 | 25.89 | 827.3 | 271.9 | 1110.0 |
| b: window 8 | 26.30 | 26.01 | 823.1 | 267.8 | 1101.7 |

Draft statistics, summed from the engine logs (`engine-replay-*-summary.txt`):

| Mode | Drafts accepted / offered (MTP + suffix) | Suffix windows | Suffix accepted / offered | Windows (estimate) | Tokens per window (estimate) |
|---|---:|---:|---:|---:|---:|
| c | 13,670 / 17,813 (76.7%) | 0 | 0 / 0 | 7,974 | 2.714 |
| a | 13,678 / 18,005 (76.0%) | 439 | 1,051 / 1,305 | 7,966 | 2.717 |
| b | 13,846 / 18,397 (75.3%) | 375 | 1,475 / 1,979 | 7,798 | 2.776 |

- **Measured:** decode tok/s, accepted/offered counts, suffix window counts, prompt and wall times.
- **Estimated:** window count = generated − accepted. This assumes each window emits its accepted drafts plus
  one token, and that request boundaries add nothing.

### Reading

- Suffix drafting fires in about 5% of windows, and those drafts are accepted at a high rate (80% in a, 75% in
  b). The windows it takes over would mostly have succeeded with MTP drafts anyway. In a, the window count
  falls by 8 of 7,974 (0.1%).
- In b, the 8-wide window saves 2.2% of windows. But each extra window token costs about 13.4 ms
  (`evidence/strata-gyro-profile-sub8-graph-20261008-v1`), and the net is +0.2% against c. That is inside the noise.
- **Conclusion:** on this workload, suffix drafting gives no measurable decode gain, in either window size.
  The serving config (a) is within about 0.5% of off (c); one run each cannot tell which is faster. Serving
  is left unchanged. The repetitive part of agent output (tool arguments, rewritten code) is already
  predicted by MTP.

### Prompt reads are a quarter of agent wall time

In mode a, prompt work was 271.9 s of 1110 s of wall time. Per request (`replay-a2-timings.json`, engine log):

| Tokens read (after cache reuse) | Requests | Tokens | Time | Rate | Per request |
|---|---:|---:|---:|---:|---:|
| ≥ 2,048 | 15 | 55,558 | 129.6 s | 429 tok/s | 8.6 s |
| 512-2,047 | 14 | 15,793 | 63.8 s | 247 tok/s | 4.6 s |
| 64-511 | 32 | 8,483 | 78.4 s | 108 tok/s | 2.5 s |

- A linear fit over reads under 2,048 tokens gives about 1.85 s + 2.36 ms per token. Reads of 97-109 tokens on
  top of a cached prefix of 4-16K tokens take 1.6-1.9 s.
- So every agent turn pays a fixed cost of about 1.5 s before decoding starts. Across the 61 requests, that
  is on the order of 80-90 s, or 7-8% of wall time. This is a fit, not a breakdown. Where the fixed cost goes
  (checkpoint save/restore, conversation-cache parking, prefill graph setup, expert reads for a small batch)
  has not been measured.

## Reproduce

1. Serve with `scripts/serve-strata-gyro-config.sh` and a `strata-gyro-replay-<x>.json` config (copied to the
   Jetson as `server-replay-<x>.json`), or run `run-replay2.sh`, which stops servers, waits for the GPU lock,
   starts each mode, replays and stops.
2. `uv run --no-project python scripts/replay-strata-requests.py work/hermes-capture-20261008/requests http://127.0.0.1:18092 replay-<x>.json 1 61`
3. `uv run --no-project python scripts/compare-strata-replays.py replay-c.json replay-b.json replay-a2.json`

The full replay outputs (`replay-*.json`, containing generated code and text that quotes local files) and the
recorded requests stay in the gitignored capture dir. Only timings and output hashes are committed here.
