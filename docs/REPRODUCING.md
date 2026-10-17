# Reproducing the Jetson AGX Orin Gyro-M experiments

This guide builds the preserved Orin source in this checkout and reconstructs the October 7–9, 2026 text-inference experiments described in [OPTIMIZATION.md](OPTIMIZATION.md). The main target is the original Gyro-M checkpoint with the `f16cache` engine, the `enko` MTP vocabulary, INT8 KV, and a 64K context limit. Later abliterated weights, vision support, and 262K deployment are separate configurations.

The published engine tree was checked against the preserved final Git tree. Commands and configuration fields were inspected, but this documentation pass did **not** perform a fresh download, CUDA build, or inference run. The original measurements remain the evidence for execution on Orin. See the [source and publication record](../orin/README.md).

## 1. What is reproducible from this repository?

| Target | Available material | Boundary |
|---|---|---|
| Build the optimized source | This checkout, pinned GGML revision, and its shared-codebook patch | The engine tree matches the recorded final source; publication commits have new IDs. |
| Build and serve the original model | Build flags, packing tools, model revision/hash, and server configuration | Requires an Orin system, upstream downloads, and sufficient memory/storage. |
| Fixed 2K/25K benchmarks | Prompt generator, exact token/output contract, mode configurations, and summarizer | Rebuild the variants required by the chosen protocol and record the new environment. |
| The **27.02 tok/s** result | 61-request replay configuration, comparison code, summaries, and output checks | The original request capture is Git-excluded. A clone alone cannot reproduce that exact replay. |
| Later 262K/vision deployments | Separate validation and deployment records | They do not reproduce the earlier 64K replay measurement. |

The 27.02 tok/s figure is aggregate server-reported **decode** throughput for 21,644 generated tokens across 61 recorded requests. It excludes prompt processing. A new prompt, different weights, or a fresh agent session is a different workload.

## 2. Hardware, software, and workspace

The recorded machine was a **Jetson AGX Orin 64GB**, SM87 with 16 SMs, running Ubuntu 24.04, Jetson Linux 39.2.1, JetPack 7.2.1, and CUDA Toolkit 13.2.2 (`nvcc` 13.2.86). Benchmarks used MAXN mode 0; GPU clock observations reached 1,300.5 MHz. These identify the tested environment, rather than establish compatibility with every other BSP or power profile. See the [recorded environment](../orin/README.md#recorded-environment) and [initial port](../orin/evidence/strata-gyro-orin-20261007/README.md).

You need Git, a C++ build toolchain, CUDA development tools, CMake, Ninja, `uv`, Python 3.12 on the tested Ubuntu image, and `curl`. The pinned Strata `requirements.txt` specifies the Python packages. The original GGUF alone is **91,965,774,272 bytes**; allow additional space for MTP tensors, packs, builds, logs, and downloads. Its GPU-resident expert portion is about 29.88 GiB, not the size of the complete GGUF.

Run build, kernel validation, and inference commands **on the Jetson**. Use a fresh directory and an idle board for GPU work: the historical full-resident model and another full model cannot be assumed to fit together. These examples do not replace or restart an existing production service.

Clone this repository on the Jetson, then set the following variables from its root in one shell. Use your own unused workspace path; do not overwrite an existing runtime.

```bash
set -euo pipefail
export GYRO_REPO="$(pwd)"
export ORIN_SUPPORT="$GYRO_REPO/orin"
export GYRO_RUN="$HOME/gyro-orin-repro"
export GYRO_SRC="$GYRO_RUN/src/strata"
export GYRO_GGML="$GYRO_RUN/src/agention-llama"
export GYRO_BUILD="$GYRO_RUN/build-sm87-f16cache"
export GYRO_PY="$GYRO_RUN/venv/bin/python"
test -f "$ORIN_SUPPORT/patches/strata-orin-arm64.patch"
test ! -e "$GYRO_RUN"
mkdir -p "$GYRO_RUN/src" "$GYRO_RUN/receipts"
ln -s "$GYRO_REPO" "$GYRO_SRC"
```

Inspect the local observation setup before GPU work:

```bash
command -v astop
astop --help-agent
astop agent-summary
nvpmodel -q
nvcc --version
cmake --version
```

Use the installed ASTOP server address and `/v1/capabilities` for the version you have. The original launchers used a separate observer on port 19724; that is a testbed convention, not a required port for Strata. Record unavailable fields as unknown. `nvpmodel -q` supplies the power-profile identity that a GPU-utilization reading does not establish.

## 3. Use this checkout and prepare the pinned GGML source

The ARM64 port and final optimization are already applied. Do not apply the cumulative Strata patches to this checkout again. The publication retains upstream history and consolidates the experimental commits without changing the final engine tree.

| Source state | Identifier |
|---|---|
| Strata upstream base | `434e136f73a7da2bfd11cde3bb0aa0acaf1328c6` |
| Historical ARM64 port | `bb98d09538fd4188feb806443da21870903057f9` |
| Published ARM64 base tag | `orin-arm64-base` |
| Expected ARM64 tree | `d0e8fb38226b84841036fba16a585b4e6d576284` |
| Historical final source | `9acaf8eacb4d56af6e9f4369e0573ce92b1f3de3` |
| Published final-code tag | `orin-f16cache-code` |
| Expected final-code tree | `24603390b080e9011aa02fc779b818a79ea8f9db` |
| GGML fork base | `93a24e52cc12687213c0432af41c7fe2ab1a0eda` |

The checks below verify the source pins and that the build-related paths still match the final-code tag. Configure your Git author identity before committing the local GGML patch.

```bash
test "$(git -C "$GYRO_REPO" rev-parse 'orin-arm64-base^{tree}')" = d0e8fb38226b84841036fba16a585b4e6d576284
test "$(git -C "$GYRO_REPO" rev-parse 'orin-f16cache-code^{tree}')" = 24603390b080e9011aa02fc779b818a79ea8f9db
git -C "$GYRO_REPO" diff --exit-code orin-f16cache-code -- CMakeLists.txt setup.py include src tests tools

git init "$GYRO_GGML"
git -C "$GYRO_GGML" remote add origin https://github.com/agentionai/llama.cpp.git
git -C "$GYRO_GGML" fetch --depth=1 origin 93a24e52cc12687213c0432af41c7fe2ab1a0eda
git -C "$GYRO_GGML" checkout --detach FETCH_HEAD
git -C "$GYRO_GGML" apply --check "$ORIN_SUPPORT/patches/agention-llama-tq-mmq-shared-lut.patch"
git -C "$GYRO_GGML" apply --index "$ORIN_SUPPORT/patches/agention-llama-tq-mmq-shared-lut.patch"
git -C "$GYRO_GGML" commit -m "Stage TQ MMQ codebook in shared memory"
```

The [source verification receipt](../orin/source-verification.json) records the tree matches and the read-only check against the deployed staged source. Upstream licenses and attribution are retained. The archived Strata patches are for reconstructing older experiment variants from the base, and the separate GGML patch remains necessary for this build.

## 4. Build for SM87

The [build helper](../orin/tools/build.sh) uses the recorded SM87 flags and is also used for the optional benchmark variants below.

```bash
gyro_build() { bash "$ORIN_SUPPORT/tools/build.sh" "$1" "$2" "$3"; }
gyro_build "$GYRO_SRC" "$GYRO_BUILD" "$GYRO_GGML"
sha256sum "$GYRO_BUILD/strata" > "$GYRO_RUN/receipts/binary.sha256"
```

Keep `STRATA_PREFILL_MMQ=OFF` as recorded; do not infer from that option's name that the native expert MMQ path is disabled. The separate GGML source and shared-codebook patch are still part of this build. Record your compiler, dependencies, tree IDs, and binary hash. A newly compiled binary is not assumed to match the historical `996f228f…` binary byte-for-byte.

## 5. Fetch and prepare the original checkpoint

The input is the original `agentionai/Qwen3.8-Flash-Next-Gyro-GGUF` revision `e40c350d6a8974dc1b70cd46ec451a6e63561cb6`, file `Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf`, SHA256 `5edfedf7398e18cb9d7cc32f40035acfc5f66bb9051e8216a053ac72d5324ee9`. Use the [download manifest](../orin/config/gyro-m-download.json) with a local cache path:

```bash
uv run --no-project python - <<'PY'
import json, os
from pathlib import Path
run = Path(os.environ["GYRO_RUN"])
manifest = json.loads((Path(os.environ["ORIN_SUPPORT"]) / "config/gyro-m-download.json").read_text())
manifest["cache"] = str(run / "hf-cache/hub")
(run / "download.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY
uv run --no-project python "$ORIN_SUPPORT/tools/prefetch-gyro-model.py" \
  "$GYRO_RUN/download.json" "$GYRO_RUN/download-state"
export GYRO_MODEL="$GYRO_RUN/hf-cache/hub/models--agentionai--Qwen3.8-Flash-Next-Gyro-GGUF/snapshots/e40c350d6a8974dc1b70cd46ec451a6e63561cb6/Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf"
printf '%s  %s\n' 5edfedf7398e18cb9d7cc32f40035acfc5f66bb9051e8216a053ac72d5324ee9 "$GYRO_MODEL" | sha256sum -c -

uv venv --python /usr/bin/python3 --no-python-downloads "$GYRO_RUN/venv"
uv pip install --python "$GYRO_PY" -r "$GYRO_SRC/requirements.txt"
export STRATA_GGUF_PY="$GYRO_GGML/gguf-py"
uv run --offline --no-project --python "$GYRO_PY" --no-python-downloads \
  python "$GYRO_SRC/tools/iq_pack.py" --gguf "$GYRO_MODEL" \
  --out "$GYRO_RUN/packs/gyro-m" --compat-bf16
```

The downloader checks size, range responses, and the complete hash. If using an existing copy, set `GYRO_MODEL` to that file and perform the same hash check instead of downloading again. `--compat-bf16` reproduces the recorded dense compatibility conversion; it is not a claim that every converted dense value is unchanged. The original conversion record distinguishes exact and rounded tensors.

### MTP tensors, packs, and the `enko` vocabulary

Use the pinned original MTP tensor revision, not the current model's latest branch. The fetch utility has a fallback if a revision disappears; a fallback is not an exact reproduction. Require successful pinned verification and inspect the inventory's `repo` revision.

```bash
export STRATA_MTP_REVISION=de4b8e4d43b917e7706784d8bb445c9af86a3540
uv run --offline --no-project --python "$GYRO_PY" python "$GYRO_SRC/tools/mtp_fetch.py" fetch --out "$GYRO_RUN/mtp"
uv run --offline --no-project --python "$GYRO_PY" python "$GYRO_SRC/tools/mtp_fetch.py" verify --out "$GYRO_RUN/mtp"
uv run --no-project python - <<'PY'
import json, os
from pathlib import Path
inventory = json.loads((Path(os.environ["GYRO_RUN"]) / "mtp/mtp-inventory.json").read_text())
assert inventory["repo"].endswith("/resolve/de4b8e4d43b917e7706784d8bb445c9af86a3540/")
assert len(inventory["tensors"]) == 31
assert inventory["total_bytes"] == 5214301696
PY
uv run --offline --no-project --python "$GYRO_PY" python "$GYRO_SRC/tools/mtp_pack.py" \
  --src "$GYRO_RUN/mtp" --experts q2_0 --out "$GYRO_RUN/mtp/mtp-q2_0.gguf"
uv run --offline --no-project --python "$GYRO_PY" python "$GYRO_SRC/tools/mtp_rt.py" \
  --gguf "$GYRO_RUN/mtp/mtp-q2_0.gguf" --out "$GYRO_RUN/mtp/rt"
cp "$GYRO_SRC/data/draft_vocab.bin" "$GYRO_RUN/mtp/rt/draft_vocab.bin"

mkdir "$GYRO_RUN/mtp/rt-enko"
for name in dense.bin dense.txt experts.bin; do
  ln -s "../rt/$name" "$GYRO_RUN/mtp/rt-enko/$name"
done
uv run --offline --no-project --python "$GYRO_PY" python "$GYRO_SRC/tools/draft_vocab.py" \
  --gguf "$GYRO_MODEL" --base "$GYRO_SRC/data/draft_vocab_en.bin" \
  --add hangul,cjk_punct --out "$GYRO_RUN/mtp/rt-enko/draft_vocab.bin"
printf '%s  %s\n' 8de2bec33e6a866311c70ec35ca58eec8048c9645b49ee94ef5bce523055174f \
  "$GYRO_RUN/mtp/rt-enko/draft_vocab.bin" | sha256sum -c -
```

`uv --offline` constrains dependency resolution; it does not prevent `mtp_fetch.py` from making its own HTTP requests. The original MTP subset contains 31 tensors totaling 5,214,301,696 bytes. The default vocabulary has 106,299 tokens; `enko` has 47,493 and limits Chinese/Japanese drafting. Keep both directories: the fixed-input benchmarks use `mtp/rt`, while the final replay uses `mtp/rt-enko`. Compare pack outputs with the [derived-artifact hashes](../orin/evidence/strata-gyro-orin-20261007/derived-artifacts.sha256) and [vocabulary hashes](../orin/evidence/strata-gyro-agent-round2-20261008-v1/draft-vocab-sha256.txt).

## 6. Validate the build before serving

These are GPU validation commands, separate from downloading and packing. On an idle Jetson, record ASTOP observations and direct command exits alongside the results.

```bash
"$GYRO_BUILD/strata-device" --selftest
cuobjdump --list-elf "$GYRO_BUILD/strata"
ctest --test-dir "$GYRO_BUILD" --output-on-failure \
  -R '^(native_hadamard_test|router_dot_parity|native_expert_parity_(hadamard_)?tq2_t_tq2_t|gdn_parity|gdn_rec_parity|gemm_bf16_parity|mmq_gather_group_test)$'
"$GYRO_BUILD/native_expert_parity" --load-check "$GYRO_RUN/packs/gyro-m" "$GYRO_MODEL"
```

Confirm SM87 device/code identity, successful selected tests, and a load check reporting zero failures. A build exit of 0 alone does not establish correct loading or output parity. These checks also do not establish the 27.02 tok/s result.

## 7. Create a local 64K configuration and serve

Start from the frozen [64K replay configuration](../orin/config/strata-gyro-replay-f16cache.json). The generic [server configuration](../orin/config/strata-gyro-orin-server.json) was subsequently updated to 262K and newer abliterated weights, so it is not the template for this measurement. The following rewrites only filesystem locations, the engine executable, and the local listening port; inference arguments retain the replay settings.

```bash
uv run --no-project python - <<'PY'
import json, os
from pathlib import Path
repo, run = Path(os.environ["ORIN_SUPPORT"]), Path(os.environ["GYRO_RUN"])
cfg = json.loads((repo / "config/strata-gyro-replay-f16cache.json").read_text())
old_runtime = "${RUNTIME}"
old_model = cfg["args"][cfg["args"].index("--native") + 1]
def relocate(value):
    if isinstance(value, str):
        return value.replace(old_model, os.environ["GYRO_MODEL"]).replace(old_runtime, str(run))
    if isinstance(value, list):
        return [relocate(item) for item in value]
    if isinstance(value, dict):
        return {key: relocate(item) for key, item in value.items()}
    return value
cfg = relocate(cfg)
cfg["exe"] = str(run / "build-sm87-f16cache/strata")
cfg["host"], cfg["port"] = "127.0.0.1", 18094
cfg["log"] = str(run / "server-engine.log")
(run / "server-64k.json").write_text(json.dumps(cfg, indent=2) + "\n")
PY
cd "$GYRO_SRC"
STRATA_APR=1 uv run --offline --no-project --python "$GYRO_PY" \
  python -m serve.server --engine strata --config "$GYRO_RUN/server-64k.json" \
  --host 127.0.0.1 --port 18094
```

This foreground server uses a separate loopback port; stop it with Ctrl-C when finished. In a second terminal, wait for bounded readiness and perform a small functional request:

```bash
curl --fail --show-error --retry 60 --retry-connrefused --retry-delay 2 \
  --retry-max-time 600 --max-time 5 http://127.0.0.1:18094/health
curl --fail --show-error http://127.0.0.1:18094/v1/models
curl --fail --show-error --max-time 120 http://127.0.0.1:18094/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data '{"model":"qwen38-flash-next-gyro-m","messages":[{"role":"user","content":"What is 2 + 2?"}],"temperature":0,"max_tokens":32}'
```

Check the engine log for all 24,576 expert slots, `arena_mib 0`, INT8 KV, 65,536 context, 8,192-token owned prefill, MTP using `rt-enko`, and dense FP16 cache population. The cache's first-read allocation cost is expected. This request verifies API operation; it is not the published replay benchmark.

## 8. Reproduce the fixed-input benchmark family

The [benchmark harness](../orin/tools/strata-gyro-benchmark.py) constructs and saves exact tokenized prompts itself. It uses 2,048 and 25,000 input tokens, 512 greedy output tokens, one warmup and three measured requests per length/mode, no prefix reuse, INT8 KV, and 32,768 context. The unreachable EOS ID forces output length; this is not an answer-quality test. Keep the protocol's mode order and timer definitions.

| Protocol | Build directories expected below `GYRO_RUN` | Strata patch after `orin-arm64-base` | GGML |
|---|---|---|---|
| [MTP A/B](../orin/config/strata-gyro-mtp-benchmark-v1.json) | `build-sm87` for both modes | None | Unpatched pinned fork |
| [stage-lut A-B-B-A](../orin/config/strata-gyro-tq2t-stage-lut-ab-v1.json) | `build-sm87`, `build-sm87-tq2t-prefetch` | Candidate: `strata-orin-tq2t-stage-lut.patch` | Unpatched pinned fork |
| [sub8 A-B-B-A](../orin/config/strata-gyro-sub8-ab-v1.json) | `build-sm87-mmq-lut`, `build-sm87-sub8` | Baseline: `strata-orin-tq2t-stage-lut.patch`; candidate: `strata-orin-tq2t-rpw4-sub8.patch` | Shared-LUT patch for both |

For example, build the MTP A/B baseline from the published ARM64 base tag. This preserves the final serving checkout separately:

```bash
git -C "$GYRO_SRC" worktree add --detach "$GYRO_RUN/src/strata-base" orin-arm64-base
git -C "$GYRO_GGML" worktree add --detach "$GYRO_RUN/src/agention-llama-base" 93a24e52cc12687213c0432af41c7fe2ab1a0eda
gyro_build "$GYRO_RUN/src/strata-base" "$GYRO_RUN/build-sm87" "$GYRO_RUN/src/agention-llama-base"
export GYRO_RESULT="$GYRO_RUN/results/mtp-ab-001"
test ! -e "$GYRO_RESULT"
mkdir -p "$GYRO_RESULT"
uv run --no-project python - <<'PY'
import json, os
from pathlib import Path
protocol = json.loads((Path(os.environ["ORIN_SUPPORT"]) / "config/strata-gyro-mtp-benchmark-v1.json").read_text())
protocol["runtime_directory"] = os.environ["GYRO_RUN"]
(Path(os.environ["GYRO_RESULT"]) / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
PY
cp "$ORIN_SUPPORT/tools/strata-gyro-benchmark.py" "$GYRO_RESULT/strata-gyro-benchmark.py"
printf 'GYRO_MODEL_PATH=%s\nGYRO_MODEL_REVISION=%s\n' "$GYRO_MODEL" \
  e40c350d6a8974dc1b70cd46ec451a6e63561cb6 > "$GYRO_RUN/runtime.env"
```

The harness parses `runtime.env` as literal `KEY=value` lines, so write the expanded model path. It uses `build-sm87/strata` for its base binary receipt even when `mode_build` selects other binaries; keep that baseline build available. For a candidate, create another worktree from `orin-arm64-base`, apply **one** Strata candidate patch, commit it, and call `gyro_build` with the build-directory name required by the protocol and the GGML tree indicated above.

The machine-specific benchmark wrapper is not distributed. To invoke the unchanged harness directly, stop the foreground server first, verify the intended power profile, and record ASTOP telemetry before, during, and after the run:

```bash
astop agent-summary > "$GYRO_RESULT/system-before.txt"
nvpmodel -q > "$GYRO_RESULT/power-before.txt"
grep -q MAXN "$GYRO_RESULT/power-before.txt"
astop --record "$GYRO_RESULT/telemetry.ndjson" --interval 1s > "$GYRO_RESULT/telemetry.log" 2>&1 &
gyro_observer_pid=$!
timeout --signal=INT --kill-after=30s 110m uv run --offline --no-project --python "$GYRO_PY" \
  python "$GYRO_RESULT/strata-gyro-benchmark.py" "$GYRO_RUN" "$GYRO_RESULT" \
  > "$GYRO_RESULT/run.log" 2>&1 &
gyro_bench_pid=$!
astop watch start --name gyro-repro-mtp-ab-001 --pid "$gyro_bench_pid" --owner gyro-repro
astop wait gyro-repro-mtp-ab-001 --until exit --timeout 2h --json > "$GYRO_RESULT/observer-terminal.json" || true
gyro_bench_rc=0
wait "$gyro_bench_pid" || gyro_bench_rc=$?
printf '%s\n' "$gyro_bench_rc" > "$GYRO_RESULT/exit-code.txt"
kill -INT "$gyro_observer_pid" 2>/dev/null || true
wait "$gyro_observer_pid" || true
astop agent-summary > "$GYRO_RESULT/system-after.txt"
uv run --no-project python "$ORIN_SUPPORT/tools/summarize-strata-gyro-benchmark.py" "$GYRO_RESULT"
```

A timeout or an ASTOP `disappeared`/`unknown` event is not success. Preserve the terminal event, direct exit, and result files. After reviewing the saved event, ACK its `event_id` using `astop event ack EVENT_ID --consumer gyro-repro`, then remove only `gyro-repro-mtp-ab-001` with `astop watch stop`. If execution fails, retain the output directory and use a new directory for the next attempt.

The summarizer checks wrapper and engine exits, request counts, token lengths, zero prefix reuse, script/protocol/input hashes, and outputs. It reports output comparisons; inspect them rather than treating every successful summary as proof that every candidate's outputs match. Record actual mode arguments and binary hashes along with per-request prefill, TTFT, stream decode, and total time. GPU activity/clocks, memory pressure, swap, and temperature availability affect interpretation. Use an independent board source for rail power if ASTOP does not expose it, and label that source.

## 9. The 61-request replay and later deployments

Exact replay additionally requires the original `*-v1_chat_completions.json` capture files for requests 1–61. They are not in Git. Without them, the preserved summaries can be inspected but the 27.02 tok/s run cannot be rerun exactly. New captures must be labeled as a new workload. The historical capture also contains two tasks that searched for missing input files and stopped; it is not a successful coding-task suite.

With those request files available, a freshly started 64K server using the configuration in section 7, and a fresh output path, run the replay below. The comparison also requires `replay-reference.json`: a separately recorded run of the same 61 requests under the reference configuration. That file is not supplied by this repository.

```bash
uv run --no-project python "$ORIN_SUPPORT/tools/replay-strata-requests.py" \
  "$GYRO_RUN/capture/requests" http://127.0.0.1:18094 "$GYRO_RUN/replay-f16cache.json" 1 61
uv run --no-project python "$ORIN_SUPPORT/tools/compare-strata-replays.py" \
  "$GYRO_RUN/replay-reference.json" "$GYRO_RUN/replay-f16cache.json"
```

Run each comparison configuration from the same initial cache state and preserve the original request order. A fresh server begins empty; replay then intentionally reuses prefixes and conversation state. Confirm **61 common requests**, not just zero differences on a smaller intersection: the comparison script joins runs by their common request names. It compares content and tool calls while excluding random tool-call IDs. Aggregate decode is 1,000 × Σ predicted_n / Σ predicted_ms; wall time is the sum of request HTTP durations. Per-request rates are not averaged to obtain the headline.

The [262K validation](../orin/evidence/strata-gyro-262k-persistent-20261009-v1/README.md), [October 11 vision deployment](../orin/evidence/gyro-abliterated-vision-deploy-20261011-v1/README.md), and [October 15 re-encode update](../orin/evidence/gyro-abliterated-update-20261015-v1/README.md) have separate model/configuration contracts. Their short generation rates and synthetic recall checks cannot substitute for the 61-request result. Check those records when reproducing those deployments; this guide does not apply their service changes.
