# Checkpoint Spread and Gyro-L on Jetson AGX Orin

Measurements: October 19, 2026. Reference publication: October 20, 2026.

This report extends the [initial Gyro-M optimization work](OPTIMIZATION.md) on Jetson AGX Orin with two targeted additions:

1. **Prompt checkpoint spread:** an eviction policy that retains checkpoints at wider intervals farther back in a conversation which can reduce prompt recomputation when a request returns to an older shared prefix.
2. **Native Gyro-L support:** end-to-end execution paths for Gyro-L's pair-trellis quantized expert weights.

Both changes are selective backports onto the Strata engine reporting version 0.1.39 and preserve all previously established Orin optimizations; they do not represent an update to the complete current upstream codebase.

Three primary experimental tracks address distinct operational questions:

- **Returning to a shared parent (same-model A/B):** In one synthetic branch-and-return fixture using Gyro-M and the same executable, replacing least-recently-used (LRU) checkpoint eviction with the spread policy reduced the final return request's HTTP time to first token (TTFT) from **128.620 s to 1.639 s** (**78.47×** faster). All eight request hashes and assistant answers matched between modes.
- **Running Gyro-L within the 64 GB memory budget:** Using a text-only, INT8 KV-cache configuration with a 6,144 MiB VRAM reserve, Gyro-L passed all seven bounded API and long-context checks, including a prompt of **253,952 input tokens** within a 262,144-token context capacity. Although that configuration left room for 8,192 output tokens, the test capped generation at 512 tokens and did not evaluate 8,192-token output generation.
- **Comparing Gyro-M and Gyro-L under common engine settings:** Under a matched protocol (`chunk1024`, with dense FP16 weight caching disabled), Gyro-M generated tokens at **23.77–25.00 tok/s** and Gyro-L at **18.45–19.24 tok/s** across 2,048-, 25,000-, and 80,000-token inputs. These figures measure stream decode throughput rather than end-to-end request latency or output quality.

The [evidence index](../orin/evidence/gyro-l-spread-20261020-v1/README.md) links the selected numerical records, source bindings, and scope boundaries. Publication validation reconstructs the recorded source trees and verifies the exported artifacts; it adds no new GPU measurements.

---

## 1. Baseline Environment, Source Provenance, and Model Lineages

### Hardware and software baseline

All GPU measurements ran on a Jetson AGX Orin 64GB developer kit (CUDA compute capability SM87) operating in MAXN power mode. The recorded software stack was Ubuntu 24.04, Jetson Linux 39.2.1, JetPack 7.2.1, and CUDA Toolkit 13.2.2. The earlier [environment record](../orin/README.md#recorded-environment) provides the full baseline system context.

### Engine and GGML source provenance

Prior to this campaign, the deployed engine checkout on the Orin target contained staged optimization changes beyond its Git `HEAD`. Those changes were frozen before applying the checkpoint spread and Gyro-L extensions. The resulting baseline source tree (`24603390b080e9011aa02fc779b818a79ea8f9db`) is identical to the published `orin-f16cache-code` tree, anchoring both the earlier Gyro-M work and the current experiments to the same optimized starting point.

| Component | Pinned source and treatment |
|---|---|
| Initial complete publication | `orin-reference-20261018`, commit `b3b9424c4e9e298a0a7739d24dd6ec38c40c2841` |
| Spread policy | [Niko Strata `477ac13c`](https://github.com/Niko1221/Strata/tree/477ac13cbe8e582c20484871ce459d20d82df09d); helper and tests integrated with Orin checkpoint recycling |
| Pair-trellis reference | Agention GitLab [CPU `50f7a8a2`](https://gitlab.com/agentionai-group/llama.cpp/-/commit/50f7a8a2083605785f6cddbd7901b573312aead6) and [CUDA `e6ab9d50`](https://gitlab.com/agentionai-group/llama.cpp/-/commit/e6ab9d50b70baf354a86133ce31e7fddcc6c727b); selected implementation imported while retaining the Orin shared-LUT path |
| Measured unified Strata | Historical commit `a144885a4b165d6e8fb4834a37e008f970a575da`, tree `b07039e21728a3b974b12118a1c70ce3f9873274` |
| Published unified code | `orin-gyro-l-spread-code-20261020`; engine, build, packing, and test paths match the measured source |
| Measured GGML | Base `93a24e52cc12687213c0432af41c7fe2ab1a0eda` plus the shared-LUT and Gyro-L patches; resulting tree `61915afdbd3ce8d6d83560d470a6b1f77a688f3b` |

The public repository adds reports, evidence, and helpers to the measured engine checkout, so its whole-tree hash differs. [Source bindings](../orin/evidence/gyro-l-spread-20261020-v1/source-bindings.json) map the executables to the measured source trees; publication validation checks that the engine, build, packing, and test paths match.

### Model lineages

The experiments used two distinct model checkpoints:

| Model | Revision | Complete GGUF bytes | SHA256 |
|---|---|---:|---|
| Huihui abliterated Gyro-M | `aca61b2202b6e0a91a761236b65c151a68f88c16` | 91,965,774,560 | `9f31a8d1f449f67cfd3b7eb6d500f4ae37d95be6e3f83b8d4a1572646b99874e` |
| Original-lineage Gyro-L | `aa0a0c3e24f9e08716ae806d40489abc7db6237f` | 108,323,559,808 | `5428a5646eaeaa56cc00a892f92e7546aeadb255233503969ad25cf48c7f359b` |

See the pinned [M download manifest](../orin/config/gyro-l-spread/m-download.json), [L download manifest](../orin/config/gyro-l-spread/l-download.json), and [complete L hash verification](../orin/evidence/gyro-l-spread-20261020-v1/l-integrity.json). Although their tokenizer metadata matched, the two models come from different training/post-processing lineages and have different weights. Consequently, the M/L results represent an operational comparison between the tested models rather than an isolated comparison of quantization formats.

---

## 2. Prompt Checkpoint Spread Policy and Same-Model A/B

### Mechanism and integration scope

To resume generation from a shared prompt prefix without recomputing earlier tokens, the engine periodically saves prompt checkpoints. With a limited checkpoint budget, LRU favors recent positions while retaining the protected root checkpoint. When a client branches into several temporary sub-requests and then returns to a longer parent conversation, LRU can evict the checkpoints near the end of the parent, leaving a large uncached gap.

The spread policy selects eviction victims based on token-position spacing, retaining checkpoints at progressively wider distances back from the end of the conversation. Existing protections for the root and newest checkpoints, as well as Orin's checkpoint buffer recycling, remain intact.

The policy is enabled with `--prompt-cache-spread` and is off by default. While the imported [policy helper](../include/strata/program/conv_cache.hpp) supports pin and tail metadata and has accompanying tests for them, the Orin [integration patch](../orin/patches/strata-orin-spread-20261019.patch) passes no pin or tail metadata into the helper. This backport does not expose a request-level prefix-pin API, does not promise to hard-pin a main conversation, and makes no changes to Hermes request fields or client code.

### Same-model A/B: returning to a parent conversation after six reviews

The evaluation fixture modeled a **49,152-token** parent conversation (including **7,695 tokens of system-message content**), followed by six temporary review requests sharing a prefix with that parent, and finally a return request continuing the parent conversation (49,197 prompt tokens total). Both LRU and spread runs used Gyro-M and the exact same spread-capable executable, configured with six prompt checkpoints, whole-conversation parking disabled, `chunk1024`, and no dense FP16 weight cache.

| Return request | Prompt tokens | Cached tokens | Tokens read | Internal prompt time (s) | HTTP TTFT (s) | HTTP total (s) |
|---|---:|---:|---:|---:|---:|---:|
| LRU | 49,197 | 7,699 | 41,498 | 128.324 | 128.620 | 130.220 |
| Spread | 49,197 | 49,145 | 52 | 1.326 | 1.639 | 3.181 |

By retaining a checkpoint near the parent's end, spread reduced the return request's recomputation from **41,498 tokens to 52**.

For that final return request, the observed HTTP TTFT speedup was **78.47×** and the HTTP total-time speedup was **40.94×**. The supporting artifacts are the [LRU](../orin/evidence/gyro-l-spread-20261020-v1/cache-lru.json) and [spread](../orin/evidence/gyro-l-spread-20261020-v1/cache-spread.json) result records and the unchanged [fixture generator/client](../orin/tools/gyro-l-spread/bench-gyro-cache-branch.py). All eight request hashes and assistant answers matched across both modes.

**Scope boundary:** This test demonstrates checkpoint reuse in a single synthetic shared-prefix branching sequence. It does not establish the same speedup for independent conversations, arbitrary branch lifetimes, every Hermes review workflow, or token decode rates. Whole-conversation parking remained disabled throughout.

---

## 3. Native Gyro-L Support and Memory Budget on 64 GB Orin

### Pair-trellis expert quantization

Gyro-L encodes its mixture-of-experts (MoE) weights using three pair-trellis quantization formats: **TQP5, TQP6, and TQP7**, corresponding to GGML type IDs **147, 148, and 149**. Each block of 128 values occupies **42, 50, and 58 bytes**, respectively. The backport integrates these formats across the GGUF reader, weight packer, native CUDA decoder, matrix-vector decode path, and batched prefill path. Packed metadata references expert tensors directly in the original GGUF file, avoiding a separate 45 GiB copy of the expert weights on disk.

### Full-resident smoke test versus reserved-memory validation

An initial full-resident Gyro-L smoke test loaded all **24,576 experts** (about **45.12 GiB**) and generated a **5-token output** from a **32-token prompt**, with multi-token prediction (MTP) weights loaded but speculative draft generation disabled by a cap of 1 (`cap1`). After model load, the engine reported only **410 MiB** of free CUDA memory, while ASTOP recorded a minimum available memory of **0.3245 GiB**, a peak used memory of **61.080 GiB**, and `critical` memory pressure. Passing that minimal smoke test did not validate full-resident long-context operation with MTP `cap4`.

Before running long-input validation and performance benchmarks, the configuration added `--vram-reserve-mib 6144`. It retained a 262,144-token context capacity, INT8 KV cache, vision off, `chunk1024`, and owned prefill buffers, with dense FP16 weight caching and whole-conversation parking disabled. Because runtime free memory determines how many experts fit within the budget, actual cache admission must be recorded rather than inferred from the requested 24,576-expert target:

- The long-context validation process admitted **22,805 experts / 41.86 GiB**.
- The separate common-performance benchmark process admitted **23,106 / 42.41 GiB**.

### Unified DRAM, expert residency, and KV configuration

Although Jetson AGX Orin's CPU and GPU share physical DRAM, this engine allocates a dedicated CUDA expert cache and copies host weights into it. Shrinking that CUDA cache via `--vram-reserve-mib 6144` leaves unified memory available for KV cache and prefill buffers; it does not create a separate physical pool of CPU RAM.

In the fixed-profile decode path, expert lookups that miss the GPU cache can be routed to the CPU worker pool. The 6,144 MiB reserve was selected strictly to maintain memory headroom for long contexts, not because a CPU-versus-GPU speed comparison showed a throughput advantage. Note also that `kv_resident=0` in the evidence records denotes the default full-GPU KV cache configuration, not CPU KV offload.

---

## 4. Matched Gyro-M/Gyro-L Performance and Gyro-M Regression Check

### Common benchmark protocol

The [common protocol](../orin/config/gyro-l-spread/common-performance.json) held the following engine and workload parameters constant across both models:

- Concurrency and context: one request at a time; a context capacity of 262144; INT8 KV cache
- Prefill and caching: prefill `chunk1024` with `--no-prefill-borrow`; dense FP16 weight cache disabled; prompt/prefix cache disabled; parking disabled; short-read disabled
- Speculative decoding and workers: suffix drafting disabled; MTP verifier `cap4/max-t4/confidence0.5`; two CPU pool workers; temperature 0
- Expert cache budget: requested expert count 24576 and reserve 6144 MiB

At each input length (2,048, 25,000, and 80,000 tokens), the harness ran one warmup request followed by three measured requests, each generating exactly 512 tokens. For every run, the harness verified identical input token IDs between modes, clean tokenizer round trips, matching input hashes, zero reused prefix tokens, and a final prefill position equal to the input length minus one. No natural end-of-sequence (EOS) token occurred during the measured runs. Under the 6,144 MiB reserve, Gyro-M admitted all **24,576 experts / 29.88 GiB**, whereas Gyro-L admitted **23,106 / 42.41 GiB**.

| Model | Input tokens | Internal prefill (tok/s) | IPC TTFT (s) | Stream decode (tok/s) | IPC total (s) |
|---|---:|---:|---:|---:|---:|
| M | 2,048 | 342.36 | 6.039 | 25.00 | 26.484 |
| L | 2,048 | 224.61 | 9.196 | 19.24 | 35.751 |
| M | 25,000 | 333.93 | 74.943 | 24.59 | 95.730 |
| L | 25,000 | 185.56 | 134.814 | 18.67 | 162.186 |
| M | 80,000 | 324.73 | 246.475 | 23.77 | 267.970 |
| L | 80,000 | 180.67 | 442.915 | 18.45 | 470.635 |

Table values are three-run means. [Common results](../orin/evidence/gyro-l-spread-20261020-v1/common-performance-v4.json) stores the means and engine settings; [summary.json](../orin/evidence/gyro-l-spread-20261020-v1/summary.json) provides standard deviations, ranges, internal decode timers, and per-request ASTOP, CPU-core, and process statistics; [comparisons.json](../orin/evidence/gyro-l-spread-20261020-v1/comparisons.json) records the cross-model ratios.

Under these matched settings, Gyro-M's internal prefill rate was **1.524–1.800×** Gyro-L's, and its stream decode rate was **1.289–1.317×** Gyro-L's. This experiment measures overall operational speed under shared settings; it does not isolate the individual contributions of quantization kernels, weight supply, and actual expert residency, nor does it rank model answer quality.

### Did Gyro-L support regress Gyro-M?

The [bounded M regression check](../orin/evidence/gyro-l-spread-20261020-v1/m-2k-regression-v4.json) compared the spread-only and unified executables on the 2,048-token workload. Both builds generated the exact same 512 output token IDs for the fresh-process warmup request and all three measured 2K requests. Mean stream decode rates were **24.989 and 24.996 tok/s**, and mean internal prefill rates were **341.738 and 342.361 tok/s**, respectively. No regression was observed in this 2K test. (Bitwise output equality was required only between the two Gyro-M builds, not between the distinct Gyro-M and Gyro-L models.)

Note that the common M/L table intentionally runs Gyro-M at `chunk1024` with dense FP16 weight caching disabled to match Gyro-L's memory-constrained settings. It does not represent Gyro-M's restored production configuration (`chunk8192` with automatic dense FP16 weight caching). Likewise, the [older 32K chunk sweep](../orin/evidence/strata-gyro-prefill-chunk-own-20261007-v2/README.md) is a separate experiment at a 32K context capacity and must not be mixed with this 262K table.

### How to interpret timers and telemetry counters

The [counter definitions](../orin/evidence/gyro-l-spread-20261020-v1/counter-definitions.json) map each reported metric to its exact source location:

- **Internal prefill (`tok/s`):** Computed as `(input_tokens - 1) / prompt_seconds`. The internal timer spans state reset and target/draft prompt processing; the final input token is consumed during the first speculative verification window.
- **IPC TTFT and IPC total (`s`):** Measured across the inter-process communication (IPC) boundary from immediately before serializing and sending the `GEN` command until receipt of the first token (TTFT) or the terminal `DONE` message (total). These timers exclude HTTP transport and initial model loading, and IPC TTFT is not a pure prefill-kernel timer.
- **Stream decode (`tok/s`):** Computed as `511 / (last_token_received - first_token_received)`. Because speculative decoding can emit multiple accepted tokens in a single step, tokens can arrive in bursts. A separate internal decode rate in `summary.json` is computed as `512 / decode_seconds` and includes the first verification window plus completion bookkeeping.
- **CPU routing fraction:** In the fixed-profile decode graph, CPU-routed entries equal total expert lookups minus GPU cache hits, including target-model work for speculative positions that are subsequently rejected. They count lookup events rather than unique experts or worker jobs. For Gyro-L, CPU-routed fractions were **0.0986%, 0.0845%, and 0.0856%** at 2K, 25K, and 80K input tokens, respectively. Because Gyro-M's full-GPU fast path bypasses host lookup counters entirely, its zero denominator leaves the ratio `null`. Decode CPU-routing percentages apply only to token generation and cannot explain prefill latency, which uses a separate batched/streaming execution path.
- **Outside-cache GPU offload:** The `DONE` message's `offloaded` counter tracks GPU execution through direct/peer paths rather than CPU fallback; these counters were zero in all common benchmark runs. Similarly, logical file-read counters track engine read calls rather than physical NVMe SSD or PCIe bus traffic.

---

## 5. Bounded Long-Context and API Validation

Running under the reserved-memory configuration (`l-on-v4`), Gyro-L passed all **7/7 checks** in the API validation suite:

1. Short Korean text generation
2. Three long-context dispersed-identifier recall requests (at 64,000, 128,000, and 253,952 input tokens)
3. Structured tool call generation
4. Tool-result continuation
5. English Server-Sent Events (SSE) streaming with usage metadata and terminal `DONE` receipt

The three long-context requests used independent synthetic inputs rather than progressively cached conversation turns.

| Input tokens, verified from response usage | Beginning/middle/end identifiers | HTTP total (s) |
|---:|---|---:|
| 64,000 | All three recovered | 370.482 |
| 128,000 | All three recovered | 739.135 |
| 253,952 | All three recovered | 1,491.998 |

At **253,952** input tokens within the 262,144-token context window, **8,192 tokens of capacity** remained for generation. However, the test request set `max_tokens` to **512**, and the model finished naturally before hitting that limit. This validates long-input prefill and a short response, without establishing 8,192-token output generation, comprehensive long-context quality, or multi-user concurrency. Whole-conversation parking restore was explicitly skipped. See the [API results](../orin/evidence/gyro-l-spread-20261020-v1/l-context-result.json) for per-check records.

Across Gyro-L startup, the three long-context requests, the remaining API checks, and process cleanup, ASTOP recorded:

- **Peak used memory:** **62,273,916,928 bytes (about 57.997 GiB)**
- **Minimum headroom (total minus used):** **3,658,629,120 bytes (3.407 GiB)**
- **Peak GPU clock and temperature:** **1,300.5 MHz** and **59.218°C**
- **Peak swap occupancy and memory pressure:** **1,530,839,040 bytes**, with pressure states remaining within `normal/warning`

Swap occupancy alone does not establish ongoing swap I/O or thrashing. The [Telemetry summary](../orin/evidence/gyro-l-spread-20261020-v1/l-context-telemetry.json) preserves unavailable power and per-process GPU fields as `null`.

---

## 6. Prefill Chunk Candidate (`chunk2048`) and Campaign Validation Limits

### Candidate check: `chunk2048` on 2K inputs

A separate Gyro-L process evaluated `chunk2048` against `chunk1024` under the same 2,048-token input and 512-token output protocol (one warmup followed by three measured runs).

| L configuration | Actual expert slots | Prefill (tok/s) | IPC TTFT (s) | Stream decode (tok/s) | IPC total (s) |
|---|---:|---:|---:|---:|---:|
| chunk1024 | 23,106 | 224.61 | 9.196 | 19.24 | 35.751 |
| chunk2048 | 22,858 | 282.92 | 7.324 | 18.98 | 34.242 |

Chunk2048 improved internal prefill throughput by **1.260×** and reduced total IPC time by **1.044×**, but admitted fewer GPU-resident experts. Owned prefill buffers compete with expert residency for memory, so larger buffers can increase decode cache misses. The allocation precheck uses `Prefill::bytes_needed` plus 512 MiB of headroom. Faster prefill at 2K does not by itself establish a better overall configuration.

Output hashes differed between `chunk1024` and `chunk2048`, and Gyro-L outputs were not bit-identical across the three repeats of a single setting either. Neither logit/state equivalence nor equal answer quality was established. Chunk2048 remains a bounded 2K candidate check, while chunk1024 is the configuration validated across the long-context suite. Chunk1024 is not a hard model limit; chunks4096/8192 were not tested for Gyro-L. See the [chunk protocol](../orin/config/gyro-l-spread/chunk2048.json) and [comparison record](../orin/evidence/gyro-l-spread-20261020-v1/comparisons.json).

### Verification boundaries and campaign findings

- **MTP guards in draft-off versus draft-on smoke tests:** Initial attempts to run with `--spec 0` or without MTP weights were rejected by native pack and server validation guards. The successful draft-off smoke test therefore kept MTP weights loaded and disabled drafting via `cap1/suffix0`, recording `offered/accepted0/0` (0 offered, 0 accepted). The draft-on smoke test used `cap4` and observed `3/3` accepted draft tokens on a five-token arithmetic response. Note that earlier `off_scope` wording in the frozen protocol referring to “MTP removed” describes the rejected initial candidate rather than the final draft-off smoke test.
- **MMQ parity stream-ordering fix:** An initial quantized matrix-multiplication (MMQ) control mismatch also reproduced on the pre-extension baseline GGML tree. Inspection showed that the test's non-blocking CUDA stream lacked ordering guarantees for its host-to-device input copy. Placing input copies, sentinel setup, and kernel compute onto the same CUDA stream fixed the race condition without relaxing numerical tolerances. The final parity suite passed all **78 matrix products**: six `Q2_0` controls plus `M/TQP5/TQP6/TQP7` formats across three routed batch patterns, `gate-up/down` projections, and three repeats. See the [parity log](../orin/evidence/gyro-l-spread-20261020-v1/mmq-parity.log).
- **Compute Sanitizer unavailability:** NVIDIA Compute Sanitizer could not run because GPU debugging features were disabled on the target Orin system; no sanitizer pass is claimed. Arithmetic verification relies on direct CPU/GGML reference comparisons, native expert parity checks at layers 0/16/32, and Gyro-L Hadamard transform checks.
- **Strict run-acceptance criteria:** Early in the campaign, some interrupted worker scripts recorded a raw exit status of `0` (`exit0`) via a shell `EXIT` trap. Those runs were rejected. Final acceptance required a completed result JSON, direct native binary exit statuses, worker exit statuses, and verified sequence/service completion. Where ASTOP reported a process exit disposition as `disappeared` or `unknown`, that status was preserved as unknown rather than treated as success.

---

## 7. Reproduce This Snapshot

Run these steps on an idle Jetson AGX Orin 64GB with the recorded toolchain and enough disk space for both GGUF models, MTP tensors, packed artifacts, build trees, and output receipts. The commands create an isolated workspace under `$HOME/gyro-l-spread-repro` and do not modify an active system service. Because the model and API tests need sufficient available memory, run the two models and the A/B server configurations sequentially.

### 1. Source checkout and build

Clone this repository, check out the `orin-reference-20261020` tag, and ensure a Git user name and email are configured for the local GGML patch commits. (The earlier Gyro-M baseline campaign remains independently reproducible from `orin-reference-20261018` via [REPRODUCING.md](REPRODUCING.md).)

```bash
set -euo pipefail
git checkout orin-reference-20261020
export REPO="$(pwd)"
export RUN="$HOME/gyro-l-spread-repro"
export RUNTIME="$RUN"
export SUPPORT="$REPO/orin"
export CLIENTS="$SUPPORT/tools/gyro-l-spread"
export CFG="$SUPPORT/config/gyro-l-spread"
export GGML="$RUN/src/ggml-l"
export PY="$RUN/venv/bin/python"
test ! -e "$RUN"
mkdir -p "$RUN/src" "$RUN/results" "$RUN/logs" "$RUN/configs"
ln -s "$REPO" "$RUN/src/strata"
git diff --exit-code orin-gyro-l-spread-code-20261020 HEAD -- CMakeLists.txt setup.py include src tests tools

git init "$GGML"
git -C "$GGML" remote add origin https://github.com/agentionai/llama.cpp.git
git -C "$GGML" fetch --depth 1 origin 93a24e52cc12687213c0432af41c7fe2ab1a0eda
git -C "$GGML" checkout --detach FETCH_HEAD
git -C "$GGML" apply --index "$SUPPORT/patches/agention-llama-tq-mmq-shared-lut.patch"
test "$(git -C "$GGML" write-tree)" = 01f81d2075654714d00ee5413cfbec2b7887388f
git -C "$GGML" commit -m 'Apply the measured Orin shared-LUT baseline'
git -C "$GGML" worktree add --detach "$RUN/src/ggml-baseline" HEAD
git -C "$GGML" apply --index "$SUPPORT/patches/agention-ggml-gyro-l-20261019.patch"
test "$(git -C "$GGML" write-tree)" = 61915afdbd3ce8d6d83560d470a6b1f77a688f3b
git -C "$GGML" commit -m 'Apply the measured Gyro-L pair-trellis support'

command -v astop
astop --help-agent
astop agent-summary
nvpmodel -q
nvcc --version
bash "$SUPPORT/tools/build.sh" "$REPO" "$RUN/build-l" "$GGML"
"$RUN/build-l/conv_cache_test"
```

For the spread A/B comparison and the Gyro-M regression baseline, also build the published spread-only commit (`9ecdbabb204867ae8fb272f9f92f6d4a1141a1e2`) against the baseline GGML worktree. Since that commit already includes the frozen Orin baseline, do not re-apply the historical cumulative `f16cache` patches.

```bash
git -C "$REPO" worktree add --detach "$RUN/src/spread" 9ecdbabb204867ae8fb272f9f92f6d4a1141a1e2
bash "$SUPPORT/tools/build.sh" "$RUN/src/spread" "$RUN/build-spread" "$RUN/src/ggml-baseline"
```

### 2. Model files, packs, MTP setup, and arithmetic parity checks

The model prefetch tool verifies full file byte counts and SHA256 digests against the pinned manifests. If you reuse pre-downloaded GGUF files on disk, point `MODEL_M` and `MODEL_L` to their paths and verify them against the same manifests. Do not use the original-lineage Gyro-M checkpoint from the earlier reproduction guide, as this campaign's M/L comparison uses the Huihui abliterated Gyro-M checkpoint.

```bash
uv venv --python /usr/bin/python3 --no-python-downloads "$RUN/venv"
uv pip install --python "$PY" -r "$REPO/requirements.txt"
for model in m l; do
  uv run --offline --no-project python "$CLIENTS/render-config.py" "$CFG/$model-download.json" "$RUN/$model-download.json"
  uv run --offline --no-project python "$SUPPORT/tools/prefetch-gyro-model.py" "$RUN/$model-download.json" "$RUN/download-$model"
done
export MODEL_M="$RUN/hf-cache/hub/models--agentionai--Qwen3.8-Flash-Next-Gyro-abliterated-Huihui-GGUF/snapshots/aca61b2202b6e0a91a761236b65c151a68f88c16/Qwen3.8-Flash-Next-Gyro-M-abliterated-TQ2_0.gguf"
export MODEL_L="$RUN/hf-cache/hub/models--agentionai--Qwen3.8-Flash-Next-Gyro-GGUF/snapshots/aa0a0c3e24f9e08716ae806d40489abc7db6237f/Qwen3.8-Flash-Next-Gyro-L-TQ3_0.gguf"
export STRATA_GGUF_PY="$GGML/gguf-py"
uv run --offline --no-project --python "$PY" python "$REPO/tools/iq_pack.py" --gguf "$MODEL_M" --out "$RUN/packs/gyro-m" --compat-bf16
uv run --offline --no-project --python "$PY" python "$REPO/tools/iq_pack.py" --gguf "$MODEL_L" --out "$RUN/packs/gyro-l" --compat-bf16

export STRATA_MTP_REVISION=de4b8e4d43b917e7706784d8bb445c9af86a3540
uv run --offline --no-project --python "$PY" python "$REPO/tools/mtp_fetch.py" fetch --out "$RUN/mtp"
uv run --offline --no-project --python "$PY" python "$REPO/tools/mtp_fetch.py" verify --out "$RUN/mtp"
uv run --offline --no-project --python "$PY" python "$REPO/tools/mtp_pack.py" --src "$RUN/mtp" --experts q2_0 --out "$RUN/mtp/mtp-q2_0.gguf"
uv run --offline --no-project --python "$PY" python "$REPO/tools/mtp_rt.py" --gguf "$RUN/mtp/mtp-q2_0.gguf" --out "$RUN/mtp/rt"
export MTP="$RUN/mtp/rt-enko"
mkdir "$MTP"
for name in dense.bin dense.txt experts.bin; do ln -s "../rt/$name" "$MTP/$name"; done
cp "$SUPPORT/evidence/strata-gyro-agent-round2-20261008-v1/draft_vocab_enko.bin" "$MTP/draft_vocab.bin"
printf '%s  %s\n' 8de2bec33e6a866311c70ec35ca58eec8048c9645b49ee94ef5bce523055174f "$MTP/draft_vocab.bin" | sha256sum -c -
```

Verify that the generated MTP inventory's `repo` URL contains `/resolve/de4b8e4d43b917e7706784d8bb445c9af86a3540/`, totaling 31 tensors and 5,214,301,696 bytes; the fetch script can fall back if an upstream revision disappears, which would not match the pinned input. Note that `uv --offline` restricts Python package resolution rather than HTTP requests made by the model download scripts. Both Gyro-M and Gyro-L use the same 47,493-token `enko` draft vocabulary.

Before running benchmarks, execute the native arithmetic parity checks and record both their logs and direct exit statuses:

```bash
"$RUN/build-l/native_expert_parity" --load-check "$RUN/packs/gyro-l" "$MODEL_L"
"$RUN/build-l/native_expert_parity" "$MODEL_M" 0 16 32
"$RUN/build-l/native_expert_parity" "$MODEL_L" 0 16 32
"$RUN/build-l/native_hadamard_test" --pack "$RUN/packs/gyro-l" --gguf "$MODEL_L" 0 0 16 17 32 31
MMQ_PARITY_REPEAT=3 "$RUN/build-l/prefill_tqp_parity"
```

### 3. API configuration rendering and checkpoint spread A/B

The configuration templates under `$CFG` contain literal path placeholders that must be rendered before starting the server (`render-config.py` refuses to overwrite existing files):

- `m-lru` and `m-spread` target the spread-only Gyro-M build (`$RUN/build-spread`).
- `l-on-v4` is the `reserve 6144`/`cap4` long-context configuration for Gyro-L, and `l-off-v4` is the `cap1` draft-off smoke configuration.
- `m-production` and `l-production` capture the end-of-campaign service templates and are distinct from the fixed-performance benchmark protocol.

```bash
for mode in m-lru m-spread m-unified l-on-v4 l-off-v4 m-production l-production; do
  uv run --offline --no-project python "$CLIENTS/render-config.py" "$CFG/$mode.json" "$RUN/configs/$mode.json"
done
cp "$RUN/configs/m-spread.json" "$RUN/server.json"
uv run --offline --no-project --python "$PY" python "$CLIENTS/bench-gyro-cache-branch.py" freeze --runtime "$RUN" --fixture "$RUN/cache-branch.json"
export STRATA_APR=1
cd "$REPO"
uv run --offline --no-project --python "$PY" python -m serve.server --engine strata --config "$RUN/configs/m-lru.json" --host 127.0.0.1 --port 18094
```

The server runs in the foreground. In a second shell with the same environment variables exported, poll `/health` until it reports `loaded: true`, then run the LRU benchmark client:

```bash
curl --fail --silent http://127.0.0.1:18094/health
uv run --offline --no-project --python "$PY" python "$CLIENTS/bench-gyro-cache-branch.py" run --fixture "$RUN/cache-branch.json" --out "$RUN/results/cache-lru.json" --url http://127.0.0.1:18094 --label lru
```

Once the LRU run finishes, stop the foreground server cleanly, launch a fresh server instance with `$RUN/configs/m-spread.json`, wait for `/health` to report `loaded: true`, and re-run the client with `--out "$RUN/results/cache-spread.json"` and `--label spread`. Never run both server instances concurrently or overwrite the first result file. Confirm `status=completed`, verify that all eight request hashes and assistant answers match between `cache-lru.json` and `cache-spread.json`, and check the cached/read token counts on the final return request (server readiness alone is not completion evidence).

### 4. Gyro-L long-context and API validation suite

The preserved context validation client checks two historical path names for metadata provenance. The symlink and config copy below route those paths to the current unified build without substituting the earlier binary. Freeze the deterministic test inputs once, start a fresh foreground server using `$RUN/configs/l-on-v4.json` on port 18094, and invoke the validation client from a second shell:

```bash
mkdir "$RUN/build-sm87-f16cache"
ln -s ../build-l/strata "$RUN/build-sm87-f16cache/strata.f16cache"
cp "$RUN/configs/l-on-v4.json" "$RUN/server-262k-validation.json"
uv run --offline --no-project --python "$PY" python "$CLIENTS/validate-gyro-context.py" freeze "$RUN" "$RUN/context"
```

```bash
uv run --offline --no-project --python "$PY" python "$CLIENTS/validate-gyro-context.py" run "$RUN" "$RUN/context" --url http://127.0.0.1:18094 --model-id qwen38-flash-next-gyro-l --no-parking-check
```

Verify that the output JSON reports `status=completed`, all seven checks pass, and the response usage fields confirm actual long-prompt input token counts of 64,000, 128,000, and 253,952. Always inspect the saved manifest and per-response usage; a server's declared context capacity alone does not prove a completed long-context request.

### 5. Fixed-input Gyro-M/Gyro-L and chunk benchmarks

Stop the HTTP test server before starting the IPC benchmarks, as `strata-gyro-benchmark.py` spawns and manages its own native worker processes sequentially. (The harness's frozen synthetic prompts contain Korean text, which is kept unchanged to preserve the exact input token contract; all report text and instructions remain in English.)

```bash
printf 'GYRO_MODEL_PATH=%s\n' "$MODEL_M" > "$RUN/runtime.env"
mkdir "$RUN/perf" "$RUN/perf-2048"
uv run --offline --no-project python "$CLIENTS/render-config.py" "$CFG/common-performance.json" "$RUN/perf/protocol.json"
uv run --offline --no-project python "$CLIENTS/render-config.py" "$CFG/chunk2048.json" "$RUN/perf-2048/protocol.json"
uv run --offline --no-project --python "$PY" python "$CLIENTS/strata-gyro-benchmark.py" "$RUN" "$RUN/perf"
uv run --offline --no-project --python "$PY" python "$CLIENTS/strata-gyro-benchmark.py" "$RUN" "$RUN/perf-2048"
```

The common performance protocol runs 28 requests in total (covering warmups, the Gyro-M baseline regression check, and the 2K/25K/80K M and L runs), while the `chunk2048` candidate protocol runs four requests. For every run, verify a completed result JSON, a direct native exit status of 0 for each mode, exactly 512 output tokens, zero reused input tokens, and matching prompt hashes. Record natural-EOS positions as well; none occurred in the published measured runs. The harness refuses to reuse its existing prompt directory, so use a fresh result directory for any repeat run.

Throughout execution, use the installed ASTOP utilities to capture CPU/core utilization, memory and swap usage, GPU clock and temperature, and native process lifecycle events before, during, and after each run. For long-running native processes, register the exact PID under a unique named watch, use server-side `wait`, persist the terminal event, and acknowledge that event only after validating the benchmark output file. Always record direct process exit codes alongside observer telemetry: an observer disposition of `disappeared` or `unknown` does not establish a successful run. Confirm MAXN mode with `nvpmodel -q` (the reproduction scripts do not alter the board's power mode).

---

## 8. Scope Boundaries and Post-Campaign Service State

These measurements remain bounded by the tested protocols and do not establish:

- Broad conversational or reasoning quality, or equal answer quality between Gyro-M and Gyro-L
- Full-resident Gyro-L stability at the 262,144-token context limit
- 8,192-token output generation
- Concurrent multi-user throughput
- Checkpoint retention across arbitrary conversation branch lifetimes or independent sessions
- Any throughput advantage from CPU expert routing
- Long-context validation for the `chunk2048` candidate, or Compute Sanitizer coverage on the tested Orin system

At the conclusion of the October 19 campaign, the local service was restored to **Gyro-M plus checkpoint spread**, running with `chunk8192` and the previously established automatic dense FP16 weight-cache settings. While native Gyro-L support and its bounded validation configurations remain available in the unified engine, the restored Gyro-M production deployment and the common M/L benchmark table represent distinct configurations. The initial Gyro-M optimization report and reproduction contract remain archived under `orin-reference-20261018`, while this document and `orin-reference-20261020` provide the source, model, and workload contract for the checkpoint spread and Gyro-L extensions.
