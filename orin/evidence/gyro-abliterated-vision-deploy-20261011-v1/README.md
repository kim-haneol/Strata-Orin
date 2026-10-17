# Abliterated Gyro-M plus CUDA vision, 262K INT8

Status: completed and deployed. User authorized a new pack and
vision binary, tests, and deployment after successful tests. Preserve 262144
context, INT8 KV, and the existing strataorin profile.

## Frozen inputs and preparation

- HF revision `bc755182a9171dd3ef082e8cd42d10e7ed6b23fb`.
- Full model SHA256 `9a43d2978584c02d9512bfe7bacf5fdebb2b05f8a26fcc04a89ed6138f0e7b28`.
- mmproj SHA256 `f456cd796fdbdef0cadb22710b54e0071b6cdf22c07963365baa898267aec517`.
- Both files were reconstructed/downloaded and fully verified in the preceding
  stage; see `../gyro-abliterated-vision-stage-20261011-v1/README.md`.
- New standalone pack via `iq_pack.py --compat-bf16`; native expert/PLE tensor
  bytes stay in the new GGUF. Do not reuse the original dense pack.
- Reuse existing f16cache text engine and `mtp/rt-enko`. Build `strata-vision`
  with CUDA SM87, Release, Ninja, six build workers, against agention-llama
  `93a24e52cc12687213c0432af41c7fe2ab1a0eda` and Strata `bb98d095...`.
- Keep expert slots 24576, prefill 8192, context 262144, KV int8 and 4096 MiB
  conversation cache. CUDA encoder cap 1024 image tokens; warm-up reserves its
  working memory before the text engine sizes its allocations.
- The text engine also requires `--vision` to enable GENI requests and its
  M-RoPE table. The Python `vision` block alone does not enable this engine flag.
- Preserve API alias `qwen38-flash-next-gyro-m` and both existing loopback
  tunnel ports 22868/18092. Model variant is identified by paths and hashes.

## Validation contract

1. Freeze synthetic PNGs and requests before GPU tests. Test OCR and distinct
   shapes/colors on two different images, image reuse, and a two-image request.
2. Same abliterated pack/engine/settings and identical requests with vision OFF
   and ON: one warm-up, three cold-prefix requests, 128 output tokens each.
   Record output identity, prompt/cache counts, internal timings and HTTP time.
   This bounds text overhead for these requests, not every workload.
3. Reuse immutable synthetic recall inputs of exactly 64000, 128000 and 253952
   rendered tokens, with a new config/model-bound manifest. Three dispersed
   identifiers, tool call/result, conversation restore, SSE with usage: 11 checks.
4. Combine the largest recall text with an image; require all three identifiers,
   image code and actual prompt tokens above 253952 but within 262144 capacity.
   This is not an 8192-output-token or broad semantic-quality test.
5. Actual research Hermes image-tool/file fixture through host port 22868.
   No messaging tools or external chat messages.
6. ASTOP raw recordings before/during/after build and validation, named watches,
   server-side waits, direct wrapper exit receipts and retained systemd results.
   ASTOP disappeared/unknown is not completion authority.
7. Promote only a passing candidate. Preserve the original 262K config backup,
   restart persistent service, test actual generation/images, verify enabled and
   active services and preserved strataorin hash.

Remote run directory:
`${TEST_HOME}/.local/share/jetson-orin/gyro-vision-deploy-20261011`.
Raw telemetry, requests, models and builds stay outside Git; summaries, receipts,
scripts and final deployment config are committed.

The initial two-worker build ran beside the original serving model. At 95/284
build steps, the original API was stopped, its config backed up, and the build
was deliberately interrupted to resume with six workers and sufficient memory.
Completed Ninja objects were retained. This interrupted attempt is not a build
success; its separate unit/ASTOP receipts are retained in the run directory.

The first ON candidate omitted the engine's `--vision` flag. The encoder started
and `/health` reported images enabled, but the first image request returned HTTP
400: `this engine was started without --vision`. No image quality result exists
for that attempt. The failed unit exited 1; its error/config/benchmark receipts
are preserved. The flag was added, the candidate manifest regenerated, and the
ON benchmark and all image/context tests restarted under a separate v2 unit.

## Completed preparation and bounded text comparison

The new dense pack SHA256 is `e9588a855a18984a01c4202eaa295fb7211b341d49c440b0f203b5c310048954`.
The CUDA SM87 vision executable SHA256 is
`711998650c4c7ec032c7e655d32594718fe95c7ba9ded891640d0ca99c2b4aaa`.
Both upstream source checkouts used to build the encoder were clean. The resumed
build and standalone encoder test have retained `CLD_EXITED` / status 0 receipts.

Standalone encoding produced finite, nonzero FP32 embeddings of width 2560:
640x480 fixtures were 300 tokens (20x15), 262/275 ms; the 2048x2048 fixture was
1024 tokens (32x32), 1753 ms. Those times exclude text-model inference.
During that standalone interval, native ASTOP recorded 99.9% GPU utilization
while the text API was stopped (`encoder-gpu-evidence.json`).

The final comparison includes the engine's `--vision` flag and the resident CUDA
encoder. Same abliterated weights/pack, 3414 input tokens, 128 output tokens,
one warm-up and three measured cold-prefix requests per mode (`cache_n=0`).
All output strings and usage counts matched, including the warm-up.

| Mode | Median HTTP seconds | Median prefill seconds | Median decode tok/s |
|---|---:|---:|---:|
| Vision OFF | 11.026 | 7.428 | 36.2 |
| Vision ON | 10.978 | 7.377 | 36.1 |

HTTP change is -0.44%; there is no meaningful slowdown in this bounded comparison.
It does not establish overhead for every prompt or agent workload, nor compare
original and abliterated model quality/performance. The integer-list output is
especially easy for the unchanged MTP drafter.

The five basic image API checks passed: distinct OCR/color/shape answers for A
and B, A after B, the large 1024-image-token fixture, and both images together.
HTTP seconds were 3.99, 4.05, 1.97, 6.91 and 3.17 respectively.

## Context and API validation

All 11 context/API checks passed with the final candidate, including exact
rendered input counts and 3/3 dispersed-identifier recovery. All three long text
requests were cold (`cache_n=0`). HTTP seconds were 114.09 at 64000 tokens,
235.56 at 128000, and 497.96 at 253952. Tool calling and continuation, A/B/A/B
conversation restoration with 4937 cached tokens, and complete SSE with usage
also passed. See `context-result.json` for raw synthetic answers and timings.

## Persistent deployment and rollback

Promotion uses `scripts/deploy-gyro-vision.sh` only after core validation and the
actual research Hermes fixture pass. It preserves the original config at
`strata-gyro-orin-20261007/server.json.before-abliterated-vision-20261011` and the
Jetson user service at
`~/.config/systemd/user/jetson-gyro-api.service.before-abliterated-vision-20261011`.
It writes the tested config atomically, enables/starts the persistent service,
then probes real text generation. A failed promotion automatically restores the
previous 262K config and unit and restarts the previous API.

Research's YAML change is only `supports_vision: true` under
`providers.strata-local.models.qwen38-flash-next-gyro-m`; comments and unrelated
settings are preserved and a complete parsed-YAML comparison refuses other
changes. Backup: `${HERMES_HOME}/config.yaml.before-gyro-vision-20261011`.
This enables native image content in the main model's user/tool messages instead
of relying on inference from an unknown custom model name.

Manual rollback: stop `jetson-gyro-api.service`, restore the two Jetson backups
above to `server.json` and the original `.service` path, run
`systemctl --user daemon-reload`, then start the API. As `research`, restore its
YAML backup over `.hermes/config.yaml` and restart `hermes-gateway.service`.
The old model, old pack and strataorin files are retained. The host USB-NCM tunnel
and loopback ports 22868/18092 remain unchanged.

## Joint context/vision, Hermes and final state

The joint request passed with **255001 actual input tokens**, including 1024
image tokens: 3/3 identifiers plus exact image code `ORIN 7319`, 62 output tokens,
500.52 HTTP seconds, 497.33 internal prefill seconds, `cache_n=0`. The entire
validation wrapper exited 0 (`vision-unit.txt`); all 11 API/context and 6 image
checks passed. This is a short answer at near-maximum input, not a maximum-output
or general vision/long-context quality benchmark.

Research's actual default Hermes used native image tool content, read the
disposable input, and wrote the correct image fields, sum 89 and marker
`ORIN_VISUAL_20261011`. Its 6 API calls generated 1187 output tokens and reused
31542 cached tokens. The direct fixture exit and retained unit status were 0.
The YAML config hash is `e81a34a667a6e220426b1fedbe7d992fff98f53c0e922768cac75a6701752916`.
The gateway was restarted and is enabled/active, PID 2394138. No chat message
was sent by this validation.

Promotion exited 0, then a real image request through host port 22868 again
passed all OCR/color/shape fields (`deployed-image-22868.json`). Both host ports
report loaded=true, images=true, context=262144. Production
`jetson-gyro-api.service` is enabled/active with `Restart=always`, PID 471471;
its exact tested config SHA256 is
`a76a4752fbe8a6a8cc55401c378f02c224c13f0231fe218eeb3a20078b84aba5`.
The main engine is still f16cache + enko, with INT8 KV and 24576 expert slots.
The host tunnel's PID/state/restart count is unchanged. Jetson, haneol and
research user lingering are all enabled. Strataorin's original config hash
matches byte for byte. No model, profile or replay data was deleted.

ASTOP recorded the full operation natively at 2-second intervals. The final v2
validation/deployment interval contains 1079 snapshots: peak used RAM 57.055 GiB
of 61.404 GiB, minimum total-minus-used headroom 4.349 GiB. OS free RAM separately
reached 605286400 bytes; the headroom metric includes reclaimable capacity and
must not be confused with raw free RAM. Pressure was normal/warning. Existing
swap rose only 2531328 bytes from the interval's first to last sample, and its
maximum was 424599552 bytes. GPU utilization reached 100%, observed frequency
1300.5 MHz, and available temperature samples peaked at 68.531 C. GPU power and
GPU/process GPU memory were unavailable and remain null.

Completion authority is the numeric wrapper/retained systemd exit receipts and
validated HTTP/file artifacts. ASTOP terminal dispositions were unknown and
are preserved without treating them as success. The background recorder did
not stop on SIGINT; after rechecking its exact PID/command, it was stopped with
SIGTERM, absence was confirmed, and the complete recording parsed successfully.
Only this task's watches/events were acknowledged/removed. The corrected first
failed validation unit was reset after preserving its failure receipt.

Persistent startup configuration and actual startup/image requests passed.
Neither host was rebooted and the USB cable was not physically disconnected.
