# Abliterated Gyro-M re-encode update

Status: deployed and verified on 2026-10-15, production ready at 07:13 UTC. Target revision
`aca61b2202b6e0a91a761236b65c151a68f88c16`, official model SHA256
`9f31a8d1f449f67cfd3b7eb6d500f4ae37d95be6e3f83b8d4a1572646b99874e`.

## Deployment result

All 17 candidate checks passed: 11 context/API checks and six vision checks.
Preparation, apply and workflow direct exit receipts are all 0; systemd records
`Result=success`, `ExecMainCode=1/CLD_EXITED`, `ExecMainStatus=0`. ASTOP's
disappeared/unknown terminal records are retained but are not success authority.

| Check | Result |
|---|---|
| Exact 64,000-token dispersed recall | Pass, 114.21 s |
| Exact 128,000-token dispersed recall | Pass, 235.81 s |
| Exact 253,952-token dispersed recall | Pass, 498.22 s |
| Short Korean, tool call/result, A/B/A/B cached restore, SSE | All pass |
| OCR/shapes, image A/B/A restore, large image, two images | All pass; restored image reused 348 tokens |
| Long context + image | Pass, 255,001 input tokens, 501.24 s |
| Promoted production API | Real text generation and image check pass |
| Host forwarding | 22868 real generation/models pass; legacy 18092 health pass |

The enabled `jetson-gyro-api.service` is active with zero automatic restarts;
the existing 22868 USB-NCM tunnel and Paseo daemon retain their original PIDs
and restart counts. Live process arguments prove one text engine using the new
pack/GGUF and the CUDA vision helper using the new snapshot. Context 262144,
INT8 KV, expert cache 24576, prefill 8192, dense FP16 cache, MTP enko and the
4096 MiB/four-slot conversation pool remain unchanged. No parallel inference
slots were added. Production config SHA256 is
`561306126d4d7f7c385a412a657356d24782a7a27dc2ba3d44405f0a66c94d3c`;
the host tracked default config now matches it.

Hermes `strataorin`, OMP model/settings and Paseo config hashes all match their
pre-update values. The original model/pack/config remain local for rollback.
Receipts: [summary](summary.json), [context/API](context-result.json),
[vision](vision-result.json), clients (testbed record; not included in this archive),
live arguments (testbed record; not included in this archive).

## Client probe limitation

The native OMP probe using the original profile unexpectedly sent 181,742
tokens and hit the 120-second smoke-test cap during prefill; no output was
generated. A minimal prompt still sent 176,990 tokens; a fresh empty cwd still
sent 176,908. All three were cancelled, and the server returned idle.

An ephemeral `PI_CODING_AGENT_DIR` containing byte-identical copies of the
original `config.yml`/`models.yml`, an empty MCP config and a minimal prompt
sent only 279 tokens and returned `GYRO_OMP_OK`, exit 0, through the same 22868
provider. The original profile has a `rea` MCP server. This implicates the
original profile's MCP payload, but individual tool/schema contributions were
not audited. It is a client payload issue to investigate separately, not a
completed normal-profile latency test or a reason to claim a model regression.
No original OMP settings, MCP config, mobile chat or session were changed.
[Engine receipts](omp-probes-engine-log.txt) retain token counts/cancellation.
The isolated check uses `--no-session`, no built-in tools/extensions/skills,
no title generation, explicit minimal system prompt and closed stdin.

Subsequent controlled [payload audit](../omp-payload-audit-20261015-v1/README.md)
settled this limitation: REA contributes 170 advertised tools, whose schemas
are expanded by OMP; the tool definitions alone add 176,546 tokens in a fresh
minimal request. The original profile and MCP configuration remain unchanged.

## Resource observations

Native ASTOP recording retained 993 samples across download, reconstruction,
pack generation, all candidate tests and production startup. Minimum measured
headroom was 4.37 GiB; last recorded headroom was 8.16 GiB. Swap was already
0.88 GiB before the update and was 0.39 GiB at the last sample. Pressure states
were normal/warning. GPU utilization reached 100%; frequency reached 1300.5
MHz. Temperature was available in 327/993 samples (41.75–64 C); power and
per-process GPU memory/utilization were unavailable and remain null.

These observations mix different cache and service lifecycle states; they do
not establish a reduction in the new model's RAM usage. After client probes,
ASTOP reported idle and about 7.9 GiB headroom. All task watches were recorded,
acknowledged and removed; temporary workflow/telemetry jobs were stopped.
[Telemetry aggregate](telemetry-summary.json); raw NDJSON remains on Jetson.

## Frozen protocol

1. Preserve the current old model, readonly HF snapshot, dense pack, runtime,
   config, unit, client settings and `strataorin`. Back up the production config
   under a new update-specific name; rollback reuses local files without another
   model download. Do not change context 262144, INT8 KV, expert slots 24576,
   prefill 8192, dense FP16 cache, MTP enko, conversation cache or CUDA vision.
2. Retrieve only the audited 144 routed gate/up/down tensors (32,086,425,600
   bytes) and the new header. Reuse the unchanged projector. Verify strict HTTP
   206 ranges, source complete SHA256, reconstruct separately, and verify target
   complete SHA256 before publication. The prior bounded audit did not certify
   all other tensor bytes; a hash mismatch is a failed candidate and must never
   be deployed. Expand the download only if required by this final check.
3. Build a separately named dense pack with the existing packer. Keep engine and
   vision binaries unchanged because the verified layout/types are unchanged.
4. Before switching production, freeze a new config/model-bound validation
   manifest using the existing immutable synthetic requests. Run short Korean,
   exact 64K/128K/253952 recall, tool call/result, conversation restore, SSE,
   images/OCR/shapes and a long prompt plus image. This bounds API compatibility
   and synthetic recall, not broad quality or coding improvement. Keep a small
   matched old/new text workload as a regression diagnostic.
5. Candidate testing requires a bounded production interruption because two
   full models cannot coexist in RAM. A wrapper must automatically restart the
   old service/config on failure. Only publish the new production config after
   validation passes; check readiness, real generation, preserved 22868 tunnel,
   client settings, service enablement and rollback path.
6. ASTOP before/during/after, named watches and server-side waits; preserve direct
   process exit receipts. Unknown/disappeared observation status is not success.
   Summaries, manifests and checksums are committed; raw model payload and
   telemetry remain outside Git.

Remote staging directory:
`${TEST_HOME}/.local/share/jetson-orin/gyro-abliterated-update-20261015`.
Previous deployment remains in
`strata-gyro-orin-20261007/packs/gyro-m-abliterated-20261011`.

## Verified reconstruction

The delta process directly exited 0 (systemd `Result=success`,
`ExecMainCode=1/CLD_EXITED`, `ExecMainStatus=0`). Its
[receipt](download-complete.json) verifies both the complete old SHA256
`9a43d2978584c02d9512bfe7bacf5fdebb2b05f8a26fcc04a89ed6138f0e7b28`
and complete target SHA256 above. Only 32,086,425,600 changed tensor bytes
were received, plus the small header/metadata requests. The
[header comparison](header-comparison.json) proves identical headers. Therefore
the previous audit's 229 unresolved tensors are now settled: exactly 144
tensors changed and the other 1,080 are byte-identical.

The unchanged 904,004,128-byte projector was reused after SHA256 verification
(`f456cd796fdbdef0cadb22710b54e0071b6cdf22c07963365baa898267aec517`).
The new dense pack was generated separately; its hashes and unchanged engine /
vision binary hashes are retained in [pack-hashes.txt](pack-hashes.txt) and
[binary-hashes.txt](binary-hashes.txt).

## Matched short text diagnostic

One warmup plus three measured requests per model, each 3,414 input tokens,
128 generated tokens and zero prompt-cache tokens. Same engine, device and
runtime options; old was measured immediately before maintenance, new on the
candidate service. The medians are:

| Metric | Previous weights | Updated weights |
|---|---:|---:|
| HTTP wall seconds | 10.818 | 11.115 |
| Reported prefill seconds | 7.223 | 7.500 |
| Reported decode tokens/second | 36.2 | 36.0 |

The measured HTTP difference is +2.75%, principally in prefill. This is one
small workload with three repeats, not a statistically established general
speed change or a coding-quality benchmark. All four old/new outputs were
identical. There is no fresh task-correctness comparison supporting a claim
that these re-encoded weights improve coding quality. Order was old then new,
without randomized crossover. GPU temperature was intermittently available
and power was unavailable, so no thermal/power explanation is established.

## Implementation and preparation checks

- Six delta-transfer regression checks passed, including the distinct
  ordinary-to-abliterated and re-encode selections, strict ranges, positional
  writes, verified resume and reuse of one projector blob in two snapshots.
- Five mocked transaction checks passed: preflight refusal, failed candidate,
  failed production probe, verified success, and a failed rollback probe that
  must never be reported as a successful recovery. These do not replace the
  subsequent real API/model checks.
- The initial four-connection transfer was deliberately stopped and resumed
  with sixteen connections. Completed chunks are hash-verified before reuse;
  per-worker verification/download buffers stay at 4 MiB. Initial interruption
  receipts are retained separately, and are not success evidence.
- A supervised workflow waits for the exact download process and then checks
  the direct exit receipt and completed target hash. It builds a separate pack,
  freezes test inputs/config, collects the old text baseline and waits for the
  GPU to be idle before maintenance. Candidate tests use private loopback 18094;
  18092 is promoted only after all checks pass. Host 22868 forwarding is retained.
- The old config backup is `server.json.before-reencode-20261015`, SHA256
  `a76a4752fbe8a6a8cc55401c378f02c224c13f0231fe218eeb3a20078b84aba5`.
  Engine/encoder binaries, runtime tuning and client configs are preserved.
- Rollback helper: `scripts/rollback-gyro-reencode-update.sh`, also copied into
  the remote update directory. The previous model remains local and verified.
- Native ASTOP recording runs every five seconds; download and workflow watches
  use exact PID/start identities. API/tensor/telemetry receipts are the final
  result authority, not an unknown process-exit observation.

The machine-specific rollback launcher is not distributed in this reference archive.
