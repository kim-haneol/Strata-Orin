# Orin reference experiments

This directory contains selected configuration, patches, measurement tools, and evidence for Jetson AGX Orin. Each report binds its measurements to a source snapshot and a workload. Occasional experiment publications do not imply ongoing user support.

| Experiment | Report | Complete snapshot |
|---|---|---|
| Initial Gyro-M optimization, October 7–15 | [Optimization](../docs/OPTIMIZATION.md) and [reproduction](../docs/REPRODUCING.md) | `orin-reference-20261018` |
| Checkpoint spread, Gyro-L, and bounded 262K validation, October 19 | [Report and reproduction](../docs/GYRO_L_SPREAD.md), [evidence](evidence/gyro-l-spread-20261020-v1/README.md) | `orin-reference-20261020` |

## Source and attribution

- Upstream: [agentionai/Strata](https://github.com/agentionai/Strata), revision `434e136f73a7da2bfd11cde3bb0aa0acaf1328c6`, in the [Niko1221/Strata](https://github.com/Niko1221/Strata) lineage. Upstream history and the MIT license are retained.
- ARM64 work: [shi3z/Strata-DGX-Spark](https://github.com/shi3z/Strata-DGX-Spark), revision `324b0ef56c84f2d92e9f956354bfe38e031776fa`, incorporated with the Gyro Hadamard correction in the historical Orin port `bb98d09538fd4188feb806443da21870903057f9`.
- Initial final engine: historical `9acaf8eacb4d56af6e9f4369e0573ce92b1f3de3`. The publication's `orin-f16cache-code` tag has the same complete Git tree, `24603390b080e9011aa02fc779b818a79ea8f9db`. Later rejected MMQ address-table commits are not applied.
- Initial GGML: [agentionai/llama.cpp](https://github.com/agentionai/llama.cpp), revision `93a24e52cc12687213c0432af41c7fe2ab1a0eda`, plus [the shared-codebook patch](patches/agention-llama-tq-mmq-shared-lut.patch).
- Gyro-L/spread extension: `orin-gyro-l-spread-code-20261020` preserves the measured engine paths from historical `a144885a4b165d6e8fb4834a37e008f970a575da`. Add the [Gyro-L GGML patch](patches/agention-ggml-gyro-l-20261019.patch) after the shared-codebook patch; the resulting GGML tree is `61915afdbd3ce8d6d83560d470a6b1f77a688f3b`. The [new report](../docs/GYRO_L_SPREAD.md) attributes the selective upstream backports. This is not a full upstream upgrade.

The local experimental commits were consolidated for publication. Their historical IDs remain in the records for provenance; they are not all reachable commits in this repository. Use the published tags and retained patches for reconstruction. Default-off experimental paths already present in the measured final tree remain unchanged.

## Recorded environment

The historical testbed was a Jetson AGX Orin 64GB developer kit (P3701-0005), CUDA SM87 with 16 SMs, Ubuntu 24.04, Jetson Linux 39.2.1, JetPack 7.2.1, and CUDA Toolkit 13.2.2 (`nvcc` 13.2.86). Benchmarks used MAXN and observed GPU clocks up to 1,300.5 MHz. This is the measured environment, not a compatibility claim for other BSPs or power profiles.

The main optimization comparison used the original Gyro-M weights. The later 262K/vision and abliterated-weight deployments are separately dated records. Their workload contracts should not be mixed with the fixed 2K/25K or 61-request replay results.

## Included material

- `config/`: dated benchmark protocols, source/model pins, and historical server templates. `${RUNTIME}`, `${HF_CACHE}`, and other placeholders must be replaced as shown in the reproduction guide; Strata does not automatically expand them in JSON.
- `patches/`: required GGML changes and historical experiment patches. Earlier Strata variant patches are cumulative against the ARM64 base. The dated October 19 extension patches are incremental; their Strata changes are already applied to the current checkout. Follow the relevant report's order and do not reapply them.
- `tools/`: the SM87 build helper and the original portable Python measurement, download, profiling, and replay utilities. The request generators and timer definitions are preserved.
- `evidence/`: selected experiment narratives, comparison tables, numerical results, output hashes, and profiler summaries. The small `enko` vocabulary is retained with its hash and generation recipe.
- [Everyday agent timing record](evidence/everyday-agent-usage-20261017/README.md): the observation behind the root README's approximate 25 tokens/s statement; separate from the controlled optimization measurements.

## Evidence scope

Historical narrative records can mention original runtime filenames, private orchestration scripts, and artifacts not distributed here. Machine-specific paths and account identifiers in selected records have been replaced with placeholders. Measurements, model hashes, and code changes are preserved. The [initial publication manifest](publication-manifest.json) applies to `orin-reference-20261018`; its hashes are historical, not a manifest of the evolving `main`. The extension has its own [export manifest](evidence/gyro-l-spread-20261020-v1/export-manifest.json). Redacted or selected copies are identified explicitly.

This archive excludes personal conversation/request captures, SSH and client configuration, service-management receipts, full telemetry/profiler captures, downloaded weights, model packs, and compiled engines. In particular, the exact 61-request input capture is private: the timing/output-identity evidence is included, but that replay cannot be regenerated from a clone alone. The public fixed-input harness creates its own synthetic prompts.

Publication validation checks source identity, documentation, examples, and selected CPU tests. It is not a fresh CUDA build, benchmark, or inference run. The [initial source verification](source-verification.json) and [initial validation](publication-validation.json) apply to the initial snapshot. The extension's [publication validation](evidence/gyro-l-spread-20261020-v1/publication-validation.json) records its separate checks.
