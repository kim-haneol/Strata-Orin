# Orin reference archive

This directory contains the selected configuration, patches, measurement tools, and evidence for the Jetson AGX Orin Gyro-M work. The root README introduces the reference fork; [OPTIMIZATION.md](../docs/OPTIMIZATION.md) explains the experiments and [REPRODUCING.md](../docs/REPRODUCING.md) builds the preserved source.

## Source and attribution

- Upstream: [agentionai/Strata](https://github.com/agentionai/Strata), revision `434e136f73a7da2bfd11cde3bb0aa0acaf1328c6`, in the [Niko1221/Strata](https://github.com/Niko1221/Strata) lineage. Upstream history and the MIT license are retained.
- ARM64 work: [shi3z/Strata-DGX-Spark](https://github.com/shi3z/Strata-DGX-Spark), revision `324b0ef56c84f2d92e9f956354bfe38e031776fa`, incorporated with the Gyro Hadamard correction in the historical Orin port `bb98d09538fd4188feb806443da21870903057f9`.
- Final engine: historical `9acaf8eacb4d56af6e9f4369e0573ce92b1f3de3`. The publication's `orin-f16cache-code` tag has the same complete Git tree, `24603390b080e9011aa02fc779b818a79ea8f9db`. Later rejected MMQ address-table commits are not applied.
- GGML: [agentionai/llama.cpp](https://github.com/agentionai/llama.cpp), revision `93a24e52cc12687213c0432af41c7fe2ab1a0eda`, plus [the shared-codebook patch](patches/agention-llama-tq-mmq-shared-lut.patch).

The local experimental commits were consolidated for publication. Their historical IDs remain in the records for provenance; they are not all reachable commits in this repository. Use the published tags and retained patches for reconstruction. Default-off experimental paths already present in the measured final tree remain unchanged.

## Recorded environment

The historical testbed was a Jetson AGX Orin 64GB developer kit (P3701-0005), CUDA SM87 with 16 SMs, Ubuntu 24.04, Jetson Linux 39.2.1, JetPack 7.2.1, and CUDA Toolkit 13.2.2 (`nvcc` 13.2.86). Benchmarks used MAXN and observed GPU clocks up to 1,300.5 MHz. This is the measured environment, not a compatibility claim for other BSPs or power profiles.

The main optimization comparison used the original Gyro-M weights. The later 262K/vision and abliterated-weight deployments are separately dated records. Their workload contracts should not be mixed with the fixed 2K/25K or 61-request replay results.

## Included material

- `config/`: dated benchmark protocols, source/model pins, and historical server templates. `${RUNTIME}`, `${HF_CACHE}`, and other placeholders must be replaced as shown in the reproduction guide; Strata does not automatically expand them in JSON.
- `patches/`: the required GGML change and historical experiment patches. Strata variant patches are cumulative against the ARM64 base, unless explicitly stated otherwise.
- `tools/`: the SM87 build helper and the original portable Python measurement, download, profiling, and replay utilities. The request generators and timer definitions are preserved.
- `evidence/`: selected experiment narratives, comparison tables, numerical results, output hashes, and profiler summaries. The small `enko` vocabulary is retained with its hash and generation recipe.
- [Everyday agent timing record](evidence/everyday-agent-usage-20261017/README.md): the observation behind the root README's approximate 25 tokens/s statement; separate from the controlled optimization measurements.

## Evidence scope

Historical narrative records can mention original runtime filenames, private orchestration scripts, and artifacts not distributed here. Machine-specific paths and account identifiers in the selected records have been replaced with placeholders. Their measurements, model hashes, and code changes are preserved. The [publication manifest](publication-manifest.json) maps each exported file to its original and published SHA256, so a redacted copy is not presented as byte-identical evidence.

This archive excludes personal conversation/request captures, SSH and client configuration, service-management receipts, full telemetry/profiler captures, downloaded weights, model packs, and compiled engines. In particular, the exact 61-request input capture is private: the timing/output-identity evidence is included, but that replay cannot be regenerated from a clone alone. The public fixed-input harness creates its own synthetic prompts.

Publication validation checks the source tree, documentation, examples, and selected CPU tests. It is not a fresh CUDA build, benchmark, or inference run. See [source verification](source-verification.json) and [publication validation](publication-validation.json).
