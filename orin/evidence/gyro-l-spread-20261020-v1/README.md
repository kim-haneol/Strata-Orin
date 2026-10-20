# Gyro-L and checkpoint-spread evidence

This is a selected export of the October 19, 2026 Orin campaign, published October 20. The English [technical report and reproduction procedure](../../../docs/GYRO_L_SPREAD.md) separates the same-model spread A/B, M regression, common-settings M/L comparison, memory validation, and larger-chunk candidate.

| Question | Record |
|---|---|
| What source and executables ran? | [Source bindings](source-bindings.json), [execution bindings](execution-bindings-v4.json), [published commit mapping](publication-source-map.json) |
| Did the GGUF pass complete hash verification? | [L integrity](l-integrity.json) |
| What changed on return to the parent? | [LRU](cache-lru.json), [spread](cache-spread.json) |
| Did the integrated engine change M's measured 2K output? | [M regression](m-2k-regression-v4.json), [M output comparison](m-comparison-v4.json) |
| What are the matched M/L results? | [Common performance](common-performance-v4.json), [ratios](comparisons.json) |
| Where are ranges, standard deviations, and resource measurements? | [Aggregate summary](summary.json) |
| What passed at long context? | [API results](l-context-result.json), [telemetry](l-context-telemetry.json) |
| What did the short full-resident and draft-off/on probes cover? | [Initial resident smoke](l-off-v3.json), [reserved draft-off](l-off-v4.json), [draft-on](l-on-v4.json) |
| What do CPU/offload counters mean? | [Definitions and source locations](counter-definitions.json) |
| What arithmetic checks passed after fixing test-stream ordering? | [MMQ log](mmq-parity.log) |
| Which early worker results were interrupted? | [Interrupted workers](interrupted-workers.json) |
| What was exported and checked for publication? | [Export manifest](export-manifest.json), [publication validation](publication-validation.json) |

The report's performance figures are historical measurements. Publication validates their source correspondence and selected export, without launching a new GPU build or inference run. The prior complete publication is retained as `orin-reference-20261018`; this extension is `orin-reference-20261020`.

The export replaces host/account-specific paths with placeholders. `summary.json` retains aggregate statistics, telemetry, direct-exit records, and comparisons; its duplicated nested case/service receipts are omitted. Selected final cases are provided separately. Original source hashes and published hashes are both recorded in the manifest. Synthetic request/answer text is preserved verbatim, including Korean fixtures; the report and instructions are English. No personal agent conversations or request captures are included.

Some source bindings describe preliminary candidates. In particular, the unversioned `l-off.json` config in the historical bindings is the rejected `--spec 0` attempt. Reproduce the successful draft-off case with `l-off-v4.json`, which still loads MTP weights. The frozen common protocol also retains an early `off_scope` description of MTP removal; the report corrects that interpretation explicitly.

The aggregate direct-exit map deliberately retains failed attempts. The initial `parity`/`mmq-l` failures are not final passes: the accepted MMQ run is `mmq-ordered-stream=0`, with the log above. Interrupted workers with misleading raw zero exits are identified separately. ASTOP disappeared/unknown events are not completion authority; the accepted final runs also have completed result JSON and direct native/worker completion records.

Raw telemetry streams, full build/profiler logs, service-management files, model weights, compiled binaries, and the author's editing transcript are outside this export. Use the included deterministic clients to create new fixtures and results in a fresh workspace. Historical raw-path references are provenance, not a promise that every referenced artifact is distributed here.
