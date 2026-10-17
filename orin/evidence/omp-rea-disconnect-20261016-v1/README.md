# Disconnect REA MCP from OMP

Applied 2026-10-16 KST at the user's request. Status: verified.

Removed `mcpServers.rea` from `${OMP_HOME}/agent/mcp.json` and added `rea`
to `disabledServers`, preventing the same server name from being rediscovered
through another OMP configuration source. No other server entries existed.
The effective config is recorded in
`config/omp-mcp-rea-disconnected-20261016.json`.

The original file is backed up privately at
`${OMP_HOME}/agent/mcp.json.before-rea-disconnect-20261016T124343+0900`.
Before SHA256:
`a2c4a6c3ea39bb7954652613ab93fcfd60e76825a33da3b5f4af7cc97b86221d`.
After SHA256:
`0906d3a8072f7a32ee7ae40be6ba7f6a6ba05e67498671bc81452ef5f8cf1054`.
The backup preserves the original mode/timestamps; replacement is atomic.
Restore that backup and reload affected OMP sessions if reconnecting later.

## Verification

- The native request-capture helper, using a copy of the current MCP policy,
  advertises **zero tools** under the same no-built-in-tools probe. Its request
  is 3,117 bytes, versus the earlier 563,006-byte/170-tool REA request. This is
  a synthetic transport capture with a canned response and no model inference.
- A separate real native OMP process reads the **actual default agent
  directory**, uses the normal coding system prompt and the existing 22868
  provider, and returns `OMP_REA_OFF_OK`, direct exit 0, in 20.56 seconds.
  Built-in tools, skills, rules, extensions and title generation are disabled
  for this bounded smoke; no persistent chat is created. This is not a matched
  latency benchmark against the earlier large/cancelled requests.
  The engine log records 1,981 input tokens and 24 generated tokens for this
  smoke (4.743 s prefill and 0.694 s decode; CLI wall time includes startup).
- Reloaded both existing idle OMP agents via the supported Paseo reload API.
  User chat `<PRIVATE_CHAT>` reports 61 timeline items;
  fixture `<FIXTURE_CHAT>` reports 16. Both return
  `status=reloaded` and remain idle under the same IDs, providers and workspaces.
  All three Paseo agent IDs remain present; the unrelated Codex agent is kept.
- ASTOP pinned the old OMP PID/start identities before reload. Both old
  processes are gone; terminal records were saved, acknowledged and watches
  removed. Unknown/disappeared exit observations are not success authority:
  direct API reload receipts and native exit/output are the authority.
- Other client config hashes are unchanged: OMP models/settings, Paseo config,
  and Hermes `strataorin`. Paseo PID 2607903/restarts 0 and the existing tunnel
  PID 1307733/restarts 1 are preserved. No daemon restart or re-pairing occurred.
- Jetson API PID 636731/restarts 0 and config SHA256
  `561306126d4d7f7c385a412a657356d24782a7a27dc2ba3d44405f0a66c94d3c`
  are unchanged. Context 262144, INT8 KV, vision, prefill and conversation-cache
  settings are retained. REA's installed CLI/package and other clients are not
  uninstalled or reconfigured.

Raw Paseo inspection/reload captures and native logs remain gitignored under
`work/omp-rea-disconnect-20261016/`; sanitized receipts are committed here.
The prior payload audit remains a dated snapshot:
`evidence/omp-payload-audit-20261015-v1/README.md`.

Reproduce the no-inference verification with:

```bash
uv run --offline --no-project python scripts/audit-omp-request-payload.py \
  --current-only work/omp-rea-disconnect-20261016/capture-recheck
```
