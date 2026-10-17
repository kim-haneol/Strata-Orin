# OMP / REA MCP context expansion audit

Follow-up: the user requested disconnection on 2026-10-16. REA is now removed
and denied in OMP's default MCP config, both existing idle OMP agents were
reloaded, and current-policy capture advertises zero MCP tools. See the
[disconnect receipt](../omp-rea-disconnect-20261016-v1/README.md). The findings
below retain the original 2026-10-15 configuration snapshot.

Inspected 2026-10-15. Status: cause reproduced; original client configuration
and production model service were not changed. OMP 18.9.0, REA MCP 6.4.0.

## Findings

The earlier description that an "OMP profile adds 180K" was imprecise. Most
of the input is the attached REA tool catalog, rather than the normal coding
system prompt or conversation history. Both a large source catalog and schema
expansion in the OMP request path contribute.

| Controlled minimal client | HTTP request bytes | Advertised tools | Actual tokenizer input tokens |
|---|---:|---:|---:|
| Empty MCP config | 3,108 | 0 | 799 |
| Original REA MCP config | 563,006 | 170 | 177,428 |
| Same REA request with only `tools` removed offline | — | 0 | 882 |

Therefore the tool definitions add **176,546 tokens** in this request, about
67.35% of the 262,144-token context capacity before useful conversation.
The controlled cwd was inside this repository and included its ordinary
project instructions; that explains why the empty case is larger than the
previous 279-token empty-directory probe. Native full-prompt probes in the
deployment record received 181,742 / 176,990 / 176,908 tokens and were cancelled
during prefill; this audit does not repeat those expensive inference requests.

Direct MCP `initialize` and `tools/list` returned the same 170 tools. Their
canonical input schemas total 365,646 bytes; OMP's outgoing schemas total
483,373 bytes. The largest, `compare_web_captures`, expands from 57,175 to
152,420 bytes. Its 198 `$ref` nodes and `$defs` block become zero on the wire,
while object nodes grow from 1,125 to 2,718. The aggregate outgoing tool
catalog, with descriptions and function wrappers, is 559,421 bytes.

This establishes context expansion, not a duplicated-chat-history bug or an
LLM-weight regression. Whether to label the expansion an upstream defect is
an engineering judgment: the full catalog is large already, and expanding
references makes it worse. The source explicitly dereferences schemas during
normalization. `--no-tools` means built-in tools only; leaving MCP tools present
under that flag agrees with its documented scope.

For ordinary coding, a narrow tool surface and an on-demand/separate REA
profile avoids paying for all 170 reverse-engineering tool definitions. No
such change was applied in this audit; original REA availability is preserved.

## Protocol and result authority

`scripts/audit-omp-request-payload.py` starts an ephemeral loopback HTTP endpoint
that captures a synthetic request and returns a canned assistant response.
Two fresh `PI_CODING_AGENT_DIR` directories contain copies of the same original
settings/model definitions; only MCP config differs. The model URL points to
the temporary capture endpoint in both cases. Both OMP processes directly exit
0 and return the sentinel. There is no upstream forwarding or model inference.
Flags disable built-in tools, extensions, skills, rules, titles and persistence;
stdin is closed. The test executes no model-selected tools.

`scripts/audit-rea-mcp-schemas.py` reads source metadata through `initialize`
and `tools/list` only, with no `tools/call`. The metadata audit helper exits 0.
Original schemas and captured OpenAI-compatible schemas are matched by all
170 function names. Production tokenizer/template CPU code on Jetson counts
the captured requests and a `tools`-removed counterfactual; no GPU inference is
launched. ASTOP reported Jetson idle before this count. Raw requests and raw
MCP metadata remain gitignored under `work/omp-payload-audit-20261015/`.
Only sizes, hashes, names, token counts and schema-shape summaries are committed.

Reproduce from this repository, using the existing installed OMP/REA:

```bash
uv run --offline --no-project python scripts/audit-omp-request-payload.py work/omp-payload-audit-20261015
uv run --offline --no-project python scripts/audit-rea-mcp-schemas.py work/omp-payload-audit-20261015/rea-metadata
```

Primary source support:

- [OMP CLI: no-tools disables built-in tools](https://github.com/can1357/oh-my-pi/blob/v18.9.0/docs/cli-reference.md)
- [OMP 18.9.0 schema normalization](https://github.com/can1357/oh-my-pi/blob/v18.9.0/packages/ai/src/utils/schema/normalize.ts#L1192)
- [MCP runtime tool registration](https://github.com/can1357/oh-my-pi/blob/v18.9.0/docs/mcp-runtime-lifecycle.md)
