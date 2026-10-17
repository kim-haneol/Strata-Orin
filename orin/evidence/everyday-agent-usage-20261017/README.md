# Everyday agent generation timing

Read-only inspection on 2026-10-17 found the following recorded generation rates. No new inference was run.

| Recorded workload | Completed responses | Generated tokens | Decode seconds | Aggregate tokens/s |
|---|---:|---:|---:|---:|
| October 11 OMP user questions and tool use | 20 | 12,515 | 492.672 | 25.40 |
| Conversation interval after the October 15 weight update; client unconfirmed | 64 | 20,636 | 827.551 | 24.94 |

The OMP responses were matched one-to-one to engine records by input-plus-cached token count and output token count. The initial greeting and separate controlled fixtures were excluded. Prompts ranged from 7,981 to 41,855 tokens; individual decode rates ranged from 22.4 to 31.3 tokens/s.

The later service interval had prompts from 28,007 to 78,120 tokens. Two cancelled prefill requests were excluded. Its client could not be established from the available records, so these requests are not attributed to Hermes. The earlier and later rows used different dated model deployments and are not a matched speed comparison.

Aggregate speed is the sum of generated tokens divided by the sum of engine decode time. It excludes prompt processing, tool execution, and user idle time. It supports the approximate everyday-use figure in the root README, not a guarantee for every prompt or context length.

[Summary and source hashes](summary.json), [OMP timings](omp-timings.json), [service timings](service-timings.json). Only numeric timing metadata is published; private conversation text and request payloads are excluded.
