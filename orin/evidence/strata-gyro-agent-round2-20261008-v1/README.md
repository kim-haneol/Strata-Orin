# Agent workload, round 2: checkpoint buffers, a native short-chunk expert path, a smaller draft head (2026-10-08)

Follow-up to `evidence/strata-gyro-prompt-fixed-cost-20261008-v1` (serving: `dqpark`). The protocol is the same:
the 61 recorded Hermes requests, replayed greedily in order with `scripts/replay-strata-requests.py`, compared
with `scripts/compare-strata-replays.py` (`compare.txt`). Each configuration ran once. Decode spread between
identical-code runs here is about 0.5% (dqpark 26.31 vs rec 26.19 tok/s).

Source: `work/gyro-strata-port/strata-ptime` (branch `exp-ptime`) on top of the serving `sub8` revision.
Patches (each the whole diff from `bb98d09`): `strata-orin-sub8-native.patch` (round-2 experiment with the native
path on by default) and `strata-orin-sub8-recycle.patch` (the same with it off: the adopted build).

## 1. Checkpoint buffer recycling (adopted)

A turn-boundary checkpoint copies ~118 MB of running state into freshly allocated vectors. Each save then pays a
page fault per page and a zero fill. Now the checkpoint the LRU evicts keeps its buffers, and the next save reuses
them (single GPU only). `STRATA_CKPT_RECYCLE=0` restores fresh buffers.

- Outputs identical (61/61 against dqpark, `rec` and `nat0`).
- Checkpoint save total: 5.8 s → 2.6 s, about 50 ms per turn. Prompt total: 222.9 → 219.9 s.

## 2. Short chunks on `native_expert_grouped` (rejected, off by default)

A chunk of at most `STRATA_PF_NATIVE_MAX` tokens with all routed experts resident ran its experts on the decode
path's grouped kernels. Those kernels read each expert in place from its cache slot, so the gather copy into MMQ
group buffers (about 200 ms per short read) went away.

- **Not faster.** Reads of about 100 tokens took 769-814 ms against 775-844 ms with MMQ + gather. The fit over
  reads under 2,048 tokens got worse: 1,139 ms + 2.20 ms per token, against 806 + 2.35
  (`prompt-timing-nat1.txt`). With ~4 entries per expert, the grouped kernels are no cheaper than MMQ plus the
  copy.
- **Not bitwise the same.** It uses q8_1 activations and h like the verify window, not MMQ's products. 54 of 61
  outputs changed (`nat1`), starting with request 1, whose 101-token second part went through it.
- The code stays behind `STRATA_PF_NATIVE_MAX=N`, which defaults to 0.

## 3. Draft head over English, code and Hangul (adopted)

The MTP drafter computes its logits over a token subset (`rt/draft_vocab.bin`). The default subset holds 106,299
tokens, 275.7 MiB of head rows that every draft step reads. It includes all 55,328 Han and 5,476 kana tokens. The
new subset is the shipped English and code subset (`data/draft_vocab_en.bin`, 40,525) plus all Hangul and CJK
punctuation tokens:

    tools/draft_vocab.py --gguf <model>.gguf --base data/draft_vocab_en.bin --add hangul,cjk_punct --out mtp/draft_vocab_enko.bin

That gives 47,493 tokens and 123.2 MiB (`draft_vocab_enko.bin`, sha256 in `draft-vocab-sha256.txt`). The runtime
dir `mtp/rt-enko` links `dense.bin`, `dense.txt` and `experts.bin` from `mtp/rt` and holds the new
`draft_vocab.bin`.

| Run (same binary `recycle`) | Draft vocabulary | Decode tok/s | Draft ms/window | Verify ms/window | Accepted / offered | Wall s |
|---|---|---:|---:|---:|---:|---:|
| rec (draft numbers: `p`, the same head) | 106,299 | 26.19 | 13.13 | 89.53 | 13,681 / 17,991 | 1057.8 |
| enko | 47,493 | **27.09** | **9.98** | 89.41 | 13,681 / 17,997 | **1030.6** |

- **Measured:** outputs identical (61/61). The verify step decides every token, so drafts change only speed.
  Accepted drafts are the same 13,681. Draft time per window fell by 3.15 ms. Decode +3.4% against rec, +3.0%
  against dqpark.
- Window figures come from `STRATA_DECODE_TIMING`, weighted over 8,034 windows. The `rec` run had no decode timing,
  so its draft/verify columns come from the `p` run (the same 106,299-token head). The `rec` run supplies decode
  tok/s and wall time.
- **Cost:** a token outside the subset can never be drafted. Chinese or Japanese output now gets almost no MTP
  drafts: 36 Han and 85 kana tokens remain. Korean and English keep theirs. This workload's Korean summary (t3)
  lost no acceptance.

## Serving now

`config/strata-gyro-orin-server.json`: `build-sm87-recycle/strata.recycle` (sha256 `4ea3f309…`) with
`--mtp …/mtp/rt-enko`. The previous Jetson config is saved as `server.json.dqpark-20261008`.

| Build | Decode tok/s | Prompt s | Wall s |
|---|---:|---:|---:|
| sub8 (serving at the start of the day) | 26.16 | 271.9 | 1110.0 |
| dqpark | 26.31 | 222.9 | 1056.3 |
| **recycle + enko** | **27.09** | **220.0** | **1030.6** |

On this agent workload, wall time for the same outputs is −7.2% against the morning's serving build.

## Reproduce

`run-native.sh` (build, then replays `nat1` and `nat0` with `STRATA_PF_NATIVE_MAX=0`) and `run-recycle.sh`
(build, then the `rec` replay). The enko replay is
`work/hermes-capture-20261008/run-replay-env.sh enko enko "STRATA_PROMPT_TIMING=1 STRATA_DECODE_TIMING=1" 1 61`
with `config/strata-gyro-replay-enko.json`.
