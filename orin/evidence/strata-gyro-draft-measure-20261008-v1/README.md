# Decode after the enko draft head: MTP policy re-screen and where the window's remaining time goes (2026-10-08)

Serving build `build-sm87-recycle/strata.recycle` with `mtp/rt-enko`
(`evidence/strata-gyro-agent-round2-20261008-v1`). A window currently takes 99.5 ms: verify 89.5 ms, draft 10.0 ms.
That is 27.05 tok/s on the 61-request Hermes replay. Reaching 30 tok/s needs about 10 ms less per window.

## 1. MTP policy re-screen on the replay workload

The 106K → 47K draft head changes both the draft cost and the vocabulary that the confidence (`--spec-min-p`) is
computed over. So the earlier optimum, cap 4 / confidence 0.5 (chosen on a 2K explanation prompt), was re-checked
on the replay. Order: base, 0.4, 0.6, cap 5, cap 3, base. All outputs were identical to the enko replay
(`compare-policy.txt`).

| Policy | Decode tok/s | Windows | avg T | Tokens/window | ms/window (verify + draft) |
|---|---:|---:|---:|---:|---:|
| cap 4, 0.5 (base, 3 runs incl. `enko`) | 27.00 / 27.07 / 27.09 | 8,034 | 3.24 | 2.694 | 99.5-99.8 (89.5 + 10.0) |
| cap 4, 0.6 | 27.04 | 8,479 | 2.92 | 2.553 | 94.4 (85.0 + 9.3) |
| cap 5, 0.5 | 27.09 | 7,307 | 3.74 | 2.962 | 109.3 (97.4 + 11.9) |
| cap 4, 0.4 | 26.86 | 7,746 | 3.54 | 2.793 | 104.0 (93.4 + 10.6) |
| cap 3, 0.5 | 26.27 | 9,150 | 2.70 | 2.366 | 90.1 (82.3 + 7.7) |

- Base spread across three runs: 0.3%. Confidence 0.6 and cap 5 fall inside it. 0.4 is 0.7% slower, cap 3 2.9%.
- **Kept cap 4 / 0.5.** The optimum is a plateau: cap 5 and 0.6 trade tokens per window against window cost
  almost exactly.

## 2. Where a window's time goes (Nsight Systems)

Two traces. Both replay requests 1-3 untraced and then trace requests 4-5 (52 and 76 generated tokens):

- `mtpnode`: the serving build plus enko, `--cuda-graph-trace=node`. Per-kernel times inside graphs, but launch
  costs are inflated.
- `dqnsys`: the dqpark build with the 106K head, `--cuda-graph-trace=graph`. Gaps and launch calls with low
  overhead.

Per-graph figures are in `nsys-graphs.json`.

| Item | Measured | Share of a ~99.5 ms window |
|---|---:|---:|
| Draft round: input prep + K/V repeated for the selected row (fc_embedding, fc_hidden, HC read, K/V) | 0.39-0.42 ms per round | ~0.4% |
| Draft step (one per extra draft), enko | 2.54 ms GPU: head mmvq 1.38, MTP attention 0.33 (grows with context), router gate 0.18, HC 0.20 | 1.24 steps per window → 3.2 ms |
| Draft round, enko | 3.7 ms GPU (K/V pass 0.5-0.7 + full row 3.0) | 3.7 ms |
| Host gaps: last draft → verify | 1.09 ms (the 1,858-node verify graph's `cudaGraphLaunch` takes ~1.0 ms on the host) | ~1.1% |
| Host gaps: verify → commit, between draft graphs | 0.25 ms; 0.07 ms each | ~0.35% |
| GDN state commit after verify (`gdn_step_commit_kernel`, all GDN layers) | 1.68 ms of a 2.1 ms commit graph | ~1.7% |

- **Measured:** the duplicate prep in the draft round really exists, but it costs ~0.4 ms per window.
- The other host-side cost is the ~1.0 ms verify graph launch. The window's T is known only after the drafts
  (the confidence stop), so the graph cannot simply be launched earlier.
- **Corrected (v1 said "copies"):** the GDN commit is not a copy. Verify computes the candidates from the old state
  and does not store per-candidate states. Once the accepted count is known, `gdn_step_commit_kernel`
  (`src/kernels/cuda/verify_kernels.cu`) recomputes the state update for those `n_keep` tokens, one token at a
  time, and writes the live state. A pointer swap has no accepted-state buffer to point to. Avoiding the recompute
  would mean storing per-candidate states during verify, or some other restore scheme, and the stores and memory
  traffic that adds would have to be subtracted. The 1.68 ms is an upper bound on what could be saved, not a saving.
- **Upper bound, not a prediction:** if the costs above vanished and nothing else grew, 1.68 ms would give about
  +0.46 tok/s, 2.08 ms (with the duplicate prep) +0.58, and 3.5 ms (with the graph gaps) +0.99 at 27.05 tok/s.
  The profiles also come from two requests and two builds. None of this reaches the 10 ms needed for 30 tok/s. These are the remaining mid-sized items of the current
  structure. The draft head mmvq (1.38 ms per step) remains the largest single draft item; a lower-precision head
  was set aside on purpose.

## Reproduce

`run-policy.sh` (configs `strata-gyro-replay-pol-*.json`). For the traces:
`GRAPH_TRACE=node run-replay-nsys.sh mtpnode 3 5 enko` and `run-replay-nsys.sh dqnsys 3 5 dqpark`. Reports and
sqlite exports stay on the Jetson (`profile-*.nsys-rep`, `.sqlite`; not committed).
