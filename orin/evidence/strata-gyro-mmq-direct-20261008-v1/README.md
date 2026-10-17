# MMQ straight from the expert cache: rejected (2026-10-08)

Proposal under test: keep the prefill's MMQ products but let them read each routed expert in place from its cache
slot, instead of copying it into a 16-expert group buffer first. That copy (`copy16_kernel`) costs about 200 ms of
a ~100-token read (`evidence/strata-gyro-prompt-fixed-cost-20261008-v1`). Serving was `recycle` with `rt-enko`; the
serving binary does not contain this code.

## What was built (`patches/strata-orin-sub8-direct.patch`, commit `0386c9a`)

For a chunk below the streamed-walk size (< 1,024 tokens) whose routed experts were all resident:

- rows ordered by cache slot;
- slots `[smin, smax]` passed to MMQ as channels (`expert_bytes` = slot stride, per-channel `expert_bounds`);
- one gate/up product and one down product per layer.

Runtime checks: every slot holds an expert, the stride equals the blob size, gate/up and down have one type, and
the last routed slot is not the arena's last. These cover MMQ reading up to a tile past each expert's last down
row; on Gyro-M's TQ2_T blobs (1,305,600 B, a multiple of 256 and of 34) that over-read lands on the next slot's
whole gate blocks.

## Result (61-request replay, `compare.txt`)

| Run | Outputs vs enko | Decode tok/s | Prompt s | ~100-token reads |
|---|---|---:|---:|---:|
| dir0 (`STRATA_PF_MMQ_DIRECT=0`) | identical | 27.03 | 219.4 | 772-842 ms |
| dir1 (direct on) | **52 of 61 differ** | 26.60 (other outputs) | **448.0** | **3,978-4,317 ms** |

Nsight on request 4 (99 tokens, `nsys-request4-kernels.json`, `prefill-timing-request4.txt`): `mul_mat_q` took
4,279 ms in 100 launches, **347,804 blocks per launch on average**. The GPU timeline was 4,561 ms, of which gate/up
was 1,579 and down 2,708.

## Why

- **The premise was wrong for this cache.** The block count means ~23,000 channels per layer (gate/up 10 row
  tiles, down 20, one column tile each). The expert profile places a layer's experts all over the 24,576-slot
  arena, not in a 512-slot run. So one product per layer spans almost the whole arena, and nearly all its tiles
  are empty slots of other layers. Each still costs a block.
- **And a bug:** the per-channel bounds were written to `bounds_dev`, which holds `2 * (512 + 32 + 2)` = 1,092
  ints. A range of ~23,000 channels overran it (and the mapped staging region), corrupting adjacent memory. The 52
  changed outputs are that corruption, not an MMQ numeric effect, and say nothing about the path's numerics.
- The fix in source (`29e581b`, `strata-orin-sub8-direct-off.patch`): the path is off by default
  (`STRATA_PF_MMQ_DIRECT=1` to enable) and refuses any slot range larger than the bounds buffer.

## What this does and does not show

- dir1's 5x slowdown rejects **this implementation**: one product per layer over the whole slot range, with the
  range overrunning the bounds buffer. It is not a measurement of a correct direct read. Such a read would keep
  the 16-expert groups and pass the routed experts' addresses through a table.
- dir0 is the build with the new path switched off. It shows the off state is unchanged, not that a fixed direct
  read is correct or fast.
- So this experiment alone does not show that every remaining optimization is under 1%.

## What a working version would need

Either of:

- a channel → slot pointer (or 64-bit offset) table inside the fork's MMQ kernels, so that only the routed experts
  are channels. The x offset is an `int` today, and slot offsets reach ~30 GB, so it must be a pointer or 64-bit.
  This touches the xy-tiling, stream-k and fixup paths. Concurrent reads across ~228 experts spread over the arena
  might also cost in the GPU TLB; that is unmeasured.
- a cache layout that keeps each layer's experts in one contiguous slot run. This would change how decode's
  residency and adaptation place experts.

The upper bound of the payoff is the copy itself: about 200 ms per short read. On the replay that is roughly 1% of
wall time (`copy16` is ~25% of a short read's GPU time, and short and mid reads add up to ~110 s of the 1,030 s).
Not pursued further.
