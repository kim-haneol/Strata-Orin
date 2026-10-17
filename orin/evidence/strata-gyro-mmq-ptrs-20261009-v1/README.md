# MMQ reading resident experts in place through an address table: rejected (2026-10-09)

The corrected form of the direct-read idea (`evidence/strata-gyro-mmq-direct-20261008-v1`). It keeps the 16-expert
groups, launches, rows, tiles and stream-k, and passes each group's expert addresses to MMQ instead of copying the
experts into the group buffers. The baseline is the serving build `f16cache`.

## Change

- **Fork** (agention llama.cpp `mmq.cuh`): `mmq_args::x_ptrs`, a device array of per-channel weight addresses.
  Channel z's tiles read from `x_ptrs[z]` instead of `x + z * stride_channel_x`. The addresses are 64-bit, so the
  ~30 GB arena is fine.
- **Strata** (`src/prefill/prefill.cpp`, source `eaa98f3`): for a short read whose routed experts are all resident,
  one table per layer holds the gate/up and down addresses of the routed experts. Each group's two products get
  `w_ptrs` and skip the gather and the zero tail. The down over-read past an expert's last row lands on the next
  slot's whole gate blocks; this is checked per expert: the slot ends where its blob ends, every slot is filled,
  it is not the arena's last slot, and gate/up and down have one type. `STRATA_PF_MMQ_PTRS` switches it.

## Attempt 1: runtime branch in the kernel (`agention-llama-tq-mmq-lut-xptrs.patch`, build `mmqptr`)

- Outputs identical everywhere (`compare-mmqptr-1-12.txt`, `compare-61.txt`).
- Within the build, pointers on vs off: −120 to −230 ms per short or mid read (`prompt-timing-mmqptr-run1..4.txt`).
- But the `x_ptrs ? … : …` branch changed the code generation of **every** MMQ kernel, pointers used or not. The
  TQ2_T gate/up product got 35% slower: 1,454 → 1,965 ms on the 1,713-token read, 651 → 903 ms on 427
  (`prefill-timing-f16cache-vs-mmqptr.txt`; f16cache was re-measured before and after, so this is not clock
  drift).
  - Registers changed in 81 of 256 MMQ kernels, without spills (`res-*.txt`).
  - So the "off" side of the comparison above was itself slowed, and the 61-request prompt time rose from 211.1 s
    to 240.3 s.

## Attempt 2: a template argument (`agention-llama-tq-mmq-lut-xptrs-tmpl.patch`, build `mmqptr2`)

`mul_mat_q<…, bool xptrs = false>`. The launcher picks the instantiation from `args.x_ptrs`, so the existing
kernels compile as before: all 256 non-pointer MMQ kernels have the same register and stack use as in `f16cache`
(`res-f16cache.txt`, `res-mmqptr2.txt`).

Requests 1-12, crossed, plus `f16cache` as a reference (`compare-mmqptr2-1-12.txt`,
`prompt-timing-mmqptr2-run*.txt`); outputs identical:

| Read (tokens) | pointers on − off (mean of two pairs) | off vs f16cache |
|---:|---:|---|
| 99-104, 153 | −8 to +7 ms | equal |
| 193 | +64 ms | equal |
| 319-378 | +150 to +161 ms | equal |
| 427 | +181 ms | equal |
| 477 | +254 ms | equal |
| 1,713 and 4,346 (streamed walk, not affected) | +5, +37 ms | equal |

All 61 with pointers on (`compare-61.txt`): outputs identical, prompt 216.7 s vs `f16cache` 211.1 s.

## Reading

- **Measured:** with the kernel regression removed, reading resident experts in place is no faster than gathering
  them on ~100-token reads, and 150-250 ms slower on 300-480-token reads. The copy it saves (~20% of a short read's
  expert time after the group gather) is outweighed by MMQ reading weights spread across the arena instead of a
  compact group buffer.
- Where that loss comes from is not measured. Candidates are GPU TLB reach and L2 locality, and the pointer
  kernels' slightly different registers (TQ2_T J = 32: 156 → 168).
- **Rejected:** `STRATA_PF_MMQ_PTRS` now defaults to off (source `0a67529`, `strata-orin-sub8-mmqptr-off.patch`).
  Serving stays on `f16cache`.
- This closes the expert-copy line for short reads. Group gather (adopted) is what is left of the copy saving.
