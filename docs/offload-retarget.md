# Page-reusing expert eviction (`ZEROMOE_OFFLOAD_RETARGET`)

Code: `memory/offload_store.hpp` (`VirtualArena::retarget`, `ExpertResidency::begin`).
Results: `docs/GLM5.3-baseline.md`, `configs/bench/perf-baselines.md` (Trial 47).

## What it does

With SSD offload, routed experts live in a DDR pool (`OFFLOAD_DDR_GIB`). When a needed expert is missing and the pool is full, the
cache evicts a victim and loads the new expert into its slot.

| | Without (`ZEROMOE_OFFLOAD_RETARGET=0`) | With (default) |
|---|---|---|
| Victim's pages | released with `madvise(MADV_DONTNEED)` | moved to the incoming expert with `mremap(MREMAP_MAYMOVE\|MREMAP_FIXED\|MREMAP_DONTUNMAP)` |
| Incoming expert | page-faults fresh pages; the kernel zero-fills them | reads straight into pages that are already populated |
| Other cost | TLB shootdowns across all threads for every released page | none |

`mremap` leaves an empty anonymous mapping behind at the victim's address, with the same VMA flags (for example `NOHUGEPAGE`).
Only the common `min(strideA, strideB)` bytes of each plane move; any tail is released with `madvise`.

## Why it was slow

nsys on GLM-5.3 showed both GPUs idle about 85% of a decode step, waiting on the CPU expert lane (about 236 ms of math plus about
165 ms of SSD split-load stall per token). Inside that lane, freeing and refaulting pages on eviction was the largest remaining
cost: every miss paid a `madvise`, a fault, a zero-fill and a round of TLB shootdowns on a 12-thread process.

## Behaviour and safety

- Runs under the arena lock, after the `before_evict` hook, exactly as `evict()` does.
- Pairs one victim with the incoming expert per miss (`a.inserted && !paired`); further victims are evicted normally.
- If `mremap` fails for a plane, the victim's remaining planes are freed and the call falls back to normal eviction.
- Output is unchanged. Only page management differs (GLM-5.3 text read and coherent).
- Needs Linux 5.7+ for `MREMAP_DONTUNMAP` (the dev box runs 7.2).

## Default and scope

- **On by default.** `ExpertResidency` only exists when SSD offload is active, so the setting reaches only models that use the
  offload store. Set `ZEROMOE_OFFLOAD_RETARGET=0` to turn it off, for example for an A/B.
- Models not using the offload store (Qwen, DeepSeek and others) never reach this code.

| Model | Status |
|---|---|
| GLM-5.3 (3090 + 5060 Ti, SSD offload) | measured: 1.74-1.78 -> 2.05-2.09 tok/s first-to-last (about +17-19%) |
| GLM-5.3-Flash and other glm5next-graph models with `OFFLOAD_DDR_GIB` | same mechanism, not measured; gain scales with the eviction rate |
| Ling 3.1 Flash | not measured; affected only if its graph goes through this store |

The gain is largest with a small pool and a high miss rate. A model whose experts all fit in the pool evicts rarely and sees little.

## How to verify on a new model

1. Run the same recipe twice, with `ZEROMOE_OFFLOAD_RETARGET=0` and with it unset, swap-free and warmed.
2. Compare first-to-last tok/s (`REQUEST-DECODE-OUTPUT`) and the lane line `[GLM5NEXT-LANE] breakdown`. `begin_load` and `phase2_graph` drop.
3. Read the output text. Run-to-run spread is about +-0.05-0.07 tok/s, so only a larger gap counts.

## Default-on check (2026-10-20)

Peak recipe, 80 tokens, 64 GiB pool, swap-free, `ZEROMOE_OFFLOAD_RETARGET` unset: 2.08 tok/s first-to-last, Rates decode 1.95,
output text coherent (`logs/split_ab_default_on.*`).
