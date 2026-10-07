# Dynamic DDR expert cache component (2026-10-08)

Host-side building block for the GLM-5.3-Flash SSD offload (goal: an SSD-offload version of GLM-5.3-Flash in the
full graph). Standalone: no engine, CUDA or ggml dependencies. Next step is the miss node in the GLM graph that calls
`ExpertResidency::ensure()`.

## Parts
- `include/expert_cache.hpp` (pure C++): `ExpertCache` decides which (layer, expert) slabs are resident.
  Policies `Static` (seeded set, no updates) and `Lru` (hit moves to most-recent; a miss inserts and evicts the least
  recent non-pinned entry when full; if nothing is evictable the miss is a counted bypass). `seed()` takes a hottest-first
  list, truncates to capacity, pins the hottest `floor(capacity * pinned_fraction)` and makes hotter entries more
  recent. A per-layer residency bitmap ((n_experts + 63) / 64 words) is kept in sync, with `all_resident(layer, ids, n)`
  for a graph node (negative ids = skipped routes). `rank_hottest_first()` turns a usage file into the seed order
  (ties by (layer, expert), as in the simulator). The semantics are deliberately identical to bench/expert_cache_sim.py.
- `include/offload_store.hpp` (Linux): `OffloadStore` reads the store from `gguf_tool offload`; each expert is ONE
  `preadv(O_DIRECT)` scattering gate/up/down straight into three 4096-aligned destination planes (zero copy), with a
  buffered per-plane fallback for misaligned destinations, non-tight slabs or filesystems that refuse O_DIRECT.
  `ModelReader` reads experts from the packed model (fallback source, e.g. for a hot expert that was evicted).
  `VirtualArena` maps every (layer, projection) at full size with slot = expert id (`mmap` NORESERVE); only populated
  slots cost memory and `evict()` returns a slot's pages (`MADV_DONTNEED`), so `MUL_MAT_ID` over the arena needs no
  remapping. `ExpertResidency::ensure(layer, ids, n)` accesses the cache, evicts victims, loads misses from the first
  source that has them, and reports misses/evictions/loads/errors; a miss that cannot be loaded is dropped from the cache
  and counted in `errors` (the caller must fail the request).
- Not thread-safe by design: call from the decode thread (async loading is a later step).

## Tests (tests/test_expert_dram_cache.cpp, wired into tests/run_components.sh; 455,610 checks)
- Random equivalence with a reference LRU that transcribes the simulator (all combinations of capacity, pinned
  fraction and seeding), including victim identity, size, bitmap and residency of every key.
- Edge cases: capacity 0, all pinned, seed truncation/duplicates/out-of-range, static policy, erase, bitmap words and
  `all_resident`, usage ranking tie-breaks.
- Synthetic stores/models in the real file format: aligned zero-copy reads, misaligned fallback, non-tight store
  (this found a real bug: the fallback assumed contiguous planes), not-found/invalid files.
- Arena: no memory before population, per-expert residency by mincore, neighbours untouched on evict, evicted slots read zero.
- End to end: 6,000 ensure() calls over three layers with data verified after every eviction/reload, bitmap == cache ==
  arena residency, an injected unloadable expert reported and never cached.
- **Acceptance**: replaying the committed router trace (docs/ssd-offload/data) with the committed old-survey seed
  reproduces all 28 results of the Python simulator exactly (mean, first-20, after-20 misses per token at 32/40/48/56 GiB,
  static, LRU-empty, static+LRU and pinned+LRU, per-prompt and session).
- Real data (optional): all 3,173 experts of the 19.35 GiB store read through the component and compared with the main
  model: 0 mismatches, 2.32 ms per expert on average (2.82 GB/s), every read on the O_DIRECT path.

## Not done yet
Asynchronous/prefetched loads, prefill streaming that bypasses the cache, huge-page vs 4 KiB eviction granularity,
the graph miss node, and wiring the engine's startup preload to the cache's seed set.
