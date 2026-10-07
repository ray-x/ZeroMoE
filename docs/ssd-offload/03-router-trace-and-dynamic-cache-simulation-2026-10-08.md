# GLM-5.3-Flash: router trace of the 5-prompt survey and DDR-cache simulation (2026-10-08)

Follow-up to doc 02 (a static cold set misses 15-25% of routes on new prompts). This measures what a DYNAMIC DDR
cache over the SSD store would do, from a real router trace.

## 1. Trace
The same 5 prompts were re-run with `log_level = "VERBOSE"`, `trace_filter = "TRACE-ROUTER"` (layered path, temperature
0). The engine prints the picks of the last token of each batch: the last prompt token, then every decode token.
Parsed: 5 prompts, decode tokens 159/159/159/159/80 (the fifth stops early), 34 CPU layers x 8 routes = 272 per token.
Validation: decode counts from the trace equal `usage_decode - usage_prefill` of the first run EXACTLY (194,752 hits,
diff 0), so the parse is right and routing is deterministic across runs. (The decode usage file counts prefill tokens
too; the prefill file counts prefill only.) Data: docs/ssd-offload/data/glm53flash_survey5_router_trace.txt.

## 2. Working sets
Distinct CPU-layer experts used in decode (of 9,792): prompt 1 6,757 (69%), prompt 2 6,272, prompt 3 5,878,
prompt 4 5,236, prompt 5 4,956; the union of all five is 9,154 (93%). One prompt already touches 50-69% of the experts;
a handful of prompts touch nearly all of them. A DDR cache therefore needs thousands of experts to be effective.

## 3. Cache simulation (tools: bench/expert_cache_sim.py; initial set = top-C of the OLD survey, independent of these prompts)
Misses per decode token (272 routes), session = cache persists across the 5 prompts, per-prompt = reset at each prompt.
Cost at 2.2 ms per exposed miss (QD1 read of one expert slab; no overlap assumed).

| DDR for experts | experts C | static only | LRU from empty (session) | static+LRU per-prompt | static+LRU session | exposed ms/token (static+LRU) |
|---|---|---|---|---|---|---|
| 32 GiB | 5,294 | 106.8 | 28.1 | 25.5 | 24.8 | 55 |
| 40 GiB | 6,617 | 70.8 | 19.3 | 15.1 | 14.2 | 31-33 |
| 48 GiB | 7,941 | 39.9 | 14.2 | 7.3 | 6.5 | 14-16 |
| 56 GiB | 9,264 | 11.4 | 12.8 | 2.1 | 1.4 | 3-5 |

- Dynamic caching cuts misses 5-10x versus a static set of the same size. At 48 GiB: 6.5-7.3 misses/token (2.4-2.7% of
  routes, 42-47 MB/token), about 14-16 ms/token if fully exposed on a 138 ms decode; at 56 GiB: 1.4-2.1 misses/token.
- Starting from a survey-ranked set matters for the first tokens (first 20 tokens at 48 GiB: 8.5-10.6 misses vs 28.8
  for LRU from empty). "Pinned half + LRU" is about equal to plain static+LRU here.
- Misses happen almost only on the first use of an expert in a session plus capacity churn; the exposed cost can be
  reduced further by issuing the read when routing is known and by prefetch (not simulated).
- A "64 GiB total" host limit leaves about 56 GiB for experts, i.e. almost no misses; to exercise the SSD path use
  40-48 GiB for experts.

## 4. Design that follows
- The unit that must be cached is a (layer, expert) slab; the cache is dynamic (promote on miss, evict least recent),
  seeded from usage statistics.
- Idea to avoid changing graph tensors: keep a full-size virtual host arena per layer (all 288 slots at fixed
  addresses; slot index = expert id) and populate only resident experts' pages. MUL_MAT_ID then works unchanged on the
  arena; a miss reads the expert from the store (aligned O_DIRECT into the slot) before the op; eviction frees that
  slot's pages (MADV_DONTNEED). The "conditional" part is a small CPU node before MUL_MAT_ID that checks a residency
  bitmap for the layer's ids and loads only missing experts (synchronous first, then issued at routing time).
- Open issues: the arena is projection-compact (gate/up/down are separate planes), so one expert slab maps to three
  destination ranges (three aligned reads or one read plus copies); per-expert eviction needs 4 KiB page granularity
  while the arenas currently run on 94% transparent huge pages, which the CPU path likes, so the trade-off must be
  measured; prefill touches many experts and should bypass the cache (streaming) rather than thrash it.

## 5. Caveats
- Only 5 prompts and 160 decode tokens each (the last stops at 80); the old survey that supplies the initial set has an
  unknown prompt mix. The simulator replays decode tokens only (no prefill effects on the cache) and counts a miss as a
  full 6.49 MB read (IQ2_S-down layers are larger).
