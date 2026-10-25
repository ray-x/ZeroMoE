# SSD expert loading for GLM-5.3-Flash: read cost and miss estimates (2026-10-08)

> **Correction (same day):** the miss estimates in section 2 come from a random split of one survey and are far too
> optimistic. With independent prompts the route miss rate at 19 GiB on SSD is 15-25%, not 1.5%. See
> `02-survey-offload-store-and-miss-correction-2026-10-08.md`. The read-cost numbers (section 1) are unaffected.

Goal: SSD offload must be *optional* for GLM (conditional load on a miss), inside the full graph. This is the
cost model that fixes how big the problem is before any graph code is written.

## 1. Read cost (bench/expert_read_bench.py, GLM-5.3-Flash-AJ-IQ2_XXS shards)
- File on btrfs over a Samsung 980 PRO. Expert = three planes of 2,162,688 B each = 6.49 MB (IQ2_XXS layers).
  `general.alignment=32`; every plane starts 3,456 bytes past a 4096 boundary, so O_DIRECT needs a one-page-wider
  aligned superset read (about +0.2%).
- 120-240 random (layer 11-44, expert) reads per cell; buffered "cold" = posix_fadvise(DONTNEED) first.
  Codex was compiling in the root tree during the run (about one core busy).

| per-expert (6.5 MB) | 1 reader | 2 | 4 | 8 | 16 | ceiling |
|---|---|---|---|---|---|---|
| O_DIRECT p50 latency (ms) | 2.57 | 4.19 | 7.76 | 14.8 | 28.9 | 3.5 GB/s |
| buffered cold p50 (ms) | 4.72 | 5.24 | 8.29 | 14.9 | 29.3 | 3.5 GB/s |
| page-cache hit p50 (ms) | 0.80 | 0.90 | 1.19 | 2.51 | 4.08 | ~22 GB/s |

- O_DIRECT bypasses the page cache on this btrfs file (fincore growth 2.2 MB for ~780 MB read).
- One expert costs about 2.6 ms with O_DIRECT and one reader, and about 1.9 ms per expert at the device ceiling.
  Buffered cold reads are about 1.8x slower per expert at one reader and, unlike O_DIRECT, fill the page cache that
  competes with the DDR tier. A CPU MoE layer costs about 3.5 ms (118 ms / 34 layers), so one miss is about as
  expensive as a whole layer if exposed, but fits inside it if issued at routing time.

## 2. Miss estimates from `.zeromoe_usage_glm53flash` (layers 11-44, global_topn ranking)
Ranking from half the survey hits, misses measured on the other half (random split: optimistic, a floor).
A Good-Turing estimate of mass on never-seen experts is 0.73% of routes (about 2 of 272 per token).

| DDR GiB for experts | experts on SSD | SSD GiB | route miss | misses/token | layers with a miss | exposed ms/token @2.6 ms/miss |
|---|---|---|---|---|---|---|
| 59.3 (all) | 0 | 0 | 0 | 0 | 0 | 0 |
| 56 | 525 | 3.2 | 0.37% | 1.0 | 0.94 | 2.6 |
| 52 | 1,187 | 7.2 | 0.72% | 2.0 | 1.83 | 5.1 |
| 48 | 1,849 | 11.2 | 1.03% | 2.8 | 2.62 | 7.3 |
| 44 | 2,511 | 15.2 | 1.32% | 3.6 | 3.35 | 9.3 |
| 40 | 3,173 | 19.2 | 1.59% | 4.3 | 4.03 | 11.2 |
| 32 | 4,497 | 27.2 | 3.73% | 10.1 | 8.78 | 26.4 |

- Misses are spread thinly: almost every missing layer misses exactly one expert, so the workload is a few
  independent ~6.5 MB reads per token, not a burst. Worst layers (L39, L32, L38, L36) miss 2.3-2.7% of their routes.
- All 34 CPU layers' experts need only 59.2 GiB. A "64 GiB" host limit therefore leaves about nothing on SSD; to
  exercise the path use 40-48 GiB for experts (1,850-3,170 experts, 11-19 GiB on SSD).
- Decode baseline is about 138 ms/token, so even fully exposed misses cost 2-8% at 40-56 GiB.

## 3. Consequences for the design
- Issue the read when the layer's route ids are known (before the CPU works on the resident experts) and read
  with O_DIRECT in a bounded worker; one outstanding read per layer is hidden by about 3.5 ms of CPU work.
- Compute missed experts through a second MUL_MAT_ID over a small staging ring; with no misses its ids are all
  negative and the existing negative-id-skip makes it free. The first version may read synchronously.
- Per-expert (not per-layer) residency is needed in the loader: `memcpy` if the slot has a host pointer, else read.

## 4. Caveats
- Usage survey is small (130k routed hits); the split is random, not per prompt. Real route telemetry on fresh
  prompts is the next measurement. Experts of the 8 IQ2_S-down layers are somewhat larger than 6.49 MB.
- Read benchmark ran with one core busy elsewhere; the page-cache numbers include that noise.
