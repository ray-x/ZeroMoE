# Predictive MoE Loading: RAM/VRAM Prefetch & Promotion Thresholds

How the pilot router's layer-ahead predictions turn into real RAM fetches and
VRAM promotions, and exactly which config knob decides "load it" vs "skip
it" at each stage. All line references are `src/execution_layer.cpp`'s pilot
block (~2900-3282) and `include/dense_dispatch.hpp`'s config-holder
functions, current as of 2026-08-17. See `CLAUDE.md` for the real-hardware
A/B history behind each default value.

## The pipeline, in order

For every token, near the end of layer `l`'s processing (using `l`'s fully
combined post-attention/post-FFN hidden state), the pilot router runs a
cheap gate pass to predict layer `l+1`'s top-K expert picks, and optionally
layer `l+2`'s. Each predicted `(layer, expert)` pick then goes through the
same four stages:

```
predict --> classify residency --> RAM hint (maybe) --> VRAM promote (maybe) --> eviction-protect (maybe)
```

### Stage 0 — Classify residency (always runs, no threshold)

Every predicted pick is checked against both tiers before anything else
happens (`execution_layer.cpp:3005-3040` for l+1, `:3221-3232` for l+2):

| Bucket | Meaning | What happens next |
|---|---|---|
| `already_ram` | resident in the RAM-tier LRU, not yet in VRAM | candidate for RAM→VRAM promotion (Stage 2) |
| `already_vram_pinned` | resident in VRAM and pinned | nothing to do — can never be evicted anyway |
| `already_vram_lru` | resident in VRAM, evictable | nothing to do, but still added to the eviction-protect set (Stage 3) |
| `needs_fetch` | resident nowhere | candidate for a real RAM fetch (Stage 1) |

This check costs one `RAMSafetyManager::is_ram_resident` lock + a linear
scan of `expert_vram_cache` — cheap, and it's why a correct prediction of
something already resident costs nothing beyond this classification.

### Stage 1 — RAM (DDR) fetch: `needs_fetch` only

Only the `needs_fetch` bucket calls `async_io_queue.push_hint(layer, expert)`
— a lock-free ring-buffer push, not a synchronous read. A background
`LockFreePrefetcher` worker later pulls the hint and calls
`RAMSafetyManager::claim_or_wait(..., RamClaimCaller::PrefetchWorker)` — the
**same** unified claim/publish state machine the decode thread's real cache
misses use. If the decode thread already resolved that expert first, the
worker's claim finds it Ready/Loading and does zero real I/O
(`claims_beaten_by_decode`). Only if the worker wins the claim does a real
disk read happen, straight into a real RAM-tier LRU slot.

**Threshold — l+1 (`prefetch_pilot_k` / `prefetch_pilot_k_per_layer`):**
how many ranks the pilot predicts *and* is willing to fetch for l+1. Every
`needs_fetch` rank within this count gets hinted — there is no separate,
smaller "fetch cutoff" at depth-1; prediction count *is* the fetch count.
Default 8, current live config: 8 (no per-layer override set).

**Threshold — l+2 (`prefetch_pilot_depth` + `prefetch_pilot_k_depth2`):**
two gates, not one:
1. `prefetch_pilot_depth` must equal `2` (default `1` = l+2 measurement-only,
   no real hints ever pushed). **Current live config: `1` — l+2 real
   prefetch is OFF.** Measurement (recall counters) always runs regardless,
   so the "Pilot recall by rank" report is populated either way.
2. Even when depth-2 is on, only the top `prefetch_pilot_k_depth2` ranks
   (`execution_layer.cpp:3207`, `depth2_push_cutoff = min(prefetch_pilot_k_depth2, pilot_k2)`)
   get a real hint — deeper l+2 ranks are still classified for the report
   but never fetched. This is a **separate, usually smaller** cutoff than
   l+1's own `prefetch_pilot_k`, because l+2's prediction is structurally
   weaker (see "Why l+2 is weaker" below). Current live config:
   `prefetch_pilot_k_depth2 = 1` (rank 0 only, if depth-2 were ever turned
   back on).

**Why l+2 is weaker:** l+2's prediction reuses l's *pre*-l+1-FFN hidden
state (`execution_layer.cpp:3134`, "naive reuse") — it never sees l+1's own
FFN contribution, the exact term the pilot's own l+1 prediction already
needs to be accurate. An FFN-reconstruction alternative was tried and found
statistically tied despite costing real compute (removed; see CLAUDE.md).

### Stage 2 — RAM→VRAM speculative promotion: `already_ram` only, l+1 only

Depth-2 has **no VRAM promotion path at all** — grep-confirmed, no
`promote_ram_to_vram` call exists in the l+2 block. Only l+1's `already_ram`
bucket is eligible.

**Threshold A — general vault pool (`ram_vram_promote_top_k` /
`ram_vram_promote_adaptive`, mutually exclusive):**
- **Static** (`ram_vram_promote_top_k = N > 0`): promote if `k < N` (rank
  strictly less than N). Current live config: `N = 2` (ranks 0-1 only).
  Real A/B history (CLAUDE.md): `top_k=3` measurably regressed cache-hit
  rate via VRAM churn; `top_k=1` vs `top_k=2` flipped which was "best"
  between a 40-token and an 80-token prompt — `top_k=2` is the current
  best-tested value, not a proven optimum (`top_k=3` was never retested at
  the longer length).
- **Adaptive** (`ram_vram_promote_adaptive = true`): self-calibrates from
  the session's own live `n_pilot_ram_only_hits_by_rank` /
  `n_pilot_ram_only_checks_by_rank` counters. Walks rank 0, 1, 2, ... and
  promotes while **both**:
  - `checks[rank] >= ram_vram_promote_min_samples` (default 30 — the
    cold-start guard; this metric accumulates passively regardless of
    whether promotion is even acting on it, so there's no bootstrap
    problem, just a thin-sample window early in a session), **and**
  - `hits[rank] / checks[rank] >= ram_vram_promote_adaptive_threshold`
    (default `0.30`, i.e. 30% — provisional, not separately re-tuned from
    the static sweep above).
  Stops at the **first rank that fails either condition** — a contiguous-
  prefix cutoff (`adaptive_promote_still_eligible`,
  `execution_layer.cpp:2967`): a later rank clearing the bar alone is not
  trusted, matching the monotonic-decay shape seen in every real capture so
  far. Currently OFF in the live config (`ram_vram_promote_top_k=2` is used
  instead).
  Promotion always uses **COPY semantics** (`allow_move=false` inside
  `promote_ram_to_vram`) — the RAM copy is never freed on a speculative
  promotion, specifically because this is a bet with real-world accuracy in
  the 15-44% range depending on rank (see "Pilot recall by rank" report); a
  wrong bet degrades to a cheap RAM hit later, never a disk re-read.

**Threshold B — GPU1 dual-role slice ("Idea 1", `predict_gpu1_rank_start` /
`predict_gpu1_rank_end` / `predict_gpu1_cap_per_layer`):** a second,
independent decision over the *same* `already_ram` population and the same
per-rank `best_exp`, not an else-branch of Threshold A — a rank can
satisfy both if the rank windows happen to overlap (in practice they're
configured disjoint). Targets deeper, weaker ranks (4+) specifically,
because a wrong bet here can only cost churn in GPU1's small, isolated
dual-role slice (`GpuTopology::dual_role_gpu_id()`), never the shared vault
pool's own eviction candidates — the same weak-rank idea that measured
badly on the general pool is a defensible experiment when blast-radius is
this contained.
- **Enable gate:** `predict_gpu1_rank_start >= 0` (this is what actually
  flips `Gpu1PredictPromoteConfig::enabled`, per `main.cpp:389` —
  `rank_end`/`cap_per_layer` alone do nothing if `rank_start` is negative).
  **Current live config: `predict_gpu1_rank_start = -1` → disabled**,
  despite `predict_gpu1_rank_end=4` / `predict_gpu1_cap_per_layer=1` also
  being set in the file — those two values are inert until `rank_start` is
  set to a non-negative rank.
- **Rank window:** `rank_start <= k <= rank_end`, both inclusive, both
  required in `[0, 7]`.
- **Cap:** `predict_gpu1_cap_per_layer` promotions per **layer transition**
  (not per token) — a local counter (`gpu1_predict_promotions_this_transition`)
  reset each transition, not persisted.

### Stage 3 — Eviction protection: both depths, both `already_ram` and `already_vram_lru`

Anything confirmed resident this transition — from **both** l+1 and l+2
(when depth-2 real hints are on), excluding pinned entries since those
can't be evicted anyway — is folded into one `PilotConfirmedEntry` array
(cap 16: up to 8 from l+1 + up to 8 from l+2) and published **once**, after
the full k-loop for both depths (`execution_layer.cpp:3273-3277`), via
`publish_pilot_confirmed_resident()`. This is a **wholesale replace**, not
an accumulating window — the previous transition's set is fully discarded.

While published, this set is excluded from eviction-victim selection at all
three call sites: `GpuTopology::evict_lru`, the RAM-tier D2H-salvage victim
scan, and `RAMSafetyManager::claim_or_wait`'s fresh-miss victim scan
(`include/gpu_topology.hpp`, `include/ram_manager.hpp`). No config flag
gates this — it is unconditionally compiled in at all three sites.

**Protection span** (verified by tracing the code, not assumed): roughly
one layer's worth of processing — from being predicted (during layer `l`)
through being used (during layer `l+1`'s own dispatch), ending when layer
`l+1→l+2`'s own transition republishes and wholesale-replaces the set. It
does not persist across multiple tokens or many layers.

For l+2 specifically, only picks within `depth2_push_cutoff` (the same
`prefetch_pilot_k_depth2`-bounded set that gets a real fetch) are eligible
for protection (`execution_layer.cpp:3240` `depth2_active_here`) — nothing
gets "protected" that was never actually relied on for a real fetch.

## "What is the threshold for a predicted MoE to be loaded?" — quick answer

There is no single threshold; it's a chain of independent gates, one per
stage, each with its own knob:

| Stage | Gate | Config knob | Default | Live (this box) |
|---|---|---|---|---|
| RAM fetch, l+1 | rank `< prefetch_pilot_k`, and not already resident | `prefetch_pilot_k` (+ per-layer override) | 8 | 8 |
| RAM fetch, l+2 (enable) | depth must be 2 | `prefetch_pilot_depth` | 1 (off) | 1 (off) |
| RAM fetch, l+2 (count) | rank `< prefetch_pilot_k_depth2` | `prefetch_pilot_k_depth2` | 8 | 1 |
| VRAM promote, general pool | rank `< ram_vram_promote_top_k` (static) | `ram_vram_promote_top_k` | 0 (off) | 2 |
| VRAM promote, general pool (alt) | live per-rank accuracy `>= ram_vram_promote_adaptive_threshold` with `>= ram_vram_promote_min_samples` observations, contiguous-prefix | `ram_vram_promote_adaptive` / `_threshold` / `_min_samples` | off / 0.30 / 30 | off |
| VRAM promote, GPU1 slice (enable) | `predict_gpu1_rank_start >= 0` | `predict_gpu1_rank_start` | -1 (off) | -1 (off) |
| VRAM promote, GPU1 slice (window/cap) | `rank_start <= k <= rank_end`, `<= cap_per_layer`/transition | `predict_gpu1_rank_end` / `predict_gpu1_cap_per_layer` | -1 / 1 | 4 / 1 (inert — enable gate is off) |
| Eviction protection | resident this transition (l+1, and l+2 if its own fetch cutoff applies) | *(unconditional — no flag)* | always on | always on |

The two mutually-exclusive VRAM-promote modes (static top-k vs. adaptive)
are enforced by `EngineConfig`'s own validation at load time — setting both
disables both with a startup warning.

## Cost model when a prediction is wrong

There is no explicit "failure" path — the whole system is speculative and
self-correcting, not retry-based:

- A `needs_fetch` hint that turns out unnecessary: real disk bandwidth
  spent for nothing, but bounded by `claim_or_wait`'s collision check
  (`claims_beaten_by_decode` costs zero I/O) and by l+2's own smaller,
  separately-tunable cutoff.
- A correctly-fetched-but-never-used RAM slot: no extra I/O by itself, but
  occupies cache capacity until evicted, which can indirectly evict
  something genuinely needed soon. Measured directly as "Unified RAM
  waste" (~4% at l+1-only in the last clean measurement, ~7.9% when l+2 was
  on — see CLAUDE.md's depth-2 A/B).
- A wrong VRAM promotion bet: never a disk re-read, because promotion is
  always COPY semantics — the RAM fallback survives. Worst case is a RAM
  hit instead of the free VRAM win.

## Open items

- No config flag exists to disable the Stage 3 eviction protection for an
  A/B — it's unconditional. Worth adding if further tuning needs a clean
  on/off comparison.
- `ram_vram_promote_top_k=3` was only tested at a 40-token prompt (where it
  regressed) and never retested at 80 tokens (where `top_k=2` beat
  `top_k=1`) — the true optimum for this hardware/model pairing is not yet
  fully pinned down.
- Depth-2 (`prefetch_pilot_depth=2`) has not been re-tested since Stage 3's
  eviction protection was added; its earlier-measured higher waste rate
  (7.9% vs. l+1's 4.0%) is exactly the failure mode protection targets, so
  the earlier "not a throughput win" verdict may not still hold.
