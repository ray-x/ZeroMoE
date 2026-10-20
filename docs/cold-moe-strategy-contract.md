# ColdMoEStrategy contract (R2)

Status: **specification** (2026-10-21). This is the contract the R2 strategy-adapters work
implements. It builds on `docs/graph-design-optimization-plan.md` §R2 and the lessons of
`docs/graph-design-conformance-2026-10-08.md`. No implementation yet — review this first.

---

## 1. Goal

One shared backend owns cold-MoE execution: route classification, expert residency (DDR/SSD),
transfer (H2D/D2D), and compute (CPU lane / GPU assist / cache-backed runner). A model family
supplies only **addressing** (which experts are cold, and where their weights live); it never
re-implements caching, staging, pilot, or the CPU lane.

Today the same job exists twice, entangled with the families:

| Concern | Full-graph path | Cache-backed path |
|---|---|---|
| route classification | `moe_partial_route.hpp` (`RouteClassification`, `HotColdSplit`) | same file (already shared) |
| residency | `memory/offload_store.hpp` (`ExpertResidency` + `VirtualArena`) | `RAMSafetyManager` + `GpuTopology` (native cache) |
| transfer | `EarlyReadyLane` (pinned bounce) | `backends/ggml/ggml_moe_transfer_manager.cpp` (staging ring) |
| compute | `EarlyReadyLane` CPU graph + GPU assist | `backends/ggml/ggml_moe_runner.cpp` (per-layer callbacks) |

`moe_partial_route.hpp` is already the shared **route** layer and is the model to follow: it owns
the *route slot* (not the expert id) as the unit of work, so a router selecting the same expert
twice still yields two slots each with its own probability contribution.

---

## 2. The interface

A `ColdMoEStrategy` exposes one `layer_invoke` for one layer, and is driven by a per-token
**invocation descriptor** that every branch (resident GPU, staged GPU, CPU, cache-backed) reads
from the same immutable object:

```cpp
// One token's routing for one MoE layer. Immutable for the lifetime of the invocation;
// produced once, consumed by every branch, freed after the last consumer.
struct ColdMoEInvocation {
  uint64_t generation;              // generation identity (see §3.5)
  int      layer;
  moe_partial::RouteClassification routes;   // route-slot ownership + exact-once weights (§3.2)
};

// The adapter a family supplies. It owns NO cache/staging/lane machinery.
struct ColdMoEAdapter {
  // Addressing only: which experts are cold this layer, and their gate/up/down bytes.
  // Returns the plane pointers for `expert`, or false when it cannot address it.
  std::function<bool(int expert, const void *planes[3])> expert_planes;
  // Begin making the invocation's cold experts resident (background); returns the
  // experts that are not ready yet. finish_load() blocks until they are.
  std::function<bool(const int32_t *experts, int n, std::vector<int32_t> &pending)> begin_load;
  std::function<bool()> finish_load;
};

class ColdMoEStrategy {
 public:
  // State machine per invocation. A route is Pending until its expert's planes are
  // fully resident and valid, Ready when computable, Error when unrecoverable.
  enum class State : uint8_t { Pending, Ready, Error };

  // Classify + begin residency. Returns false only on an unrecoverable error.
  virtual bool begin(const ColdMoEInvocation &, const ColdMoEAdapter &) = 0;
  // True once every owned route is Ready (or the invocation has failed).
  virtual bool ready(const ColdMoEInvocation &) const = 0;
  // Wait for readiness; returns false on error. Re-entrant on the decode thread.
  virtual bool finish(const ColdMoEInvocation &) = 0;
  // Free the invocation's held buffers; callable after the last consumer.
  virtual void release(const ColdMoEInvocation &) = 0;
  virtual ~ColdMoEStrategy() = default;
};
```

This is deliberately a **superset** of the callbacks `EarlyReadyLane` already exposes
(`begin_load` / `finish_load` / `expert_planes`); the lane becomes one implementation of the
strategy, and `ggml_moe_runner` + `RAMSafetyManager`/`GpuTopology` become another.

### Positioning within `ExecutionPolicy`

`ColdMoEStrategy` is the `MoEExecutionStrategy` component of the broader `ExecutionPolicy` from
`docs/graph-design-optimization-plan.md` §4.7 — the *last* of five independently selectable
policies, all of which live under the "execution plan" layer and vary independently of the model's
mathematics:

```
ExecutionPolicy
├── CachePolicy          (admission / eviction)
├── Predictor            (future experts + confidence + target layer)
├── PrefetchPolicy       (spend bandwidth/memory on predictions)
├── PromotionPolicy      (move experts into faster storage)
└── MoEExecutionStrategy (resident/staged/CPU → combined)   ← this contract
```

The four siblings are specified alongside this contract (their seams already exist: the expert
store + admission in `offload_store.hpp`/`expert_cache.hpp`, the predictor in the ridge/pilot
tables, prefetch in `async_prefetcher.hpp`, promotion in `GpuTopology`). `MoEExecutionStrategy`
depends on the capabilities those siblings expose, so the interfaces are kept narrow and the
supported combinations are validated explicitly.

---

## 3. The invariants

Each is a hard requirement; a strategy that violates one is not mergeable.

### 3.1 Route-slot ownership
The **route slot** (position in the router's top-k output), not the expert id, is the unit of
ownership. A router may select the same expert twice; each selected slot still carries its own
weight contribution. *Already enforced by `RouteAssignment::original_route_slot` and
`RouteClassification`.*

### 3.2 Exact-once weighting
Every selected route slot contributes its prefix-normalised weight **exactly once**, across all
branches, in original route-slot order. *Already checked by `RouteClassification::exact_once()`.*
The strategy must preserve it end-to-end: no branch may drop or double-count a slot, and the merge
sums in route-slot order (the reduction order is numerically significant — see the DeepSeek
reduction fix in `docs/deepseek_v41_decode.md`).

### 3.3 Pending / Ready / Error
An unavailable route stays **Pending** (or fails the request) — it is never computed against a
half-filled or zeroed slot. *Partial today:* `ExpertResidency` already marks "loading" experts and
waits for them (`begin`/`finish`, `is_loading`, `data_ready`), but the cache-backed path does not
expose the same states to graph code. The strategy must make `Pending/Ready/Error` visible to the
graph so an unavailable route fails or waits, not silently computes garbage.

### 3.4 Expert-buffer lifetimes
An expert's planes stay valid from the moment residency publishes them until the **last consumer**
releases them — never evicted or overwritten mid-compute. *Partial today:*
`hold_for_compute()` and `pending_copy_event` cover the full-graph path; the cache-backed
transfer manager has `DeferredVramHold` for the same purpose. One shared release discipline
replaces both.

### 3.5 Generation identity
Every per-token invocation carries a **generation** (epoch) so a route never binds to a slot that
was reused for a different expert between classification and compute. *Already present as
`slot_generation` in `RouteAssignment` and `stage_source->epoch`;* the strategy must thread one
generation through all three branches, not just staging.

### 3.6 Bounded memory
DDR, VRAM and pinned pools are **globally bounded** by the placement plan; a layer cannot allocate
a fresh `2*topk` bounce + pinned bank per invocation. *Not met today:* the conformance audit
(§"Reusable dispatch/completion graph segments") found per-layer pool allocation; this is the
largest open gap.

### 3.7 Shutdown ordering
CUDA-backed state (buffers, events, streams, thread pools) is released **before** the shared
backend it borrows from, on the device that created it. *Broken today:* the benign teardown crash
(`CUDA error: driver shutting down` in `DenseArenaOwner`/`StageRuntime` destructors) is exactly a
violation. The strategy's `release()`/`shutdown()` must encode this order.

### 3.8 State/lifetime contract (generation + Pending/Ready/Error)

The strategy is the single place that makes a per-token invocation's **generation** and **state**
visible. Both paths (lane and cache-backed) expose the same three states, keyed by generation:

- **Pending** — the invocation's cold experts are not yet all resident; compute must not read them.
- **Ready** — every owned expert is resident and valid; compute may proceed.
- **Error** — residency failed unrecoverably, *or* the invocation's generation no longer matches the
  path's current epoch (a route classified at generation `g` must never bind to a slot reused at `g+1`).

Generation source:
- **Lane (full-graph):** `EarlyReadyLane::calls`, advanced once per token at the end of `run()`; the
  staging epoch is `calls + 1`. `completed` is reset to false in the decode loop and set true at the
  end of `run()`, so `completed` *is* the Pending→Ready transition.
- **Cache-backed:** (defined when that path is wired; the `RAMSafetyManager`/`GpuTopology` epoch).

Rules, all paths:
1. `state(inv)` must be consulted before any compute reads an owned expert's planes; **Ready** is the
   only state in which compute may proceed.
2. `finish(inv)` blocks until Ready (lane: no-op — `run()` already drove residency on the scheduler
   bridge; cache-backed: waits on residency). It fails, rather than reporting Ready, on a stale
   invocation.
3. `release(inv)` is idempotent, and a no-op once the generation has advanced (a stale release after a
   slot reuse must not free a live buffer).

*Partial today:* the lane enforces "never compute against a missing expert" through `finish_load()`
blocking + `std::abort()` on load error, but it does not yet expose that as the strategy's
`Pending/Ready/Error`. The generation cross-check (`inv.generation != lane_->calls`) is the first
visible thread; wiring per-token `state()`/`finish()` into the decode loops is the remaining part of
migration step 2.

---

## 4. Migration (incremental, adapters first)

1. **Adapters over the existing paths** — do not rewrite either stack. Wrap
   `EarlyReadyLane` (full-graph) and `ggml_moe_runner` + `RAMSafetyManager`/`GpuTopology`
   (cache-backed) behind `ColdMoEStrategy`; the adapters call the existing code unchanged.

   The cache-backed path's residency is **split across two phases in two files** (grounded in
   `cold_moe_strategy_cache_backed.hpp`): RAM claim in `core/execution_layer.cpp`'s `ram_needed_k`
   loop (`claim_or_wait` → `Hit`/`Loading`/`Claimed`, then `wait_for_ram_slot`), and VRAM hold +
   transfer + compute in `core/batched_dispatch_pipeline.cpp`'s `run_ggml_moe_layer`
   (`hold_for_compute` → `begin_layer` → `run_layer*` → `release_hold`/`defer_vram_hold`). The
   adapter therefore wraps **both** phases and needs the `StreamingEngine` + per-layer dispatch
   context, not just `(runner, ram, topology)` — that gap is threaded in at the wiring step.

2. **Thread generation + State through the graph** — make `Pending/Ready/Error` and the
   generation visible to both paths; this is the correctness-critical step. State mapping for the
   cache-backed path: `RamClaimStatus::Hit` → Ready; `Loading`/`Claimed` → Pending; load error →
   Error (abort today).
3. **Unify residency + transfer** — collapse the two transfer paths (pinned bounce vs staging
   ring) into one bounded-pool implementation, then unify residency.
4. **Fix shutdown ordering** — encode release-before-backend order (also fixes the teardown crash).

Each step is gated (byte-identical logits + `bench/regression_gate.sh` per the tiered policy in
`docs/models/README.md`).

## 5. Acceptance

**Ling and one other family run through the same `ColdMoEStrategy` with no family-specific cache,
staging, or CPU-lane machinery.** Until that holds, the shared backend does not yet exist — only
its route layer (`moe_partial_route.hpp`) and the ridge-table dedup do.
