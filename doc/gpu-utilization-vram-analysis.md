# GPU utilization / VRAM analysis: RTX 3090 (ExpertVault) vs RTX 5060 Ti (CoreEngine)

Investigation date: 2026-08-16. Config under test: `configs/glm52_3090_5060_sdd2.toml` (GLM-5.2,
`UD-IQ2_M_align`). Observed symptom: GPU 0 (3090) runs noticeably busier than GPU 1 (5060 Ti)
during generation (~45% vs ~20% utilization by casual `nvidia-smi` observation).

## 1. Architecture: the two GPUs have different jobs, not a shared workload

- **GPU 0 (RTX 3090, "ExpertVault")** — 24GB card (25.30GB reported by CUDA). Holds the main
  19.5GB MoE expert cache (`moe_cache_gb = 19.5`, `role = "moe"`) plus a 256-slot active-layer
  area (3.77GB) for speculative decode-time pre-warming. No dense weights, no KV cache.
  **Handles ~94-99.7% of all routed-expert FFN compute** (of 1410-1460 total VRAM cache slots
  across both GPUs, only the small dual-role slice on GPU 1 sits outside this pool).
- **GPU 1 (RTX 5060 Ti, "CoreEngine")** — 16GB card (16.65GB reported by CUDA). Holds all dense
  weights (attention Q/K/V/O projections, embeddings, shared-expert FFN), plus a small "dual-role"
  MoE expert-cache slice (currently 1.3GB) for local hits that avoid a PCIe round-trip to GPU 0.
  **Handles all dense/attention GPU dispatch** (`dense_gpu`, `attn_qabsorb_gpu`, `attn_score_gpu`
  paths) plus that thin sliver of expert hits.

## 2. Why utilization is skewed: workload split + no cross-GPU overlap

From a real `[REQUEST]` line (40 decode + ~13 prefill tokens, ~60-65s wall time):

| bucket | wall time | share | GPU |
|---|---|---|---|
| `expert=` (MoE FFN + cache mgmt) | ~45-49s | ~75% | GPU 0 |
| `attn=` | ~11-17s | ~19-25% | GPU 1 (mostly) |
| `dense_gpu=` (subset of `attn=`) | ~9-10s | ~15-16% | GPU 1 |

Dispatch is essentially **sequential** on the decode thread: attention (GPU 1) finishes, *then*
routing + expert FFN (GPU 0) runs, with no cross-GPU pipelining (confirmed elsewhere in this
project's own investigation history: 0% cross-stream contention measured — nothing genuinely
overlaps). So each GPU is either fully busy or fully idle at any given moment; GPU 0's "on" window
is simply ~4x longer than GPU 1's because expert FFN dominates wall-clock, which is what produces
the utilization gap over a sampled window.

**GPU 0 itself is not compute-bound either**, worth knowing before chasing more raw GPU throughput
on either card. Of the `expert=` bucket, actual GPU kernel execution (`gpu cmp=`) is only ~7s.
Most of the rest is real H2D bandwidth (active-layer pre-warm transfers, ~18-22s — already
extensively investigated elsewhere in this project and found to be genuine, non-fixable
bandwidth-bound work) and CPU-side disk-read waiting (`ram_scan`/`phase_b`/`phase_c`, ~10-13s).

## 3. Real VRAM breakdown for GPU 1 — the config numbers are not accurate

The config's `dense_gb`/`kv_cache_gb` values are planning labels, not enforced reservations.
Pulled from the engine's own `[VRAM-BREAKDOWN]` log output (byte-accurate, not estimated) at the
live config (`moe_cache_gb = 1.3`):

Real card capacity: **16.65 GB** (not the nominal "16.0" in the config).

| component | config says | real allocation |
|---|---|---|
| dense pool (weights) | `dense_gb = 12.0` | **12.9315 GB — ~8% over the configured budget** |
| &nbsp;&nbsp;↳ `attn_output` (78 tensors) | — | 5.411 GB (largest single piece) |
| &nbsp;&nbsp;↳ `other_attention` (q/k/v/kv_a proj) | — | 3.751 GB |
| &nbsp;&nbsp;↳ `shared_expert` (shexp) | — | 2.078 GB |
| &nbsp;&nbsp;↳ embedding + LM head | — | 1.190 GB |
| &nbsp;&nbsp;↳ leading dense FFN | — | 0.497 GB |
| &nbsp;&nbsp;↳ norms | — | 0.005 GB |
| `attn_output` BF16 fast-path | (implicit, would shrink the row above by ~50%) | **0.000 GB — 0/78 layers converted.** ~~Needs a 12.93GB worst-case reservation it will never get~~ CORRECTED (§5.1 below): blocked by a hard type gate, not VRAM budget — `attn_output.weight` is Q5_K on disk, and this fast-path only accepts F32-sourced tensors. No amount of freed VRAM changes this. |
| KV cache | `kv_cache_gb = 3.0` | **0 GB on GPU.** `[GRAPH] KV cache: host-only (0.746 GB) — GPU KV mirror unused by the live attention path, not allocated`. The 3GB budget doesn't correspond to any real GPU1 reservation this run — MLA's compressed latent KV lives host-side. |
| dual-role expert cache | `moe_cache_gb = 1.3` | 1.300 GB — this one *is* enforced exactly (contrast the log's `"(moe_cache_gb enforced)"` annotation on GPU 0, which has no equivalent for `dense_gb`) |
| CUDA ctx/cuBLAS/scratch (static) | — | 0.601 GB |
| **static total (startup snapshot)** | ~16.3 GB implied (12+3+1.3) | **14.83 GB / 16.65 GB, 1.82 GB "free"** |

### Live, not just startup

The static snapshot above is taken right after model load, before any tokens run. Sampling
`nvidia-smi` through a whole real request instead:

- **GPU 1 peak real usage: 15.51 GB / 16.65 GB (93%) — only ~1.14 GB genuinely free at peak**,
  tighter than the 1.82GB the startup snapshot implies.
- The ~0.7GB gap is dynamically-growing scratch (`ensure_buffers` for the qabsorb/score GPU
  paths, `GpuExpertScratch::ensure` for expert-FFN) that only claims its real footprint once
  generation is actually running.

## 4. Experiment: grow GPU 1's dual-role expert-cache slice

Hypothesis: since GPU 1 sits at ~1.8GB "free" per the startup snapshot, growing
`moe_cache_gb` from 1.3GB toward that ~1.8GB ceiling (or beyond) would shift more expert-dispatch
hits onto the otherwise-idle GPU 1, raising its utilization.

Method: real hardware A/B, `freeze_usage_stats = true` (fixed routing ranking across runs, no
usage-file drift), same prompt (`"Write a python hello world program."`), same
`--log-path`/config, `nvidia-smi --query-gpu=index,utilization.gpu,memory.used -lms 500` sampled
through the entire session (load → generate → shutdown).

| | baseline (1.3GB) | 1.8GB | 2.5GB |
|---|---|---|---|
| overall tok/s | 0.68 | 0.62 (-9%) | 0.56 (-18%) |
| decode tok/s | 0.85 | 0.77 | 0.74 |
| prefill tok/s | 1.08 | 1.07 (~flat) | 0.77 |
| GPU1 avg util (whole session) | 11.9% | 11.5% (flat/slightly down) | 9.5% (down) |
| GPU1 peak real mem | 15.51 GB | 15.50 GB (barely moved) | — |
| CUDA `out of memory` errors on device 1 | 0 | 0 | **593** |
| CPU expert-fallback calls | 118 | 126 | 432 |

**Result: conclusively negative at both tested sizes, not just a wash.** Neither run raised GPU1
utilization — the metric the change was meant to improve moved in the wrong direction both times.

**Root cause (2.5GB run, directly confirmed via 593 real `[ERROR] ... out of memory` log lines,
not inferred):** GPU 1's free margin is shared headroom that several *dynamically-growing* scratch
buffers depend on at runtime — dense attention scratch (`ensure_buffers`, used by the
qabsorb/score GPU paths) and the dual-role slice's own expert-FFN scratch
(`GpuExpertScratch::ensure`). Handing more of it to the static `moe_cache_gb` pool left those
dynamic buffers unable to grow, so they started failing allocation mid-run, forcing hundreds of
experts onto the slow CPU fallback path — which cost far more than the bigger cache slice could
ever save.

The smaller 1.8GB bump avoided outright OOM but still regressed (-9% overall throughput, GPU1
utilization flat-to-down) — evidence this card genuinely has no spare margin for this pool at any
tested increment, not just that 2.5GB specifically was too aggressive.

**Conclusion:** reverted to `moe_cache_gb = 1.3` (the pre-experiment value). Do not raise this
again without first freeing *real* margin — shrinking `dense_gb`/`kv_cache_gb`'s actual footprint
(e.g. funding `attn_output`'s BF16 fast-path would both shrink the dense pool by up to ~2.7GB
*and* likely speed up `dense_gpu`'s dominant `cmp=` cost, a plausibly better use of any freed VRAM
than the expert-cache slice) — not by reallocating the sliver of margin that already exists.

## 5. GPU 0's own breakdown, and where `shared_expert`/norms actually live

Real card capacity: **25.30 GB** (not the nominal "24.0" in config). Straight from the engine's
own `[VRAM-BREAKDOWN]` log:

| component | size |
|---|---|
| MoE expert cache pool (`moe_cache_gb = 19.5`, enforced) | **19.500 GB** |
| &nbsp;&nbsp;↳ of which: active-layer area (256-slot speculative pre-warm) | 3.775 GB (a sub-region inside the same allocation, not separate) |
| CUDA context / cuBLAS handles | ~0.279 GB |
| **total used** | **23.55 GB / 25.30 GB (93%), 1.74 GB free** |

**GPU 0 holds no `shared_expert`, no per-layer norms, no router weights at all.** Those tensors
land on GPU 1 instead — confirmed directly in GPU1's own `[VRAM-BREAKDOWN]` dump (§3 above):
`shared_expert (shexp): 2.078 GB (225 tensors)` and `norms: 0.005 GB (312 tensors)` are both listed
as part of GPU1's dense pool composition. The split isn't "MoE stuff → GPU0, everything else →
GPU1" — it's **routed vs. always-active**: GPU 0 only holds tensors that are *conditionally*
fetched per token based on top-k routing (the config's GPU0 block explicitly sets
`kv_cache_gb = 0.0, dense_gb = 0.0`). GPU 1 holds everything computed on *every* token regardless
of routing — attention, embeddings, norms, the router itself (`gate_inp`), and the "shared"
expert, which despite the name is functionally a dense layer (always active alongside whichever
routed experts get picked, not part of the top-k selection).

Live `nvidia-smi` sampling during generation showed GPU0 peaking at **21.94GB**, about 1.6GB
*below* the 23.55GB the startup log reports as allocated — likely the `cudaMalloc`'d pool
reserving virtual capacity up front while physical pages aren't all touched/committed yet, not a
sign anything is actually smaller than logged. Not chased further.

### If `shared_expert`/norms/dense weights didn't fit in GPU1's pool

Not hypothetical — this is a real, already-implemented fallback path (`src/main.cpp`'s dense-offload
block + `dense_dispatch.hpp`), just not one that's currently triggered (see below).

- Dense tensors fill GPU1's fixed `dense_gb` pool **largest-first, deliberately**: "prioritize
  keeping large, expensive-to-CPU-dequant tensors ... on GPU over small ones ... cheap on CPU
  either way." Fill priority for this model: `attn_output` (5.41GB) → `other_attention` (3.75GB) →
  `shared_expert` (2.08GB) → embedding+LM head (1.19GB) → leading_dense_ffn (0.50GB) → `norms`
  (0.005GB, dead last).
- If the pool doesn't fit at full size, the engine backs off to 95% of real free VRAM and retries;
  whatever individual tensors still don't fit — smallest/last in fill order first — simply stay
  **host-resident** rather than failing to load.
- At inference time, every dense call site (`dense_proj()`, `rmsnorm_proj()`) checks the tensor's
  GPU pointer and falls through to a **real, correct CPU AVX-512 kernel** if absent
  (`cpu_quant_matmul`, with dedicated fused paths for this model's actual dense quant types —
  Q5_K/Q6_K/Q8_0 — not the generic "unsupported type, zero the output" case). Correctness is
  identical either way; only speed changes.
- **Norms are essentially risk-free** either way — tiny F32 tensors, CPU fallback is a direct
  reinterpret of the host bytes, O(D) per row, negligible cost on CPU. That's exactly why they sit
  last in fill priority: losing GPU residency for norms costs almost nothing.
- **`shared_expert` is the one that would actually sting.** Unlike routed experts (sparse — only
  some tokens touch a given expert), the shared expert runs on *every* token, so a CPU fallback
  there adds a real, recurring per-token cost, not a rare one.
- On every run tested so far, none of this has triggered: `"[CUDA] Dense offload: 861 tensors on
  VRAM"` with zero `offload_failed` — the full 12.9315GB pool fits, so `shared_expert` and every
  norm are 100% GPU1-resident right now.
- **This is a separate allocation from, and unaffected by, the dual-role slice experiment in §4.**
  The dense pool (with `shared_expert`/norms inside it) is a one-time static allocation committed
  at startup, *before* the dual-role expert-cache slice is even sized (log order: "after dense
  tensor offload" precedes "after CoreEngine dual-role expert cache"). What broke in the 2.5GB
  test was a different, later-allocated category — per-call runtime scratch buffers
  (`ensure_buffers`, `GpuExpertScratch::ensure`) that grow dynamically during generation — not the
  static dense weights.

### 5.1 Correction: `attn_output`'s BF16 fast-path is blocked by type, not VRAM budget

An earlier pass through this doc (and the conversation that produced it) framed `attn_output`'s
0/78 conversion rate as an *underfunded* budget problem — "needs a 12.93GB worst-case reservation
it will never get." That's wrong, found by actually reading `upload_dequantized_bf16`
(`cuda_manager.hpp`) rather than inferring from the budget log line alone:

```cpp
// HARD TYPE GATE: only genuinely lossless-relative-to-storage F32
// source data may be rounded to BF16 here.
if (qt.meta.type != GGUFType::F32) return false;
```

**`attn_output.weight` is Q5_K on disk for this model** (this project's dense tensors are
Q5_K/Q6_K/Q8_0, never F32) — so this check rejects every candidate before the VRAM-budget math
even runs. The "net budget was 0.7476 GB, worst-case pool reservation 12.9315 GB" figures in the
startup log are real numbers the code computes, but they're not why 0 layers convert — 0 layers
would convert with *any* budget, including infinite. `gate_inp`'s fast-path, by contrast, hits
75/75 (100%) precisely because `gate_inp` genuinely is F32 in this checkpoint — the deciding
factor is source type, not size or available headroom. The gate exists deliberately: converting an
already-quantized tensor here would silently discard that format's own encoding on the way through
an F32 detour, a real precision/behavior change the gate is designed to prevent.

**Conclusion: there is no lever here at all** — not a small one, not a hard-to-reach one. No
config change, no freed VRAM, no `dense_gb`/`kv_cache_gb` shrinking makes this fast-path apply to
this model. §7's "fund the BF16 fast-path" option (below) is closed for this reason, not sized.

## 6. Data flow: what happens where around a MoE layer, and why nothing overlaps

Traced the real per-layer sequence in `execution_layer.cpp`/`attention_layer.cpp` around a routed
MoE layer (layer 5, say) to answer where each piece actually executes:

1. **`attn_norm`** (pre-attention norm) → `rmsnorm_proj()`, dispatched to **GPU1** if resident
   (currently yes, for all 312 norm tensors).
2. Attention compute → mostly **GPU1** (`dense_proj` for Q/K/V/O projections; the CPU-only
   `score` sub-phase is the one documented exception, see CLAUDE.md's attention timing entries).
3. Residual add → **CPU** (plain elementwise add on the host `hidden_states` buffer).
4. **`ffn_norm`** (the norm that feeds *both* the router and every expert's FFN input) → **always
   CPU**, unconditionally. Traced directly in `execution_layer.cpp`: this one calls
   `rmsnorm_row()` directly, never through the GPU-dispatch path at all, regardless of whether the
   weight is GPU1-resident. (A code comment there references a past S>1 prefill bug tied to this
   exact call — looks like a historical implementation detail rather than a deliberate
   architecture choice, but that's what actually runs.)
5. Router (`ffn_gate_inp`) → `dense_proj`, **GPU1** if resident (currently yes, BF16-converted).
6. Top-k selection → **CPU**.
7. **Routed-expert FFN → GPU0**, consuming the CPU-computed `ffn_norm` output. Results come back
   via `accumulate` — **CPU**, AVX-512 weighted combine of GPU0's downloaded outputs.
8. **Shared-expert FFN** (`ffn_gate_shexp`/`ffn_up_shexp`/`ffn_down_shexp`) → `dense_proj`,
   **GPU1** if resident, reusing the *same* already-computed `ffn_norm` output from step 4 — no
   new norm needed here.
9. Residual add → **CPU** — this becomes layer 6's `hidden_states`.
10. Back to step 1 for layer 6: **`attn_norm` → GPU1**, if resident.

**The key structural finding: neither GPU talks to the other directly.** `dense_proj`/
`rmsnorm_proj` take host `float*` pointers and do a full H2D-upload → compute → D2H-download round
trip internally, on *every* call, regardless of which GPU it targets. So layer 5's real flow is:
GPU0 (expert FFN) → **host** → GPU1 (shared expert) → **host** → CPU (residual add) → **host** →
GPU1 (next `attn_norm`) → ... — host RAM is the hub for every single op, on every layer, with no
peer-to-peer GPU0↔GPU1 transfer at any point.

This is the concrete mechanism behind §2's "sequential, not overlapped" finding: as long as every
step is bookended by a host round-trip, GPU1 has nothing to do while GPU0's round-trip is in
flight, and vice versa — which is exactly why option 1 below (cross-GPU pipelining) is a real
architecture change, not a scheduling tweak.

### Sizing the shared-expert/routed-expert overlap (2026-08-16)

Before committing to the thread-safety work the overlap idea needs, sized it with a real
measurement rather than estimating from the existing `shared_expert=` bucket alone: added a
temporary nested `PERF_SCOPE` isolating just the three `dense_proj` (GPU) calls from the CPU-only
SiLU/`elem_mul`/clamp activation between them, rebuilt, ran once, then reverted the instrumentation
(diagnostic-only, not meant to persist).

**Result: 98.9% of the bucket is genuine, overlappable GPU-dispatch time.**

```
Shared-expert GPU1 dispatch (of 2.215s total): 2.189s (98.9%) is the three dense_proj calls
alone -- an UPPER BOUND on how much could be hidden behind GPU0's routed-expert Pass A/B if run
concurrently on a separate thread; the remaining 0.025s (SiLU/elem_mul/clamp activation + the
final accumulate) is CPU-only and would stay on the critical path regardless of threading.
```

This is a better result than the cautious estimate in §6 assumed — almost none of the bucket is
CPU-bound activation that would eat into the achievable overlap. If the thread-safety work (separate
scratch buffers for the shared-expert path, join before the residual `+=`) is implemented, the
realistic ceiling is close to the *full* ~2.2s bucket (~3-4% of a ~60-70s request), not a
fraction of it — the risk in pursuing this is now mostly about implementation correctness (the
thread-safety/scratch-buffer work), not about the win being smaller than it looks once CPU
overhead is accounted for.

Not yet implemented — this was a sizing measurement only, run once. The dominant remaining costs
in a real request are still `expert=`'s own `active_layer_prewarm_issue` (~19s, already
extensively investigated elsewhere in this project and found to be genuine bandwidth-bound work,
not fixable) and `phase_b`/`disk=` (real disk reads), both far larger than this ~2.2s opportunity
— worth keeping in perspective before investing in the thread-safety work.

## 7. Strategies going forward, ranked

1. **Cross-GPU/cross-stage pipelining (highest payoff, real engineering effort, not attempted).**
   Since attention (GPU 1) → route → expert FFN (GPU 0) currently serializes, overlapping token
   N's expert FFN on GPU 0 with token N+1's (or the next layer's) attention on GPU 1 would raise
   *both* GPUs' utilization **and** cut wall-clock time — not just rebalance the split. Flagged
   in this project's own history as an open, unexplored investigation ("worth a dedicated future
   investigation into WHY so little genuine temporal overlap materializes in practice"). Needs a
   real async dispatch/pipeline redesign, not a config change.
2. **Fund `attn_output`'s BF16 fast-path — closed, not a VRAM problem.** See §5.1: this fast-path
   has a hard type gate (F32-sourced tensors only) and `attn_output.weight` is Q5_K on disk for
   this model. 0/78 layers convert regardless of budget — no config change or freed VRAM makes
   this apply. Ruled out conclusively, not sized to "small."
3. **Shared-expert/routed-expert GPU0/GPU1 sibling overlap — sized, real, small.** See §6's
   sizing subsection: 98.9% of the `shared_expert` bucket (2.19s of 2.22s) is genuine,
   theoretically-overlappable GPU-dispatch time, but the bucket itself is only ~3-4% of a real
   request's wall time. Real, implementable (separate scratch buffers + a second thread + a
   race-free join before the residual `+=`), but low priority given the effort-to-payoff ratio.
4. **Grow GPU1's dual-role expert-cache slice — closed, don't repeat.** Two independent real A/Bs
   (1.8GB, 2.5GB) both regressed throughput and GPU1 utilization at every tested increment. See
   §4 above.
5. **Accept the utilization split as largely inherent to this architecture.** MoE FFN compute
   dominates by construction (expert weights are the model), concentrated by design on the
   bigger/faster-PCIe card. Total wall-clock, not utilization balance, is the metric that matters
   — and the biggest remaining wall-clock levers (disk prefetch accuracy, active-layer transfer
   volume) are already at a wall per this project's own prior investigations without a genuine
   overlap redesign (option 1). By this point, options 2-4 are all closed or sized-to-marginal —
   the two live threads left are (a) whether concurrent multi-request serving is a real scenario
   (reopens option 1 in a genuinely viable form) and (b) accepting that single-request
   GPU-utilization-balance tuning has run out of levers on this hardware/model.

## 8. Conclusion (2026-08-16)

Confirmed with the user: concurrent multi-request serving is not a near-term scenario on this
hardware ("far away, unless we upgrade the hardware"). That closes option 1 as well — not just
sized down, genuinely off the table until/unless the deployment shape changes.

**This closes the GPU-utilization-balance investigation.** Final state of all four ranked
options: (1) cross-GPU pipelining — inapplicable to the current single-request deployment; (2)
`attn_output` BF16 fast-path — inapplicable to this model's quant scheme (§5.1), not a tuning
target; (3) shared-expert/routed-expert sibling overlap — real but small (~3-4% of wall time),
implementable if this code is being touched anyway; (4) grow GPU1's dual-role slice — tested
twice, conclusively negative (§4).

**No further GPU-balance-specific work is recommended on this hardware as configured.** The
~45%/20% utilization split observed at the start of this investigation is architecturally
inherent (§2) and not a symptom of any fixable inefficiency — it reflects MoE FFN genuinely
dominating wall-clock time, concentrated by design on the card built to hold it.

If more throughput is wanted next, the productive direction is different from anything in this
doc: per this project's own broader investigation history (see the project memory index),
prefetch lead-time/accuracy is the one remaining lever repeatedly flagged as still "live" across
multiple closed-out investigations (disk throughput itself is already at its measured ceiling;
active-layer transfer volume is already confirmed genuine, non-fixable bandwidth-bound work) —
worth a fresh look there rather than continuing to search for more GPU-balance-shaped answers on
this specific question.
