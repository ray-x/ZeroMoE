# Model-family framework: implement only what is unique

Status: **proposal — nothing in sections 4–6 is implemented.** Section 3 lists what already exists on
the `feat/mimo-v26-flash` branch. Written 2026-10-22 after the MiMo V2.6 Flash bring-up.
Builds on, and must stay consistent with: `docs/graph-framework-architecture.md` (layers),
`docs/cold-moe-strategy-contract.md` and `models/graph/execution_policy.hpp` (R2 strategy
interfaces), and `docs/execution-profile-spec.md` (R4 profiles, spec only).

## 1. Why this exists

MiMo V2.6 Flash went from **0.34 tok/s decode / 0.4 tok/s prefill** (first end-to-end run, 11-token prompt) to
**13.6–15.3 tok/s decode / 74–77 tok/s prefill (long prompts)** in one day (numbers and setup: `docs/models/mimo26-flash.md`).
Almost none of that was model-specific work. The causes of the slow phase were common settings or
shared-code traps that the new family silently did not inherit:

| Trap | Effect (measured) | How it hid |
|---|---|---|
| Decode CPU backend created with 1 thread (family code) | decode 3.3 → 15 tok/s once fixed | the family set its own value; Ling/GLM use 10/6 |
| `min_prefill_batch()` default 64 > MiMo's 16-token chunk | every prompt token after the first chunk ran as a one-token decode step; 516-token prefill 14.6 → 74 tok/s | silent fallback, one INFO line |
| Quant type missing from a helper (MXFP4 in the offload store's row-size table; in GLM's mmq check) | 28 load errors / NaN in prefill | each helper had its own list |
| Placeholder expert id repeated within a token on the batched MUL_MAT_ID path | NaN output | needs the negative-id skip, gated by a per-quant check |
| Expert coverage below 100 % | SSD reads during decode | warned, not enforced |
| 8–13 GiB VRAM idle | smaller hot set | no measurement of what the graph really needs |
| Arena not pinned while op-offload copies | ~7 GiB/s copies | pinning was tied to the assist ring |

The pattern: **a new family reimplements glue and misses a default.** The design below makes the glue
shared, makes defaults inherited, and makes divergence explicit and visible.

## 2. Requirements

1. A new model implements only its unique part (attention/state, residual structure, metadata,
   prompt format).
2. A specific model can be tuned without touching shared code.
3. A change to the common setup applies to all models by default.
4. A model can override the common setup — explicitly, with a recorded reason.

## 3. What exists today

| Piece | Where | State |
|---|---|---|
| `FullGraphModel` / `FullGraphModelBase<Derived>` / `GraphCore` | `models/graph/` | in master |
| Strategy interfaces (`ExecutionPolicy`, `MoEExecutionStrategy`, `ColdMoEStrategy`) | `models/graph/execution_policy.hpp`, `moe_execution_strategy.hpp` | in master (R2) |
| Shared MoE block (hot/cold branches, early-ready lane fork, merge, shared expert) | `models/graph/moe_block.hpp` | **this branch** |
| Hot-side placement provider (VRAM-cache aliasing, route maps, holds, byte verification) | `models/graph/aliased_expert_placement.hpp` | **this branch** |
| Cold-side provider (offload store config, DDR seeding, plane tensors, assist ring) and its runtime | `models/graph/offload_expert_provider.hpp`, `offload_expert_runtime.hpp` | **this branch** |
| Conformance gate (no direct `moe_forward_gguf`, no `ExpertStore`/`EarlyReadyLane` in family graphs) | `scripts/check_moe_conformance.sh`, `make moe_conformance` | **this branch** |
| Execution profiles keyed by model × hardware × workload | `docs/execution-profile-spec.md` | spec only |

Used by: GLM/Ling (block + providers, via `build_ffn`), MiMo (block + providers, via its own
`moe_ingraph`). **Not shared yet:** the glue around the block — placement setup, capability/policy
resolution, hook wiring, chunk/offload policy — exists once in GLM's `build_ffn` and once in MiMo's
`moe_ingraph`.

## 4. Design

### 4.1 What a family provides

```cpp
struct FamilyDescriptor {
  ModelSpec         spec;       // dims, per-layer types, router spec, quant types in use
  AttentionFactory  attention;  // layer -> AttentionBlock
  LayerPlan         plan;       // per layer: attention + FFN block + residual style
  ProfileOverrides  overrides;  // optional, sparse, each with a reason (4.2)
};
```

Interfaces (graph-build time only; no per-token cost):

* `AttentionBlock`: `state_spec(layer)` (KV heads/widths/window, recurrent state), `declare_inputs()` /
  `set_inputs()` for its own host inputs (positions, masks, KV row tables), and
  `build(x, layer, phase, S)`. The skeleton allocates state from `state_spec`.
* `FfnBlock`: a dense SwiGLU block, or the shared MoE block. A model may mix them per layer (MiMo layer 0
  is dense, layers 1–47 are MoE).
* `LayerPlan` owns the residual structure. DeepSeek-V4.1 hyper-connections keep several residual
  streams; an interface covering only attention and FFN would not fit it.

Everything else is the skeleton's: layer loop, MoE block, placement, offload tier, early-ready lane,
prefill chunking and continuation, op-offload policy, VRAM/coverage preflight, standard report.

### 4.2 Settings: one resolved value per knob, with a source

```
common defaults  →  family overrides  →  profile (frozen, measured)  →  run config (TOML / env)
```

* **Common defaults**: one struct with every tunable (lane threads, prefill chunk and minimum batch,
  op-offload policy and threshold, arena pinning, assist slots, cache headroom policy, …).
* **Family overrides**: optional fields. Each override carries a mandatory `reason`
  (e.g. GLM: "512-token chunks measured 90 vs 250 tok/s"). An override equal to the default is rejected
  as redundant.
* **Profile**: the R4 `ExecutionProfile` — validated, measured values for a key
  (hardware × model dims × quant signature × budgets × workload).
* **Run config**: TOML/env, for tuning one model or run without a code change.

At startup the resolver prints every knob with its value and source, e.g.
`prefill_offload  off  family:ling (reason: …)` / `moe_cache_gb[0]  12  toml`.

**Tension with R4, and the proposed resolution.** R4 §5 makes profiles *immutable once frozen*
so a re-tune never silently changes another run. Requirement 3 needs a change to a common default to
propagate. These agree only if a profile **pins just the fields it measured** and every other field
keeps following the defaults. A pinned field is frozen with its provenance (changed only by an explicit
re-baseline); an unpinned field follows the live default. So: profiles store a sparse set of pinned
values, not a full copy of the settings. *This needs a decision from the R4 owner before
implementation.*

### 4.3 Enforcement

* **Shared-code rule:** no `family ==` comparisons in `models/graph/` or the engine's cache sizing
  (today: Ling branches inside GLM files and in `engine_runner_cache.cpp`, plus the MiMo special case
  added on this branch). Extend `moe_conformance` to fail on it. Differences go through spec fields and
  capability flags on the graph model (aliases the VRAM cache, wants op-offload, minimum chunk).
* **Required hooks are pure virtual**; optional ones have defaults that are *logged* at startup.
* **Preflight on the resolved settings** (warn in dev, fail in performance mode):
  chunk < minimum batch; lane threads below the machine default; coverage < 100 %;
  VRAM headroom after load above a threshold; op-offload with an unpinned arena;
  a tensor quant type missing from the shared quant table; decode-time store misses > 0.
* **One quant table** (row bytes, mmq eligibility, kernel support) used by every helper, replacing the
  per-helper lists.
* **Per-family gates:** a correctness parity check and a performance floor (the Ling gate is the model:
  tokens identical, decode within a band). A change to a common default runs every family's gates; a
  family that regresses adds an explicit override with a reason instead of diverging silently.

### 4.4 Onboarding a new model

1. Detect the family and read hyper-parameters (existing step; see the init-order guard in CLAUDE.md).
2. Correctness first: block-level parity of the attention block against llama.cpp, then end-to-end.
3. Write the `FamilyDescriptor`. Inherit all settings.
4. Run the standard bring-up report (coverage, VRAM headroom, lane breakdown, decode-time misses,
   per-chunk prefill speed). These `[…-LANE]`, `[…-VRAM]`, `[…-PREFILL]` lines currently exist only for MiMo
   and must come from the shared block.
5. Plan caches and the DDR tier from measured VRAM and expert bytes (a planner emits the TOML); estimate
   the achievable tok/s from expert size, cold fraction and bandwidth before long runs.
6. Freeze what was measured as pinned profile fields; record the baseline in `docs/models/`.

### 4.5 What stays family-specific

Attention and recurrent/windowed state (MLA, DSA indexer, GDN, sliding window + sinks), residual
structure (hyper-connections), embeddings/head quirks, tokenizer and prompt format, and the router's
mathematics (sigmoid + bias, top-k normalisation).

## 5. Rollout

1. Settings struct + resolver + printed table; **no behaviour change** (defaults equal today's
   values). Move hardcoded and env-read knobs in, MiMo and Ling first.
2. Extract MiMo's `moe_ingraph` into a shared function and make Ling's `build_ffn` call it; the Ling gate
   must stay token-identical and in band.
3. Capability flags on the graph model; remove the scattered family checks; gate them.
4. Quant table and the preflight checks.
5. Attention/FFN/`LayerPlan` interfaces, prototyped on MiMo's attention (small, already near-isolated).
6. Migrate the remaining families one at a time (GLM last: ~10k lines).

Do not big-bang. Each step keeps the existing gates green.

## 6. Open questions

* R4 immutability vs propagation (4.2) — who owns the decision.
* Where recurrent-state carry across prefill chunks lives (today `supports_prefill_continuation`).
* Backend placement per block (today pinned by hand in each graph).
* Deprecation path for the family-prefixed env vars (`MOESTREAM_GLM5NEXT_*`, `MOESTREAM_LING_*`,
  `MOESTREAM_MIMO2_*`), which currently are the run-config layer.
* Whether other families benefit from the MiMo findings is **untested**. Candidates, in order: Ling
  prefill with 512-token chunks + pinned arena + op-offload (GLM already has the mechanism behind an
  off-by-default flag; Ling disables it in code); larger caches from measured VRAM; the MXFP4 fixes for
  any MXFP4 model.

## Appendix: MiMo V2.6 Flash measurements (this branch, RTX 3090 + RTX 5060 Ti)

| Prompt tokens | Prefill tok/s | Path | Decode tok/s (64 tok) |
|---:|---:|---|---:|
| 21 | 25.0 | CPU cold experts | 13.6 |
| 146 | 41.6 | CPU cold experts | 15.0 |
| 516 | 74.1 | GPU op-offload, 512-token chunks | 15.3 |
| 1022 | 77.1 | GPU op-offload, 512-token chunks | 14.6 |

Decode had 0 store misses/loads. Ling gate (short decode, 32 tokens): 10.07 tok/s with tokens identical
to the pre-refactor build. Not yet done: llama.cpp parity for MiMo, a frozen surveyed usage profile,
assist-ring A/B, a Ling rerun after the last edits.
