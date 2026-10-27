# Model-family framework: implement only what is unique

Status: **implementation in progress** on `feat/mimo-v26-flash`. The assembly refactor's rollout is
recorded in section 5; runtime-policy ownership is the next refactor in section 5d. Written
2026-10-22 after MiMo V2.6 Flash bring-up.

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
5. CUDA MoE decode uses the shared full-graph runtime. A family-specific graph
   adapter is required; model-specific equations belong in the adapter, not in
   a parallel layer-by-layer CUDA executor. `plan_decode_memory()` must reject
   unsupported architectures/topologies before allocation. Layered decode is
   available only for an `Insufficient` plan with byte counts and a reason;
   `Unknown` requires graph decode and `Unsupported` fails startup. See
   `docs/design/full-graph-model-onboarding.md` for the required graph gate.

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
values, not a full copy of the settings. *Decided: adopted.*

**Drift guard.** Following live defaults means a large change to a common default (say
`prefill_chunk`) can change a frozen profile's behaviour without anyone re-baselining it. So each
profile records, when frozen, a `baseline_default_hash` (computed from the table of common defaults, so parallel branches do not
conflict on a version number) and the baseline values of the *unpinned* fields it was measured with. At startup the resolver
compares them with the live ones. A mismatch is a preflight warning (an error in performance mode):
"profile measured with chunk=128, live default is 512 — re-tune or pin it." The change is therefore
visible, not silent, and fixing it is an explicit re-baseline. The drift check is part of R4's
decision, not an addition on top of it.

The hash covers only fields that can change performance or numerics. Benign fields (log level, file
paths, report formatting) are masked out, listed in one place, so changing them never trips the guard.
Enforcement matches the preflight policy: warning in development, **hard failure in performance mode**.
Only a failure honours R4's rule against silent execution changes; the user must accept the new baseline
or re-tune.

**Fallback if R4 rejects sparse profiles:** full snapshotting. A profile freezes every resolved default at
creation; the engine rejects a mismatch until an explicit `--rebase-profile` run updates the frozen file
and flags it for re-measurement. This keeps R4 intact but gives up automatic propagation (requirement 3):
every common-default change then needs a rebase pass over all profiles.

**Capabilities are a separate resolved table.** Capability flags (aliases the VRAM cache, wants
op-offload, supports the early-ready lane, supports prefill continuation, minimum batch) default to
**off / the safe mode**. A new family opts in explicitly, so an advanced path such as `op_offload` is
never on by inheritance. The startup table prints every resolved *capability* as well as every
tunable, each with its source, and any capability that fell back (e.g. lane requested but the policy
resolved to the cache-backed strategy) is printed as a fallback with the reason, at graph-build time.

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

0. **Decision (taken):** sparse pins with a defaults-hash drift guard (4.2) are adopted. R4 §5 is
   amended accordingly when R4 is implemented; the full-snapshot fallback is not used.
1. Settings struct + resolver + printed table; **no behaviour change** (defaults equal today's
   values; a test asserts each default against the value the code used before). Move hardcoded and
   env-read knobs in, MiMo and Ling first. Add the computed defaults hash.
2. Extract MiMo's `moe_ingraph` into a shared function and make Ling's `build_ffn` call it; the Ling gate
   must stay token-identical and in band.
   **Done in part** (`models/graph/moe_ingraph.hpp`): the order of operations around the block
   (classify placement, resolve the policy for the phase, record the adopted lane, warn when a requested
   lane is unavailable) is shared by MiMo and GLM/Ling. What stays per family is where the hot/cold
   tensors and route maps come from and the hooks. A single function that also builds placement and
   the cold tier (MiMo's whole `moe_ingraph`) cannot serve GLM yet: its staging, split-load, prefetch
   and per-GPU hot sets are not on the shared providers. Closing that gap is part of step 6.
3. Capability flags on the graph model (default off); remove the scattered family checks; print resolved
   capabilities and fallbacks at graph-build time. The `family ==` conformance rule (4.3) must be
   **enforced in CI before any GLM migration** (step 6), not after.
   **Done in part:** `GraphEngineCaps` (`aliases_vram_cache`, `ram_cache_with_preload`,
   `replaces_active_layer_area`, all default off) and `FullGraphModel::bind_vram_cache`; the engine's
   cache sizing and session setup use them instead of MiMo/Ling/GLM `ModelFamily` checks. A
   `[GRAPH-CAPS]` line prints every resolved capability at startup. `moe_conformance` fails when the
   remaining counts grow. Left, on purpose: the Ling native-graph cache flag (opt-in, changes another
   code path), per-family prompt templates (tokenizer format is family-specific), GLM-only checks in
   `glm_graph_in_use`, and the Qwen4Exp / DeepSeek-V4.1 checks (other families, other gates).
   **Measured:** MiMo's active-layer pool (16 slots/GPU) is unused by its in-graph path; switching it off
   (`replaces_active_layer_area=true`) frees 0.17 GiB per GPU, decode 13.7–15.4 tok/s, prefill 25/42/75/78 tok/s
   and tokens unchanged, so MiMo now sets it.
4. Quant table and the preflight checks. Run the Ling decode gate in the same change: the shared table
   must leave every existing tensor byte calculation unchanged (assert row bytes for each type against
   the old per-helper tables before deleting them).
   **Done in part:** `memory/quant_table.hpp` (block, bytes, MMQ eligibility per ggml type) now serves the
   GGUF loader's size formula, the offload store's row size and the MMQ check (the provider's and GLM's
   copies), and `tests/test_quant_table.cpp` checks all 31 rows against the linked ggml. It found two
   drifts: the offload store sized Q2_0 at 10 bytes per 32 elements (the real block is 18 per 64) and
   knew none of the legacy quants, and GLM's MMQ list lacked MXFP4. Preflight: `settings_findings`
   (decode lane threads below 10, op-offload with an unpinned arena, chunk below the minimum batch),
   a model-level check (tensor type missing from the table, max chunk below the minimum batch), logged
   as `[PREFLIGHT]` warnings; `graph_preflight_strict=true` in TOML (or
   `--graph-preflight-strict true`) aborts instead. Run-time checks (`models/graph/graph_health.hpp`, pure functions): expert coverage below 100 %,
   expert loads during decode after the first measured window, and idle VRAM above 4 GiB after the decode
   graph is built. MiMo reports them as `[PREFLIGHT] mimo2:` warnings (abort under
   `graph_preflight_strict=true`); the first real finding was 5.4 GiB idle on the 3090. Not wired yet for the
   GLM graph.
5. Attention/FFN/`LayerPlan` interfaces, prototyped on MiMo's attention (small, already near-isolated).
   **Prototype done on MiMo** (`models/graph/layer_plan.hpp`): `AttentionBlock` (`declare_inputs`,
   `set_inputs`, `build`), `FfnBlock`, `LayerPlan`, `PreNormResidualPlan` (cur += attention; cur += ffn,
   with a per-layer FFN choice and a backend-placement hook) and `build_layer_stack`. MiMo's attention
   (masks, KV row tables, subgraph), its dense layer 0 and its MoE layers are blocks, and its layer loop is
   `build_layer_stack`; the built graph is identical (same node and split counts, same tokens). What the
   prototype showed: the blocks need the family's graph object (MiMo downcasts `GraphCore` to its `Graph`),
   so a block interface over `GraphCore` alone is not enough yet; `positions` and the logits head still
   live in the family's `create_graph`; KV state allocation from a state spec (`state_spec`) is not
   done. Not tried: a family with another residual structure (DeepSeek-V4.1 hyper-connections), which
   is the real test of `LayerPlan`.
   **Follow-ups done:** blocks now hold the family's graph object, so the interfaces carry only the layer and
   tensors and nothing downcasts `GraphCore`; `declare_positions`, `build_output_head` and
   `KvLayerSpec`/`allocate_kv_cache` (KV state as data) are shared skeleton helpers MiMo uses. Still open:
   recurrent state (GDN, DSA indexer) has no helper, and a family with another residual structure
   (DeepSeek-V4.1 hyper-connections) is only shown to fit: `tests/test_layer_plan.cpp` (target
   `layer_plan_test`, CPU backend) runs the same blocks under `PreNormResidualPlan` and under a
   multi-stream plan ([D, HC, S] streams, blocks see a mixed row, outputs added to every stream) without
   changing the block interfaces. It uses a stand-in mixing, not DeepSeek-V4.1's math or code, which is
   under active work in another session and was not touched.
6. Migrate the remaining families one at a time (GLM last: ~10k lines).
   **Settings boundary for the GLM graph (done):** the tunables of the supported Ling/GLM configurations
   (offload tier, GPU assist, lane and staging switches, DMA pinning, usage file) come from the resolver
   for the concrete model family (`Family::Glm` for GLM-5.x/DSA, `Family::Ling` for Ling; Ling inherits
   the Glm row and uses the remaining `ZEROMOE_GLM5NEXT_*` graph aliases for the shared implementation).
   Direct `ZEROMOE_GLM5NEXT_/LING_` environment reads remain in feature-specific experimental or
   diagnostic (dynamic-cache heat, pilot ridge/score/lookahead tuning, GPU1 hot-set usage and layers, prefill
   prefetch variants, dumps, traces, CPU thread counts); `moe_conformance` ratchets that count so it can only
   shrink. `ModelFamily` references in `core/` are ratcheted the same way (112: Qwen4Exp and
   DeepSeek-V4/V4.1 special cases, GLM-only paths, prompt templates); the Qwen4Exp and DeepSeek-V4.1 ones belong to
   those families' own migration and were not touched, because DeepSeek-V4.1 is under active work in another
   session. The coverage and idle-VRAM checks run for the GLM graph too (decode-time misses are MiMo only).

Do not big-bang. Each step keeps the existing gates green.

## 5b. Status (2026-10-23)

Steps 0–5 are done, and step 6 is done for MiMo, Ling and the supported GLM configurations.

| Step | State |
|---|---|
| 0 R4 decision | sparse pins + defaults-hash drift guard adopted; `docs/execution-profile-spec.md` §5 and §10 amended |
| 1 settings resolver | done: MiMo, Ling, and the GLM graph's offload tier, lane and assist switches |
| 2 shared in-graph MoE glue | done (`moe_ingraph.hpp`); MiMo's whole `moe_ingraph` cannot serve GLM until its staging/split/prefetch paths use the shared providers |
| 3 capabilities | done for GLM-5.x, GLM-DSA, Ling, MiMo; `[GRAPH-CAPS]` logged |
| 4 quant table + preflight | done, including run-time checks for MiMo and the GLM graph (`graph_health.hpp`) |
| 5 attention/FFN/LayerPlan | done on MiMo (blocks hold their graph, shared helpers, KV as data); the multi-stream structure is shown to fit by `layer_plan_test` |
| 6 migrate families | MiMo, Ling, the GLM graph (`GlmHcPlan`/`GlmAttention`/`GlmFfn`) and the DeepSeek-V4.1 decode graph (`DsvHcPlan`) done; 81 experimental/diagnostic GLM env reads ratcheted; Qwen4Exp and DeepSeek-V4/V4.1 (112 `ModelFamily` references in `core/` ratcheted) left to their owners |

GLM/Ling on the interfaces (done 2026-10-23): `build_graph` in `ggml_glm5next_graph_build.cpp` runs its layer loop
through `GlmHcPlan` (hyper-connection or plain residual), with `GlmAttention` (KDA and MLA/DSA variants, their state
and the indexer carry-over) and `GlmFfn` (pre-norm, pilot staging, `build_ffn`) as blocks. The built graph is
identical (S=2: 11,507 nodes / 409 splits; decode: 11,061 / 273), Ling tokens identical, 10.3 tok/s. This is a real
hyper-connection structure on the `LayerPlan` interface. Recurrent state is data too: `StateSpec` /
`allocate_state_set` (tested); GLM still allocates its decode state from its own arena (`new_decode_state_tensor`),
so it does not use that helper, **by decision**: GLM's decode state (KV, KDA history/state, indexer caches) lives in one
pre-sized arena (`g.state`) owned by the engine and carried across decode-graph rebuilds, which is what keeps a
prompt's state alive between graphs. `allocate_state_set` allocates graph-owned state; replacing the arena with it
would change that lifetime and the memory budget for no gain. A family whose state is owned by its graph (MiMo's KV,
a new family) uses the helpers; GLM keeps its arena.

Not done, and why:
- DeepSeek-V4.1 decode graph on the interfaces (done 2026-10-23, `ggml_deepseek_v41_graph_decode.cpp`):
  `DsvAttention`, `DsvFfn` and `DsvHcPlan` (hyper-connection streams with the carried pre-mix, GPU or host nodes, the
  engram hook). Same decode graph (1,876 nodes / 161 splits), 64 greedy tokens identical to the master binary, and
  speed within noise (new 3.75 / 3.86 / 3.99, master 4.08 / 4.05 / 3.92 tok/s; band 3.5-4.6). Its prefill graph and
  its `ModelFamily` references in `core/` are not migrated.
- A frozen surveyed MiMo usage profile (the file knows 7,099 of 12,032 experts; with 100 % DDR coverage it only
  changes which experts are VRAM-hot) (needs a long calibration run).
- llama.cpp parity for MiMo: **passed 2026-10-23** (8 steps, tokens identical, cosine 0.9995-0.9999; `docs/models/mimo26-flash.md`).
- The unexplained one-off 900 ms/token stretch and 3.3 s tail chunk seen once during MiMo bring-up did not
  reproduce in more than a dozen later runs on the same prompts.
- Ling prefill with 512-token chunks and prefill op-offload: **tested 2026-10-23** on a 1,771-token prompt
  (`bench/ling/reference-audit/run.sh` with `LING_PROMPT_FILE`, 64 GiB DDR pool, pinned arena). One 1,771-token
  graph: 22.2 tok/s; 512-token chunks: 21.4 tok/s; 512-token chunks with `PREFILL_OP_OFFLOAD=1`: 21.6 tok/s
  (each 512-token chunk 19.0–19.2 s either way). So chunking alone does not help Ling, and the op-offload flag
  has no effect because Ling's graph path never engages scheduler op-offload (the cold experts run on the CPU
  lane at about 27 tok/s). Ling's IQ1_S experts are small, so copying them over PCIe could beat that; wiring
  op-offload into the GLM/Ling graph is a feature to build, not a setting to flip.

## 5c. What the framework does not own yet (review, 2026-10-23)

An honest limit of the work above: it standardised how a family is *assembled* and how its settings are
resolved, not where all the optimisation knowledge lives.

- **Staging and scheduling of cold experts is still partly per family.** `build_moe_block` owns branch
  construction and ordering and the early-ready lane fork, but each family supplies its plan and hooks: MiMo the
  offload miss node and the second-GPU branch; GLM/Ling the pilot/stage nodes and a custom cold-branch runner
  (split load, prefetched compact weights). Deciding *how* to stage cold experts is not yet one shared policy.
- **`ExecutionPolicy` selects, it does not yet plug in a stack.** It chooses an execution kind from placement and
  capabilities. The cache-backed implementation (`cold_moe_strategy_cache_backed.hpp`) is live through `CacheBackedMoeRuntime`:
  MiMo fallback, Ling-native and DeepSeek-V4.1 delegate to the existing shared native pipeline. The
  claim/hold/transfer/compute/release lifecycle is still owned inside that pipeline, not by an asynchronous strategy. So what the
  refactor shares concretely is the full-graph early-ready path (placement, offload tier, lane, assist); cache
  admission, prediction, prefetch and promotion remain separate per path.
- **Consequence for onboarding:** a new family gets the shared block, providers, settings, capabilities and
  preflight for free, but will still write its own staging hooks if it needs more than hot/cold + early-ready.

Closing this means wiring one cache-backed strategy to the real claim/hold/compute/release lifecycle and moving
GLM's staging behind the strategy interface; it is not started.

## 5d. Second refactor: own the runtime policy (review, 2026-10-23)

The first refactor made MiMo assemblable from shared parts; it did not make a new family fast by default, because
the speed came from backend setup, chunk/min-batch policy, cache and placement sizing, pinning/op-offload and
cold-expert staging, all still family-owned. The next refactor should standardise those runtime policies and use the
current MiMo implementation as the golden baseline. In order:

1. **Invert graph construction.** A `FamilyDescriptor` plus a shared graph builder: the family provides the spec,
   attention/FFN math, router math and placement data; the skeleton owns backends, state allocation, the layer loop,
   chunking and reporting. Add `AttentionBlock::state_spec(layer)` so KV/recurrent state is data, with an explicit
   opt-out for GLM's engine-owned arena.
2. **Replace `MoEBlockHooks` with named cold-MoE policies** (early-ready lane, GLM stage/ring, GLM split/prefetch,
   MiMo miss/GPU1, cache-backed callback, GPU op-offload). `ExecutionPolicy` constructs a `ColdMoEStrategy` stack;
   migrate GLM's `run_cold` first, then MiMo's hooks, and delete the raw lambdas.
3. **Move setup into the skeleton:** CPU backend/threads, prefill chunk and minimum batch, continuation, cache
   sizing, pinning, op-offload, second-GPU hot set, and the coverage/idle-VRAM/decode-miss preflight, derived from the
   model spec and hardware or declared as family overrides with a reason.
4. **Finish the settings layers:** typed overrides for all field types, profiles/TOML/CLI, the drift guard, and one
   prefill-knob model; remove the remaining direct GLM env reads.
5. **Real gates:** a frozen MiMo performance gate (decode >= 10 tok/s, a prefill floor, 0 decode store misses, 100 %
   coverage) and long-prompt/sliding-window parity over several prompts; turn `validate_vs_llama.sh` into a pass/fail
   gate. Placement is not token-neutral (the 12 -> 15 GiB cache change moved tokens from step 13: GPU and CPU
   kernels differ), so a gate needs a fixed placement or an explicit tolerance.
6. **Share the survey and the cache planner:** the route survey is now a shared side tap
   (`graph_route_survey.hpp`, MiMo and the GLM graph); next is a placement/cache planner that consumes its output and
   emits the TOML, replacing the manual 15 GiB / 14.5 GiB tuning.

## 5d+. Common VRAM policies (decided 2026-10-23)

These apply to every graph family by default, as skeleton behaviour, not as a MiMo setting:

- **Embedding table on CPU/DDR.** `embedding_device = "auto"` keeps `token_embd.weight` out of the dense VRAM pool for every model
  (every lookup reads the host copy); `"gpu"` forces the device copy. A family whose graph embeds on the GPU overrides
  `FullGraphModel::embedding_on_gpu()`. The recorded MiMo A/B completed; Ling, DeepSeek-V4.1, Qwen and GLM-5.3
  have not been re-gated for this change. Baseline reruns are deferred at the user's request.
- **Prefill VRAM area reuse (implemented, opt-in).** `PhaseExpertCache` adds a separate decode-hot allocation,
  released before prefill and refilled from pinned DDR afterwards. The permanent cache does not move. Both MiMo and
  GLM/Ling use it; graph leases and placement generations prevent stale CUDA graph replay. Keep the permanent cache's
  prefill reserve. See `docs/design/phase-cache.md` for settings, ordering and validation.

## 5e. Planned: a hardware probe that writes a factory config (requested 2026-10-23)

A script (`scripts/probe_hardware.sh` / a `zeromoe --probe` mode) that measures the machine and a model and emits a tuned
default TOML, so a copied or stale config cannot waste VRAM or RAM. It would collect: per-GPU total/free VRAM, PCIe
generation and width, and measured H2D bandwidth; CPU cores, AVX-512/VNNI, and DDR size and read bandwidth; the model's
real dense requirement and expert sizes (the numbers the preflight already computes, so `dense_gb` is derived, not
typed); and storage read speed (the USB SSD seed is the dominant start-up cost). Output: `moe_cache_reserve_gb` per GPU
(the VRAM-fit setting below), `host_ram_gb`, DDR pool, CPU threads, prefill chunk, assist slots, and the family's profile
pins, each with the measurement behind it. It builds on pieces that exist: `[VRAM-FIT]`/preflight sizing, the route
survey and its hot-set coverage curve, and the settings resolver's provenance table; it is the cache planner of the
second refactor (5d.6).

**VRAM fit (done 2026-10-23):** `moe_cache_reserve_gb` in a `[[gpus]]` entry makes the expert cache fill the VRAM that dense
weights, KV and workspace leave (dual-role GPU: free after dense - reserve; vault GPU: total - reserve), ignoring
`moe_cache_gb`, and logs `[VRAM-FIT]`.

## 5f. Runtime-memory continuation (2026-10-23)

Implemented in the MiMo worktree:

- `memory/resident_memory_policy.hpp` supplies embedding placement and expert-cache budgeting to the engine.
  Dedicated vaults and dual-role GPUs use the same budget calculation. Fit mode overrides a stale explicit
  cache size; an exhausted fit remains zero instead of falling back to legacy auto allocation. Vault allocation
  is capped by measured free VRAM. Reserves retain their existing bases: total VRAM on a vault, free VRAM after
  dense allocation on a dual-role GPU.
- `embedding_device` rejects unknown values, and cache reserves reject negative/non-finite/out-of-range values.
  `gguf_tool plan --embedding-device auto|cpu|gpu` uses the same embedding policy and reports the embedding
  bytes and payload-equivalent additional expert slots. Baseline configs and pin counts are not changed.
- MiMo and GLM/Ling coverage preflight measures unique expert identities ready in DDR or mapped in VRAM after
  seeding. It logs overlaps and missing identities; DDR capacity is reported separately. A large, empty pool or
  duplicated hot experts can no longer masquerade as complete coverage.
- DeepSeek-V4.1 prefill and decode host callbacks use `CacheBackedMoeRuntime`, preserving the native pipeline
  and its cache ownership. The direct-dispatch conformance exception is removed. This does not extract the
  native pipeline's asynchronous lifecycle or migrate DeepSeek's prefill layer structure.

Validation: Release engine build, resident-memory policy tests, settings/coverage tests, synthetic-GGUF advisor
test, CLI rejection of invalid placement/reserves, and conformance. No model baselines rerun.

Header-only estimates from the available model files (decimal GB; extra slots use the largest expert payload,
before pool alignment and workspace constraints):

| Model | Embedding VRAM avoided | Payload-equivalent extra expert slots |
|---|---:|---:|
| MiMo V2.6 Flash | 0.664 GB | 61 |
| Ling 3.1 Flash | 0.684 GB | 127 |
| DeepSeek V4.1 Flash | 0.217 GB | 16 |
| GLM 5.3 Flash | 0.520 GB | 74 |
| Qwen 3.8 Flash | 0.358 GB | 202 |

These are sizing advice, not measured throughput gains or permission to consume the eviction/staging budget.
The existing MiMo embedding A/B changed the fitted hot set; its token files differ, so it does not establish
bitwise parity for fixed placement. The logs show a dense-pool reduction from 6.503 GB to 5.839 GB.

**Phase-aware VRAM reuse is now implemented with a separate allocation**, rather than a tail of the permanent
`cudaMalloc`. `PhaseExpertCache` owns the shared refill/release code; `PlacementEpoch` refuses retirement while a
borrower exists. MiMo and GLM/Ling bind generations and preserve their existing KV/recurrent-state owners. The
separately releasable extension avoids retargeting permanent topology slots.

Hardware-cost calibration and named staging policies from section 5d remain open; a faster PCIe link does not
select a different strategy automatically. The extension stays opt-in pending steady-state performance gates.

### Phase-cache validation (2026-10-23)

- Release build, route ownership and placement lease tests, settings/default-snapshot tests, layer/state tests,
  and conformance pass. The validation harness also has a regression test proving a changed decode frame is
  detected even when the first (prefill) logits match.
- MiMo: `logs/phase-cache-mimo-decode-validation/result.json`; 3 requests, 18 teacher-forced steps including
  a 693-token chunked prefill. Minimum control/candidate logit cosine 0.9993794; repeated short request has
  identical logits across all 8 steps. The same 117 experts (1,070,393,600 bytes) refill in 89.9–96.3 ms, release
  in 0.6–2.6 ms, and decode store loads remain zero. Sampled peak VRAM: 23,628 MiB / 15,356 MiB.
- The test exposed an unused private MiMo embedding upload which defeated CPU embedding placement and caused
  long-prefill OOM. The graph now omits that unused tensor from accounting and uploads; long prefill passes.
- Ling with adaptive GPU assist enabled completed 3 allocation cycles, but its fixed-cache control itself
  changes decode logits after the long prompt. The current isolated check disables GPU assist in both control
  and candidate. The no-assist control reproduces all repeated-prompt logits exactly; candidate validation **failed**: minimum short-request cosine 0.9932799, long-request cosine 0.9732832
  (required 0.999). The phase placement itself reproduces repeated-request logits exactly. Root cause is unresolved;
  see `docs/handoff-2026-10-23-phase-cache.md`. Its standard DDR+VRAM profile has incomplete identity coverage and existing SSD demand loads;
  those are distinct from extension refills, which read only pinned, ready DDR sources.

No frozen performance bands were rerun, and these short correctness runs do not establish a steady-state
throughput improvement. Complete reproduction details are in `docs/design/phase-cache.md`.

## 6. Open questions

* R4 immutability vs propagation (4.2) — **decided**: sparse pins plus the defaults-hash drift check. R4's
  spec text still says "immutable" and needs the matching amendment.
* Drift check: warning in development, error in performance mode (proposed). Which fields are masked
  as benign needs an agreed list.
* Where recurrent-state carry across prefill chunks lives (today `supports_prefill_continuation`).
* Backend placement per block (today pinned by hand in each graph).
* Deprecation path for the family-prefixed env vars (`ZEROMOE_GLM5NEXT_*`, `ZEROMOE_LING_*`,
  `ZEROMOE_MIMO2_*`), which currently are the run-config layer.
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
