# Graph design optimization plan

Status: **proposed** (2026-10-21). This is the plan driving the follow-up refactor work. It
complements — and does not replace — `docs/graph-framework-architecture.md` (the framework
structure) and `docs/source-layout-2026-10-18.md` (the folder split and the "one implementation
per family" rule).

---

## 1. Why this exists

The engine has six supported model families (DeepSeek-Flash, Qwen 3.8 Flash, GLM 5.3, GLM 5.3
Flash, Ling 3.1 Flash, plus more planned) and is heavily dependent on CUDA graphs. The shared
framework has already been refactored twice — the 2026-10-11 `graph-framework` pass and the
2026-10-18 `source-split` pass — and the shared layer is in good shape. What remains painful is
**per-model tuning**: every new model starts from zero, and the knowledge from tuning one model
tends to be lost or silently overridden when the next model is tuned.

This plan separates the problem into:

1. **Architectural** fixes that reduce the "start from zero" burden (R1–R6 below).
2. **Process** fixes that stop tuning knowledge from being lost or overridden (the regression
   gate in `bench/regression_gate.sh` and the per-model playbook in `docs/models/`).

Both are perf-neutral by construction: they touch graph *build* / teardown / placement, not the
graph *replay* path (build is ~7 ms of a ~53 ms prepare, ~0.2 % of a request — see
`docs/source-layout-2026-10-18.md`).

---

## 2. Current state: what is already good

The shared layer, reached through `engine.family_ops.graph_model`, is solid and should be the
model for further work:

| Piece | File | Role |
|---|---|---|
| `GraphCore` | `models/graph/graph_core.hpp` | family-independent ggml handles, three `DeviceArena`s (`dense`/`expert`/`state`), host allocation helpers |
| `FullGraphModel` | `models/graph/full_graph_model.hpp` | the only thing the engine talks to; CRTP `FullGraphModelBase<Derived>` owns request validation + prefill/decode dispatch |
| `FamilyOps` | `models/base/family_ops.{hpp,cpp}` | single `ModelFamily -> "what does this family do"` table (attention fn, cache allocator, HC hooks, router spec) |
| `moe_routed_ggml.hpp` | `models/graph/moe_routed_ggml.hpp` | shared router / expert-FFN / top-k-sum builders |
| `ExpertStore` | `memory/offload_store.hpp` | model-agnostic SSD-offload DDR tier (`OffloadStore`, `VirtualArena`, `ExpertResidency`) |
| `EarlyReadyLane` | `models/graph/early_ready_lane.hpp` | shared CPU lane for cold experts (with Qwen-only staging extras) |
| `graph_config.hpp` | `models/graph/graph_config.hpp` | typed env readers + `[GRAPH-CONFIG]` logging |

---

## 3. Root cause: three responsibilities are fused

`docs/source-layout-2026-10-18.md` proved the *truly duplicated builder code is only ~20 lines*,
so the pain is **not** shared-code duplication. It is that three responsibilities that change for
different reasons are fused in one per-family graph builder:

| Part | Owns | Changes when |
|---|---|---|
| **Model definition** | layer sequence, attention, normalization, router mathematics, expert FFN, residual/HC, persistent state | model architecture changes |
| **Execution plan** | CPU/GPU placement, cache budgets, expert residency, staging, pilot & promotion policies | hardware, workload or optimization changes |
| **Graph runtime** | allocation, binding, capture, replay, synchronization, resource lifetime & teardown | the framework gains a reusable capability |

`ModelFamily` is used for all three at once, which is why Ling's `bootstrap()/prepare()/run()` in
`models/graph/full_graph_models.cpp` is a runtime `if/else` over
`bailingmoe3_{hybrid,native}_graph_enabled()`, and why each new model re-derives the same
offload/pilot/lane machinery. Concrete symptoms (each a copy a new model re-derives):

1. **Two cold-MoE execution stacks** — full-graph (`EarlyReadyLane` + `ExpertStore`) vs
   cache-backed (`backends/ggml/ggml_moe_runner.cpp` + `moe::` builders).
2. **Three pilot/prefetch implementations** — GLM (`PilotRidgeTable`), Qwen
   (`Qwen4ExpRidgeTable`), DeepSeek (`ZEROMOE_DSV41_PILOT_K`).
3. **Two shared MoE topology builders** — `moe_routed_ggml.hpp` vs `ggml_moe_graph_builder.cpp`.

---

## 4. The target architecture

One shared graph framework compiles a graph per model + execution phase; reuse comes from common
graph blocks, resource management and execution strategies. A model supplies its mathematical
differences and persistent state.

```
model definition ─┐
execution plan  ──┼──> shared graph compiler ──> graph instance ──> shared request runtime (prefill/decode/replay)
hardware budgets ─┘                                                             │
      ^                                                                         │
      └──────────────────────── measurements (feedback) ────────────────────────┘
```

### 4.1 Common unit = a graph block

A block has explicit inputs, outputs and state. Many models compose the familiar layer: input
transform → attention → residual/HC → normalization → router → routed experts (+ shared) →
residual/HC. Provide reusable blocks; a model controls ordering and may supply a custom region.

Each block declares: input/output shapes + dtypes; persistent state it reads/updates; scratch +
placement requirements; dependencies; whether it supports prefill, decode and chunk continuation.
A new model adds a unique attention block without re-implementing allocation, capture or request
lifecycle. *Starting points:* `GraphCore`, `FullGraphModel`, `FamilyOps` (extend incrementally).

### 4.2 A clear boundary around routed MoE execution

The **model** defines the mathematics (router → ids/weights → expert FFNs → weighted combination);
the **strategy** decides where/how (resident GPU + staged GPU + CPU → combined). The shared MoE
interface needs more than tensor handles:

| Model supplies | Strategy supplies |
|---|---|
| expert shapes, layouts, quantization | storage / cache access |
| activation + clamp semantics | CPU/GPU partitioning |
| router IDs + mathematical weights | transfers, readiness, synchronization |
| shared-expert computation | bounded staging resources |
| required output shape | execution + combination dependencies |

Preserve original route slots and apply each router weight exactly once; buffer ownership stays
valid through the final consumer. *Adapt both existing stacks to this interface, then consolidate
their internals as equivalence is demonstrated.* (This is `docs/cold-moe-strategy-contract.md`.)

### 4.3 Separate cache, pilot and promotion policies

Independently replaceable components: **expert store** (locate/load weights, residency/leases),
**admission/eviction policy**, **predictor** (future experts + confidence + target layer),
**prefetch scheduler** (spend bandwidth/memory on predictions), **promotion policy** (move into
faster storage), **executor** (compute routed work). Prediction stays speculative: real routing
drives required work, and demand loading must complete correctly on a miss.

### 4.4 Preserve graph stability while policies evolve

Divide policy decisions by when they take effect: at **graph preparation** (strategy, capacities,
placement, dependencies); during **replay** (inputs, IDs, weights, buffer contents); at a
**transition boundary** (placement/binding changes with sync + recapture). Promotion must know
whether the compiled graph supports stable slots with updated contents or a controlled rebuild.

### 4.5 Separate prefill and decode plans over the same model definition

Shared mathematics; different plans — prefill: batched expert work, streaming/batch reuse, chunk
sizing + continuation, larger temporary workspace; decode: sparse per-token work, residency +
latency, CPU/GPU overlap + prediction, stable small replay buffers. Define state handoff
explicitly: KV, recurrent, HC state and borrowed weight allocations survive the transition.

### 4.6 Make adding a model a bounded task

Onboard: map metadata/tensors → compose existing blocks (implement only genuinely new math) →
declare capabilities/state → select an existing execution strategy + initial profile → validate.
A new model inherits the cache manager, offload loader, pilot scheduler, staging pools and replay
lifecycle. Profiles keyed by hardware, shapes, quantization, budgets and workload.

### Mapping to the earlier R1–R6 tactics

The concrete file-level targets from the earlier plan remain valid and map onto §4:
**R1** → §4.1 (graph blocks) + §4.6 (onboarding); **R2** → §4.2 (MoE boundary) + §4.3 (policies);
**R3** → the "execution plan" layer + §4.4; **R4** → §4.6 profiles; **R5** → diagnostic cleanup
(separate); **R6** → §4.2 strategy selection.

### 4.7 Design patterns to evolve around

The codebase already uses most of these; the change is to make the *boundaries* explicit.

| Pattern | Present? | Role here |
|---|---|---|
| **Strategy** | yes — `FullGraphModel` / `FamilyOps` select behavior | make cache, pilot, promotion and MoE execution *independently selectable* strategies |
| **Composition** | partial — graph-building functions | `ModelProgram` = ordered blocks (norm, attention, residual/HC, router, MoE, …); models compose existing blocks and add new ones |
| **Builder** | yes — graph builders | one shared compile boundary: `GraphInstance compile(ModelProgram, ExecutionPlan, RuntimeResources)` |
| **Adapter** | yes — `full_graph_models.cpp` wrappers | wrap the two existing paths behind the common interface during migration |
| **Template Method** | yes — `FullGraphModelBase<Derived>` | keep it small (lifecycle reuse only); put new flexibility in composition + strategies |
| **RAII** | partly — owned objects + manual teardown | owned resource objects for leases, staging slots, workers, CUDA resources |
| **Registry/Factory** | yes — `full_graph_model()`, family resolution | resolve implementations once at model load |
| **Bridge** | no | model architecture × execution strategy vary independently |

#### The strategy hierarchy for optimization choices

```
ExecutionPolicy
├── CachePolicy          (admission / eviction)
├── Predictor            (future experts + confidence + target layer)
├── PrefetchPolicy       (spend bandwidth/memory on predictions)
├── PromotionPolicy      (move experts into faster storage)
└── MoEExecutionStrategy (resident/staged/CPU → combined)   ← ColdMoEStrategy today
```

A model uses a resident-GPU strategy on one machine and CPU/GPU offload on another, or swaps the
predictor, without touching its mathematical definition. Keep interfaces narrow; validate
supported combinations (some policies depend on capabilities the executor supplies).

#### Speculation with reconciliation (a recurring shape)

The speculative parts of the system all share one contract — **predict → verify → reconcile** —
and the *reconciliation* (not the prediction) is the part that must be specified:

| Mechanism | Predicts | Verifies (authoritative) | Reconciles when wrong |
|---|---|---|---|
| **MTP / token drafting** | future tokens | target model re-verifies each draft | rejected proposals dropped; speculative model state reset to the last accepted token |
| **Expert pilot** (`Predictor`) | experts likely needed | the layer's real router (top-k) | demand loading supplies the actual experts; predicted-but-unused prefetch is wasted, reclaimed by `CachePolicy` eviction |
| **Prefetch** (`PrefetchPolicy`) | what to fetch early | actual demand | wasted prefetch evicted |
| **Promotion** (`PromotionPolicy`) | which experts become hot | observed re-hit counts | mis-promoted experts demoted |

The rule that makes these safe is the same in every row: **a wrong prediction costs wasted work,
never a wrong result.** The verifier is always the authoritative path (the target model, the real
router, real demand), and the reconciler is what bounds the waste. `ExecutionPolicy`'s `Predictor`
documents this explicitly; the same shape should be recognized (not re-invented) wherever a new
speculation is added.

#### The concrete compile boundary

```cpp
GraphInstance compile(const ModelProgram &model, const ExecutionPlan &plan,
                      RuntimeResources &resources);
```

The builder combines mathematical blocks with placement + execution strategies. `GraphInstance`
owns the graph and its bindings. Model and strategy extension points run mainly during
*preparation*; per-token work reuses the compiled machinery.

**The biggest missing piece** is therefore the explicit separation between *model mathematics*,
*optimization policy*, and *runtime resource ownership* (RAII: a GPU resource's lifetime is not
established by object scope alone — destruction must respect outstanding work and borrowed
allocations). Those boundaries decide how much engineering is reusable when the next popular
model arrives.

---

## 5. Post-merge findings (2026-10-21)

After GLM/DeepSeek/Ling were merged into master, the duplication got *more* concrete, not less:

- **Registry grew to five classes** — added `GlmDsaGraphModel` (a copy of `Glm5NextGraphModel`).
- **Second cold-MoE stack landed** in `backends/ggml/` (cache-backed: `ggml_moe_runner` ~96 KB,
  `ggml_moe_backend` ~67 KB, `ggml_moe_transfer_manager` ~46 KB, `ggml_router_runner` ~24 KB).
- **Third pilot** in DeepSeek (`ZEROMOE_DSV41_PILOT_K`).
- **Page-reusing eviction** (`ZEROMOE_OFFLOAD_RETARGET`, `mremap MREMAP_DONTUNMAP`) landed at
  the *page* level; the VRAM-slot dynamic promotion ("topology retarget") is still not built
  (`docs/offload-retarget.md`, `docs/ssd-offload/10-*.md` §6).

---

## 6. Sequencing & dependencies

The recommended sequence, updated to the reviewer's priority order — common MoE interface →
state/lifetime contracts → reusable profiles → LayerPlan (after two architectures migrate) — with
current status:

1. **Repair the gate and capture reproducible profiles** — prerequisite for every gated step below.
2. **Common MoE execution interface** (`ColdMoEStrategy` + `ExecutionPolicy`) — done (R2 branch).
3. **Explicit state/lifetime contracts** (generation + Pending/Ready/Error, contract §3.8) — done
   (R2 branch).
4. **Migrate two architectures through the interface** — the full-graph lane is wired (GLM); the
   cache-backed adapter is grounded and awaits the gated extraction. This is the precondition for
   the broader LayerPlan.
5. **Reusable execution profiles** (R4) — the first-class deliverable below; next up.
6. **Broader LayerPlan** (R1's full layer table) — only after step 4 lands.
7. **Diagnostic cleanup** (R5) — a separate workstream, not on the critical path.

R3 (placement planning) and R6 (strategy selection) fold in where they fit: R6 with step 2, R3 with
the profiles and the two migrations.

### Tuning profiles (first-class deliverable)

Shared code alone does not preserve tuning knowledge. Each model's profile records, keyed by
**hardware, model dimensions, quantization, memory budgets and execution phase**:

- prefill throughput, **time-to-first-token**, decode throughput, **peak memory** (RSS + VRAM), and
  the **selected strategy** (placement / offload / pilot / lane / assist);
- validated on **short prompts, long prompts and chunk continuation**, not one decode length.

These profiles are the frozen baselines the gate compares against, and live in `docs/models/`.
The *machine-loadable* half of the same knowledge — the selected strategy + tunables as data the
engine loads — is the reusable **execution profile**, specced in `docs/execution-profile-spec.md`
(R4). The two stay in lock-step: a `docs/models/*.md` recipe freezes the measurement; the execution
profile carries the config that produced it.

---

## 7. Gates & acceptance criteria

Every structural change must pass, before it lands:

- **Qwen:** `bench/graph_framework_step.sh <name> <candidate>` — logits byte-identical, free tokens identical.
- **GLM:** `ARMS="lru lane" bench/graph_framework_glm_gate.sh <name> <candidate>`.
- **Throughput band (all models):** `bench/regression_gate.sh <candidate>` — decode tok/s within
  the frozen band in that script (and `docs/models/README.md`).
- **Native suite:** `zeromoe_tests` (the six pre-existing `qabsorb`/`wuv` failures documented
  in `docs/pickup-refactor-2026-10-18.md` are expected).

Structural rule enforced throughout: **node-creation order inside a graph must stay bit-for-bit
identical.** A refactor that reorders `ggml_*` calls is a behavior change, not a refactor, and is
rejected.

Noise floor: ~±2.4 % per six-pair mean (documented in the handoff docs); the models are also
*not bit-deterministic* under near-tie routing, so "coherence" is judged by long-run output, not
exact token equality — see `docs/models/README.md`.

### Acceptance milestone

**Ling and one other family migrate onto the same `ColdMoEStrategy` without adding any
family-specific cache, staging or CPU-lane machinery.** Until that holds, the shared backend does
not yet exist — only its ridge-table dedup does.

---

## 8. What NOT to do (already evaluated and rejected)

- **Push CRTP deeper into the builders** — `docs/source-layout-2026-10-18.md` measured ~20 lines
  of real duplication; the large functions genuinely diverge.
- **Restore removed machinery** — conditional-node master graph, Phase-2 3-graph dispatch,
  storage-service/completion-graph (removed for a reason; documented as un-adopted).
- **Unify attention builders across MLA families** — `build_dsa` / `build_bailing_mla` /
  `build_full_attention` differ enough that a forced shared base adds indirection without payoff.

---

## 9. Related docs

- `docs/graph-framework-architecture.md` — how the framework is structured; how to add a family.
- `docs/source-layout-2026-10-18.md` — folder split, "one family = one implementation" rule.
- `docs/graph-design-conformance-2026-10-08.md` — six-requirement design audit.
- `docs/handoff-2026-10-11-graph-framework-refactor.md` — the last big refactor's commits + gates.
- `docs/execution-profile-spec.md` — R4: machine-loadable execution profiles (keyed by hardware /
  shapes / quant / budgets / workload; the data that selects + parameterises `ExecutionPolicy`).
- `docs/models/README.md` — per-model playbook and the frozen baseline bands.
