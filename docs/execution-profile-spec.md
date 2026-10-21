# Reusable execution profiles (R4)

Status: **spec — not yet implemented**. Companion to `docs/graph-design-optimization-plan.md`
§4.6/§6. The strategy *interfaces* this references (`ExecutionPolicy`) are specced on the R2
branch (`models/graph/execution_policy.hpp`); this document defines the *data* that selects and
parameterises them.

## 1. Problem

A model's execution configuration today is scattered across three disconnected places:

1. **`MOESTREAM_*` env vars** read through `models/graph/graph_config.hpp` (`graphcfg::flag_on` /
   `integer` / `atoi_or` / …): e.g. `MOESTREAM_GLM5NEXT_OFFLOAD_DDR_GIB`,
   `MOESTREAM_GLM5NEXT_EARLY_READY`, `MOESTREAM_DSV41_PILOT_K`.
2. **Per-family constants and `static` decision lambdas** inside the graph builders.
3. **`docs/models/*.md` recipes** — human-readable, not loadable by the engine.

Three consequences, all of which the design review flagged:

- **Tune-from-zero.** Adding a model means re-deriving every knob from first principles.
- **Clobbering.** The knobs are a flat namespace, so running model B can silently inherit or
  override model A's tuning (`MOESTREAM_GLM5NEXT_*` vs `MOESTREAM_DSV41_*` is a convention, not a
  mechanism).
- **The selected strategy is not a datum.** It is implicit in *which* env vars happen to be set,
  so "what strategy ran" cannot be read off, diffed, or versioned.

## 2. Goal / non-goals

**Goal:** a first-class, machine-loadable `ExecutionProfile` that records, per
model × hardware × workload, the *selected* execution configuration (policy identifiers +
tunables), keyed so a new model looks up and reuses an existing profile instead of tuning from
zero.

**Non-goals:** this is *not* the `ExecutionPolicy` (that is the behavior/interface layer), not the
placement *algorithm* (R3), and not the frozen *measurement* baselines (those stay in
`docs/models/` — see §6). The profile is the *data* in between.

## 3. The profile key

The key is what makes a profile reusable: two runs that share a key share a profile. It captures
everything that determines the execution config, and nothing more.

| Dimension | Fields | Source |
|---|---|---|
| Hardware | GPU model/VRAM, device count, PCIe link class, host RAM | `GpuTopology::devices` + `cudaMemGetInfo` |
| Model dimensions | `n_layers`, `d_model`, `moe_inter`, `n_experts`, `topk` | `ArchitectureConfig` |
| Quant signature | `gate_type`, `up_type`, `down_type` | `ArchitectureConfig` / GGUF |
| Budgets | DDR residency bytes, VRAM cache bytes, pinned bytes | `EngineConfig` + preflight |
| Workload class | prefill / decode / chunk-continuation | request shape |

The quant signature matters because it selects the MoE runner and kernels
(`GgmlMoeRunner::instance_for(gate_type, up_type, down_type)` already keys on exactly this). The
hardware/VRAM matters because it selects the placement (resident-GPU vs offload vs CPU lane).

## 4. The profile body

The body records the *selected* configuration as data, not code:

- **Strategy selection** (by identifier, resolved to `ExecutionPolicy` by the builder):
  `cache`, `predictor`, `prefetch`, `promotion`, `moe`.
- **Placement:** `offload_first_layer`, resident-layer prefix, `ddr_budget`, `pinned_fraction`.
- **Speculation depths:** `pilot_k`, `pilot_prefetch`, `pilot_lookahead`, `assist_slots`.
- **Thresholds:** `pin_threshold`, `promote_hit_threshold`.
- **Provenance:** when validated, against which `docs/models/*.md` baseline.

Identifiers, not pointers: the profile references strategies by name so it is serialisable,
diffable, and independent of the `ExecutionPolicy` headers.

## 5. The resolution rule

1. On model load, derive the key from `ArchitectureConfig` + hardware + budgets + workload.
2. Look up a profile by key.
3. **Hit → reuse** the profile's config. No tuning.
4. **Miss → contribute** a new profile with the model's initial config, then validate and freeze it.

This is the rule that makes onboarding a *bounded task* (§4.6 of the plan): a new model reuses
existing strategy implementations and contributes only the delta where its key genuinely differs.
It also removes the clobbering: a profile is selected by key, never by "which env vars leaked in".

## 6. Relationship to `ExecutionPolicy` and the compile boundary

```
ExecutionPolicy  = the strategy INTERFACES (behavior)          [R2 branch]
ExecutionProfile = the DATA that selects + parameterises them   [this spec]
ExecutionPlan    = a resolved profile + the concrete policy objects built from it
compile(ModelProgram, ExecutionPlan, RuntimeResources) -> GraphInstance
```

The "execution plan" layer of the three-way split (§3 of the plan doc) is exactly the resolved
profile. The builder turns `ExecutionProfile` (identifiers + values) into `ExecutionPolicy`
(instantiated strategy objects) at load time; per-token work never touches the profile.

## 7. Mapping to existing tunables

The profile body is a structured re-homing of the flat knobs. Representative mapping (not
exhaustive — the full list comes from `graph_config.hpp`'s `log_consulted()` output):

| Profile field | Today's knob (or constant) |
|---|---|
| `moe` = lane vs cache-backed | `MOESTREAM_GLM5NEXT_EARLY_READY` (+ which stack the family builds) |
| `offload_first_layer` | `MOESTREAM_GLM5NEXT_OFFLOAD_FIRST_LAYER` |
| `ddr_budget` | `MOESTREAM_GLM5NEXT_OFFLOAD_DDR_GIB` |
| `pinned_fraction` | `MOESTREAM_GLM5NEXT_OFFLOAD_PINNED` |
| `cache` (admission/eviction) | `MOESTREAM_GLM5NEXT_OFFLOAD_POLICY` (lru/static/segmented) |
| `pilot_k` | `MOESTREAM_DSV41_PILOT_K` |
| `pilot_prefetch` / `pilot_stage` / `pilot_copy` | `MOESTREAM_GLM5NEXT_PILOT_PREFETCH` / `_PILOT_STAGE` / `_PILOT_COPY` |
| `predictor` (ridge/pilot) | `MOESTREAM_QWEN_EARLY_READY_RIDGE` + `pilot_ridge.hpp` |
| `assist_slots` / `assist2` | `MOESTREAM_GLM5NEXT_ASSIST2_SLOTS` / `_ASSIST2_DEVICE` / `_ASSIST2_KEEP` |
| `pin_threshold` / `promote_hit_threshold` | `RAMSafetyManager` fields (hard-coded today) |
| `moe` device / chunk pipeline | `MOESTREAM_GGML_MOE_DEVICE` / `MOESTREAM_GGML_CHUNK_PIPELINE` |

`graph_config.hpp`'s knob registry (`Registry::snapshot()`) is the natural migration path: it
already records *which* knobs a run consults, so the profile can be populated by replaying a
validated run's consulted-knob list instead of hand-editing.

## 8. Validation & provenance

A profile is validated exactly like today: byte-identical logits (Qwen/GLM gates) + a throughput
measurement within the frozen band, per the tiered policy in `docs/models/README.md`. The profile's
provenance field points at the `docs/models/*.md` baseline that froze it, so the machine-readable
profile and the human-readable recipe stay in lock-step. A profile whose key matches but whose
band has drifted is a regression, not a new profile.

## 9. The type (sketch — header comes with the implementation)

```cpp
// models/graph/execution_profile.hpp  (design sketch; not yet wired)

struct ProfileKey {
  // hardware
  std::string gpu; int devices = 0; size_t vram_bytes = 0, ram_bytes = 0;
  // model dimensions
  int n_layers = 0, d_model = 0, moe_inter = 0, n_experts = 0, topk = 0;
  // quant signature
  uint32_t gate_q = 0, up_q = 0, down_q = 0;
  // budgets
  size_t ddr_budget = 0, vram_cache_budget = 0;
  // workload class
  enum class Workload { Prefill, Decode, ChunkContinuation };
  Workload workload = Workload::Decode;

  auto operator<=>(const ProfileKey&) const = default;  // full structural equality
};

struct ExecutionProfile {
  ProfileKey key;

  // strategy selection by identifier (resolved to ExecutionPolicy by the builder)
  std::string cache_policy, predictor, prefetch_policy, promotion_policy, moe_strategy;

  // placement
  int offload_first_layer = 0; size_t ddr_gib = 0; double pinned_fraction = 0.0;
  // speculation depths
  int pilot_k = 0, pilot_prefetch = 0, pilot_lookahead = 0, assist_slots = 0;
  // thresholds
  uint32_t pin_threshold = 0, promote_hit_threshold = 0;

  // provenance
  std::string validated_against;   // e.g. "docs/models/glm53-flash.md"
  std::string committed_at;        // git rev
};
```

Resolution lives behind a small registry:

```cpp
const ExecutionProfile *resolve_profile(const ProfileKey &key);  // hit -> reuse
void contribute_profile(ExecutionProfile profile);               // miss -> validate + freeze
```

## 10. Open questions (to settle at implementation time)

1. **Serialisation format** — a typed file (TOML/JSON) vs. generated C++ defaults. TOML is
   preferred: it matches the existing `*.toml` run configs and stays human-diffable.
2. **Key granularity** — is `(gpu, vram)` enough, or must the key include PCIe link class (x16 vs
   x8) after the 5060 Ti finding in `run_ggml_moe_layer`? Leaning: include it; it changed the
   selected device once already.
3. **Contribute vs override** — when a key matches but the owner wants to re-tune, does
   `contribute_profile` refuse (immutable) or record a new revision? Leaning: immutable once frozen,
   with an explicit `--rebaseline` to rewrite after a validated change.
