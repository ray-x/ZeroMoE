# Reusable execution profiles (R4)

Status: **spec — not yet implemented**. Companion to `docs/graph-design-optimization-plan.md`
§4.6/§6. The strategy *interfaces* this references (`ExecutionPolicy`) are specced on the R2
branch (`models/graph/execution_policy.hpp`); this document defines the *data* that selects and
parameterises them.

## 1. Problem

A model's execution configuration today is scattered across three disconnected places:

1. **`ZEROMOE_*` env vars** read through `models/graph/graph_config.hpp` (`graphcfg::flag_on` /
   `integer` / `atoi_or` / …): e.g. `ZEROMOE_GLM5NEXT_OFFLOAD_DDR_GIB`,
   `ZEROMOE_GLM5NEXT_EARLY_READY`, `ZEROMOE_DSV41_PILOT_K`.
2. **Per-family constants and `static` decision lambdas** inside the graph builders.
3. **`docs/models/*.md` recipes** — human-readable, not loadable by the engine.

Three consequences, all of which the design review flagged:

- **Tune-from-zero.** Adding a model means re-deriving every knob from first principles.
- **Clobbering.** The knobs are a flat namespace, so running model B can silently inherit or
  override model A's tuning (`ZEROMOE_GLM5NEXT_*` vs `ZEROMOE_DSV41_*` is a convention, not a
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
| Hardware | **per-device** GPU model/VRAM + **PCIe gen × width**, host RAM | `GpuTopology::devices` + `cudaMemGetInfo` |
| Model dimensions | `n_layers`, `d_model`, `moe_inter`, `n_experts`, `topk` | `ArchitectureConfig` |
| Quant signature | `gate_type`, `up_type`, `down_type` | `ArchitectureConfig` / GGUF |
| Budgets | DDR residency bytes, VRAM cache bytes, pinned bytes | `EngineConfig` + preflight |
| Workload class | prefill / decode / chunk-continuation | request shape |

The quant signature matters because it selects the MoE runner and kernels
(`GgmlMoeRunner::instance_for(gate_type, up_type, down_type)` already keys on exactly this). The
hardware/VRAM matters because it selects the placement (resident-GPU vs offload vs CPU lane).

**PCIe link class is per-device, not per-system.** A box can mix generations and widths (the
baseline box is already exactly this: an RTX 3090 on PCIe 4 ×16 alongside a 5060 Ti on PCIe 3 ×8,
~12.3 GB/s vs ~6.8 GB/s H2D). Because link bandwidth decides the H2D ceiling and therefore whether
a `copy_expert_h2d`-style resident layer or a D2D reuse path wins, the key records a **vector of
per-device specs** (`gpu`, `vram`, `pcie_gen`, `pcie_width`) rather than a single system-wide
scalar. Two runs differ in the key if they differ on *any* device's link class, even when the GPU
models are identical.

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

**A frozen profile pins only what it measured** (amended 2026-10-23, see
`docs/design/settings-resolver.md` §4). The profile body stores a sparse set of pinned fields; every other
setting follows the live common defaults, so a change to a common default reaches every model that did not
measure that field. Pinned fields are immutable: `contribute_profile` on an existing key is rejected, and a
re-tune must go through an explicit `--rebaseline`, which rewrites the pins only after the change passes the
gate and the new baseline is recorded in `docs/models/`. A frozen profile is never silently mutated by
another model's run.

To keep inheritance from changing a frozen profile's behaviour unseen, the profile also records
`baseline_default_hash` (a hash of the table of common defaults when it was frozen) and the baseline value
of every unpinned field. At startup, if the live defaults hash differs, the resolver compares each unpinned
field and reports the ones that changed (`prefill_chunk 128 -> 512`): a warning in development, a failure in
performance mode, until the profile is re-measured (`--accept-baseline`) or the field is pinned. Diagnostic
settings are a separate struct and never enter the hash.

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
| `moe` = lane vs cache-backed | `ZEROMOE_GLM5NEXT_EARLY_READY` (+ which stack the family builds) |
| `offload_first_layer` | `ZEROMOE_GLM5NEXT_OFFLOAD_FIRST_LAYER` |
| `ddr_budget` | `ZEROMOE_GLM5NEXT_OFFLOAD_DDR_GIB` |
| `pinned_fraction` | `ZEROMOE_GLM5NEXT_OFFLOAD_PINNED` |
| `cache` (admission/eviction) | `ZEROMOE_GLM5NEXT_OFFLOAD_POLICY` (lru/static/segmented) |
| `pilot_k` | `ZEROMOE_DSV41_PILOT_K` |
| `pilot_prefetch` / `pilot_stage` / `pilot_copy` | `ZEROMOE_GLM5NEXT_PILOT_PREFETCH` / `_PILOT_STAGE` / `_PILOT_COPY` |
| `predictor` (ridge/pilot) | `ZEROMOE_QWEN_EARLY_READY_RIDGE` + `pilot_ridge.hpp` |
| `assist_slots` / `assist2` | `ZEROMOE_GLM5NEXT_ASSIST2_SLOTS` / `_ASSIST2_DEVICE` / `_ASSIST2_KEEP` |
| `pin_threshold` / `promote_hit_threshold` | `RAMSafetyManager` fields (hard-coded today) |
| `moe` device / chunk pipeline | `ZEROMOE_GGML_MOE_DEVICE` / `ZEROMOE_GGML_CHUNK_PIPELINE` |

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
// Serialised as TOML (matches the existing *.toml run configs, stays human-diffable).

struct DeviceSpec {                 // per-device; order = the model's device order
  std::string gpu;                  // e.g. "rtx3090", "rtx5060ti"
  size_t vram_bytes = 0;
  int pcie_gen = 0;                 // 3 | 4  (a mixed 3.0+4.0 box differs per device)
  int pcie_width = 0;               // 8 | 16
  bool operator==(const DeviceSpec&) const = default;
};

struct ProfileKey {
  // hardware (per-device, so mixed PCIe generations/widths are captured)
  std::vector<DeviceSpec> devices;
  size_t ram_bytes = 0;
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
// miss -> validate + freeze. Refuses (returns error) when the key already exists:
// a frozen profile's pinned fields are immutable; re-tune only via an explicit rebaseline path.
bool contribute_profile(ExecutionProfile profile);
bool rebaseline_profile(ExecutionProfile profile);              // gated rewrite of a frozen key
```

## 10. Decisions

1. **Serialisation: TOML.** Matches the existing `*.toml` run configs and stays human-diffable;
   the C++ type above is the in-memory form only.
2. **Key granularity: include PCIe link class, per device.** The link class changed the selected
   device once already (the 5060 Ti finding in `run_ggml_moe_layer`). Because a box can mix
   generations/widths (e.g. PCIe 4 ×16 + PCIe 3 ×8), the class is recorded **per device**
   (`DeviceSpec.pcie_gen`/`pcie_width`), not as a single system-wide scalar.
3. **Contribute vs override: pinned fields are immutable once frozen** (amended 2026-10-23: a profile pins
   only the fields it measured; the rest follow live defaults, guarded by `baseline_default_hash`, see §5).
   `contribute_profile` refuses on an existing key; re-tuning goes through `rebaseline_profile`, which only
   rewrites after the change passes the gate and its new baseline lands in `docs/models/`.
