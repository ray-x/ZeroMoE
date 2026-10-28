# Settings resolver and sparse profile pins

Status: **defaults, family overrides, canonical root TOML, compatibility environment
aliases and generated graph CLI flags implemented**. Qwen whole-layer placement and
shared CPU thread caps also use the resolver. Profile pins and capability negotiation
remain design proposals. User-facing behavior and commands: [configuration guide](../configuration.md).
The sections below describe the broader target design, not all shipped behavior.
Code: `models/graph/graph_settings.hpp`
(pure), `graph_settings_global.hpp` (real env/CLI binding + `[SETTINGS]` log), `tests/test_graph_settings.cpp`. Detail for rollout steps 0–1 of
`docs/design/model-family-framework.md` (§4.2, §5). Assumes the sparse-pin proposal: a profile stores
only measured fields, everything else follows live defaults, and a version-plus-hash drift check
guards it. If the R4 owner rejects that, §7 gives the fallback. Extends `docs/execution-profile-spec.md`
(R4) and replaces the read side of `models/graph/graph_config.hpp`.

## 1. Goals and non-goals

Goals: one typed settings struct; one resolved value per knob with its source; step 1 changes **no
behaviour**; profiles pin only what they measured; a change to a common default is either inherited or
loudly flagged.

Non-goals: choosing new default values, changing strategies (`ExecutionPolicy` stays as is), the
`FamilyDescriptor` attention/FFN interfaces (separate step), a tuner.

## 2. Data model

```cpp
// models/graph/graph_settings.hpp
enum class Source : uint8_t { Default, Family, Profile, Toml, Env, Cli };

template <class T> struct Resolved { T value{}; Source src = Source::Default; const char *note = ""; };

struct GraphSettings {            // tunables: may change performance or numerics
  Resolved<int>    cpu_threads_decode, cpu_threads_prefill;
  Resolved<int>    prefill_chunk, prefill_min_batch;
  Resolved<int>    prefill_offload_min_tokens;   // op-offload threshold
  Resolved<bool>   prefill_op_offload;
  Resolved<double> offload_ddr_gib;              // 0 = auto
  Resolved<double> offload_pinned_fraction;
  Resolved<int>    gpu_assist_slots;
  Resolved<bool>   pin_arena_for_dma;
  Resolved<std::string> offload_usage_file;
  // ... one field per knob in the inventory (§5)
};

struct GraphCapabilities {        // what the family supports; all default false
  bool aliases_vram_cache = false, early_ready_lane = false,
       prefill_continuation = false, wants_op_offload = false;
  int  min_prefill_batch = 0;     // 0 = unset, resolver rejects use
};

struct DiagnosticSettings { /* trace modes, dumps, verify flags: never hashed */ };
```

Rules: tunables, capabilities and diagnostics are three separate structs. Diagnostics (trace, dump,
verify, node-timing knobs) are the "benign" set: masked out of the drift hash and not subject to
provenance checks. A knob is added to exactly one struct, in one PR, with its default and a one-line
meaning.

## 3. Layers and precedence

```
common default  <  family override  <  profile pin  <  TOML  <  env  <  CLI
```

* **Common default:** a `constexpr` initialiser table in `graph_settings.cpp`.
* **Family override:** `FamilyDescriptor::overrides`, a sparse list of `{field, value, reason}`. The
  `reason` string is mandatory. An override equal to the default is rejected at startup as redundant.
* **Profile pin:** only the fields stored in the matched profile (§4). Beats a family override because
  it is measured for this exact key; losing this would let a code edit silently undo a measurement.
* **TOML / env / CLI:** run-level tuning. Remaining environment aliases are transitional; MiMo2
  and Ling graph-setting aliases have been removed as maintained launchers moved to canonical TOML
  and `--graph-*` options. A remaining alias hit logs once with the canonical name. Unknown `ZEROMOE_*` names are warned (a typo
  today is silently ignored).

Capabilities are not layered: the family sets them in code, the resolver only reads them. They may
still be *requested off* by run config (e.g. disable op-offload) but never requested on beyond what the
family declares.

## 4. Sparse pins, versioning and the drift check

Profile file (TOML, same format family as R4 §10):

```toml
[key]                       # R4 ProfileKey, unchanged
model = "mimo-v2.6-flash"; quant = "gsq-rco-3bit"; devices = ["rtx3090:24:4x16", "rtx5060ti:16:3x8"]
[pins]                      # ONLY measured fields
prefill_chunk            = 512
prefill_offload_min_tokens = 256
cpu_threads_decode       = 10
[baseline]
baseline_default_hash = "c41e…"   # hash of the base defaults table when frozen
unpinned_hash    = "9f3c…"  # hash of resolved unpinned tunables at freeze time
validated_against = "docs/models/mimo26-flash.md"
committed_at     = "<git rev>"
```

* **`defaults_hash`:** no hand-edited integer (two branches that both change a default would conflict
  on it). It is computed at startup from the base defaults table: `(field name, default value)` for
  every `GraphSettings` field plus the capability defaults, sorted by name. The profile records it as
  `baseline_default_hash` when frozen. A checked-in snapshot file of the defaults table is diffed by a
  test, so a default change shows up in review as a snapshot edit.
* **`unpinned_hash`:** hash over `(field name, resolved value)` of every **tunable not pinned by this
  profile**, sorted by name, using the value that resolved *without* TOML/env/CLI (so a one-off run
  override cannot trip or mask it). Diagnostics are excluded by construction (they live in another
  struct). Capabilities are included, since they change behaviour.
* **Check at resolve time:** recompute the hash with the live defaults.
  The cheap check comes first: if the live `defaults_hash` equals `baseline_default_hash`, nothing
  changed and the check ends. Otherwise compare each unpinned field. Equal values → silent (a default
  changed that this profile does not inherit). Mismatch → report which unpinned fields differ from the frozen baseline (the profile
  also stores the per-field values it saw, so the message can say `prefill_chunk 128 -> 512`, not just
  "hash differs"). Development: warning. Performance mode: **hard failure** with the same diff and the
  two ways out (re-tune, or `--accept-baseline`, which rewrites `[baseline]` after a measurement run).
* **Pinned fields are never part of the hash,** so retuning a default only drifts profiles that
  inherit it.
* **Lookup miss:** no profile means no pins and no drift check; settings are default + family + run
  layers, and the table prints `profile: none`.

Storing per-field baseline values makes the file larger but makes every drift report actionable.

## 5. Knob inventory (step 1 scope: MiMo and Ling)

Step 1 moves these in with today's defaults, no change in value:

| Field | MiMo today | Ling today | Source today |
|---|---|---|---|
| `cpu_threads_decode` | 10 | 10 | Qwen CPU-worker alias remains; MiMo2 uses `graph_cpu_threads_decode` / `--graph-cpu-threads-decode` |
| `prefill_chunk` | 512 | per Ling code | `…_PREFILL_CHUNK` |
| `prefill_min_batch` | 8 | 64 (base default) | `min_prefill_batch()` override |
| `prefill_op_offload`, `prefill_offload_min_tokens` | auto, 256 | disabled in code | `…_PREFILL_OP_OFFLOAD`, `…_OFFLOAD_MIN` |
| `offload_ddr_gib`, `offload_pinned_fraction`, `pin_arena_for_dma` | 0 auto, 0.xx, on | per Ling code | `…_OFFLOAD_*` |
| `gpu_assist_slots` | 0 | per Ling code | `…_GPU_ASSIST` |
| `offload_usage_file` | env | env | `…_OFFLOAD_USAGE` |

(The exact Ling column is filled from the code during implementation; the migration test in §6
fails if any value differs.) GLM's ~50 knobs stay on `graphcfg::env` until step 6; the resolver
coexists with it and `log_effective()` keeps printing consulted-but-unset knobs.

## 6. Interface and tests

```cpp
GraphSettings resolve_settings(const FamilyDescriptor &fam, const ProfileView *prof,
                               const TomlConfig &toml, const EnvView &env, const Cli &cli,
                               std::string *err);
std::string format_settings(const GraphSettings &, const GraphCapabilities &);
```

`EnvView` is an injectable snapshot so tests never touch `setenv`. Resolution runs once, before graph
build, and the result is passed down as `const&`; graph code stops calling `getenv` for tunables.

Printed table (one block per run, e.g. `[SETTINGS]`):

```
prefill_chunk             512   family:mimo2 (reason: 512-token chunks measured 74 vs 41 tok/s)
cpu_threads_decode         10   default
moe_cache_gb[0]            12   toml
prefill_op_offload         on   default      capability: wants_op_offload=true (family:mimo2)
profile: mimo-v2.6-flash/3090+5060ti  defaults hash c41e (live c41e)  drift: none
```

Tests:
1. **No-behaviour-change:** a table of `{family, field, old resolved value}` taken from the code before
   the change; resolving with empty TOML/env must equal it. Also run with each legacy env var set.
2. **Precedence:** each layer beats the one below, per field.
3. **Redundant override rejected; missing reason rejected.**
4. **Defaults snapshot file vs the computed `defaults_hash`.**
5. **Drift:** change one unpinned default → mismatch with a field-level diff; change a pinned field or a
   diagnostic → no mismatch; env override of an unpinned field → no mismatch.
6. **Gates:** Ling decode gate token-identical and in band; MiMo short run unchanged; `moe_conformance`
   extended to reject new `getenv("ZEROMOE_…")` calls for tunables in MiMo/Ling graph files.

## 7. Fallback: full snapshot (not used)

Kept for reference only; the sparse-pin rule was adopted. If it were ever reversed, `[pins]` becomes the whole resolved tunable set and `[baseline]` is
dropped. Resolution compares the live resolved values to the snapshot field by field and fails in
performance mode on any difference, until `--rebase-profile` rewrites it and marks it unvalidated.
Layers, struct, table and tests above are unchanged; only §4 differs. Every default change then needs
a rebase pass over all profiles.

## 8. Open items

* **Decided:** the CLI layer is the existing `core/main.cpp` parser (`--config` plus legacy flags); it
  writes into the resolved struct at highest precedence. No new CLI is built for this work; the only new
  flag is `--accept-baseline`. TOML and env remain the main run-time overrides.
* **Decided:** the version is a computed hash of the defaults table, not an integer (§4). The masked
  list does not exist; diagnostics are a separate struct.
* **Decided:** per-device settings (`moe_cache_gb`, assist slots per device) are vectors indexed by
  device, parsed from TOML arrays, because topologies are asymmetric. A resolved vector is hashed and
  compared element-wise, and its length must match the device count or preflight fails.
* The family override `reason` text should be lint-checked for non-empty only; quality is a review
  matter.

## 9. Step 1 as built

* Migrated: MiMo graph settings and Ling's graph-specific settings are now canonical TOML/CLI fields;
  their family-prefixed environment aliases were removed. GLM aliases remain and apply to Ling when
  it uses the shared GLM graph.
* `Family` names the model, not the graph implementation. `Family::Glm` covers GLM-5.x and
  GLM-DSA; `Family::Ling` inherits Glm family defaults and aliases because Ling's shared graph
  uses the GLM implementation. Ling no longer has duplicate `ZEROMOE_LING_*` graph-tuning aliases.
* Capability is declared per field (`Field::claims`), not inferred from aliases. TOML acceptance
  uses the model's active adapter list: a Ling run that uses the shared graph lists `Family::Glm`,
  while the native Ling graph does not claim the GLM offload knobs. Alias visibility for
  TOML/env resolution is tracked separately (`detail::reads`).
* Remaining environment aliases keep their old parse (strict strtol, atoi, on-unless-0). These aliases
  are being retired in favor of the canonical TOML/CLI surface.
* Not migrated: the `ZEROMOE_MIMO2_INGRAPH_MOE` / `_FULL_GRAPH` gates (feature gates, not tunables),
  GLM's other ~45 knobs, Ling's native/hybrid graph gates, `ZEROMOE_LING_NO_*_CACHE_ALIAS`.
* Ling's chunk: TOML/CLI value wins; otherwise the engine's prefill batch or 2048. The default stays
  0 = "the model decides" because that fallback needs the engine.
* `defaults_table()` / `defaults_hash()` exist; `tests/graph_settings_defaults.snapshot` is diffed by the
  unit test. The drift check against a profile is not built (no profiles exist yet).
* `moe_conformance` now fails on a direct `ZEROMOE_MIMO2_/LING_` tunable read in the MiMo graph and
  `full_graph_models.cpp`.
* Run `make test-settings BUILD_DIR=build-cpu` after configuring a CPU build, or
  `cmake --build --preset settings-tests && ctest --preset settings-tests` after
  `cmake --preset cpu`.

### Added after step 1

* `usage_record_file` (diagnostic, outside the defaults hash): canonical `graph_usage_record_file` TOML/CLI setting; no environment alias. Turns on the shared route survey (`models/graph/graph_route_survey.hpp`). Only the in-graph MoE path is tapped; MiMo's synchronous fallback (`ZEROMOE_MIMO2_INGRAPH_MOE=0`) and layered path record nothing, so the file stays empty there.
* One prefill-chunk model for graph families: env (`prefill_chunk` aliases) > the TOML's `prefill_batch_tokens` >
  the family override (MiMo 512) > default. MiMo's configs no longer set `prefill_batch_tokens = 16`, which the
  MiMo graph used to ignore and which would otherwise have capped the chunk at 16.
