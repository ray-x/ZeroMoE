# Model porting guide: host setup and runtime settings

This guide complements [Full graph model design and onboarding](full-graph-model-onboarding.md). That guide covers architecture validation and graph lifecycle. This one covers the host assumptions, configuration sources, and environment-variable behavior that can make a correct port fail at startup or run far below expected speed.

It describes the current code. The typed graph-settings resolver resolves defaults, family overrides, root TOML settings, legacy environment aliases, and generated `--graph-*` CLI flags, in that precedence order. Profiles remain a proposal.

## 1. Record the target system before tuning

Capture the hardware and build state for every correctness and performance result. Useful Linux commands include:

```sh
nvidia-smi
nvidia-smi -L
nvidia-smi topo -m
free -h
lscpu
numactl --hardware
ulimit -l
env | sort | rg '^(CUDA|OMP|ZEROMOE)_'
```

Record the CPU model and core count, NUMA layout, available host RAM, GPU models and free VRAM, GPU interconnect, CUDA driver/runtime, ZeroMoE commit, ggml revision and patches, and build options. Also record whether other sessions are using the machine. The engine has startup resource checks because two processes can each pass their own configured RAM and VRAM budgets while exhausting the shared host.

`CUDA_VISIBLE_DEVICES` filters and reorders the devices visible to CUDA. Treat the `device_id` values in the runtime profile as indices in that visible list, and confirm the device names printed at startup. Do not assume they still refer to the same physical GPU after changing `CUDA_VISIBLE_DEVICES`.

Host RAM and GPU memory are separate budgets. `ram_preload_all_experts` controls whether the full expert source is populated in host RAM when the model path needs it. `pin_ram_fraction` controls how much of the RAM expert tier is selected for pinning, while `ram_pinned_budget_gb` caps pinned host memory. A full pageable DDR source and a pinned DMA source are different states; check both in the logs. Size them against actual available RAM, other processes, and model weight residency.

The graph adapter's `device_expert_layers()` is a placement request, not a promise. The arena allocator may fit fewer complete layers after reserving dense weights and graph workspace. Record the resolved arena size and layer range, not only the requested count.

## 2. Know which settings system owns a value

There is no universal “environment beats TOML” rule. A setting's owner determines its source and precedence. Before documenting or changing a knob, trace the exact read site.

| Setting source | Current behavior | Where to inspect |
|---|---|---|
| Engine TOML | `EngineConfig::load()` reads the key and uses the in-code default when absent. | `core/engine_config.hpp` |
| CLI | Supported CLI options are applied after config loading and override their corresponding config fields. Only options implemented in `core/main.cpp` do this. | `core/main.cpp` |
| Graph runtime policy | `GraphRuntimeTuning` runs after model discovery and before cache planning. It fills defaults only when the corresponding key was absent. A model can require a resource; an explicit contradictory setting can fail startup. | `models/graph/full_graph_model.hpp`, `core/engine_runner.cpp` |
| Typed graph settings | Resolve in this order: common default, family override, then the first present environment alias. These settings are cached once per family and reported with their source. | `models/graph/graph_settings.hpp`, `models/graph/graph_settings_global.hpp` |
| Legacy graph environment settings | Read by the feature or builder itself, often through `graphcfg`. Parsing and fallback are defined at each read site. | `models/graph/graph_config.hpp`, model graph source files |
| Hardware allocation | The allocator turns placement and memory budgets into actual arenas and may clamp or decline a request. | `core/engine_runner_cache.cpp`, family arena builders |

The graph-settings resolver currently has `Default`, `Family`, and `Env` sources; its `Profile`, `Toml`, and `Cli` layers are future design. Engine TOML and CLI settings exist outside that resolver. A model may also read an environment variable directly, so check whether it overrides a TOML-backed field or only enables a feature.

The typed resolver is currently scoped to the families and fields declared in `models/graph/graph_settings.hpp`; it is not a complete environment-variable catalog. Other family settings, including many Qwen and DeepSeek graph controls, still use `graphcfg` helpers or direct environment reads.

`GraphRuntimeTuning` is the place for model-specific requirements that should not silently inherit generic defaults. Declare whether the graph requires a complete DDR expert source and provide measured pin defaults when appropriate. New adapters must explicitly declare runtime tuning, CPU-lane support, device expert count, and placement direction; do not leave those choices implicit.

## 3. Parse environment variables according to the actual helper

Environment-variable spellings are not enough to know how a value is parsed. `graphcfg` helpers intentionally preserve different conventions used by existing graph code:

| Helper | Meaning |
|---|---|
| `flag_on(name)` | On only when the value starts with `1`. |
| `flag_unless_off(name)` | On when unset; off only when the value starts with `0`. |
| `integer(name, default)` | Whole-string integer parse; malformed values use the default. |
| `atoi_or(name, default)` | Legacy `atoi` behavior; a numeric prefix is accepted. |
| `atof_or(name, default)` | Legacy `atof` behavior. |

Typed `graphsettings` entries declare their parser and aliases centrally. For a setting with several aliases, the first present alias for that family wins. Some flag settings combine aliases; read the field definition before assuming how conflicting aliases behave. Malformed strict integers leave the previous default or family value in place.

Set environment variables before launching ZeroMoE. Typed graph settings are resolved once and cached per family. Legacy `graphcfg` reads can happen later in graph construction, and direct `std::getenv` calls bypass the typed resolver. Changing an environment variable while the process is running is not a reliable way to change a graph's resolved configuration.

### Find a model's environment surface

For a new family, inspect the adapter's `enabled()` implementation first: feature gates commonly decide whether prefill or decode uses the graph. Then search the model graph for configuration reads:

```sh
rg -n 'ZEROMOE_|std::getenv|graphcfg::|settings_for\(' \
  models/<family> models/graph/full_graph_models.cpp
```

For every variable used in a recipe, document its owner, accepted values, default, family/phase scope, config overlap, and failure/fallback behavior. Separate feature gates from tunables and diagnostics. Prefer `graphsettings` for reusable typed tunables and `graphcfg` for feature-specific knobs; avoid adding another private parser without a reason.

## 4. Use startup logs to verify resolution

Read startup output in this order:

1. **`[PREFLIGHT]`**: topology or resource findings. `graph_preflight_strict=true` in TOML or `--graph-preflight-strict true` turns findings into a startup refusal.
2. **`[SETTINGS]`**: typed graph settings, their resolved value, and whether the source was `default`, `family`, `toml`, `env`, or `cli`.
3. **`[GRAPH-CAPS]`**: whether prefill/decode graph paths are enabled, CPU-lane support, graph-only status, device expert layer request/direction, and batch limits.
4. **`[GRAPH-TUNING]`**: model-required host expert source and pin defaults, plus defaults applied by the common setup.
5. **`[GRAPH-CONFIG]`**: `ZEROMOE_*` values known to the graph configuration registry, including names not read through that registry. A set value marked “not read yet” may belong to a later phase; at shutdown, `[GRAPH-CONFIG]` reports the knobs consulted over the run.
6. **Arena and preload messages**: actual VRAM expert arena capacity, complete resident layer range, DDR preload/pinning results, and any allocation clamp or fallback.

`[GRAPH-CONFIG]` only tracks reads routed through `graphcfg`. Direct `std::getenv` calls do not appear in its consulted-knob registry. `[SETTINGS]` only covers the typed resolver. Use source search when a variable is missing from those tables.

For a graph-only model, a zero device-layer request is a likely performance problem unless the graph has a deliberate alternative. Confirm a complete host source for every nonresident layer, CPU-lane capability where the graph needs it, and pinning budget if transfers depend on pinned pages. A graph that builds successfully has not yet passed performance validation.

## 5. Port and validate in controlled stages

1. **Baseline**: use the reference implementation and a known prompt. Record hardware/build details, model hash, quantization, context, prompt token count, prefill batch, and environment.
2. **Topology**: detect the exact GGUF architecture and validate target versus auxiliary layers, tensor shapes/types, attention map, dimensions, and required state before allocating large buffers.
3. **Adapter policy**: explicitly set graph gates, graph-only/fallback behavior, runtime tuning, CPU-lane support, device expert placement and direction, cache ownership, and prefill limits.
4. **Correctness smoke**: use a short prompt and small generation limit. Verify graph construction, finite outputs, state handoff, reset, and token/logit comparison against the reference path before throughput tuning.
5. **Performance smoke**: measure prefill and decode separately. Record first-run/setup time separately from steady output-token time. Preserve the resolved logs listed above.
6. **One change at a time**: vary one TOML or environment setting per run. Revert it explicitly before the next comparison. Do not compare measurements taken with different residency, context, prompt batch, quantization, or concurrent host load.

For CPU thread settings, budget all concurrent pools together: `omp_max_threads`, expert-fetch workers, prefetch workers, and any model-specific graph lane. `OMP_NUM_THREADS` influences OpenMP defaults when the config leaves `omp_max_threads` unset; some graph families have their own resolved CPU-thread setting. Avoid setting each pool to all cores independently, which can oversubscribe the host and slow both CPU work and input/output.

Keep the complete launch command with each baseline. For example, a Qwen4Exp graph run uses the default prefill/decode graph and a TOML profile:

```sh
CUDA_VISIBLE_DEVICES=0 \
./build/zeromoe --config configs/testing/qwen4exp_3090_only_decode_batched.toml --predict 8
```

That command illustrates the default graph path, not a portable placement or memory recipe. Choose the profile and placement values for the target hardware, then verify the resolved values and actual arena in the logs.
