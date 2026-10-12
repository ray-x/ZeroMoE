# Full-graph framework: structure and how to add a model

The whole-model GGML graph of each family is split into a small number of
layers, each with one job. This replaces the former single 11k-line
`ggml_qwen4exp_graph.cpp` and 7k-line `ggml_glm5next_graph.cpp`.

```
include/graph_core.hpp          GraphCore, DeviceArena, HostAllocation, host helpers
include/graph_config.hpp        typed env readers + [GRAPH-CONFIG] log
include/expert_store.hpp        DDR tier of the expert hierarchy (model-agnostic)
include/moe_routed_ggml.hpp     router / expert FFN / top-k-sum builders (shared)
include/early_ready_lane.hpp    CPU lane for cold experts (shared)
models/full_graph_model.hpp     FullGraphModel interface + FullGraphModelBase<Derived>
models/full_graph_models.cpp    one small class per family + registry
src/ggml_<family>_graph*.cpp    the family's graph implementation, split by topic
```

## Layers

1. **`FullGraphModel`** (`models/full_graph_model.hpp`) is the only thing the
   engine talks to: `enabled / bootstrap / build / run / shutdown`,
   `device_expert_layers`, and the policy hooks `router_spec` and
   `supports_cpu_lane`. It is reached through
   `engine.family_ops.graph_model` (nullptr = layer-by-layer path).
2. **`FullGraphModelBase<Derived>`** (CRTP) owns the request validation and the
   prefill/decode dispatch every family used to repeat. A family implements
   `prefill_prepare / prefill_run / decode_prepare / decode_run` and may hide
   `accepts()` to refuse request shapes. Dispatch is static; only the registry
   boundary is virtual.
3. **`GraphCore`** is the family-independent base of each family's `FullGraph`:
   ggml handles (`cuda`, `cpu`, `sched`, `ctx`, `graph`), the common tensors,
   the three `DeviceArena`s (`dense`, `expert`, `state`), owned host memory and
   `alloc_host / bind_host / host_input`. `release_core()` is the shared tail of
   every `destroy()`.
4. **Family implementation**, one namespace per family (`qwen4exp_graph`,
   `glm5next_graph`). Each family's `FullGraph : GraphCore` adds only what is
   specific to it. The `.cpp` files are cut by concern:

| Topic | Qwen4Exp file suffix | GLM5Next file suffix |
|---|---|---|
| settings / env gates | `_settings` | `_settings` |
| memory placement, arenas | `_arena`, `_placement` | `_arena`, `_partial` |
| expert / dense weights | `_expert_bind` | `_weights`, `_offload`, `_stage` |
| building blocks | `_build_blocks`, `_build_moe` | `_build_blocks`, `_build` |
| whole-graph build | `_build` | `_build` |
| lifecycle (public API) | (no suffix) | (no suffix) |
| decode / run | `_decode_prepare`, `_run` | `_decode`, `_run` |
| diagnostics | `_diag` | |

Each family has one internal header (`ggml_qwen4exp_graph_internal.hpp`,
`ggml_glm5next_graph_impl.hpp`) with its types, process-wide state and function
declarations. The `.cpp` files include only that header.

## Adding a model family

1. **Detect it**: add a `ModelFamily` value and its detection in
   `models/model_factory.cpp`; read its hyper-parameters into
   `ArchitectureConfig` *before* `resolve_family_ops()` (see CLAUDE.md,
   initialization-order guard).
2. **Graph state**: declare `struct FullGraph : moestream::graph::GraphCore` in
   your family's internal header. Add only family-specific members; use
   `dense`, `expert`, `state` for arenas and `alloc_host` / `host_input` for
   host inputs. Call `release_core()` last in `destroy()`.
3. **Graph builder**: build the trunk with the shared pieces
   (`moe_routed::gate_probs / select_ids / expert_ffn / sum_topk`,
   `ExpertStore` for SSD-offloaded experts, `EarlyReadyLane` for the CPU lane).
   Only attention / recurrent-state / hyper-connection code is yours.
4. **Model class**: in `models/full_graph_models.cpp` derive
   `FullGraphModelBase<YourModel>`, map the four prepare/run hooks onto your
   implementation, override `router_spec` / `supports_cpu_lane` if they differ,
   and add a `case` to `full_graph_model()`.
5. **Settings**: put env gates in `_settings.cpp` using the typed readers in
   `graph_config.hpp` so they show up in the `[GRAPH-CONFIG]` block.

## Process-wide state

Each family keeps its live graph(s) in file-level globals declared in its
internal header (`g_graph`, owner slots, config flags). `shutdown()` must reset
every CUDA-backed global explicitly (see the Phase-2 shutdown crash note in
project memory); adding a new CUDA-backed global means adding it there.

## Verification

- Build: `cmake --build build_base --target moestream` and `moestream_tests`.
- Qwen correctness gate: `bench/graph_framework_step.sh` (logits byte-identical,
  free-running tokens identical).
- GLM gate: `ARMS="lru lane" bench/graph_framework_glm_gate.sh`.
- A refactor that changes only structure must keep both byte-identical.
