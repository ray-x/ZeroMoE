# Source layout: what changed in the 2026-10-18 split, and what is left

Companion to `graph-framework-architecture.md` (the full-graph layer, already
organised the way llama.cpp organises its graph builders). This note covers the
other three files that were over 5k lines and the rest of the oversize list.

## Done on branch `refactor/source-split`

| was | now | how it is organised |
|---|---|---|
| `include/perf_timer.hpp` (5.9k, in 90 TUs) | `include/perf_timer.hpp` (132) facade; `include/perf/counters_NN_*.hpp` (8 slices, ~390 each); `include/perf/perf_counters.hpp` (252); `src/perf_report_{summary,dump_a,dump_b,stats}.cpp` | `PerfCounters` derives from a linear chain of counter slices, so `perf().t_x` is unchanged. Reporting moved out of the header (it was compiled into every TU). `dump_report()` is six section methods. |
| `src/main_test.cpp` (7.5k) | `src/main_test.cpp` (1.1k: `main()` and every EXL3 test, untouched) + `tests/native/tests_*.cpp` (8 files) + `native_tests.hpp` / `native_test_common.hpp` | one file per topic (basic CUDA, CPU quant, CPU MoE, GPU attention, quant prototypes, expert cache, GPU kernels, config). |
| `src/main.cpp` (5.0k, one 4,160-line function) | `src/main.cpp` (828) + `include/engine_runner.hpp` + `src/engine_runner*.cpp` (7 files) | `EngineRunner` owns the long-lived subsystems and exposes one method per startup phase: `apply_engine_config`, `init_host_and_gpu`, `load_model`, `plan_expert_cache`, `setup_vram_expert_cache`, `offload_dense_tensors`, `init_ram_tier`, `preload_experts`, `start_workers`, `run_repl`, `shutdown_and_report`. |

All three are mechanical: bodies are verbatim, and each commit message states the
check used (identical report output for the counters; identical multiset of source
lines for the other two).

## Folder layout (reorganised 2026-10-18)

```
core/              orchestration: engine, EngineRunner phases, execution/dispatch pipelines, MoE router,
                   GPU topology, layered attention, MTP, perf counters (core/perf/), logger, main.cpp
memory/            expert caches and resource management: ram_manager, expert_cache/store, scratch buffers,
                   offload store, direct_io/io_uring/cufile, prefill streaming, prefetcher
models/base/       model_base/family/factory, family_ops, GGUF + safetensors loaders, tokenizer, graph_initializer
models/graph/      shared full-graph framework: graph_core, graph_config, early_ready_lane, routed-MoE builders
models/qwen/       Qwen4Exp + Qwen MoE: cpu_qwen4exp, ggml_qwen4exp_graph*, stage pool, PLE, GDN, offload ring
models/glm/        GLM5Next (Flash) graph files, glm_dsa.hpp, GLM projection backend
models/deepseek/   DeepSeek-V4.1 graph, engram, attention helpers
backends/cpu/      CPU engine, CPU model backend, kernels_cpu (+neon/metal), dequant, ggml grids
backends/cuda/     CudaManager, kernel dispatcher, cublas, cuda_raii
backends/ggml/     ggml MoE backend/runner/transfer manager, HC and GDN runners, resource guards
backends/exl3/     EXL3 (WIP; moved as-is, only compile-checked)
kernels/quant/     per-quant matmul/dequant .cu
kernels/math/      generic .cu (matmul, rmsnorm, expert accum/activation, attention score, MLA, GDN conv)
kernels/models/    model-specific .cu (GLM KDA/router, Qwen HC/stage, DSV4 score)
tests/native/      native harness incl. main_test.cpp (moved from src/)
tools/ bench/ tests/ configs/ patches/ third_party/ docs/ cmake/   unchanged
```

Rules: headers live next to their sources. `CMakeLists.txt` defines `MOESTREAM_INCLUDE_DIRS`
(every header directory above) and uses it for all targets, so `#include "name.hpp"` lines
are unchanged and a file can move between these folders without editing the files that include
it. File names are unique across the tree (a prerequisite, checked when the move was made).
The move changed no file content except 8 include lines that were path-qualified
(`"../models/full_graph_model.hpp"` and similar): 241 of 249 moves are exact (100%) renames in
git, the rest 98-99%. Old path -> new path for every moved file: `docs/layout-map-2026-10-18.tsv`.

Differences from the sketch that was proposed: files named there that do not exist
(`engine.cpp`, `ram_manager.cpp`, `expert_cache.cpp`, `offload_store.cpp`, `qwen4exp_attn.cu`,
`ggml_deepseek_v4_graph.cpp`, `qwen4exp_layer13_staging.cpp`) were not invented; `ram_manager`,
`expert_cache`, `engine` and `offload_store` are header-only here. Added `models/graph/`
(the shared full-graph framework is used by Qwen, GLM and DeepSeek, so it belongs to none of
them), `core/perf/`, and `tests/native/`.

## How this maps to llama.cpp

| llama.cpp | MoEStream |
|---|---|
| `llama_model` / `llama_model_loader` | `StreamingEngine::model_file`, `gguf_loader.hpp`, `GGUFGraphInitializer` |
| `llama_context` (owns backends, sched, memory, run loop) | `EngineRunner` (new): owns engine, RAM tier, GPU topology, prefetcher; one method per phase |
| `llm_graph_context` + `src/models/<arch>.cpp` | `FullGraphModel` + `GraphCore` + `src/ggml_<family>_graph*.cpp` (see the framework doc) |
| `llama_kv_cache` / `llama_memory` | `RAMSafetyManager` (DDR expert tier) + `GpuTopology` (VRAM expert tier) |
| `ggml-cuda/*.cu`, one file per op | `cuda/*.cu` (legacy layered kernels, custom graph ops: see `cuda-direct-usage-audit-2026-10-18.md`) |
| small public header, big `.cpp` | not yet: several 3-5k-line headers hold their implementation |

Naming follows llama.cpp's flat, prefix-named style (`engine_runner_*.cpp`,
`perf_report_*.cpp`) rather than adding directories, so `git mv` churn and merge
conflicts with parallel sessions stay low. `include/perf/` is the one new directory.

## CPU engine state

`CPUEngine` carried seven Qwen4Exp members (`qwen4exp_conv_history`, `_recur_state`,
`_hc_stream_buf`, `_hc_inject`, `_ple_conv_history`, `_ple_token_history`,
`_ple_next_pos`). They are now one `Qwen4ExpCpuState qwen4exp;`
(`include/qwen4exp_cpu_state.hpp`), so the next architecture adds its own small state
struct instead of more members here. An aggregate member, not a polymorphic
`unique_ptr<ModelExecutionState>`: the types are known at compile time, so a base class
and casts would add indirection and no safety. Scope is `CPUEngine` only; the same names
on `StreamingEngine` are used by ~80 sites in the graph files other sessions edit, so
they are left alone. Touches `exl3_decode.cpp` (rename only); EXL3 is WIP and is
verified by compiling, not running.

## Families and versions: one implementation per family

Rule: a `ModelFamily` is one implementation; a version of that family is data, not a
copy. Concretely:

* A new version of an existing architecture (same GGUF tensors, different sizes or
  flags) adds no enum value and no files. Differences come from GGUF metadata
  (`ArchitectureConfig`) or, if needed, a capability flag on `FamilyOps`.
* A new enum value is only for a genuinely different tensor layout. Its
  `FamilyOps` row is then built from the same shared pieces as its siblings, not
  pasted from them.
* Files are named by family (`ggml_<family>_graph*.cpp`), never by version.

How the current code measures up:

| case | state |
|---|---|
| GLM-5.2 and GLM-5.3 (non-Flash) | One family, `GLM_DSA`, shared by both (and by `DeepSeek_V3`, which reuses the same `FamilyOps` row for its MLA core). This is the model to follow. |
| GLM-5.3-Flash | `GLM5NEXT` is a different upstream architecture (recurrent KDA layers plus absorbed MLA), so it stays its own family, but it must reuse shared pieces instead of copies. |
| DeepSeek-V4 vs V4.1 | Were two families with separate graph builders and attention paths: the counter-example. V4 is deprecated (commit `DEPRECATE DeepSeek-V4`); V4.1 is the DeepSeek family going forward. |
| Hyper-connections | Was pasted into three `FamilyOps` rows (V4, V4.1, GLM5Next). Now `use_hyper_connections(ops, c)` in `models/family_ops.cpp` is the single place; V4.1 and GLM5Next call it. |

Open duplication to fold into shared pieces when the owners of those paths are ready:
the `dsv4_*` helpers still living beside the V4.1 code and used through
`attention_layer.cpp`; the `deepseek_v4_full_graph_output_ready` flag (V4.1 sets it
under a V4 name); and the near-identical per-family `FullGraphModel` classes in
`models/full_graph_models.cpp`.

## CRTP for the full-graph builders: what a survey of the code found

The model boundary is already CRTP (`FullGraphModelBase<Derived>` in
`models/full_graph_model.hpp`: validation and prefill/decode dispatch are static; only the
registry lookup is virtual, 15 virtuals each called once per graph build or run). Pushing it
deeper would unify what the Qwen4Exp and GLM5Next builders duplicate, so I compared the
same-named functions after normalising family names (similarity 1.0 = identical):

| function | similarity | lines (qwen / glm) |
|---|---:|---|
| `type_ok` | 1.00 | 3 / 3 |
| `env_device_expert_layers` | 0.98 | 10 / 10 |
| `expert_weight` | 0.73 | 5 / 5 |
| `env_enabled` | 0.69 | 6 / 8 |
| `resident_tensor` | 0.40 | 85 / 23 |
| `load_full_weight` | 0.36 | 254 / 98 |
| `device_expert_bytes` | 0.33 | 111 / 60 |
| `graph_dense_weight_name` | 0.16 | 66 / 18 |
| `build_graph` | 0.00 | 1090 / 534 |

So the genuinely duplicated code is about 20 lines; everything large has diverged because
the architectures differ (GDN layers vs KDA layers, PLE, per-layer expert placement). A
shared `GraphBuilderBase<Derived>` skeleton would have almost nothing to hold. The earlier
shared pieces (`GraphCore`, `moe_routed_ggml.hpp`, `expert_store.hpp`) already took the
duplication that existed. I did not touch these files: Qwen/GLM5Next graph code is under
active tuning and the gain would be ~20 lines.

Performance: graph construction is once per request (Qwen decode setup: `graph=7.3ms` of
`prepare_total=53ms`, about 0.24% of a 64-token request); per-token work replays the built
graph. Static vs virtual dispatch at the build boundary therefore cannot move tokens/second.
Not applicable here: expression templates with `static_assert` on shapes (dimensions come
from the model file at runtime) and `index_sequence` unrolling over layers (`n_layers` is a
runtime value; it would instantiate a separate huge function per model size).

## Still over 3k lines (not touched)

| file | lines | why left alone |
|---|---:|---|
| `include/engine_config.hpp` | 4.8k | config struct + TOML parse + validation; every tuning session adds keys here, so a split now conflicts with all of them. |
| `src/attention_layer.cpp` | 4.5k | layered MLA/GDN attention: the GLM-5.3 non-Flash path. Explicitly out of scope. |
| `include/ram_manager.hpp` | 4.4k | header-only `RAMSafetyManager`; used by the GLM legacy path. |
| `src/kernels_cpu.cpp` | 4.0k | CPU kernels; GLM legacy path. |
| `include/cuda_manager.hpp` | 3.4k | header-only `CudaManager` (pools + layered `launch_*` wrappers); see audit, follow-up 1. |
| `include/gpu_topology.hpp`, `include/engine.hpp` | 3.4k, 3.4k | header-only; hot in DSV4/GLM tuning. |
| `tests/test_components.cpp` | 3.5k | under the 5k threshold. |

Next candidates, in order of value and risk: (1) move the implementations of the
header-only giants into `.cpp` files (build time, no behaviour change); (2) split
`preload_experts` (1.5k-line method) and `offload_dense_tensors` (0.7k) into
sub-phases with the same member-state technique; (3) a shared
`DeepSeekGraphBase` for `ggml_deepseek_v4_graph.cpp` / `ggml_deepseek_v41_graph.cpp`,
once the DSV4 tuning session is done.

## Merging notes for parallel sessions

Moved code is verbatim, so a hunk written against the old location applies by
identifier: `run_engine` hunks go to the `EngineRunner::<phase>` that contains the
same lines; `PerfCounters` fields and `dump_report` hunks go to
`include/perf/counters_*.hpp` and `src/perf_report_dump_*.cpp`; `main_test.cpp`
hunks go to `tests/native/tests_*.cpp`.
