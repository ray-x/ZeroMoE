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
