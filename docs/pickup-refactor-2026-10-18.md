# Picking up the 2026-10-18 source refactor (for the DeepSeek, GLM and other sessions)

Branch `refactor/source-split` (worktree `~/work/moestream-refactor`), on top of the
rewritten `master` (`703dde9`). Not merged. The merge decision is the user's.

## 0. First: master was rewritten, most branches are on the old history

`git merge-base master <branch>` prints nothing for `deepseek-v41-flash`,
`graph-framework`, `prefetch-prediction`, `graph-ssd-offload`, `gguf-pack-interleave`
and `exl3-followup`: they share no ancestor with master (`git merge` says "refusing to
merge unrelated histories"). Their commit hashes are not in `.git/filter-repo/commit-map`
either (an earlier rewrite, the trailer strip, produced different hashes again). Only
`glm53-layered-speed` is on the new history.

Do not try to rebase by hash. Transplant by content:

1. `git cherry master <your-branch>` lists your commits as `+` when master lacks the
   patch and `-` when an equivalent is already in master (ignore the old `init commit`).
2. Cherry-pick the `+` commits, oldest first, onto a branch cut from master:
   `git switch -c <name>-on-master master && git cherry-pick -x <c1> <c2> ...`

## 1. What the refactor changes (code is verbatim, only its location moved)

| old | new |
|---|---|
| `src/main.cpp` `run_engine()` (4,160 lines) | `EngineRunner` (`include/engine_runner.hpp`); phases in `src/engine_runner*.cpp`: `apply_engine_config` and `init_host_and_gpu` (engine_runner.cpp), `load_model` (_model), `plan_expert_cache` and `setup_vram_expert_cache` (_cache), `offload_dense_tensors` (_dense), `init_ram_tier` (_ram), `preload_experts` (_preload, includes the `[PIN]` block), `start_workers`, `run_repl`, `shutdown_and_report` (_session) |
| `include/perf_timer.hpp` (5.9k lines) | `include/perf/counters_NN_*.hpp` (new counter: append to the LAST slice), `include/perf/perf_counters.hpp`, `src/perf_report_{summary,dump_a,dump_b,stats}.cpp` (report line: the matching section) |
| `src/main_test.cpp` | `tests/native/tests_*.cpp`; `main()` and EXL3 tests stay in `main_test.cpp` |
| `src/ggml_deepseek_v4_graph.cpp` (DeepSeek-V4, not V4.1) | deleted; `graph_alloc_vram_headroom_ok` is in `src/ggml_graph_vram_headroom.cpp` |
| hyper-connection wiring pasted in 3 rows | `use_hyper_connections(ops, c)` in `models/family_ops.cpp` |
| 7 `qwen4exp_*` members of `CPUEngine` (not `StreamingEngine`) | `CPUEngine::qwen4exp` (`include/qwen4exp_cpu_state.hpp`); touches only `cpu_qwen4exp.cpp`, `exl3_decode.cpp`, `cpu_main.cpp` |
| `early_ready_lane.hpp`, `qwen4exp_stage_pool.hpp` hand-freed CUDA resources | `include/cuda_raii.hpp` owners (`DeviceBuffer`, `PinnedBuffer`, `CudaEvent`, `CudaStream`), test `cuda_raii_test` |

How to move a `main.cpp` hunk: take the refactored `main.cpp`, then
`git show <commit> -- src/main.cpp | sed 's#src/main.cpp#src/<phase file>#' | git apply --recount -C1 --whitespace=nowarn --reject`.
The hunk context is verbatim code, so each hunk lands in whichever phase file contains
those lines (the `--reject` output says which did not).

## 2. Rehearsal that was actually run (git only, nothing built or executed)

Branch `integration/refactor-ds-glm-rehearsal`, worktree `~/work/rehearsal-ds`, tip
`8a9ddf3`, 16 commits on master: my refactor + the DeepSeek commits + the GLM commit.

| step | result |
|---|---|
| DeepSeek commits master lacks (`git cherry`): `84a8e77`, `0c26807`, `38a88e5`, `2b02cf1`, `3394afa`, `69eeaae` | cherry-pick onto master: 0 conflicts. (Cherry-picking only the last three conflicts in 4 files, because master lacks `v41_build_cache_moe` from the first two.) |
| merge `refactor/source-split` into that | 1 conflict, `src/main.cpp` only, as predicted |
| `3394afa`'s `main.cpp` hunks (5) | hunk 1 (`dsv41_graph_rows`, expert-stride block) -> `engine_runner_cache.cpp`; hunks 2-5 (`pin_global_tiers`, 109 lines) -> `engine_runner_preload.cpp`; 115 insertions / 1 deletion, same as the original |
| GLM `5d0aacb` (`GLM_DSA` 4 KiB stride) | applies cleanly to `engine_runner_cache.cpp` on top of the DeepSeek hunk; the two edit the same block but different conditions, no interaction |
| syntax-only compile (`g++ -fsyntax-only`, 12 files: both new engine_runner files, main.cpp, the DSV4.1 graph files, cpu_expert_executor, graph_initializer, attention_layer, execution_layer, ...) | 0 errors |

NOT done: no link, no run, no benchmark of the combined tree (both other sessions were
using the machine). Treat it as "applies and compiles per file", not "verified".

## 3. What each session does

* DeepSeek V4.1 (`moestream-9f`): your 7 branch commits reduce to the 6 listed above on
  master. Pick up `integration/refactor-ds-glm-rehearsal`, or redo steps 1-2 yourself.
  Either way, your future `main.cpp` edits go into the phase files above.
* GLM-5.3 (`moestream-cc`): `5d0aacb` is already on the new history; its single hunk
  lands in `engine_runner_cache.cpp` (see the rehearsal). New counters: last
  `include/perf/counters_NN` slice + the matching `perf_report_dump_*` section.
* Everyone: nothing here touches `src/ggml_deepseek_v41_graph*.cpp`,
  `attention_layer.cpp`, `ram_manager.hpp`, `token_dispatch_pipeline.cpp`,
  `kernels_cpu.cpp`, `cuda_manager.hpp` or `engine_config.hpp`.
