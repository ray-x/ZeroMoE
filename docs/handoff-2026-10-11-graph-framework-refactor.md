# Handoff: graph framework refactor (2026-10-11)

Branch `graph-framework`, worktree `~/work/moestream-graph-framework`, merged into `master` at `aa28a2a`
(base was `b8cf7ff`; master had moved by 5 commits of striped-store work, one trivial conflict resolved).

## What landed

| Commit | Change | Gate result (Qwen Q2_0 split44 early-ready, 6 pairs; GLM offload) |
|---|---|---|
| `7a78079` | `include/graph_config.hpp`: drop-in `getenv` that records lookups, typed readers, `[GRAPH-CONFIG]` block at bootstrap/shutdown | logits identical, +0.75%, native 171 |
| `7f210cb` | `include/expert_store.hpp`: ExpertCache + VirtualArena + readers + residency behind one object; GLM uses it | logits identical, +0.17%; GLM lru/prefetch/stage identical |
| `9ff6ef6` | `include/moe_routed_ggml.hpp`: `gate_probs/select_ids/route_weights/expert_ffn/sum_topk` shared by GLM and Qwen builders | identical, -0.46%; GLM identical |
| `208633a` | `include/early_ready_lane.hpp` (was `qwen4exp_early_ready.hpp`) generalised, GLM CPU lane; conditional-node master graph, Phase 2 dynamic dispatch, completion-stage builders removed | identical, -0.13%; GLM lru+lane identical |
| `89e4fba` | storage-service / completion-graph / adaptive-policy headers + tests, `Phase2Mode` and ~80 `phase2_*` members removed; shared arena members renamed (`expert_source_graph`, `dense_source_graph`, `state_source_graph`, ...) | identical, -0.40% |
| `d9921ff` | DeepSeek-V4 split into `src/ggml_deepseek_v4_graph.cpp`; `FullGraph` in `src/glm5next_graph_internal.hpp` (namespace `glm5next_graph`) | identical, +0.12% |
| `a890005` | typed readers at 39 call sites, `moe_routed::router_spec()` | identical, +0.74% |
| `f9613ad` | `models/family_spec.hpp` (`FamilySpec`: router_spec, supports_cpu_lane, graph_ops); GLM CPU lane **on by default** | identical, +0.16%; GLM lru (lane on) 4.22 -> 4.29 |
| `aa28a2a` | merge of master (striped store `open_auto`, now called from `expert_store.hpp`) | identical, -0.21% (rerun), native 172 |

Step 4 (lane only, before removals) was also gated on its own: logits identical, 34.00 -> 34.00 tok/s (git note on `208633a`).

Sizes: `ggml_qwen4exp_graph.cpp` 16,097 -> ~11,200 lines; `ggml_glm5next_graph.cpp` 8,513 -> ~6,900.

## Behaviour changes to know about

- **GLM CPU lane is ON by default** (`MOESTREAM_GLM5NEXT_EARLY_READY=0` opts out). Four interleaved pairs, same binary:
  4.53 -> 4.62 tok/s (+1.9%, 4/4 wins), logits and tokens identical. Applies only to offloaded GLM layers on
  families whose `FamilySpec::supports_cpu_lane()` is true; declines when pilot staging is active or `hot1` is non-empty.
- **Removed features** (all were off by default): CUDA conditional-node master graph, Phase 2 (3-graph per-token GPU/CPU dispatch),
  completion-staged/storage-service path, Phase 1 static partial-layer GPU commit (`qwen4exp_partial_layer_committed` is now unused),
  `cuda/qwen4exp_master_graph.cu` (its staging-ring kernels live in `cuda/qwen_stage_kernels.cu`),
  `MOESTREAM_QWEN4EXP_STORAGE_SERVICE` / `COMPLETION_GRAPH` / `PARTIAL_LAYER_PHASE2` env knobs.
  Tag `gf-before-phase2-removal` marks the tree before removal (it predates steps 2-4; use `7a78079` + `git show` for the old code).
- `moestream_ggml_sched_early_cpu` is now defined once in `ggml_qwen4exp_graph.cpp`. An `inline` definition in the shared header is not
  emitted unless used and silently disables early-ready (this broke the first Step 4 build; the Qwen gate caught it).

## Not done / open

- **DeepSeek-V4 is compile- and link-checked only** (no model on the box). Its builder moved files and its stubs; run it when a model is available.
- Numeric `getenv` readers with non-trivial parsing (strict `strtol`, clamps, multi-line) still use `graphcfg::env()`; they are recorded in
  the effective-config log but not converted. `atoi_or` / `atof_or` preserve legacy semantics where the idiom matched exactly.
- The shared `Phase2Mode`-era comments in `ggml_qwen4exp_graph.cpp` (about 25 mentions) are stale text only.
- GLM prefetch/stage arms abort with a CUDA error at process exit in the baseline binary too (decode completes first); not from this work.
- `FamilySpec` sits on top of the existing `ModelGraphOps` dispatch in `models/family_ops.cpp`; it does not replace it.
- Bigger speed levers from the review are untouched: pinned DDR slab pool for DMA staging (GLM), cheaper IQ2_XXS CPU kernel,
  prefill-histogram admission into the DDR cache. See the review in the 2026-10-11 session notes / `docs/ssd-offload/09-*.md`.

## How to re-run the gates

- Frozen baseline binary: `~/work/gf-baseline/moestream-master-b8cf7ff` (built from master `b8cf7ff` against the prebuilt accepted ggml
  `~/Projects/moestream-build-upstream-master/ggml-build`, `-DMOESTREAM_GGML_PREBUILT_DIR=...`, CUDA arch 75).
- Qwen: `bench/graph_framework_step.sh <name> <candidate-binary>` -> `logs/graph_framework/<name>/{correctness,free,perf}`;
  needs `logs/graph_framework/inputs/{prompt,Q2}.tokens` (copied from `logs/quant_decode_compare_chat_20261004/`).
- GLM: `ARMS="lru lane" bench/graph_framework_glm_gate.sh <name> <candidate-binary>`; set `WAIT_IDLE` (seconds) for the Qwen harness.
- Both need the 3090 free of other compute processes and hold `~/work/.no-arms` while running (the prefetch session's queue yields to it).
  A foreign process appearing mid-run aborts the Qwen harness; re-run.
- Noise floor: about +-2.4% per six-pair mean (baseline binary against itself: 32.6 vs 33.4).
- Native suite: configure a tests tree with `-DMOESTREAM_EXLLAMAV3_SRC=~/Projects/exllamav3/exllamav3/exllamav3_ext -DCMAKE_CUDA_ARCHITECTURES=86`
  and run `moestream_tests`; without the EXL3 source the EXL3 CUDA test fails (build-config gap, not a code bug).

## Gotchas hit

- Shell here is zsh: `set VAR a b c` is wrong, use arrays; `pkill -f <name>` can kill your own shell; fish heredocs do not work.
- `pgrep -c ... || echo 0` prints two lines when the count is 0 and never equals `0`; use `| wc -l`.
- Names containing `phase2_` were not all Phase 2: `phase2_bind_or_allocate_expert`, `phase2_expert_ptr_text`, `phase2_expert_arena_owner`
  and the `phase2_*_source_graph` members are the prefill->decode arena-borrowing core (renamed, kept).
- Other sessions use the same GPU; gates starve for hours if another session runs back-to-back arms.

Artifacts: gate logs under `logs/graph_framework/` (git-ignored) in the worktree, frozen binaries and edit scripts under `~/work/gf-baseline/`.
