# Configuration knob cleanup handoff — 2026-10-30

## Worktree state

- Worktree: `.worktrees/ling-env-cleanup`
- Branch: `codex/ling-env-cleanup`
- Base HEAD before the checkpoint: `0938cd4b36a01a2e0420b187043af4c600b8814c`; the checkpoint commit is the current tip of `codex/ling-env-cleanup`.
- Checkpoint contents: 125 modified tracked files plus this handoff; pre-commit tracked diff SHA-256 `f365b9189650c4f9695a25954323a6f77a0a448410c64fcfa84c869e2904a982`.
- The accumulated worktree state is committed as a checkpoint on this branch; `master` was not changed.
- `master` advanced independently to `58c68f0edbf59148432daf43d937bd7e785cae73` while this branch stayed at `0938cd4`; the branch is 19 commits behind and 0 ahead. No merge or rebase was attempted, and the dirty tree was preserved. Master's new commits also touch six files that are dirty here (`README.md`, `docs/models/glm53-flash.md`, `models/glm/ggml_glm5next_graph.cpp`, `models/glm/ggml_glm5next_graph_decode.cpp`, `models/glm/ggml_glm5next_graph_weights.cpp`, `models/graph/early_ready_lane.hpp`), so a future merge may need manual reconciliation. The router, Qwen settings, resolver, inventory tool, and `docs/env-vars.md` are untouched by that master range.

## Completed in this round

Removed seven live environment names, taking the runtime inventory from 129 to 122:

1. `ZEROMOE_QWEN_EARLY_READY_ROUTE_TRACE` — route records now emit at DEBUG under `[EARLY-READY-ROUTE]`. `bench/option_b_topic_check.py` filters capture and candidate engine logs into the existing route JSONL artifacts.
2. `ZEROMOE_GLM5NEXT_GRAPH_STRICT` — the shared planner already requires a graph for a `Fits` plan. `bench/glm53/run_dsa_graph2gpu.sh` now runs `check_full_graph_moe_run.py --full-fit` after successful inference.
3. `ZEROMOE_QWEN4EXP_PARTIAL_LAYER_PREFILL_CAPTURE` — removed the override that restored a prefill capture unused by Phase 2 and known to perturb later-layer routing.
4. `ZEROMOE_QWEN4EXP_PARTIAL_LAYER_PHASE2` — removed its stale prefill-state and capture bypasses; the conditional-node Phase 2 implementation was already gone.
5. `ZEROMOE_QWEN4EXP_PARTIAL_LAYER`
6. `ZEROMOE_QWEN4EXP_PARTIAL_LAYER_MIN_COVERAGE`
7. `ZEROMOE_QWEN4EXP_PARTIAL_LAYER_TOP_N`

The last three selectors and their now-unreachable Phase 1 partial-layer planner, GPU-only graph branch, prefill route taps, and legacy predictive partial-pinning diagnostics were removed. The accepted early-ready and Option B hot/cold paths remain. Historical benchmark and how-to records remain for provenance; they are not live readers.

No MTP-related environment variable was changed. The partial-layer feature was opt-in, so this cleanup does not establish a new default decode speed. No model run was made and no new throughput claim is made; the recorded partial-graph performance result remains historical.

## Follow-up in this round

- Removed the opt-in DeepSeek V4.1 cache-aware routing experiment, `ZEROMOE_DSV41_CACHE_BONUS_RAM` and `ZEROMOE_DSV41_CACHE_BONUS_VRAM`. The default unbiased top-k router path is unchanged; the experiment changed routed experts and its quality cost was never measured. No maintained script or test set either variable.
- Removed dead assignments/comments for already-removed names from `tools/check_adaptive_placement_accuracy.py`, `tools/bench_qwen4exp_mtp_suite.py`, `bench/glm53/run_ab.sh`, `configs/qwen4exp_q2_0_option_b_scheduler.env`, and `models/qwen/ggml_qwen4exp_graph_settings.cpp`. The placement gate now pins the TOML `qwen4exp_full_graph` / `qwen4exp_full_graph_decode` keys instead of setting the retired environment names.
- Corrected 74 audit rows whose `removal_status` still said `NOT STARTED` even though their `review_decision` and the current inventory already recorded the removal.
- Updated the memory-guard documentation: `ZEROMOE_MEM_GUARD_FLOOR_MB` defaults to 10240 MiB, not 20480; the shipped 16384 value raises the floor rather than relaxing a 20 GiB default.

## Current inventory

`tools/config_inventory.py --json` reports:

| Surface | Count |
| --- | ---: |
| Runtime environment names | 120 |
| Test-only environment names | 18 |
| CLI options | 101 |
| Graph settings | 35 |
| TOML settings | 232 |
| Macros | 7 |
| Build cache inputs | 4 |
| ZeroMoE CMake options | 1 |

Model-prefix counts among runtime environment names: Ling 0, GLM5Next 17, DeepSeek 9 (`ZEROMOE_DEEPSEEK_V41_*`) plus 7 `ZEROMOE_DSV41_*` pilot controls, MiMo 1, and Qwen 38. The remaining MiMo control, `ZEROMOE_MIMO2_INGRAPH_MOE`, stays pending routed-expert parity and its regression witness. `ZEROMOE_GLM5NEXT_ROUTE_TAP` remains because it creates usage-profile data. GLM prefill graph selection remains opt-in while its documented CUDA route-compaction limitation remains unresolved. Qwen probe, prompt, and MTP controls were not broadly pruned in this batch.

The audit JSON and markdown summary are refreshed at [configuration-knob-audit.json](../design/configuration-knob-audit.json) and [configuration-knob-audit.md](../design/configuration-knob-audit.md).

## Verification

- `python3 tools/config_inventory.py --check`: pass at 120 runtime environment names.
- Python syntax: `py_compile` for `tools/check_adaptive_placement_accuracy.py` and `tools/bench_qwen4exp_mtp_suite.py`; AST parse for `bench/option_b_topic_check.py` and `tools/check_full_graph_moe_run.py`: pass.
- Shell syntax: `bash -n` for `bench/glm53/run_ab.sh`, `bench/glm53/run_dsa_graph2gpu.sh`, and `configs/qwen4exp_q2_0_option_b_scheduler.env`: pass.
- `git diff --check`: pass.
- `cmake -S . -B build-cuda`: pass after the source edits refreshed the configure-time provenance snapshot.
- `cmake --build build-cuda --target zeromoe -j2`: pass, including the post-build provenance stamp.
- `python3 tools/build_provenance.py check --binary build-cuda/zeromoe --repo .`: pass before the checkpoint at source `0938cd4b36a01a2e0420b187043af4c600b8814c`, dirty worktree, binary SHA-256 `29c43ab61e77ec51ddff86b0c33ebf3615cc8c0fa975ab665affcafcd81d0bb1`, `ggml_base=unknown`.
- After the checkpoint commit, the binary is rebuilt and re-stamped against the clean commit, and the post-commit provenance check passes. The adjacent `build-cuda/zeromoe.build-info.json` is authoritative for the current commit and binary hash; `ggml_base` remains `unknown`.
- No tests or model validation runs were performed.

## Work left

- Reconcile this branch with `master` before merging; it is 19 commits behind, and six overlapping dirty files may conflict.
- Continue auditing the remaining 120 runtime environment names model by model. Preserve MTP variables, user-critical usage/pin/offload/prompt controls, and TOML settings unless evidence supports a specific change.
- Review the remaining GLM and Qwen experimental controls against current maintained callers and benchmark evidence. Do not interpret retained controls as validated defaults.
- Keep DeepSeek validation skipped as instructed; any future benchmark needs its persistent artifacts and provenance checks.
- Review the remaining seven macros, four build-cache inputs, and one CMake option after the environment pass.
- Run the affected model's baseline only when its cleanup is complete and the user-authorized validation conditions are met.
