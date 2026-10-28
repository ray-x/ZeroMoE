# Handoff, 2026-10-05: legacy VRAM cache removal, expert-budget sweep, global-pool verdict

Everything done on 2026-10-05. Read together with `docs/option-b-vram-budget-2026-10-05.md` (findings, tables, how to reproduce) and
`docs/handoff-2026-10-03-option-b-global-pool.md` (global-pool background; its section 8 holds the teacher-forced numerics).
Hardware: RTX 3090 24 GiB (220 W limit), RTX 5060 Ti, Xeon W-2235 (Gen3 PCIe, 6C/12T), 125 GB RAM. Model for everything below: Q2_0
(`/path/to/models/qwen3.8-flash-next/Q2_0`). Engine: `../moestream-build-upstream-master/moestream`, built from this tree.

## 1. Standing rules (from the user)
- Commit as `ray-x`, **no Claude co-author trailer**, only when asked. Use the upstream integration ggml.
- **Design policy (new today): no layered decode; every supported model needs a full-graph decode unless exempted; a failed graph is a failed request.**
  Other families (GLM5Next, DeepSeek-V4, ...) still use the shared layered/legacy-cache code: do not delete it for them. Memory note: `design-policy-full-graph-only-no-layered`.
- A consistent 1.5-4% gain over ~6 runs is a valid change. Read the generated text before trusting any tok/s.
- Shell pitfalls: zsh does not word-split variables (`set -- $x`, `env $e`): pass arguments explicitly. 12 OpenMP threads collapse 3-25x under any other CPU load: use
  `ZEROMOE_QWEN_CPU_THREADS=10` when anything else runs. Another session (ExLlamaV3, Codex's graph refactor) commits to the same repo concurrently: re-read `git log` first.

## 2. Commits made today (all ray-x, no trailer)

| Commit | Content |
|---|---|
| `99fd1ac` | Option-B global expert-pool selection, off by default (`qwen4exp_option_b_global_top_k`, `..._global_floor_per_layer`), config `configs/testing/qwen4exp_q2_0_option_b_global_topk48.toml`, `bench/option_b_global_pool_ab.sh`, unit test |
| `4d97b26` | Qwen4Exp full graph: legacy VRAM expert cache and active-layer area never created when the decode graph is on |
| `2138f09` | Fix: `main.cpp` block was above the declaration of `qwen_full_graph_decode` (did not compile) |
| `2cc77bf` | Fix: `tests/test_components.cpp` global-pool test was inside another function (did not compile) |
| `e207c1c` | Opt-in `ZEROMOE_QWEN4EXP_GRAPH_ALLOC_FLOOR_MIB` (64..4096) override of the graph allocator admission floor; default unchanged |
| `bb9d713` | `docs/option-b-vram-budget-2026-10-05.md`, `bench/option_b_vram_probe.sh`, `bench/option_b_vram_stress.sh`, `bench/option_b_survey_coverage.py`, handoff sections 8-9 |

**Lesson:** `99fd1ac`/`4d97b26` were made by staging zero-context hunks (`git apply --cached --unidiff-zero`), which mis-placed code in two files; the working tree I had built and tested
was correct, the committed tree was not. After any partial-staging commit, export HEAD (`git archive HEAD include src tests | tar -x -C /tmp/x`) and compile it with the zeromoe target flags
(`../moestream-build-upstream-master/CMakeFiles/zeromoe.dir/flags.make`, paths rewritten) with `-fsyntax-only`. Clean HEAD compiles `src/main.cpp`, `src/ggml_qwen4exp_graph.cpp` and `tests/test_components.cpp` as of `bb9d713`.

## 3. Results (details and tables: `docs/option-b-vram-budget-2026-10-05.md`)

1. **Legacy VRAM cache = 6.44 GiB of dead VRAM.** 4,650 slots allocated, `0 / 4650 slots filled`, `0 hit / 0 miss`, `Preloaded 0 VRAM-pinned`. Config A had `moe_cache_gb = 11.5` inherited from the older non-graph config.
   Off: VRAM used at decode 20.65 -> 14.64 GiB (free 3.35 -> 9.36 GiB), speed/output unchanged, 402-token prefill unchanged, plain 22/26 still works.
2. **Global ranking is not the lever.** Cache off, K = 8,421, 7 order-balanced reps: A (right-sized 44 layers) 31.00 tok/s; V1 (global ranking, same 44 layers) 31.06 (+0.2%); B (global, all 48 layers) 31.49 (+1.6%, 7/7). The gain is from adding hot experts to layers 0, 1, 2, 22 (48/0), not from the ranking.
3. **More hot experts in the freed VRAM is the big lever.** K 8,421 -> ~14,500 (34% -> 59% of 24,576 experts): ~30 -> ~39 tok/s (single runs). Slab 1.318 MiB/expert; survey-expected coverage 76.3% -> 93.3%.
4. **Admission floor.** K = 14,746 leaves 0.98 GiB < the 1.00 GiB floor at the decode-graph build -> engine falls back to layered decode, now with no cache: **all-CPU, ~835 ms/token**. K <= ~14,729 passes by default;
   with `ZEROMOE_QWEN4EXP_GRAPH_ALLOC_FLOOR_MIB=512` K = 14,746 builds (steady free 1.38 GiB, ~39 tok/s, coherent).
5. **Stress sessions** (5 requests 28/402/28/402/28 tokens, one process): K = 13,100 / 14,458 / 14,554 / 14,746 all completed with no fallback or errors; min free VRAM 2.89 / 1.56 / 1.44 / 1.19 GiB; 28-token requests 39.4 / 41.9 / 41.6 / 42.1 tok/s.
   The engine releases the arena before a prefill that needs more workspace than is free (policy: 1.0 GiB floor + 0.5 GiB + 7.5 MiB/token) and rebuilds it for decode (3.5-3.9 s); at K >= 14,554 even a 17-token prompt triggered that on the first request.
6. **Numerics (teacher-forced vs config A, `bench/option_b_global_pool_check.py`).** Determinism control bit-identical. 4 ordinary prompts x 96 tokens: global layouts mean KL 0.0007-0.0009, top-1 ~99% (same as per-layer). Identifier-copy prompt: all layouts diverge (steps 16, 12, 28, 34), step 12 flips only when layers 0/1/2/22 are split layers. Not run at K >= 13,100.
7. Read-only review of `graph-design.md` (a 5 MB Codex transcript at the repo root): Codex committed an early-ready scheduler (private CPU cold-expert graph started by an input-ready event; per its docs +6.4%, 31.85 -> 33.87 tok/s, byte-identical logits), staged-GPU execution **rejected on parity evidence**, shape-class workers, SSD storage service, adaptive-policy foundation. I did not verify those numbers or edit that work. Its docs: `docs/graph-refactor-current-state-2026-10-05.md`, `docs/option-b-graph-refactor-steps-2026-10-04.md`.

## 4. State of the tree
- Committed: everything in section 2. **Untracked, mine:** `configs/testing/qwen4exp_q2_0_option_b_rightsize_split44_x0_cov80.toml` (baseline config A used by all scripts and the howto; other `configs/qwen4exp_*option_b*` files are older untracked configs from earlier sessions) and scratch runners in `logs/` (`vram_headroom_probe.sh`, `vram_stress_session.sh`, `zeromoe_q2_0_3090_run.sh`, `rank_only_ab.sh`, `zeromoe_global_matrix*.sh`, `gp_check_multi.sh`, `zeromoe_copy_test.sh`, `glm53_neurall_run.sh`) plus their outputs (`logs/stress_K*`, `logs/option_b_global_pool_check_20261005*`, `logs/rank_only_ab_20261005.log`, `logs/ms_*`). The committed scripts in `bench/` supersede the scratch ones.
- **Not mine, do not commit with this work:** ExLlamaV3 changes (`include/exl3_*`, `src/exl3_*`, `exllamav3_backend`, `docs/exllamav3-backend.md`, `main.cpp`/`engine_config.hpp`/`test_components.cpp` exl3 hunks), Codex's graph WIP, `bench/option_b_overlap_spike.cu`, `docs/GLM5.3-Flash-baseline.md`, other modified configs/docs.
- Memory notes added or updated (in `~/.claude/projects/-home-ray-Projects-MoEStream/memory/`): `design-policy-full-graph-only-no-layered`, `option-b-global-pool-handoff`, `pcie-gen3-limits-expert-staging-5060ti-baseline`, `strata-bench-inflated-by-suffix-drafts-and-garbled-pcie-frac`.

## 5. Reproduce (repo root, idle 3090, engine built, model verified with `gguf_tool verify`)
```bash
python3 bench/option_b_survey_coverage.py                                       # sizing/coverage table
bench/option_b_vram_probe.sh 14554                                              # steady VRAM + decode tok/s
ZEROMOE_QWEN4EXP_GRAPH_ALLOC_FLOOR_MIB=512 bench/option_b_vram_probe.sh 14746 # above the default floor
bench/option_b_vram_stress.sh 14554                                             # 5-request session
bench/option_b_global_pool_ab.sh 0 7 128                                        # A/B/C/D, 7 reps
python3 bench/option_b_global_pool_check.py --prompt-file <1-line prompt> --out-dir logs/<dir> --ngen 64   # teacher-forced vs A
```

## 6. Decisions and next steps (in the order I would take them)
1. **Choose the shipped expert budget.** K = 14,458-14,554 (59%) passes the default floor, keeps 1.4-1.6 GiB free, and is within ~1% of K = 14,746 on speed. Decide whether to keep the floor override (K = 14,746) or not.
2. **Stress what is untested:** prompts of ~100-300 and 1,000+ tokens, a 15+ request session, filling the 2,048 context, a larger context size; log free VRAM inside the engine instead of 250 ms nvidia-smi samples (it can miss spikes). Teacher-forced check at the chosen K.
3. **Apply the budget to the configs:** the Q2_0 and IQ1_M Option-B configs still carry the 8,421-expert right-sized sets and `moe_cache_gb = 11.5`. Decide whether the shipped config becomes a global pool (48/0) with the new K, and re-derive K for IQ1_M (different slab size) and for the 5060 Ti.
4. **Policy stage 2:** make full-graph decode the Qwen4Exp default and turn the five layered-decode fallback exits in `src/ggml_qwen4exp_graph.cpp` (CPU-offload without host-node bridge, decode state not seeded, layer without a complete DDR arena, KV capacity growth x2, Phase-2 mode) into hard errors that say why. **Stage 3:** delete the Qwen4Exp layered leftovers (legacy-cache code paths for this family, layer-13 staging proof, staged-GPU path, old partial-layer phases). Coordinate with Codex's graph refactor first (same files).
5. Possible follow-ups: stack this with Codex's early-ready scheduler (untested together); an adaptive hot-set tier fed by the freed VRAM (needs updatable remap/mask tables, which Codex's resident-table work may provide).

## 7. Known weak points of today's evidence
Single runs for the K sweep and stress; only F2 (ranking vs 48 layers) has 7 reps. Coverage numbers are survey predictions. Two prompt sizes only. Output text checked by eye except in the teacher-forced runs.
Why later requests showed fewer arena release messages than I expected (1 release in 5 requests at K >= 14,554) is not understood. The floor override has no safety evidence beyond the runs listed.
