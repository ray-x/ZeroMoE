# Handoff, 2026-10-03: Option-B global expert pool (0/48/0), host-node GPU reduce, Strata comparison

**Continuation:** [numeric validation and design decision](option-b-global-pool-validation-2026-10-03.md)
records the teacher-forced comparisons, compact/negative-route controls,
global setup fixes and the decision to keep global mode experimental.
Section 4 below describes the original unresolved state.

Written so a fresh session can continue without this conversation. Everything below was measured on this box:
RTX 3090 (24 GB, **220 W power limit**), RTX 5060 Ti (16 GB, PCIe Gen3 x8), Xeon W-2235 (6C/12T, Gen3 only), 125 GB DDR4.
Standing rules from the user: use the upstream integration ggml (`../llama.cpp-moestream-integration`, build
`../moestream-build-upstream-master`); a consistent 1.5-4% gain averaged over ~6 runs is a valid change; commit under the
user's name only when asked; read the generated text before trusting any tok/s ([[coherence-before-throughput]] in memory).
In the shell, **zsh does not word-split variables** (`set -- $x`, `env $e` fail): pass arguments explicitly.

## 1. State of the working tree (nothing from this session's last phase is committed)

Last commit of mine: `b5e97c9` (host-node GPU weighted reduce). Uncommitted, **mine** (global pool feature):

| File | Change |
|---|---|
| `include/option_b_global_hot.hpp` (new) | header-only `option_b_global::select()`: pooled top-K over (layer, expert) usage counts, per-layer floor and ceiling, deterministic |
| `include/engine_config.hpp`, `include/engine.hpp` | keys `qwen4exp_option_b_global_top_k` (0 = off) and `qwen4exp_option_b_global_floor_per_layer` (default 1) |
| `src/main.cpp` | startup Option-B pass uses the helper in global mode (hot masks, committed layers, coverage); VRAM estimate uses it too |
| `src/ggml_qwen4exp_graph.cpp` | `plan_expert_vram_split` Option-B branch uses the helper; coverage gate, right-sizing and fill_vram are skipped in global mode |
| `tests/test_components.cpp` | `test_option_b_global_pool` (passes: `make test-components ARGS="option_b_global_pool"`) |
| `configs/testing/qwen4exp_q2_0_option_b_global_topk48.toml` (new) | all 48 layers selected, `top_n=512` (per-layer ceiling = none), `global_top_k=8421` |
| `bench/option_b_global_pool_ab.sh` (new) | 4-way A/B (see section 3) |

Other modified files in `git status` (bench/*, other configs, safetensors_adapter, tools/*, glm5next, batched_dispatch) are **not mine**: do not
commit them with this work. Built into `../moestream-build-upstream-master` (`make -j6 zeromoe`).

## 2. What the global pool is, and the prediction

Option-B per-layer mode gives each selected layer its own top-N. Global mode ranks every (layer, expert) pair of the selected layers by survey
usage (`tests/moe_usage_survey/zeromoe_usage_survey_qwen4exp_usage_tap`, text lines `layer expert count`; every layer routes 159,000 so counts compare
across layers) and keeps the top K. Each layer with usage keeps >= 1 expert, so listing all 48 layers gives 0/48/0.

**Important correction to the premise.** The current "0/44/4" config is *right-sized*, not 50% per layer: `right_size_top_n=true` trims each layer to the
smallest set reaching 80% survey coverage, so it holds **8,421 experts (10.84 GiB, mean 191/layer, 132-246)** in 44 layers; layers 0, 1, 2, 22 stay plain CPU
(low coverage). Survey-expected hit rate: current 73.4%; global K=8,421: 76.3% (+2.9 pts, misses/layer 2.66 -> 2.37); global K=11,264 (14.5 GiB): 86.0%.
Per-expert slab = ~1.38 MB, uniform Q2_0 across all 48 layers (checked in the GGUF header).

## 3. Measured results (Q2_0, 3090, chat prompt, 128 tokens, 12 threads, quiet box, 7 order-balanced reps)

`bench/option_b_global_pool_ab.sh` reproduces this.

| Config | VRAM | mean tok/s (sd) | vs A |
|---|---|---|---|
| A right-sized per-layer 0/44/4 | 10.84 GiB | 30.78 (0.25) | |
| B global pool, K=8,421 (equal VRAM) | 10.84 GiB | 31.11 (0.29) | **+1.1%** (6/7 reps positive; below the 1.5% bar) |
| C per-layer top-256 x 44 (right-size off) | 14.5 GiB | 34.63 (0.12) | +12.5% |
| D global pool, K=11,264 | 14.5 GiB | 35.21 (0.47) | +14.4% (D vs C +1.7%, partly run-order dependent) |

**Takeaways:** (1) the global ranking alone is worth ~1% at equal VRAM, roughly a third of what the survey predicts; (2) **using ~3.7 GiB more VRAM is the real
lever (+12.5%)** and works with the old per-layer mode; the cov80 configs leave that VRAM unused (dense 4.1 + KV 2 + arena 10.84 of 24 GB). Not yet decided
whether to raise the default budgets.

## 4. OPEN: correctness of the global layouts (do this first)

- On a 402-token mid-sentence prompt ("summarize" of an env-var-heavy doc fragment), configs **B and D (and layer 2 added to the 44-layer set) produced the identical degraded
  summary** (`ZEROMOE_ZEROMOE_...`), while A, C, V1 (global over the original 44 layers), +layer 0, +1, +22, +1&2 quoted `ZEROMOE_QWEN4EXP_HOST_NODE` correctly.
  Non-monotonic in the layer set, all layers are plain Q2_0, so layer 2 is not structurally special. Could be a numerical near-tie flip in a 2-bit model, or a real split-path defect
  that shows when layers 0/1/2/22 (flat routing, few hot experts) become split layers. **Not resolved.**
- The 128-token chat prompt was coherent for all four configs. The 8-prompt exact-identifier copy test I tried is **saturated at ~5% for every config including the baseline**
  (the model truncates long identifiers), so it cannot discriminate; do not reuse it.
- Needed: a teacher-forced comparison (`ZEROMOE_TEACHER_FORCE_TOKEN_FILE`, see `docs/qwen4exp-tg400-state-handoff.md`) of top-1 agreement / KL against config A (or the all-CPU path)
  on real text, for B, D and V1; plus a check of a split layer with very few hot experts (floor=1) for an edge case in the compact-tensor / negative-id-skip path. Until then, treat B/D as unvalidated.
- Long prompts: a 402-token prompt crosses the arena-release boundary for both A (needs ~4.44 GiB, 2.66 free) and D (2.40 free); no CUDA/OOM errors, decode rebuilds the arena. A ~2,000-token
  prompt takes many minutes to prefill on this path (about 30 s per layer) and was abandoned: also note context_size=2048 is too small for a 1,987-token prompt plus generation.

## 5. Earlier work this session (all committed unless noted)

- Host-node GPU reduce: fixed on after the measured +1.65% result; the env opt-out has been removed. (commit `b5e97c9`; documented in `docs/runtime-settings.md` and howto s20).
- **12 OpenMP threads collapse (3-25x) under any other CPU load** (ZeroMoE, llama.cpp, any spinning pool). With a background job use `ZEROMOE_QWEN_CPU_THREADS=10|11`.
  `PRELOAD_LAYERS=0` fails on the 5060 Ti (dense arena declined -> legacy path, 0.26 tok/s): use 1.
- CPU kernel cost by quant (my benchmark `bench/expert_cost_by_quant_bench.cpp`; the mixed gate/up+down variant is only in the session scratchpad): codebook types
  (IQ1/IQ2/IQ3) stream 12-20 GB/s on this Xeon, K-quants/Q4/Q5/Q6/Q8 30-47 GB/s; page size (THP), thread count 10 vs 12 and cache are not factors. Engine ~= benchmark + shared-expert/routing nodes.
  All-CPU rule of thumb: tok/s ~= 1000 / (n_cpu_layers x ~1.2 ms). Real checkpoints: `GSQ-RCO-IQ2_XS_RCO50` is gate/up IQ2_S/IQ2_XXS/IQ1_M + down Q2_0; `UD-IQ1_M` is IQ1_M or IQ2_XXS gate/up + IQ4_NL down.
- PCIe: 5060 Ti Gen3 x8 = 6.9 GB/s H2D; 3090 Gen3 x16 ~12.3 GB/s (Strata probe). Staging all top-k experts is a net loss (howto s12). A partial PCIe share running *in parallel* with CPU work
  is a different, untested design (my earlier "staging loses" analysis was either/or and wrong for that case).
- 5060 Ti baseline (pruned checkpoint, 27 GPU / 21 CPU layers): 25.2 tok/s with threads 10-11. 47 CPU layers: 15.8 tok/s. llama.cpp all-experts-CPU (`-ncmoe 48`): UD-IQ1_M 17.3, Q2_0 16.9 tok/s; GLM-5.3-Flash 3.0bit
  (110 GB, needs neurall's llama.cpp for `glm5-next`): 9.8-10.0 tok/s.
- `/path/to/Strata` (same model family) compared: its `run.md`/`bench_inference.py` 42-47 tok/s is inflated (repeated truncated prompt + suffix-draft speculation; `--spec 4` without `--mtp` still drafts);
  real chat decode 29-30 tok/s vs ZeroMoE Q2_0 0/44/4 29.4-29.8 (same prompt): **equal**. Strata is GPU-bound (util 89.5%, mean 201 W at the 220 W cap, mem util 20%), ZeroMoE CPU-bound (util 41.7%).
  Strata's paper numbers need MTP and a Ryzen 5 7600 + DDR5. User decided to set the Strata comparison aside and **not** to test MTP (adds noise).
  Ideas not taken from Strata that fit our design: adaptive expert swapping, doorbell handoff, parallel PCIe share.

## 6. Suggested next steps

1. Validate section 4 (teacher-forced numeric comparison) before adopting global mode or committing it.
2. If valid: decide the shipped budget (K or GiB) given the free VRAM, and update `qwen4exp_option_b_global_*` docs (`include/engine_config.hpp` default-config comment block, `docs/runtime-settings.md`, howto).
3. Test the larger-VRAM per-layer config on IQ1_M and on the 5060 Ti (smaller card: budget differs); re-derive K from the real arena size.
4. Optional levers: adaptive hot-set swapping; hot-GPU / cold-CPU overlap inside a layer (bounded ~3 ms/token earlier); the 220 W power limit (needs root) caps the 3090 clocks at ~1.7-1.9 GHz.

The overlap lever now has a [standalone real-kernel graph experiment and integration design](option-b-scheduler-overlap-design-2026-10-04.md).

## 7. Files and logs (untracked unless noted)

`logs/ms_cmp_matrix*_20261003.log`, `logs/ms_cmp_*_r*.{stdout,engine.log}` (the 7-rep A/B), `logs/ms_mid_*_M.stdout` (402-token texts), `logs/ms_copy_*` (saturated copy test),
`logs/zeromoe_q2_0_3090_run.sh` and `logs/zeromoe_global_matrix*.sh` (scratch runners), `logs/strata_pcie_frac_20261003.log`, `logs/glm53flash_neurall_allcpu_20261003.log`.
Scratch configs for the variants are regenerated by `bench/option_b_global_pool_ab.sh`. Memory notes (in `~/.claude/projects/-home-ray-Projects-MoEStream/memory/`):
`pcie-gen3-limits-expert-staging-5060ti-baseline.md`, `strata-bench-inflated-by-suffix-drafts-and-garbled-pcie-frac.md`.

## 8. Resolution of section 4 (teacher-forced check, 2026-10-05)

`bench/option_b_global_pool_check.py` (committed harness), idle 3090, 10 CPU threads, engine built 2026-10-05, model verified with `gguf_tool verify`. Logits compared against A.
- Determinism control: A repeated = 100% top-1, KL 0, max delta 0 (bit-identical), so every difference below is real.
- 4 ordinary prompts (96 tokens each; `logs/gp_check_prompts/p1..p4.txt`), mean over prompts, vs A: B (global K=8421) KL 0.00094 / top-1 99.2%; C (per-layer 256, 14.5 GiB) 0.00074 / 99.0%; D (global K=11264) 0.00088 / 99.2%; V1 (global, 44 layers) 0.00086 / 99.2%; floor1 (1 hot expert per layer) 0.00091 / 99.5%. Global layouts are not distinguishable from the per-layer one.
- 402-token identifier prompt (`logs/option_b_global_pool_check_prompt.txt`, 64 tokens): KL B 0.0217, C 0.0188, D 0.0195, V1 0.0047, floor1 0.0219. Dominated by step 16 (KL ~0.7 in B, C, D and floor1, so NOT global-specific) and step 12. Steps 28 and 34 flip top-1 in every layout (reference margins 0.022 and 0.07 logits = pure near-ties). Step 12 (margin 0.59 logits) flips only in B, D and floor1, the layouts that make layers 0, 1, 2 and 22 split layers; C and V1 keep them plain CPU. One event; its cause is not isolated.
- Reading: divergence from A grows with how many experts change device placement (V1 closest to A, lowest KL); it is numerical sensitivity at fragile identifier-copy positions of a 2-bit model, not a global-pool routing bug. The earlier degraded text was one near-tie flip at step 12, amplified by greedy decoding.
- Still open (minor): whether a split layer with very few hot experts shifts logits slightly more than a plain-CPU layer (B 0.00094 vs C 0.00074 on ordinary prompts is within noise). Isolate by comparing hidden states per layer, or more identifier-style prompts.
- Decision input: global ranking gains +1.1% at equal VRAM (below the 1.5% bar); the VRAM budget is the real lever (+12.5%, quality-neutral per the numbers above). Global mode stays OFF by default.
Raw results: `logs/option_b_global_pool_check_20261005*/` (summary.json, <layout>.comparison.json).

## 9. Update 2026-10-05: legacy cache removal and the VRAM budget

The 6.4 GiB legacy VRAM expert cache was allocated but never filled in Qwen4Exp full-graph runs; it is now disabled automatically (`4d97b26`, fixes `2138f09`, `2cc77bf`).
The freed VRAM, spent on hot experts, is the dominant speedup (K = 8,421 -> ~14,500: ~30 -> ~39 tok/s); global ranking alone is worth ~0.2%, all-48-layers +1.6%.
Findings, the K sweep, the admission-floor failure at K = 14,746 and the stress sessions, with reproduction commands: `docs/option-b-vram-budget-2026-10-05.md`.
