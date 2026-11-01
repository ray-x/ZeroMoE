# Qwen4Exp MTP shared handoff

Updated 2026-10-30. Promo and MTP prefill-overlap controls are now typed Qwen graph settings (`graph_promo_*`, `graph_mtp_prefill_overlap`); `ZEROMOE_MTP_GREEDY_FAST_PATH`, `ZEROMOE_QWEN4EXP_EXPERT_PROMOTION`, and `ZEROMOE_QWEN4EXP_ROUTE_IDS_DUMP` were removed. Refresh the current HEAD and worktree list before using any
binary or citing a result.

## v0.0.1 baseline: plain decode and MTP (2026-10-28, locally measured)

This supersedes every rate quoted further down. Measured source: `ea2b200` = master `197cdfa` merged with `swap-b`, plus two compile fixes to master (`2528868` `core/main.cpp`, `ea2b200` GLM5NEXT decode diagnostics; master did not build without them); the tagged `v0.0.1` commit adds only documentation on top. Built in a clean worktree; the binary passes `tools/build_provenance.py check` (source `ea2b200`, `dirty=False`, ggml base `8baabd53b`, native `sm_86` only). Toolchain: CUDA 13.3, driver 610.57.04, GCC 16.2, RTX 3090 only (`CUDA_VISIBLE_DEVICES=0`; the 5060 Ti is not used), Xeon W-2235 (6C/12T, DDR4). ggml: `patches/ggml` 0001-0017 plus `patches/ggml-host-node` 0001-0003, built with `scripts/build_ggml.sh`.
Config: `configs/bench/qwen38-flash-q2_0-mtp.toml` (13,000-expert global hot set, `expert_promotion = "on"`, 3 draft tokens, Q4_K_M draft). Greedy decoding, 3 passes per cell, medians (min-max), tok/s:

| prompt | prompt tok | generated tok | plain decode | MTP decode | speed-up | draft acceptance | MTP end-to-end* | prefill (MTP run) |
|---|---|---|---|---|---|---|---|---|
| palindrome (code, low miss) | 29 | 463 | 48.3 (47.7-48.6) | **83.0** (83.0-83.3) | 1.72x | 79.3% | 72.8 | 0.73 s |
| code review | 115 | 500 | 49.3 (49.1-50.3) | **78.3** (77.9-79.3) | 1.59x | 70.9% | 52.0 | 3.18 s |
| B-tree | 47 | 500 | 50.0 (49.8-50.6) | **79.9** (79.6-80.1) | 1.60x | 68.1% | 53.3 | 3.05 s |
| math | 89 | 500 | 47.1 (46.8-47.3) | **80.7** (80.6-81.9) | 1.71x | 84.0% | 53.3 | 3.12 s |
| leetcode | 128 | 246 | 45.0 (44.8-45.8) | **74.3** (73.7-74.3) | 1.65x | 86.8% | 37.4 | 3.20 s |
| story (high entropy) | 43 | 500 | 48.6 (48.4-49.5) | **64.1** (63.4-64.5) | 1.32x | 52.4% | 45.8 | 3.07 s |

* End-to-end is the engine's `Rates: overall=` (prompt plus generated tokens over prefill plus decode, model load excluded). Decode is `[REQUEST-DECODE-WALL]`, which includes the one-time decode-graph setup (plus about 0.23 s of
  draft-graph setup with MTP); no separate steady-state rate is logged. Prefill is 0.74 s for the 29-token prompt but about 3.1 s for 43-128 tokens (probably a one-time prefill-graph build; not investigated).

Engine gate line on every run (36 of 36): `policy=required memory_plan=unknown layered_s1_forwards=0 legacy_s1_expert_dispatches=0`. `tools/check_full_graph_moe_run.py` has only `--full-fit` and `--allow-memory-fallback`
modes, neither of which fits a hot/cold split with an Unknown plan, so it was not run. The llama.cpp ratio required by `AGENTS.md` was not measured for this model (no matched reference build).

What changed since the 2026-10-27 baseline below (53.5 wall / 57.7 steady on the palindrome prompt):
- `patches/ggml/0016` Q2_0 VNNI dot with a 6-op weight unpack (bit-identical, 2.2x per thread): the largest single gain, +4 to +11 tok/s.
- `patches/ggml/0015` fused hyper-connection CUDA kernels (bit-identical logits at S=1 and S=4): +1 to +3.
- The draft head keeps its own rows after a verify instead of re-running a step per accepted token: +1 to +5, acceptance moves by -1.7 to +5.9 points depending on the prompt.
- The greedy MTP token selector is vectorised: +3 on every prompt. The MTP profile uses a workload-specific six-thread CPU override: +1 to +2.
- `patches/ggml/0017` (IQ3_XXS dot) does not affect this model (Q2_0 experts); it matters for IQ3_XXS recipes (see `configs/bench/perf-baselines.md`, Trial 25 follow-up).

Tried and rejected (details in `docs/design/mtp-window-timeline.md`): draft count 2, 4 and 5 (3 is best on every prompt), CPU cold step as a CUDA host callback overlapping the GPU (-15%), fusing the routed-expert
reduction (-480 nodes, no speed change), more hot experts (13,500 fails at preload), faster IQ3_S dot (<=1.11x), bit-identical window kernels (-4% with a BF16 vector-kernel patch).

Open items and owners:
- **Promotion default (done after v0.0.1).** The key is now `expert_promotion` (the old `qwen4exp_expert_promotion` is rejected at startup with a rename message; it is not `graph_expert_promotion` because root keys starting with `graph_` are reserved for the typed graph-settings table and are rejected as unknown settings unless registered there) and defaults to `auto`: on when graph-native MTP is on, off otherwise; an explicit value wins,
  and the engine logs the resolved mode (`[EXPERT-PROMOTION] ...`). Promotion is worth +4% (leetcode) to +104% (story), +61% on math (MTP without it: 77.2 / 66.2 / 56.9 / 50.3 / 69.8 / 31.6 for palindrome / review / B-tree / math / leetcode / story, pass 1).
  The env override `ZEROMOE_QWEN4EXP_EXPERT_PROMOTION` is unchanged. `docs/design/configuration-knob-audit.json` still lists the old key until its owner regenerates it.
- **Text equality.** Plain and MTP greedy output are identical on the palindrome prompt but differ on story (character 55) and math (character 46); a per-column exactness experiment (branch `exact-window-wip`, row 0 byte-identical) did not remove it,
  so rows 1-3 of the window or another S-dependent path still differ. Speculation here is therefore not yet output-exact.
- `graph_settings_test` aborts on master itself (`--graph-offload-async: expected true or false`), unchanged by this merge; `graph_settings_binding` reports BAD_COMMAND under ctest. Owner: the settings session.
- GLM-5.3-Flash (`/data/sda/models/glm53flash/GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf`, Q2_K experts, no MTP head) is the next target; which of these optimizations carry over is assessed in the session notes, not yet measured.

Regression check: the same suites on the previous build (`1950bee`, before master's last cleanup commit) agree within +-2.5% per cell; the worst cell is -0.5% for MTP and -0.7% for plain decode (rule: report anything below -5%). Draft acceptance is identical on every prompt.

Artifacts (exact TOML, prompts, command, stdout, engine log, manifest, environment record): `build_mtp_results/2026-10-28-ea2b200-mtp-suite/` and `...-plain-suite/` (the numbers above); the same suites on the previous build are in `2026-10-28-1950bee-*-suite/`, and `2026-10-28-1950bee-mtp-suite-promotion-off/` is the invalid first attempt (promotion left at its default), kept for the record.
The suites were run from the worktree `ZeroMoE-wt-land` (binary `build/zeromoe`, sha256 in `environment.txt`); the raw per-run summaries are `summary.csv` in each directory.

## Master status (2026-10-27)

`master` is at `a8a1c7d`. Six changes landed after the `b78dd9b` binary that
produced the matched raw-story numbers below. The first five are not runtime
changes on the CUDA engine; the sixth changes the default graph/preload
surface:

- `5781401` removes the always-on `ZEROMOE_USE_GGML` and
  `ZEROMOE_BUILD_SERVER` build macros. The guards were true in every supported
  build, so the compiled decode path is unchanged; only the option surface and
  the dead fallback branches are gone.
- `d2317b2` adds the build-provenance stamp and the CUDA configure repair:
  the duplicate `graph_settings_test` target no longer breaks a non-CPU
  configure, and `graph_settings_test` passes against the defaults snapshot.
- `7f86269` stops the stamp from claiming the pinned ggml commit when a local
  `ZEROMOE_GGML_ROOT` is used.
- `0bd23c1` removes the `ZEROMOE_USE_GGML_CUDA` option. The macro is now
  derived from `ZEROMOE_CPU_ONLY`, so every CUDA build links ggml-cuda; the
  previous default `OFF` failed to link. Build commands no longer pass the
  flag. `982dedd` keeps `test_components` assertions live in Release builds.
- `19100ff` defaults `ZEROMOE_METAL` ON for Apple Silicon and limits MPS to
  prefill-sized F32 matmul (`S>=32`), so macOS decode stays on NEON.
  `GGML_METAL` is forced OFF; the CPU-only engine never selects a ggml
  backend. This only affects macOS.
- `a8a1c7d` removes the Qwen/GLM-DSA/Ling/DeepSeek-V4.1 full-graph env gates
  and the Qwen decode gate. Full graph is now the default; only the Qwen TOML
  keys, GLM-DSA MTP, Ling's native selector and the V4.1 decode comparison
  remain explicit fallbacks. Qwen whole-layer preload defaults to 2 (`0` is
  still the global-pool sentinel) and trailing placement defaults to false.
  See `docs/handoff/env-surface-full-graph-2026-10-27.md`.

`build/zeromoe` is still stamped from `b78dd9b` and fails
`python3 tools/build_provenance.py check --binary build/zeromoe --repo .`
against current master. Do not quote new rates from it; rebuild first. The
validation build in `build_verify_master/` is current (CPU_ONLY=OFF, arch 86,
no `ZEROMOE_USE_GGML_CUDA` input); its manifest reports `source=a8a1c7d
dirty=True` because `models/base/family_ops.cpp` remains uncommitted in its
owning session.

Outstanding items: the plain-mode rerun on current master (the `658a212` plain
run never finished because the driver stopped answering `nvidia-smi`), and the
Qwen family dispatch change in `models/base/family_ops.cpp`, which remains
uncommitted in its owning session.

## Exact MTP reproduction

Prompt: “Write a Python function that checks whether a string is a palindrome,
and explain it.” The model received 29 prompt tokens and stopped at EOS after
463 output tokens.

- Model: Qwen3.8-Flash-Next GSQ-RCO Q2_0, two shards.
- MTP draft: Q4_K_M shared checkpoint at
  `/path/to/models/qwen3.8-flash-next/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf`.
- Config: context 1536, max_new_tokens 800, temperature 0, top-p 0.95,
  global Option-B K=13,000, host_ram_gb 80 GiB, 10 OpenMP threads, frozen
  usage files, three draft tokens.
- GPU: RTX 3090 only (`CUDA_VISIBLE_DEVICES=0`). CPU: Xeon W-2235. Toolkit /
  driver: CUDA 13.3.73 / 610.57.04.
- The measured global-pool placement uses
  `--graph-device-expert-layers 0`. MTP phase timing is emitted at DEBUG
  verbosity (`--verbosity 5`). Full-graph prefill/decode are now unconditional; the old
  `ZEROMOE_QWEN4EXP_FULL_GRAPH*` switches are removed. Keep
  `ZEROMOE_BENCH_IGNORE_STOP` and `ZP_NOCHAIN` unset.

Persistent configs, prompt, stdout, engine logs, and build info:

- Worktree reproduction: `build_mtp_results/repro-ff9a894-2026-10-27/`.
- Master reproduction: `build_mtp_results/repro-master-658a212-2026-10-27/`.

| Source snapshot | Plain decode wall | MTP decode wall | MTP loop steady | Acceptance | Output parity |
|---|---:|---:|---:|---:|---|
| ff9a894 MTP worktree | 46.02 tok/s | 54.10 tok/s | 58.4 tok/s | 81.0% | byte-identical |
| master 658a212 | plain rerun pending | 53.50 tok/s | 57.9 tok/s | 81.0% | not compared on this run |

On master, MTP speculation took 8.003 s across 135 windows: 6.154 s target-window
run, 0.943 s draft steps (762 steps), 0.759 s driver/sampling, and 0.132 s for
embedding, preparation, copies, and commit. That is 3.43 emitted tokens per
window and 45.6 ms per target window. `[REQUEST-DECODE-WALL]` includes one-time
graph/draft setup; `[MTP-PROFILE] speculate` excludes it. Request-wide
throughput includes prefill and is not a decode rate.

The worktree plain run took 10.060 s decode wall (46.02 tok/s) and matched MTP
byte for byte. A master 27d5d33 plain run with the same prompt/config took
10.065 s (46.00 tok/s). A fresh plain run on 658a212 is outstanding because
the NVIDIA driver stopped answering `nvidia-smi` after the MTP process exited.

## User-reported comparison and Strata evidence

The report supplied in this conversation gives 57.7 tok/s steady and 53.5 tok/s
wall for MTP3 at 463 tokens, against 44 tok/s plain; a later plain rerun gave
46.65 tok/s. The 658a212 MTP result matches those figures within measurement
noise. It is one prompt and one run per setting.

Strata MTP has not been run locally. The supplied report/source analysis gives
Strata's Q2_0 4K-prompt cycle at about 34.3 ms (GPU 14.6 ms, CPU expert wait
13.4 ms, draft 2.0 ms, driver/sampling 4.3 ms). Hardware differs, so treat this
as a comparison target, not a same-device benchmark.

## Matched raw-story follow-up (2026-10-27)

Workload: raw prompt `Tell a story.` (4 tokens), context 512, 200 generated
tokens, 10 threads, Q2_0 target, shared Q4_K_M MTP sidecar, RTX 3090 target
configuration K=13,000 and three drafts. The runs use the verified `b78dd9b`
main-checkout binary and patched GGML upstream `8baabd53b`.

| Engine/mode | Decode wall | Rate | Draft acceptance | Output vs Zero plain |
| --- | ---: | ---: | ---: | --- |
| ZeroMoE plain | 5.482 s | 36.48 tok/s | — | reference |
| ZeroMoE MTP, batched window S=4 | 8.013 s | 24.96 tok/s | 121/232 (52.2%) | diverged near character 44 |
| ZeroMoE MTP, batched window S=2 | 6.093 s | 32.83 tok/s | 86/113 (76.1%) | diverged near character 44 |
| ZeroMoE MTP, stepwise verification | 6.474 s | 30.89 tok/s | 125/216 (57.9%) | byte-identical |
| Bee saved raw-prompt plain | 12.727 s | 15.64 tok/s | — | not compared |
| Bee saved raw-prompt MTP | 14.369 s | 13.85 tok/s | 120/156 (76.9%) | not compared |

The saved Bee records match prompt length, output length, context, and thread
count; their logs do not record an explicit GPU layer/CPU-MoE split, so the
throughput ratio is provisional as an offload comparison. The ZeroMoE batched
rates are not acceptable performance results because both S=2 and S=4 differ
from plain generation. Stepwise mode preserves output but remains below plain
decode because it verifies each token separately.

For S=4, target graph execution averaged 5.744 s / 78 windows = 73.6 ms per
window, versus the Strata report's 28.0 ms target window. ZeroMoE emitted
200/78 = 2.56 tokens per cycle, versus roughly 3.2–3.4 in the estimate. The
window cost and lower acceptance explain the gap from the 60–90 tok/s estimate;
the correctness failure must be resolved before optimizing this path.

`ZEROMOE_QWEN4EXP_MTP_WINDOW=0` stepwise control matched plain. This localizes
the current failure to the batched target window/prefix-commit path. A temporary
prefix-commit bypass was prepared but not run: `nvidia-smi` showed the separate
`ZeroMoE-wt-mtp2/build/zeromoe` process using 21 GiB on the 3090. The diagnostic
change was reverted and the main binary rebuilt with prefix commit enabled.
Do not start another 3090 run until that process releases the card. No files in
`ZeroMoE-wt-mtp2` were edited.

### GDN window operation-sequence update

The S>1 target window now uses one `ggml_gated_delta_net(..., K=S)` operation
with the per-token snapshots in GGML's latest-first order, matching Bee commit
`3823c9eb6` `src/models/delta-net-base.cpp::build_recurrent_attn`. This replaces
the earlier ZeroMoE window's S chained `K=1` GDN operations. The main binary
build succeeded and its provenance check passed (`b78dd9b`, GGML `8baabd53b`,
binary SHA `520de7a6…`). Runtime parity and speed for this change are not yet
measured. The post-change plain run was 34.19 tok/s; the MTP row rates above are
pre-change and must not be used to assess the GDN window update. `nvidia-smi`
stopped responding after that run, so I did not launch the updated MTP run.

## Teacher-forced draft prediction comparison

I re-derived top-1 results from the saved fixed-input trace dumps rather than
using the earlier summary script. On the 32-row fixture,
`zero_seed_trace_gpu_norms_r1.dump` and Bee's
`bee_seed_trace_s1_stages_r1.dump` agree on draft-logit top-1 for 32/32 rows.
Against the fixture's following token, both engines are correct on 15/31
transitions. Repeating with ZeroMoE's F32-attention trace gives the same counts.
An index-offset check gives 0/32 against the same-row token and 2/31 against
the previous-row token, so the following-token alignment is the best of these
shifts.

The production-shaped masked capture independently agrees for its three tail
rows: Bee getter hidden and internal `h_nextn` are bit-identical on rows 29–31;
the ZeroMoE/Bee draft argmaxes agree 3/3, and both predict the following fixture
token on 2/2 transitions. This supports the getter path for those rows, but is
too short to validate the full teacher-forced sequence.

There is still a capture-mode discrepancy. In the 32-row S=1 diagnostic bundle,
Bee getter hidden versus internal `h_nextn` differs by max abs 244.38, mean abs
1.44, and minimum row cosine 0.0589, while getter logits and internal
`result_output` are bit-identical across all 32 rows. The production-shaped
masked bundle does not reproduce that hidden mismatch. Bee commit `3823c9eb6`
sets `t_h_nextn` to `flat_out` in masked mode and the getter returns the saved
`t_h_nextn` rows, so keep the large-batch diagnostic discrepancy scoped to
that harness until its output-row mapping is explained.

The separate unmasked production trace is not a valid hidden comparison: its
harness queries getter index 0 for positions 29–31, but Bee documents unmasked
indices as raw token positions. Do not use that trace's hidden values as a
parity result.

Reproduce with:

```sh
python3 tools/compare_mtp_predictions.py \
  build_mtp_results/seed_fixture_chat_r2.txt \
  build_mtp_results/zero_seed_trace_gpu_norms_r1.dump \
  build_mtp_results/bee_seed_trace_s1_stages_r1.dump

python3 tools/compare_mtp_predictions.py \
  build_mtp_results/seed_fixture_chat_r2.txt \
  build_mtp_results/zero_seed_trace_gpu_norms_r1.dump \
  build_mtp_results/bee_seed_trace_prod_r3.dump
```

This establishes draft-decision parity for this fixture and these reference
runs. It does not estimate live speculative acceptance: the sample is short,
and the same-input fixture accuracy is only 15/31. Use a longer natural-text
teacher-forced trace to compare per-position draft agreement and target
acceptance before treating draft accuracy as a performance explanation.

## Q2_0 VNNI status

Q2_0 VNNI is already in the applied GGML patch series
(`patches/ggml/0001-ggml-cpu-q2_0-vnni.patch`); the GGML build has
`GGML_AVX512_VNNI=ON` and `GGML_NATIVE=ON`. Full-graph MTP cold experts use GGML
`MUL_MAT_ID`, whose Q2_0 dot path contains a VNNI dispatch. This MTP work did
not add that kernel. A same-workload VNNI-on/off and effective-bandwidth
measurement is still outstanding; build flags do not prove a particular call
used the VNNI branch or reached 41 GB/s.

## Build and cross-session provenance

- Current master at this update: b78dd9b (`Default server and ggml on; require ggml`).
- Persistent binary: `build/zeromoe`; manifest: `build/zeromoe.build-info.json`.
- Validate before a run with `python3 tools/build_provenance.py check --binary build/zeromoe --repo .`.
- GGML base: `8baabd53b77d1c48c6ef8d3b620436292f3f6231`; generated `GGML_COMMIT`
  reports this upstream hash. `scripts/build_ggml.sh` now passes and checks the
  resolved source commit.
- The MTP worktree `ZeroMoE-wt-mtp2` is detached at ff9a894 and has an
  uncommitted `ZP_NOCHAIN` guard. Its 11:48 binary was built with that guard
  present but the variable unset, so it uses the normal chained GDN path. Do not
  call that binary a master build.
- Other sessions have separate worktrees. Check `git worktree list --porcelain`
  and each worktree's status before using or editing them. Preserve their
  uncommitted files.

## Open work and next owner

1. When the 3090 is free, run a prefix-commit bypass control for K=1 on the
   `Tell a story.` fixture, then compare S=2/S=4 window logits and wide states
   against sequential target steps.
2. Extend teacher-forced draft prediction/target-acceptance comparison beyond
   the current 32-row trace and inspect the initial divergence position.
3. Verify Bee's target GPU/offload split in a matched run; the saved raw-story
   timing log does not record it.
4. Continue profiling graph build and draft/target execution only after the
   batched path is output-equivalent. Leave the `ZeroMoE-wt-mtp2` refactor
   untouched.

## Ornith 1.5 122B full-graph recovery (2026-10-29, locally measured)

Owner/worktree: current Codex recovery task, `recovery/ornith-122b-offload` at ZeroMoE source `7d48e01de3ea2804ea6375b37858c23c909ad1d2`; uncommitted and not merged. This work ports GGUF architecture `qwen35moe` through `Qwen35MoEGraphModel` and the shared full-graph runtime. It is not Qwen3.6-35B support.

Locally measured on RTX 5060 Ti, two CPU threads, 3-token raw prompt, 200 generated tokens, context 203, prefill chunk 8. Planner selected expert layers `[38,48)` for VRAM and retained all other experts in pinned DDR. Two runs passed `check_full_graph_moe_run.py --full-fit --require-full-graph-prefill`; each recorded one graph prefill, 199 decode graph replays, and zero layered or legacy expert dispatches. Per-run rates:

| Run | Prefill | Request decode | Graph-gate decode | End-to-end request |
|---|---:|---:|---:|---:|
| 1 | 4.43 tok/s | 5.87 tok/s | 6.27 tok/s | 5.76 tok/s |
| 2 | 4.72 tok/s | 5.98 tok/s | 6.39 tok/s | 5.86 tok/s |

Teacher-forced comparisons against llama.cpp `src/models/qwen35moe.cpp` at `8baabd53b77d1c48c6ef8d3b620436292f3f6231` are bit-identical over 48 layers for pre-MoE hidden states, router probabilities, normalized routed weights, and ordered top-8 expert IDs. Details and artifacts are in `build_mtp_results/2026-10-29-ornith-122b-smoke-5060ti/` and `build_mtp_results/2026-10-29-ornith-122b-perf-5060ti/`.

Performance reference status: an older three-repetition llama.cpp row measured 5.651 tok/s for the same GGUF, GPU, context, token counts, threads, and CPU/GPU expert placement (`-ngl 99 -ncmoe 38`). Both performance-checker ratios passed (1.110 and 1.130), but that binary has no source-worktree snapshot and the result remains provisional. A clean llama.cpp build from commit `8baabd53b77d1c48c6ef8d3b620436292f3f6231` was completed with matching `sm_120a` CUDA code target. Its benchmark could not load the GGUF because `/mnt/usb-ssd` is currently not mounted and the file is absent from the mounted model volumes checked. Do not treat the provisional ratio as final acceptance.

Next action: reconnect `/mnt/usb-ssd` and wait until the RTX 3090 is idle. The 5060 Ti/2-thread 10-layer run is exploratory, not the target profile. Rebuild the updated worktree, collect an Ornith-specific router-usage survey, then measure (1) a trailing-22-layer/leading-26-CPU split at 10 threads against llama.cpp `-ngl 99 -ncmoe 26` with the 16-token/400-token workload (user target 28 tok/s), and (2) usage-ranked expert hotset on the 3090 with a model-byte budget and 10 threads (user target 42 tok/s). Keep the second target pending until effective hotset bytes and routing coverage are logged; do not copy Qwen3.8's raw 14,554 expert count to Ornith's larger IQ3_XXS expert footprint. Use context/thread/GPU-layer/CPU-MoE arguments with the performance checker for each matched run. Worktree owner: current `recovery/ornith-122b-offload` task. Artifacts: `build_mtp_results/2026-10-29-ornith-122b-perf-5060ti/`; model note: `docs/models/ornith-1.5-122b.md`.

User-reported Qwen 3.8 Flash throughput figures are contextual only and were not used as llama.cpp reference measurements for this Ornith workload.

## Ornith usage survey and VRAM hotset (2026-10-30)

- Owner/worktree: current recovery task, branch `recovery/ornith-122b-offload`, project source `7d48e01de3ea2804ea6375b37858c23c909ad1d2` (dirty, unmerged). Binary SHA256 `6d428529ddcc17f80f8c1b87dccbec4fb04a95309d1e74e822008b207ada662e`; GGML base `8baabd53b77d1c48c6ef8d3b620436292f3f6231`.
- Fixed the empty usage survey: the Qwen35 graph builder now passes routed IDs into the shared tap. Five prompts completed with 1,030 decode callbacks, 395,520 route selections, and 11,615 distinct `(layer, expert)` pairs across all 48 layers. Artifact: `build_mtp_results/2026-10-29-ornith-122b-perf-5060ti/ornith-usage-survey.txt`.
- RTX 3090 usage-ranked placement selected 4,831 experts across all layers, spent 14.998 GiB, and covered 73.38% of surveyed route volume. A 24-token / 200-token generation measured 22.64 tok/s from first to last output, 27.59 graph steady replay tok/s, and 6.94 prefill tok/s. The full-fit graph gate passed with zero layered forwards and zero legacy expert dispatches. This is a placement check, not a matched llama.cpp comparison.
- Early-ready all-split is opt-in as documented. The hotset check used `ZEROMOE_QWEN_EARLY_READY_ALL_SPLIT=0`. The planner now labels this `usage-hotset-only` rather than reporting all-DDR placement.
- Artifacts: `build_mtp_results/2026-10-29-ornith-122b-perf-5060ti/ornith-usage-survey-retry7-20261030.*` and `ornith-hotset15-retry4-20261030.*`; model note: `docs/models/ornith-1.5-122b.md`.
- Next: build or repair a llama.cpp CUDA benchmark binary that enumerates this host's GPUs, then compare the same Ornith GGUF, 3090, context, prompt/generation lengths, and thread count. Keep onboarding in progress until the 90%-of-llama.cpp performance gate passes.

## Ornith official survey and matched benchmark update (2026-10-30)

- The standard `tests/moe_usage_survey/run_survey.py` runner had an `args.ngen`/`args.predict` typo; fixed it and used five repository prompt fixtures through the persistent harness. All five are `OK`. The official histogram contains 11,524 `(layer, expert)` pairs across all 48 layers and 405,504 route selections (1,056 callbacks; 8,448 per layer). File: `build_mtp_results/2026-10-30-ornith-122b-3090-match-24p200g10t/official-survey-5/ornith-usage-survey.txt`.
- The official histogram's 15-GiB usage hotset selected 4,853 experts across 48 layers, spent 15.000 GiB, and covered 73.35% of that survey's traffic. Full-fit graph gate passed. Its single-prompt run overlapped high shared-host CPU load, so its 14.64 request / 16.58 steady tok/s rates are exploratory, not a performance acceptance measurement.
- Matched workload already exercised on the earlier five-generic-prompt ranking: 24 prompt + 200 generation tokens, context 224, 10 threads, RTX 3090, `n_cpu_moe=31`. ZeroMoE's three-run graph steady rate was 21.97 tok/s. The prior llama-bench row averaged 13.67 tok/s and the checker passed 1.608x, but that binary embeds llama.cpp commit `3cf03257f`, not the current clean checkout commit `8baabd53b77d1c48c6ef8d3b620436292f3f6231`; keep the ratio provisional.
- A fresh `llama-bench` build is running from clean llama.cpp commit `8baabd53b77d1c48c6ef8d3b620436292f3f6231`, CUDA target sm_86, on the same upstream GGML base. Build command/log and CMake/source provenance are in `build_mtp_results/2026-10-30-ornith-122b-3090-match-24p200g10t/`.
- Next: finish the fresh reference build; run the exact `-pg 24,200 -t 10 -ngl 99 -ncmoe 31` reference row; rerun the ZeroMoE three-request workload against the official histogram when shared-host load is low; run `check_full_graph_moe_perf.py` with that fresh reference. Keep onboarding active until the matched 90% gate passes and the user-context 28/42 tok/s expectations are addressed.

## Ornith latest acceptance and VRAM audit (2026-10-30)

This section supersedes the earlier pending-reference notes above. The clean
llama.cpp reference and official-histogram matched ZeroMoE runs now exist.

- Owner/worktree: `recovery/ornith-122b-offload`, source commit
  `7d48e01de3ea2804ea6375b37858c23c909ad1d2`, dirty and unmerged. Latest
  checked ZeroMoE binary SHA256 is
  `f769656f8109d12ecb4a3da89f6729ba157e15c4cd7d0d88955d7655c7c1a60b`;
  `build_provenance.py check` passes. GGML base is
  `8baabd53b77d1c48c6ef8d3b620436292f3f6231`; applied patch manifest SHA256
  is `bc15a89242db58386ab4335048ea55bb15233f18c40fe23881802097fe1f4f71`.
  ZeroMoE was built for sm_86 and sm_120a; the matched device is the sm_86
  RTX 3090. Per-run build manifests are in `official-survey-5/`.
- Fresh llama.cpp reference: clean commit
  `8baabd53b77d1c48c6ef8d3b620436292f3f6231`, GGML 0.27.0, CUDA sm_86,
  same GGML base, CUDA 13.3.73 and GNU toolchain. For the Ornith GGUF,
  24 prompt + 200 generation tokens, context 224, 10 threads, CUDA0,
  `-ngl 99 -ncmoe 31`, three-run steady decode averaged 15.726 tok/s
  (samples 15.992, 16.385, 14.803). Reference build and raw JSON:
  `llama-build-sm86-upstream8ba/`, `llama-bench-pg-upstream8ba.json`.
- Official five-prompt usage survey: 1,056 decode callbacks, 405,504 route
  selections, 11,524 distinct `(layer, expert)` pairs across 48 layers;
  histogram at `official-survey-5/ornith-usage-survey.txt`.
- Usage-ranked full-graph fit sweep:

  | Hotset budget | Experts | Survey coverage | Planner required / available | Steady decode | Ratio vs llama.cpp |
  | ---: | ---: | ---: | ---: | ---: | ---: |
  | 15.0 GiB | 4,853 | 73.35% | Fits | 20.63 tok/s | 1.312x |
  | 16.25 GiB | 5,275 | 76.52% | 22.96 / 23.28 GiB | 21.33 tok/s | 1.357x |
  | 16.5 GiB | 5,375 | 77.23% | 23.21 / 23.28 GiB | 20.99 tok/s | 1.335x |

  Full-graph and performance checker outputs for 16.25 and 16.5 are saved
  under `official-survey-5/`; all four pass. Each per-run build manifest is
  present. The 16.25 and 16.5 binaries have different working-tree snapshot
  hashes, although source commit, upstream GGML base, patch manifest,
  toolchain, and GPU target agree; retain each run's own manifest when
  comparing the budget profiles.
- User absolute hotset target check: `check_full_graph_moe_perf.py --minimum-tps 42`
  fails both current matched runs: 21.33 tok/s at 16.25 GiB and 20.99 tok/s
  at 16.5 GiB. The default 90%-of-llama.cpp gate passes, but that does not
  satisfy the user's Qwen3.8 hotset target. Logs: `official-survey-5/`
  `official-hotset16_25-minimum42-gate.txt` and
  `official-hotset16_5-minimum42-gate.txt`.
- The separate 28 tok/s layer-split target remains unmet too: the corrected
  19-GPU-layer / 29-CPU-layer profile measured 18.26–18.62 tok/s. The
  16-prompt / 400-generation split workload has no matched same-GGUF llama.cpp
  row yet.
- Memory accounting correction: the dense graph arena is 4.35 GiB
  (`4,669,931,520` bytes), not 6.7 GiB. The planner's roughly 6.7 GiB
  fixed/non-hotset term also contains F32 KV/GDN state, its 2 GiB workspace
  reserve, and 1% alignment. At the 16.5 GiB split the graph has about
  1.99 GiB free and estimates about 1.70 GiB for the following prefill, so
  only about 0.29 GiB remains beyond that shape's estimate. The 16.5 profile
  is the tested aggressive ceiling; 16.25 leaves more margin. Their source
  snapshot hashes differ, so the small decode-rate difference is not a strict
  same-binary A/B.
- Correctness evidence: llama.cpp and ZeroMoE teacher-forced hidden states,
  router probabilities and normalized weights match bit-for-bit over 190
  rows; ordered route IDs match 96/96 rows. GDN prefill-to-first-decode state
  handoff matches bit-for-bit for 36/36 recurrent layers. The mapping and
  caveats are in `docs/models/ornith-1.5-122b.md`.
- Remaining validation scope: `ZEROMOE_QWEN_EARLY_READY_ALL_SPLIT` was off
  for accepted runs and remains an opt-in path; post-first-decode-step GDN
  state is not included in the saved llama.cpp parity comparison. The model
  work remains unmerged. User-reported Qwen3.8 rates are target context for
  this port, while the same-GGUF llama.cpp ratio is a separate acceptance
  gate.

## Qwen3.8 speed-gap diagnosis for Ornith (2026-10-30)

- The full Ornith GGUF is smaller (49.5 GB) than the documented Qwen3.8 Q2_0
  shards (~99 GB), but this does not mean its routed workload is smaller.
  Available Qwen3.8 sibling metadata (IQ3_XXS; same architecture dimensions)
  reports hidden 2,560, 512 experts/layer, top-10, width 640. Ornith is
  hidden 3,072, 256 experts/layer, top-8, width 1,024.
- Gate/up/down shape counts estimate 3.62B active routed matrix weights per
  Ornith token versus 2.36B for Qwen3.8, a 1.54x work ratio. Qwen's measured
  28.66 tok/s split rate divided by 1.54 predicts 18.61 tok/s; corrected
  Ornith split runs measured 18.26–18.62 tok/s. Active expert work explains
  most of the layer-split gap, subject to the caveat that Qwen's Q2_0 GGUF and
  original raw log are absent from this checkout.
- Qwen's documented hotset held 14,554 expert records / 18.74 GiB and reached
  about 42–43 tok/s. Ornith holds 5,375 / 16.498 GiB and its five-prompt
  survey covers 77.23% of observed routes. The Qwen model note reports about
  93% hotset route coverage. The survey corpora differ, so coverage is a
  directional comparison. Larger Ornith expert records and higher active
  work per token both contribute to the remaining hotset speed gap.
- Ornith's 16.5 GiB hotset leaves only about 0.29 GiB beyond its measured
  next-prefill estimate under the current memory plan. More expert VRAM
  residency requires reducing other graph memory or changing the prefill
  allocation lifecycle; raising the budget alone is not a valid fit.
- The split-run timing audit shows graph setup (~82 ms) is negligible beside
  repeated decode time (~21.5 s per 400 tokens). The current INFO-mode hotset
  logs do not split cold-expert CPU time by layer/op. Once CUDA works again,
  the next useful diagnostic is a DEBUG timing run on the 16.5 GiB profile,
  followed by an uninstrumented measurement if a code change improves it.
  Detailed saved analysis: `build_mtp_results/2026-10-29-ornith-122b-perf-5060ti/ornith-qwen38-timing-and-profile-audit.md`.
- Current environment check: `/mnt/usb-ssd` is mounted and the Ornith GGUF is
  readable, but `nvidia-smi` failed to communicate with the NVIDIA driver.
  No new GPU run was started; retry the DEBUG timing run only after CUDA is
  available.

## Ornith cold expert CPU fusion tuning (2026-10-30)

- Found a quant-specific path gap: `qwen4exp_cold::supported()` accepted only
  Q2_0 weights / Q8_0 activations. Qwen3.8 Q2_0 could use the shared fused
  cold-expert CPU node; Ornith IQ3_XXS/IQ2_S/IQ4_XS could not and instead used
  separate gate, up, SwiGLU, and down `MUL_MAT_ID` nodes.
- Generalized `models/qwen/cold_ffn_fused.cpp` to select per-tensor CPU dot
  traits and a shared activation quant type. The Ornith IQ combinations use
  Q8_K. The op now dispatches according to the kernel row contract: IQ kernels
  use `nrc=1`; Q2_0 retains the patched four-row VNNI path when supported.
- `cold_ffn_fused_test` passed 180 bit-exact cases against the stock GGML
  chain: 60 Q2_0, 60 IQ3_XXS, and 60 mixed IQ2_S/IQ3_XXS/IQ4_XS cases. Each
  set covers 5 thread counts, S=1/2/4, and four route masks. An initial IQ3
  attempt using four-row dot calls failed; the code now follows GGML's
  one-row IQ3 contract and all expanded cases pass.
- Reconfigured the existing build tree to refresh its dirty-source snapshot,
  rebuilt `zeromoe`, and passed `build_provenance.py check`. Latest binary:
  SHA256 `65947859c774df82695c172199b7b981db009ddb47f1a8a8651cd8ecbbcf8a67`;
  source snapshot SHA256
  `c5672e44ab40695b672bcaf14c89a90f1ca6d87a275476df292bbb157d235888`;
  GGML base and patch manifest unchanged.
- No GPU decode result for this change: the NVIDIA driver still does not
  answer `nvidia-smi`. CPU equivalence is proven; speedup is not. The complete
  requested handoff is
  [`ornith-122b-cold-expert-fusion-2026-10-30.md`](ornith-122b-cold-expert-fusion-2026-10-30.md).
- Next: once CUDA is available and the 3090 is idle, use the prepared
  `official-hotset16_5-cold-fused-debug.toml` profile in `official-survey-5/`
  to confirm fused cold nodes are active and collect per-layer CPU timings,
  then run an uninstrumented matched 3-request decode and rerun both
  full-graph and llama.cpp performance gates. At the latest recheck,
  `nvidia-smi` still failed and host load was 16.10 on 12 allowed CPUs; no
  diagnostic or decode run was started.

## Expanded Ornith fused-kernel coverage (2026-10-30)

- The saved 48-layer quant inventory lists 13 gate/up/down weight triplets.
  `tests/test_cold_ffn_fused.cpp` now includes all 13 model combinations plus
  a Q2_0 control and one mixed-format case. The original three test cases
  have 180 bit-exact passes; the 12 remaining model-specific combinations
  compile but remain unrun.
- A static set comparison of the saved `[QUANT]` inventory with the test
  source found all 13 model triplets covered; the two extra test cases are
  the Q2_0 control and the alternate mixed-format combination.
- Those 72 cases use two CPU thread counts, S=1, and sparse/all/mixed cold
  masks. Two more short checks use Ornith's actual D=3072/FF=1024 dimensions.
  The latest host-load recheck was 15.76 on 12 allowed CPUs, so the remaining
  CPU test was deferred to avoid competing with other sessions.
- Current full test target rebuild command:
  `cmake --build build_mtp_results/2026-10-29-ornith-122b-smoke-5060ti/build --target cold_ffn_fused_test -j1`.

## Latest Ornith cold-fusion test update (2026-10-30)

The expanded test fixture now quantizes IQ2_XXS/IQ1_S through
`ggml_quantize_chunk` with a unit importance matrix. A low-priority smoke run
(`cold_ffn_fused_test --smoke-ornith-types`) passed 14 cases bit-identically:
all 12 model-specific quant triplets beyond the already-tested IQ3_XXS case,
plus two Ornith actual-shape cases at `D=3072`, `FF=1024`, E=8, top-8. The
updated output is saved in
`build_mtp_results/2026-10-30-ornith-122b-3090-match-24p200g10t/official-survey-5/cold-fused-type-smoke-rerun.log`.
The broader multithread/token/mask matrix still has 70 cases pending, and the
full-model GPU timing remains outstanding because `nvidia-smi` still cannot
reach the driver. The CPU test target compiles. The recovery worktree remains
active and unmerged.

## Ornith full CPU matrix rerun status (2026-10-30)

The low-priority full `cold_ffn_fused_test` rerun completed its Q2_0 control
(60/60 bit-exact), then was stopped with SIGINT when host load rose to 14.63
on 12 allowed CPUs (exit 130). This is an interrupted shared-host run, not a
kernel failure. The prior one-thread all-cold smoke passed all 14 pending
quant/shape cases. The 70 remaining thread/token/mask cases await a quieter
host; raw command, log, and exit status are under `official-survey-5/`.

Latest probe: `nvidia-smi` still fails, and the clean matching
`llama-bench --list-devices` reports no CUDA-capable device. Host load at the
probe was 14.45 on 12 allowed CPUs. No GPU diagnostic was launched.

## CUDA state diagnosis for Ornith follow-up (2026-10-30)

Read-only checks: NVIDIA kernel module 610.57.04 is loaded and `/proc/driver`
lists both GPUs with their device nodes present; `CUDA_VISIBLE_DEVICES` is
unset. But `nvidia-smi` fails and the clean matching `llama-bench
--list-devices` sees no CUDA device. No XID appeared in the searched boot
journal lines. No shared-driver reset or device-node change was attempted.
This is an external host CUDA state requirement for the pending DEBUG/GPU
performance run, not a model startup fallback.
