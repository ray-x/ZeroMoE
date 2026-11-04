# Qwen4Exp MTP shared handoff

Updated 2026-11-04. Promo and MTP prefill-overlap controls are now typed Qwen graph settings (`graph_promo_*`, `graph_mtp_prefill_overlap`); `ZEROMOE_MTP_GREEDY_FAST_PATH`, `ZEROMOE_QWEN4EXP_EXPERT_PROMOTION`, and `ZEROMOE_QWEN4EXP_ROUTE_IDS_DUMP` were removed. Refresh the current HEAD and worktree list before using any
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

## DeepSeek V4.1 Flash Engram status (2026-11-03)
Main checkout remains branch `master`, HEAD `e2167121bef0748954a435e5a8d3fad8ccbb4969`, with uncommitted SSD Engram support and the same-layer dummy-route fix. Use `/mnt/usb-ssd/@/home/ray/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO/` for the five main shards and separate Engram sidecar. The current clean binary passes provenance with GGML base `8baabd53b77d1c48c6ef8d3b620436292f3f6231`, CUDA 13.3, native ZeroMoE targets sm86/sm120.

**NaN root cause:** cached graph MoE gave a GPU branch dummy IDs from another layer for zero-weight routes. The current layer's down-projection view interpreted those bytes and generated NaNs; multiplying by zero preserved them. Same-input A/B showed identical layer-7 input/routes, cached output 332,037/496,640 NaNs, host callback output finite. Corrected pool-order analysis showed selected routes were finite; only zero-weight dummy routes were non-finite. Selected cache slabs matched DDR bytes. Fix: dummy IDs come only from the current layer; reuse a same-layer slot if fewer than top-k distinct slots exist.

A clean-binary guarded smoke with this fix completed the 40-layer full-graph prefill and generated eight tokens without NaN/Inf reports. It used the SSD Engram sidecar, 97-token prompt, 85 GB host budget, 15 GiB pinned limit, and 26 GiB emergency floor. Minimum MemAvailable was 71.34 GiB; no emergency stop. Artifact: `build_mtp_results/2026-11-03-e216712-e216712-deepseek41-engram-same-layer-clean-smoke/`. Process cleanup still exits -6 in CUDA `cudaFree`. Rates are 0.74 tok/s prefill, 0.71 tok/s decode, 0.63 tok/s decode wall, below 100/10 targets. Full-fit checker fails because memory plan is `Unknown`; no matched llama.cpp reference or teacher-forced parity yet.

Historical comparison: detached `b8b47a4` before typed DeepSeek cache-MoE settings reproduced the same NaN class using the original environment flags. Sidecar format support was backported for the test; 5060 Ti MoE cache reduced from 10 to 8.5 GB to preserve the earlier revision's 2 GiB VRAM floor. Artifacts: `build_mtp_results/2026-11-03-b8b47a4-deepseek41-engram-legacy-env-layer7-cache85/`. This confirms the settings migration did not introduce the bug.

Next: teacher-forced reference parity, a cache-backed memory plan, CUDA teardown repair, then optimize and measure performance on a matched build/workload. Source changes remain uncommitted.


### Why the pre-fix rate looked higher (2026-11-03)
The ~4–5 tok/s decode / 11.5–27.9 tok/s prefill rows came from runs with NaN routing after layer 7. Their late-layer router collapsed to only six distinct experts per layer; the run loaded 1,080 DDR expert blocks (8.61 GB) and uploaded 8.87 GB to VRAM, with seven `SLOW-PREAD` events. The corrected output selected a normal, diverse expert set: 4,271 unique blocks (36.87 GB) and 42.00 GB H2D uploads. Cold-clean and immediate-repeat fixed runs both stayed around 0.74 tok/s prefill / 0.71–0.76 tok/s decode. Each logged 161 slow SSD reads around 200–250 ms and about 0.03 GB/s real read bandwidth. Therefore the old rate was artificially high due to corrupted routing; the corrected workload now exposes SSD expert reads as the bottleneck. Async H2D overlap was prepared but did not start because the guard saw a process in `ZeroMoE-wt-prefill-tune` and only 64.93 GiB MemAvailable. Artifacts: `build_mtp_results/2026-11-03-e216712-deepseek41-engram-same-layer-clean-smoke/`, `...-same-layer-warm-repeat/`, and `...-same-layer-async-upload/`. Next speed work should reduce cold expert-read cost while preserving the finite route path; do not use the pre-fix rates as a valid comparison.


### Why the corrected rate is lower (2026-11-03)
The pre-fix fast run was not a valid baseline: after its layer-7 NaNs, late-layer routing collapsed to six unique experts per layer. It loaded 1,080 unique expert blocks, read 8.61 GB, and uploaded 8.87 GB. The corrected run produced finite routing, loaded 4,271 expert blocks for the prompt (36.87 GB), and uploaded 42 GB; the same prompt/config on the previous run path showed 161 `SLOW-PREAD` warnings (>200 ms each) and only 0.03 GB/s accumulated read throughput. An immediate repeat stayed at 0.74 tok/s and still had 142 slow reads, so the slow rate persists across repeats. Earlier high rates reflect corrupted routing plus a much smaller expert set, not a regression in correct math. Async-upload A/B was refused by the guard because `/home/ray/Projects/ZeroMoE-wt-prefill-tune` had a GPU0 process and MemAvailable was 64.93 GiB (below the 85 GB host-cap + 26 GiB start floor); that process was left alone.

Capacity explains why SSD reads remain: `gguf_tool plan` reports 15,360 routed experts and 157.13 GB for full MoE residency without Engram tables. The smoke config allows 85 GB host RAM, caps the RAM expert cache at 6,000 slots, and configures 21 GB of VRAM MoE cache; 85+21=106 GB is already less than 157.13 GB before dense/state overhead, and `ram_preload_all_experts` is not enabled. Runtime is on-demand: the clean run had 0 RAM-cache hits and 4,271 main-thread expert claims, with 4,271 DDR slots populated after prefill (36.87 GB). The 161 slow-read count is only calls over the 200 ms log threshold, not the total number of SSD reads. Full plan output and parsed summary are in the clean-smoke artifact (`model-capacity-plan.txt`, `cache-capacity-and-read-summary.json`).

### SSD reads explain the apparent rate drop (2026-11-03)
The 4–5 tok/s decode / 11–28 tok/s prefill figures came from the NaN-corrupted route path. Its late layers collapsed to six unique experts per layer, loading 1,080 expert blocks total (8.61 GB) and uploading 8.87 GB. The corrected clean run loaded 4,271 unique expert blocks for the same prompt (36.87 GB) and uploaded 42.0 GB. Its 0.74 tok/s prefill / 0.71 tok/s decode repeated at 0.74/0.76 on an immediate rerun; each reported ~0.03 GB/s real SSD read throughput. A clean `gguf_tool plan` confirms all MoE experts require 157.13 GB (15,360 slots) without Engram; config is capped at 85 GB host, 6,000 DDR slots, and 21 GB MoE VRAM, with `ram_preload_all_experts` unset. Runtime reports 0 RAM hits, 4,271 main-thread claims, 36.87 GB read. `SLOW-PREAD` means one individual projection read exceeded the 200 ms threshold (161 warnings in clean run), not the total read count. The model does not fully preload into DDR+VRAM; inactive experts stay on SSD and are fetched on demand.

The `graph_cache_moe_async_upload=true` A/B completed correctly but did not improve prefill: 0.74 tok/s prefill, 0.78 tok/s decode, 42.11 GB H2D, 36.97 GB SSD reads. The prior async attempt was correctly refused while `ZeroMoE-wt-prefill-tune` used GPU0 and MemAvailable was 64.93 GiB; the retry started later when idle. It still exits -6 during CUDA cleanup and the full-fit checker remains blocked on `memory_plan=Unknown`. Async comparison artifact: `build_mtp_results/2026-11-03-e216712-deepseek41-engram-same-layer-async-upload-retry3/`.

### 110 GiB DDR request vs current host capacity (2026-11-03)
Current host `/proc/meminfo`: MemTotal 125.48 GiB, MemAvailable 119.26 GiB. Preserving the established 26 GiB emergency floor leaves only 93.26 GiB currently available for all new host allocations before ZeroMoE resident/scratch overhead. A 110 GiB MoE payload plus the 26 GiB reserve requires at least 136 GiB physical RAM, and more once runtime overhead is included. The model plan reports 157.13 GB decimal for full MoE residency alone. I did not raise `host_ram_gb`, preallocate memory, or lower the guard; 110 GiB cannot safely fit on this host. The async upload A/B also completed with no useful prefill gain (0.74 tok/s; 36.97 GB SSD reads, 42.11 GB H2D), confirming that SSD expert reads still dominate. Its artifact is `build_mtp_results/2026-11-03-e216712-deepseek41-engram-same-layer-async-upload-retry3/`.

### DeepSeek host reserve update (2026-11-03)
Per user request, the DeepSeek starter and dual-GPU benchmark profile now set `mem_guard_stop_mb=10240` (10 GiB); historical run artifacts keep their original guards. Current `/proc/meminfo` showed MemTotal 125.48 GiB and MemAvailable 119.26 GiB. A 110 GiB DDR payload plus a 10 GiB reserve would require 120 GiB currently available, already 0.74 GiB above the observed value before ZeroMoE resident/scratch overhead. The configured 110 GiB pool was therefore not allocated. The profile host budget remains 107 GB decimal; the 85 GB smoke configurations remain historical. Continue enforcing live physical-memory admission.


### DeepSeek preload usage-profile provenance (2026-11-03)
The `route_usage.txt` and `prefill_usage.txt` copied into the DDR retry profiles came from `build_mtp_results/2026-11-03-e216712-deepseek41-engram-usage-train-10gib-floor-retry1/`. Their SHA256 values are `6e4d9c47f37ea47230d89baf272225020f02d61114b8ac03b8992dc5851c4412` and `0e7bcbff3a9976f7c1d5d4f75bd8850a0d62742251a6e22d2dbf1be2b4598158`; all retry copies are byte-identical. The source config used the same six-shard DeepSeek-V4.1-Flash-1017-GSQ-RCO model and separate Engram sidecar.
The source run recorded one valid 97-token prompt and eight generated tokens in `bench.json` (`valid=true`, `status=ok`), with the 40-layer graph and no NaN/Inf report in its engine log. The route histogram has 4,547 `(layer,expert)` records over 40 layers and 24,960 selections; the prefill histogram has 4,283 records and 23,280 selections, consistent with one 97-token prefill plus seven routed decode positions. The source process later exited `-6` during CUDA teardown, after writing the histograms; the request itself was valid.
This is a valid model-compatible one-request routing sample for seeding a preload ranking, but it is not a representative usage survey. It has one short prompt and eight generated tokens. Use multi-prompt/multi-turn observations before treating its hot-set coverage as representative.

## DeepSeek V4.1 usage-ranked preload smoke (2026-11-03)

**Locally measured on the dirty `master` checkout** (`e2167121bef0748954a435e5a8d3fad8ccbb4969`), binary SHA `ebadd4a5a314aa330077e4083359e9ea2f483718f279854f67ef23c5b66a5c54`, GGML base `8baabd53b77d1c48c6ef8d3b620436292f3f6231`, CUDA 13.3, driver 610.57.04, native targets sm86/sm120. Persistent artifacts: `build_mtp_results/2026-11-03-e216712-deepseek41-engram-usage-preload-test-retry4/`. This was a guarded smoke (97 prompt tokens, 8 generated, context 4096, temperature 0), not the final performance workload.

The full graph built and completed all 40 prefill layers and seven graph decode runs; the run gate reports zero one-token layered forwards and zero legacy per-expert dispatches. However, the memory plan remains `Unknown`, so `tools/check_full_graph_moe_run.py --full-fit` fails. `bench.json` reports prefill 5.73 tok/s, decode 1.06 tok/s, and decode wall 0.89 tok/s; the engine’s steady graph decode counter reports 0.78 tok/s. End-to-end active request rate was 0.30 tok/s. These are far below the user's 100 tok/s prefill / 10 tok/s decode targets. The 8-token run is too short to accept as a performance result.

The usage-ranked preload filled 6,000 DDR slots (51.59 GB) and 1,705 VRAM slots (20.98 GB), 7,705 slots total out of 15,360 (50.2% theoretical disjoint coverage). Prefill recorded 2,285 RAM hits, 572 main-thread claims, and zero prefetch-worker wins. Decode still spent 744.77 ms/token in SSD reads (`ram_batch` 746.72 ms/token; async upload 0.82 ms/token). This confirms that this 6,000-slot preload did not remove cold expert reads. The next speed work should focus on route-hotset coverage and making the graph’s DDR misses overlap with GPU work; the log warns the DeepSeek adapter lacks a common early-ready CPU lane.

The run completed its request with no NaN/Inf log report and no emergency stop (minimum MemAvailable 59.62 GiB), then exited `-6` during CUDA `cudaFree` teardown (`driver shutting down`). This smoke used an 85 GB host cap, 15 GiB pin budget, 20 GiB startup margin, and the requested 10 GiB runtime emergency floor. It does not verify the requested 110 GiB DDR configuration, which remains unsafe on this host. The GLM source-reported comparison and material configuration differences are in the artifact's `reference-settings.md`; this smoke is not a matched llama.cpp performance comparison.


## DeepSeek DDR-capacity A/B: 6,000 vs 8,000 slots (2026-11-03)

Locally measured, same 97-token prompt (same SHA), 8 generated tokens, context 4096, temperature 0, same binary (`ebadd4a5...`), same DDR usage files, host cap 85 GB, pin cap 15 GiB, emergency floor 10 GiB, and 11/10 GB VRAM caches. The only runtime setting changed was `ram_cache_total_experts` from 6,000 to 8,000. Artifacts: `build_mtp_results/2026-11-03-e216712-deepseek41-engram-usage-preload-test-retry4/` and `...-retry5-8k-ddr/`.

| DDR slots | Prefill | Bench decode | Graph steady decode | Decode wall | SSD disk read/decode token | min MemAvailable |
|---:|---:|---:|---:|---:|---:|---:|
| 6,000 | 5.73 tok/s | 1.06 tok/s | 0.78 tok/s | 0.89 tok/s | 744.77 ms | 59.62 GiB |
| 8,000 | 6.36 tok/s | 1.07 tok/s | 0.74 tok/s | 0.84 tok/s | 727.67 ms | 43.73 GiB |

The 8,000-slot run preloaded 68.79 GB (64.06 GiB) DDR expert slabs plus 20.98 GB (19.54 GiB) in VRAM, for about 83.6 GiB of physical cache pools. It reduced main-thread prefill claims from 572 to 455, but prefetch-worker wins remained zero and decode still spent about 728 ms/token reading from SSD. Increasing DDR from 6,000 to 8,000 therefore did not solve decode. The unchanged 11/10 GB VRAM allocation is below the user's requested capacity; the next capacity profile will increase both DDR and VRAM while retaining startup admission and live 10 GiB emergency guards.

Both runs used full-graph prefill/decode (40 prefill layers, seven decode graph runs, zero layered one-token forwards or legacy expert dispatches), but both fail `check_full_graph_moe_run.py --full-fit` because the graph memory plan is `Unknown`. Both requests were numerically coherent with no NaN/Inf log, but teacher-forced parity remains open. Both processes exited -6 at CUDA cleanup (`driver shutting down`); neither hit the emergency memory stop. The 8-token measurements are smoke diagnostics, not the 10 tok/s decode / 100 tok/s prefill acceptance workload.


## DeepSeek 105 GiB DDR capacity attempt: VRAM headroom failure (2026-11-03)

Retry7 used the rebuilt binary with a DeepSeek-only 8 GiB startup margin, 8 GiB emergency floor, a 120 GB decimal process cap, 13,100 DDR expert-slot target, and 18.25/14.50 GB requested VRAM MoE caches. The byte-aware RAM planner capped the DDR pool at 12,560 slots / 107.998 GB (100.58 GiB) after resident and workspace/state reserves. VRAM pools allocated 32.1204 GB (29.91 GiB): 17.6249 GB on the 3090 and 14.4955 GB on the 5060 Ti. It reported maximum disjoint expert-identity coverage of 15,169/15,360.

Graph prefill failed before token generation: GPU0 had 0.97 GiB free, below the explicit 1.00 GiB graph allocation floor. MemAvailable minimum was 12.53 GiB, so the configured 8 GiB host stop did not fire. The process exits -6 during CUDA cleanup. This was not a performance result. The full-fit planner still reports `Unknown`; the run gate cannot accept `--full-fit`.

The host cannot provide a true 105 GiB DDR expert pool with the current resident/state/slack estimates: that requires about 124.65 GB process budget, plus the 8 GiB startup reserve, exceeding the observed available physical memory. Current effective safe DDR expert capacity is about 100.6 GiB. The next retry lowers the 3090's cache request by 0.5 GB to retain more graph headroom while preserving the large DDR pool and high 5060 cache.


## DeepSeek high-capacity preload retry8: swap-pressure abort (2026-11-03)

Artifact: `build_mtp_results/2026-11-03-e216712-deepseek41-engram-105gib-ddr-vram-headroom-retry8/`. Source `e2167121bef0748954a435e5a8d3fad8ccbb4969` (dirty), binary SHA `9e09ac1e284f287f759cb764224b66ab9d07133190c808886f01e54add703edb`, GGML base `8baabd53b77d1c48c6ef8d3b620436292f3f6231`. Both GPUs were idle at admission; initial MemAvailable was 120.42 GiB. Config requested 13,100 DDR slots (~105 GiB), startup/live reserve 8 GiB, and 17.75/14.4955 GB MoE VRAM pools.

The planner still capped DDR at 12,560 slots / 100.581 GiB (host budget 120 GB, resident 6.15967 GB, workspace/state slack 5.74826 GB). It allocated 17.6249 GB MoE on the 3090 and 14.4955 GB on the 5060 Ti; maximum disjoint coverage was 15,169/15,360. The model has `general.alignment=32`, so the run used buffered io_uring (`direct_io=false`, `drop_pages=false`).

The run was terminated after 302.9 seconds, before preload completion or token generation. The guard recorded minimum MemAvailable 19.01 GiB and did not trigger its 8 GiB stop, while swap usage climbed from ~2.1 to ~27 GiB. A host sample showed ~315 MiB/s swap-out, ~50% I/O wait, and memory PSI avg10 8.37%. SIGTERM did not promptly stop the model in `io_uring` wait; only the retry8 `zeromoe` PID was then SIGKILLed. After exit, swap usage returned to ~2.5 GiB and active swap-out stopped. No throughput result was produced.

This establishes that MemAvailable-only guarding is insufficient for this preload: the run entered severe reclaim/swapping well above the 8 GiB floor. Buffered source-page caching is a likely contributor, but not yet isolated. Next run requires `drop_pages=true`, continuous swap-out/PSI tripwires that stop before severe pressure, and an idle/recovery gate. Do not raise the DDR allocation cap until this is addressed. Independently, the engine emitted `[PLACEMENT-WARN]`: DeepSeek does not advertise a common early-ready CPU lane, so mixed VRAM/DDR execution serializes MoE branches and can explain a large decode penalty.


## Post-attempt correction and safety follow-up (2026-11-03)

Retry8's engine log shows the effective in-process MemAvailable floor was 10 GiB (`10240 MB`), because `graph_mem_guard_floor_mb` was still `-1`; setting `mem_guard_stop_mb=8192` alone did not lower the built-in 10 GiB minimum. The external Python watcher had an 8 GiB floor, but retry8's minimum was 19.01 GiB, so neither threshold fired. The active DeepSeek profiles now set `graph_mem_guard_floor_mb=8192` explicitly. They also set `drop_pages=true` so successful buffered io_uring reads advise source pages away.

The runtime guard now blocks further host allocations after two consecutive samples of high memory PSI or swap-out (full PSI >=2%, some PSI >=20%, or swap-out >=64 MiB/s). The new binary built from dirty source e2167121 has SHA `2cbed8d2b4fc8d0da21bde984ffbfd5d9347a031c392a5b1a1fbf71f7462881c`; the focused `host_budget_admission` component test passed. Retry8 remains an aborted preload with no token timing.


## DeepSeek retry9: drop-pages A/B and PSI threshold (2026-11-03)

Artifact: `build_mtp_results/2026-11-03-e216712-deepseek41-engram-105gib-ddr-vram-headroom-retry9/`. Retry9 used the rebuilt pressure-aware binary, `drop_pages=true`, explicit `graph_mem_guard_floor_mb=8192`, and the same 120 GB host cap / 13,100-slot DDR request / 17.75+14.4955 GB VRAM requests. Planner cap remained 12,560 slots (100.581 GiB); VRAM pools were 17.6249 and 14.4955 GB.

At 214.7 seconds the preload had filled roughly 57 GiB. MemAvailable was 58.94 GiB, `VmSwap=0`, swap-out was ~0.008 MiB/s, but memory PSI avg10 reached 2.34%. The first pressure guard blocked allocations at that point and the runner stopped the process. The host returned to ~120 GiB MemAvailable with both GPUs free. No generation or performance timing was produced. Compared with retry8 (27 GiB swap and ~315 MiB/s swap-out at the same stage), page dropping eliminated that swap behavior through the measured point. The 2% PSI-only cutoff was over-conservative under the SSD read workload.

Next change: retain an unconditional swap-out rate stop (64 MiB/s for two polls), but gate PSI thresholds on low MemAvailable (full PSI >=4% below 32 GiB; some PSI >=20% below 48 GiB). This keeps protection against the observed reclaim incident without stopping solely on read-related PSI while RAM is plentiful.


## DeepSeek retry10: early transient swap-pressure stop (2026-11-03)

Artifact: `build_mtp_results/2026-11-03-e216712-deepseek41-engram-105gib-ddr-vram-headroom-retry10/`. Retry10 passed idle admission after the `/tmp/opencode/survey2.toml` workload exited. `drop_pages=true`, actual 8 GiB engine floor, host cap120 GB, DDR request13,100 (planner12560 / 100.581 GiB), and VRAM caches17.6249/14.4955 GB were active. At 63.8 GiB process RSS, MemAvailable was53.746 GiB and this process `VmSwap` remained0. Global swap-out peaked at217.7 MiB/s and memory PSI at3.3%; runtime allocation admission blocked briefly and recovered. The external wrapper then stopped after two samples (exit -9), before the 8 GiB floor. No tokens or performance data were produced; host and GPUs recovered.

Read-only cgroup inspection showed `memory.max=max` for the user/session scopes; host `vm.swappiness=150`. This points to transient system reclaim rather than a 64 GiB cgroup cap. `drop_pages=true` improved on retry8's 27 GiB swap growth, but retry10's wrapper threshold is too aggressive. Next runner should let the allocator block allocations during short reclaim bursts and terminate only for sustained severe swap-out, process VmSwap growth, or a low MemAvailable/PSI combination.


## DeepSeek retry11: DDR slabs swapped after the 15 GiB pin budget (2026-11-03)

Artifact: `build_mtp_results/2026-11-03-e216712-deepseek41-engram-105gib-ddr-vram-headroom-retry11/`. The stable-idle gated run used the current binary, 8 GiB effective floor, `drop_pages=true`, 120 GB host cap, 13,100-slot DDR request (planner capped to 12,560 / 100.581 GiB), and 17.75/14.4955 GB requested VRAM caches. It reached 70.06 GiB RSS with MemAvailable 47.905 GiB, but process VmSwap reached2.002 GiB, cumulative global swap-out2.103 GiB, and PSI full/some avg10 peaked5.7%. It was stopped by the external cumulative-swap guard before the MemAvailable floor; no tokens or timing were produced. The host recovered.

Cgroup files show `memory.max=max`; host `/proc/sys/vm/swappiness` is150. With `ram_pinned_budget_gb=15`, most of the planned DDR pool remains pageable, consistent with ZeroMoE's own slabs moving into zram during preload. Retry12 will pin 105 GiB (enough for the planned 100.581 GiB pool) without increasing the host allocation budget; all 8 GiB and swap/PSI guards remain.


## DeepSeek retry12: 105 GiB RAM pin-budget A/B (2026-11-03)

Artifact: `build_mtp_results/2026-11-03-e216712-deepseek41-engram-105gib-ddr-vram-headroom-retry12/`. Retry12 kept the 8 GiB floor, `drop_pages=true`, 120 GB host cap, and planned 100.581 GiB DDR + 29.91 GiB VRAM pools, but raised `ram_pinned_budget_gb` from15 to105. At 66.84 GiB process RSS, MemAvailable was50.203 GiB, process VmSwap1.771 GiB, global swap-out2.075 GiB, and memory PSI4.15%. The runtime blocked allocations during swap bursts; the wrapper stopped at2 GiB cumulative swap, exit -9. No tokens/timing; host recovered. No pin failure warnings occurred, but shutdown counters did not run, so actual slab pin success is still unverified.

Pin budget alone did not avoid early swap; next attempt will log per-process VmPin/VmLck and allow a bounded larger swap total (8 GiB global / 6 GiB process) while preserving the 8 GiB MemAvailable floor and severe PSI/swap-rate stops.


## DeepSeek retry13: full DDR preload, graph VRAM headroom failure (2026-11-03)

Artifact: `build_mtp_results/2026-11-03-e216712-deepseek41-engram-105gib-ddr-vram-headroom-retry13/`. Retry13 used `drop_pages=true`, `ram_pinned_budget_gb=105`, the effective8GiB floor, host cap120GB, requested DDR slots13,100 (byte planner cap12,560 / 100.581GiB), and VRAM caches17.75/14.4955GB. Log confirms `[PIN] Preloaded 1024 VRAM-pinned + 640 VRAM-unpinned + 12560 RAM`.

MemAvailable minimum was10.636GiB, above the floor. Process RSS peaked100.971GiB; process VmSwap5.879GiB; global swap-out6.266GiB; memory PSI peaked8.05%. The VmSwap amount is near the planner's 6.15967GB resident-tensor accounting, so it may be host resident/dense copies rather than the MoE slab pool; no shutdown pin counters were captured, and the process VmPin metric did not show the CUDA registrations. The 105GiB pin budget was configured and no cudaHostRegister failure warnings appeared.

The request generated no tokens because the prefill graph refused to build at 0.80GiB free on the 3090, below its1.00GiB safety floor. Exit -6 during CUDA cleanup; no performance result. Retry14 lowers the 3090 cache to17.25GB, preserving the 5060 Ti and DDR settings, to restore graph headroom. The decode graph memory plan still reports `Unknown`; the graph path still warns that no early-ready CPU lane is advertised.


## DeepSeek retry14: PSI-only guard blocked before graph-headroom check (2026-11-03)

Artifact: `build_mtp_results/2026-11-03-e216712-deepseek41-engram-105gib-ddr-vram-headroom-retry14/`. The 3090 cache was reduced17.75->17.25 GB; the 5060 Ti remained14.4955 GB; DDR request13100, planner cap12560 /100.581GiB, pin budget105GiB, `drop_pages=true`, and actual8GiB engine floor. The wrapper stopped during DDR preload at81.59GiB RSS /30.79GiB MemAvailable when PSI reached8.17%. Process VmSwap peaked5.696GiB, cumulative swap5.860GiB; it did not hit the8GiB MemAvailable floor, and no graph or tokens ran. Host recovered.

Retry15's result is recorded below. Its guarded runner stopped after MemAvailable fell below 8 GiB, so it is not a valid performance run.
## DeepSeek retry15: DDR preload crossed the configured floor (2026-11-03)
Artifact: `build_mtp_results/2026-11-03-e216712-deepseek41-engram-105gib-ddr-vram-headroom-retry15/` in the isolated `codex/deepseek-v41-continuation` worktree. The binary was the earlier `e2167121...` dirty build (snapshot `9f921991...`), not the current post-placement-plan source; the log reports `memory_plan=unknown`.
The run preloaded 12,560 DDR expert slots (100.581 GiB) plus 1,664 VRAM cache slots, then completed the 97-token full-graph prefill through all 40 layers at 20.27 tok/s and generated one token. The wrapper stopped the request before its requested eight tokens after MemAvailable crossed the 8 GiB floor. `bench.json` marks the measurement invalid; no decode or performance result is accepted.
Preflight admitted 120 GB host RAM plus an 8 GiB margin with only about 0.2 GB of additional MemAvailable (128.8 GB observed vs 128.6 GB required). The wrapper low-water was 7.2596 GiB; the engine reported 5,010 MB. Process RSS peaked at 104.845 GiB, process VmSwap at 5.903 GiB, global swap-out at 6.531 GiB, PSI at 8.4%, and swap-out at 336 MiB/s. Preload ended at 12:49:31 and full-graph prefill ended at 12:49:37; the floor was crossed during decode around layer 16 at 12:49:39. Between two wrapper samples, MemAvailable fell 1.357 GiB while process RSS rose 1.578 GiB. The exact source of this decode-time RSS growth remains unresolved. The guard logged the low-water violation but did not interrupt the in-flight graph; the outer wrapper stopped it. Trace decode graph allocations, persistent pinning, and cache-upload paths before another full-pool run. Do not rerun this host cap unchanged.


## DeepSeek retry16: memory-safe decode graph smoke (2026-11-03)
Artifact: `build_mtp_results/2026-11-03-c0e9963-deepseek41-engram-114gb-safe-decode-retry16/` in `codex/deepseek-v41-continuation`. The current-master dirty build passed provenance check (`c0e9963`, GGML `8baabd53`). The focused `host_budget_admission` component test passed.
Retry16 lowered host cap to114GB while retaining the8GiB engine floor; the planner selected11,840 DDR slots /94.815GiB. It preloaded all selected DDR experts, ran 40-layer graph prefill for 97 tokens at19.20 tok/s, built full-graph decode, and completed one decode step. The full-fit run gate passed. The wrapper's11GiB stop fired before the requested eight tokens; wrapper minimum was10.1517GiB and engine low-water was9,752MB, both above the8GiB floor. Process swap peaked0.396GiB, global swap-out1.251GiB, memory PSI5.55%, and swap-out300.9MiB/s. `bench.json` is invalid and the one-token rate is not a performance result.
Retry16 shows the graph path and memory planner pass their smoke gates with the reduced pool. Retry17 lowers host cap by another1GB to113GB and keeps the11GiB wrapper stop to test whether all eight decode tokens complete above the engine floor. It still does not test the100.581GiB DDR pool or accept correctness/performance; teacher-forced boundary parity and a matched llama.cpp performance check remain required.

## DeepSeek retry17: two graph decode steps completed above the floor (2026-11-03)
Artifact: `build_mtp_results/2026-11-03-c0e9963-deepseek41-engram-113gb-safe-decode-retry17/`. Host cap113GB produced11,720 DDR slots /93.854GiB. It completed full-graph prefill (97 tokens at19.79 tok/s), then two full-graph decode steps. `check_full_graph_moe_run.py --full-fit` passed. The wrapper's11GiB stop interrupted the eight-token request after two tokens; `bench.json` is invalid and the0.49 tok/s two-token wall rate is not performance data.
Wrapper minimum MemAvailable was10.641GiB; engine low-water was10,916MB, above the8GiB floor. Process swap stayed0, global swap-out peaked0.257GiB, and the engine reported21 allocation refusals plus one reclaim block. The next test keeps this host cap and wrapper stop but uses three bench repetitions in one loaded process, providing more prefill/decode samples without reloading between reps. This remains smoke-only; llama.cpp boundary parity and a matched performance check remain required.

## DeepSeek retry18: one complete multi-rep smoke (2026-11-03)
Artifact: `build_mtp_results/2026-11-03-c0e9963-deepseek41-engram-112gb-multirep-retry18/`. The 112 GB host-cap run attempted three repetitions of a 97-token prompt plus eight decode tokens. The first request completed; the wrapper stopped during repetition two when MemAvailable reached10.975GiB, above the8GiB engine floor. No `bench.json` or final graph-gate summary was written.
First complete request: prefill20.34 tok/s; decode1.24 tok/s by the engine counter and0.99 tok/s over decode wall; p50/p95 decode latency798.8/962.2ms. The full request took12.91s including prefill (0.62 tok/s overall). Placement plan selected11,640 DDR experts /93.213GiB; post-prefill occupancy was11,672/11,760 DDR slots (100.37GB) and2,578/2,578 VRAM slots (31.729GB). No NaN/Inf report; process swap peaked0, global swap-out0.091GiB, memory PSI2.78%. `check_full_graph_moe_run.py --full-fit` failed because interruption prevented the final `[MOE-GRAPH-GATE]` summary. The graph completed one prefill and eight decode tokens, but this is not onboarding acceptance or a matched llama.cpp performance result.
This run finally provided separate prefill and decode rates after one load. The wrapper stopped the second repetition's prefill at11GiB. The remaining host-memory growth between requests needs tracing before more repeated long runs; the rates remain far below100 tok/s prefill /10 tok/s decode.

## DeepSeek survey-tier checks: retries19-20 (2026-11-03)
Retry19's first sample-only plan was stopped before inference because it requested1,024 pinned +2,554 VRAM-LRU slots for a2,578-slot VRAM pool. Retry20 lowered the LRU fraction but set DDR capacity2,100; per-layer limits then loaded only1,777 DDR entries. Startup preload was 1,024 pinned +1,520 VRAM-LRU +1,777 DDR =4,321 of the4,547 survey identities.
Retry20's first request completed 97-token graph prefill at3.94 tok/s and eight decode tokens at0.48 tok/s engine rate (0.52 tok/s wall). It still read70.95 GB expert data from SSD; decode logged1,407ms/token SSD read. RAM/VRAM overlap reached1,002 identities. A second request errored; the full-fit gate passed but this is not performance acceptance. MemAvailable stayed above90GiB and no swap occurred. Retry21 raises DDR capacity to5,000 (125 slots/layer) but leaves `ram_preload_all_experts=false`, aiming to preload the remaining sampled experts without adding history-missing identities. Check `[PIN] Preloaded` against4,547 before interpreting the next SSD-read result.

## DeepSeek decode CPU/VRAM split lever (2026-11-03)

A code trace found an existing DeepSeek-specific decode split in `core/token_dispatch_pipeline.cpp`: when `cpu_offload_max_experts>0` on S=1, selected DDR experts outside the active-layer VRAM set can be forced onto the CPU executor while already-issued GPU expert kernels remain asynchronous. The DeepSeek route keeps per-expert outputs and accumulates in original k order, preserving the existing reduction tree. The benchmark profile had this knob at0, so the path was disabled. Retry15 now sets it to6 (the model's top-k) for an A/B. This is the shared cache-backed executor path; it does not make `DeepseekV41GraphModel::supports_cpu_lane()` true, and the full-graph memory plan remains `Unknown`. Measure with correctness checks before treating it as a performance improvement.


## DeepSeek graph CPU-cold path found (2026-11-03)

The existing `V41CacheMoeSite::cpu_cold_graph` path in `models/deepseek/ggml_deepseek_v41_graph.cpp` is enabled by typed `graph_cache_moe_cpu_cold`. It runs DDR-owned selected experts in an async CPU job and leaves VRAM-owned experts in the graph's GPU branches, then merges outputs. Retry15 enabled this path, but its guarded run ended after one generated token and is too short to assess CPU/GPU overlap or speed. `cpu_offload_max_experts=6` remains only for the cache-backed fallback. Do not overlap the next run with the external prefill workload.


## DeepSeek V4.1 five-prompt usage survey (2026-11-03; exploratory)

Owner/worktree: DeepSeek continuation session, branch codex/deepseek-v41-continuation. Source commit c0e996355051a277086644dd40ffe995bd8da2df was dirty; the binary manifest records source snapshot SHA256 67ded5e9561a41d3a787b8de81d842d7385aae34e0a5c36f224caaaf69721477. Binary SHA256 4511529f9f9fd05205cc5b2a7b8e4b0a3d4459cb7b3e29a116cb2e2869cc0c43. Shared GGML base 8baabd53b77d1c48c6ef8d3b620436292f3f6231; ZeroMoE GGML patch-manifest SHA256 5c4c17e7164237cf7da1570a5dcf1194fbd700df0ee3ca980bf96dcc4530f95b. Toolchain: CUDA 13.3.73 / driver 610.57.04, GCC 16.2.1, CUDA targets 86;120. Hardware: RTX 3090 + RTX 5060 Ti. Runtime environment: CUDA_VISIBLE_DEVICES=0,1, OMP_NUM_THREADS=10, NVIDIA_TF32_OVERRIDE=0.

Workload: main GGUF DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00006.gguf plus Engram sidecar from /mnt/usb-ssd/@/home/ray/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO/; context 4096, temperature 0.8, full graph prefill/decode, host cap 85 GB, 8 GiB startup margin, 11 GiB wrapper stop. Five project prompts were run in one model process, one repetition each, 160 output tokens each, no warmup: Python CSV (22 tokens), C++ delete review (61), OS deadlock (22), world history (12), distributed-systems summary (141). All five request rows are valid; 800 output tokens total.

| Prompt | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: |
| Python CSV average | 0.48 | 0.79 |
| C++ delete review | 2.02 | 0.31 |
| OS deadlock | 1.22 | 0.30 |
| World history | 0.94 | 0.28 |
| Distributed systems summary | 2.22 | 0.26 |

Median decode is 0.30 tok/s, aggregate decode 0.33 tok/s; the first prompt is a faster outlier. The full-fit graph run checker passed. This was a usage survey, not performance acceptance, and has no matched llama.cpp row. The wrapper measured 64.923 GiB minimum MemAvailable, zero process swap, zero global swap-out, and no emergency/pressure stop. After saving all five rows and usage files, the process exited -6 in CUDA teardown (CUDA error: driver shutting down at cudaFree); treat request data as complete but retain this teardown issue.

Usage artifacts: build_mtp_results/2026-11-03-c0e9963-deepseek41-five-prompt-route-survey/. route_usage.txt is prefill plus decode (11,654 unique layer/expert pairs, 252,720 selections); prefill_usage.txt has 7,913 pairs / 61,920 selections. decode_usage.txt is derived by subtracting prefill counts from total route counts per pair (10,609 pairs / 190,800 selections). Hashes and prompt provenance are in usage-summary.json and prompt-manifest.json in that directory. No prior survey experts were preloaded for this run.

The runtime summary still shows a cold-miss bottleneck: 12,833 expert-cache hits vs 94,144 misses (12% hit rate), zero GPU expert calls vs 96,321 CPU expert calls, 811.86 GB of real expert SSD reads at 0.26 GB/s accumulated, and 113.58 GB H2D uploads. Follow-up should use decode_usage.txt as a decode-weighted VRAM-first/DDR-second placement source, then verify actual pinned/resident counts, GPU expert-call counts, and SSD-read reduction. Do not call this a performance improvement until the matched llama.cpp check meets the repository's relative bar.


### Five-prompt usage ranking vs the original one-prompt file

The old 4,547-identity histogram has 24,960 selections, of which 23,280 (93.3%) are prefill and only 1,680 are the seven routed decode positions from its one request. Against the new five-prompt sample, old vs current top-2,000 overlap is 716/2,000 (35.8%) for combined usage and 616/2,000 (30.8%) for decode-only. The old top-2,000 identities cover 33.5% of current decode selections; the current decode-ranked top-2,000 covers 72.3%.

The old route file has only 4,547 active identities. The current combined top-10,000 contains 3,997/4,547 old identities and captures 99.29% of current combined selections. The current decode top-10,000 contains 3,781/4,547 old identities and captures 99.68% of current decode selections; the entire old set captures 54.66% of current decode selections. Old route minus old prefill leaves only 809 identities across seven routed decode positions, which is too small to be a stable decode-rank reference. Top-2,000 cutoff ties are substantial, so selection-mass coverage should accompany identity overlap.

Detailed comparison and reproducible script: build_mtp_results/2026-11-03-c0e9963-deepseek41-five-prompt-route-survey/usage-rank-comparison.md and compare_usage_rankings.py. The placement candidate for this workload is the new decode_usage.txt, with VRAM-first and DDR-second ranking; verify actual resident counts and SSD-read reduction before accepting it.

## DeepSeek V4.1 frozen-survey replay and full-load status (2026-11-03)

Source/worktree: codex/deepseek-v41-continuation, project commit c0e996355051a277086644dd40ffe995bd8da2df, dirty build snapshot 67ded5e9561a41d3a787b8de81d842d7385aae34e0a5c36f224caaaf69721477. The selected binary provenance check passed (binary SHA256 4511529f9f9fd05205cc5b2a7b8e4b0a3d4459cb7b3e29a116cb2e2869cc0c43; GGML base 8baabd53b77d1c48c6ef8d3b620436292f3f6231; patch manifest 5c4c17e7164237cf7da1570a5dcf1194fbd700df0ee3ca980bf96dcc4530f95b). CUDA 13.3.73, driver 610.57.04, GCC 16.2.1, targets 86;120, RTX 3090 + RTX 5060 Ti.

The same five survey prompts were replayed with the decode histogram as pin_file, prefill histogram as prefill_pin_file, and freeze_usage_stats=true. Placement used 10,000 DDR slots, 1,024 pinned VRAM experts, and 1,520 VRAM-LRU experts. All five 160-token requests were valid; full-graph gate passed. Per-prompt prefill rates were 5.28, 3.08, 2.10, 1.38, 2.64 tok/s; decode wall rates were 1.28, 1.15, 0.77, 0.48, 0.35 tok/s. Median decode was 0.77 tok/s; aggregate decode 0.63 tok/s. Minimum MemAvailable was 24.135 GiB, zero process swap/global swap-out, no pressure stop. Exit -6 occurred only during CUDA teardown after results were saved.

This was not a full-MoE-resident run. Planner coverage was at most 12,578/15,360 identities (DDR plan 80.08 GiB; VRAM cache 29.55 GiB). Runtime counters: zero GPU expert calls / 66,656 CPU expert calls, cache hit rate 47.7% (36,884 hits / 40,361 misses), 458.43 GB real expert SSD reads at 0.28 GB/s, 407 SLOW-PREAD events, and 168.41 GB H2D uploads. Expert source path is the USB SSD at /mnt/usb-ssd/@/home/ray/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO. Prompt phase logs include main-thread RAM claims that fell through to real disk reads. The current run therefore confirms SSD reads in both the prefill and decode workloads.

The user supplied the active-parameter comparison: GLM-5.3-Flash 18B active vs DeepSeek V4.1 Flash 16B during decode and 8B during prefill. The user-reported GLM-Flash rate is 14 tok/s; it is not a matched local run in this artifact. The active compute difference is too small to explain the observed gap by itself. DeepSeek logs show the more likely path issue: its full graph owns zero MoE weight bytes and dispatches experts through the existing VRAM/DDR/CPU cache path; the DeepSeek adapter warns it has no common early-ready CPU lane to overlap hot-GPU and cold-CPU branches. The 10 tok/s decode and 100 tok/s prefill targets remain unmet.

The user then required full MoE residency before further performance testing. A full-target attempt requested 13,100 DDR slots, but byte-aware admission allowed only 12,320 plus 2,578 VRAM slots (14,898/15,360 identities). It was stopped before prompts as instructed; no speeds were recorded. Min MemAvailable was 18.477 GiB, process swap 2.496 GiB, cumulative global swap-out 3.437 GiB, and there is no active process now. Current machine admission is short by 462 DDR identities even with the full 2,578-slot VRAM cache. Do not restart performance testing until the full-load admission constraint is addressed; keep the 8 GiB reserve and 11 GiB wrapper stop.

Artifacts: build_mtp_results/2026-11-03-c0e9963-deepseek41-frozen-10000-ddr-nearfull-host116-replay5/ (valid partial-capacity performance run) and build_mtp_results/2026-11-03-c0e9963-deepseek41-frozen-full-moeresident-replay6/ (full-load plan, stopped before prompt).

### DeepSeek V4.1 full residency with a 4 GiB startup reserve

The user lowered the startup reserve to 4 GiB. The supported live engine stop
floor remains 8 GiB; setting graph_mem_guard_floor_mb=4096 is rejected by
configuration validation. With host cap123 GB and startup margin4 GiB, the
planner admitted 12,880 DDR slots plus a 2,578-slot VRAM cache. Actual preload
loaded 12,740 RAM + 2,544 VRAM entries (15,284/15,360 identities). The engine
then refused 964 request-time host allocations; all five bench rows errored
with zero prompt tokens. Low-water MemAvailable was 12.045 GB, process swap
5.766 GiB, global swap-out 6.222 GiB; no wrapper pressure stop fired. No
performance result is valid. There is no active model process. The full-load
requirement is not met; wait for more host headroom or change the placement
capacity before another performance run.

## GLM-5.3-Flash cache-hit reference and DeepSeek full-load blocker (2026-11-03)

The user-provided comparison is GLM-5.3-Flash (18B active) at a reported
14 tok/s versus DeepSeek V4.1-Flash (16B active in decode, 8B in prefill).
The available log at /home/ray/work/moestream-graph/glm_runs/baseline.engine.log
for the AJ IQ2_XXS GGUF reports 2,951 cache hits / 0 misses (100%) and
6.38 tok/s decode (96 output tokens); it is not the reported 14 tok/s run.
Other local GLM-5.3-Flash pinab traces report 56.9-57.1% cache hits and
4.06-4.14 tok/s decode. The log matching the 14 tok/s report and its cache
counter have not been identified; treat 14 tok/s as user-reported, not a
matched local benchmark. The referenced external worktree's .git metadata is
invalid, so its project revision cannot be verified.

DeepSeek's valid frozen five-prompt replay at
build_mtp_results/2026-11-03-c0e9963-deepseek41-frozen-10000-ddr-nearfull-host116-replay5/
measured decode wall rates 1.28, 1.15, 0.77, 0.48, 0.35 tok/s (median 0.77;
aggregate 0.63) and prefill 5.28, 3.08, 2.10, 1.38, 2.64 tok/s. Despite the
full-graph gate passing, runtime counters show 0 GPU expert calls, 66,656 CPU
expert calls, 47.7% expert-cache hit rate, 458.43 GB SSD reads, and 407
slow-pread warnings. The adapter logs that it cannot overlap CPU-cold and
GPU-hot branches; graph-owned MoE bytes are zero. This dispatch/placement
path, rather than the 16B-vs-18B active-parameter count, is the leading
explanation for the speed gap. These runs are not prompt-matched to the
user-reported GLM 14 tok/s result.

The full-load gate remains blocked. Under host cap123 GB and startup margin
4 GiB, the planner allowed 12,880 DDR slots plus 2,578 VRAM slots, while the
actual preload reached 12,740 RAM + 2,544 VRAM identities (15,284/15,360).
The engine refused 964 further host allocations; all bench requests errored
before tokenization, with zero generated tokens. MemAvailable low-water was
12.045 GiB, process swap 5.766 GiB, and cumulative global swap-out 6.222 GiB.
No further performance run should start until the full 15,360-expert MoE set
can be admitted while preserving the configured live floor. No DeepSeek
process is running at handoff.

### SSD-read phase and GLM cache-hit evidence

For the DeepSeek near-full replay, prefill RAM-claim diagnostics recorded
4,269 main-thread claims that required a real disk read across the five
prompts. The 407 SLOW-PREAD warnings were observed around prefill phases.
The shutdown report separately counted 27,922 decode-thread io_uring batches,
with only 200 marked in_prefill, so decode also issued many SSD-backed reads.
The 458.43 GB expert-read total includes startup preload plus runtime reads;
the log does not provide an exact decode-only byte subtotal.

The GLM recipe's referenced baseline.engine.log reports 2,951 expert-cache
hits and zero misses (100%) for a 96-token run, but that log's decode rate is
6.38 tok/s, not the user's 14 tok/s result. Other GLM-5.3-Flash pinab logs
show about 56.9-57.1% hits and 4.06-4.14 tok/s. I have not found a cache-hit
counter for the user's 14 tok/s run, so do not attach those other runs' hit
rates to that figure. The user-provided GLM result remains the intended
performance reference.

Current full-residency attempt still does not meet the user requirement. The
full target process was stopped before any prompt speed could be accepted:
byte-aware DDR cap 12,880, actual preload 12,740 RAM + 2,544 VRAM, and 964
host allocations refused. No model process is active.

### User correction: GLM SSD references are distinct

The user clarifies that GLM-5.3-Flash at 14 tok/s has no SSD reads during
decode, while GLM-5.3 non-Flash at about 2 tok/s does read from SSD. Do not
conflate them. The DeepSeek replay's 407 slow pread warnings were during
prefill, but its decode-thread io_uring report counted 27,922 Phase-B calls
with only 200 in_prefill, indicating ongoing decode-path SSD I/O. The engine's
458.43 GB disk-read total includes preload and runtime; there is no
decode-only byte counter. Full MoE residency is still not achieved, so this
does not isolate the SSD-free DeepSeek performance ceiling.


### 2026-11-03 DeepSeek V4.1 full-residency placement result

A strict placement path was added in the DeepSeek worktree. It fills the complete uniform-stride VRAM pool, pins all VRAM assignments, gives each layer’s exact remainder to DDR, and checks the union of both tiers before accepting requests. The guarded load passed: **15,360/15,360 experts resident** (12,782 DDR / 2,578 VRAM), with zero misses and zero tier overlap. DDR slabs used 109.79 GB against a 110.749 GB safe expert budget; VRAM cache used 31.729 GB. This supersedes the earlier 15,284-expert partial placement.

The host stayed above the 8 GiB stop floor (11.512 GiB low-water) and the wrapper did not emergency-stop. The engine memory guard blocked growth once; process swap peaked at 3.975 GiB and global swap-out at 4.481 GiB. The one-token smoke request was rejected by CLI validation (`--bench-ngen` minimum 8), so no inference request ran and no prefill/decode speed or request-time SSD behavior was measured.

Artifacts: `build_mtp_results/2026-11-03-c0e9963-deepseek41-full-residency-layout-check/` (config, command/environment, binary manifest, stdout/stderr, memory samples, and wrapper summary). Tested binary SHA-256 is in that directory’s `binary.build-info.json`; source commit `c0e996355051a277086644dd40ffe995bd8da2df`, GGML base `8baabd53b77d1c48c6ef8d3b620436292f3f6231`. Current rebuilt binary has matching worktree provenance; only the residency-plan log wording changed after the load. Next performance request must use at least 8 generated tokens and should be treated as a separate run.


### Follow-up: persistent prompt test stopped by pressure guard

The requested long-prompt/64-token attempt again passed strict residency (15,360/15,360: 12,782 DDR + 2,578 VRAM), but the external guard stopped the process during the post-load memory-recovery wait before any request was sent. Process swap reached 5.007 GiB against the 5 GiB limit; cumulative swap-out was 5.229 GiB, PSI peaked at 10.13%, and MemAvailable stayed above the 8 GiB floor (11.998 GiB low-water). No prefill/decode speeds or request-time SSD measurements were produced. This is the second pressure-related stop; follow the user's instruction to wait for further direction before another load. Artifacts: `build_mtp_results/2026-11-03-c0e9963-deepseek41-full-resident-longprompt-64gen/`.


### Persistent long-prompt API attempt: strict-pin/cache interaction

Strict residency again passed (15,360/15,360: 12,782 DDR + 2,578 VRAM), and the API health endpoint returned 200. The host guard recovery gate cleared before request dispatch. The long-prompt request returned HTTP 500 after 14.29 seconds with zero prompt tokens processed. The exact error was `upload_expert: evict_lru found NO victim for layer=0 expert=0 (cache_cap=2578, unpinned_slots=0)`, followed by `DeepSeek cache graph failed to hold VRAM expert`. Strict mode pinned every VRAM slot to preserve residency, while prefill attempted to upload a DDR expert into that full cache. The API runner treated the graph failure as fatal and exited; no signal was sent. No speed or request-time SSD result exists. The user instructed to wait after a repeated failure: **do not restart until directed**. Artifacts: `build_mtp_results/2026-11-03-c0e9963-deepseek41-persistent-server-longprompt-64gen/`. Next implementation must let the cache graph consume DDR-resident experts without requiring an evictable VRAM victim, or provide a safe DDR-backed replacement path, while preserving full-set residency.


### Source fix after strict-pin prefill failure (not runtime tested)

The API process exited because `server.cpp` treated a generation exception as server-fatal. It now returns HTTP 500 for an inference exception while keeping the API loop alive. The strict VRAM layout now reserves 32 empty, evictable upload slots; the corresponding experts remain in the DDR complement, and strict RAM promotion keeps its DDR source copy. The 3090 cache is lowered by 0.60 GB and the 5060 Ti cache raised by 0.60 GB to increase graph-GPU prefill workspace without increasing total VRAM pool bytes. Strict placement now also requires an explicit 0.65 GB host request/prefill reserve in addition to expert slabs. These changes compile and pass build-provenance checks but have **not** been loaded or tested after the HTTP 500. Per the user's instruction, do not restart until directed.


### Strict-residency follow-up source changes (compiled, not runtime tested)

After the no-victim prefill failure, the strict profile now leaves 32 main-cache slots empty/evictable for graph staging and places their source experts in the DDR complement. Strict placement reserves an explicit 0.65 GB host budget for prompt/request activations; the exact DDR slab bytes plus this headroom must fit before preload starts. The dual-GPU profile shifts 0.60 GB of expert-pool allocation from the 3090 to the 5060 Ti to increase graph-GPU prefill workspace. Strict mode retains DDR source copies on VRAM promotion. These settings have no hidden defaults: strict mode requires the TOML to specify both dynamic slots and host headroom.

The API server now turns a generation exception into HTTP 500 for that request and continues serving, instead of setting the server-wide stop flag. The 30-second pre-request pressure-recovery gate was removed; the request-time MemoryGuard remains the admission check. Build completed and provenance check passed (binary SHA-256 `88c7d1a0132b56f231a679ff0527f8bfb553875ed0678382b5430189771892ab`). **These latest changes have not been loaded or prompt-tested.** No model process is active. Do not reload until the user directs, per their instruction after the repeated prompt failure.


### Persistent-server retry: startup path error

The current server retry exited before loading the model: `GGUFModel::load_header` reported the configured absolute first-shard path as nonexistent. The same path parses from TOML and is readable by the shell (`stat`/`GGUF` magic), so this remains an unresolved startup-path discrepancy. No prompt ran, no memory pressure stop occurred, and no `zeromoe` process remains. Per the user's instruction after repeated failure, do not retry until directed. Artifact: `build_mtp_results/2026-11-03-c0e9963-deepseek41-persistent-server-longprompt-64gen-v3/`.


### Reload retry: repeated GGUF path-not-found before model load

After the user authorized a reload, a persistent non-PTY `--serve` launch again failed in `GGUFModel::load_header` with “model path does not exist” for the configured first shard. The same absolute path and Engram file parse from TOML and are readable with `stat`/`read` from the host shell. The failure occurred before weight loading; no prompt ran and no model process remains. Per the user's instruction after a repeated failure, no further load attempt has been started. Artifacts: `build_mtp_results/2026-11-03-c0e9963-deepseek41-persistent-server-longprompt-64gen-v3/`.


### Persistent server result with 32 staging slots

The revised profile loaded and passed strict residency: 15,360/15,360 (12,814 DDR + 2,546 VRAM), with 32 dynamic staging slots and 0.65 GB host headroom. The long prompt (`S=959`) failed in prefill at layer 0 expert 59; a follow-up `hello` request failed at layer 0 expert 218. Both errors were `evict_lru found NO victim` with 32 unpinned slots; all were occupied/temp-pinned/upload-pending during the graph prefill. The request path returned HTTP 500, and `/health` remained 200—the server stayed loaded, as requested. No decode token or timing was produced, and no retry was sent. This confirms 32 staging slots are insufficient for the 959-token whole-prompt graph. The live server remains running; wait for user direction before another request or layout change. Artifact details: `build_mtp_results/2026-11-03-c0e9963-deepseek41-persistent-server-longprompt-64gen-v3/`.


### Current loaded-server memory snapshot

The persistent API still returns health 200 after the no-victim errors. Current `/proc/meminfo` reports MemAvailable 11.260 GB (10.49 GiB) with an 8 GiB floor, leaving about 2.49 GiB above the floor; system SwapUsed is 45.342 GB (42.24 GiB). No additional prompt was sent. The remaining issue is graph staging/temporary VRAM holds, not absence of DDR-resident model experts.


## DeepSeek V4.1 Flash prefill crash and guarded follow-up (2026-11-04)

The `codex/deepseek-v41-continuation` worktree (source commit `c0e996355051a277086644dd40ffe995bd8da2df`) loaded all 15,360 MoE experts (12,814 DDR + 2,546 VRAM; 32 staging slots). The 959-token full prefill graph built (40 layers, 6,318 nodes, 0.63 GiB GPU workspace), then the process segfaulted in the asynchronous CPU cold-prefill lane at layer 0 before decode. Core metadata showed two out-of-range IDs (`1301014451`, `1759629055`) in a 226-entry spill list being written into a 384-entry mask. The remaining spill IDs were valid. No prefill/decode rate was measured. systemd produced a 31.6 GB core at `/var/lib/systemd/coredump/`; it remains untouched.

The DeepSeek adapter now snapshots the CPU spill IDs, validates them before launching the worker and after DDR lookup, and reuses the already loaded/pinned DDR blocks for CPU overflow instead of issuing a second batch lookup. This compiled and passed provenance; current candidate SHA-256 is `c46f18d96981d104d0300c921f82287f6b715b333607558e35f502222d4b13ef`. It has not been runtime-tested.

The guarded follow-up load was refused before model allocation: MemAvailable was 124.7 GB versus the configured 127.3 GB requirement (123 GB host cap + 4.3 GB startup margin). Current MemAvailable is 121.9 GB; both GPUs passed five idle queries. No `--force-start` was used and no model process is active. The next safe run requires at least the configured headroom and should keep core dumps disabled. Full prefill/decode acceptance and the 100/10 tok/s targets remain unverified. Artifacts: `build_mtp_results/2026-11-04-c0e9963-deepseek41-prefill-overflow-959prompt-64gen-v1/` and `...-v2-id-guard/`.


### DeepSeek CPU spill-block handoff follow-up (2026-11-04)

The layer-0 core showed the async CPU cold-prefill worker indexing a 384-entry spill mask with two corrupted IDs from a 226-entry list. The worker now copies and validates that ID list before/after DDR lookup. Prefill also transfers the already loaded and pinned `MoEExpertBlock` pointers from the route-staging batch into the CPU lane, avoiding a second `get_experts_batch` call. Candidate binary SHA-256: `c46f18d96981d104d0300c921f82287f6b715b333607558e35f502222d4b13ef`; build and provenance check passed. This source has not been runtime-tested.

A fresh strict-residency start was refused before model load: MemAvailable 124.7 GB vs 127.3 GB required; current observed MemAvailable is 126.78 GB (still 0.52 GB below the configured threshold). No `--force-start` was used. A prepared, core-dump-disabled run is at `build_mtp_results/2026-11-04-c0e9963-deepseek41-prefill-overflow-ddr-blockreuse-959p-64gen-v3/`. The 100 tok/s prefill and 10 tok/s decode targets remain outstanding.


### DeepSeek V4.1 Flash DDR/host-cap recheck (2026-11-04)

The first pinned-block candidate (binary SHA-256 `c46f18d96981d104d0300c921f82287f6b715b333607558e35f502222d4b13ef`) builds and passes provenance. Its DeepSeek prefill CPU lane now reuses DDR blocks fetched and pinned during the main route-staging batch; it does not issue a second `get_experts_batch`. CPU spill IDs are snapshotted and range-checked. No runtime prompt has validated this change.

A 122 GB host-cap trial passed startup preflight at 126.5 GB but failed the full-residency byte check: safe DDR budget 110.092 GB versus the required DDR complement plus 0.65 GB headroom (about 110.726 GB). The prepared full-residency candidate restores the 123 GB cap and 4 GiB startup margin; its preflight threshold is 127.295 GB. A one-shot core-disabled launcher waits for three consecutive MemAvailable samples above that threshold, then relies on ZeroMoE's own per-GPU preflight; it never force-starts or retries. Current MemAvailable has remained about 126.7 GB, so no model is loaded and no performance result exists. Candidate/run artifacts: `build_mtp_results/2026-11-04-c0e9963-deepseek41-prefill-overflow-pinnedblocks-959p-64gen-v4/`.


### DeepSeek versus GLM graph reference (2026-11-04)

The current DeepSeek candidate keeps the model-specific CPU overflow path; GLM source offers two useful, model-specific patterns: prefill GPU op-offload with a per-layer asynchronous cold-expert ring split across PCIe links, and an early-ready CPU cold lane for decode. The user-specified GLM AJ IQ2_XXS recipe reports ~272 prefill/~15 decode tok/s; the readable `~/work/moestream-graph-framework` performance record reports 272.7 prefill and 11.7-12.9 decode, but its `.git` pointer is invalid so its project commit is unverified. DeepSeek retest settings and material differences are recorded in `build_mtp_results/2026-11-04-c0e9963-deepseek41-prefill-overflow-pinnedblocks-959p-64gen-v4/README.md`.


### DeepSeek current-memory VRAM rebalance candidate (2026-11-04)

The 123 GB host cap plus 4 GiB startup margin remains below current MemAvailable by about 0.6 GB. A 122 GB cap with the original VRAM split failed strict DDR planning by 0.634 GB. Candidate v5 therefore shifts 0.70 GB of MoE cache to GPU0/3090 (17.35 GB total pool there; GPU1 remains 15.0955 GB) while keeping 32 dynamic slots and the 0.65 GB host prompt reserve. This should move roughly 0.7 GB of model experts from DDR to the faster x16 3090 and preserve full expert coverage; the runtime's per-layer byte planner is the authority and must reject it before preload if exact bytes do not fit. The v5 run folder has host cap122, startup margin4 GiB, core limit0, and the saved 959-token/64-output workload. Its gate threshold is 126.295 GB; no model has started. Artifacts: `build_mtp_results/2026-11-04-c0e9963-deepseek41-vram3090-17p35-ddr-cap122-959p-64gen-v5/`.


### DeepSeek full-residency prefill/decode attempt v21 (2026-11-04)
Source/worktree: `codex/deepseek-v41-continuation`, commit `c0e996355051a277086644dd40ffe995bd8da2df`, dirty snapshot. Binary SHA-256 `9365978283c3c40c7523626e65fb24bb62b803807dcd8df84fd110bf1f2b6a0a`; GGML base `8baabd53b77d1c48c6ef8d3b620436292f3f6231`; patch manifest SHA-256 `5c4c17e7164237cf7da1570a5dcf1194fbd700df0ee3ca980bf96dcc4530f95b`; native targets 86/120. Runtime artifacts: `build_mtp_results/2026-11-04-c0e9963-deepseek41-fullgraph-prefill-decode-v21/`.

The guarded run loaded the user-specified USB GGUF and Engram through host-context symlink aliases (no GGUF copies). Full residency passed: 15,360/15,360 experts (12,757 DDR, 2,603 VRAM), with 32 dynamic VRAM slots reserved. The saved 959-token prompt completed all 40 full-graph prefill layers with CPU cold-expert offload: 62.507 s, 15.3423 tok/s, 6,492 CPU expert calls, zero GPU expert calls, and zero MoE disk reads counted by the prefill engine. Prefill is far below the 100 tok/s target.

The same request asked for 64 decode tokens, but required full-graph decode construction failed before token generation: GPU 0 reported 0.98 GiB free against a 1.00 GiB graph allocator floor. HTTP 500 returned with zero generated tokens; decode speed is unmeasured, and no layered fallback occurred. The server remained healthy (`/health` 200) and was left running; the user-directed no-retry-after-request-failure rule is in force. During the request, host MemAvailable remained above 10.015 GiB, process swap was zero, and memory PSI remained zero. No retry or restart pending user direction.


### DeepSeek v22 full-graph prefill/decode and decode-throughput diagnosis (2026-11-04)
Source/worktree: `codex/deepseek-v41-continuation`, commit `c0e996355051a277086644dd40ffe995bd8da2df`, dirty snapshot. Binary SHA-256 `9365978283c3c40c7523626e65fb24bb62b803807dcd8df84fd110bf1f2b6a0a`; GGML base `8baabd53b77d1c48c6ef8d3b620436292f3f6231`; patch manifest SHA-256 `5c4c17e7164237cf7da1570a5dcf1194fbd700df0ee3ca980bf96dcc4530f95b`; CUDA architectures 86/120. Artifacts: `build_mtp_results/2026-11-04-c0e9963-deepseek41-fullgraph-prefill-decode-v22/`.

The guarded run used the USB GGUFs via symlink aliases (no payload copies), host cap 123 GB, startup margin 4 GiB, 4.5 GiB live stop, and 32 dynamic VRAM slots. Full residency passed: 15,360/15,360 experts (12,789 DDR + 2,571 VRAM). The 959-token full-graph prefill with CPU spill completed all 40 layers in 62.803 s (15.27 tok/s), below the 100 tok/s target. It made 6,492 CPU expert calls and zero GPU expert calls.

Required full-graph decode completed all 64 tokens at 1.00 tok/s (p50 1,015 ms, p95 1,113 ms), below the 10 tok/s target. Decode logged 16,441 CPU expert calls, zero GPU expert calls, and zero routed DDR experts staged to VRAM. CPU cold workers accounted for 102.85 s summed worker time across the request; decoded experts were present in DDR, and the engine reported zero real MoE disk reads in prefill/decode. Host `/proc` read-bytes rose 3.165 GB during the request (source not attributed by those counters). MemAvailable bottomed at 5.253 GiB, process swap stayed zero, system swap used 1.233 GiB, and peak memory PSI was 1.26%. The server remains healthy (`/health` 200).

The next decode lever is phase-specific policy: retain batched CPU DDR spill for prefill, but stage routed DDR misses into reserved VRAM slots during decode. A tested unit policy helper and `graph_cache_moe_cpu_cold_decode` setting are in the dirty worktree; v22 still used the compatibility default. The current loaded server is the old binary and remains running. The full-graph run checker has not passed because its shutdown summary is absent; do not call onboarding complete.


### DeepSeek v23 phase-specific DDR-to-VRAM decode staging (2026-11-04)
Source/worktree: `codex/deepseek-v41-continuation`, commit `c0e996355051a277086644dd40ffe995bd8da2df`, dirty snapshot. Binary SHA-256 `2dc333e8816ca0a4441910fe51eb6d9f793446543a9921a9f5738d07a98ee583`; GGML base `8baabd53b77d1c48c6ef8d3b620436292f3f6231`; patch manifest SHA-256 `5c4c17e7164237cf7da1570a5dcf1194fbd700df0ee3ca980bf96dcc4530f95b`; CUDA targets 86/120. Artifacts: `build_mtp_results/2026-11-04-c0e9963-deepseek41-fullgraph-prefill-decode-v23/`.

The new setting `graph_cache_moe_cpu_cold_decode=false` keeps CPU DDR spill active for prefill while allowing decode misses to stage asynchronously from resident DDR into VRAM. The guarded run again verified 15,360/15,360 expert identities (12,789 DDR + 2,571 VRAM; 32 dynamic slots). One 959-token/64-token request returned HTTP 200. Full-graph prefill took 61.792 s (15.52 tok/s) with CPU offload; full-graph decode completed 64 tokens in 30.851 s (2.07 tok/s, peak 2.21, p50 460 ms, p95 544 ms). This doubles v22's decode rate, but remains well below the 10 tok/s goal, and prefill is still far below 100 tok/s.

Decode logged 6,571 DDR misses staged to GPU, 15,120 held route slots, zero CPU spills, and about 11.24 s aggregate upload time; this indicates H2D/cache transfer remains a major lever. Engine counters reported zero real MoE disk reads; `/proc` read-bytes rose 0.923 GB during the request and 0.04 GB during decode, not attributed to MoE (Engram/table I/O may account for non-MoE reads). MemAvailable bottomed at 5.388 GiB, process swap stayed zero, system swap used peaked at 1.407 GiB, and PSI peaked at 4.86%. Server remained healthy (`/health` 200) after the request. It was later sent SIGTERM to rebuild for the next tuning pass; shutdown aborted in `DenseArenaOwner` with CUDA driver-shutting-down during teardown, but systemd reports no core file. The full-graph checker passed after shutdown summary was written (`python3 tools/check_full_graph_moe_run.py ... --full-fit`). On subsequent SIGTERM, the old binary aborted during `DenseArenaOwner` CUDA teardown; systemd recorded no core file. The next binary now resets DeepSeek-borrowed dense arena state during graph shutdown. Onboarding performance acceptance and both speed targets remain open.


### DeepSeek v24 async DDR-to-VRAM decode staging and clean shutdown (2026-11-04)
Source/worktree: `codex/deepseek-v41-continuation`, commit `c0e996355051a277086644dd40ffe995bd8da2df`, dirty snapshot. Binary SHA-256 `475ec22454ec73e4e689553dc5cb49f63bc496d111a6f41e1d8ccc135f592931`; GGML base `8baabd53b77d1c48c6ef8d3b620436292f3f6231`; patch manifest SHA-256 `5c4c17e7164237cf7da1570a5dcf1194fbd700df0ee3ca980bf96dcc4530f95b`; CUDA targets 86/120. Artifacts: `build_mtp_results/2026-11-04-c0e9963-deepseek41-fullgraph-prefill-decode-v24/`.

The v24 runtime used CPU cold DDR spill for prefill and disabled it for decode, staging DDR misses asynchronously into the VRAM cache. Strict residency passed: 15,360/15,360 experts (12,789 DDR + 2,571 VRAM; 32 dynamic slots). One 959-token/64-token request returned HTTP 200. Full-graph prefill completed at 15.40 tok/s, still below the >100 target. Full-graph decode completed 64 tokens at 2.25 tok/s steady (1.91 wall, p50 448 ms, p95 478 ms), improved slightly over v23's 2.07 but still below 10. Decode logged 6,283 DDR misses staged, zero CPU spills, 10.24 s aggregate asynchronous upload time, and 63 decode graph runs. The full-fit checker passed with zero layered S=1 forwards or legacy dispatches.

Host MemAvailable bottomed at 5.388 GiB; process swap peaked at 0.706 GiB, system swap at 2.657 GiB, PSI at 28.71%, and the memory guard blocked two allocations before recovering. Engine counters reported zero MoE disk reads; host `/proc` read-bytes rose 0.923 GB during the request and 0.04 GB during decode, not attributed to MoE (Engram/table reads may account for non-MoE I/O). The model server shut down with exit 0 and no core file after the DeepSeek shutdown path explicitly released the shared GLM dense arena before CUDA teardown.

The decode path no longer spills experts to CPU, but per-token GPU uploads/cache churn remain the main measured throughput lever; CPU-side full-graph prefill also remains far below target. No matched llama.cpp benchmark has been recorded, so the relative performance gate remains open.


### DeepSeek v25 persistent-pin hint A/B (2026-11-04)
Binary SHA-256 `9adbb80fbbfc9d6d34435db5c0a2357cb02efbb996a8beae2d76ff918b19ab94`; same GGML base/patch manifest as v24. Artifact: `build_mtp_results/2026-11-04-c0e9963-deepseek41-fullgraph-prefill-decode-v25/`. Full residency and full-graph checker passed. The only code change from v24 passed `MoEExpertBlock::slab_pinned` into `upload_expert_async` to skip redundant transient registration when the slab is already pinned. Decode was 2.14 tok/s wall / 2.22 steady versus v24 1.91 wall / 2.25 steady (no material improvement). Async registration plus deferred unregistration remained 10.24 s over 71.90 GB; 2,488/14,847 calls (16.8%) were already pinned. MemAvailable low-water 5.456 GiB, zero process swap, PSI peak 1.26%. Shutdown exited 0 without core. The pin-hint micro-optimization is safe but not the 10 tok/s lever; next work should address H2D transfer volume/latency or hotset/cache policy, and separately accelerate the prefill CPU path.


## DeepSeek v26 bounce-upload attempt and swap incident (2026-11-04)

Source worktree: `codex/deepseek-v41-continuation`, HEAD `de4d7f50c35949fbe026eb713fe89987a0040df3`, with an uncommitted opt-in DeepSeek async bounce-upload change. Binary SHA-256 `ccbb54bc3bd46b8f50b901d4dd9cf483746e25cbd116d8c14c14c32b5e045973`; GGML base `8baabd53b77d1c48c6ef8d3b620436292f3f6231`; patch manifest `5c4c17e7164237cf7da1570a5dcf1194fbd700df0ee3ca980bf96dcc4530f95b`. Artifact: `build_mtp_results/2026-11-04-de4d7f5-deepseek41-bounce-upload-ab-v26/`.

The guarded startup gate passed with MemAvailable 129.255 GB, both GPUs idle, CUDA device count 2, and 1.317 GiB system swap in use. Full residency loaded 15,360/15,360 experts. The 959-token full-graph prefill with CPU offload completed all 40 layers in 62.654 s (15.306 tok/s) and reported zero MoE disk reads. The first decode graph token failed before generating output; therefore v26 has no decode rate and the bounce-upload speed effect is inconclusive. The second request was not sent.

The host-process monitor later recorded MemAvailable 0 GiB, system swap 39.625 GiB, process VmSwap 28.005 GiB, and memory PSI avg10 about 98.6%/98.4%. The user reports that the machine froze and required a power reset. The server log's last application message was a CUDA invalid-resource-handle error on GPU 1, but this does not establish the cause of the whole-host freeze. The previous-boot kernel journal was not retained. No new model or GPU test should run until swap-aware live protection is added and the user authorizes another attempt.

The v26 launcher only gated initial MemAvailable and swap-out rate, and its monitor only recorded host memory. It did not stop when the model process or system began swapping. That guard is insufficient. The next safety work is to add a live watchdog that records and stops the run on process VmSwap growth, system swap growth, sustained memory PSI, or crossing the configured MemAvailable floor; keep ordinary HTTP/model failures separate so a failed prompt still leaves the server alive when host memory remains safe.


### Post-reset swap guard retrofit (not runtime-tested)

The v26 run’s guarded-start script required initial MemAvailable and a low swap-out rate, but its live monitor only recorded host memory and did not stop the server as swap grew. After the power reset, the v26 artifact was retrofitted with a stricter startup gate (system SwapUsed <=2 GiB, growth <=256 MiB, swap-out <1 MiB/s) and a live watchdog. The watchdog sends SIGTERM only for host-safety violations: configured MemAvailable floor, process VmSwap >0.25 GiB, system SwapUsed growth >0.50 GiB, PSI-full avg10 >5% for two samples, or swap-out >8 MiB/s for two samples. Ordinary HTTP/model failures still leave ZeroMoE alive when host memory is safe.

A synthetic check confirms the v25 low-swap sample passes and v26’s final monitor sample triggers the watchdog conditions. The script passes Python compilation and `bash -n`. This retrofit has not been run on hardware. No further GPU/model run has been started after the power reset; wait for user authorization before another attempt.


### Reusable swap watchdog (post-reset safety work)

The passive per-run monitor used in v26 is retained as historical evidence. A reusable replacement now lives at `tools/host_pressure_watchdog.py`; the saved guarded launcher calls it after startup. It records every second and sends SIGTERM only for explicit host-safety conditions: MemAvailable below the configured 4.5 GiB floor, ZeroMoE VmSwap above 0.25 GiB, system swap growth above 0.50 GiB from launch, PSI-full avg10 above 5% for two samples, or swap-out above 8 MiB/s for two samples. Ordinary HTTP/model failure alone leaves ZeroMoE up. Startup also includes the configured MoE request headroom and rejects system swap above 2 GiB, growth above 256 MiB, or swap-out >=1 MiB/s. Four synthetic policy tests pass; the watchdog has not been run on hardware. No new model run has been started after the power reset.


### Bounce-upload experiment disabled after failed decode

The v26 opt-in bounce-upload path was reverted from the active source after its only guarded request failed on the first decode token and produced no decode measurement. The v26 config and logs remain as immutable run evidence; do not rerun that artifact. The active worktree now matches the committed v25 engine source for inference, plus the separately tested host swap watchdog utility. No GPU/model test has been run after the power reset.

### Post-reset active state (2026-11-04)

The unverified bounce-upload change was removed from the active inference source; the failed v26 config/logs remain in its artifact directory. The active DeepSeek engine sources now match commit `de4d7f50c35949fbe026eb713fe89987a0040df3`. A fresh non-inference build passed provenance (binary SHA-256 `a1e422990b50299d136af83fbf39fe260fd667321041419582014882fd3fb556`). `graph_settings`, `graph_settings_binding`, and `vram_dynamic_slot_plan` pass (3/3), and the new swap-watchdog synthetic tests pass (4/4). The v26 host watchdog is now backed by reusable `tools/host_pressure_watchdog.py`; no hardware rerun has occurred after the power reset.


### v26 swap-threshold chronology from the passive monitor

The saved 2 s monitor samples first show MemAvailable below the configured 4.5 GiB floor at `2026-11-04T05:43:39Z` (2.917 GiB), four seconds after `[PREFILL-DONE]` at `05:43:35Z`. PSI-full avg10 crossed 5% at `05:43:41Z`; ZeroMoE VmSwap exceeded 0.25 GiB and system SwapUsed growth exceeded 0.5 GiB at `05:43:43Z`. The watchdog would trigger on the first low-MemAvailable sample, before the recorded swap growth; this is a replay of saved samples, not a hardware intervention. Two chronology-focused synthetic tests were added, so the watchdog suite is now 6/6. Hardware validation remains unapproved and unperformed.


### DeepSeek next cache-capacity A/B prepared (not launched)

A v27 full-residency candidate is prepared at `build_mtp_results/2026-11-04-7f76290-deepseek41-dynamic64-swapguard-v27/`. Its only runtime change from v25 is raising the dynamic VRAM staging reserve from 32 to 64 slots; bounce upload stays disabled. The expected DDR increase is about 0.26 GiB, with exact placement checked by the runtime planner. The saved guarded launcher includes the reusable live swap watchdog and two sequential 959-token/64-token requests; it sends the second only after HTTP 200 for the first.

Current host MemAvailable is 125.973 GB with about 0.002 GiB swap used; this candidate's startup gate requires 127.945 GB including host cap, startup margin, and request headroom, so it will not start in the current state. No server or GPU process is running. No rerun has started after the power reset; wait for user authorization.

The v27 64-slot reserve also passes the offline placement-planner test: 34 dynamic slots on the 3090, 30 on the 5060 Ti, with 2,571 static-resident slots retained. The runtime DDR/VRAM placement remains unverified. Current MemAvailable (125.95 GB) is below the guarded candidate’s 127.945 GB startup requirement, so it is not launchable now; the saved launcher would wait. No hardware load has been started.


### v26 decode-handle diagnosis and corrected opt-in path (2026-11-04; runtime unverified)

Static review found a concrete candidate for v26’s first decode-token invalid-resource-handle error: GpuTopology had one bounce-buffer completion event per host buffer, created once in whichever CUDA device context was current, then recorded against copy streams from both GPUs. NVIDIA’s multi-device guide says cudaEventRecord fails when the event and stream belong to different devices ([CUDA Programming Guide, multi-device stream/event behavior](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html#multi-device-stream-event-and-memory-copy-behavior)). This candidate explains the logged first-token decode failure; it does not explain the separate later host freeze, whose monitor evidence is swap exhaustion.

The opt-in path now creates a completion event per device per portable bounce buffer, waits all device events before reusing a buffer, checks copy size against buffer capacity, and drains in-flight events at teardown. The cache_moe_async_bounce_upload setting defaults off. CPU-only compile and tests pass, including the event-device mapping/capacity checks. There has been no hardware test; candidate artifact is build_mtp_results/2026-11-04-b43915e-deepseek41-bounce-device-events-v28/.


## DeepSeek V4.1 Flash full-graph v30 (2026-11-04; locally measured)

Owner/worktree: `codex/deepseek-v41-continuation`; the measured run used runtime source commit `2b5a16503f9dd17842051052b1c27365e2579b0b`. Current HEAD is `9629a7cc4fbe8fe2e4c71f74c8ea7434a0043ab2` after the watchdog-only code commit and this handoff update; no inference source changed after the run. Binary SHA256 `2c5ec2eea13bb182ad4174658ee2204e1d414db56ba2f1296bafe594325dbc69`; GGML base `8baabd53b77d1c48c6ef8d3b620436292f3f6231`; GGML patch manifest `5c4c17e7164237cf7da1570a5dcf1194fbd700df0ee3ca980bf96dcc4530f95b`. Toolchain CUDA 13.3, GCC 16.2.1, driver 610.57.04, native targets `sm_86;sm_120`. Dual GPUs: RTX 3090 + RTX 5060 Ti, `CUDA_VISIBLE_DEVICES=0,1`, OMP threads 10.

Input layout was confirmed after user correction: numbered main GGUF shards 1–5 plus `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` as logical component 6. Main GGUF reported 1,042 tensors; Engram sidecar reported 2 tables. No GGUF data was copied.

Strict full residency passed: 15,360/15,360 experts, 12,821 DDR (110.141 GB), 2,539 VRAM, and 64 dynamic staging slots (34 on GPU0, 30 on GPU1). VRAM pools were 16.950 GB on the 3090 and 15.096 GB on the 5060 Ti; 5.705 GB dense tensors were on the 3090. The full-fit graph checker passed.

Locally measured first request: 959 prompt tokens, 64 generated tokens, HTTP 200, 78.690 s total. Full-graph CPU-offload prefill: 39.335 s, 24.38 tok/s. Full-graph decode: 2.37 steady tok/s, 2.30 wall tok/s, 27.822 s decode wall; p50 427.32 ms and p95 461.546 ms/token. These do not meet the working targets of 100 tok/s prefill and 10 tok/s decode. A matched llama.cpp benchmark was not available; this is not relative-performance acceptance.

The routed decode cache recorded 6,630 DDR-to-VRAM misses of 15,120 top-6 route slots (56.15% VRAM hits, 43.85% misses). The layer-miss report found zero real SSD fetches across 2,503 decode calls. The generic `Expert cache: 17032 hit / 0 miss` counter is not this routed VRAM hit rate. Per-layer hit rate ranged from 25.9% (layer 38) to 72.2% (layer 29). An offline approximation from the frozen five-prompt survey estimated 77.3% coverage for 2,539 resident entries; compact per-layer caps and tie ordering were not modeled, so treat that estimate as provisional. The 21-point difference from the measured single-prompt route hit rate warrants route-distribution tracing before changing the hot set.

The sequential repeat was refused before tokenization (HTTP 500): the host request envelope needed about 0.558 GB while free memory was at the 4.5 GiB engine floor. MemAvailable later reached 4.483 GiB; process VmSwap stayed 0, system SwapUsed stayed 0.002 GiB, and swap-out/PSI remained zero. The 4.5 GiB watchdog sent SIGTERM and escalated after two seconds; no ZeroMoE process remains. No request was retried. The user then requested a separate 1 GiB watchdog emergency floor; code commit `187d751` adds `host_pressure_watchdog_floor_mb`, leaving the engine's 4.5 GiB request floor intact. v31 was prepared but not launched. Follow-up v32 supersedes it: v30 measured 64 active-layer slots allocated (0.787692 GB) with zero uploads/hits, so v32 disables that unused pool and moves the same bytes into the main MoE pools (3090: 17.540769 GB, 5060 Ti: 15.292438 GB), keeping 64 dynamic slots and total requested VRAM unchanged. v32 is prepared but unlaunched at `build_mtp_results/2026-11-04-c07a0d0-deepseek41-reuse-inactive-active-vram-v32/`; the runtime planner must confirm exact placement.

Next action: do not reload or resend requests until the user instructs. Use v35, whose admission gate aborts immediately on swap-out or >128 MiB swap growth. If it passes, confirm the strict full-residency plan, run the long request, and inspect route-usage output before changing the hot set. Full run artifacts are under `build_mtp_results/2026-11-04-2b5a165-deepseek41-bounce-dynamic64-v30/`.


### v34 admission-gate freeze incident (2026-11-04; no model started)

After the system rebooted, v34's saved gate log showed no `wait-gate-result.txt`, no `server.log`, and no ZeroMoE PID/port: the launcher never passed admission and did not load the model. The last recorded sample had MemAvailable 26.188 GB versus 127.945 GB required, SwapUsed 1.175 GiB from a 0.893 GiB baseline (+0.282 GiB), and swap-out 62 MiB/s; GPU/CUDA probes still returned two idle devices. Previous-boot kernel logs were unavailable, so the exact freeze cause cannot be proven. The gate's failure was continuing to poll after it observed active swap pressure.

### v35 fail-closed admission gate (prepared; not launched)

The startup gate now aborts immediately on total SwapUsed above 2 GiB, growth above 128 MiB from its baseline, or any `pswpout` delta, recording the sample and reason before exiting. It retains the stable MemAvailable/GPU/CUDA requirements and the v32/v34 placement plan. A replay of v34's final saved sample triggers the abort before model launch. This is an offline policy check, not a hardware run. Candidate: `build_mtp_results/2026-11-04-2530fbc-deepseek41-active-reuse-host123-failclosedgate-v35/`.
