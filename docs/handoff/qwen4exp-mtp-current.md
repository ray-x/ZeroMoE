# Qwen4Exp MTP shared handoff

Updated 2026-10-27. Refresh the current HEAD and worktree list before using any
binary or citing a result.

## Master status (2026-10-27)

`master` is at `19100ff`. Five changes landed after the `b78dd9b` binary that
produced the matched raw-story numbers below, and none is a runtime change on
the CUDA engine:

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

`build/zeromoe` is still stamped from `b78dd9b` and fails
`python3 tools/build_provenance.py check --binary build/zeromoe --repo .`
against current master. Do not quote new rates from it; rebuild first. The
validation build in `build_verify_master/` is current (CPU_ONLY=OFF, arch 86,
no `ZEROMOE_USE_GGML_CUDA` input); its manifest reports `source=19100ff
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
  `/home/ray/vllm-workspace/qwen3.8-flash-next/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf`.
- Config: context 1536, max_new_tokens 800, temperature 0, top-p 0.95,
  global Option-B K=13,000, host_ram_gb 80 GiB, 10 OpenMP threads, frozen
  usage files, three draft tokens.
- GPU: RTX 3090 only (`CUDA_VISIBLE_DEVICES=0`). CPU: Xeon W-2235. Toolkit /
  driver: CUDA 13.3.73 / 610.57.04.
- Required environment: `ZEROMOE_QWEN4EXP_FULL_GRAPH=1`,
  `ZEROMOE_QWEN4EXP_FULL_GRAPH_DECODE=1`,
  `ZEROMOE_QWEN4EXP_GRAPH_PRELOAD_LAYERS=0`,
  `ZEROMOE_QWEN4EXP_GRAPH_PRELOAD_TRAILING=1`, and
  `ZEROMOE_MTP_PROFILE=1` for the MTP run. Keep
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
