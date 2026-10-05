# Graph refactor — final handoff, 2026-10-05

The reviewed graph/scheduler work and staged A/B sequence are complete for the
Q2_0 Option-B 0/44/4 profile on the RTX 3090. The graph work is committed
through `490380a`; unrelated ongoing work in the shared tree remains outside
those commits. All generated tests, logs, captured histories, and discarded
attempts are preserved.

## Selected design

Keep the existing static expert placement: 8,421 hot experts, 10.84 GiB, with
layers 0, 1, 2 and 22 on CPU. The scheduler change consists of:

1. A persistent private CPU cold-expert graph per split layer. The main thread
   starts it after that layer's input-ready event, while queued hot/shared GPU
   work continues. It avoids running a nested graph inside CPU worker threads.
2. Immutable routing/remap tables uploaded to owned GPU buffers once, removing
   repeated scheduler input transfers of constants.
3. Owned pinned cold outputs and GPU output buffers. A nonblocking upload
   stream records completion; the model stream waits on that event before the
   merge. Whole-token synchronization protects reuse. Arithmetic and selected
   experts are unchanged.

The graph and buffer addresses remain stable during decode. GGML owns GPU
capture/replay; this path does not reconstruct the graph per token. The hook
is opt-in and requires the patched integration fork, so global defaults remain
off. GDS/cuFile is excluded and the BF16 dispatch experiment remains deferred.

The later staging refactor uses one shared worker service while retaining
independent per-layer GPU weight arenas and epoch mailboxes. Sharing one
shape-compatible arena would let a later upload overwrite weights still in
use by an earlier layer in the captured graph. The storage completion service
and adaptive timing policy are foundations for a future partial-layer path;
they are not enabled by the production profile.

The staged route prototype also has an explicit copy-stream completion join and
uses valid masked dummy IDs for inactive GPU slots. A no-request control is
byte-identical, but real layer-13 staging still changes the later hidden states
through a small CPU-versus-GPU quantization arithmetic difference. Its verifier
compares against a GPU reference and is therefore insufficient for production
parity; the path remains rejected.

## Final results and evidence

Six balanced permutations, ten CPU threads, identical forced 64-token history,
same binary/config/placement, no tracing or verification in timing:

| Mode | Decode tok/s | Change |
| --- | ---: | ---: |
| Original graph/scheduler | 31.848 | baseline |
| Early-ready + resident tables | 33.294 | +4.538%, 6/6 wins |
| Above + async output | 33.874 | +6.361% overall; +1.744% incremental, 6/6 wins |

Timing: `logs/option_b_scheduler_final_three_arm/summary.json`. Admission
version 2 watches compiler/test/benchmark processes and compute processes on
all GPUs; contaminated attempts were excluded and preserved before retry.
All three modes produced 64 byte-identical raw logit records. Numerical
evidence: `logs/option_b_scheduler_final_three_arm_logits/comparison.json`
and `third_comparison.json`; those dump timings are excluded.

Final current-worktree native check passed:
`CUDA_VISIBLE_DEVICES=0 make test BUILD_DIR=../moestream-build-upstream-master JOBS=4`.
Log: `logs/option_b_scheduler_final_native.log`. Shell/Python syntax checks and
`git diff --check` passed too. Do not generalize these throughput figures to
other quantizations, layouts, prompt distributions or GPU configurations.

## Rejected experiments, retained

- All-hot CPU skip: +0.029%, 2/6 wins; CPU savings were hidden under GPU work.
- Async output alone: +1.356%, 6/6 wins; below the 1.5% bar in that earlier
  configuration. It meets the bar on the resident-table base in the final
  contemporaneous comparison, so the selected bundle includes it.
- Predictor alone: −0.049%, 3/6 wins on the neural-network history.
- Layer-13 staging: −0.432%, 2/6 wins.
- All 42 eligible target-layer staging: −12.636%, 0/6 wins. Leave forecasting
  and dynamic staging off in the selected runtime profile.

Staging correctness passed independent GPU comparisons against original DDR
expert weights for all 42 layers, including topical code/creative histories.
The strict 100-ms worker-delay test produced 64 byte-identical fallback
records with zero staged routes. A separate GPU-delay test let two transfers
arrive before use because its delay kernels slowed the model; retain that as
contention evidence, not an all-fallback proof.

Code/creative/code free-running requests completed in each mode in one engine
lifetime. Code outputs matched across modes and before/after the creative
request. The creative continuations differed coherently and reached the
512-token cap. Staging passed 1,050 reference diagnostic batches across that
sequence. The bounded code trace included post-EOS continuation; retained
natural-prefix summaries exclude it and show full top-choice agreement for
the actual answer. The old exact-identifier-copy score remains unsuitable.

Artifacts are under `logs/option_b_topics_stage_all_verified/`,
`logs/option_b_multi_request_free/`, `logs/option_b_stage_all_worker_late/`,
and the individual `option_b_step*` directories. First allocation failures
and all interrupted attempts are retained. The worklog records their causes
and corrections, including CUDA quantized-buffer padding, separate prediction
and selected-route pins, worker mailbox lifetimes, shared verifier workspace,
and bank holds through DMA and compute.

## Run the selected profile

From the repository root, in a shell without benchmark-only overrides:

```bash
source configs/qwen4exp_q2_0_option_b_scheduler.env
../moestream-build-upstream-master/moestream \
  --config configs/qwen4exp_q2_0_option_b_rightsize_split44_x0_cov80.toml
```

The integration checkout is `../llama.cpp-moestream-integration`. The saved
scheduler patch is `patches/ggml-scheduler-early-ready-layer13.patch`; despite
its historical filename, the application hook supports all eligible split
layers. Existing CUDA CPY bridge support in that fork is also required.

## Files and ownership

Core: `include/qwen4exp_early_ready.hpp`, `include/qwen4exp_stage_pool.hpp`,
`src/ggml_qwen4exp_graph.cpp`, and `cuda/qwen4exp_master_graph.cu`.
Only the tiny-cache budget change in Test 23 of `src/main_test.cpp` belongs to
this scheduler effort; preserve its other edits. The Qwen graph source also
contains the earlier global-pool experiment; that mode is not selected here.

Drivers: `bench/option_b_early_ready_check.py`,
`bench/analyze_qwen_route_prediction.py`, `bench/option_b_topic_check.py`,
and `bench/option_b_request_check.py`. Runtime overlay:
`configs/qwen4exp_q2_0_option_b_scheduler.env`. Documentation: the short README
link, this handoff, and `docs/option-b-graph-refactor-steps-2026-10-04.md`.

The native rebuild refreshed the binary after unrelated shared-tree changes.
The additional current-binary check in `logs/option_b_scheduler_post_native_logits/`
passed: all three modes' 64 raw records match each other and their pre-native
records byte for byte. CPU compilation from another session made this dump
run very slow; its timings are invalid and do not supersede the guarded
performance batch. The current binary's hash matches that parity manifest.
No graph-refactor acceptance gate remains pending. Separate future work may
revisit other models/hardware or speculative staging with a different memory
and scheduling design. Do not silently enable the rejected experiments or
reuse dump/reference timings as performance evidence.
