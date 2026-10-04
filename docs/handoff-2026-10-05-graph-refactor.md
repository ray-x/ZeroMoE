# Graph refactor handoff — 2026-10-05

**Historical checkpoint:** this file records the interrupted work-in-progress
state. The completed decisions and verification are in
[the final handoff](handoff-2026-10-05-graph-refactor-complete.md).

The active objective remains “refact the graph.” Work is experimental and
uncommitted. Continue from this worktree and preserve existing generated logs,
test files, and unrelated Exl3 changes. GDS/cuFile is unsupported on this box
and out of scope. BF16 routing experiments remain disabled.

Resume update: host GPU access was rechecked and is available. The updated
all-layer run `logs/option_b_step4_stage_all_verified_v2/` succeeded in
full-graph mode across 42 target layers. Its 32-visit diagnostics record 1,344
independent reference comparisons, all with zero maximum delta; the full run
passed subsequent assertions as well. Model-logit top-1 agreement was 62/64,
mean KL 0.0008802. This supersedes the pending-GPU status below, but not the
remaining quality and throughput gates.

## Decisions and acceptance

- Keep static Option-B placement as the baseline. It moves no work at runtime.
- Use the opt-in early-ready scheduler hook to overlap CPU cold-expert work
  with already-submitted GPU work. The external llama.cpp integration fork is
  `../llama.cpp-moestream-integration`; its scheduler patch is
  `patches/ggml-scheduler-early-ready-layer13.patch`.
- Do not use CUDA conditional graphs for mixed hot/cold routing: their
  conditional bodies cannot contain host work, and an all-or-nothing test
  cannot execute a partial hot set correctly. `cudaLaunchHostFunc` is not
  appropriate inside that conditional body either.
- Async cold-output copy is promising but needs combined numerical parity and
  a repeat timing batch. Dynamic staging stays experimental until it passes
  whole-graph allocation, staged-output reference, late fallback, multi-request
  lifetime, topical-prompt, and timing checks.
- The established throughput bar is +1.5% average over six order-balanced
  pairs. Correctness dumps and tracing are excluded from timing.

## Measurements already established

- All-hot CPU skip: +0.029%, 2/6 wins. Reject; it saves CPU time already hidden
  behind GPU work. Details: `logs/option_b_step1_all_hot_perf/summary.json`.
- Async output copy alone: +1.356%, 6/6 wins. It is just under the bar and
  remains opt-in. Details: `logs/option_b_step2_async_output_perf/summary.json`.
- Immutable resident route tables: +2.505%, 6/6 wins. Accepted experimental
  base. Details: `logs/option_b_step2b_resident_tables_perf/summary.json`.
- Next-layer predictor overhead: −0.049%, 3/6 wins, effectively flat on the
  tested neural-network prompt. It still needs code and creative histories.
  Predictor recall on that history was 62.72%, precision 53.87%, full mixed
  coverage 35.84%; transfer-byte figures are lower bounds, not achieved
  deadlines. See `logs/option_b_step3_prediction_logits/prediction_scores.json`.
- Layer-13 dynamic staging: −0.432%, 2/6 wins; do not adopt. Independent GPU
  reference outputs matched exactly on 32 checks. Artificial 100-ms DMA delay
  preserved all 64 baseline logits by falling back to CPU. See
  `logs/option_b_step4_stage13_perf/summary.json`,
  `logs/option_b_step4_stage13_verified/`, and
  `logs/option_b_step4_stage13_late/`.
- Async output copy on the resident-table base: +1.820%, 6/6 wins (pairs
  1.387, 0.977, 1.639, 1.326, 3.879, 1.747%). This batch clears the bar, but
  one contaminated attempt was excluded and retried. It still needs an
  identical-history raw-logit comparison before adoption. Results:
  `logs/option_b_resident_async_combined_perf/summary.json`.

## Latest all-layer staging attempt

`logs/option_b_step4_stage_all_verified/` is invalid evidence. Candidate
initialization successfully created pools across the 42 eligible target
layers, but the GPU had only 0.73 GiB free when the graph allocator checked
its existing 1-GiB safety floor. The graph declined and inference fell back
to layered decode. Compilation from the concurrent Exl3 work also invalidated
the attempted timing. Preserve the entire directory; do not count its
baseline timing or its candidate as an A/B result.

Two fixes were applied after that attempt:

1. Reference-output verification now shares one GPU reference graph and weight
   workspace between compatible layer pools, with shape/type/device checks.
   Each layer still keeps its own actual output snapshot. This reduces
   diagnostic VRAM use from one reference workspace per layer to one total.
2. `Qwen4ExpStagePool::token_complete` waits for the worker to finish reading
   the current pinned prediction mailbox before the next replay may overwrite
   it. This waits for the small host-side forecast read only; expert DMA stays
   asynchronous and its bank remains held through transfer and consumer use.

The updated engine build succeeded:
`logs/option_b_stage_mailbox_shared_reference_build.log` (`[100%] Built target
moestream`). Python syntax checks succeeded for
`bench/option_b_topic_check.py` and `bench/analyze_qwen_route_prediction.py`.
The updated binary has not yet run on the GPU. The last observed environment
check reported that `nvidia-smi` could not communicate with the NVIDIA driver;
recheck availability before launching inference. Do not weaken the graph
allocator's 1-GiB safety floor to force the test through.

## Next actions

1. When the GPU is available and the host idle, run the 64-token baseline vs
   all-eligible-layer candidate with `STAGE_VERIFY=1`. Keep resident tables and
   the predictor enabled in both arms. Confirm full-graph mode is active in
   both logs, there are no allocator refusals, and every actually staged
   output's verifier delta is zero. If memory still fails, measure and report
   pool and graph-reserve costs, then bound staging residency without changing
   the safety floor.
2. Rebuild/run native `make test BUILD_DIR=../moestream-build-upstream-master
   JOBS=4` with `CUDA_VISIBLE_DEVICES=0`; retain its log.
3. Run the same-binary 64-record raw-logit comparison for resident tables plus
   async output against resident tables alone. Use `--allow-busy-correctness`
   only for this dump run and mark its timing invalid.
4. Run `bench/option_b_topic_check.py` for code and creative prompts. Capture
   predictor-only and staged runs, check route accuracy and actual staged IDs,
   then compare free-running output quality. The old exact-identifier copy
   score was ~5% for every configuration and is not a useful quality test.
5. Only after full-graph correctness, parity, and late-transfer fallback pass,
   measure staging over all eligible layers with six guarded pairs. Compare
   equal VRAM policies and keep predictor/branch overhead in the candidate.
   Reconfirm async-output timing with a fresh six-pair run before adopting it.
6. Audit graph teardown, repeated requests/epoch reset, default-off flags,
   docs and test coverage. No changes are committed.

## Files added or edited for this effort

- `include/qwen4exp_early_ready.hpp`
- `include/qwen4exp_stage_pool.hpp`
- `src/ggml_qwen4exp_graph.cpp` (contains other pre-existing work; preserve it)
- `cuda/qwen4exp_master_graph.cu`
- `src/main_test.cpp` (only the localized small-buffer cache-budget fix here
  belongs to this graph work; preserve the other edits)
- `bench/option_b_early_ready_check.py`
- `bench/analyze_qwen_route_prediction.py`
- `bench/option_b_topic_check.py`
- `docs/option-b-graph-refactor-steps-2026-10-04.md`
- `docs/option-b-scheduler-overlap-design-2026-10-04.md`
- `docs/handoff-2026-10-04-option-b-early-ready.md`
- `patches/ggml-scheduler-early-ready-layer13.patch` in the main checkout;
  applied to the sibling integration checkout.

Preserve unrelated working-tree changes, especially the concurrent Exl3
implementation, configuration, and test edits.
