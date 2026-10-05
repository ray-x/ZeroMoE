# Qwen graph refactor: recorded state

This file records the committed graph work and the remaining unrelated
ExLlamaV3, CPU, configuration, test and generated-artifact changes in the
shared worktree. Graph commits are authored by `ray-x`; the latest foundation
commit is `490380a` (`Add graph tier telemetry and adaptive policy foundation`).

## Selected runtime

The selected Q2_0 RTX 3090 profile is Option-B 0/44/4:

- 8,421 usage-selected experts remain resident in 10.84 GiB of VRAM.
- Layers 0, 1, 2 and 22 use the plain CPU path.
- Ten CPU threads are used while other host work may be present.
- BF16 MMVF routing remains disabled for this A/B.
- SSD/GDS integration and dynamic staged-GPU execution are disabled.

Source `configs/qwen4exp_q2_0_option_b_scheduler.env` together with
`configs/qwen4exp_q2_0_option_b_rightsize_split44_x0_cov80.toml` when using the
patched `../moestream-build-upstream-master` binary.

## Graph changes in this commit

The early-ready scheduler path gives each split layer a private CPU cold-expert
graph. An input-ready event starts that graph from the scheduler integration
hook while already-submitted GPU work proceeds. The path keeps stable private
input/output buffers and does not reconstruct the main graph per token.

Resident remap and mask tables are uploaded once into owned GPU buffers. The
CPU graph retains its host tables, while the main GPU graph reads immutable
device copies. Cold output can use an owned pinned host buffer and an upload
stream joined by an event before the merge.

The experimental epoch-tagged staging pool is retained behind flags for
correctness research. It publishes all three quantized planes only after their
copies complete, validates the epoch and expert ID on the GPU, holds banks
through the consuming token, and falls back to CPU when publication is late.
It is not enabled by the selected profile.

The CUDA conditional graph remains an optional pure-GPU all-or-nothing path.
Conditional bodies contain no host callbacks. The current early-ready path is
the accepted scheduler design; the conditional path is not combined with a
host callback in one child graph.

## Measurements

The final six-permutation, order-balanced decode comparison used the same
binary, model configuration, forced 64-token history and ten CPU threads:

| Mode | tok/s | Result |
| --- | ---: | --- |
| Original graph/scheduler | 31.848 | baseline |
| Early-ready + resident tables | 33.294 | +4.538%, 6/6 wins |
| Above + asynchronous output | 33.874 | +6.361% overall, +1.744% incremental, 6/6 wins |

Evidence is in `logs/option_b_scheduler_final_three_arm/summary.json`.
The current-binary follow-up in `logs/option_b_scheduler_post_native_logits/`
and the original parity run in
`logs/option_b_scheduler_final_three_arm_logits/` contain byte-identical
64-record logits for all three modes, with zero KL, TV and maximum delta.

To reproduce the three-arm measurement, first verify the Q2_0 directory, make
sure the host and GPU are idle, then run:

```bash
python3 bench/option_b_early_ready_check.py \
  --engine ../moestream-build-upstream-master/moestream \
  --out-dir logs/option_b_scheduler_repro \
  --prompt-ids logs/quant_decode_compare_chat_20261004/prompt.tokens \
  --continuation logs/quant_decode_compare_chat_20261004/Q2.tokens \
  --reps 6 --all-split --wait-idle 300 \
  --candidate-env MOESTREAM_QWEN_EARLY_READY_RESIDENT_TABLES=1 \
  --third-env MOESTREAM_QWEN_EARLY_READY_RESIDENT_TABLES=1 \
  --third-env MOESTREAM_QWEN_EARLY_READY_ASYNC_OUTPUT=1
```

The three arms are the original scheduler, early-ready plus resident tables,
and that bundle plus asynchronous output. The script records admission state,
rejects compiler/inference interference, and writes `summary.json` with the
paired gains. For a normal run rather than an A/B, source the runtime overlay
and launch the configured engine:

```bash
source configs/qwen4exp_q2_0_option_b_scheduler.env
../moestream-build-upstream-master/moestream \
  --config configs/qwen4exp_q2_0_option_b_rightsize_split44_x0_cov80.toml
```

The rejected experiments remain recorded:

- all-hot CPU skip: +0.029%, 2/6 wins;
- predictor alone: −0.049%, 3/6 wins;
- Layer-13 staging: −0.432%, 2/6 wins;
- all-layer staging: −12.636%, 0/6 wins.

The all-layer verifier nevertheless passed 1,344 independent staged-output
checks at the diagnostic boundary, with maximum delta zero. The strict worker
delay test produced byte-identical fallback logits and zero staged routes.
Three natural-stop requests (code, creative, code) completed in one process;
the first and third code answers matched before and after the creative request.

The native suite passed with:

```text
CUDA_VISIBLE_DEVICES=0 make test BUILD_DIR=../moestream-build-upstream-master JOBS=4
```

The retained log is `logs/option_b_scheduler_final_native.log`.

## Milestone 1: immutable route classification

The first follow-up improvement adds `moe_partial::RouteClassification` to
`include/moe_partial_route.hpp`. It creates an immutable assignment for each
original router slot, carrying the expert ID, original slot, router weight,
execution target, branch-local weight slot and slot generation. Resident,
staged and CPU ownership are mutually exclusive; a CPU assignment is the
explicit fallback when no published staged slot exists. Duplicate expert IDs
remain separate route slots, so their router weights cannot be accidentally
collapsed.

The classifier currently sits beside the graph code and is not yet wired into
the accepted decode path. That keeps the production baseline unchanged while
the exact-once contract is tested before adding conditional or host-node
branches.

The focused test covers all-hot, all-cold, mixed, duplicate-route, overlapping
ownership, invalid IDs, generation propagation and both negative-sentinel and
dummy-route modes:

```bash
g++ -std=c++23 -O2 -Iinclude tests/test_moe_partial_route.cpp \
  -o /tmp/test_moe_partial_route_route_classification
for mode in capable stock optout; do
  /tmp/test_moe_partial_route_route_classification "$mode"
done
```

All three modes passed. The next milestone can consume this descriptor from a
DDR-only partial-layer graph without mutating shared remap or mask tables.

## Milestone 2: DDR-only partial-layer graph prototype

An opt-in prototype now uses `MOESTREAM_QWEN4EXP_PARTIAL_GRAPH=1` together
with `MOESTREAM_QWEN4EXP_PARTIAL_GRAPH_LAYER=13`. It keeps the static resident
CUDA branch and inserts a bounded `cudaLaunchHostFunc` callback for the DDR
complement. The callback uses a prepared GGML CPU graph with immutable CPU
route tables, then uploads a per-route weighted cold result for the graph
merge. It does not read SSD, use speculative staging, or alter the accepted
early-ready path.

The first parity comparison against the original static split path produced
63/64 top-1 agreement and mean KL 0.00129. This is diagnostic evidence only;
the path is not accepted as numerically identical yet. The guarded six-pair
comparison against the accepted early-ready/resident-table/async-output bundle
measured 33.451 -> 26.526 tok/s (-20.70%, 0/6 wins):
`logs/option_b_partial_layer13_ddr_perf/summary.json`. The callback is
serialized on the graph stream and currently adds too much latency.

The existing Phase-2 prefix/suffix graph can provide the pure-CUDA conditional
body while the suffix CPU graph carries the host callback. A top-256 proof
constructed all three graph instances but declined the extra 11.17-GiB expert
arena; a top-64 retry still declined a 10.93-GiB arena. Preserve
`logs/option_b_partial_phase2_layer13_logits/` and
`logs/option_b_partial_phase2_layer13_top64_logits/`. Resource sharing must be
fixed before treating the conditional proof as runnable. Prefix construction
now prebinds the compact expert planes into its owner graph so suffixes can
reuse stable addresses; the remaining failure is allocator admission before
that borrowed ownership is reached. The prototype remains
opt-in and the production profile is unchanged.

## Milestone 4 foundation: storage dispatch/completion boundary

`include/qwen4exp_storage_service.hpp` adds a host-only bounded service with
immutable `StorageRequest` records and explicit `Ready`/`Error` completions.
Every completion carries both `model_generation` and `request_generation`; a
stale handle cannot consume a later request's result. A callback may submit a
request and return immediately. The worker owns the DDR/SSD loader operation,
and a later completion graph can consume the matching result. The service does
not call CUDA or wait for I/O from a CUDA host callback.

The focused test is `tests/test_qwen4exp_storage_service.cpp`:

```bash
g++ -std=c++23 -O2 -pthread -Iinclude \
  tests/test_qwen4exp_storage_service.cpp -o /tmp/test_qwen4exp_storage_service
/tmp/test_qwen4exp_storage_service
```

It passed ready, error, stale-generation and release cases. Runtime SSD
loading and completion-graph submission are still separate work; the current
production profile does not use this service yet.

## Milestone 5: shared staging worker service

`Qwen4ExpStageWorkService` owns one bounded worker queue for all staged layers.
`FullGraph` creates one service and every stage pool submits bank jobs to it;
each layer retains independent CUDA weights, metadata, copy stream and
double-buffered bounce storage. The independent GPU storage is deliberate:
sharing one shape-compatible weight arena across layers would let a later
upload overwrite weights still consumed by an earlier layer in the same graph.
The shared queue removes the per-layer worker-thread overhead without creating
that lifetime race. The production profile still keeps staging disabled.

The all-layer verifier remains the required next runtime check. The Python A/B
harness currently cannot be run from this restricted session because a CUDA
process spawned by Python cannot see the driver, while the same binary launched
directly from a shell does; this is an execution-environment limitation rather
than a graph result. No staging speed claim is made from that failed attempt.

A later escalated one-repetition run did reach the GPU. The shared worker
reported `verified=32` and `max_delta=0` for every staged layer, but the full
token logits were not parity-safe: 96.875% top-1 agreement, mean KL 0.000967,
and 33.15 -> 22.07 tok/s. This is retained in
`logs/option_b_stage_all_shared_worker_verified_v5/` as a rejected staging
candidate; it does not change the production baseline.

The isolated layer-13 diagnosis is now complete. With stage requests disabled,
the prediction graph and dummy staged branch are byte-identical to the
no-staging control. With real requests, the first selected staged route changes
the layer-13 snapshot by only about 3.7e-9, but later layers amplify that
quantization-path difference; the one-repetition run reached 98.4375% top-1
agreement and mean KL 0.000589. The staged verifier still reports zero delta
because it compares GPU staging with a GPU reference, not with the CPU cold
branch used by the accepted scheduler. The evidence is in
`logs/option_b_stage_layer13_alllayers_v1/` and
`logs/option_b_stage_layer13_no_request_v1/`. This path is therefore rejected
for production until its CPU/GPU arithmetic contract is resolved.

## Milestone 6: storage telemetry and deferred adaptive policy

`StorageService` now publishes queue and service durations with every
`Ready`/`Error` completion and exposes aggregate submitted/rejected/ready/error
counters. The model and request generations remain part of the completion, so
an SSD/DDR completion graph can reject a stale result before it touches a
staging slot. Loader exceptions are converted to an explicit error completion.
The service remains host-only: a CUDA host callback may enqueue an immutable
request, but the worker does not call CUDA or wait for a graph operation.

`include/qwen4exp_adaptive_policy.hpp` contains the later decision rule and is
not wired into the default scheduler. It chooses staged GPU only when its
measured transfer queue, H2D, GPU compute and (when needed) storage time is
strictly below measured CPU queue plus CPU compute time. A pending or invalid
staging/storage estimate falls back to CPU. This policy is covered by a
focused test and must be connected to real stage/storage samples only after a
DDR partial-layer completion graph has passed exact-once and logits parity
checks.

The shared stage worker now exposes queue/run aggregates and each pool reports
completed DMA time. These are diagnostic inputs for the policy; they do not
change route selection yet.

## Scope and reproducibility

The scoped graph files are:

- `src/ggml_qwen4exp_graph.cpp`
- `cuda/qwen4exp_master_graph.cu`
- `include/qwen4exp_early_ready.hpp`
- `include/qwen4exp_stage_pool.hpp`
- `include/option_b_global_hot.hpp`
- `patches/ggml-scheduler-early-ready-layer13.patch`
- `bench/option_b_early_ready_check.py`
- `bench/analyze_qwen_route_prediction.py`
- `bench/option_b_topic_check.py`
- `bench/option_b_request_check.py`
- `configs/qwen4exp_q2_0_option_b_scheduler.env`
- the graph handoff and worklog documents

The patch filename retains its historical Layer-13 name; the scheduler hook
now supports all eligible split layers. The commit does not include generated
logs, build directories, unrelated README changes, or unrelated ExLlamaV3 and
native-test edits. Those files remain in the worktree for their owners.
