# Qwen graph refactor: recorded state

## Cross-session update: graph API review and execution policy

The legacy-cache/VRAM handoff changes the current placement context. The
earlier 33.87 tok/s graph A/B used the right-sized 0/44/4 set (8,421 experts).
The later VRAM sweep removed a 6.44-GiB empty legacy cache and measured a
global 48-layer set of roughly 14.5k experts at about 39 tok/s in single-run
probes. The two results have not been stacked: no six-pair early-ready A/B
against the 48-layer, 14.5k placement exists yet. Treat +2.28% in the
2026-10-04 handoff as early-ready over its 0/44/4 baseline; the later
contemporaneous three-arm run measured +4.54% with resident route tables and
+6.36% with asynchronous output added. See
`docs/handoff-2026-10-05-legacy-cache-and-vram-budget.md` before choosing the
next production placement.

The user's current architecture rule is full-graph execution for every model
MoEStream continues to support; layered execution is not a future support
path. Qwen full-graph prefill/decode now default on, and an enabled Qwen graph
failure is returned as a request error instead of silently entering the layer
loop. Native tests passed after this change. This enforcement is currently
Qwen-specific: GLM5NEXT has conditional graph coverage, DeepSeek-V4 has only
prefill graph coverage, and other families still use layered execution. The
all-family graph-only migration remains open; do not describe it as complete.

The attempted no-Option-B top-64 Phase-2 run is under
`logs/phase2_conditional_no_optionb_top64_v5.*`. All three full-graph instances
built, but the CUDA master conditional rejected prefix/suffix embedding because
their scheduler graphs had 28/69 splits, not single CUDA backends. It therefore
used the host branch between separate full-graph submissions; this did not
exercise a CUDA conditional body. The 64-step forced-history comparison had
93.75% top-1 agreement and mean KL 0.00285; token IDs match because they were
teacher-forced. This is correctness evidence for the split execution only,
not a conditional-graph pass or a valid throughput A/B.

The CUDA API review is recorded in
`docs/cuda-graph-api-review-2026-10-05.md`. It rejects stream priority as a DMA
priority mechanism, retains GGML-owned stream capture, defers graph memory
nodes and TMA, and distinguishes captured external events from host callbacks.

## Original graph-tier requirements: current audit

| Requirement | Status and evidence |
| --- | --- |
| Immutable route classification and exact-once cases | The classifier, fixed-buffer branch projection and all-hot/all-cold/mixed/duplicate tests are implemented. The opt-in DDR callback now consumes a per-replay classification for its CPU branch; the accepted path is unchanged and staged ownership is not yet populated by the runtime graph. |
| Reusable DDR partial-layer graph with resident conditional, cold-only host callback and staged conditional | **Partially implemented.** The opt-in DDR graph combines resident GPU work with a cold-only CPU host callback, and now has a fail-closed three-way conditional composer (`MOESTREAM_QWEN4EXP_MASTER_GRAPH_STAGED`) with separate resident/staged child handles and a device route-mode predicate. It will only instantiate single-CUDA child graphs; no staged generation currently publishes a pure-GPU child, so the three-way composer remains inactive. The existing Phase-2 prototype's 28/69 scheduler splits still prevent its ordinary master conditional from running. |
| A/B partial graph against accepted early-ready | **A/B completed; candidate rejected.** After fixing both the per-token weight transfer and activation dependency, the corrected six-pair decode comparison measured 34.05 tok/s for early-ready and 26.70 tok/s for the partial callback: −21.61% mean paired gain, 0/6 wins. The 64-step correctness gate is byte-identical, generated tokens match in all six pairs, and all 63 partial callbacks per run succeeded. See `logs/option_b_partial_layer13_activation_fix_perf_tty/results.json`. |
| SSD dispatch/completion with ready/error generations | Host-only bounded `StorageService` has model/request generations, explicit Ready/Error, stale-handle rejection, timing, shared payload ownership, atomic `wait_consume`, and a completion submitter hook. An opt-in Qwen runtime owner now reads expert planes with `pread` from the model shards into shared payloads (`MOESTREAM_QWEN4EXP_STORAGE_SERVICE=1`) and has a generation-tagged Ready/Error self-test. **A separate CUDA completion-graph submission is still not integrated.** |
| Shared shape-class staging pools | Layers sharing top-k, tensor shapes and quant types share a worker service. GPU expert storage, metadata and DMA banks remain per layer for consumer lifetime safety. A shared GPU slot arena is not implemented. |
| Adaptive CPU versus staged-GPU selection | A timing policy helper and queue/run/DMA/storage telemetry exist. The policy is not wired to live selection or measured in an A/B; the measured staged route path also fails CPU-baseline numerical parity. |
| Full-graph-only policy | Qwen graph prefill/decode now default on and an enabled Qwen graph failure aborts the request rather than entering layered execution. The native suite passes (`logs/fullgraph_only_policy_final_native.log`). **The all-family migration is incomplete:** GLM5NEXT and DeepSeek-V4 have partial graph adapters, and other families still use the layered path. |

The latest VRAM handoff changes the performance baseline for placement work.
It reports 14.5k globally selected experts across 48 layers at about 39 tok/s
in single-run probes, while the current graph A/B's 2.28% early-ready result
was on 0/44/4. The early-ready scheduler has not yet been A/B tested on the
48-layer 14.5k placement. The newer result therefore does not close the
partial-graph or conditional-body requirements.

The four-step overlap plan is complete as an experiment sequence:

| Step | Result |
| --- | --- |
| Skip prepared CPU graph on all-hot routes | Implemented with copy/merge retained; +0.029%, 2/6 wins; disabled. |
| Measure transfer/join cost | Measured. Async output alone +1.356%, 6/6; with resident tables it added +1.744%, 6/6, and remains in the accepted bundle. No `cudaEventQuery` polling variant was tested. |
| Evaluate next-layer prediction | Measured offline: 62.72% proxy cold recall, 53.87% precision, 4.51 GB estimated unused bytes of 9.77 GB candidates over 2,646 visits; predictor throughput −0.05%, 3/6. These traces do not establish deadline success by themselves. |
| Add stable-slot dynamic staging | Slot generations, post-copy publication, copy-stream joins, and token-lifetime holds are implemented. Late-transfer fallback passes. Staged outputs match a GPU oracle, but CPU/GPU placement arithmetic shifts later hidden states; dynamic staging is rejected and off. |

This file records the committed graph work and the remaining unrelated
ExLlamaV3, CPU, configuration, test and generated-artifact changes in the
shared worktree. Graph commits are authored by `ray-x`; the latest scoped
follow-ups are `c9685b9` (partial activation and staged conditional composer),
`8a5cd68` (shared shape-class worker test), `db35ec8` (opt-in storage
completion boundary), `3ee2feb` (atomic completion consumption), and
`cfec8d1` (pread-backed shard loader and Ready/Error runtime self-test).

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

The current tree also contains a fail-closed three-way composer for a future
published staged generation. `cuda/qwen4exp_master_graph.cu` classifies each
route set on-device as resident, staged, or CPU fallback and writes two
conditional handles. `try_launch_three_way` accepts only single-CUDA child
graphs, so a staged child containing the DDR gather callback is rejected rather
than placed inside a conditional body. The engine carries a separate staged
expert-ID publication vector, and `publish_qwen4exp_staged_completion` now
validates a Ready completion before publishing it. The opt-in pread service
does not submit staged completions to this method yet, so the composer remains
inactive by default.

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

The classifier remains outside the accepted decode path. The opt-in DDR
partial-layer callback now builds a per-replay classification from its D2H
route snapshot and projects the CPU-owned routes into preallocated CPU-graph
inputs. The current prototype supplies only the resident map; its staged map
is empty until staged-slot publication is integrated. The production baseline
therefore remains unchanged.

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

All three modes passed, including branch projection that preserves each
original router weight on exactly one target and reuses preallocated output
buffers. The opt-in callback uses that projection without mutating the raw
D2H IDs/weights or the graph's immutable remap/mask tables.

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

Current code review found a correctness defect in that run: the partial CPU
graph multiplies by its per-token router weights, but the CUDA marker bridge
previously skipped the D2H weight copy whenever `gpu_reduce` was enabled. The
opt-in partial callback now copies weights, classifies every route slot, and
projects only CPU-owned slots into its private CPU inputs. The old parity and
speed numbers remain historical and do not judge the corrected graph. It also
skips CPU graph compute for an all-GPU route set and uploads a freshly
zero-initialized cold contribution.

The corrected implementation was built and run on the RTX 3090. Two defects
found during validation were fixed before the final A/B: the CUDA marker now
copies the current token's router weights, and the partial marker carries the
real post-attention activation as a separate graph dependency instead of
reading its dummy output source. The corrected 64-step forced-history
comparison against early-ready is byte-identical (zero KL, total variation and
maximum delta). The final six-pair order-balanced decode run measured 34.05
tok/s early-ready versus 26.70 tok/s partial, −21.61% mean paired gain, 0/6
wins. All six token files match and every partial run reports 63 successful
host callbacks. Full data and input hashes are in
`logs/option_b_partial_layer13_activation_fix_perf_tty/manifest.json` and
`results.json`. The partial graph remains opt-in and rejected for production
because its bounded callback and graph-stream serialization cost about 21.6%.

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

The shape registry has a CPU-only focused test at
`tests/test_qwen4exp_stage_shape_pool.cpp`. It confirms that identical
`(top-k, projection shape, quantization type)` keys share one worker service,
different keys remain isolated, and all submitted jobs complete. Compile it
with the integration GGML headers and run it before any CUDA staging A/B.

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
Requests may optionally carry a shared payload buffer; the loader fills it and
the matching completion returns the same object for a later graph submission,
avoiding a second SSD/DDR copy. `StorageCompletionDispatcher` provides the
single host handoff that consumes and generation-validates a completion before
calling a future graph submitter.

The optional runtime owner is created after graph-RAM preload, before the
prefetch workers start. It reads packed or unpacked expert planes with
generation-safe `pread` requests into the shared payload and preserves the same
Ready/Error generation contract. Enable
`MOESTREAM_QWEN4EXP_STORAGE_SERVICE=1` and, for a one-request Ready/Error
check, `MOESTREAM_QWEN4EXP_STORAGE_SELFTEST=1`. This does not change the
production profile and intentionally does not claim SSD or CUDA completion
integration.

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

The worker topology is now a shape-class registry: layers with matching
`topk`, projection dimensions and quantization types share one worker service;
layers with different shapes get separate services. Weight arenas, bounce
banks and epoch metadata remain per layer to preserve consumer lifetimes.

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
