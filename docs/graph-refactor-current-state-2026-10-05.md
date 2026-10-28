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
ZeroMoE continues to support; layered execution is not a future support
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

## 2026-10-06 staged-completion investigation

The current uncommitted graph source now reserves the payload-backed staged
child's additional `topk` expert planes in `device_expert_bytes()`. Before this
reservation, the historical staged commits reached construction with an arena
capacity error: the completion child needed about 4.6 MiB beyond the planned
partial arena. After the reservation, the RTX 3090 run reaches
`QWEN4EXP-COMPLETION-STAGED`, publishes a Ready payload, builds the resident /
staged conditional parent, and exercises mixed and all-staged route modes
without `ARENA-BIND-FAIL`.

The parity blocker is separate from payload unpacking. A bounded four-token
run with the current staged environment produces `8160, 8160, 271, 271`, while
the full-graph control produces `8160, 369, 264, 4145`. Forcing the same
phase-2 continuation through the CPU suffix produces the same mismatch, so the
staged GPU child is not the first fault. The phase-2 / prefix handoff remains
experimental and is not eligible for the production scheduler. The accepted
early-ready path remains the default.

The source now publishes Route-graph normalized weights and has explicit
publication buffers for the Prefix/Route HC input and inject values. These
changes are guarded by the phase-2 flags and are retained for the next state
handoff probe; the runtime parity gate is still failing. `make test-components`
passes all 44 tests and the CUDA engine target builds successfully after the
allocator/publication changes.

The opt-in `ZEROMOE_QWEN4EXP_PHASE2_STATE_TRACE=1` probe hashes recurrent and
KV storage after each synchronized replay. It first exposed post-split GDN
state writes stuck at their initial value; explicit phase-2 state terminals and
host mirrors now make those source and destination hashes change. The phase-2
path still diverges at the first continuation token, so the remaining fault is
the split-layer HC/route continuation output rather than staged payload upload.

## Original graph-tier requirements: current audit

| Requirement | Status and evidence |
| --- | --- |
| Immutable route classification and exact-once cases | The classifier, fixed-buffer branch projection and all-hot/all-cold/mixed/duplicate tests are implemented. The opt-in DDR callback now consumes a per-replay classification for its CPU branch; the accepted path is unchanged and staged ownership is not yet populated by the runtime graph. |
| Reusable DDR partial-layer graph with resident conditional, cold-only host callback and staged conditional | **Implemented behind opt-in flags; smoke-validated on RTX 3090.** The DDR graph retains the bounded CPU callback. The split-only `SplitStaged` child owns compact top-k CUDA tensors, aliases the prefix route IDs, and consumes a device-only `[gate|up|down]` payload unpack inside the staged conditional body; the mixed CPU/GPU tail runs afterward. The composer remains fail-closed unless the conditional children are single-CUDA graphs. |
| A/B partial graph against accepted early-ready | **A/B completed; candidate rejected.** After fixing both the per-token weight transfer and activation dependency, the corrected six-pair decode comparison measured 34.05 tok/s for early-ready and 26.70 tok/s for the partial callback: −21.61% mean paired gain, 0/6 wins. The 64-step correctness gate is byte-identical, generated tokens match in all six pairs, and all 63 partial callbacks per run succeeded. See `logs/option_b_partial_layer13_activation_fix_perf_tty/results.json`. |
| SSD dispatch/completion with ready/error generations | Host-only bounded `StorageService` has model/request generations, explicit Ready/Error, stale-handle rejection, timing, shared payload ownership, atomic `wait_consume`, and a completion submitter hook. The opt-in Qwen owner reads expert planes with `pread` from model shards (SSD→host; no GDS), and the completion graph uploads a stable payload. The staged suffix now consumes that payload through the conditional graph; RTX runtime validation is still pending. |
| Shared shape-class staging pools | Layers sharing top-k, tensor shapes and quant types share a worker service. GPU expert storage, metadata and DMA banks remain per layer for consumer lifetime safety. A shared GPU slot arena is not implemented. |
| Adaptive CPU versus staged-GPU selection | The opt-in policy warms four measured CPU-suffix executions, then forces one staged calibration sample. Later decisions happen before storage submission, so a CPU choice skips the SSD read. Inputs use storage queue/service timings, CUDA-event H2D duration, device-event conditional-graph execution, measured tail execution, and host dispatch/payload-preparation time. CPU execution is synchronous with no separate request queue, so its measured queue component is zero and the complete suffix latency is recorded as execution. Runtime A/B and logits parity remain pending. |
| Full-graph-only policy | Qwen graph prefill/decode now default on and an enabled Qwen graph failure aborts the request rather than entering layered execution. The native suite passes (`logs/fullgraph_only_policy_final_native.log`). **The all-family migration is incomplete:** GLM5NEXT and DeepSeek-V4 have partial graph adapters, and other families still use the layered path. |

The cold-route host callback previously called `RouteClassification::exact_once()`,
which allocated a temporary `seen` vector once per replay despite the
allocation-free callback contract. The classifier stores its immutable records
in original route-slot order, so `exact_once()` now validates slot `i` at
position `i` in O(top-k) time without scratch allocation. The focused route
test counts global allocations around the reserved classify/validate/project
sequence and covers all-hot, all-cold, mixed and duplicate-route cases. It is
now run by `make test-components` in all three negative-route capability
modes. That CPU target also compiles and runs the storage-generation,
staged-boundary and adaptive-policy tests, plus the shared shape-class pool
test when CUDA/GGML headers are available. The latest `make test-components`
run passed all 44 existing component tests and each focused graph-tier test.

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
completion boundary), `3ee2feb` (atomic completion consumption), `cfec8d1`
(pread-backed shard loader), `b945d47` (host completion dispatcher hook),
and `62527d1` (captured completion-graph payload upload), `18d2903`
(payload offsets), `ee621b6` (external-event join fix and RTX 3090 runtime
evidence), and `dc6896a` (runtime conditional topology contract).
The persistent completion-graph handle is `d468d64`; `a196237` composes its
captured payload child before the staged route predicate.
`66f8d18` also includes the completion child in the master-graph rebuild key,
so prepare/reset changes cannot leave a stale child graph attached.
`080b65e` wires the payload-backed `SuffixStaged` child and its generation-
validated runtime dispatch; it is intentionally not the production default.
`202b440` adds the measured opt-in adaptive CPU/staged-GPU decision and its
warmup counters.
`3eaa43b` factors uniform payload-slot validation into the storage boundary and
tests duplicate route slots plus malformed offsets.
`b7a5c2d` adds a host-only staged-storage/adaptive-boundary regression test.
`441ccd4` makes the Phase-2 prefix borrow the bootstrap expert arena instead
of allocating a second copy; the first RTX run exposed this 10.93-GiB OOM.
`ee1c8dc` extends the CUDA topology contract to execute the production payload
unpack kernel inside the staged conditional body.
`0f958dd` separates the pure-GPU split child from the mixed CPU/GPU tail, and
`8a733af` allows the continuation graph to omit the already-executed prefix
child handle.

## Selected runtime

The selected Q2_0 RTX 3090 profile is Option-B 0/44/4:

- 8,421 usage-selected experts remain resident in 10.84 GiB of VRAM.
- Layers 0, 1, 2 and 22 use the plain CPU path.
- Ten CPU threads are used while other host work may be present.
- BF16 MMVF routing remains disabled for this A/B.
- SSD/GDS integration and dynamic staged-GPU execution are disabled.

Source `configs/qwen4exp_q2_0_option_b_scheduler.env` together with
`configs/testing/qwen4exp_q2_0_option_b_rightsize_split44_x0_cov80.toml` when using the
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
submits a matching completion before the continuation graph is launched. When
a completion graph is prepared, its captured payload copy is a parent-graph
child before the route predicate, and a device-only unpack kernel fills the
staged tensors only in the staged conditional body. Any storage, generation,
topology, or graph-admission failure clears publication and runs the existing
CPU suffix instead.

The topology contract is executable in
`tests/test_qwen4exp_conditional_graph.cu`: it runs all-resident, all-staged,
and mixed route sets through pure-GPU conditional children and verifies an
unconditional parent host node runs after the branch. On the RTX 3090 it
reports `PASS: resident/staged conditionals with parent host node`. This test
covers topology and payload unpacking, but it does not replace the model
runtime A/B. Build and run it with:

```bash
/opt/cuda/bin/nvcc -std=c++17 -O2 -arch=sm_86 \
  tests/test_qwen4exp_conditional_graph.cu \
  cuda/qwen4exp_master_graph.cu -o /tmp/test_qwen4exp_conditional_graph
CUDA_VISIBLE_DEVICES=0 /tmp/test_qwen4exp_conditional_graph
```

The payload-backed staged A/B is enabled only for an explicit experiment:

```bash
ZEROMOE_QWEN4EXP_STORAGE_SERVICE=1 \
ZEROMOE_QWEN4EXP_COMPLETION_GRAPH=1 \
ZEROMOE_QWEN4EXP_COMPLETION_STAGED=1 \
ZEROMOE_QWEN4EXP_MASTER_GRAPH_STAGED=1 \
ZEROMOE_GGML_CUDA_GRAPH_CAPTURE_ON_FIRST_CALL=1 \
ZEROMOE_QWEN4EXP_ADAPTIVE_POLICY=1 \
ZEROMOE_QWEN4EXP_PARTIAL_LAYER=13 \
ZEROMOE_QWEN4EXP_FULL_GRAPH=1 \
ZEROMOE_QWEN4EXP_FULL_GRAPH_DECODE=1 \
./build_ggmlcuda/zeromoe --config configs/testing/qwen4exp_q2_0_option_b_rightsize_split44_x0_cov80.toml
```

The prefix runs once, the storage worker loads the exact route slots with
`pread`, and a continuation graph selects resident, staged, or CPU fallback.
The first integration uses synchronous request completion so generation and
ownership correctness can be measured before adding overlap. A CUDA runtime
measurement of this new path is still required; the existing storage self-test
and synthetic conditional topology test do not substitute for that A/B.

The first RTX 3090 smoke run of the split-child path reached `route mode=2`
(all routes staged), published a Ready completion with 10 route slots and
13,824,000 payload bytes, and completed the common tail without hidden-state
NaNs. This is a correctness smoke result using a top-1 resident set; it is
not a throughput comparison against the production 0/44/4 profile.
The matched no-storage Phase-2 control on this binary currently reports
all-NaN hidden states, so its timing is rejected as an A/B baseline. Use the
already validated early-ready results for production comparison until that
control is repaired.

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
  --candidate-env ZEROMOE_QWEN_EARLY_READY_RESIDENT_TABLES=1 \
  --third-env ZEROMOE_QWEN_EARLY_READY_RESIDENT_TABLES=1 \
  --third-env ZEROMOE_QWEN_EARLY_READY_ASYNC_OUTPUT=1
```

The three arms are the original scheduler, early-ready plus resident tables,
and that bundle plus asynchronous output. The script records admission state,
rejects compiler/inference interference, and writes `summary.json` with the
paired gains. For a normal run rather than an A/B, source the runtime overlay
and launch the configured engine:

```bash
source configs/qwen4exp_q2_0_option_b_scheduler.env
../moestream-build-upstream-master/moestream \
  --config configs/testing/qwen4exp_q2_0_option_b_rightsize_split44_x0_cov80.toml
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

## Retired milestone 2: DDR-only partial-layer graph prototype

The prototype, env selectors, and partial-only host-node callback path were removed on 2026-10-28. The following results are retained as historical rejection evidence.


The historical prototype used `ZEROMOE_QWEN4EXP_PARTIAL_GRAPH=1` together
with `ZEROMOE_QWEN4EXP_PARTIAL_GRAPH_LAYER=13`; both selectors were removed
on 2026-10-28 after the measured candidate was rejected. It kept the static resident
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
`results.json`. The partial-graph prototype and callback implementation were removed because
the bounded callback and graph-stream serialization cost about 21.6%.

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

It passed ready, error, stale-generation, release, duplicate-route and malformed
payload-slot cases. Runtime SSD loading and completion-graph submission are
implemented behind opt-in flags; the current production profile does not use
this service.

The host-only staged boundary simulation is
`tests/test_qwen4exp_staged_boundary.cpp`:

```bash
g++ -std=c++23 -O2 -pthread -Iinclude \
  tests/test_qwen4exp_staged_boundary.cpp -o /tmp/test_qwen4exp_staged_boundary
/tmp/test_qwen4exp_staged_boundary
```

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

## Milestone 6: storage telemetry and opt-in adaptive policy

`StorageService` now publishes queue and service durations with every
`Ready`/`Error` completion and exposes aggregate submitted/rejected/ready/error
counters. The model and request generations remain part of the completion, so
an SSD/DDR completion graph can reject a stale result before it touches a
staging slot. Loader exceptions are converted to an explicit error completion.
The storage worker remains host-only: a CUDA host callback may enqueue an
immutable request, but the worker does not call CUDA or wait for a graph
operation. The separate completion graph owns only the post-Ready H2D copy.
Requests may optionally carry a shared payload buffer; the loader fills it and
the matching completion returns the same object for a later graph submission,
avoiding a second SSD/DDR copy. `StorageCompletionDispatcher` provides the
single host handoff that consumes and generation-validates a completion before
calling a future graph submitter.

The optional runtime owner is created after graph-RAM preload, before the
prefetch workers start. It reads packed or unpacked expert planes with
generation-safe `pread` requests into the shared payload and preserves the same
Ready/Error generation contract. Enable
`ZEROMOE_QWEN4EXP_STORAGE_SERVICE=1` and, for a one-request Ready/Error
check, `ZEROMOE_QWEN4EXP_STORAGE_SELFTEST=1`. Add
`ZEROMOE_QWEN4EXP_COMPLETION_GRAPH=1` prepares a persistent payload arena
before Qwen graph bootstrap, and `ZEROMOE_QWEN4EXP_COMPLETION_GRAPH_SELFTEST=1`
captures and launches the stable-address H2D completion graph and joins its
consumer event. These remain opt-in and do not change the production profile;
the continuation graph now unpacks the prepared payload in the staged
conditional body, while malformed, stale, mixed-geometry or graph-admission
failures use the CPU suffix. A
direct RTX 3090 self-test produced `submitted=1`, `bytes=1382400`, `ready=1`
and a separate generation-tagged `error=1 error_code=404` record in
`logs/qwen4exp_q2_0_option_b_split44_x0_cov80.log`.

`include/qwen4exp_adaptive_policy.hpp` contains the opt-in decision rule. With
`ZEROMOE_QWEN4EXP_ADAPTIVE_POLICY=1`, four CPU suffix executions seed the
baseline. The first staged route is then run as a calibration sample; later
decisions use those measurements before submitting storage work, so choosing
CPU skips that request's SSD read. Storage queue/service come from the matching
generation-tagged completion. H2D duration now comes from CUDA events around the
captured upload, while the prior wall-clock `upload_wait_ms()` is retained only
as submit-to-observation latency. Device events time the resident/staged
conditional graph; a separate suffix replay measures the tail; host payload
preparation and graph-dispatch overhead are recorded separately. CPU suffix
execution is synchronous in GGML's backend, so its measured queue component is
zero and its complete replay duration is used as execution cost. Invalid or
unmeasured targets fail closed to CPU. Focused CPU tests cover unavailable
samples and each cost component; an RTX runtime A/B and logits-parity result are
still required.

The shared stage worker now exposes queue/run aggregates and each pool reports
completed DMA time. These are diagnostic inputs for the opt-in policy; the
production route selection remains unchanged.

The worker topology is now a shape-class registry: layers with matching
`topk`, projection dimensions and quantization types share one worker service;
layers with different shapes get separate services. Weight arenas, bounce
banks and epoch metadata remain per layer to preserve consumer lifetimes.

## Experimental staged-graph memory telemetry

At the time of this 2026-10-05 record, the staged graph memory trace was
selected by `ZEROMOE_QWEN4EXP_MEMORY_TRACE=1`. Current code emits the same
telemetry at VERBOSE logging (`--verbosity 4`). It records CUDA free/total VRAM,
`MemAvailable`, process RSS/peak RSS, cgroup memory usage, graph-construction
mode, storage completion, completion-graph submission, conditional graph
instantiation and graph launch. CUDA allocation, graph-instantiation and
graph-launch failures are tagged with a separate memory/OOM record. The trace
is disabled by default and does not alter the accepted early-ready scheduler.

The first post-reboot two-token diagnostic used the checked-in Q2_0 Option-B
profile with the staged flags and the known finite fallback overrides. It
completed without a driver reset or OOM: VRAM stayed above 7.67 GiB free at
conditional launch, host `MemAvailable` stayed above 84 GiB, the 13.8 MB
payload upload reached `Pending`, and route mode 3 executed. This is a memory
stability smoke result, not a staged-vs-early-ready correctness or throughput
result. The experimental path remains uncommitted pending logits parity.

The first post-instrumentation bootstrap caught and fixed a classifier shape
error: the raw `[hidden, HC]` residual was passed directly to the router,
which expects the HC-mixed `[hidden]` shape. The corrected bootstrap now builds
Prefix, Route, resident, staged and tail graphs with no GGML assertion. A full
token replay remains gated until the staged route has a fresh parity result.

A subsequent two-token replay completed with finite logits and route mode 3
(mixed resident/staged). The trace reported RSS 36.8--37.5 GiB, peak RSS
108.3 GiB, `MemAvailable` above 83.4 GiB and VRAM free above 7.67 GiB at the
conditional launch. No OOM marker, CUDA error or hidden-state corruption was
reported. This validates memory stability and graph execution only; it is not
yet logits parity or a throughput acceptance result.

## 2026-10-06 route-boundary investigation

The clean bounded teacher-forced comparison used one 64-token repetition with
all graph diagnostics disabled. The validated early-ready arm measured 32.40
tok/s. The experimental staged arm measured 12.85 tok/s, and the logits were
not identical (top-1 agreement 1.6%; first disagreement at decode step 1).
The staged log proves Ready completion plus mixed/pure staged conditional
dispatch, so this is a real correctness failure before any performance tuning;
the production early-ready path remains the reference and the staged flags
remain opt-in. An earlier 31.04/12.74 comparison used debug graph snapshots
and is retained only as historical diagnosis.

A control with the partial-layer planner enabled but Phase-2 disabled previously
attempted to allocate an additional 10.9 GiB compact arena and hit CUDA OOM
during graph setup. A pre-allocation admission guard now measures free VRAM
against the compact-arena estimate plus a 768 MiB margin; the reproduced case
now declines cleanly at 8.95 GiB free (`need=10.92 GiB`) before `cudaMalloc`
and aborts the experimental request without a driver fault. The risky
allocation can be explicitly forced for diagnosis with
`ZEROMOE_QWEN4EXP_PHASE2_ALLOW_RISKY_ALLOC=1`.

The diagnostic then compared the same Prefix/Route handoff with a debug CPU
suffix. The CPU suffix produced the same wrong next token as the staged path,
so payload unpack and resident/staged merge are not the first failure. An
ordinary full-graph control, with the same model and prompt and without
Phase-2, produced the expected step-1 token (`369`). The isolated Route graph
received a finite Layer-13 HC input, but its route IDs differed from the
ordinary control (`[195,329,118,128,420,412,220,367,481,472]` versus the
control route `[273,50,385,387,35,347,93,286,6,423]`). The Prefix route-output
tensor itself was stale (`[0..9]` and zero probabilities), so it is not a
valid classifier source. The Route graph is currently used as the staged
classifier, while the stale Prefix output keeps the non-staged Phase-2 branch
unvalidated.

The Prefix handoff was finite (min `-2.058`, max `1.276`), and memory telemetry
still showed about 7.67 GiB free VRAM, about 85 GiB `MemAvailable`, RSS around
37.5 GiB and peak RSS around 108.3 GiB. No OOM, CUDA error, or hidden-state
corruption marker appeared. The remaining correctness issue is therefore a
split-boundary graph/state mismatch, observed before staged payload execution;
no performance conclusion should be drawn from the 12.74 tok/s run. The route
dependency chain is present and assigned to CUDA in Prefix (softmax → argsort
→ view → contiguous), so the failure is deeper than a missing backend
assignment. Prefix's captured HC classifier snapshots are zero while Route
recomputation from the same residual handoff is finite; an opt-in attempt to
omit the duplicate Prefix classifier was unsafe during bootstrap and was
reverted.

The investigation also fixed two lifecycle errors: the isolated Route graph
now receives the per-token input handoff and its PLE embedding is refreshed
with the other Phase-2 graphs. The generic F32 diagnostic snapshot now sizes
its destination by element count, avoids under-allocation for BF16/F16 sources,
and marks its copy nodes as graph outputs so allocator reuse cannot leave an
uninitialized trace. These fixes are covered by a successful source build and
the 44/44 CPU component tests, but a fresh logits-parity run is still required.

The CPU continuation now has an explicit host-visible handoff: Prefix performs
a D2H copy into the SuffixCpu input before replay, and that leaf is assigned to
the CPU backend. The bounded smoke confirms the host handoff is finite, but
the CPU HC result still does not match the ordinary full graph, so this tier is
not accepted yet.

The latest source inspection found a concrete lifetime hazard in the Prefix
handoff refresh. The graph copies the residual into a separately allocated,
persistent `handoff_out`, but after replay the host also copied from
`prefix.final_streams` into that buffer. `final_streams` is an intermediate
graph tensor; its allocator lifetime ends at its last graph consumer, so
reading it after replay can observe storage reused by other graph outputs.
This is a plausible cause of the split-boundary mismatch, not yet a proven
sole cause. The diagnostic now marks the persistent destination as an output,
removes an extra source-residual output that produced all-NaN probes, and adds
`ZEROMOE_QWEN4EXP_P2_GRAPH_HANDOFF_ONLY=1` to test the graph-written buffer
without the post-replay overwrite. A read-only pre-refresh dump is enabled by
`ZEROMOE_QWEN4EXP_STAGED_DEBUG=1`. The modified graph translation unit
compiles successfully. The current full target now also builds successfully:
`cmake --build build_ggmlcuda -j2 --target zeromoe`. No new inference run has
been made, so handoff parity remains unverified; the early-ready scheduler
remains the default.

The first guarded graph-handoff-only probe used a temporary binary linked from
the updated graph object and cached objects. It stopped at `[MEM-GUARD] cache
initialization exceeded the process host budget` before phase-2 graph setup and
then exited with status `-11`. Host `MemAvailable` remained above 120 GiB and
VRAM was initially 23.3 GiB free, so this was not a Linux OOM or measured CUDA
allocation failure. The mixed-object probe is not correctness evidence. The
full target now builds, but subsequent idle `nvidia-smi` checks have
intermittently failed to contact the driver; no inference was retried. The next
parity check must use the consistent `build_ggmlcuda/zeromoe` binary and the
previously successful bounded Phase-2 configuration after driver health is
stable.

The storage-service unit test also covers a full bounded queue, rejection of a
second request, and safe reuse of the freed slot under a new request
generation. The test passes alongside the existing Ready/Error, stale-handle,
exception, and payload-shape cases.

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

## 2026-10-07 update: staged-arena alias fixed

The staged child previously followed only one `phase2_expert_source_graph`
edge when allocating expert tensors. When Prefix borrowed the retained decode
arena, a later staged child borrowed from Prefix and allocated with Prefix's
reset bump cursor. Its 4.6 MiB staged up tensor landed inside the 91.2 MiB
Layer-3 Option-B up tensor. This matched the 174/198 resident-plane corruption
observed after Prefix replay and the staged decode's first-token divergence.

`phase2_bind_or_allocate_expert()` and named expert lookups now resolve the
full source chain to the true buffer owner before looking up or allocating an
address. Two bounded teacher-forced runs verified the fix: all 198 resident
planes remained byte-identical after Prefix, and the staged conditional path
consumed a Ready payload on four consecutive decode steps. The five generated
tokens matched the early-ready reference (`8160, 369, 264, 4145, 15673`); the
five-step top-1 comparison was 100% with mean KL `0.000382`. No OOM kill or
NVIDIA Xid was observed. Logs and resource samples are retained under
`logs/phase2_layer3_up_weight_verify_20261007/run19_kernel_param_scan/`,
`run20_forced_cold_stage/`, and `run21_forced_cold_stage_5tokens/`.

This is a correctness result, not a throughput acceptance run. The staged
path remains opt-in and early-ready remains the default. Next: run a
representative paired performance comparison, then exercise the bounded CPU
callback and mixed resident/staged routing cases before considering adoption.
