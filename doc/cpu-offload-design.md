# Qwen4Exp CPU-Offload Design

Status: design baseline for the `CPUOffload` branch.

This branch explores a hybrid routed-expert path:

- hot experts remain in VRAM and execute with the existing CUDA kernels;
- cold experts remain permanently resident in the DDR arena and execute on
  the CPU;
- only the routed activation/result vectors cross the CPU/GPU boundary;
- dense attention, shared experts, and all non-target layers retain their
  existing GPU path unless a later experiment explicitly enables more layers.

The baseline branch remains the performance control. The CPU-offload path must
be opt-in and must never change dispatch when its flag is disabled.

## Current code that can be reused

The repository already contains most of the low-level pieces:

| Existing component | Location | Reuse |
| --- | --- | --- |
| Complete fixed-stride DDR expert arena | `src/token_dispatch_pipeline.cpp` | Supplies gate/up/down planes and global expert IDs without per-token weight copies |
| Persistent CPU GGML expert runner | `src/ggml_qwen_cpu_moe.cpp` | CPU `MUL_MAT_ID` execution over the DDR arena |
| Direct grouped quantized CPU path | `src/token_dispatch_pipeline.cpp` and CPU kernels | Faster IQ1_M/IQ4_NL path for the production checkpoint |
| Pinned activation scratch | `include/scratch_buffers.hpp` | `LayerScratch::normed` and `qwen_cpu_moe_out` are already `cudaMallocHost`-backed when allocation succeeds |
| Shared-expert GPU graph | `src/ggml_qwen_shared_runner.cpp` | `run_shared_only()` and `run_host_routed_add()` provide the existing overlap shape |
| Per-expert hot/cold classification | `src/token_dispatch_pipeline.cpp` | `qwen4exp_option_b_hot_expert()` and `force_cpu_offload` already identify cold routes |

The current path is not yet the target implementation. It generally computes
the whole routed result on one side, then hands the final `[D]` vector to the
other side. The new path must split the top-k set into hot and cold subsets in
one token/layer dispatch.

## Target execution sequence

For one decode token at an enabled layer:

```text
router/top-k
    |
    +--> hot IDs  ----> GPU routed experts -------------------+
    |                                                         |
    +--> cold IDs ---> persistent CPU worker, DDR weights     +--> GPU combine
                                                              |
shared expert -----------------------------------------------+
```

The first implementation uses the existing host-visible normalized activation
buffer. If a future path has only a device-resident activation, it must first
queue a D2H copy into a preallocated pinned input buffer; it must not pass a
device pointer to the CPU runner.

### 1. Route partition

After top-k selection, build fixed-capacity arrays in `LayerScratch`:

- `hot_ids[]`, `hot_weights[]`;
- `cold_ids[]`, `cold_weights[]`;
- `cold_count` and `hot_count`.

These arrays must be allocated during scratch initialization. No `vector` growth,
expert-weight packing, or heap allocation is permitted in the decode path.

The partition must preserve the router order and weight associated with every
expert. The CPU runner receives only the cold IDs and weights; the GPU runner
receives only the hot IDs and weights.

### 2. CPU cold path

The CPU path uses the complete immutable DDR arena:

```text
input:       pinned [D] float activation
weights:     DDR [512 experts × gate/up/down]
route data:  preallocated cold IDs and weights
output:      pinned [D] float cold routed result
```

The existing `GgmlQwenCpuMoeRunner::run_strided()` is the correctness fallback.
The direct grouped IQ path is the performance candidate. Both must accept a
cold subset with `cold_count`, rather than assuming all top-k experts are CPU
owned.

The worker must be persistent for the process lifetime of the decode request.
The branch will expose the worker policy in telemetry and test with:

```text
OMP_WAIT_POLICY=active
```

This is a test variable, not a correctness requirement. If active waiting is
needed to reach the target, that cost must be reported explicitly.

### 3. GPU hot path and shared expert

The GPU submits hot routed experts and the shared expert without waiting for
the CPU. The existing GPU expert accumulator can continue to own the hot
result. The final combine needs three terms:

```text
shared_gpu + hot_gpu + cold_cpu
```

The first safe implementation may download the hot result to the existing
pinned result buffer and use the current host-routed add path. The optimized
implementation should add the cold CPU vector directly to the device-side
shared-plus-hot result with one small device combine graph/kernel.

The optimized combine is important: otherwise the experiment can accidentally
replace large weight traffic with an unnecessary GPU-to-host activation copy.

## Completion and stream ownership

There are two milestones.

### Milestone A: overlap-safe proof

Use the already implemented scheduling shape:

1. submit GPU shared/hot work;
2. start the persistent CPU cold job;
3. wait only for CPU completion on the host;
4. queue one pinned activation upload and the GPU combine;
5. continue the layer pipeline.

This is not fully host-nonblocking, but GPU shared/hot work runs while the CPU
computes. It is the first correctness and timing checkpoint.

### Milestone B: device-fenced completion

After Milestone A is numerically stable, add a per-slot completion record:

```text
CPU worker:
    compute cold result in pinned host slot
    cudaMemcpyAsync(host slot -> device cold slot, copy_stream)
    cudaEventRecord(done_event, copy_stream)

GPU submission:
    compute_stream waits on done_event
    combine graph consumes device cold slot
```

The slot must not be reused until the compute stream has consumed it. Use a
two- or three-slot ring and an event per slot. The implementation must not use
`cudaStreamSynchronize()` in the decode path.

The CPU worker is allowed to enqueue the H2D copy after its math completes;
the event represents completion of both the copy and the preceding CPU work.
The event/copy-stream behavior must be validated with a small standalone test
before it is connected to the model graph.

## CUDA Graph boundary

A monolithic captured graph cannot dynamically perform arbitrary CPU expert
math. The first CPU-offload proof therefore uses an explicit Layer-13
dispatch seam while keeping the surrounding GPU work unchanged.

`cudaHostNode` is not the first implementation choice. A host node callback
would make CPU scheduling part of graph execution and complicate buffer
lifetime, error propagation, and CUDA API calls from the callback. It can be
evaluated later, after the explicit seam has a measured benefit.

The control matrix is:

| Profile | Surrounding layers | Layer 13 | Purpose |
| --- | --- | --- | --- |
| Baseline | Existing full graph | Existing full graph | A/A control |
| CPU proof | Existing path | Explicit hot/cold seam | Correctness and timing |
| CPU optimized | Existing path | Device-fenced combine | Handoff overhead |
| Scale test | Existing path | Usage-selected layers | Throughput and tail behavior |

## Configuration proposal

Add a separate opt-in configuration family rather than reusing the Option B
static-pinning switch:

```toml
qwen4exp_cpu_offload = true
qwen4exp_cpu_offload_layers = [13]
qwen4exp_cpu_offload_hot_source = ".moestream_usage_qwen4exp_7prompts_isolated"
qwen4exp_cpu_offload_hot_fraction = 0.50
qwen4exp_cpu_offload_max_cold = 10
qwen4exp_cpu_offload_event_fence = false
qwen4exp_cpu_offload_ring_slots = 2
qwen4exp_cpu_offload_timing = true
```

The initial profile should keep all other model layers on the established
baseline path. `event_fence=false` selects Milestone A; only after the A/B
result is understood should it be changed to `true`.

## Required telemetry

For every enabled layer and decode token, record separate totals for:

- route partition time;
- hot expert GPU issue and finish time;
- cold CPU queue wait;
- cold CPU compute time;
- input D2H time, if needed;
- output H2D issue time and device event completion time;
- host wait time in Milestone A;
- device combine time;
- complete Layer-13 time;
- cold count and hot count.

The log must distinguish `H2D issue` from `H2D completion/wait`. A fast
`cudaMemcpyAsync` call alone is not evidence that the transfer was hidden.

## Correctness gates

1. CPU cold result matches the existing all-CPU routed result within the
   established tolerance.
2. Mixed hot/cold result matches an all-GPU/all-CPU oracle for the same route
   IDs and weights.
3. No expert-weight H2D occurs during decode for the CPU-offload path.
4. Pinned input/output buffers remain alive until the consumer event completes.
5. A failed CPU worker, event, or combine falls back to the existing safe path
   and never returns an incomplete MoE result.
6. With the feature disabled, the baseline graph and token output are
   unchanged.

## Test sequence

1. Build and run existing component/native tests.
2. Run Layer 13 with one forced cold expert and compare outputs.
3. Run Layer 13 with real top-k routing and a 50% usage-ranked hot set.
4. Run 32-64 decode tokens with Milestone A timing.
5. Enable the event-fenced ring and repeat the same A/B run.
6. Only then select additional layers from the usage histogram.

The first success criterion is not a global throughput target. It is a clean
Layer-13 ledger showing that CPU cold compute overlaps GPU work and that the
only PCIe payload is the activation/result vector.

## Unified-graph follow-up

A single graph launch is a valid optimization target, but it is a later
milestone rather than the first implementation. The proposed host callback
must obey one important rule:

> A host callback that queues work to another CPU worker and immediately
> returns does not make downstream graph nodes wait for that worker.

If the callback returns immediately, a following H2D memcpy node can read the
output buffer before the CPU worker has written it. A later
`cudaStreamWaitEvent()` is also not automatically safe when the same event is
reused across graph replays: the wait can observe the previous replay's
completed event unless the event lifecycle is explicitly synchronized.

There are two implementable variants:

### Unified graph, synchronous host node

The host node callback performs the CPU cold-expert computation synchronously
and returns only after the pinned output buffer is complete. A fixed graph H2D
memcpy node follows it, then the GPU combine node consumes the device buffer.
This is graph-correct and keeps the main decode thread to one graph launch, but
the host callback thread still pays the CPU compute time. GPU work on an
independent graph branch can overlap it; the CPU work itself is not eliminated.

### Unified graph, externally signalled worker

The callback dispatches a persistent worker, but the graph needs a verified
per-replay completion protocol:

1. reserve a ring slot before launching the graph;
2. pass fixed host/device slot addresses to the graph instance;
3. let the worker write the host result and enqueue the H2D copy on a copy
   stream;
4. record the slot's completion event after that copy;
5. ensure the graph wait observes the new event generation, not a prior
   replay;
6. retire the slot only after the combine graph has consumed it.

This requires an event-generation or per-replay slot protocol. It must be
validated with a standalone CUDA test before graph integration. A plain
`cudaStreamWaitEvent(compute_stream, reused_event)` placed during capture is
not sufficient evidence of correctness.

The unified graph also needs explicit fixed pinned buffers. Arbitrary GGML
tensor pointers are not host buffers, and `get_pinned_memory(node->src[0])`
cannot be assumed to work for a device-resident activation. If the graph owns
the activation, the graph must contain an explicit D2H node into a pinned
input slot before the host node.

For this reason the rollout order is:

1. prove the mixed Layer-13 path with the explicit scheduler seam;
2. prove the host-node and fixed-buffer protocol with a synthetic graph;
3. compare synchronous-host-node and externally-signalled-worker variants;
4. only then move the Layer-13 seam into the full model graph.

The expected outcome is a reduction in host submission overhead, not a
guaranteed 29.44 tok/s result. The 1.92 ms gap and 71 ms outlier must be
re-measured after the graph conversion; graph capture alone cannot prove that
the outlier originated in scheduler transitions.
