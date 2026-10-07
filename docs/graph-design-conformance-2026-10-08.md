# Graph design conformance audit — 2026-10-08

The supplied design is only partially implemented. The current uncommitted
Qwen prototype contains the main topology of the first milestone: independent
resident/staged CUDA conditionals, a sibling cold-only host node, output
initialization, and a three-way merge. It is not an accepted all-layer,
asynchronous SSD execution system with globally shared storage pools.

## Audit scope and version distinction

Inspected `/home/ray/Projects/MoEStream`, branch `cleanup/ggml-exl3-only`, HEAD
`a28b4616f10d7f28501aa32aadf515a135cec525`, including its uncommitted files.
This checkout differs from the isolated `521d67a` bug-fix worktrees. No engine
code was changed and no new GPU run was launched for this audit.

The distinction matters: the committed route predicate still selects one of
all-resident, all-staged, or CPU fallback. The uncommitted
`cuda/qwen4exp_master_graph.cu` instead reports independent resident/staged/CPU
presence bits and can enable both GPU branches. Historical measurements and
topology PASS logs cannot certify this newer source unless their manifests
identify the matching build.

Audited source SHA256:

- `src/ggml_qwen4exp_graph.cpp`: `1568fbafa2a17959fe49e018c70890ff9a196d755615600844127e6fd0b0fb3b`
- `cuda/qwen4exp_master_graph.cu`: `35d4234323ba845d43a3046b9cedbc0807646a470d97428ee2f4af5f5d02aa9c`
- `include/moe_partial_route.hpp`: `8f0c653c0df035db8b45c6eeae949faacd760c058e5521481cd228c81311765a`
- `include/qwen4exp_stage_pool.hpp`: `238475ba0f161e583acf34dabaddcceb6886111eee79b9afbd4438bbb0aeb5f2`

## Comparison with the six design requirements

| Design requirement | Current implementation | Remaining gap / verdict |
| --- | --- | --- |
| 1. Three-tier policy, immutable assignments, exact-once router weights | Option-B static placement exists. `RouteAssignment` carries expert, original route slot, router weight, target, weight slot and generation. `RouteClassification` validates disjoint ownership, preserves duplicate route slots, and supports allocation-free projection into the cold CPU graph. Storage requests have Pending/Ready/Error states. | **Partial.** The new CPU callback and GPU predicate classify independently; one immutable descriptor is not yet the common authority for every branch. Unpublished routes map to CPU only because the callback requires available DDR weights. A genuinely SSD-only route needs a pending execution assignment until loading finishes. The legacy experimental Phase-1 GPU-only path still explicitly permits dropping an out-of-set expert; it violates this design's exact miss invariant. |
| 2. Coarse conditional branches with cold-only host node outside bodies | The uncommitted `try_launch_three_way()` clears resident/staged/CPU outputs, sets handles and staged masks in a GPU kernel, builds two pure-CUDA conditional children, adds a sibling CPU host node with parent D2H/H2D copies, and merges after all three dependencies. The CPU callback skips compute and freshly zeros output when it owns no routes. Split children write separate MoE outputs and Tail consumes the merge. | **Structurally implemented for the opt-in single partial layer; acceptance incomplete.** The CUDA topology test is out of sync with the current kernel signatures and modes, so current test coverage is inadequate. This structure is not the production path across all layers. |
| 3. External SSD service and readiness-driven dispatch/completion | `StorageService` has a bounded queue, a host worker, model/request generations, Ready/Error, atomic consumption, timeout retirement and exception handling. The loader copies published DDR views or uses `pread`; `CompletionGraph` owns stable pinned/device payload buffers and a ready event. CUDA work remains outside the storage loader. | **Partial.** The decode thread submits the request and calls `wait_consume_for()` before launching the continuation. This is a host-side synchronous orchestration boundary, not the design's short request callback plus independent readiness-driven completion submission. Resident warm-up overlaps the read, but its output is cleared/recomputed in the later parent graph. The path does not yet provide SSD→bounded DDR cache→CPU/staged GPU service across all layers/requests. |
| 4. Global bounded DDR, VRAM and pinned pools; consumer lifetime | Shape-compatible layers share `Qwen4ExpStageWorkService`. Each layer has two banks with epochs, expert IDs, copy events and token-completion holds. Reuse requires both consumer-token completion and completed DMA. | **Partial worker sharing; shared storage pools missing.** Every `Qwen4ExpStagePool::prepare()` still allocates `2 * topk` GPU slots plus two pinned bounce banks per layer. The completion upload is a separate single-flight arena. These do not implement the requested globally bounded VRAM/pinned pools or unified DDR cache. Existing layer identity is implicit in each pool; globally reused slots need explicit model/layer/expert/slot-generation identity. |
| 5. Speculative future-layer prefetch with exact demand fallback | The early-ready predictor and stage pool can prefetch a following layer, publish after all planes are copied, match actual routes by epoch/expert, and retain CPU fallback for an unavailable prediction. Route analysis reports cold recall, precision and estimated wasted bytes; late-publication experiments exist. | **Experimental and disabled.** Historical gains did not justify adoption. No accepted all-layer pipeline demonstrates useful bytes arriving before their consuming deadline. The staged storage queue is FIFO; a unified demand-over-speculation policy and global byte/queue budgets are not integrated. Prediction accuracy alone is insufficient. |
| 6. Choose CPU/staged GPU from measured completion time | The opt-in policy checks valid timing samples, warms CPU measurements, calibrates staging, and includes H2D, graph execution/dispatch and storage service time in its staged estimate. The newer code decides a CPU path before submitting storage. | **Prototype, disabled, no accepted model A/B.** Its observations describe complete suffix paths rather than a general per-route completion scheduler. It is not yet conditioned on global cache/queue state. For SSD-only CPU execution, storage delay must be included in that alternative too; the current CPU fallback assumes DDR is available. |

## Current graph topology

The inspected parent builder uses this dependency order:

```text
prepared completion upload → zero resident/staged/CPU outputs → GPU predicate
    ├─ IF any resident: resident CUDA child ────────────────────────┐
    ├─ IF any staged: payload unpack kernel → staged CUDA child ────┤
    └─ D2H activation/IDs/weights → cold-only host node → H2D ──────┤
                                                                  ↓
                                                        merge → common Tail
```

The payload unpack is a kernel inside a conditional, which is permitted.
Transfers to/from the CPU callback are in the parent graph. The host callback
is not nested in either GPU child. This matches the supplied structural rule:
[NVIDIA conditional-node requirements](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaConditionalNodeParams.html).
The callback must also avoid CUDA API calls and synchronization on unordered
CUDA work; independent host functions may be serialized:
[NVIDIA host-function contract](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html).

The source gives the CPU classifier a staged expert map derived from a sorted,
deduplicated publication vector, while the staged GPU dispatch addresses
payloads by original route slot. That currently suffices for CPU exclusion,
but the descriptor's `weight_slot` is not a demonstrated common physical-slot
contract. Resolve that mapping before using the descriptor for shared staging.

## Concrete test drift in the current tree

`tests/test_qwen4exp_conditional_graph.cu` cannot currently validate the newer
three-output topology:

- At line 134, `zero_args` supplies resident, staged and count. The production
  kernel now expects resident, staged, **CPU**, and count.
- At line 194, `merge_args` supplies two partial outputs, destination and count.
  The production kernel now expects **three** partial outputs, destination and
  count.
- At line 225, the resident-plus-CPU case `{1, 7}` expects mode `0`; the current
  bitmask predicate produces `1 | 4 == 5`.

CUDA kernel pointers are untyped at the graph-node argument interface, so a
successful C++/CUDA compile alone does not check these argument lists. Repair
this test before launching it. Add CPU-only and resident+staged+CPU cases,
duplicate slots, a skipped branch after a nonzero previous replay, and checks
that the CPU contribution reaches the final merge. The existing classifier
unit tests establish route bookkeeping; they do not prove full-model output
parity or CUDA consumer lifetimes.

## Validation and adoption

The accepted early-ready/resident-table/async-output bundle remains the
production reference. The historical 31.848→33.874 tok/s measurement and
byte-identical logits apply to that bundle and its recorded build/config.

The corrected DDR callback's historical A/B was byte-identical but 21.61%
slower. Dynamic staged arithmetic and speculative staging were rejected.
The later 14.37% slowdown measurement had all payload publications rejected,
so it did not benchmark the new resident+staged+CPU parent shown above.
None of those measurements proves acceptance of the current uncommitted
combined topology.

The user's first milestone is therefore **present in source structure but
not fully validated/adopted**. The external SSD service, completion upload
and lifetime helpers are useful foundations. The final all-layer shared-pool,
asynchronous three-tier design remains unfinished.

## Recommended next work

1. Repair the current CUDA topology test and establish a single immutable
   invocation mapping shared by CPU and GPU branches. Cover all ownership
   combinations, duplicate routes, generation changes and skipped-output reuse.
2. Validate the one-layer combined parent with a common Tail against early-ready
   using identical prefill, placement, weights and teacher-forced history. Run
   a separate no-dump, order-balanced A/B before adoption.
3. Replace the decode thread's storage wait with explicit pending requests and
   readiness/error-driven completion submission. Run available resident work
   once and retain its result through the SSD boundary. CPU fallback is valid
   only after its weights are available; unavailable routes must remain pending
   or fail the request.
4. Introduce globally bounded shape-class VRAM/pinned pools and a DDR cache,
   with stable graph addresses and ownership through the last consumer. Extend
   across layers only after slot-reuse and late-completion tests pass.
5. Add consumed-before-deadline byte accounting and demand priority, then
   evaluate prediction and adaptive placement from those measurements.

Reusable dispatch/completion graph segments are compatible with the user's
full-graph execution policy; this design does not require restoring the legacy
per-layer execution loop. No runtime defaults or implementation files were
changed by this audit.
