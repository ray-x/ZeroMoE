# Graph-native MTP: adding it to another model

The speculation driver, draft wrapper and reporting are model independent. A
model supplies a target, a draft runtime and an embedding lookup.

| Piece | File | Model-specific? |
|---|---|---|
| Accept/reject math, checkpoints, draft rebuild, EOS/capacity | `models/graph/mtp_speculation.hpp` (`speculate`) | no |
| `SpecDraft` over a graph draft runtime | `models/graph/mtp_graph_draft.hpp` (`GraphMtpDraft`) | no |
| Draft graph runtime (K/V, rewind, scheduler, admission) | `models/graph/mtp_graph_runtime.{hpp,cpp}`, `mtp_graph_contract.hpp` | no; takes a model `Builder` |
| Counters and `[MTP-GRAPH]` line | `core/mtp_graph_report.hpp` | no |
| Draft graph builder | `models/qwen/qwen4exp_mtp_graph.cpp` | yes (the NextN block) |
| `SpecTarget` over the production full-graph decode | `core/qwen4exp_graph_mtp.cpp` (`GraphTarget`) | yes |
| Distribution (penalties, temperature, top-p) | `core/generation_loop.hpp` | shared |

## What a model implements

1. **Draft builder** for `MtpGraphRuntime::create(shape, budget, builder)`. The
   draft gets its own K/V and is rewindable. Weights are uploaded to the
   device; the vocabulary head must run on the GPU too (a borrowed host head
   cost 13.6 ms per draft step on Qwen4Exp, against 0.4 ms for the draft layer).
2. **`SpecTarget`**:
   * `step(token, position, wide, logits)`: one token through the production
     decode graph; `wide` is the pre-head residual the draft consumes.
   * `save` / `restore` / `release`: a checkpoint of everything a step mutates
     that is not rewritten by position (recurrent state, conv history, n-gram
     history). KV that is written at `position` and masked above it needs none.
   * optional `supports_batch()` / `step_batch(tokens, position, wide, logits)`:
     see below.
3. **Prompt export**: prefill must leave the wide residual of every prompt token
   and the last-token logits (`SpecPrompt`).
4. **Admission order**: build the target's decode graph before the draft claims
   device memory, or the draft starves it and the target silently changes
   configuration.

## Stepwise vs batched verification

The target is authoritative either way; output is exact. They differ in cost.

* **Stepwise** (default) judges each draft as soon as the step that yields its
  distribution has run and stops at the first rejection. The live state then
  already stands after the accepted prefix, so there is no checkpoint, no
  restore and no wasted step: target steps equal emitted tokens. MTP in this
  mode costs plain decoding plus the draft and a fixed setup, so it can match
  but never beat plain decoding; it is the correctness reference and the
  fallback.
* **Batched** costs one pass over `k+1` tokens plus, on a rejection at `a < k`,
  a restore and a replay of `y..d_a`.

Batching pays off only if a `k+1` token pass costs well under `k+1` steps.
With acceptance `α` and `k` drafts a cycle emits `E = (1-α^(k+1))/(1-α)`
tokens (about 3.3 at `α = 0.8`, `k = 3`). Implement `supports_batch()` for a
model only after measuring `T(k+1)/T(1)` on the target placement you care
about, and require `T(k+1)/T(1) < E` with margin for the draft and replay.

Qwen4Exp on the 3090 hybrid recipe (CPU-resident routed experts) is the
counter-example: a step is dominated by CPU expert work that scales with the
number of distinct experts, so `T(4)/T(1)` is close to 4 and batching cannot
help. It helps when the routed experts are device resident.

## Measurements (Qwen4Exp, Q2_0, RTX 3090 only, 13,000 hot experts, 3 drafts)

| | rate |
|---|---|
| plain decode | 48 tok/s |
| MTP stepwise, host-CPU draft head | 18.7 tok/s |
| MTP stepwise, GPU draft head, verify all then restore | 32 tok/s |
| MTP stepwise, lazy verify (no waste, no restore) | **36.8 tok/s** (199 target steps for 200 tokens) |

Draft step 14 ms -> 1.3 ms; target step 20-22 ms; ~0.8-1 s one-time setup.
Acceptance 74-94% by draft count (k=1 94%, k=3 80%, k=4 74%); rate is ~36
tok/s for k=1..4 because the floor is plain decoding plus draft and setup.

## Why batching does not help the 3090 hybrid recipe (measured)

Router ids of a 199-token plain run (a temporary per-token id dump added to the
router-usage tap in `qwen4exp_cuda_usage_track_callback`, not committed; 13,000 hot experts): 0.82 of the 10 routed experts per layer are
CPU-resident per token, and consecutive tokens barely share them. Distinct
CPU experts per layer for a window of 1/2/3/4/5 tokens: 0.82/1.49/2.09/2.67/3.23,
so a 4-token pass needs 3.25x the CPU expert work of one token. Even if only
half of a step is CPU expert time, the expected MTP speedup is <= 1.03x
(k=1) and falls with larger k; at 90% CPU share it is 0.68-0.84x. No local
configuration keeps the routed experts on the GPU (models are 62-70 GB, the
3090 has 24 GB), so batched verification is not built for Qwen4Exp.
