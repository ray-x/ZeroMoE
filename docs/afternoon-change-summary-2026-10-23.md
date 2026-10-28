# Afternoon change summary — 2026-10-23, Sydney

Scope: work performed by Codex after 12:00 PM AEDT in
`/path/to/ZeroMoE/.worktrees/mimo-v26-flash`, branch `feat/mimo-v26-flash`.
Paused at the user's request after the latest test finished, approximately 16:48 AEDT.

**The refactor is not complete.** MiMo passes the phase-cache correctness/memory test. Ling completes the
allocation transitions and reproduces its own repeated-request logits, but fails the strict comparison
against the fixed-cache control. The extension remains opt-in and defaults to zero.

No commits or merges were made. All changes are in the existing worktree on top of
`6ad3a87d80f9d37f36e336e303516e9da428d08b`.

## Work inherited, rather than originated this afternoon

The worktree already contained unfinished changes for `embedding_device`, CPU embedding defaults,
`FullGraphModel::embedding_on_gpu()`, `moe_cache_reserve_gb`, the MiMo dual-GPU reserve settings,
an embedding A/B option in `bench/mimo_survey.sh`, and design notes. Those changes were preserved and
continued. `.ccls-cache/` and the user's `mimo_tunning.md` transcript were also already untracked.

## 1. Shared resident-memory policy

- Added `memory/resident_memory_policy.hpp` for common embedding-placement and expert-cache budget decisions.
- Refactored both dedicated-vault and dual-role GPU cache sizing to use the same calculation.
- Fixed exhausted auto-fit budgets being interpreted as the legacy "zero means auto" sentinel, which could
  allocate most of VRAM instead of disabling the cache.
- Capped vault allocation by measured free VRAM. Existing reserve accounting is retained: total VRAM for a
  dedicated vault, and free VRAM after dense allocations for a dual-role GPU.
- Added rejection of unknown `embedding_device` values and negative/non-finite/unrepresentable cache reserves.
- Added `tests/test_resident_memory_policy.cpp` and CLI rejection checks.

## 2. Accurate coverage preflight

- Changed MiMo and GLM/Ling preflight to count the union of actual expert identities in VRAM and ready in DDR.
- Capacity, duplicate identities, and uncovered identities are now distinct. An empty pool, an in-flight load,
  or duplicated hot experts cannot make the report claim full coverage.
- Also check the GLM cold-only case without a partial hot-set descriptor.
- Added overlap and empty-DDR regression cases to `tests/test_graph_settings.cpp`.

This exposed that the Ling test profile really covers about 68% of its routed expert identities, so its
ordinary workload can have SSD demand misses. Extension refills are a separate operation and issue no
demand/model-file loads.

## 3. DeepSeek shared-adapter migration

- Routed DeepSeek V4.1's prefill and decode host callbacks through `CacheBackedMoeRuntime`.
- Preserved the native cache, kernels and claim/hold/transfer/compute/release implementation underneath.
- Removed DeepSeek's direct `moe_forward_gguf` exception from the conformance script and added it to the
  families checked for use of the policy boundary.
- Corrected stale comments claiming `CacheBackedStrategy` had no callers. It is live; extracting the native
  pipeline's asynchronous lifecycle is still separate, unfinished work.

## 4. Read-only model-sizing advisor

- Added `gguf_tool plan --embedding-device auto|cpu|gpu`, sharing the embedding policy with the engine.
- The advisor excludes CPU embeddings from its dense VRAM estimate and reports payload-equivalent extra
  expert slots. It reads model metadata; it does not run inference or modify configs.
- Added `tests/test_gguf_embedding_plan.py` with a synthetic GGUF fixture.

| Model | Embedding VRAM avoided, decimal GB | Approximate extra expert payloads |
|---|---:|---:|
| MiMo V2.6 Flash | 0.664 | 61 |
| Ling 3.1 Flash | 0.684 | 127 |
| DeepSeek V4.1 Flash | 0.217 | 16 |
| GLM 5.3 Flash | 0.520 | 74 |
| Qwen 3.8 Flash | 0.358 | 202 |

These are metadata estimates before pool alignment and workspace constraints, not measured speedups or
instructions to pin every available slot. Baseline configs and pin counts were not retuned.

## 5. Shared phase-cache allocator and graph lifetime

Added `models/graph/phase_expert_cache.hpp`, used by MiMo and the GLM/Ling in-graph path:

- A separate decode-hot CUDA allocation extends the permanent cache without moving its addresses.
- Before prefill, graph borrowers are retired and the extension is drained/freed.
- After prefill workspace is retired, the same selected experts are restored directly from pinned, ready DDR.
- Selection excludes permanent hot experts. Pinning the DDR sources prevents prefill eviction from creating
  unexpected SSD reads during restoration.
- Per-layer strides follow the engine's row-alignment rules, including mixed quantization.
- A shared additional GPU branch uses the extension. Its identities are removed from cold ownership without
  changing authoritative route slots or weights.
- `memory/phase_route_partition.hpp` validates disjoint ownership before mutating maps.
- `memory/placement_epoch.hpp` gives graph borrowers leases and monotonically increasing generations.
  Retirement refuses while a graph borrower is live.
- `GraphCore` stores the generation and lease. MiMo rebuilds on generation changes; GLM/Ling checks the
  generation in prepare/reuse and execution paths. KV/recurrent state remains with its original owner.
- Rollback rejects stale bindings on cleanup failure; insufficient headroom or eligible DDR sources leaves
  the extension inactive rather than demand-loading weights.

Common settings:

```text
ZEROMOE_PHASE_CACHE_MIB=0                 default: disabled
ZEROMOE_PHASE_CACHE_RESERVE_MIB=512       free VRAM reserved for decode workspace
```

Added route-conservation and placement-lease tests in `tests/test_phase_routes.cpp`, and updated the settings
tests/default snapshot. This is not yet automatic hardware-cost planning or migration of Qwen/DeepSeek to
this phase allocator.

## 6. MiMo private-embedding bug found by CUDA validation

The first long-prefill control ran out of VRAM even after the common engine stopped uploading embeddings.
MiMo's model-state builder was privately uploading every non-expert resident tensor, including the unused
embedding table. It therefore consumed the recovered 0.66 GB again.

MiMo now omits `token_embd.weight` from graph-private upload/accounting because embedding lookup happens
before its graph. An explicit GPU embedding option still controls the engine's copy. The 693-token
512+181 chunked prefill subsequently passed.

## 7. Targeted CUDA validation and diagnostics

- Added `bench/phase_cache_transition.py`: short → chunked long → repeated short requests under the existing
  kernel-enforced memory caps. It captures logits and tokens per request, placement events, residency counters,
  and sampled GPU memory.
- Fixed its EOS handling: `REQUEST-DECODE-OUTPUT` is not guaranteed for short/EOS responses, so completion
  uses `REQUEST-DECODE-WALL`.
- Added saved-control reuse with a configuration check and a separate assist-isolation option.
- Strengthened comparison from first-step/prefill logits to **every decode frame**. Added
  `ZEROMOE_TEACHER_FORCE_REQUEST_DIR` to the existing diagnostic teacher-forcing mechanism; the original
  single-file mode remains unchanged when the new variable is unset.
- Added `tests/test_phase_cache_validation.py`, including the regression where prefill matches but a later
  decode frame differs. First-step matching alone must not be accepted as decode correctness.
- Added final offload-store summaries to make decode loads observable at shutdown.

## Results at pause

| Check | Result |
|---|---|
| Release engine build | Passed; `build_mimo/zeromoe` last built around 16:27 AEDT |
| Resident-memory policy tests | Passed |
| Settings/default-snapshot/coverage tests | Passed |
| Layer/state tests | Passed |
| Route ownership and placement-lease tests | Passed |
| Synthetic GGUF embedding advisor test | Passed |
| Validation-harness decode regression tests | Passed |
| Conformance | Passed, 27 graph files |
| MiMo complete teacher-forced decode/transition test | Passed |
| Ling complete teacher-forced decode/transition test | **Failed numerical gate**, including with GPU assist disabled |

MiMo: 117 extra experts, 1,070,393,600 bytes; three restores, 89.9–96.3 ms each; releases 0.6–2.6 ms;
zero decode store loads; sampled peak VRAM 23,628 / 15,356 MiB. All 18 checked frames pass the 0.999
cosine floor (minimum 0.9993794). The repeated short request is bitwise identical over all eight frames.

Ling, assist disabled: 314 extra experts, 1,071,420,320 bytes; three restores, 187.7–199.1 ms; releases
2.15–2.24 ms; sampled peak VRAM 22,138 / 13,958 MiB. Its repeated short request is bitwise identical, but
phase versus fixed-cache minimum cosine is 0.9932799 for the short request and **0.9732832** for the long
request, below the required 0.999. Maximum absolute logit delta on the long request is 3.7901.

The root cause is not isolated. It may involve quantized CPU/GPU computation differences or the new
branch/layout/routing; it must not be dismissed as precision noise without verification. No confirmed
steady-state decode speedup or regression has been established. No frozen model performance gates were rerun.

See [the handoff](handoff-2026-10-23-phase-cache.md) for artifacts, commands, constraints and the next investigation.
