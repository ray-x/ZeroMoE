# Predicated Static Graph / Master-Graph Unification — Session Summary

Branch: `CPUOffload`. Nothing in this session was committed or pushed —
everything described below is sitting in the working tree (and one file in
the separate `beellama.cpp/ggml` checkout). Read this before continuing in
a fresh session.

## Origin

Started from a design doc pitching a "Predicated Static Graph" architecture
to eliminate host-orchestration overhead in the Qwen4Exp mixed CPU/GPU MoE
decode path (the same problem `doc/cpu-offload-design.md`'s "Unified-graph
follow-up" section already flagged as a later milestone, not yet
implemented). Several of that pitch's implementation specifics turned out
to be fabricated (wrong CUDA type names, a nonexistent `graphs` field on
`cudaConditionalNodeParams`, invented ggml function signatures/line
numbers) — nothing from it was used without independent verification on
real hardware first.

## What was validated in isolation first

A standalone project, **`/path/to/predicated-graph-poc`**
(separate from ZeroMoE, no shared code), built up the mechanism
incrementally, each step verified on this box's actual RTX 3090 before
trusting it:

1. **Predicated CPU host node** (activation-sized payloads): a graph host
   node that always launches but early-exits internally when there's
   nothing to compute (`<2us` measured). No CUDA graph conditional node
   needed here — conditional nodes were confirmed (empirically, not just
   from docs) to **not support host nodes inside their bodies**
   (`cudaGraphInstantiate` fails with "invalid argument" if you try).
2. **Real `cudaGraphCondTypeIf` conditional nodes** for large payloads
   (weight-tensor-sized transfers, tens of MB): proved a genuine skip (not
   "run and discard") — sentinel-buffer test confirmed the H2D copy simply
   never happens when the predicate is false, even immediately after a
   real transfer on the previous replay.
3. **Segmented master graph assembly**: prefix (normal stream capture) +
   device predicate kernel + conditional node (body populated via
   `cudaStreamBeginCaptureToGraph`, the API the original doc omitted) +
   suffix (another independent capture), all stitched with the explicit
   graph API (`cudaGraphAddChildGraphNode`) into one master graph.
4. **ggml-cuda's internal graph-handle stability**: empirically confirmed
   (against real ggml, not synthetic code) that `ggml_backend_cuda_context`'s
   cached `cudaGraph_t` is stable and safe to hold across replay-only calls
   (`cuda_graph_update_required == false`), only getting destroyed/recaptured
   on a real shape/property change.
5. **Real ggml graphs spliced into a real conditional node**: two
   independently-captured ggml graphs (standing in for `sgpu`/`scpu`) added
   as the two bodies of one conditional node, launched repeatedly with a
   toggling predicate — correct branch every time, other branch's output
   buffer provably untouched (sentinel poisoning check), one graph launch
   per token, zero host sync/branch.

That project's `tests/` (ctest, all passing) and `bench/` targets remain a
reusable reference if this mechanism needs re-validating later, e.g. after
a ggml version bump.

## Critical risk found and avoided

Before touching real ZeroMoE code, investigated `src/ggml_qwen4exp_graph.cpp`'s
actual `prefix -> read ids -> host branch -> sgpu or scpu` dispatch
(`qwen4exp_full_graph_decode_run_phase2`). Found that **`scpu`'s
`ggml_backend_sched_t` is genuinely multi-backend** (`{cuda, cpu}`, and it
actually gets split across both — it requires CPU-buffer-backed expert
weight tensors unconditionally). Harvesting a single `cudaGraph_t` for it
(the way the mechanism above works) would have **silently dropped whatever
ggml placed on the CPU backend** — a real, would-have-been-silent
correctness bug, caught via a dedicated investigation before any code was
written. `sgpu`, by contrast, only ever touches CUDA-backed tensors (its
"compact all-GPU path"), so it's safe to harvest.

**Decision (explicit, already made): unify only the `sgpu` fast path.**
`scpu` keeps its existing multi-backend dispatch unchanged. Fully unifying
`scpu` would require rebuilding its CPU compute as an explicit host node
(bypassing `ggml_backend_sched`'s automatic CPU/GPU placement entirely) —
bigger scope, deliberately deferred.

## What was actually implemented (this session)

| File | Status | What |
|---|---|---|
| `beellama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu` | modified (not committed there either) | New `zeromoe_ggml_cuda_get_captured_graph()` accessor, same pattern as the existing `zeromoe_ggml_backend_cuda_get_stream` bridge. |
| `patches/ggml-cuda-conditional-graph-harvest.patch` | new | Checked-in patch capturing the above, for the fetch+patch flow. Verified it applies cleanly, in sequence, on top of a fresh vanilla ggml + the two existing patches. |
| `CMakeLists.txt` | modified | Wires the new patch into the existing apply-once flow (same idempotent check pattern as the stream-bridge patch); adds `cuda/qwen4exp_master_graph.cu` to the source list. |
| `cuda/qwen4exp_master_graph.cu` | new | Device-side predicate kernel — replaces the host `std::unordered_set` membership check with a GPU-resident hot-set bitmap lookup, driving `cudaGraphSetConditional`. |
| `src/ggml_qwen4exp_graph.cpp` | modified | New `qwen4exp_master_graph` namespace (safety check via `ggml_backend_sched_get_n_splits`, hot-bitmap upload/cache, master-graph build/cache/launch) + the per-token dispatch block rewritten to try the fast path first, falling back to the original code otherwise. |
| `bench/qwen_cpu_moe_ggml_bench.cpp` | modified (earlier in session) | `d_model`/`d_ff`/`topk` made env-overridable (`ZEROMOE_QWEN_CPU_BENCH_D_MODEL`/`_D_FF`/`_TOPK`), defaults unchanged — used to get real GLM-shape (2048x4096) CPU FFN numbers instead of only the hardcoded Qwen4Exp shape. |

**Safety properties baked into the new code:**
- Fully opt-in: `ZEROMOE_QWEN4EXP_MASTER_GRAPH` env var, unset by default.
  Existing decode path is byte-for-byte unchanged unless explicitly enabled.
- Self-disabling: checks `ggml_backend_sched_get_n_splits(sched) == 1` for
  both `prefix` and `sgpu` before ever trusting the fast path. If either is
  ever multi-backend-split (e.g. a future config change adds CPU-affinity
  layers before the split point), it logs
  `[QWEN4EXP-MASTER-GRAPH] disabled for this session: ...` once and falls
  back permanently for that session — never guesses.
- `scpu` dispatch is completely untouched.

**Build status:** full `zeromoe` target builds clean (`cmake --build
build_ggmlcuda --target zeromoe`), binary runs (`--help` verified). The
host-node follow-up below includes the subsequent real-model decode result.

## Host-node parallel follow-up

The deferred CPU-offload path was exercised on the real Qwen decode engine with
`ZEROMOE_QWEN4EXP_HOST_NODE_PARALLEL=1`. Two small opt-in patches were added:

| File | Purpose |
|---|---|
| `patches/ggml-cuda-host-node-parallel.patch` | Reuses ggml-cuda's existing multi-stream graph optimizer for the explicit host-marker/shared-work branch. |
| `patches/ggml-cuda-host-node-parallel-gate.patch` | Allows only this marker optimizer during prefill; the unrelated experimental graph optimizer remains disabled unless `GGML_CUDA_GRAPH_OPT=1` is set. |

The synthetic branched-graph sandbox still passes its overlap check, and the
live 22/26 host-node run completed without a prefill crash, produced coherent
output, and logged all 26 host-node branches. The non-instrumented decode
measurement was approximately **77.0 ms/token (13.0 tok/s)**, versus the
serial host-node result of approximately **78.1 ms/token (12.8 tok/s)** and
the 22/26 baseline of approximately **35.3 ms/token (28.3 tok/s)**.

This is a structural success, not a throughput success. The shared GPU work
can overlap with the host callback, but the CPU routed FFN result is required
before the next layer can consume the residual, so most of the CPU callback
remains on the critical path. Runs with graph/layer phase instrumentation are
not performance measurements because the diagnostic brackets add substantial
overhead.

The direct native harness was rerun after the patch with
`CUDA_VISIBLE_DEVICES=0`; all available CPU, CUDA, GGML, IQ, GDN, attention,
and shared-expert tests passed. The only skips were tests requiring an
external model fixture or explicitly opt-in real-data inputs.

## Repo hygiene note (pre-existing, not something I changed)

`git status` on this branch shows most of the repo — including
`CMakeLists.txt`, `configs/`, `doc/`, and now `patches/` — as **untracked**
(`??`), not modified. These files were apparently never `git add`ed in this
repo's history. This means my edits to `CMakeLists.txt` don't show as a
clean `git diff` against any prior commit; a first real commit here would
need to add a large amount of pre-existing untracked content, not just the
new work from this session. Worth sorting out before this branch is ever
merged or shared.

## Next steps

1. **Keep the host-node parallel path laboratory-only.** It is correct and
   useful for validating graph topology, but its measured gain is too small
   to justify a production rollout. Do not enable it by default.
2. **Oracle-parity check (do this first, before anything else).** Pick a
   fixed prompt, `temp=0` (greedy), run once with
   `ZEROMOE_QWEN4EXP_MASTER_GRAPH` unset (baseline) and once with it set,
   diff the generated token sequence exactly. Watch for the
   `[QWEN4EXP-MASTER-GRAPH] enabled: ...` log line to confirm the safety
   check actually passed (vs. silently falling back to the old path every
   token, which would make the parity check meaningless).
3. **Perf measurement**, only after parity holds: tok/s and per-token
   latency, flag on vs. off, on a workload where most tokens route entirely
   inside `qwen4exp_partial_layer_gpu_ids` (the case this unifies) —
   compare against the two-scheduler-submission cost the function's own
   comment already calls out as "the real, measured cost this feature
   trades against."
4. **If parity + perf both check out**, decide whether to pursue the
   `scpu` unification too (rebuild its CPU dispatch as an explicit host
   node so both branches can live in one master graph) — bigger, deferred
   scope from earlier in this session.
5. Separately, and not blocking on the above: `changelog.txt` documents an
   **unresolved** coherence bug (eval-callback registration possibly
   silently disabling CUDA graph capture, diverging decode output) in this
   same area. Explicitly treated as a known, separate issue this session
   (not investigated further) — still open.
6. Decide what to do with `/path/to/predicated-graph-poc` — keep
   it as a reusable mechanism-validation harness (recommended, e.g. for
   re-checking the ggml-handle-stability assumption after a ggml version
   bump), or archive it once the real integration is trusted.
7. Nothing here has been committed. When ready, review the diff yourself
   before committing/pushing — this was explicitly not done automatically,
   per the repo's own AGENTS.md guidance on AI-generated changes.
