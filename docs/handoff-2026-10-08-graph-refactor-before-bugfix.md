# Handoff: graph refactor branch before the coherence bug-fix split

**Scope clarification:** this handoff records historical branch-point work and
its measurements. The current checkout also contains newer uncommitted
resident/staged/CPU graph composition. Its predicate supports independent
branches, so the historical statement about mixed-route CPU fallback does not
describe all of the current source. See
[the design conformance audit](graph-design-conformance-2026-10-08.md) for the
current code, missing requirements, and a concrete CUDA test-signature mismatch.

## Branch and snapshot

The graph-refactor work was on `cleanup/ggml-exl3-only`. At the bug-fix
worktree split, that branch was at `521d67a773b3b9e22f48bb5fcad0f6840ca663a6`
(`Fix Qwen4Exp Option-B staged arena ownership`, 2026-10-07 01:32 +1100).
The isolated bug-fix checkout was created from that commit. The current main
checkout has since advanced to `11c6100` on the same branch; it also contains
later commits `35b6d14` (skip Phase-1 prefill router capture under Phase 2),
`5aa754a` (load staged payloads from resident DDR rather than `pread`), and
`11c6100` (enable early-ready by default). Treat the current root checkout as
a later, separate state when reproducing the original bug-fix work.

There is no surviving branch named `graph-refactor`. The accepted scheduler
work was recorded as complete through `490380a` (`Add graph tier telemetry
and adaptive policy foundation`); the staged/Phase-2 experiments continued
after that, through the `521d67a` arena-ownership fix. The current surviving
bug-fix worktree is `/path/to/moestream-claude-perf`, branch
`claude/staged-coherence-phase2`, still based at `521d67a`.

## Completed and accepted production path

The production scheduler change kept Option B as the static placement policy
and added early-ready CPU work for cold experts. Its accepted bundle is:

- a private CPU graph per split layer, started when its input is ready while
  queued GPU work continues;
- immutable resident routing/remap tables uploaded once;
- pinned cold outputs with asynchronous upload and an event join before the
  merge.

The measured Q2_0 RTX 3090 Option-B 0/44/4 set contains 8,421 resident experts
in 10.84 GiB; layers 0, 1, 2, and 22 stay on CPU. Six order-balanced pairs
measured 31.848 tok/s for the original scheduler, 33.294 tok/s with resident
tables, and 33.874 tok/s with async output added. The final bundle was +6.361%
over baseline (6/6 wins), and all three paths had byte-identical 64-record
logits. See `docs/handoff-2026-10-05-graph-refactor-complete.md` and
`logs/option_b_scheduler_final_three_arm/`.

That production result is distinct from the opt-in graph experiments below.
The accepted early-ready implementation did not reconstruct the graph per
token. It relies on the patched GGML scheduler hook in
`patches/ggml-scheduler-early-ready-layer13.patch` and the patched integration
checkout documented in the final handoff.

## Experimental work completed on the branch

The branch also accumulated foundations for partial-layer execution:

- immutable route-slot classification, exact-once projection, and CPU tests
  for all-hot, all-cold, mixed, duplicate, and invalid routes;
- an opt-in DDR partial-layer callback using a bounded CPU graph;
- a payload-backed storage/completion service with model/request generations,
  Ready/Error outcomes, stale-completion rejection, a persistent upload graph,
  and conditional staged-GPU child composition;
- a shared shape-class worker service. GPU expert arenas and DMA banks remain
  per layer to protect consumer lifetimes;
- an opt-in adaptive CPU/staged timing policy and route/staging telemetry;
- a CUDA topology test for resident/staged conditional children with a parent
  host node. The staged arena alias that let a child overwrite an Option-B
  resident tensor was fixed at `521d67a`; bounded forced-cold runs then
  preserved all 198 resident planes and matched five generated tokens.

The graph API review rejected stream priorities as a DMA-priority control,
kept stream capture under GGML ownership, and deferred CUDA graph memory nodes,
TMA, and GDS/cuFile. SSD access on this machine must remain SSD→host memory,
then CPU or H2D; no GDS path is available.

## What was still pending or rejected at the split

The production early-ready bundle was accepted. The broader partial/staged
graph design was not production-ready:

- The DDR partial-layer callback passed a 64-step forced-history correctness
  comparison, but its corrected six-pair A/B measured 34.05 tok/s early-ready
  versus 26.70 tok/s partial callback (−21.61%, 0/6 wins). It remained
  opt-in/rejected.
- Layer-13 dynamic staging measured −0.432% (2/6 wins); all-eligible-layer
  staging measured −12.636% (0/6 wins). The predictor alone measured −0.049%
  (3/6 wins). These modes remain disabled.
- Phase-2/conditional composition was fail-closed unless its conditional
  children were single-CUDA graphs. A no-Option-B run had 28/69 scheduler
  splits, so it fell back to host dispatch between full-graph submissions; it
  did not exercise the CUDA conditional body.
- The SSD/DDR service and completion primitives had unit and topology tests,
  but an end-to-end staged model path still needed coherence and performance
  validation. Adaptive CPU-versus-staged selection likewise lacked a valid
  runtime A/B. A shared GPU staging arena was not implemented.
- Qwen full-graph prefill/decode was enforced, but the user's full-graph-only
  policy was not migrated across every supported MoE family; GLM5NEXT and
  DeepSeek-V4 had only partial graph adapters, and other families still used
  layered execution.
- The all-layer 48-layer / roughly 14.5k-expert placement had single-run
  results near 39 tok/s but no matched six-pair early-ready comparison against
  the 0/44/4 placement. Do not combine those results as if they were one A/B.

Later work isolated a Phase-2 prefill graph perturbation: marking layer-13
router IDs/probabilities/bias as outputs changes allocator reuse/fusion.
Skipping that Phase-1 capture under Phase 2 makes prefill bit-identical to
early-ready; forcing the old capture reproduces the divergence. A later
six-pair Phase-2/staged A/B measured −14.37%, but every candidate completion
was rejected (`published=0`) and fell back to the CPU suffix. It therefore
measured the mixed-route fallback, not staged-GPU compute. That run also used
the pre-short-circuit code and read about 0.86 GB of rejected payload per
candidate run. The isolated bug-fix worktree has a later resident-overlap
read short-circuit, but a matched A/B of that version has not completed.

The residual step-27 top-1 flip had a reference margin of about 0.00398 and
also appeared in an early-ready-only Option-B noise-floor sweep. This weakens
the stale-state/corruption hypothesis, but staged-GPU coherence is not proven
when the staged branch never publishes. Separately, ZeroMoE prefill still
differs from standard llama.cpp GPU-dense prefill at layer 0 (RMS `2.11e-5`,
max `1.31e-3`); the user's llama.cpp prefill gate remains unmet.

## Resume checklist

1. Use the isolated bug-fix worktree based on `521d67a`, not the current root
   checkout, unless intentionally testing the later `cleanup/ggml-exl3-only`
   commits. Preserve unrelated ExLlamaV3 and generated files.
2. Keep the accepted early-ready path as the comparison baseline. Use the
   exact `configs/testing/qwen4exp_q2_0_option_b_rightsize_split44_x0_cov80.toml`
   placement for both arms; it is 8,421 experts / 10.842 GiB.
3. Re-establish stable CUDA discovery before inference. The most recent target
   build completed with `-j10`, but its launch returned
   `cudaGetDeviceCount()==0` and the following `nvidia-smi` query lost driver
   communication. OOM-kill count was zero. Do not interpret that as a model
   result.
4. First run exact prefill equality between early-ready and Phase 2, then the
   standard llama.cpp prefill gate. Keep correctness dumps separate from
   no-dump timing. For timing use six order-balanced pairs, `-j10` for builds,
   ten CPU threads, and record VRAM, RAM, and OOM counters.
5. The performance work is mixed-route execution: compose resident GPU and
   staged/CPU cold route contributions once each, while preserving original
   router weights and a common tail. Keep the overlap check until exact-once
   mixed-route integration tests pass. Do not enable staging/adaptive policy
   in production until the route branch actually publishes and beats
   early-ready without coherence loss.

No commit was created for this handoff. The existing worktree and run logs are
preserved; this file is a navigation aid, not a claim that the pending gates
have passed.
