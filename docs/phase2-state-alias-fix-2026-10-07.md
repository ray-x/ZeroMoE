# Qwen4Exp phase-2 route-switch state fix (2026-10-07)

## Finding

The isolated `bugfix/staged-coherence-521d67a` branch exposed a second
phase-2 state bug after the Prefix handoff correction. In the completion-staged
graph setup, the GPU split child was followed by a `Tail` graph, while the cold
fallback used a separate `SuffixCpu` graph. Both graphs allocated their own
KV and recurrent-state buffers. When decode routing changed from all-resident
to a route needing CPU fallback, `SuffixCpu` resumed with stale state for
layers after the split.

The first observed route change was at decode position 29, immediately after
the split at layer 13. Before the fix, the first large hidden-state divergence
was layer 14 (RMS `1.31e-3`, max absolute delta `2.11e-2`), and logits first
disagreed at step 3 (KL `0.233`). The fix builds `SuffixCpu` with the `Tail`
graph as its state source, so both graph handles bind to the same post-split
KV/GDN/PLE storage. The current dirty workspace already contains this
`tail.get()` state-source relationship, along with newer route-graph work.

## Verification on the isolated 521d67a branch

The branch binary built with `cmake --build build_ggmlcuda --target moestream
-j10`. To avoid recompiling an unchanged GGML CUDA backend, this temporary
build linked the existing GGML 0.22 libraries built from the matching
Beellama source; the MoEStream target itself compiled and linked in the
isolated worktree. GGML CUDA graph support was confirmed active in the runtime
log.

The monitored 64-token, same-binary teacher-forced pair compared the validated
early-ready path with the phase-2 staged candidate. It completed with no OOM
kills and no supervisor stop condition. Top-1 agreement was 62/64 (96.875%);
the two remaining disagreements were near-ties at steps 6 and 27. At step 6,
the reference margin was `0.0131`, its top token ranked second in the
candidate, and KL was `1.48e-4`. The split boundary itself matched to float
noise: at position 29, layer 13 RMS was `3.59e-10` and layer 14 RMS was
`1.30e-9`. Later layer-14 differences stayed around `1e-5` to `3e-5`, rather
than the `1.3e-3` first mismatch seen before the state-sharing fix.

The no-dump, six-pair performance check on this same 521d67a candidate measured
17.77 tok/s for early-ready and 15.06 tok/s for phase-2 staged (`-15.25%`, 0/6
paired wins). This is a timing-valid result for that isolated branch. It also
showed that mixed resident/cold routes were not using the staged GPU branch:
completion publication logged `published=0`, followed by CPU dispatch. The
route predicate in this version chooses GPU execution only when every route is
resident or every route is staged; mixed routes fall back to the CPU suffix.
That explains the speed gap and remains a separate limitation of the
521d67a staged path. The current dirty workspace has newer route-graph code,
so these throughput numbers do not characterize that newer code.

The default early-ready scheduler remains the production path. This
verification fixes and measures the isolated state-continuity defect; it does
not establish that the staged-GPU path is ready to become the default.

## Reproduction artifacts

All generated logs and hidden-state files were retained in
`/tmp/moestream_bugfix_521d67a_20261007/logs/`:

- `phase2_bugfix_hidden_pair_20261007_r5/` is the pre-state-sharing reference.
- `phase2_bugfix_hidden_pair_20261007_r7/` has the post-fix 64-token logits,
  per-layer hidden snapshots, comparison JSON, and resource samples.
- `phase2_bugfix_perf_pair_20261007_r1/` has the six-pair no-dump timing
  results and monitored resource samples.

The r7 hidden comparison is
`phase2_bugfix_hidden_pair_20261007_r7/hidden_comparison.json`; the logits
comparison is `phase2_bugfix_hidden_pair_20261007_r7/comparison.json`. The
runner script is the branch-local
`bench/qwen4exp_hidden_state_pair.py`.
