# Partial/staged graph worktree bisect (2026-10-06)

This note records the bounded CUDA checks used to locate the first correctness
and decode-cost regressions without touching the main checkout. Native
`git worktree add` was unavailable because this environment exposes `.git` as
read-only, so the checks used an isolated clone under `/tmp/moestream-bisect`.
The clone received only the historical EXL3 source files needed to make old
commits link; those files are unrelated to the Qwen graph result.

## Reference run

The clean teacher-forced A/B is in
`logs/option_b_partial_staged_clean_20261006/`:

| arm | decode/core timing | result |
| --- | ---: | --- |
| accepted early-ready baseline | 32.397 tok/s, 30.16 ms core | reference top-1 8160, 369, 264, ... |
| current staged candidate | 12.851 tok/s, 77.21 ms core | first disagreement at step 1; candidate top-1 8160, 8160, ... |

The logits agree at step 0 and first diverge at step 1 (reference 369,
candidate 8160). The candidate's staged path was exercised: a Ready completion,
mixed/pure staged route modes, and `dispatch=staged-gpu` all appear in the log.
The 63 measured phase-2 records have a median of 75.65 ms and a mean of
77.05 ms, so this is both a parity failure and a large decode regression.

## Historical checkpoints

The following checks used the same prompt, Q2_0 model, ten CPU threads, and a
short four-token teacher-forced continuation. They are diagnostic checkpoints,
not acceptance runs: old commits were built with a compatible CUDA source list
and their original staged arena sizing.

* `5c87237` (parent of the scheduler refactor), with phase 2 disabled and
  early-ready disabled, produced `8160, 369, 264, 4145` at about 70 ms/token.
* `b1a2fdf` with phase 2 enabled and a forced zero coverage threshold produced
  the same first four tokens. Its phase-2 records were 66--83 ms/token.
  This shows that the phase-2 split itself is already in the slow legacy
  scheduler regime when early-ready is disabled, but it is not the source of
  the step-1 mismatch in the accepted baseline comparison.
* `c9685b9` with phase 2 enabled but without a staged completion also produced
  the reference first four tokens (`8160, 369, 264, 4145`).
* `080b65e` and `0f958dd`, with the completion/staged environment enabled and
  the 64-expert budget, did not reach decode. They fail while constructing the
  completion child: the expert arena is filled to its planned capacity and the
  additional roughly 4.6 MiB payload-backed expert slab cannot be bound. This
  is an arena-admission bug at the staged completion boundary, not a logits
  result.

## Conclusion

There is no valid commit-level parity bisect across the staged series because
the historical staged commits fail before replay under the current 64-expert
budget. The earliest successful staged replay is the current post-fix tree,
and it diverges at decode step 1. The slowdown has two parts:

1. Disabling the accepted early-ready scheduler moves the run into the old
   phase-2 scheduler regime (about 66--83 ms/token in the historical check).
2. The current staged completion path adds its own graph/storage orchestration
   and measures about 77 ms/token, but it is not parity-safe yet.

The production decision remains unchanged: keep the early-ready private CPU
graph, immutable resident routing tables, and asynchronous cold-output upload
as the default. Do not adopt staged completion until its arena admission and
step-1 parity are fixed and rechecked against the same early-ready reference.

## Layer-13 boundary evidence and bisect correction

The last currently verified good source checkpoint is commit
`c9685b9fd4471e2c7fb1d6e718165c806c0f71b8` (`c9685b9`). Its short Phase-2 run
produced the expected first four tokens (`8160, 369, 264, 4145`). The temporary
checkout used for that run (`/tmp/moestream-known-good-phase2-20261006`) is no
longer present in the current environment, but the commit object remains
reachable from the current repository history. Treat it as the last verified
good checkpoint, not as proof that every later commit is bad.

There is no verified oldest bad commit yet. Commits `080b65e` and `0f958dd`
fail while constructing the completion child at the tested 64-expert budget,
before decode, so they cannot establish a logits regression. The first
decode-bad result is from the current working-tree Route-graph candidate on
top of `4651151`, with logits matching at step 0 and disagreeing at step 1.
The Route graph/mode is a candidate boundary, not a proven cause; `4651151`
itself has not yet been run as the matched staged control. `080b65e` adds
`SuffixStaged`; it does not add a `Phase2Mode::Route`.

A two-token staged smoke is retained at
`logs/option_b_partial_staged_routegraph_smoke_20261005/`: it records staged
child construction, a Ready payload completion, and tokens `8160, 369` (one
decode step). Its manifest's engine SHA256 is
`6b1070556d2ddc60cb2bb4197cb6a503a9b6cf4eaa8919d05d14d28993d54299`, which
matches neither the current `build_ggmlcuda/moestream`
(`e8ecea9c1b12f881ca6f762ac6e0fa0eaf013d0c734235dc8ff89b5f939ec805`) nor the
currently present upstream-master binary (`033e4ef3e00168ecb5723a6a5407322d4b4b8239f7e051b87c9cb7de8786332e`). Its TOML comment labels it an upstream-master build, but the exact binary path and source revision are absent. This smoke proves only that an unidentified binary ran one staged decode step; it is not evidence that any post-`c9685b9` commit built or decoded. The later 64-token matched run from the dirty current worktree built and reached decode but diverged at step 1.

The all-layer decode dump is not valid for locating a first bad layer. The
Phase-2 file `/tmp/alllayers-phase2.step28.f32` is only 13 layers long, and
layers 3--12 are filled with the same `0x7fffffff` poison word. Do not infer
that Layer 3 is the model's first divergence from this file. The prefill
snapshots are usable: they match bit-for-bit through layer 34; the first
non-identical values are at layer 35 (17,999/286,720 values, maximum delta
`2.38418579e-7`). This small drift is not the observed decode failure.

The strongest valid decode-boundary evidence is from
`/tmp/preffn-full.engine.log` and `/tmp/preffn-phase2.engine.log` at the first
decode position (28):

* The ordinary full graph routes Layer 13 to
  `[273,50,385,387,35,347,93,286,6,423]`.
* The Prefix graph's exported route IDs are still `[0,1,2,3,4,5,6,7,8,9]`;
  its probabilities were logged as zero. They are not a usable route result.
* The isolated Route graph emits
  `[195,329,118,128,420,412,220,367,481,472]`.
* The saved ordinary and Route graph router inputs have the same 2,560-F32
  shape but no bitwise-equal values; their measured RMS difference is `0.5204`
  and maximum absolute difference is `5.7907`. The first generated token is
  `8160` in both runs, so this is not explained by feeding different generated
  tokens at the first decode boundary.

This places the observed wrong route before expert execution, payload unpack,
and resident/staged merge. It narrows the fault to the split-layer input or
its publication. The current source refreshes the persistent handoff after
replay by copying from `prefix.final_streams`, an intermediate whose GGML
allocator lifetime does not guarantee post-replay contents. The graph-owned
handoff probe avoids that refresh, but the latest probe read `0/10240` finite
values from the persistent destination. Thus the lifetime hazard is a strong
lead, while the internal graph copy is also not yet proven correct; neither
path is accepted as a fix.

The shortest next correctness check is to capture, from one matched first
decode step, both the Prefix post-attention residual and its persistent
handoff destination, then compare those with the ordinary full graph's
Layer-13 post-attention residual. Only after those three values agree should
the Route graph's HC-mixed router input and IDs be compared. Keep this to the
existing two-token, memory-traced run; the all-layer writer should remain off
until its snapshot copies are made explicit graph outputs and poison-checked.

## Matched position-28 handoff result

The two-token teacher-forced rerun uses the same 28 prompt IDs, Q2_0 profile,
10 CPU threads and first continuation token (`8160`). `--ngen 2` is required:
`--ngen 1` returns the first token from prefill and never executes decode
position 28. The baseline produced `8160, 369`; Phase 2 produced `8160, 8160`.

At position 28, the ordinary full graph's Layer-13 pre-FFN residual has FNV64
`0x3fdf5513cc863a9d`. The finite Prefix `final_streams` residual has FNV64
`0xfc837c3f0e4322cd`. Comparing the saved 10,240-F32 arrays gives 0 exact
matches, maximum absolute delta `2.0872666`, and RMS delta `0.1532113`. The
external D2D refresh copies Prefix `final_streams` into the persistent handoff
exactly (10,240/10,240 values identical), so that copy is not introducing the
residual difference. The graph-owned CPY destination remains uninitialized
poison (0/10,240 finite values).

The resulting Route graph input has FNV64 `0x1aae28008a06c66f`; the ordinary
router input has FNV64 `0xea9691564497abf3`. Their arrays have 0/2,560 equal
values, maximum absolute delta `5.7906948`, RMS delta `0.520398`. The ordinary
route is `[273,50,385,387,35,347,93,286,6,423]`; Route emits
`[195,329,118,128,420,412,220,367,481,472]`. Prefix still publishes
`[0..9]`, so its route terminal is stale. This confirms the first valid mismatch
is already at the Prefix split-layer residual, before the Route graph and
before expert payload execution. It does not yet identify which earlier
decode layer first diverges.

An offline comparison grouped the 10,240 residual elements as ten contiguous
HC streams of 1,024 features each. No value matched the ordinary residual.
Per-stream RMS deltas were `0.2750, 0.2685, 0.1856, 0.0649, 0.0827, 0.0769,
0.0759, 0.1294, 0.0856, 0.0762`; this is a broad hidden-state mismatch, not a
small rounding difference isolated to the router or one HC stream. The grouping
follows the tensor shape `[hidden=1024, HC=10]`; it does not locate the first
upstream layer.

The attempt to retain every intermediate hidden tensor as a graph output
changed Phase-2 execution: layers 0--2 remained finite and equal, layer 3 was
partly non-finite, and layers 4 onward were poison; the generated token also
changed to 0. That trace is invalid and the instrumentation change was
reverted. Its files are retained under `/tmp/qwen-p2-layertrace-20261006/` for
forensics, but must not be used as model-state evidence. The original host-CPY
all-layer dump remains invalid after layer 2 as noted above.

The clean Phase-2 diagnostic run completed with no host or VRAM OOM: minimum
`MemAvailable` was 83.46 GiB and minimum free 3090 VRAM was 7.76 GiB. One
post-run `nvidia-smi` query failed, then subsequent read-only checks recovered
and showed both GPUs idle. A later two-token run without all-layer capture
completed under the same guards and produced the matched position-28 captures
below; the driver query again recovered to idle afterward.

Current localization: the wrong Layer-13 residual is in the Prefix graph
calculation or the Prefix graph's upstream decode state. The attempted
post-replay copy from `prefix.final_streams` faithfully copies those wrong
bytes, while the graph-owned persistent CPY fails to publish at all. Next work
must compare Prefix against the ordinary graph without adding outputs to the
captured graph; use a one-boundary-at-a-time diagnostic or a separate
non-captured reference path, and first resolve why the stable graph CPY output
is not written. Keep early-ready as the production default.

Static state-path audit rules out one tempting explanation for this run: when
`qwen4exp_partial_layer` is set (13 here), prefill does not defer recurrent
state readback for direct device handoff. It synchronizes GDN state to the
engine's host arrays, and the Phase-2 Prefix seeds its private state tensors
from those arrays. This does not prove the seeded values match the ordinary
decode graph, but the deferred-prefill alias path is not active in this test.

A lower-priority source-level question is the current uncommitted Phase-2
state-publication change in `src/ggml_qwen4exp_graph.cpp`: it marks GDN state
CPY nodes as `GGML_TENSOR_FLAG_OUTPUT` in every non-Off graph, including Prefix.
GGML CUDA has a GDN/cache fusion pass that rejects fusion when the immediately
following state CPY is an output node. However, MoEStream expands the Prefix's
route and handoff roots before appending the state-update CPYs, while the fusion
requires the CPY to be the next real node after GDN in one CUDA graph split.
No saved log shows that fusion firing in the ordinary graph, and the current
builder order makes adjacency doubtful. Treat this as unconfirmed and do not
spend a runtime A/B on the flag unless a node/split trace first proves that the
fusion is active in the control path. The first bad replay used a dirty
worktree, so no changed line can yet be attributed as the root cause.

The final non-invasive position-28 run saved the exact Prefix residual at
`/tmp/qwen-p2-layertrace-20261006/phase2-clean.handoff.prefix-final-streams-before-refresh.f32`
and the ordinary reference at
`/tmp/qwen-p2-layertrace-20261006/baseline2.layer13.pre_ffn_residual.f32`.
The corresponding Route input hash and IDs are in
`/tmp/qwen-p2-layertrace-20261006/phase22.engine.log`; the raw Route input is
also retained at `/tmp/preffn-phase2-route-input.f32` and has the same hash.
The baseline route input is
`/tmp/qwen-p2-layertrace-20261006/baseline2.router-input.f32`.

The invasive every-layer-output probe was reverted after it changed the
Phase-2 run: the trace stayed finite and equal through layer 2, became partly
non-finite at layer 3, and was poison from layer 4. Its candidate output was
`8160,0` instead of the non-instrumented Phase-2 run's `8160,8160`. The
position-28 file and logs remain under
`/tmp/qwen-p2-layertrace-20261006/phase22.*` as evidence of probe
interference, not as valid model state. The opt-in handoff dump, node binding
trace and source/CPY scheduler eval probe are retained; the engine target builds.
Pre-existing dirty changes are untouched.

A build trace after `ggml_backend_sched_alloc_graph` found exactly one CPY
node whose `src[1]` is the persistent handoff. The CPY output and destination
retain the expected persistent address and are assigned to `CUDA0`. A
two-token run with `MOESTREAM_GGML_CUDA_GRAPH_CAPTURE_ON_FIRST_CALL` unset
logged no first-call capture and showed the same failure: the graph-owned
handoff was poison, the source-to-handoff D2D copy was byte-exact, the source
and destination were both `CUDA0` buffers, and the candidate still produced
`8160,8160` with 10/10 route IDs different from the ordinary graph. This
rules out the opt-in first-call capture path and a host-buffer source as the
cause.

A temporary `LD_PRELOAD` tracer at `/tmp/libmoe_cuda_copytrace.so` then caught
two eager 40-KiB D2D copies to the handoff address, both returning
`cudaSuccess` with `capture=0`. The first is the graph CPY; the second is the
post-replay refresh. Both use the same source pointer and stream. After the
first copy, the persistent destination still has the poison hash
`0x69135d8732dc4325`; after the second, it matches the finite source hash
`0xfc837c3f0e4322cd`. This proves both copy calls were submitted successfully
and the later refresh copied the finite source byte-for-byte. It does not show
what source bytes the first copy observed. With both nodes assigned to CUDA0
and CPY using that backend's main stream, a missing cross-stream join is not
established; the scheduler split and source readiness at the first copy remain
to be traced.
The copy-call trace is in
`/tmp/qwen-p2-layertrace-20261006/cpy-trace.stdout`.

The graph allocation scan locates the source tensor at node 1945 and its CPY
at node 1981. Other graph outputs alias that allocation only at earlier nodes
(the last is node 1889); no later writer using that address appears before
handoff. Inspection of the GGML implementation confirms `ggml_cpy(src, dst)`
returns a view of `dst`, and CUDA CPY copies `src` into `src1` on the backend's
main stream. Both handoff nodes are assigned to `CUDA0`, which makes a simple
wrong-destination or wrong-stream selection unlikely. The graph-owned copy
still sees poison after replay while the later external refresh copies the
finite source exactly, so execution-time ordering or an unexpected write
remains unresolved; the scheduler split and CUDA stream for nodes 1945/1981
must be established before attributing this to a missing cross-stream join.

The one-node scheduler callback did run, but splitting the scheduler at the
source/CPY boundary can itself perturb cross-stream ordering; its poison
values are not treated as an unmodified model-state trace. Marking the source
as an additional graph output also made the continuation NaN, so that
retention probe was reverted. Omitting the bare destination leaf as a graph
output had no effect and was also reverted. The scheduler eval callback also
reads poison at its split boundary, but that callback changes scheduler
segmentation, so it is not normal-path source evidence.

The latest CPU-only `make test-components` passed all 44 component tests and
the focused graph-tier tests; the engine target builds. Guarded inference runs
recorded no host or VRAM OOM. NVIDIA queries still fail intermittently after a
run; the latest guarded run completed and its immediate query recovered. Keep
the next stream/event trace to one two-token run with host and VRAM guards.
