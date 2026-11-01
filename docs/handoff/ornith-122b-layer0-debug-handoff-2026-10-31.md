# Ornith-1.5-122B layer-0 drift handoff — 2026-10-31

## State at handoff

**Paused for the next session to continue the correctness investigation.**
Ornith is not correctness-accepted. The mixed VRAM/DDR teacher-forced result
still has a first observed difference at layer 0's routed MoE output. The
exact operation or backend responsible has not been identified. Do not use
the performance results below as correctness evidence.

The key distinction is:

1. A prior, heavily instrumented layer-0 mismatch was not reproduced after
   disabling invasive graph timing/probe callbacks and reducing capture to
   the already-consumed routed output.
2. A later minimal-tap comparison of the mixed hotset path still differed at
   layer 0, but its llama.cpp reference ran MoE on CPU while ZeroMoE ran a
   mixture of VRAM and DDR experts. That comparison locates a discrepancy; it
   does not identify its cause.

The fastest useful next step is an actual-weight, one-layer teacher-forced
comparison with explicitly matched placement. There is already a valid
approximately 2 GiB layer-0 GGUF fixture and a layer-0 llama.cpp trace helper;
the corresponding ZeroMoE fixture run was not completed because CUDA device
discovery intermittently returned zero devices.

## Source tree and artifact provenance

- Use worktree `/home/ray/Projects/ZeroMoE/.worktrees/ornith-122b-offload`.
- Branch: `recovery/ornith-122b-offload`.
- HEAD at handoff: `7d48e01de3ea2804ea6375b37858c23c909ad1d2`.
- The worktree is **dirty with extensive in-progress changes**. Preserve it;
  do not reset, clean, or copy its source into another worktree without
  recording the exact diff. Main worktree was clean at `a16fa373246a1c43d930548d9c5439aca77aa022`.
- GGUF: `/mnt/usb-ssd/@/home/ray/models/Ornith-1.5-122B-UD-IQ3_XXS.gguf`,
  49,515,667,712 bytes. GGUF architecture is `qwen35moe`, with 49 blocks
  including one MTP/next-token block; the main decoder has 48 layers.
- Reference llama.cpp source: commit
  `8baabd53b77d1c48c6ef8d3b620436292f3f6231`. Keep the llama.cpp runtime and
  its GGML libraries from the same build tree; a prior mixed-library attempt
  failed backend discovery and yielded no correctness result.
- The R27 ZeroMoE binary was built from the dirty worktree snapshot at the
  recorded source commit above, against GGML base `8baabd...`; its SHA-256 is
  `eca796ab653c60b52d1292e74254723bf27a3d5a1f041303a3bd7eba3fe93822`.
  Its saved provenance check passed. Re-run the provenance check and rebuild
  if any source changes before using it.
- GPU: use only RTX 3090, UUID
  `GPU-fbddd113-32a1-56af-9c8e-7f2c877a9736` (24 GiB). The host is shared.
  Do not select the RTX 5060 Ti. Existing runner environment uses
  `CUDA_VISIBLE_DEVICES` set to the 3090 UUID, `OMP_NUM_THREADS=10`,
  `ZEROMOE_QWEN_CPU_THREADS=10`, `CUDA_HOME=/opt/cuda`, and the coherent
  `iq1s-vnni-off` GGML libraries from the R30 matched llama.cpp build tree.
  Full recorded environment and build details are under the R27 artifact
  directory below.
- The R27 host preflight requires 1-minute load average <= 2.0 and I/O PSI
  `some avg10` <= 1.0. R27's preflight passed (load 0.08 then 0.06; I/O PSI
  0.00); CUDA runtime discovery still failed for the model process. Treat
  `cudaGetDeviceCount=0` as an execution-environment failure until the
  direct probe and engine see the same device; it is not evidence of a model
  or graph failure.

## Teacher-forced workload and observed results

Use the same input sequence in both engines:

- Prompt text: `Hello there.`
- Prompt token IDs: `9419, 1017, 13`.
- Forced decode token: `353`.
- Context: 128; batch and microbatch: 8; threads: 10.
- The reference helper manually evaluates the prompt, then the forced token;
  it saves both token files and per-tensor metadata. Verify token IDs byte for
  byte before comparing any tensors.

### Minimal all-CPU/DDR control: R22

R22 is the strongest completed correctness control. ZeroMoE used its full
graph with all routed experts in host DDR/CPU, matching llama.cpp's CPU MoE
placement. It captured only final routed MoE outputs, avoiding the large
intermediate-probe/timing callbacks. The comparison reports **zero unequal
values at all 48 layers for both the prompt and forced decode passes**.

This confirms that the tested all-CPU/DDR routed-output path can match the
llama.cpp output on this token sequence. It does not establish mixed VRAM/DDR
parity, all input sequences, or independent equality of router IDs and
weights. See:

- [`R22 minimal comparison`](../../build_mtp_results/2026-10-31-ornith-recovery-3090/correctness/teacher-forced-routed-output-minimal-3090-r22/minimal-routed-output-comparison.txt)
- [`R22 layer table`](../../build_mtp_results/2026-10-31-ornith-recovery-3090/correctness/teacher-forced-routed-output-minimal-3090-r22/all-routed-comparison.md)
- R22 run inputs, configs, source provenance, and logs are in that same directory.

### Invasive all-boundary capture: R20/R21 (diagnostic, not a clean control)

R20/R21 enabled broad tensor probes and timing/evaluation callbacks. The
boundary report shows tiny floating-point deltas at layer 0 and 88 ordered
expert-ID mismatches across the full trace, but this capture path was later
found to alter graph execution/scheduling. Do not treat those ID mismatches
as evidence of the normal path without reproducing them with minimal taps.

The Qwen `ZEROMOE_QWEN4EXP_LAYER_TIMING` callback could split CUDA graph
subgraphs into one-node submissions and disable fused MoE kernels. Timing was
made opt-in; ordinary debug logging alone is not meant to install it. With
callbacks/probes unset, a subsequent teacher-forced routed-output comparison
matched llama.cpp bitwise across all 48 layers. This explains the earlier
instrumentation-sensitive observation, **not** the later mixed-placement R24
delta.

- [`R21 boundary report`](../../build_mtp_results/2026-10-31-ornith-recovery-3090/correctness/teacher-forced-boundary-cpu-match-3090-r21/teacher-forced-boundary-comparison.md)
- Timing/probe caveats are also recorded in
  [`docs/env-vars.md`](../env-vars.md).

### Mixed VRAM/DDR minimal comparison: R24 (still unresolved)

R24 used the same prompt/forced token, 17 GiB exact-8x usage-ranked hotset,
full graph, split early-ready path, synchronous output copy, and final routed
output taps. Prefill matched bitwise. The first observed forced-decode
difference was layer 0 routed output, maximum absolute error
`4.395432e-05`; it grew to `0.050619` at layer 44, then decreased.

**Placement was not matched:** llama.cpp ran experts on CPU, while ZeroMoE
ran hot experts in VRAM and cold experts from DDR. Therefore this result does
not distinguish a CUDA/CPU numerical difference from an adapter operation,
layout, router, or expert-dispatch bug. It is still a valid reason to stop
acceptance and isolate layer 0. R24's raw engine log was overwritten by the
invalid R25 launch; the routed tensors and comparison report were recorded
as retained in the recovery notes. R25's incident explains the overwrite:
its copied TOML retained R24's `log_path`, and its output tensor directory
was absent.

- [`R25 artifact incident`](../../build_mtp_results/2026-10-31-ornith-recovery-3090/correctness/layer0-selective-boundary-probe-r25/incident-and-next-step.md)
- The R24 numeric result and sequence are also summarized in
  [`the recovery handoff`](ornith-122b-recovery-2026-10-31.md), section
  “R20-R25”.

### VNNI-on candidate (separate unresolved build-specific signal)

A separate VNNI-on ZeroMoE build differed from the CPU reference at prompt
pass 0, layer 0 routed output (max abs `2.33e-10`, 7,787 unequal floats),
with a much larger downstream delta by layer 47. Disabling the custom cold
fusion did not change this result. This points below that helper, possibly
to a CPU VNNI dot path, but router IDs/weights and pre-MoE inputs were not
captured in this VNNI run. Do not use that binary. This is a separate
configuration-specific symptom, not a proven explanation for R24's mixed
VRAM/DDR discrepancy.

- [`VNNI-on fused-path report`](../../build_mtp_results/2026-10-31-ornith-recovery-3090/vnni-on/correctness-cpu-all-moe/correctness-status.md)
- [`VNNI-on stock-path report`](../../build_mtp_results/2026-10-31-ornith-recovery-3090/vnni-on/correctness-cpu-stock-vnni-on/correctness-status.md)

## What has and has not been established

| Claim | Status and evidence |
|---|---|
| The llama.cpp reference can be reproduced on the selected teacher-forced sequence. | Yes. R22 stores prompt IDs, forced ID, reference tensors, source revision, and comparison. |
| All-CPU/DDR ZeroMoE routed outputs match llama.cpp on that sequence. | Yes, bitwise in R22 for 48 layers and both passes. Scope is this output boundary and workload. |
| The mixed VRAM/DDR result is correct. | No. R24 differs at L0; placement is not matched. |
| Attention is the first known cause. | No. The R24 first difference is at routed MoE output, and the old attention concern was not reproduced with the normal callback-free path. Earlier hidden-state/probe differences were instrumentation-sensitive. |
| Router IDs/weights are equal in R24. | Unknown. R24 only captured final routed outputs. The broad R21 router mismatch is probe-sensitive and cannot substitute for a clean R24 route trace. |
| The custom CPU cold-expert fused calculation itself matches stock GGML for Ornith tensor shapes. | Yes for the synthetic-weight unit matrix, not for actual model activations or GPU execution. |
| VNNI-on is safe. | No. A separate VNNI-on candidate diverged and has not been localized. Keep VNNI off for the reference investigation. |
| A complete layer-0 actual-weight A/B has run. | No. Fixture and helpers exist, but the ZeroMoE fixture run did not load due to intermittent CUDA device discovery. |

## Layer-0 fixture and next experiment

### Existing fixture and prepared tooling

The valid actual-weight fixture is:

`build_mtp_results/2026-10-31-ornith-recovery-3090/correctness/layer0-production-node-taps-r27/ornith-qwen35moe-layer0.gguf`

Its manifest records source bytes/hash inputs, 22 tensors, about 1.9 GiB of
tensor data, `block_count=1`, `nextn_predict_layers=0`, and the original
recurrent layer-0 flag. It contains layer-0 weights plus required global
embedding/output tensors. The source 49.5 GB model was not modified. Fixture
creation utility: `slice_qwen35_layer0.py` in the same directory.

Prepared helper/tools in that R27 directory:

- `llama_ref_trace_layer0.cpp`, `llama-ref-trace-layer0`, and
  `run-llama-ref-layer0.sh`: test-only llama.cpp callback trace, filtering
  actual eval tensors to layer 0. It can save pre-MoE hidden state, router
  logits/probabilities/top-k/weights, routed output, and selected expert gate,
  up, SwiGLU, down, and weighted tensors when the corresponding nodes are
  present.
- `compare_layer0_boundaries.py`: compares hidden state, router values,
  ordered IDs/weights, and routed output.
- `compare_layer0_expert_stages.py` is prepared in R26 for intermediate expert
  stages. Check its expected tensor names against the actual helper output.
- `layer0-mini-fixture-run/zeromoe.toml`, `usage-layer0.txt`, token files,
  and a comparator are present. **There is no completed ZeroMoE result and no
  verified mini-fixture runner yet.** Inspect the config/paths and create a
  fresh isolated output directory before running. Do not use an old `log_path`.
- `GraphCore` now has the shared boundary-tap type and the helper
  `models/graph/full_graph_boundary_probe.hpp`. The shared opt-in variables
  are `ZEROMOE_FULL_GRAPH_PROBE_DIR` and `ZEROMOE_FULL_GRAPH_PROBE_LAYER`.
  Initial Qwen and GLM hooks capture existing semantic boundary tensors.
  This is in-progress instrumentation, not yet proven observer-free or
  uniformly complete for every family. Compare its result to a minimal
  routed-output run before trusting it. Do not enable it for performance.

### Recommended diagnostic order

Do not load the full 49.5 GB GGUF to debug a layer-0 first difference.

1. Preserve the current R22 all-CPU/DDR trace as the clean control. Recheck
   its token files and provenance before reusing it.
2. Complete the mini-fixture ZeroMoE and llama.cpp runs on the same
   teacher-forced sequence. First use **matched CPU expert placement** on both
   engines to validate fixture, state progression, and tap mapping. Then run
   **matched CUDA expert placement** (all layer-0 experts resident on the
   3090 in both engines) to expose a backend-specific delta. Finally, if both
   matched controls pass, reproduce the R24 mixed hotset placement. Explicitly
   log actual residency; do not infer placement from config alone.
3. For a minimal observer-effect check, capture final routed output alone
   first. Then add stage taps at layer 0 only and confirm that enabling those
   taps leaves that final output unchanged. Keep scheduler evaluation timing
   callbacks unset.
4. Compare in order and stop at the first mismatch:
   block input / pre-MoE hidden; router logits; router probabilities;
   ordered top-k IDs; normalized selected weights; selected expert gate and
   up projections; activation/clamp; down projection; per-slot weighted
   expert results; final top-k reduction/routed output. Save raw tensor bytes
   plus dtype, shape, stride, backend, token/pass, and actual expert IDs.
5. If dense/block input matches and router outputs match but a projection
   differs, compare llama.cpp's exact GGML node sequence, input layout,
   weight layout, dtype/quant type, and backend for the selected expert. Do
   not “fix” it by algebraically equivalent fusion/reordering before a
   same-input boundary trace.
6. Only after L0 passes on the matched CUDA path should the next test use the
   full model and the intended mixed-hotset survey. Run a single combined
   teacher-forced survey workload in both engines, compare per-token route
   IDs/weights first, and derive usage counts only from parity-passing
   reference routes. Do not run a performance sweep before that gate.

For matched all-CUDA placement, the 2 GiB fixture should fit the 3090, but
verify GGUF tensor residency and free memory in both logs. The prepared llama
runner currently uses `--n-cpu-moe 48`, which is an all-CPU expert setup; do
not reuse it unchanged for the matched-CUDA experiment. Use `--n-gpu-layers
99 --n-cpu-moe 0` for the all-CUDA reference and configure ZeroMoE so all
128 layer-0 experts are actually resident. Keep the R22 all-CPU control as a
separate matched placement. A ZeroMoE usage file with only surveyed experts
is not proof that every layer-0 expert is resident.

### Common unit tests already run

1. `cold_ffn_fused_test --ornith-expanded-matrix ornith-iq3_xxs-actual-shape`
   **passed**: 6 cases bit-identical. It compares the custom CPU cold-expert
   fused FFN calculation with stock GGML for synthetic values using the
   actual Ornith IQ3_XXS expert dimensions/types. It does **not** use the
   actual GGUF expert weights/activations; it does not exercise routing,
   actual GDN state, CUDA, residency, mixed VRAM/DDR, or full-model decode.
   Result: `correctness/layer0-production-node-taps-r27/cold-ffn-unit-run.log`.
2. `qwen_moe_router_unit_test` **passed** against a scalar CPU reference for
   deterministic synthetic Qwen-style router values: logits, softmax,
   ordered top-k IDs, selected-weight normalization, and scale. The harness
   was fixed to read strided/non-contiguous GGML views correctly. It does
   **not** load actual router weights or hidden states, exercise a CUDA
   kernel, prove model-family config, or compare with llama.cpp on an actual
   token. Current result:
   `correctness/layer0-production-node-taps-r27/router-unit-run.log`.
   The initial harness-layout failure is preserved separately in
   `router-unit-run-initial-view-layout-failure.log`.
3. `test_moe_partial_route` passed the small route-classification/branch-mask
   check (`inactive id=-1` in decode versus prefill). It does not test any
   numerical MoE output or routing parity. Binary and source are in the R27
   artifact directory.

These tests are useful guards, not substitutes for the actual-weight
teacher-forced layer-0 boundary A/B.

## Performance context (not correctness acceptance)

The 17 GiB exact-8x benchmark-weighted five-prompt usage profile measured
42.50, 42.52, and 42.54 steady graph tok/s on the 3090, versus llama.cpp
16.52 tok/s. This weighting duplicates one benchmark question eight times;
it is experimental and only about 70% benchmark-prompt route observations.
The equal-weight five-prompt profile measured 35.60 steady graph tok/s. Thus
the 42 tok/s result does not establish the expected general five-prompt
workload, and neither profile is accepted while mixed-placement correctness
is unresolved. Other measured stages (prefill, graph setup, end-to-end decode)
are recorded separately in `ornith-122b-recovery-2026-10-31.md` and its linked
run directories. Per-layer GPU timing/counter attribution remains open.

## Why this took multiple long sessions without an accepted result

1. **The correctness gate was applied too late.** Route surveys and multiple
   performance runs (including R23) happened before a clean mixed-placement
   teacher-forced comparison. Route coverage derived from ZeroMoE's own
   routes was treated as useful hotset evidence before proving those routes
   matched llama.cpp. The user's stated rule should have stopped that work at
   the first layer-0 divergence.
2. **The comparison changed placement.** CPU reference experts versus mixed
   CUDA/DDR ZeroMoE cannot distinguish operation drift from backend numeric
   differences. Several runs also changed instrumentation or build options,
   so their results were not interchangeable controls.
3. **Diagnostics changed the graph.** Broad timing/probe callbacks altered
   graph submission/fusion. Their differences were initially interpreted as
   model correctness failures. Minimal captures and explicit “observer
   effect” controls should have come first.
4. **Test-run hygiene failed once.** R25 reused an R24 TOML log path and had
   no tensor output directory; it overwrote the R24 raw engine log and
   produced no useful stage capture. Every run needs a unique output root,
   preflight assertions for every input/output path, and a post-run artifact
   manifest before the next run.
5. **Too much work used full-model loading.** A failure first seen at layer 0
   can be isolated using an actual-weight one-layer fixture. The fixture and
   helper were prepared late; then intermittent CUDA discovery prevented
   completion. Continue with the fixture rather than repeating 50 GB loads.
6. **The shared host/runtime visibility was not stabilized.** The 3090 was
   observed idle, but CUDA/NVML discovery was intermittent across processes.
   Distinguish host preflight, full inventory, UUID-filtered process list,
   direct CUDA probe, and engine device discovery; save each result. Do not
   attribute startup failure to the model when it occurs before GGUF load.
7. **Run IDs obscured the evidence.** R-series numbering accumulated surveys,
   placement tests, diagnostics, and benchmarks with different purposes. Use
   named artifacts that encode correctness/performance, placement, workload,
   and source snapshot; keep one table of hypothesis, controlled change,
   result, and next decision.

The process improvement is a hard sequence: freeze provenance and workload;
save llama.cpp reference tensors from the first correctness run; compare
ZeroMoE after every single change; stop and save a small reproducer at the
first boundary mismatch; only then validate route survey/hotset behavior;
only after correctness passes run matched performance checks. One run should
combine the chosen prompt set and teacher-forced route capture where feasible
instead of splitting short and long survey work without a specific diagnostic
reason.

## Artifact index for the next session

- Current recovery/performance/historical run map:
  [`ornith-122b-recovery-2026-10-31.md`](ornith-122b-recovery-2026-10-31.md)
- Earlier drift and CPU placement/build ablations:
  [`ornith-122b-drift-investigation-2026-10-30.md`](ornith-122b-drift-investigation-2026-10-30.md)
- R22 clean minimal routed-output A/B:
  `build_mtp_results/2026-10-31-ornith-recovery-3090/correctness/teacher-forced-routed-output-minimal-3090-r22/`
- R20/R21 broad boundary probes (observer-effect caution):
  `build_mtp_results/2026-10-31-ornith-recovery-3090/correctness/teacher-forced-boundary-3090-r20/`
  and `.../teacher-forced-boundary-cpu-match-3090-r21/`
- R25 overwrite incident and R26 prepared-but-not-run diagnostic:
  `build_mtp_results/2026-10-31-ornith-recovery-3090/correctness/layer0-selective-boundary-probe-r25/`
  and `.../layer0-selective-boundary-probe-r26/`
- R27 actual-weight fixture, reference helper, failed device probes, test
  logs, config, and source/build provenance:
  `build_mtp_results/2026-10-31-ornith-recovery-3090/correctness/layer0-production-node-taps-r27/`
- VNNI-on experiment:
  `build_mtp_results/2026-10-31-ornith-recovery-3090/vnni-on/`
- Full-graph boundary probe implementation and first adapter hooks:
  `models/graph/full_graph_boundary_probe.hpp`,
  `models/graph/graph_core.hpp`,
  `models/qwen/ggml_qwen4exp_graph_build.cpp`,
  `models/qwen/ggml_qwen4exp_graph_build_moe.cpp`, and
  `models/glm/ggml_glm5next_graph_build.cpp`.

## 2026-11-01 continuation notes

- Rebuilt the R27 binary after fixing two in-progress build breaks:
  `full_graph_boundary_probe_enabled_for_layer` calls in
  `models/qwen/ggml_qwen4exp_graph_build.cpp` and
  `models/qwen/ggml_qwen4exp_graph_build_moe.cpp` were missing the
  `zeromoe::graph::` qualifier; and the Qwen35 layer-map validation in
  `models/base/graph_initializer.cpp` rejected the single-layer fixture
  (`n_gdn >= n_layers`). Relaxed that check to `n_layers > 1`.
  Provenance now passes on the rebuilt binary.
- Completed the first ZeroMoE mini-fixture run
  (`layer0-mini-fixture-run-r1`, all experts in DDR/CPU, full graph).
  Prompt rows 0-2 match the R22 llama reference bitwise; forced-decode row
  differs at layer 0 with max abs `4.462243e-05`, all 3072 values unequal.
  This reproduces the R24 forced-decode symptom on a one-layer, all-CPU
  fixture, ruling out mixed VRAM/DDR placement as the cause.
- The same forced-decode row also differs from the full-model R22 ZeroMoE
  run (`4.462243e-05`), while R22 full-model forced decode matched the
  llama reference. So the discrepancy is specific to the single-layer fixture
  / forced-decode state handoff, not to placement.
- `prefill_batch_tokens=1` crashed (core dump) on the fixture; not pursued.
- The fixture cannot be run in llama.cpp directly (single GDN layer trips
  the hybrid-cache buffer assert); comparison used the saved R22 llama
  reference tensors instead.

No further model run or tuning was performed while writing this handoff.
