# Ornith-1.5-122B recovery update — 2026-10-31

## Status

The Qwen35 MoE adapter and required full graph decode run on the RTX 3090.
The 42 tok/s steady graph target passes three times with an explicit 8x
benchmark-prompt weight inside a five-prompt survey: 42.50, 42.52, and
42.54 tok/s. The equal-weight five-prompt profile reaches 35.60 tok/s.

The 8x profile is benchmark-weighted: the exact gradient-descent question
contributes about 70% of survey observations. Treat it as experimental until
that weighting reflects expected use. The current focus is the full-graph
profile, per the user's instruction; the old layer-split result is historical
and outside this recovery scope.

## Attention and the earlier drift

The attention implementation was not the first numerical mismatch. The prior
teacher-forced trace had matching pre-MoE inputs, router values and expert IDs;
the first reported delta was the layer-0 routed expert result. The cause was
implicit Qwen graph timing callbacks: `log_level=DEBUG` installed an evaluation
callback, split CUDA subgraphs into one-node submissions, and disabled fused
MoE kernels. Making timing opt-in restored the fusion path. With callbacks and
probes unset, routed outputs matched llama.cpp bitwise across all 48 layers in
both teacher-forced passes. The older, instrumentation-induced delta is not an
attention divergence.

The short Nsight Systems trace of the normal hotset path also shows attention
is a small part of decode time: GDN plus full-attention kernels totalled about
0.43 ms per token, while the graph submit interval averaged 50 ms per token.
The dominant work is the CPU and routed-MoE path.

## Matched performance results

The table below records earlier profiles from the recovery session. The
latest equal-weight and benchmark-weighted survey results are separated below
so that historical failures are not confused with the current candidate.

All runs use the same GGUF and only the RTX 3090, masked by UUID. The
llama.cpp reference source is clean at `8baabd53b77d1c48c6ef8d3b620436292f3f6231`.
The matched llama.cpp binaries have the same upstream GGML base; ZeroMoE also
has its project GGML patches. The hotset placement differs from llama.cpp's
contiguous `n_cpu_moe` split, so relative ratios are provisional.

These earlier rows are historical. The user has since narrowed the active work to full-graph profiles.

| ZeroMoE profile | Workload | Decode | llama.cpp | Absolute target | Full-graph status |
| --- | --- | ---: | ---: | ---: | --- |
| Split, Qwen map | 16 prompt / 400 generated, context 416, 10 threads | 18.69 tok/s steady | 17.37 tok/s | 28 tok/s: fail | Decode and prefill graph gates pass |
| Usage hotset 16.5 GiB | 24 prompt / 200 generated, context 224, 10 threads | 20.38 tok/s steady | 16.52 tok/s | 42 tok/s: fail | Decode and prefill graph gates pass |
| Usage hotset 17.0 GiB, original five-standard survey | same 24 / 200 workload, repeated twice | 19.82 / 20.20 tok/s steady | 16.52 tok/s | 42 tok/s: fail | Decode and prefill graph gates pass both times |
| Usage hotset 17.25 GiB, default expert prefill | same 24 / 200 workload | 20.89 tok/s steady | 16.52 tok/s | 42 tok/s: fail | Decode passes; prefill gate fails |

The 17.0 GiB profile uses `ZEROMOE_QWEN4EXP_PREFILL_OP_OFFLOAD=0` to lower
prefill workspace demand while retaining full-graph prefill. It selected 5,544
experts and covered 78.43% of the Ornith survey routes. This setting did not
improve decode: the two measured runs averaged 20.01 tok/s. Host load rose
above 18 during the repeats, so those rates are contended. The llama.cpp
hotset reference averaged 16.52 tok/s; the relative gate passes, while the
42 tok/s absolute gate fails.

The 17.25 GiB experiment selected 5,632 experts and covered 79.04% of survey
routes. Its decode graph reached 20.89 tok/s, but the 8-token full-graph
prefill needed 1.70 GiB with only 1.42 GiB free. It fell back to legacy
prefill, so it is not a full-graph prefill result and is rejected by the
corrected `--require-full-graph-prefill` gate. Increasing the budget by
0.75 GiB gained only 1.81 route-coverage points.

## Early ready correctness and scheduler fix

Before the graph-root fix, all-split early ready with async output enabled
built private CPU lanes but pruned their MAP_CUSTOM1 markers: the decode graph
had only two CPU nodes, and the scheduler reported lane calls=0. Setting
ASYNC_OUTPUT=0 was a safe workaround and byte-identical on the same placement.

The builder now explicitly roots the side-effect CPU marker when async output
is enabled. The V3 build logs architecture=qwen35moe,
async_output_copy=1, source=adapter default; the graph has 47 MAP_CUSTOM1 lane
nodes, all 47 lanes complete, the full-graph gate passes, and the five-prompt
teacher-forced outputs match the sync-output baseline byte for byte across all
48 layers and both trace passes:
[default-on correctness run](../../build_mtp_results/2026-10-31-ornith-recovery-3090/profile-aligned/early-ready-qwen35-default-async-correctness-benchmark5/).

The first default-on throughput attempt started at host load 13.11 and rose to
20.68. It was interrupted after 377 generated tokens and is not valid
performance evidence. The Qwen profile's host-load guidance is now enforced in
the reference runner; any performance rerun must be preflighted at load <=2.

## Prompt-specific usage profile

The exact gradient-descent benchmark prompt was surveyed for 200 forced
tokens. With its single-prompt usage histogram, 17.0 GiB hotset, prefill
offload disabled, and all split early ready using synchronous output copy,
the 24/200 benchmark passed three times:

| Run | Steady graph decode | End to end decode | Prefill | Overall |
| --- | ---: | ---: | ---: | ---: |
| r1 | 43.31 tok/s | 35.09 tok/s | 9.39 tok/s | 24.23 tok/s |
| r2 | 43.34 tok/s | 35.14 tok/s | 9.51 tok/s | 24.35 tok/s |
| r3 | 43.49 tok/s | 35.21 tok/s | 9.47 tok/s | 24.34 tok/s |

The mean steady graph rate is 43.38 tok/s. The matched llama.cpp rate is
16.52 tok/s, giving a 2.62 ratio. Each run selected 5,661 experts and reported
93.26% survey coverage. These runs validate this exact prompt ranking; they
do not validate the requested five-prompt usage ranking.

## Requested five-prompt usage profile

The benchmark-inclusive survey contains the exact gradient-descent prompt and
four standard Python, Go, OS, and general prompts. It has 11,396 layer/expert
records and 337,152 route observations. At 17.0 GiB, the profile selected
5,561 experts with 81.58% survey coverage. It passed full graph decode and
prefill gates and reached 35.60 steady tok/s (29.87 end-to-end decode,
9.48 prefill, 21.67 overall). The 90% llama.cpp ratio gate passed at 2.155;
the 42 tok/s absolute gate failed.

The 17.25 GiB version with prefill offload disabled and batch8 reached
35.62 tok/s but recorded two full-graph prefill fallback markers and is
rejected. A 17.25 GiB batch4 run passes strict prefill but reaches only
35.24 tok/s; the extra hotset does not improve steady decode.

The exact per-run configs, prompt file, five-prompt usage file, environment,
binary provenance, logs and gates are in
[the prompt-specific runs](../../build_mtp_results/2026-10-31-ornith-recovery-3090/profile-aligned/early-ready-allsplit-sync-output-repeats.md),
[the requested five-prompt run](../../build_mtp_results/2026-10-31-ornith-recovery-3090/profile-aligned/early-ready-benchmark5-sync-output-perf-r1/),
and [the rejected 17.25 GiB run](../../build_mtp_results/2026-10-31-ornith-recovery-3090/profile-aligned/early-ready-benchmark5-17g25-sync-output-perf-r1/).

## Workload-weighted five-prompt profile

The experimental usage file preserves the five unique prompts but adds seven
copies of the exact 200-token gradient-descent survey counts. The benchmark
prompt therefore has eight times the weight of each of the four standard
prompts, about 70% of total route observations. The 17.0 GiB hotset selects
5,576 experts and covers 86.39% of the weighted observations.

The weighted profile passed the full graph and performance gates in three
matched runs:

| Run | Steady graph decode | End to end decode | Prefill | Overall |
| --- | ---: | ---: | ---: | ---: |
| r1 | 42.50 tok/s | 34.57 tok/s | 9.35 tok/s | 23.95 tok/s |
| r2 | 42.52 tok/s | 34.53 tok/s | 9.50 tok/s | 24.04 tok/s |
| r3 | 42.54 tok/s | 34.59 tok/s | 9.46 tok/s | 24.04 tok/s |

Mean steady decode is 42.52 tok/s against llama.cpp at 16.52 tok/s (2.57x).
Each run generated 600 tokens, with 9 prefill graph executions and 597 decode
graph executions; graph decode took 14.03–14.05 seconds. The steady graph
metric passes the 42 tok/s gate. End-to-end decode and prefill rates remain
separate because user-visible generation includes extra request and token-loop
work.

Teacher-forced routed-output A/B for this exact weighted usage placement is
byte-identical across all 48 layers and both trace passes:
[baseline](../../build_mtp_results/2026-10-31-ornith-recovery-3090/profile-aligned/benchmark5-exact8x-baseline-routed-parity/)
and [early-ready candidate](../../build_mtp_results/2026-10-31-ornith-recovery-3090/profile-aligned/benchmark5-exact8x-earlyready-routed-parity/).
Per-run provenance, usage-file hash, configs, commands, environment, logs, and
gates are in the
[three run directories](../../build_mtp_results/2026-10-31-ornith-recovery-3090/profile-aligned/benchmark5-exact8x-17g-perf-r1/)
and [usage frontier summary](../../build_mtp_results/2026-10-31-ornith-recovery-3090/profile-aligned/benchmark5-usage-profile-frontier.md).

This result is benchmark-weighted, not a general default unless the 8x prompt
frequency matches expected use. The equal-weight five-prompt survey remains
35.60 tok/s.

## VNNI-on candidate rejected

The host CPU advertises AVX-512 VNNI, while the measured ZeroMoE binary was
built with GGML_AVX512_VNNI=OFF. A separate VNNI-on build passes full-graph
execution but fails the teacher-forced routed-output comparison at prompt
pass 0, layer 0: max absolute delta 2.33e-10 across 7,787 unequal floats.
The delta grows through later layers, reaching 1.289 at decode pass 1,
layer 47.

Disabling custom cold fusion did not change the mismatch, so it is below that
helper and likely in the VNNI dot path. Router IDs and weights were not
captured in these VNNI runs. Do not use this binary until the pre-MoE hidden
state, route logits, ordered IDs, and normalized weights are traced at the
first divergent boundary. No VNNI performance run was made. Details are in
the [fused-path check](../../build_mtp_results/2026-10-31-ornith-recovery-3090/vnni-on/correctness-cpu-all-moe/correctness-status.md)
and [stock-path check](../../build_mtp_results/2026-10-31-ornith-recovery-3090/vnni-on/correctness-cpu-stock-vnni-on/correctness-status.md).

## Current workflow timing

The matched workload has three 24-token prompts, 200 generated tokens each,
context 224, and 10 decode threads. The performance checker uses steady graph
decode; the user-visible end-to-end decode rate includes request and token-loop
overhead.

- Single-prompt usage profile: 43.31–43.49 steady tok/s; 35.09–35.21
  end-to-end decode; 9.39–9.51 prefill; 24.23–24.35 overall.
- Equal-weight five-prompt survey: 35.60 steady tok/s; 29.87 end-to-end
  decode; 9.48 prefill; 21.67 overall.
- Five-prompt 8x benchmark-weighted profile: 42.50–42.54 steady tok/s;
  34.53–34.59 end-to-end decode; 9.35–9.50 prefill; 23.95–24.04 overall.

For the weighted profile, each run reports 9 prefill graph executions and
597 decode graph executions. The decode graphs take 14.03–14.05 seconds for
600 generated tokens. Startup expert preload reads 46.82 GB from the model
file in about 109.9 seconds; that is separate from prefill and steady decode.

Per-layer CPU lane timings are available from the built-in early-ready
scheduler counters without an evaluation callback. At calls=192, the weighted
profile sums to 9.97 ms of CPU lane wall time per token across 47 lanes
(mean 212 us per lane); the equal-weight five-prompt profile sums to 14.42 ms
(mean 307 us). These lane intervals overlap downstream GPU work, so they do
not add directly to request wall time. The per-layer rows and slowest lanes are
in the [early-ready lane timing report](../../build_mtp_results/2026-10-31-ornith-recovery-3090/profile-aligned/benchmark5-earlyready-lane-timing.md).

GPU counter sampling remains unavailable because host Nsight permissions deny
the required counters. Earlier normal CUDA tracing found GDN plus full
attention kernels at about 0.43 ms per token. Per-layer GPU compute attribution
remains open.

## Timing breakdown

The uninstrumented split run recorded 3.45 s prefill (4.63 tok/s), 82 ms
decode-graph preparation, and 18.48 tok/s request decode-wall; steady graph
decode was 18.69 tok/s. The hotset run's first request spent 3.12 s in prefill
(7.68 tok/s); subsequent prefill requests were about 1.9 s. The first decode
graph setup was 1.63 s; retained graph rebuilds were about 98 ms. These are
workflow timings, not per-layer GPU timings.

For a separate 20-token Nsight diagnostic (16.5 GiB profile, context 64), the
19 decode submit ranges totalled 951 ms. CUDA kernel time was 213 ms in total,
or 11.2 ms per token. H2D copies inside those ranges totalled about 92 MB and
10.2 ms. The 36 GDN and 12 full-attention layer kernels together used about
8.1 ms over all 19 tokens. The CPU scheduler trace showed ten ZeroMoE threads
busy for nearly the full decode window (about 10 core-seconds per wall-second).
This isolates CPU-lane saturation; it does not provide per-layer CPU timings.
Nsight GPU performance-counter sampling was denied by host permissions, but
CUDA kernel tracing succeeded without those counters.

Per-layer CPU timing is now available through
`ZEROMOE_QWEN4EXP_COLD_FUSED_TIMING=1`, which records inside the existing CPU
custom node without installing the graph evaluation callback. The 20-token
17.0 GiB diagnostic reported 48 layers, 19 calls per layer, and ten tasks per
node. The sum of per-layer mean wall times was 25.515 ms per token; layers 46,
43, and 44 were slowest at 870, 838, and 828 microseconds per call. Mean cold
route slots ranged from 0.89 to 3.84 by layer. The timer adds start/end barriers
inside the node, so this is a diagnostic, not an accepted throughput result.
Full rows are in [`cold-node-layer-timing.tsv`](../../build_mtp_results/2026-10-31-ornith-recovery-3090/profile-aligned/cold-node-layer-timing.tsv)
and the [summary](../../build_mtp_results/2026-10-31-ornith-recovery-3090/profile-aligned/cold-node-layer-timing.md).
With the timing variable unset, the fused-versus-stock CPU test passed 264 bit-identical cases.

The older `ZEROMOE_QWEN4EXP_LAYER_TIMING` scheduler callback remains unsuitable
for the normal path because it changes CUDA graph partitioning and disables
fusion. A follow-up design improvement is to generalize the fused-node timer
as a shared graph-runtime CPU timing interface; per-layer GPU timing remains
open.

## Memory and planner finding

The historical 16.5 GiB usage hotset selected 5,375 experts and covered
77.23% of survey routes. The current accepted 17.0 GiB profile selects 5,661
experts and covers 93.26% of the single benchmark-prompt survey; it passes strict
full-graph prefill with ZEROMOE_QWEN4EXP_PREFILL_OP_OFFLOAD=0 at context 224.
The separate 17.25 GiB experiment still fails its prefill workspace check.
These profiles use different contexts and surveys, so the later 17.0 GiB
result does not make the older budget comparison a same-context A/B.

The decode planner's `Fits` state does not prove that every prefill graph
fits. The run checker now rejects any logged `decode=0` graph-build failure or
prefill fallback when `--require-full-graph-prefill` is requested. Before this
fix, it accepted a log with successful prefill graphs even when other prefill
requests fell back.

See the [Qwen3.8-to-Ornith profile transfer audit](qwen38-to-ornith-profile-transfer-2026-10-31.md) for the exact settings that were transferred, adapted, or missed.

The Qwen3.8-to-Ornith profile audit records why several reference settings were not applied and the missing benchmark inputs: [profile transfer audit](qwen38-to-ornith-profile-transfer-2026-10-31.md).

## Artifacts

- Corrected matched profiles, runs, gates and provenance:
  [`2026-10-31-ornith-recovery-3090/profile-aligned`](../../build_mtp_results/2026-10-31-ornith-recovery-3090/profile-aligned/)
- Earlier 16.5 GiB post-fusion runs and the matched llama.cpp rows:
  [`2026-10-30-ornith-postfusion-3090/profile-aligned`](../../build_mtp_results/2026-10-30-ornith-postfusion-3090/profile-aligned/)
- Nsight trace, statistics and decode-window SQLite:
  [`hotset16_5-nsys20-host-only.nsys-rep`](../../build_mtp_results/2026-10-30-ornith-postfusion-3090/profile-aligned/hotset16_5-nsys20-host-only.nsys-rep),
  [`hotset16_5-nsys20-host-only.stats.txt`](../../build_mtp_results/2026-10-30-ornith-postfusion-3090/profile-aligned/hotset16_5-nsys20-host-only.stats.txt)
- Five-prompt Ornith usage survey:
  [`ornith-usage-survey.txt`](../../build_mtp_results/2026-10-30-ornith-122b-3090-match-24p200g10t/official-survey-5/ornith-usage-survey.txt)

## Next work

The 42 tok/s absolute target remains unmet. Qwen early-ready output selection
is architecture-specific: Qwen4Exp keeps the measured Qwen3.8 async default;
qwen35moe defaults to sync output. Strict memory preflight now refuses an
`Insufficient` plan before prompt execution rather than silently entering
layered fallback.

The 17.0 and 17.25 GiB batch-4 runs measured 34.44 and 34.71 tok/s, with only
0.54 points of survey-coverage gain. The 18.0 GiB candidate is unfit and was
rejected before generation. Do not increase the VRAM budget without a live
`Fits` plan.

The 80% per-layer floor prototype ran twice at 17.0 GiB. Both runs passed
graph gates and raised minimum layer coverage to 80.03%, but measured 33.70 and
33.53 tok/s versus 34.44 tok/s for the global-ranking batch-4 control. CPU PSI
peaked at 1.63 and 1.21 in the floor runs, so this did not establish a clean
speed regression or gain. The runtime knob was removed; the offline simulation
and lane reports remain as evidence. A future planner experiment needs
per-token route bundles and critical-lane timing, not marginal per-layer counts
alone. Any new placement still needs teacher-forced routed-output parity.

The five-prompt survey ranks the hotset; the matched runtime prompt repeats
gradient descent three times. Keep that 24/200 workload for the 42 tok/s gate,
and test a separate five-distinct-prompt runtime suite to validate survey
generalization. Async/sync lane telemetry points to different CPU/GPU overlap,
but GPU per-layer attribution remains open. Keep all acceptance on full-graph
decode and strict full-graph prefill. Do not reopen layer-split optimization
or change model equations without same-input teacher-forced boundary traces.
Continue using host-level CUDA execution and the 3090-only UUID.

## Latest guarded full-graph run

Run `early-ready-qwen35-default-equal5-perf-r2` used the Qwen3.8-compatible
async-output default on the RTX 3090, with the equal-weight five-prompt usage
profile, 17.0 GiB hotset, context 224, 10 threads, and strict full-graph
prefill. The run completed 600 generated tokens across three requests. Its
full-graph gate passed with 9 prefill graph runs, 597 decode graph runs, and
zero one-token layered forwards or legacy expert dispatches.

Steady graph decode measured 29.43 tok/s. The run's aggregate rates were
25.29 tok/s end-to-end decode, 8.43 tok/s prefill, and 18.59 tok/s overall.
The relative llama.cpp ratio passed at 1.782x, but the user-set 42 tok/s gate
failed. Host load was 0.67 / 6.35 / 9.84 at launch (1 / 5 / 15 minutes) and
19.05 / 13.06 / 11.90 at completion. Treat this as a host-contended diagnostic,
not a valid performance measurement. The current one-minute-only preflight did
not establish a stable run window.

Run artifacts, including stdout, engine log, TOML, prompt, GPU snapshots,
provenance, and gates, are in
[`early-ready-qwen35-default-equal5-perf-r2`](../../build_mtp_results/2026-10-31-ornith-recovery-3090/profile-aligned/early-ready-qwen35-default-equal5-perf-r2/).



## Latest host-level full-graph run

`early-ready-qwen35-default-equal5-perf-r3` used the Qwen3.8-compatible
async-output default on the RTX 3090, the 17.0 GiB equal-weight five-prompt
survey hotset, context 224, 10 CPU threads, and strict full-graph prefill.
The actual runtime prompt file repeated the same 24-token gradient-descent
prompt for three 200-token requests; it did not execute five distinct prompts.

The graph selected 5,561 experts, spent 16.999 GiB, and reported 81.58% route
coverage. The full-graph gate passed with 9 prefill and 597 decode graph runs,
zero one-token layered forwards, and zero legacy expert dispatches. Steady
graph decode was 32.60 tok/s; per-request decode rates were 32.32, 32.26, and
32.33 tok/s. Aggregate decode was 27.76 tok/s, prefill 9.58 tok/s, and overall
20.60 tok/s. The matched llama.cpp rate was 16.52 tok/s (1.974x), but the
42 tok/s absolute gate failed.

Built-in early-ready counters, which do not install a graph evaluation
callback, report 47 CPU lanes per request. Across the three requests, summed
CPU-node intervals were 14.399–14.488 ms/token and wait intervals were
11.844–11.937 ms/token. The slowest CPU lanes were layer 18 (443–446 us),
layer 31 (420–424 us), layer 19 (404–409 us), layer 15 (396–399 us), and
layer 13 (about 389 us). These lane intervals overlap CUDA work and must not
be added as serial wall time. Initial graph construction took 612.9 ms,
including 605.5 ms to copy dense weights, with 98.5 ms scheduler allocation;
the first retained-shape rebuild took 11.8 ms. Full-graph decode work was
18.315 s across 597 graph executions (30.68 ms per decode execution). No
per-layer GPU timing was collected; the one NVIDIA-SMI sample during graph
construction predates hot-expert placement and decode.

The two launch preflight samples were load averages 0.07 and 0.06. During
startup the one-minute average peaked at 28.81 and ended at 18.98. Captured
CPU PSI `some avg10` remained 0.00; concurrent visible processes included
Python at about 11%, Codex at 5.4%, and two OpenCode processes at 3.9% and
1.4%. This is diagnostic evidence, not a repeatable performance acceptance
run. The run's host monitor records the process and pressure samples; the
selected RTX 3090 was confirmed, while `nvidia-smi` intermittently failed in
the runner context.

The workspace sandbox could not enumerate CUDA (`cudaGetDeviceCount=0`). A
host-level execution with the user's 3090-only UUID succeeded: ZeroMoE selected
the RTX 3090 and reported 23.30 GiB free before model allocation. The model
file is 49,515,667,712 bytes. The resulting artifacts are in
[`early-ready-qwen35-default-equal5-perf-r3`](../../build_mtp_results/2026-10-31-ornith-recovery-3090/profile-aligned/early-ready-qwen35-default-equal5-perf-r3/).

The binary manifest now records GGML base
`8baabd53b77d1c48c6ef8d3b620436292f3f6231`, recovered from the build's pinned
`ZEROMOE_GGML_COMMIT`; the generated `GGML_COMMIT` header reports the same
revision. `tools/build_provenance.py` now falls back to this CMake pin when the
copied source tree has no Git metadata, and rejects a mismatch. The verified
manifest and checker output are included with the run artifacts.

## Matched sync-output control

`early-ready-qwen35-sync-equal5-perf-r4` used the same source snapshot,
binary, five-prompt survey hotset, 17.0 GiB expert budget, repeated
gradient-descent prompt, 24/200 workload, context 224, and 10 threads as r3.
The only runtime setting changed was
`ZEROMOE_QWEN_EARLY_READY_ASYNC_OUTPUT=0`.

The full-graph gate passed with 9 prefill and 597 decode graph executions,
zero one-token layered forwards, and zero legacy expert dispatches. The hotset
again selected 5,561 experts (16.999 GiB; 81.58% survey coverage). Steady
graph decode was 34.46 tok/s, compared with 32.60 tok/s in async-default r3
(about 5.7% faster). The r4 aggregate rates were 29.10 tok/s decode,
9.56 tok/s prefill, and 21.32 tok/s overall. It passed the relative llama.cpp
gate at 2.087x but failed the absolute 42 tok/s gate.

Both modes reported the same decode graph shape: 4,202 CUDA nodes, 49 CPU
nodes, and 98 scheduler splits. Lane counters differed: sync output recorded
GPU work pending in 182–192 of 192 lane calls, while async-default recorded
zero. Summed CPU lane work was 14.94 ms/token in r4 versus 14.44 ms/token in
r3; summed lane wait was 9.23 versus 11.89 ms/token. This suggests the
async-output fence changes CPU/GPU overlap, but the pair is not repeatable yet
and host conditions differed. The async branch enqueues an H2D copy on a
separate stream and then places a `cudaStreamWaitEvent` on the producer stream.
That fence is a candidate cause of the reduced overlap; the timing correlation
is an inference and needs a repeatable same-host A/B.

The RTX 3090 decode sample showed 22,596 MiB used, 49% utilization, 272 W,
with the ZeroMoE process as the only listed compute process. The sync run's
captured CPU PSI `some avg10` peaked at 1.89; other sessions were active, so
repeat the pair twice with CPU pressure held low before changing the adapter
default. Per-run logs, gates, lane timing, host samples, and the selected-GPU
snapshot are in
[`early-ready-qwen35-sync-equal5-perf-r4`](../../build_mtp_results/2026-10-31-ornith-recovery-3090/profile-aligned/early-ready-qwen35-sync-equal5-perf-r4/).


## Sync-output repeat and architecture default

A second explicit sync-output run, `early-ready-qwen35-sync-equal5-perf-r5`,
again passed full-graph gates and measured 34.06 tok/s steady decode (2.062x
llama.cpp), below the 42 tok/s absolute target. Its aggregate decode, prefill,
and overall rates were 28.84, 9.52, and 21.15 tok/s. Captured CPU PSI
`some avg10` peaked at 3.0, so this run is diagnostic for repeatability but
not a clean acceptance measurement.

The Qwen shared builder now selects the output mode by exact GGUF architecture:
`qwen4exp` keeps the measured Qwen3.8 async default; `qwen35moe` defaults to
sync output. The explicit `ZEROMOE_QWEN_EARLY_READY_ASYNC_OUTPUT` override still
supports either path. This is based on two explicit sync results and one async
result; the new default was then verified in a fresh isolated build.

Build `early-ready-qwen35-sync-default-clean-build` used source commit
`7d48e01de3ea2804ea6375b37858c23c909ad1d2`, binary SHA256
`672c1a2bcecdc57ce1feb5cc8992d9103c50097e8c7b94863882dc3bcb2c3b07`, CUDA
architecture 86, and GGML base `8baabd53b77d1c48c6ef8d3b620436292f3f6231`.
The r6 run logged `async_output_copy=0 source=adapter default`, selected the
3090, and passed strict full-graph prefill/decode gates with 5,561 experts at
16.999 GiB and 81.58% survey coverage. It measured 35.55 tok/s steady graph
decode (2.153x llama.cpp), 29.84 tok/s aggregate decode, 9.61 tok/s prefill,
and 21.74 tok/s overall. The 42 tok/s absolute gate remains unmet. Host PSI
`some avg10` peaked at 0.08; load average peaked at 24.42 during model load,
indicating that the high load average did not correspond to comparable CPU
pressure in this run.

The offline byte-stride budget estimate is in
[`ornith-budget-frontier-2026-10-31.md`](ornith-budget-frontier-2026-10-31.md).

## Strict memory-plan fail-fast verification

The r7 18.0 GiB attempt logged `memory_plan=insufficient` (23.97 GiB required,
23.28 GiB available), disabled graph prefill/decode, and then ran 597 layered
forwards despite `graph_preflight_strict=true`. The full-graph checker rejected
it. Its layered speed is not Ornith full-graph evidence.

I changed `resolve_cuda_moe_graph_policy()` so strict graph preflight rejects
an `Insufficient` plan before prompt execution. The rebuilt binary's r8
verification produced the explicit strict-preflight error, no request stats,
no graph-fallback gate, and zero generated tokens. Details and the validation
record are in
[`early-ready-qwen35-strict-insufficient-failfast-r8`](../../build_mtp_results/2026-10-31-ornith-recovery-3090/profile-aligned/early-ready-qwen35-strict-insufficient-failfast-r8/).

## 17.25 GiB strict full-graph placement run

The r9 candidate used the sync-output adapter default, an 17.25 GiB equal
survey hotset, prefill batch 4, and strict graph prefill. The live memory plan
reported `Fits` with 23.22 GiB required and 23.28 GiB available. It selected
5,643 experts with 82.12% survey route coverage. The graph gate passed with
3 prefill and 657 decode graph runs, zero layered forwards, and zero legacy
expert dispatches.

Steady graph decode measured 34.71 tok/s (2.102x llama.cpp), so the 42 tok/s
absolute gate failed. Aggregate decode was 33.48 tok/s, prefill 12.89 tok/s,
and overall 25.52 tok/s. The decode-time RTX 3090 sample showed 22,852 MiB
used and 48% utilization, with the ZeroMoE process as the only listed GPU
process. CPU PSI `some avg10` peaked at 1.75 and load average at 25.19; treat
this as a valid full-graph execution result but not a clean acceptance sample.
The 17.25 GiB profile adds only 0.54 survey-coverage points over 17.0 GiB and
did not improve decode in this run.

The strict gate passed; performance did not. Compare 17.0 and 17.25 GiB with
the same prefill batch under low CPU pressure before deciding whether the extra
residency helps. Artifacts are in
[`early-ready-qwen35-sync-default-17g25-equal5-perf-r9`](../../build_mtp_results/2026-10-31-ornith-recovery-3090/profile-aligned/early-ready-qwen35-sync-default-17g25-equal5-perf-r9/).


## Matched 17.0 GiB control and rejected layer-floor prototype

R10 repeated the qwen35moe sync-default workload at 17.0 GiB with prefill
batch 4 to match the r9 17.25 GiB candidate. The memory plan fit at 22.96 GiB
required / 23.28 GiB available; strict graph gates passed with 3 prefill and
657 decode runs and zero layered forwards. Steady decode was 34.44 tok/s
(2.087x llama.cpp), with 28.84 tok/s aggregate decode, 9.52 tok/s prefill,
and 21.15 tok/s overall. CPU PSI `some avg10` remained 0.00.

A temporary runtime prototype applied an 80% per-layer route-coverage floor
at the same 17.0 GiB budget. R11/R12 both passed strict full-graph gates and
selected 5,587 experts at 81.67% total coverage and 80.03% minimum layer
coverage. They measured 33.70 and 33.53 tok/s. CPU PSI peaked at 1.63 and 1.21
in those runs versus 0.00 in R10, so the throughput comparison is not a clean
A/B. The floor lowered CPU work in some layers but moved the slowest lane to
layer 31 (479/488 us), then layers 35 and 39. It did not demonstrate a speed
gain and was removed from the runtime code/config; the offline simulator and
run evidence remain.

The r11/r12 binaries are archived as `zeromoe-used-layerfloor` with their
run manifests. They are historical builds and should not be used as the
current default binary. The current source and strict sync-default build use
global hottest-first placement with the per-layer floor unset. A future
placement objective needs per-token route bundles and critical-path timing,
not marginal per-layer counts alone. Per-layer GPU attribution remains open.

R10 artifacts are in
[`early-ready-qwen35-sync-default-17g-batch4-equal5-perf-r10`](../../build_mtp_results/2026-10-31-ornith-recovery-3090/profile-aligned/early-ready-qwen35-sync-default-17g-batch4-equal5-perf-r10/).
The exploratory floor runs are in
[`R11`](../../build_mtp_results/2026-10-31-ornith-recovery-3090/profile-aligned/early-ready-qwen35-sync-layerfloor80-17g-equal5-perf-r11/)
and [`R12`](../../build_mtp_results/2026-10-31-ornith-recovery-3090/profile-aligned/early-ready-qwen35-sync-layerfloor80-17g-equal5-perf-r12/).

## R13 route-bundle capture: layer attribution is missing

Ran the prepared five-prompt, 64-token persistent survey on the RTX 3090 only,
using the provenance-checked sync-default binary and the 17.0 GiB usage-ranked
profile. The graph policy reported `required` / `fits` with 24,658,173,050
required bytes and 24,996,347,904 available bytes. The survey exited successfully
and wrote five prompt rows. This was an instrumented placement diagnostic, not
a performance measurement; no tok/s result is accepted from it.

The trace contains 33,652 `[MOE-ROUTE-BUNDLE]` records, but every record says
`layer=-1`. The summary parser therefore reports zero usable records and zero
layers. The capture cannot establish per-layer route co-occurrence, so it must
not drive a new VRAM hotset. The source logger or its caller is missing the
layer index; fix that data handoff and make the trace reject negative layer
IDs before repeating the survey. The run and raw logs are in

[`route-bundle-survey5-64t-r13`](../../build_mtp_results/2026-10-31-ornith-recovery-3090/profile-aligned/route-bundle-survey5-64t-r13/).

The run was pinned to GPU UUID `GPU-fbddd113-32a1-56af-9c8e-7f2c877a9736`
(RTX 3090). One live sample showed 22,596 MiB and 32% utilization; after the
survey process exited, usage returned to 526 MiB. The host was initially quiet,
but its one-minute load average reached 23.35 during preload on this shared
machine, so this run is not a clean performance sample. No matched timing was
collected, and the 42 tok/s target remains unmet; the last uninstrumented
steady-state result remains R6 at 35.55 tok/s.

Next, propagate the actual layer index into the trace and add a fail-fast
assertion for missing layer IDs. After resuming, rerun the five-prompt capture
only when the shared host is quiet, then use valid per-layer bundles alongside
the critical-lane timing before changing placement. Keep Ornith in progress
until correctness and the matched performance gates pass. Work was paused at
that R13 handoff per the user's request, then resumed; the continuation results
and open work are recorded below.

## R14-R19: layer trace repair and longer route-bundle sample

R13 exposed that every route record had `layer=-1`. The Qwen builder knew the
layer, but constructed `EarlyReadyLane` without passing it; the shared graph
builder used by GLM and other adapters also default-constructed the lane. The
lane now requires a nonnegative layer at construction, both builders pass their
existing layer index, and opt-in route tracing asserts the layer is valid. The
R14 binary was built separately so the archived R6 binary was not replaced.
Its provenance check passed for source `7d48e01de3ea2804ea6375b37858c23c909ad1d2`
and GGML base `8baabd53b77d1c48c6ef8d3b620436292f3f6231`; binary SHA256 is
`be8f8ef875ef2932538fd6ecd4bd3f2c712078b13ad85051ee4fa9aea9a73b6d`.

The first R14 launch had a runner path error and did not start the model. R14
and R15 then confirmed that the 17 GiB full-graph profile eagerly reads the
complete 38.77 GiB DDR expert source. Disabling that preload made Qwen's
full-graph source binding reject the request, so it is required for this
adapter. Idle-class I/O priority did not remove the read cost; R17 kept
preload concurrency at 1/1 and completed without CPU pressure. The startup
read still raises I/O PSI on the shared host, so continue to require an idle
3090 and a low pre-run host-load gate.

R17 completed the five standard prompts at 64 generated tokens each. Its
strict full-graph gate passed: the memory plan fit, with zero one-token layered
forwards and zero legacy expert dispatches. The repaired trace has 33,652
records, exactly 716 complete decode events across all 47 early-ready layers.
An offline reconstruction from the GGUF tensor byte sizes selected the same
5,561 experts and 81.58% historical usage coverage as the runtime plan. That
cross-check validates the route-to-layer labels and the selection calculation.

R18 requested 200 output tokens at context224. The full-graph memory plan fit,
but the first request exhausted KV capacity after 139 generated tokens. It is
an incomplete capture, not a placement failure or performance result. R19
repeated the five prompts at context384; the strict plan fit at
24,666,037,370 required bytes / 24,996,347,904 available. All five prompts
completed at 200 tokens and `check_full_graph_moe_run.py --full-fit` passed.
R19 recorded 65,612 route rows: 1,396 complete token/layer events, including
401 prompt-side continuation events and 995 generated-token forward events.

The 17 GiB marginal-frequency hotset covered 72.43% of routed expert IDs and
74.43% of route weight on generated tokens in R19, but only 18.70% of
generated token/layer bundles were fully hot. The generated path averaged
2.21 cold experts per layer event. Layers 12-18 are the weakest: their
all-hot rates range from 6.5% to 9.5%. R6's uninstrumented lane counters put
layers 18, 19, 13, 15, 14, 11, and 16 among the highest CPU-work lanes; across
layers, R6 CPU work correlates with R19 cold experts at Pearson 0.479 and
anti-correlates with all-hot rate at -0.621. This is cross-run diagnostic
evidence, not a causal performance measurement.

I simulated within-layer hotset swaps at the same per-layer byte budget.
With the 64-token R17 trace, optimizing four prompts raised training all-hot
coverage from 23.84% to 45.77% but reduced the held-out prompt from 15.23% to
11.89%. On the longer R19 trace, an eight-swap cap raised four-prompt training
coverage from 21.32% to 29.89%; the held-out prompt moved only from 8.22% to
8.31%, while its average cold routes worsened from 2.92 to 3.04. These
experiments show that a per-token bundle objective can overfit the small prompt
set. No candidate hotset was applied to the runtime.

R17 and R19 are instrumented diagnostics, not performance acceptance runs.
R19's route logging changes lane overhead, and its preload worker count differs
from the performance profile. The last accepted uninstrumented result remains
R6 at 35.55 tok/s steady-state decode; the 42 tok/s target and the relative
llama.cpp comparison remain open. Do not use R17/R19 rates as replacement
performance results.

Artifacts:

- [`R17 route trace and analysis`](../../build_mtp_results/2026-10-31-ornith-recovery-3090/profile-aligned/route-bundle-survey5-64t-r17/)
- [`R18 context failure`](../../build_mtp_results/2026-10-31-ornith-recovery-3090/profile-aligned/route-bundle-survey5-200t-r18/)
- [`R19 longer route trace and analysis`](../../build_mtp_results/2026-10-31-ornith-recovery-3090/profile-aligned/route-bundle-survey5-200t-context384-r19/)

Next steps: run the required same-input llama.cpp versus ZeroMoE teacher-forced
block-boundary comparison for pre-layer hidden states and routed expert IDs /
weights; stop at the first mismatch and save the small repro. Then use a larger,
held-out route-bundle sample to develop a regularized placement objective. Only
after correctness passes should a new uninstrumented, matched 3090 performance
run test whether placement reaches the 42 tok/s absolute target and the default
90% llama.cpp bar.

## R20-R25: mixed-hotset correctness is still open

R22 verified the current full-graph all-CPU/DDR expert path against the
minimal llama.cpp routed-output trace: all 96 layer/pass outputs were
bitwise equal. This did not cover the 17 GiB mixed VRAM/DDR hotset used for
performance.

R23 was a performance-only run. The 17 GiB exact-8x benchmark-weighted
profile measured 42.71 tok/s steady graph decode against 16.52 tok/s in the
matched llama.cpp benchmark; both the 42 tok/s absolute and 90% relative
gates passed for that profile. The 8x weighting is benchmark-prompt-specific
and remains experimental. It does not establish numerical correctness.

R24 was the first minimally tapped teacher-forced check of the mixed hotset.
It made no source-code changes: it used the existing R14 binary, context 128,
the three-token `Hello there.` prompt and forced token 353, the 17 GiB exact-8x
profile, all-split early-ready with synchronous output, and final routed
output taps only. Prompt prefill outputs matched bitwise. The forced decode
pass first differed at the layer-0 routed output (maximum absolute error
`4.395432e-05`). The output delta grew through later layers, reaching
`0.050619` at layer 44 before decreasing. The report was produced
at 2026-10-31 13:53:06 +1100. This is a mismatch against llama.cpp with all
MoE experts on CPU while ZeroMoE used mixed VRAM/DDR experts; it locates the
first difference but is not a same-placement CUDA parity result. Do not
claim mixed-hotset correctness yet.

The R17/R19 usage surveys recorded ZeroMoE's own routed experts; they did not
compare llama.cpp and ZeroMoE expert IDs/weights over the five-prompt survey.
Thus figures such as 81.58% coverage describe the current ZeroMoE routes only.
If its router chooses different experts, those survey rankings and hotset
coverage do not describe the reference model. Treat the survey-derived
placement and R23 throughput as provisional until teacher-forced route IDs and
weights match on the intended prompts/tokens. The next long paired trace
should both compare every layer's routing and collect survey usage from those
same forced sequences; a one-token trace is only a fast reproducer.

The layer-0 mismatch should have been gated before R23 performance work. R22
covered only the CPU/DDR placement, R23 had no correctness capture, and the
earlier all-stage probes changed graph scheduling. Those runs did not satisfy
the mixed-path correctness gate.

R25 added `ZEROMOE_QWEN4EXP_PROBE_DUMP_LAYER` so a diagnostic graph can retain
full stage taps for one selected layer. The first R25 process did load and
run on the RTX 3090, but the copied TOML still pointed `log_path` at R24 and
the R25 tensor directory had not been created. The gate therefore failed;
the run produced no usable stage capture. It overwrote R24's raw engine log,
which was copied into the R25 directory. R24's routed tensors and comparison
report remain intact. This artifact mistake is recorded in
[`R25 incident notes`](../../build_mtp_results/2026-10-31-ornith-recovery-3090/correctness/layer0-selective-boundary-probe-r25/incident-and-next-step.md).

Do not rerun while the shared-host load gate is above threshold. At 14:26,
one-minute load was 6.43. The 3090 remained visible in the full inventory at
1 MiB used and zero compute processes were reported for its UUID. R26 is prepared with isolated log/tensor paths and copied
reference tensors; its layer-0-only llama.cpp reference helper is built
against the recorded `8baabd` source and has no unresolved shared libraries.
The preflight now queries full `nvidia-smi` inventory/process output and
filters by the 3090 UUID, because NVML queries failed intermittently but then
succeeded on retry. After the host gate passes, verify R26's layer-0 captures,
then run the layer-0-only llama.cpp reference trace with the same
teacher-forced prompt/token. First classify hidden input, router
logits/probabilities/IDs/weights, and expert outputs; only then decide whether
the difference is expected CPU/CUDA arithmetic or a graph-path correctness
bug. No R25 correctness or performance result is accepted yet.
