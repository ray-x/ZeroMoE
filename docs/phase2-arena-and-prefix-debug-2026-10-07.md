# Phase-2 arena and prefix corruption debug (2026-10-07)

## Current evidence

The retained two-token run is
`logs/phase2_layer3_up_weight_verify_20261007/run18_optionb_cuda_node_scan/`.
It establishes two separate findings:

1. The Layer-3 Option-B up-weight tensor matches its DDR source when the
   decode graph is built (`candidate.engine.log:358`), but 174 of 198 planes
   differ after Prefix replay and before the split-layer suffix
   (`candidate.engine.log:1275`). Thus the Prefix execution path has already
   changed resident weights before the completion payload is uploaded.
2. The staged suffix graph also has an arena allocation defect. One staged
   up-weight tensor overlaps the Option-B Layer-3 resident tensor by 4.6 MiB
   (`candidate.engine.log:953` and `:1287`); another sibling graph's staged
   tensor does not overlap (`candidate.engine.log:542`). The completion
   payload upload can therefore overwrite resident weights later in the same
   token. This is a confirmed graph-arena bug, but it cannot explain the
   earlier post-Prefix mismatch by itself.

The captured graph had no overlapping memcpy or memset destinations and the
GGML graph scan had no overlapping output tensors. The partial-layer-only
Layer-3 path preserved its weights, which makes the generic Q2_0 MMID kernel
less likely, but that is not a matched Option-B control.

## Root cause and verification

`phase2_bind_or_allocate_expert()` and named expert lookup now follow the
`phase2_expert_source_graph` chain to the graph that owns the shared arena,
address map, and bump cursor. This fixes the case where a staged child borrows
from Prefix while Prefix itself borrows the retained decode arena; the old
one-hop lookup allocated from Prefix's reset cursor. Opt-in
`QWEN4EXP-P2-STAGE-ALLOC` records graph, direct source, canonical owner,
cursor, and address. The engine target builds successfully.

This was the cause of the reproduced corruption. The staged child had bound
its 4.6 MiB up-weight payload tensor inside the Layer-3 Option-B resident
range. The later graph borrowed from a borrower, whose cursor/map did not own
the shared buffer. Following the chain restored the arena owner's address map
and cursor: the staged range no longer overlaps the resident range, and the
resident up-weight data remains identical after Prefix replay.

## Runtime verification

The retained runner directories contain their exact config, input command,
resource samples, model output, and logs. Both reference comparisons use the
same 28 prompt IDs and the same five teacher-forced continuation IDs.

| Run | Phase-2 resident set | Staged path | Result |
| --- | ---: | --- | --- |
| `run19_kernel_param_scan` | top 64 | not selected; all 10 routes resident | `8160, 369`; post-Prefix weights match 198/198; stage overlap 0 |
| `run20_forced_cold_stage` | top 1 | selected, Ready payload, staged conditional | `8160, 369`; top-1 agrees on 2/2 logits; stage overlap 0 |
| `run21_forced_cold_stage_5tokens` | top 1 | selected on all four decode steps | `8160, 369, 264, 4145, 15673`; top-1 agrees on 5/5 logits; stage overlap 0 |

The five-token comparison against the retained early-ready reference measured
mean KL `0.000382`, mean total variation `0.00680`, and no top-1 disagreement.
The captured-kernel scan reported zero pointer arguments inside the protected
weight range (205 kernel parameter scans were unavailable, so that scan alone
is not a complete writer proof); independently, the post-Prefix full-plane
comparison matched all 198/198 planes at positions 28 through 31.

RAM/VRAM monitoring recorded no OOM kills, no Xid event, minimum sampled
`MemAvailable` of about 81.8 GiB, and peak RTX 3090 use of about 16.2 GiB.
The pre- and post-run GPU queries showed both devices idle. These are bounded
correctness checks, not a throughput A/B.

The forced-cold five-token run includes the normal Option-B CPU-resident
layers in its Prefix/Tail graphs (44 of 48 layers have a static hot set; the
other four remain on CPU). Its setup-inclusive rate was about 1.02 tok/s with
`PARTIAL_LAYER_TOP_N=1`, which deliberately sends nearly every Layer-13 route
through storage and staged GPU execution. This stress number is not comparable
to the accepted 0/44/4 scheduler result and is not a measured production
regression. The default early-ready path is unchanged.

## Post-commit staged-path performance check

The 64-token diagnostic A/B used the validated upstream-master early-ready
binary for the baseline and the current binary for the staged candidate. In
the logit-dump run, top-1 agreement was 95.31%, with the first disagreement
at step 6: the reference token ranked second in the candidate, with a 0.147
reference margin. Because logits were being written, its timing is invalid. A
separate no-dump one-pair run measured 33.79 tok/s for early-ready and 14.13
tok/s for staged (−58.2%). This is a diagnosis, not an acceptance result.

Before the follow-up handoff edit below, removing the handoff-only override
made the staged graph diverge at step 1 (`369` reference versus `8160` staged).

An opt-in adaptive diagnostic produced one CPU warmup average near 45 ms and a
staged estimate near 65.7 ms. The staged sample measured storage queue 0.069 ms,
storage service 18.691 ms, H2D 1.136 ms, GPU work 3.974 ms, and 41.823 ms in the
remaining dispatch/payload-preparation component. The policy selected CPU
after calibration. Its one-pair no-dump run measured 34.22 tok/s early-ready
versus 16.99 tok/s adaptive (−50.4%) and first diverged at token 27. The large
dispatch residual and slow CPU suffix point to split-graph orchestration and
the CPU path as current performance bottlenecks; data transfer and GPU kernel
work are much smaller. These single-pair measurements guide profiling but do
not establish a final throughput ranking.

## Handoff follow-up in the worktree

The controlled flag test showed that the graph-owned persistent CPY output
matches the early-ready reference, while the post-replay copy from
`prefix.final_streams` does not. `refresh_prefix_handoff()` now treats the
retained CPY output as authoritative by default and no longer refreshes it
from that scheduler intermediate. This follow-up is not committed yet.

With no handoff debug flag and no frozen-FFN/weight overrides, a five-token
teacher-forced run matched the early-ready reference (top-1 5/5, mean KL
`0.000581`). A 64-token no-dump one-pair run then measured 33.18 tok/s
early-ready versus 14.18 tok/s staged (−57.3%). The staged tokens matched the
reference for the first six IDs and first diverged at step 6 (`1439` versus
`28758`), again a near-tie. The handoff correction removes the step-1 failure;
the later numerical drift and large throughput gap remain open.

## Remaining verification

The arena-alias defect is fixed in commit `521d67a`. The handoff follow-up and
performance work remain uncommitted. The staged path still needs multi-pair
throughput testing after the numerical drift is resolved, plus a CPU
host-callback run and a mixed resident/staged route case. Keep early-ready as
the production default until those checks establish an acceptable gain without
coherence loss.

## 2026-10-07 staged-path phase timing

The existing coarse P2 timer grouped storage waiting, payload upload,
conditional dispatch, and the tail replay under `suffix_submit_sync`; on the
staged path `route_decision` also remained zero. At the time, the source emitted a phase ledger under
`ZEROMOE_QWEN4EXP_GRAPH_PHASE_TIMING=1`; that one-off setting has since been removed. Current per-layer timing follows
`--verbosity 5`. The historical source emitted a
`GGML-QWEN-FULL-DECODE-P2-STAGED-PHASES` ledger for each completed token. It
separates storage request wall time and service queue/service measurements,
payload preparation and H2D, both split-child scheduler replays, event-wait
enqueue costs, the three-way conditional's host and CUDA-event spans, the tail
replay, and upload-event synchronization. The staged route timer now includes
its route graph replay and host classification instead of reporting zero.

`cmake --build build_ggmlcuda -j2 --target zeromoe` passed with this
instrumentation. The retained bounded attempt is
`logs/phase2_layer3_up_weight_verify_20261007/run32_phase_breakdown_8tok/`.
It did not reach model inference: the baseline process reported
`cudaGetDeviceCount` returned zero and exited before creating an engine log.
An immediate full `nvidia-smi` query also temporarily failed, while a later
`nvidia-smi -L` again enumerated both GPUs. The preflight sample had 123 GiB
`MemAvailable`; a follow-up had 117 GiB, and neither `dmesg` nor the kernel
journal showed an OOM or NVIDIA Xid entry. Because device access was
intermittent, no retry was made at that point. A later full query recovered;
the candidate-only run in `run33_phase_breakdown_candidate_8tok/` then
completed under active memory sampling, as recorded below. Run 32 remains a
failed preflight, not a performance or correctness result.

The eight-token run matched the reference token IDs at 7/8 positions and first
diverged at step 6 (`1439` reference, `28758` candidate), reproducing the
64-token near-tie. This is a short correctness/profile run; its output rate is
not a throughput acceptance measurement. For staged positions 30–34, mean P2
time was 67.25 ms: prefix/handoff 15.22 ms, route replay and classification
0.16 ms, post-storage dispatch/tail 43.09 ms, and output readback 0.19 ms.
Within the staged ledger, storage service averaged 4.88 ms, payload
preparation 2.42 ms, H2D 1.14 ms, conditional-graph host time 2.78 ms, and
conditional device time 2.74 ms. The tail replay averaged 40.31 ms wall time,
while its scheduler synchronization took 1.15 ms. The remaining roughly 39.2
ms is therefore on the `ggml_backend_sched_graph_compute_async()` submission
side of the tail replay, making tail graph submission the leading measured
cost. The exact reason for that host-side cost is still unverified; the
non-invasive replay ledger is the next diagnostic.

The monitored run sampled minimum host `MemAvailable` at 79.86 GiB and peak
RTX 3090 allocation at 16,296 MiB. The `/proc/vmstat` OOM-kill count stayed
0; no NVIDIA Xid or OOM entry appeared in the kernel log, and the in-run GPU
queries had no failures. The cgroup memory-event file was unavailable. The
resource samples, generated tokens, phase rows, input hashes, and runner are
retained in `run33_phase_breakdown_candidate_8tok/` and
`run33_phase_breakdown_candidate_8tok.py`.

An additional graph-replay-ledger run was prepared in
`run34_graph_replay_ledger_8tok.py`, but it did not launch inference: the GPU
admission query returned status 9 with an NVIDIA driver communication error.
The A/B runner had treated any status 9 as “no compute processes”; it now
accepts that status as idle only when the command emitted no diagnostic text.
The mocked regression check passed for both empty-status-9 and
status-9-with-driver-error cases. Retry the replay ledger only after the full
GPU management queries are consistently healthy.

## 2026-10-07 scheduler profile: staged-tail bottleneck

The run-33 phase ledger showed a roughly 40 ms tail replay but did not separate
CPU backend work from scheduler dispatch. The replay ledger also initially
reported stale warmup timings for the tail; the worktree now records the tail's
own submit/sync values and labels its entry `instance=tail`. It also records
Prefix and Route separately when the staged continuation replays them outside
the parent conditional.

The built-in GGML scheduler profiler is available in this checkout through
`BEE_GGML_PROFILE=1` and `BEE_GGML_PROFILE_SPLITS=1`. It does not enable the
eval callback that made the conditional child handles unavailable in
`run39_cpu_layer_ledger_16tok/`. The monitored staged sample is
`run40_scheduler_profile_8tok/`; its mixed resident/staged branch reached
Ready and generated 8 tokens, with the same first top-1 mismatch at step 6.
The matched early-ready reference in `run38_early_ready_replay_ledger_8tok/`
matched all 8 token IDs.

On staged positions 30–34, total P2 time averaged 66.4 ms. The corresponding
early-ready full-graph replay took about 25–27 ms. The staged tail scheduler
reported 69 splits: 34 CPU splits, each a four-node `MUL_MAT_ID` graph, and 35
GPU splits. Across the six post-first-stage tail replays, the scheduler spent
39.05 ms per token, including 27.05 ms of CPU graph compute, 0.40 ms submitting
GPU work, and 1.14 ms in GPU synchronization. The CPU splits averaged about
0.80 ms of compute and 1.08 ms of host time apiece. The pass also reported
3.76 MB of H2D data, 0.35 MB of D2H data, and no event wait or backend-sync
time. These measurements identify CPU `MUL_MAT_ID` work and per-split handling
as the staged tail's main cost; storage service averaged about 4.9 ms and
upload about 1.1 ms. The opt-in early-ready composition below moves those CPU
nodes to private graphs, but the serialized tail scheduler still remains on
the critical path.

This is a diagnosis, not an accepted performance result. Keep early-ready as
the production default. The next implementation work should make the staged
Layer-13 branches coexist with the validated early-ready execution for the
rest of the full graph, then repeat coherence and paired throughput checks.
Run 40 sampled minimum host `MemAvailable` of 79.63 GiB, peak RTX 3090 use of
15,960 MiB, no GPU query failures, no OOM-kill counter increase, and no kernel
OOM/Xid entry. Its resource samples and raw scheduler profile are retained
with the run artifacts.

## 2026-10-07 Phase-2 early-ready composition experiment

To test whether the staged tail could retain the accepted private CPU graphs,
the worktree adds an opt-in `ZEROMOE_QWEN4EXP_PHASE2_EARLY_READY=1` path for
non-split layers in Prefix and Tail. A matching scheduler hook patch lives in
`patches/ggml-host-node-beellama/0004-ggml-scheduler-early-ready-private-cpu.patch`.
The test build applied it to a copy of the Beellama GGML tree under `/tmp` and
built the patched `ggml-base`; it reused the existing CUDA backend. The shared
Beellama checkout was not edited. The test executable exported the callback
and preloaded the copied `libggml-base`.

The first run (`run42_phase2_earlyready_8tok/`) caught a Phase-2 lifecycle bug:
Prefix and Tail private CPU graphs kept their construction-time
`expected_pos=-1`, while the ordinary full-graph path resets `expected_pos`
and `completed` every token. The run aborted on that assertion before
producing a token; OOM-kill counts stayed zero. Phase-2 now resets those fields
in each participating graph before replay. Runs 43 and 45 then completed 8 and
32 tokens. Both kept the staged conditional Ready path; run 45 matched 29/32
token IDs, with disagreements at steps 6, 21 and 27. For the 24-call counters,
early-ready ran in every sampled Prefix/Tail layer, with average event wait
around 0.22–0.23 ms and private CPU graph compute around 0.80–0.83 ms per
layer. Yet the staged P2 loop stayed near 65–70 ms per token. This opt-in
composition therefore did not recover the accepted path's throughput, and
early-ready remains the default.

Run 48 repeated the 32-token Phase-2 early-ready test without the verbose BEE
split profiler. It matched 29/32 token IDs at the same three positions and
reported a mean 68.57 ms P2 time for positions 30–58, with a 14.08 tok/s
first-to-last output interval. Its sampled early-ready callbacks still took
about 0.83 ms CPU work and 0.23 ms event wait per layer. This confirms that
adding the private CPU graph markers to Prefix/Tail alone does not make the
staged path competitive.

Run 49 is the same-binary 32-token diagnostic pair: full-graph early-ready
measured 17.77 tok/s, while Phase-2 early-ready plus resident/staged
conditionals measured 14.58 tok/s (−18.0%, 0/1 paired wins). The generated
tokens agreed at 29/32 positions, first diverging at step 6. It is one pair,
not an acceptance result, but it confirms that the tested Phase-2 composition
still loses against early-ready on a controlled binary and history.

A clean 40-token full-graph early-ready run on the current test executable
matched all 40 reference IDs and reported 17.65 tok/s. It used a different
GGML build from the retained accepted 33.87 tok/s binary and was a single
diagnostic run. Run 49 provides the better same-binary comparison, while the
cross-build throughput gap still needs separate investigation. Runs
43/45/47/48/49 kept minimum sampled `MemAvailable` near 79–80 GiB, peak 3090
usage near 15.4–16.3 GiB, zero in-run query failures and no OOM-kill-count
increase; no NVIDIA Xid/OOM log appeared.

The direct CPU-only unit checks for route classification (all-hot,
all-cold, mixed and duplicate slots), storage Ready/Error generations,
staged/adaptive boundaries, the adaptive policy, and the shared shape-class
staging registry all pass. They were compiled directly to avoid the aggregate
component script's unrelated EXL3 support sources. The current `zeromoe`
target also builds successfully with the added opt-in timing and Phase-2
early-ready code.

## Bounded storage wait follow-up

The staged request path previously waited indefinitely for the storage worker
after submitting a generation-tagged request. It now uses
`ZEROMOE_QWEN4EXP_STORAGE_TIMEOUT_MS` (default 100 ms, clamped to 30 s).
On timeout the current token takes the existing exact CPU suffix. The storage
read itself is not cancelled; when its worker completes, it discards the late
result and releases that generation's bounded slot. This prevents a late
completion from being mistaken for the current token and prevents timed-out
requests from permanently consuming service slots.

`tests/test_qwen4exp_storage_service.cpp` now covers a deliberately delayed
read, timeout reporting, stale-handle wakeup, and reuse of the retired slot
under a new request generation. The CPU-only storage test passes. The CUDA
engine target also rebuilds successfully. The inference-level timeout fallback
has not yet been exercised; no GPU inference or hardware test was run for this
change.

## Mixed resident, staged, and cold-host parent graph

The Phase-2 staged parent now includes all three route owners in one merge:
the resident and staged GGML children remain pure CUDA conditional bodies,
while a pre-warmed CPU callback is a sibling host node in the parent graph.
The parent copies the split-layer activation and route snapshot to pinned host
buffers, classifies each original route slot once, computes only CPU-owned
routes from the existing DDR expert arena, uploads that partial output, and
merges resident + staged + CPU contributions. The CPU runner handles one
cold route at a time, so it does not recompute resident or staged experts.
All branches retain the original router weight for their own route slots.

The completion copy stays on its dedicated upload stream. The parent joins
its ready event after route-bitmap preparation, then unpacks the published
payload inside the staged conditional. This removes the previous duplicate
payload copy from the parent graph. A storage timeout/error leaves the staged
bitmap empty; the same parent graph still runs any resident routes and sends
the cold remainder through the CPU callback. Resident child-handle warm-up is
queued after the storage request so it can overlap the read.

The `zeromoe` target builds, and direct CPU tests for exact-once route
classification, storage generations/timeouts, and adaptive timing pass. The
current `nvidia-smi` preflight returned a driver communication error, so this
new graph composition has not had a GPU replay or post-change A/B. Keep the
validated early-ready scheduler as the default. The older same-binary Phase-2
run remains a rejection result and does not validate this new composition.
