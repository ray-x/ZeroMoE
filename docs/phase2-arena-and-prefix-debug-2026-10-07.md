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
regression. No representative post-fix throughput A/B has been run yet; the
default early-ready path is unchanged.

## Remaining verification

The arena-alias defect and its fix are now reproduced and verified. The staged
path still needs a representative multi-pair throughput comparison against
early-ready, plus a CPU host-callback run and a mixed resident/staged route
case. Keep early-ready as the production default until those checks establish
an acceptable gain without coherence loss.
