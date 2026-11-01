# Ornith 1.5 122B

## Support status

Ornith is a `qwen35moe` GGUF and uses the Qwen35 MoE full-graph adapter. The
port is in the unmerged `recovery/ornith-122b-offload` worktree. It is not the
Qwen3.6 35B model. CUDA decode remains full-graph only; model-specific block
math lives in the Qwen35 graph builder and uses the shared graph runtime,
routed-MoE builder, usage survey, and split expert arena.

The inspected reference is llama.cpp
`8baabd53b77d1c48c6ef8d3b620436292f3f6231`, at
`src/models/qwen35moe.cpp` and `src/models/models.h`. The family path is:

1. `detect_model_family()` recognizes the Qwen35 GDN gate plus expert metadata
   and returns `ModelFamily::Qwen`.
2. `full_graph_model(ModelFamily::Qwen)` selects `Qwen35MoEGraphModel`.
3. The adapter whitelists `general.architecture = "qwen35moe"` and validates
   the exact required tensor shapes before graph allocation.
4. `build_qwen35moe_graph_layer()` builds Qwen35 residual/block order while
   reusing the shared full-attention, GDN, routed-MoE, cache, and offload code.

## Current recovery status — 2026-10-31

Support remains in the unmerged `recovery/ornith-122b-offload` worktree. The
adapter accepts only `general.architecture = "qwen35moe"`; CUDA decode is
full-graph-only. Teacher-forced routed outputs match llama.cpp on the supported
path, and strict full-graph prefill/decode gates pass with zero layered
forwards or legacy expert dispatches.

The Qwen early-ready default now follows the exact architecture: `qwen4exp`
keeps async output for Qwen3.8, while `qwen35moe` uses scheduler-managed sync
output. The new default-sync build passed provenance checks with GGML base
`8baabd53b77d1c48c6ef8d3b620436292f3f6231`. On the RTX 3090, a 17.0 GiB
hotset selected 5,561 experts and covered 81.58% of the equal-weight survey.
Steady full-graph decode measured 35.55 tok/s; end-to-end decode was
29.84 tok/s, prefill 9.61 tok/s, and overall 21.74 tok/s. The 42 tok/s target
is still unmet.

The matched output-copy comparison measured 32.60 tok/s async-default and
34.46/34.06 tok/s in two explicit sync runs; the new adapter-default sync run
reached 35.55 tok/s. The async/sync lane counters suggest different CPU/GPU
overlap, but the comparison needs further repeatable measurements. The
five-prompt survey ranks placement; the runtime benchmark repeats the
24-token gradient-descent prompt three times. A five-distinct-prompt runtime
suite remains open.

An 18.0 GiB candidate required 23.97 GiB with 23.28 GiB available and was
rejected before generation by strict graph preflight. A 17.25 GiB, prefill-
batch-4 run fit at 23.22/23.28 GiB, passed full-graph gates, and measured
34.71 tok/s with 82.12% survey coverage; it did not meet 42 tok/s.
An 80% per-layer floor prototype ran twice at 17.0 GiB with strict graph
gates. It raised minimum survey coverage to 80.03% but measured 33.70/33.53
tok/s versus 34.44 tok/s for global ranking; CPU PSI was higher in the floor
runs. The runtime knob was removed; the offline simulation remains documented,
but no speed benefit was established.
See the [recovery handoff](../handoff/ornith-122b-recovery-2026-10-31.md),
[Qwen profile audit](../handoff/qwen38-to-ornith-profile-transfer-2026-10-31.md),
and [profile artifacts](../../build_mtp_results/2026-10-31-ornith-recovery-3090/).

## Latest correctness and performance gates

The 17 GiB exact-8x benchmark-weighted profile measured 42.71 tok/s steady
graph decode in R23, versus 16.52 tok/s for the matched llama.cpp run. It
passes the 42 tok/s and 90% relative throughput thresholds for that exact
ZeroMoE route workload; the weighting remains benchmark-specific. This rate
does not establish that its route choices match the reference model.

R22's all-CPU/DDR expert path matched the minimal llama.cpp routed outputs
bitwise. R24's mixed VRAM/DDR hotset prefill also matched, but its forced
decode first differed at the layer-0 routed output (max absolute error
`4.395432e-05`), growing to `0.050619` at layer 44. The R24 reference kept MoE experts on CPU, so this is not a
same-placement CUDA comparison and the cause is still unknown. Mixed-hotset
correctness remains unverified; do not treat onboarding as complete. See the
handoff for the R25 diagnostic status and next steps.

R17/R19 route surveys recorded ZeroMoE's own expert IDs without comparing
them against llama.cpp on those survey sequences. Their route coverage and
hotset rankings are therefore provisional. Do not use them as reference-model
placement evidence until teacher-forced IDs and selected weights match over
the intended prompts and tokens.

