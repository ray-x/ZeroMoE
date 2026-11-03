# Ornith 1.5 122B

## Support status

Ornith is a `qwen35moe` GGUF and uses the Qwen35 MoE full-graph adapter. It is
**supported but untuned for prefill**: the full-graph path works (prefill and
decode pass with zero layered forwards or legacy expert dispatches), but it does
not implement the Qwen3.8 prefill workflow (whole-layer VRAM expert block plus
op-offload staging of the batch's cold experts). Prompt throughput therefore
stays well below llama.cpp. It is not the Qwen3.6 35B model. Model-specific
block math lives in the Qwen35 graph builder and uses the shared graph runtime,
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

## Measured on the RTX 3090 (2026-11-03)

Workload: 1088-token prompt, 100 generated tokens, context 4096, 10 threads,
batch/ubatch 1024. RTX 3090 only. The prefill bar for this port is **> 100
tok/s**; the numbers below met it.

| Metric | Value |
|---|---:|
| Prefill | **103.4 tok/s** (target: > 100) |
| Decode (steady) | 23.5-30.9 tok/s, peak ~30 |
| Prefill VRAM | ~22 GiB of 25 GiB |
| Prefill hotset | 13.58 GiB (4,352 experts, 71.3% of prefill traffic) for the 1024-token batch, 16.16 GiB (77.6%) for the tail chunk |
| Decode hotset | 17.00 GiB (5,516 experts, 77.8% of decode traffic) |

Config: `prefill_batch_tokens = 1024`, `qwen4exp_graph_expert_vram_budget_gb = 17.0`,
`graph_prefill_op_offload = true`, `graph_early_ready_all_split = false`,
`ram_preload_all_experts = true`, `ram_pinned_budget_gb = 45.0`, the merged Ornith
usage histograms, and `max_new_tokens <= 100` for the decode leg.

What makes this path work, and what it costs:

- The prefill hotset is ranked from `prefill_pin_file` while decode is ranked
  from `pin_file` minus the prefill counts (decode-only). One VRAM window serves
  both phases: prefill caps the resident set to what its batch workspace leaves
  (`estimate_prefill_workspace_bytes` now takes the real op-offload flag:
  2.7 MiB/token instead of 8), and `decode_prepare` releases the prefill-sized
  arena when it is smaller than the configured budget so decode rebuilds its own.
- The hot/cold split serves batched prefill too: a foreign route carries a
  negative expert id, which the patched CPU op and the batched CUDA MMQ path skip
  for free. `patches/ggml/0018-ggml-sched-strided-weight-copy.patch` copies a
  strided host expert plane (pinned DDR arena, padded per-expert stride) into a
  dense device tensor; without it, op-offload staging produced all-NaN prefill.
- Usage histograms must be produced by the graph path: run with
  `qwen4exp_track_router_usage = true`, `freeze_usage_stats = false`, and
  `qwen4exp_graph_expert_vram_budget_gb = 0` (an empty histogram cannot satisfy
  the budget gate at startup). The tap now records prefill batches as well as
  decode tokens, so `prefill_pin_file` is a real prefill survey instead of a copy
  of the combined file.

## Gap to Qwen3.8 Flash

| | Ornith 122B (this port) | Qwen3.8 Flash |
|---|---|---|
| Expert placement | per-expert usage-ranked hot/cold split (48 GPU<->CPU transitions per batch) | whole-layer VRAM block plus op-offload staging |
| Expert bytes vs VRAM | 38.8 GiB of experts, ~14-16 GiB resident (~40%) | smaller expert working set, most of it resident |
| Prefill expert compute | hot experts on GPU, cold experts on the CPU path (serialized) | nearly all expert FLOPs on CUDA via staged copies |
| Prefill workspace | 2.7 MiB/token with op-offload (4.0 GiB at 1024) | same op-offload shape |
| Dense prefill fast paths | `dense_bf16_prefill_gemm` / `dense_bf16_kquant` default off, not validated on this quant mix | enabled and measured |
| Decode | 23.5-30.9 tok/s | 22-30 tok/s (Q2_0 hotset) |

The remaining prefill gap is structural, not a tuning knob: Ornith gives up
whole-layer residency (the whole-layer default is 22 layers, which starves the
batch workspace down to 1.9 GiB and forces 48-64 token chunks), so every layer
alternates between GPU and CPU and the CPU cold-expert branch cannot overlap the
GPU hot branch. llama.cpp's matched row (`-ngl 22 -ncmoe 26`, ub 2048) is
~183 tok/s at pp512; Ornith is at 0.56x of that, while decode is ahead of the
llama.cpp decode row (15.7 tok/s).

Closing it needs the Qwen3.8 workflow end to end: a whole-layer VRAM block sized
for the batch workspace, op-offload staging for the remaining layers, and the
dense BF16 prefill paths verified for this quantization mix.

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
29.84 tok/s. The 42 tok/s target
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

