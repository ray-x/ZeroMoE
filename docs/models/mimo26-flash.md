# MiMo V2.6 Flash RL — onboarding in progress

Branch: `feat/mimo-v26-flash`, based on merged R2 (`ac513ac`). This is an
experimental text-generation implementation. Correctness parity and a regression
witness are pending; there is no frozen performance baseline yet.

## Actual export

`/path/to/models/MiMo-V2.6-Flash-RL-GSQ-RCO-3bit.gguf`

The header audit and `mimo2_spec_test` confirm:

- `mimo2`, 48 layers, hidden width 4096, 64 query heads, vocabulary 152576.
- Layer 0 has a dense FFN; layers 1–47 have 256 experts and route top 8.
- Full attention on layers 0, 5, 11, 17, 23, 29, 35, 41 and 47; the other
  layers use a 128-token sliding window and attention sinks.
- Full-attention layers have 4 KV heads; sliding-window layers have 8.
- Key head width 192, value head width 128, partial NEOX rotary width 64.
- Sigmoid routing with selection bias and normalized selected weights.
- Expert matrices use Q2_K, Q3_K and MXFP4. Non-expert weights occupy about
  6.06 GiB; the whole file is 107.73 GiB.
- `nextn_predict_layers=0` and
  `export_scope=text_trunk_without_mtp_or_modality_companions`. This file
  cannot supply MTP, vision or audio companion computation.

The operator ordering follows
[llama.cpp's MiMo2 implementation](https://github.com/ggml-org/llama.cpp/blob/master/src/models/mimo2.cpp),
including attention value scaling after the output projection. The embedded
chat template, rather than the generic Qwen wrapper, defines text prompting:
no separator newline between messages and exactly `<think></think>` for a
direct answer.

## Initial recipe

`configs/mimo26-flash.toml` uses a 2048-token context, bounded
native DDR/VRAM expert caches and two GPUs. It is a correctness profile,
not a performance recommendation. Expert reads use the model's backing file;
no separate offload store is created.

```bash
ZEROMOE_DISABLE_ROUTER_BF16=1 CUDA_VISIBLE_DEVICES=0,1 \
  scripts/run_capped.sh --limit-gb 100 -- build_mimo/zeromoe \
  --config configs/mimo26-flash.toml --predict 16
```

Enter a prompt through stdin/the REPL, then `:q` to exit. In this engine's
native execution path, `--prompt` alone does not submit a REPL request.
Check for competing model processes before running. The kernel cap includes
`MemorySwapMax=0`; do not replace it with a RAM-only cap.

The registered MiMo2 graph is the CUDA path and cannot be disabled into a
request-time fallback. There is no compatible layered attention fallback.
`graph_prefill_chunk` sets the graph's maximum prompt chunk (default 16,
range 1–256); use `--graph-prefill-chunk` for a per-run override. Larger chunks increase workspace and cold expert
fan-out; changing this is a tuning experiment after correctness validation.
The BF16-router override above preserves original F32 routing for the initial
comparison, rather than making a claim about the faster setting's accuracy.

## GLM 5.3 Flash comparison and MiMo measurements

**Acceptance target: at least 10 tok/s steady-state decode on this machine.**
MiMo belongs to the DDR+VRAM-resident performance class alongside Qwen 3.8
Flash, GLM 5.3 Flash and Ling 3.1 Flash. A result below 10 tok/s is an
optimization failure, not an acceptable model-size limit.

The GLM no-offload profile is a useful cache-policy reference. Its
AJ-IQ2_XXS export is 87.3 GB with 74.21 GiB of routed experts: 42 MoE layers
x 288 experts. MiMo's export is 107.73 GiB with 101.672 GiB of expert
matrices: 47 x 256. The expert counts are almost equal; MiMo's expert payload
is about 37% larger, but still fits the intended DDR+VRAM operating envelope.
Ling's validated short-decode profile is the execution-strategy reference:
its no-copy offload graph, early CPU lane and GPU assist reach 10.15 tok/s.

GLM's validated recipe in `docs/GLM5.3-Flash-baseline.md` loads about 80 GB
of experts into DDR (`host_ram_gb=106`, `ram_preload_all_experts=true`),
keeps eight complete MoE layers in 15.8 GB of VRAM, and runs the remaining
layers on CPU. Its prefill and decode histories were collected separately
from five varied prompts, then frozen for repeatable timing. Decode had no
demand SSD misses and used no secondary offload store.

MiMo started with a much weaker profile: 80 GB host budget,
`ram_preload_all_experts=false`, and no established route survey. The
first cold run measured 0.71 tok/s over four output tokens. I then tested
the GLM-style DDR preload with a 100 GB host budget, 20 GB of expert VRAM,
1,600 pinned VRAM entries, and 9,776 DDR entries, under a 110 GiB cap with
swap disabled. It took about 6m35s to preload from the USB-backed GGUF.

The first DDR-preload run averaged **0.41 tok/s** over 59 generated tokens.
The globally ranked/disjoint-tier run averaged **0.69 tok/s** over 64 tokens
(0.61 tok/s median for steps 20 onward); prefill was 0.25 tok/s. That run
still fetched 2,986 expert blocks from the GGUF, about 46.7 blocks per output
token. At roughly 8.9 MiB per expert, this is around 400 MiB of cold reads
per token. It also ended with 834 experts duplicated across RAM and VRAM
(7.54 GB), despite the disjoint startup ranking. The change improved speed,
but the result remains far below the 10 tok/s target.

The benchmark history contains one water-cycle prompt, not a representative
survey. The profile is still experimental; these numbers are diagnostic, not
a frozen baseline. They show that DDR preload alone does not solve MiMo's
decode path. *(Superseded 2026-10-22: MiMo now runs its routed experts in the graph with the early-ready lane, see
"Status" and the measured results; the text below is the history of the first cache-backed bring-up.)* MiMo then called the generic synchronous
`moe_forward_gguf` path from 47 graph callbacks. Ling's 10 tok/s path instead
splits resident GPU work from cold CPU/offload work, runs them concurrently,
and uses small GPU-assist rings. Reusing that strategy is the primary path to
the target. The remaining work is to eliminate decode-time SSD reads, stop
prefill from recreating RAM/VRAM overlap, and port the Ling early-ready lane
and GPU-assist strategy to MiMo's attention graph.

An intermediate grouped-decode experiment now allows MiMo to use the shared
GGML routed runner and retains all ten Q2_K/Q3_K/MXFP4 projection signatures
instead of rebuilding a default runner between layers. It is opt-in through
`moe_ggml_mul_mat_id=true`; performance validation with populated DDR is
pending. This can reduce per-expert graph submissions, but it still stages
cold expert bytes serially and therefore cannot replace the early-ready lane.
Collect separate varied prefill and decode histories before tuning either
policy. The llama.cpp prompt/logit parity run has not passed, and an existing
model regression witness remains pending.

Geometry tests and the R2 lifecycle test pass. The release build passes.
The corrected prompt wrapper produces the requested answer in a short smoke,
but that is not a correctness comparison. Do not extrapolate this
implementation to the advertised million-token context: sliding-window
layers currently allocate full-capacity device KV, with a VRAM admission
check. A ring cache remains future work. MTP is absent from this GGUF and is
not enabled.

## What R2 made reusable, and what remains

> **Superseded in part (2026-10-23).** The paragraphs below describe the first, cache-backed bring-up. MiMo's
> default path is now the in-graph hot/cold MoE on the shared `build_moe_block` (`models/graph/moe_ingraph.hpp`,
> the aliased-placement and offload providers), not `CacheBackedMoeRuntime`; the synchronous callback remains as the
> fallback when `ZEROMOE_MIMO2_INGRAPH_MOE=0`. R2's `CacheBackedStrategy` is still an unwired specification.

MiMo reuses `FullGraphModel` registration/lifecycle, `GraphCore`, the engine's
resident dense weights, and the live `moe_forward_gguf` pipeline. Routing,
cache residency, prefetch, promotion, expert quantization kernels and dispatch
were not copied into the family graph. New family code defines metadata
validation, attention math, KV layout and the dense/MoE block sequence.

This onboarding needed three shared lifecycle additions: a native-cache
capability, runtime binding, and a graph-only/fail-closed capability. It also
needed a generic completion flag because the existing flags were named for
individual families. These are evidence of remaining framework work, rather
than a claim that model onboarding is already declarative.

In particular, R2's `CacheBackedStrategy` is still a fail-closed specification
whose residency/compute methods contain wiring TODOs. MiMo calls the working
native pipeline through `CacheBackedMoeRuntime`; it does not complete that R2
extraction. Keep that distinction when judging how much R2 reduced the work.

Recommended next framework work:

1. Wire one cache-backed strategy to the real native claim/hold/compute/release
   lifecycle and migrate two existing families before generalizing further.
2. Share graph state/weight binding and attention building blocks after a
   second family demonstrates matching requirements. Keep rotary convention,
   head dimensions, sinks, masking and value scaling explicit in model specs.
3. Represent dense placement and expert execution/cache policy as separate
   capabilities. New popular models should select policies independently of
   their attention family.
4. Consolidate graph completion, logits and failure fields into one request
   result; MiMo still uses the established legacy-named shared logits fields.
5. Add speculation through a separate draft/verify state contract when real
   MTP weights and acceptance measurements are available. Ordinary decode
   remains the correctness reference.

## In-graph hot/cold MoE status (2026-10-22)

MiMo's routed MoE runs through the shared `build_moe_block` (hot experts alias
the VRAM expert cache, cold experts come from the shared offload DDR tier, with
the early-ready CPU lane for decode). Config: `configs/testing/mimo26_flash_gsq_rco_3bit_ingraph_3090.toml`
(one GPU) and `..._ingraph_dual.toml` (3090 + 5060 Ti second hot set, untested).

Measured (3090 only, 15 GiB cache = 1,647 hot experts, 80 GiB offload DDR, assist off,
64-token DNS prompt, coherent output): **2.87 tok/s decode** (`REQUEST-DECODE-OUTPUT`).
This is below the 10 tok/s target. About 870 of 12,032 experts are in no tier at that
budget, so decode still reads them from the SSD.

Pitfalls found while bringing it up:
- Batched MUL_MAT_ID must not repeat a placeholder expert id within a token (outputs
  are left unwritten -> NaN). Use the patched negative-id skip (`AutoBatched`) when the
  layer's quant takes the MMQ path (`layer_experts_take_mmq`).
- The offload store's row-size table needed MXFP4 (ggml type 39).
- Aliased cache slots need row-aligned strides (engine pads them when MiMo's in-graph
  MoE is enabled); the engine RAM tier must not preload on top of the offload tier.
- `--graph-node-trace 1|2|3|4` is a debug aid; trace modes install an eval
  callback that disables the early-ready bridge, so never use them for timing.

Next: run the dual-GPU config, freeze a surveyed usage profile, A/B the GPU-assist ring.

## Measured results (2026-10-22, dual GPU: RTX 3090 + RTX 5060 Ti)

Config `configs/bench/mimo26_flash_gsq_rco_3bit_ingraph_dual.toml` (12 GiB / 14.5 GiB expert caches =
1,317 + 1,592 hot experts), 80 GiB offload DDR tier, assist ring off, 10 CPU threads. Run via
registered `graph_*` TOML settings or generated `--graph-*` options. Placement
preflight: coverage 12032/12032.

| Prompt tokens | Prefill tok/s | Path | Decode tok/s (64 tok) |
|---:|---:|---|---:|
| 21 | 25.0 | CPU cold experts | 13.6 |
| 146 | 41.6 | CPU cold experts | 15.0 |
| 516 | 74.1 | GPU op-offload, 512-token chunks | 15.3 |
| 1022 | 77.1 | GPU op-offload, 512-token chunks | 14.6 |

Decode: 0 store misses / 0 loads over the run (no SSD reads); the CPU lane's compute is ~40 ms/token.
Prefill: GPU op-offload copies ~52 GiB of experts per chunk (nearly constant per chunk), so it pays
off from ~256 tokens (auto threshold, `graph_prefill_offload_min_tokens`).

Measured graph VRAM (`[MIMO2-VRAM]`): compute buffers ~0.004 GiB per GPU, KV 0.1 GiB at 512 context;
the engine's fixed 6 GiB "graph overhead reserve" only affects automatic cache sizing.

What made the difference (in order): decode CPU backend was set to 1 thread (now 10); second GPU as a
second hot set; larger caches (coverage 100%); min_prefill_batch default of 64 made every prefill chunk
after the first fall back to one-token steps (MiMo now declares 8); op-offload + pinned arena for long prefill.

Not yet done: llama.cpp parity (GPU and CPU expert kernels differ numerically), a frozen surveyed usage
profile (the profile used knows only 7,099 of 12,032 experts), assist-ring A/B, Ling regression rerun
after the final shared edits. One unexplained anomaly: a 146-token prompt once stepped at ~900 ms/token
for ~2 minutes (not reproduced; no competing process seen in the next runs).

## 2026-10-23: 3090 cache 12 -> 15 GiB

The new idle-VRAM preflight check found 5.4 GiB free on the 3090 after the graph was built (with the unused
active-layer pool dropped). Raising `moe_cache_gb` for GPU 0 to 15 GiB in
`configs/bench/mimo26_flash_gsq_rco_3bit_ingraph_dual.toml` leaves 2.6 GiB free (5060 Ti unchanged at 1.7 GiB) and
gave decode 13.86 / 15.65 / 15.84 / 15.33 tok/s (was 13.43 / 15.05 / 15.43 / 14.73) and prefill
25.8 / 43.8 / 76.8 / 80.2 tok/s (was 24.9 / 41.4 / 74.7 / 77.8). The generated tokens differ from the
12 GiB run from token 13 of the first prompt (a different hot set moves experts between the GPU and CPU
kernels); the text is coherent in both.

## Open (2026-10-23)

- llama.cpp parity passed on one short prompt (above); a long-prompt run (past the sliding window) has not been done.
- The usage file knows 7,099 of 12,032 routed experts; a surveyed profile only changes the VRAM-hot set,
  because the DDR tier covers every expert (0 SSD reads in decode).
- One 146-token prompt once stepped at ~900 ms/token and one S=12 tail chunk took 3.3 s during bring-up; neither
  reproduced in more than a dozen later runs.

### Parity against llama.cpp, 2026-10-23 (passed)

Prompt `The capital of France is` (ids 785 6722 315 9625 374), 8 greedy steps, `--raw-prompt --temp 0`. ZeroMoE:
dual-GPU in-graph config (3090 cache 12 GiB), an earlier local build tree; llama.cpp `b9739`-class unpatched CPU build
(`build_mimo/mimo26-llama-logits`, `tools/compare_logits.py`; recipe in README, "Validating a new model").

| step | max_abs | rmse | cosine | top-1 (ZeroMoE / llama.cpp) |
|---:|---:|---:|---:|---|
| 0 (prompt's last token) | 0.461 | 0.0799 | 0.99987 | 12095 / 12095 |
| 1 | 0.697 | 0.1066 | 0.99950 | 13 / 13 |
| 2 | 0.942 | 0.1215 | 0.99967 | 1084 / 1084 |
| 3 | 0.739 | 0.1364 | 0.99964 | 374 / 374 |
| 4 | 0.492 | 0.0887 | 0.99987 | 7407 / 7407 |
| 5 | 0.711 | 0.1378 | 0.99968 | 304 / 304 |
| 6 | 0.851 | 0.1362 | 0.99974 | 279 / 279 |
| 7 | 0.467 | 0.0820 | 0.99985 | 10200 / 10200 |

All eight tokens are identical and the cosine never drops below 0.9995, so attention, routing, the shared expert and the
Q2_K/Q3_K/MXFP4 expert paths agree with llama.cpp up to kernel rounding. One short prompt: a longer one and a second
prompt would test sliding-window layers beyond the window (4096 on this model is far above this prompt).

## Surveying the usage profile (2026-10-23)

The in-graph path bypasses the engine's usage statistics (`freeze_usage_stats = false` records nothing there), so the
profile is recorded from the graph: `bench/mimo_survey.sh survey <profile> <binary>` runs 20 varied prompts
(`bench/mimo_survey_prompts.txt`) with `--graph-usage-record-file`, a side tap that leaves the real route ids
alone and counts decode and prefill separately (`<profile>` is decode, `<profile>.prefill` is prefill; a 512-token
prefill chunk has far more route slots than a decode step and would dominate a combined file).
`bench/mimo_survey.sh eval <profile>` runs four held-out prompts (`bench/mimo_eval_prompts.txt`) for an A/B.
The tap adds a CPU node per layer: use it for surveys, not timing. Coverage and the A/B result are below.

### Surveyed profile A/B (2026-10-23, held-out prompts, dual GPU, 3090 cache 15 GiB)

Profiles in `bench/profiles/`: `mimo26_decode.usage` (decode routes), `mimo26_prefill.usage`, `mimo26_combined.usage`
(sum). The earlier profile covered 7,099 of 12,032 experts; the survey saw 11,385 (decode) and 9,852 (prefill).

| Profile | Decode, 4 prompts (tok/s) | Short-prompt prefill, 4 prompts (tok/s) |
|---|---|---|
| earlier (`.zeromoe_usage_mimo26_gsq_rco3`) | 15.35 / 13.82 / 15.39 / 14.57 and 15.29 / 14.23 / 15.86 / 14.69 | 28.8 / 32.7 / 36.9 / 35.5 and 28.5 / 33.3 / 37.9 / 35.9 |
| decode-only | 16.72 / 14.76 / 16.66 / 16.23 and 16.67 / 14.62 / 16.69 / 16.17 | 23.5 / 28.2 / 27.1 / 30.0 and 23.4 / 27.6 / 27.1 / 29.9 |
| combined | 15.31 / 11.58 / 16.53 / 16.20 and 16.99 / 14.73 / 16.37 / 16.23 | 25.2 / 32.0 / 33.1 / 35.7 and 27.5 / 33.6 / 33.7 / 36.0 |

Why decode-only slows short prefill: with the same configuration, the same 3,239 hot experts and 0 store misses, a ~25-token
prefill runs its cold experts on the CPU, and replaying the surveyed routes against each hot set gives decode-route
coverage 52 % (earlier) / 67 % (decode-only) / 65 % (combined) and prefill-route coverage 67 % / 55 % / 68 %. The earlier
profile came from prefill-heavy runs, so it favoured prefill. The combined profile is adopted (decode about +8 %, short
prefill level with the earlier one).

**Long prompts do not care which experts are hot.** 600- and 1,000-token prompts (`bench/mimo_survey.sh long`), prefill
tok/s decode-only / prefill-only / combined: 54.1 / 53.3 / 54.5 and 63.7 / 62.5 / 63.3. A 512-token chunk touches
nearly every expert in every layer, so the copy volume (about 60 GiB per chunk, 21-22k expert copies, 7.2-7.4 s) depends
on how many experts are cold, not which. A prefill-specific hot set, or swapping hot sets between prefill and decode,
would not speed up long prefill; what does is fewer cold bytes per token, i.e. a larger chunk (the copy volume per chunk
is the same, so tokens per chunk is the lever) or more VRAM.

### GPU utilisation and where VRAM goes

A test run is ~90 % loading (1 min 46 s VRAM pin preload + 3 min 47 s DDR seed from the USB SSD at ~0.35 GB/s, GPUs idle,
iowait 35 %) and ~40 s of compute. In decode the GPUs wait on the CPU cold-expert lane (62 ms/token: ~36 ms CPU lane,
~16 ms GPU wait). VRAM per GPU: 3090 dense 6.5 GB + cache 15 GB + 0.3 GB context + 0.4 GB KV, 2.5 GiB free at decode but
only 1.25 GiB at a 512-token prefill (compute and copy buffers 1.19 GiB); 5060 Ti cache 14.5 GB, 1.6 GiB free. Experts
hold about 70 % of VRAM. Decode-route coverage vs experts in VRAM (surveyed): 3,239 -> 66.6 %, 3,519 -> 69.5 %, 4,000 -> 73.9 %,
5,000 -> 81.7 %, 6,500 -> 90.2 % (estimated decode 16 / 17 / 18.5 / 22 / 27 tok/s on a 26 ms fixed + 36 ms cold model).
