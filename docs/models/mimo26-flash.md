# MiMo V2.6 Flash RL — onboarding in progress

Branch: `feat/mimo-v26-flash`, based on merged R2 (`ac513ac`). This is an
experimental text-generation implementation. Correctness parity and a regression
witness are pending; there is no frozen performance baseline yet.

## Actual export

`/mnt/usb-ssd/@/home/ray/models/MiMo-V2.6-Flash-RL-GSQ-RCO-3bit.gguf`

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

`configs/mimo26_flash_gsq_rco_3bit.toml` uses a 2048-token context, bounded
native DDR/VRAM expert caches and two GPUs. It is a correctness profile,
not a performance recommendation. Expert reads use the model's backing file;
no separate offload store is created.

```bash
MOESTREAM_DISABLE_ROUTER_BF16=1 CUDA_VISIBLE_DEVICES=0,1 \
  scripts/run_capped.sh --limit-gb 100 -- build_mimo/moestream \
  --config configs/mimo26_flash_gsq_rco_3bit.toml --ngen 16
```

Enter a prompt through stdin/the REPL, then `:q` to exit. In this engine's
native execution path, `--prompt` alone does not submit a REPL request.
Check for competing model processes before running. The kernel cap includes
`MemorySwapMax=0`; do not replace it with a RAM-only cap.

`MOESTREAM_MIMO2_FULL_GRAPH=0` disables this family graph and fails the request;
there is no compatible layered attention fallback.
`MOESTREAM_MIMO2_PREFILL_CHUNK` sets the graph's maximum prompt chunk
(default 16, range 1–256). Larger chunks increase workspace and cold expert
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
decode path. MiMo currently calls the generic synchronous
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
the early-ready CPU lane for decode). Config: `configs/mimo26_flash_gsq_rco_3bit_ingraph_3090.toml`
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
- `MOESTREAM_MIMO2_NODE_TRACE=1|2|3|4` are debug aids; trace modes install an eval
  callback that disables the early-ready bridge, so never use them for timing.

Next: run the dual-GPU config, freeze a surveyed usage profile, A/B the GPU-assist ring.

## Measured results (2026-10-22, dual GPU: RTX 3090 + RTX 5060 Ti)

Config `configs/mimo26_flash_gsq_rco_3bit_ingraph_dual.toml` (12 GiB / 14.5 GiB expert caches =
1,317 + 1,592 hot experts), 80 GiB offload DDR tier, assist ring off, 10 CPU threads. Run via
`MOESTREAM_MIMO2_*` env (see ggml_mimo2_graph.cpp). Placement preflight: coverage 12032/12032.

| Prompt tokens | Prefill tok/s | Path | Decode tok/s (64 tok) |
|---:|---:|---|---:|
| 21 | 25.0 | CPU cold experts | 13.6 |
| 146 | 41.6 | CPU cold experts | 15.0 |
| 516 | 74.1 | GPU op-offload, 512-token chunks | 15.3 |
| 1022 | 77.1 | GPU op-offload, 512-token chunks | 14.6 |

Decode: 0 store misses / 0 loads over the run (no SSD reads); the CPU lane's compute is ~40 ms/token.
Prefill: GPU op-offload copies ~52 GiB of experts per chunk (nearly constant per chunk), so it pays
off from ~256 tokens (auto threshold, `MOESTREAM_MIMO2_PREFILL_OFFLOAD_MIN`).

Measured graph VRAM (`[MIMO2-VRAM]`): compute buffers ~0.004 GiB per GPU, KV 0.1 GiB at 512 context;
the engine's fixed 6 GiB "graph overhead reserve" only affects automatic cache sizing.

What made the difference (in order): decode CPU backend was set to 1 thread (now 10); second GPU as a
second hot set; larger caches (coverage 100%); min_prefill_batch default of 64 made every prefill chunk
after the first fall back to one-token steps (MiMo now declares 8); op-offload + pinned arena for long prefill.

Not yet done: llama.cpp parity (GPU and CPU expert kernels differ numerically), a frozen surveyed usage
profile (the profile used knows only 7,099 of 12,032 experts), assist-ring A/B, Ling regression rerun
after the final shared edits. One unexplained anomaly: a 146-token prompt once stepped at ~900 ms/token
for ~2 minutes (not reproduced; no competing process seen in the next runs).
