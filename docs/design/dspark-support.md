# DSpark (DFlash) speculative-decoding draft for DeepSeek-V4.1 Flash: design and plan

Status (2026-11-08): WP1 config parser and tensor-topology validator implemented and tested on the real file (`models/deepseek/dspark_config.hpp`, test `ZEROMOE_TEST_ONLY_DSPARK_CONFIG=1 build_dsv41_retry16/zeromoe_tests`); WP2 decode taps implemented (below); WP3-7 not started. Decision recorded: the draft runs on the GPU and the target's expert pins are shrunk by ~590 slots (~9 GB) to make room. File inspected:
`~/vllm-workspace/DeepSeek-V41-Flash/dspark-DeepSeek-V4.1-Flash-1017-BF16.gguf` (8,506,281,280 bytes, GGUF v3, 78 tensors, 57 KV).
Reference implementation: `~/Projects/llama.cpp` at the pinned commit `8baabd53`, `src/models/dflash.cpp`
(`llama_model_dflash::graph_dsv4`, `build_dspark_markov_head`), arch id `LLM_ARCH_DFLASH` ("dflash").

## What the file is

A **draft model** for speculative decoding of DeepSeek-V4.1 Flash, not a standalone LLM.
`general.architecture = dflash`; in llama.cpp's words "DSpark = DFlash + a semi-autoregressive Markov head and Confidence head",
with a DeepSeek-V4 backbone.

| Item | Value |
|---|---|
| Stages | 3 full DSV4 blocks (`dflash.block_count = 3`), no compressed attention (`compress_ratios = [0,0,0]`), sliding window 128 |
| Width | `embedding_length 5120`, 64 heads, 1 KV head, key/value length 512 (rope 64, base 10000), q_lora 1280, output groups 8 / lora rank 1024 |
| Hyper-connections | `count 4`, sinkhorn 20, eps 1e-6, `single_pass = True` (tensors `hc_attn_*`, `hc_ffn_*` per block) |
| MoE | 128 experts, top-3, 1 shared expert, expert FFN 2304, gating func 4 (sqrt-softplus), weights scale 1.5, normalized, swiglu clamp 10 |
| Draft block | `dflash.block_size = 5`, `sample_from_anchor = True` (slot 0 is predicted too), `has_confidence_head = True` |
| Target taps | `dflash.target_layers = [37, 38, 39]`: hidden states of those target layers feed the draft |
| Extra tensors | `fc.weight (15360 -> 5120)`, `enc.output_norm.weight`, `markov_w1/w2.weight (256 x 129280)`, `conf_proj.weight (5376 x 1)`, `output_norm.weight` |
| Not in the file | token embedding and LM head (taken from the **target** model); **`hc_head_fn/scale/base`**: llama.cpp's `graph_dsv4` calls `build_hc_head(inpL, model.hc_head_fn, model.hc_head_scale, model.hc_head_base)` and `build_hc_head` has no null handling, yet this 78-tensor file has none of them. Open question: the final 4-stream collapse probably reuses the **target's** `hc_head_*` tensors (the draft also shares the target's embedding and LM head); confirm against the target GGUF's `output_hc_*`/`hc_head_*` names before WP4 |
| Types | 9 expert tensors MXFP4 (type 39, ~7.2 GB), 27 BF16 matrices (type 30), 42 F32 vectors |

Per-block size: experts 3 x 802 MB, BF16 attention + shared expert ~140 MB. Total draft VRAM if fully resident: **~8.5 GB**.

## Algorithm (from the reference graph)

1. **Encoder / KV injection (per accepted position).** Take the target's hidden states at layers 37, 38, 39, concatenate
   (3 x 5120 = 15360), `fc` to 5120, RMS norm (`enc.output_norm`). For each draft stage: `wkv` (attn_kv) then kv norm, rope on the
   trailing 64 dims, and write the K (K == V) into that stage's sliding-window ring cache (128 positions). This gives the
   draft cheap, target-conditioned context for every committed token.
2. **Draft block.** Input is a block of `block_size = 5` tokens (the last committed token as anchor plus mask/noise slots). Embed with
   the **target's** `tok_embd`, replicate x4 for hyper-connections, run the 3 stages (HC pre, attention over the ring plus the block,
   HC post, HC pre, MoE top-3 + shared expert, HC post), collapse (`hc_head`), `output_norm`, **target LM head** -> logits.
3. **Markov head.** For block positions i = 0..4, bias the base logits with `markov_w2 . markov_w1[prev_token]` where `prev` is the argmax of
   the previous biased column (greedy chain; rank 256). This is the "semi-autoregressive" part.
4. **Confidence head.** `sigmoid(conf_proj . [pre-norm hidden(i); markov_w1[prev(i)]] + b)` per position: predicted acceptance probability,
   used to decide how many drafted tokens to verify.
5. **Verify** with the target in one batched forward over the drafted tokens; accept the longest matching prefix (greedy) plus the target's
   own next token; feed the newly committed positions' target features back to step 1.

## What ZeroMoE has and lacks

Has: the DeepSeek-V4.1 graph blocks (HC pre/post/head, MLA attention, sqrt-softplus router, cache-backed MoE), MXFP4 GPU matvec
(`launch_mxfp4_matmul`), BF16 matched-cuBLAS path, `ZEROMOE_TEACHER_FORCE_TOKEN_FILE`, a Qwen4Exp MTP pipeline for reference
(`docs/handoff/qwen4exp-mtp-current.md`), GGUF split loader.
Lacks: the `dflash` architecture adapter, target-hidden-state taps at layers 37-39, the draft graph (non-causal block attention over a
128-position ring + the block), the Markov/confidence heads, and above all **multi-token verification with state rollback** in the
DeepSeek target path.

## Work packages (in dependency order)

| WP | Content | Size | Main risk |
|---|---|---|---|
| 0 | Reference capture: run the pinned llama.cpp with a `deepseek4`-capable target? Not possible: pinned llama.cpp loads `deepseek4`, our main GGUF is `deepseek41` (see handoff). So the draft cannot be validated end to end against llama.cpp; validate stage-by-stage against the CPU ZeroMoE reference and against `dflash.cpp` by reading. | small | no external oracle |
| 1 | `dflash` adapter: register arch, parse KV, validate topology and tensor shapes/types, load tensors (MXFP4 experts, BF16 matrices). Unit test on the real file (shapes, types, `hc_head_*` question). | small-medium | tensor naming (`enc.*`, `fc`, `markov_*`, `conf_proj`) |
| 2 | Target taps (definition confirmed from llama.cpp, `deepseek4.cpp` ~1238 and `common/speculative.cpp` ~1133/1230): for each id in `target_layers` (37, 38, 39) take the residual stream **entering** that target layer (`llama_set_embeddings_layer_inp`), reduced by `build_hc_mean`: the arithmetic mean over the 4 hyper-connection streams, shape `[n_embd]` per token (`x[:, 0..hc-1]` summed then scaled by 1/hc). Concatenate the three means per token (3 x n_embd = 15360) in `target_layers` order as the input of `fc`. In ZeroMoE this is the `res` tensor at the top of `build_layer(res, l)` in `ggml_deepseek_v41_graph_decode.cpp` (decode) and the equivalent in the prefill graph (`ggml_deepseek_v41_graph.cpp`); export for every position (prefill: all tokens; decode/verify: per step). | medium | graph output plumbing in prefill + decode nodes; must not change target numerics; adds 3 small GPU reductions per step |
| 3 | Draft encoder + ring KV: `fc` (BF16 15360 -> 5120) + `enc.output_norm` RMS norm + per-stage `attn_kv` (5120 -> 512) + `attn_kv_a_norm` + rope on the trailing 64 dims + write K (K == V) into the stage's 128-position ring, device resident. Rope is **plain, interleaved pairs** (llama.cpp `LLAMA_ROPE_TYPE_NORM` for the DSV4 draft; `ggml_rope_ext` with ext_factor 0, freq_scale 1, base `dflash.rope.freq_base` = 10000), applied at the committed token's position; the existing `rope_yarn_inplace` with ext_factor 0 is the matching host routine. | medium | rope offsets / ring indexing; no external oracle, so test against an independent float64 reimplementation and the structural invariants |
| 4 | Draft block forward: 3 DSV4 stages on the existing blocks; MoE with 128 x 3 MXFP4 experts; attention over ring plus the 5 block tokens (check the block mask in `dflash.cpp`/`build_attention`). | large | expert residency (below), HC `single_pass` semantics |
| 5 | Markov + confidence heads, target LM head reuse; unit tests of the chain vs a float64 reference. | small-medium | vocab 129280 x rank 256 matmuls |
| 6 | Speculative loop + **batched verify** in the target: S = 1 + drafted tokens at pos_base > 0 through the full-graph path (cache-backed MoE for S > 1, attention for S > 1 with existing KV), then commit/rollback of the DeepSeek state: SWA window slots, compressed-attention pending accumulators and indexer keys (host `Dsv41CompressedLayerState`), GPU mirrors (`dsv41_head_attention_invalidate` granularity), Engram row cache. | very large | state rollback correctness; verify cost with S=5 (up to 30 distinct experts per layer vs 6 now) |
| 7 | Acceptance policy using the confidence head, metrics (acceptance length, tokens/step), A/B vs plain decode with identical tokens; bit-exact check that spec-decode output equals plain greedy output. | medium | greedy-equality must hold exactly |

## Open decisions (need the user)

1. **Where do the draft's 3 x 128 experts live?** They are ~7.2 GB (MXFP4) and the 3090 + 5060 Ti pools are fully used by target pins
   (3,566 resident slots + dynamic reserve). Options: (a) shrink target pins by ~590 slots (~9 GB) and keep the draft fully on the GPU
   (costs target decode hit rate); (b) keep the draft experts in DDR and run its MoE on the CPU (3 layers x top-3 x MXFP4; needs an MXFP4 CPU
   kernel; DDR is also full: 101 GB used of 118.6 GB cap); (c) put the draft on the 5060 Ti only and give up that many target pins there.
2. **Scope and success bar.** Expected gain is workload dependent: decode is 76 ms/token now; a spec step costs draft (~3 DSV4 layers + heads) plus a
   batched verify that touches several times more experts than a single token. Needs a measured acceptance length before it can promise a speed-up.
   Suggested bar: bit-identical greedy output to plain decode, and a measured end-to-end tokens/s gain on the 959+64 workload.
3. **Validation reference.** llama.cpp cannot load our target (`deepseek41` vs `deepseek4`), so the greedy-equality check against plain ZeroMoE decode
   is the primary correctness oracle.


## Implementation log

- **WP1 + WP2 (decode side), 2026-11-08.** New config key `dspark_model_path` (EngineConfig). At load (`core/engine_runner_model.cpp`) the draft header is read, `parse_config` and `validate_tensors` run, and the draft width and `target_layers` are checked against the target GGUF metadata (`deepseek41.embedding_length`, `deepseek41.block_count`; `architecture_config` is not populated yet at that point). The decode graph (`ggml_deepseek_v41_graph_decode.cpp`, `DsvHcPlan::build_layer`) adds, for each target layer in `target_layers`, a GPU node that averages the HC streams of the residual entering that layer (`sum / HC`, same as llama.cpp `build_hc_mean`), and after each decoded token copies the three vectors into `StreamingEngine::dspark_features` (`3 x n_embd`, in `target_layers` order) with `dspark_features_pos`. First-token log: `[DSPARK] feature tap layer=37 pos=959 l2=164.5 finite=1` (layers 38: 179.2, 39: 191.1). Run `build_mtp_results/2026-11-08-dspark-taps-v107/` (source bf7976b, 3 requests): decode 13.06 / 13.26 / 13.17 tok/s and prefill 92.5 / 94.5 / 94.4 tok/s, i.e. no measurable cost versus v104 (13.05 / 13.22 / 13.19; 91.6 / 94.2 / 94.3); graph checker PASS. Not yet done: the prefill-side taps (all prompt positions), and any consumer of the features.
- Note: this config parser does not accept TOML single-quoted literal strings; use double quotes for paths.

## Second draft file: Lucebox-DSpark-draft-Q2K-Q4K (2026-11-08)

`~/vllm-workspace/DeepSeek-V41-Flash/Lucebox-DSpark-draft-Q2K-Q4K.dflash.sp.gguf`, 4,928,288,512 bytes (vs 8,506,281,280 for the BF16 draft: 58%). Same `dflash` architecture, same hyper-parameters (3 stages, width 5120, 128 experts top-3 + shared, block_size 5, target layers 37/38/39, Markov rank 256, vocab 129,280, 78 tensors), validated by the same loader (`ZEROMOE_TEST_ONLY_DSPARK_CONFIG=1 ZEROMOE_TEST_DSPARK_GGUF=<path>`).

| | BF16 draft | Lucebox draft |
|---|---|---|
| Experts (gate/up/down x 3 stages) | MXFP4 (type 39), ~7.2 GB | **Q2_K (type 10)**, ~4.5 GB |
| Dense matrices (attention, shared expert, `fc`) | BF16 (type 30) | **Q4_K (type 12)** |
| HC `fn`, `markov_w1/w2`, `conf_proj` | F32 / BF16 | **F16 (type 1)** |
| Router, norms, sinks, biases | F32 | F32 |
| Extra keys | none | `dflash.rope.scaling.type=yarn`, factor 16, original context 65,536, beta_fast 32, beta_slow 1; `dflash.dsv41_semantics=true`; indexer keys (head_count 32, key_length 128, top_k 512; unused, `compress_ratios=[0,0,0]`); `hash_layer_count 0`; `compress_rope_freq_base 160000` |
| Missing keys | - | `sample_from_anchor` (reference default: true), `has_confidence_head` (present iff `conf_proj.weight` exists) |

Consequences:
- **Memory.** About 4.9 GB instead of 8.5 GB: roughly 400 target expert pins to give up instead of roughly 690 (one expert slot is ~12.3 MB), so a smaller hit on the target's decode.
- **Kernels.** Q2_K and Q4_K dense/expert matvecs already exist in ZeroMoE (`kquant_matmul.cu`, Q5_K lane8 pattern, `launch_q2_K_*`, `launch_q4_K_matmul`); the BF16 draft would additionally need MXFP4 expert matvec and BF16 attention matvecs. The Lucebox draft is the cheaper one to run.
- **Rope.** This draft declares YaRN scaling (factor 16, original context 65,536), unlike the BF16 draft's plain rope; the encoder's K rope (WP3) must use `rope_yarn_inplace` with the YaRN parameters and the DSV4 `attn_factor`/mscale convention of the target, not ext_factor 0. Confirm against llama.cpp `dflash.cpp` before WP3.
- **Quality.** A Q2_K-expert / Q4_K-dense draft will be less accurate than the BF16 one; the acceptance study decides whether the smaller size costs more in acceptance length than it gains in speed.
- **Performance so far.** There is no speculative loop yet, so neither draft has a speed number. The only measurable effect today is the load check and the target feature taps, which cost nothing (v107).
