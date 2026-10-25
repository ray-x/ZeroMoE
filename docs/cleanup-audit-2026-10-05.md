# Cleanup audit — 2026-10-05

Goal: ggml (+ exllamav3) become the only tensor libraries. Branch:
`cleanup/ggml-exl3-only` (base: `bd8eb96`). Nothing is deleted; moved files go
to `obsolete/`.

Method: callers found by symbol name (`grep -w`) across `src/ include/ models/
tests/ tools/`. Name-matching can miss indirect use (function pointers, macros),
so every "move" is confirmed by a build; every "dead" claim is only as good as
that build.

## Ground rule found during the audit

ggml full graphs exist for Qwen4Exp (decode + prefill), GLM-5.3-Flash
(`glm5next`, opt-in) and DeepSeek-V4 (prefill only, opt-in). Full GLM-5.2/5.3
(79 layers) and DeepSeek-V4 decode still run on the native stack
(`attention_layer`, `kernels_cpu`, `execution_layer`, `*_dispatch_pipeline`,
`cuda_manager`, `cuda_kernel_dispatcher`). That stack stays until new ggml
graphs exist (step 5, deferred).

## `cuda/*.cu`

| File | Callers (outside cuda/) | Verdict |
|---|---|---|
| `gguf_cuda.cu` | none | **moved** to `obsolete/cuda/` |
| `iq2m_dequant.cu` | none | **moved** |
| `kv_quant.cu` | none (`kv_quantized` in engine.hpp is only a flag) | **moved** |
| `exl3_matmul.cu` | `exl3_gpu.cpp`, `exl3_cuda.hpp`, tests | keep (exl3) |
| `ggml_moe_pack.cu` | `ggml_moe_backend/transfer_manager` | keep (ggml) |
| `expert_accum.cu` | `ggml_moe_backend/runner` (+ `cuda_manager` wrapper) | keep |
| `qwen4exp_hc_q8.cu` | `ggml_hc_backend.cpp` | keep |
| `qwen4exp_master_graph.cu` | `qwen4exp_stage_pool/early_ready.hpp` | keep |
| `iq1/iq2/iq3/iq4xs/q2_0/q5_1/q8_0/mxfp4/kquant_matmul.cu`, `matmul.cu`, `dequant_bf16.cu` | `cuda_kernel_dispatcher.cpp`, `cuda_manager.hpp`, tests | keep until GLM/DSV4 on ggml |
| `mla_attention.cu`, `attn_score_softmax.cu`, `dsv4_score_softmax.cu`, `rmsnorm.cu`, `wuv_int8_dequant.cu` | `cuda_manager.hpp`, `family_ops.cpp`, `attention_layer.cpp` | keep until GLM/DSV4 on ggml |
| `glm5next_kda.cu`, `glm5next_router.cu` | `cuda_manager.hpp` wrappers | **unconfirmed** — check whether the glm5next graph still uses the wrappers |
| `gdn_conv1d.cu` | `qwen4exp_conv1d_silu_gpu_host`, called by Qwen GDN **and by GLM5Next KDA** (`attention_layer.cpp`, `glm5next_kda_gpu`) | **keep** — not Qwen-only |
| `qwen4exp_attn.cu`, `qwen4exp_gdn.cu` | `cuda_manager.hpp` → `attention_layer.cpp` (layered Qwen path) | **moved** to `obsolete/cuda/` (see below) |

## Qwen-only native path — REMOVED and verified

Removed: `CudaManager::qwen4exp_attn_gpu_*` (9 methods), `qwen4exp_prepare_qkv_device`,
`qwen4exp_gdn_gpu_host`, their device scratch buffers/KV cache/norm cache and
extern decls (`cuda_manager.hpp` -739 lines +  ~90 more); the GPU branches in
`attention_layer.cpp` (CPU fallbacks untouched); the raw-device-attn-output
branch in `execution_layer.cpp`; `device_full_attn_input` in
`hyper_connections.cpp`; the two GPU tests in `main_test.cpp`.
`qwen4exp_attn_gpu` / `qwen4exp_gdn_gpu` config keys are still parsed but are now
**deprecated no-ops** (~60 configs set them; they must keep loading). The related
`perf_timer.hpp` counters are left in place (print only when nonzero).

Kept: `qwen4exp_conv1d_silu_gpu_host` + `gdn_conv1d.cu` (also used by GLM KDA).

Verification (build_ggmlcuda, `configs/qwen4exp_tg400_iq1m_22_26.toml`,
`--ngen 64`, prompt "Tell a story.", greedy):
- generated text byte-identical before/after
- decode-wall: before 23.20 (cold, first run) / 25.13 (warm rerun); after 25.63 tok/s
  => no regression; the 23.2 was run-order warm-up, not a difference
- `attn=0.00s` in the baseline: this config never entered the removed paths
- `zeromoe_tests`: exit 0, 0 failures
- NOT covered: layered (non-graph) Qwen decode no longer has a GPU attention
  path; policy says graph failure aborts the request, so that is intended.
Logs: `logs/qwen_flash_decode_{baseline,after}_*_2026-10-05.log`,
pre-edit copies `logs/*.pre_qwen_removal`.

## `kernels_cpu.cpp` (4,033 lines) — audit result: do NOT hand-delete

Method: for each of the 63 public symbols in `include/kernels_cpu.hpp`, callers
split into ggml-stack files (`ggml_*`, `cpu_qwen*`, `qwen_*`), legacy files
(`attention_layer`, `execution_layer`, `*_dispatch_pipeline`, `moe_*`,
`*_expert_executor`, ...) and tests. Intra-file references checked too.

Findings
- **Qwen's CPU experts already use ggml directly.** `ggml_qwen_cpu_moe.cpp`
  calls `ggml_vec_dot_iq1_m_q8_K / iq2_xxs_q8_K / iq3_xxs_q8_K / iq4_nl_q8_0`.
  The hand kernels are not on the Qwen decode hot path.
- **The hand kernels' real consumer is `cpu_quant_matmul`** (`kquant_dequant.hpp:663`),
  called by `cpu_expert_executor`, `moe_router_service`, `shared_expert_executor`,
  `dense_dispatch`, `token_dispatch_pipeline` — i.e. the legacy GLM/DeepSeek stack.
- ~50 of the 63 symbols have **no external caller**, but most are reached through
  `cpu_quant_matmul`'s per-type `switch` or `kquant_dequant.hpp` (2 refs inside
  the .cpp means "definition + dispatch"). Name-grep alone cannot prove them
  dead, and deleting by that list would break GLM/DeepSeek CPU paths.
- Truly unreferenced (0 refs anywhere): `dot_f32` only. Everything else is
  either dispatched or used by tests / the NEON file.
- Keep regardless: `matmul_qwen_q2_0_q2_0_fused` (ggml-stack only, Q2_0 VNNI),
  `matmul_qwen_iq1_m/iq2_xxs/q4_0_*_fused`, `cpu_*_enabled` gates.

Recommendation (belongs with step 5, GLM/DSV4 -> ggml)
Replace `cpu_quant_matmul`'s hand-written per-type kernels with one generic
ggml-backed routine: `ggml_get_type_traits_cpu(type)` -> `from_float` for the
activation into `vec_dot_type`, then `vec_dot` per output row (OpenMP over rows).
That is ~100 lines and would make ~3,000 lines of `kernels_cpu.cpp` removable.
Risk: numerics change (the hand kernels use their own int8 activation
quantization; tests at `main_test.cpp` ~205-420 pin tolerances against them)
and per-type performance may differ. Needs an A/B on real GLM-5.x CPU-expert
layers, so it was NOT done in this cleanup pass.

## Native CUDA matmul kernels — no technical reason to keep them

Question raised by the owner: if ggml/exllamav3 are the tensor libraries, why keep
hand-written mat-mul kernels? Answer: only because two call sites still route to them.

- Matmul-family files: `q2_0 88, mxfp4 126, matmul 127, q5_1 181, iq2 190, iq4xs 213,
  q8_0 221, iq3 252, iq1_dequant 404, kquant 524` = ~2,330 lines (+ `dequant_bf16` 251
  as the cuBLAS BF16 prepack).
- ggml-cuda (upstream 3cf03257f) supports every type they cover in its `mmvq` decode
  path (Q4_0 Q5_1 Q8_0 Q2_K..Q6_K IQ1_S/M IQ2_XXS/XS/S IQ3_XXS/S IQ4_NL/XS MXFP4 Q2_0),
  and all but IQ1_M in `mmq` (prefill). Coverage is not a reason.
- Two choke points, both `(type, device y, device x, device w, S, I, O)`:
  1. `CudaKernelDispatcher::matmul` (`cuda_kernel_dispatcher.cpp:109`) — called only by
     `gpu_expert_executor.cpp` (legacy routed-expert GPU compute).
  2. `CudaManager::dense_matmul_device_host` (`cuda_manager.hpp` ~2851, ~3139) — dense
     attention/FFN projections.
- Who still reaches them: GLM-5.x and DeepSeek-V4 only. `configs/glm53_*` do not enable
  `moe_ggml_mul_mat_id` (default false), so the legacy per-expert native path is live there.
  Qwen4Exp (full graph) and GLM-5.3-Flash (full graph) do not use them.
- exl3: `exl3_matmul.cu` stays (separate library, not a ggml type).

Proposed replacement (not done): one ggml-backed single-op `mul_mat` helper behind both
choke points — weights/x/y bound as external-pointer tensors (the HC backend already does
this; patch `0004-ggml-cuda-external-pointer-padding` exists for it), graph + gallocr cached
per (type, I, O, S), CUDA-graph replay for S=1. ~150 lines replaces ~2,330.
Risks to measure, not assume:
- numerics: native kernels use fp32 activations; ggml mmvq quantizes x to Q8_1 (what
  llama.cpp does). Needs a coherent-text + logit A/B on real GLM-5.x, not a unit tolerance.
- S=1 decode overhead: ~8 experts x 3 projections x 78 layers = ~1,900 calls/token on the
  legacy path; per-call graph launch cost must be small next to GLM-5.x's ~1 s/token.
This is the same GLM/DSV4 work as the `cpu_quant_matmul` replacement above and should be
done together (one A/B on real model runs).

## Not yet audited

- Other `src/*.cpp` / `include/*.hpp` classification (native-stack vs ggml vs shared).
- `cpu_*` engine files, `models/*` (`cpu_qwen_moe.hpp` etc.).
- Remaining `main_test.cpp` / `test_components.cpp` tests tied to removed code.
- Configs: only GLM-5.2 (deferred by owner) and root clutter handled so far.
