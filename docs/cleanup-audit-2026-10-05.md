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
- `moestream_tests`: exit 0, 0 failures
- NOT covered: layered (non-graph) Qwen decode no longer has a GPU attention
  path; policy says graph failure aborts the request, so that is intended.
Logs: `logs/qwen_flash_decode_{baseline,after}_*_2026-10-05.log`,
pre-edit copies `logs/*.pre_qwen_removal`.

## Not yet audited

- `src/*.cpp` / `include/*.hpp` classification (native-stack vs ggml vs shared).
- `kernels_cpu.cpp` (4,033 lines) — which functions ggml's CPU backend can replace.
- `cpu_*` engine files, `models/*` (`cpu_qwen_moe.hpp` etc.).
- `main_test.cpp` / `test_components.cpp` tests tied to removed code.
- Configs: only GLM-5.2 (deferred by owner) and root clutter handled so far.

## Already moved to `obsolete/` (step 1)

Root debug logs, scratch dumps, one-off scripts, perf data, tgz archives,
CLAUDE.v1-4, session transcripts, unreferenced build dirs (`build_lf`,
`build_prof`, `build_verify`, `build_llamacpp_ggml`, `.lcb-run`, `llama_logs`).
Left in place because scripts/docs reference them: `build_b11401`,
`build-bee-instrumented`, `build_fork_diag`, `bench_tmp`, `doc`, `Testing`.
`CMakeLists.txt` backup before edits: `obsolete/cmake_backup/`.
