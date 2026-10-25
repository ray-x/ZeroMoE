# Where ZeroMoE still uses CUDA directly (not through ggml)

Audit of 2026-10-18 on `master` (`703dde9`). Method: static references only
(`git grep` of the exported entry points of every `cuda/*.cu`, and of raw
`cuda*`/`cublas*`/`<<<` calls in `src/`, `include/`, `models/`). Nothing here was
verified by running a model, so "reachable" below means "referenced", not "executed
in a run". Out of scope as requested: EXL3 (WIP) and the GLM-5.3 non-Flash path.

## Short answer

Not everything is on ggml. Three kinds of direct CUDA remain, and only the last is a
candidate for replacement.

| kind | where | replaceable by ggml? |
|---|---|---|
| 1. Memory and transfer management | `gpu_topology`, `cuda_manager` pools, `ggml_moe_transfer_manager`, `early_ready_lane`, `qwen4exp_stage_pool`, `active_layer_upload_workers`, `scratch_buffers`, `prefill_expert_stream`, `cufile_io` | No. ggml has no expert cache, DMA scheduling or pinned staging. The ggml graphs alias these buffers. |
| 2. Custom ops installed into ggml graphs | `cuda/ggml_moe_pack.cu`, `expert_accum.cu`, `qwen4exp_hc_q8.cu` (via `ggml_hc_backend`), `kquant_matmul.cu` (DSV4.1 `v41_native_f32` hook), `qwen_stage_kernels.cu`, `mla_attention.cu` (via `family_ops`) | Case by case. These exist because ggml lacks the op or the fused form. |
| 3. The layered (non-graph) compute stack | `CudaManager` (`include/cuda_manager.hpp`, 3.4k lines, ~250 raw CUDA calls) plus `cuda/{matmul,dequant_bf16,iq1_dequant,iq2_matmul,iq3_matmul,iq4xs_matmul,kquant_matmul,mxfp4_matmul,q2_0_matmul,q5_1_matmul,q8_0_matmul,rmsnorm,attn_score_softmax,dsv4_score_softmax,gdn_conv1d,glm5next_kda,glm5next_router,wuv_int8_dequant,expert_activation}.cu` (~4.3k lines) | Yes in principle, for any model family that has a full graph. Not for families that only have the layered path (GLM-5.x non-Flash). |

## Per-kernel references

`used_by` is the set of non-`.cu` files that mention an exported entry point.

| file | loc | referenced by | belongs to |
|---|---:|---|---|
| `attn_score_softmax.cu` | 97 | `cuda_manager.hpp` | layered MLA attention |
| `dsv4_score_softmax.cu` | 94 | `cuda_manager.hpp` | layered DSV4 attention |
| `gdn_conv1d.cu` | 61 | `cuda_manager.hpp` | layered Qwen GDN (Qwen is graph-only by policy, see below) |
| `glm5next_kda.cu` | 88 | `cuda_manager.hpp` | layered GLM5Next (Flash) |
| `glm5next_router.cu` | 107 | `cuda_manager.hpp` | layered GLM5Next (Flash) |
| `rmsnorm.cu` | 78 | `cuda_manager.hpp` | layered |
| `wuv_int8_dequant.cu` | 80 | `cuda_manager.hpp`, `attention_layer.cpp` | layered MLA |
| `matmul.cu`, `dequant_bf16.cu`, `iq1_dequant.cu`, `iq2/3/4xs_matmul.cu`, `mxfp4_matmul.cu`, `q2_0/q5_1/q8_0_matmul.cu` | 128 to 404 each | `cuda_manager.hpp`, `cuda_kernel_dispatcher.cpp`, `kquant_dequant.hpp`, `main_test.cpp` | layered dense GEMV/dequant for every quant type |
| `kquant_matmul.cu` | 728 | `ggml_f32_quant_backend.hpp`, `ggml_deepseek_v41_graph.cpp`, DSV4.1 tests (NOT `cuda_manager.hpp`) | **DSV4.1 graph, kind 2**: `zeromoe_install_f32_quant_graph_hook()` routes ggml ops to q2_K/q3_K kernels |
| `expert_accum.cu` | 60 | `cuda_manager.hpp`, `ggml_moe_backend.cpp`, `ggml_moe_runner.cpp` | kind 2 and 3 |
| `expert_activation.cu` | 70 | `gpu_expert_executor.cpp` | layered expert FFN |
| `ggml_moe_pack.cu` | 179 | `ggml_moe_backend.cpp`, `ggml_moe_transfer_manager.cpp` | kind 2 |
| `qwen4exp_hc_q8.cu` | 172 | `ggml_hc_backend.cpp`, `q8_0_matmul.cu` | kind 2 (Qwen HC graph op) |
| `qwen_stage_kernels.cu` | 85 | `early_ready_lane.hpp`, `qwen4exp_stage_pool.hpp` | kind 1/2 (staging) |
| `mla_attention.cu` | 214 | `family_ops.cpp` | layered MLA |

## Observations

* DSV4.1's graph is the only graph family that calls hand-written quant kernels from
  inside the graph (`kquant_matmul.cu`, kind 2). It also asks `cudaMemGetInfo`
  directly in `ggml_deepseek_v4_graph.cpp` (sizing only). Left untouched: DSV4
  tuning is in progress elsewhere.
* The layered stack is reachable from any family without a required full graph. Qwen4Exp
  has the "full graph required, failed graph fails the request" policy, so
  `gdn_conv1d.cu` and the Qwen parts of `CudaManager` should be dead in production; they
  are still exercised by `tests/native/tests_gpu_kernels.cpp` and
  `tests_gpu_attention.cpp`. Deleting them is a separate change that needs the user's
  go-ahead because the same file also serves GLM.
* GLM5Next (Flash) still has layered kernels (`glm5next_kda`, `glm5next_router`)
  because its full graph is opt-in (`ZEROMOE_GLM5NEXT_FULL_GRAPH`) and not yet
  required for every request shape.
* GLM-5.3 (non-Flash) and older DeepSeek paths depend on all of kind 3. Replacing it
  means building a full graph for that family first, which is the GLM work planned in
  another session.
* `CudaManager` is also a pool/stream owner (kind 1), so it cannot simply be deleted
  when the last layered family goes: its allocation and stream code would have to move
  to a smaller class first.

## Suggested follow-ups (not done here)

1. Split `cuda_manager.hpp` (3.4k lines) into `DeviceMemory` (pools, pinned buffers,
   streams: kind 1) and `LayeredKernels` (the `launch_*` wrappers: kind 3). This is a
   prerequisite for retiring kind 3 per family, and is mechanical.
2. Once Qwen4Exp's layered code is confirmed dead, delete its kernels and tests in
   their own commit.
3. Decide whether DSV4.1 keeps its `kquant_matmul` hook or moves to ggml's own
   `MUL_MAT`/`MUL_MAT_ID` with the patched-in quant types.
