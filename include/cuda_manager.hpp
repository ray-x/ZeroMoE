// cuda_manager.hpp — Multi-GPU pool manager with topology-aware execution
// Implements the asymmetric dual-GPU pipeline from optimize.md:
//   GPU 0 (CoreEngine): Dense layers + KV cache + activations
//   GPU 1+ (ExpertVault): MoE expert LRU cache in VRAM
#pragma once
#include "engine.hpp"
#include "engine_config.hpp"
#include "perf_timer.hpp"
#include "gpu_topology.hpp"
#include "cublas_matmul.hpp"
#include "kquant_dequant.hpp"  // dequant_tensor_f32, for upload_dequantized_f32
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <limits>
#include <new>

// External CUDA kernel hooks
extern "C" void launch_gguf_q4_0_dp4a(float* y, const float* x, const void* weights, int S, int I, int O);
extern "C" void launch_q8_0_matmul(float* y, const float* x, const void* weights, int S, int I, int O);
extern "C" void launch_q5_1_matmul(float* y, const float* x, const void* weights, int S, int I, int O);
extern "C" void launch_dequant_q8_0_to_bf16(void* dst_bf16, const void* src_q8_0, int64_t n_elements);
// Type-dispatched BF16 decoder (cuda/dequant_bf16.cu): Q8_0/Q5_1/Q4_K/Q5_K/Q6_K.
extern "C" bool launch_dequant_to_bf16(int ggml_type, void* dst_bf16, const void* src, int64_t n_elements);
extern "C" void launch_f32_to_bf16(void* dst_bf16, const float* src_f32, int64_t n);
extern "C" void launch_q2_K_matmul(float* y, const float* x, const void* weights, int S, int I, int O);
extern "C" void launch_q3_K_matmul(float* y, const float* x, const void* weights, int S, int I, int O);
extern "C" void launch_q4_K_matmul(float* y, const float* x, const void* weights, int S, int I, int O);
extern "C" void launch_q5_K_matmul(float* y, const float* x, const void* weights, int S, int I, int O);
extern "C" void launch_q6_K_matmul(float* y, const float* x, const void* weights, int S, int I, int O);
extern "C" void launch_iq2_xxs_matmul(float* y, const float* x, const void* weights, int S, int I, int O);
extern "C" void launch_iq2_xs_matmul(float* y, const float* x, const void* weights, int S, int I, int O);
extern "C" void launch_iq2_s_matmul(float* y, const float* x, const void* weights, int S, int I, int O);
extern "C" void launch_iq3_xxs_matmul(float* y, const float* x, const void* weights, int S, int I, int O);
extern "C" void launch_iq4_xs_matmul(float* y, const float* x, const void* weights, int S, int I, int O);
extern "C" void launch_iq4_nl_matmul(float* y, const float* x, const void* weights, int S, int I, int O);
// IQ3_S: real grid-codebook kernel (cuda/iq3_matmul.cu, verbatim port of
// ggml's dequantize_row_iq3_s), added 2026-09 -- IQ3_S previously had no
// fused CUDA kernel at all (see kquant_dequant.hpp's gpu_unsupported_reason,
// pre-this-change).
extern "C" void launch_iq3s_matmul(float* y, const float* x, const void* weights, int S, int I, int O);
// IQ1_S: real grid-codebook kernel (cuda/iq1_dequant.cu, verbatim port of
// ggml's dequantize_row_iq1_s), added 2026-09 after finding this file's
// pre-existing "iq1_matmul_kernel_optimized" does NOT implement the real
// format (see that kernel's own updated comment for the full finding).
extern "C" void launch_iq1s_matmul(float* y, const float* x, const void* weights, int S, int I, int O);
// IQ1_M: real grid-codebook kernel (cuda/iq1_dequant.cu, verbatim port of
// ggml's dequantize_row_iq1_m), added 2026-09 alongside IQ1_S -- reuses the
// SAME 2048-entry c_iq1s_grid table (IQ1_M's dequant reference indexes the
// identical iq1s_grid), just a different per-block scale/index-packing
// layout (block_iq1_m: qs[32]+qh[16]+scales[8], no per-block `d` half --
// the super-block scale is itself packed into the top nibbles of
// scales[]'s 4 uint16 words). Not present in the real GLM-5.3 checkpoint
// this project currently tests against (confirmed via a direct header
// parse of every downloaded shard: 0 IQ1_M tensors) -- verified via
// Test 5's synthetic-data path only, no real production run behind this
// one the way IQ1_S got.
extern "C" void launch_iq1m_matmul(float* y, const float* x, const void* weights, int S, int I, int O);
// MXFP4: dormant kernel (see kquant_dequant.hpp's gpu_type_supported()) --
// wired here for Test 5 harness parity even though this switch is only
// reached once gpu_type_supported() itself has already gated the type.
extern "C" void launch_mxfp4_matmul(float* y, const float* x, const void* weights, int S, int I, int O);
extern "C" void launch_rmsnorm(float* y, const float* x, const float* w, int S, int D, float eps);
// GPU-side scatter-accumulation of routed-expert outputs. See cuda/expert_accum.cu.
extern "C" void launch_expert_scatter_accum(const float* src, const int* tok, const float* w,
                                            float* dst, int n, int D, int S, cudaStream_t stream);

// qwen4exp chunkwise-parallel GDN (gated delta rule). See cuda/qwen4exp_gdn.cu.
extern "C" void launch_gdn_gather(const float*, const float*, const float*,
                                  float*, float*, float*, int, int, int, int, int, int, cudaStream_t);
extern "C" void launch_gdn_prep_scalars(const float*, const float*, float*, float*,
                                        float*, float*, float*, int*, int, int, int, int, cudaStream_t);
extern "C" void launch_gdn_build_M_rhs(const float*, const float*, const float*, const float*,
                                       const float*, const float*, const float*, const int*,
                                       float*, float*, int, int, int, cudaStream_t);
extern "C" bool launch_gdn_forward_subst(const float*, const float*, float*, int, int, int, cudaStream_t);
extern "C" void launch_gdn_build_P_out(const float*, const float*, const float*, const float*,
                                       const float*, const float*, const int*, float*, float*,
                                       int, int, int, float, cudaStream_t);
extern "C" void launch_gdn_scale_state(const float*, const float*, const float*, float*, float*,
                                       int, int, int, cudaStream_t);
extern "C" void launch_gdn_scatter(const float*, float*, int, int, int, int, int, cudaStream_t);

// GDN's causal short conv + SiLU (EngineConfig::qwen4exp_gdn_conv_gpu). See
// cuda/gdn_conv1d.cu.
extern "C" void launch_gdn_conv1d_silu(const float*, const float*, const float*,
                                       float*, int, int, int, cudaStream_t);
// GLM5NEXT KDA recurrent delta rule. Unlike Qwen4Exp GDN, GLM's decay is
// per key channel, so it needs its own kernel rather than the chunkwise GDN
// formulation above.
extern "C" void launch_glm5next_kda_delta_rule(
    const float*, const float*, const float*, const float*, const float*,
    float*, float*, int, int, int, cudaStream_t);
// GLM5NEXT learned-router gating/top-k. The router matmul itself uses the
// normal device-resident dense dispatcher; this kernel performs the remaining
// bias-aware selection and weight preparation without a host round trip.
extern "C" void launch_glm5next_router_topk(
    float*, const float*, int32_t*, float*, int, int, int, int, int, float,
    cudaStream_t);

// qwen4exp dense full-attention: scale + causal mask + row softmax over the
// [H, n_q, T] score matrix. See cuda/qwen4exp_attn.cu.
extern "C" void launch_qwen4exp_scale_softmax_causal(
    float* scores, int H, int n_q, int T, int q_base_abs, float scale,
    cudaStream_t stream);
extern "C" void launch_qwen4exp_sigmoid_gate(
    float *x, const float *gate, int n, cudaStream_t stream);
extern "C" void launch_qwen4exp_prepare_qkv(
    const float *q_full, const float *k_raw, const float *v_raw,
    float *q, float *gate, float *k, float *v,
    const float *q_norm, const float *k_norm,
    int S, int H, int Hkv, int hd, int n_rot, int pos_base,
    float theta_base, float eps, cudaStream_t stream);
// EngineConfig::attn_score_gpu (cuda/attn_score_softmax.cu) -- see
// score_gpu_batched_host below for the full design.
extern "C" void launch_combine_scale_softmax(
    const float* scores_nope, const float* scores_rope, float* scores,
    int H, int T, float scale, int S, int pos_base, cudaStream_t stream);
// int8-persistent WUV coverage fix (cuda/wuv_int8_dequant.cu) -- see
// upload_wuv_int8/wuv_int8_batched_host below for the full design.
extern "C" void launch_dequant_wuv_int8(
    const int8_t* weight_int8, const float* scale, uint16_t* out_bf16,
    int H, int O, int I, cudaStream_t stream);
// EngineConfig::dsv4_score_gpu (cuda/dsv4_score_softmax.cu) -- see
// dsv4_score_gpu_batched_host below for the full design.
extern "C" void launch_dsv4_score_softmax(
    float* scores, const float* sinks, int H, int T, float scale, cudaStream_t stream);

class CudaManager {
public:
    GpuTopology topology;
    int device_id = 0;  // Core engine GPU device ID
    cublasHandle_t cublas_handle = nullptr;  // cuBLAS handle, bound to core compute_stream

    // Dedicated stream + cuBLAS handle for grouped_output_proj_host
    // (dsv4_outproj_gpu) -- deliberately NOT the shared cublas_handle above,
    // which is left on CUDA's legacy default stream (stream 0, see that
    // field's own init comment). Real-hardware A/B found that sharing
    // stream 0 for this call site cost MORE in MoE expert dispatch than it
    // saved in attention: the legacy default stream implicitly
    // synchronizes with EVERY other stream on the device unless per-thread
    // default streams are used, and this call fires 43x/token (once per
    // layer) -- each one a synchronization barrier landing right at the
    // layer boundary where MoE's own async Pass A/Pass B pipeline
    // (core.compute_stream/copy_stream) needs to keep flowing. Measured:
    // attn saved ~3.9s, expert='s own untimed/dark portion grew by ~3.3s,
    // net wash on total throughput. A genuinely separate, non-default
    // stream (created once, never touched by anything else) can never
    // trigger that implicit cross-stream sync, isolating this call site
    // from MoE's dispatch entirely.
    cudaStream_t outproj_stream = nullptr;
    cublasHandle_t outproj_cublas_handle = nullptr;
    float* d_outproj_x = nullptr;
    float* d_outproj_y = nullptr;
    size_t d_outproj_x_cap = 0;
    size_t d_outproj_y_cap = 0;

    // Persistent device buffers on core GPU (never freed mid-session)
    float* d_x = nullptr;
    float* d_y = nullptr;
    size_t d_x_cap = 0;
    size_t d_y_cap = 0;
    // Grow-only scratch for the BF16-matched fast-path's per-call activation
    // conversion (dense_matmul_bf16_matched_host) -- separate from d_x/d_y
    // above since those are F32-typed and shared by the fused-kernel /
    // mixed-type dispatch paths; this one holds the freshly-converted BF16
    // activation each call, half the byte size of the equivalent d_x.
    void* d_x_bf16 = nullptr;
    size_t d_x_bf16_cap = 0;
    // Grow-only scratch holding ONE dequantised weight in BF16, for the
    // prefill GEMM path below. Sized by the largest projection actually
    // dispatched (attn_q at 2560x12288 -> 62.9 MB on qwen4exp); a single
    // reused buffer, not per-layer. Keeping weights permanently resident in
    // BF16 instead would need 252 GB for this model -- see cuda/dequant_bf16.cu.
    void* d_w_bf16 = nullptr;
    size_t d_w_bf16_cap = 0;

    // qwen4exp dense-attention GPU path scratch (qwen4exp_attn_gpu_host).
    // Grow-only and reused across all 12 dense layers and every call --
    // same pattern as d_score_* above. Freed in the destructor.
    float* d_q4_Q = nullptr;      size_t d_q4_Q_cap = 0;
    float* d_q4_qfull = nullptr;  size_t d_q4_qfull_cap = 0;
    float* d_q4_gate = nullptr;   size_t d_q4_gate_cap = 0;
    float* d_q4_K = nullptr;      size_t d_q4_K_cap = 0;
    float* d_q4_V = nullptr;      size_t d_q4_V_cap = 0;
    float* d_q4_qnorm = nullptr;  size_t d_q4_qnorm_cap = 0;
    float* d_q4_knorm = nullptr;  size_t d_q4_knorm_cap = 0;
    float* d_q4_scores = nullptr; size_t d_q4_scores_cap = 0;
    float* d_q4_out = nullptr;    size_t d_q4_out_cap = 0;

    // GDN causal conv1d+SiLU GPU scratch (qwen4exp_conv1d_silu_gpu_host).
    // Grow-only, shared across all 36 GDN layers and every call. Weight and
    // history are small (163 KB / 123 KB at qwen4exp's real dims) and
    // re-uploaded every call rather than cached per-layer on device --
    // negligible next to x/out's own size, not worth the extra bookkeeping.
    // Freed in the destructor.
    float* d_gdnconv_x = nullptr;    size_t d_gdnconv_x_cap = 0;
    float* d_gdnconv_w = nullptr;    size_t d_gdnconv_w_cap = 0;
    float* d_gdnconv_h = nullptr;    size_t d_gdnconv_h_cap = 0;
    float* d_gdnconv_out = nullptr;  size_t d_gdnconv_out_cap = 0;

    // GLM5NEXT KDA recurrent delta-rule scratch. The legacy host bridge below
    // still uses d_glm_kda_state for its self-contained correctness/test path.
    // Production GLM execution uses d_glm_kda_state_by_layer instead: each
    // KDA layer keeps its recurrent matrix on the device across calls, so the
    // large H*hd*hd state does not cross PCIe once per layer/token.
    float* d_glm_kda_q = nullptr;     size_t d_glm_kda_q_cap = 0;
    float* d_glm_kda_k = nullptr;     size_t d_glm_kda_k_cap = 0;
    float* d_glm_kda_v = nullptr;     size_t d_glm_kda_v_cap = 0;
    float* d_glm_kda_beta = nullptr;  size_t d_glm_kda_beta_cap = 0;
    float* d_glm_kda_g = nullptr;     size_t d_glm_kda_g_cap = 0;
    float* d_glm_kda_state = nullptr; size_t d_glm_kda_state_cap = 0;
    float* d_glm_kda_out = nullptr;   size_t d_glm_kda_out_cap = 0;

    struct Glm5nextKdaDeviceState {
        float* data = nullptr;
        size_t cap = 0;
        bool initialized = false;
    };
    std::vector<Glm5nextKdaDeviceState> d_glm_kda_state_by_layer;

    // GLM5NEXT native CUDA router scratch. The router input/logits are
    // grow-only per-call buffers; IDs/weights come back to the host because
    // the existing RAM/VRAM expert-cache bookkeeping consumes that compact
    // table. No model weights are duplicated here.
    float* d_glm_router_x = nullptr;       size_t d_glm_router_x_cap = 0;
    float* d_glm_router_logits = nullptr;  size_t d_glm_router_logits_cap = 0;
    float* d_glm_router_bias = nullptr;    size_t d_glm_router_bias_cap = 0;
    int32_t* d_glm_router_ids = nullptr;   size_t d_glm_router_ids_cap = 0;
    float* d_glm_router_weights = nullptr; size_t d_glm_router_weights_cap = 0;

    // qwen4exp chunkwise GDN scratch (qwen4exp_gdn_gpu_host). Grow-only,
    // shared across all 36 GDN layers and every chunk. Freed in the destructor.
    // Per-layer routed-expert output accumulator, one per GPU (experts for one
    // layer can land on different devices in the dual-role config). Grow-only.
    static constexpr int kMaxAccumGpus = 8;
    float* d_acc[kMaxAccumGpus]      = {nullptr};  size_t d_acc_cap[kMaxAccumGpus]  = {0};
    int*   d_acc_tok[kMaxAccumGpus]  = {nullptr};  size_t d_acc_tok_cap[kMaxAccumGpus] = {0};
    float* d_acc_w[kMaxAccumGpus]    = {nullptr};  size_t d_acc_w_cap[kMaxAccumGpus] = {0};
    bool   acc_active[kMaxAccumGpus] = {false};

    float* d_gdn_Q=nullptr;     size_t d_gdn_Q_cap=0;
    float* d_gdn_K=nullptr;     size_t d_gdn_K_cap=0;
    float* d_gdn_V=nullptr;     size_t d_gdn_V_cap=0;
    float* d_gdn_beta=nullptr;  size_t d_gdn_beta_cap=0;
    float* d_gdn_g=nullptr;     size_t d_gdn_g_cap=0;
    float* d_gdn_out=nullptr;   size_t d_gdn_out_cap=0;
    float* d_gdn_state=nullptr; size_t d_gdn_state_cap=0;
    float* d_gdn_Qc=nullptr;    size_t d_gdn_Qc_cap=0;
    float* d_gdn_Kc=nullptr;    size_t d_gdn_Kc_cap=0;
    float* d_gdn_Vc=nullptr;    size_t d_gdn_Vc_cap=0;
    float* d_gdn_Kcs=nullptr;   size_t d_gdn_Kcs_cap=0;
    float* d_gdn_KK=nullptr;    size_t d_gdn_KK_cap=0;
    float* d_gdn_QK=nullptr;    size_t d_gdn_QK_cap=0;
    float* d_gdn_M=nullptr;     size_t d_gdn_M_cap=0;
    float* d_gdn_P=nullptr;     size_t d_gdn_P_cap=0;
    float* d_gdn_R=nullptr;     size_t d_gdn_R_cap=0;
    float* d_gdn_Qs=nullptr;    size_t d_gdn_Qs_cap=0;
    float* d_gdn_U0=nullptr;    size_t d_gdn_U0_cap=0;
    float* d_gdn_U=nullptr;     size_t d_gdn_U_cap=0;
    float* d_gdn_outc=nullptr;  size_t d_gdn_outc_cap=0;
    float* d_gdn_gcs=nullptr;   size_t d_gdn_gcs_cap=0;
    float* d_gdn_At=nullptr;    size_t d_gdn_At_cap=0;
    float* d_gdn_ecum=nullptr;  size_t d_gdn_ecum_cap=0;
    float* d_gdn_cf=nullptr;    size_t d_gdn_cf_cap=0;
    float* d_gdn_betac=nullptr; size_t d_gdn_betac_cap=0;
    int*   d_gdn_fast=nullptr;  size_t d_gdn_fast_cap=0;
    // Persistent HOST-side scratch for the same per-call activation
    // conversion, grow-only and reused across calls. Originally this was a
    // fresh std::vector<uint16_t> allocated (and freed) on every single
    // call -- real-hardware use showed visibly elevated per-token CPU usage
    // after the BF16 fast-path landed, traced to exactly this: the old
    // dense_matmul_host path did a direct cudaMemcpy straight from x_host
    // with zero host-side CPU work, so the heap alloc/free churn here (once
    // per converted layer, per token, on the single main decode thread) was
    // new cost that didn't exist before. Reusing one grow-only buffer
    // removes that churn; the conversion loop itself stays scalar (not
    // AVX) -- cheap enough at real S=1 decode widths that vectorizing it
    // wasn't judged worth the risk here, unlike this project's other
    // per-element dequant loops which run over far larger data.
    uint16_t* h_x_bf16_scratch = nullptr;
    size_t h_x_bf16_scratch_cap = 0;
    size_t h_x_bf16_scratch_reservation_bytes = 0;

    // MLA qabsorb BF16 fast-path (EngineConfig::attn_qabsorb_gpu) -- see
    // upload_wuk_bf16/qabsorb_bf16_batched_host below for the full design.
    // Per-layer BF16-converted WUK (attn_k_b), one cudaMalloc per layer,
    // populated once at startup by warm_mla_absorb_cache
    // (attention_layer.cpp) and read-only for the rest of the process
    // lifetime -- same "convert once, dispatch many times" shape as
    // QuantizedTensor::gpu_fast_ptr above, just keyed by layer index
    // instead of living on a QuantizedTensor (WUK isn't one -- it's a
    // dequant-cached std::vector<float>, wuk_cache in attention_layer.cpp).
    // Empty vector (not yet sized) means "never attempted"; a null entry
    // at a valid layer index means "attempted and failed/skipped" (VRAM
    // budget exhausted, missing weights, etc.) -- both cases make
    // qabsorb_bf16_batched_host's d_W_bf16-null check return false and the
    // caller falls back to the CPU path for that layer, same discipline as
    // every other best-effort GPU fast-path in this file.
    std::vector<void*> wuk_bf16_by_layer;
    // Grow-only scratch for qabsorb's per-call activation, separate from
    // d_x_bf16/h_x_bf16_scratch above (dense_matmul_bf16_matched_host's own
    // scratch) since qabsorb's shape is [H,S,nope] -- a different size
    // multiplier (H heads) than a single dense_proj call ever uses, and
    // reusing the same buffer would mean re-growing on every alternating
    // call between the two call sites instead of settling at a stable size.
    void* d_qabsorb_x_bf16 = nullptr;
    size_t d_qabsorb_x_bf16_cap = 0;
    float* d_qabsorb_y = nullptr;
    size_t d_qabsorb_y_cap = 0;
    uint16_t* h_qabsorb_x_bf16_scratch = nullptr;
    size_t h_qabsorb_x_bf16_scratch_cap = 0;
    size_t h_qabsorb_x_bf16_scratch_reservation_bytes = 0;

    // WUV (attn_v_b) BF16 fast-path -- the score GPU path's own down-
    // projection step, added 2026-08-14 after a real production run showed
    // the CPU per-head OMP-parallel WUV loop costing 0.916ms/call (2.86s
    // over one 40-token decode run), the SAME "OMP team-spawn cost exceeds
    // a tiny per-call workload" pattern qabsorb's own gather fix already
    // hit and fixed at a different call site. Structurally identical to
    // wuk_bf16_by_layer above (H independent per-head [vh,kvl] GEMMs, same
    // strided-batched dispatch), just a different static per-layer weight.
    std::vector<void*> wuv_bf16_by_layer;
    void* d_wuv_ctx_bf16 = nullptr;
    size_t d_wuv_ctx_bf16_cap = 0;
    float* d_wuv_out = nullptr;
    size_t d_wuv_out_cap = 0;
    uint16_t* h_wuv_ctx_bf16_scratch = nullptr;
    size_t h_wuv_ctx_bf16_scratch_cap = 0;
    size_t h_wuv_ctx_bf16_scratch_reservation_bytes = 0;

    // 2026-08-29: int8-persistent WUV storage -- see upload_wuv_int8/
    // wuv_int8_batched_host below for the full design (real motivation:
    // wuv_bf16_by_layer above only covers a minority of layers on GPU1
    // because it competes for an already-exhausted VRAM budget with
    // qabsorb/the score mirror -- storing int8 instead of BF16 roughly
    // halves the per-layer footprint, buying real coverage in the SAME
    // existing budget). Per-row (per (head,output-row) pair) scale,
    // matching this project's own "prefer finer-grained scales" convention
    // -- wuv_int8_scale_by_layer[l] is a SEPARATE small device buffer of
    // H*vh floats per layer, uploaded once alongside the int8 weights.
    // d_wuv_dequant_bf16 is TRANSIENT, grow-only scratch (like d_wuv_ctx_
    // bf16/d_wuv_out above) sized to exactly ONE layer's worth -- shared
    // across every dispatch call regardless of which layer, not one
    // buffer per layer, so its VRAM cost is a fixed one-time ~16.8MB (at
    // GLM-5.2's real dims) added ONCE, not multiplied by layer count.
    std::vector<int8_t*> wuv_int8_by_layer;
    std::vector<float*> wuv_int8_scale_by_layer;
    uint16_t* d_wuv_dequant_bf16 = nullptr;
    size_t d_wuv_dequant_bf16_cap = 0;

    // MLA score GPU fast-path (EngineConfig::attn_score_gpu) -- see
    // ensure_lc_rc_gpu/upload_lc_rc_rows/score_gpu_batched_host below for the
    // full design. Real-hardware feasibility bench (tests/bench_score_cpu_
    // vs_gpu.cu): 5x-51x speedup over CPU, growing with context length,
    // since unlike qabsorb this phase's CPU cost is UNBOUNDED with context
    // (no sliding window for GLM-5.2). Persistent per-layer GPU mirror of
    // the KV latent cache (engine.Lc/Rc, attention_layer.cpp) -- F32, not
    // BF16, since correctness (not VRAM budget) is the first concern here
    // and the raw compute cost is already tiny per the bench; sized to
    // engine.max_t so it never needs to grow/realloc mid-generation (a
    // realloc would require re-uploading the WHOLE cache, defeating the
    // point of an incrementally-grown mirror). Same null-means-"never
    // attempted-or-failed" discipline as wuk_bf16_by_layer above.
    std::vector<float*> d_lc_by_layer;
    std::vector<float*> d_rc_by_layer;
    std::vector<int> lc_rc_max_t_by_layer;  // allocated capacity per layer, for bounds checks
    // Grow-only scratch, shared across layers/calls (same rationale as
    // qabsorb's own scratch above -- settle at a stable size instead of
    // re-growing every alternating call). Sized to the largest H*T seen so
    // far; T can be up to engine.max_t.
    float* d_score_q_absorbed = nullptr; size_t d_score_q_absorbed_cap = 0;
    float* d_score_q_rope = nullptr;     size_t d_score_q_rope_cap = 0;
    float* d_score_nope_buf = nullptr;   size_t d_score_nope_buf_cap = 0;
    float* d_score_rope_buf = nullptr;   size_t d_score_rope_buf_cap = 0;
    float* d_score_combined = nullptr;   size_t d_score_combined_cap = 0;
    float* d_score_context = nullptr;    size_t d_score_context_cap = 0;

    // DeepSeek-V4 raw-SWA/HCA score GPU fast-path (EngineConfig::
    // dsv4_score_gpu) -- see ensure_dsv4_kvc_gpu/upload_dsv4_kvc_row/
    // ensure_dsv4_hca_gpu/upload_dsv4_hca_row/dsv4_score_gpu_batched_host
    // below for the full design. Scoped to ratio==0 (raw-SWA) and
    // ratio==128 (HCA) layers only -- CSA (ratio==4) stays entirely on CPU,
    // see dsv4_score_softmax.cu's own header comment for why. Two
    // independent per-layer mirrors:
    //   - d_dsv4_kvc_by_layer: the raw SWA ring buffer
    //     (engine.KVc[layer_idx]), [window,qh] F32, FIXED size (the ring
    //     capacity never grows), updated one row at a time at its ring slot
    //     -- NOT append-only, unlike d_lc_by_layer/d_rc_by_layer above.
    //   - d_dsv4_hca_by_layer: the HCA compressed-block cache
    //     (Dsv4CompressedLayerState::keys), [cache_capacity,qh] F32,
    //     append-only by block index (one new row every 128 tokens).
    // Both null-means-"never attempted-or-failed", same discipline as
    // wuk_bf16_by_layer/d_lc_by_layer above.
    std::vector<float*> d_dsv4_kvc_by_layer;
    std::vector<int>    dsv4_kvc_window_by_layer;   // allocated ring capacity per layer
    std::vector<float*> d_dsv4_hca_by_layer;
    std::vector<int>    dsv4_hca_capacity_by_layer; // allocated block capacity per layer
    // Grow-only scratch, shared across layers/calls -- same rationale as
    // d_score_* above (settle at a stable size instead of re-growing every
    // call). d_dsv4_score_kv holds the compacted (de-wrapped) combined K,
    // sized to the largest T = past_context_len+hca_visible seen so far.
    float* d_dsv4_score_q       = nullptr; size_t d_dsv4_score_q_cap       = 0;
    float* d_dsv4_score_kv      = nullptr; size_t d_dsv4_score_kv_cap      = 0;
    float* d_dsv4_score_scores  = nullptr; size_t d_dsv4_score_scores_cap  = 0;
    float* d_dsv4_score_sinks   = nullptr; size_t d_dsv4_score_sinks_cap   = 0;
    float* d_dsv4_score_context = nullptr; size_t d_dsv4_score_context_cap = 0;

    // Pinned host staging buffer for async H2D transfers
    float* h_pinned = nullptr;
    size_t h_pinned_cap = 0;
    size_t h_pinned_reservation_bytes = 0;

    // Single-pool arena for bulk dense-tensor offload (main.cpp's startup
    // loop over resident_layers) — one cudaMalloc, offset-based
    // suballocation for every tensor, instead of one cudaMalloc PER tensor
    // (~1182 separate allocations previously). Eliminates per-allocation
    // fragmentation the same way GpuTopology::allocate_expert_cache's slab
    // allocator already does for the expert VRAM cache; ~1182 tensors of
    // wildly varying sizes fragmenting a near-full device was the remaining
    // cause of dense tensors not fitting even after freeing dead allocations
    // elsewhere (see gpu_topology.hpp's removed d_hidden/d_scratch and
    // graph_initializer.cpp's removed GPU KV cache mirror).
    uint8_t* dense_pool = nullptr;
    size_t dense_pool_cap = 0;
    size_t dense_pool_used = 0;

    // Reports actual driver-reported free/used/total VRAM for every known
    // device (not MoEStream's own bookkeeping — this is ground truth from
    // cudaMemGetInfo, so it reflects everything on the GPU including
    // whatever else is running on the machine). Call at checkpoints around
    // allocation-heavy steps to see exactly where headroom runs out, rather
    // than inferring it from which individual tensors failed to offload.
    void report_vram_usage(const char* label) {
        for (auto& dev : topology.devices) {
            if (dev.role == GpuRole::Unused) continue;
            cudaSetDevice(dev.id);
            size_t free_b = 0, total_b = 0;
            if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) continue;
            const char* role_str = (dev.role == GpuRole::CoreEngine) ? "CoreEngine" : "ExpertVault";
            LOG_INFO("[VRAM] %-28s device=%d (%s, %s): used=%.2f GB free=%.2f GB total=%.2f GB\n",
                     label, dev.id, dev.name.c_str(), role_str,
                     (total_b - free_b) / 1e9, free_b / 1e9, total_b / 1e9);
        }
        cudaSetDevice(device_id);  // restore core device as current, matching pre-call state
    }

    // Categorized VRAM breakdown across every known device -- goes beyond
    // report_vram_usage's single used/free/total number by attributing
    // each device's usage to a specific subsystem: CoreEngine = regular
    // dense pool (dense_pool_used, real ground truth from
    // offload_tensor_to_pool, not the pre-allocation estimate) +
    // attn_output's BF16 fast-path buffers (passed in, since only the
    // caller in main.cpp tracks how many layers converted and how many
    // bytes that used); ExpertVault = MoE expert cache pool
    // (dev.expert_cache_size, already tracked by allocate_expert_cache).
    // Whatever's left after subtracting the known subsystems is logged as
    // "other" -- a REMAINDER against real cudaMemGetInfo ground truth
    // (CUDA context, cuBLAS workspace, pinned staging, small scratch
    // buffers like d_x/d_y/d_x_bf16), not a second independent estimate
    // that could silently drift from what's actually on the device.
    //
    // Added specifically because a real run made it hard to tell, from
    // report_vram_usage's one number alone, how much of the CoreEngine
    // GPU's usage the BF16 attn_output fast-path itself was responsible
    // for versus the (much larger) regular dense pool -- see CLAUDE.md's
    // "BF16 fast-path" section for the hand-derived math this log line is
    // meant to make visible directly instead.
    void report_vram_breakdown(const char* label,
                                size_t attn_output_fastpath_bytes,
                                int attn_output_layers_converted) {
        for (auto& dev : topology.devices) {
            if (dev.role == GpuRole::Unused) continue;
            cudaSetDevice(dev.id);
            size_t free_b = 0, total_b = 0;
            if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) continue;
            size_t used_b = total_b - free_b;

            if (dev.role == GpuRole::CoreEngine) {
                // dev.expert_cache_size is normally 0 here -- only nonzero
                // once add_dual_role_expert_cache (gpu_topology.hpp) has
                // given this device its own small MoE cache slice. Included
                // in `accounted` so that slice doesn't get misattributed to
                // "other" once it exists.
                size_t accounted = dense_pool_used + attn_output_fastpath_bytes + dev.expert_cache_size;
                size_t other_b = (used_b > accounted) ? (used_b - accounted) : 0;
                LOG_INFO("[VRAM-BREAKDOWN] %s device=%d (%s, CoreEngine):\n",
                         label, dev.id, dev.name.c_str());
                LOG_INFO("  dense pool (quantized, regular path): %.3f GB\n",
                         dense_pool_used / 1e9);
                LOG_INFO("  attn_output fast-path (BF16):         %.3f GB (%d layer(s))\n",
                         attn_output_fastpath_bytes / 1e9, attn_output_layers_converted);
                if (dev.expert_cache_size > 0) {
                    LOG_INFO("  MoE expert cache (dual-role slice):   %.3f GB\n",
                             dev.expert_cache_size / 1e9);
                }
                LOG_INFO("  other (CUDA ctx / cuBLAS / scratch):  %.3f GB\n", other_b / 1e9);
            } else {
                size_t moe_b = dev.expert_cache_size;
                size_t other_b = (used_b > moe_b) ? (used_b - moe_b) : 0;
                LOG_INFO("[VRAM-BREAKDOWN] %s device=%d (%s, ExpertVault):\n",
                         label, dev.id, dev.name.c_str());
                LOG_INFO("  MoE expert cache pool:                %.3f GB\n", moe_b / 1e9);
                LOG_INFO("  other (CUDA ctx / cuBLAS / scratch):  %.3f GB\n", other_b / 1e9);
            }
            LOG_INFO("  ---> used=%.2f GB free=%.2f GB total=%.2f GB\n",
                     used_b / 1e9, free_b / 1e9, total_b / 1e9);
        }
        cudaSetDevice(device_id);  // restore core device as current, matching pre-call state
    }

    // Free bytes on the core GPU right now (ground truth via cudaMemGetInfo,
    // not MoEStream's own bookkeeping). Used to size the dense-tensor pool
    // when the full requested size doesn't fit — see main.cpp's backoff.
    size_t core_free_vram_bytes() {
        cudaSetDevice(device_id);
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        return free_b;
    }

    bool initialize() {
        if (!topology.initialize()) return false;

        device_id = (topology.core_gpu >= 0) ? topology.devices[topology.core_gpu].id : 0;
        cudaSetDevice(device_id);

        // Enable P2P access between device pairs (for cross-GPU vector transfers)
        for (size_t i = 0; i < topology.devices.size(); ++i) {
            if (topology.devices[i].role == GpuRole::Unused) continue;
            for (size_t j = 0; j < topology.devices.size(); ++j) {
                if (i == j) continue;
                if (topology.devices[j].role == GpuRole::Unused) continue;
                int can_peer = 0;
                cudaDeviceCanAccessPeer(&can_peer, topology.devices[i].id, topology.devices[j].id);
                if (can_peer) {
                    cudaSetDevice(topology.devices[i].id);
                    cudaDeviceEnablePeerAccess(topology.devices[j].id, 0);
                    // Ignore errors (may already be enabled)
                }
            }
        }
        cudaSetDevice(device_id);

        // cuBLAS handle for unquantized dense matmul (F32/F16/BF16). Left on the
        // DEFAULT stream (not bound to compute_stream) so dense_matmul_host can
        // mix synchronous H2D/D2H memcpy + dequant kernel launches + cuBLAS on a
        // single stream without cross-stream ordering hazards. Async overlap is
        // a later optimization.
        cublasCreate(&cublas_handle);

        // Dedicated stream + handle for grouped_output_proj_host -- see
        // that field's own comment (above) for why this must NOT share
        // cublas_handle/stream 0. cublasSetStream binds this handle to the
        // new stream for the rest of the process's lifetime; every
        // grouped_output_proj_host call reuses it, never touching stream 0.
        cudaError_t stream_err = cudaStreamCreate(&outproj_stream);
        if (stream_err == cudaSuccess) {
            cublasCreate(&outproj_cublas_handle);
            cublasSetStream(outproj_cublas_handle, outproj_stream);
        } else {
            LOG_STREAM_ERROR() << "[CUDA] outproj_stream creation failed: "
                               << cudaGetErrorString(stream_err)
                               << " -- dsv4_outproj_gpu will be unavailable "
                                  "(grouped_output_proj_host checks for a null handle).\n";
        }

        // Allocate 4 MB pinned staging.  Pinned CUDA host memory is still
        // real DDR, so it must consume the same process-wide host budget as
        // RAM slabs and graph buffers.
        constexpr size_t kPinnedBytes = 4 << 20;
        const bool pinned_admitted =
            MemoryGuard::try_reserve_host_bytes(kPinnedBytes);
        if (pinned_admitted &&
            cudaMallocHost((void**)&h_pinned, kPinnedBytes) == cudaSuccess &&
            h_pinned) {
            h_pinned_cap = kPinnedBytes;
            h_pinned_reservation_bytes = kPinnedBytes;
        } else {
            cudaGetLastError();
            if (pinned_admitted)
                MemoryGuard::release_host_bytes(kPinnedBytes);
            h_pinned = nullptr;
            h_pinned_cap = 0;
        }

        return true;
    }

    ~CudaManager() {
        cudaSetDevice(device_id);
        for (int g = 0; g < kMaxAccumGpus; ++g) {
            if (d_acc[g]) cudaFree(d_acc[g]);
            if (d_acc_tok[g]) cudaFree(d_acc_tok[g]);
            if (d_acc_w[g]) cudaFree(d_acc_w[g]);
        }
        for (float** b : {&d_gdn_Q,&d_gdn_K,&d_gdn_V,&d_gdn_beta,&d_gdn_g,&d_gdn_out,
                          &d_gdn_state,&d_gdn_Qc,&d_gdn_Kc,&d_gdn_Vc,&d_gdn_Kcs,&d_gdn_KK,
                          &d_gdn_QK,&d_gdn_M,&d_gdn_P,&d_gdn_R,&d_gdn_Qs,&d_gdn_U0,&d_gdn_U,
                          &d_gdn_outc,&d_gdn_gcs,&d_gdn_At,&d_gdn_ecum,&d_gdn_cf,&d_gdn_betac})
            if (*b) cudaFree(*b);
        if (d_gdn_fast) cudaFree(d_gdn_fast);
        if (d_q4_Q) cudaFree(d_q4_Q);
        if (d_q4_qfull) cudaFree(d_q4_qfull);
        if (d_q4_gate) cudaFree(d_q4_gate);
        if (d_q4_K) cudaFree(d_q4_K);
        if (d_q4_V) cudaFree(d_q4_V);
        if (d_q4_qnorm) cudaFree(d_q4_qnorm);
        if (d_q4_knorm) cudaFree(d_q4_knorm);
        if (d_q4_scores) cudaFree(d_q4_scores);
        if (d_q4_out) cudaFree(d_q4_out);
        if (d_gdnconv_x) cudaFree(d_gdnconv_x);
        if (d_gdnconv_w) cudaFree(d_gdnconv_w);
        if (d_gdnconv_h) cudaFree(d_gdnconv_h);
        if (d_gdnconv_out) cudaFree(d_gdnconv_out);
        if (d_glm_kda_q) cudaFree(d_glm_kda_q);
        if (d_glm_kda_k) cudaFree(d_glm_kda_k);
        if (d_glm_kda_v) cudaFree(d_glm_kda_v);
        if (d_glm_kda_beta) cudaFree(d_glm_kda_beta);
        if (d_glm_kda_g) cudaFree(d_glm_kda_g);
        if (d_glm_kda_state) cudaFree(d_glm_kda_state);
        if (d_glm_kda_out) cudaFree(d_glm_kda_out);
        for (auto& state : d_glm_kda_state_by_layer) {
            if (state.data) cudaFree(state.data);
            state.data = nullptr;
            state.cap = 0;
            state.initialized = false;
        }
        if (d_glm_router_x) cudaFree(d_glm_router_x);
        if (d_glm_router_logits) cudaFree(d_glm_router_logits);
        if (d_glm_router_bias) cudaFree(d_glm_router_bias);
        if (d_glm_router_ids) cudaFree(d_glm_router_ids);
        if (d_glm_router_weights) cudaFree(d_glm_router_weights);
        if (cublas_handle) cublasDestroy(cublas_handle);
        if (outproj_cublas_handle) cublasDestroy(outproj_cublas_handle);
        if (outproj_stream) cudaStreamDestroy(outproj_stream);
        if (d_outproj_x) cudaFree(d_outproj_x);
        if (d_outproj_y) cudaFree(d_outproj_y);
        if (d_x) cudaFree(d_x);
        if (d_y) cudaFree(d_y);
        if (d_x_bf16) cudaFree(d_x_bf16);
        if (h_x_bf16_scratch) delete[] h_x_bf16_scratch;
        MemoryGuard::release_host_bytes(h_x_bf16_scratch_reservation_bytes);
        h_x_bf16_scratch_reservation_bytes = 0;
        for (void* p : wuk_bf16_by_layer) if (p) cudaFree(p);
        if (d_qabsorb_x_bf16) cudaFree(d_qabsorb_x_bf16);
        if (d_qabsorb_y) cudaFree(d_qabsorb_y);
        if (h_qabsorb_x_bf16_scratch) delete[] h_qabsorb_x_bf16_scratch;
        MemoryGuard::release_host_bytes(h_qabsorb_x_bf16_scratch_reservation_bytes);
        h_qabsorb_x_bf16_scratch_reservation_bytes = 0;
        for (void* p : wuv_bf16_by_layer) if (p) cudaFree(p);
        if (d_wuv_ctx_bf16) cudaFree(d_wuv_ctx_bf16);
        if (d_wuv_out) cudaFree(d_wuv_out);
        if (h_wuv_ctx_bf16_scratch) delete[] h_wuv_ctx_bf16_scratch;
        MemoryGuard::release_host_bytes(h_wuv_ctx_bf16_scratch_reservation_bytes);
        h_wuv_ctx_bf16_scratch_reservation_bytes = 0;
        for (int8_t* p : wuv_int8_by_layer) if (p) cudaFree(p);
        for (float* p : wuv_int8_scale_by_layer) if (p) cudaFree(p);
        if (d_wuv_dequant_bf16) cudaFree(d_wuv_dequant_bf16);
        for (float* p : d_lc_by_layer) if (p) cudaFree(p);
        for (float* p : d_rc_by_layer) if (p) cudaFree(p);
        if (d_score_q_absorbed) cudaFree(d_score_q_absorbed);
        if (d_score_q_rope) cudaFree(d_score_q_rope);
        if (d_score_nope_buf) cudaFree(d_score_nope_buf);
        if (d_score_rope_buf) cudaFree(d_score_rope_buf);
        if (d_score_combined) cudaFree(d_score_combined);
        if (d_score_context) cudaFree(d_score_context);
        for (float* p : d_dsv4_kvc_by_layer) if (p) cudaFree(p);
        for (float* p : d_dsv4_hca_by_layer) if (p) cudaFree(p);
        if (d_dsv4_score_q) cudaFree(d_dsv4_score_q);
        if (d_dsv4_score_kv) cudaFree(d_dsv4_score_kv);
        if (d_dsv4_score_scores) cudaFree(d_dsv4_score_scores);
        if (d_dsv4_score_sinks) cudaFree(d_dsv4_score_sinks);
        if (d_dsv4_score_context) cudaFree(d_dsv4_score_context);
        if (h_pinned) cudaFreeHost(h_pinned);
        MemoryGuard::release_host_bytes(h_pinned_reservation_bytes);
        h_pinned_reservation_bytes = 0;
        if (dense_stage) cudaFreeHost(dense_stage);
        MemoryGuard::release_host_bytes(dense_stage_reservation_bytes);
        dense_stage = nullptr;
        dense_stage_cap = 0;
        dense_stage_reservation_bytes = 0;
        if (dense_pool) cudaFree(dense_pool);
    }

    // Allocate the single dense-tensor pool. Call once, sized to the sum of
    // every tensor's aligned byte requirement, BEFORE any
    // offload_tensor_to_pool() calls. Returns false if even this one
    // allocation doesn't fit (caller should re-size and retry, or fall back
    // to per-tensor offload_tensor_to_vram for graceful partial degradation).
    bool init_dense_pool(size_t total_bytes) {
        if (total_bytes == 0) return true;  // nothing to allocate, trivially fine
        cudaSetDevice(device_id);
        cudaError_t err = cudaMalloc((void**)&dense_pool, total_bytes);
        if (err != cudaSuccess) {
            LOG_ERROR("[CUDA] Dense tensor pool allocation failed: %s (requested %.2f GB)\n",
                      cudaGetErrorString(err), total_bytes / 1e9);
            dense_pool = nullptr;
            dense_pool_cap = 0;
            return false;
        }
        dense_pool_cap = total_bytes;
        dense_pool_used = 0;
        return true;
    }

    // Suballocate qt's bytes from the pool (256-byte aligned — matches
    // cudaMalloc's own natural alignment guarantee, small enough to waste
    // negligible space across ~1000 tensors, large enough for coalesced
    // access) and H2D-copy qt.host_ptr into it. False if the pool doesn't
    // have room left — since the pool is normally sized to fit everything
    // requested, this should only trigger if init_dense_pool was sized to a
    // deliberately-reduced budget (see main.cpp's free-VRAM-based backoff).
    bool offload_tensor_to_pool(QuantizedTensor& qt) {
        if (qt.device_ptr) return true;  // already offloaded
        if (!qt.host_ptr || !dense_pool) return false;

        constexpr size_t ALIGN = 256;
        size_t aligned_used = (dense_pool_used + ALIGN - 1) & ~(ALIGN - 1);
        if (aligned_used + qt.meta.size_bytes > dense_pool_cap) return false;

        cudaSetDevice(device_id);
        qt.device_ptr = dense_pool + aligned_used;
        PERF_START(upload);
        cudaMemcpy(qt.device_ptr, qt.host_ptr, qt.meta.size_bytes, cudaMemcpyHostToDevice);
        PERF_END(upload, t_gpu_upload);
        perf().bytes_gpu_uploaded.fetch_add(qt.meta.size_bytes, std::memory_order_relaxed);

        dense_pool_used = aligned_used + qt.meta.size_bytes;
        return true;
    }

    // Offload a dense tensor permanently to the core engine GPU's VRAM.
    // Returns true if it actually ended up device-resident.
    bool offload_tensor_to_vram(QuantizedTensor& qt) {
        if (qt.device_ptr) return true;  // already offloaded
        if (!qt.host_ptr) return false;

        cudaSetDevice(device_id);
        PERF_START(upload);
        cudaError_t err = cudaMalloc((void**)&qt.device_ptr, qt.meta.size_bytes);
        if (err != cudaSuccess) {
            // No budget check exists anywhere in this path — it just tries
            // to malloc every matched tensor until the GPU physically runs
            // out of room, previously with zero error checking, so a
            // tensor that didn't fit silently stayed host-only forever
            // (indistinguishable from "never attempted" in dense_proj's
            // fallback log). Loud now so this is diagnosable: check total
            // resident dense-tensor size against the core GPU's actual free
            // VRAM (minus KV cache / driver overhead) if you see this.
            LOG_ERROR("[CUDA] Failed to offload %.2f MB to VRAM (device %d): %s — "
                      "staying host-resident (CPU dequant-matmul fallback)\n",
                      qt.meta.size_bytes / 1e6, device_id, cudaGetErrorString(err));
            qt.device_ptr = nullptr;
            return false;
        }
        cudaMemcpy(qt.device_ptr, qt.host_ptr, qt.meta.size_bytes, cudaMemcpyHostToDevice);
        PERF_END(upload, t_gpu_upload);
        perf().bytes_gpu_uploaded.fetch_add(qt.meta.size_bytes, std::memory_order_relaxed);
        return true;
    }

    // Dequantize a static per-layer dense weight to BF16 ONCE (host-side, via
    // the same ggml reference dequant_row used by the CPU fallback, then
    // rounded to bf16 via fp32_to_bf16) and upload the result to its OWN
    // standalone cudaMalloc buffer (deliberately NOT a dense_pool
    // suballocation, so this can be called independent of pool sizing/
    // lifetime) — qt.gpu_fast_ptr. dense_proj then dispatches through
    // dense_matmul_bf16_matched_host (cuBLAS, matched BF16 weight/BF16
    // activation) instead of qt's original-quant-type fused kernel
    // (Q2_K..Q6_K/IQ2*/IQ3_XXS), which has no GEMV specialization and is
    // disproportionately slow at S=1 decode — see attn_output's ~623ms/
    // token cost in CLAUDE.md, the motivating case. host_ptr/device_ptr/
    // meta.type are untouched, so the CPU fallback stays correct regardless.
    //
    // RE-ADDED AFTER TWO PRIOR ROUNDS (see CLAUDE.md for the full history):
    // (1) an earlier F32 version of this fast-path worked but only covered
    // ~19% of layers within a real VRAM budget; (2) a bf16 variant using a
    // MISMATCHED-type cublasGemmEx call (BF16 weight, F32 activation) was
    // tried, produced real NaN corruption on real hardware, and was fully
    // reverted -- root-caused MUCH later (via real perf/nsys profiling and
    // a separate, unrelated concurrency-race fix) to NOT actually be
    // caused by BF16 itself; a dedicated isolated test (Test 9,
    // main_test.cpp) subsequently confirmed on real hardware that the
    // mismatched-type combination is cleanly REJECTED by cublasGemmEx
    // (CUBLAS_STATUS_NOT_SUPPORTED), not silently wrong -- so it was never
    // going to work at all, mismatch or not. This version uses the
    // MATCHED-type combination instead (BF16 weight AND BF16 activation),
    // verified correct in isolation first via Test 10 (real hardware,
    // max_rel_diff ~1.18e-5) before being wired in here.
    //
    // Best-effort: returns false (leaving gpu_fast_ptr null, caller keeps
    // using the existing slower path) if the bf16 expansion doesn't fit in
    // max_bytes, the shape is malformed, or the quant type has no CPU
    // reference dequant. bf16 is ~2-3x the size of a K-quant/IQ original
    // (e.g. ~2.9x for Q5_K, vs ~5.8x it would be for F32), so this is still
    // capacity-checked per call rather than assumed to fit — callers should
    // pass real remaining VRAM headroom, not the tensor's own original size.
    bool upload_dequantized_bf16(QuantizedTensor& qt, size_t max_bytes) {
        if (qt.gpu_fast_ptr) return true;  // already converted
        // HARD TYPE GATE: only genuinely lossless-relative-to-storage F32
        // source data may be rounded to BF16 here. Every call site (gate_inp,
        // attn_output) matches by tensor NAME only, never by qt.meta.type --
        // without this gate, any quantized tensor matching those name
        // patterns (Q5_K/Q6_K/Q8_0/IQ2_*/IQ3_XXS, whatever this checkpoint
        // happens to use) would silently get dequantized to F32 then rounded
        // to BF16 here, discarding that quant format's own encoding entirely
        // -- a real precision/behavior change the user explicitly does NOT
        // want for anything already quantized. F32 is the only case where
        // "round to BF16" is the ONLY loss involved (halves storage, no
        // format switch); for every other type, the CPU-fallback dequant
        // reference distinguishes cleanly between quant formats and this
        // fast-path must not collapse that distinction. Gated here (the
        // single choke point both call sites route through) rather than at
        // each call site, so no future caller can bypass it either.
        if (qt.meta.type != GGUFType::F32) return false;
        if (!qt.host_ptr || qt.meta.shape.size() < 2) return false;
        int O = (int)qt.meta.shape[0];
        int I = (int)qt.meta.shape[1];
        if (O <= 0 || I <= 0) return false;
        size_t n = (size_t)O * (size_t)I;
        size_t bf16_bytes = n * sizeof(uint16_t);
        if (bf16_bytes > max_bytes) return false;

        std::vector<float> host_f32(n);
        if (!dequant_tensor_f32(qt.meta.type, qt.host_ptr, O, I, host_f32.data())) {
            LOG_WARN("[CUDA] Fast-path BF16 dequant unsupported for type=%u — "
                     "keeping original quantized GPU path\n", (unsigned)qt.meta.type);
            return false;
        }
        std::vector<uint16_t> host_bf16(n);
        for (size_t i = 0; i < n; ++i) host_bf16[i] = fp32_to_bf16(host_f32[i]);

        cudaSetDevice(device_id);
        cudaError_t err = cudaMalloc((void**)&qt.gpu_fast_ptr, bf16_bytes);
        if (err != cudaSuccess) {
            LOG_ERROR("[CUDA] Fast-path BF16 upload failed (%.2f MB): %s — "
                      "keeping original quantized GPU path\n",
                      bf16_bytes / 1e6, cudaGetErrorString(err));
            qt.gpu_fast_ptr = nullptr;
            return false;
        }
        cudaMemcpy(qt.gpu_fast_ptr, host_bf16.data(), bf16_bytes, cudaMemcpyHostToDevice);
        perf().bytes_gpu_uploaded.fetch_add(bf16_bytes, std::memory_order_relaxed);
        return true;
    }

    // Per-call dispatch for a BF16-fast-path-converted weight (qt.gpu_fast_ptr,
    // set by upload_dequantized_bf16 above). Unlike the weight (converted
    // ONCE at load and cached for the process lifetime), the activation x
    // is different every call, so it must be converted to bf16 fresh each
    // time -- real, new per-call work that the earlier F32 fast-path never
    // had (F32 activations needed no conversion, just upload). Host-side
    // conversion (reusing fp32_to_bf16, the same function Test 10 verified
    // in isolation) keeps this simple: no new CUDA kernel, just a smaller
    // H2D upload than the F32 path would have needed.
    bool dense_matmul_bf16_matched_host(const void* d_W_bf16,
                                         const float* x_host, float* y_host,
                                         int S, int I, int O) {
        if (!d_W_bf16) return false;
        size_t n_x = (size_t)S * I;
        size_t x_bf16_bytes = n_x * sizeof(uint16_t);
        size_t y_bytes = (size_t)S * O * sizeof(float);

        cudaSetDevice(device_id);
        if (x_bf16_bytes > d_x_bf16_cap) {
            if (d_x_bf16) { cudaFree(d_x_bf16); d_x_bf16 = nullptr; }
            d_x_bf16_cap = 0;
            cudaError_t err = cudaMalloc(&d_x_bf16, x_bf16_bytes);
            if (err != cudaSuccess) {
                LOG_ERROR("[CUDA] dense_matmul_bf16_matched_host: d_x_bf16 grow to %.2f MB failed: %s\n",
                          x_bf16_bytes / 1e6, cudaGetErrorString(err));
                return false;
            }
            d_x_bf16_cap = x_bf16_bytes;
        }
        if (!ensure_buffers(0, y_bytes)) return false;  // reuses/grows d_y (F32 output buffer)

        // Grow-only reuse of h_x_bf16_scratch (see member comment) instead
        // of a fresh std::vector every call -- eliminates the heap
        // alloc/free churn that was the real source of elevated per-token
        // CPU usage observed on real hardware after this fast-path landed.
        if (!ensure_host_scratch(h_x_bf16_scratch,
                                 h_x_bf16_scratch_cap,
                                 h_x_bf16_scratch_reservation_bytes, n_x,
                                 "dense BF16 activation"))
            return false;
        // Elementwise and independent, so parallelising is bit-identical.
        // This ran SERIALLY and, worse, outside every PERF bracket, so it was
        // invisible: at qwen4exp's prefill shape it converts ~1.13M floats per
        // call across ~370 dense calls per prefill. Timed now (into the same
        // bucket as the upload it feeds) so it can never hide again.
        // TRIED 2026-09-05 and REVERTED: an `#pragma omp parallel for` here cut
        // attn 2.67s -> 2.39s but blew the expert bucket up 2.71s -> 4.62s
        // (its staging memcpy 1.02s -> 2.17s), for a net 6.14s -> 7.73s. This
        // function is called ~370 times per prefill, so it adds ~370 OMP team
        // spawns interleaved with the one big staging region that actually
        // needs the cores -- the small region wins locally and loses globally.
        // Same class as this project's other OMP-spawn findings; kept serial.
        PERF_START(dense_bf16_cvt);
        for (size_t i = 0; i < n_x; ++i)
            h_x_bf16_scratch[i] = fp32_to_bf16(x_host[i]);
        PERF_END(dense_bf16_cvt, t_dense_gpu_up);

        PERF_START(dense_bf16_up);
        cudaMemcpy(d_x_bf16, h_x_bf16_scratch, x_bf16_bytes, cudaMemcpyHostToDevice);
        PERF_END(dense_bf16_up, t_dense_gpu_up);

        PERF_START(dense_bf16_cmp);
        cublasStatus_t status = cublas_dense_matmul_bf16_matched(
            cublas_handle, d_W_bf16, d_x_bf16, d_y, S, I, O);
        PERF_END(dense_bf16_cmp, t_dense_gpu_compute);
        if (status != CUBLAS_STATUS_SUCCESS) return false;

        PERF_START(dense_bf16_dn);
        cudaMemcpy(y_host, d_y, y_bytes, cudaMemcpyDeviceToHost);
        PERF_END(dense_bf16_dn, t_dense_gpu_down);
        return true;
    }

    // MLA qabsorb BF16 fast-path (EngineConfig::attn_qabsorb_gpu), part 1:
    // one-time per-layer weight upload. WUK_host is wuk_cache[layer_idx]
    // (attention_layer.cpp), already dequantized to F32 host memory by
    // ensure_wuk_wuv_cached -- laid out [H,kvl,nope] (H heads, each an
    // independent [kvl,nope] row-major weight, contiguous per-head, matching
    // the CPU matmul() call this replaces: matmul(..., WUK.data()+h*nope*kvl,
    // S, nope, kvl) => per-head W is [O=kvl,I=nope] row-major). Rounds the
    // WHOLE [H,kvl,nope] block to BF16 in one pass and uploads it as one
    // cudaMalloc/cudaMemcpy -- callers already loop layers themselves
    // (warm_mla_absorb_cache), so no separate per-layer capacity-tracking
    // loop is needed here beyond the max_bytes check every fast-path in this
    // file already does.
    //
    // Best-effort, matching upload_dequantized_bf16's own contract: returns
    // false (leaving wuk_bf16_by_layer[layer_idx] null) on any failure --
    // wrong shape, doesn't fit in max_bytes, or the cudaMalloc itself fails
    // -- caller (warm_mla_absorb_cache) just logs and continues; the real
    // attention call falls back to the CPU path for that layer, same
    // discipline as every other best-effort GPU fast-path here.
    bool upload_wuk_bf16(int layer_idx, const std::vector<float>& WUK_host,
                          int H, int nope, int kvl, size_t max_bytes) {
        if (layer_idx < 0) return false;
        size_t n = (size_t)H * (size_t)nope * (size_t)kvl;
        if (n == 0 || WUK_host.size() != n) return false;
        size_t bf16_bytes = n * sizeof(uint16_t);
        if (bf16_bytes > max_bytes) return false;

        if ((int)wuk_bf16_by_layer.size() <= layer_idx) {
            wuk_bf16_by_layer.resize(layer_idx + 1, nullptr);
        }
        if (wuk_bf16_by_layer[layer_idx]) return true;  // already uploaded

        std::vector<uint16_t> host_bf16(n);
        for (size_t i = 0; i < n; ++i) host_bf16[i] = fp32_to_bf16(WUK_host[i]);

        cudaSetDevice(device_id);
        void* d_wuk = nullptr;
        cudaError_t err = cudaMalloc(&d_wuk, bf16_bytes);
        if (err != cudaSuccess) {
            LOG_WARN("[CUDA] qabsorb WUK BF16 upload failed layer=%d (%.2f MB): %s -- "
                     "layer stays on the CPU qabsorb path\n",
                     layer_idx, bf16_bytes / 1e6, cudaGetErrorString(err));
            return false;
        }
        cudaMemcpy(d_wuk, host_bf16.data(), bf16_bytes, cudaMemcpyHostToDevice);
        wuk_bf16_by_layer[layer_idx] = d_wuk;
        perf().bytes_gpu_uploaded.fetch_add(bf16_bytes, std::memory_order_relaxed);
        return true;
    }

    // MLA qabsorb BF16 fast-path, part 2: per-call batched dispatch,
    // Q_absorbed[H,S,kvl] = q_nope_gather[H,S,nope] @ WUK[H,kvl,nope]^T (per
    // head), replacing attention_layer.cpp's #pragma omp parallel for over
    // heads calling the CPU matmul() once per head. ONE H2D upload, ONE
    // cublasGemmStridedBatchedEx call (batches all H heads on-device instead
    // of H separate host-issued cublas calls), ONE D2H download -- avoiding
    // the "per-head loop pays per-head overhead" mistake this codebase's own
    // CPU qabsorb history already hit once (see attention_layer.cpp's
    // qabsorb comment on the OMP-spawn-per-head regression), just for GPU
    // launch/transfer overhead instead of OMP team spawns.
    //
    // x_host: [H,S,nope] row-major F32, host (caller gathers this from Q's
    //   real [S,H,qh] layout -- see mla_attention_absorbed_gguf's call site).
    // y_host: [H,S,kvl] row-major F32, host -- SAME layout Q_absorbed already
    //   uses (Q_absorbed.data() + h*S*kvl), so the caller can write straight
    //   into the existing buffer with no reshuffle.
    // Returns false (caller falls back to the CPU per-head loop) if this
    // layer's WUK was never uploaded (upload_wuk_bf16 never called, or it
    // failed) or any CUDA/cublas call fails -- never a hard requirement.
    bool qabsorb_bf16_batched_host(int layer_idx, const float* x_host, float* y_host,
                                    int H, int S, int nope, int kvl) {
        if (layer_idx < 0 || (size_t)layer_idx >= wuk_bf16_by_layer.size()) return false;
        void* d_wuk = wuk_bf16_by_layer[layer_idx];
        if (!d_wuk || H <= 0 || S <= 0 || nope <= 0 || kvl <= 0) return false;

        size_t n_x = (size_t)H * (size_t)S * (size_t)nope;
        size_t n_y = (size_t)H * (size_t)S * (size_t)kvl;
        size_t x_bf16_bytes = n_x * sizeof(uint16_t);
        size_t y_bytes = n_y * sizeof(float);

        cudaSetDevice(device_id);
        if (x_bf16_bytes > d_qabsorb_x_bf16_cap) {
            if (d_qabsorb_x_bf16) { cudaFree(d_qabsorb_x_bf16); d_qabsorb_x_bf16 = nullptr; }
            d_qabsorb_x_bf16_cap = 0;
            if (cudaMalloc(&d_qabsorb_x_bf16, x_bf16_bytes) != cudaSuccess) {
                LOG_ERROR("[CUDA] qabsorb_bf16_batched_host: d_qabsorb_x_bf16 grow to %.2f MB failed\n",
                          x_bf16_bytes / 1e6);
                return false;
            }
            d_qabsorb_x_bf16_cap = x_bf16_bytes;
        }
        if (y_bytes > d_qabsorb_y_cap) {
            if (d_qabsorb_y) { cudaFree(d_qabsorb_y); d_qabsorb_y = nullptr; }
            d_qabsorb_y_cap = 0;
            if (cudaMalloc((void**)&d_qabsorb_y, y_bytes) != cudaSuccess) {
                LOG_ERROR("[CUDA] qabsorb_bf16_batched_host: d_qabsorb_y grow to %.2f MB failed\n",
                          y_bytes / 1e6);
                return false;
            }
            d_qabsorb_y_cap = y_bytes;
        }
        if (!ensure_host_scratch(h_qabsorb_x_bf16_scratch,
                                 h_qabsorb_x_bf16_scratch_cap,
                                 h_qabsorb_x_bf16_scratch_reservation_bytes,
                                 n_x, "MLA qabsorb BF16 activation"))
            return false;
        for (size_t i = 0; i < n_x; ++i) h_qabsorb_x_bf16_scratch[i] = fp32_to_bf16(x_host[i]);

        PERF_START(qabsorb_bf16_up);
        cudaMemcpy(d_qabsorb_x_bf16, h_qabsorb_x_bf16_scratch, x_bf16_bytes, cudaMemcpyHostToDevice);
        PERF_END(qabsorb_bf16_up, t_qabsorb_gpu_up);

        PERF_START(qabsorb_bf16_cmp);
        cublasStatus_t status = cublas_dense_matmul_bf16_matched_batched(
            cublas_handle,
            d_wuk, (long long)nope * kvl,
            d_qabsorb_x_bf16, (long long)S * nope,
            d_qabsorb_y, (long long)S * kvl,
            S, nope, kvl, H);
        PERF_END(qabsorb_bf16_cmp, t_qabsorb_gpu_compute);
        if (status != CUBLAS_STATUS_SUCCESS) return false;

        PERF_START(qabsorb_bf16_dn);
        cudaMemcpy(y_host, d_qabsorb_y, y_bytes, cudaMemcpyDeviceToHost);
        PERF_END(qabsorb_bf16_dn, t_qabsorb_gpu_down);
        perf().n_qabsorb_gpu_calls.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    // WUV BF16 fast-path, part 1: one-time per-layer upload -- byte-for-
    // byte the same shape/logic as upload_wuk_bf16 above, just a different
    // static weight (WUV/attn_v_b instead of WUK/attn_k_b). Per-head WUV is
    // [vh,kvl] row-major (matches the CPU matmul() call it replaces:
    // matmul(..., WUV.data()+h*kvl*vh, 1, kvl, vh) => per-head W is
    // [O=vh,I=kvl]). Best-effort: false leaves wuv_bf16_by_layer[layer_idx]
    // null, caller falls back to the CPU per-head loop for that layer.
    bool upload_wuv_bf16(int layer_idx, const std::vector<float>& WUV_host,
                          int H, int kvl, int vh, size_t max_bytes) {
        if (layer_idx < 0) return false;
        size_t n = (size_t)H * (size_t)kvl * (size_t)vh;
        if (n == 0 || WUV_host.size() != n) return false;
        size_t bf16_bytes = n * sizeof(uint16_t);
        if (bf16_bytes > max_bytes) return false;

        if ((int)wuv_bf16_by_layer.size() <= layer_idx) {
            wuv_bf16_by_layer.resize(layer_idx + 1, nullptr);
        }
        if (wuv_bf16_by_layer[layer_idx]) return true;  // already uploaded

        std::vector<uint16_t> host_bf16(n);
        for (size_t i = 0; i < n; ++i) host_bf16[i] = fp32_to_bf16(WUV_host[i]);

        cudaSetDevice(device_id);
        void* d_wuv = nullptr;
        cudaError_t err = cudaMalloc(&d_wuv, bf16_bytes);
        if (err != cudaSuccess) {
            LOG_WARN("[CUDA] WUV BF16 upload failed layer=%d (%.2f MB): %s -- "
                     "layer stays on the CPU WUV path\n",
                     layer_idx, bf16_bytes / 1e6, cudaGetErrorString(err));
            return false;
        }
        cudaMemcpy(d_wuv, host_bf16.data(), bf16_bytes, cudaMemcpyHostToDevice);
        wuv_bf16_by_layer[layer_idx] = d_wuv;
        perf().bytes_gpu_uploaded.fetch_add(bf16_bytes, std::memory_order_relaxed);
        return true;
    }

    // WUV BF16 fast-path, part 2: per-call batched dispatch -- byte-for-byte
    // the same shape as qabsorb_bf16_batched_host above (same strided-
    // batched cuBLAS call, same grow-only scratch pattern), just renamed
    // roles: x_host=[H,kvl] is score's own context output (the score GPU
    // path's `context_host`, S folded in since S==1 always here), y_host=
    // [H,vh] is head_contexts. Replaces attention_layer.cpp's CPU per-head
    // OMP-parallel matmul() loop, which real production data showed costing
    // 0.916ms/call -- OMP team-spawn overhead exceeding an ~8.4M-FLOP
    // per-call workload, the same pattern qabsorb's own gather fix already
    // diagnosed and fixed at a different call site. Best-effort: false
    // (this layer's WUV was never uploaded, or a CUDA/cublas call failed)
    // falls back to the existing CPU loop, unchanged.
    // 2026-09-01: S added so prefill can batch this the same way the score
    // path now does. x_host is [H,S,kvl] (h-major, matching score_gpu_batched_
    // host's own output); y_host is [S,H,vh] (s-major, matching
    // attention_layer.cpp's head_contexts, which indexes (s*H+h)*vh).
    //
    // Those two layouts disagree, and the transpose is free: passing
    // ldc = H*vh with strideC = vh puts batch h's column s at
    // h*vh + s*H*vh == (s*H+h)*vh, i.e. the GEMM scatters straight into the
    // caller's buffer. At S==1 there is only one column, so ldc is unused and
    // strideC=vh reproduces the pre-existing packed [H,vh] output exactly --
    // decode is byte-for-byte unaffected.
    bool wuv_bf16_batched_host(int layer_idx, const float* x_host, float* y_host,
                                int H, int kvl, int vh, int S = 1) {
        if (layer_idx < 0 || (size_t)layer_idx >= wuv_bf16_by_layer.size()) return false;
        void* d_wuv = wuv_bf16_by_layer[layer_idx];
        if (!d_wuv || H <= 0 || kvl <= 0 || vh <= 0 || S <= 0) return false;

        size_t n_x = (size_t)H * (size_t)S * (size_t)kvl;
        size_t n_y = (size_t)S * (size_t)H * (size_t)vh;
        size_t x_bf16_bytes = n_x * sizeof(uint16_t);
        size_t y_bytes = n_y * sizeof(float);

        cudaSetDevice(device_id);
        if (x_bf16_bytes > d_wuv_ctx_bf16_cap) {
            if (d_wuv_ctx_bf16) { cudaFree(d_wuv_ctx_bf16); d_wuv_ctx_bf16 = nullptr; }
            d_wuv_ctx_bf16_cap = 0;
            if (cudaMalloc(&d_wuv_ctx_bf16, x_bf16_bytes) != cudaSuccess) {
                LOG_ERROR("[CUDA] wuv_bf16_batched_host: d_wuv_ctx_bf16 grow to %.2f MB failed\n",
                          x_bf16_bytes / 1e6);
                return false;
            }
            d_wuv_ctx_bf16_cap = x_bf16_bytes;
        }
        if (y_bytes > d_wuv_out_cap) {
            if (d_wuv_out) { cudaFree(d_wuv_out); d_wuv_out = nullptr; }
            d_wuv_out_cap = 0;
            if (cudaMalloc((void**)&d_wuv_out, y_bytes) != cudaSuccess) {
                LOG_ERROR("[CUDA] wuv_bf16_batched_host: d_wuv_out grow to %.2f MB failed\n",
                          y_bytes / 1e6);
                return false;
            }
            d_wuv_out_cap = y_bytes;
        }
        if (!ensure_host_scratch(h_wuv_ctx_bf16_scratch,
                                 h_wuv_ctx_bf16_scratch_cap,
                                 h_wuv_ctx_bf16_scratch_reservation_bytes,
                                 n_x, "MLA WUV BF16 activation"))
            return false;
        for (size_t i = 0; i < n_x; ++i) h_wuv_ctx_bf16_scratch[i] = fp32_to_bf16(x_host[i]);

        PERF_START(wuv_bf16_up);
        cudaMemcpy(d_wuv_ctx_bf16, h_wuv_ctx_bf16_scratch, x_bf16_bytes, cudaMemcpyHostToDevice);
        PERF_END(wuv_bf16_up, t_wuv_gpu_up);

        PERF_START(wuv_bf16_cmp);
        cublasStatus_t status = cublas_dense_matmul_bf16_matched_batched(
            cublas_handle,
            d_wuv, (long long)kvl * vh,
            d_wuv_ctx_bf16, (long long)S * kvl,
            d_wuv_out, (long long)vh,
            S, kvl, vh, H, /*ldc=*/H * vh);
        PERF_END(wuv_bf16_cmp, t_wuv_gpu_compute);
        if (status != CUBLAS_STATUS_SUCCESS) return false;

        PERF_START(wuv_bf16_dn);
        cudaMemcpy(y_host, d_wuv_out, y_bytes, cudaMemcpyDeviceToHost);
        PERF_END(wuv_bf16_dn, t_wuv_gpu_down);
        perf().n_wuv_gpu_calls.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    // int8-persistent WUV, part 1: one-time per-layer upload -- same role
    // as upload_wuv_bf16 above (2026-08-29), but roughly halves the
    // persistent per-layer footprint (int8 weight + a small per-row scale,
    // vs. BF16's flat 2 bytes/element) specifically to buy real coverage
    // within GPU1's already-exhausted attn_score_gpu_max_vram_gb budget --
    // see CLAUDE.md's WUV-coverage entries for the real numbers that
    // motivated this (only 21/78 layers covered at BF16). Per-ROW
    // (per (head,output-row) pair) symmetric quantization -- matches this
    // project's own "prefer finer-grained scales over one tensor-wide
    // value" convention used by every other lossy quant path already in
    // this codebase (Q6_K/Q8_0's own per-block scales, etc.), not just a
    // single scale for the whole layer or even the whole head.
    //
    // WUV_host layout: same as upload_wuv_bf16's own (per-head [vh,kvl]
    // row-major, flattened as h*vh*kvl + o*kvl + i) -- unchanged, so this
    // function is a drop-in replacement at the call site, not a new
    // layout contract.
    //
    // STATUS (2026-09-02): reachable in production ONLY via the opt-in
    // MOESTREAM_WUV_INT8=1 environment toggle -- see that toggle's own full
    // write-up in attention_layer.cpp (wuv_int8_enabled) for the real
    // hardware A/B numbers and the reasoning behind leaving it OFF. Short
    // version: it works and is correct (0 errors across 4 real runs; the
    // 2026-08-29 NaN cascade that shelved it was the slab_capacity=0 bug,
    // fixed 2026-09-01, not this path), it nearly doubles WUV GPU coverage
    // (18/78 -> 35/78) and cuts the score= bucket 8.2%, but whole-request
    // throughput is FLAT because score= is only ~4.7% of a request dominated
    // by expert streaming. Worth enabling when VRAM is the binding
    // constraint; not worth int8's accuracy cost otherwise.
    bool upload_wuv_int8(int layer_idx, const std::vector<float>& WUV_host,
                          int H, int kvl, int vh, size_t max_bytes) {
        if (layer_idx < 0) return false;
        size_t n = (size_t)H * (size_t)kvl * (size_t)vh;
        if (n == 0 || WUV_host.size() != n) return false;
        size_t n_rows = (size_t)H * (size_t)vh;
        size_t int8_bytes = n;                        // 1 byte/element
        size_t scale_bytes = n_rows * sizeof(float);   // H*vh scales/layer
        if (int8_bytes + scale_bytes > max_bytes) return false;

        if ((int)wuv_int8_by_layer.size() <= layer_idx) {
            wuv_int8_by_layer.resize(layer_idx + 1, nullptr);
            wuv_int8_scale_by_layer.resize(layer_idx + 1, nullptr);
        }
        if (wuv_int8_by_layer[layer_idx]) return true;  // already uploaded

        std::vector<int8_t> host_q(n);
        std::vector<float> host_scale(n_rows);
        for (size_t row = 0; row < n_rows; ++row) {
            const float* w_row = WUV_host.data() + row * (size_t)kvl;
            float max_abs = 0.0f;
            for (int i = 0; i < kvl; ++i) max_abs = std::max(max_abs, std::fabs(w_row[i]));
            float scale = max_abs / 127.0f;
            host_scale[row] = scale;
            int8_t* q_row = host_q.data() + row * (size_t)kvl;
            if (scale > 0.0f) {
                float inv_scale = 1.0f / scale;
                for (int i = 0; i < kvl; ++i) {
                    float qf = std::round(w_row[i] * inv_scale);
                    qf = std::max(-127.0f, std::min(127.0f, qf));
                    q_row[i] = (int8_t)qf;
                }
            } else {
                // Every element in this row is exactly 0 -- q=0 regardless
                // of scale (avoids a 0/0 division above).
                std::memset(q_row, 0, (size_t)kvl);
            }
        }

        cudaSetDevice(device_id);
        int8_t* d_q = nullptr;
        float* d_scale = nullptr;
        if (cudaMalloc(&d_q, int8_bytes) != cudaSuccess) {
            LOG_WARN("[CUDA] WUV int8 weight upload failed layer=%d (%.2f MB) -- "
                     "layer stays on the CPU WUV path\n", layer_idx, int8_bytes / 1e6);
            return false;
        }
        if (cudaMalloc(&d_scale, scale_bytes) != cudaSuccess) {
            LOG_WARN("[CUDA] WUV int8 scale upload failed layer=%d (%.2f MB) -- "
                     "layer stays on the CPU WUV path\n", layer_idx, scale_bytes / 1e6);
            cudaFree(d_q);
            return false;
        }
        cudaMemcpy(d_q, host_q.data(), int8_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_scale, host_scale.data(), scale_bytes, cudaMemcpyHostToDevice);
        wuv_int8_by_layer[layer_idx] = d_q;
        wuv_int8_scale_by_layer[layer_idx] = d_scale;
        perf().bytes_gpu_uploaded.fetch_add(int8_bytes + scale_bytes, std::memory_order_relaxed);
        perf().n_wuv_int8_layers.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    // int8-persistent WUV, part 2: per-call batched dispatch -- same role
    // and same strided-batched cuBLAS call as wuv_bf16_batched_host above,
    // with ONE extra step first: reconstruct a transient BF16 buffer from
    // this layer's persistent int8+scale storage (launch_dequant_wuv_int8,
    // cuda/wuv_int8_dequant.cu), then hand that buffer to the EXACT SAME,
    // unmodified cublas_dense_matmul_bf16_matched_batched call the BF16
    // path already used and this project already proved correct -- the
    // only new correctness surface here is the dequant kernel itself, not
    // a new GEMM precision path. d_wuv_dequant_bf16 is shared, grow-only
    // scratch (like d_wuv_ctx_bf16/d_wuv_out), reused fresh every call
    // regardless of which layer -- there's no cross-call reuse to exploit
    // since decode visits a different layer on every call.
    //
    // MEASURED 2026-09-02, because the per-call re-expansion looks expensive
    // enough that it is worth not guessing: the dequant kernel costs 0.015s
    // total, 0.011ms/call over 1435 calls at real GLM-5.3 dims -- essentially
    // free, and NOT the reason int8's bucket-level win fails to reach wall
    // time. perf_timer.hpp prints it as the "int8-persistent WUV variant
    // active" line whenever t_wuv_int8_dequant is nonzero. Do not re-derive
    // the "the dequant must be eating it" argument; it has been measured.
    //
    // Opt-in only (MOESTREAM_WUV_INT8=1) -- see wuv_int8_enabled() in
    // attention_layer.cpp for the full A/B and the OFF-by-default rationale.
    bool wuv_int8_batched_host(int layer_idx, const float* x_host, float* y_host,
                                int H, int kvl, int vh, int S = 1) {
        if (layer_idx < 0 || (size_t)layer_idx >= wuv_int8_by_layer.size()) return false;
        int8_t* d_wuv_q = wuv_int8_by_layer[layer_idx];
        float* d_wuv_scale = wuv_int8_scale_by_layer[layer_idx];
        if (!d_wuv_q || !d_wuv_scale || H <= 0 || kvl <= 0 || vh <= 0 || S <= 0) return false;

        // 2026-09-02: S added to mirror wuv_bf16_batched_host's own batched
        // form. Without it this function silently computed only ONE query row
        // while the caller had already sized its [H,S,kvl] input and [S,H,vh]
        // output for S -- i.e. re-enabling the int8 path after the prefill
        // score batching landed would have produced garbage for a reason
        // entirely unrelated to the 2026-08-29 corruption this path was
        // originally shelved for, and looked like a confirmation of it.
        size_t n_x = (size_t)H * (size_t)S * (size_t)kvl;
        size_t n_y = (size_t)S * (size_t)H * (size_t)vh;
        size_t n_w = (size_t)H * (size_t)vh * (size_t)kvl;
        size_t x_bf16_bytes = n_x * sizeof(uint16_t);
        size_t y_bytes = n_y * sizeof(float);
        size_t w_bf16_bytes = n_w * sizeof(uint16_t);

        cudaSetDevice(device_id);
        if (x_bf16_bytes > d_wuv_ctx_bf16_cap) {
            if (d_wuv_ctx_bf16) { cudaFree(d_wuv_ctx_bf16); d_wuv_ctx_bf16 = nullptr; }
            d_wuv_ctx_bf16_cap = 0;
            if (cudaMalloc(&d_wuv_ctx_bf16, x_bf16_bytes) != cudaSuccess) {
                LOG_ERROR("[CUDA] wuv_int8_batched_host: d_wuv_ctx_bf16 grow to %.2f MB failed\n",
                          x_bf16_bytes / 1e6);
                return false;
            }
            d_wuv_ctx_bf16_cap = x_bf16_bytes;
        }
        if (y_bytes > d_wuv_out_cap) {
            if (d_wuv_out) { cudaFree(d_wuv_out); d_wuv_out = nullptr; }
            d_wuv_out_cap = 0;
            if (cudaMalloc((void**)&d_wuv_out, y_bytes) != cudaSuccess) {
                LOG_ERROR("[CUDA] wuv_int8_batched_host: d_wuv_out grow to %.2f MB failed\n",
                          y_bytes / 1e6);
                return false;
            }
            d_wuv_out_cap = y_bytes;
        }
        if (w_bf16_bytes > d_wuv_dequant_bf16_cap) {
            if (d_wuv_dequant_bf16) { cudaFree(d_wuv_dequant_bf16); d_wuv_dequant_bf16 = nullptr; }
            d_wuv_dequant_bf16_cap = 0;
            if (cudaMalloc(&d_wuv_dequant_bf16, w_bf16_bytes) != cudaSuccess) {
                LOG_ERROR("[CUDA] wuv_int8_batched_host: d_wuv_dequant_bf16 grow to %.2f MB failed\n",
                          w_bf16_bytes / 1e6);
                return false;
            }
            d_wuv_dequant_bf16_cap = w_bf16_bytes;
        }
        if (!ensure_host_scratch(h_wuv_ctx_bf16_scratch,
                                 h_wuv_ctx_bf16_scratch_cap,
                                 h_wuv_ctx_bf16_scratch_reservation_bytes,
                                 n_x, "DSV4 WUV BF16 activation"))
            return false;
        for (size_t i = 0; i < n_x; ++i) h_wuv_ctx_bf16_scratch[i] = fp32_to_bf16(x_host[i]);

        PERF_START(wuv_int8_dq);
        launch_dequant_wuv_int8(d_wuv_q, d_wuv_scale, d_wuv_dequant_bf16, H, vh, kvl, 0);
        PERF_END(wuv_int8_dq, t_wuv_int8_dequant);

        PERF_START(wuv_bf16_up);
        cudaMemcpy(d_wuv_ctx_bf16, h_wuv_ctx_bf16_scratch, x_bf16_bytes, cudaMemcpyHostToDevice);
        PERF_END(wuv_bf16_up, t_wuv_gpu_up);

        PERF_START(wuv_bf16_cmp);
        cublasStatus_t status = cublas_dense_matmul_bf16_matched_batched(
            cublas_handle,
            d_wuv_dequant_bf16, (long long)kvl * vh,
            d_wuv_ctx_bf16, (long long)S * kvl,
            d_wuv_out, (long long)vh,
            S, kvl, vh, H, /*ldc=*/H * vh);
        PERF_END(wuv_bf16_cmp, t_wuv_gpu_compute);
        if (status != CUBLAS_STATUS_SUCCESS) return false;

        PERF_START(wuv_bf16_dn);
        cudaMemcpy(y_host, d_wuv_out, y_bytes, cudaMemcpyDeviceToHost);
        PERF_END(wuv_bf16_dn, t_wuv_gpu_down);
        perf().n_wuv_gpu_calls.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    // MLA score GPU fast-path, part 1: allocate this layer's persistent KV
    // latent cache mirror ([max_t,kvl] + [max_t,rope], F32), sized ONCE to
    // engine.max_t so it never needs a realloc mid-generation (a realloc
    // would force re-uploading the whole cache from scratch, defeating the
    // point of an incrementally-grown mirror -- see upload_lc_rc_rows).
    // Budget-checked against max_bytes the same way upload_wuk_bf16 is,
    // never assumes headroom. Best-effort: returns false (leaving this
    // layer's entries null) on any failure -- caller (warm_mla_absorb_cache)
    // logs and continues; the real attention call falls back to the CPU
    // score path for that layer, same discipline as every other best-effort
    // GPU fast-path in this file. Idempotent -- a second call for an
    // already-allocated layer is a cheap no-op returning true.
    bool ensure_lc_rc_gpu(int layer_idx, int max_t, int kvl, int rope, size_t max_bytes) {
        // GLM5NEXT DSA is nope-only (rope==0). It still needs the Lc mirror;
        // a zero-byte Rc mirror is intentionally represented by nullptr.
        if (layer_idx < 0 || max_t <= 0 || kvl <= 0 || rope < 0) return false;
        if ((int)d_lc_by_layer.size() <= layer_idx) {
            d_lc_by_layer.resize(layer_idx + 1, nullptr);
            d_rc_by_layer.resize(layer_idx + 1, nullptr);
            lc_rc_max_t_by_layer.resize(layer_idx + 1, 0);
        }
        if (d_lc_by_layer[layer_idx] && (rope == 0 || d_rc_by_layer[layer_idx])) return true;  // already allocated

        size_t lc_bytes = (size_t)max_t * (size_t)kvl * sizeof(float);
        size_t rc_bytes = (size_t)max_t * (size_t)rope * sizeof(float);
        if (lc_bytes + rc_bytes > max_bytes) return false;

        cudaSetDevice(device_id);
        float* d_lc = nullptr;
        float* d_rc = nullptr;
        cudaError_t err1 = cudaMalloc(&d_lc, lc_bytes);
        cudaError_t err2 = (err1 == cudaSuccess && rc_bytes > 0)
                               ? cudaMalloc(&d_rc, rc_bytes) : err1;
        if (err1 != cudaSuccess || err2 != cudaSuccess) {
            LOG_WARN("[CUDA] score Lc/Rc GPU mirror alloc failed layer=%d (%.2f MB): %s -- "
                     "layer stays on the CPU score path\n",
                     layer_idx, (lc_bytes + rc_bytes) / 1e6,
                     cudaGetErrorString(err1 != cudaSuccess ? err1 : err2));
            if (d_lc) cudaFree(d_lc);
            if (d_rc) cudaFree(d_rc);
            return false;
        }
        // Zero-init: positions >= the real (still-growing) context length are
        // never read by score_gpu_batched_host's T bound, but zeroing avoids
        // ever feeding uninitialized VRAM bytes into a GEMM if a future
        // caller mistake ever passes too-large a T.
        cudaMemset(d_lc, 0, lc_bytes);
        if (rc_bytes > 0) cudaMemset(d_rc, 0, rc_bytes);
        d_lc_by_layer[layer_idx] = d_lc;
        d_rc_by_layer[layer_idx] = d_rc;
        lc_rc_max_t_by_layer[layer_idx] = max_t;
        perf().bytes_gpu_uploaded.fetch_add(lc_bytes + rc_bytes, std::memory_order_relaxed);
        return true;
    }

    // MLA score GPU fast-path, part 2: push newly-written KV latent cache
    // rows to this layer's GPU mirror. Called right after
    // attention_layer.cpp writes the same rows into engine.Lc[layer_idx]/
    // Rc[layer_idx] (mla_attention_absorbed_gguf's own KV-write step) --
    // host and device buffers share the IDENTICAL [max_t,*] row-major
    // layout and stride, so this is a single contiguous memcpy per array,
    // no reshuffling. n_rows is normally 1 (decode) or the prefill batch
    // size S; pos_base+n_rows must not exceed this layer's allocated max_t
    // (caller's own existing `if (pos >= engine.max_t) continue` bound
    // already guarantees this in practice).
    // Best-effort: false means this layer's mirror was never allocated
    // (ensure_lc_rc_gpu never called or failed) or the row range doesn't
    // fit -- caller should treat this the same as any other GPU fast-path
    // miss, i.e. score_gpu_batched_host will also return false for this
    // layer since its cache is stale/incomplete.
    bool upload_lc_rc_rows(int layer_idx, const float* Lc_host, const float* Rc_host,
                            int pos_base, int n_rows, int kvl, int rope) {
        if (layer_idx < 0 || (size_t)layer_idx >= d_lc_by_layer.size()) return false;
        float* d_lc = d_lc_by_layer[layer_idx];
        float* d_rc = d_rc_by_layer[layer_idx];
        if (!d_lc || n_rows <= 0 || pos_base < 0 || rope < 0) return false;
        if (rope > 0 && (!d_rc || !Rc_host)) return false;
        if (pos_base + n_rows > lc_rc_max_t_by_layer[layer_idx]) return false;

        cudaSetDevice(device_id);
        cudaMemcpy(d_lc + (size_t)pos_base * kvl, Lc_host + (size_t)pos_base * kvl,
                   (size_t)n_rows * kvl * sizeof(float), cudaMemcpyHostToDevice);
        if (rope > 0)
            cudaMemcpy(d_rc + (size_t)pos_base * rope, Rc_host + (size_t)pos_base * rope,
                       (size_t)n_rows * rope * sizeof(float), cudaMemcpyHostToDevice);
        return true;
    }

    // MLA score GPU fast-path, part 3: per-token dispatch. Computes
    // context[H,kvl] = softmax((Q_absorbed[H,kvl] @ Lc[T,kvl]^T +
    //                            Q_rope[H,rope] @ Rc[T,rope]^T) * scale)
    //                   @ Lc[T,kvl]
    // via 3 plain (non-batched) GEMMs + one combine/softmax kernel --
    // unlike qabsorb, MLA's latent K/V is SHARED across every head, so this
    // is NOT H independent per-head ops and needs no strided-batched GEMM;
    // every head just multiplies its own query against the one shared
    // cache. See tests/bench_score_cpu_vs_gpu.cu for the real-hardware
    // feasibility numbers this design is based on (5x-51x speedup,
    // verified correctness) -- this is the same 3-GEMM+softmax shape,
    // wired to the persistent per-layer Lc/Rc mirror instead of a
    // benchmark's one-shot upload.
    //
    // context_host: [H,kvl] row-major F32, out -- this is the SAME
    //   quantity the CPU score loop calls `latent_accumulation` per head;
    //   caller still applies the per-head WUV down-projection afterward
    //   (unchanged CPU code, out of scope here -- see this function's own
    //   file-header comment for why: WUV is H independent per-head GEMMs,
    //   structurally identical to qabsorb's already-proven batched
    //   approach, not new work needing separate validation).
    // Returns false (caller falls back to the CPU per-head loop) if this
    // layer's Lc/Rc mirror isn't ready or doesn't cover [0,T), or any CUDA/
    // cublas call fails -- never a hard requirement.
    // 2026-09-01: S/pos_base added so PREFILL (S>1) can use this path. The
    // caller previously gated on S==1 because "prefill has a DIFFERENT
    // past_context_len per row within the same batch" -- that per-row causal
    // limit now lives in the softmax kernel (see cuda/attn_score_softmax.cu),
    // which is the only place it was ever needed.
    //
    // Why no strided-batched GEMM is required even at S>1: MLA's latent K/V
    // (Lc/Rc) is shared across every head AND every query, so the [H,S] set of
    // queries collapses into ONE GEMM row dimension. Q_absorbed is already laid
    // out [H,S,kvl] (attention_layer.cpp's qabsorb loop indexes h*S*kvl +
    // s*kvl), so viewing it as an [H*S, kvl] matrix is exact -- each of the
    // three GEMMs below needs literally one change, H -> rows, and the ldb/lda
    // are untouched.
    //
    // Shapes: Q_absorbed_host [H,S,kvl], Q_rope_host [H,S,rope],
    //         context_host (out) [H,S,kvl]. Row (h,s) is at index h*S+s
    //         throughout. T is the total key count (pos_base+S for prefill,
    //         pos_base+1 for decode).
    bool score_gpu_batched_host(int layer_idx, const float* Q_absorbed_host, const float* Q_rope_host,
                                 float* context_host, int H, int T, int kvl, int rope, float scale,
                                 int S = 1, int pos_base = 0) {
        if (layer_idx < 0 || (size_t)layer_idx >= d_lc_by_layer.size()) return false;
        float* d_lc = d_lc_by_layer[layer_idx];
        float* d_rc = d_rc_by_layer[layer_idx];
        if (!d_lc || (rope > 0 && !d_rc) || H <= 0 || T <= 0 || kvl <= 0 || rope < 0 || S <= 0) return false;
        if (T > lc_rc_max_t_by_layer[layer_idx]) return false;
        // Every row must have at least one key to attend to, and the mirror
        // must cover the last query's own position -- otherwise the causal
        // mask would clamp and silently drop real context.
        if (pos_base < 0 || pos_base + S > T) return false;
        const int rows = H * S;   // one GEMM row per (head, query) pair

        cudaSetDevice(device_id);
        auto grow = [&](float** buf, size_t* cap, size_t need_bytes, const char* label) -> bool {
            if (need_bytes <= *cap) return true;
            if (*buf) { cudaFree(*buf); *buf = nullptr; }
            *cap = 0;
            if (cudaMalloc((void**)buf, need_bytes) != cudaSuccess) {
                LOG_ERROR("[CUDA] score_gpu_batched_host: %s grow to %.2f MB failed\n", label, need_bytes / 1e6);
                return false;
            }
            *cap = need_bytes;
            return true;
        };
        if (!grow(&d_score_q_absorbed, &d_score_q_absorbed_cap, (size_t)rows * kvl * sizeof(float), "d_score_q_absorbed")) return false;
        if (rope > 0 && !grow(&d_score_q_rope, &d_score_q_rope_cap,
                               (size_t)rows * rope * sizeof(float), "d_score_q_rope")) return false;
        if (!grow(&d_score_nope_buf, &d_score_nope_buf_cap, (size_t)rows * T * sizeof(float), "d_score_nope_buf")) return false;
        if (!grow(&d_score_rope_buf, &d_score_rope_buf_cap, (size_t)rows * T * sizeof(float), "d_score_rope_buf")) return false;
        if (!grow(&d_score_combined, &d_score_combined_cap, (size_t)rows * T * sizeof(float), "d_score_combined")) return false;
        if (!grow(&d_score_context, &d_score_context_cap, (size_t)rows * kvl * sizeof(float), "d_score_context")) return false;

        const float alpha = 1.0f, beta = 0.0f;
        PERF_START(score_gpu_up);
        cudaMemcpy(d_score_q_absorbed, Q_absorbed_host, (size_t)rows * kvl * sizeof(float), cudaMemcpyHostToDevice);
        if (rope > 0) {
            if (!Q_rope_host || cudaMemcpy(d_score_q_rope, Q_rope_host,
                                           (size_t)rows * rope * sizeof(float),
                                           cudaMemcpyHostToDevice) != cudaSuccess)
                return false;
        }
        // The combine kernel always adds both score buffers. For nope-only
        // DSA, make the rope contribution an explicit zero buffer instead of
        // declining the entire GPU path and re-entering the CPU O(T) loop.
        if (rope == 0 && cudaMemset(d_score_rope_buf, 0,
                                    (size_t)rows * T * sizeof(float)) != cudaSuccess)
            return false;
        PERF_END(score_gpu_up, t_score_gpu_up);

        PERF_START(score_gpu_cmp);
        // scores_nope[H,T] = Q_absorbed[H,kvl] @ Lc[T,kvl]^T -- same "Y=X@W^T"
        // pattern as cublas_dense_matmul (Lc plays the role of a [T,kvl] weight).
        cublasStatus_t st1 = cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, T, rows, kvl, &alpha,
                                          d_lc, kvl, d_score_q_absorbed, kvl, &beta, d_score_nope_buf, T);
        cublasStatus_t st2 = CUBLAS_STATUS_SUCCESS;
        if (rope > 0)
            st2 = cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, T, rows, rope, &alpha,
                              d_rc, rope, d_score_q_rope, rope, &beta, d_score_rope_buf, T);
        if (st1 != CUBLAS_STATUS_SUCCESS || st2 != CUBLAS_STATUS_SUCCESS) { PERF_END(score_gpu_cmp, t_score_gpu_compute); return false; }

        launch_combine_scale_softmax(d_score_nope_buf, d_score_rope_buf, d_score_combined, H, T, scale, S, pos_base, 0);

        // context[H,kvl] = attn_weights[H,T] @ Lc[T,kvl] -- plain row-major
        // A@B (Lc NOT transposed here, unlike the two GEMMs above), standard
        // row-major-via-column-major trick: swap operand order/dims.
        cublasStatus_t st3 = cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, kvl, rows, T, &alpha,
                                          d_lc, kvl, d_score_combined, T, &beta, d_score_context, kvl);
        PERF_END(score_gpu_cmp, t_score_gpu_compute);
        if (st3 != CUBLAS_STATUS_SUCCESS) return false;

        PERF_START(score_gpu_dn);
        cudaMemcpy(context_host, d_score_context, (size_t)rows * kvl * sizeof(float), cudaMemcpyDeviceToHost);
        cudaStreamSynchronize(0);
        PERF_END(score_gpu_dn, t_score_gpu_down);
        perf().n_score_gpu_calls.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    // DSV4 raw-SWA/HCA score fast-path, part 1a: allocate this layer's
    // persistent raw-window ring-buffer mirror, [window,qh] F32, sized to
    // the model's FIXED sliding_window (never grows -- the ring buffer's
    // capacity is constant for the whole run, unlike Lc/Rc's max_t sizing
    // above). Zero-init, same "never feed uninitialized VRAM into a GEMM"
    // rationale as ensure_lc_rc_gpu. Best-effort: false leaves this layer's
    // mirror entry null, caller falls back to the CPU score loop for that
    // layer. Idempotent -- a second call for an already-allocated layer is
    // a cheap no-op.
    bool ensure_dsv4_kvc_gpu(int layer_idx, int window, int qh, size_t max_bytes) {
        if (layer_idx < 0 || window <= 0 || qh <= 0) return false;
        if ((int)d_dsv4_kvc_by_layer.size() <= layer_idx) {
            d_dsv4_kvc_by_layer.resize(layer_idx + 1, nullptr);
            dsv4_kvc_window_by_layer.resize(layer_idx + 1, 0);
        }
        if (d_dsv4_kvc_by_layer[layer_idx]) return true;  // already allocated

        size_t bytes = (size_t)window * (size_t)qh * sizeof(float);
        if (bytes > max_bytes) return false;

        cudaSetDevice(device_id);
        float* d_kvc = nullptr;
        cudaError_t err = cudaMalloc((void**)&d_kvc, bytes);
        if (err != cudaSuccess) {
            LOG_WARN("[CUDA] DSV4 KVc ring GPU mirror alloc failed layer=%d (%.2f MB): %s -- "
                     "layer stays on the CPU score path\n",
                     layer_idx, bytes / 1e6, cudaGetErrorString(err));
            return false;
        }
        cudaMemset(d_kvc, 0, bytes);
        d_dsv4_kvc_by_layer[layer_idx] = d_kvc;
        dsv4_kvc_window_by_layer[layer_idx] = window;
        perf().bytes_gpu_uploaded.fetch_add(bytes, std::memory_order_relaxed);
        return true;
    }

    // DSV4 raw-SWA/HCA score fast-path, part 1b: push one newly-written
    // ring row to this layer's mirror. Called right after
    // attention_layer.cpp writes the same row into
    // engine.KVc[layer_idx][slot] -- host and device share the identical
    // [window,qh] row-major layout, so this is a single qh-wide memcpy at
    // the ring slot (NOT append-only, unlike upload_lc_rc_rows -- a ring
    // buffer overwrites the same slot repeatedly as pos wraps).
    bool upload_dsv4_kvc_row(int layer_idx, const float* kv_row_host, int slot, int qh) {
        if (layer_idx < 0 || (size_t)layer_idx >= d_dsv4_kvc_by_layer.size()) return false;
        float* d_kvc = d_dsv4_kvc_by_layer[layer_idx];
        if (!d_kvc || slot < 0 || slot >= dsv4_kvc_window_by_layer[layer_idx] || qh <= 0) return false;

        cudaSetDevice(device_id);
        cudaMemcpy(d_kvc + (size_t)slot * qh, kv_row_host, (size_t)qh * sizeof(float), cudaMemcpyHostToDevice);
        return true;
    }

    // DSV4 raw-SWA/HCA score fast-path, part 2a: allocate this layer's
    // persistent HCA compressed-block mirror, [cache_capacity,qh] F32 --
    // same shape/discipline as ensure_dsv4_kvc_gpu above, just append-only
    // by block index instead of ring-indexed. Only ever called for
    // ratio==128 layers.
    bool ensure_dsv4_hca_gpu(int layer_idx, int cache_capacity, int qh, size_t max_bytes) {
        if (layer_idx < 0 || cache_capacity <= 0 || qh <= 0) return false;
        if ((int)d_dsv4_hca_by_layer.size() <= layer_idx) {
            d_dsv4_hca_by_layer.resize(layer_idx + 1, nullptr);
            dsv4_hca_capacity_by_layer.resize(layer_idx + 1, 0);
        }
        if (d_dsv4_hca_by_layer[layer_idx]) return true;  // already allocated

        size_t bytes = (size_t)cache_capacity * (size_t)qh * sizeof(float);
        if (bytes > max_bytes) return false;

        cudaSetDevice(device_id);
        float* d_hca = nullptr;
        cudaError_t err = cudaMalloc((void**)&d_hca, bytes);
        if (err != cudaSuccess) {
            LOG_WARN("[CUDA] DSV4 HCA cache GPU mirror alloc failed layer=%d (%.2f MB): %s -- "
                     "layer stays on the CPU score path\n",
                     layer_idx, bytes / 1e6, cudaGetErrorString(err));
            return false;
        }
        cudaMemset(d_hca, 0, bytes);
        d_dsv4_hca_by_layer[layer_idx] = d_hca;
        dsv4_hca_capacity_by_layer[layer_idx] = cache_capacity;
        perf().bytes_gpu_uploaded.fetch_add(bytes, std::memory_order_relaxed);
        return true;
    }

    // DSV4 raw-SWA/HCA score fast-path, part 2b: push one newly-committed
    // HCA compressed block to this layer's mirror (called right after
    // dsv4_commit_compressed_key writes the same row into
    // Dsv4CompressedLayerState::keys, attention_layer.cpp).
    bool upload_dsv4_hca_row(int layer_idx, const float* key_host, int block_idx, int qh) {
        if (layer_idx < 0 || (size_t)layer_idx >= d_dsv4_hca_by_layer.size()) return false;
        float* d_hca = d_dsv4_hca_by_layer[layer_idx];
        if (!d_hca || block_idx < 0 || block_idx >= dsv4_hca_capacity_by_layer[layer_idx] || qh <= 0) return false;

        cudaSetDevice(device_id);
        cudaMemcpy(d_hca + (size_t)block_idx * qh, key_host, (size_t)qh * sizeof(float), cudaMemcpyHostToDevice);
        return true;
    }

    // qwen4exp dense full-attention on the GPU (EngineConfig::qwen4exp_attn_gpu).
    //
    // Replaces attention_layer.cpp's per-(token,head) CPU loop -- which does
    // `dot_f32_avx512` per (query,key) pair and an AVX-512 axpy per key -- with
    // the standard softmax(Q*K^T)*V decomposition: two cuBLAS strided-batched
    // GEMMs around one softmax kernel. Measured at 46-50s of a ~300s qwen4exp
    // prefill across the 12 dense layers, versus 0.35s for ALL routed-expert
    // GPU compute, so this is the dominant remaining prefill cost.
    //
    // GQA maps onto strided-batched cleanly and needs NO host transpose:
    //  * Query heads sharing one kv head are CONTIGUOUS (h/group == kv means
    //    h in [kv*group, kv*group+group)), so one call per kv head with
    //    strideA = 0 (same K/V for the whole group) covers them all. With
    //    Hkv=2 that is 2 calls per GEMM, not H.
    //  * Q is [S,H,hd] and out is [S,H,hd] row-major; head h is exactly a
    //    column-major hd x n_q block with ld = H*hd at offset h*hd, and the
    //    next head is +hd away -- so leading dimension + strideB express the
    //    layout directly. Same for the [T,Hkv,hd] K/V cache with ld = Hkv*hd.
    //
    // F32 throughout, deliberately: this is the first cut of a numerically
    // sensitive path and the codebase has a documented NaN-cascade history,
    // so correctness comes before the tensor-core BF16 variant.
    //
    // Q_host:   [S,H,hd] row-major F32 (post QK-norm, post RoPE).
    // K_host/V_host: engine.Kc/Vc[layer] base, [T,Hkv,hd] row-major F32.
    // out_host: [S,H,hd] row-major F32, written in full (beta=0). When
    // device_out is non-null, the result is retained in d_q4_out instead and
    // the caller can hand that raw CUDA buffer to the next device operation.
    // T must be the real context length (pos_base+S, clamped to max_t).
    // Returns false on any allocation/cuBLAS failure -- caller falls back to
    // the unchanged CPU loop, never a hard requirement.
    bool qwen4exp_attn_gpu_host(const float* Q_host, const float* K_host,
                                const float* V_host, float* out_host,
                                int S, int T, int H, int Hkv, int hd,
                                int pos_base, float scale,
                                void **device_out = nullptr) {
        if (device_out) *device_out = nullptr;
        if (!Q_host || !K_host || !V_host || (!out_host && !device_out))
            return false;
        if (S <= 0 || T <= 0 || H <= 0 || Hkv <= 0 || hd <= 0) return false;
        if (H % Hkv != 0) return false;              // GQA grouping assumption
        const int group = H / Hkv;

        cudaSetDevice(device_id);
        auto grow = [&](float** buf, size_t* cap, size_t need, const char* label) -> bool {
            if (need <= *cap) return true;
            if (!vram_ok_to_grow(need - *cap)) return false;   // watermark
            if (*buf) { cudaFree(*buf); *buf = nullptr; }
            *cap = 0;
            if (cudaMalloc((void**)buf, need) != cudaSuccess) {
                LOG_WARN("[CUDA] qwen4exp_attn_gpu_host: %s grow to %.1f MB failed -- "
                         "layer falls back to the CPU attention loop\n", label, need / 1e6);
                return false;
            }
            *cap = need;
            return true;
        };

        // Chunk over queries so the [H,n_q,T] score matrix stays bounded: at
        // H=24, S=T=3511 the full matrix is 1.18 GB, which would both risk the
        // CoreEngine GPU's budget and evict everything else from cache.
        const size_t kScoreBudget = 512ull * 1024 * 1024;
        const size_t per_q = (size_t)H * T * sizeof(float);
        int n_q = (int)std::min<size_t>((size_t)S, std::max<size_t>(1, kScoreBudget / std::max<size_t>(1, per_q)));

        const size_t q_bytes  = (size_t)S * H * hd * sizeof(float);
        const size_t kv_bytes = (size_t)T * Hkv * hd * sizeof(float);
        const size_t sc_bytes = (size_t)H * n_q * T * sizeof(float);
        if (!grow(&d_q4_Q, &d_q4_Q_cap, q_bytes, "d_q4_Q")) return false;
        if (!grow(&d_q4_K, &d_q4_K_cap, kv_bytes, "d_q4_K")) return false;
        if (!grow(&d_q4_V, &d_q4_V_cap, kv_bytes, "d_q4_V")) return false;
        if (!grow(&d_q4_scores, &d_q4_scores_cap, sc_bytes, "d_q4_scores")) return false;
        if (!grow(&d_q4_out, &d_q4_out_cap, q_bytes, "d_q4_out")) return false;

        PERF_START(q4_up);
        if (cudaMemcpy(d_q4_Q, Q_host, q_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d_q4_K, K_host, kv_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d_q4_V, V_host, kv_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            LOG_WARN("[CUDA] qwen4exp_attn_gpu_host: H2D failed -- CPU fallback\n");
            return false;
        }
        PERF_END(q4_up, t_qwen4exp_attn_gpu_up);
        perf().bytes_gpu_uploaded.fetch_add(q_bytes + 2 * kv_bytes, std::memory_order_relaxed);

        const float alpha = 1.0f, beta = 0.0f;
        PERF_START(q4_cmp);
        for (int s0 = 0; s0 < S; s0 += n_q) {
            const int nq = std::min(n_q, S - s0);
            for (int kv = 0; kv < Hkv; ++kv) {
                const int h0 = kv * group;
                // scores[h] (nq x T) = Q[h] (nq x hd) * K[kv]^T (hd x T)
                if (cublasSgemmStridedBatched(
                        cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        T, nq, hd, &alpha,
                        d_q4_K + (size_t)kv * hd, Hkv * hd, 0,
                        d_q4_Q + (size_t)s0 * H * hd + (size_t)h0 * hd, H * hd, hd,
                        &beta,
                        d_q4_scores + (size_t)h0 * nq * T, T, (long long)nq * T,
                        group) != CUBLAS_STATUS_SUCCESS) {
                    LOG_WARN("[CUDA] qwen4exp_attn_gpu_host: QK^T GEMM failed -- CPU fallback\n");
                    return false;
                }
            }
            launch_qwen4exp_scale_softmax_causal(d_q4_scores, H, nq, T,
                                                 pos_base + s0, scale, 0);
            for (int kv = 0; kv < Hkv; ++kv) {
                const int h0 = kv * group;
                // out[h] (nq x hd) = scores[h] (nq x T) * V[kv] (T x hd)
                if (cublasSgemmStridedBatched(
                        cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        hd, nq, T, &alpha,
                        d_q4_V + (size_t)kv * hd, Hkv * hd, 0,
                        d_q4_scores + (size_t)h0 * nq * T, T, (long long)nq * T,
                        &beta,
                        d_q4_out + (size_t)s0 * H * hd + (size_t)h0 * hd, H * hd, hd,
                        group) != CUBLAS_STATUS_SUCCESS) {
                    LOG_WARN("[CUDA] qwen4exp_attn_gpu_host: scores*V GEMM failed -- CPU fallback\n");
                    return false;
                }
            }
        }
        if (cudaStreamSynchronize(0) != cudaSuccess) {
            LOG_WARN("[CUDA] qwen4exp_attn_gpu_host: sync failed -- CPU fallback\n");
            return false;
        }
        PERF_END(q4_cmp, t_qwen4exp_attn_gpu_cmp);

        if (device_out) {
            *device_out = d_q4_out;
        } else {
            PERF_START(q4_dn);
            if (cudaMemcpy(out_host, d_q4_out, q_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
                LOG_WARN("[CUDA] qwen4exp_attn_gpu_host: D2H failed -- CPU fallback\n");
                return false;
            }
            PERF_END(q4_dn, t_qwen4exp_attn_gpu_dn);
        }
        perf().n_qwen4exp_attn_gpu_calls.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    // Device-input counterpart used by Qwen4Exp prefill.  The three
    // projection results and Q/K normalization/RoPE have already been
    // produced in d_q4_Q/K/V by qwen4exp_prepare_qkv_device(), so this
    // launches the same attention GEMMs without the Q/K/V H2D round trip.
    // The pointers are checked against the manager's persistent buffers to
    // prevent a caller from handing us a buffer whose lifetime is not held by
    // this object.
    bool qwen4exp_attn_gpu_device_inputs(const void *Q_device,
                                         const void *K_device,
                                         const void *V_device,
                                         int S, int T, int H, int Hkv, int hd,
                                         int pos_base, float scale,
                                         const void **device_out) {
        if (device_out) *device_out = nullptr;
        if (!device_out || !Q_device || !K_device || !V_device ||
            Q_device != d_q4_Q || K_device != d_q4_K || V_device != d_q4_V ||
            S <= 0 || T <= 0 || H <= 0 || Hkv <= 0 || hd <= 0 || H % Hkv)
            return false;
        if ((size_t)S * H * hd * sizeof(float) > d_q4_Q_cap ||
            (size_t)T * Hkv * hd * sizeof(float) > d_q4_K_cap ||
            (size_t)T * Hkv * hd * sizeof(float) > d_q4_V_cap)
            return false;

        cudaSetDevice(device_id);
        const size_t kScoreBudget = 512ull * 1024 * 1024;
        const size_t per_q = (size_t)H * T * sizeof(float);
        const int group = H / Hkv;
        const int n_q = (int)std::min<size_t>(
            (size_t)S, std::max<size_t>(1, kScoreBudget / std::max<size_t>(1, per_q)));
        const size_t q_bytes = (size_t)S * H * hd * sizeof(float);
        const size_t sc_bytes = (size_t)H * n_q * T * sizeof(float);
        auto grow = [&](float **buf, size_t *cap, size_t need,
                        const char *label) -> bool {
            if (need <= *cap) return true;
            if (!vram_ok_to_grow(need - *cap)) return false;
            if (*buf) { cudaFree(*buf); *buf = nullptr; }
            *cap = 0;
            if (cudaMalloc((void **)buf, need) != cudaSuccess) {
                LOG_WARN("[CUDA] qwen4exp_attn_gpu_device_inputs: %s grow failed\n",
                         label);
                return false;
            }
            *cap = need;
            return true;
        };
        if (!grow(&d_q4_scores, &d_q4_scores_cap, sc_bytes, "d_q4_scores") ||
            !grow(&d_q4_out, &d_q4_out_cap, q_bytes, "d_q4_out"))
            return false;

        const float alpha = 1.0f, beta = 0.0f;
        PERF_START(q4_cmp);
        for (int s0 = 0; s0 < S; s0 += n_q) {
            const int nq = std::min(n_q, S - s0);
            for (int kv = 0; kv < Hkv; ++kv) {
                const int h0 = kv * group;
                if (cublasSgemmStridedBatched(
                        cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        T, nq, hd, &alpha,
                        d_q4_K + (size_t)kv * hd, Hkv * hd, 0,
                        d_q4_Q + (size_t)s0 * H * hd + (size_t)h0 * hd,
                        H * hd, hd, &beta,
                        d_q4_scores + (size_t)h0 * nq * T, T,
                        (long long)nq * T, group) != CUBLAS_STATUS_SUCCESS)
                    return false;
            }
            launch_qwen4exp_scale_softmax_causal(d_q4_scores, H, nq, T,
                                                 pos_base + s0, scale, 0);
            for (int kv = 0; kv < Hkv; ++kv) {
                const int h0 = kv * group;
                if (cublasSgemmStridedBatched(
                        cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        hd, nq, T, &alpha,
                        d_q4_V + (size_t)kv * hd, Hkv * hd, 0,
                        d_q4_scores + (size_t)h0 * nq * T, T,
                        (long long)nq * T, &beta,
                        d_q4_out + (size_t)s0 * H * hd + (size_t)h0 * hd,
                        H * hd, hd, group) != CUBLAS_STATUS_SUCCESS)
                    return false;
            }
        }
        if (cudaStreamSynchronize(0) != cudaSuccess) return false;
        PERF_END(q4_cmp, t_qwen4exp_attn_gpu_cmp);
        *device_out = d_q4_out;
        perf().n_qwen4exp_attn_gpu_calls.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    // Project and prepare Q/K/V directly on CUDA for Qwen4Exp prefill.  The
    // caller supplies the already device-resident HC activation and the three
    // resident projection weights.  This method intentionally implements the
    // plain-RoPE case first; the model's YaRN parameters are checked by the
    // caller and use the established host path when they are non-default.
    bool qwen4exp_prepare_qkv_device(
        GGUFType q_type, const void *q_weight,
        GGUFType k_type, const void *k_weight,
        GGUFType v_type, const void *v_weight,
        const float *device_x, const float *q_norm_host,
        const float *k_norm_host, int S, int D, int H, int Hkv, int hd,
        int n_rot, int pos_base, float theta_base, float eps,
        const void **q_device, const void **gate_device,
        const void **k_device, const void **v_device) {
        if (q_device) *q_device = nullptr;
        if (gate_device) *gate_device = nullptr;
        if (k_device) *k_device = nullptr;
        if (v_device) *v_device = nullptr;
        if (!q_device || !gate_device || !k_device || !v_device ||
            !q_weight || !k_weight || !v_weight || !device_x ||
            !q_norm_host || !k_norm_host || S <= 0 || D <= 0 || H <= 0 ||
            Hkv <= 0 || hd <= 0 || n_rot <= 0 || n_rot > hd ||
            (n_rot & 1))
            return false;

        cudaSetDevice(device_id);
        auto grow = [&](float **buf, size_t *cap, size_t need,
                        const char *label) -> bool {
            if (need <= *cap) return true;
            if (!vram_ok_to_grow(need - *cap)) return false;
            if (*buf) { cudaFree(*buf); *buf = nullptr; }
            *cap = 0;
            if (cudaMalloc((void **)buf, need) != cudaSuccess) {
                LOG_WARN("[CUDA] qwen4exp_prepare_qkv_device: %s grow failed\n",
                         label);
                return false;
            }
            *cap = need;
            return true;
        };

        const size_t qfull_bytes = (size_t)S * H * 2 * hd * sizeof(float);
        const size_t q_bytes = (size_t)S * H * hd * sizeof(float);
        const size_t kv_bytes = (size_t)S * Hkv * hd * sizeof(float);
        const size_t norm_bytes = (size_t)hd * sizeof(float);
        if (!grow(&d_q4_qfull, &d_q4_qfull_cap, qfull_bytes, "d_q4_qfull") ||
            !grow(&d_q4_Q, &d_q4_Q_cap, q_bytes, "d_q4_Q") ||
            !grow(&d_q4_gate, &d_q4_gate_cap, q_bytes, "d_q4_gate") ||
            !grow(&d_q4_K, &d_q4_K_cap, kv_bytes, "d_q4_K") ||
            !grow(&d_q4_V, &d_q4_V_cap, kv_bytes, "d_q4_V") ||
            !grow(&d_q4_qnorm, &d_q4_qnorm_cap, norm_bytes, "d_q4_qnorm") ||
            !grow(&d_q4_knorm, &d_q4_knorm_cap, norm_bytes, "d_q4_knorm"))
            return false;

        if (!dense_matmul_device(q_type, q_weight, device_x, d_q4_qfull,
                                 S, D, H * 2 * hd) ||
            !dense_matmul_device(k_type, k_weight, device_x, d_q4_K,
                                 S, D, Hkv * hd) ||
            !dense_matmul_device(v_type, v_weight, device_x, d_q4_V,
                                 S, D, Hkv * hd))
            return false;
        if (cudaMemcpy(d_q4_qnorm, q_norm_host, norm_bytes,
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d_q4_knorm, k_norm_host, norm_bytes,
                       cudaMemcpyHostToDevice) != cudaSuccess)
            return false;
        launch_qwen4exp_prepare_qkv(
            d_q4_qfull, d_q4_K, d_q4_V, d_q4_Q, d_q4_gate, d_q4_K, d_q4_V,
            d_q4_qnorm, d_q4_knorm, S, H, Hkv, hd, n_rot, pos_base,
            theta_base, eps, 0);
        if (cudaGetLastError() != cudaSuccess) return false;
        *q_device = d_q4_Q;
        *gate_device = d_q4_gate;
        *k_device = d_q4_K;
        *v_device = d_q4_V;
        return true;
    }

    // Device-handoff form: Q/K/V still use the existing host-backed path, but
    // the final attention result stays on CUDA for HC post-combine.
    bool qwen4exp_attn_gpu_device(const float* Q_host, const float* K_host,
                                  const float* V_host, int S, int T, int H,
                                  int Hkv, int hd, int pos_base, float scale,
                                  const void **device_out) {
        if (!device_out) return false;
        void *out = nullptr;
        if (!qwen4exp_attn_gpu_host(Q_host, K_host, V_host, nullptr, S, T, H,
                                    Hkv, hd, pos_base, scale, &out))
            return false;
        *device_out = out;
        return out != nullptr;
    }

    // Apply the Qwen4Exp sigmoid output gate and attn_output projection while
    // the attention context is still on CUDA.  gate_host is the host-side
    // half of the fused attn_q projection; it is uploaded once into the
    // already-sized q4 scratch buffer.  The projection weight is the normal
    // resident device tensor used by dense_proj().
    bool qwen4exp_attn_gpu_post_device(const void *context_device,
                                       const float *gate_host,
                                       GGUFType output_type,
                                       const void *output_weight_device,
                                       int S, int H, int hd, int D,
                                       const void **device_out) {
        if (!context_device || !gate_host || !output_weight_device ||
            !device_out || S <= 0 || H <= 0 || hd <= 0 || D <= 0 ||
            context_device != d_q4_out)
            return false;
        const size_t context_bytes = (size_t)S * H * hd * sizeof(float);
        if (context_bytes > d_q4_out_cap || context_bytes > d_q4_Q_cap)
            return false;
        if (!ensure_buffers(context_bytes, (size_t)S * D * sizeof(float)))
            return false;
        cudaSetDevice(device_id);
        if (cudaMemcpy(d_q4_Q, gate_host, context_bytes,
                       cudaMemcpyHostToDevice) != cudaSuccess)
            return false;
        // Preserve the raw attention context for a safe host fallback if the
        // output projection type or allocation is unsupported.
        if (cudaMemcpy(d_x, d_q4_out, context_bytes,
                       cudaMemcpyDeviceToDevice) != cudaSuccess)
            return false;
        launch_qwen4exp_sigmoid_gate(d_x, d_q4_Q,
                                     (int)((size_t)S * H * hd), 0);
        if (!dense_matmul_device(output_type, output_weight_device, d_x,
                                 d_y, S, H * hd, D))
            return false;
        *device_out = d_y;
        return true;
    }

    // Same post-attention operation as qwen4exp_attn_gpu_post_device(), with
    // the sigmoid gate already resident on CUDA from the device QKV prepare
    // path.  This avoids downloading the fused Q projection's gate half and
    // uploading it again before HC post-combine.
    bool qwen4exp_attn_gpu_post_device_gate(
        const void *context_device, const void *gate_device,
        GGUFType output_type, const void *output_weight_device,
        int S, int H, int hd, int D, const void **device_out) {
        if (!context_device || !gate_device || !output_weight_device ||
            !device_out || context_device != d_q4_out ||
            gate_device != d_q4_gate || S <= 0 || H <= 0 || hd <= 0 || D <= 0)
            return false;
        const size_t context_bytes = (size_t)S * H * hd * sizeof(float);
        if (context_bytes > d_q4_out_cap || context_bytes > d_q4_gate_cap)
            return false;
        if (!ensure_buffers(context_bytes, (size_t)S * D * sizeof(float)))
            return false;
        cudaSetDevice(device_id);
        if (cudaMemcpy(d_x, d_q4_out, context_bytes,
                       cudaMemcpyDeviceToDevice) != cudaSuccess)
            return false;
        launch_qwen4exp_sigmoid_gate(d_x, (const float *)d_q4_gate,
                                     (int)((size_t)S * H * hd), 0);
        if (!dense_matmul_device(output_type, output_weight_device, d_x,
                                 d_y, S, H * hd, D))
            return false;
        *device_out = d_y;
        return true;
    }

    // Recovery path if a downstream raw device handoff rejects the output.
    bool qwen4exp_attn_gpu_copy_device_output(const void *device_out,
                                              float *host_out, size_t bytes) {
        if (!device_out || !host_out ||
            (device_out != d_q4_out && device_out != d_y) ||
            (device_out == d_q4_out ? bytes > d_q4_out_cap : bytes > d_y_cap))
            return false;
        return cudaMemcpy(host_out, device_out, bytes,
                          cudaMemcpyDeviceToHost) == cudaSuccess;
    }

    // VRAM headroom gate shared by every DYNAMIC grow-only scratch path below
    // (dense BF16 weights, qwen4exp attention, chunkwise GDN, expert
    // accumulator). Static pools sized from config at startup are budgeted
    // separately -- this is for the buffers that grow at run time, which are
    // the ones that historically OOM'd mid-run.
    // Returns true if `extra` more bytes keeps this device at or under
    // EngineConfig::vram_max_fraction. On query failure it allows the
    // allocation: a missing reading must not silently disable a fast path.
    bool vram_ok_to_grow(size_t extra) {
        if (extra == 0) return true;
        size_t vfree = 0, vtotal = 0;
        if (cudaMemGetInfo(&vfree, &vtotal) != cudaSuccess) return true;
        if (vram_has_headroom(extra, vfree, vtotal, vram_min_free_bytes())) return true;
        perf().n_vram_watermark_refused.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    // ---- GPU-side routed-expert accumulation (EngineConfig::moe_gpu_accum) ----
    // Replaces "download every expert's [n,D] output, weighted-add on the CPU"
    // with "accumulate on the device, download [S,D] once per layer". See
    // cuda/expert_accum.cu for the traffic arithmetic (~16.9 GB -> ~1.7 GB).
    //
    // begin() is called lazily the first time a layer dispatches an expert to a
    // given GPU, because which devices a layer touches is not known upfront.
    bool expert_accum_begin(int gpu_id, int S, int D,
                            const int* tok_host, const float* w_host, int total_pairs) {
        if (gpu_id < 0 || gpu_id >= kMaxAccumGpus || S <= 0 || D <= 0) return false;
        cudaSetDevice(gpu_id);
        auto grow = [&](void** b, size_t* cap, size_t need) -> bool {
            if (need <= *cap) return true;
            if (!vram_ok_to_grow(need - *cap)) return false;   // watermark
            if (*b) { cudaFree(*b); *b = nullptr; }
            *cap = 0;
            if (cudaMalloc(b, need) != cudaSuccess) return false;
            *cap = need; return true;
        };
        const size_t acc_b = (size_t)S * D * sizeof(float);
        if (!grow((void**)&d_acc[gpu_id], &d_acc_cap[gpu_id], acc_b)) return false;
        if (total_pairs > 0) {
            if (!grow((void**)&d_acc_tok[gpu_id], &d_acc_tok_cap[gpu_id],
                      (size_t)total_pairs * sizeof(int))) return false;
            if (!grow((void**)&d_acc_w[gpu_id], &d_acc_w_cap[gpu_id],
                      (size_t)total_pairs * sizeof(float))) return false;
            if (cudaMemcpy(d_acc_tok[gpu_id], tok_host, (size_t)total_pairs * sizeof(int),
                           cudaMemcpyHostToDevice) != cudaSuccess) return false;
            if (cudaMemcpy(d_acc_w[gpu_id], w_host, (size_t)total_pairs * sizeof(float),
                           cudaMemcpyHostToDevice) != cudaSuccess) return false;
        }
        if (cudaMemset(d_acc[gpu_id], 0, acc_b) != cudaSuccess) return false;
        acc_active[gpu_id] = true;
        return true;
    }

    // Scatter this expert's device-resident output rows into the accumulator.
    // pair_off indexes into the tok/w arrays uploaded by begin().
    bool expert_accum_add(int gpu_id, const float* d_expert_out,
                          int pair_off, int n, int D, int S) {
        if (gpu_id < 0 || gpu_id >= kMaxAccumGpus || !acc_active[gpu_id]) return false;
        if (!d_expert_out || n <= 0) return false;
        cudaSetDevice(gpu_id);
        launch_expert_scatter_accum(d_expert_out, d_acc_tok[gpu_id] + pair_off,
                                    d_acc_w[gpu_id] + pair_off, d_acc[gpu_id], n, D, S, 0);
        return true;
    }

    // One [S,D] download per layer per GPU, ADDED into the host output (which
    // may already hold contributions from CPU-dispatched experts).
    bool expert_accum_end(int gpu_id, float* out_host, int S, int D) {
        if (gpu_id < 0 || gpu_id >= kMaxAccumGpus || !acc_active[gpu_id]) return false;
        cudaSetDevice(gpu_id);
        if (cudaStreamSynchronize(0) != cudaSuccess) { acc_active[gpu_id] = false; return false; }
        static thread_local std::vector<float> tmp;
        tmp.resize((size_t)S * D);
        PERF_START(acc_dn);
        const cudaError_t e = cudaMemcpy(tmp.data(), d_acc[gpu_id],
                                         (size_t)S * D * sizeof(float), cudaMemcpyDeviceToHost);
        PERF_END(acc_dn, t_moe_accum_dn);
        acc_active[gpu_id] = false;
        if (e != cudaSuccess) return false;
        for (size_t i = 0; i < tmp.size(); ++i) out_host[i] += tmp[i];
        perf().n_moe_gpu_accum_layers.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    bool expert_accum_is_active(int gpu_id) const {
        return gpu_id >= 0 && gpu_id < kMaxAccumGpus && acc_active[gpu_id];
    }

    // qwen4exp chunkwise-parallel GDN on the GPU (EngineConfig::qwen4exp_gdn_gpu).
    //
    // Port of include/qwen4exp_gdn.hpp's qwen4exp_gdn_delta_chunk, which is
    // itself derived from and unit-tested against this project's sequential
    // qwen4exp_gdn_delta_step oracle. See cuda/qwen4exp_gdn.cu for the notation.
    //
    // Chunks are sequential BY CONSTRUCTION (the recurrent state carries), but
    // all H_v heads and all hd columns run in parallel within a chunk, which is
    // exactly what the CPU form could not exploit: measured at 82.8s of a
    // 263.8s prefill (31%), the single largest remaining bucket.
    //
    // Q/K are [S,H_k,hd] and shared across value heads (hk = h % H_k); V/out are
    // [S,H_v,hd]; beta/g_log are [S,H_v]; state is [H_v,hd,hd] indexed [k][v].
    // Returns false on any failure -- caller falls back to the CPU path.
    bool qwen4exp_gdn_gpu_host(const float* Q, const float* K, const float* V,
                               const float* beta, const float* g_log,
                               float* state, float* out,
                               int S, int H_k, int H_v, int hd, int chunk) {
        if (!Q || !K || !V || !beta || !g_log || !state || !out) return false;
        if (S <= 1 || H_k <= 0 || H_v <= 0 || hd <= 0 || chunk <= 0) return false;
        if (H_v % H_k != 0) return false;

        cudaSetDevice(device_id);
        auto grow = [&](float** b, size_t* cap, size_t need, const char* label) -> bool {
            if (need <= *cap) return true;
            if (!vram_ok_to_grow(need - *cap)) return false;   // watermark
            if (*b) { cudaFree(*b); *b = nullptr; }
            *cap = 0;
            if (cudaMalloc((void**)b, need) != cudaSuccess) {
                LOG_WARN("[CUDA] qwen4exp_gdn_gpu_host: %s grow to %.1f MB failed -- CPU fallback\n",
                         label, need / 1e6);
                return false;
            }
            *cap = need; return true;
        };
        const int C = chunk;
        const size_t qk_b = (size_t)S * H_k * hd * sizeof(float);
        const size_t v_b  = (size_t)S * H_v * hd * sizeof(float);
        const size_t sc_b = (size_t)S * H_v * sizeof(float);
        const size_t st_b = (size_t)H_v * hd * hd * sizeof(float);
        const size_t ch_b = (size_t)H_v * C * hd * sizeof(float);
        const size_t cc_b = (size_t)H_v * C * C * sizeof(float);
        const size_t hc_b = (size_t)H_v * C * sizeof(float);
        if (!grow(&d_gdn_Q,&d_gdn_Q_cap,qk_b,"Q") || !grow(&d_gdn_K,&d_gdn_K_cap,qk_b,"K") ||
            !grow(&d_gdn_V,&d_gdn_V_cap,v_b,"V") || !grow(&d_gdn_out,&d_gdn_out_cap,v_b,"out") ||
            !grow(&d_gdn_beta,&d_gdn_beta_cap,sc_b,"beta") || !grow(&d_gdn_g,&d_gdn_g_cap,sc_b,"g") ||
            !grow(&d_gdn_state,&d_gdn_state_cap,st_b,"state") ||
            !grow(&d_gdn_Qc,&d_gdn_Qc_cap,ch_b,"Qc") || !grow(&d_gdn_Kc,&d_gdn_Kc_cap,ch_b,"Kc") ||
            !grow(&d_gdn_Vc,&d_gdn_Vc_cap,ch_b,"Vc") || !grow(&d_gdn_Kcs,&d_gdn_Kcs_cap,ch_b,"Kcs") ||
            !grow(&d_gdn_R,&d_gdn_R_cap,ch_b,"R") || !grow(&d_gdn_Qs,&d_gdn_Qs_cap,ch_b,"Qs") ||
            !grow(&d_gdn_U0,&d_gdn_U0_cap,ch_b,"U0") || !grow(&d_gdn_U,&d_gdn_U_cap,ch_b,"U") ||
            !grow(&d_gdn_outc,&d_gdn_outc_cap,ch_b,"outc") ||
            !grow(&d_gdn_KK,&d_gdn_KK_cap,cc_b,"KK") || !grow(&d_gdn_QK,&d_gdn_QK_cap,cc_b,"QK") ||
            !grow(&d_gdn_M,&d_gdn_M_cap,cc_b,"M") || !grow(&d_gdn_P,&d_gdn_P_cap,cc_b,"P") ||
            !grow(&d_gdn_gcs,&d_gdn_gcs_cap,hc_b,"gcs") || !grow(&d_gdn_At,&d_gdn_At_cap,hc_b,"At") ||
            !grow(&d_gdn_ecum,&d_gdn_ecum_cap,hc_b,"ecum") || !grow(&d_gdn_cf,&d_gdn_cf_cap,hc_b,"cf") ||
            !grow(&d_gdn_betac,&d_gdn_betac_cap,hc_b,"betac")) return false;
        if ((size_t)H_v * sizeof(int) > d_gdn_fast_cap) {
            if (d_gdn_fast) cudaFree(d_gdn_fast);
            if (cudaMalloc((void**)&d_gdn_fast, (size_t)H_v * sizeof(int)) != cudaSuccess) return false;
            d_gdn_fast_cap = (size_t)H_v * sizeof(int);
        }

        // TRIED 2026-09-05 and REVERTED: staging these 6 small (0.5-69 MB)
        // transfers through a pinned dense_stage buffer (the SAME pattern
        // that WON for dense_matmul_bf16_host, see that function's own
        // comment) measured WORSE here -- up 0.788s -> 1.219s across 36
        // layers x 6 calls each. Unlike the ~30-115 MB single-shot dense_proj
        // transfers, these are numerous and small enough that the fixed cost
        // of an extra CPU-side memcpy (plus host_ptr_is_pinned's own
        // cudaPointerGetAttributes probe, x8 calls/layer here vs x2 there)
        // isn't amortized by anything the pinned DMA saves. Same class as
        // this file's other "a fix that won at one call site regressed at a
        // structurally different one" findings -- kept plain.
        PERF_START(gdn_up);
        if (cudaMemcpy(d_gdn_Q, Q, qk_b, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d_gdn_K, K, qk_b, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d_gdn_V, V, v_b, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d_gdn_beta, beta, sc_b, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d_gdn_g, g_log, sc_b, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d_gdn_state, state, st_b, cudaMemcpyHostToDevice) != cudaSuccess) {
            LOG_WARN("[CUDA] qwen4exp_gdn_gpu_host: H2D failed -- CPU fallback\n");
            return false;
        }
        PERF_END(gdn_up, t_qwen4exp_gdn_gpu_up);

        const float one = 1.0f, zero = 0.0f;
        const float q_scale = 1.0f / std::sqrt((float)hd);
        PERF_START(gdn_cmp);
        for (int s0 = 0; s0 < S; s0 += C) {
            const int Cc = std::min(C, S - s0);
            launch_gdn_gather(d_gdn_Q, d_gdn_K, d_gdn_V, d_gdn_Qc, d_gdn_Kc, d_gdn_Vc,
                              s0, S, Cc, hd, H_k, H_v, 0);
            launch_gdn_prep_scalars(d_gdn_beta, d_gdn_g, d_gdn_gcs, d_gdn_At, d_gdn_ecum,
                                    d_gdn_cf, d_gdn_betac, d_gdn_fast, s0, S, Cc, H_v, 0);
            // KK = Kc @ Kc^T, QK = Qc @ Kc^T   (row-major A@B^T)
            cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, Cc, Cc, hd,
                &one, d_gdn_Kc, hd, (long long)Cc*hd, d_gdn_Kc, hd, (long long)Cc*hd,
                &zero, d_gdn_KK, Cc, (long long)Cc*Cc, H_v);
            cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, Cc, Cc, hd,
                &one, d_gdn_Kc, hd, (long long)Cc*hd, d_gdn_Qc, hd, (long long)Cc*hd,
                &zero, d_gdn_QK, Cc, (long long)Cc*Cc, H_v);
            // R = Kc @ S0, Qs = Qc @ S0   (row-major A@B)
            cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, hd, Cc, hd,
                &one, d_gdn_state, hd, (long long)hd*hd, d_gdn_Kc, hd, (long long)Cc*hd,
                &zero, d_gdn_R, hd, (long long)Cc*hd, H_v);
            cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, hd, Cc, hd,
                &one, d_gdn_state, hd, (long long)hd*hd, d_gdn_Qc, hd, (long long)Cc*hd,
                &zero, d_gdn_Qs, hd, (long long)Cc*hd, H_v);
            launch_gdn_build_M_rhs(d_gdn_KK, d_gdn_Vc, d_gdn_R, d_gdn_gcs, d_gdn_ecum,
                                   d_gdn_At, d_gdn_betac, d_gdn_fast, d_gdn_M, d_gdn_U0,
                                   Cc, hd, H_v, 0);
            if (!launch_gdn_forward_subst(d_gdn_M, d_gdn_U0, d_gdn_U, Cc, hd, H_v, 0)) {
                LOG_WARN("[CUDA] qwen4exp_gdn_gpu_host: forward-substitution needs %.0f KB "
                         "shared memory, more than this device allows -- CPU fallback\n",
                         (double)Cc * hd * sizeof(float) / 1024.0);
                return false;
            }
            launch_gdn_build_P_out(d_gdn_QK, d_gdn_Qs, d_gdn_gcs, d_gdn_ecum, d_gdn_At,
                                   d_gdn_betac, d_gdn_fast, d_gdn_P, d_gdn_outc,
                                   Cc, hd, H_v, q_scale, 0);
            // outc += P @ U
            cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, hd, Cc, Cc,
                &one, d_gdn_U, hd, (long long)Cc*hd, d_gdn_P, Cc, (long long)Cc*Cc,
                &one, d_gdn_outc, hd, (long long)Cc*hd, H_v);
            launch_gdn_scale_state(d_gdn_Kc, d_gdn_cf, d_gdn_At, d_gdn_state, d_gdn_Kcs,
                                   Cc, hd, H_v, 0);
            // state += Kcs^T @ U   (row-major A^T@B)
            cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, hd, hd, Cc,
                &one, d_gdn_U, hd, (long long)Cc*hd, d_gdn_Kcs, hd, (long long)Cc*hd,
                &one, d_gdn_state, hd, (long long)hd*hd, H_v);
            launch_gdn_scatter(d_gdn_outc, d_gdn_out, s0, S, Cc, hd, H_v, 0);
        }
        if (cudaStreamSynchronize(0) != cudaSuccess) {
            LOG_WARN("[CUDA] qwen4exp_gdn_gpu_host: sync failed -- CPU fallback\n");
            return false;
        }
        PERF_END(gdn_cmp, t_qwen4exp_gdn_gpu_cmp);

        PERF_START(gdn_dn);
        if (cudaMemcpy(out, d_gdn_out, v_b, cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(state, d_gdn_state, st_b, cudaMemcpyDeviceToHost) != cudaSuccess) {
            LOG_WARN("[CUDA] qwen4exp_gdn_gpu_host: D2H failed -- CPU fallback\n");
            return false;
        }
        PERF_END(gdn_dn, t_qwen4exp_gdn_gpu_dn);
        perf().n_qwen4exp_gdn_gpu_calls.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    // GDN causal short conv + SiLU on the GPU (EngineConfig::qwen4exp_gdn_conv_gpu).
    // See cuda/gdn_conv1d.cu for the kernel and its design rationale.
    //
    // x/out/history are all HOST pointers -- same shapes/layout as the CPU
    // reference (attn_causal_conv1d_silu): x/out are [S,channels] row-major,
    // w is [channels,kernel] row-major (the on-disk layout, no transpose
    // needed here unlike the CPU function's own SIMD-motivated transpose),
    // history is [channels,kernel-1] row-major. The history-update tail
    // (computing the new persistent state for the NEXT call) is done on the
    // HOST after the D2H, reusing the exact same formula as the CPU
    // reference byte-for-byte -- it's cheap (channels*(kernel-1) floats,
    // e.g. 123 KB at qwen4exp's real dims) and doing it here avoids a
    // second kernel plus keeps this function's correctness trivially
    // comparable to the reference it replaces.
    bool qwen4exp_conv1d_silu_gpu_host(const float* x, const float* w, float* out,
                                       int S, int channels, int kernel, float* history) {
        if (!x || !w || !out) return false;
        if (S <= 0 || channels <= 0 || kernel <= 1) return false;
        const int hist = kernel - 1;
        if (hist > 0 && !history) return false;

        cudaSetDevice(device_id);
        auto grow = [&](float** b, size_t* cap, size_t need, const char* label) -> bool {
            if (need <= *cap) return true;
            if (!vram_ok_to_grow(need - *cap)) return false;
            if (*b) { cudaFree(*b); *b = nullptr; }
            *cap = 0;
            if (cudaMalloc((void**)b, need) != cudaSuccess) {
                LOG_WARN("[CUDA] qwen4exp_conv1d_silu_gpu_host: %s grow to %.1f MB failed -- "
                         "CPU fallback\n", label, need / 1e6);
                return false;
            }
            *cap = need; return true;
        };
        const size_t x_b = (size_t)S * channels * sizeof(float);
        const size_t w_b = (size_t)channels * kernel * sizeof(float);
        const size_t h_b = (size_t)channels * hist * sizeof(float);
        if (!grow(&d_gdnconv_x, &d_gdnconv_x_cap, x_b, "x") ||
            !grow(&d_gdnconv_w, &d_gdnconv_w_cap, w_b, "w") ||
            !grow(&d_gdnconv_out, &d_gdnconv_out_cap, x_b, "out") ||
            (h_b > 0 && !grow(&d_gdnconv_h, &d_gdnconv_h_cap, h_b, "history")))
            return false;

        PERF_START(gdnconv_up);
        if (cudaMemcpy(d_gdnconv_x, x, x_b, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d_gdnconv_w, w, w_b, cudaMemcpyHostToDevice) != cudaSuccess ||
            (h_b > 0 && cudaMemcpy(d_gdnconv_h, history, h_b, cudaMemcpyHostToDevice) != cudaSuccess)) {
            LOG_WARN("[CUDA] qwen4exp_conv1d_silu_gpu_host: H2D failed -- CPU fallback\n");
            return false;
        }
        PERF_END(gdnconv_up, t_gdn_conv1d_gpu_up);

        PERF_START(gdnconv_cmp);
        launch_gdn_conv1d_silu(d_gdnconv_x, d_gdnconv_w, d_gdnconv_h, d_gdnconv_out,
                               S, channels, kernel, 0);
        if (cudaStreamSynchronize(0) != cudaSuccess) {
            LOG_WARN("[CUDA] qwen4exp_conv1d_silu_gpu_host: sync failed -- CPU fallback\n");
            return false;
        }
        PERF_END(gdnconv_cmp, t_gdn_conv1d_gpu_cmp);

        PERF_START(gdnconv_dn);
        if (cudaMemcpy(out, d_gdnconv_out, x_b, cudaMemcpyDeviceToHost) != cudaSuccess) {
            LOG_WARN("[CUDA] qwen4exp_conv1d_silu_gpu_host: D2H failed -- CPU fallback\n");
            return false;
        }
        PERF_END(gdnconv_dn, t_gdn_conv1d_gpu_dn);

        // History update, host-side, byte-identical to the CPU reference's
        // own tail (attn_causal_conv1d_silu, src/attention_layer.cpp): the
        // new history is the last `hist` positions of [old history ++ x].
        if (hist > 0) {
            static thread_local std::vector<float> new_hist;
            new_hist.resize((size_t)channels * hist);
            for (int c = 0; c < channels; ++c) {
                const float* h = history + (size_t)c * hist;
                for (int j = 0; j < hist; ++j) {
                    const int pos = S + j;
                    new_hist[(size_t)c * hist + j] =
                        (pos < hist) ? h[pos] : x[(size_t)(pos - hist) * channels + c];
                }
            }
            std::memcpy(history, new_hist.data(), (size_t)channels * hist * sizeof(float));
        }

        perf().n_gdn_conv1d_gpu_calls.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    // GLM5NEXT KDA recurrent delta rule on CUDA. GLM's KDA differs from
    // Qwen4Exp GDN: decay is a vector over the key channels and the state is
    // [H,hd,hd]. The CUDA kernel keeps the token loop sequential (the state
    // dependency is real) while parallelising all heads/state elements.
    //
    // This host bridge intentionally copies the persistent state in and out
    // on every call. That is a little more PCIe traffic than a device-only
    // state, but it preserves the existing prompt-cache and generation
    // checkpoint/rollback semantics without introducing a second mutable
    // state store that could diverge from StreamingEngine.
    bool glm5next_kda_delta_rule_gpu_host(const float* Q, const float* K, const float* V,
                                          const float* beta, const float* g_log,
                                          float* state, float* out,
                                          int S, int H, int hd) {
        if (!Q || !K || !V || !beta || !g_log || !state || !out)
            return false;
        if (S <= 0 || H <= 0 || hd <= 0 || hd > 1024)
            return false;

        cudaSetDevice(device_id);
        auto grow = [&](float** b, size_t* cap, size_t need, const char* label) -> bool {
            if (need <= *cap) return true;
            if (!vram_ok_to_grow(need - *cap)) {
                LOG_WARN("[CUDA] glm5next_kda_delta_rule_gpu_host: %s grow to %.1f MB "
                         "blocked by VRAM watermark -- CPU fallback\n",
                         label, need / 1e6);
                return false;
            }
            if (*b) { cudaFree(*b); *b = nullptr; }
            *cap = 0;
            cudaError_t e = cudaMalloc((void**)b, need);
            if (e != cudaSuccess) {
                LOG_WARN("[CUDA] glm5next_kda_delta_rule_gpu_host: %s grow to %.1f MB "
                         "failed: %s -- CPU fallback\n",
                         label, need / 1e6, cudaGetErrorString(e));
                return false;
            }
            *cap = need;
            return true;
        };

        const size_t qkv_bytes = (size_t)S * H * hd * sizeof(float);
        const size_t beta_bytes = (size_t)S * H * sizeof(float);
        const size_t state_bytes = (size_t)H * hd * hd * sizeof(float);
        if (!grow(&d_glm_kda_q, &d_glm_kda_q_cap, qkv_bytes, "Q") ||
            !grow(&d_glm_kda_k, &d_glm_kda_k_cap, qkv_bytes, "K") ||
            !grow(&d_glm_kda_v, &d_glm_kda_v_cap, qkv_bytes, "V") ||
            !grow(&d_glm_kda_beta, &d_glm_kda_beta_cap, beta_bytes, "beta") ||
            !grow(&d_glm_kda_g, &d_glm_kda_g_cap, qkv_bytes, "g_log") ||
            !grow(&d_glm_kda_state, &d_glm_kda_state_cap, state_bytes, "state") ||
            !grow(&d_glm_kda_out, &d_glm_kda_out_cap, qkv_bytes, "out"))
            return false;

        PERF_START(glm_kda_up);
        const bool copied =
            cudaMemcpy(d_glm_kda_q, Q, qkv_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(d_glm_kda_k, K, qkv_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(d_glm_kda_v, V, qkv_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(d_glm_kda_beta, beta, beta_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(d_glm_kda_g, g_log, qkv_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(d_glm_kda_state, state, state_bytes, cudaMemcpyHostToDevice) == cudaSuccess;
        PERF_END(glm_kda_up, t_glm5next_kda_gpu_up);
        if (!copied) {
            LOG_WARN("[CUDA] glm5next_kda_delta_rule_gpu_host: H2D failed -- CPU fallback\n");
            return false;
        }

        PERF_START(glm_kda_cmp);
        launch_glm5next_kda_delta_rule(
            d_glm_kda_q, d_glm_kda_k, d_glm_kda_v, d_glm_kda_beta, d_glm_kda_g,
            d_glm_kda_state, d_glm_kda_out, S, H, hd, 0);
        cudaError_t launch_err = cudaGetLastError();
        cudaError_t sync_err = (launch_err == cudaSuccess)
                                  ? cudaStreamSynchronize(0) : launch_err;
        PERF_END(glm_kda_cmp, t_glm5next_kda_gpu_cmp);
        if (sync_err != cudaSuccess) {
            LOG_WARN("[CUDA] glm5next_kda_delta_rule_gpu_host: kernel/sync failed: %s "
                     "-- CPU fallback\n", cudaGetErrorString(sync_err));
            return false;
        }

        PERF_START(glm_kda_dn);
        const bool copied_back =
            cudaMemcpy(out, d_glm_kda_out, qkv_bytes, cudaMemcpyDeviceToHost) == cudaSuccess &&
            cudaMemcpy(state, d_glm_kda_state, state_bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
        PERF_END(glm_kda_dn, t_glm5next_kda_gpu_dn);
        if (!copied_back) {
            LOG_WARN("[CUDA] glm5next_kda_delta_rule_gpu_host: D2H failed -- CPU fallback\n");
            return false;
        }
        perf().n_glm5next_kda_gpu_calls.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    // Production GLM5NEXT KDA path with a persistent device-resident state.
    // The recurrent update is already one fused CUDA kernel; the important
    // distinction from glm5next_kda_delta_rule_gpu_host() is that `state` is
    // seeded only on the first call for a layer (or after reset/restore) and
    // is never copied back after each token batch. This removes the largest
    // avoidable PCIe transfer from every KDA layer invocation while retaining
    // the host vector as the checkpoint/prompt-cache authority at explicit
    // synchronization points.
    bool glm5next_kda_delta_rule_gpu_device(const float* Q, const float* K, const float* V,
                                            const float* beta, const float* g_log,
                                            float* host_state, float* out,
                                            int layer_idx, int S, int H, int hd,
                                            bool reset_state) {
        if (!Q || !K || !V || !beta || !g_log || !host_state || !out)
            return false;
        if (layer_idx < 0 || S <= 0 || H <= 0 || hd <= 0 || hd > 1024)
            return false;

        cudaSetDevice(device_id);
        auto grow = [&](float** b, size_t* cap, size_t need, const char* label) -> bool {
            if (need <= *cap) return true;
            if (!vram_ok_to_grow(need - *cap)) {
                LOG_WARN("[CUDA] glm5next_kda_delta_rule_gpu_device: %s grow to %.1f MB "
                         "blocked by VRAM watermark -- CPU fallback\n",
                         label, need / 1e6);
                return false;
            }
            if (*b) { cudaFree(*b); *b = nullptr; }
            *cap = 0;
            cudaError_t e = cudaMalloc((void**)b, need);
            if (e != cudaSuccess) {
                LOG_WARN("[CUDA] glm5next_kda_delta_rule_gpu_device: %s grow to %.1f MB "
                         "failed: %s -- CPU fallback\n",
                         label, need / 1e6, cudaGetErrorString(e));
                return false;
            }
            *cap = need;
            return true;
        };

        const size_t qkv_bytes = (size_t)S * H * hd * sizeof(float);
        const size_t beta_bytes = (size_t)S * H * sizeof(float);
        const size_t state_bytes = (size_t)H * hd * hd * sizeof(float);
        if ((size_t)layer_idx >= d_glm_kda_state_by_layer.size())
            d_glm_kda_state_by_layer.resize((size_t)layer_idx + 1);
        auto& device_state = d_glm_kda_state_by_layer[(size_t)layer_idx];
        if (reset_state) {
            // The caller has already reset the host mirror for a new
            // sequence. Do not copy the previous conversation back over that
            // zero seed if a best-effort CUDA allocation fails below.
            device_state.initialized = false;
        }

        // If a best-effort GPU allocation/copy fails after a previous GPU
        // update, restore the host mirror before attention falls back to its
        // scalar CPU recurrence. Without this, the fallback would continue
        // from the host's state from the beginning of the request.
        auto sync_back_on_failure = [&]() {
            if (device_state.data && device_state.initialized) {
                cudaMemcpy(host_state, device_state.data, state_bytes, cudaMemcpyDeviceToHost);
            }
            // The caller will execute the CPU recurrence after a false
            // return. Its host result must become the next device seed;
            // otherwise the next GPU call would continue from the pre-failure
            // device matrix and silently discard the CPU fallback update.
            device_state.initialized = false;
        };

        if (!grow(&d_glm_kda_q, &d_glm_kda_q_cap, qkv_bytes, "Q") ||
            !grow(&d_glm_kda_k, &d_glm_kda_k_cap, qkv_bytes, "K") ||
            !grow(&d_glm_kda_v, &d_glm_kda_v_cap, qkv_bytes, "V") ||
            !grow(&d_glm_kda_beta, &d_glm_kda_beta_cap, beta_bytes, "beta") ||
            !grow(&d_glm_kda_g, &d_glm_kda_g_cap, qkv_bytes, "g_log") ||
            !grow(&d_glm_kda_out, &d_glm_kda_out_cap, qkv_bytes, "out")) {
            sync_back_on_failure();
            return false;
        }
        if (!grow(&device_state.data, &device_state.cap, state_bytes, "persistent KDA state")) {
            sync_back_on_failure();
            return false;
        }

        PERF_START(glm_kda_up);
        const bool copied_inputs =
            cudaMemcpy(d_glm_kda_q, Q, qkv_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(d_glm_kda_k, K, qkv_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(d_glm_kda_v, V, qkv_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(d_glm_kda_beta, beta, beta_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(d_glm_kda_g, g_log, qkv_bytes, cudaMemcpyHostToDevice) == cudaSuccess;
        // A reset or an invalidated state means the host mirror is the new
        // authoritative seed (zero for a fresh sequence, or restored prompt
        // state for a prompt-cache hit). Otherwise the device state already
        // contains the result of all preceding KDA calls for this layer.
        const bool seed_state = reset_state || !device_state.initialized;
        if (copied_inputs && seed_state) {
            if (cudaMemcpy(device_state.data, host_state, state_bytes,
                           cudaMemcpyHostToDevice) != cudaSuccess) {
                cudaGetLastError();
                PERF_END(glm_kda_up, t_glm5next_kda_gpu_up);
                LOG_WARN("[CUDA] glm5next_kda_delta_rule_gpu_device: state seed H2D failed "
                         "-- CPU fallback\n");
                return false;
            }
        }
        PERF_END(glm_kda_up, t_glm5next_kda_gpu_up);
        if (!copied_inputs) {
            sync_back_on_failure();
            LOG_WARN("[CUDA] glm5next_kda_delta_rule_gpu_device: H2D failed -- CPU fallback\n");
            return false;
        }

        PERF_START(glm_kda_cmp);
        launch_glm5next_kda_delta_rule(
            d_glm_kda_q, d_glm_kda_k, d_glm_kda_v, d_glm_kda_beta, d_glm_kda_g,
            device_state.data, d_glm_kda_out, S, H, hd, 0);
        cudaError_t launch_err = cudaGetLastError();
        cudaError_t sync_err = (launch_err == cudaSuccess)
                                  ? cudaStreamSynchronize(0) : launch_err;
        PERF_END(glm_kda_cmp, t_glm5next_kda_gpu_cmp);
        if (sync_err != cudaSuccess) {
            // The kernel may have partially updated the state. Copy the
            // device contents to the host before the caller enters its CPU
            // fallback; this preserves a coherent continuation even when a
            // driver reports an error after launching the kernel.
            device_state.initialized = true;
            sync_back_on_failure();
            LOG_WARN("[CUDA] glm5next_kda_delta_rule_gpu_device: kernel/sync failed: %s "
                     "-- state synchronized before CPU fallback\n",
                     cudaGetErrorString(sync_err));
            return false;
        }

        PERF_START(glm_kda_dn);
        const cudaError_t out_err =
            cudaMemcpy(out, d_glm_kda_out, qkv_bytes, cudaMemcpyDeviceToHost);
        PERF_END(glm_kda_dn, t_glm5next_kda_gpu_dn);
        if (out_err != cudaSuccess) {
            sync_back_on_failure();
            LOG_WARN("[CUDA] glm5next_kda_delta_rule_gpu_device: output D2H failed: %s "
                     "-- CPU fallback\n", cudaGetErrorString(out_err));
            return false;
        }
        device_state.initialized = true;
        perf().n_glm5next_kda_gpu_calls.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    // Explicit state boundary used by prompt-cache save/checkpoint code. It
    // is intentionally not called from every KDA layer invocation.
    bool glm5next_kda_sync_state_to_host(int layer_idx, float* host_state, size_t elements) {
        if (layer_idx < 0 || !host_state || elements == 0 ||
            (size_t)layer_idx >= d_glm_kda_state_by_layer.size())
            return true; // no device state means the host mirror is current
        auto& device_state = d_glm_kda_state_by_layer[(size_t)layer_idx];
        if (!device_state.data || !device_state.initialized) return true;
        cudaSetDevice(device_id);
        const size_t bytes = elements * sizeof(float);
        cudaError_t err = cudaMemcpy(host_state, device_state.data, bytes,
                                     cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            LOG_WARN("[CUDA] glm5next_kda_sync_state_to_host: layer=%d D2H failed: %s\n",
                     layer_idx, cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    // Host prompt-cache restore replaces the authoritative state. The next
    // KDA call must seed the device from that restored host vector instead of
    // continuing from the prior conversation.
    void glm5next_kda_invalidate_device_states() {
        for (auto& state : d_glm_kda_state_by_layer) state.initialized = false;
    }

    // GLM5NEXT learned router on CUDA. This is deliberately a small host
    // bridge rather than a GGML graph: the router weight is already resident
    // on the core GPU, the dense matmul stays on that GPU, and only the
    // compact [token, top-k] route table is copied back for the established
    // expert cache/prefetch machinery. The CPU fallback remains available if
    // the router weight was not offloaded or any CUDA operation fails.
    bool glm5next_router_gpu_host(const QuantizedTensor& router,
                                  const float* normed, const float* bias,
                                  int S, int D, int n_experts, int topk,
                                  GatingFunc gating_func, bool normalize,
                                  float weight_scale, int32_t* ids_host,
                                  float* weights_host) {
        if (!router.device_ptr || !normed || S <= 0 || D <= 0 ||
            n_experts <= 0 || topk <= 0 || topk > n_experts ||
            !ids_host || !weights_host)
            return false;

        cudaSetDevice(device_id);
        auto grow = [&](void** b, size_t* cap, size_t need,
                        const char* label) -> bool {
            if (need <= *cap) return true;
            if (!vram_ok_to_grow(need - *cap)) {
                LOG_WARN("[CUDA] glm5next_router_gpu_host: %s grow to %.1f MB "
                         "blocked by VRAM watermark -- CPU fallback\n",
                         label, need / 1e6);
                return false;
            }
            if (*b) { cudaFree(*b); *b = nullptr; }
            *cap = 0;
            cudaError_t e = cudaMalloc(b, need);
            if (e != cudaSuccess) {
                LOG_WARN("[CUDA] glm5next_router_gpu_host: %s grow to %.1f MB "
                         "failed: %s -- CPU fallback\n",
                         label, need / 1e6, cudaGetErrorString(e));
                return false;
            }
            *cap = need;
            return true;
        };

        const size_t x_bytes = (size_t)S * D * sizeof(float);
        const size_t logits_bytes = (size_t)S * n_experts * sizeof(float);
        const size_t bias_bytes = (size_t)n_experts * sizeof(float);
        const size_t route_count = (size_t)S * topk;
        if (!grow(reinterpret_cast<void**>(&d_glm_router_x),
                  &d_glm_router_x_cap, x_bytes, "activation") ||
            !grow(reinterpret_cast<void**>(&d_glm_router_logits),
                  &d_glm_router_logits_cap, logits_bytes, "logits") ||
            !grow(reinterpret_cast<void**>(&d_glm_router_ids),
                  &d_glm_router_ids_cap, route_count * sizeof(int32_t), "ids") ||
            !grow(reinterpret_cast<void**>(&d_glm_router_weights),
                  &d_glm_router_weights_cap, route_count * sizeof(float), "weights"))
            return false;
        if (bias && !grow(reinterpret_cast<void**>(&d_glm_router_bias),
                          &d_glm_router_bias_cap, bias_bytes, "bias"))
            return false;

        PERF_START(glm_router_up);
        cudaError_t h2d = cudaMemcpy(d_glm_router_x, normed, x_bytes,
                                     cudaMemcpyHostToDevice);
        if (h2d == cudaSuccess && bias)
            h2d = cudaMemcpy(d_glm_router_bias, bias, bias_bytes,
                             cudaMemcpyHostToDevice);
        PERF_END(glm_router_up, t_glm5next_router_gpu_up);
        if (h2d != cudaSuccess) {
            LOG_WARN("[CUDA] glm5next_router_gpu_host: H2D failed: %s -- "
                     "CPU fallback\n", cudaGetErrorString(h2d));
            return false;
        }

        PERF_START(glm_router_cmp);
        if (!dense_matmul_device(router.meta.type, router.device_ptr,
                                 d_glm_router_x, d_glm_router_logits,
                                 S, D, n_experts)) {
            PERF_END(glm_router_cmp, t_glm5next_router_gpu_cmp);
            LOG_WARN("[CUDA] glm5next_router_gpu_host: router matmul is not "
                     "supported on device -- CPU fallback\n");
            return false;
        }
        const int gating = static_cast<int>(gating_func);
        launch_glm5next_router_topk(
            d_glm_router_logits, bias ? d_glm_router_bias : nullptr,
            d_glm_router_ids, d_glm_router_weights, S, n_experts, topk,
            gating, normalize ? 1 : 0, weight_scale, 0);
        cudaError_t launch_err = cudaGetLastError();
        cudaError_t sync_err = (launch_err == cudaSuccess)
                                  ? cudaStreamSynchronize(0) : launch_err;
        PERF_END(glm_router_cmp, t_glm5next_router_gpu_cmp);
        if (sync_err != cudaSuccess) {
            LOG_WARN("[CUDA] glm5next_router_gpu_host: matmul/router failed: %s "
                     "-- CPU fallback\n", cudaGetErrorString(sync_err));
            return false;
        }

        PERF_START(glm_router_dn);
        cudaError_t d2h =
            cudaMemcpy(ids_host, d_glm_router_ids,
                       route_count * sizeof(int32_t), cudaMemcpyDeviceToHost);
        if (d2h == cudaSuccess)
            d2h = cudaMemcpy(weights_host, d_glm_router_weights,
                             route_count * sizeof(float), cudaMemcpyDeviceToHost);
        PERF_END(glm_router_dn, t_glm5next_router_gpu_dn);
        if (d2h != cudaSuccess) {
            LOG_WARN("[CUDA] glm5next_router_gpu_host: D2H route readback failed: "
                     "%s -- CPU fallback\n", cudaGetErrorString(d2h));
            return false;
        }
        perf().n_glm5next_router_gpu_calls.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    // DSV4 raw-SWA/HCA score fast-path, part 3: per-token dispatch, decode
    // (S==1) only -- same convention as dsv4_outproj_gpu's own S==1 gate in
    // attention_layer.cpp (prefill's per-row-varying past_context_len is
    // out of scope for this first cut, same reasoning as attn_score_gpu's
    // own decode-only note). Computes ONE joint softmax over [raw window
    // rows (ring-order, oldest-to-newest) | HCA compressed rows (block 0..
    // hca_visible)] plus an optional per-head sink logit -- see
    // dsv4_score_softmax.cu's header for why this must be one joint
    // softmax, not two merged ones, and why CSA (ratio==4) layers never
    // call this.
    //
    // The raw window's ring buffer can be split across the buffer wrap
    // point (oldest_slot + past_context_len > window once the ring has
    // filled, which is the STEADY-STATE case after the first `window`
    // tokens, not an edge case) -- rather than a wrap-aware GEMM, this
    // compacts both the (possibly-wrapped) raw window and the HCA prefix
    // into one contiguous [T,qh] scratch buffer via up to 3
    // device-to-device memcpys, then reuses exactly score_gpu_batched_
    // host's own 1-GEMM+softmax+1-GEMM shape -- simpler here than MLA score
    // since DSV4's raw K==V has no separate nope/rope latent to combine.
    //
    // Q_host: [H,qh] row-major F32, host (the SAME per-head Q slice the CPU
    //   loop's `head_q` already computes -- caller passes it unchanged).
    // sinks_host: [H] F32 or nullptr (no attention-sink tensor for this
    //   layer).
    // context_host: [H,qh] row-major F32, out -- this IS head_contexts'
    //   per-token slice; caller writes it straight in, no reshuffle, still
    //   applies the same de-rope step afterward as the CPU path does.
    // Returns false (caller falls back to the CPU per-head loop, unchanged)
    // if either mirror isn't ready/doesn't cover this call's range, or any
    // CUDA/cublas call fails -- never a hard requirement.
    bool dsv4_score_gpu_batched_host(
            int layer_idx, const float* Q_host, int H, int qh,
            int window, int past_context_len, int oldest_slot,
            int hca_visible, const float* sinks_host, float scale,
            float* context_host) {
        if (layer_idx < 0 || (size_t)layer_idx >= d_dsv4_kvc_by_layer.size()) return false;
        float* d_kvc = d_dsv4_kvc_by_layer[layer_idx];
        if (!d_kvc || H <= 0 || qh <= 0 || past_context_len <= 0) return false;
        if (past_context_len > dsv4_kvc_window_by_layer[layer_idx]) return false;
        if (oldest_slot < 0 || oldest_slot >= window) return false;

        float* d_hca = nullptr;
        if (hca_visible > 0) {
            if ((size_t)layer_idx >= d_dsv4_hca_by_layer.size()) return false;
            d_hca = d_dsv4_hca_by_layer[layer_idx];
            if (!d_hca || hca_visible > dsv4_hca_capacity_by_layer[layer_idx]) return false;
        }

        const int T = past_context_len + hca_visible;

        cudaSetDevice(device_id);
        auto grow = [&](float** buf, size_t* cap, size_t need_bytes, const char* label) -> bool {
            if (need_bytes <= *cap) return true;
            if (*buf) { cudaFree(*buf); *buf = nullptr; }
            *cap = 0;
            if (cudaMalloc((void**)buf, need_bytes) != cudaSuccess) {
                LOG_ERROR("[CUDA] dsv4_score_gpu_batched_host: %s grow to %.2f MB failed\n", label, need_bytes / 1e6);
                return false;
            }
            *cap = need_bytes;
            return true;
        };
        if (!grow(&d_dsv4_score_q, &d_dsv4_score_q_cap, (size_t)H * qh * sizeof(float), "d_dsv4_score_q")) return false;
        if (!grow(&d_dsv4_score_kv, &d_dsv4_score_kv_cap, (size_t)T * qh * sizeof(float), "d_dsv4_score_kv")) return false;
        if (!grow(&d_dsv4_score_scores, &d_dsv4_score_scores_cap, (size_t)H * T * sizeof(float), "d_dsv4_score_scores")) return false;
        if (!grow(&d_dsv4_score_context, &d_dsv4_score_context_cap, (size_t)H * qh * sizeof(float), "d_dsv4_score_context")) return false;
        if (sinks_host && !grow(&d_dsv4_score_sinks, &d_dsv4_score_sinks_cap, (size_t)H * sizeof(float), "d_dsv4_score_sinks")) return false;

        const float alpha = 1.0f, beta = 0.0f;
        PERF_START(dsv4_score_gpu_up);
        // Compact the (possibly-wrapped) raw window into
        // d_dsv4_score_kv[0:past_context_len), then append the HCA prefix.
        const int tail_len = std::min(past_context_len, window - oldest_slot);
        const int head_len = past_context_len - tail_len;
        cudaMemcpy(d_dsv4_score_kv, d_kvc + (size_t)oldest_slot * qh,
                   (size_t)tail_len * qh * sizeof(float), cudaMemcpyDeviceToDevice);
        if (head_len > 0) {
            cudaMemcpy(d_dsv4_score_kv + (size_t)tail_len * qh, d_kvc,
                       (size_t)head_len * qh * sizeof(float), cudaMemcpyDeviceToDevice);
        }
        if (hca_visible > 0) {
            cudaMemcpy(d_dsv4_score_kv + (size_t)past_context_len * qh, d_hca,
                       (size_t)hca_visible * qh * sizeof(float), cudaMemcpyDeviceToDevice);
        }
        cudaMemcpy(d_dsv4_score_q, Q_host, (size_t)H * qh * sizeof(float), cudaMemcpyHostToDevice);
        if (sinks_host) {
            cudaMemcpy(d_dsv4_score_sinks, sinks_host, (size_t)H * sizeof(float), cudaMemcpyHostToDevice);
        }
        PERF_END(dsv4_score_gpu_up, t_dsv4_score_gpu_up);

        PERF_START(dsv4_score_gpu_cmp);
        // scores[H,T] = Q[H,qh] @ kv_combined[T,qh]^T -- same "Y=X@W^T"
        // pattern as score_gpu_batched_host's own nope/rope GEMMs
        // (kv_combined plays the role of a [T,qh] weight, shared across
        // every head since DSV4's raw/HCA cache is MQA-style K==V, one
        // copy for all H heads).
        cublasStatus_t st1 = cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, T, H, qh, &alpha,
                                          d_dsv4_score_kv, qh, d_dsv4_score_q, qh, &beta, d_dsv4_score_scores, T);
        if (st1 != CUBLAS_STATUS_SUCCESS) { PERF_END(dsv4_score_gpu_cmp, t_dsv4_score_gpu_compute); return false; }

        launch_dsv4_score_softmax(d_dsv4_score_scores, sinks_host ? d_dsv4_score_sinks : nullptr, H, T, scale, 0);

        // context[H,qh] = attn_weights[H,T] @ kv_combined[T,qh] -- plain
        // row-major A@B, same operand-swap trick as score_gpu_batched_
        // host's own st3 GEMM.
        cublasStatus_t st2 = cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, qh, H, T, &alpha,
                                          d_dsv4_score_kv, qh, d_dsv4_score_scores, T, &beta, d_dsv4_score_context, qh);
        PERF_END(dsv4_score_gpu_cmp, t_dsv4_score_gpu_compute);
        if (st2 != CUBLAS_STATUS_SUCCESS) return false;

        PERF_START(dsv4_score_gpu_dn);
        cudaMemcpy(context_host, d_dsv4_score_context, (size_t)H * qh * sizeof(float), cudaMemcpyDeviceToHost);
        cudaStreamSynchronize(0);
        PERF_END(dsv4_score_gpu_dn, t_dsv4_score_gpu_down);
        perf().n_dsv4_score_gpu_calls.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    // Allocate expert VRAM cache on vault GPU(s) based on model geometry
    void setup_expert_cache(size_t expert_slab_bytes, int n_experts, const EngineConfig& config,
                            const std::vector<size_t>& layer_slab_bytes = {}) {
        topology.allocate_expert_cache(expert_slab_bytes, n_experts, &config, layer_slab_bytes);

        // Three-tier caching: Allocate active layer area for temporary expert storage
        // This provides a workspace for current-layer experts without evicting LRU residents
        if (config.use_active_layer_area) {
            // 2026-08-28: build the multi-GPU request list from config.gpus
            // (any [[gpus]] entry with active_layer_weight > 0 opts in) --
            // allocate_active_layer_area_multi itself falls back to the
            // plain single-GPU path when fewer than 2 entries qualify, so
            // this call is safe/inert for any config not using the feature.
            std::vector<GpuTopology::ActiveLayerGpuRequest> al_requests;
            for (auto& gc : config.gpus) {
                if (gc.active_layer_weight > 0) {
                    al_requests.push_back({gc.device_id, gc.active_layer_weight, gc.active_layer_capacity_slots});
                }
            }
            if (!topology.allocate_active_layer_area_multi(expert_slab_bytes, config.active_layer_capacity,
                                                             al_requests)) {
                LOG_STREAM_WARN() << "[GPU] Active layer area allocation failed - "
                                   "three-tier caching will fall back to standard LRU behavior\n";
            } else {
                LOG_STREAM_INFO() << "[GPU] Three-tier: active layer area enabled for temporary expert storage\n";
            }
        } else {
            LOG_STREAM_INFO() << "[GPU] Three-tier: active layer area disabled by configuration\n";
        }
    }

    // Grow-only persistent device buffers for matmul I/O. Returns false if
    // a grow was needed and the cudaMalloc failed -- previously unchecked,
    // meaning a failed grow silently left d_x/d_y null (or, worse, freed
    // but with a stale cap that made a LATER same-size call skip growing
    // again) while every caller proceeded to memcpy/kernel-launch into it
    // regardless. That's this project's worst-case failure signature
    // (silent corruption from a bad pointer, not an obvious crash) applied
    // to a path this session's prefill-scratch-sizing investigation showed
    // really can grow by hundreds of MB for a long prompt. On failure, the
    // cap is reset to 0 (not left at the stale pre-failure value) so the
    // NEXT call retries the allocation cleanly instead of wrongly assuming
    // a null buffer is already sized.
    // --- Pinned staging for dense_matmul_host's H2D/D2H -------------------
    // dense_matmul_host used a plain blocking cudaMemcpy straight from the
    // caller's host buffer. Most dense_proj callers hand it an ordinary
    // std::vector, i.e. PAGEABLE memory -- and a pageable H2D cannot DMA
    // directly, so the driver stages it through an internal pinned bounce
    // buffer synchronously. MEASURED 2026-09-05 on qwen4exp prefill:
    // dense_gpu up=1.50s + dn=0.83s over 216 calls, ~0.65 GB/s against this
    // box's 12.3 GB/s link -- 19x below the wire, i.e. essentially all
    // overhead. Staging through our OWN persistent pinned buffer turns it
    // into (CPU memcpy at ~10 GB/s) + (real DMA at 12.3 GB/s).
    //
    // A caller that ALREADY hands us pinned memory (LayerScratch::normed and
    // friends are PinnedFloatBuffer) skips the staging copy entirely -- hence
    // the cudaPointerGetAttributes probe rather than staging unconditionally.
    // Deliberately NOT cudaHostRegister on the caller's buffer: this project
    // has already measured that failing at scale on this box ("part or all of
    // the requested memory range is already mapped") once ram_pinned_budget_gb
    // has registered enough of the address space.
    static bool host_ptr_is_pinned(const void* p) {
        cudaPointerAttributes attr{};
        if (cudaPointerGetAttributes(&attr, p) != cudaSuccess) {
            cudaGetLastError();  // probing an unregistered pointer is not an error we care about
            return false;
        }
        return attr.type == cudaMemoryTypeHost;
    }

    // Grow one of the small host-side BF16 conversion scratch arrays while
    // charging its actual capacity to the same process-wide host ledger as
    // RAM slabs, graph arenas, and MTP scratch. These arrays used to be plain
    // new[] allocations, so a long-prompt growth could cross host_ram_gb even
    // though the surrounding CUDA allocations were admitted.
    template <typename T>
    bool ensure_host_scratch(T *&ptr, size_t &capacity,
                             size_t &reservation_bytes, size_t count,
                             const char *name) {
        if (count <= capacity && ptr) return true;
        if (count == 0 || count > std::numeric_limits<size_t>::max() / sizeof(T))
            return false;
        const size_t bytes = count * sizeof(T);
        if (!MemoryGuard::try_reserve_host_bytes(bytes)) {
            LOG_STREAM_WARN() << "[MEM-GUARD] host scratch refused: "
                              << (name ? name : "unnamed") << " bytes="
                              << bytes << " reserved="
                              << MemoryGuard::reserved_host_bytes()
                              << " budget=" << MemoryGuard::host_budget_bytes()
                              << "\n";
            return false;
        }
        T *replacement = new (std::nothrow) T[count];
        if (!replacement) {
            MemoryGuard::release_host_bytes(bytes);
            LOG_STREAM_WARN() << "[HOST-SCRATCH] allocation failed: "
                              << (name ? name : "unnamed") << " bytes="
                              << bytes << "\n";
            return false;
        }
        if (ptr) delete[] ptr;
        MemoryGuard::release_host_bytes(reservation_bytes);
        ptr = replacement;
        capacity = count;
        reservation_bytes = bytes;
        return true;
    }

    bool ensure_dense_stage(size_t bytes) {
        if (bytes <= dense_stage_cap && dense_stage) return true;
        cudaSetDevice(device_id);
        if (bytes == 0) return false;
        // Keep the old staging buffer alive until the replacement succeeds;
        // reserving the full new size first accounts for the realloc peak.
        if (!MemoryGuard::try_reserve_host_bytes(bytes)) {
            LOG_STREAM_WARN() << "[MEM-GUARD] dense staging refused: "
                              << bytes / 1e6 << " MB\n";
            return false;
        }
        void* p = nullptr;
        if (cudaMallocHost(&p, bytes) != cudaSuccess || !p) {
            cudaGetLastError();
            MemoryGuard::release_host_bytes(bytes);
            return false;  // caller falls back to the original direct copy
        }
        if (dense_stage) {
            cudaFreeHost(dense_stage);
            MemoryGuard::release_host_bytes(dense_stage_reservation_bytes);
        }
        dense_stage = (float*)p;
        dense_stage_cap = bytes;
        dense_stage_reservation_bytes = bytes;
        return true;
    }
    void* dense_stage = nullptr;
    size_t dense_stage_cap = 0;
    size_t dense_stage_reservation_bytes = 0;

    // Dedicated, NON-BLOCKING stream for dense_matmul_host's H2D/D2H.
    // These were plain cudaMemcpy on the LEGACY DEFAULT stream, which must
    // wait for every other blocking stream on the device before it can even
    // start. MEASURED 2026-09-05: ~0.6 GB/s per call against a 12.3 GB/s
    // link -- i.e. essentially all drain, not transfer. Correctness is
    // unchanged because each copy is still followed by a synchronize before
    // anything reads it; only the device-wide wait goes away.
    cudaStream_t dense_copy_stream = nullptr;
    cudaStream_t dense_stream() {
        if (!dense_copy_stream)
            cudaStreamCreateWithFlags(&dense_copy_stream, cudaStreamNonBlocking);
        return dense_copy_stream;  // nullptr on failure == legacy stream, still correct
    }
    void dense_copy(void* dst, const void* src, size_t n, cudaMemcpyKind kind) {
        cudaStream_t st = dense_stream();
        if (!st) { cudaMemcpy(dst, src, n, kind); return; }
        cudaMemcpyAsync(dst, src, n, kind, st);
        cudaStreamSynchronize(st);
    }

    bool ensure_buffers(size_t x_bytes, size_t y_bytes) {
        cudaSetDevice(device_id);
        if (x_bytes > d_x_cap) {
            if (d_x) { cudaFree(d_x); d_x = nullptr; }
            d_x_cap = 0;
            cudaError_t err = cudaMalloc((void**)&d_x, x_bytes);
            if (err != cudaSuccess) {
                LOG_ERROR("[CUDA] ensure_buffers: d_x grow to %.2f MB failed: %s\n",
                          x_bytes / 1e6, cudaGetErrorString(err));
                return false;
            }
            d_x_cap = x_bytes;
        }
        if (y_bytes > d_y_cap) {
            if (d_y) { cudaFree(d_y); d_y = nullptr; }
            d_y_cap = 0;
            cudaError_t err = cudaMalloc((void**)&d_y, y_bytes);
            if (err != cudaSuccess) {
                LOG_ERROR("[CUDA] ensure_buffers: d_y grow to %.2f MB failed: %s\n",
                          y_bytes / 1e6, cudaGetErrorString(err));
                return false;
            }
            d_y_cap = y_bytes;
        }
        return true;
    }

    // Same grow-only pattern as ensure_buffers above, but for the DEDICATED
    // outproj staging buffers (d_outproj_x/d_outproj_y) -- kept fully
    // separate from d_x/d_y so grouped_output_proj_host never shares a
    // buffer with anything dispatched on the default stream, matching
    // outproj_stream/outproj_cublas_handle's own isolation.
    bool ensure_outproj_buffers(size_t x_bytes, size_t y_bytes) {
        cudaSetDevice(device_id);
        if (x_bytes > d_outproj_x_cap) {
            if (d_outproj_x) { cudaFree(d_outproj_x); d_outproj_x = nullptr; }
            d_outproj_x_cap = 0;
            cudaError_t err = cudaMalloc((void**)&d_outproj_x, x_bytes);
            if (err != cudaSuccess) {
                LOG_ERROR("[CUDA] ensure_outproj_buffers: d_outproj_x grow to %.2f MB failed: %s\n",
                          x_bytes / 1e6, cudaGetErrorString(err));
                return false;
            }
            d_outproj_x_cap = x_bytes;
        }
        if (y_bytes > d_outproj_y_cap) {
            if (d_outproj_y) { cudaFree(d_outproj_y); d_outproj_y = nullptr; }
            d_outproj_y_cap = 0;
            cudaError_t err = cudaMalloc((void**)&d_outproj_y, y_bytes);
            if (err != cudaSuccess) {
                LOG_ERROR("[CUDA] ensure_outproj_buffers: d_outproj_y grow to %.2f MB failed: %s\n",
                          y_bytes / 1e6, cudaGetErrorString(err));
                return false;
            }
            d_outproj_y_cap = y_bytes;
        }
        return true;
    }

    // Frees d_x/d_y/d_x_bf16 back to empty (cap=0), WITHOUT touching
    // h_x_bf16_scratch (host RAM, not VRAM -- not what this is about) or
    // any other persistent buffer. Safe to call any time between requests:
    // the next dense_proj call regrows whatever it actually needs via the
    // SAME grow-only logic above/in dense_matmul_bf16_matched_host, just
    // starting from 0 instead of a stale larger cap. Added specifically
    // because these buffers grow to fit a prefill's full S (dense/attention
    // projections run on the whole batch in one GEMM) and, without this,
    // stay that size for the rest of the session even though decode only
    // ever needs S=1 -- real, measured impact for a long prompt (~120 MB
    // at S=1000, ~240 MB at S=2000 for this project's GLM-5.2 dims) against
    // a CoreEngine GPU that can be left with under 1.2 GB free. Called from
    // generate_text right after the single prefill call completes, before
    // the decode loop begins.
    void shrink_dense_scratch() {
        cudaSetDevice(device_id);
        if (d_x) { cudaFree(d_x); d_x = nullptr; }
        if (d_y) { cudaFree(d_y); d_y = nullptr; }
        if (d_x_bf16) { cudaFree(d_x_bf16); d_x_bf16 = nullptr; }
        d_x_cap = 0;
        d_y_cap = 0;
        d_x_bf16_cap = 0;
    }

    bool ensure_pinned(size_t bytes) {
        if (bytes == 0) return false;
        if (bytes <= h_pinned_cap && h_pinned) return true;
        // Reserve before allocating while the old buffer remains live. This
        // models the real peak during growth and prevents an apparently
        // harmless resize from crossing host_ram_gb.
        if (!MemoryGuard::try_reserve_host_bytes(bytes)) {
            LOG_STREAM_WARN() << "[MEM-GUARD] CUDA pinned staging refused: "
                              << bytes / 1e6 << " MB\n";
            return false;
        }
        void* p = nullptr;
        if (cudaMallocHost(&p, bytes) != cudaSuccess || !p) {
            cudaGetLastError();
            MemoryGuard::release_host_bytes(bytes);
            return false;
        }
        if (h_pinned) {
            cudaFreeHost(h_pinned);
            MemoryGuard::release_host_bytes(h_pinned_reservation_bytes);
        }
        h_pinned = static_cast<float*>(p);
        h_pinned_cap = bytes;
        h_pinned_reservation_bytes = bytes;
        return true;
    }

    // Execute quantized matmul on core GPU: persistent buffers + pinned + streams
    void execute_matmul(const QuantizedTensor& w, const float* x_host, float* y_host, int S) {
        int I = w.meta.shape[1];
        int O = w.meta.shape[0];
        size_t x_bytes = (size_t)S * I * sizeof(float);
        size_t y_bytes = (size_t)S * O * sizeof(float);

        if (!ensure_buffers(x_bytes, y_bytes)) {
            LOG_ERROR("[CUDA] execute_matmul: buffer grow failed, aborting this call\n");
            return;
        }
        if (!ensure_pinned(std::max(x_bytes, y_bytes))) {
            LOG_STREAM_WARN() << "[MEM-GUARD] execute_matmul: pinned staging "
                              << "unavailable; aborting this GPU call safely\n";
            return;
        }

        cudaSetDevice(device_id);
        auto& core = topology.devices[topology.core_gpu >= 0 ? topology.core_gpu : 0];

        // Async H2D via pinned staging
        PERF_START(upload);
        std::memcpy(h_pinned, x_host, x_bytes);
        cudaMemcpyAsync(d_x, h_pinned, x_bytes, cudaMemcpyHostToDevice, core.copy_stream);
        cudaStreamSynchronize(core.copy_stream);
        PERF_END(upload, t_gpu_upload);
        perf().bytes_gpu_uploaded.fetch_add(x_bytes, std::memory_order_relaxed);

        // Kernel launch
        PERF_START(compute);
        launch_gguf_q4_0_dp4a(d_y, d_x, w.device_ptr, S, I, O);
        cudaStreamSynchronize(core.compute_stream);
        PERF_END(compute, t_gpu_compute);

        // D2H result
        cudaMemcpy(y_host, d_y, y_bytes, cudaMemcpyDeviceToHost);
    }

    // Unquantized dense matmul on the core GPU via cuBLAS:
    //   y[S,O] = x[S,I] @ W^T,  W stored row-major [O,I] at d_W (device).
    // Handles F32/F16/BF16 weights (output always F32); returns false for any
    // quant type so the caller routes those to the dequant-matmul kernels.
    // d_W/d_X/d_Y must be device-resident — this does no host staging. The
    // handle runs on the core compute_stream, so callers staging x on another
    // stream must order/sync first.
    bool dense_matmul(GGUFType wtype,
                      const void* d_W, const float* d_X, float* d_Y,
                      int S, int I, int O) {
        if (!cublas_handle) return false;
        return cublas_dense_matmul(cublas_handle, wtype, d_W, d_X, d_Y, S, I, O)
               == CUBLAS_STATUS_SUCCESS;
    }

    // Device-resident dispatch counterpart used by Qwen4Exp's full-attention
    // handoff. Unlike dense_matmul(), this also covers the fused quantized
    // kernels already used by dense_matmul_host(), so an attn_output tensor
    // does not need to fall back to a host round trip merely because its GGUF
    // type is quantized.
    bool dense_matmul_device(GGUFType wtype, const void *d_W,
                             const float *d_X, float *d_Y,
                             int S, int I, int O) {
        if (!d_W || !d_X || !d_Y || S <= 0 || I <= 0 || O <= 0)
            return false;
        switch (wtype) {
            case GGUFType::F32:
            case GGUFType::F16:
            case GGUFType::BF16:
                return dense_matmul(wtype, d_W, d_X, d_Y, S, I, O);
            case GGUFType::Q8_0:      launch_q8_0_matmul(d_Y, d_X, d_W, S, I, O); return true;
            case GGUFType::Q5_1:      launch_q5_1_matmul(d_Y, d_X, d_W, S, I, O); return true;
            case GGUFType::Q2_K:      launch_q2_K_matmul(d_Y, d_X, d_W, S, I, O); return true;
            case GGUFType::Q3_K:      launch_q3_K_matmul(d_Y, d_X, d_W, S, I, O); return true;
            case GGUFType::Q4_K:      launch_q4_K_matmul(d_Y, d_X, d_W, S, I, O); return true;
            case GGUFType::Q5_K:      launch_q5_K_matmul(d_Y, d_X, d_W, S, I, O); return true;
            case GGUFType::Q6_K:      launch_q6_K_matmul(d_Y, d_X, d_W, S, I, O); return true;
            case GGUFType::IQ2_XXS:   launch_iq2_xxs_matmul(d_Y, d_X, d_W, S, I, O); return true;
            case GGUFType::IQ2_XS:    launch_iq2_xs_matmul(d_Y, d_X, d_W, S, I, O); return true;
            case GGUFType::IQ2_S:     launch_iq2_s_matmul(d_Y, d_X, d_W, S, I, O); return true;
            case GGUFType::IQ3_XXS:   launch_iq3_xxs_matmul(d_Y, d_X, d_W, S, I, O); return true;
            case GGUFType::IQ4_XS:    launch_iq4_xs_matmul(d_Y, d_X, d_W, S, I, O); return true;
            case GGUFType::IQ4_NL:    launch_iq4_nl_matmul(d_Y, d_X, d_W, S, I, O); return true;
            case GGUFType::IQ1_S:     launch_iq1s_matmul(d_Y, d_X, d_W, S, I, O); return true;
            case GGUFType::IQ1_M:     launch_iq1m_matmul(d_Y, d_X, d_W, S, I, O); return true;
            case GGUFType::IQ3_S:     launch_iq3s_matmul(d_Y, d_X, d_W, S, I, O); return true;
            case GGUFType::MXFP4:     launch_mxfp4_matmul(d_Y, d_X, d_W, S, I, O); return true;
            default: return false;
        }
    }

    // Device-input, host-output bridge for the Qwen4Exp full-attention
    // projections. The input stays on CUDA through the matmul; only the
    // projection result is downloaded because the existing normalization,
    // RoPE, and KV-cache code still consumes host arrays.
    bool dense_matmul_device_host(GGUFType wtype, const void *d_W,
                                  const float *d_X, float *y_host,
                                  int S, int I, int O) {
        if (!d_W || !d_X || !y_host || S <= 0 || I <= 0 || O <= 0)
            return false;
        const size_t y_bytes = (size_t)S * O * sizeof(float);
        if (!ensure_buffers(0, y_bytes)) return false;
        cudaSetDevice(device_id);
        if (!dense_matmul_device(wtype, d_W, d_X, d_y, S, I, O))
            return false;
        if (cudaStreamSynchronize(0) != cudaSuccess) return false;
        return cudaMemcpy(y_host, d_y, y_bytes, cudaMemcpyDeviceToHost) ==
               cudaSuccess;
    }

    // Host-staged dense matmul for the forward path: uploads x[S,I] to device,
    // dispatches by weight type (cuBLAS for F32/F16/BF16, a fused dequant kernel
    // for Q8_0; Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/IQ2_*/IQ3_XXS land here as they're
    // written), downloads y[S,O]. W must already be device-resident (device_ptr set by
    // offload_tensor_to_vram). Returns false for any type without a GPU kernel
    // yet — callers must treat that as "unsupported" and NOT silently recompute.
    // Everything runs on the default stream with synchronous memcpy, so ordering
    // is correct-by-construction (no async overlap yet).
    //
    // BF16 is special-cased and delegated to dense_matmul_bf16_matched_host
    // BEFORE the generic F32 x-upload below, rather than going through the
    // dense_matmul()/cublas_dense_matmul() path like F32/F16 still do. Why:
    // cublas_dense_matmul's BF16 case calls cublasGemmEx with a MISMATCHED
    // type combination (Atype=CUDA_R_16BF weight, Btype=CUDA_R_32F
    // activation) that Test 9 (src/main_test.cpp) proved cuBLAS rejects
    // outright on real hardware (CUBLAS_STATUS_NOT_SUPPORTED) -- so any
    // dense tensor that's natively BF16 in the GGUF (e.g. a model file
    // pre-converted to BF16 to avoid runtime dequant) would silently fail
    // GPU dispatch here and fall through to the CPU path every time, with
    // no crash and no obvious signal beyond a DEBUG-level "no GPU
    // dispatch" log line. dense_matmul_bf16_matched_host uses the MATCHED
    // combination instead (weight AND activation both BF16), verified
    // correct via Test 10 on real hardware -- this reuses that same,
    // already-verified function rather than adding a second mismatched-
    // type call site.
    // Prefill GEMM path: dequantise the weight to BF16 on-device and hand the
    // whole thing to cuBLAS, instead of this project's own fused kernels.
    //
    // Those fused kernels are all GEMV-shaped -- one output row per block,
    // streaming the activation -- so each of the O blocks re-reads the entire
    // [S, I] activation matrix. That is right at S=1 (decode) and 20-60x
    // wrong at prefill batch sizes. Measured at attn_qkv's real shape
    // (S=3189, I=2560, O=10240): ours 194.41 ms vs dequant->BF16 + GemmEx
    // 3.11 ms. See cuda/dequant_bf16.cu for the full table.
    //
    // Everything stays on-device: the activation F32->BF16 conversion is a
    // kernel, NOT the host-side loop dense_matmul_bf16_matched_host uses,
    // because at S=3189 that would be a real cost rather than a rounding
    // error (measured 0.06 ms on-device).
    // 2026-09-05: generalized from Q8_0-only to any type launch_dequant_to_bf16
    // can decode. Motivation, measured: the Qwen3.8-Flash-Next UD-IQ1_M
    // checkpoint keeps only 244 of its dense tensors in Q8_0 and puts 212 in
    // Q5_K + 40 in Q6_K, so the Q8_0-only gate left more than half its dense
    // pool on the slower fused kernels -- dense_gpu cmp 0.51s (all-Q8_0
    // Q4_K_XL) vs 3.51s (this model) at the same 512-token prefill.
    // The block-size guard is per-type: Q8_0/Q5_1 are 32-element blocks,
    // the K-quants are 256.
    bool dense_matmul_bf16_host(GGUFType wtype, const void* d_W_q8_0,
                                const float* x_host,
                                float* y_host, int S, int I, int O) {
        if (!d_W_q8_0 || !cublas_handle) return false;
        const size_t blk = (wtype == GGUFType::Q8_0 || wtype == GGUFType::Q5_1) ? 32 : 256;
        if (((size_t)I % blk) != 0) return false;  // whole blocks only

        cudaSetDevice(device_id);
        const size_t x_bytes = (size_t)S * I * sizeof(float);
        const size_t y_bytes = (size_t)S * O * sizeof(float);
        const size_t xb_bytes = (size_t)S * I * sizeof(uint16_t);
        const size_t wb_bytes = (size_t)O * I * sizeof(uint16_t);
        if (!ensure_buffers(x_bytes, y_bytes)) return false;

        auto grow = [&](void*& buf, size_t& cap, size_t need, const char* what) {
            if (need <= cap) return true;
            if (buf) { cudaFree(buf); buf = nullptr; }
            cap = 0;
            cudaError_t e = cudaMalloc(&buf, need);
            if (e != cudaSuccess) {
                LOG_ERROR("[CUDA] dense_matmul_bf16_host: %s grow to %.1f MB failed: %s\n",
                          what, need / 1e6, cudaGetErrorString(e));
                return false;
            }
            cap = need;
            return true;
        };
        // VRAM budget pre-check. Widening this path from Q8_0-only to
        // Q5_K/Q6_K (2026-09-05) put 390 more GLM tensors through here, some
        // 201 MB, on a GPU already holding 23.7 of 24 GB -- so the grow()
        // below started FAILING and logging an error per call (136 in one
        // run) before falling back. Falling back is correct, but attempting
        // an allocation we can see will not fit is pure waste, and the error
        // spam buries real problems. Ask first; fall back quietly.
        if (wb_bytes > d_w_bf16_cap) {
            if (!vram_ok_to_grow(wb_bytes - d_w_bf16_cap)) {
                perf().n_dense_bf16_skipped_vram.fetch_add(1, std::memory_order_relaxed);
                return false;   // caller uses the fused kernel instead
            }
        }
        if (!grow(d_x_bf16, d_x_bf16_cap, xb_bytes, "d_x_bf16")) return false;
        if (!grow(d_w_bf16, d_w_bf16_cap, wb_bytes, "d_w_bf16")) return false;

        // TRIED 2026-09-05 and REVERTED: staged x_host/y_host through the
        // SAME pinned dense_stage pattern that helps dense_matmul_host's own
        // fallback path (small, numerous transfers). REGRESSED here instead
        // -- a real, position-controlled A/B (throwaway warmup immediately
        // before every measured run, both arms, to rule out the sequence-
        // position warmup effect documented below) found this OFF beats ON
        // by ~8% cleanly, 3 runs each, no overlap: 23.68-23.88s (off) vs
        // 25.62-25.75s (on). Root cause: this call site's transfers are FEW
        // and LARGE (attn_qkv's output alone is S*conv_dim*4 = ~116 MB at a
        // 2824-token prefill), where a plain cudaMemcpy from pageable memory
        // is already pipelined efficiently by the driver -- staging through
        // an explicit full-buffer CPU memcpy first just duplicates that work
        // AND adds host_ptr_is_pinned's own cudaPointerGetAttributes probe
        // (x2/call, x424 calls here) on top. Same "a fix that wins for many
        // small transfers can lose for a few large ones" lesson as this
        // file's own upload_expert_active_async history -- kept plain.
        //
        // FIRST-PASS METHODOLOGY WARNING, kept because the mistake is worth
        // not repeating: an earlier, non-position-controlled A/B on this
        // exact change showed the OPPOSITE result (pinned ~8% FASTER) --
        // caused by a real, separately-confirmed artifact where whichever
        // config runs SECOND in a back-to-back pair is ~1.8s faster
        // regardless of which config it is (GPU/driver warm-up between
        // consecutive process launches, not page-cache -- PLE's own cost was
        // checked and stayed flat across both arms). Never trust an A/B pair
        // ordered (config1, config2, config1, config2, ...); always give
        // every measured run an identical warmup immediately before it.
        PERF_START(dense_up);
        cudaMemcpy(d_x, x_host, x_bytes, cudaMemcpyHostToDevice);
        PERF_END(dense_up, t_dense_gpu_up);

        PERF_START(dense_cmp);
        launch_f32_to_bf16(d_x_bf16, d_x, (int64_t)S * I);
        if (!launch_dequant_to_bf16((int)wtype, d_w_bf16, d_W_q8_0, (int64_t)O * I)) {
            PERF_END(dense_cmp, t_dense_gpu_compute);
            return false;   // no BF16 decoder for this type -- caller falls back
        }
        cublasStatus_t st = cublas_dense_matmul_bf16_matched(
            cublas_handle, d_w_bf16, d_x_bf16, d_y, S, I, O);
        cudaStreamSynchronize(0);
        PERF_END(dense_cmp, t_dense_gpu_compute);
        if (st != CUBLAS_STATUS_SUCCESS) {
            LOG_ERROR("[CUDA] dense_matmul_bf16_host: cublas status %d "
                      "(S=%d I=%d O=%d) -- caller falls back\n", (int)st, S, I, O);
            return false;
        }

        PERF_START(dense_dn);
        cudaMemcpy(y_host, d_y, y_bytes, cudaMemcpyDeviceToHost);
        PERF_END(dense_dn, t_dense_gpu_down);
        // dense_matmul_host's own tail (below) increments this only for ITS
        // own dispatch -- a successful call here returns directly and would
        // otherwise never be counted, undercounting the report line's
        // "calls=" figure against the real number of GPU dispatches.
        perf().n_dense_gpu_calls.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    bool dense_matmul_host(GGUFType wtype, const void* d_W,
                           const float* x_host, float* y_host,
                           int S, int I, int O) {
        if (!d_W) return false;
        if (wtype == GGUFType::BF16) {
            return dense_matmul_bf16_matched_host(d_W, x_host, y_host, S, I, O);
        }
        // Opt-in prefill fast path (see dense_matmul_q8_0_bf16_host above).
        // S==1 (decode) deliberately stays on the fused kernel: the GEMV
        // shape is CORRECT there, and a per-call weight dequantisation would
        // be pure overhead for a single token.
        // Q8_0 rides dense_bf16_prefill_gemm alone (verified on GLM-5.3 and
        // qwen4exp); Q5_K/Q6_K additionally require dense_bf16_kquant, which
        // is qwen4exp-verified only. See dense_bf16_kquant()'s comment.
        const bool bf16_capable = (wtype == GGUFType::Q8_0) ||
                                  (dense_bf16_kquant() &&
                                   (wtype == GGUFType::Q5_K || wtype == GGUFType::Q6_K));
        if (bf16_capable && S > 1 && dense_bf16_prefill_gemm() &&
            dense_matmul_bf16_host(wtype, d_W, x_host, y_host, S, I, O)) {
            return true;
        }
        size_t x_bytes = (size_t)S * I * sizeof(float);
        size_t y_bytes = (size_t)S * O * sizeof(float);
        if (!ensure_buffers(x_bytes, y_bytes)) return false;  // caller (dense_proj) falls back to CPU
        cudaSetDevice(device_id);

        // Phase split -- added after attn_output's dense_proj call (which
        // routes through here) was found costing ~623ms/token, ~5x its
        // qproj sibling calls, with zero CPU fallback logged (so it IS
        // successfully GPU-dispatching every time) -- this pins down
        // whether the cost is the H2D copy, the kernel itself, or the D2H
        // copy, same method that found the real expert-FFN GPU issue
        // earlier this session.
        PERF_START(dense_up);
        if (host_ptr_is_pinned(x_host) || !ensure_dense_stage(x_bytes)) {
            cudaMemcpy(d_x, x_host, x_bytes, cudaMemcpyHostToDevice);
        } else {
            std::memcpy(dense_stage, x_host, x_bytes);
            cudaMemcpy(d_x, dense_stage, x_bytes, cudaMemcpyHostToDevice);
        }
        PERF_END(dense_up, t_dense_gpu_up);

        PERF_START(dense_cmp);
        bool ok = false;
        switch (wtype) {
            case GGUFType::F32:
            case GGUFType::F16:
                // BF16 no longer reaches this switch -- handled by the
                // early-return delegation to dense_matmul_bf16_matched_host
                // above (see that early-return's comment for why).
                ok = dense_matmul(wtype, d_W, d_x, d_y, S, I, O);
                break;
            case GGUFType::Q8_0:
                launch_q8_0_matmul(d_y, d_x, d_W, S, I, O);
                ok = true;
                break;
            // Q5_1 (cuda/q5_1_matmul.cu, 2026-09-04). Written for the
            // ROUTED-EXPERT path (qwen4exp's ffn_down_exps), but wired here
            // too: it is the same launcher, this switch is the one Test 5
            // actually drives, and a dense Q5_1 tensor would otherwise hit
            // the loud CPU-fallback warning for no reason.
            case GGUFType::Q5_1:
                launch_q5_1_matmul(d_y, d_x, d_W, S, I, O);
                ok = true;
                break;
            case GGUFType::Q2_K:
                launch_q2_K_matmul(d_y, d_x, d_W, S, I, O);
                ok = true;
                break;
            case GGUFType::Q3_K:
                launch_q3_K_matmul(d_y, d_x, d_W, S, I, O);
                ok = true;
                break;
            case GGUFType::Q4_K:
                launch_q4_K_matmul(d_y, d_x, d_W, S, I, O);
                ok = true;
                break;
            case GGUFType::Q5_K:
                launch_q5_K_matmul(d_y, d_x, d_W, S, I, O);
                ok = true;
                break;
            case GGUFType::Q6_K:
                launch_q6_K_matmul(d_y, d_x, d_W, S, I, O);
                ok = true;
                break;
            case GGUFType::IQ2_XXS:
                launch_iq2_xxs_matmul(d_y, d_x, d_W, S, I, O);
                ok = true;
                break;
            case GGUFType::IQ2_XS:
                launch_iq2_xs_matmul(d_y, d_x, d_W, S, I, O);
                ok = true;
                break;
            case GGUFType::IQ2_S:
                launch_iq2_s_matmul(d_y, d_x, d_W, S, I, O);
                ok = true;
                break;
            case GGUFType::IQ3_XXS:
                launch_iq3_xxs_matmul(d_y, d_x, d_W, S, I, O);
                ok = true;
                break;
            case GGUFType::IQ4_XS:
                launch_iq4_xs_matmul(d_y, d_x, d_W, S, I, O);
                ok = true;
                break;
            case GGUFType::IQ4_NL:
                // Added 2026-09-04 alongside the expert-path dispatch in
                // execution_layer.cpp -- both switches must know a type or
                // Test 5 (which drives dense_matmul_host) reports "no kernel"
                // even though routed experts dispatch fine.
                launch_iq4_nl_matmul(d_y, d_x, d_W, S, I, O);
                ok = true;
                break;
            case GGUFType::IQ1_S:
                launch_iq1s_matmul(d_y, d_x, d_W, S, I, O);
                ok = true;
                break;
            case GGUFType::IQ1_M:
                launch_iq1m_matmul(d_y, d_x, d_W, S, I, O);
                ok = true;
                break;
            case GGUFType::IQ3_S:
                launch_iq3s_matmul(d_y, d_x, d_W, S, I, O);
                ok = true;
                break;
            // MXFP4: reachable here (dense_matmul_host is called directly by
            // src/main_test.cpp's Test 5 harness, bypassing gpu_type_supported()'s
            // gate) so the kernel can be validated on real hardware BEFORE
            // gpu_type_supported() flips it on for the live expert-FFN path.
            case GGUFType::MXFP4:
                launch_mxfp4_matmul(d_y, d_x, d_W, S, I, O);
                ok = true;
                break;
            // IQ1_* / IQ2_M : no GPU fused kernel — CPU dequant (libggml) handles
            // IQ1_* via the caller's cpu_quant_matmul fallback; IQ2_M has no ggml
            // dequant export and loud-zeros.
            default:
                ok = false;
                break;
        }
        // Kernel launches (and dense_matmul's cuBLAS call) are async -- an
        // explicit sync here is required for the timing window to capture
        // real device execution time, not just launch overhead. The
        // downstream D2H cudaMemcpy below would implicitly wait for this
        // anyway (default-stream blocking semantics), but that would fold
        // compute time into t_dense_gpu_down instead of attributing it
        // correctly.
        cudaStreamSynchronize(0);
        PERF_END(dense_cmp, t_dense_gpu_compute);
        if (!ok) return false;

        PERF_START(dense_dn);
        if (host_ptr_is_pinned(y_host) || !ensure_dense_stage(y_bytes)) {
            cudaMemcpy(y_host, d_y, y_bytes, cudaMemcpyDeviceToHost);
        } else {
            cudaMemcpy(dense_stage, d_y, y_bytes, cudaMemcpyDeviceToHost);
            std::memcpy(y_host, dense_stage, y_bytes);
        }
        PERF_END(dense_dn, t_dense_gpu_down);
        perf().n_dense_gpu_calls.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    // One-time upload of a static (never-changes-across-tokens) F32 buffer
    // to a persistent VRAM allocation -- caller owns the returned pointer
    // for the lifetime of the process (mirrors this project's existing
    // "cache dequantized static weights, never free" convention, e.g.
    // attention_layer.cpp's woa_cache/wuk_cache/wuv_cache, just VRAM- rather
    // than host-resident). Returns nullptr on any allocation/copy failure
    // -- caller falls back to the CPU path, never a hard failure.
    float* upload_persistent_f32(const float* host_data, size_t n_floats) {
        cudaSetDevice(device_id);
        float* d_ptr = nullptr;
        size_t bytes = n_floats * sizeof(float);
        cudaError_t err = cudaMalloc((void**)&d_ptr, bytes);
        if (err != cudaSuccess) {
            LOG_STREAM_ERROR() << "[GPU] upload_persistent_f32: cudaMalloc failed for "
                               << bytes / 1e6 << " MB: " << cudaGetErrorString(err) << "\n";
            return nullptr;
        }
        err = cudaMemcpy(d_ptr, host_data, bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            LOG_STREAM_ERROR() << "[GPU] upload_persistent_f32: cudaMemcpy H2D failed for "
                               << bytes / 1e6 << " MB: " << cudaGetErrorString(err) << "\n";
            cudaFree(d_ptr);
            return nullptr;
        }
        return d_ptr;
    }

    // Grouped (block-diagonal) output projection: o_groups independent
    // [1, group_in] x [o_rank, group_in]^T matmuls, y[g] = d_WOA[g] @ x[g].
    // Added specifically for dsv4_attention_gguf's S==1 (decode) outproj
    // step (attention_layer.cpp) -- real-hardware profiling found this the
    // single largest remaining cost in attention (~6.8s of a real 40-token
    // run), root-caused to memory bandwidth, not compute: the weight matrix
    // (o_groups*o_rank*group_in floats, ~134MB for this model) has to be
    // re-read from host RAM on EVERY decode token since no other layer's
    // work fits in cache between two calls to the same layer, and this
    // CPU's RAM bandwidth is the bottleneck (~0.5 FLOP/byte arithmetic
    // intensity -- a textbook memory-bound matrix-vector multiply, not a
    // parallelism problem; confirmed by a standalone CPU-vs-GPU
    // microbenchmark using this exact shape and the real weight-cycling
    // access pattern, tests/bench_outproj_cpu_vs_gpu.cu, before this was
    // wired into the live path -- GPU VRAM bandwidth (~900+ GB/s on
    // consumer cards vs. ~30-50 GB/s CPU DDR) gave a measured 12x+ speedup
    // once the weight is already VRAM-resident, verified numerically
    // matching (max rel diff ~2e-3, normal float32 accumulation-order
    // difference between AVX-512 and cuBLAS, not a bug). d_WOA MUST already
    // be VRAM-resident (via upload_persistent_f32 above, called once per
    // layer on first use) -- this function only runs the GEMV, it never
    // uploads the weight itself.
    bool grouped_output_proj_host(const float* d_WOA, int o_groups, int o_rank, int group_in,
                                  const float* x_host, float* y_host) {
        // outproj_cublas_handle/outproj_stream, NOT cublas_handle/stream 0
        // -- see outproj_stream's own declaration comment for the real-
        // hardware regression (net-wash on total throughput) this fixes.
        if (!d_WOA || !outproj_cublas_handle || !outproj_stream) return false;
        size_t x_bytes = (size_t)o_groups * group_in * sizeof(float);
        size_t y_bytes = (size_t)o_groups * o_rank * sizeof(float);
        if (!ensure_outproj_buffers(x_bytes, y_bytes)) return false;  // caller falls back to CPU
        cudaSetDevice(device_id);

        PERF_START(dense_up);
        cudaMemcpyAsync(d_outproj_x, x_host, x_bytes, cudaMemcpyHostToDevice, outproj_stream);
        PERF_END(dense_up, t_dense_gpu_up);

        PERF_START(dense_cmp);
        const float alpha = 1.0f, beta = 0.0f;
        bool ok = true;
        for (int g = 0; g < o_groups; ++g) {
            // Row-major W[o_rank, group_in] reinterpreted as column-major
            // A'(m=group_in, n=o_rank); CUBLAS_OP_T gives y = W @ x exactly,
            // with no data reshuffling -- same derivation validated in the
            // standalone benchmark above.
            cublasStatus_t st = cublasSgemv(outproj_cublas_handle, CUBLAS_OP_T,
                group_in, o_rank, &alpha,
                d_WOA + (size_t)g * o_rank * group_in, group_in,
                d_outproj_x + (size_t)g * group_in, 1,
                &beta, d_outproj_y + (size_t)g * o_rank, 1);
            if (st != CUBLAS_STATUS_SUCCESS) { ok = false; break; }
        }
        // Sync ONLY outproj_stream -- never stream 0. This is the entire
        // point of the dedicated stream: unlike cudaStreamSynchronize(0),
        // this cannot force any other stream (MoE's compute_stream/
        // copy_stream, or dense_proj's own stream-0 work) to wait or be
        // waited on. Real per-call work is tiny (a few microseconds of
        // GEMV), so synchronous-from-the-caller's-perspective is still
        // fine here -- decode needs the result immediately either way.
        cudaStreamSynchronize(outproj_stream);
        PERF_END(dense_cmp, t_dense_gpu_compute);
        if (!ok) return false;

        PERF_START(dense_dn);
        cudaMemcpyAsync(y_host, d_outproj_y, y_bytes, cudaMemcpyDeviceToHost, outproj_stream);
        cudaStreamSynchronize(outproj_stream);
        PERF_END(dense_dn, t_dense_gpu_down);
        perf().n_dense_gpu_calls.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    // Host-staged F32 RMSNorm on GPU: y[S,D] = rmsnorm(x[S,D], w[D]).
    // w must be device-resident (F32 norm weights are offloaded to VRAM).
    bool rmsnorm_gpu_host(const void* d_w, const float* x_host, float* y_host,
                          int S, int D, float eps) {
        if (!d_w) return false;
        size_t bytes = (size_t)S * D * sizeof(float);
        if (!ensure_buffers(bytes, bytes)) return false;  // caller falls back to CPU rmsnorm_row
        cudaSetDevice(device_id);
        cudaMemcpy(d_x, x_host, bytes, cudaMemcpyHostToDevice);
        launch_rmsnorm(d_y, d_x, (const float*)d_w, S, D, eps);
        cudaMemcpy(y_host, d_y, bytes, cudaMemcpyDeviceToHost);
        return true;
    }

    // Transfer a small vector between GPUs (expert output → core for residual add)
    // Uses P2P if available, otherwise bounces through pinned host buffer
    void cross_device_transfer(float* dst, int dst_dev,
                               const float* src, int src_dev, size_t bytes) {
        topology.cross_device_transfer(dst, dst_dev, src, src_dev, bytes);
    }

    // Get hardware profile for conditional logic
    HardwareProfile get_profile() const { return topology.profile; }
    int get_core_device() const { return device_id; }
    int get_vault_device(int idx = 0) const {
        if (idx < (int)topology.vault_gpus.size())
            return topology.devices[topology.vault_gpus[idx]].id;
        return device_id; // fallback to core
    }
};
