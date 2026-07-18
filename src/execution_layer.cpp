#include "engine.hpp"
#include "kernels_cpu.hpp"
#include "dense_dispatch.hpp"
#include "kquant_dequant.hpp"
#include "perf_timer.hpp"
#include "ram_manager.hpp"
#include "async_prefetcher.hpp"
#include "logger.hpp"
#include <cuda_runtime.h>
#include <immintrin.h>
#include <vector>
#include <algorithm>
#include <thread>
#include <cstring>
#include <cmath>

extern void mla_attention_absorbed_gguf(StreamingEngine& engine, int layer_idx, const float* Q_roped, int S, int pos_base, float* out);
extern void dense_attention_gguf(StreamingEngine& engine, int layer_idx, const float* hidden_states, int S, int pos_base, float* out);

// CUDA kernel launchers
extern "C" void launch_gguf_q4_0_dp4a(float* y, const float* x, const void* weights, int S, int I, int O);
extern "C" void launch_iq2m_matmul(float* y, const float* x, const void* weights, int S, int I, int O);
extern "C" void launch_q2_K_matmul(float* y, const float* x, const void* weights, int S, int I, int O);
extern "C" void launch_q3_K_matmul(float* y, const float* x, const void* weights, int S, int I, int O);
extern "C" void launch_q4_K_matmul(float* y, const float* x, const void* weights, int S, int I, int O);
extern "C" void launch_q5_K_matmul(float* y, const float* x, const void* weights, int S, int I, int O);
extern "C" void launch_q6_K_matmul(float* y, const float* x, const void* weights, int S, int I, int O);
extern "C" void launch_iq2_xxs_matmul(float* y, const float* x, const void* weights, int S, int I, int O);
extern "C" void launch_iq2_xs_matmul(float* y, const float* x, const void* weights, int S, int I, int O);
extern "C" void launch_iq2_s_matmul(float* y, const float* x, const void* weights, int S, int I, int O);
extern "C" void launch_iq3_xxs_matmul(float* y, const float* x, const void* weights, int S, int I, int O);
extern "C" void launch_tq1_matmul(float* y, const float* x, const void* weights, const void* scales, int S, int I, int O);
extern "C" void launch_iq1s_matmul(float* y, const float* x, const void* weights, int S, int I, int O);
extern "C" void launch_iq1m_matmul(float* y, const float* x, const void* weights, int S, int I, int O);

// ============================================================
// Scratch buffers: pre-allocated once, reused every layer to
// eliminate per-layer heap allocation (was ~5% overhead measured)
// ============================================================
// GPU-resident scratch buffers for expert FFN dispatch (lazily grown,
// reused across calls to avoid per-token cudaMalloc/cudaFree churn).
struct GpuExpertScratch {
    float* d_x = nullptr;     // [S*D] input activation
    float* d_gate = nullptr;  // [S*moe_inter]
    float* d_up = nullptr;    // [S*moe_inter]
    float* d_prod = nullptr;  // [S*moe_inter] SiLU(gate) * up, re-uploaded for down_proj
    float* d_down = nullptr;  // [S*D] final output
    size_t cap_sd = 0, cap_si = 0;
    int device_id = -1;       // which GPU these buffers currently live on

    // target_device is the vault GPU actually holding the resident expert
    // (loc->gpu_id) — NOT whatever device happens to be ambient on this
    // thread. This is thread_local static, so on first use it would
    // otherwise silently allocate on whatever device some unrelated earlier
    // call last left current, and in a multi-vault-GPU setup (triple_gpu.toml)
    // a later call could target a DIFFERENT vault than these buffers live on,
    // mixing pointers from two devices in one kernel launch. Reallocate
    // fresh whenever the target device changes.
    void ensure(int S, int D, int moe_inter, int target_device) {
        if (target_device != device_id) {
            if (device_id >= 0 && (d_x || d_gate || d_up || d_prod || d_down)) {
                cudaSetDevice(device_id);
                if (d_x)    { cudaFree(d_x);    d_x = nullptr; }
                if (d_down) { cudaFree(d_down); d_down = nullptr; }
                if (d_gate) { cudaFree(d_gate); d_gate = nullptr; }
                if (d_up)   { cudaFree(d_up);   d_up = nullptr; }
                if (d_prod) { cudaFree(d_prod); d_prod = nullptr; }
            }
            cap_sd = 0;
            cap_si = 0;
            device_id = target_device;
        }
        cudaSetDevice(device_id);

        size_t need_sd = (size_t)S * D;
        size_t need_si = (size_t)S * moe_inter;
        if (cap_sd < need_sd) {
            if (d_x) cudaFree(d_x);
            if (d_down) cudaFree(d_down);
            cudaMalloc((void**)&d_x, need_sd * sizeof(float));
            cudaMalloc((void**)&d_down, need_sd * sizeof(float));
            cap_sd = need_sd;
        }
        if (cap_si < need_si) {
            if (d_gate) cudaFree(d_gate);
            if (d_up) cudaFree(d_up);
            if (d_prod) cudaFree(d_prod);
            cudaMalloc((void**)&d_gate, need_si * sizeof(float));
            cudaMalloc((void**)&d_up, need_si * sizeof(float));
            cudaMalloc((void**)&d_prod, need_si * sizeof(float));
            cap_si = need_si;
        }
    }
};

// Diagnostic: count NaN/Inf in a buffer, used to bisect exactly which stage
// of the GPU-resident expert FFN pipeline (gate/up projection vs. down
// projection -- different weight tensors, potentially different quant
// kernels, e.g. IQ2_XXS for gate/up vs. IQ3_XXS for down in this session's
// dominant combo) first introduces corruption, rather than only seeing NaN
// much further downstream at the final logits.
static bool check_finite(const char* label, int layer_idx, int expert_id, const float* data, size_t n) {
    int nan_count = 0, inf_count = 0;
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(data[i])) ++nan_count;
        else if (std::isinf(data[i])) ++inf_count;
    }
    if (nan_count > 0 || inf_count > 0) {
        LOG_ERROR("[EXPERT-GPU] %s layer=%d expert=%d: %d/%zu NaN, %d/%zu Inf.\n",
                  label, layer_idx, expert_id, nan_count, n, inf_count, n);
        return false;
    }
    return true;
}

// gpu_type_supported() now lives in kquant_dequant.hpp (shared with
// ram_manager.hpp / async_prefetcher.hpp's VRAM-promotion decisions, so none
// of them waste PCIe bandwidth promoting an expert type that can never
// actually dispatch to GPU). Checked BEFORE touching the GPU here (uploading
// activations etc.) so a VRAM-resident expert whose quant type simply has no
// kernel doesn't pay for a wasted H2D copy only to fall back to CPU anyway.

// Maps a GGUF quant type to its GPU matmul launcher. Only call this after
// gpu_type_supported() has confirmed the type has a kernel.
static bool gpu_matmul_dispatch(GGUFType type, float* y, const float* x, const void* w,
                                 int S, int I, int O) {
    switch (type) {
        case GGUFType::Q4_0:  launch_gguf_q4_0_dp4a(y, x, w, S, I, O); return true;
        case GGUFType::Q2_K:  launch_q2_K_matmul(y, x, w, S, I, O);   return true;
        case GGUFType::Q3_K:  launch_q3_K_matmul(y, x, w, S, I, O);   return true;
        case GGUFType::Q4_K:  launch_q4_K_matmul(y, x, w, S, I, O);   return true;
        case GGUFType::Q5_K:  launch_q5_K_matmul(y, x, w, S, I, O);   return true;
        case GGUFType::Q6_K:  launch_q6_K_matmul(y, x, w, S, I, O);   return true;
        case GGUFType::IQ2_XXS: launch_iq2_xxs_matmul(y, x, w, S, I, O); return true;
        case GGUFType::IQ2_XS:  launch_iq2_xs_matmul(y, x, w, S, I, O);  return true;
        case GGUFType::IQ2_S:   launch_iq2_s_matmul(y, x, w, S, I, O);   return true;
        case GGUFType::IQ3_XXS: launch_iq3_xxs_matmul(y, x, w, S, I, O); return true;
        default: return false;
    }
}

// Context needed to opportunistically dispatch an expert's FFN on GPU.
// `loc` is the expert's current VRAM residency (nullptr if not cached).
struct ExpertGpuContext {
    GpuTopology* topo = nullptr;
    const ExpertLocation* loc = nullptr;
    int layer_idx = -1;
    int expert_id = -1;
    const uint8_t* slab_buffer = nullptr;
    size_t slab_capacity = 0;
};

struct ExpertScore { int id; float score; };

// One GPU-path expert whose gate/up kernels have been issued (async) but not
// yet collected -- queued during moe_forward_gguf's Pass A so Pass B can
// finish them (sync + download + down-proj) after the CPU has had time to
// churn through the CPU-path experts in between. `loc` is resolved once at
// issue time so finish doesn't need to re-derive VRAM residency.
struct PendingGpuExpert {
    int k = -1;
    MoEExpertBlock* exp = nullptr;
    const ExpertLocation* loc = nullptr;
};

struct LayerScratch {
    std::vector<float> attn_out;
    std::vector<float> moe_out;
    std::vector<float> normed;
    std::vector<float> router_logits;
    std::vector<float> gate_buf;   // expert gate_proj output [S * moe_inter]
    std::vector<float> up_buf;     // expert up_proj output [S * moe_inter]
    std::vector<float> down_buf;   // expert down_proj output [S * D]
    std::vector<float> expert_acc; // accumulated expert output [S * D]

    // Token-loop scratch (pre-allocated to eliminate per-token heap churn)
    std::vector<ExpertScore> scores;   // [n_experts]
    std::vector<MoEExpertBlock*> expert_ptrs; // [topk]
    std::vector<float> pilot_nrm;      // [D]
    std::vector<float> pilot_logits;   // [n_experts]

    // Up to `topk` independent GPU device-buffer sets, so up to `topk`
    // GPU-path experts can have gate/up kernels in flight simultaneously
    // (issued but not yet collected) instead of each expert fully
    // round-tripping (upload -> kernel -> sync -> download) before the next
    // one starts. See moe_forward_gguf's Pass A / Pass B and
    // expert_ffn_gpu_issue / expert_ffn_gpu_finish.
    std::vector<GpuExpertScratch> gpu_pool; // [topk]
    std::vector<PendingGpuExpert> pending_gpu; // capacity topk, cleared+refilled per token

    // Whole-batch router lookahead (prefill only, S > 1): the router already
    // knows every token's top-k choice before any expert runs, since
    // router_logits is one batched matmul over all S tokens. Cache each
    // token's decision here so it's computed once, and tally how often each
    // expert is chosen across the batch so cold-start VRAM warm-up can fill
    // free slots with the most in-demand experts first.
    std::vector<int> tok_topk_ids;      // [S * topk]
    std::vector<float> tok_topk_weights; // [S * topk]
    std::vector<int> expert_freq;       // [n_experts]

    void resize(int S, int D, int n_experts, int moe_inter, int topk, int n_expert_shared = 0, int dense_inter = 0) {
        attn_out.resize(S * D);
        moe_out.resize(S * D);
        normed.resize(S * D);
        router_logits.resize(S * n_experts);
        // gate_buf/up_buf are reused per-token (S=1 per call) at whichever of
        // the routed, shared-expert, or leading-dense-layer intermediate
        // width is largest — the shared-expert width is
        // moe_inter * n_expert_shared when >1 shared expert is present.
        int shared_inter = moe_inter * std::max(1, n_expert_shared);
        int buf_inter = std::max({moe_inter, shared_inter, dense_inter});
        gate_buf.resize(S * buf_inter);
        up_buf.resize(S * buf_inter);
        down_buf.resize(S * D);
        expert_acc.resize(S * D);
        scores.resize(n_experts);
        expert_ptrs.resize(topk);
        pilot_nrm.resize(D);
        pilot_logits.resize(n_experts);
        tok_topk_ids.resize(S * topk);
        tok_topk_weights.resize(S * topk);
        expert_freq.resize(n_experts);
        if ((int)gpu_pool.size() < std::max(1, topk)) gpu_pool.resize(std::max(1, topk));
        pending_gpu.reserve(std::max(1, topk));
    }
};

// ============================================================
// Expert FFN: gate_proj → SiLU, up_proj, element-wise multiply, down_proj
//
// Dispatch policy: GPU is only worth it when the PCIe weight-transfer cost
// is already sunk (expert is VRAM-resident) or amortized across a large
// enough batch. For decode (S=1) a cold expert almost always loses to CPU,
// since host DDR bandwidth beats a one-shot PCIe upload — only VRAM cache
// *hits* take the GPU path there. Unsupported quant types on GPU (e.g.
// 1-bit formats) transparently fall back to CPU.
//
// Split into issue/finish (GPU path) + a standalone CPU path, instead of one
// monolithic call that fully round-trips before returning. Perf data showed
// GPU-path experts were cheap (~1.5ms/call) but CPU-path experts were not
// (~27ms/call for a K-quant/IQ dequant+matmul), and the two never overlapped
// -- expert_ffn used to block on cudaStreamSynchronize before the next
// expert (GPU or CPU) even started, so the GPU sat idle during every CPU
// expert and vice versa, both far below 100% utilization simultaneously.
// moe_forward_gguf's Pass A now issues every GPU-path expert's gate/up
// kernels (async, no sync) and immediately runs CPU-path experts on the host
// while those kernels execute in the background; Pass B then finishes each
// issued GPU expert (sync + download + down-proj), which by then should
// return near-instantly since the CPU work already gave the GPU time to
// finish. Each in-flight GPU expert needs its OWN device buffer set
// (LayerScratch::gpu_pool) since issue and finish are no longer adjacent in
// time for a single expert -- reusing one shared scratch across concurrent
// in-flight experts would let a later issue() overwrite an earlier expert's
// still-unread gate/up buffers.
// ============================================================
static constexpr int GPU_COLD_BATCH_THRESHOLD = 4;  // amortizes a cold upload

// Decides GPU vs CPU for one expert and, for the GPU case, resolves VRAM
// residency (including the synchronous cold-batch upload for large S).
// Returns nullptr if the CPU path should be taken.
static const ExpertLocation* expert_ffn_resolve_gpu_loc(
    const ExpertGpuContext* gpu_ctx, int S,
    const GGUFTensor& gate_meta, const GGUFTensor& up_meta, const GGUFTensor& down_meta
) {
    const ExpertLocation* loc = gpu_ctx ? gpu_ctx->loc : nullptr;
    bool vram_resident = loc && loc->device_ptr && loc->layer_idx >= 0;
    if (cpu_only_mode()) vram_resident = false;  // diagnostic: force CPU expert path

    // Batch large enough to amortize a cold upload: fetch synchronously now,
    // since (unlike background promotion) this call needs the data
    // immediately to compute this token's result.
    if (!vram_resident && gpu_ctx && gpu_ctx->topo && gpu_ctx->slab_buffer
        && gpu_ctx->slab_capacity > 0 && S >= GPU_COLD_BATCH_THRESHOLD) {
        loc = gpu_ctx->topo->upload_expert(gpu_ctx->layer_idx, gpu_ctx->expert_id,
                                            gpu_ctx->slab_buffer, gpu_ctx->slab_capacity);
        vram_resident = loc && loc->device_ptr;
    }
    if (!vram_resident) return nullptr;

    if (!gpu_type_supported(gate_meta.type) || !gpu_type_supported(up_meta.type) ||
        !gpu_type_supported(down_meta.type)) {
        // VRAM-resident but at least one projection's quant type has no
        // GPU kernel — was previously a completely silent CPU fallback
        // (indistinguishable from "not resident yet"), which made a real
        // capability gap look like a caching bug. Skip the GPU path
        // entirely rather than wasting an H2D upload that goes unused.
        // Only names the projection(s) actually blocking dispatch (gate/
        // up are commonly fine — it's usually just down, e.g. Unsloth
        // Dynamic quant's gate/up=IQ2_XXS + down=IQ3_XXS) rather than
        // dumping all three types regardless of whether they're the
        // problem.
        if (Logger::instance().would_log(LogLevel::DEBUG)) {
            std::string why;
            if (!gpu_type_supported(gate_meta.type))
                why += std::string("gate=") + gpu_type_name(gate_meta.type) + " (" + gpu_unsupported_reason(gate_meta.type) + ") ";
            if (!gpu_type_supported(up_meta.type))
                why += std::string("up=") + gpu_type_name(up_meta.type) + " (" + gpu_unsupported_reason(up_meta.type) + ") ";
            if (!gpu_type_supported(down_meta.type))
                why += std::string("down=") + gpu_type_name(down_meta.type) + " (" + gpu_unsupported_reason(down_meta.type) + ") ";
            LOG_DEBUG("[MOE-GPU]   layer=%d expert=%d VRAM-resident but blocked: %s-- CPU dequant-matmul fallback\n",
                      gpu_ctx ? gpu_ctx->layer_idx : -1, gpu_ctx ? gpu_ctx->expert_id : -1, why.c_str());
        }
        return nullptr;
    }
    return loc;
}

// Phase A: upload input + launch gate/up kernels, no sync. Returns
// immediately once the (small, S*D-element) H2D copy completes -- the
// kernel launches themselves are async and keep executing on the device
// after this call returns.
static void expert_ffn_gpu_issue(
    GpuExpertScratch& gpu_scratch, const ExpertLocation* loc,
    const float* input, const GGUFTensor& gate_meta, const GGUFTensor& up_meta,
    int S, int D, int moe_inter
) {
    // Slab layout mirrors the host copy: gate at offset 0, up at its
    // original file-offset delta from gate (see load_expert_slab).
    const uint8_t* base = loc->device_ptr;
    const uint8_t* gate_dev = base;
    const uint8_t* up_dev   = base + (up_meta.offset - gate_meta.offset);

    gpu_scratch.ensure(S, D, moe_inter, loc->gpu_id);  // pins the correct vault GPU

    PERF_START(gexp_up1);
    cudaMemcpy(gpu_scratch.d_x, input, (size_t)S * D * sizeof(float), cudaMemcpyHostToDevice);
    PERF_END(gexp_up1, t_gpu_upload);

    gpu_matmul_dispatch(gate_meta.type, gpu_scratch.d_gate, gpu_scratch.d_x, gate_dev, S, D, moe_inter);
    gpu_matmul_dispatch(up_meta.type,   gpu_scratch.d_up,   gpu_scratch.d_x, up_dev,   S, D, moe_inter);
}

// Phase B: sync the gate/up kernels issued earlier, download, SiLU+mul on
// CPU, then upload/launch/sync/download the down projection. Called after
// the caller has (ideally) already spent enough time on other work that the
// gate/up kernels are done, so the first cudaStreamSynchronize here returns
// near-instantly instead of actually blocking.
static void expert_ffn_gpu_finish(
    GpuExpertScratch& gpu_scratch, const ExpertLocation* loc,
    float* gate_buf, float* up_buf, float* output,
    const GGUFTensor& gate_meta, const GGUFTensor& down_meta,
    int layer_idx, int expert_id,
    int S, int D, int moe_inter
) {
    // Each in-flight expert's issue() may have run on a different vault GPU
    // (multi-GPU config) and left a different device "current" on this
    // thread -- re-pin explicitly rather than trusting ambient state.
    cudaSetDevice(loc->gpu_id);
    const uint8_t* down_dev = loc->device_ptr + (down_meta.offset - gate_meta.offset);

    PERF_START(gexp_cmp1);
    cudaStreamSynchronize(0);
    PERF_END(gexp_cmp1, t_gpu_compute);

    // SiLU + elementwise multiply is cheap (moe_inter elements) —
    // done on CPU rather than adding a dedicated kernel for now.
    PERF_START(gexp_dn1);
    cudaMemcpy(gate_buf, gpu_scratch.d_gate, (size_t)S * moe_inter * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(up_buf,   gpu_scratch.d_up,   (size_t)S * moe_inter * sizeof(float), cudaMemcpyDeviceToHost);
    PERF_END(gexp_dn1, t_gpu_download);
    check_finite("gate_buf (post gate+up matmul)", layer_idx, expert_id, gate_buf, (size_t)S * moe_inter);
    check_finite("up_buf   (post gate+up matmul)", layer_idx, expert_id, up_buf,   (size_t)S * moe_inter);
    silu_inplace(gate_buf, S * moe_inter);
    elem_mul(gate_buf, gate_buf, up_buf, S * moe_inter);

    PERF_START(gexp_up2);
    cudaMemcpy(gpu_scratch.d_prod, gate_buf, (size_t)S * moe_inter * sizeof(float), cudaMemcpyHostToDevice);
    PERF_END(gexp_up2, t_gpu_upload);

    PERF_START(gexp_cmp2);
    gpu_matmul_dispatch(down_meta.type, gpu_scratch.d_down, gpu_scratch.d_prod, down_dev, S, moe_inter, D);
    cudaStreamSynchronize(0);
    PERF_END(gexp_cmp2, t_gpu_compute);

    PERF_START(gexp_dn2);
    cudaMemcpy(output, gpu_scratch.d_down, (size_t)S * D * sizeof(float), cudaMemcpyDeviceToHost);
    PERF_END(gexp_dn2, t_gpu_download);
    check_finite("output   (post down matmul)", layer_idx, expert_id, output, (size_t)S * D);
    perf().n_expert_gpu_calls.fetch_add(1, std::memory_order_relaxed);
}

// CPU fallback path: per-projection dequant by the projection's OWN type, so
// a mixed Q2_K_XL slab (gate=Q2_K / up=Q5_K / down=Q6_K) routes each weight
// correctly. Correct for F32/F16/BF16/Q8_0/Q4_0; K-quants/IQ2 dequant is
// pending transcription (dequant_row zeroes them loudly, never silent).
static void expert_ffn_cpu(
    const float* input, float* output,
    const uint8_t* gate_data, const GGUFTensor& gate_meta,
    const uint8_t* up_data, const GGUFTensor& up_meta,
    const uint8_t* down_data, const GGUFTensor& down_meta,
    float* gate_buf, float* up_buf,
    int S, int D, int moe_inter
) {
    PERF_START(cexp);
    cpu_quant_matmul(gate_meta.type, gate_data, input, gate_buf, S, D, moe_inter);
    cpu_quant_matmul(up_meta.type,   up_data,   input, up_buf,   S, D, moe_inter);
    silu_inplace(gate_buf, S * moe_inter);
    elem_mul(gate_buf, gate_buf, up_buf, S * moe_inter);
    cpu_quant_matmul(down_meta.type, down_data, gate_buf, output, S, moe_inter, D);
    PERF_END(cexp, t_cpu_expert);
    perf().n_expert_cpu_calls.fetch_add(1, std::memory_order_relaxed);
}

// ============================================================
// MoE forward with router, top-k selection, expert execution.
// Supports parallel expert I/O via threading.
// ============================================================
void moe_forward_gguf(
    StreamingEngine& engine,
    RAMSafetyManager& ram_mgr,
    int layer_idx,
    const float* hidden_states,
    int S,
    float* output,
    LayerScratch& scratch
) {
    auto& c = engine.model_file.architecture_config;
    int D = c.hidden;
    int topk = c.topk;
    int n_experts = c.n_experts;
    int moe_inter = c.moe_inter;

    if (n_experts <= 0 || moe_inter <= 0) {
        std::fill(output, output + S * D, 0.0f);
        return;
    }

    // --- Router: compute expert probs and select top-k. Reference algorithm
    // ported from llama.cpp's build_moe_ffn (llama-graph.cpp):
    //   1) probs = gate_func(logits)     [sigmoid or softmax, per c.use_sigmoid_gate]
    //   2) selection_probs = probs + bias  [bias only shifts WHICH experts get
    //      picked — the DeepSeek-V3/GLM-5.2 e_score_correction_bias tensor]
    //   3) top-k selected BY selection_probs (biased)
    //   4) weights gathered from probs (UNBIASED) at the selected indices
    //   5) optionally normalize the top-k weights to sum to 1
    //   6) optionally scale by expert_weights_scale
    {
        PERF_SCOPE(t_moe_route);
        std::string norm_key = "blk." + std::to_string(layer_idx) + ".ffn_norm.weight";
        auto norm_it = engine.resident_layers.find(norm_key);
        if (norm_it != engine.resident_layers.end() && norm_it->second.host_ptr) {
            // This reinterprets host_ptr as float* unconditionally -- only
            // correct if ffn_norm.weight is actually F32 (see CLAUDE.md's
            // "quantized weight correctness" gotcha). Norm weights are
            // almost always F32 in practice, but if THIS one somehow isn't,
            // every element downstream (router logits AND every expert's
            // FFN input, since both consume scratch.normed) goes to garbage
            // simultaneously -- which would exactly explain gate_buf/up_buf
            // and router weight going NaN together on the very first expert
            // touched, with no GPU kernel or router-math bug required.
            if (norm_it->second.meta.type != GGUFType::F32) {
                LOG_ERROR("[MOE-ROUTE] layer=%d ffn_norm.weight type=%u (NOT F32) -- rmsnorm_row "
                          "reinterprets host_ptr as float* unconditionally; this WILL produce garbage.\n",
                          layer_idx, (unsigned)norm_it->second.meta.type);
            }
            // Check the RAW WEIGHT VALUES themselves, not just rmsnorm's
            // output -- the load-time [LOAD-CHECK] pass only verified the
            // TYPE tag is F32, never the actual bytes. This distinguishes
            // "the source data for this specific layer is corrupted" (a
            // loading/offset/file-corruption bug) from "rmsnorm_row's math
            // has an edge case" (e.g. a legitimately all-zero hidden_states
            // row combined with a zero norm weight -> Inf*0 = NaN even with
            // a nonzero eps).
            check_finite("ffn_norm.weight (raw, pre-rmsnorm)", layer_idx, -1,
                         reinterpret_cast<float*>(norm_it->second.host_ptr), (size_t)D);
            // ROOT CAUSE (found by tracing per-expert output norms against
            // colibri): rmsnorm_row normalizes exactly ONE row (D floats).
            // This used to be called ONCE, unconditionally, regardless of S
            // -- correct only for S==1 (decode). During prefill (S>1) every
            // token past s=0 read whatever was already sitting in
            // scratch.normed (leftover from a PREVIOUS layer's rmsnorm,
            // since this is a reused thread_local buffer, never zeroed
            // between layers) as its router+expert input. That's why every
            // MoE layer's routed-expert output was exactly 0.0 for tokens
            // s>=1 throughout the whole prefill pass (dead/stale input ->
            // degenerate router logits -> saturated sigmoid, same
            // weight=0.3125 for every top-k expert -- exactly what the
            // trace showed), while single-token decode (S==1) never hit it.
            const float* norm_w = reinterpret_cast<float*>(norm_it->second.host_ptr);
            for (int s = 0; s < S; ++s) {
                rmsnorm_row(scratch.normed.data() + (size_t)s * D, hidden_states + (size_t)s * D,
                            norm_w, D, c.eps);
            }
            check_finite("scratch.normed (post ffn_norm rmsnorm, router+expert input)", layer_idx, -1,
                         scratch.normed.data(), (size_t)S * D);
        } else {
            std::memcpy(scratch.normed.data(), hidden_states, S * D * sizeof(float));
        }

        std::string router_key = "blk." + std::to_string(layer_idx) + ".ffn_gate_inp.weight";
        auto router_it = engine.resident_layers.find(router_key);
        if (router_it == engine.resident_layers.end() || !router_it->second.host_ptr) {
            std::fill(output, output + S * D, 0.0f);
            return;
        }
        dense_proj(engine, "blk." + std::to_string(layer_idx) + ".ffn_gate_inp.weight",
                   scratch.normed.data(), scratch.router_logits.data(), S, D, n_experts);

        // Gating activation: turns raw logits into probs in-place. These are
        // the UNBIASED values later gathered as expert weights.
        for (int s = 0; s < S; ++s) {
            float* row = scratch.router_logits.data() + (size_t)s * n_experts;
            if (c.use_sigmoid_gate) {
                for (int e = 0; e < n_experts; ++e) row[e] = 1.0f / (1.0f + std::exp(-row[e]));
            } else {
                float max_l = row[0];
                for (int e = 1; e < n_experts; ++e) max_l = std::max(max_l, row[e]);
                float sum = 0.0f;
                for (int e = 0; e < n_experts; ++e) { row[e] = std::exp(row[e] - max_l); sum += row[e]; }
                float inv = 1.0f / sum;
                for (int e = 0; e < n_experts; ++e) row[e] *= inv;
            }
        }
        // No prior diagnostic covered this: expert_ffn's GPU-resident branch
        // only checks each expert's gate/up/down buffers AFTER routing has
        // already picked them and weighted their contribution -- if the
        // router's own dense_proj (ffn_gate_inp.weight, itself a dense-tensor
        // GPU-offload candidate per main.cpp's is_dense_offload_candidate)
        // or the gating math above produces NaN, nothing upstream would ever
        // catch it, and it would silently poison every expert's weight this
        // layer without a single gate_buf/up_buf/output log firing.
        check_finite("router_logits (post gate_inp matmul + gating)", layer_idx, -1,
                     scratch.router_logits.data(), (size_t)S * n_experts);
    }

    // Selection bias (DeepSeek-V3/GLM-5.2 style expert_selection_bias):
    // present only on some architectures, absent -> selection uses raw probs.
    // GGUF name confirmed against llama-arch.cpp: "blk.N.exp_probs_b.bias"
    // (previously looked up under a name — "ffn_gate_inp.e_score_correction_bias"
    // — that doesn't exist in real GGUFs; see the matching fix in
    // graph_initializer.cpp's preload manifest for the full explanation).
    const float* bias_row = nullptr;
    {
        std::string bias_key = "blk." + std::to_string(layer_idx) + ".exp_probs_b.bias";
        auto bias_it = engine.resident_layers.find(bias_key);
        if (bias_it != engine.resident_layers.end() && bias_it->second.host_ptr) {
            bias_row = reinterpret_cast<float*>(bias_it->second.host_ptr);
            check_finite("exp_probs_b.bias (raw)", layer_idx, -1, bias_row, (size_t)n_experts);
        }
    }

    // Zero output accumulator
    std::memset(output, 0, S * D * sizeof(float));

    // --- Phase 1: batch-wide top-k selection (router already knows every
    // token's choice up front, since router_logits above is one batched
    // matmul over all S tokens) — compute once per token, cache the result,
    // and tally how often each expert is picked across the whole batch. ---
    std::fill(scratch.expert_freq.begin(), scratch.expert_freq.end(), 0);
    for (int s = 0; s < S; ++s) {
        float* token_probs = scratch.router_logits.data() + s * n_experts;

        for (int e = 0; e < n_experts; ++e) {
            float sel = token_probs[e] + (bias_row ? bias_row[e] : 0.0f);
            scratch.scores[e] = {e, sel};
        }
        std::partial_sort(scratch.scores.begin(), scratch.scores.begin() + topk, scratch.scores.begin() + n_experts,
                         [](const ExpertScore& a, const ExpertScore& b) { return a.score > b.score; });

        float sum_w = 0.0f;
        for (int k = 0; k < topk; ++k) {
            int id = scratch.scores[k].id;
            float w = token_probs[id];  // unbiased weight
            scratch.tok_topk_weights[s * topk + k] = w;
            scratch.tok_topk_ids[s * topk + k] = id;
            scratch.expert_freq[id]++;
            sum_w += w;
        }
        if (c.expert_weights_norm) {
            float inv_sum = 1.0f / std::max(sum_w, 6.103515625e-5f);
            for (int k = 0; k < topk; ++k) scratch.tok_topk_weights[s * topk + k] *= inv_sum;
        }
        if (c.expert_weights_scale != 0.0f) {
            for (int k = 0; k < topk; ++k) scratch.tok_topk_weights[s * topk + k] *= c.expert_weights_scale;
        }

        // Router-selection trace: same VERBOSE gate as layers_forward_gguf's
        // [TRACE] hidden-norm dump, but for CONTENT (which experts, what
        // weight) rather than magnitude -- a routing bug (wrong experts
        // picked, e.g. from a bias-tensor or gating-function mixup) would
        // still produce numerically "healthy" norms downstream, since the
        // FFN math itself is correct, just applied to the wrong experts.
        // Only the LAST token (s==S-1) -- that's the one whose output
        // actually feeds the next-token prediction.
        if (s == S - 1 && Logger::instance().would_log(LogLevel::VERBOSE)) {
            LogStream ls(LogLevel::VERBOSE);
            ls << "[TRACE l=" << layer_idx << "] router picks:";
            for (int k = 0; k < topk; ++k) {
                ls << " e" << scratch.tok_topk_ids[s * topk + k]
                   << "(w=" << scratch.tok_topk_weights[s * topk + k] << ")";
            }
            ls << "\n";
        }
    }

    // --- Phase 2: cold-start VRAM warm-up (prefill only, S > 1) ---
    // A brand-new turn's prefill sees every prompt token in one shot, so
    // unlike steady-state decode (one token, one uncertain guess at a time)
    // we already have a full-batch demand signal *before* running a single
    // expert. Mirror how an OS warms an empty page cache on first touch:
    // load into any still-EMPTY VRAM slot immediately, prioritized by
    // in-batch frequency, since filling empty space costs nothing (no
    // eviction). Once the cache is full, fall back to the steady-state
    // hotness-gated promotion in RAMSafetyManager::get_expert() — same as
    // an OS reverting to LRU swap-out once the cache is warm.
    if (S > 1 && ram_mgr.vram_cache) {
        std::vector<int> candidates;
        candidates.reserve(n_experts);
        for (int e = 0; e < n_experts; ++e) if (scratch.expert_freq[e] > 0) candidates.push_back(e);
        std::sort(candidates.begin(), candidates.end(), [&](int a, int b) {
            return scratch.expert_freq[a] > scratch.expert_freq[b];
        });

        for (int expert_id : candidates) {
            if (!ram_mgr.vram_cache->has_free_slot()) break; // cache warm now — stop preloading
            if (ram_mgr.vram_cache->find_expert(layer_idx, expert_id)) continue; // already resident

            MoEExpertBlock* blk = ram_mgr.get_expert(engine, layer_idx, expert_id);
            if (!blk || !blk->slab_buffer) continue;

            // Same all-three-projections-must-be-GPU-supported gate as
            // RAMSafetyManager::get_expert's steady-state promotion — a
            // Dynamic-quant model where down uses an unimplemented format
            // (e.g. IQ3_XXS) can never dispatch to GPU regardless of VRAM
            // residency, so warming those into VRAM only wastes a free slot
            // that a genuinely GPU-capable expert could use later.
            bool gpu_capable = gpu_type_supported(blk->gate_proj.meta.type)
                             && gpu_type_supported(blk->up_proj.meta.type)
                             && gpu_type_supported(blk->down_proj.meta.type);
            if (!gpu_capable) continue;

            LOG_DEBUG("[MOE-VRAM]  WARM    layer=%d expert=%d freq=%d/%d tokens (cold-start prefill, free slot)\n",
                      layer_idx, expert_id, scratch.expert_freq[expert_id], S);
            ram_mgr.vram_cache->upload_expert_async(layer_idx, expert_id, blk->slab_buffer, blk->slab_capacity);
        }
    }

    // --- Phase 3: execute each token's already-selected experts ---
    for (int s = 0; s < S; ++s) {
        // FIX: both routed and shared expert FFNs must consume the SAME
        // ffn_norm'd input the router scored (matching the reference: llama.cpp
        // builds one normed `cur` tensor and feeds it to both the router and
        // build_moe_ffn's per-expert matmuls). This previously fed the raw,
        // un-normalized hidden_states into every expert FFN — router weights
        // were correct but the FFN itself was operating on the wrong input.
        const float* token_hidden = scratch.normed.data() + s * D;
        float* token_output = output + s * D;

        // --- Load all needed experts (LRU cache handles hits/misses; most
        // will already be warm here thanks to Phase 1/2 above) ---
        for (int k = 0; k < topk; ++k) {
            scratch.expert_ptrs[k] = ram_mgr.get_expert(engine, layer_idx, scratch.tok_topk_ids[s * topk + k]);
        }

        // Weighted accumulation + VERBOSE trace, shared by both the CPU path
        // (Pass A, accumulates immediately) and the GPU path (Pass B,
        // accumulates after finish()) -- same logic either way, just called
        // at a different point in the pipeline depending on path.
        auto accumulate = [&](int k, MoEExpertBlock* exp, const float* down_buf_ptr) {
            // Per-expert output norm, VERBOSE-gated: distinguishes "expert_ffn
            // itself computed all-zero" from "it computed something real but
            // the weighted accumulation below never applied it" -- check_finite
            // only catches NaN/Inf, never a suspiciously-exact zero, which is
            // exactly the symptom under investigation (moe_out|=0 every MoE
            // layer during prefill, colibri shows nonzero).
            if (Logger::instance().would_log(LogLevel::VERBOSE)) {
                double ss = 0.0; for (int d = 0; d < D; ++d) ss += (double)down_buf_ptr[d] * down_buf_ptr[d];
                LOG_STREAM_VERBOSE() << "[TRACE l=" << layer_idx << "] expert k=" << k
                                     << " id=" << exp->expert_id << " requested_id="
                                     << scratch.tok_topk_ids[s * topk + k]
                                     << " |down_buf|=" << std::sqrt(ss) << "\n";
            }

            // Weighted accumulation: output += weight * expert_output
            float w = scratch.tok_topk_weights[s * topk + k];
            if (std::isnan(w) || std::isinf(w)) {
                LOG_ERROR("[EXPERT-GPU] router weight itself is non-finite: layer=%d token=%d k=%d "
                          "expert=%d w=%f -- expert_ffn's own output can be perfectly clean and this "
                          "still poisons token_output; check router_logits/normalization above.\n",
                          layer_idx, s, k, exp->expert_id, w);
            }
            __m512 v_w = _mm512_set1_ps(w);
            int i = 0;
            for (; i + 16 <= D; i += 16) {
                __m512 v_out = _mm512_loadu_ps(token_output + i);
                __m512 v_exp = _mm512_loadu_ps(down_buf_ptr + i);
                _mm512_storeu_ps(token_output + i, _mm512_fmadd_ps(v_w, v_exp, v_out));
            }
            for (; i < D; ++i) token_output[i] += w * down_buf_ptr[i];
        };

        // --- Pass A: issue every GPU-path expert's gate/up kernels (async,
        // no sync), then immediately run each CPU-path expert synchronously
        // -- the GPU chews through its queued kernels in the background
        // while the CPU does this unrelated work, instead of the two never
        // overlapping. See the expert_ffn_gpu_issue/_finish header comment. ---
        scratch.pending_gpu.clear();
        for (int k = 0; k < topk; ++k) {
            MoEExpertBlock* exp = scratch.expert_ptrs[k];
            if (!exp || !exp->gate_proj.host_ptr || !exp->down_proj.host_ptr) {
                // Previously a completely silent skip -- if this fires for
                // EVERY k this would exactly explain moe_out staying at
                // exactly 0.0 for a whole layer (not just NaN-poisoned,
                // which check_finite already covers) with zero prior
                // diagnostic pointing at it.
                LOG_ERROR("[EXPERT-GPU] layer=%d token=%d k=%d expert_id=%d: expert_ptrs[k] is "
                          "%s -- skipping this expert's contribution entirely (exp=%p).\n",
                          layer_idx, s, k, scratch.tok_topk_ids[s * topk + k],
                          !exp ? "NULL" : (!exp->gate_proj.host_ptr ? "missing gate_proj.host_ptr"
                                                                     : "missing down_proj.host_ptr"),
                          (void*)exp);
                continue;
            }

            ExpertGpuContext gpu_ctx;
            gpu_ctx.topo = ram_mgr.vram_cache;
            gpu_ctx.layer_idx = layer_idx;
            gpu_ctx.expert_id = exp->expert_id;
            gpu_ctx.slab_buffer = exp->slab_buffer;
            gpu_ctx.slab_capacity = exp->slab_capacity;
            gpu_ctx.loc = ram_mgr.vram_cache ? ram_mgr.vram_cache->find_expert(layer_idx, exp->expert_id) : nullptr;

            const ExpertLocation* resolved = expert_ffn_resolve_gpu_loc(
                &gpu_ctx, 1, exp->gate_proj.meta, exp->up_proj.meta, exp->down_proj.meta);

            if (resolved) {
                GpuExpertScratch& gs = scratch.gpu_pool[scratch.pending_gpu.size()];
                expert_ffn_gpu_issue(gs, resolved, token_hidden, exp->gate_proj.meta, exp->up_proj.meta,
                                      1, D, moe_inter);
                scratch.pending_gpu.push_back({k, exp, resolved});
            } else {
                expert_ffn_cpu(
                    token_hidden, scratch.down_buf.data(),
                    exp->gate_proj.host_ptr, exp->gate_proj.meta,
                    exp->up_proj.host_ptr, exp->up_proj.meta,
                    exp->down_proj.host_ptr, exp->down_proj.meta,
                    scratch.gate_buf.data(), scratch.up_buf.data(),
                    1, D, moe_inter);
                accumulate(k, exp, scratch.down_buf.data());
            }
        }

        // --- Pass B: finish every GPU-issued expert queued above (sync +
        // download + down-proj), in issue order. By now the CPU has spent
        // Pass A's CPU-path time on unrelated experts, so the first sync per
        // expert should mostly find the gate/up kernels already done. ---
        for (size_t pi = 0; pi < scratch.pending_gpu.size(); ++pi) {
            PendingGpuExpert& p = scratch.pending_gpu[pi];
            GpuExpertScratch& gs = scratch.gpu_pool[pi];
            expert_ffn_gpu_finish(
                gs, p.loc,
                scratch.gate_buf.data(), scratch.up_buf.data(), scratch.down_buf.data(),
                p.exp->gate_proj.meta, p.exp->down_proj.meta,
                layer_idx, p.exp->expert_id, 1, D, moe_inter);
            accumulate(p.k, p.exp, scratch.down_buf.data());
        }

        check_finite("token_output (post routed experts, pre-shared)", layer_idx, -1, token_output, (size_t)D);

        // Shared expert (always-active) — GLM-5.2/DeepSeek-V3/Hunyuan MoE
        // layers can have n_expert_shared always-on experts whose combined
        // FFN (intermediate width n_expert_shared * moe_inter — NOT just
        // moe_inter, confirmed against deepseek2.cpp's shexp tensor shapes)
        // is added to every token's routed-expert output. Unscaled: only the
        // routed weights get expert_weights_scale (see build_moe_ffn — w_scale
        // is applied to the gathered top-k weights, never to the shared path).
        std::string sp = "blk." + std::to_string(layer_idx) + ".";
        if (c.n_expert_shared > 0 && engine.resident_layers.count(sp + "ffn_gate_shexp.weight")) {
            int shared_inter = moe_inter * c.n_expert_shared;
            dense_proj(engine, sp + "ffn_gate_shexp.weight", token_hidden, scratch.gate_buf.data(), 1, D, shared_inter);
            dense_proj(engine, sp + "ffn_up_shexp.weight",   token_hidden, scratch.up_buf.data(),   1, D, shared_inter);
            silu_inplace(scratch.gate_buf.data(), shared_inter);
            elem_mul(scratch.gate_buf.data(), scratch.gate_buf.data(), scratch.up_buf.data(), shared_inter);
            dense_proj(engine, sp + "ffn_down_shexp.weight", scratch.gate_buf.data(), scratch.down_buf.data(), 1, shared_inter, D);
            // Previously completely uninstrumented -- the shared-expert path
            // has no equivalent of expert_ffn's check_finite calls, so a bug
            // here (e.g. a dense-tensor GPU dispatch through the same fused
            // K-quant kernels that turned out to have the Q3_K struct-layout
            // bug) would silently poison every token's output with no log
            // ever pointing at it.
            check_finite("shexp down_buf (pre-accumulate)", layer_idx, -1, scratch.down_buf.data(), (size_t)D);
            for (int i = 0; i < D; ++i) token_output[i] += scratch.down_buf[i];
            check_finite("token_output (post shared expert)", layer_idx, -1, token_output, (size_t)D);
        }
    }
}

// ============================================================
// Dense (non-MoE) FFN for leading dense layers (l < first_dense) in an
// otherwise-MoE model — plain gate_proj -> SiLU, up_proj, elementwise
// multiply, down_proj on ffn_{gate,up,down}.weight. No router, no expert
// cache. Used by both GLM-5.2's leading dense layers and Hunyuan/Hy-V3
// layers before the MoE portion of the stack begins.
// ============================================================
void dense_ffn_gguf(
    StreamingEngine& engine,
    int layer_idx,
    const float* hidden_states,
    int S,
    float* output,
    LayerScratch& scratch
) {
    auto& c = engine.model_file.architecture_config;
    int D = c.hidden;
    int inter = c.dense_inter;
    if (inter <= 0) {
        std::fill(output, output + S * D, 0.0f);
        return;
    }

    std::string p = "blk." + std::to_string(layer_idx) + ".";
    auto norm_it = engine.resident_layers.find(p + "ffn_norm.weight");
    if (norm_it != engine.resident_layers.end() && norm_it->second.host_ptr) {
        // Same bug as moe_forward_gguf's router norm (see its comment):
        // rmsnorm_row only normalizes ONE row -- must loop over S or every
        // token past s=0 reads stale scratch.normed during prefill.
        const float* norm_w = reinterpret_cast<float*>(norm_it->second.host_ptr);
        for (int s = 0; s < S; ++s) {
            rmsnorm_row(scratch.normed.data() + (size_t)s * D, hidden_states + (size_t)s * D,
                        norm_w, D, c.eps);
        }
    } else {
        std::memcpy(scratch.normed.data(), hidden_states, (size_t)S * D * sizeof(float));
    }

    for (int s = 0; s < S; ++s) {
        const float* token_hidden = scratch.normed.data() + (size_t)s * D;
        float* token_output = output + (size_t)s * D;

        dense_proj(engine, p + "ffn_gate.weight", token_hidden, scratch.gate_buf.data(), 1, D, inter);
        dense_proj(engine, p + "ffn_up.weight",   token_hidden, scratch.up_buf.data(),   1, D, inter);
        silu_inplace(scratch.gate_buf.data(), inter);
        elem_mul(scratch.gate_buf.data(), scratch.gate_buf.data(), scratch.up_buf.data(), inter);
        dense_proj(engine, p + "ffn_down.weight", scratch.gate_buf.data(), token_output, 1, inter, D);
    }
}

// ============================================================
// Main layer-by-layer forward pass with pre-allocated scratch,
// speculative pilot prefetch, and performance instrumentation.
// ============================================================
void layers_forward_gguf(StreamingEngine& engine, RAMSafetyManager& ram_mgr,
                         LockFreePrefetcher& async_io_queue, float* hidden_states, int S, int pos_base) {
    auto& c = engine.model_file.architecture_config;
    int D = c.hidden;

    // Pre-allocate scratch once (avoids per-layer heap churn)
    static thread_local LayerScratch scratch;
    scratch.resize(S, D, c.n_experts, c.moe_inter, c.topk, c.n_expert_shared, c.dense_inter);

    // Diagnostic: L2 norm of the LAST TOKEN's hidden state (D floats, not the
    // whole S*D prefill batch -- matches colibri's identical trace exactly,
    // see colibri_trace.log comparison, so the two are directly comparable
    // without a sqrt(S) skew). EVERY layer now (not just a 5-point spread):
    // that spread showed MoEStream's growth compounding to ~30x colibri's by
    // the last few layers while starting within ~2x at layer 0, meaning the
    // excess accumulates gradually across many layers rather than in one bad
    // layer -- full resolution is needed to see exactly where it accelerates.
    auto l2norm = [](const float* p, int n) {
        double s = 0.0; for (int i = 0; i < n; ++i) s += (double)p[i] * p[i];
        return std::sqrt(s);
    };
    int n = S * D;  // hoisted above the loop -- was declared after its first use at "[L0] in" below
    bool trace = Logger::instance().would_log(LogLevel::VERBOSE);  // gate BEFORE paying for l2norm, not just before printing
    for (int l = 0; l < c.n_layers; ++l) {
        const float* last_tok = hidden_states + (size_t)(S - 1) * D;  // hidden_states is mutated in place below, not reallocated -- always valid
        if (trace) LOG_STREAM_VERBOSE() << "[TRACE l=" << l << "] in      |hidden|=" << l2norm(last_tok, D) << "\n";
        // --- Attention: MLA (GLM-5.2/DeepSeek-style, absorbed + latent KV
        // cache) vs dense GQA causal attention (Hunyuan/Hy-V3/Llama-style,
        // real Kc/Vc cache) ---
        if (c.is_mla) {
            mla_attention_absorbed_gguf(engine, l, hidden_states, S, pos_base, scratch.attn_out.data());
        } else {
            dense_attention_gguf(engine, l, hidden_states, S, pos_base, scratch.attn_out.data());
        }

        // Residual add
        int i = 0;
        for (; i + 16 <= n; i += 16) {
            __m512 h = _mm512_loadu_ps(hidden_states + i);
            __m512 a = _mm512_loadu_ps(scratch.attn_out.data() + i);
            _mm512_storeu_ps(hidden_states + i, _mm512_add_ps(h, a));
        }
        for (; i < n; ++i) hidden_states[i] += scratch.attn_out[i];

        // Per-layer, per-stage checkpoint: pinpoints the FIRST layer/stage
        // where hidden_states goes non-finite, instead of only learning
        // "somewhere in this forward pass" from the aggregate
        // after-prefill/after-decode-step checks in generation_loop.hpp.
        // check_finite() is a no-op (silent) unless it actually finds
        // NaN/Inf, so this is cheap on the (expected) all-finite path.
        check_finite("hidden_states (post attention residual)", l, -1, hidden_states, (size_t)n);
        if (trace) LOG_STREAM_VERBOSE() << "[TRACE l=" << l << "] post-attn |attn_out|=" << l2norm(scratch.attn_out.data() + (size_t)(S - 1) * D, D)
                                      << " |hidden|=" << l2norm(last_tok, D) << "\n";

        // --- Speculative pilot prefetch for next layer's experts ---
        if (l + 1 < c.n_layers && c.n_experts > 0) {
            PERF_START(pilot);
            std::string norm_key = "blk." + std::to_string(l + 1) + ".ffn_norm.weight";
            std::string router_key = "blk." + std::to_string(l + 1) + ".ffn_gate_inp.weight";
            auto norm_it = engine.resident_layers.find(norm_key);
            auto router_it = engine.resident_layers.find(router_key);
            
            if (norm_it != engine.resident_layers.end() && router_it != engine.resident_layers.end()
                && norm_it->second.host_ptr && router_it->second.host_ptr) {
                // Use only last token for pilot prediction (saves compute)
                const float* last_hidden = hidden_states + ((S - 1) * D);
                rmsnorm_row(scratch.pilot_nrm.data(), last_hidden,
                           reinterpret_cast<float*>(norm_it->second.host_ptr), D, c.eps);

                dense_proj(engine, router_key,
                           scratch.pilot_nrm.data(), scratch.pilot_logits.data(), 1, D, c.n_experts);

                int pilot_k = std::min(8, c.topk);
                for (int k = 0; k < pilot_k; ++k) {
                    int best_exp = 0;
                    for (int e = 1; e < c.n_experts; ++e) {
                        if (scratch.pilot_logits[e] > scratch.pilot_logits[best_exp]) best_exp = e;
                    }
                    scratch.pilot_logits[best_exp] = -2e30f;
                    async_io_queue.push_hint(l + 1, best_exp);
                }
            }
            PERF_END(pilot, t_prefetch);
        }

        // --- FFN: leading dense layers (l < first_dense) use a plain dense
        // FFN — no router, no expert cache. Applies to both GLM-5.2's leading
        // dense layers and, since Hunyuan/Hy-V3 share the same first_dense
        // convention, to any dense layers there too. ---
        {
            PERF_SCOPE(t_moe_expert);
            if (l < c.first_dense) {
                dense_ffn_gguf(engine, l, hidden_states, S, scratch.moe_out.data(), scratch);
            } else {
                moe_forward_gguf(engine, ram_mgr, l, hidden_states, S, scratch.moe_out.data(), scratch);
            }
        }

        // Residual add
        i = 0;
        for (; i + 16 <= n; i += 16) {
            __m512 h = _mm512_loadu_ps(hidden_states + i);
            __m512 m = _mm512_loadu_ps(scratch.moe_out.data() + i);
            _mm512_storeu_ps(hidden_states + i, _mm512_add_ps(h, m));
        }
        for (; i < n; ++i) hidden_states[i] += scratch.moe_out[i];

        check_finite("hidden_states (post FFN/MoE residual)", l, -1, hidden_states, (size_t)n);
        if (trace) LOG_STREAM_VERBOSE() << "[TRACE l=" << l << "] post-ffn  |moe_out|=" << l2norm(scratch.moe_out.data() + (size_t)(S - 1) * D, D)
                                      << " |hidden|=" << l2norm(last_tok, D) << "\n";
    }
}