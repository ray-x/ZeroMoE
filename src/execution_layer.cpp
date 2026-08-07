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
#include <array>
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
extern "C" void launch_iq4_xs_matmul(float* y, const float* x, const void* weights, int S, int I, int O);
extern "C" void launch_tq1_matmul(float* y, const float* x, const void* weights, const void* scales, int S, int I, int O);
extern "C" void launch_iq1s_matmul(float* y, const float* x, const void* weights, int S, int I, int O);
extern "C" void launch_iq1m_matmul(float* y, const float* x, const void* weights, int S, int I, int O);
// MXFP4: dormant kernel, wired in but not yet dispatched to -- see
// kquant_dequant.hpp's gpu_type_supported() comment.
extern "C" void launch_mxfp4_matmul(float* y, const float* x, const void* weights, int S, int I, int O);

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
    // Returns false if a needed grow's cudaMalloc failed -- previously
    // unchecked, so a failed grow left d_x/d_gate/d_up/d_prod/d_down
    // either null or (worse) a stale non-null pointer from a SMALLER prior
    // allocation while cap_sd/cap_si were already bumped to the bigger
    // "needed" size, so a later call with the SAME size would wrongly skip
    // reallocating and the caller would proceed to read/write past the
    // real allocation. On any failure here, cap_sd/cap_si are reset to 0
    // (not left at the just-computed "needed" value) so the NEXT call
    // retries cleanly instead of trusting a partially-grown state.
    bool ensure(int S, int D, int moe_inter, int target_device) {
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
            if (d_x)    { cudaFree(d_x);    d_x = nullptr; }
            if (d_down) { cudaFree(d_down); d_down = nullptr; }
            cap_sd = 0;
            cudaError_t err1 = cudaMalloc((void**)&d_x, need_sd * sizeof(float));
            cudaError_t err2 = (err1 == cudaSuccess)
                ? cudaMalloc((void**)&d_down, need_sd * sizeof(float)) : err1;
            if (err1 != cudaSuccess || err2 != cudaSuccess) {
                LOG_ERROR("[EXPERT-GPU] GpuExpertScratch::ensure: d_x/d_down grow to %.2f MB "
                          "failed on device %d: %s\n", (need_sd * sizeof(float)) / 1e6,
                          device_id, cudaGetErrorString(err1 != cudaSuccess ? err1 : err2));
                if (d_x)    { cudaFree(d_x);    d_x = nullptr; }
                if (d_down) { cudaFree(d_down); d_down = nullptr; }
                return false;
            }
            cap_sd = need_sd;
        }
        if (cap_si < need_si) {
            if (d_gate) { cudaFree(d_gate); d_gate = nullptr; }
            if (d_up)   { cudaFree(d_up);   d_up = nullptr; }
            if (d_prod) { cudaFree(d_prod); d_prod = nullptr; }
            cap_si = 0;
            cudaError_t err1 = cudaMalloc((void**)&d_gate, need_si * sizeof(float));
            cudaError_t err2 = (err1 == cudaSuccess)
                ? cudaMalloc((void**)&d_up, need_si * sizeof(float)) : err1;
            cudaError_t err3 = (err2 == cudaSuccess)
                ? cudaMalloc((void**)&d_prod, need_si * sizeof(float)) : err2;
            if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess) {
                cudaError_t first_err = err1 != cudaSuccess ? err1 : (err2 != cudaSuccess ? err2 : err3);
                LOG_ERROR("[EXPERT-GPU] GpuExpertScratch::ensure: d_gate/d_up/d_prod grow to %.2f MB "
                          "failed on device %d: %s\n", (need_si * sizeof(float)) / 1e6,
                          device_id, cudaGetErrorString(first_err));
                if (d_gate) { cudaFree(d_gate); d_gate = nullptr; }
                if (d_up)   { cudaFree(d_up);   d_up = nullptr; }
                if (d_prod) { cudaFree(d_prod); d_prod = nullptr; }
                return false;
            }
            cap_si = need_si;
        }
        return true;
    }

    // Frees every buffer back to empty (cap=0). Safe any time between
    // requests -- the next issue() call regrows via ensure() above,
    // starting from 0 instead of a stale cap. In practice this buffer
    // never grows much (expert FFN dispatch is always called with S=1,
    // even during prefill -- see layers_forward_gguf's per-token loop --
    // so this is defense-in-depth / consistency with shrink_dense_scratch
    // rather than reclaiming a large amount, unlike that one).
    void shrink() {
        if (device_id < 0) return;
        cudaSetDevice(device_id);
        if (d_x)    { cudaFree(d_x);    d_x = nullptr; }
        if (d_down) { cudaFree(d_down); d_down = nullptr; }
        if (d_gate) { cudaFree(d_gate); d_gate = nullptr; }
        if (d_up)   { cudaFree(d_up);   d_up = nullptr; }
        if (d_prod) { cudaFree(d_prod); d_prod = nullptr; }
        cap_sd = 0;
        cap_si = 0;
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
        case GGUFType::IQ4_XS:  launch_iq4_xs_matmul(y, x, w, S, I, O);  return true;
        // MXFP4: dead code until kquant_dequant.hpp's gpu_type_supported()
        // flips MXFP4 to true (post real-hardware Test 5 pass) -- wired
        // here now so that flip is a genuine one-line change with nothing
        // else left to plumb.
        case GGUFType::MXFP4:   launch_mxfp4_matmul(y, x, w, S, I, O);   return true;
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
    // Non-owning pointer to the RAM-tier MoEExpertBlock slab_buffer/
    // slab_capacity above were copied out of. Added to close two of the
    // "[TIER-OVERLAP]" leaks documented in CLAUDE.md ("Cold-Miss GPU
    // Dispatch Never Frees the RAM Slot"): expert_ffn_resolve_gpu_loc's
    // own synchronous upload_expert/upload_expert_active calls promote
    // this RAM data into VRAM but, without this pointer, had no way to
    // tell the RAM tier its copy is now redundant. Null is safe (no
    // marking attempted) for any future caller that doesn't have a RAM-
    // tier block to reference.
    MoEExpertBlock* exp = nullptr;
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
    // True iff expert_ffn_resolve_gpu_loc took a temporary
    // hold_for_compute() on `loc` (the decode-time cold-miss branch) that
    // must be released once this expert's kernels are confirmed done --
    // see expert_ffn_gpu_finish. False for the pre-existing S>=4 cold-batch
    // path and the already-VRAM-resident case, neither of which take a hold.
    bool needs_release = false;
    // Which pool `loc` resolved from, threaded through so the eventual
    // release_hold() call (expert_ffn_gpu_finish, possibly much later than
    // the matching hold_for_compute) can search only that one pool instead
    // of guess-scanning both -- see hold_for_compute/release_hold's own
    // comments in gpu_topology.hpp. Only meaningful when needs_release is
    // true.
    bool loc_from_active_layer = false;
    // Captured from gpu_ctx.expert_id at Pass A issue time, BEFORE
    // expert_ffn_resolve_gpu_loc runs (which can promote this exact `exp`
    // into the main VRAM cache and, on a real RAM-tier reclaim, clear
    // exp->expert_id to -1). Pass B's expert_ffn_gpu_finish call and this
    // struct's own release_hold rollback path both need the REAL identity
    // to find the right VRAM slot -- previously both read exp->expert_id
    // FRESH at their own (much later) call time instead of using a
    // captured value, which is exactly what let a same-dispatch RAM-tier
    // reclaim corrupt release_hold's lookup (a real, observed crash: see
    // CLAUDE.md's active-reclaim entries). This field exists so Pass B
    // never needs to trust exp's mutable state again for this one purpose.
    int expert_id = -1;
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

    // Metadata-only "shadow" blocks (StreamingEngine::populate_vram_shadow)
    // for experts already confirmed GPU-resident before the per-token
    // dispatch loop runs -- avoids the real RAM LRU load (a real disk read
    // + a real RAM slot) that moe_forward_gguf used to pay unconditionally
    // for EVERY routed expert regardless of GPU residency (the
    // "[TIER-OVERLAP]" finding, see CLAUDE.md). One slot per topk index,
    // reused across calls like every other LayerScratch buffer.
    std::vector<MoEExpertBlock> vram_shadow_blocks; // [topk]
    // The find_expert() result from the shadow pre-check above, threaded
    // through to Pass A so it doesn't have to call find_expert() AGAIN for
    // the same expert a few lines later -- a real, measured cost (a real
    // run showed Cache mutex wait roughly double, 4.185s vs 1.4-2.0s
    // before, from this exact redundant second lookup). Safe to reuse:
    // hold_for_compute() (called later in Pass A) does its own independent,
    // fresh identity-based re-scan of expert_vram_cache regardless of which
    // pointer is passed in -- it never trusts this pointer's current
    // contents, only (layer, expert_id) -- so a stale entry here just makes
    // the hold correctly fail, exactly like a stale entry from a second
    // find_expert() call would. nullptr for any k that wasn't shadowed.
    std::vector<const ExpertLocation*> vram_loc_hint; // [topk]

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

    // Pilot RECALL instrumentation (diagnostic only, no behavior change):
    // the pilot's full predicted top-K for pilot_pred_layer, captured
    // BEFORE the already-resident skip filter so this measures pure
    // prediction accuracy -- completely independent of the downstream
    // serving pipeline (worker thread speed, staging pool size) that
    // "Prefetch served %" conflates together. Consumed (and reset
    // to -1) the next time moe_forward_gguf runs for pilot_pred_layer, at
    // the same s==S-1 "last token" position the pilot itself predicted
    // from -- an apples-to-apples comparison for both decode (S=1, the
    // only token) and prefill (S>1, same last-token position both sides
    // use). -1 = no prediction currently pending for any layer.
    std::array<int, 8> pilot_pred_ids{};
    int pilot_pred_count = 0;
    int pilot_pred_layer = -1;
    // Parallel to pilot_pred_ids: whether pilot_pred_ids[k] was actually
    // "fetch-worthy" (not already cached anywhere) at prediction time, and
    // so was actually pushed as a hint. Lets the recall-check site below
    // compute a SECOND accuracy figure restricted to the subset of
    // predictions that would actually trigger a real fetch -- the number
    // that determines whether prefetch is worth enabling, as opposed to
    // pilot_recall's raw top-K accuracy (which includes already-cached
    // picks whose correctness doesn't matter to that question).
    std::array<bool, 8> pilot_pred_was_hint{};

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
        vram_shadow_blocks.resize(std::max(1, topk));
        vram_loc_hint.assign(std::max(1, topk), nullptr);
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
// residency (including the synchronous cold-batch upload for large S, or
// the decode-time cold-miss synchronous upload below). Returns nullptr if
// the CPU path should be taken. *out_needs_release is set true iff this
// call took a temporary hold_for_compute() on the returned slot that the
// caller MUST match with exactly one GpuTopology::release_hold() once its
// GPU kernels are confirmed done reading that slot (see
// expert_ffn_gpu_finish) -- true for EVERY path that returns a non-null
// location (already-VRAM-resident, the S>=4 cold-batch upload, and the
// decode-time cold-miss upload alike, see the hold_for_compute() call at
// the end of this function). Was previously true ONLY for the decode-time
// branch, which was a real bug -- see that call site's own comment.
static const ExpertLocation* expert_ffn_resolve_gpu_loc(
    const ExpertGpuContext* gpu_ctx, int S,
    const GGUFTensor& gate_meta, const GGUFTensor& up_meta, const GGUFTensor& down_meta,
    bool* out_needs_release, bool* out_loc_from_active_layer,
    RAMSafetyManager& ram_mgr
) {
    *out_needs_release = false;
    const ExpertLocation* loc = gpu_ctx ? gpu_ctx->loc : nullptr;
    // cpu_affinity_layers (config): this specific layer is pinned to CPU --
    // a genuine early return, NOT the same trick cpu_only_mode() uses just
    // below (forcing vram_resident=false alone would NOT be enough here:
    // the cold-batch/decode-miss/active-layer upload branches further down
    // all key on `!vram_resident`, so forcing it false would actually
    // TRIGGER an upload attempt rather than prevent one -- confirmed by
    // reading through to those branches before writing this). Returning
    // here, before any of those branches run, guarantees no
    // upload_expert/upload_expert_active call is ever reached for this
    // layer from this function, matching the config's contract ("never
    // DMA into VRAM for these layers").
    if (gpu_ctx && is_cpu_affinity_layer(gpu_ctx->layer_idx)) {
        perf().n_cpu_fallback_affinity_layer.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    bool vram_resident = loc && loc->device_ptr && loc->layer_idx >= 0;
    if (cpu_only_mode()) vram_resident = false;  // diagnostic: force CPU expert path
    // Tracks which pool `loc` most recently resolved from, so hold_for_compute/
    // release_hold (called once this function settles on a final `loc`) can
    // search ONLY that pool instead of guess-scanning both -- see their own
    // comments in gpu_topology.hpp for the real cost this avoids. Every
    // assignment to `loc` below must keep this in sync: false for anything
    // resolved via the main expert_vram_cache (gpu_ctx->loc, upload_expert),
    // true for anything resolved via find_active_expert/upload_expert_active.
    bool loc_from_active_layer = false;

    // Batch large enough to amortize a cold upload: fetch synchronously now,
    // since (unlike background promotion) this call needs the data
    // immediately to compute this token's result.
    if (!vram_resident && gpu_ctx && gpu_ctx->topo && gpu_ctx->slab_buffer
        && gpu_ctx->slab_capacity > 0 && S >= GPU_COLD_BATCH_THRESHOLD) {
        loc = gpu_ctx->topo->upload_expert(gpu_ctx->layer_idx, gpu_ctx->expert_id,
                                            gpu_ctx->slab_buffer, gpu_ctx->slab_capacity);
        vram_resident = loc && loc->device_ptr;
        // FIX ("[TIER-OVERLAP]" leak, see CLAUDE.md): upload_expert is
        // SYNCHRONOUS -- by the time it returns, the H2D copy is
        // unconditionally complete, so (unlike an async promotion) it's
        // immediately safe to reclaim the source RAM block, same
        // safe-confirmed-point reasoning get_expert's own MOVE branch
        // already uses. This is the S>=GPU_COLD_BATCH_THRESHOLD cold-batch
        // path -- gpu_ctx->exp is the real RAM-tier block slab_buffer came
        // from (null only for a shadow, which never reaches here).
        // Upgraded from also_in_vram-only marking to finish_sync_promotion
        // (active reclaim) once PendingGpuExpert started capturing
        // expert_id at issue time (see that struct's own comment) -- that
        // fix closed the Pass A->B/rollback identity-read-late hazard for
        // EVERY synchronous-upload site in this function, not just
        // find_active_expert's, so this site no longer needs to avoid an
        // active MOVE either.
        if (vram_resident && gpu_ctx->exp) ram_mgr.finish_sync_promotion(*gpu_ctx->exp, gpu_ctx->layer_idx, gpu_ctx->expert_id);
    }

    // Decode-time cold miss (S < GPU_COLD_BATCH_THRESHOLD, in practice
    // always S==1): closes the "miss pays CPU too" gap -- previously a
    // decode-time miss ALWAYS fell to the ~9ms CPU dequant-matmul fallback,
    // even when the expert's quant type has a GPU kernel that would cost
    // ~2ms, because promotion only ever happened in the BACKGROUND after a
    // LATER hit proved the expert hot. A prior attempt at this exact fix
    // used an ASYNC upload ordered via stream-0 launch order alone, which
    // caused real memory corruption (see CLAUDE.md's "Decode-time cache-
    // miss GPU dispatch: attempted, reverted" section) -- the async copy's
    // completion event fired as soon as the COPY landed, well before the
    // kernels reading that data had actually finished executing, so a
    // DIFFERENT thread's background eviction could steal the slot mid-
    // compute. This version is different in kind, not degree: (1) uses the
    // SYNCHRONOUS upload_expert() (same call the S>=4 branch above already
    // uses safely) so there is no copy-vs-kernel-launch race to reason
    // about at all -- by the time this call returns, the data is
    // unconditionally present; (2) additionally holds the slot via
    // hold_for_compute() (reference-counted, evictlru-respected) for the
    // ENTIRE issue-to-finish window, released only after
    // expert_ffn_gpu_finish's cudaStreamSynchronize(0) confirms the
    // asynchronous kernel EXECUTION (not just the copy) is actually done --
    // closing the race the previous attempt missed. Type support is
    // checked FIRST, before spending the synchronous upload's real latency
    // on a type that could never dispatch anyway.

    // Three-tier caching: Try active layer area first for temporary storage
    // This avoids evicting valuable LRU residents for current-layer experts
    if (!vram_resident && gpu_ctx && gpu_ctx->topo && gpu_ctx->slab_buffer
        && gpu_ctx->slab_capacity > 0 && S < GPU_COLD_BATCH_THRESHOLD
        && gpu_type_supported(gate_meta.type) && gpu_type_supported(up_meta.type)
        && gpu_type_supported(down_meta.type)) {

        // Check for an existing active-layer resident FIRST -- a real hit
        // (this expert was already uploaded here for an earlier token still
        // within this layer's window). find_active_expert bumps hit_count
        // and, if it crosses active_layer_promote_threshold, promotes this
        // expert into the real VRAM LRU cache right away (freeing this
        // active-layer slot) instead of waiting for cleanup_active_layer_area
        // at the next layer transition.
        //
        // FIX (real bug, found while chasing [TIER-OVERLAP]): a promoted hit
        // returns a pointer into the MAIN expert_vram_cache, NOT the active-
        // layer pool -- but loc_from_active_layer was previously set `true`
        // unconditionally for every non-null return here, regardless of
        // which pool it actually came from. That mislabeling made every
        // SUBSEQUENT hold_for_compute() call for this dispatch search ONLY
        // active_layer_slots for a device_ptr that actually lives in the
        // separate expert_cache_pool allocation -- guaranteed not found,
        // silently forcing a CPU fallback for a genuinely VRAM-resident
        // expert every time a promotion happened via this path. promoted_to_main
        // (set by find_active_expert's on_promoted callback, which fires
        // exactly when the returned loc is main-cache) fixes this.
        //
        // REGRESSION, FOUND AND ROOT-CAUSE-FIXED ON REAL HARDWARE (two
        // rounds): this callback previously called
        // ram_mgr.finish_sync_promotion(*gpu_ctx->exp, ...) here (an active
        // MOVE, clearing gpu_ctx->exp->expert_id=-1 immediately) to close
        // the separate [TIER-OVERLAP] leak. A real run hit this
        // immediately: "[ERROR] [MOE-VRAM] release_hold found no matching
        // resident slot for layer=X expert=-1" right after the corresponding
        // "ACTIVE-LAYER HIT ... (promoted to main cache)" line for the SAME
        // expert. Root cause: gpu_ctx->exp is the SAME MoEExpertBlock
        // currently being dispatched for THIS k, about to be pushed into
        // scratch.pending_gpu (see dispatch_expert_k below) for Pass B to
        // process later -- and Pass B used to read p.exp->expert_id FRESH
        // at release time (not a captured value) to call release_hold().
        // Per release_hold's own failure-path consequence (see CLAUDE.md's
        // ram_temp_pin_count entry), every occurrence permanently leaked
        // the real VRAM slot as un-evictable, measurably tanking that run
        // (cache hit rate down to 47%, disk back up to 97s/230GB,
        // throughput down to 0.21 tok/s). ROUND 1 fix reverted to
        // marking-only here (safe, but leaves [TIER-OVERLAP] barely
        // improved on the dominant promotion path). ROUND 2 (this version):
        // fixed the ACTUAL root cause instead of avoiding it -- PendingGpuExpert
        // now captures expert_id at Pass A issue time, BEFORE this callback
        // can ever run, so Pass B (and this function's own immediate-
        // rollback release_hold call below, same bug) no longer depend on
        // exp's mutable state at all. With that dependency gone, actively
        // reclaiming the RAM slot here is safe again -- re-enabled.
        bool promoted_to_main = false;
        loc = gpu_ctx->topo->find_active_expert(gpu_ctx->layer_idx, gpu_ctx->expert_id,
            [&](int layer, int expert_id) {
                promoted_to_main = true;
                if (gpu_ctx->exp) ram_mgr.finish_sync_promotion(*gpu_ctx->exp, layer, expert_id);
            });
        vram_resident = loc && loc->device_ptr;
        if (vram_resident) {
            loc_from_active_layer = !promoted_to_main;
            LOG_DEBUG("[MOE-VRAM]  ACTIVE-LAYER HIT layer=%d expert=%d hit_count=%u%s\n",
                      gpu_ctx->layer_idx, gpu_ctx->expert_id, loc->hit_count,
                      promoted_to_main ? " (promoted to main cache)" : "");
        } else {
            // First touch this layer-window -- try active layer area (decode-
            // thread-only, no eviction of the main LRU cache)
            loc = gpu_ctx->topo->upload_expert_active(gpu_ctx->layer_idx, gpu_ctx->expert_id,
                                                       gpu_ctx->slab_buffer, gpu_ctx->slab_capacity);
            vram_resident = loc && loc->device_ptr;

            if (vram_resident) {
                loc_from_active_layer = true;
                // REVERTED (real, measured regression -- see CLAUDE.md):
                // this branch used to also set gpu_ctx->exp->also_in_vram =
                // true here, on the theory that it's "the same synchronous-
                // therefore-safe reasoning as the cold-batch branch above."
                // That reasoning doesn't hold for the ACTIVE LAYER
                // specifically: it's an explicitly TEMPORARY pool (~90% of
                // uploads are freed at the next layer transition WITHOUT
                // ever being promoted to the durable main cache -- a real
                // run showed 17546 active-layer uploads vs only 1795
                // promoted). Marking the RAM source "redundant, evict
                // first" at UPLOAD time -- before any hit even confirms
                // this expert will survive past this token's layer window
                // -- means the RAM tier aggressively evicted slots whose
                // VRAM backing had ALREADY vanished by the time anything
                // tried to reuse the freed capacity, forcing an unnecessary
                // disk reload on the very next real need. Measured
                // fallout: RAM-tier hit rate collapsed from the usual
                // 42-49% to 28.5%, and RAM-tier fill dropped from 99.9% to
                // 71.2% (slots churning empty faster than being kept
                // useful). The cold-batch and decode-miss-fallback
                // upload_expert sites (this function's OTHER two
                // also_in_vram markings) do NOT have this problem -- they
                // promote into the MAIN cache, a durable LRU pool where a
                // resident genuinely stays until legitimately evicted by
                // score, matching the assumption get_expert's own COPY
                // branch already relies on safely. This revert only stopped
                // the regression -- it deliberately did NOT mark also_in_vram
                // here (upload time, temporary residency, unsafe) nor attempt
                // to notify the RAM tier once a promotion to the durable main
                // cache actually happens later. That correct notification is
                // now wired up separately, at the point it's actually safe:
                // see the find_active_expert call above (its on_promoted
                // callback + the also_in_vram marking inside it) and
                // cleanup_active_layer_area's call in moe_forward_gguf, both
                // of which fire only once promote_active_to_main's
                // synchronous D2D copy has genuinely landed in the main
                // cache -- closing the "[TIER-OVERLAP]" leak this comment
                // used to describe as still-open (see CLAUDE.md).
                LOG_DEBUG("[MOE-VRAM]  ACTIVE-LAYER layer=%d expert=%d -> temporary upload (no LRU eviction)\n",
                          gpu_ctx->layer_idx, gpu_ctx->expert_id);
            } else {
                // Fallback: cold-miss synchronous upload to regular VRAM cache
                loc = gpu_ctx->topo->upload_expert(gpu_ctx->layer_idx, gpu_ctx->expert_id,
                                                    gpu_ctx->slab_buffer, gpu_ctx->slab_capacity);
                vram_resident = loc && loc->device_ptr;
                loc_from_active_layer = false;  // main cache, even though the active-layer branches above ran first
                if (vram_resident) {
                    // FIX ("[TIER-OVERLAP]" leak, see CLAUDE.md): third of
                    // three synchronous-upload sites in this function.
                    // Upgraded to active reclaim -- see the cold-batch
                    // site's own comment above for why this is now safe.
                    if (gpu_ctx->exp) ram_mgr.finish_sync_promotion(*gpu_ctx->exp, gpu_ctx->layer_idx, gpu_ctx->expert_id);
                    LOG_DEBUG("[MOE-VRAM]  COLD-GPU layer=%d expert=%d -> synchronous H2D upload on decode-time "
                              "miss (avoids CPU dequant-matmul fallback)\n",
                              gpu_ctx->layer_idx, gpu_ctx->expert_id);
                }
            }
        }
    }

    if (!vram_resident) {
        // Attribute WHY this call is about to fall back to CPU -- previously
        // conflated into one unmeasured "!vram_resident" branch. THREE
        // different root causes, not two: (1) this expert's quant type has
        // no GPU kernel at all, so the upload block above never even
        // attempted a promotion -- a fixed, model-intrinsic cost, entirely
        // independent of cache/pin sizing; (2) the type WAS supported, no
        // host buffer was available (gpu_ctx->slab_buffer null) so NO
        // upload could be structurally attempted at all -- every upload
        // guard above requires a non-null slab_buffer, which a shadow-fast-
        // pathed expert (exp->vram_shadow) never carries by design, so this
        // is the signature of a shadow pre-check racing and losing its VRAM
        // residency before this dispatch (see the caller's "emergency RAM
        // reload" comment -- previously assumed rare, never counted); vs
        // (3) the type was supported, a host buffer WAS available, and
        // upload_expert was genuinely called but failed -- see its own
        // n_upload_expert_no_victim/n_upload_expert_memcpy_fail counters
        // for the specific cudaMemcpy/evict_lru cause. Split (2) out from
        // (3) after a real run showed the WHOLE bucket (2668 calls) with
        // only ONE actual upload_expert error logged -- proof almost none
        // of them were genuine upload attempts. Added after hold_for_
        // compute's own fail counter came back at 0 on a real run with a
        // rising CPU-fallback count, ruling that mechanism out.
        bool type_ok = gpu_type_supported(gate_meta.type) && gpu_type_supported(up_meta.type)
                     && gpu_type_supported(down_meta.type);
        if (!type_ok) {
            perf().n_cpu_fallback_type_unsupported.fetch_add(1, std::memory_order_relaxed);
        } else if (!gpu_ctx || !gpu_ctx->slab_buffer) {
            perf().n_cpu_fallback_no_host_buffer.fetch_add(1, std::memory_order_relaxed);
        } else if (gpu_ctx->slab_capacity == 0) {
            // Expected, benign: this RAM slot was just MOVE-promoted to
            // VRAM (get_expert/get_experts_batch's move_on_promotion),
            // which zeroes slab_capacity immediately while leaving
            // gate_proj/up_proj/down_proj's host pointers untouched --
            // slab_buffer is still non-null (the allocation itself isn't
            // freed), but every upload guard above requires slab_capacity
            // > 0, so no upload can be attempted THIS call. The CPU
            // fallback below still reads valid data via the unaffected
            // gate_proj/down_proj pointers -- this is a real, if brief,
            // missed-GPU-dispatch cost (the expert needs one more
            // reactive hit before it's usable via the VRAM path), not a
            // failure. Split out after a real run showed the WHOLE
            // upload_failed bucket (2668 calls) with only ONE genuine
            // upload_expert error -- this bucket, not a hidden bug, was
            // the dominant explanation once the shadow-race hypothesis
            // (no_host_buffer) also came back low.
            perf().n_cpu_fallback_just_promoted.fetch_add(1, std::memory_order_relaxed);
        } else {
            perf().n_cpu_fallback_upload_failed.fetch_add(1, std::memory_order_relaxed);
        }
        return nullptr;
    }

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
        // Same underlying reason as the !vram_resident branch's type_ok
        // check above -- an already-resident expert (via some OTHER path,
        // e.g. reactive promotion) whose type still can't dispatch. Counted
        // into the SAME bucket since both mean "this expert's quant type
        // has no GPU kernel", just reached via a different code path.
        perf().n_cpu_fallback_type_unsupported.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }

    // FIX (real bug, found by re-reading this function after reports of
    // non-deterministic NaN that came and went independent of any single
    // recent code change -- exactly the signature of a race, not a per-
    // kernel logic bug): hold_for_compute() used to be called ONLY inside
    // the decode-time-cold-miss branch above, meaning EVERY OTHER path
    // that reaches this point with a valid `loc` -- an expert that was
    // ALREADY VRAM-resident before this function was even called (the
    // MOST COMMON case, and more so now with this session's more
    // aggressive pin_threshold/promote_hit_threshold settings keeping more
    // experts resident longer), or the S>=GPU_COLD_BATCH_THRESHOLD cold-
    // batch upload above -- returned completely UNPROTECTED. Nothing
    // stopped evict_lru (called from the prefetcher thread's own
    // background promotions, or even a LATER k in this SAME Pass A loop
    // needing a free slot) from picking this exact slot as an eviction
    // victim and overwriting it mid-flight while Pass B's kernels (issued
    // via expert_ffn_gpu_issue, potentially many other experts' worth of
    // time earlier in Pass A, and not confirmed done until
    // expert_ffn_gpu_finish's cudaStreamSynchronize) were still reading
    // it -- silent corruption, not a crash, exactly the "fluent but wrong"
    // signature this project has hit before. Held uniformly here now,
    // regardless of which branch above resolved `loc`, so every path that
    // returns a non-null location gets equal protection for the entire
    // issue-to-finish window. If the hold fails (slot identity changed in
    // the narrow window between the residency check and this call), treat
    // as a miss rather than trust a possibly-stale pointer -- same
    // reasoning the decode-time branch already used, just applied
    // everywhere now instead of only there.
    if (gpu_ctx && gpu_ctx->topo) {
        bool held = gpu_ctx->topo->hold_for_compute(gpu_ctx->layer_idx, gpu_ctx->expert_id, loc->device_ptr,
                                                     loc_from_active_layer);
        if (!held) {
            perf().n_hold_for_compute_fail.fetch_add(1, std::memory_order_relaxed);
            LOG_WARN("[MOE-VRAM]  layer=%d expert=%d: hold_for_compute failed for an apparently-"
                     "resident slot -- identity changed underneath us; falling back to CPU for this "
                     "call rather than trust a possibly-stale pointer\n",
                     gpu_ctx->layer_idx, gpu_ctx->expert_id);
            return nullptr;
        }
        *out_needs_release = true;
        *out_loc_from_active_layer = loc_from_active_layer;
    }
    return loc;
}

// Phase A: upload input + launch gate/up kernels, no sync. Returns
// immediately once the (small, S*D-element) H2D copy completes -- the
// kernel launches themselves are async and keep executing on the device
// after this call returns. Returns false if gpu_scratch.ensure's grow
// failed (real VRAM exhaustion) -- caller must NOT queue this expert for
// Pass B (expert_ffn_gpu_finish) in that case, and should fall back to
// the CPU path for it instead, same as any other GPU-dispatch failure.
static bool expert_ffn_gpu_issue(
    GpuExpertScratch& gpu_scratch, const ExpertLocation* loc,
    const float* input, const GGUFTensor& gate_meta, const GGUFTensor& up_meta,
    int S, int D, int moe_inter
) {
    // Slab layout mirrors the host copy: gate at offset 0, up at its
    // original file-offset delta from gate (see load_expert_slab).
    const uint8_t* base = loc->device_ptr;
    const uint8_t* gate_dev = base;
    const uint8_t* up_dev   = base + (up_meta.offset - gate_meta.offset);

    if (!gpu_scratch.ensure(S, D, moe_inter, loc->gpu_id)) return false;  // pins the correct vault GPU

    PERF_START(gexp_up1);
    cudaMemcpy(gpu_scratch.d_x, input, (size_t)S * D * sizeof(float), cudaMemcpyHostToDevice);
    PERF_END(gexp_up1, t_expert_gpu_upload);

    gpu_matmul_dispatch(gate_meta.type, gpu_scratch.d_gate, gpu_scratch.d_x, gate_dev, S, D, moe_inter);
    gpu_matmul_dispatch(up_meta.type,   gpu_scratch.d_up,   gpu_scratch.d_x, up_dev,   S, D, moe_inter);
    return true;
}

// DeepSeek-V4 per-layer SwiGLU clamp (ArchitectureConfig::swiglu_clamp_exp/
// shexp, gguf_loader.hpp): bounds the SwiGLU activation (post SiLU*up, pre
// down-proj) to [-clamp, clamp]. clamp<=0 is "inactive" (empty vector at
// this layer, or the file doesn't set one) -- a strict no-op, matching
// every other DeepSeek-V4 field's "absent = feature off" default. Shared
// by the routed-expert (GPU/CPU) and shared-expert SwiGLU sites below --
// deliberately factored out once rather than copy-pasted at each of the 3
// call sites.
static inline void swiglu_clamp_inplace(float* x, int n, float clamp) {
    if (clamp <= 0.0f) return;
    for (int i = 0; i < n; ++i) x[i] = std::max(-clamp, std::min(clamp, x[i]));
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
    int S, int D, int moe_inter,
    GpuTopology* topo, bool needs_release, bool loc_from_active_layer,
    float swiglu_clamp
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
    swiglu_clamp_inplace(gate_buf, S * moe_inter, swiglu_clamp);

    PERF_START(gexp_up2);
    cudaMemcpy(gpu_scratch.d_prod, gate_buf, (size_t)S * moe_inter * sizeof(float), cudaMemcpyHostToDevice);
    PERF_END(gexp_up2, t_expert_gpu_upload);

    PERF_START(gexp_cmp2);
    gpu_matmul_dispatch(down_meta.type, gpu_scratch.d_down, gpu_scratch.d_prod, down_dev, S, moe_inter, D);
    cudaStreamSynchronize(0);
    PERF_END(gexp_cmp2, t_gpu_compute);

    // This sync just confirmed EVERY kernel that reads this slab (gate, up,
    // AND down -- the last of the three) has actually finished executing,
    // not just that their launches completed -- so it's safe to release
    // the temporary hold taken in expert_ffn_resolve_gpu_loc's decode-time
    // cold-miss branch now, exactly here and not any earlier. Releasing
    // before this point would reopen the exact eviction-during-compute
    // race the hold exists to prevent.
    if (needs_release && topo) topo->release_hold(layer_idx, expert_id, loc->device_ptr, loc_from_active_layer);

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
    int S, int D, int moe_inter,
    float swiglu_clamp
) {
    PERF_START(cexp);
    cpu_quant_matmul(gate_meta.type, gate_data, input, gate_buf, S, D, moe_inter);
    cpu_quant_matmul(up_meta.type,   up_data,   input, up_buf,   S, D, moe_inter);
    silu_inplace(gate_buf, S * moe_inter);
    elem_mul(gate_buf, gate_buf, up_buf, S * moe_inter);
    swiglu_clamp_inplace(gate_buf, S * moe_inter, swiglu_clamp);
    cpu_quant_matmul(down_meta.type, down_data, gate_buf, output, S, moe_inter, D);
    PERF_END(cexp, t_cpu_expert);
    perf().n_expert_cpu_calls.fetch_add(1, std::memory_order_relaxed);
}

// Gating activation: turns one row of raw router logits into probs
// in-place, per ArchitectureConfig::gating_func (GatingFunc, gguf_loader.
// hpp). Shared by moe_forward_gguf's real router (below) AND the pilot
// router's own prediction ranking (layers_forward_gguf's "second,
// independent misprediction source" fix, this file) -- deliberately ONE
// function, not two independently-maintained copies of this switch, since
// the pilot's whole reason for existing is to rank experts the SAME way the
// real router will; a gating formula that drifts out of sync between the
// two sites silently degrades prefetch accuracy without ever crashing, the
// exact bug class both sites' own inline comments already warn about.
static void apply_gating_activation(GatingFunc func, float* row, int n) {
    switch (func) {
    case GatingFunc::Sigmoid:
        for (int e = 0; e < n; ++e) row[e] = 1.0f / (1.0f + std::exp(-row[e]));
        break;
    case GatingFunc::SqrtSoftplus:
        // sqrt(softplus(x)), softplus(x) = log(1+exp(x)) -- DeepSeek-V4's
        // hard-required router activation (see llama.cpp's
        // LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS). Naive log1p(exp(x))
        // overflows exp() for large positive x; softplus(x) = x +
        // log1p(exp(-x)) for x > 0 is the standard numerically-stable form
        // (exp(-x) -> 0 as x grows, so this never overflows, and reduces to
        // the direct log1p(exp(x)) for x <= 0 where THAT form is safe).
        for (int e = 0; e < n; ++e) {
            float x = row[e];
            float softplus = (x > 0.0f) ? (x + std::log1p(std::exp(-x)))
                                         : std::log1p(std::exp(x));
            row[e] = std::sqrt(softplus);
        }
        break;
    case GatingFunc::Softmax:
    default: {
        float max_l = row[0];
        for (int e = 1; e < n; ++e) max_l = std::max(max_l, row[e]);
        float sum = 0.0f;
        for (int e = 0; e < n; ++e) { row[e] = std::exp(row[e] - max_l); sum += row[e]; }
        float inv = 1.0f / sum;
        for (int e = 0; e < n; ++e) row[e] *= inv;
        break;
    }
    }
}

// ============================================================
// MoE forward with router, top-k selection, expert execution.
// Supports parallel expert I/O via threading.
// ============================================================
void moe_forward_gguf(
    StreamingEngine& engine,
    RAMSafetyManager& ram_mgr,
    LockFreePrefetcher& async_io_queue,
    int layer_idx,
    const float* hidden_states,
    int S,
    float* output,
    LayerScratch& scratch,
    const int* token_ids
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

    // DeepSeek-V4 per-layer SwiGLU clamp (swiglu_clamp_inplace's own
    // comment) -- resolved once here, passed to both the routed-expert
    // (CPU/GPU) and shared-expert SwiGLU sites below. 0.0f ("inactive") for
    // any layer index the file's clamp array doesn't cover, matching
    // ArchitectureConfig::swiglu_clamp_exp/shexp's own "empty = off"
    // default for every non-DeepSeek-V4 file.
    float clamp_exp = (layer_idx < (int)c.swiglu_clamp_exp.size())
                           ? c.swiglu_clamp_exp[layer_idx] : 0.0f;
    float clamp_shexp = (layer_idx < (int)c.swiglu_clamp_shexp.size())
                             ? c.swiglu_clamp_shexp[layer_idx] : 0.0f;

    // DeepSeek-V4 hash-routed layers (first c.hash_layer_count layers):
    // expert SELECTION is a direct token-id -> expert-id lookup via
    // ffn_gate_tid2eid.weight (one row of `topk` int32 expert ids per
    // vocab token, indexed by token id -- ggml's real ggml_get_rows(
    // tid2eid, token_id) semantics, no runtime hash computation despite
    // the tensor's name; verified against llama.cpp's real DeepSeek-V4
    // implementation, src/models/deepseek4.cpp + llama-graph.cpp's
    // build_moe_ffn). The router's own logits/probs (computed
    // unconditionally below, every layer -- see graph_initializer.cpp's
    // preload-manifest fix) still supply the WEIGHTS for whichever
    // experts this resolves to; only the selection step branches, in the
    // per-token loop further down.
    const int32_t* tid2eid = nullptr;
    if (layer_idx < c.hash_layer_count) {
        std::string tid2eid_key = "blk." + std::to_string(layer_idx) + ".ffn_gate_tid2eid.weight";
        auto tid2eid_it = engine.resident_layers.find(tid2eid_key);
        // Validate the on-disk type/shape BEFORE reinterpreting host_ptr as
        // int32_t* -- added after a real run showed EVERY hash-route lookup
        // failing its bounds check (100% of tokens, all hash layers), which
        // is far more consistent with reading the wrong bytes entirely
        // (e.g. the tensor's real ggml type isn't I32, so calculate_size()
        // undersized the read and left the tail as uninitialized heap
        // garbage -- see GGUFType::I64's own comment) than with a subtle
        // per-token routing quirk. Previously this reinterpret_cast was
        // unconditional on host_ptr alone, so a type/width mismatch would
        // silently produce garbage ids instead of a diagnosable error.
        const GGUFTensor* meta_ptr = (tid2eid_it != engine.resident_layers.end())
                                          ? &tid2eid_it->second.meta : nullptr;
        bool shape_ok = meta_ptr && !meta_ptr->shape.empty() &&
                        (int64_t)meta_ptr->shape[0] == (int64_t)topk;
        if (meta_ptr && tid2eid_it->second.host_ptr &&
            meta_ptr->type == GGUFType::I32 && shape_ok) {
            tid2eid = reinterpret_cast<const int32_t*>(tid2eid_it->second.host_ptr);
        } else if (meta_ptr && tid2eid_it->second.host_ptr) {
            // Tensor loaded, but its type or row width doesn't match what
            // this raw-int32 read path assumes -- loud, PER-LAYER (see
            // below), and named specifically so a real run pinpoints
            // exactly which assumption broke instead of just seeing every
            // token's row fail the downstream bounds check with no clue why.
            static std::vector<bool> tid2eid_shape_warned_per_layer;
            if ((int)tid2eid_shape_warned_per_layer.size() <= layer_idx)
                tid2eid_shape_warned_per_layer.resize(layer_idx + 1, false);
            if (!tid2eid_shape_warned_per_layer[layer_idx]) {
                LOG_ERROR("[MOE-ROUTE] layer=%d ffn_gate_tid2eid.weight loaded but type/shape "
                          "mismatch (type=%u expected I32=%u, shape[0]=%llu expected topk=%d) -- "
                          "falling back to the normal learned-router top-k for this layer.\n",
                          layer_idx, (unsigned)meta_ptr->type, (unsigned)GGUFType::I32,
                          meta_ptr->shape.empty() ? 0ull : (unsigned long long)meta_ptr->shape[0], topk);
                tid2eid_shape_warned_per_layer[layer_idx] = true;
            }
        } else {
            // Loud, PER-LAYER warning -- FIX: the old version of this
            // warning was a single function-static bool SHARED across
            // every hash-routed layer, so only the very first one to hit
            // this path ever logged anything; every other hash-routed
            // layer's identical failure was completely silent. Falls back
            // to the normal argsort-over-learned-router-probs path below
            // for this layer -- a REAL, working fallback now that
            // ffn_gate_inp.weight is unconditionally loaded (see the
            // preload-manifest fix), unlike the previous bug where the
            // router itself was also missing and this whole layer's MoE
            // output was silently zero-filled instead of falling back to
            // anything.
            static std::vector<bool> tid2eid_warned_per_layer;
            if ((int)tid2eid_warned_per_layer.size() <= layer_idx)
                tid2eid_warned_per_layer.resize(layer_idx + 1, false);
            if (!tid2eid_warned_per_layer[layer_idx]) {
                LOG_ERROR("[MOE-ROUTE] layer=%d is hash-routed (hash_layer_count=%d) but "
                          "ffn_gate_tid2eid.weight failed to load (missing/short-read) -- "
                          "falling back to the normal learned-router top-k for this layer, "
                          "which will select DIFFERENT experts than the real hash table would.\n",
                          layer_idx, c.hash_layer_count);
                tid2eid_warned_per_layer[layer_idx] = true;
            }
        }
    }

    // --- Router: compute expert probs and select top-k. Reference algorithm
    // ported from llama.cpp's build_moe_ffn (llama-graph.cpp):
    //   1) probs = gate_func(logits)     [sigmoid/softmax/sqrtsoftplus, per c.gating_func -- see apply_gating_activation above]
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
            // DMA): rmsnorm_row normalizes exactly ONE row (D floats).
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
            apply_gating_activation(c.gating_func, row, n_experts);
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
        float sum_w = 0.0f;

        // Records one (expert_id, weight) pick into scratch's per-token
        // arrays and the routing-frequency histograms -- shared by both
        // the hash-routed and normal argsort selection branches below, so
        // the two paths can never diverge in what bookkeeping happens once
        // an expert is actually picked (weight is always the UNBIASED
        // token_probs[id], same convention either way).
        auto record_pick = [&](int k, int id) {
            float w = token_probs[id];
            scratch.tok_topk_weights[s * topk + k] = w;
            scratch.tok_topk_ids[s * topk + k] = id;
            scratch.expert_freq[id]++;
            // Phase A accumulate the cross-conversation
            // routing-frequency histogram (engine.route_freq), dumped to a
            // usage file at shutdown and read back at the NEXT process
            // startup to preload the measured hot set into VRAM. This is
            // the true demand signal (what the router actually picked),
            // not a cache-hit proxy
            engine.bump_route_freq(layer_idx, id);
            // Additionally track prefill-only selections in their own
            // histogram (route_freq_prefill) -- see that field's own
            // comment for why decode's higher routing volume would
            // otherwise crowd out prefill's signal in the combined one.
            // Never a replacement for the line above, purely additive.
            if (S > 1) engine.bump_route_freq_prefill(layer_idx, id);
            sum_w += w;
        };

        // Hash-routed SELECTION: direct token-id -> expert-id lookup,
        // bypassing argsort entirely for this token. Every guard here is
        // defensive (this is a brand-new raw-int32 read path with no prior
        // real-hardware mileage): token_ids may legitimately be null (a
        // caller that doesn't have per-token ids), an out-of-range token
        // id would read past the tensor's real extent, and a corrupt/
        // garbage row would hand a downstream expert-array access an
        // out-of-bounds id -- any of these falls through to the normal
        // argsort path below for just this token, never crashes.
        bool used_hash_route = false;
        if (tid2eid && token_ids) {
            int tok = token_ids[s];
            if (tok >= 0 && tok < c.vocab) {
                const int32_t* row = tid2eid + (size_t)tok * topk;
                bool ids_valid = true;
                for (int k = 0; k < topk; ++k) {
                    if (row[k] < 0 || row[k] >= n_experts) { ids_valid = false; break; }
                }
                if (ids_valid) {
                    for (int k = 0; k < topk; ++k) record_pick(k, row[k]);
                    used_hash_route = true;
                } else {
                    LOG_ERROR("[MOE-ROUTE] layer=%d token_id=%d: ffn_gate_tid2eid row contains "
                              "an out-of-range expert id (n_experts=%d) -- falling back to the "
                              "learned router's top-k for this token only.\n",
                              layer_idx, tok, n_experts);
                }
            }
        }
        if (!used_hash_route) {
            for (int e = 0; e < n_experts; ++e) {
                float sel = token_probs[e] + (bias_row ? bias_row[e] : 0.0f);
                scratch.scores[e] = {e, sel};
            }
            std::partial_sort(scratch.scores.begin(), scratch.scores.begin() + topk, scratch.scores.begin() + n_experts,
                             [](const ExpertScore& a, const ExpertScore& b) { return a.score > b.score; });
            for (int k = 0; k < topk; ++k) record_pick(k, scratch.scores[k].id);
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

        // Pilot RECALL check (always on, independent of log level -- a
        // perf counter, not a log line): if the pilot predicted THIS exact
        // layer's routing one iteration ago (LayerScratch::pilot_pred_ids),
        // compare its predicted set against what the router really just
        // picked for the same "last token" position, then consume/reset
        // the pending prediction either way so a stale value never lingers
        // into a later layer's unrelated comparison.
        if (s == S - 1 && scratch.pilot_pred_layer == layer_idx) {
            int overlap = 0;
            int hint_overlap = 0, hint_denom = 0;
            for (int pk = 0; pk < scratch.pilot_pred_count; ++pk) {
                int pred_id = scratch.pilot_pred_ids[pk];
                bool matched = false;
                for (int k = 0; k < topk; ++k) {
                    if (scratch.tok_topk_ids[s * topk + k] == pred_id) { matched = true; break; }
                }
                if (matched) ++overlap;
                // Restricted tally: of the picks that were actually
                // fetch-worthy (not already cached, so a real hint got
                // pushed for them), how many turned out correct? This is
                // the decision-relevant number -- see
                // n_pilot_recall_hint_hits's own comment in perf_timer.hpp.
                if (scratch.pilot_pred_was_hint[pk]) {
                    ++hint_denom;
                    if (matched) ++hint_overlap;
                }
            }
            perf().n_pilot_recall_hits.fetch_add((uint64_t)overlap, std::memory_order_relaxed);
            perf().n_pilot_recall_checks.fetch_add(1, std::memory_order_relaxed);
            perf().n_pilot_recall_topk_sum.fetch_add((uint64_t)topk, std::memory_order_relaxed);
            perf().n_pilot_recall_hint_hits.fetch_add((uint64_t)hint_overlap, std::memory_order_relaxed);
            perf().n_pilot_recall_hint_checks_denom.fetch_add((uint64_t)hint_denom, std::memory_order_relaxed);
            scratch.pilot_pred_layer = -1;
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
    if (S > 1 && ram_mgr.vram_cache && !is_cpu_affinity_layer(layer_idx)) {
        PERF_START(prefill_warm);
        std::vector<int> candidates;
        candidates.reserve(n_experts);
        for (int e = 0; e < n_experts; ++e) if (scratch.expert_freq[e] > 0) candidates.push_back(e);
        std::sort(candidates.begin(), candidates.end(), [&](int a, int b) {
            return scratch.expert_freq[a] > scratch.expert_freq[b];
        });

        // prefill_warm_via_prefetcher=true needs a GPU-type-capability check
        // WITHOUT ever calling ram_mgr.get_expert() -- that synchronous
        // RAM/disk fetch is exactly the blocking cost this mode exists to
        // avoid. expert_layer_geometry is a pure tensor-table lookup (same
        // expert-0-representative convention Phase 3's own shadow-check
        // uses below), computed once per layer rather than per candidate.
        bool hint_mode_gpu_capable = true;
        if (engine.prefill_warm_via_prefetcher) {
            size_t p2_gb = 0, p2_ub = 0, p2_db = 0;
            const GGUFTensor *p2_gM = nullptr, *p2_uM = nullptr, *p2_dM = nullptr;
            hint_mode_gpu_capable = engine.expert_layer_geometry(layer_idx, p2_gb, p2_ub, p2_db,
                                                                   p2_gM, p2_uM, p2_dM)
                && gpu_type_supported(p2_gM->type) && gpu_type_supported(p2_uM->type)
                && gpu_type_supported(p2_dM->type);
        }

        for (int expert_id : candidates) {
            if (!ram_mgr.vram_cache->has_free_slot()) break; // cache warm now — stop preloading
            if (ram_mgr.vram_cache->find_expert(layer_idx, expert_id)) continue; // already resident

            if (engine.prefill_warm_via_prefetcher) {
                // Async path (config prefill_warm_via_prefetcher=true): push
                // a hint to the SAME queue/worker-thread pipeline decode's
                // own PILOT prediction already uses, instead of loading
                // synchronously right here. Trade-off documented on the
                // config field itself -- this legitimately might not finish
                // in time for THIS layer's own Phase 3 dispatch a moment
                // later, falling through to the ordinary (already-correct)
                // miss path for this call; never a correctness issue.
                if (!hint_mode_gpu_capable) continue;
                async_io_queue.push_hint(layer_idx, expert_id);
                perf().n_prefill_hint_pushed.fetch_add(1, std::memory_order_relaxed);
                continue;
            }

            // ---- Legacy synchronous path (default, unchanged behavior) ----
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
            // FIX (real, measured "[TIER-OVERLAP]" leak -- see CLAUDE.md's
            // "third, independent tier-overlap source" entry): this call
            // used to bypass get_expert's own MOVE-vs-COPY promotion
            // bookkeeping entirely, since it's a direct external
            // upload_expert_async call, not a hit routed through
            // get_expert's internal Step-1 branch. apply_promotion_
            // bookkeeping mirrors that same logic for this call site.
            perf().n_prefill_warm_attempted.fetch_add(1, std::memory_order_relaxed);
            ExpertLocation* promoted =
                ram_mgr.vram_cache->upload_expert_async(layer_idx, expert_id, blk->slab_buffer, blk->slab_capacity);
            if (promoted) perf().n_prefill_warm_succeeded.fetch_add(1, std::memory_order_relaxed);
            else perf().n_prefill_warm_failed.fetch_add(1, std::memory_order_relaxed);
            ram_mgr.apply_promotion_bookkeeping(*blk, layer_idx, expert_id, promoted);
        }
        PERF_END(prefill_warm, t_prefill_warm_time);
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

        // Weighted accumulation + VERBOSE trace, shared by both the CPU path
        // (dispatch_expert_k's own synchronous fallback) and the GPU path
        // (Pass B, accumulates after finish()) -- same logic either way, just
        // called at a different point in the pipeline depending on path.
        // Moved up (was previously defined between the loading block and the
        // old Pass A loop) so dispatch_expert_k below can capture it.
        auto accumulate = [&](int k, MoEExpertBlock* exp, const float* down_buf_ptr) {
            // Per-expert output norm, VERBOSE-gated: distinguishes "expert_ffn
            // itself computed all-zero" from "it computed something real but
            // the weighted accumulation below never applied it" -- check_finite
            // only catches NaN/Inf, never a suspiciously-exact zero, which is
            // exactly the symptom under investigation (moe_out|=0 every MoE
            // layer during prefill)
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

        // Issues expert k's GPU kernels (async, no sync) against whatever
        // scratch.expert_ptrs[k] already holds, falling back to synchronous
        // CPU compute if GPU dispatch isn't possible. Pulled out of the old
        // single "Pass A" loop (which used to run only AFTER get_experts_
        // batch) so it can also be called INLINE for shadowed (already
        // VRAM-resident) experts below, BEFORE get_experts_batch's blocking
        // disk read for this layer's genuine misses -- expert_ffn_gpu_issue
        // never syncs, so those kernels execute on the GPU stream WHILE this
        // thread blocks on io_uring for the rest of the batch, instead of
        // the two running strictly back-to-back as before (real compute/IO
        // overlap, not just async-within-GPU overlap).
        auto dispatch_expert_k = [&](int k) {
            MoEExpertBlock* exp = scratch.expert_ptrs[k];
            // A shadow block (see the loading loop below) legitimately has
            // null host_ptr by design -- it's metadata-only, GPU dispatch
            // reads weight bytes from ExpertLocation::device_ptr, never
            // from here. Only a REAL (non-shadow) block missing host_ptr
            // is the genuine "something went wrong" case this check exists
            // to catch.
            if (!exp || (!exp->vram_shadow && (!exp->gate_proj.host_ptr || !exp->down_proj.host_ptr))) {
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
                return;
            }

            ExpertGpuContext gpu_ctx;
            gpu_ctx.topo = ram_mgr.vram_cache;
            gpu_ctx.layer_idx = layer_idx;
            gpu_ctx.expert_id = exp->expert_id;
            gpu_ctx.slab_buffer = exp->slab_buffer;
            gpu_ctx.slab_capacity = exp->slab_capacity;
            // Null for a shadow block (exp->vram_shadow) -- correct: a
            // shadow's slab_buffer is already null too, so none of the
            // synchronous-upload branches below (which all require
            // gpu_ctx->slab_buffer) can ever fire for it anyway; also_in_vram
            // marking is only relevant for a REAL RAM-tier block being
            // promoted, never a metadata-only shadow.
            gpu_ctx.exp = exp;
            // Reuse the shadow pre-check's find_expert() result instead of
            // calling it a second time -- a real, measured cost (Cache
            // mutex wait roughly doubled, 4.185s vs 1.4-2.0s, from this
            // exact redundant lookup on a real run). Safe: hold_for_compute
            // (below, inside expert_ffn_resolve_gpu_loc) does its own fresh
            // identity-based re-scan regardless of which pointer is passed
            // in, so a stale hint here just makes the hold correctly fail,
            // exactly as a stale result from a second find_expert() call
            // would. nullptr (not shadowed, or shadow lookup found nothing)
            // falls back to the original single find_expert() call.
            gpu_ctx.loc = (exp->vram_shadow && scratch.vram_loc_hint[k])
                ? scratch.vram_loc_hint[k]
                : (ram_mgr.vram_cache ? ram_mgr.vram_cache->find_expert(layer_idx, exp->expert_id) : nullptr);

            bool needs_release = false;
            bool loc_from_active_layer = false;
            const ExpertLocation* resolved = expert_ffn_resolve_gpu_loc(
                &gpu_ctx, 1, exp->gate_proj.meta, exp->up_proj.meta, exp->down_proj.meta,
                &needs_release, &loc_from_active_layer, ram_mgr);

            bool gpu_issued = false;
            if (resolved) {
                GpuExpertScratch& gs = scratch.gpu_pool[scratch.pending_gpu.size()];
                gpu_issued = expert_ffn_gpu_issue(gs, resolved, token_hidden,
                                                   exp->gate_proj.meta, exp->up_proj.meta,
                                                   1, D, moe_inter);
                if (gpu_issued) {
                    scratch.pending_gpu.push_back({k, exp, resolved, needs_release, loc_from_active_layer,
                                                    gpu_ctx.expert_id});
                    // Protect this RAM slot from being reclaimed by ANY
                    // RAM-tier victim-selection (get_expert/get_experts_
                    // batch/demote_from_vram) until Pass B below has
                    // finished with this k -- see MoEExpertBlock::
                    // ram_temp_pin_count's own comment for the real
                    // release_hold-identity-mismatch bug this closes.
                    // Harmless no-op if exp is a shadow scratch block
                    // (never a RAM-tier victim-selection candidate). Goes
                    // through RAMSafetyManager::ram_pin (not a direct field
                    // increment) -- a LockFreePrefetcher worker thread's own
                    // claim_or_wait victim-selection scan now reads this
                    // exact field too, under ram_mutex; an unlocked write
                    // here would race that locked read. See ram_pin's own
                    // comment (ram_manager.hpp).
                    ram_mgr.ram_pin(*exp);
                } else {
                    // gpu_scratch.ensure's cudaMalloc genuinely failed (VRAM
                    // exhaustion) -- this expert never got Pass B queued, so
                    // release any hold taken above HERE rather than leaving
                    // it dangling forever (expert_ffn_gpu_finish, the only
                    // other place that releases it, will never run for this
                    // expert). Falls through to the CPU path below like any
                    // other GPU-dispatch failure.
                    //
                    // Uses gpu_ctx.expert_id (captured before
                    // expert_ffn_resolve_gpu_loc ran), NOT exp->expert_id --
                    // that call can promote this exact `exp` into the main
                    // VRAM cache and, on a RAM-tier reclaim, clear
                    // exp->expert_id to -1 before we ever get here. Same
                    // "capture, don't re-read late" fix as PendingGpuExpert's
                    // own expert_id field.
                    if (needs_release && gpu_ctx.topo) {
                        gpu_ctx.topo->release_hold(layer_idx, gpu_ctx.expert_id, resolved->device_ptr,
                                                    loc_from_active_layer);
                    }
                    LOG_ERROR("[EXPERT-GPU] layer=%d expert=%d: GPU issue failed (scratch buffer "
                              "grow), falling back to CPU for this call\n", layer_idx, gpu_ctx.expert_id);
                }
            }
            if (!gpu_issued && exp->vram_shadow) {
                // Rare: a shadow-fast-pathed expert (confirmed VRAM-resident
                // moments ago, in the loading loop below) failed to actually
                // dispatch on GPU after all -- either a genuine race
                // (evicted in the narrow window between that check and
                // here) or expert_ffn_gpu_issue's scratch buffer grow
                // failing. Either way we never loaded real RAM data for it,
                // since the VRAM fast path was expected to succeed -- load
                // it now, synchronously, on this rare/slow path, so the CPU
                // fallback below has real bytes instead of the shadow's
                // null host_ptr.
                MoEExpertBlock* real = ram_mgr.get_expert(engine, layer_idx, exp->expert_id);
                if (real && real->gate_proj.host_ptr && real->down_proj.host_ptr) {
                    perf().n_shadow_race_reload_success.fetch_add(1, std::memory_order_relaxed);
                    exp = real;
                    scratch.expert_ptrs[k] = real;
                } else {
                    perf().n_shadow_race_reload_hard_fail.fetch_add(1, std::memory_order_relaxed);
                    LOG_ERROR("[EXPERT-GPU] layer=%d expert=%d: VRAM fast-path failed AND emergency "
                              "RAM reload also failed -- skipping this expert's contribution "
                              "entirely rather than read a null/stale pointer\n",
                              layer_idx, exp->expert_id);
                    return;
                }
            }
            if (!gpu_issued) {
                expert_ffn_cpu(
                    token_hidden, scratch.down_buf.data(),
                    exp->gate_proj.host_ptr, exp->gate_proj.meta,
                    exp->up_proj.host_ptr, exp->up_proj.meta,
                    exp->down_proj.host_ptr, exp->down_proj.meta,
                    scratch.gate_buf.data(), scratch.up_buf.data(),
                    1, D, moe_inter, clamp_exp);
                accumulate(k, exp, scratch.down_buf.data());
            }
        };
        scratch.pending_gpu.clear();

        // --- Load all needed experts (LRU cache handles hits/misses; most
        // will already be warm here thanks to Phase 1/2 above). Batched:
        // get_experts_batch parallelizes the disk reads for whichever of
        // this layer's topk picks are genuine misses (not already staged
        // via prefetch), instead of the old one-pread-at-a-time sequential
        // loop 
        //
        // FIX (real, measured "[TIER-OVERLAP]" bug -- see CLAUDE.md): this
        // used to call get_experts_batch UNCONDITIONALLY for the entire
        // topk list, even for experts already GPU-resident -- wasting a
        // real disk read and a real RAM slot on data whose BYTES the GPU
        // path never reads (it reads weight bytes from
        // ExpertLocation::device_ptr, and the per-projection TYPE from the
        // packed base tensor, uniform across every expert in a layer --
        // neither needs a RAM copy). A real run showed 66% of everything
        // VRAM-resident was ALSO RAM-resident as a direct result, 12+ GB
        // double-counted. Fixed by checking VRAM residency first (a real
        // find_expert lookup, same one dispatch_expert_k above already does
        // for its own purposes) and building a metadata-only "shadow" block
        // (populate_vram_shadow, zero I/O) for anything already resident +
        // GPU-type-supported, instead of loading it into RAM at all. Only
        // the remainder -- not yet VRAM-resident, or an unsupported type --
        // still goes through the real get_experts_batch path below,
        // batched exactly as before.
        //
        // COMPUTE/IO OVERLAP: a shadowed expert's kernels are issued via
        // dispatch_expert_k IMMEDIATELY below, inline in this same k-loop --
        // NOT deferred until after get_experts_batch runs, as they used to
        // be -- so they run on the GPU stream while this thread goes on to
        // block on this layer's disk-bound misses instead of the two
        // running strictly sequentially.
        {
            size_t shadow_gb = 0, shadow_ub = 0, shadow_db = 0;
            const GGUFTensor *shadow_gM = nullptr, *shadow_uM = nullptr, *shadow_dM = nullptr;
            // FIX (real, high-impact bug found while chasing a persistent
            // [TIER-OVERLAP] baseline that didn't respond to any promotion-
            // reclaim fix): this used to call engine.packed_slab_geometry(...)
            // here, which only recognizes the PACKED tensor convention. For
            // an unpacked_expert_format=true model (this project's own
            // recommended format for the single-DMA-per-expert read path),
            // packed_slab_geometry returns false for EVERY layer -- meaning
            // layer_gpu_capable was false everywhere, the entire shadow-
            // fast-path below (the mechanism whose whole job is "skip the
            // RAM-tier lookup for anything already VRAM-resident") never
            // fired even once, and EVERY routed expert on EVERY token --
            // including all of pinned_vram_capacity's PERMANENTLY VRAM-
            // resident experts, which never go through any promotion event
            // and therefore have no other path to ever getting their RAM
            // copy reclaimed -- fell through to the unconditional RAM-tier
            // load below. This is almost certainly why [TIER-OVERLAP]
            // stayed flat (~520-720) across four rounds of active-layer-
            // promotion-reclaim fixes: those fixes only address experts
            // freshly PROMOTED into VRAM, never the much larger set that's
            // ALREADY VRAM-resident (pinned or otherwise) and gets
            // redundantly RAM-loaded anyway because this gate never lets
            // them skip that load. The fix already exists a few lines
            // below for a DIFFERENT feature (cuFile eligibility) that hit
            // the exact same packed-only limitation and was updated to use
            // expert_layer_geometry (the unified dispatcher supporting
            // both formats) -- that fix was never carried back to this
            // older variable. Now uses the same unified dispatcher here too.
            bool layer_gpu_capable = !cpu_only_mode() && !is_cpu_affinity_layer(layer_idx) && ram_mgr.vram_cache &&
                engine.expert_layer_geometry(layer_idx, shadow_gb, shadow_ub, shadow_db,
                                              shadow_gM, shadow_uM, shadow_dM) &&
                gpu_type_supported(shadow_gM->type) && gpu_type_supported(shadow_uM->type) &&
                gpu_type_supported(shadow_dM->type);

            // cuFile (GPUDirect Storage) direct disk->VRAM eligibility for
            // this layer -- kept as a SEPARATE check from layer_gpu_capable
            // above (not merged into one, even though both now use
            // expert_layer_geometry) since the two conditions are otherwise
            // independent (cuFile also needs cuda_cufile_active) and
            // merging them would make a future change to one silently
            // affect the other. Layer-level pre-filter only (cheap,
            // tensor-table lookups) -- the actual per-expert absolute file
            // offset is resolved again inside the k-loop below, where the
            // specific expert_id is known.
            bool cufile_layer_eligible = engine.cuda_cufile_active && !is_cpu_affinity_layer(layer_idx)
                && ram_mgr.vram_cache;
            if (cufile_layer_eligible) {
                size_t cf_gb = 0, cf_ub = 0, cf_db = 0;
                const GGUFTensor *cf_gM = nullptr, *cf_uM = nullptr, *cf_dM = nullptr;
                cufile_layer_eligible = engine.expert_layer_geometry(layer_idx, cf_gb, cf_ub, cf_db,
                                                                       cf_gM, cf_uM, cf_dM)
                    && gpu_type_supported(cf_gM->type) && gpu_type_supported(cf_uM->type)
                    && gpu_type_supported(cf_dM->type);
            }

            static thread_local std::vector<int> ram_needed_ids;
            static thread_local std::vector<int> ram_needed_k;
            static thread_local std::vector<MoEExpertBlock*> ram_needed_ptrs;
            ram_needed_ids.clear(); ram_needed_k.clear();
            ram_needed_ids.reserve(topk); ram_needed_k.reserve(topk);

            for (int k = 0; k < topk; ++k) {
                int expert_id = scratch.tok_topk_ids[s * topk + k];
                bool shadowed = false;
                scratch.vram_loc_hint[k] = nullptr;  // explicit every layer/token -- resize()'s
                                                      // one-time clear only covers the FIRST call
                if (layer_gpu_capable) {
                    const ExpertLocation* loc = ram_mgr.vram_cache->find_expert(layer_idx, expert_id);
                    if (loc && loc->device_ptr) {
                        // Measurement only (expert_fetch_tracker.hpp) -- this
                        // is a genuine VRAM-tier HIT via the shadow fast
                        // path, resolved WITHOUT ever calling consume_staged.
                        // Was this exact expert ALSO recently touched by the
                        // async prefetcher?
                        record_hit_if_recently_fetched(layer_idx, expert_id, /*is_vram_tier=*/true);
                        MoEExpertBlock& shadow = scratch.vram_shadow_blocks[k];
                        engine.populate_vram_shadow(shadow, layer_idx, expert_id);
                        scratch.expert_ptrs[k] = &shadow;
                        scratch.vram_loc_hint[k] = loc;
                        shadowed = true;
                    }
                }
                // cuFile direct disk->VRAM: only ever for a GENUINE double
                // miss (not shadowed above -- i.e. not already VRAM-
                // resident -- AND not already RAM-resident). Re-reading
                // from disk via cuFile when a perfectly good copy already
                // sits in host RAM would waste real NVMe bandwidth for a
                // slower net result than the existing RAM->VRAM promotion
                // path, which is exactly what this check exists to prevent
                // -- see RAMSafetyManager::is_ram_resident's own comment.
                if (!shadowed && cufile_layer_eligible) {
                    if (ram_mgr.is_ram_resident(layer_idx, expert_id)) {
                        perf().n_cufile_skipped_ram_resident.fetch_add(1, std::memory_order_relaxed);
                    } else {
                        // Resolve THIS expert's absolute file offsets --
                        // packed format needs expert_id*per_expert_size
                        // added to the shared _exps tensor's base offset;
                        // unpacked format's per-expert tensor offset IS
                        // already absolute (mirrors build_expert_read_
                        // requests' identical packed-vs-unpacked branching).
                        size_t eg = 0, eu = 0, ed = 0;
                        const GGUFTensor *egM = nullptr, *euM = nullptr, *edM = nullptr;
                        bool have_geom = engine.unpacked_expert_format
                            ? engine.unpacked_expert_geometry(layer_idx, expert_id, eg, eu, ed, egM, euM, edM)
                            : engine.packed_slab_geometry(layer_idx, eg, eu, ed, egM, euM, edM);
                        if (have_geom) {
                            uint64_t gate_off = egM->offset
                                + (engine.unpacked_expert_format ? 0 : (uint64_t)expert_id * eg);
                            uint64_t up_off = euM->offset
                                + (engine.unpacked_expert_format ? 0 : (uint64_t)expert_id * eu);
                            uint64_t down_off = edM->offset
                                + (engine.unpacked_expert_format ? 0 : (uint64_t)expert_id * ed);
                            const cufile_io::FileHandle* h = engine.model_file.cufile_handle_for_tensor(*egM);
                            if (h && h->valid) {
                                const ExpertLocation* cf_loc = ram_mgr.vram_cache->upload_expert_cufile(
                                    layer_idx, expert_id, *h, gate_off, eg, up_off, eu, down_off, ed);
                                if (cf_loc) {
                                    MoEExpertBlock& shadow = scratch.vram_shadow_blocks[k];
                                    engine.populate_vram_shadow(shadow, layer_idx, expert_id);
                                    scratch.expert_ptrs[k] = &shadow;
                                    scratch.vram_loc_hint[k] = cf_loc;
                                    shadowed = true;
                                }
                            }
                        }
                    }
                }
                if (!shadowed) {
                    ram_needed_ids.push_back(expert_id);
                    ram_needed_k.push_back(k);
                } else {
                    // Already VRAM-resident (or cuFile-loaded moments ago
                    // above) -- issue its GPU kernels now instead of waiting
                    // for the blocking get_experts_batch call below, so they
                    // run on the GPU stream concurrently with that call's
                    // disk I/O for this layer's genuine misses.
                    dispatch_expert_k(k);
                }
            }
            if (!ram_needed_ids.empty()) {
                ram_needed_ptrs.resize(ram_needed_ids.size());
                ram_mgr.get_experts_batch(engine, layer_idx, ram_needed_ids.data(),
                                           (int)ram_needed_ids.size(), ram_needed_ptrs.data());
                for (size_t i = 0; i < ram_needed_ids.size(); ++i)
                    scratch.expert_ptrs[ram_needed_k[i]] = ram_needed_ptrs[i];
                // Dispatch whatever get_experts_batch just resolved. The
                // shadowed experts above already had their kernels issued
                // inline, before this blocking call ran -- only the genuine
                // misses are dispatched here, after their data is ready.
                for (size_t i = 0; i < ram_needed_k.size(); ++i)
                    dispatch_expert_k(ram_needed_k[i]);
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
                layer_idx, p.expert_id, 1, D, moe_inter,
                ram_mgr.vram_cache, p.needs_release, p.loc_from_active_layer, clamp_exp);
            // Matches Pass A's increment -- release the RAM-tier protection
            // now that expert_ffn_gpu_finish's own release_hold() call
            // (which uses p.expert_id, captured at Pass A issue time -- see
            // PendingGpuExpert's own comment for why NOT p.exp->expert_id)
            // has already happened. Must be the LAST thing done with
            // p.exp's identity in this iteration; this slot becomes a valid
            // victim-selection candidate again immediately after this
            // decrement. Goes through RAMSafetyManager::ram_unpin -- see
            // Pass A's matching ram_pin call for why this is no longer a
            // direct field decrement.
            ram_mgr.ram_unpin(*p.exp);
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
            swiglu_clamp_inplace(scratch.gate_buf.data(), shared_inter, clamp_shexp);
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

    // Three-tier caching: Cleanup active layer area after processing this layer
    // This reclaims temporary workspace used for current-layer experts
    if (ram_mgr.vram_cache && ram_mgr.vram_cache->active_layer_pool) {
        // FIX ("[TIER-OVERLAP]" leak, see CLAUDE.md): the rare end-of-layer
        // promotion path (a slot that crossed active_layer_promote_threshold
        // but couldn't promote inline via find_active_expert, e.g. because
        // it was still held mid-compute at that moment) has the same
        // durable-main-cache-now-exists fact to report as the inline hit
        // path does -- but unlike that call site, there's no single
        // MoEExpertBlock* in scope here (this is a sweep over up to
        // active_layer_capacity slots, not one dispatch's gpu_ctx), so this
        // goes through RAMSafetyManager::finish_sync_promotion's lookup
        // overload (a real RAM-tier lookup by (layer, expert_id)) rather
        // than gpu_ctx->exp directly -- see that function's own comment in
        // ram_manager.hpp for why this reclaims the slot immediately rather
        // than just marking it (a real run showed marking-only barely moves
        // [TIER-OVERLAP] within a short test).
        ram_mgr.vram_cache->cleanup_active_layer_area(layer_idx,
            [&](int l, int eid) { ram_mgr.finish_sync_promotion(engine, l, eid); });
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

// Pre-allocated once, reused every layer/call (avoids per-layer heap
// churn) -- promoted from a function-local static to file scope so
// shrink_moe_gpu_scratch() below can reach the SAME instance
// layers_forward_gguf uses, without changing either function's own
// behavior (still exactly one instance per thread, still only ever
// touched from the single main decode thread).
static thread_local LayerScratch g_layer_scratch;

// Frees every GpuExpertScratch slot's device buffers back to empty.
// Exposed (declared extern in generation_loop.hpp) so generate_text can
// call this right after a request's single prefill call completes, before
// settling into the decode loop. In practice this reclaims very little --
// expert FFN dispatch is always called with S=1 even during prefill (see
// the per-token loop in moe_forward_gguf), so these buffers never grow
// much regardless of prompt length -- this exists mainly for consistency
// with CudaManager::shrink_dense_scratch (the one that actually matters,
// see CLAUDE.md) and as defense-in-depth. Safe to call between requests:
// the next expert dispatch regrows via GpuExpertScratch::ensure's existing
// grow-only logic.
void shrink_moe_gpu_scratch() {
    for (auto& gs : g_layer_scratch.gpu_pool) gs.shrink();
}

// ============================================================
// Main layer-by-layer forward pass with pre-allocated scratch,
// speculative pilot prefetch, and performance instrumentation.
// ============================================================
void layers_forward_gguf(StreamingEngine& engine, RAMSafetyManager& ram_mgr,
                         LockFreePrefetcher& async_io_queue, float* hidden_states, int S, int pos_base,
                         const int* token_ids) {
    auto& c = engine.model_file.architecture_config;
    int D = c.hidden;

    // S > 1 = prefill (whole prompt in one pass), S == 1 = decode (one
    // token at a time) -- see RAMSafetyManager::in_prefill's own comment
    // for why the prefetch miss path needs this split.
    ram_mgr.in_prefill = (S > 1);

    // Measurement only (expert_fetch_tracker.hpp) -- one call to this
    // function IS one token's worth of processing (prefill's single S>1
    // call, or one decode step's S=1 call), so this is the exact right
    // place to advance the token-step counter record_hit_if_recently_
    // fetched uses to distinguish a same-token race from a much later,
    // unrelated token rediscovering an earlier token's stale fetch record.
    advance_token_step();

    LayerScratch& scratch = g_layer_scratch;
    scratch.resize(S, D, c.n_experts, c.moe_inter, c.topk, c.n_expert_shared, c.dense_inter);

    // Diagnostic: L2 norm of the LAST TOKEN's hidden state
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
        // --- Attention: dispatched through engine.family_ops.attention_fn
        // (models/family_ops.hpp) -- MLA absorbed (GLM-5.2/DeepSeek-V3), MLA
        // raw/SWA (DeepSeek-V4), or dense GQA causal (Hunyuan/Hy-V3/Llama),
        // depending on which family this file resolved to at load time. The
        // hc_pre_fn/hc_post_fn bracket is the hyper-connection pre-collapse/
        // post-combine hook (DeepSeek-V4); for every other family it's the
        // shared no-op / plain residual-add (hc_pre_noop/hc_post_plain_add,
        // family_ops.cpp) -- bit-for-bit the same two operations this loop
        // used to inline directly, just reachable through one function
        // pointer instead of a bool check with no third option. ---
        auto& fo = engine.family_ops;
        fo.hc_pre_fn(engine, l, /*sub_block=*/0, hidden_states, S, D);
        fo.attention_fn(engine, l, hidden_states, S, pos_base, scratch.attn_out.data());
        fo.hc_post_fn(engine, l, /*sub_block=*/0, hidden_states, scratch.attn_out.data(), S, D);

        // Per-layer, per-stage checkpoint: pinpoints the FIRST layer/stage
        // where hidden_states goes non-finite, instead of only learning
        // "somewhere in this forward pass" from the aggregate
        // after-prefill/after-decode-step checks in generation_loop.hpp.
        // check_finite() is a no-op (silent) unless it actually finds
        // NaN/Inf, so this is cheap on the (expected) all-finite path.
        check_finite("hidden_states (post attention residual)", l, -1, hidden_states, (size_t)n);
        if (trace) LOG_STREAM_VERBOSE() << "[TRACE l=" << l << "] post-attn |attn_out|=" << l2norm(scratch.attn_out.data() + (size_t)(S - 1) * D, D)
                                      << " |hidden|=" << l2norm(last_tok, D) << "\n";

        // --- FFN: leading dense layers (l < first_dense) use a plain dense
        // FFN — no router, no expert cache. Applies to both GLM-5.2's leading
        // dense layers and, since Hunyuan/Hy-V3 share the same first_dense
        // convention, to any dense layers there too. dense_ffn_gguf/
        // moe_forward_gguf are unchanged by the FamilyOps refactor -- they
        // still read/write plain [S,D] hidden_states/scratch.moe_out
        // regardless of family; only the hc_pre_fn/hc_post_fn bracket around
        // them (sub_block=1, matching attention's sub_block=0 above) knows
        // about hyper-connections. ---
        fo.hc_pre_fn(engine, l, /*sub_block=*/1, hidden_states, S, D);
        {
            PERF_SCOPE(t_moe_expert);
            if (l < c.first_dense) {
                dense_ffn_gguf(engine, l, hidden_states, S, scratch.moe_out.data(), scratch);
            } else {
                moe_forward_gguf(engine, ram_mgr, async_io_queue, l, hidden_states, S, scratch.moe_out.data(), scratch, token_ids);
            }
        }
        fo.hc_post_fn(engine, l, /*sub_block=*/1, hidden_states, scratch.moe_out.data(), S, D);

        check_finite("hidden_states (post FFN/MoE residual)", l, -1, hidden_states, (size_t)n);
        if (trace) LOG_STREAM_VERBOSE() << "[TRACE l=" << l << "] post-ffn  |moe_out|=" << l2norm(scratch.moe_out.data() + (size_t)(S - 1) * D, D)
                                      << " |hidden|=" << l2norm(last_tok, D) << "\n";

        // --- Speculative pilot prefetch for next layer's experts ---
        // Deliberately placed HERE -- AFTER this layer's own FFN/MoE
        // residual has been added -- not right after the attention
        // residual. hidden_states at this point is EXACTLY what layer
        // l+1's real router will see when the loop reaches it next
        // iteration. Predicting from the pre-FFN hidden_states (the old
        // position, right after the attention residual above) meant the
        // pilot's router pass ran on a vector missing this layer's own
        // MoE FFN contribution entirely -- routinely the single largest
        // per-layer compute cost in this model -- so its guessed top-8
        // experts had no reason to match what the real router at l+1
        // would actually select. Real-run prefetch instrumentation
        // (n_prefetch_hits_prefill/decode, n_prefetch_miss_still_queued) showed
        // 0% served in BOTH prefill and decode with 0/N misses even
        // finding a leftover trace of a hint -- exactly what stale-input
        // misprediction looks like (a timing/pool-size problem would
        // still show occasional lucky matches, not a clean zero).
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

                // Rank by EXACTLY the same "selection_probs" the real router
                // uses (moe_forward_gguf above: gate_func(logits) + bias),
                // not raw dense_proj output. Missing this was a second,
                // independent misprediction source on top of the residual-
                // timing fix above -- confirmed by comparing against
                // colibri's own pilot (glm.c), which predicts from the SAME
                // "post-attention, pre-FFN" state we used to use (a strictly
                // WORSE input than what we now use) but still measures 71.6%
                // recall, because it correctly applies sigmoid+bias before
                // ranking. GLM-5.2/DeepSeek-V3's exp_probs_b.bias exists
                // specifically to shift WHICH experts get selected for
                // load-balancing (auxiliary-loss-free routing) -- ranking on
                // the un-biased score means the pilot's top-k has no reason
                // to match the real, bias-corrected top-k the router
                // actually dispatches, independent of how accurate the input
                // hidden_states is.
                apply_gating_activation(c.gating_func, scratch.pilot_logits.data(), c.n_experts);
                {
                    std::string bias_key = "blk." + std::to_string(l + 1) + ".exp_probs_b.bias";
                    auto bias_it = engine.resident_layers.find(bias_key);
                    if (bias_it != engine.resident_layers.end() && bias_it->second.host_ptr) {
                        const float* bias_row = reinterpret_cast<float*>(bias_it->second.host_ptr);
                        for (int e = 0; e < c.n_experts; ++e) scratch.pilot_logits[e] += bias_row[e];
                    }
                }

                int pilot_k = std::min(pilot_k_for_layer(l + 1), c.topk);
                // Recall instrumentation: capture the full predicted set
                // BEFORE the already-resident skip below, so this measures
                // pure prediction accuracy against the real router's later
                // selection -- see LayerScratch::pilot_pred_ids's own
                // comment. moe_forward_gguf compares and resets this the
                // next time layer l+1 actually runs.
                scratch.pilot_pred_count = pilot_k;
                scratch.pilot_pred_layer = l + 1;
                // Collected below, published once after the k-loop --
                // feeds was_pilot_confirmed_resident's eviction-race
                // measurement (dense_dispatch.hpp). Only RAM-tier and
                // VRAM-UNPINNED-LRU picks are worth tracking here: a
                // pinned VRAM slot can never be chosen as an evict_lru
                // victim in the first place (see that function's pinned
                // check), so including it would just waste the fixed-size
                // buffer on something that could never match.
                PilotConfirmedEntry confirmed_resident_this_transition[PILOT_CONFIRMED_CAP];
                int confirmed_resident_count = 0;
                for (int k = 0; k < pilot_k; ++k) {
                    int best_exp = 0;
                    for (int e = 1; e < c.n_experts; ++e) {
                        if (scratch.pilot_logits[e] > scratch.pilot_logits[best_exp]) best_exp = e;
                    }
                    scratch.pilot_logits[best_exp] = -2e30f;
                    scratch.pilot_pred_ids[k] = best_exp;

                    // Checked so a hint is never pushed for an expert that's
                    // ALREADY resident -- a guaranteed instant hit needing no
                    // read at all once actually reached, so speculatively
                    // prefetching it anyway only steals scarce worker-thread
                    // I/O bandwidth from hints that genuinely need a disk
                    // read. Routed through RAMSafetyManager::is_ram_resident
                    // (locked internally) rather than a raw scan of
                    // engine.layers_experts[l+1] -- a LockFreePrefetcher
                    // worker thread can now concurrently claim/evict slots
                    // in that exact array via claim_or_wait, so an unlocked
                    // read of expert_id here would be a real data race, not
                    // just an occasionally-stale peek.
                    bool already_ram = ram_mgr.is_ram_resident(l + 1, best_exp);
                    // FIX: this used to only check the RAM tier. Before the
                    // [TIER-OVERLAP] fix, a VRAM-resident expert was almost
                    // always ALSO RAM-resident (moe_forward_gguf loaded
                    // everything into RAM unconditionally), so this check
                    // incidentally caught most VRAM-resident cases too. Now
                    // that VRAM-resident-but-NOT-RAM-resident is the common
                    // case (the fast path skips the RAM load entirely), this
                    // check was missing it -- a real run showed prefetch-served
                    // misses collapse from ~24-25% to 0.5% as a direct
                    // result: hints kept getting pushed for experts that,
                    // once reached, never touch the staging pool at all
                    // (the shadow fast path bypasses get_expert/
                    // consume_staged entirely), so the staged bytes just sat
                    // unused. Direct read of expert_vram_cache's public
                    // fields (NOT vram_cache->find_expert, which mutates
                    // hit_count/last_used_clock as a side effect -- this is
                    // a speculative "is it worth hinting" check on the NEXT
                    // layer, not a genuine use yet, same non-mutating style
                    // the RAM-tier check above already uses).
                    bool already_vram = false, vram_pinned = false;
                    if (!already_ram && ram_mgr.vram_cache) {
                        for (const auto& loc : ram_mgr.vram_cache->expert_vram_cache) {
                            if (loc.layer_idx == l + 1 && loc.expert_id == best_exp) {
                                already_vram = true;
                                vram_pinned = loc.pinned;
                                break;
                            }
                        }
                    }
                    // Classify into exactly one bucket for the "Pilot
                    // prediction coverage" diagnostic (perf_timer.hpp) --
                    // pure bookkeeping alongside the pre-existing
                    // skip/push decision below, doesn't change which
                    // branch fires.
                    bool is_hint_worthy = !already_ram && !already_vram;
                    if (already_ram) {
                        perf().n_pilot_pred_already_ram.fetch_add(1, std::memory_order_relaxed);
                        if (confirmed_resident_count < PILOT_CONFIRMED_CAP)
                            confirmed_resident_this_transition[confirmed_resident_count++] = {l + 1, best_exp};
                    } else if (already_vram && vram_pinned) {
                        perf().n_pilot_pred_already_vram_pinned.fetch_add(1, std::memory_order_relaxed);
                        // Deliberately NOT added to confirmed_resident_this_transition
                        // -- a pinned slot can never be an evict_lru victim, see
                        // that buffer's own comment above.
                    } else if (already_vram) {
                        perf().n_pilot_pred_already_vram_lru.fetch_add(1, std::memory_order_relaxed);
                        if (confirmed_resident_count < PILOT_CONFIRMED_CAP)
                            confirmed_resident_this_transition[confirmed_resident_count++] = {l + 1, best_exp};
                    } else {
                        perf().n_pilot_pred_needs_fetch.fetch_add(1, std::memory_order_relaxed);
                    }
                    scratch.pilot_pred_was_hint[k] = is_hint_worthy;
                    if (is_hint_worthy) {
                        async_io_queue.push_hint(l + 1, best_exp);
                    } else {
                        perf().n_pilot_hints_skipped.fetch_add(1, std::memory_order_relaxed);
                    }
                }
                // Publish AFTER the full k-loop (not per-pick) so a reader
                // (evict_lru/RAM-tier eviction, running on the decode
                // thread same as this loop but conceptually racy in
                // principle) always sees a complete, internally-consistent
                // set for this transition rather than a partially-filled
                // one -- see was_pilot_confirmed_resident's own comment on
                // why this is measurement-only, not a hard correctness
                // dependency either way.
                publish_pilot_confirmed_resident(confirmed_resident_this_transition, confirmed_resident_count);
            }
            PERF_END(pilot, t_prefetch);
        }
    }
}