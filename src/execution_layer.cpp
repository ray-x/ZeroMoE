#include "active_layer_upload_workers.hpp"
#include "async_prefetcher.hpp"
#include "dense_dispatch.hpp"
#include "engine.hpp"
#include "ggml_deepseek_v4_graph.hpp"
#include "ggml_gdn_proj_backend.hpp"
#include "ggml_glm5next_graph.hpp"
#include "ggml_hc_backend.hpp"
#include "ggml_moe_backend.hpp"
#include "ggml_qwen4exp_graph.hpp"
#include "kernels_cpu.hpp"
#include "kquant_dequant.hpp"
#include "logger.hpp"
#include "mtp_speculative.hpp"
#include "perf_timer.hpp"
#include "prefill_expert_stream.hpp"
#include "ram_manager.hpp"
#include "tensor_trace.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fstream>
#include <immintrin.h>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

extern void mla_attention_absorbed_gguf(StreamingEngine &engine, int layer_idx,
                                        const float *Q_roped, int S,
                                        int pos_base, float *out);
extern void dense_attention_gguf(StreamingEngine &engine, int layer_idx,
                                 const float *hidden_states, int S,
                                 int pos_base, float *out);
extern bool qwen4exp_hc_device_live();
extern bool qwen4exp_hc_post_combine_device(StreamingEngine &engine,
                                            int layer_idx, int sub_block, int S,
                                            int D,
                                            const void *device_sub_block_out);
extern bool qwen4exp_hc_post_combine_device_raw(StreamingEngine &engine,
                                                int layer_idx, int sub_block,
                                                int S, int D,
                                                const void *device_data);
extern bool qwen4exp_hc_prepare_host_layer_input(StreamingEngine &engine,
                                                 int layer_idx, int S);
extern bool qwen4exp_hc_download_streams(StreamingEngine &engine, int S);
extern bool qwen4exp_hc_commit_host_layer_input(StreamingEngine &engine,
                                                int layer_idx, int S);
extern bool qwen4exp_copy_gdn_device_output_to_host(const void *device_output,
                                                    float *host, int S);

// CUDA kernel launchers
extern "C" void launch_gguf_q4_0_dp4a(float *y, const float *x,
                                      const void *weights, int S, int I, int O);
extern "C" void launch_iq2m_matmul(float *y, const float *x,
                                   const void *weights, int S, int I, int O);
extern "C" void launch_q2_K_matmul(float *y, const float *x,
                                   const void *weights, int S, int I, int O);
extern "C" void launch_q3_K_matmul(float *y, const float *x,
                                   const void *weights, int S, int I, int O);
extern "C" void launch_q4_K_matmul(float *y, const float *x,
                                   const void *weights, int S, int I, int O);
extern "C" void launch_q5_K_matmul(float *y, const float *x,
                                   const void *weights, int S, int I, int O);
extern "C" void launch_q6_K_matmul(float *y, const float *x,
                                   const void *weights, int S, int I, int O);
extern "C" void launch_iq2_xxs_matmul(float *y, const float *x,
                                      const void *weights, int S, int I, int O);
extern "C" void launch_iq2_xs_matmul(float *y, const float *x,
                                     const void *weights, int S, int I, int O);
extern "C" void launch_iq2_s_matmul(float *y, const float *x,
                                    const void *weights, int S, int I, int O);
extern "C" void launch_iq3_xxs_matmul(float *y, const float *x,
                                      const void *weights, int S, int I, int O);
// IQ3_S: kquant_dequant.hpp's gpu_type_supported() has advertised this type
// as GPU-capable (cuda/iq3_matmul.cu's real, Test-5-verified launch_iq3s_
// matmul) since it was added for real Unsloth "UD-IQ3_XXS"-labeled GLM-5.3
// files (gate/up=IQ2_S, down=IQ3_S on the large majority of MoE layers) --
// but this extern declaration, and the matching case below, were never
// actually added. gpu_matmul_dispatch's switch fell through to its
// `default: return false`, and every one of its 3 call sites (gate/up/down
// dispatch) discards that bool return value -- so the down-proj kernel
// launch that should have populated gpu_scratch.d_down simply never
// happened, silently leaving it at whatever stale/uninitialized bytes were
// already there. Found 2026-09 chasing real, fluent-but-completely-
// incoherent generation output on a real UD-IQ3_XXS run (0 logged errors,
// since nothing ever threw -- gpu_type_supported() and gpu_matmul_dispatch()
// simply disagreed with each other) -- confirmed by reading both switches
// side by side, not by trial and error.
extern "C" void launch_iq3s_matmul(float *y, const float *x,
                                   const void *weights, int S, int I, int O);
extern "C" void launch_iq4_xs_matmul(float *y, const float *x,
                                     const void *weights, int S, int I, int O);
extern "C" void launch_tq1_matmul(float *y, const float *x, const void *weights,
                                  const void *scales, int S, int I, int O);
extern "C" void launch_iq1s_matmul(float *y, const float *x,
                                   const void *weights, int S, int I, int O);
extern "C" void launch_expert_swiglu(float *prod, const float *gate,
                                     const float *up, int n, float clamp,
                                     int fused, int *d_bad,
                                     cudaStream_t stream);
extern "C" void launch_iq4_nl_matmul(float *y, const float *x,
                                     const void *weights, int S, int I, int O);
extern "C" void launch_iq1m_matmul(float *y, const float *x,
                                   const void *weights, int S, int I, int O);
// Q8_0: same launcher the DENSE path already uses (declared independently in
// cuda_manager.hpp); re-declared here because this translation unit does not
// include that header. Newly reachable from routed experts as of 2026-09-04
// -- see gpu_matmul_dispatch's own Q8_0 case.
extern "C" void launch_q8_0_matmul(float *y, const float *x,
                                   const void *weights, int S, int I, int O);
// Q5_1: cuda/q5_1_matmul.cu. New kernel (2026-09-04), not a rewiring like
// Q8_0 above -- there was no Q5_1 kernel anywhere before it.
extern "C" void launch_q5_1_matmul(float *y, const float *x,
                                   const void *weights, int S, int I, int O);
// Dequantise-to-BF16 + cuBLAS path for routed experts (cuda/dequant_bf16.cu).
extern "C" bool launch_dequant_to_bf16(int ggml_type, void *dst_bf16,
                                       const void *src, int64_t n_elements);
extern "C" void launch_f32_to_bf16(void *dst_bf16, const float *src_f32,
                                   int64_t n);
// MXFP4: dormant kernel, wired in but not yet dispatched to -- see
// kquant_dequant.hpp's gpu_type_supported() comment.
extern "C" void launch_mxfp4_matmul(float *y, const float *x,
                                    const void *weights, int S, int I, int O);

// ============================================================
// Scratch buffers: pre-allocated once, reused every layer to
// eliminate per-layer heap allocation (was ~5% overhead measured)
// ============================================================
// GPU-resident scratch buffers for expert FFN dispatch (lazily grown,
// reused across calls to avoid per-token cudaMalloc/cudaFree churn).
struct GpuExpertScratch {
  // 8-byte device counter pair (NaN, Inf) for the GPU activation path's own
  // finiteness check -- replaces downloading 2*S*moe_inter floats just so the
  // host can scan them. Grow-once, freed with the rest of the scratch.
  int *d_bad = nullptr;
  int *ensure_bad_counter() {
    if (!d_bad) {
      if (cudaMalloc((void **)&d_bad, 2 * sizeof(int)) != cudaSuccess) {
        d_bad = nullptr;
      } else {
        // MUST zero on allocation. The kernel atomicAdds into this and the
        // readback is now SAMPLED rather than per-call, so there is no longer
        // a per-call memset to hide an uninitialised start -- reading raw VRAM
        // as ints produced counts like -1085005065 (old float bit patterns),
        // i.e. three spurious "non-finite" reports on a run whose output was
        // byte-identical to the CPU path.
        cudaMemset(d_bad, 0, 2 * sizeof(int));
      }
    }
    return d_bad;
  }

  float *d_x = nullptr;    // [S*D] input activation
  float *d_gate = nullptr; // [S*moe_inter]
  float *d_up = nullptr;   // [S*moe_inter]
  float *d_prod =
      nullptr; // [S*moe_inter] SiLU(gate) * up, re-uploaded for down_proj
  float *d_down = nullptr; // [S*D] final output
  size_t cap_sd = 0, cap_si = 0;
  int device_id = -1; // which GPU these buffers currently live on

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
        if (d_x) {
          cudaFree(d_x);
          d_x = nullptr;
        }
        if (d_down) {
          cudaFree(d_down);
          d_down = nullptr;
        }
        if (d_gate) {
          cudaFree(d_gate);
          d_gate = nullptr;
        }
        if (d_up) {
          cudaFree(d_up);
          d_up = nullptr;
        }
        if (d_prod) {
          cudaFree(d_prod);
          d_prod = nullptr;
        }
      }
      cap_sd = 0;
      cap_si = 0;
      device_id = target_device;
    }
    cudaSetDevice(device_id);

    size_t need_sd = (size_t)S * D;
    size_t need_si = (size_t)S * moe_inter;
    if (cap_sd < need_sd) {
      if (d_x) {
        cudaFree(d_x);
        d_x = nullptr;
      }
      if (d_down) {
        cudaFree(d_down);
        d_down = nullptr;
      }
      cap_sd = 0;
      cudaError_t err1 = cudaMalloc((void **)&d_x, need_sd * sizeof(float));
      cudaError_t err2 =
          (err1 == cudaSuccess)
              ? cudaMalloc((void **)&d_down, need_sd * sizeof(float))
              : err1;
      if (err1 != cudaSuccess || err2 != cudaSuccess) {
        LOG_ERROR(
            "[EXPERT-GPU] GpuExpertScratch::ensure: d_x/d_down grow to %.2f MB "
            "failed on device %d: %s\n",
            (need_sd * sizeof(float)) / 1e6, device_id,
            cudaGetErrorString(err1 != cudaSuccess ? err1 : err2));
        if (d_x) {
          cudaFree(d_x);
          d_x = nullptr;
        }
        if (d_down) {
          cudaFree(d_down);
          d_down = nullptr;
        }
        return false;
      }
      cap_sd = need_sd;
    }
    if (cap_si < need_si) {
      if (d_gate) {
        cudaFree(d_gate);
        d_gate = nullptr;
      }
      if (d_up) {
        cudaFree(d_up);
        d_up = nullptr;
      }
      if (d_prod) {
        cudaFree(d_prod);
        d_prod = nullptr;
      }
      cap_si = 0;
      cudaError_t err1 = cudaMalloc((void **)&d_gate, need_si * sizeof(float));
      cudaError_t err2 =
          (err1 == cudaSuccess)
              ? cudaMalloc((void **)&d_up, need_si * sizeof(float))
              : err1;
      cudaError_t err3 =
          (err2 == cudaSuccess)
              ? cudaMalloc((void **)&d_prod, need_si * sizeof(float))
              : err2;
      if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess) {
        cudaError_t first_err =
            err1 != cudaSuccess ? err1 : (err2 != cudaSuccess ? err2 : err3);
        LOG_ERROR("[EXPERT-GPU] GpuExpertScratch::ensure: d_gate/d_up/d_prod "
                  "grow to %.2f MB "
                  "failed on device %d: %s\n",
                  (need_si * sizeof(float)) / 1e6, device_id,
                  cudaGetErrorString(first_err));
        if (d_gate) {
          cudaFree(d_gate);
          d_gate = nullptr;
        }
        if (d_up) {
          cudaFree(d_up);
          d_up = nullptr;
        }
        if (d_prod) {
          cudaFree(d_prod);
          d_prod = nullptr;
        }
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
    if (device_id < 0)
      return;
    cudaSetDevice(device_id);
    if (d_x) {
      cudaFree(d_x);
      d_x = nullptr;
    }
    if (d_down) {
      cudaFree(d_down);
      d_down = nullptr;
    }
    if (d_gate) {
      cudaFree(d_gate);
      d_gate = nullptr;
    }
    if (d_up) {
      cudaFree(d_up);
      d_up = nullptr;
    }
    if (d_prod) {
      cudaFree(d_prod);
      d_prod = nullptr;
    }
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
static bool check_finite(const char *label, int layer_idx, int expert_id,
                         const float *data, size_t n) {
  int nan_count = 0, inf_count = 0;
  for (size_t i = 0; i < n; ++i) {
    if (std::isnan(data[i]))
      ++nan_count;
    else if (std::isinf(data[i]))
      ++inf_count;
  }
  if (nan_count > 0 || inf_count > 0) {
    LOG_ERROR("[EXPERT-GPU] %s layer=%d expert=%d: %d/%zu NaN, %d/%zu Inf.\n",
              label, layer_idx, expert_id, nan_count, n, inf_count, n);
    trace_abort_if_nan("EXPERT-GPU", true);
    return false;
  }
  if (trace_gate("EXPERT-GPU")) {
    std::string labeled =
        std::string(label) + " expert=" + std::to_string(expert_id);
    trace_row("EXPERT-GPU", layer_idx, labeled, data, n);
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
// Prefill-only expert GEMM: dequantise the weight to BF16 and hand it to
// cuBLAS, instead of the fused GEMV-shaped kernels below.
//
// Same finding as the dense side (see dense_dispatch.hpp's
// dense_bf16_prefill_gemm_flag): every fused kernel here is one-output-row-
// per-block and streams the activation, so each of the O blocks re-reads all
// of [S, I]. Right at S=1, badly wrong at prefill batch sizes. Measured on an
// RTX 3090: a [62x2560]x[2560x640] BF16 GEMM (one expert's real prefill shape
// after moe_prefill_batched groups its tokens) is 0.0109 ms, so all 73,728
// expert GEMMs of a 3,189-token prefill would total ~0.8s against the 112.9s
// the fused kernels currently take.
//
// Everything is already device-resident here (unlike dense_matmul_host, which
// takes host pointers), so this adds no PCIe traffic at all -- only a weight
// dequantisation into scratch.
//
// Correctness: the four dequant kernels were checked against these very fused
// kernels, which decode the same bytes -- agreement to 0.20-0.30% of |y|max,
// i.e. BF16's own 8-bit mantissa precision and nothing more.
#ifndef MAX_CUDA_DEVICES_MM
#define MAX_CUDA_DEVICES_MM 16
#endif
static bool gpu_matmul_dispatch_bf16(GGUFType type, float *y, const float *x,
                                     const void *w, int S, int I, int O) {
  const int t = (int)type;
  if (t != 7 && t != 8 && t != 12 && t != 13)
    return false; // Q5_1/Q8_0/Q4_K/Q5_K

  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess)
    return false;
  if (dev < 0 || dev >= MAX_CUDA_DEVICES_MM)
    return false;

  // Per-device lazily-created handle + grow-only scratch. The expert dispatch
  // path is decode-thread-only (the active-layer upload workers never reach
  // here), so plain statics are safe without a lock -- same assumption the
  // surrounding gpu_pool/pending_gpu bookkeeping already makes.
  static cublasHandle_t handles[MAX_CUDA_DEVICES_MM] = {};
  static void *w_bf16[MAX_CUDA_DEVICES_MM] = {};
  static size_t w_cap[MAX_CUDA_DEVICES_MM] = {};
  static void *x_bf16[MAX_CUDA_DEVICES_MM] = {};
  static size_t x_cap[MAX_CUDA_DEVICES_MM] = {};

  if (!handles[dev] && cublasCreate(&handles[dev]) != CUBLAS_STATUS_SUCCESS) {
    handles[dev] = nullptr;
    return false;
  }
  auto grow = [](void *&buf, size_t &cap, size_t need) {
    if (need <= cap)
      return true;
    if (buf) {
      cudaFree(buf);
      buf = nullptr;
    }
    cap = 0;
    if (cudaMalloc(&buf, need) != cudaSuccess)
      return false;
    cap = need;
    return true;
  };
  const size_t wb = (size_t)O * I * sizeof(uint16_t);
  const size_t xb = (size_t)S * I * sizeof(uint16_t);
  if (!grow(w_bf16[dev], w_cap[dev], wb))
    return false;
  if (!grow(x_bf16[dev], x_cap[dev], xb))
    return false;

  if (!launch_dequant_to_bf16(t, w_bf16[dev], w, (int64_t)O * I))
    return false;
  launch_f32_to_bf16(x_bf16[dev], x, (int64_t)S * I);

  const float alpha = 1.0f, beta = 0.0f;
  const cublasStatus_t st = cublasGemmEx(
      handles[dev], CUBLAS_OP_T, CUBLAS_OP_N, O, S, I, &alpha, w_bf16[dev],
      CUDA_R_16BF, I, x_bf16[dev], CUDA_R_16BF, I, &beta, y, CUDA_R_32F, O,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  return st == CUBLAS_STATUS_SUCCESS;
}

static bool gpu_matmul_dispatch(GGUFType type, float *y, const float *x,
                                const void *w, int S, int I, int O) {
  // !! THIS BLOCK MUST STAY ABOVE `switch (type)` !!
  // It used to sit BETWEEN `switch (type) {` and the first `case`, where C++
  // never executes it: control jumps straight to a case label, so every
  // statement before the first one is unreachable. That is why this feature
  // measured as "no effect" when it was first tried (2026-09-04) and why the
  // conclusion recorded then -- "the expert GEMM was never the bottleneck" --
  // was drawn from code that had never run. Found 2026-09-05 by putting an
  // UNCONDITIONAL one-shot log at the top of the block and observing that it
  // never fired even though the enclosing function demonstrably ran.
  // MEASURED AND OFF BY DEFAULT -- this gave nothing, and the reason is worth
  // recording. The expert GEMM was never the bottleneck: a real 3,189-token
  // prefill shows expert=111.93s of which the actual matmul (`cmp`) is
  // 0.24s. The bucket is dominated by ram_scan=50.92s (get_expert=49.58s,
  // ~3.2 ms per call on a 100%-cache-hit preloaded RAM tier), then
  // resolve_loc=13.61s and gpu_finish=13.37s (that last one is the
  // D2H->CPU-SiLU->H2D round trip, dn=15.98s).
  //
  // The earlier "15 TFLOP at 134 GFLOPS, ~140x available" projection was a
  // MISATTRIBUTION: it divided the expert bucket's total flops by the WHOLE
  // bucket wall time, which is mostly RAM-tier bookkeeping, not compute.
  // Switching it on measured 0.24 -> 0.26s on cmp and prefill 291.08 ->
  // 299.84s, i.e. nothing, or slightly worse.
  //
  // Deliberately NOT sharing dense_bf16_prefill_gemm's flag: that one is a
  // real 47x win on the dense side and must not drag this along. The code and
  // its four verified dequant kernels stay because they become useful the
  // moment get_expert is fixed and compute actually matters here.
  if (S > 1 && moe_expert_bf16_gemm()) {
    static std::atomic<uint64_t> ok{0}, fail{0};
    if (gpu_matmul_dispatch_bf16(type, y, x, w, S, I, O)) {
      if ((ok.fetch_add(1) & 0xFFFFF) == 0)
        LOG_STREAM_INFO() << "[EXPERT-BF16] engaged " << ok.load()
                          << " / failed " << fail.load() << std::endl;
      return true;
    }
    if ((fail.fetch_add(1) & 0xFFFFF) == 0)
      LOG_STREAM_INFO() << "[EXPERT-BF16] FAILED for type=" << (int)type
                        << " S=" << S << " I=" << I << " O=" << O
                        << " (engaged " << ok.load() << ")" << std::endl;
  }

  switch (type) {
  case GGUFType::Q4_0:
    launch_gguf_q4_0_dp4a(y, x, w, S, I, O);
    return true;
  case GGUFType::Q2_K:
    launch_q2_K_matmul(y, x, w, S, I, O);
    return true;
  case GGUFType::Q3_K:
    launch_q3_K_matmul(y, x, w, S, I, O);
    return true;
  case GGUFType::Q4_K:
    launch_q4_K_matmul(y, x, w, S, I, O);
    return true;
  case GGUFType::Q5_K:
    launch_q5_K_matmul(y, x, w, S, I, O);
    return true;
  case GGUFType::Q6_K:
    launch_q6_K_matmul(y, x, w, S, I, O);
    return true;
  case GGUFType::IQ2_XXS:
    launch_iq2_xxs_matmul(y, x, w, S, I, O);
    return true;
  case GGUFType::IQ2_XS:
    launch_iq2_xs_matmul(y, x, w, S, I, O);
    return true;
  case GGUFType::IQ2_S:
    launch_iq2_s_matmul(y, x, w, S, I, O);
    return true;
  case GGUFType::IQ3_XXS:
    launch_iq3_xxs_matmul(y, x, w, S, I, O);
    return true;
  // IQ3_S / IQ1_S: real kernels, advertised as GPU-capable by
  // gpu_type_supported() (kquant_dequant.hpp), but never actually wired
  // into this switch until now -- see this file's own extern declaration
  // comment for launch_iq3s_matmul for the full incident. IQ1_S's extern
  // was already declared above; it had the identical gap.
  case GGUFType::IQ3_S:
    launch_iq3s_matmul(y, x, w, S, I, O);
    return true;
  case GGUFType::IQ1_S:
    launch_iq1s_matmul(y, x, w, S, I, O);
    return true;
  // IQ1_M: the IDENTICAL gap this block's own comment describes for IQ3_S/
  // IQ1_S, still open -- kquant_dequant.hpp's gpu_type_supported() has
  // returned true for IQ1_M since its kernel landed, but it was never added
  // here. Found 2026-09-04 by diffing the two switches mechanically rather
  // than by reading them. This one was worse than the IQ3_S/IQ1_S version:
  // every caller below DISCARDS this function's bool return, so an IQ1_M
  // expert would have been promoted to VRAM (burning PCIe + a cache slot),
  // reached this switch, fallen through to `default: return false`, and then
  // had its d_gate/d_up/d_down read as whatever stale bytes the scratch
  // buffer already held -- silent garbage, not a clean CPU fallback. No real
  // IQ1_M checkpoint has been run through this engine yet (see that type's
  // own comment in gpu_type_supported), so this was latent, not a live
  // corruption -- but it was one real IQ1_M file away from being one.
  case GGUFType::IQ1_M:
    launch_iq1m_matmul(y, x, w, S, I, O);
    return true;
  // IQ4_NL: added 2026-09-04. Down projection of ALL 48 MoE layers in the
  // Qwen3.8-Flash-Next UD-IQ1_M checkpoint -- without it gpu_type_supported()
  // rejected every layer (gate/up/down must ALL be supported), so the whole
  // model ran on CPU with both GPUs idle.
  case GGUFType::IQ4_NL:
    launch_iq4_nl_matmul(y, x, w, S, I, O);
    return true;
  // Q8_0: launch_q8_0_matmul has existed and been exercised for a long time
  // -- but only by the DENSE path (cuda_manager.hpp's dense_matmul_host),
  // never by routed experts, because it was missing from BOTH this switch
  // and gpu_type_supported(). Added 2026-09-04 for qwen4exp
  // (Qwen3.8-Flash-Next), whose Unsloth UD-Q4_K_XL checkpoint stores
  // ffn_down_exps as Q8_0 on 5 of 48 MoE layers (Q5_1 on the other 43, which
  // still has no kernel) -- and since gpu_type_supported() is an AND over
  // gate/up/down, that one tensor was forcing those whole layers' experts
  // onto CPU at the ~9-10x per-call tax this project has already measured.
  // Same kernel, same launcher, same Test 5 coverage (which includes
  // attn_q_b's real production shape at S=1) -- this is a wiring fix, not a
  // new kernel.
  case GGUFType::Q8_0:
    launch_q8_0_matmul(y, x, w, S, I, O);
    return true;
  // Q5_1: the format 43 of qwen4exp's 48 MoE layers store ffn_down_exps in.
  // See cuda/q5_1_matmul.cu's header for the ggml reference it was ported
  // from and why the affine form collapses to two accumulators per block.
  case GGUFType::Q5_1:
    launch_q5_1_matmul(y, x, w, S, I, O);
    return true;
  case GGUFType::IQ4_XS:
    launch_iq4_xs_matmul(y, x, w, S, I, O);
    return true;
  // MXFP4: dead code until kquant_dequant.hpp's gpu_type_supported()
  // flips MXFP4 to true (post real-hardware Test 5 pass) -- wired
  // here now so that flip is a genuine one-line change with nothing
  // else left to plumb.
  case GGUFType::MXFP4:
    launch_mxfp4_matmul(y, x, w, S, I, O);
    return true;
  default:
    return false;
  }
}

// Context needed to opportunistically dispatch an expert's FFN on GPU.
// `loc` is the expert's current VRAM residency (nullptr if not cached).
struct ExpertGpuContext {
  GpuTopology *topo = nullptr;
  const ExpertLocation *loc = nullptr;
  int layer_idx = -1;
  int expert_id = -1;
  const uint8_t *slab_buffer = nullptr;
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
  MoEExpertBlock *exp = nullptr;
  // EngineConfig::cpu_offload_max_experts: this k was chosen (this token/
  // layer, this call only) to redirect to CPU compute instead of a VRAM
  // upload -- set by dispatch_expert_k from a per-token force_cpu_
  // offload_k flag, itself computed right after get_experts_batch returns
  // (see that call site's own comment). expert_ffn_resolve_gpu_loc checks
  // this immediately, mirroring is_cpu_affinity_layer's early-return
  // pattern exactly, so no upload branch below it can ever fire.
  bool force_cpu_offload = false;
};

struct ExpertScore {
  int id;
  float score;
};

// One GPU-path expert whose gate/up kernels have been issued (async) but not
// yet collected -- queued during moe_forward_gguf's Pass A so Pass B can
// finish them (sync + download + down-proj) after the CPU has had time to
// churn through the CPU-path experts in between. `loc` is resolved once at
// issue time so finish doesn't need to re-derive VRAM residency.
struct PendingGpuExpert {
  int k = -1;
  MoEExpertBlock *exp = nullptr;
  const ExpertLocation *loc = nullptr;
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
  // Decode-time device accumulation has already consumed this expert's
  // d_down with expert_accum_add(). Keep the VRAM hold and RAM pin until the
  // accumulator's final stream wait below, then release them together.
  bool accumulated_on_device = false;
};

// 2026-08-29: minimal grow-only pinned-host float buffer, backed by
// cudaMallocHost instead of cudaHostRegister-ing an existing std::vector's
// heap allocation. This sidesteps the real, confirmed "already mapped"
// cudaHostRegister failure a plain std::vector allocation hits once
// EngineConfig::ram_pinned_budget_gb is raised past ~10-30GB (see
// LayerScratch::resize's own comment below) -- a fresh cudaMallocHost
// allocation comes straight from CUDA's own pinned-memory pool, never
// colliding with anything this codebase separately registers elsewhere.
// API surface deliberately kept to exactly what LayerScratch's own 4
// call sites use (data()/size()/resize()/operator[]), confirmed via a
// full grep sweep of every scratch.normed/gate_buf/up_buf/down_buf use
// in this file -- not a general-purpose vector replacement.
class PinnedFloatBuffer {
public:
  PinnedFloatBuffer() = default;
  ~PinnedFloatBuffer() { free_buf(); }
  PinnedFloatBuffer(const PinnedFloatBuffer &) = delete;
  PinnedFloatBuffer &operator=(const PinnedFloatBuffer &) = delete;

  // Grow-only, same as std::vector::resize() growing this codebase's own
  // call sites already assume (every use here writes the buffer fully
  // before reading it back within the same call, so old content is never
  // relied upon across a resize -- confirmed via the same grep sweep).
  bool resize(size_t n, const char *name = nullptr) {
    if (n > capacity_) {
      size_t new_cap = std::max(n, capacity_ * 2);
      if (new_cap > std::numeric_limits<size_t>::max() / sizeof(float))
        return false;
      const size_t new_bytes = new_cap * sizeof(float);
      if (!MemoryGuard::try_reserve_host_bytes(new_bytes)) {
        if (name) {
          LOG_STREAM_WARN()
              << "[MEM-GUARD] scratch allocation refused: " << name
              << " bytes=" << new_bytes
              << " reserved=" << MemoryGuard::reserved_host_bytes()
              << " budget=" << MemoryGuard::host_budget_bytes() << "\n";
        }
        return false;
      }
      float *new_ptr = nullptr;
      cudaError_t err = cudaMallocHost((void **)&new_ptr, new_bytes);
      bool now_pinned = (err == cudaSuccess && new_ptr != nullptr);
      if (!now_pinned) {
        // Never a hard failure -- fall back to a plain (unpinned)
        // heap allocation, matching this codebase's own standing
        // "best-effort, never a hard requirement" discipline for
        // every other GPU fast-path.
        new_ptr = (float *)std::malloc(new_bytes);
      }
      if (!new_ptr) {
        MemoryGuard::release_host_bytes(new_bytes);
        if (name) {
          LOG_STREAM_WARN() << "[SCRATCH] allocation failed: " << name
                            << " bytes=" << new_bytes << "\n";
        }
        return false;
      }
      if (name) {
        LOG_STREAM_INFO() << "[SCRATCH-PIN] " << name << ": "
                          << (now_pinned ? "pinned (cudaMallocHost)"
                                         : "unpinned fallback")
                          << " bytes=" << new_bytes
                          << (now_pinned ? ""
                                         : (std::string(" (") +
                                            cudaGetErrorString(err) + ")"))
                          << "\n";
      }
      // A grow FREES the old buffer, which may still be the source of an
      // in-flight cudaMemcpyAsync -- cudaFreeHost does NOT wait for that.
      // Harmless for the original users here (they grow once at startup and
      // their copies were effectively synchronous anyway), but a real
      // use-after-free for any caller whose uploads are genuinely async.
      // MEASURED 2026-09-05: the prefill expert path's activation arena hit
      // exactly this and produced fluent-but-WRONG, run-to-run-varying
      // output with zero errors reported. Growth is rare (doubling), so a
      // full device sync here costs nothing measurable.
      if (pinned_ && ptr_)
        cudaDeviceSynchronize();
      free_buf();
      ptr_ = new_ptr;
      capacity_ = new_cap;
      pinned_ = now_pinned;
      reserved_bytes_ = new_bytes;
    }
    size_ = n;
    return true;
  }
  float *data() { return ptr_; }
  const float *data() const { return ptr_; }
  size_t size() const { return size_; }
  float &operator[](size_t i) { return ptr_[i]; }
  const float &operator[](size_t i) const { return ptr_[i]; }

private:
  void free_buf() {
    if (ptr_) {
      if (pinned_)
        cudaFreeHost(ptr_);
      else
        std::free(ptr_);
      ptr_ = nullptr;
      MemoryGuard::release_host_bytes(reserved_bytes_);
      reserved_bytes_ = 0;
      size_ = 0;
      capacity_ = 0;
    }
  }
  float *ptr_ = nullptr;
  size_t size_ = 0;
  size_t capacity_ = 0;
  bool pinned_ = false;
  size_t reserved_bytes_ = 0;
};

struct LayerScratch {
  std::vector<float> attn_out;
  std::vector<float> moe_out;
  // Only populated for the Qwen4Exp + GGML HC handoff. It is a non-owning
  // pointer into GgmlMoeRunner's persistent graph output and is consumed
  // immediately after moe_forward_gguf returns; all other models leave it
  // null and retain the original host path.
  const void *moe_device_output = nullptr;
  // The GGML MoE runner that owns moe_device_output.  Qwen4Exp can use more
  // than one persistent runner because its layers have different quantization
  // signatures; active_instance() may therefore no longer be the producer by
  // the time a fallback/readback is needed.
  GgmlMoeRunner *moe_device_runner = nullptr;
  // True when moe_device_output is owned by GgmlQwenSharedRunner rather than
  // GgmlMoeRunner.  This lets the HC fallback read the combined device output
  // without mistaking it for a routed-expert runner output.
  bool moe_shared_device_output = false;
  bool moe_device_output_ready = false;
  PinnedFloatBuffer normed;
  std::vector<float> router_logits;
  PinnedFloatBuffer gate_buf;    // expert gate_proj output [S * moe_inter]
  PinnedFloatBuffer up_buf;      // expert up_proj output [S * moe_inter]
  PinnedFloatBuffer down_buf;    // expert down_proj output [S * D]
  std::vector<float> expert_acc; // accumulated expert output [S * D]
  // Pinned destination for the Qwen CPU routed result.  The mixed CPU/GPU
  // path uploads it with ggml_backend_tensor_set_async() into the shared
  // expert graph's persistent routed input.
  PinnedFloatBuffer qwen_cpu_moe_out;

  // Token-loop scratch (pre-allocated to eliminate per-token heap churn)
  std::vector<ExpertScore> scores;           // [n_experts]
  std::vector<MoEExpertBlock *> expert_ptrs; // [topk]
  std::vector<float> pilot_nrm;              // [D]
  std::vector<float> pilot_logits;           // [n_experts]

  // Qwen4Exp-only router pilot. The HC graph's mixed activation is not
  // exposed as a host tensor between layers, so this explicitly records a
  // stream-0 proxy from the device-authoritative HC state. It is kept
  // separate from the conventional pilot fields above so other families
  // cannot accidentally consume or overwrite Qwen measurements.
  static constexpr int kQwenPilotMax = 16;
  std::array<int, kQwenPilotMax> qwen_pilot_pred_ids{};
  std::vector<float> qwen_pilot_logits;      // [n_experts]
  std::vector<int32_t> qwen_pilot_gpu_ids;   // [topk], device-router output
  std::vector<float> qwen_pilot_gpu_weights; // [topk], device-router output
  bool qwen_pilot_gpu_pending = false;
  int qwen_pilot_gpu_pending_layer = -1;
  int qwen_pilot_pred_count = 0;
  int qwen_pilot_pred_layer = -1;
  int qwen_pilot_prefetch_count = 0;

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
  std::vector<const ExpertLocation *> vram_loc_hint; // [topk]

  // Up to `topk` independent GPU device-buffer sets, so up to `topk`
  // GPU-path experts can have gate/up kernels in flight simultaneously
  // (issued but not yet collected) instead of each expert fully
  // round-tripping (upload -> kernel -> sync -> download) before the next
  // one starts. See moe_forward_gguf's Pass A / Pass B and
  // expert_ffn_gpu_issue / expert_ffn_gpu_finish.
  std::vector<GpuExpertScratch> gpu_pool; // [topk]
  std::vector<PendingGpuExpert>
      pending_gpu; // capacity topk, cleared+refilled per token

  // Whole-batch router lookahead (prefill only, S > 1): the router already
  // knows every token's top-k choice before any expert runs, since
  // router_logits is one batched matmul over all S tokens. Cache each
  // token's decision here so it's computed once, and tally how often each
  // expert is chosen across the batch so cold-start VRAM warm-up can fill
  // free slots with the most in-demand experts first.
  std::vector<int> tok_topk_ids;       // [S * topk]
  std::vector<float> tok_topk_weights; // [S * topk]
  std::vector<int> expert_freq;        // [n_experts]
  // EngineConfig::prefill_hint_order_by_earliest_token -- companion to
  // expert_freq, filled in the SAME Phase 1 loop at zero extra scan cost:
  // for each expert, the LOWEST token index s that picked it this layer
  // (INT_MAX if never picked). Phase 2's hint-push order can use this
  // instead of pure expert_freq-descending -- see that config field's own
  // comment for the motivation (later tokens in Phase 3's dispatch order
  // already accumulate real lead time "for free" from the tokens ahead of
  // them; the earliest tokens get almost none, regardless of how often
  // their own experts are used elsewhere in the layer).
  std::vector<int> first_needed_by_token; // [n_experts]

  // prefill_warm_via_prefetcher=true: experts Phase 2 hinted to the async
  // prefetcher but hadn't yet confirmed VRAM-resident. Unlike decode's own
  // speculative ram_vram_promote_top_k pilot promotion, opportunistically
  // promoting these is NOT a bet -- every entry here is a CONFIRMED real
  // router pick for a token already in this exact prefill batch (built
  // from expert_freq, the batch's own already-computed router output),
  // so it WILL be dispatched to before this prefill call returns; the
  // only open question is timing, not correctness or "was this the right
  // guess". Checked at the START of each subsequent layer's own Phase 2
  // (cross-layer -- a same-layer check right after push_hint essentially
  // never finds the fetch has landed yet, see push_hint's own call-site
  // comment) and opportunistically promoted via a decode-thread
  // upload_expert_async call, mirroring exactly what the legacy
  // synchronous Phase-2 path already did inline for every candidate.
  // Phase 3's own synchronous dispatch remains the correctness backstop
  // regardless -- an entry that ages out here unpromoted just costs
  // exactly what it always cost before this opportunistic path existed,
  // never a hang or a miss. static thread_local LayerScratch persists
  // across the whole layer loop (and across requests), so this is a
  // real, if small, cross-layer relay, not per-call scratch.
  struct PendingPrefillPromotion {
    int layer_idx;
    int expert_id;
    int age_layers;
  };
  static constexpr int PENDING_PREFILL_CAP = 32;
  std::array<PendingPrefillPromotion, PENDING_PREFILL_CAP>
      pending_prefill_promotions{};
  int pending_prefill_count = 0;

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

  // EngineConfig::pilot_widen_all_tokens (MEASUREMENT ONLY) -- same idea
  // as pilot_pred_ids above, but the UNION of every token's own top-k
  // picks (not just the last token's) -- see that config field's own
  // comment. 128 = generous headroom (min(pilot_k,topk)=8 per token x
  // up to 16 tokens' worth of fully-disjoint growth before real overlap
  // stops adding new entries -- a short 10-20 token test prompt should
  // never come close to overflowing this).
  static constexpr int kPilotWideMax = 128;
  std::array<int, kPilotWideMax> pilot_wide_pred_ids{};
  int pilot_wide_pred_count = 0;
  int pilot_wide_pred_layer = -1;
  std::vector<float>
      pilot_wide_nrm; // [D] -- separate from pilot_nrm, own scratch
  std::vector<float> pilot_wide_logits; // [n_experts] -- separate from
                                        // pilot_logits, own scratch
  // Parallel to pilot_pred_ids: whether pilot_pred_ids[k] was actually
  // "fetch-worthy" (not already cached anywhere) at prediction time, and
  // so was actually pushed as a hint. Lets the recall-check site below
  // compute a SECOND accuracy figure restricted to the subset of
  // predictions that would actually trigger a real fetch -- the number
  // that determines whether prefetch is worth enabling, as opposed to
  // pilot_recall's raw top-K accuracy (which includes already-cached
  // picks whose correctness doesn't matter to that question).
  std::array<bool, 8> pilot_pred_was_hint{};
  // Parallel to pilot_pred_ids, same idea as pilot_pred_was_hint but for
  // the OTHER bucket that matters for RAM->VRAM promotion: was
  // pilot_pred_ids[k] classified "already_ram" (RAM-resident, NOT yet
  // VRAM-resident) at prediction time? This is a genuinely different
  // population from pilot_pred_was_hint's "needs_fetch" bucket -- disjoint,
  // in fact, since a pick is classified into exactly one of the four
  // buckets (needs_fetch / already_vram_pinned / already_vram_lru /
  // already_ram). Neither the raw top-K recall (which mixes all four
  // buckets together) nor the fetch-worthy accuracy (which measures ONLY
  // needs_fetch, the population the EXISTING disk-prefetch path cares
  // about) says anything about this one -- the population a RAM->VRAM
  // promotion decision actually needs: of the picks already sitting in
  // RAM with nothing further to fetch, how often does the prediction
  // that put them there turn out to match the real router's later pick?
  std::array<bool, 8> pilot_pred_was_ram_only{};

  // DEPTH-2 measurement (approach (a), "naive reuse" only) -- mirrors
  // pilot_pred_ids/count/layer/was_hint, one hop further out. Predicts
  // a slot's target layer's routing by reusing the SAME pre-(l+1)-
  // attention hidden_states vector the l+1 pilot above already computed
  // from, i.e. gated on a vector missing l+1's own FFN contribution
  // entirely -- the single term already proven fatal to omit at depth-1
  // (see the pilot loop's own comment on the 0%-recall failure the
  // post-FFN timing fix corrected). Deliberately NOT wired to
  // async_io_queue.push_hint or publish_pilot_confirmed_resident --
  // this is pure recall measurement to settle whether depth-2 is worth
  // building at all, not a real prefetch path.
  //
  // TWO-SLOT, not one: a depth-2 prediction set while processing layer l
  // (targeting l+2) has to survive a FULL intervening iteration (l+1) --
  // whose OWN pilot block ALSO fires and would overwrite a single shared
  // field with its own l+3 target before layer l+2 is ever reached to
  // consume the l+2 entry. A single scalar here silently loses ~every
  // prediction except the last one or two near the tail of the layer
  // stack (caught via a suspiciously low check count: exactly
  // ~1/layer-count of the expected volume, one surviving match per
  // token instead of one per (layer, token) pair). Indexed by l % 2:
  // l and l+2 always share parity, so slot[l % 2] is only ever
  // overwritten again at iteration l+2 itself -- AFTER that iteration's
  // own check (at the top of moe_forward_gguf) has already consumed it.
  // l+1 (opposite parity) writes to the other slot and never collides.
  struct Pilot2Pending {
    std::array<int, 8> ids{};
    int count = 0;
    int layer = -1;
    std::array<bool, 8> was_hint{};
  };
  std::array<Pilot2Pending, 2> pilot_pred2_slots{};

  // ATTN-PILOT measurement (2026-08-25, MEASUREMENT ONLY) -- see the
  // firing site's own comment (moe_forward_gguf, right after the
  // attention-residual check_finite call) for the full design. Single
  // scalar, same survival-window reasoning as pilot_pred_ids above (one
  // intervening layer boundary, not two).
  std::array<int, 8> pilot_attn_pred_ids{};
  int pilot_attn_pred_count = 0;
  int pilot_attn_pred_layer = -1;
  // Parallel to pilot_attn_pred_ids -- was this pick fetch-worthy (not
  // already cached anywhere) at prediction time? Same purpose as
  // pilot_pred_was_hint above, mirrored for ATTN-PILOT's own real-hint
  // wiring and fetch-worthy accuracy reporting.
  std::array<bool, 8> pilot_attn_pred_was_hint{};
  // Snapshot of the last-token hidden state taken right after l's
  // attention residual (before l's FFN runs) -- the actual dense_proj/
  // gating prediction has to be DEFERRED until after l's own moe_forward_
  // gguf call has already consumed whatever was pending in pilot_attn_
  // pred_layer (see the firing site's own comment for the real ordering
  // bug this fixes), by which point hidden_states itself has moved on to
  // POST-FFN -- so the pre-FFN input this prediction needs has to be
  // captured here and carried forward instead of read live later.
  std::vector<float> pilot_attn_snapshot;

  // 2026-08-29: TRIED pinning normed/gate_buf/up_buf/down_buf (the 4
  // host buffers behind every routed-expert GPU dispatch's blocking
  // cudaMemcpy calls) via cudaHostRegister, motivated by real nsys
  // profiling (cudaMemcpy+cudaMemcpyAsync+cudaStreamSynchronize together
  // = 83.7% of all CUDA API time across a real request) -- REVERTED. Two
  // real problems found via a diagnostic log line, not assumed: (1)
  // cudaHostRegister failed on EVERY call, always with "part or all of
  // the requested memory range is already mapped". (2) resize() (and so
  // the attempted registration) fires roughly once/second in a real run,
  // not once per request as assumed -- meaning the failed registration
  // attempts were ALSO adding pure per-call overhead with no offsetting
  // benefit.
  //
  // ROOT-CAUSED same day via a follow-up one-shot diagnostic (since
  // removed) that swept EngineConfig::ram_pinned_budget_gb: registration
  // succeeds cleanly at 0GB and 10GB, fails at 30GB and the live 45GB --
  // a real, confirmed causal interaction, not coincidence (the RAM
  // tier's OWN "RAM slab persistent pin" report line independently shows
  // 655 registration failures of its OWN slabs at 30GB too, not just
  // budget-exhaustion skips -- the same phenomenon, scaling with how
  // much of the process's virtual address space is already registered).
  // The exact mechanism (real page-level collision vs. a driver-internal
  // VMA-tracking quirk) isn't confirmed, only the real, reproducible
  // threshold. This creates a genuine conflict: ram_pinned_budget_gb=45
  // is ITSELF an already-proven, real win (see that field's own comment)
  // for a different reason (upload_expert_active_async's DMA becoming
  // genuinely async) -- closing THIS gap would mean trading away some of
  // THAT gap, not a free win. Properly resolving this needs a real A/B
  // across candidate budget values measuring BOTH effects together, not
  // attempted this session. See CLAUDE.md for the full writeup.

  bool resize(int S, int D, int n_experts, int moe_inter, int topk,
              int n_expert_shared = 0, int dense_inter = 0) {
    attn_out.resize(S * D);
    moe_out.resize(S * D);
    if (!normed.resize(S * D, "normed"))
      return false;
    router_logits.resize(S * n_experts);
    // gate_buf/up_buf are reused per-token (S=1 per call) at whichever of
    // the routed, shared-expert, or leading-dense-layer intermediate
    // width is largest — the shared-expert width is
    // moe_inter * n_expert_shared when >1 shared expert is present.
    int shared_inter = moe_inter * std::max(1, n_expert_shared);
    int buf_inter = std::max({moe_inter, shared_inter, dense_inter});
    if (!gate_buf.resize(S * buf_inter, "gate_buf") ||
        !up_buf.resize(S * buf_inter, "up_buf") ||
        !down_buf.resize(S * D, "down_buf"))
      return false;
    expert_acc.resize(S * D);
    scores.resize(n_experts);
    expert_ptrs.resize(topk);
    vram_shadow_blocks.resize(std::max(1, topk));
    vram_loc_hint.assign(std::max(1, topk), nullptr);
    pilot_nrm.resize(D);
    pilot_logits.resize(n_experts);
    qwen_pilot_logits.resize(n_experts);
    qwen_pilot_gpu_ids.resize(std::max(1, topk));
    qwen_pilot_gpu_weights.resize(std::max(1, topk));
    qwen_pilot_pred_count = 0;
    qwen_pilot_pred_layer = -1;
    qwen_pilot_prefetch_count = 0;
    qwen_pilot_gpu_pending = false;
    qwen_pilot_gpu_pending_layer = -1;
    pilot_wide_nrm.resize(D);
    pilot_wide_logits.resize(n_experts);
    pilot_attn_snapshot.resize(D);
    tok_topk_ids.resize(S * topk);
    tok_topk_weights.resize(S * topk);
    expert_freq.resize(n_experts);
    first_needed_by_token.resize(n_experts);
    if ((int)gpu_pool.size() < std::max(1, topk))
      gpu_pool.resize(std::max(1, topk));
    pending_gpu.reserve(std::max(1, topk));
    return true;
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
static constexpr int GPU_COLD_BATCH_THRESHOLD = 4; // amortizes a cold upload

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
static const ExpertLocation *expert_ffn_resolve_gpu_loc(
    const ExpertGpuContext *gpu_ctx, int S, const GGUFTensor &gate_meta,
    const GGUFTensor &up_meta, const GGUFTensor &down_meta,
    bool *out_needs_release, bool *out_loc_from_active_layer,
    RAMSafetyManager &ram_mgr, StreamingEngine &engine) {
  PERF_SCOPE(t_resolve_gpu_loc_total);
  *out_needs_release = false;
  const ExpertLocation *loc = gpu_ctx ? gpu_ctx->loc : nullptr;
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
    perf().n_cpu_fallback_affinity_layer.fetch_add(1,
                                                   std::memory_order_relaxed);
    return nullptr;
  }
  // EngineConfig::cpu_offload_max_experts: this specific k was capped to
  // CPU compute for THIS token/layer only (not a standing per-layer
  // affinity like cpu_affinity_layers above) -- same early-return shape,
  // same reasoning (every upload branch below keys on !vram_resident, so
  // this must return before any of them run, not just force vram_resident
  // false). The counter is incremented at the call site that SET this
  // flag (execution_layer.cpp's dispatch loop), not here, since that's
  // where the real vs. capacity-exhausted split is known.
  if (gpu_ctx && gpu_ctx->force_cpu_offload) {
    return nullptr;
  }
  bool vram_resident = loc && loc->device_ptr && loc->layer_idx >= 0;
  if (cpu_only_mode())
    vram_resident = false; // diagnostic: force CPU expert path
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
  if (!vram_resident && gpu_ctx && gpu_ctx->topo && gpu_ctx->slab_buffer &&
      gpu_ctx->slab_capacity > 0 && S >= GPU_COLD_BATCH_THRESHOLD) {
    loc = gpu_ctx->topo->upload_expert(gpu_ctx->layer_idx, gpu_ctx->expert_id,
                                       gpu_ctx->slab_buffer,
                                       gpu_ctx->slab_capacity);
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
    if (vram_resident && gpu_ctx->exp)
      ram_mgr.finish_sync_promotion(*gpu_ctx->exp, gpu_ctx->layer_idx,
                                    gpu_ctx->expert_id,
                                    &engine.layers_experts[gpu_ctx->layer_idx]);
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
  if (!vram_resident && gpu_ctx && gpu_ctx->topo && gpu_ctx->slab_buffer &&
      gpu_ctx->slab_capacity > 0 && S < GPU_COLD_BATCH_THRESHOLD &&
      gpu_type_supported(gate_meta.type) && gpu_type_supported(up_meta.type) &&
      gpu_type_supported(down_meta.type)) {

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
    loc = gpu_ctx->topo->find_active_expert(
        gpu_ctx->layer_idx, gpu_ctx->expert_id, [&](int layer, int expert_id) {
          promoted_to_main = true;
          if (gpu_ctx->exp)
            ram_mgr.finish_sync_promotion(*gpu_ctx->exp, layer, expert_id,
                                          &engine.layers_experts[layer]);
        });
    vram_resident = loc && loc->device_ptr;
    if (vram_resident) {
      loc_from_active_layer = !promoted_to_main;
      LOG_DEBUG(
          "[MOE-VRAM]  ACTIVE-LAYER HIT layer=%d expert=%d hit_count=%u%s\n",
          gpu_ctx->layer_idx, gpu_ctx->expert_id, loc->hit_count,
          promoted_to_main ? " (promoted to main cache)" : "");
    } else {
      // First touch this layer-window -- try active layer area (decode-
      // thread-only, no eviction of the main LRU cache)
      loc = gpu_ctx->topo->upload_expert_active(
          gpu_ctx->layer_idx, gpu_ctx->expert_id, gpu_ctx->slab_buffer,
          gpu_ctx->slab_capacity);
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
        LOG_DEBUG("[MOE-VRAM]  ACTIVE-LAYER layer=%d expert=%d -> temporary "
                  "upload (no LRU eviction)\n",
                  gpu_ctx->layer_idx, gpu_ctx->expert_id);
      } else {
        // Fallback: cold-miss synchronous upload to regular VRAM cache
        loc = gpu_ctx->topo->upload_expert(
            gpu_ctx->layer_idx, gpu_ctx->expert_id, gpu_ctx->slab_buffer,
            gpu_ctx->slab_capacity);
        vram_resident = loc && loc->device_ptr;
        loc_from_active_layer = false; // main cache, even though the
                                       // active-layer branches above ran first
        if (vram_resident) {
          // FIX ("[TIER-OVERLAP]" leak, see CLAUDE.md): third of
          // three synchronous-upload sites in this function.
          // Upgraded to active reclaim -- see the cold-batch
          // site's own comment above for why this is now safe.
          if (gpu_ctx->exp)
            ram_mgr.finish_sync_promotion(
                *gpu_ctx->exp, gpu_ctx->layer_idx, gpu_ctx->expert_id,
                &engine.layers_experts[gpu_ctx->layer_idx]);
          LOG_DEBUG("[MOE-VRAM]  COLD-GPU layer=%d expert=%d -> synchronous "
                    "H2D upload on decode-time "
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
    bool type_ok = gpu_type_supported(gate_meta.type) &&
                   gpu_type_supported(up_meta.type) &&
                   gpu_type_supported(down_meta.type);
    if (!type_ok) {
      perf().n_cpu_fallback_type_unsupported.fetch_add(
          1, std::memory_order_relaxed);
    } else if (!gpu_ctx || !gpu_ctx->slab_buffer) {
      perf().n_cpu_fallback_no_host_buffer.fetch_add(1,
                                                     std::memory_order_relaxed);
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
      perf().n_cpu_fallback_just_promoted.fetch_add(1,
                                                    std::memory_order_relaxed);
    } else {
      perf().n_cpu_fallback_upload_failed.fetch_add(1,
                                                    std::memory_order_relaxed);
    }
    return nullptr;
  }

  if (!gpu_type_supported(gate_meta.type) ||
      !gpu_type_supported(up_meta.type) ||
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
        why += std::string("gate=") + gpu_type_name(gate_meta.type) + " (" +
               gpu_unsupported_reason(gate_meta.type) + ") ";
      if (!gpu_type_supported(up_meta.type))
        why += std::string("up=") + gpu_type_name(up_meta.type) + " (" +
               gpu_unsupported_reason(up_meta.type) + ") ";
      if (!gpu_type_supported(down_meta.type))
        why += std::string("down=") + gpu_type_name(down_meta.type) + " (" +
               gpu_unsupported_reason(down_meta.type) + ") ";
      LOG_DEBUG("[MOE-GPU]   layer=%d expert=%d VRAM-resident but blocked: "
                "%s-- CPU dequant-matmul fallback\n",
                gpu_ctx ? gpu_ctx->layer_idx : -1,
                gpu_ctx ? gpu_ctx->expert_id : -1, why.c_str());
    }
    // Same underlying reason as the !vram_resident branch's type_ok
    // check above -- an already-resident expert (via some OTHER path,
    // e.g. reactive promotion) whose type still can't dispatch. Counted
    // into the SAME bucket since both mean "this expert's quant type
    // has no GPU kernel", just reached via a different code path.
    perf().n_cpu_fallback_type_unsupported.fetch_add(1,
                                                     std::memory_order_relaxed);
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
    bool held =
        gpu_ctx->topo->hold_for_compute(gpu_ctx->layer_idx, gpu_ctx->expert_id,
                                        loc->device_ptr, loc_from_active_layer);
    if (!held) {
      perf().n_hold_for_compute_fail.fetch_add(1, std::memory_order_relaxed);
      LOG_WARN("[MOE-VRAM]  layer=%d expert=%d: hold_for_compute failed for an "
               "apparently-"
               "resident slot -- identity changed underneath us; falling back "
               "to CPU for this "
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
static bool expert_ffn_gpu_issue(GpuExpertScratch &gpu_scratch,
                                 const ExpertLocation *loc, const float *input,
                                 const GGUFTensor &gate_meta,
                                 const GGUFTensor &up_meta, int S, int D,
                                 int moe_inter, GpuTopology *topo) {
  PERF_SCOPE(t_gpu_issue_total);
  // Real, confirmed bug (found 2026-09 chasing a real, fully deterministic
  // GPU-path NaN on a genuinely mixed per-expert model, real multi-GPU
  // config): this function launches the gate/up kernels via a bare
  // <<<>>> (every .cu launcher in this codebase does -- always targets
  // whatever the CURRENT device happens to be) but never pinned it itself
  // -- unlike its own sibling, expert_ffn_gpu_finish_stage1/stage2, which
  // BOTH explicitly re-pin with the exact comment "Each in-flight expert's
  // issue() may have run on a different vault GPU (multi-GPU config) and
  // left a different device 'current' on this thread -- re-pin explicitly
  // rather than trusting ambient state." That comment is a direct
  // admission the ambient device is NOT reliable across pending experts in
  // this dual-GPU config, and nothing upstream of THIS call (the one that
  // actually launches the kernels reading loc->device_ptr) ever set it --
  // a real, latent gap, kept fixed here regardless of the investigation
  // below. NOTE: real-hardware testing (2026-09) confirmed this fix alone
  // does NOT explain a separate, still-unresolved deterministic NaN on a
  // genuinely mixed per-expert model (layer=30 expert=113 on one real
  // config) -- weight bytes and activation were independently verified
  // byte-correct at every checkpoint (RAM, VRAM-pre-dispatch, isolated
  // kernel test with the real dumped activation) right up to the kernel
  // launch, and neither this fix nor a synchronous-activation-upload test
  // nor disabling both active-layer speculative-fill mechanisms changed
  // the outcome. Root cause of THAT bug is still open; this fix is kept
  // because it's independently correct, not because it's confirmed to be
  // the cause.
  cudaSetDevice(loc->gpu_id);
  // Slab layout mirrors the host copy: gate at offset 0, up at its
  // original file-offset delta from gate (see load_expert_slab).
  const uint8_t *base = loc->device_ptr;
  const uint8_t *gate_dev = base;
  const uint8_t *up_dev = base + (up_meta.offset - gate_meta.offset);

  // Raw-weight-bytes probe: distinguishes "the VRAM weight bytes for this
  // expert are genuinely never/wrongly written" from "written but the
  // matmul/SiLU pipeline zeroes them downstream". A plain cudaMemcpy with
  // no explicit stream (legacy default stream) implicitly synchronizes
  // with every prior stream's work first, so a post-copy all-zero read
  // here means the bytes really are zero in VRAM at this exact address --
  // not merely "not yet" written by some still-in-flight async upload.
  //
  // REVISED from a first cut that only sampled the first 64 bytes of each
  // tensor: the corrupted matmul output is the FULL 2048-element gate/up
  // vector coming back zero, which (for a ~2MB quantized tensor) requires
  // a much larger swath to actually be zero than 64 bytes out of ~2M could
  // ever confirm or rule out -- that first version could not distinguish
  // "genuinely all zero" from "zero happened to land in the first 64
  // bytes, real data follows". Now samples NUM_WINDOWS windows spread
  // evenly across each tensor's real extent (gate: [0, up_off_delta); up:
  // [up_off_delta, loc->size)) so a corrupted region anywhere in either
  // tensor is very likely to land in at least one window, without paying
  // for a full-tensor download every dispatch.
  if (trace_gate("VRAM-BYTE-PROBE")) {
    constexpr int NUM_WINDOWS = 16;
    constexpr size_t WIN = 64;
    size_t up_off = (size_t)(up_meta.offset - gate_meta.offset);
    size_t gate_extent = std::min(up_off, loc->size);
    size_t up_extent = (loc->size > up_off) ? (loc->size - up_off) : (size_t)0;
    uint8_t buf[WIN];
    int gate_zero_windows = 0, gate_windows = 0;
    int up_zero_windows = 0, up_windows = 0;
    for (int w = 0; w < NUM_WINDOWS && gate_extent > 0; ++w) {
      size_t off = (gate_extent > WIN)
                       ? (size_t)((double)w / NUM_WINDOWS * (gate_extent - WIN))
                       : 0;
      size_t n = std::min(WIN, gate_extent - off);
      if (n == 0)
        continue;
      cudaMemcpy(buf, gate_dev + off, n, cudaMemcpyDeviceToHost);
      ++gate_windows;
      bool all_zero = true;
      for (size_t i = 0; i < n; ++i)
        if (buf[i] != 0) {
          all_zero = false;
          break;
        }
      if (all_zero)
        ++gate_zero_windows;
    }
    for (int w = 0; w < NUM_WINDOWS && up_extent > 0; ++w) {
      size_t off = (up_extent > WIN)
                       ? (size_t)((double)w / NUM_WINDOWS * (up_extent - WIN))
                       : 0;
      size_t n = std::min(WIN, up_extent - off);
      if (n == 0)
        continue;
      cudaMemcpy(buf, up_dev + off, n, cudaMemcpyDeviceToHost);
      ++up_windows;
      bool all_zero = true;
      for (size_t i = 0; i < n; ++i)
        if (buf[i] != 0) {
          all_zero = false;
          break;
        }
      if (all_zero)
        ++up_zero_windows;
    }
    LOG_STREAM_VERBOSE() << "[VRAM-BYTE-PROBE l=" << loc->layer_idx
                         << " e=" << loc->expert_id
                         << "] device_ptr=" << (const void *)loc->device_ptr
                         << " size=" << loc->size
                         << " upload_ready=" << loc->upload_ready
                         << " gate_extent=" << gate_extent
                         << " up_extent=" << up_extent
                         << " gate_zero_windows=" << gate_zero_windows << "/"
                         << gate_windows
                         << " up_zero_windows=" << up_zero_windows << "/"
                         << up_windows
                         << " gate_type=" << gpu_type_name(gate_meta.type)
                         << " up_type=" << gpu_type_name(up_meta.type) << "\n";
  }

  // d-scale probe: the byte-window probe above only proves the tensor
  // isn't ALL zero -- it can't tell "real nonzero qs/sign bits sitting
  // next to a zero dequant scale" from genuinely-live data, because the
  // scale is a small fraction of each block's bytes. iq2_matmul.cu's
  // dequant_iq2_xxs_coop multiplies every one of a block's 256 elements
  // by a single shared per-block scale db = d * (...) * 0.25, where d
  // is that block's own leading fp16 header (2 of its 66 bytes). If d
  // is 0 for (nearly) every block of a tensor, EVERY dequantized value
  // is exactly 0.0 regardless of what the qs/sign bits look like --
  // exactly the "raw bytes look real but matmul output is all zero"
  // pattern the byte-window probe couldn't explain. Sampled block-
  // aligned (not byte-window-aligned) so every sample lands exactly on
  // a `d` header instead of an arbitrary offset into qs data.
  // Independently gated from VRAM-BYTE-PROBE above (own trace tag) even
  // though it shares that block's gate_extent/up_extent derivation --
  // both are cheap arithmetic, not worth threading a shared gate through.
  if (trace_gate("VRAM-DSCALE-PROBE") && (gate_meta.type == GGUFType::IQ2_XXS ||
                                          up_meta.type == GGUFType::IQ2_XXS)) {
    size_t up_off = (size_t)(up_meta.offset - gate_meta.offset);
    size_t gate_extent = std::min(up_off, loc->size);
    size_t up_extent = (loc->size > up_off) ? (loc->size - up_off) : (size_t)0;
    // Named IQ2_BLOCK_BYTES, not BLOCK_SIZE: <linux/fs.h> #defines a
    // bare BLOCK_SIZE macro ((1<<10)), pulled in transitively via
    // async_prefetcher.hpp -> io_uring_backend.hpp's real Linux
    // branch (#include <linux/io_uring.h>) -- invisible from the
    // Mac-side lint workflow, which always takes that header's
    // non-Linux stub branch instead. A local `constexpr size_t
    // BLOCK_SIZE` silently got rewritten to `constexpr size_t
    // (1<<10)` by the preprocessor, producing the confusing real
    // GCC error "expected primary-expression before 'constexpr'".
    constexpr size_t IQ2_BLOCK_BYTES =
        66; // sizeof(block_iq2_xxs_cu): 2(d) + 32*2(qs)
    constexpr int NUM_SAMPLES = 8;
    auto sample_d = [&](const uint8_t *dev, size_t extent, int &zero_d,
                        int &sampled, float &min_d, float &max_d) {
      size_t num_blocks = extent / IQ2_BLOCK_BYTES;
      if (num_blocks == 0)
        return;
      for (int i = 0; i < NUM_SAMPLES; ++i) {
        size_t block_idx =
            (num_blocks > 1)
                ? (size_t)((double)i / (NUM_SAMPLES - 1) * (num_blocks - 1))
                : 0;
        uint16_t d_raw = 0;
        cudaMemcpy(&d_raw, dev + block_idx * IQ2_BLOCK_BYTES, sizeof(d_raw),
                   cudaMemcpyDeviceToHost);
        float d = fp16_to_fp32(d_raw);
        ++sampled;
        if (d == 0.0f)
          ++zero_d;
        min_d = std::min(min_d, d);
        max_d = std::max(max_d, d);
      }
    };
    int gate_zero_d = 0, gate_d_sampled = 0, up_zero_d = 0, up_d_sampled = 0;
    float gate_min_d = 1e30f, gate_max_d = -1e30f, up_min_d = 1e30f,
          up_max_d = -1e30f;
    if (gate_meta.type == GGUFType::IQ2_XXS)
      sample_d(gate_dev, gate_extent, gate_zero_d, gate_d_sampled, gate_min_d,
               gate_max_d);
    if (up_meta.type == GGUFType::IQ2_XXS)
      sample_d(up_dev, up_extent, up_zero_d, up_d_sampled, up_min_d, up_max_d);
    LOG_STREAM_VERBOSE() << "[VRAM-DSCALE-PROBE l=" << loc->layer_idx
                         << " e=" << loc->expert_id
                         << "] gate_zero_d=" << gate_zero_d << "/"
                         << gate_d_sampled << " gate_d_range=[" << gate_min_d
                         << "," << gate_max_d << "]"
                         << " up_zero_d=" << up_zero_d << "/" << up_d_sampled
                         << " up_d_range=[" << up_min_d << "," << up_max_d
                         << "]\n";
  }

  // One-shot real-activation capture for the DeepSeek-V4-Flash garbled-
  // output investigation (see main_test.cpp's Test 11 header comment):
  // that test already proved expert 199's real weight bytes produce
  // NONZERO output on both CPU and GPU when fed a synthetic random `x`,
  // ruling out an unconditional GPU-kernel dequant bug -- but it can't
  // rule out a caching/pipeline bug that only manifests with the ACTUAL
  // activation vector at the real failing decode step. This pulls that
  // exact real `x` (the same buffer about to be uploaded as d_x below)
  // out to a flat binary file the moment it matches env-selected
  // (layer, expert) coordinates, so it can be replayed offline through
  // the same isolated cpu_quant_matmul/dense_matmul_host harness Test 11
  // already uses on the real dumped weight bytes. Inert unless both env
  // vars are set; fires at most once per process (an expert can be
  // dispatched many times across decode steps -- only the first hit is
  // captured, matching how the original all-zero trace was a single
  // observed instant, not a steady-state property to average over).
  {
    static std::atomic<bool> activation_dumped{false};
    if (!activation_dumped.load(std::memory_order_relaxed)) {
      const char *target_layer_env =
          std::getenv("MOESTREAM_DUMP_ACTIVATION_LAYER");
      const char *target_expert_env =
          std::getenv("MOESTREAM_DUMP_ACTIVATION_EXPERT");
      if (target_layer_env && target_expert_env &&
          loc->layer_idx == std::atoi(target_layer_env) &&
          loc->expert_id == std::atoi(target_expert_env)) {
        bool expected = false;
        if (activation_dumped.compare_exchange_strong(expected, true)) {
          const char *out_path_env =
              std::getenv("MOESTREAM_DUMP_ACTIVATION_OUT");
          const std::string out_path =
              out_path_env ? out_path_env : "/tmp/real_activation.bin";
          std::ofstream f(out_path, std::ios::binary);
          f.write(reinterpret_cast<const char *>(input),
                  (std::streamsize)((size_t)S * D * sizeof(float)));
          LOG_STREAM_VERBOSE()
              << "[ACTIVATION-DUMP l=" << loc->layer_idx
              << " e=" << loc->expert_id << "] wrote S=" << S << " D=" << D
              << " floats to " << out_path << "\n";
        }
      }
    }
  }

  // Triage aid for the "every layer shows post-SiLU all-zero" finding
  // (debug.log grep of [SILU-TRACE-GPU] range=[0, 0] matched at every
  // traced layer, not one suspect expert as originally hypothesized --
  // see conversation). Distinguishes "activation feeding this expert is
  // already degenerate before the matmul" (points upstream, to
  // attention/RMSNorm/embedding -- see the residual-norm divergence
  // memory note) from "real nonzero input still produces zero gate_buf"
  // (points at the matmul/weight-dispatch path itself). Cheap (S*D scan,
  // same VERBOSE gate as the existing SILU-TRACE-GPU it's meant to be
  // read alongside) so left unconditional rather than env-var-gated.
  if (trace_gate("INPUT-TRACE-GPU")) {
    float in_min = input[0], in_max = input[0];
    int in_zero = 0;
    for (int i = 0; i < S * D; ++i) {
      in_min = std::min(in_min, input[i]);
      in_max = std::max(in_max, input[i]);
      if (input[i] == 0.0f)
        ++in_zero;
    }
    LOG_STREAM_VERBOSE() << "[INPUT-TRACE-GPU l=" << loc->layer_idx
                         << " e=" << loc->expert_id << "] activation range=["
                         << in_min << ", " << in_max << "] zeros=" << in_zero
                         << "/" << (S * D) << "\n";
  }

  if (!gpu_scratch.ensure(S, D, moe_inter, loc->gpu_id))
    return false; // pins the correct vault GPU

  // 2026-08-29: cudaMemcpyAsync on compute_stream instead of a plain
  // blocking cudaMemcpy (legacy default/stream-0). Found while
  // investigating why this bucket grew after moving upload_expert_
  // active's fallback off the default stream (see active_layer_sync_
  // fallback_stream's own comment, gpu_topology.hpp): this call was
  // ALREADY exposed to the same trap (a plain cudaMemcpy with no stream
  // argument implicitly waits for EVERY other blocking-type stream on
  // the device to drain first) -- adding that new stream just gave it
  // one more thing to wait behind. No explicit cudaStreamSynchronize
  // needed here: the gate/up kernel launches immediately below are
  // still on stream 0 (gpu_matmul_dispatch's launchers all use a bare
  // <<<>>> launch, unchanged), and stream 0's own implicit-wait
  // semantics guarantee THEY won't start reading d_x until this async
  // copy on compute_stream has completed -- the exact same mechanism
  // that made the old blocking form correct is what makes this async
  // form correct too, just without paying the CPU-side wait here. The
  // caller's own existing cudaStreamSynchronize(0) batch sync (Pass A/B,
  // see t_gpu_batch_sync's own comment) is what actually confirms
  // completion, transitively covering compute_stream through that
  // ordering -- unchanged, no new sync point added. compute_stream
  // reused deliberately rather than a new dedicated stream, specifically
  // to NOT repeat the "added stream becomes a new thing stream-0 callers
  // wait behind" mistake this whole investigation started from --
  // compute_stream already existed and was already part of that wait
  // set (confirmed near-0% busy at every other call site that's
  // checked it), so this adds no NEW default-stream exposure anywhere
  // else.
  PERF_START(gexp_up1);
  cudaMemcpyAsync(
      gpu_scratch.d_x, input, (size_t)S * D * sizeof(float),
      cudaMemcpyHostToDevice,
      (topo && loc->gpu_id >= 0 && (size_t)loc->gpu_id < topo->devices.size())
          ? topo->devices[loc->gpu_id].compute_stream
          : nullptr);
  PERF_END(gexp_up1, t_expert_gpu_upload);

  gpu_matmul_dispatch(gate_meta.type, gpu_scratch.d_gate, gpu_scratch.d_x,
                      gate_dev, S, D, moe_inter);
  gpu_matmul_dispatch(up_meta.type, gpu_scratch.d_up, gpu_scratch.d_x, up_dev,
                      S, D, moe_inter);
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
static inline void swiglu_clamp_inplace(float *x, int n, float clamp) {
  if (clamp <= 0.0f)
    return;
  for (int i = 0; i < n; ++i)
    x[i] = std::max(-clamp, std::min(clamp, x[i]));
}

// ggml_swiglu_clamp's EXACT semantics (llama.cpp ggml-cpu ops.cpp), used by
// the reference for GLM5NEXT (and DEEPSEEK4) instead of this file's default
// silu→mul→clamp-product order:
//   gate = min(gate_raw, limit);  up = clamp(up_raw, -limit, limit);
//   out  = silu(gate) * up       (product itself NOT clamped)
// Gated on swiglu_clamp_fused() (dense_dispatch.hpp) at each call site --
// see that flag's own comment for why DeepSeek-V4 deliberately stays on the
// old order for now. limit<=0 reduces to plain silu(gate)*up (no clamp),
// matching swiglu_clamp_inplace's own "inactive" contract.
static inline void swiglu_clamp_fused_inplace(float *gate, const float *up,
                                              int n, float limit) {
  for (int i = 0; i < n; ++i) {
    float g = gate[i];
    float u = up[i];
    if (limit > 0.0f) {
      g = std::min(g, limit);
      u = std::max(-limit, std::min(limit, u));
    }
    gate[i] = g / (1.0f + std::exp(-g)) * u;
  }
}

// Phase B, batched (2026-08-29): SPLIT from the original single-function
// design specifically to cut cudaStreamSynchronize call COUNT, not to
// change what gets computed. Real nsys profiling of a whole request found
// cudaStreamSynchronize alone at 99,791 calls/21.5s/26.4% of all CUDA API
// time, while "0% stream busy" had already been independently confirmed 4
// separate times elsewhere in this codebase's history -- meaning each sync
// call was mostly waiting on NOTHING, pure fixed per-call overhead paid up
// to 2x PER EXPERT (one before gate/up download, one after down-proj) when
// Pass A had already issued every pending expert's gate/up kernels before
// Pass B ever runs. The caller (dispatch_expert_k's own Pass-B loop) now
// does ONE shared cudaStreamSynchronize per DISTINCT vault GPU involved in
// the whole batch (not one per expert), calling stage1 for every pending
// expert first (no sync inside), then the shared sync, then stage2.
// Correctness-preserving: identical math, identical per-expert data, just
// fewer, larger synchronization points instead of many small ones -- see
// dispatch_expert_k's own comment for the exact new call sequence and why
// grouping by loc->gpu_id (not a single unconditional sync) is required
// for correctness in a multi-vault-GPU config.
//
// Stage 1: download gate/up (assumes caller's shared sync already
// confirmed these kernels are done), SiLU+mul+clamp on CPU, upload the
// SwiGLU product, issue the down-proj kernel. Deliberately does NOT sync
// after issuing down-proj -- the caller's own second shared sync (after
// calling stage1 for every pending expert) covers that.
static void expert_ffn_gpu_finish_stage1(
    GpuExpertScratch &gpu_scratch, const ExpertLocation *loc, float *gate_buf,
    float *up_buf, const GGUFTensor &gate_meta, const GGUFTensor &down_meta,
    int layer_idx, int expert_id, int S, int D, int moe_inter,
    float swiglu_clamp, GpuTopology *topo) {
  PERF_SCOPE(t_gpu_finish_total);
  // Each in-flight expert's issue() may have run on a different vault GPU
  // (multi-GPU config) and left a different device "current" on this
  // thread -- re-pin explicitly rather than trusting ambient state.
  cudaSetDevice(loc->gpu_id);
  const uint8_t *down_dev =
      loc->device_ptr + (down_meta.offset - gate_meta.offset);

  // SiLU + elementwise multiply is cheap (moe_inter elements) —
  // done on CPU rather than adding a dedicated kernel for now.
  // GPU activation fast path (EngineConfig::expert_gpu_activation): compute
  // SwiGLU straight into d_prod and skip the download/CPU/upload round trip
  // entirely. Only taken when no host-side trace needs the raw gate/up values
  // -- those traces exist to debug this exact path, so they must still be able
  // to see it. check_finite's NaN/Inf guard is NOT skipped: the kernel counts
  // non-finite inputs and we read back 8 bytes instead of 2*S*moe_inter floats.
  bool act_done = false;
  const bool act_gpu =
      expert_gpu_activation() && !trace_gate("SILU-TRACE-GPU") &&
      !trace_gate("SWIGLU-TRACE-GPU") && !trace_gate("EXPERT-GPU") &&
      !trace_gate("INPUT-TRACE-GPU-PAIRED");
  if (act_gpu) {
    int *d_bad = gpu_scratch.ensure_bad_counter();
    if (d_bad) {
      PERF_START(gexp_act);
      launch_expert_swiglu(gpu_scratch.d_prod, gpu_scratch.d_gate,
                           gpu_scratch.d_up, S * moe_inter, swiglu_clamp,
                           swiglu_clamp_fused() ? 1 : 0, d_bad, 0);
      // Sampled readback. A synchronous 8-byte copy PER EXPERT is a full
      // device sync per expert -- measured 12.649s across 12,827 calls, i.e.
      // as expensive as the download this path exists to remove. The counter
      // accumulates on-device and is sampled instead; NaN propagates through
      // the residual stream, so detecting it within a layer is sufficient.
      static thread_local int act_sample = 0;
      if ((++act_sample & 511) == 0) {
        int bad[2] = {0, 0};
        cudaMemcpy(bad, d_bad, 2 * sizeof(int), cudaMemcpyDeviceToHost);
        if (bad[0] || bad[1]) {
          LOG_ERROR(
              "[EXPERT-GPU] gate/up non-finite (device-side, cumulative): "
              "%d NaN, %d Inf by layer=%d expert=%d\n",
              bad[0], bad[1], layer_idx, expert_id);
          cudaMemset(d_bad, 0, 2 * sizeof(int));
        }
      }
      PERF_END(gexp_act, t_expert_gpu_activation);
      perf().n_expert_gpu_activation.fetch_add(1, std::memory_order_relaxed);
      act_done = true; // product already in d_prod; no download/upload needed
    }
  }

  if (!act_done) {
    PERF_START(gexp_dn1);
    cudaMemcpy(gate_buf, gpu_scratch.d_gate,
               (size_t)S * moe_inter * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(up_buf, gpu_scratch.d_up, (size_t)S * moe_inter * sizeof(float),
               cudaMemcpyDeviceToHost);
    PERF_END(gexp_dn1, t_gpu_download);
    check_finite("gate_buf (post gate+up matmul)", layer_idx, expert_id,
                 gate_buf, (size_t)S * moe_inter);
    check_finite("up_buf   (post gate+up matmul)", layer_idx, expert_id, up_buf,
                 (size_t)S * moe_inter);

    // Mirrors expert_ffn_cpu's own [SILU-TRACE] instrumentation -- this GPU
    // path runs the IDENTICAL silu_inplace/elem_mul/swiglu_clamp_inplace CPU
    // functions on the downloaded gate_buf/up_buf (see this function's own
    // header comment: no dedicated CUDA activation kernel exists), so the
    // same SiLU/FTZ-saturation-vs-upstream-corruption question applies here
    // too. Needed because the CPU-path trace alone can't distinguish "this
    // dispatch went through the GPU path instead" from "the CPU path itself
    // never zeroes" -- a zero-`down_buf` call with no CPU SILU-TRACE nearby
    // may simply have been GPU-dispatched and thus invisible to that trace.
    float pre_min = 0.0f, pre_max = 0.0f;
    bool trace_silu_gpu = trace_gate("SILU-TRACE-GPU");
    if (trace_silu_gpu) {
      pre_min = gate_buf[0];
      pre_max = gate_buf[0];
      for (int i = 1; i < S * moe_inter; ++i) {
        pre_min = std::min(pre_min, gate_buf[i]);
        pre_max = std::max(pre_max, gate_buf[i]);
      }
    }
    // Downloads gpu_scratch.d_x -- the EXACT device buffer this call's own
    // gate/up matmul just read -- rather than relying on issue()'s own
    // separate [INPUT-TRACE-GPU] log line to correlate by (layer, expert):
    // Pass A issues every in-flight expert's gate/up kernel before Pass B
    // (this function) downloads/finishes any of them, so issue()'s trace
    // lines for a whole batch print consecutively, followed later by this
    // function's SILU-TRACE-GPU lines for the same batch in a possibly
    // different order -- adjacent-line greps (or even matching (l,e) keys,
    // if the same expert is re-dispatched across tokens) can pair the wrong
    // occurrences. Logged in the SAME statement grouping as SILU-TRACE-GPU
    // below so the two are always about the literal same call, no
    // correlation-by-eye required.
    if (trace_gate("INPUT-TRACE-GPU-PAIRED")) {
      std::vector<float> x_check((size_t)S * D);
      cudaMemcpy(x_check.data(), gpu_scratch.d_x, (size_t)S * D * sizeof(float),
                 cudaMemcpyDeviceToHost);
      float x_min = x_check[0], x_max = x_check[0];
      int x_zero = 0;
      for (size_t i = 0; i < x_check.size(); ++i) {
        x_min = std::min(x_min, x_check[i]);
        x_max = std::max(x_max, x_check[i]);
        if (x_check[i] == 0.0f)
          ++x_zero;
      }
      LOG_STREAM_VERBOSE() << "[INPUT-TRACE-GPU-PAIRED l=" << layer_idx
                           << " e=" << expert_id << "] d_x range=[" << x_min
                           << ", " << x_max << "] zeros=" << x_zero << "/"
                           << x_check.size() << "\n";
    }
    // Tightly wraps only the three real math calls (not the trace-only
    // blocks in between, which are no-ops off the VERBOSE path and would
    // otherwise pollute this measurement when tracing happens to be on) --
    // see t_expert_silu_activation's own comment (PerfCounters) for why
    // this was added: three prior investigation rounds instrumented
    // GpuTopology/RAMSafetyManager functions looking for the gap between
    // expert= and every other sub-timer and all came back near-zero; this
    // is the first timer on this specific CPU-side activation path.
    PERF_START(gexp_silu);
    if (swiglu_clamp_fused()) {
      // GLM5Next: ggml_swiglu_clamp semantics (fused min/clamp/silu/mul) --
      // replaces the silu→mul→clamp-product sequence below entirely.
      swiglu_clamp_fused_inplace(gate_buf, up_buf, S * moe_inter, swiglu_clamp);
      PERF_END(gexp_silu, t_expert_silu_activation);
    } else {
      silu_inplace(gate_buf, S * moe_inter);
      PERF_END(gexp_silu, t_expert_silu_activation);
      if (trace_silu_gpu) {
        int n_zero = 0;
        for (int i = 0; i < S * moe_inter; ++i)
          if (gate_buf[i] == 0.0f)
            ++n_zero;
        LOG_STREAM_VERBOSE()
            << "[SILU-TRACE-GPU l=" << layer_idx << " e=" << expert_id
            << "] pre-SiLU gate range=[" << pre_min << ", " << pre_max
            << "] post-SiLU zeros=" << n_zero << "/" << (S * moe_inter) << "\n";
      }
      PERF_START(gexp_mulclamp);
      elem_mul(gate_buf, gate_buf, up_buf, S * moe_inter);
      swiglu_clamp_inplace(gate_buf, S * moe_inter, swiglu_clamp);
      PERF_END(gexp_mulclamp, t_expert_silu_activation);
    }
    if (trace_gate("SWIGLU-TRACE-GPU")) {
      int n_zero = 0;
      for (int i = 0; i < S * moe_inter; ++i)
        if (gate_buf[i] == 0.0f)
          ++n_zero;
      LOG_STREAM_VERBOSE() << "[SWIGLU-TRACE-GPU l=" << layer_idx
                           << " e=" << expert_id
                           << "] post-mul+clamp product zeros=" << n_zero << "/"
                           << (S * moe_inter) << "\n";
    }

    // 2026-08-29: same conversion as gexp_up1 above (see its own comment
    // for the full mechanism/safety argument) -- cudaMemcpyAsync on
    // compute_stream instead of a plain blocking cudaMemcpy. Safe here for
    // an even more direct reason: the down-proj kernel launch immediately
    // below is ALREADY documented issue-only/no-sync (still on stream 0),
    // so it already relied on stream 0's implicit wait-for-others
    // semantics to be correct before this change -- this just stops that
    // wait from ALSO blocking the CPU thread on the upload call itself.
    // The product upload belongs INSIDE this guard: on the GPU-activation path
    // the kernel has already written d_prod, and re-uploading gate_buf here
    // would overwrite it with stale host data. That was a real bug -- it
    // produced garbage output while every NaN check still passed, because the
    // data was finite, just wrong.
    PERF_START(gexp_up2);
    cudaMemcpyAsync(
        gpu_scratch.d_prod, gate_buf, (size_t)S * moe_inter * sizeof(float),
        cudaMemcpyHostToDevice,
        (topo && loc->gpu_id >= 0 && (size_t)loc->gpu_id < topo->devices.size())
            ? topo->devices[loc->gpu_id].compute_stream
            : nullptr);
    PERF_END(gexp_up2, t_expert_gpu_upload);
  } // !act_done

  // Issue-only -- no sync here. gexp_cmp2/t_gpu_compute previously wrapped
  // the sync too; now it measures issue time alone (real, but usually
  // negligible: an async kernel launch just enqueues work). The GPU
  // execution + wait time this used to include now shows up in the
  // caller's own shared t_gpu_batch_sync instead -- the two together
  // still reconcile to the same real work, just fewer, larger buckets.
  PERF_START(gexp_cmp2);
  gpu_matmul_dispatch(down_meta.type, gpu_scratch.d_down, gpu_scratch.d_prod,
                      down_dev, S, moe_inter, D);
  PERF_END(gexp_cmp2, t_gpu_compute);
}

// Stage 2: assumes the caller's second shared sync (after stage1 ran for
// every pending expert in this batch) already confirmed every down-proj
// kernel -- the last kernel that reads/writes this expert's slab -- has
// actually finished executing, not just that its launch completed. Safe to
// release the temporary hold now, exactly as in the original single-
// function design (releasing any earlier would reopen the exact eviction-
// during-compute race the hold exists to prevent -- that invariant is
// unchanged, just now satisfied by the caller's batched sync instead of a
// per-call one).
static void expert_ffn_gpu_finish_stage2(const ExpertLocation *loc,
                                         float *output, int layer_idx,
                                         int expert_id, int S, int D,
                                         GpuTopology *topo, bool needs_release,
                                         bool loc_from_active_layer,
                                         GpuExpertScratch &gpu_scratch) {
  cudaSetDevice(loc->gpu_id);
  if (needs_release && topo)
    topo->release_hold(layer_idx, expert_id, loc->device_ptr,
                       loc_from_active_layer);

  PERF_START(gexp_dn2);
  cudaMemcpy(output, gpu_scratch.d_down, (size_t)S * D * sizeof(float),
             cudaMemcpyDeviceToHost);
  PERF_END(gexp_dn2, t_gpu_download);
  check_finite("output   (post down matmul)", layer_idx, expert_id, output,
               (size_t)S * D);
  perf().n_expert_gpu_calls.fetch_add(1, std::memory_order_relaxed);
}

// CPU fallback path: per-projection dequant by the projection's OWN type, so
// a mixed Q2_K_XL slab (gate=Q2_K / up=Q5_K / down=Q6_K) routes each weight
// correctly. Correct for F32/F16/BF16/Q8_0/Q4_0; K-quants/IQ2 dequant is
// pending transcription (dequant_row zeroes them loudly, never silent).
static void expert_ffn_cpu(const float *input, float *output,
                           const uint8_t *gate_data,
                           const GGUFTensor &gate_meta, const uint8_t *up_data,
                           const GGUFTensor &up_meta, const uint8_t *down_data,
                           const GGUFTensor &down_meta, float *gate_buf,
                           float *up_buf, int S, int D, int moe_inter,
                           float swiglu_clamp, int layer_idx, int expert_id) {
  PERF_START(cexp);
  cpu_quant_matmul(gate_meta.type, gate_data, input, gate_buf, S, D, moe_inter);
  cpu_quant_matmul(up_meta.type, up_data, input, up_buf, S, D, moe_inter);

  // Pre-SiLU gate_buf extremes + post-SiLU exact-zero count, VERBOSE-gated:
  // distinguishes a legitimate SiLU/FTZ saturation (x/(1+exp(-x)) flushing
  // to exactly 0.0f for a sufficiently negative pre-activation -- plausible
  // now that ~100+ magnitude hidden states are confirmed legitimate model
  // behavior, not corruption, per the attention-magnitude investigation)
  // from a real bug upstream of this point (e.g. gate_buf already
  // zero/garbage before SiLU ever runs). Only computed under VERBOSE,
  // matching every other high-volume per-expert trace in this file.
  float pre_min = 0.0f, pre_max = 0.0f;
  bool trace_silu = trace_gate("SILU-TRACE");
  if (trace_silu) {
    pre_min = gate_buf[0];
    pre_max = gate_buf[0];
    for (int i = 1; i < S * moe_inter; ++i) {
      pre_min = std::min(pre_min, gate_buf[i]);
      pre_max = std::max(pre_max, gate_buf[i]);
    }
  }
  if (swiglu_clamp_fused()) {
    // GLM5Next: ggml_swiglu_clamp semantics (fused min/clamp/silu/mul) --
    // replaces the silu→mul→clamp-product sequence below entirely.
    swiglu_clamp_fused_inplace(gate_buf, up_buf, S * moe_inter, swiglu_clamp);
  } else {
    silu_inplace(gate_buf, S * moe_inter);
    if (trace_silu) {
      int n_zero = 0;
      for (int i = 0; i < S * moe_inter; ++i)
        if (gate_buf[i] == 0.0f)
          ++n_zero;
      LOG_STREAM_VERBOSE() << "[SILU-TRACE l=" << layer_idx
                           << " e=" << expert_id << "] pre-SiLU gate range=["
                           << pre_min << ", " << pre_max
                           << "] post-SiLU zeros=" << n_zero << "/"
                           << (S * moe_inter) << "\n";
    }

    elem_mul(gate_buf, gate_buf, up_buf, S * moe_inter);
    swiglu_clamp_inplace(gate_buf, S * moe_inter, swiglu_clamp);
  }
  if (trace_gate("SWIGLU-TRACE")) {
    int n_zero = 0;
    for (int i = 0; i < S * moe_inter; ++i)
      if (gate_buf[i] == 0.0f)
        ++n_zero;
    LOG_STREAM_VERBOSE() << "[SWIGLU-TRACE l=" << layer_idx
                         << " e=" << expert_id
                         << "] post-mul+clamp product zeros=" << n_zero << "/"
                         << (S * moe_inter) << "\n";
  }
  cpu_quant_matmul(down_meta.type, down_data, gate_buf, output, S, moe_inter,
                   D);
  PERF_END(cexp, t_cpu_expert);
  perf().n_expert_cpu_calls.fetch_add(1, std::memory_order_relaxed);
}

// Qwen4Exp CPU-affinity fast path for the quantized expert projections.
// Llama.cpp's CPU MUL_MAT_ID path groups the selected matrices, quantizes the
// shared gate/up activation once, and uses ggml's quantized dot kernels rather
// than dequantizing every output row. Qwen uses IQ1_M or IQ2_XXS gate/up and
// IQ4_NL down in this checkpoint. Keep this helper family- and config-gated so
// mixed-quant and non-Qwen models retain the established fallback.
static bool expert_ffn_cpu_qwen_grouped_quantized(
    const float *input, float *output,
    const std::vector<MoEExpertBlock *> &experts, const float *weights, int D,
    int moe_inter, float swiglu_clamp, int layer_idx) {
  const int n = static_cast<int>(experts.size());
  if (!input || !output || !weights || n <= 0 || D <= 0 || moe_inter <= 0)
    return false;
  std::vector<const uint8_t *> gate_w((size_t)n);
  std::vector<const uint8_t *> up_w((size_t)n);
  std::vector<const uint8_t *> down_w((size_t)n);
  std::vector<float *> gate((size_t)n);
  std::vector<float *> up((size_t)n);
  std::vector<float *> down((size_t)n);
  std::vector<float> gate_storage((size_t)n * moe_inter);
  std::vector<float> up_storage((size_t)n * moe_inter);
  std::vector<float> down_storage((size_t)n * D);
  GGUFType gate_type = GGUFType::F32;
  GGUFType up_type = GGUFType::F32;
  GGUFType down_type = GGUFType::F32;
  bool types_initialized = false;
  for (int i = 0; i < n; ++i) {
    MoEExpertBlock *e = experts[(size_t)i];
    if (!e || e->vram_shadow || !e->gate_proj.host_ptr ||
        !e->up_proj.host_ptr || !e->down_proj.host_ptr)
      return false;
    if (!types_initialized) {
      gate_type = e->gate_proj.meta.type;
      up_type = e->up_proj.meta.type;
      down_type = e->down_proj.meta.type;
      types_initialized = true;
    }
    if (e->gate_proj.meta.type != gate_type ||
        e->up_proj.meta.type != up_type || e->down_proj.meta.type != down_type)
      return false;
    gate_w[(size_t)i] = e->gate_proj.host_ptr;
    up_w[(size_t)i] = e->up_proj.host_ptr;
    down_w[(size_t)i] = e->down_proj.host_ptr;
    gate[(size_t)i] = gate_storage.data() + (size_t)i * moe_inter;
    up[(size_t)i] = up_storage.data() + (size_t)i * moe_inter;
    down[(size_t)i] = down_storage.data() + (size_t)i * D;
  }

  PERF_START(qwen_cpu_group);
  // Prefer the persistent GGML CPU MUL_MAT_ID graph.  It groups all selected
  // experts and all three projections into one CPU backend submission, which
  // is the important difference from the older per-projection OpenMP helper
  // below.  Keep the old implementation as a correctness-preserving
  // fallback for unsupported/mixed tensor layouts or non-GGML builds.
  // The persistent graph currently models plain SwiGLU.  Do not route a
  // clamped Qwen variant through it until its clamp semantics are represented
  // in the graph; the established CPU path remains the safe fallback.
  static const int cpu_graph_mode = [] {
    const char *env = std::getenv("MOESTREAM_QWEN_CPU_GRAPH");
    if (env && env[0] == '0')
      return 0; // force the established path
    if (env && env[0] == '2')
      return 2; // validate, then use established path
    return 1;   // graph, unless it cannot represent it
  }();
  std::vector<float> graph_reference;
  std::vector<float> output_before;
  if (cpu_graph_mode == 2)
    output_before.assign(output, output + D);
  if (cpu_graph_mode != 0 && GgmlQwenCpuMoeRunner::compiled_in() &&
      swiglu_clamp <= 0.0f) {
    std::vector<GgmlQwenCpuExpertRef> refs((size_t)n);
    for (int i = 0; i < n; ++i) {
      MoEExpertBlock *e = experts[(size_t)i];
      refs[(size_t)i] = {e->gate_proj.host_ptr,
                         e->up_proj.host_ptr,
                         e->down_proj.host_ptr,
                         e->gate_proj.meta.size_bytes,
                         e->up_proj.meta.size_bytes,
                         e->down_proj.meta.size_bytes,
                         static_cast<uint32_t>(e->gate_proj.meta.type),
                         static_cast<uint32_t>(e->up_proj.meta.type),
                         static_cast<uint32_t>(e->down_proj.meta.type)};
    }
    std::vector<float> graph_output((size_t)D, 0.0f);
    if (GgmlQwenCpuMoeRunner::instance().run(input, graph_output.data(),
                                             refs.data(), n, weights, D,
                                             moe_inter, n)) {
      static std::atomic<bool> logged_cpu_graph{false};
      if (!logged_cpu_graph.exchange(true, std::memory_order_relaxed))
        LOG_STREAM_INFO()
            << "[CPU-MOE] Qwen persistent GGML MUL_MAT_ID CPU graph active "
               "(topk="
            << n << ", threads=" << omp_get_max_threads() << ")\n";
      if (cpu_graph_mode == 1) {
        for (int d = 0; d < D; ++d)
          output[d] += graph_output[(size_t)d];
        PERF_END(qwen_cpu_group, t_cpu_expert);
        perf().n_expert_cpu_calls.fetch_add((uint64_t)n,
                                            std::memory_order_relaxed);
        (void)layer_idx;
        return true;
      }
      graph_reference = std::move(graph_output);
    }
  }

  auto grouped_gate_up = [&](float *const *dst, const uint8_t *const *weights,
                             GGUFType type) {
    switch (type) {
    case GGUFType::IQ1_M:
      return matmul_iq1_m_ggml_q8k_grouped(dst, input, weights, n, 1, D,
                                           moe_inter);
    case GGUFType::IQ2_XXS:
      // Qwen's grouped CPU path uses GGML's production kernel directly,
      // matching Llama.cpp's MUL_MAT_ID implementation.  Keep the existing
      // AVX2-prototype gate for all non-Qwen callers; this switch is reached
      // only by the Qwen4Exp-specific helper below.
      return matmul_iq2_xxs_ggml_q8k_grouped(dst, input, weights, n, 1, D,
                                             moe_inter);
    case GGUFType::IQ3_XXS:
      return matmul_iq3_xxs_ggml_q8k_grouped(dst, input, weights, n, 1, D,
                                             moe_inter);
    default:
      return false;
    }
  };
  if (!grouped_gate_up(gate.data(), gate_w.data(), gate_type) ||
      !grouped_gate_up(up.data(), up_w.data(), up_type)) {
    PERF_END(qwen_cpu_group, t_cpu_expert);
    return false;
  }
  for (int i = 0; i < n; ++i) {
    if (swiglu_clamp_fused()) {
      swiglu_clamp_fused_inplace(gate[(size_t)i], up[(size_t)i], moe_inter,
                                 swiglu_clamp);
    } else {
      silu_inplace(gate[(size_t)i], moe_inter);
      elem_mul(gate[(size_t)i], gate[(size_t)i], up[(size_t)i], moe_inter);
      swiglu_clamp_inplace(gate[(size_t)i], moe_inter, swiglu_clamp);
    }
  }
  bool down_grouped = false;
  // The IQ4_NL q8_0 kernel is kept behind an explicit diagnostic switch until
  // its numerical output is checked against cpu_quant_matmul on this model's
  // exact GGUF rows.  The validated default remains the old down projection;
  // gate/up grouping is independent and remains enabled below.
  static const bool enable_grouped_iq4_nl = [] {
    const char *env = std::getenv("MOESTREAM_QWEN4EXP_GROUPED_IQ4_NL");
    return env && env[0] == '1';
  }();
  if (enable_grouped_iq4_nl && down_type == GGUFType::IQ4_NL) {
    std::vector<const float *> act((size_t)n);
    for (int i = 0; i < n; ++i)
      act[(size_t)i] = gate[(size_t)i];
    down_grouped = matmul_iq4_nl_ggml_q8_0_grouped_inputs(
        down.data(), act.data(), down_w.data(), n, 1, moe_inter, D);
  }
  if (!down_grouped) {
    for (int i = 0; i < n; ++i) {
      if (!cpu_quant_matmul(down_type, down_w[(size_t)i], gate[(size_t)i],
                            down[(size_t)i], 1, moe_inter, D)) {
        PERF_END(qwen_cpu_group, t_cpu_expert);
        return false;
      }
    }
  }
  for (int d = 0; d < D; ++d) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i)
      sum += weights[i] * down[(size_t)i][d];
    output[d] += sum;
  }
  if (!graph_reference.empty()) {
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    for (int d = 0; d < D; ++d) {
      const float legacy = output[d] - output_before[(size_t)d];
      const float graph = graph_reference[(size_t)d];
      max_abs = std::max(max_abs, std::abs(legacy - graph));
      max_rel = std::max(max_rel, std::abs(legacy - graph) /
                                      std::max(1.0f, std::abs(legacy)));
    }
    static std::atomic<bool> logged_graph_compare{false};
    if (!logged_graph_compare.exchange(true, std::memory_order_relaxed))
      LOG_STREAM_INFO()
          << "[CPU-MOE-COMPARE] Qwen persistent GGML graph vs established "
             "CPU path: layer="
          << layer_idx << " topk=" << n << " max_abs=" << max_abs
          << " max_rel=" << max_rel << " (graph result not used)\n";
  }
  PERF_END(qwen_cpu_group, t_cpu_expert);
  perf().n_expert_cpu_calls.fetch_add((uint64_t)n, std::memory_order_relaxed);
  (void)layer_idx;
  return true;
}

// Llama.cpp-style source view for a packed Qwen4Exp MoE layer. The source is
// the already-populated DDR RAM arena, not the GGUF file mmap. Each expert is
// one fixed-stride [gate|up|down] slab, which GGML MUL_MAT_ID can index via
// tensor->nb[2] without copying selected weights for the token.
struct QwenRamCpuMoeLayer {
  GgmlQwenCpuExpertRef meta;
  const void *gate_plane = nullptr;
  const void *up_plane = nullptr;
  const void *down_plane = nullptr;
  size_t expert_stride = 0;
};

static bool qwen4exp_ram_cpu_moe_layer(const StreamingEngine &engine,
                                       int layer_idx, int n_experts,
                                       QwenRamCpuMoeLayer &out) {
  if (layer_idx < 0 || n_experts <= 0 || engine.layer_is_unpacked(layer_idx) ||
      !engine.graph_ram_layer_ready(layer_idx))
    return false;

  const auto *arena = engine.graph_ram_arena(layer_idx);
  if (!arena || arena->n_experts != n_experts || !arena->base ||
      arena->stride == 0 || arena->gate_bytes == 0 || arena->up_bytes == 0 ||
      arena->down_bytes == 0)
    return false;

  const auto &slots = engine.layers_experts[(size_t)layer_idx];
  if (slots.size() < static_cast<size_t>(n_experts))
    return false;
  const MoEExpertBlock &first = slots[0];
  if (!first.slab_buffer || first.slab_capacity < arena->stride ||
      !first.gate_proj.host_ptr || !first.up_proj.host_ptr ||
      !first.down_proj.host_ptr)
    return false;
  for (int expert = 0; expert < n_experts; ++expert) {
    const MoEExpertBlock &slot = slots[(size_t)expert];
    if (slot.expert_id != expert || !slot.slab_from_shared_arena ||
        slot.slab_buffer != arena->base + (size_t)expert * arena->stride ||
        slot.slab_capacity < arena->stride || slot.ram_loading ||
        !slot.gate_proj.host_ptr || !slot.up_proj.host_ptr ||
        !slot.down_proj.host_ptr)
      return false;
  }

  out.meta = {arena->base,
              arena->base + arena->gate_bytes,
              arena->base + arena->gate_bytes + arena->up_bytes,
              arena->gate_bytes,
              arena->up_bytes,
              arena->down_bytes,
              static_cast<uint32_t>(first.gate_proj.meta.type),
              static_cast<uint32_t>(first.up_proj.meta.type),
              static_cast<uint32_t>(first.down_proj.meta.type)};
  out.gate_plane = arena->base;
  out.up_plane = arena->base + arena->gate_bytes;
  out.down_plane = arena->base + arena->gate_bytes + arena->up_bytes;
  out.expert_stride = arena->stride;
  return true;
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
static void apply_gating_activation(GatingFunc func, float *row, int n) {
  switch (func) {
  case GatingFunc::Sigmoid:
    for (int e = 0; e < n; ++e)
      row[e] = 1.0f / (1.0f + std::exp(-row[e]));
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
      float softplus =
          (x > 0.0f) ? (x + std::log1p(std::exp(-x))) : std::log1p(std::exp(x));
      row[e] = std::sqrt(softplus);
    }
    break;
  case GatingFunc::Softmax:
  default: {
    float max_l = row[0];
    for (int e = 1; e < n; ++e)
      max_l = std::max(max_l, row[e]);
    float sum = 0.0f;
    for (int e = 0; e < n; ++e) {
      row[e] = std::exp(row[e] - max_l);
      sum += row[e];
    }
    float inv = 1.0f / sum;
    for (int e = 0; e < n; ++e)
      row[e] *= inv;
    break;
  }
  }
}

// Keep the routed-pair graph at one high-water shape when requested. This
// avoids rebuilding a graph merely because a layer's routing histogram is
// slightly different. Zero preserves the existing grow-to-observed behavior.
static int ggml_fixed_chunk_pair_capacity() {
  static const int value = [] {
    const char *env = std::getenv("MOESTREAM_GGML_FIXED_PAIR_CAPACITY");
    if (!env || !env[0])
      return 0;
    const int requested = std::atoi(env);
    if (requested <= 0)
      return 0;
    return ((requested + 255) / 256) * 256;
  }();
  return value;
}

// The reference GGML graph keeps the expert axis fixed at the model's full
// expert count.  The compact MoE adapter normally rounds the routed set to a
// 64-expert high-water mark, which still causes a graph/allocator rebuild
// whenever a layer exposes a new high-water mark.  Opt-in fixed slots let us
// measure the same stable graph topology without changing the default memory
// budget.  Unused slots are never addressed by ids and are initialized by the
// existing quantized-tail clear path.
static int ggml_fixed_expert_slots() {
  static const int value = [] {
    const char *env = std::getenv("MOESTREAM_GGML_FIXED_EXPERT_SLOTS");
    if (!env || !env[0])
      return 0;
    const int requested = std::atoi(env);
    if (requested <= 0)
      return 0;
    return ((requested + 63) / 64) * 64;
  }();
  return value;
}

// Qwen's `ram_preload_all_experts` is a source-ownership contract, not merely
// a hint to improve the file-backed path.  Once enabled, every Qwen expert
// miss must come from the already-populated RAM tier.  In particular, do not
// silently fall back to the GGUF mmap when an optional grouped/chunked path
// rejects a layer: mmap is file-backed model storage (and can fault from the
// SSD), whereas the resident DDR tier is the source selected by the config.
// This is deliberately Qwen-only so GLM/DeepSeek/other model families retain
// their existing mmap fallback behavior.
static bool qwen_ddr_only_expert_source(const StreamingEngine &engine) {
  return engine.family_ops.family == ModelFamily::Qwen4Exp &&
         engine.qwen_ram_preload_all_experts;
}

// Diagnostic A/B for the Qwen CPU graph.  The normal DDR path presents the
// already-loaded [gate|up|down] expert slabs directly through a strided
// MUL_MAT_ID view.  Bee's graph sees three independently contiguous expert
// tensors.  When enabled, copy only the exact routed experts from the same
// DDR slabs into the runner's compact contiguous storage before computing.
// This is deliberately Qwen-only and opt-in: it changes neither the default
// source-ownership contract nor any other model family's CPU path.
static bool qwen_cpu_compact_graph_enabled() {
  static const bool enabled = [] {
    const char *env = std::getenv("MOESTREAM_QWEN_CPU_COMPACT_GRAPH");
    return env && env[0] == '1';
  }();
  return enabled;
}

// Qwen-only A/B for the Bee-style post-down partition.  The CPU graph still
// owns all three quantized projections and the DDR strided arena; this switch
// only decides whether the CPU also performs router weighting/accumulation.
// Keep it enabled by default when qwen4exp_cpu_gpu_overlap is enabled, while
// retaining an environment escape hatch for numerical/performance bisects.
static bool qwen_cpu_gpu_combine_enabled() {
  static const bool enabled = [] {
    const char *env = std::getenv("MOESTREAM_QWEN_CPU_GPU_COMBINE");
    // This path is retained as an explicit experiment.  It adds a separate
    // CUDA graph launch for each CPU-affinity layer and, until the weighting
    // is fused into the shared graph, the extra launch/shape conversion is
    // slower than the established CPU weighted-result path on Qwen decode.
    return env && env[0] == '1';
  }();
  return enabled;
}

// The Qwen decode fast path normally uses the grouped GGML runner, so it does
// not call RAMSafetyManager::get_experts_batch().  Record the first actual
// DDR-backed fetch boundary from both paths through this one helper.  The
// timestamp is consumed once per predicted target layer; active-layer/D2D
// reuse deliberately does not count because it never starts a DDR fetch.
static void qwen4exp_record_pilot_fetch_start(int layer_idx) {
  if (layer_idx < 0 || layer_idx >= PerfCounters::kMaxTrackedLayersForMiss)
    return;
  const int64_t fetch_start_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count();
  const int64_t router_ready_ns =
      perf().qwen_pilot_router_ready_ns[layer_idx].exchange(
          0, std::memory_order_acq_rel);
  const int64_t prediction_ready_ns =
      perf().qwen_pilot_prediction_for_fetch_ns[layer_idx].exchange(
          0, std::memory_order_acq_rel);
  if (router_ready_ns > 0 && fetch_start_ns >= router_ready_ns) {
    PerfCounters::atomic_add(
        perf().t_qwen_pilot_router_to_fetch,
        (double)(fetch_start_ns - router_ready_ns) * 1e-9);
    perf().n_qwen_pilot_router_to_fetch.fetch_add(1,
                                                  std::memory_order_relaxed);
  }
  if (prediction_ready_ns > 0 && fetch_start_ns >= prediction_ready_ns) {
    PerfCounters::atomic_add(
        perf().t_qwen_pilot_ready_to_fetch,
        (double)(fetch_start_ns - prediction_ready_ns) * 1e-9);
    perf().n_qwen_pilot_ready_to_fetch.fetch_add(1,
                                                 std::memory_order_relaxed);
  }
}

// ============================================================
// ggml MUL_MAT_ID path for one prefill layer's routed experts (opt-in via
// moe_ggml_mul_mat_id). Returns true only if the ENTIRE layer's routed
// contribution has been computed and added into `output`; on false the caller
// runs its own per-expert path unchanged, so every rejection below is a
// fallback, never a dropped expert.
//
// Structurally this replaces Phase 3b's per-expert-bucket loop with three
// GGML_OP_MUL_MAT_ID nodes in one backend graph. The token-to-expert grouping
// Phase 3b does on the host (CSR build + OMP activation gather) is done inside
// ggml-cuda on the GPU instead, from `ids` alone -- which is why this function
// does not need offs[]/slot[]/act_arena at all, only the raw routing table.
// ============================================================
// Chunked variant of the device MoE path. The ordinary runner submits one
// graph after all compact expert weights have landed. This version keeps two
// compact weight ranges in flight: while GGML computes chunk N, the copy
// stream stages chunk N+1 into the other range. Each graph gathers only the
// token rows routed to that chunk and a CUDA scatter-add reduces its weighted
// output into one [S,D] device accumulator.
static bool run_ggml_moe_layer_chunked(
    StreamingEngine &engine, RAMSafetyManager &ram_mgr, GpuTopology *topo,
    int layer_idx, int S, int D, int moe_inter, int topk, int n_experts,
    const std::vector<int> &used, const int *tok_topk_ids,
    const float *tok_topk_weights, int chunk_experts, bool unpacked_layer,
    int chosen_device, size_t gb0, size_t ub0, size_t db0, const GGUFTensor *g0,
    const GGUFTensor *u0, const GGUFTensor *d0, const void *device_input,
    const void **device_output) {
  if (!device_input || !device_output || unpacked_layer || S <= 1 ||
      topk <= 0 || used.empty() || chunk_experts <= 0 || !topo)
    return false;

  // This graph requires one uniform 3-D weight geometry, just like the
  // non-chunked MUL_MAT_ID path. Reject mixed layers before allocating any
  // state so the caller can use its established fallback safely.
  for (int e : used) {
    size_t gb, ub, db;
    const GGUFTensor *gm, *um, *dm;
    if (!engine.expert_geometry_for(layer_idx, e, gb, ub, db, gm, um, dm) ||
        gb != gb0 || ub != ub0 || db != db0 || gm->type != g0->type ||
        um->type != u0->type || dm->type != d0->type)
      return false;
  }

  struct ChunkPairs {
    std::vector<int32_t> token;
    std::vector<int32_t> local;
    std::vector<float> weight;
  };
  const int chunk = std::min(chunk_experts, (int)used.size());
  const int n_chunks = ((int)used.size() + chunk - 1) / chunk;
  std::vector<ChunkPairs> pairs((size_t)n_chunks);
  std::vector<int> local((size_t)n_experts, -1);
  for (int i = 0; i < (int)used.size(); ++i)
    local[(size_t)used[(size_t)i]] = i;
  int max_pairs = 0;
  for (int s = 0; s < S; ++s) {
    for (int k = 0; k < topk; ++k) {
      const size_t flat = (size_t)s * topk + k;
      const int e = tok_topk_ids[flat];
      if (e < 0 || e >= n_experts || local[(size_t)e] < 0)
        return false;
      const int compact = local[(size_t)e];
      const int ci = compact / chunk;
      const int slot = compact % chunk;
      pairs[(size_t)ci].token.push_back((int32_t)s);
      pairs[(size_t)ci].local.push_back((int32_t)slot);
      pairs[(size_t)ci].weight.push_back(tok_topk_weights[flat]);
    }
  }
  for (const auto &p : pairs)
    max_pairs = std::max(max_pairs, (int)p.token.size());
  if (max_pairs <= 0)
    return false;
  // Keep the graph shape stable across nearby layers. Padding is cheap: the
  // padded route weights are zero, so the device scatter contributes nothing.
  int pair_capacity = ((max_pairs + 255) / 256) * 256;
  const int fixed_pair_capacity = ggml_fixed_chunk_pair_capacity();
  if (fixed_pair_capacity > 0) {
    if (max_pairs > fixed_pair_capacity) {
      LOG_WARN("[GGML-MOE] fixed pair capacity %d is below layer=%d "
               "maximum routed pairs=%d; keeping observed capacity\n",
               fixed_pair_capacity, layer_idx, max_pairs);
    } else {
      pair_capacity = fixed_pair_capacity;
    }
  }

  GgmlMoeRunner &runner = GgmlMoeRunner::instance_for(
      (uint32_t)g0->type, (uint32_t)u0->type, (uint32_t)d0->type);
  GgmlMoeLayerPlan plan;
  plan.device_id = chosen_device;
  plan.n_experts = 2 * chunk; // two alternating compact slot ranges
  plan.n_expert_slots = 2 * chunk;
  plan.n_tokens = pair_capacity; // padded routed-pair count
  plan.source_tokens = S;
  plan.topk = 1;
  plan.d_model = D;
  plan.d_ff = moe_inter;
  plan.gate_type = (uint32_t)g0->type;
  plan.up_type = (uint32_t)u0->type;
  plan.down_type = (uint32_t)d0->type;
  plan.chunk_experts = chunk;
  plan.chunked = true;
  if (!runner.begin_layer(plan))
    return false;
  // If this signature was used by an earlier layer, its copy stream may still
  // be reading the old VRAM-cache sources. Drain and release those holds only
  // when this runner is reused, allowing the other signature runner to make
  // progress between the two uses.
  runner.release_completed_vram_holds();
  if (runner.gate_expert_bytes() != gb0 || runner.up_expert_bytes() != ub0 ||
      runner.down_expert_bytes() != db0)
    return false;

  struct D2DHold {
    int expert_id;
    const uint8_t *device_ptr;
  };
  struct RamH2DHold {
    MoEExpertBlock *block;
  };
  std::vector<D2DHold> d2d_holds;
  std::vector<RamH2DHold> ram_h2d_holds;
  d2d_holds.reserve(used.size());
  ram_h2d_holds.reserve(used.size());
  int n_reused = 0;
  size_t mmap_staged_total = 0;

  auto abort_pipeline = [&]() {
    // A failed chunk may have queued work on both streams. Drain it before
    // releasing held slabs and reset the singleton so the caller's fallback
    // cannot reuse partially published graph state.
    cudaSetDevice(chosen_device);
    cudaDeviceSynchronize();
    for (const auto &h : d2d_holds)
      topo->release_hold(layer_idx, h.expert_id, h.device_ptr, false);
    for (const auto &h : ram_h2d_holds)
      ram_mgr.ram_unpin(*h.block);
    runner.release_completed_vram_holds();
    runner.shutdown();
    return false;
  };

  const size_t ram_cap =
      layer_idx >= 0 && layer_idx < (int)engine.layers_experts.size()
          ? std::min(engine.layers_experts[(size_t)layer_idx].size(),
                     (size_t)std::max(1, ram_mgr.cap_for_layer(layer_idx)))
          : 0;
  const bool qwen_ddr_only_source = qwen_ddr_only_expert_source(engine);

  for (int ci = 0; ci < n_chunks; ++ci) {
    const int first = ci * chunk;
    const int count = std::min(chunk, (int)used.size() - first);
    const int buf = ci % GgmlMoeRunner::NBUF;
    const int slot_base = buf * chunk;
    runner.wait_stage_free(buf);

    std::vector<unsigned char> staged((size_t)count, 1);
    if (!runner.begin_copy_batch(buf))
      return abort_pipeline();

    for (int i = 0; i < count; ++i) {
      const int e = used[(size_t)(first + i)];
      const ExpertLocation *rl = topo->find_expert(layer_idx, e);
      if (rl && rl->device_ptr && rl->upload_ready &&
          rl->gpu_id == chosen_device &&
          topo->hold_for_compute(layer_idx, e, rl->device_ptr, false)) {
        if (runner.copy_expert_d2d(buf, slot_base + i, rl->device_ptr)) {
          staged[(size_t)i] = 0;
          d2d_holds.push_back({e, rl->device_ptr});
          ++n_reused;
        } else {
          topo->release_hold(layer_idx, e, rl->device_ptr, false);
        }
      }
    }
    if (!runner.mark_copy_d2d_done(buf))
      return abort_pipeline();
    cudaSetDevice(chosen_device);

    std::vector<const uint8_t *> ram_sources((size_t)count, nullptr);
    std::vector<MoEExpertBlock *> ram_stage_holds;
    if (ram_cap > 0) {
      auto &ram_slots = engine.layers_experts[(size_t)layer_idx];
      for (int i = 0; i < count; ++i) {
        if (!staged[(size_t)i])
          continue;
        const uint8_t *slab = nullptr;
        size_t capacity = 0;
        MoEExpertBlock *rb = ram_mgr.find_ram_slot_if_ready_held(
            ram_slots, ram_cap, layer_idx, used[(size_t)(first + i)], &slab,
            &capacity);
        if (rb && slab && capacity >= gb0 + ub0 + db0) {
          ram_sources[(size_t)i] = slab;
          ram_stage_holds.push_back(rb);
        } else if (rb) {
          ram_mgr.ram_unpin(*rb);
        }
      }
    }

    std::vector<GgmlMoeStageSlot> slots((size_t)count);
    for (int i = 0; i < count; ++i)
      slots[(size_t)i] = runner.stage_slot(buf, i);
    std::atomic<bool> stage_ok{true};
    size_t staged_bytes = 0;
#pragma omp parallel for schedule(static) reduction(+ : staged_bytes)
    for (int i = 0; i < count; ++i) {
      if (!staged[(size_t)i])
        continue;
      const int e = used[(size_t)(first + i)];
      const GgmlMoeStageSlot &sl = slots[(size_t)i];
      const uint8_t *base = ram_sources[(size_t)i];
      const uint8_t *gs = nullptr;
      const uint8_t *us = nullptr;
      const uint8_t *ds = nullptr;
      if (base) {
        gs = base;
        us = base + gb0;
        ds = us + ub0;
      } else if (qwen_ddr_only_source) {
        // Do not let an incomplete RAM preload turn this optional pipeline
        // into a hidden file-backed expert read.  The caller will reject the
        // chunked path and use the normal DDR-aware fallback.
        stage_ok.store(false, std::memory_order_relaxed);
        continue;
      } else if (engine.model_file.mmap_active()) {
        const uint64_t goff = g0->offset + (uint64_t)e * gb0;
        const uint64_t uoff = u0->offset + (uint64_t)e * ub0;
        const uint64_t doff = d0->offset + (uint64_t)e * db0;
        gs = engine.model_file.mmap_bytes(*g0, goff, gb0);
        us = engine.model_file.mmap_bytes(*u0, uoff, ub0);
        ds = engine.model_file.mmap_bytes(*d0, doff, db0);
      }
      if (!sl.gate || !gs || !us || !ds) {
        stage_ok.store(false, std::memory_order_relaxed);
        continue;
      }
      std::memcpy(sl.gate, gs, gb0);
      std::memcpy(sl.up, us, ub0);
      std::memcpy(sl.down, ds, db0);
      staged_bytes += gb0 + ub0 + db0;
    }
    for (MoEExpertBlock *rb : ram_stage_holds)
      ram_mgr.ram_unpin(*rb);
    mmap_staged_total += staged_bytes;

    std::vector<unsigned char> mask = staged;
    if (!stage_ok.load(std::memory_order_relaxed) ||
        !runner.push_chunk(buf, slot_base, count, (const bool *)mask.data()))
      return abort_pipeline();

    std::vector<int32_t> token((size_t)pair_capacity, 0);
    std::vector<int32_t> ids((size_t)pair_capacity, slot_base);
    std::vector<float> weights((size_t)pair_capacity, 0.0f);
    const auto &p = pairs[(size_t)ci];
    for (size_t i = 0; i < p.token.size(); ++i) {
      token[i] = p.token[i];
      ids[i] = slot_base + p.local[i];
      weights[i] = p.weight[i];
    }
    const void *ignored = nullptr;
    if (!runner.run_chunk_device_input_tensor(device_input, token.data(),
                                              ids.data(), weights.data(), buf,
                                              &ignored))
      return abort_pipeline();
  }

  if (!runner.finish_chunked_output(device_output))
    return abort_pipeline();
  // Keep D2D source slots protected until this runner's copy stream reaches
  // its completion event. Reuse of this signature, or the request boundary,
  // performs the wait and releases the holds. The GGML graph and HC consumer
  // remain queued asynchronously on their own stream.
  for (const auto &h : d2d_holds)
    runner.defer_vram_hold((void *)topo, layer_idx, h.expert_id, h.device_ptr);
  for (const auto &h : ram_h2d_holds)
    ram_mgr.ram_unpin(*h.block);
  perf().n_prefill_stream_vram_hit.fetch_add((uint64_t)n_reused,
                                             std::memory_order_relaxed);
  perf().n_prefill_stream_vram_miss.fetch_add(
      (uint64_t)((int)used.size() - n_reused), std::memory_order_relaxed);
  perf().bytes_disk_read.fetch_add(mmap_staged_total,
                                   std::memory_order_relaxed);
  return true;
}

static bool run_ggml_moe_layer(
    StreamingEngine &engine, RAMSafetyManager &ram_mgr, GpuTopology *topo,
    int layer_idx, int S, int D, int moe_inter, int topk, int n_experts,
    const std::vector<int> &used, const int *tok_topk_ids,
    const float *tok_topk_weights, const float *normed, float *output,
    int chunk_experts, bool unpacked_layer,
    const void **device_output = nullptr, const void *device_input = nullptr,
    const void *device_route_ids = nullptr,
    const void *device_route_weights = nullptr,
    bool allow_active_layer_sources = false,
    GgmlMoeRunner **output_runner = nullptr) {
  if (device_output)
    *device_output = nullptr;
  if (output_runner)
    *output_runner = nullptr;
  if (!GgmlMoeRunner::compiled_in())
    return false;
  const int n_used = (int)used.size();
  if (n_used <= 0 || S < 1 || topk <= 0 || D <= 0 || moe_inter <= 0)
    return false;
  if (!topo)
    return false;
  static const bool trace_set = [] {
    const char *env = std::getenv("MOESTREAM_MOE_SET_TRACE");
    return env != nullptr && env[0] != '0';
  }();

  // A single 3-D weight tensor per projection can only describe experts that
  // share a quant type AND a byte size. Packed layers satisfy this by
  // construction (every expert is a slice of one tensor); an unpacked layer
  // has a tensor per expert and can genuinely be mixed -- GLM-5.3's UD-IQ2_M
  // layers are. Checking rather than assuming is the difference between a
  // clean fallback and reading one expert's bytes as another's.
  size_t gb0 = 0, ub0 = 0, db0 = 0;
  const GGUFTensor *g0 = nullptr, *u0 = nullptr, *d0 = nullptr;
  if (!engine.expert_geometry_for(layer_idx, used[0], gb0, ub0, db0, g0, u0,
                                  d0))
    return false;
  if (unpacked_layer) {
    for (int i = 1; i < n_used; ++i) {
      size_t gb, ub, db;
      const GGUFTensor *gM, *uM, *dM;
      if (!engine.expert_geometry_for(layer_idx, used[(size_t)i], gb, ub, db,
                                      gM, uM, dM))
        return false;
      if (gb != gb0 || ub != ub0 || db != db0 || gM->type != g0->type ||
          uM->type != u0->type || dM->type != d0->type)
        return false;
    }
  }

  // Pick the vault GPU with the most FREE VRAM, not the biggest expert pool:
  // this path allocates its own ~1 GB weight buffer and never touches the
  // expert cache, so free space is the only thing that matters. Decided once
  // per process -- cudaMemGetInfo is not free, and the answer cannot change
  // meaningfully between layers.
  static int chosen_device = [&]() -> int {
    // MOESTREAM_GGML_MOE_DEVICE pins the graph to a specific GPU. The
    // automatic choice below maximises FREE VRAM, which on a box whose vault
    // GPU is deliberately filled with expert cache picks the OTHER card --
    // and on this hardware that is the PCIe x8 5060 Ti (6.8 GB/s H2D) rather
    // than the x16 3090 (12.3 GB/s), i.e. it optimises the wrong thing. There
    // is no way to weigh "more free VRAM" against "twice the link bandwidth"
    // without knowing how much of each the layer will need, so this is a
    // manual override rather than a heuristic.
    if (const char *envd = std::getenv("MOESTREAM_GGML_MOE_DEVICE")) {
      const int d = std::atoi(envd);
      LOG_INFO("[GGML-MOE] device forced to GPU %d by "
               "MOESTREAM_GGML_MOE_DEVICE\n",
               d);
      return d;
    }
    if (moe_ggml_device() >= 0) {
      LOG_INFO("[GGML-MOE] device set to GPU %d by config moe_ggml_device\n",
               moe_ggml_device());
      return moe_ggml_device();
    }
    int best = -1;
    size_t best_free = 0;
    for (int vi : topo->vault_gpus) {
      if ((size_t)vi >= topo->devices.size())
        continue;
      const int dev = topo->devices[(size_t)vi].id;
      size_t f = 0, t = 0;
      if (cudaSetDevice(dev) != cudaSuccess)
        continue;
      if (cudaMemGetInfo(&f, &t) != cudaSuccess)
        continue;
      if (f > best_free) {
        best_free = f;
        best = dev;
      }
    }
    LOG_INFO("[GGML-MOE] selected GPU %d (%.2f GB free)\n", best,
             (double)best_free / (1024.0 * 1024.0 * 1024.0));
    return best;
  }();
  if (chosen_device < 0)
    return false;

  static const bool chunk_pipeline = [] {
    const char *env = std::getenv("MOESTREAM_GGML_CHUNK_PIPELINE");
    return env && env[0] == '1';
  }();
  if (chunk_pipeline && !device_route_ids && !device_route_weights &&
      device_output && device_input && !unpacked_layer) {
    if (run_ggml_moe_layer_chunked(
            engine, ram_mgr, topo, layer_idx, S, D, moe_inter, topk, n_experts,
            used, tok_topk_ids, tok_topk_weights, chunk_experts, unpacked_layer,
            chosen_device, gb0, ub0, db0, g0, u0, d0, device_input,
            device_output)) {
      static bool logged = false;
      if (!logged) {
        LOG_INFO("[GGML-MOE] chunked device prefill pipeline active: "
                 "GPU gather + alternating weight chunks + device scatter\n");
        logged = true;
      }
      return true;
    }
    LOG_WARN("[GGML-MOE] chunked pipeline rejected layer=%d; using the "
             "single-graph path\n",
             layer_idx);
  }

  GgmlMoeRunner &runner = GgmlMoeRunner::instance_for(
      (uint32_t)g0->type, (uint32_t)u0->type, (uint32_t)d0->type);
  GgmlMoeLayerPlan plan;
  plan.device_id = chosen_device;
  plan.n_experts = n_used;
  // A GPU route table contains global expert IDs.  Keep it only for the
  // direct resident-layer view below.  If that view is unavailable, the
  // compact tensor uses a dense [0,n_used) axis and therefore needs the
  // ordinary host route table remapped to that compact axis.
  bool use_device_route_for_graph =
      device_route_ids && device_route_weights && device_input;
  // The runner normally rounds this request to a grow-only compact-slot
  // high-water mark (64-expert increments).  A fixed slot count is an opt-in
  // Bee-style graph-shape experiment: it keeps the graph/allocator stable
  // across all layers of a quantization signature.  ids still address only
  // [0,n_used), so the unused tail is dead storage rather than extra work.
  const int fixed_expert_slots = ggml_fixed_expert_slots();
  plan.n_expert_slots =
      fixed_expert_slots > 0 ? std::max(fixed_expert_slots, n_used) : n_used;
  plan.n_tokens = S;
  plan.topk = topk;
  plan.d_model = D;
  plan.d_ff = moe_inter;
  plan.gate_type = (uint32_t)g0->type;
  plan.up_type = (uint32_t)u0->type;
  plan.down_type = (uint32_t)d0->type;
  plan.chunk_experts = std::min(chunk_experts > 0 ? chunk_experts : 32, n_used);

  // Bee's resident-layer representation is a single expert tensor.  A
  // compact cache can provide the same view only when an explicitly selected
  // full-layer region was allocated and every expert is currently resident.
  // Host-routed Qwen decode is allowed to use this path too; other model
  // families retain the existing compact-repack behavior.
  const bool allow_qwen_host_external =
      engine.family_ops.family == ModelFamily::Qwen4Exp && S == 1 &&
      device_output && tok_topk_ids && tok_topk_weights;
  const bool compact_full_prefix_layer =
      topo->expert_compact_layout && layer_idx >= 0 &&
      layer_idx < topo->expert_full_layer_prefix;
  // GGML-CUDA's quantized MUL_MAT_ID path requires the full expert-axis
  // allocation to be contiguous. A compact cache slot may include alignment
  // padding, which is valid for ordinary repack/copy use but cannot be
  // exposed as a zero-repack tensor. Leave those layers on the safe compact
  // path until the backend has a stride-aware kernel.
  const bool compact_full_prefix_raw_contiguous =
      !compact_full_prefix_layer ||
      (layer_idx >= 0 &&
       (size_t)layer_idx < topo->expert_layer_slab_bytes.size() &&
       topo->expert_layer_slab_bytes[(size_t)layer_idx] == gb0 + ub0 + db0);
  const bool external_layout_possible =
      (topo->expert_slab_size > 0 || compact_full_prefix_layer) &&
      compact_full_prefix_raw_contiguous;
  if ((use_device_route_for_graph || allow_qwen_host_external) &&
      !unpacked_layer && external_layout_possible) {
    std::vector<std::pair<int, const uint8_t *>> external_holds;
    external_holds.reserve((size_t)n_experts);
    // Compact regions use the physical per-slot stride, which may include
    // GGUF/cache alignment padding. The routed payload itself is still only
    // gb0+ub0+db0 bytes; using that raw sum here would point expert 1 into the
    // padding/tail of expert 0 and correctly fail the contiguity check.
    const size_t external_stride =
        topo->expert_compact_layout
            ? ((layer_idx >= 0 &&
                (size_t)layer_idx < topo->expert_layer_slab_bytes.size())
                   ? topo->expert_layer_slab_bytes[(size_t)layer_idx]
                   : gb0 + ub0 + db0)
            : topo->expert_slab_size;
    const uint8_t *external_base = nullptr;
    const bool contiguous = topo->find_contiguous_resident_layer(
        layer_idx, n_experts, chosen_device, external_stride, &external_base,
        &external_holds);
    if (contiguous && external_base) {
      plan.external_weights = true;
      // The external tensor preserves the model's global expert axis.  The
      // compact `used` vector may contain only the experts routed by this
      // batch; that is fine because the device route IDs still index the full
      // resident layer.
      plan.n_experts = n_experts;
      plan.n_expert_slots = n_experts;
      plan.external_expert_stride = external_stride;
      plan.external_gate = external_base;
      plan.external_up = external_base + gb0;
      plan.external_down = external_base + gb0 + ub0;
      plan.chunk_experts = n_used;
      if (runner.begin_layer(plan)) {
        bool external_ok = false;
        if (use_device_route_for_graph) {
          external_ok = runner.run_layer_device_input_tensor_device_route(
              device_input, device_route_ids, device_route_weights,
              device_output);
        } else if (device_input) {
          external_ok = runner.run_layer_device_input_tensor(
              device_input, tok_topk_ids, tok_topk_weights, device_output);
        } else if (device_output) {
          external_ok = runner.run_layer_device(
              normed, tok_topk_ids, tok_topk_weights, device_output);
        } else {
          external_ok =
              runner.run_layer(normed, tok_topk_ids, tok_topk_weights, output);
        }
        if (external_ok) {
          if (output_runner)
            *output_runner = &runner;
          for (const auto &hold : external_holds)
            runner.defer_vram_hold((void *)topo, layer_idx, hold.first,
                                   hold.second);
          perf().n_prefill_stream_vram_hit.fetch_add((uint64_t)n_used,
                                                     std::memory_order_relaxed);
          static bool logged_external_layer = false;
          if (!logged_external_layer) {
            LOG_INFO("[GGML-MOE] layer=%d external resident weights: "
                     "%d slabs, stride=%zu, no repack/H2D (%s route)\n",
                     layer_idx, n_experts, external_stride,
                     use_device_route_for_graph ? "device" : "host");
            logged_external_layer = true;
          }
          return true;
        }
        // A failed graph submission is an exceptional path. Drain the CUDA
        // work before returning the cache holds so a later fallback cannot
        // race a partially submitted graph.
        cudaSetDevice(chosen_device);
        cudaDeviceSynchronize();
      }
      for (const auto &hold : external_holds)
        topo->release_hold(layer_idx, hold.first, hold.second, false);
      plan.external_weights = false;
      plan.external_expert_stride = 0;
      plan.external_gate = plan.external_up = plan.external_down = nullptr;
      use_device_route_for_graph = false;
    } else {
      for (const auto &hold : external_holds)
        topo->release_hold(layer_idx, hold.first, hold.second, false);
      static bool logged_scatter = false;
      if (!logged_scatter) {
        LOG_INFO("[GGML-MOE] resident layer view unavailable: cache slabs "
                 "are not complete/consecutive by expert ID; using compact "
                 "repack (layer=%d stride=%zu)\n",
                 layer_idx, external_stride);
        logged_scatter = true;
      }
      use_device_route_for_graph = false;
    }
  }

  if (!runner.begin_layer(plan))
    return false;

  // ggml's own per-expert byte counts, derived from the shapes rather than
  // from the GGUF metadata, must agree with what the file actually holds --
  // otherwise the memcpy below would either truncate or overrun.
  if (runner.gate_expert_bytes() != gb0 || runner.up_expert_bytes() != ub0 ||
      runner.down_expert_bytes() != db0) {
    LOG_ERROR("[GGML-MOE] layer=%d geometry mismatch (gate %zu vs %zu, up %zu "
              "vs %zu, down %zu vs %zu) -- falling back.\n",
              layer_idx, runner.gate_expert_bytes(), gb0,
              runner.up_expert_bytes(), ub0, runner.down_expert_bytes(), db0);
    return false;
  }

  const int chunk = plan.chunk_experts;
  const int n_chunks = (n_used + chunk - 1) / chunk;
  size_t mmap_staged_total = 0;
  // Experts held in the VRAM cache for the duration of their device-to-device
  // copy. Released only after sync_copies(), never per chunk: the copy is
  // asynchronous, and a hold dropped early lets an eviction reclaim the slab
  // the copy engine is still reading.
  struct D2DHold {
    int expert_id;
    const uint8_t *device_ptr;
    bool from_active_layer;
  };
  std::vector<D2DHold> d2d_holds;
  d2d_holds.reserve((size_t)n_used);
  struct RamH2DHold {
    int expert_id;
    MoEExpertBlock *block;
  };
  std::vector<RamH2DHold> ram_h2d_holds;
  ram_h2d_holds.reserve((size_t)n_used);
  int n_reused = 0;
  int n_active_reused = 0;
  int n_ram_direct = 0;
  int n_ram_staged = 0;
  // Filled once per compact chunk by GpuTopology::find_expert_batch().
  // These pointers are only residency hints; hold_for_compute() below still
  // performs its independent identity check before the D2D enqueue.
  std::vector<ExpertLocation *> resident_locations((size_t)n_used, nullptr);
  std::vector<char> staged_flag((size_t)chunk, 1);
  // 0 = pending, 1 = copied from the selected GPU's VRAM cache,
  // 2 = copied directly from a pinned RAM slab, 3 = copied from a pageable
  // RAM slab through the pinned staging buffers, 4 = copied from mmap.
  std::vector<unsigned char> trace_source((size_t)n_used, 0);
  std::vector<int> trace_resident_any;
  std::vector<int> trace_resident_gpu;
  trace_resident_any.reserve((size_t)n_used);
  trace_resident_gpu.reserve((size_t)n_used);

  static const bool no_d2d = [] {
    const char *ev = std::getenv("MOESTREAM_GGML_NO_D2D");
    return ev && ev[0] == '1';
  }();
  // Diagnostic/optimization switch for the packed transfer design. The
  // normal path sends persistently pinned RAM slabs directly, one packed
  // record at a time. Forcing those slabs through h_stage_ lets push_chunk()
  // combine the whole 32-expert batch into one H2D plus one unpack-scatter,
  // which tests whether launch/submission overhead is larger than the extra
  // DDR->pinned-DDR memcpy.
  static const bool env_force_batch_stage = [] {
    const char *ev = std::getenv("MOESTREAM_GGML_FORCE_BATCH_STAGE");
    return ev && ev[0] == '1';
  }();
  const bool qwen_decode_batch_stage =
      engine.family_ops.family == ModelFamily::Qwen4Exp && S == 1 &&
      engine.qwen4exp_decode_batch_stage;
  const bool force_batch_stage =
      env_force_batch_stage || qwen_decode_batch_stage;
  static bool logged_batch_stage = false;
  if (force_batch_stage && !logged_batch_stage) {
    LOG_INFO("[GGML-MOE] RAM sources use packed batch staging (%s)\n",
             qwen_decode_batch_stage ? "Qwen decode config" : "environment");
    logged_batch_stage = true;
  }
  // D2D cache copies use a separate raw stream. The sync point below drains
  // only that stream in backend mode; H2D staging and graph work remain
  // continuously queued on GGML's shared stream.
  const bool use_d2d = !no_d2d;

  for (int c = 0; c < n_chunks; ++c) {
    const int c0 = c * chunk;
    const int c1 = std::min(n_used, c0 + chunk);
    const int buf = c % GgmlMoeRunner::NBUF;
    // Blocks until this buffer's PREVIOUS H2D drained, which is also what
    // overlaps chunk c's host memcpy with chunk c-1's transfer. Timed so the
    // [PREFILL] line's "CPU blocked on the double-buffer handoff" figure is
    // meaningful for this path too: a large value means the transfer is the
    // bottleneck and the staging memcpy is fully hidden; ~0 means the
    // opposite, and the two are NOT actually overlapping.
    PERF_START(gm_wait);
    runner.wait_stage_free(buf);
    PERF_END(gm_wait, t_prefill_stream_wait);

    // Start the copy-stream timing batch before VRAM reuse is queued.  This
    // lets the runner report the real D2D and H2D CUDA-event intervals
    // separately; the old placement started the timer only after D2D and
    // made the source breakdown ambiguous.
    if (!runner.begin_copy_batch(buf)) {
      runner.sync_copies();
      return false;
    }

    // Classify BEFORE staging: an expert already resident in this GPU's VRAM
    // cache is filled device-to-device (~600 GB/s) instead of being read out
    // of the mmap, memcpy'd into pinned host memory and shipped back over a
    // 12.3 GB/s link. Probe the whole compact set under one cache lock; the
    // per-expert hold below remains an independent race-safe check.
    // Only THIS device's residents qualify: a slab on the other GPU would be
    // a peer copy, which is not obviously cheaper than the H2D it replaces
    // and is not what this was measured against.
    if (use_d2d) {
      topo->find_expert_batch(layer_idx, used.data() + c0, c1 - c0,
                              resident_locations.data() + c0);
    }
    for (int i = c0; i < c1; ++i) {
      const int e = used[(size_t)i];
      staged_flag[(size_t)(i - c0)] = 1;
      // D2D reuse costs a find_expert (cache_mutex) per expert and only pays
      // off if enough of them are resident. MOESTREAM_GGML_NO_D2D=1 measures
      // whether that trade is actually positive at the current hit rate.
      const ExpertLocation *rl =
          use_d2d ? resident_locations[(size_t)i] : nullptr;
      if (trace_set && rl && rl->device_ptr && rl->upload_ready) {
        trace_resident_any.push_back(e);
        if (rl->gpu_id == chosen_device)
          trace_resident_gpu.push_back(e);
      }
      bool from_active_layer = false;
      if (!rl && use_d2d && allow_active_layer_sources) {
        bool promoted_to_main = false;
        rl = topo->find_active_expert_wait(
            layer_idx, e, [&](int, int) { promoted_to_main = true; });
        from_active_layer = rl != nullptr && !promoted_to_main;
      }
      if (rl && rl->device_ptr && rl->upload_ready &&
          rl->gpu_id == chosen_device &&
          topo->hold_for_compute(layer_idx, e, rl->device_ptr,
                                 from_active_layer)) {
        if (runner.copy_expert_d2d(buf, i, rl->device_ptr)) {
          staged_flag[(size_t)(i - c0)] = 0;
          trace_source[(size_t)i] = 1;
          d2d_holds.push_back({e, rl->device_ptr, from_active_layer});
          ++n_reused;
          if (from_active_layer)
            ++n_active_reused;
        } else {
          // Copy could not be issued -- give the hold straight back and let
          // this expert go through the normal staging path below.
          topo->release_hold(layer_idx, e, rl->device_ptr, from_active_layer);
        }
      }
    }
    if (!runner.mark_copy_d2d_done(buf)) {
      runner.sync_copies();
      return false;
    }
    // find_expert() -> finish_expert_hit() does cudaSetDevice(loc.gpu_id) and
    // does NOT restore it, so a hit on an expert resident on the other vault
    // GPU leaves that GPU current. Everything below assumes ours.
    cudaSetDevice(chosen_device);

    if (!runner.begin_copy_batch(buf)) {
      runner.sync_copies();
      for (const auto &h : d2d_holds)
        topo->release_hold(layer_idx, h.expert_id, h.device_ptr,
                           h.from_active_layer);
      for (const auto &h : ram_h2d_holds)
        ram_mgr.ram_unpin(*h.block);
      return false;
    }

    // Prefer a RAM-tier slab when it was persistently page-locked at
    // allocation time. Queue the three projection copies directly from the
    // [gate|up|down] slab and keep the RAM slot pinned until the layer's one
    // copy barrier below. Pageable RAM uses the same pinned staging buffers,
    // but the source is the resident RAM slab and the CPU memcpy completes
    // before the RAM hold is released. This avoids both mmap and CUDA's
    // hidden pageable-source staging copy.
    const size_t ram_cap =
        (layer_idx >= 0 && layer_idx < (int)engine.layers_experts.size())
            ? std::min(engine.layers_experts[(size_t)layer_idx].size(),
                       (size_t)std::max(1, ram_mgr.cap_for_layer(layer_idx)))
            : 0;
    const bool qwen_ddr_only_source = qwen_ddr_only_expert_source(engine);
    static std::atomic<bool> logged_qwen_ddr_miss{false};
    if (ram_cap > 0) {
      auto &ram_slots = engine.layers_experts[(size_t)layer_idx];
      for (int i = c0; i < c1; ++i) {
        if (!staged_flag[(size_t)(i - c0)])
          continue;
        const uint8_t *ram_slab = nullptr;
        size_t ram_capacity = 0;
        MoEExpertBlock *rb = ram_mgr.find_ram_slot_if_ready_pinned(
            ram_slots, ram_cap, layer_idx, used[(size_t)i], &ram_slab,
            &ram_capacity);
        if (!rb)
          continue;
        // Backend-stream mode used to force even persistently pinned RAM
        // slabs through h_stage_: the CPU copied [gate|up|down] into the
        // staging arena, then the copy stream uploaded that duplicate. A
        // packed direct H2D is safe here because this lookup guarantees a
        // page-locked slab and ram_h2d_holds keeps it from being evicted
        // until collect_h2d() has drained the completion event.
        const bool direct_ok = !force_batch_stage && ram_slab &&
                               ram_capacity >= gb0 + ub0 + db0 &&
                               runner.copy_expert_h2d(buf, i, ram_slab);
        if (direct_ok) {
          // This is the first actual predicted-layer DDR->VRAM H2D issue
          // when the grouped runner cannot reuse a VRAM/active-layer source.
          qwen4exp_record_pilot_fetch_start(layer_idx);
          staged_flag[(size_t)(i - c0)] = 0;
          trace_source[(size_t)i] = 2;
          ram_h2d_holds.push_back({used[(size_t)i], rb});
          ++n_ram_direct;
        } else {
          ram_mgr.ram_unpin(*rb);
        }
      }
    }

    // Resident but pageable RAM is still the preferred source when the
    // aggregate cudaHostRegister ceiling has been reached. Hold each slab
    // through the synchronous CPU copy below; after that copy the queued H2D
    // reads only the separate pinned staging buffer.
    std::vector<const uint8_t *> ram_stage_sources((size_t)(c1 - c0), nullptr);
    std::vector<MoEExpertBlock *> ram_stage_holds;
    ram_stage_holds.reserve((size_t)(c1 - c0));
    if (ram_cap > 0) {
      auto &ram_slots = engine.layers_experts[(size_t)layer_idx];
      for (int i = c0; i < c1; ++i) {
        if (!staged_flag[(size_t)(i - c0)])
          continue;
        const uint8_t *ram_slab = nullptr;
        size_t ram_capacity = 0;
        MoEExpertBlock *rb = ram_mgr.find_ram_slot_if_ready_held(
            ram_slots, ram_cap, layer_idx, used[(size_t)i], &ram_slab,
            &ram_capacity);
        if (!rb)
          continue;
        if (ram_slab && ram_capacity >= gb0 + ub0 + db0) {
          ram_stage_sources[(size_t)(i - c0)] = ram_slab;
          trace_source[(size_t)i] = 3;
          ram_stage_holds.push_back(rb);
          ++n_ram_staged;
        } else {
          ram_mgr.ram_unpin(*rb);
        }
      }
    }

    // madvise(WILLNEED) ahead of the staging memcpy. Worth it only when the
    // pages are genuinely cold: at ~3 syscalls per expert this is ~22k calls
    // per prefill, which is pure overhead once the file is in page cache.
    // MOESTREAM_GGML_NO_WILLNEED=1 measures that.
    static const bool skip_willneed = [] {
      const char *e = std::getenv("MOESTREAM_GGML_NO_WILLNEED");
      return e && e[0] == '1';
    }();
    if (!qwen_ddr_only_source && !skip_willneed &&
        engine.model_file.mmap_active()) {
      for (int i = c0; i < c1; ++i) {
        if (!staged_flag[(size_t)(i - c0)] ||
            ram_stage_sources[(size_t)(i - c0)])
          continue;
        const int e = used[(size_t)i];
        const uint64_t goff =
            unpacked_layer ? g0->offset : g0->offset + (uint64_t)e * gb0;
        const uint64_t uoff =
            unpacked_layer ? u0->offset : u0->offset + (uint64_t)e * ub0;
        const uint64_t doff =
            unpacked_layer ? d0->offset : d0->offset + (uint64_t)e * db0;
        engine.model_file.mmap_willneed(*g0, goff, gb0);
        engine.model_file.mmap_willneed(*u0, uoff, ub0);
        engine.model_file.mmap_willneed(*d0, doff, db0);
      }
    }

    // Slot pointers are resolved OUTSIDE the parallel region on purpose. They
    // do not come from a thread_local here, but the same discipline that cost
    // two SIGSEGVs in the streaming path applies: nothing a worker touches
    // should depend on which thread is asking.
    std::vector<GgmlMoeStageSlot> slots((size_t)(c1 - c0));
    for (int i = c0; i < c1; ++i)
      slots[(size_t)(i - c0)] = runner.stage_slot(buf, i - c0);

    // Pageable RAM uses a synchronous DDR->pinned staging memcpy before the
    // asynchronous H2D.  Mark the boundary immediately before that work so
    // the reported gap is not confused with the later CUDA copy service time.
    if (n_ram_staged > 0)
      qwen4exp_record_pilot_fetch_start(layer_idx);

    std::atomic<bool> stage_ok{true};
    size_t staged = 0;
    PERF_START(pf_stage);
#pragma omp parallel for schedule(static) reduction(+ : staged)
    for (int i = c0; i < c1; ++i) {
      if (!staged_flag[(size_t)(i - c0)])
        continue; // filled D2D already
      const int e = used[(size_t)i];
      const GgmlMoeStageSlot &sl = slots[(size_t)(i - c0)];
      if (!sl.gate) {
        stage_ok.store(false, std::memory_order_relaxed);
        continue;
      }
      const uint8_t *gs = nullptr;
      const uint8_t *us = nullptr;
      const uint8_t *ds = nullptr;
      const uint8_t *ram_base = ram_stage_sources[(size_t)(i - c0)];
      if (ram_base) {
        gs = ram_base;
        us = ram_base + gb0;
        ds = ram_base + gb0 + ub0;
      } else if (qwen_ddr_only_source) {
        bool expected = false;
        if (logged_qwen_ddr_miss.compare_exchange_strong(
                expected, true, std::memory_order_relaxed))
          LOG_ERROR("[GGML-MOE] Qwen ram_preload_all_experts source miss: "
                    "layer=%d expert=%d has no ready DDR slab; refusing GGUF "
                    "mmap fallback\n",
                    layer_idx, e);
        stage_ok.store(false, std::memory_order_relaxed);
        continue;
      } else if (engine.model_file.mmap_active()) {
        const uint64_t goff =
            unpacked_layer ? g0->offset : g0->offset + (uint64_t)e * gb0;
        const uint64_t uoff =
            unpacked_layer ? u0->offset : u0->offset + (uint64_t)e * ub0;
        const uint64_t doff =
            unpacked_layer ? d0->offset : d0->offset + (uint64_t)e * db0;
        gs = engine.model_file.mmap_bytes(*g0, goff, gb0);
        us = engine.model_file.mmap_bytes(*u0, uoff, ub0);
        ds = engine.model_file.mmap_bytes(*d0, doff, db0);
        trace_source[(size_t)i] = 4;
      }
      if (!gs || !us || !ds) {
        stage_ok.store(false, std::memory_order_relaxed);
        continue;
      }
      std::memcpy(sl.gate, gs, gb0);
      std::memcpy(sl.up, us, ub0);
      std::memcpy(sl.down, ds, db0);
      staged += gb0 + ub0 + db0;
    }
    PERF_END(pf_stage, t_disk_read);
    size_t ram_staged_bytes = 0;
    for (int i = c0; i < c1; ++i) {
      if (staged_flag[(size_t)(i - c0)] && ram_stage_sources[(size_t)(i - c0)])
        ram_staged_bytes += gb0 + ub0 + db0;
    }
    if (stage_ok.load(std::memory_order_relaxed))
      mmap_staged_total += staged - ram_staged_bytes;
    for (MoEExpertBlock *rb : ram_stage_holds)
      ram_mgr.ram_unpin(*rb);
    // std::vector<bool> is a bitset with no .data(), so the mask push_chunk
    // takes is built here as real bools rather than reusing staged_flag.
    std::vector<unsigned char> mask_store((size_t)(c1 - c0));
    for (int i = 0; i < c1 - c0; ++i)
      mask_store[(size_t)i] = staged_flag[(size_t)i] != 0;
    const bool *staged_mask = (const bool *)mask_store.data();
    static_assert(sizeof(bool) == sizeof(unsigned char),
                  "staged mask aliasing assumes 1-byte bool");
    if (!stage_ok.load(std::memory_order_relaxed) ||
        !runner.push_chunk(buf, c0, c1 - c0, staged_mask)) {
      // Anything already in flight has to drain before the caller reuses the
      // staging buffers on the next layer -- and before any held slab can be
      // released, since the copy engine may still be reading it.
      runner.sync_copies();
      if (runner.backend_stream_copies())
        runner.finish_h2d();
      for (const auto &h : d2d_holds)
        topo->release_hold(layer_idx, h.expert_id, h.device_ptr,
                           h.from_active_layer);
      for (const auto &h : ram_h2d_holds)
        ram_mgr.ram_unpin(*h.block);
      LOG_ERROR("[GGML-MOE] layer=%d chunk@%d: staging failed -- falling back "
                "to the per-expert path for this layer.\n",
                layer_idx, c0);
      return false;
    }
  }
  perf().bytes_disk_read.fetch_add(mmap_staged_total,
                                   std::memory_order_relaxed);

  // Every weight copy, H2D and D2D alike, must be complete before a held slab
  // can be handed back to the eviction machinery. Timed into the same bucket
  // as the per-chunk handoff wait: together they are the whole "CPU blocked
  // on the expert transfer" cost, and this one is the tail the pipeline
  // cannot hide (the last chunk's H2D only starts once its staging is done).
  PERF_START(gm_sync);
  if (!runner.backend_stream_copies())
    runner.sync_copies();
  PERF_END(gm_sync, t_prefill_stream_wait);
  if (!runner.backend_stream_copies()) {
    for (const auto &h : d2d_holds)
      topo->release_hold(layer_idx, h.expert_id, h.device_ptr,
                         h.from_active_layer);
    for (const auto &h : ram_h2d_holds)
      ram_mgr.ram_unpin(*h.block);
    ram_h2d_holds.clear();
  } else {
    // The backend-stream path has a device-side copy->graph dependency. Keep
    // both classes of source protected across the layer boundary; releasing a
    // RAM pin or VRAM hold here would let eviction overwrite bytes while the
    // queued copy stream is still reading them. finish_h2d() drains the
    // completion events once for the whole forward pass and runs these
    // releases then.
    for (const auto &h : d2d_holds)
      runner.defer_vram_hold((void *)topo, layer_idx, h.expert_id,
                             h.device_ptr);
    d2d_holds.clear();
    if (!ram_h2d_holds.empty()) {
      auto holds = std::move(ram_h2d_holds);
      runner.defer_host_release([&ram_mgr, holds = std::move(holds)]() mutable {
        for (const auto &h : holds)
          ram_mgr.ram_unpin(*h.block);
      });
    }
  }
  perf().n_prefill_stream_vram_hit.fetch_add((uint64_t)n_reused,
                                             std::memory_order_relaxed);
  perf().n_prefill_stream_vram_miss.fetch_add((uint64_t)(n_used - n_reused),
                                              std::memory_order_relaxed);

  // The ordinary graph uses compact host IDs.  The all-resident device-route
  // path deliberately keeps the router's global IDs on the device and makes
  // `used` the complete layer, so global and compact IDs are identical.
  std::vector<int32_t> ids;
  if (!use_device_route_for_graph) {
    if (!tok_topk_ids || !tok_topk_weights)
      return false;
    std::vector<int> local((size_t)n_experts, -1);
    for (int i = 0; i < n_used; ++i)
      local[(size_t)used[(size_t)i]] = i;
    ids.resize((size_t)S * topk);
    for (size_t i = 0, n = (size_t)S * topk; i < n; ++i) {
      const int e = tok_topk_ids[i];
      if (e < 0 || e >= n_experts || local[(size_t)e] < 0) {
        for (int b = 0; b < GgmlMoeRunner::NBUF; ++b)
          runner.wait_stage_free(b);
        for (const auto &h : ram_h2d_holds)
          ram_mgr.ram_unpin(*h.block);
        return false;
      }
      ids[i] = (int32_t)local[(size_t)e];
    }
  }

  PERF_START(ggml_run);
  const auto t_graph0 = std::chrono::steady_clock::now();
  const bool ok =
      use_device_route_for_graph
          ? (device_input ? runner.run_layer_device_input_tensor_device_route(
                                device_input, device_route_ids,
                                device_route_weights, device_output)
                          : false)
          : (device_output
                 ? (device_input ? runner.run_layer_device_input_tensor(
                                       device_input, ids.data(),
                                       tok_topk_weights, device_output)
                                 : runner.run_layer_device(normed, ids.data(),
                                                           tok_topk_weights,
                                                           device_output))
                 : runner.run_layer(normed, ids.data(), tok_topk_weights,
                                    output));
  PERF_END(ggml_run, t_gpu_compute);
  // Per-layer, not aggregated, DEBUG-only: this model's layers are NOT uniform (24 are
  // IQ2_XXS gate/up, 24 are IQ1_M -- see the [QUANT] startup report), and
  // ggml-cuda picks a completely different kernel family per type (MMQ for
  // one, the dequantise+cuBLAS fallback for the other). A single total would
  // average those together and hide which one actually costs.
  {
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t_graph0)
                          .count();
    int n_mmap_experts = 0;
    for (int i = 0; i < n_used; ++i)
      if (trace_source[(size_t)i] == 4)
        ++n_mmap_experts;
    perf().n_ggml_moe_d2d_experts.fetch_add((uint64_t)n_reused,
                                            std::memory_order_relaxed);
    perf().n_ggml_moe_active_d2d_experts.fetch_add((uint64_t)n_active_reused,
                                                   std::memory_order_relaxed);
    perf().n_ggml_moe_ram_direct_experts.fetch_add((uint64_t)n_ram_direct,
                                                   std::memory_order_relaxed);
    perf().n_ggml_moe_ram_staged_experts.fetch_add((uint64_t)n_ram_staged,
                                                   std::memory_order_relaxed);
    perf().n_ggml_moe_mmap_experts.fetch_add((uint64_t)n_mmap_experts,
                                             std::memory_order_relaxed);
    LOG_DEBUG("[GGML-MOE] layer=%d experts=%d gate=%u up=%u down=%u "
             "graph=%.2f ms d2d=%d active_d2d=%d ram_direct=%d "
             "ram_staged=%d mmap_staged=%.2f MB\n",
             layer_idx, n_used, (unsigned)g0->type, (unsigned)u0->type,
             (unsigned)d0->type, ms, n_reused, n_active_reused, n_ram_direct,
             n_ram_staged, (double)mmap_staged_total / (1024.0 * 1024.0));
  }

  if (trace_set) {
    auto format_ids = [](const std::vector<int> &ids) {
      std::string result;
      for (size_t i = 0; i < ids.size(); ++i) {
        if (i != 0)
          result.push_back(',');
        result += std::to_string(ids[i]);
      }
      return result;
    };
    std::vector<int> d2d_ids;
    std::vector<int> ram_direct_ids;
    std::vector<int> ram_staged_ids;
    std::vector<int> mmap_ids;
    std::vector<int> h2d_ids;
    d2d_ids.reserve((size_t)n_used);
    ram_direct_ids.reserve((size_t)n_used);
    ram_staged_ids.reserve((size_t)n_used);
    mmap_ids.reserve((size_t)n_used);
    h2d_ids.reserve((size_t)n_used);
    for (int i = 0; i < n_used; ++i) {
      const int e = used[(size_t)i];
      switch (trace_source[(size_t)i]) {
      case 1:
        d2d_ids.push_back(e);
        break;
      case 2:
        ram_direct_ids.push_back(e);
        h2d_ids.push_back(e);
        break;
      case 3:
        ram_staged_ids.push_back(e);
        h2d_ids.push_back(e);
        break;
      case 4:
        mmap_ids.push_back(e);
        h2d_ids.push_back(e);
        break;
      default:
        break;
      }
    }
    const size_t expert_bytes = gb0 + ub0 + db0;
    LOG_INFO(
        "[MOE-SET] layer=%d required=%d resident_any=%zu "
        "resident_gpu=%zu d2d=%zu h2d=%zu ram_direct=%zu "
        "ram_staged=%zu mmap=%zu h2d_bytes=%zu required_ids=%s "
        "resident_any_ids=%s resident_gpu_ids=%s d2d_ids=%s "
        "ram_direct_ids=%s ram_staged_ids=%s mmap_ids=%s h2d_ids=%s\n",
        layer_idx, n_used, trace_resident_any.size(), trace_resident_gpu.size(),
        d2d_ids.size(), h2d_ids.size(), ram_direct_ids.size(),
        ram_staged_ids.size(), mmap_ids.size(), h2d_ids.size() * expert_bytes,
        format_ids(used).c_str(), format_ids(trace_resident_any).c_str(),
        format_ids(trace_resident_gpu).c_str(), format_ids(d2d_ids).c_str(),
        format_ids(ram_direct_ids).c_str(), format_ids(ram_staged_ids).c_str(),
        format_ids(mmap_ids).c_str(), format_ids(h2d_ids).c_str());
  }

  GgmlMoeTransferStats transfer;
  runner.collect_transfer_stats(transfer);
  if (transfer.h2d_bytes || transfer.d2d_bytes) {
    perf().t_ggml_moe_d2d.fetch_add(transfer.d2d_seconds,
                                    std::memory_order_relaxed);
    perf().t_ggml_moe_h2d.fetch_add(transfer.h2d_seconds,
                                    std::memory_order_relaxed);
    perf().t_ggml_moe_h2d_dma.fetch_add(transfer.h2d_dma_seconds,
                                        std::memory_order_relaxed);
    perf().bytes_ggml_moe_d2d.fetch_add(transfer.d2d_bytes,
                                        std::memory_order_relaxed);
    perf().bytes_ggml_moe_h2d.fetch_add(transfer.h2d_bytes,
                                        std::memory_order_relaxed);
    // Preserve the existing generic report for compatibility, but it now
    // contains only actual H2D bytes/time from this runner.
    if (transfer.h2d_bytes) {
      perf().t_prefill_stream_h2d.fetch_add(transfer.h2d_seconds,
                                            std::memory_order_relaxed);
      perf().bytes_prefill_stream_h2d.fetch_add(transfer.h2d_bytes,
                                                std::memory_order_relaxed);
    }
  }
  if (ok && output_runner && device_output && *device_output)
    *output_runner = &runner;
  return ok;
}

// Run Qwen4Exp's always-on shared expert against an already device-resident
// routed result.  Keeping this as a small helper lets both the ordinary
// host-route path and the all-resident device-route path use exactly the same
// graph and fallback behavior.
static bool run_qwen4exp_shared_device(StreamingEngine &engine, int layer_idx,
                                       int S, int D, int moe_inter,
                                       const void *device_input,
                                       const void *routed_output,
                                       const ArchitectureConfig &c,
                                       const void **combined_output,
                                       const float *host_routed = nullptr) {
  static const bool debug = [] {
    const char *e = std::getenv("MOESTREAM_GGML_SHARED_DEBUG");
    return e && e[0] == '1';
  }();
  if (combined_output)
    *combined_output = nullptr;
  static const bool force_host_decode = [] {
    const char *e = std::getenv("MOESTREAM_GGML_SHARED_FORCE_HOST_DECODE");
    return e && e[0] == '1';
  }();
  if (force_host_decode && S == 1)
    return false;
  if (!device_input || (!routed_output && !host_routed) ||
      (routed_output && host_routed) || !combined_output ||
      c.n_expert_shared <= 0 || !moe_ggml_shared_expert()) {
    if (debug)
      LOG_WARN("[GGML-SHARED-DEBUG] layer=%d S=%d rejected: missing input, "
               "output, result pointer, or shared flag\n",
               layer_idx, S);
    return false;
  }

  const std::string prefix = "blk." + std::to_string(layer_idx) + ".";
  const int shared_device = moe_ggml_device() >= 0 ? moe_ggml_device() : 0;
  static const bool no_external_weights = [] {
    const char *e = std::getenv("MOESTREAM_GGML_SHARED_NO_EXTERNAL");
    return e && e[0] == '1';
  }();
  auto make_ref = [&](const std::string &name, int64_t ne0,
                      int64_t ne1) -> std::optional<GgmlSharedWeightRef> {
    auto it = engine.resident_layers.find(name);
    if (it == engine.resident_layers.end() || !it->second.host_ptr)
      return std::nullopt;
    const auto &w = it->second;
    if (w.meta.shape.size() < 2 || (int64_t)w.meta.shape[0] != ne0 ||
        (int64_t)w.meta.shape[1] != ne1)
      return std::nullopt;
    GgmlSharedWeightRef ref;
    ref.host_data = w.host_ptr;
    ref.type = (uint32_t)w.meta.type;
    ref.ne0 = ne0;
    ref.ne1 = ne1;
    ref.nbytes = w.meta.size_bytes;
    // The dense pool is allocated on the core GPU. Borrow it only when the
    // shared GGML backend is configured for that same device; otherwise the
    // runner's host upload path remains the safe fallback.
    if (!no_external_weights && engine.gpu &&
        engine.gpu->get_core_device() == shared_device && w.device_ptr) {
      ref.device_data = w.device_ptr;
      ref.device_bytes = w.meta.size_bytes;
    }
    return ref;
  };

  const int shared_inter = moe_inter * c.n_expert_shared;
  auto gate = make_ref(prefix + "ffn_gate_shexp.weight", D, shared_inter);
  auto up = make_ref(prefix + "ffn_up_shexp.weight", D, shared_inter);
  auto down = make_ref(prefix + "ffn_down_shexp.weight", shared_inter, D);
  std::optional<GgmlSharedWeightRef> gate_inp;
  auto gate_inp_it =
      engine.resident_layers.find(prefix + "ffn_gate_inp_shexp.weight");
  if (gate_inp_it != engine.resident_layers.end() &&
      gate_inp_it->second.host_ptr && !gate_inp_it->second.meta.shape.empty()) {
    GgmlSharedWeightRef ref;
    ref.host_data = gate_inp_it->second.host_ptr;
    ref.type = (uint32_t)gate_inp_it->second.meta.type;
    ref.ne0 = (int64_t)gate_inp_it->second.meta.shape[0];
    // Qwen4Exp stores this per-token scalar gate as a rank-1 [hidden]
    // vector.  The shared graph models it as [hidden,1].
    ref.ne1 = gate_inp_it->second.meta.shape.size() >= 2
                  ? (int64_t)gate_inp_it->second.meta.shape[1]
                  : 1;
    ref.nbytes = gate_inp_it->second.meta.size_bytes;
    if (!no_external_weights && engine.gpu &&
        engine.gpu->get_core_device() == shared_device &&
        gate_inp_it->second.device_ptr) {
      ref.device_data = gate_inp_it->second.device_ptr;
      ref.device_bytes = gate_inp_it->second.meta.size_bytes;
    }
    gate_inp = ref;
  }

  auto &runner = GgmlQwenSharedRunner::instance();
  const float clamp = (layer_idx < (int)c.swiglu_clamp_shexp.size())
                          ? c.swiglu_clamp_shexp[(size_t)layer_idx]
                          : 0.0f;
  if (!gate || !up || !down) {
    if (debug)
      LOG_WARN("[GGML-SHARED-DEBUG] layer=%d S=%d missing shared weight "
               "reference gate=%d up=%d down=%d\n",
               layer_idx, S, gate.has_value(), up.has_value(),
               down.has_value());
    return false;
  }
  if (!runner.ensure(shared_device, D, shared_inter, S, gate_inp.has_value(),
                     clamp)) {
    if (debug)
      LOG_WARN("[GGML-SHARED-DEBUG] layer=%d S=%d runner ensure failed "
               "device=%d D=%d inter=%d gate=%d\n",
               layer_idx, S, shared_device, D, shared_inter,
               gate_inp.has_value());
    return false;
  }
  if (!runner.ensure_layer(layer_idx, *gate, *up, *down,
                           gate_inp ? &*gate_inp : nullptr)) {
    if (debug)
      LOG_WARN("[GGML-SHARED-DEBUG] layer=%d S=%d runner ensure_layer failed\n",
               layer_idx, S);
    return false;
  }

  const bool run_ok = host_routed
                          ? runner.run_host_routed(layer_idx, device_input,
                                                   host_routed, combined_output)
                          : runner.run(layer_idx, device_input, routed_output,
                                       combined_output);
  if (!run_ok) {
    if (debug)
      LOG_WARN("[GGML-SHARED-DEBUG] layer=%d S=%d runner %s failed input=%p "
               "routed=%p host_routed=%p\n",
               layer_idx, S, host_routed ? "host-routed run" : "run",
               device_input, routed_output, (const void *)host_routed);
    return false;
  }
  static bool logged = false;
  if (!logged) {
    logged = true;
    LOG_INFO("[GGML-SHARED] Qwen4Exp routed+shared device FFN active%s\n",
             host_routed ? " (CPU routed result handoff)" : "");
  }
  return true;
}

static void run_qwen4exp_shared_host(StreamingEngine &engine, int layer_idx,
                                     int S, int D, int moe_inter,
                                     const float *normed, float *output,
                                     LayerScratch &scratch,
                                     const ArchitectureConfig &c) {
  const std::string prefix = "blk." + std::to_string(layer_idx) + ".";
  if (!engine.resident_layers.count(prefix + "ffn_gate_shexp.weight"))
    return;
  PERF_SCOPE(t_shared_expert_total);
  const int shared_inter = moe_inter * c.n_expert_shared;
  if (!scratch.gate_buf.resize((size_t)S * shared_inter, "shared_gate_buf") ||
      !scratch.up_buf.resize((size_t)S * shared_inter, "shared_up_buf") ||
      !scratch.down_buf.resize((size_t)S * D, "shared_down_buf")) {
    // The routed contribution is already present in output when this helper
    // is used as a fallback.  Keep it intact and let the caller continue
    // with a numerically complete routed-only result rather than dereference
    // a refused/null scratch buffer.
    LOG_WARN("[MEM-GUARD] shared-expert scratch refused at layer=%d; "
             "keeping the routed contribution only\n",
             layer_idx);
    return;
  }
  dense_proj(engine, prefix + "ffn_gate_shexp.weight", normed,
             scratch.gate_buf.data(), S, D, shared_inter);
  dense_proj(engine, prefix + "ffn_up_shexp.weight", normed,
             scratch.up_buf.data(), S, D, shared_inter);
  const float clamp = (layer_idx < (int)c.swiglu_clamp_shexp.size())
                          ? c.swiglu_clamp_shexp[(size_t)layer_idx]
                          : 0.0f;
  if (swiglu_clamp_fused()) {
    swiglu_clamp_fused_inplace(scratch.gate_buf.data(), scratch.up_buf.data(),
                               S * shared_inter, clamp);
  } else {
    silu_inplace(scratch.gate_buf.data(), S * shared_inter);
    elem_mul(scratch.gate_buf.data(), scratch.gate_buf.data(),
             scratch.up_buf.data(), S * shared_inter);
    swiglu_clamp_inplace(scratch.gate_buf.data(), S * shared_inter, clamp);
  }
  dense_proj(engine, prefix + "ffn_down_shexp.weight", scratch.gate_buf.data(),
             scratch.down_buf.data(), S, shared_inter, D);

  std::vector<float> gate((size_t)S, 1.0f);
  if (engine.resident_layers.count(prefix + "ffn_gate_inp_shexp.weight")) {
    dense_proj(engine, prefix + "ffn_gate_inp_shexp.weight", normed,
               gate.data(), S, D, 1);
    for (float &g : gate)
      g = 1.0f / (1.0f + std::exp(-g));
  }
  for (int s = 0; s < S; ++s) {
    float *dst = output + (size_t)s * D;
    const float *src = scratch.down_buf.data() + (size_t)s * D;
    const float g = gate[(size_t)s];
    for (int d = 0; d < D; ++d)
      dst[d] += g * src[d];
  }
  if (std::getenv("MOESTREAM_GGML_SHARED_DEBUG") && S == 1) {
    double l2 = 0.0;
    for (int d = 0; d < D; ++d)
      l2 += (double)output[d] * output[d];
    LOG_WARN("[GGML-SHARED-DEBUG] host output layer=%d S=%d l2=%g "
             "first=%g,%g,%g,%g\n",
             layer_idx, S, std::sqrt(l2), output[0], output[1], output[2],
             output[3]);
  }
  check_finite("batched shared expert down_buf", layer_idx, -1,
               scratch.down_buf.data(), (size_t)S * D);
}

static bool qwen4exp_device_route_enabled() {
  static const bool enabled = [] {
    const char *e = std::getenv("MOESTREAM_QWEN4EXP_DEVICE_ROUTE");
    return e && e[0] == '1';
  }();
  return enabled;
}

static bool qwen4exp_router_pilot_layer_allowed(const StreamingEngine &engine,
                                                int target_layer) {
  // A cpu_affinity layer is a standing CPU-only contract: its experts must
  // not be VRAM-pinned, asynchronously promoted, or used as a target for a
  // Qwen pilot graph.  The ordinary dispatch/preload paths already enforce
  // this through is_cpu_affinity_layer(); keep the Qwen-only advisory path
  // aligned so it does not spend prediction/PCIe work for a layer that can
  // never consume the predicted GPU experts.
  if (is_cpu_affinity_layer(target_layer))
    return false;
  if (!engine.qwen4exp_router_pilot_predict_layers.empty() &&
      std::find(engine.qwen4exp_router_pilot_predict_layers.begin(),
                engine.qwen4exp_router_pilot_predict_layers.end(),
                target_layer) ==
          engine.qwen4exp_router_pilot_predict_layers.end())
    return false;
  return true;
}

// Queue the same prediction without waiting for its compact route table.
// The caller consumes it at the beginning of the predicted layer, after that
// layer's attention work has had a chance to run. This moves the pilot's
// unavoidable route-table synchronization out of the layer-N -> layer-N+1
// boundary instead of merely moving the GEMV between CPU and GPU.
static bool qwen4exp_router_pilot_gpu_async(StreamingEngine &engine,
                                            int target_layer, int D,
                                            int n_experts, int topk,
                                            const void **device_ids,
                                            const void **device_weights) {
  if (!device_ids || !device_weights || target_layer < 0 || n_experts <= 0 ||
      topk <= 0 || !moe_ggml_gpu_router() || !moe_ggml_mul_mat_id() ||
      !engine.qwen4exp_device_hc_ffn_input || !qwen4exp_hc_device_live() ||
      !qwen4exp_router_pilot_layer_allowed(engine, target_layer))
    return false;
  const void *streams = GgmlHcRunner::instance().device_streams_tensor();
  if (!streams)
    return false;
  const std::string router_key =
      "blk." + std::to_string(target_layer) + ".ffn_gate_inp.weight";
  auto router_it = engine.resident_layers.find(router_key);
  if (router_it == engine.resident_layers.end() || !router_it->second.host_ptr)
    return false;
  GgmlRouterWeightRef ref;
  const auto &rw = router_it->second;
  ref.host_data = rw.host_ptr;
  ref.type = (uint32_t)rw.meta.type;
  ref.ne0 = rw.meta.shape.size() > 0 ? (int64_t)rw.meta.shape[0] : 0;
  ref.ne1 = rw.meta.shape.size() > 1 ? (int64_t)rw.meta.shape[1] : 0;
  ref.nbytes = rw.meta.size_bytes;
  const std::string bias_key =
      "blk." + std::to_string(target_layer) + ".exp_probs_b.bias";
  const float *bias = nullptr;
  auto bias_it = engine.resident_layers.find(bias_key);
  if (bias_it != engine.resident_layers.end() && bias_it->second.host_ptr &&
      bias_it->second.meta.type == GGUFType::F32)
    bias = reinterpret_cast<const float *>(bias_it->second.host_ptr);
  auto &router = GgmlRouterRunner::instance();
  const int device = moe_ggml_device() >= 0 ? moe_ggml_device() : 0;
  if (!router.ensure(
          device, D, n_experts, topk, /*n_tokens=*/1,
          engine.model_file.architecture_config.expert_weights_norm,
          engine.model_file.architecture_config.expert_weights_scale) ||
      !router.ensure_layer(target_layer, ref, bias))
    return false;
  if (!router.run_tensor_stream0_device(
          target_layer, streams, /*token_index=*/0, device_ids, device_weights))
    return false;
  // Start the tiny route-table D2H immediately after the prediction graph.
  // The consumer at the next layer polls its completion event and only waits
  // if the intervening attention work did not hide the copy.
  const bool queued = router.queue_route_to_host(target_layer);
  if (queued && target_layer < PerfCounters::kMaxTrackedLayersForMiss) {
    const int64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    perf().qwen_pilot_prediction_ready_ns[target_layer].store(
        now_ns, std::memory_order_release);
  }
  return queued;
}

static void qwen4exp_process_pilot_predictions(
    StreamingEngine &engine, RAMSafetyManager &ram_mgr,
    LockFreePrefetcher &async_io_queue, LayerScratch &scratch, int target_layer,
    int qwen_k) {
  const auto &c = engine.model_file.architecture_config;
  if (target_layer < 0 || qwen_k <= 0 || qwen_k > LayerScratch::kQwenPilotMax ||
      is_cpu_affinity_layer(target_layer))
    return;
  const int64_t prediction_ready_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count();
  if (target_layer < PerfCounters::kMaxTrackedLayersForMiss) {
    const int64_t queued_ns =
        perf().qwen_pilot_prediction_ready_ns[target_layer].exchange(
            0, std::memory_order_acq_rel);
    if (queued_ns > 0 && prediction_ready_ns >= queued_ns) {
      PerfCounters::atomic_add(
          perf().t_qwen_pilot_queue_to_ready,
          (double)(prediction_ready_ns - queued_ns) * 1e-9);
      perf().n_qwen_pilot_queue_to_ready.fetch_add(
          1, std::memory_order_relaxed);
    }
    // This timestamp is the point at which the predicted IDs are known on
    // the host and the prediction decision starts; it is distinct from the
    // earlier GPU route-graph submission above.
    perf().qwen_pilot_prediction_ready_ns[target_layer].store(
        prediction_ready_ns, std::memory_order_release);
  }
  const int qwen_prefetch_k =
      std::min({engine.qwen4exp_router_pilot_prefetch_k > 0
                    ? engine.qwen4exp_router_pilot_prefetch_k
                    : engine.qwen4exp_router_pilot_k,
                qwen_k, LayerScratch::kQwenPilotMax});
  const bool qwen_prefetch_layer_enabled =
      std::find(engine.qwen4exp_router_pilot_prefetch_exclude_layers.begin(),
                engine.qwen4exp_router_pilot_prefetch_exclude_layers.end(),
                target_layer) ==
      engine.qwen4exp_router_pilot_prefetch_exclude_layers.end();
  const bool qwen_prefetch_enabled =
      engine.qwen4exp_router_pilot_prefetch && qwen_prefetch_layer_enabled;
  perf().n_qwen_pilot_prefetch_k_observed.store(
      (uint32_t)(engine.qwen4exp_router_pilot_prefetch ? qwen_prefetch_k : 0),
      std::memory_order_relaxed);
  if (engine.qwen4exp_router_pilot_prefetch && !qwen_prefetch_layer_enabled) {
    perf().n_qwen_pilot_prefetch_excluded.fetch_add(1,
                                                    std::memory_order_relaxed);
  } else if (engine.qwen4exp_router_pilot_prefetch &&
             qwen_prefetch_k < qwen_k) {
    perf().n_qwen_pilot_prefetch_rank_skipped.fetch_add(
        (uint64_t)(qwen_k - qwen_prefetch_k), std::memory_order_relaxed);
  }
  scratch.qwen_pilot_pred_count = qwen_k;
  scratch.qwen_pilot_pred_layer = target_layer;
  scratch.qwen_pilot_prefetch_count =
      qwen_prefetch_enabled ? qwen_prefetch_k : 0;
  perf().n_qwen_pilot_predictions.fetch_add(1, std::memory_order_relaxed);

  for (int k = 0; k < qwen_k; ++k) {
    const int best = scratch.qwen_pilot_pred_ids[k];
    if (best < 0 || best >= c.n_experts)
      continue;
    // Classify without a mutating lookup. The active-layer pool is a real
    // VRAM source but is separate from the main LRU cache.
    bool already_vram =
        ram_mgr.vram_cache &&
        (ram_mgr.vram_cache->active_layer_has(target_layer, best) ||
         ram_mgr.vram_cache->main_layer_has(target_layer, best));
    const bool already_ram =
        !already_vram && ram_mgr.is_ram_resident(target_layer, best);
    const bool qwen_prefetch_this_rank =
        qwen_prefetch_enabled && k < qwen_prefetch_k;
    if (already_vram) {
      perf().n_qwen_pilot_pred_already_vram.fetch_add(
          1, std::memory_order_relaxed);
    } else if (already_ram) {
      perf().n_qwen_pilot_pred_already_ram.fetch_add(1,
                                                     std::memory_order_relaxed);
      if (qwen_prefetch_this_rank) {
        ExpertLocation *issued =
            ram_mgr.promote_ram_to_active_layer(
                engine, target_layer, best, -1, prediction_ready_ns);
        if (issued) {
          perf().n_qwen_pilot_active_prefetch_issued.fetch_add(
              1, std::memory_order_relaxed);
        } else {
          perf().n_qwen_pilot_active_prefetch_failed.fetch_add(
              1, std::memory_order_relaxed);
        }
      }
    } else {
      perf().n_qwen_pilot_pred_needs_fetch.fetch_add(1,
                                                     std::memory_order_relaxed);
      if (qwen_prefetch_this_rank) {
        async_io_queue.push_hint(target_layer, best);
        perf().n_qwen_pilot_hints_pushed.fetch_add(1,
                                                   std::memory_order_relaxed);
      }
    }
    if (qwen_prefetch_this_rank && already_vram) {
      perf().n_qwen_pilot_hints_skipped_resident.fetch_add(
          1, std::memory_order_relaxed);
    }
  }
}

// The grouped Qwen decode graph normally sources routed weights from the main
// VRAM cache/RAM tier, but its explicit allow_active_layer_sources argument
// also lets a completed prediction fill serve the graph through a D2D copy.
// Keep the older generic prewarm mechanisms family-gated so other model
// families retain their existing behavior.
static bool qwen4exp_decode_ggml_path(const StreamingEngine &engine) {
  return engine.qwen4exp_decode_ggml_moe &&
         engine.family_ops.family == ModelFamily::Qwen4Exp;
}

// Per-layer decode timing is intentionally opt-in.  It updates several
// counters once per layer call, so leaving it enabled in a production decode
// would measure the instrumentation itself (and can materially lower the
// single-token throughput).  Enable it only for a diagnostic run with
// MOESTREAM_QWEN_DECODE_LAYER_TIMING=1.
static bool qwen4exp_decode_layer_timing_enabled() {
  static const bool enabled = [] {
    const char *e = std::getenv("MOESTREAM_QWEN_DECODE_LAYER_TIMING");
    return e && e[0] == '1';
  }();
  return enabled;
}

// Decode-only, per-layer wall timer.  The existing aggregate counters tell
// us that the request is slow, but they cannot answer which layers are worth
// moving to CPU: a CPU-affinity layer includes a different cache/transfer
// path than a GPU layer.  Keep this diagnostic local to Qwen4Exp so no other
// model family changes its accounting or behavior.
struct QwenDecodeLayerTimer {
  int layer = -1;
  bool cpu_path = false;
  bool active = false;
  std::chrono::high_resolution_clock::time_point start;

  QwenDecodeLayerTimer(int layer_idx, bool is_cpu, bool enabled)
      : layer(layer_idx), cpu_path(is_cpu), active(enabled) {
    if (active)
      start = std::chrono::high_resolution_clock::now();
  }

  ~QwenDecodeLayerTimer() {
    if (!active || layer < 0 ||
        layer >= PerfCounters::kMaxTrackedLayersForMiss)
      return;
    const auto end = std::chrono::high_resolution_clock::now();
    const double elapsed =
        std::chrono::duration<double>(end - start).count();
    auto &p = perf();
    p.n_qwen_decode_layer[layer].fetch_add(1, std::memory_order_relaxed);
    PerfCounters::atomic_add(p.t_qwen_decode_layer[layer], elapsed);
    PerfCounters::atomic_add(
        cpu_path ? p.t_qwen_decode_cpu_layer[layer]
                 : p.t_qwen_decode_gpu_layer[layer],
        elapsed);
  }
};

// ============================================================
// MoE forward with router, top-k selection, expert execution.
// Supports parallel expert I/O via threading.
// ============================================================
void moe_forward_gguf(StreamingEngine &engine, RAMSafetyManager &ram_mgr,
                      LockFreePrefetcher &async_io_queue, int layer_idx,
                      const float *hidden_states, int S, float *output,
                      LayerScratch &scratch, const int *token_ids) {
  scratch.moe_device_output = nullptr;
  scratch.moe_device_runner = nullptr;
  scratch.moe_shared_device_output = false;
  scratch.moe_device_output_ready = false;
  auto &c = engine.model_file.architecture_config;
  int D = c.hidden;
  int topk = c.topk;
  int n_experts = c.n_experts;
  int moe_inter = c.moe_inter;

  const bool qwen_layer_timing =
      S == 1 && engine.family_ops.family == ModelFamily::Qwen4Exp &&
      qwen4exp_decode_layer_timing_enabled();
  QwenDecodeLayerTimer qwen_decode_layer_timer(
      layer_idx, qwen_layer_timing && is_cpu_affinity_layer(layer_idx),
      qwen_layer_timing);

  // When Qwen4Exp's HC result is already on the GGML CUDA backend, the
  // optional device router fills these in as [topk, tokens].  The existing
  // host-side bookkeeping below still consumes the arrays in its usual
  // [tokens, topk] layout, so the device path changes where routing math is
  // done without changing cache/prefetch behavior.
  bool gpu_router_handled = false;
  bool gpu_route_device = false;
  const void *gpu_route_ids_device = nullptr;
  const void *gpu_route_weights_device = nullptr;
  GgmlRouterRunner *device_router = nullptr;
  std::vector<int32_t> gpu_route_ids_ks;
  std::vector<float> gpu_route_weights_ks;

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
                        ? c.swiglu_clamp_exp[layer_idx]
                        : 0.0f;
  float clamp_shexp = (layer_idx < (int)c.swiglu_clamp_shexp.size())
                          ? c.swiglu_clamp_shexp[layer_idx]
                          : 0.0f;

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
  const int32_t *tid2eid = nullptr;
  // Real on-disk row count (ne[1]) of ffn_gate_tid2eid.weight for THIS
  // layer -- i.e. the number of vocab rows the table actually has.
  // Populated only when `tid2eid` above is populated. ROOT-CAUSE FIX:
  // the per-token lookup below used to bounds-check `tok` against
  // `c.vocab` (the engine's global vocab size, derived from
  // token_embd.weight -- see graph_initializer.cpp's own vocab_size
  // fallback history) under the assumption that it equals this
  // tensor's real ne[1]. Nothing actually enforces that the two agree:
  // if this GGUF (or a separate dense_model_path file, when
  // has_dense_model is set -- ffn_gate_tid2eid.weight loads from
  // whichever of the two `initialize_resident_memory` was given) ever
  // exported this specific table over a SMALLER vocab than c.vocab,
  // `tid2eid + tok*topk` for any real token id past that smaller
  // extent reads past the posix_memalign'd buffer entirely -- an
  // out-of-bounds heap read (UB), not a bug in the file's actual
  // content. That reads as EXACTLY this symptom: every hash-route
  // lookup "succeeds" past the type/shape check below (type IS I32,
  // shape[0] IS topk) yet every row comes back full of nonsense
  // expert ids, for every token, every hash layer -- because every
  // lookup this run happens to land past the table's real extent.
  // Bounding against the tensor's OWN shape[1] instead closes the UB
  // and turns a silent corruption into a clean, diagnosable fallback;
  // the mismatch check right after logs the smoking gun the moment it
  // fires.
  int64_t tid2eid_n_vocab = 0;
  if (layer_idx < c.hash_layer_count) {
    std::string tid2eid_key =
        "blk." + std::to_string(layer_idx) + ".ffn_gate_tid2eid.weight";
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
    const GGUFTensor *meta_ptr = (tid2eid_it != engine.resident_layers.end())
                                     ? &tid2eid_it->second.meta
                                     : nullptr;
    bool shape_ok = meta_ptr && !meta_ptr->shape.empty() &&
                    (int64_t)meta_ptr->shape[0] == (int64_t)topk;
    if (meta_ptr && tid2eid_it->second.host_ptr &&
        meta_ptr->type == GGUFType::I32 && shape_ok) {
      tid2eid = reinterpret_cast<const int32_t *>(tid2eid_it->second.host_ptr);
      tid2eid_n_vocab =
          (meta_ptr->shape.size() >= 2) ? (int64_t)meta_ptr->shape[1] : 0;
      // Loud, PER-LAYER, ONE-SHOT: fires only if the table's real
      // extent actually diverges from c.vocab -- the previous
      // silent-UB path never had a way to surface this at all.
      static std::vector<bool> tid2eid_vocab_mismatch_logged;
      if ((int)tid2eid_vocab_mismatch_logged.size() <= layer_idx)
        tid2eid_vocab_mismatch_logged.resize(layer_idx + 1, false);
      if (!tid2eid_vocab_mismatch_logged[layer_idx] && tid2eid_n_vocab > 0 &&
          tid2eid_n_vocab != (int64_t)c.vocab) {
        LOG_ERROR("[MOE-ROUTE] layer=%d ffn_gate_tid2eid.weight real n_vocab "
                  "(shape[1]=%lld) "
                  "!= engine c.vocab (%d) -- bounding token-id lookups to the "
                  "SMALLER of the "
                  "two from now on; any real token id in that gap can never "
                  "use the hash "
                  "route for this layer and falls back to the learned router "
                  "instead.\n",
                  layer_idx, (long long)tid2eid_n_vocab, c.vocab);
        tid2eid_vocab_mismatch_logged[layer_idx] = true;
      }
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
        LOG_ERROR(
            "[MOE-ROUTE] layer=%d ffn_gate_tid2eid.weight loaded but "
            "type/shape "
            "mismatch (type=%u expected I32=%u, shape[0]=%llu expected "
            "topk=%d) -- "
            "falling back to the normal learned-router top-k for this layer.\n",
            layer_idx, (unsigned)meta_ptr->type, (unsigned)GGUFType::I32,
            meta_ptr->shape.empty() ? 0ull
                                    : (unsigned long long)meta_ptr->shape[0],
            topk);
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
        LOG_ERROR(
            "[MOE-ROUTE] layer=%d is hash-routed (hash_layer_count=%d) but "
            "ffn_gate_tid2eid.weight failed to load (missing/short-read) -- "
            "falling back to the normal learned-router top-k for this layer, "
            "which will select DIFFERENT experts than the real hash table "
            "would.\n",
            layer_idx, c.hash_layer_count);
        tid2eid_warned_per_layer[layer_idx] = true;
      }
    }
  }

  // --- Router: compute expert probs and select top-k. Reference algorithm
  // ported from llama.cpp's build_moe_ffn (llama-graph.cpp):
  //   1) probs = gate_func(logits)     [sigmoid/softmax/sqrtsoftplus, per
  //   c.gating_func -- see apply_gating_activation above] 2) selection_probs =
  //   probs + bias  [bias only shifts WHICH experts get
  //      picked — the DeepSeek-V3/GLM-5.2 e_score_correction_bias tensor]
  //   3) top-k selected BY selection_probs (biased)
  //   4) weights gathered from probs (UNBIASED) at the selected indices
  //   5) optionally normalize the top-k weights to sum to 1
  //   6) optionally scale by expert_weights_scale
  {
    PERF_SCOPE(t_moe_route);
    std::string norm_key =
        "blk." + std::to_string(layer_idx) + ".ffn_norm.weight";
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
        LOG_ERROR("[MOE-ROUTE] layer=%d ffn_norm.weight type=%u (NOT F32) -- "
                  "rmsnorm_row "
                  "reinterprets host_ptr as float* unconditionally; this WILL "
                  "produce garbage.\n",
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
                   reinterpret_cast<float *>(norm_it->second.host_ptr),
                   (size_t)D);
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
      const float *norm_w = reinterpret_cast<float *>(norm_it->second.host_ptr);
      for (int s = 0; s < S; ++s) {
        rmsnorm_row(scratch.normed.data() + (size_t)s * D,
                    hidden_states + (size_t)s * D, norm_w, D, c.eps);
      }
      check_finite(
          "scratch.normed (post ffn_norm rmsnorm, router+expert input)",
          layer_idx, -1, scratch.normed.data(), (size_t)S * D);
    } else {
      std::memcpy(scratch.normed.data(), hidden_states, S * D * sizeof(float));
    }

    std::string router_key =
        "blk." + std::to_string(layer_idx) + ".ffn_gate_inp.weight";
    auto router_it = engine.resident_layers.find(router_key);
    if (router_it == engine.resident_layers.end() ||
        !router_it->second.host_ptr) {
      std::fill(output, output + S * D, 0.0f);
      return;
    }
    // Qwen4Exp's HC path has no ordinary FFN norm: the device HC result is
    // exactly the router input.  Do not key this on `norm_it == end` here.
    // The resident map may contain a metadata/cache entry for ffn_norm even
    // though Qwen's architecture does not consume it; using that incidental
    // map detail left the activation on CUDA while routing from stale host
    // data.  Keep this handoff explicitly Qwen+HC-only; other families and
    // any future norm-bearing variant remain on the established host route.
    // The graph also implements only the Qwen softmax semantics.
    // A CPU-affinity layer is a hard CPU-only contract.  Keep its route
    // table on the established host path even when GPU routing is enabled
    // for the rest of Qwen decode; otherwise the HC input would be sent
    // through a needless GPU router before the CPU expert graph consumes it.
    const bool qwen_decode_gpu_router =
        !is_cpu_affinity_layer(layer_idx) &&
        (S > 1 || engine.qwen4exp_decode_gpu_router);
    if (moe_ggml_gpu_router() && qwen_decode_gpu_router &&
        c.gating_func == GatingFunc::Softmax &&
        engine.family_ops.family == ModelFamily::Qwen4Exp &&
        engine.family_ops.uses_hyper_connections &&
        engine.qwen4exp_device_hc_ffn_input) {
      GgmlRouterWeightRef rref;
      const auto &rw = router_it->second;
      rref.host_data = rw.host_ptr;
      rref.type = (uint32_t)rw.meta.type;
      rref.ne0 = rw.meta.shape.size() > 0 ? (int64_t)rw.meta.shape[0] : 0;
      rref.ne1 = rw.meta.shape.size() > 1 ? (int64_t)rw.meta.shape[1] : 0;
      rref.nbytes = rw.meta.size_bytes;
      const float *route_bias = nullptr;
      std::string bias_key =
          "blk." + std::to_string(layer_idx) + ".exp_probs_b.bias";
      auto route_bias_it = engine.resident_layers.find(bias_key);
      if (route_bias_it != engine.resident_layers.end() &&
          route_bias_it->second.host_ptr &&
          route_bias_it->second.meta.type == GGUFType::F32)
        route_bias =
            reinterpret_cast<const float *>(route_bias_it->second.host_ptr);

      auto &router = GgmlRouterRunner::instance();
      if (router.ensure(/*device=*/moe_ggml_device() >= 0 ? moe_ggml_device()
                                                          : 0,
                        D, n_experts, topk, S, c.expert_weights_norm,
                        c.expert_weights_scale) &&
          router.ensure_layer(layer_idx, rref, route_bias)) {
        const int route_device = moe_ggml_device() >= 0 ? moe_ggml_device() : 0;
        bool all_experts_device_resident =
            qwen4exp_device_route_enabled() && moe_prefill_batched() && S > 1 &&
            !engine.layer_is_unpacked(layer_idx) && ram_mgr.vram_cache;
        if (all_experts_device_resident) {
          for (int e = 0; e < n_experts; ++e) {
            const ExpertLocation *loc =
                ram_mgr.vram_cache->find_expert(layer_idx, e);
            if (!loc || !loc->device_ptr || !loc->upload_ready ||
                loc->gpu_id != route_device) {
              all_experts_device_resident = false;
              break;
            }
          }
          cudaSetDevice(route_device);
        }
        if (all_experts_device_resident) {
          gpu_route_device = router.run_tensor_device(
              layer_idx, engine.qwen4exp_device_hc_ffn_input,
              &gpu_route_ids_device, &gpu_route_weights_device);
          gpu_router_handled = gpu_route_device;
          if (gpu_route_device)
            device_router = &router;
          if (gpu_route_device) {
            static bool logged_device = false;
            if (!logged_device) {
              logged_device = true;
              LOG_INFO("[GGML-ROUTER] Qwen4Exp all-resident device route "
                       "active: IDs/weights stay on CUDA\n");
            }
          }
        }
        if (!gpu_route_device) {
          gpu_route_ids_ks.resize((size_t)topk * S);
          gpu_route_weights_ks.resize((size_t)topk * S);
          gpu_router_handled = router.run_tensor(
              layer_idx, engine.qwen4exp_device_hc_ffn_input,
              gpu_route_ids_ks.data(), gpu_route_weights_ks.data());
        }
        if (gpu_router_handled) {
          static bool logged = false;
          if (!logged) {
            LOG_INFO("[GGML-ROUTER] Qwen4Exp device routing active: "
                     "GPU computes softmax/top-k; compact route tables %s\n",
                     gpu_route_device ? "stay on CUDA for all-resident layers"
                                      : "return to host");
            logged = true;
          }
        }

        // Opt-in correctness oracle for the Qwen GPU router.  The device
        // router is only allowed to replace the established host selection
        // once its IDs and normalized weights agree on the same HC input.
        // This deliberately downloads one [D] row and repeats the host
        // projection/top-k; it is never part of the production path.
        static const bool compare_qwen_router = [] {
          const char *e = std::getenv("MOESTREAM_QWEN_ROUTER_COMPARE");
          return e && e[0] == '1';
        }();
        if (compare_qwen_router && gpu_router_handled && S == 1 &&
            engine.family_ops.family == ModelFamily::Qwen4Exp &&
            engine.qwen4exp_device_hc_ffn_input &&
            gpu_route_ids_ks.size() >= (size_t)topk && topk >= 4) {
          std::vector<float> compare_x((size_t)D);
          if (GgmlHcRunner::instance().copy_pre_mix_to_host(
                  engine.qwen4exp_device_hc_ffn_input, compare_x.data(), 1)) {
            std::vector<float> compare_logits((size_t)n_experts);
            dense_proj(
                engine,
                "blk." + std::to_string(layer_idx) + ".ffn_gate_inp.weight",
                compare_x.data(), compare_logits.data(), 1, D, n_experts);
            std::vector<float> gpu_logits((size_t)n_experts);
            const bool gpu_logits_ok =
                router.copy_logits_to_host(layer_idx, gpu_logits.data());
            float max_raw_logit_abs = 0.0f;
            if (gpu_logits_ok) {
              for (int e = 0; e < n_experts; ++e)
                max_raw_logit_abs = std::max(
                    max_raw_logit_abs, std::abs(gpu_logits[(size_t)e] -
                                                compare_logits[(size_t)e]));
            }
            apply_gating_activation(c.gating_func, compare_logits.data(),
                                    n_experts);
            const std::string bias_key =
                "blk." + std::to_string(layer_idx) + ".exp_probs_b.bias";
            const float *compare_bias = nullptr;
            auto compare_bias_it = engine.resident_layers.find(bias_key);
            if (compare_bias_it != engine.resident_layers.end() &&
                compare_bias_it->second.host_ptr &&
                compare_bias_it->second.meta.type == GGUFType::F32)
              compare_bias = reinterpret_cast<const float *>(
                  compare_bias_it->second.host_ptr);
            std::vector<std::pair<float, int>> compare_scores;
            compare_scores.reserve((size_t)n_experts);
            for (int e = 0; e < n_experts; ++e)
              compare_scores.emplace_back(
                  compare_logits[(size_t)e] +
                      (compare_bias ? compare_bias[e] : 0.0f),
                  e);
            std::partial_sort(
                compare_scores.begin(), compare_scores.begin() + topk,
                compare_scores.end(),
                [](const auto &a, const auto &b) { return a.first > b.first; });
            std::vector<float> compare_weights((size_t)topk);
            float compare_sum = 0.0f;
            for (int k = 0; k < topk; ++k) {
              const int e = compare_scores[(size_t)k].second;
              compare_weights[(size_t)k] = compare_logits[(size_t)e];
              compare_sum += compare_weights[(size_t)k];
            }
            if (c.expert_weights_norm) {
              const float inv_sum =
                  1.0f / std::max(compare_sum, 6.103515625e-5f);
              for (float &w : compare_weights)
                w *= inv_sum;
            }
            if (c.expert_weights_scale != 0.0f)
              for (float &w : compare_weights)
                w *= c.expert_weights_scale;
            int id_mismatch = 0;
            float max_weight_abs = 0.0f;
            for (int k = 0; k < topk; ++k) {
              if (gpu_route_ids_ks[(size_t)k] !=
                  compare_scores[(size_t)k].second)
                ++id_mismatch;
              max_weight_abs = std::max(
                  max_weight_abs, std::abs(gpu_route_weights_ks[(size_t)k] -
                                           compare_weights[(size_t)k]));
            }
            LOG_WARN("[GGML-ROUTER-COMPARE] layer=%d id_mismatch=%d/%d "
                     "max_weight_abs=%g raw_logit_abs=%g gpu_ids=%d,%d,%d,%d "
                     "cpu_ids=%d,%d,%d,%d\n",
                     layer_idx, id_mismatch, topk, max_weight_abs,
                     max_raw_logit_abs, gpu_route_ids_ks[0],
                     gpu_route_ids_ks[1], gpu_route_ids_ks[2],
                     gpu_route_ids_ks[3], compare_scores[0].second,
                     compare_scores[1].second, compare_scores[2].second,
                     compare_scores[3].second);
          } else {
            LOG_WARN("[GGML-ROUTER-COMPARE] layer=%d could not download HC "
                     "input\n",
                     layer_idx);
          }
        }
      }
    }

    // GLM5NEXT native CUDA router. Unlike the optional GGML Qwen route above,
    // this keeps the route table host-visible so the existing RAM/VRAM cache,
    // prefetch, and pilot bookkeeping remain unchanged. The router weight is
    // already in the core GPU's dense pool; only the compact IDs/weights cross
    // back. Hash-routed layers must keep the CPU selection branch because
    // their expert IDs come from ffn_gate_tid2eid rather than learned top-k.
    if (!gpu_router_handled && !tid2eid && engine.glm5next_router_gpu &&
        engine.family_ops.family == ModelFamily::GLM5NEXT && engine.gpu &&
        router_it->second.device_ptr) {
      gpu_route_ids_ks.resize((size_t)topk * S);
      gpu_route_weights_ks.resize((size_t)topk * S);
      const float *route_bias = nullptr;
      std::string bias_key =
          "blk." + std::to_string(layer_idx) + ".exp_probs_b.bias";
      auto route_bias_it = engine.resident_layers.find(bias_key);
      if (route_bias_it != engine.resident_layers.end() &&
          route_bias_it->second.host_ptr &&
          route_bias_it->second.meta.type == GGUFType::F32)
        route_bias =
            reinterpret_cast<const float *>(route_bias_it->second.host_ptr);

      gpu_router_handled = engine.gpu->glm5next_router_gpu_host(
          router_it->second, scratch.normed.data(), route_bias, S, D, n_experts,
          topk, c.gating_func, c.expert_weights_norm, c.expert_weights_scale,
          gpu_route_ids_ks.data(), gpu_route_weights_ks.data());
      if (gpu_router_handled) {
        static bool logged_native_router = false;
        if (!logged_native_router) {
          logged_native_router = true;
          LOG_INFO("[GLM5NEXT-ROUTER] native CUDA routing active: "
                   "matmul/gating/top-k on core GPU, compact route table "
                   "returned to host cache bookkeeping\n");
        }
      } else {
        perf().n_glm5next_router_gpu_fallback_cpu.fetch_add(
            1, std::memory_order_relaxed);
      }
    }

    if (!gpu_router_handled) {
      dense_proj(
          engine, "blk." + std::to_string(layer_idx) + ".ffn_gate_inp.weight",
          scratch.normed.data(), scratch.router_logits.data(), S, D, n_experts);

      // Gating activation: turns raw logits into probs in-place. These are
      // the UNBIASED values later gathered as expert weights.
      for (int s = 0; s < S; ++s) {
        float *row = scratch.router_logits.data() + (size_t)s * n_experts;
        apply_gating_activation(c.gating_func, row, n_experts);
      }
      // No prior diagnostic covered this: expert_ffn's GPU-resident branch
      // only checks each expert's gate/up/down buffers AFTER routing has
      // already picked them and weighted their contribution -- if the
      // router's own dense_proj or the gating math above produces NaN,
      // nothing upstream would ever catch it.
      check_finite("router_logits (post gate_inp matmul + gating)", layer_idx,
                   -1, scratch.router_logits.data(), (size_t)S * n_experts);
    }
  }

  // Real, previously-untimed cost -- see t_moe_topk_select's own comment
  // (perf_timer.hpp) for why this got its own scope: this block (bias add
  // + argsort/partial_sort top-k selection + pilot recall bookkeeping) was
  // NOT inside t_moe_route's PERF_SCOPE above despite that counter's name
  // implying it covered "top-k selection" too. Explicit brace block (not a
  // bare statement) so this scope ends where Phase 1 actually ends, NOT
  // at the end of the whole function -- PERF_SCOPE is RAII, timed until
  // its ENCLOSING scope closes.
  if (!gpu_route_device) {
    {
      PERF_SCOPE(t_moe_topk_select);

      // Selection bias (DeepSeek-V3/GLM-5.2 style expert_selection_bias):
      // present only on some architectures, absent -> selection uses raw probs.
      // GGUF name confirmed against llama-arch.cpp: "blk.N.exp_probs_b.bias"
      // (previously looked up under a name —
      // "ffn_gate_inp.e_score_correction_bias" — that doesn't exist in real
      // GGUFs; see the matching fix in graph_initializer.cpp's preload manifest
      // for the full explanation).
      const float *bias_row = nullptr;
      {
        std::string bias_key =
            "blk." + std::to_string(layer_idx) + ".exp_probs_b.bias";
        auto bias_it = engine.resident_layers.find(bias_key);
        if (bias_it != engine.resident_layers.end() &&
            bias_it->second.host_ptr) {
          bias_row = reinterpret_cast<float *>(bias_it->second.host_ptr);
          check_finite("exp_probs_b.bias (raw)", layer_idx, -1, bias_row,
                       (size_t)n_experts);
        }
      }

      // Zero output accumulator
      std::memset(output, 0, S * D * sizeof(float));

      // --- Phase 1: batch-wide top-k selection (router already knows every
      // token's choice up front, since router_logits above is one batched
      // matmul over all S tokens) — compute once per token, cache the result,
      // and tally how often each expert is picked across the whole batch. ---
      std::fill(scratch.expert_freq.begin(), scratch.expert_freq.end(), 0);
      std::fill(scratch.first_needed_by_token.begin(),
                scratch.first_needed_by_token.end(), INT_MAX);
      for (int s = 0; s < S; ++s) {
        float *token_probs = scratch.router_logits.data() + s * n_experts;
        float sum_w = 0.0f;

        // Records one (expert_id, weight) pick into scratch's per-token
        // arrays and the routing-frequency histograms -- shared by both
        // the hash-routed and normal argsort selection branches below, so
        // the two paths can never diverge in what bookkeeping happens once
        // an expert is actually picked (weight is always the UNBIASED
        // token_probs[id], same convention either way).
        auto record_pick = [&](int k, int id, float w) {
          scratch.tok_topk_weights[s * topk + k] = w;
          scratch.tok_topk_ids[s * topk + k] = id;
          scratch.expert_freq[id]++;
          // EngineConfig::prefill_hint_order_by_earliest_token -- see
          // LayerScratch::first_needed_by_token's own comment. s only
          // ever increases across this loop, so the FIRST time an expert
          // is picked here IS its minimum -- no need to compare/take a
          // min, a plain unconditional check-and-set is correct and
          // cheaper.
          if (scratch.first_needed_by_token[id] == INT_MAX)
            scratch.first_needed_by_token[id] = s;
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
          if (S > 1)
            engine.bump_route_freq_prefill(layer_idx, id);
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
        //
        // Bound against tid2eid_n_vocab (the table's OWN on-disk ne[1]),
        // not c.vocab: they're only guaranteed to agree if this file (or,
        // under has_dense_model, whichever separate file the table
        // actually loaded from) exported the table over the full engine
        // vocab. Using c.vocab here previously let a real token id past
        // the table's real, smaller extent read straight off the end of
        // the posix_memalign'd buffer -- silent heap OOB, not a bounds
        // check -- and come back with plausible-looking but meaningless
        // int32 garbage that just happened to look like an "out-of-range
        // expert id" downstream. tid2eid_n_vocab is 0 when the table
        // failed to load at all, so this also degrades cleanly.
        bool used_hash_route = false;
        if (gpu_router_handled) {
          // The GGML tensor is shaped [topk, tokens], but the 2-D export above
          // writes compact token rows to the host buffer.  MoEStream's routing
          // arrays are [tokens, topk], so consume that row-major export here.
          // The graph has already applied the unbiased probability gather,
          // top-k normalization, and scale.
          for (int k = 0; k < topk; ++k) {
            const size_t i = (size_t)s * topk + k;
            const int id = gpu_route_ids_ks[i];
            if (id < 0 || id >= n_experts) {
              LOG_ERROR(
                  "[GGML-ROUTER] layer=%d returned invalid expert id=%d\n",
                  layer_idx, id);
              return;
            }
            record_pick(k, id, gpu_route_weights_ks[i]);
          }
          used_hash_route = true;
        } else if (tid2eid && token_ids) {
          int tok = token_ids[s];
          if (tok >= 0 && (int64_t)tok < tid2eid_n_vocab) {
            const int32_t *row = tid2eid + (size_t)tok * topk;
            bool ids_valid = true;
            for (int k = 0; k < topk; ++k) {
              if (row[k] < 0 || row[k] >= n_experts) {
                ids_valid = false;
                break;
              }
            }
            if (ids_valid) {
              for (int k = 0; k < topk; ++k)
                record_pick(k, row[k], token_probs[row[k]]);
              used_hash_route = true;
            } else {
              LOG_ERROR(
                  "[MOE-ROUTE] layer=%d token_id=%d: ffn_gate_tid2eid row "
                  "contains "
                  "an out-of-range expert id (n_experts=%d) -- falling "
                  "back to the "
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
          std::partial_sort(scratch.scores.begin(),
                            scratch.scores.begin() + topk,
                            scratch.scores.begin() + n_experts,
                            [](const ExpertScore &a, const ExpertScore &b) {
                              return a.score > b.score;
                            });
          for (int k = 0; k < topk; ++k)
            record_pick(k, scratch.scores[k].id,
                        token_probs[scratch.scores[k].id]);
        }
        if (!gpu_router_handled && c.expert_weights_norm) {
          float inv_sum = 1.0f / std::max(sum_w, 6.103515625e-5f);
          for (int k = 0; k < topk; ++k)
            scratch.tok_topk_weights[s * topk + k] *= inv_sum;
        }
        if (!gpu_router_handled && c.expert_weights_scale != 0.0f) {
          for (int k = 0; k < topk; ++k)
            scratch.tok_topk_weights[s * topk + k] *= c.expert_weights_scale;
        }

        // Router-selection trace: same VERBOSE gate as layers_forward_gguf's
        // [TRACE] hidden-norm dump, but for CONTENT (which experts, what
        // weight) rather than magnitude -- a routing bug (wrong experts
        // picked, e.g. from a bias-tensor or gating-function mixup) would
        // still produce numerically "healthy" norms downstream, since the
        // FFN math itself is correct, just applied to the wrong experts.
        // Only the LAST token (s==S-1) -- that's the one whose output
        // actually feeds the next-token prediction.
        if (s == S - 1 && trace_gate("TRACE-ROUTER")) {
          LogStream ls(LogLevel::VERBOSE);
          ls << "[TRACE l=" << layer_idx << "] router picks:";
          for (int k = 0; k < topk; ++k) {
            ls << " e" << scratch.tok_topk_ids[s * topk + k]
               << "(w=" << scratch.tok_topk_weights[s * topk + k] << ")";
          }
          ls << "\n";
        }

        // UNION coverage check (2026-08-25, measurement only) -- real
        // answer to "if DEPTH-1, DEPTH-2, and ATTN-PILOT are ALL enabled
        // and predicting this layer, is there still a real disk-cold miss
        // when we get here?" Reads all three pending predictions' CURRENT
        // state BEFORE any of their own individual checks below reset their
        // "am I pending" sentinel (pilot_pred_layer / pilot_pred2_slots[
        // layer_idx%2].layer / pilot_attn_pred_layer) -- this block must
        // run FIRST, or a later individual check's own reset would make an
        // earlier-pending prediction look absent here. Doesn't consume/
        // reset anything itself -- read-only, the three checks below are
        // still what actually resets each one.
        if (s == S - 1) {
          static thread_local std::array<int, 24> union_ids;
          int union_n = 0;
          auto add_unique = [&](int id) {
            for (int i = 0; i < union_n; ++i)
              if (union_ids[i] == id)
                return;
            if (union_n < (int)union_ids.size())
              union_ids[union_n++] = id;
          };
          bool any_pending = false;
          if (scratch.pilot_pred_layer == layer_idx) {
            any_pending = true;
            for (int pk = 0; pk < scratch.pilot_pred_count; ++pk)
              add_unique(scratch.pilot_pred_ids[pk]);
          }
          if (scratch.pilot_attn_pred_layer == layer_idx) {
            any_pending = true;
            for (int pk = 0; pk < scratch.pilot_attn_pred_count; ++pk)
              add_unique(scratch.pilot_attn_pred_ids[pk]);
          }
          const auto &union_slot2 = scratch.pilot_pred2_slots[layer_idx % 2];
          if (union_slot2.layer == layer_idx) {
            any_pending = true;
            for (int pk = 0; pk < union_slot2.count; ++pk)
              add_unique(union_slot2.ids[pk]);
          }
          if (any_pending) {
            int overlap = 0;
            for (int i = 0; i < union_n; ++i) {
              for (int k = 0; k < topk; ++k) {
                if (scratch.tok_topk_ids[s * topk + k] == union_ids[i]) {
                  ++overlap;
                  break;
                }
              }
            }
            perf().n_pilot_union_recall_hits.fetch_add(
                (uint64_t)overlap, std::memory_order_relaxed);
            perf().n_pilot_union_recall_checks.fetch_add(
                1, std::memory_order_relaxed);
            perf().n_pilot_union_recall_topk_sum.fetch_add(
                (uint64_t)topk, std::memory_order_relaxed);
            perf().n_pilot_union_pred_size_sum.fetch_add(
                (uint64_t)union_n, std::memory_order_relaxed);
          }
        }

        // EngineConfig::pilot_widen_all_tokens recall/COVERAGE check
        // (MEASUREMENT ONLY) -- two different comparisons against the
        // same widened prediction: (1) accuracy vs the last token's own
        // real picks, apples-to-apples with DEPTH-1's own recall number;
        // (2) coverage vs the WHOLE layer's real unique-expert demand
        // (scratch.expert_freq>0, valid here since every token's record_
        // pick has already run for this layer by s==S-1) -- the number
        // that actually answers "how much of what this layer needs did
        // widening actually address."
        if (s == S - 1 && scratch.pilot_wide_pred_layer == layer_idx) {
          int overlap_last = 0;
          for (int i = 0; i < scratch.pilot_wide_pred_count; ++i) {
            int pred_id = scratch.pilot_wide_pred_ids[i];
            for (int k = 0; k < topk; ++k) {
              if (scratch.tok_topk_ids[s * topk + k] == pred_id) {
                ++overlap_last;
                break;
              }
            }
          }
          int real_unique = 0;
          for (int e = 0; e < n_experts; ++e)
            if (scratch.expert_freq[e] > 0)
              ++real_unique;
          int coverage_hits = 0;
          for (int i = 0; i < scratch.pilot_wide_pred_count; ++i) {
            if (scratch.expert_freq[scratch.pilot_wide_pred_ids[i]] > 0)
              ++coverage_hits;
          }
          perf().n_pilot_wide_recall_hits_last.fetch_add(
              (uint64_t)overlap_last, std::memory_order_relaxed);
          perf().n_pilot_wide_recall_checks.fetch_add(
              1, std::memory_order_relaxed);
          perf().n_pilot_wide_recall_topk_sum.fetch_add(
              (uint64_t)topk, std::memory_order_relaxed);
          perf().n_pilot_wide_pred_size_sum.fetch_add(
              (uint64_t)scratch.pilot_wide_pred_count,
              std::memory_order_relaxed);
          perf().n_pilot_wide_coverage_hits.fetch_add(
              (uint64_t)coverage_hits, std::memory_order_relaxed);
          perf().n_pilot_wide_real_unique_sum.fetch_add(
              (uint64_t)real_unique, std::memory_order_relaxed);
          scratch.pilot_wide_pred_layer =
              -1; // consume -- never let a stale value linger
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
              if (scratch.tok_topk_ids[s * topk + k] == pred_id) {
                matched = true;
                break;
              }
            }
            if (matched)
              ++overlap;
            // Per-rank breakdown -- exact mirror of the DEPTH-2 version
            // further below (see n_pilot_recall_hits_by_rank's own
            // comment, perf_timer.hpp): pilot_pred_ids[k] is already
            // ranked by the pilot's own confidence (repeated argmax,
            // k=0 = highest), so this answers "does l+1's accuracy
            // concentrate in the high-confidence ranks" -- the number
            // that actually decides how many top ranks are safe to
            // promote RAM->VRAM speculatively.
            if (pk < 8) {
              perf().n_pilot_recall_checks_by_rank[pk].fetch_add(
                  1, std::memory_order_relaxed);
              if (matched)
                perf().n_pilot_recall_hits_by_rank[pk].fetch_add(
                    1, std::memory_order_relaxed);
            }
            // Restricted tally: of the picks that were actually
            // fetch-worthy (not already cached, so a real hint got
            // pushed for them), how many turned out correct? This is
            // the decision-relevant number -- see
            // n_pilot_recall_hint_hits's own comment in perf_timer.hpp.
            if (scratch.pilot_pred_was_hint[pk]) {
              ++hint_denom;
              if (matched)
                ++hint_overlap;
              if (pk < 8) {
                perf().n_pilot_recall_hint_checks_by_rank[pk].fetch_add(
                    1, std::memory_order_relaxed);
                if (matched)
                  perf().n_pilot_recall_hint_hits_by_rank[pk].fetch_add(
                      1, std::memory_order_relaxed);
              }
            }
            // Third, disjoint tally: of the picks that were RAM-resident
            // but NOT yet VRAM-resident at prediction time (a completely
            // different population from the fetch-worthy one just above
            // -- see pilot_pred_was_ram_only's own comment), how many
            // matched the real router's later pick? This is the number
            // that actually calibrates a RAM->VRAM speculative-promotion
            // cutoff -- neither the raw per-rank recall above (mixes in
            // already-VRAM and needs-fetch picks too) nor the
            // fetch-worthy number (a disjoint population entirely) can
            // answer it.
            if (pk < 8 && scratch.pilot_pred_was_ram_only[pk]) {
              perf().n_pilot_ram_only_checks_by_rank[pk].fetch_add(
                  1, std::memory_order_relaxed);
              if (matched)
                perf().n_pilot_ram_only_hits_by_rank[pk].fetch_add(
                    1, std::memory_order_relaxed);
            }
          }
          perf().n_pilot_recall_hits.fetch_add((uint64_t)overlap,
                                               std::memory_order_relaxed);
          perf().n_pilot_recall_checks.fetch_add(1, std::memory_order_relaxed);
          perf().n_pilot_recall_topk_sum.fetch_add((uint64_t)topk,
                                                   std::memory_order_relaxed);
          perf().n_pilot_recall_hint_hits.fetch_add((uint64_t)hint_overlap,
                                                    std::memory_order_relaxed);
          perf().n_pilot_recall_hint_checks_denom.fetch_add(
              (uint64_t)hint_denom, std::memory_order_relaxed);
          // Per-layer breakdown (2026-08-25) -- see PerfCounters::
          // n_pilot_recall_hits_by_layer's own comment. Indexed by
          // layer_idx, the TARGET layer this prediction was FOR (not the
          // layer it fired from).
          if (layer_idx >= 0 &&
              layer_idx < PerfCounters::kMaxTrackedLayersForMiss) {
            perf().n_pilot_recall_hits_by_layer[layer_idx].fetch_add(
                (uint32_t)overlap, std::memory_order_relaxed);
            perf().n_pilot_recall_topk_sum_by_layer[layer_idx].fetch_add(
                (uint32_t)topk, std::memory_order_relaxed);
            perf().n_pilot_recall_checks_by_layer[layer_idx].fetch_add(
                1, std::memory_order_relaxed);
          }
          scratch.pilot_pred_layer = -1;
        }

        // ATTN-PILOT recall check (2026-08-25, measurement only) -- exact
        // mirror of the DEPTH-1 check above (same single-scalar survival
        // pattern), comparing scratch.pilot_attn_pred_ids against this same
        // real router pick. See the firing site's own comment (moe_forward_
        // gguf, right after the attention-residual check_finite call) for
        // the full "answers two questions with one signal" design and how
        // to read this against DEPTH-1's and DEPTH-2's own numbers.
        if (s == S - 1 && scratch.pilot_attn_pred_layer == layer_idx) {
          int overlap = 0;
          int hint_overlap = 0, hint_denom = 0;
          for (int pk = 0; pk < scratch.pilot_attn_pred_count; ++pk) {
            int pred_id = scratch.pilot_attn_pred_ids[pk];
            bool matched = false;
            for (int k = 0; k < topk; ++k) {
              if (scratch.tok_topk_ids[s * topk + k] == pred_id) {
                matched = true;
                break;
              }
            }
            if (matched)
              ++overlap;
            if (pk < 8) {
              perf().n_pilot_attn_recall_checks_by_rank[pk].fetch_add(
                  1, std::memory_order_relaxed);
              if (matched)
                perf().n_pilot_attn_recall_hits_by_rank[pk].fetch_add(
                    1, std::memory_order_relaxed);
            }
            if (scratch.pilot_attn_pred_was_hint[pk]) {
              ++hint_denom;
              if (matched)
                ++hint_overlap;
              if (pk < 8) {
                perf().n_pilot_attn_recall_hint_checks_by_rank[pk].fetch_add(
                    1, std::memory_order_relaxed);
                if (matched)
                  perf().n_pilot_attn_recall_hint_hits_by_rank[pk].fetch_add(
                      1, std::memory_order_relaxed);
              }
            }
          }
          perf().n_pilot_attn_recall_hits.fetch_add((uint64_t)overlap,
                                                    std::memory_order_relaxed);
          perf().n_pilot_attn_recall_checks.fetch_add(
              1, std::memory_order_relaxed);
          perf().n_pilot_attn_recall_topk_sum.fetch_add(
              (uint64_t)topk, std::memory_order_relaxed);
          perf().n_pilot_attn_recall_hint_hits.fetch_add(
              (uint64_t)hint_overlap, std::memory_order_relaxed);
          perf().n_pilot_attn_recall_hint_checks_denom.fetch_add(
              (uint64_t)hint_denom, std::memory_order_relaxed);
          // Per-layer breakdown -- see DEPTH-1's own identical block above
          // and PerfCounters::n_pilot_attn_recall_hits_by_layer's comment.
          if (layer_idx >= 0 &&
              layer_idx < PerfCounters::kMaxTrackedLayersForMiss) {
            perf().n_pilot_attn_recall_hits_by_layer[layer_idx].fetch_add(
                (uint32_t)overlap, std::memory_order_relaxed);
            perf().n_pilot_attn_recall_topk_sum_by_layer[layer_idx].fetch_add(
                (uint32_t)topk, std::memory_order_relaxed);
            perf().n_pilot_attn_recall_checks_by_layer[layer_idx].fetch_add(
                1, std::memory_order_relaxed);
          }
          scratch.pilot_attn_pred_layer = -1;
        }

        // DEPTH-2 recall check (approach (a), measurement only) -- exact
        // mirror of the depth-1 check above, comparing the slot for THIS
        // parity (LayerScratch::pilot_pred2_slots[layer_idx % 2] -- see
        // that array's own comment for why it's two slots, not one)
        // against this same real router pick. Also buckets each pick's
        // outcome by ITS OWN RANK (pk, 0..7) into the *_by_rank counters
        // -- see n_pilot2_recall_hits_by_rank's own comment in
        // perf_timer.hpp for why: pilot_pred_ids/pilot_pred2_ids[k] is
        // already ranked by the pilot's own confidence (repeated argmax,
        // k=0 = highest), so this answers "does accuracy concentrate in
        // the high-confidence ranks" separately from the aggregate
        // overlap counters above, which only sum across the whole top-K
        // and can't distinguish "flat accuracy across all 8 ranks" from
        // "rank 0-2 are great, rank 5-7 are noise."
        {
          auto &slot2 = scratch.pilot_pred2_slots[layer_idx % 2];
          if (s == S - 1 && slot2.layer == layer_idx) {
            int overlap = 0;
            int hint_overlap = 0, hint_denom = 0;
            for (int pk = 0; pk < slot2.count; ++pk) {
              int pred_id = slot2.ids[pk];
              bool matched = false;
              for (int k = 0; k < topk; ++k) {
                if (scratch.tok_topk_ids[s * topk + k] == pred_id) {
                  matched = true;
                  break;
                }
              }
              if (matched)
                ++overlap;
              if (pk < 8) {
                perf().n_pilot2_recall_checks_by_rank[pk].fetch_add(
                    1, std::memory_order_relaxed);
                if (matched)
                  perf().n_pilot2_recall_hits_by_rank[pk].fetch_add(
                      1, std::memory_order_relaxed);
              }
              if (slot2.was_hint[pk]) {
                ++hint_denom;
                if (matched)
                  ++hint_overlap;
                if (pk < 8) {
                  perf().n_pilot2_recall_hint_checks_by_rank[pk].fetch_add(
                      1, std::memory_order_relaxed);
                  if (matched)
                    perf().n_pilot2_recall_hint_hits_by_rank[pk].fetch_add(
                        1, std::memory_order_relaxed);
                }
              }
            }
            perf().n_pilot2_recall_hits.fetch_add((uint64_t)overlap,
                                                  std::memory_order_relaxed);
            perf().n_pilot2_recall_checks.fetch_add(1,
                                                    std::memory_order_relaxed);
            perf().n_pilot2_recall_topk_sum.fetch_add(
                (uint64_t)topk, std::memory_order_relaxed);
            perf().n_pilot2_recall_hint_hits.fetch_add(
                (uint64_t)hint_overlap, std::memory_order_relaxed);
            perf().n_pilot2_recall_hint_checks_denom.fetch_add(
                (uint64_t)hint_denom, std::memory_order_relaxed);
            // Per-layer breakdown -- see DEPTH-1's own identical block
            // above and PerfCounters::n_pilot2_recall_hits_by_layer's
            // comment. layer_idx here IS the target layer (l+2) this
            // prediction was for, same convention as DEPTH-1/ATTN-PILOT
            // (this check only fires when the loop reaches slot2.layer).
            if (layer_idx >= 0 &&
                layer_idx < PerfCounters::kMaxTrackedLayersForMiss) {
              perf().n_pilot2_recall_hits_by_layer[layer_idx].fetch_add(
                  (uint32_t)overlap, std::memory_order_relaxed);
              perf().n_pilot2_recall_topk_sum_by_layer[layer_idx].fetch_add(
                  (uint32_t)topk, std::memory_order_relaxed);
              perf().n_pilot2_recall_checks_by_layer[layer_idx].fetch_add(
                  1, std::memory_order_relaxed);
            }
            slot2.layer = -1;
          }
        }
      }
    } // end t_moe_topk_select scope (Phase 1)
  }

  // Qwen4Exp pilot recall is checked at the shared host route-table point,
  // after the real IDs have been populated and before any expert dispatch.
  // Decode and the normal Qwen GPU-router fallback both use this compact host
  // route. The all-resident device-only prefill route has no host IDs here and
  // therefore contributes no false measurement; its production behavior is
  // unchanged.
  if (!gpu_route_device && S == 1 &&
      engine.family_ops.family == ModelFamily::Qwen4Exp &&
      layer_idx >= 0 && layer_idx < PerfCounters::kMaxTrackedLayersForMiss) {
    const int64_t prediction_ready_ns =
        perf().qwen_pilot_prediction_ready_ns[layer_idx].exchange(
            0, std::memory_order_acq_rel);
    if (prediction_ready_ns > 0) {
      const int64_t now_ns =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now().time_since_epoch())
              .count();
      if (now_ns >= prediction_ready_ns) {
        PerfCounters::atomic_add(
            perf().t_qwen_pilot_ready_to_router,
            (double)(now_ns - prediction_ready_ns) * 1e-9);
        perf().n_qwen_pilot_ready_to_router.fetch_add(
            1, std::memory_order_relaxed);
      }
      // Keep both origins so the later get_experts_batch boundary can report
      // the complete prediction-ready -> DDR-fetch-start gap.
      perf().qwen_pilot_prediction_for_fetch_ns[layer_idx].store(
          prediction_ready_ns, std::memory_order_release);
      perf().qwen_pilot_router_ready_ns[layer_idx].store(
          now_ns, std::memory_order_release);
    }
  }
  if (!gpu_route_device && scratch.qwen_pilot_pred_layer == layer_idx &&
      S > 0) {
    const int s = S - 1;
    int overlap = 0;
    int prefetch_overlap = 0;
    const int prefetch_count = std::min(scratch.qwen_pilot_prefetch_count,
                                        scratch.qwen_pilot_pred_count);
    const bool track_qwen_layer =
        layer_idx >= 0 && layer_idx < PerfCounters::kMaxTrackedLayersForMiss;
    for (int pk = 0; pk < scratch.qwen_pilot_pred_count; ++pk) {
      const int predicted = scratch.qwen_pilot_pred_ids[pk];
      bool matched = false;
      for (int k = 0; k < topk; ++k) {
        if (scratch.tok_topk_ids[(size_t)s * topk + k] == predicted) {
          matched = true;
          break;
        }
      }
      if (matched)
        ++overlap;
      if (pk < prefetch_count) {
        perf().n_qwen_pilot_prefetch_rank_checks[pk].fetch_add(
            1, std::memory_order_relaxed);
        if (matched) {
          ++prefetch_overlap;
          perf().n_qwen_pilot_prefetch_rank_hits[pk].fetch_add(
              1, std::memory_order_relaxed);
        }
      }
      if (pk < PerfCounters::kQwenPilotMaxRanks) {
        perf().n_qwen_pilot_rank_checks[pk].fetch_add(
            1, std::memory_order_relaxed);
        if (matched)
          perf().n_qwen_pilot_rank_hits[pk].fetch_add(
              1, std::memory_order_relaxed);
        if (track_qwen_layer) {
          perf().n_qwen_pilot_rank_checks_by_layer[layer_idx][pk].fetch_add(
              1, std::memory_order_relaxed);
          if (matched)
            perf().n_qwen_pilot_rank_hits_by_layer[layer_idx][pk].fetch_add(
                1, std::memory_order_relaxed);
        }
      }
    }
    const uint64_t predicted_slots = (uint64_t)scratch.qwen_pilot_pred_count;
    perf().n_qwen_pilot_recall_hits.fetch_add((uint64_t)overlap,
                                              std::memory_order_relaxed);
    perf().n_qwen_pilot_recall_checks.fetch_add(1, std::memory_order_relaxed);
    perf().n_qwen_pilot_recall_pred_size_sum.fetch_add(
        predicted_slots, std::memory_order_relaxed);
    if (track_qwen_layer) {
      perf().n_qwen_pilot_recall_hits_by_layer[layer_idx].fetch_add(
          (uint32_t)overlap, std::memory_order_relaxed);
      perf().n_qwen_pilot_recall_pred_size_sum_by_layer[layer_idx].fetch_add(
          (uint32_t)predicted_slots, std::memory_order_relaxed);
      perf().n_qwen_pilot_recall_checks_by_layer[layer_idx].fetch_add(
          1, std::memory_order_relaxed);
      if (prefetch_count > 0) {
        perf().n_qwen_pilot_prefetch_recall_hits_by_layer[layer_idx].fetch_add(
            (uint32_t)prefetch_overlap, std::memory_order_relaxed);
        perf()
            .n_qwen_pilot_prefetch_recall_pred_size_sum_by_layer[layer_idx]
            .fetch_add((uint32_t)prefetch_count, std::memory_order_relaxed);
        perf()
            .n_qwen_pilot_prefetch_recall_checks_by_layer[layer_idx]
            .fetch_add(1, std::memory_order_relaxed);
      }
    }
    if (prefetch_count > 0) {
      perf().n_qwen_pilot_prefetch_recall_hits.fetch_add(
          (uint64_t)prefetch_overlap, std::memory_order_relaxed);
      perf().n_qwen_pilot_prefetch_recall_checks.fetch_add(
          1, std::memory_order_relaxed);
      perf().n_qwen_pilot_prefetch_recall_pred_size_sum.fetch_add(
          (uint64_t)prefetch_count, std::memory_order_relaxed);
    }
    scratch.qwen_pilot_pred_layer = -1;
    scratch.qwen_pilot_pred_count = 0;
  }

  // --- Prefill sliding-window RAM capacity, P2 (2026-08-30) ------------
  // EngineConfig::prefill_window_cap. Placed HERE, immediately after
  // Phase 1 and BEFORE Phase 2 pushes this layer's prefetch hints, for
  // the whole reason the feature exists: Phase 2's hints are what
  // overrun the RAM tier's per-layer cap and make the prefetcher evict
  // its own not-yet-consumed work (82.2% measured untouched-waste), so
  // the cap has to already be raised by the time those hints go out.
  //
  // The demand number is EXACT, not a prediction: Phase 1 above just ran
  // the real batched router over every token in the batch, so
  // expert_freq>0 is precisely the set of experts this layer will
  // actually dispatch -- no speculation, no accuracy question, unlike
  // every cross-layer pilot mechanism in this file.
  //
  // Prefill-only by construction (S > 1). Decode keeps the uniform cap:
  // there, every layer genuinely IS live simultaneously (each token
  // traverses all of them), which is exactly the assumption
  // calculate_safe_cache_capacity's divide-by-n_layers already encodes
  // correctly.
  if (!gpu_route_device && S > 1 && ram_mgr.prefill_window_enabled) {
    int unique_needed = 0;
    for (int e = 0; e < n_experts; ++e)
      if (scratch.expert_freq[e] > 0)
        ++unique_needed;
    ram_mgr.set_prefill_window_cap(engine, layer_idx, unique_needed);
  }

  // MEASUREMENT ONLY, prefill-only -- see PerfCounters::
  // n_moe_bucket_unique_experts's own comment for the full motivation
  // (the llama.cpp -ub/mul_mat_id batching investigation). scratch.
  // expert_freq is already fully populated for this layer's whole
  // S-token batch by the Phase 1 loop just above -- expert_freq[e] IS
  // the real bucket size a batched-per-expert dispatch would see for
  // expert e this layer, and sum(expert_freq)==S*topk is today's real
  // per-(token,k) dispatch count. Read-only tally, zero effect on
  // dispatch/caching. Decode (S==1) is excluded on purpose: every
  // bucket there is trivially size <=1 by construction (one token),
  // so there's nothing to measure.
  if (!gpu_route_device && S > 1) {
    int unique_experts = 0;
    int max_bucket = 0;
    for (int e = 0; e < n_experts; ++e) {
      int f = scratch.expert_freq[e];
      if (f > 0) {
        ++unique_experts;
        if (f > max_bucket)
          max_bucket = f;
      }
    }
    int total_dispatches = S * topk;
    perf().n_moe_bucket_unique_experts.fetch_add((uint64_t)unique_experts,
                                                 std::memory_order_relaxed);
    perf().n_moe_bucket_total_dispatches.fetch_add((uint64_t)total_dispatches,
                                                   std::memory_order_relaxed);
    perf().n_moe_bucket_layers_sampled.fetch_add(1, std::memory_order_relaxed);
    double reduction = unique_experts > 0
                           ? (double)total_dispatches / (double)unique_experts
                           : 0.0;
    LOG_INFO("[MOE-BUCKET] layer=%d tokens=%d topk=%d total_dispatches=%d "
             "unique_experts=%d "
             "max_bucket=%d reduction=%.2fx\n",
             layer_idx, S, topk, total_dispatches, unique_experts, max_bucket,
             reduction);
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
  if (!gpu_route_device && S > 1 && ram_mgr.vram_cache &&
      !is_cpu_affinity_layer(layer_idx)) {
    PERF_START(prefill_warm);
    std::vector<int> candidates;
    candidates.reserve(n_experts);
    for (int e = 0; e < n_experts; ++e)
      if (scratch.expert_freq[e] > 0)
        candidates.push_back(e);
    // EngineConfig::prefill_hint_order_by_earliest_token -- see that
    // field's own comment for the motivation. Same candidate SET
    // either way (every expert with expert_freq>0); only the PUSH
    // ORDER changes.
    if (engine.prefill_hint_order_by_earliest_token) {
      std::sort(candidates.begin(), candidates.end(), [&](int a, int b) {
        return scratch.first_needed_by_token[a] <
               scratch.first_needed_by_token[b];
      });
    } else {
      std::sort(candidates.begin(), candidates.end(), [&](int a, int b) {
        return scratch.expert_freq[a] > scratch.expert_freq[b];
      });
    }

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
      hint_mode_gpu_capable =
          engine.expert_layer_geometry(layer_idx, p2_gb, p2_ub, p2_db, p2_gM,
                                       p2_uM, p2_dM) &&
          gpu_type_supported(p2_gM->type) && gpu_type_supported(p2_uM->type) &&
          gpu_type_supported(p2_dM->type);
    }

    // Active-layer BATCH pre-warm (async issue + one shared sync) --
    // a SEPARATE pool from the main vram_cache pre-warm loop below.
    // GLM's dominant cold-miss dispatch path is upload_expert_active
    // (find_active_expert/upload_expert_active), not the main cache,
    // so THIS pool is what actually targets active_up's real cost
    // (2026-08-12 prefill investigation, see CLAUDE.md: active_up was
    // ~29% of prefill's own wall time, the single largest bucket).
    // No resource-competition risk with the main-cache loop below
    // (unlike prefill_opportunistic_promote's reverted mistake) --
    // these are two independent VRAM pools with no shared gate.
    // Scoped to THIS layer only, entirely inside Phase 2 -- by the
    // time Phase 3 begins a few lines below, sync_active_layer_batch
    // has already confirmed every issued copy landed (load-bearing
    // ordering, not just tidy -- see upload_expert_active_async's own
    // correctness contract in gpu_topology.hpp). Anything not covered
    // here (not yet RAM-resident, active-layer area unavailable/full,
    // or GPU-incapable) falls through to Phase 3's existing reactive
    // path unchanged -- never a correctness issue, only a missed-
    // overlap opportunity for that specific expert.
    // The mmap stream acquires the current layer's exact expert set directly
    // in Phase 3b, so active-layer VRAM prewarm would duplicate those H2D
    // transfers and is guaranteed to be reclaimed without a dispatch hit.
    // Keep the prefetch-hint loop below active: its RAM reads warm the OS
    // page cache that mmap staging depends on, as a real A/B demonstrated.
    const bool prefill_stream_bypasses_active_prewarm =
        moe_prefill_stream() && engine.model_file.mmap_active();
    if (!prefill_stream_bypasses_active_prewarm &&
        !qwen4exp_decode_ggml_path(engine) &&
        ram_mgr.vram_cache->active_layer_capacity > 0) {
      std::vector<ExpertLocation *> active_prewarm_batch;
      std::vector<MoEExpertBlock *> active_prewarm_pinned_blocks;
      const size_t active_capacity =
          (size_t)ram_mgr.vram_cache->active_layer_capacity;
      active_prewarm_batch.reserve(
          std::min(candidates.size(), active_capacity));
      active_prewarm_pinned_blocks.reserve(
          std::min(candidates.size(), active_capacity));
      for (int expert_id : candidates) {
        // Phase 2 is a best-effort warmup, not a second unbounded dispatch
        // queue. Once every active-layer slot has an issued copy, stop
        // asking the topology for more victims. The old loop attempted all
        // 39-72 unique experts in a prefill layer with a 24-slot pool,
        // generating a warning for every impossible upload and needlessly
        // contending on cache_mutex. Phase 3 still handles the remaining
        // experts through its normal synchronous path.
        if (active_prewarm_batch.size() >= active_capacity)
          break;
        // Side-effect-free peek (active_layer_has), NOT
        // find_active_expert -- see that function's own comment
        // for why calling the real, hit-count-bumping/promotion-
        // triggering lookup speculatively here would corrupt the
        // exact state this batch is simultaneously choosing
        // victims from.
        if (ram_mgr.vram_cache->active_layer_has(layer_idx, expert_id))
          continue;
        if (ram_mgr.vram_cache->find_expert(layer_idx, expert_id))
          continue; // main-cache hit already covers it
        // Non-blocking: only pre-warm what's ALREADY RAM-resident
        // (from phase_b's own prefetch coverage) -- never trigger
        // a fresh disk read here, that would defeat the point.
        const uint8_t *ready_slab = nullptr;
        // The H2D below is asynchronous. Keep the exact RAM slot pinned from
        // this atomic lookup until sync_active_layer_batch confirms that the
        // DMA has finished; a plain find-then-async-copy leaves an eviction /
        // slab-regrow window and can upload another expert's bytes.
        MoEExpertBlock *ready = ram_mgr.find_ram_slot_if_ready_held(
            engine.layers_experts[layer_idx],
            (size_t)ram_mgr.cap_for_layer(layer_idx), layer_idx, expert_id,
            &ready_slab, nullptr);
        if (!ready) {
          perf().n_active_layer_prewarm_not_ram_ready.fetch_add(
              1, std::memory_order_relaxed);
          continue;
        }
        bool ready_gpu_capable =
            gpu_type_supported(ready->gate_proj.meta.type) &&
            gpu_type_supported(ready->up_proj.meta.type) &&
            gpu_type_supported(ready->down_proj.meta.type);
        if (!ready_gpu_capable) {
          ram_mgr.ram_unpin(*ready);
          continue;
        }
        // 2026-08-27: real coverage measurement -- see
        // n_active_layer_upload_src_pinned's own comment
        // (perf_timer.hpp).
        if (ready->slab_pinned)
          perf().n_active_layer_upload_src_pinned.fetch_add(
              1, std::memory_order_relaxed);
        else {
          perf().n_active_layer_upload_src_unpinned.fetch_add(
              1, std::memory_order_relaxed);
          perf().mark_unpinned_touch(layer_idx, expert_id);
        }
        // Real total, not slab_capacity -- see decode_async_prewarm's own
        // fix comment for the full incident.
        ExpertLocation *v = ram_mgr.vram_cache->upload_expert_active_async(
            layer_idx, expert_id, ready_slab,
            ready->gate_proj.meta.size_bytes + ready->up_proj.meta.size_bytes +
                ready->down_proj.meta.size_bytes);
        // 2026-08-30 diagnostic -- see MoEExpertBlock::consumed_by_
        // vram_upload (engine.hpp). Phase 2's own active-layer
        // pre-warm is the single largest suspected contributor to
        // the ambiguous 86.6% "Unified RAM waste" figure: it reads
        // this RAM slot's bytes straight into the active-layer VRAM
        // pool, and Phase 3's dispatch then resolves out of THAT
        // pool via find_active_expert -- so the RAM slot is
        // genuinely consumed but its own hit_count is never bumped,
        // making it indistinguishable at eviction time from a slot
        // nobody ever read.
        if (v) {
          active_prewarm_batch.push_back(v);
          active_prewarm_pinned_blocks.push_back(ready);
          ram_mgr.mark_consumed_by_vram_upload(*ready);
        } else {
          // No destination slot was claimed, so the source hold is no longer
          // needed. Do not leave an otherwise healthy RAM slot unevictable.
          ram_mgr.ram_unpin(*ready);
        }
      }
      if (!active_prewarm_batch.empty()) {
        ram_mgr.vram_cache->sync_active_layer_batch(active_prewarm_batch);
      }
      for (MoEExpertBlock *ready : active_prewarm_pinned_blocks)
        ram_mgr.ram_unpin(*ready);
    }

    for (int expert_id : candidates) {
      if (ram_mgr.vram_cache->find_expert(layer_idx, expert_id))
        continue; // already resident

      if (engine.prefill_warm_via_prefetcher) {
        // Async path (config prefill_warm_via_prefetcher=true): push
        // a hint to the SAME queue/worker-thread pipeline decode's
        // own PILOT prediction already uses, instead of loading
        // synchronously right here. Trade-off documented on the
        // config field itself -- this legitimately might not finish
        // in time for THIS layer's own Phase 3 dispatch a moment
        // later, falling through to the ordinary (already-correct)
        // miss path for this call; never a correctness issue.
        //
        // Deliberately NOT gated on vram_cache->has_free_slot() the
        // way the legacy synchronous path below still is (2026-08-
        // 12 fix). A pushed hint only ever populates the RAM tier
        // -- active_prefetch (async_prefetcher.hpp) never touches
        // VRAM from a prefetch-worker thread (see claim_or_wait's
        // own CRITICAL rule) -- so it doesn't need a free VRAM
        // slot at all, only RAM-tier capacity (host_ram_gb,
        // currently 120GB, far less contended than the ~1400-slot
        // VRAM cache that saturates partway through nearly every
        // prefill run). Gating the hint-push on VRAM capacity it
        // doesn't need was needlessly conservative: real per-
        // request [PREFILL] instrumentation showed phase_b (real,
        // decode-thread-BLOCKING disk reads inside Phase 3's own
        // miss handling) costing ~26% of prefill's own wall time
        // -- exactly the cost this gate creates, by refusing to
        // even START a background RAM fetch for any layer whose
        // candidates come up once VRAM happens to be full,
        // forcing that layer's real Phase-3 miss to pay for a
        // synchronous disk read from scratch instead of finding
        // it already RAM-resident. No VRAM-competition risk here
        // (unlike prefill_opportunistic_promote, which DID
        // compete for VRAM capacity and was reverted for exactly
        // that reason) -- this only ever adds queue pressure on
        // the prefetcher's own worker threads, not VRAM slots.
        // UNVERIFIED on real hardware -- watch "Prefetch pickup"
        // (avg ms/backlog) for a WORSE queue backlog than before
        // (this could push MORE total hints than previously,
        // since the loop no longer breaks early), which would
        // mean this trades one bottleneck for another rather than
        // removing it. See CLAUDE.md.
        if (!hint_mode_gpu_capable)
          continue;
        async_io_queue.push_hint(layer_idx, expert_id);
        perf().n_prefill_hint_pushed.fetch_add(1, std::memory_order_relaxed);
        // Track for cross-layer opportunistic VRAM promotion (see
        // LayerScratch::pending_prefill_promotions's own comment,
        // and the follow-up pass above). Silently drops the newest
        // hint if the small pending list is already full rather
        // than growing unbounded -- Phase 3's synchronous dispatch
        // is always the correctness backstop either way, so a
        // dropped tracking entry only forgoes the OPPORTUNISTIC
        // early promotion for that one expert, never correctness.
        if (!engine.prefill_opportunistic_promote) {
          // Not tracked at all -- see EngineConfig::prefill_
          // opportunistic_promote's own comment for the A/B this
          // gate exists to enable.
        } else if (scratch.pending_prefill_count <
                   LayerScratch::PENDING_PREFILL_CAP) {
          scratch.pending_prefill_promotions[scratch.pending_prefill_count++] =
              {layer_idx, expert_id, 0};
        } else {
          perf().n_prefill_pending_cap_dropped.fetch_add(
              1, std::memory_order_relaxed);
        }
        continue;
      }

      // ---- Legacy synchronous path (default, unchanged behavior) ----
      // has_free_slot() gate lives HERE now, not at the top of the
      // loop -- this path DOES promote directly into VRAM
      // (upload_expert_async below), so it genuinely needs a free
      // slot, unlike the hint-push path above. See that branch's own
      // comment for the full reasoning.
      if (!ram_mgr.vram_cache->has_free_slot())
        break; // cache warm now — stop preloading
      // This path queues an asynchronous main-cache H2D directly from the
      // RAM slab. Request an atomic caller hold from get_expert so a
      // prefetch worker cannot evict/regrow the source between the lookup
      // and upload_expert_async(). The promotion bookkeeping below transfers
      // protection to pending_copy_event once the DMA has been recorded.
      MoEExpertBlock *blk = ram_mgr.get_expert(engine, layer_idx, expert_id,
                                               /*pin_for_caller=*/true);
      if (!blk || !blk->slab_buffer) {
        if (blk)
          ram_mgr.ram_unpin(*blk);
        continue;
      }

      // Same all-three-projections-must-be-GPU-supported gate as
      // RAMSafetyManager::get_expert's steady-state promotion — a
      // Dynamic-quant model where down uses an unimplemented format
      // (e.g. IQ3_XXS) can never dispatch to GPU regardless of VRAM
      // residency, so warming those into VRAM only wastes a free slot
      // that a genuinely GPU-capable expert could use later.
      bool gpu_capable = gpu_type_supported(blk->gate_proj.meta.type) &&
                         gpu_type_supported(blk->up_proj.meta.type) &&
                         gpu_type_supported(blk->down_proj.meta.type);
      if (!gpu_capable) {
        ram_mgr.ram_unpin(*blk);
        continue;
      }

      LOG_DEBUG("[MOE-VRAM]  WARM    layer=%d expert=%d freq=%d/%d tokens "
                "(cold-start prefill, free slot)\n",
                layer_idx, expert_id, scratch.expert_freq[expert_id], S);
      // FIX (real, measured "[TIER-OVERLAP]" leak -- see CLAUDE.md's
      // "third, independent tier-overlap source" entry): this call
      // used to bypass get_expert's own MOVE-vs-COPY promotion
      // bookkeeping entirely, since it's a direct external
      // upload_expert_async call, not a hit routed through
      // get_expert's internal Step-1 branch. apply_promotion_
      // bookkeeping mirrors that same logic for this call site.
      perf().n_prefill_warm_attempted.fetch_add(1, std::memory_order_relaxed);
      // Real total, not slab_capacity -- see decode_async_prewarm's own
      // fix comment for the full incident.
      ExpertLocation *promoted = ram_mgr.vram_cache->upload_expert_async(
          layer_idx, expert_id, blk->slab_buffer,
          blk->gate_proj.meta.size_bytes + blk->up_proj.meta.size_bytes +
              blk->down_proj.meta.size_bytes,
          -1, blk->slab_pinned);
      if (promoted)
        perf().n_prefill_warm_succeeded.fetch_add(1, std::memory_order_relaxed);
      else
        perf().n_prefill_warm_failed.fetch_add(1, std::memory_order_relaxed);
      ram_mgr.apply_promotion_bookkeeping(*blk, layer_idx, expert_id, promoted);
      ram_mgr.ram_unpin(*blk);
    }

    // Cross-layer follow-up for LayerScratch::pending_prefill_promotions
    // (see that field's own comment) -- runs AFTER the candidate loop
    // above, not before. ORDERING FIX (2026-08-12): running this FIRST
    // (the original placement) let it consume vram_cache->has_free_slot()
    // capacity before THIS layer's own hint-pushing loop got to decide
    // how much of that shared budget it wanted -- confirmed via a real
    // A/B to starve LATER layers of hints entirely (n_prefill_hint_pushed
    // nearly halved, 420 vs ~930), a genuine regression (see CLAUDE.md).
    // Running it AFTER means the candidate loop above always gets first
    // claim on whatever's free; this pass only ever mops up whatever's
    // left, which can never come at another layer's expense. Still
    // fires every Phase-2 entry regardless of THIS layer's own
    // prefill_warm_via_prefetcher status, since a pending entry can
    // reference an EARLIER layer. Cheap no-op when the list is empty.
    if (engine.prefill_opportunistic_promote &&
        scratch.pending_prefill_count > 0) {
      constexpr int PENDING_PREFILL_MAX_AGE =
          12; // layers; starting guess, unverified on hardware
      int write = 0;
      for (int i = 0; i < scratch.pending_prefill_count; ++i) {
        auto &pend = scratch.pending_prefill_promotions[i];
        ++pend.age_layers;
        bool resolved = false;

        if (ram_mgr.vram_cache->find_expert(pend.layer_idx, pend.expert_id)) {
          perf().n_prefill_pending_already_resident.fetch_add(
              1, std::memory_order_relaxed);
          resolved = true; // something else already promoted it (e.g. a
                           // reactive hit) -- fine
        } else if (!ram_mgr.vram_cache->has_free_slot()) {
          perf().n_prefill_pending_no_free_slot.fetch_add(
              1, std::memory_order_relaxed);
          resolved = true; // cache is warm now -- opportunistic promotion is
                           // moot for the rest of this run
        } else {
          const uint8_t *ready_slab = nullptr;
          MoEExpertBlock *ready = ram_mgr.find_ram_slot_if_ready_held(
              engine.layers_experts[pend.layer_idx],
              (size_t)ram_mgr.cap_for_layer(pend.layer_idx), pend.layer_idx,
              pend.expert_id, &ready_slab, nullptr);
          if (!ready) {
            perf().n_prefill_pending_still_loading.fetch_add(
                1, std::memory_order_relaxed);
            resolved = (pend.age_layers >= PENDING_PREFILL_MAX_AGE);
            if (resolved)
              perf().n_prefill_pending_aged_out.fetch_add(
                  1, std::memory_order_relaxed);
          } else {
            size_t p2_gb = 0, p2_ub = 0, p2_db = 0;
            const GGUFTensor *p2_gM = nullptr, *p2_uM = nullptr,
                             *p2_dM = nullptr;
            bool ready_gpu_capable =
                engine.expert_layer_geometry(pend.layer_idx, p2_gb, p2_ub,
                                             p2_db, p2_gM, p2_uM, p2_dM) &&
                gpu_type_supported(p2_gM->type) &&
                gpu_type_supported(p2_uM->type) &&
                gpu_type_supported(p2_dM->type);
            resolved = true; // decided one way or the other below, either way
                             // don't re-check it again
            if (ready_gpu_capable && ready_slab) {
              // Real total, not slab_capacity -- see decode_async_prewarm's
              // own fix comment for the full incident.
              ExpertLocation *promoted =
                  ram_mgr.vram_cache->upload_expert_async(
                      pend.layer_idx, pend.expert_id, ready_slab,
                      ready->gate_proj.meta.size_bytes +
                          ready->up_proj.meta.size_bytes +
                          ready->down_proj.meta.size_bytes,
                      -1, ready->slab_pinned);
              // 2026-08-30 diagnostic -- see MoEExpertBlock::
              // consumed_by_vram_upload (engine.hpp).
              if (promoted)
                ram_mgr.mark_consumed_by_vram_upload(*ready);
              if (promoted)
                perf().n_prefill_pending_promoted.fetch_add(
                    1, std::memory_order_relaxed);
              ram_mgr.apply_promotion_bookkeeping(*ready, pend.layer_idx,
                                                  pend.expert_id, promoted);
            }
            // The async main-cache upload has either transferred its source
            // protection to pending_copy_event (apply_promotion_bookkeeping)
            // or did not issue a copy. In both cases this lookup hold is no
            // longer needed once the bookkeeping transition is complete.
            ram_mgr.ram_unpin(*ready);
          }
        }

        if (!resolved)
          scratch.pending_prefill_promotions[write++] =
              pend; // keep, still legitimately pending
      }
      scratch.pending_prefill_count = write;
    }

    PERF_END(prefill_warm, t_prefill_warm_time);
  }

  // --- Phase 3b (OPT-IN): expert-major batched dispatch ------------------
  // Groups this batch's (token, expert-slot) pairs BY EXPERT and runs ONE
  // expert_ffn call per expert over all the tokens routed to it, instead of
  // the legacy per-token loop's S=1 call per (token, expert).
  //
  // This is the same restructuring llama.cpp's ggml_mul_mat_id performs; its
  // CUDA fallback (ggml-cuda.cu's ggml_cuda_mul_mat_id) builds exactly this
  // CSR-style layout -- tokens_per_expert + a sorted index list -- which is
  // what the code below mirrors, using Phase 1's already-computed
  // scratch.expert_freq[] as the counts so nothing has to be recounted.
  //
  // WHY: the legacy loop issues S*n_layers*topk dispatches and re-reads each
  // expert's ~3 MB weight slab once PER TOKEN. Measured on qwen4exp: 1.53M
  // dispatches for a 3,189-token prompt, and prefill throughput flat in
  // prompt length (5.4 tok/s at S=24 vs 6.02 at S=3189). See
  // dense_dispatch.hpp's moe_prefill_batched_flag() for the full writeup.
  //
  // The generic flag is shared by all model families and remains OFF by
  // default. GLM5NEXT reaches the same corrected grouped implementation only
  // through its separate family-gated flag below; with both flags off the
  // legacy loop remains byte-for-byte unchanged for every model.
  //
  // v1's legacy per-token path deliberately does NOT pipeline across experts
  // (issue-all/finish-all the way the old Pass A/Pass B did across one token's
  // topk). The GLM5NEXT grouped-prefill opt-in below is the narrow exception:
  // it keeps one chunk of device-accumulated expert work in flight, while the
  // generic path remains conservative.
  const bool glm5next_grouped_prefill =
      engine.glm5next_moe_prefill_gpu &&
      engine.family_ops.family == ModelFamily::GLM5NEXT;
  bool batched_handled = false;
  if (gpu_route_device && S > 1 && topk > 0 && n_experts > 0 &&
      !is_cpu_affinity_layer(layer_idx)) {
    // Keep the actual routed set for compact fallback.  The previous device
    // handoff made `used` the complete [0,n_experts) range so global IDs could
    // be passed directly to the device graph.  That was correct only for the
    // all-resident view: when a layer was scattered, the compact fallback
    // consequently repacked all 512 experts even though this batch touched
    // only a subset.  The direct resident path now restores the full axis
    // internally, while compact layers use this real routed subset and the
    // host route table for a local-ID remap.
    //
    // The device-router path does not populate expert_freq: it writes the
    // routed IDs and weights on the device, then copies those two tables back
    // for the host/device handoff.  Reconstruct the counts from that table
    // here.  Using expert_freq directly made it look as though no experts
    // were used, which disabled the batched graph and fell through to the
    // legacy per-token path for the early layers.
    std::fill(scratch.expert_freq.begin(), scratch.expert_freq.end(), 0);
    for (size_t i = 0; i < scratch.tok_topk_ids.size(); ++i) {
      const int e = scratch.tok_topk_ids[i];
      if (e >= 0 && e < n_experts)
        ++scratch.expert_freq[(size_t)e];
    }
    std::vector<int> used;
    used.reserve((size_t)n_experts);
    for (int e = 0; e < n_experts; ++e)
      if (scratch.expert_freq[(size_t)e] > 0)
        used.push_back(e);
    static const bool trace_required_set = [] {
      const char *env = std::getenv("MOESTREAM_MOE_SET_TRACE");
      return env && env[0] != '0';
    }();
    if (trace_required_set && gpu_route_device && device_router) {
      // Device routing deliberately leaves IDs/weights on CUDA in normal
      // execution. Export them only for the explicit set trace so its
      // required expert set is real rather than the untouched zero-filled
      // host scratch array.
      gpu_route_ids_ks.resize((size_t)topk * S);
      gpu_route_weights_ks.resize((size_t)topk * S);
      if (!device_router->copy_route_to_host(layer_idx, gpu_route_ids_ks.data(),
                                             gpu_route_weights_ks.data())) {
        LOG_ERROR("[MOE-REQUIRED] layer=%d failed to export device route\n",
                  layer_idx);
      } else {
        scratch.tok_topk_ids = gpu_route_ids_ks;
        scratch.tok_topk_weights = gpu_route_weights_ks;
        std::fill(scratch.expert_freq.begin(), scratch.expert_freq.end(), 0);
        for (int e : gpu_route_ids_ks) {
          if (e >= 0 && e < n_experts)
            ++scratch.expert_freq[(size_t)e];
        }
        used.clear();
        for (int e = 0; e < n_experts; ++e)
          if (scratch.expert_freq[(size_t)e] > 0)
            used.push_back(e);
      }
    }
    if (trace_required_set) {
      std::string required_ids;
      required_ids.reserve(used.size() * 4);
      for (size_t i = 0; i < used.size(); ++i) {
        if (i != 0)
          required_ids.push_back(',');
        required_ids += std::to_string(used[i]);
      }
      LOG_INFO("[MOE-REQUIRED] layer=%d required=%zu route=%s ids=%s\n",
               layer_idx, used.size(), gpu_route_device ? "device" : "host",
               required_ids.c_str());
    }
    const bool qwen_device_handoff =
        engine.family_ops.family == ModelFamily::Qwen4Exp &&
        engine.family_ops.uses_hyper_connections && qwen4exp_hc_device_live() &&
        !is_cpu_affinity_layer(layer_idx);
    bool ggml_handled =
        qwen_device_handoff && clamp_exp == 0.0f && ram_mgr.vram_cache &&
        run_ggml_moe_layer(
            engine, ram_mgr, ram_mgr.vram_cache, layer_idx, S, D, moe_inter,
            topk, n_experts, used, scratch.tok_topk_ids.data(),
            scratch.tok_topk_weights.data(), scratch.normed.data(), output,
            moe_prefill_stream_chunk(), /*unpacked_layer=*/false,
            &scratch.moe_device_output, engine.qwen4exp_device_hc_ffn_input,
            gpu_route_ids_device, gpu_route_weights_device,
            /*allow_active_layer_sources=*/false, &scratch.moe_device_runner);
    if (ggml_handled) {
      scratch.moe_device_output_ready = scratch.moe_device_output != nullptr;
      bool shared_device_handled = false;
      if (moe_ggml_shared_expert() && c.n_expert_shared > 0) {
        const void *combined = nullptr;
        shared_device_handled =
            run_qwen4exp_shared_device(engine, layer_idx, S, D, moe_inter,
                                       engine.qwen4exp_device_hc_ffn_input,
                                       scratch.moe_device_output, c, &combined);
        if (shared_device_handled)
          scratch.moe_device_output = combined;
      }
      if (!shared_device_handled) {
        // The routed graph is still correct; only the optional shared graph
        // failed.  Recover the routed result and use the established host
        // shared-expert implementation so this experimental path remains
        // numerically complete.
        if (engine.qwen4exp_device_hc_ffn_input &&
            !GgmlHcRunner::instance().copy_pre_mix_to_host(
                engine.qwen4exp_device_hc_ffn_input, scratch.normed.data(),
                S)) {
          LOG_ERROR("[GGML-SHARED] layer=%d: HC input recovery failed; "
                    "shared host fallback is unsafe\n",
                    layer_idx);
        }
        if (!scratch.moe_device_runner ||
            !scratch.moe_device_runner->copy_device_output_to_host(
                scratch.moe_device_output, output)) {
          LOG_ERROR("[GGML-MOE] layer=%d: all-resident routed output recovery "
                    "failed\n",
                    layer_idx);
          return;
        }
        run_qwen4exp_shared_host(engine, layer_idx, S, D, moe_inter,
                                 scratch.normed.data(), output, scratch, c);
        scratch.moe_device_output = nullptr;
        scratch.moe_shared_device_output = false;
        scratch.moe_device_output_ready = false;
      }
      batched_handled = true;
    } else {
      // A device-route graph failure must not drop the layer.  Re-run only
      // the tiny route table to host and let the existing validated batched
      // path continue below; the large HC activation remains device-resident.
      gpu_route_device = false;
      gpu_route_ids_device = nullptr;
      gpu_route_weights_device = nullptr;
      gpu_router_handled = false;
      if (device_router) {
        gpu_route_ids_ks.resize((size_t)topk * S);
        gpu_route_weights_ks.resize((size_t)topk * S);
        gpu_router_handled = device_router->run_tensor(
            layer_idx, engine.qwen4exp_device_hc_ffn_input,
            gpu_route_ids_ks.data(), gpu_route_weights_ks.data());
      }
      if (!gpu_router_handled) {
        LOG_ERROR("[GGML-MOE] layer=%d: device-route and host-route fallback "
                  "both failed\n",
                  layer_idx);
        return;
      }
    }
  }
  // The GLM5NEXT switch is deliberately separate from the generic flag.
  // GLM's mixed per-expert quant geometry is the reason the old shared flag
  // remains opt-in/off here; this path still uses the same corrected
  // per-expert metadata and routing-consistency checks, but its activation is
  // only reachable for an explicitly enabled GLM5NEXT configuration.
  const bool grouped_prefill =
      moe_prefill_batched() || glm5next_grouped_prefill;
  if (!batched_handled && grouped_prefill && S > 1 && topk > 0 &&
      n_experts > 0) {
    PERF_SCOPE(t_moe_phase3_dispatch);

    // CSR build: offs[e] .. offs[e+1] indexes into slot[], which holds the
    // flat (s*topk + k) id of every pair routed to expert e. expert_freq is
    // Phase 1's own count of how many times each expert was picked in this
    // batch -- reused directly rather than recomputed.
    std::vector<int> offs((size_t)n_experts + 1, 0);
    for (int e = 0; e < n_experts; ++e)
      offs[(size_t)e + 1] = offs[(size_t)e] + scratch.expert_freq[(size_t)e];
    const int total_pairs = offs[(size_t)n_experts];

    std::vector<int> cursor(offs.begin(), offs.end() - 1);
    std::vector<int> slot((size_t)std::max(1, total_pairs));
    bool routing_ok = (total_pairs == S * topk);
    if (routing_ok) {
      for (int s2 = 0; s2 < S && routing_ok; ++s2) {
        for (int k = 0; k < topk; ++k) {
          const int e = scratch.tok_topk_ids[(size_t)s2 * topk + k];
          if (e < 0 || e >= n_experts) {
            routing_ok = false;
            break;
          }
          slot[(size_t)cursor[(size_t)e]++] = s2 * topk + k;
        }
      }
    }
    if (!routing_ok) {
      // expert_freq disagrees with tok_topk_ids, or an id is out of range.
      // Fall through to the legacy loop rather than compute something wrong
      // -- this should be unreachable (both come from the same Phase 1) and
      // is here so a future change to either can never silently corrupt.
      LOG_ERROR("[MOE-BATCH] layer=%d: routing table inconsistent "
                "(sum(expert_freq)=%d vs S*topk=%d) -- falling back to the "
                "per-token path for this layer.\n",
                layer_idx, total_pairs, S * topk);
    } else {
      // GPU-side accumulation (EngineConfig::moe_gpu_accum): per-pair
      // destination row and router weight, uploaded ONCE per layer, so each
      // expert's scatter kernel can read its own slice at offs[e]. Without
      // this every expert's [n,D] output has to come back to the host purely
      // so a CPU loop can do the weighted add -- ~16.9 GB/prefill at real
      // qwen4exp shape, measured as dn=16.81s against cmp=0.27s of actual
      // expert matmul.
      const bool accum_gpu = moe_gpu_accum() && ram_mgr.vram_cache;
      std::vector<int> tok_pair;
      std::vector<float> w_pair;
      if (accum_gpu) {
        tok_pair.resize((size_t)total_pairs);
        w_pair.resize((size_t)total_pairs);
        for (int i = 0; i < total_pairs; ++i) {
          const int flat = slot[(size_t)i];
          tok_pair[(size_t)i] = flat / topk;
          w_pair[(size_t)i] = scratch.tok_topk_weights[(size_t)flat];
        }
      }
      bool accum_begun[8] = {false};
      std::vector<float> batch_out;
      // One PINNED activation arena per chunk, each expert owning its own
      // slice. That is what lets the per-expert H2D actually be async: the
      // previous code gathered into a pageable std::vector, so the driver
      // staged every copy synchronously AND the buffer had to be protected
      // by a blocking cudaStreamSynchronize before the next expert could
      // reuse it. MEASURED 2026-09-05: that sync alone was 4.469 s of an
      // 8.80 s expert bucket (9,982 calls @ 0.45 ms). With a private slice
      // per expert there is no reuse hazard, and stream ordering (the
      // upload is on compute_stream, the kernels on the legacy default
      // stream, which synchronize with each other implicitly) already
      // guarantees correctness without any host round trip.
      static thread_local PinnedFloatBuffer act_arena;
      std::vector<size_t> act_off;

      // Experts actually routed to this layer, ascending id. Ascending
      // matters for the streaming path below: within a packed tensor,
      // expert e lives at base + e*stride, so walking ids forward walks the
      // file forward, which is what the kernel's readahead wants.
      std::vector<int> used;
      used.reserve((size_t)n_experts);
      for (int e = 0; e < n_experts; ++e)
        if (offs[(size_t)e + 1] > offs[(size_t)e])
          used.push_back(e);
      const int n_used = (int)used.size();
      static const bool trace_required_set = [] {
        const char *env = std::getenv("MOESTREAM_MOE_SET_TRACE");
        return env && env[0] != '0';
      }();
      if (trace_required_set) {
        std::string required_ids;
        required_ids.reserve(used.size() * 4);
        for (size_t i = 0; i < used.size(); ++i) {
          if (i != 0)
            required_ids.push_back(',');
          required_ids += std::to_string(used[i]);
        }
        LOG_INFO("[MOE-REQUIRED] layer=%d required=%zu route=host ids=%s\n",
                 layer_idx, used.size(), required_ids.c_str());
      }

      // ---- mmap streaming setup (include/prefill_expert_stream.hpp) ----
      // Bypasses the RAM tier AND the VRAM LRU for the whole layer: during
      // prefill every expert is touched exactly once, so the cache reports
      // 0% hit by arithmetic and all of its machinery is pure overhead.
      static thread_local PrefillExpertStream pstream;
      const bool unpacked_layer = engine.layer_is_unpacked(layer_idx);
      const bool qwen_ddr_only_source = qwen_ddr_only_expert_source(engine);

      // ---- ggml MUL_MAT_ID path (opt-in) -----------------------------
      // Computes this layer's ENTIRE routed contribution in one backend
      // graph and adds it into `output`. On success everything below --
      // the PrefillExpertStream setup, the activation arena, the chunk
      // loop, the device accumulator -- is skipped (n_chunks becomes 0 and
      // accum_begun[] stays false), but `batched_handled` still ends up
      // true and the shared expert still runs, so the layer's FFN is
      // complete either way. On failure nothing has been written to
      // `output` and the normal path runs unchanged.
      bool ggml_handled = false;
      bool qwen_device_handoff = false;
      bool shared_device_handled = false;
      if (moe_ggml_mul_mat_id() && clamp_exp == 0.0f && n_used > 0 &&
          !is_cpu_affinity_layer(layer_idx) && ram_mgr.vram_cache) {
        qwen_device_handoff =
            engine.family_ops.family == ModelFamily::Qwen4Exp &&
            engine.family_ops.uses_hyper_connections &&
            qwen4exp_hc_device_live();
        ggml_handled = run_ggml_moe_layer(
            engine, ram_mgr, ram_mgr.vram_cache, layer_idx, S, D, moe_inter,
            topk, n_experts, used, scratch.tok_topk_ids.data(),
            scratch.tok_topk_weights.data(), scratch.normed.data(), output,
            moe_prefill_stream_chunk(), unpacked_layer,
            qwen_device_handoff ? &scratch.moe_device_output : nullptr,
            qwen_device_handoff ? engine.qwen4exp_device_hc_ffn_input : nullptr,
            nullptr, nullptr, /*allow_active_layer_sources=*/false,
            qwen_device_handoff ? &scratch.moe_device_runner : nullptr);
        scratch.moe_device_output_ready = ggml_handled && qwen_device_handoff &&
                                          scratch.moe_device_output != nullptr;
      }

      // Qwen4Exp also has an always-on shared expert.  When the routed result
      // is already device-resident, run that FFN and add it to the routed
      // result on the same GGML stream before HC post-combine.  Keeping the
      // two results together is what lets us remove the shared expert's three
      // host dense projections and its CPU residual loop without changing the
      // HC state transition.  Any shape/type/graph failure falls back by
      // downloading only the routed result and using the existing host shared
      // path below.
      if (ggml_handled && qwen_device_handoff && moe_ggml_shared_expert() &&
          c.n_expert_shared > 0) {
        const void *combined = nullptr;
        shared_device_handled =
            run_qwen4exp_shared_device(engine, layer_idx, S, D, moe_inter,
                                       engine.qwen4exp_device_hc_ffn_input,
                                       scratch.moe_device_output, c, &combined);
        if (shared_device_handled)
          scratch.moe_device_output = combined;
        if (!shared_device_handled) {
          // The routed graph has no host output because it was handed to HC
          // as a device tensor.  Recover it before the existing shared CPU
          // block so the fallback remains an additive routed+shared result.
          if (!scratch.moe_device_runner ||
              !scratch.moe_device_runner->copy_device_output_to_host(
                  scratch.moe_device_output, output)) {
            LOG_ERROR("[GGML-SHARED] layer=%d: routed output recovery failed; "
                      "shared host fallback cannot be used safely\n",
                      layer_idx);
          }
          scratch.moe_device_output = nullptr;
          scratch.moe_shared_device_output = false;
          scratch.moe_device_output_ready = false;
        }
      }

      bool stream_enabled = false;
      int stream_chunk = n_used > 0 ? n_used : 1;
      if (!ggml_handled && !qwen_ddr_only_source && moe_prefill_stream() &&
          engine.model_file.mmap_active() && ram_mgr.vram_cache && n_used > 0) {
        size_t max_slab = 0;
        bool geom_ok = true;
        for (int e : used) {
          size_t gb, ub, db;
          const GGUFTensor *gM, *uM, *dM;
          if (!engine.expert_geometry_for(layer_idx, e, gb, ub, db, gM, uM,
                                          dM)) {
            geom_ok = false;
            break;
          }
          max_slab = std::max(max_slab, gb + ub + db);
        }
        // Prefer the vault GPU with the biggest expert pool -- on this box
        // that is the 3090 (measured H2D 12.30 GB/s) rather than the 5060 Ti
        // (6.81 GB/s), and this path is bandwidth-bound by construction.
        int vault = -1;
        size_t best = 0;
        for (int vi : ram_mgr.vram_cache->vault_gpus) {
          if ((size_t)vi >= ram_mgr.vram_cache->devices.size())
            continue;
          const auto &dv = ram_mgr.vram_cache->devices[(size_t)vi];
          if (dv.expert_cache_pool && dv.expert_cache_size > best) {
            best = dv.expert_cache_size;
            vault = dv.id;
          }
        }
        if (geom_ok && max_slab > 0 && vault >= 0) {
          stream_chunk = std::min(moe_prefill_stream_chunk(), n_used);
          if (stream_chunk < 1)
            stream_chunk = 1;
          stream_enabled = pstream.ensure(vault, max_slab, stream_chunk);
          // Ship only THIS layer's real expert bytes, not the model-wide
          // maximum the buffers had to be allocated for.
          if (stream_enabled)
            pstream.set_active_stride(max_slab);
        }
      }

      // Per-chunk classification, double-buffered alongside the transfer
      // buffers. An expert already resident in the VRAM cache needs NO
      // transfer at all -- the streaming path bypasses the cache, so
      // without this it re-ships bytes the GPU already holds. This box
      // keeps ~23 GB / 5792 expert slabs resident (see [VRAM-SLOTS]), and
      // routing is concentrated enough that a large share of any layer's
      // real picks are among them.
      std::vector<const ExpertLocation *> res_loc(
          (size_t)PrefillExpertStream::NBUF * (size_t)stream_chunk, nullptr);
      std::vector<int> slot_of(
          (size_t)PrefillExpertStream::NBUF * (size_t)stream_chunk, -1);

      // Stage a chunk's weights straight out of the mmap into the pinned
      // buffer and kick off its H2D. Returns false if anything about this
      // chunk cannot be streamed, in which case the caller runs it on the
      // legacy cached path rather than computing something wrong.
      auto stage_chunk = [&](int c) -> bool {
        const int sc0 = c * stream_chunk;
        const int sc1 = std::min(n_used, sc0 + stream_chunk);
        if (sc0 >= sc1)
          return false;
        const int buf = c % PrefillExpertStream::NBUF;

        // Wait for whatever this buffer held two chunks ago to finish
        // computing before refilling it -- the double-buffer handoff. Also
        // the point at which the PREVIOUS transfer through this buffer is
        // known complete, so its CUDA-event timing can be read back.
        PERF_START(pf_wait);
        pstream.wait_free(buf);
        PERF_END(pf_wait, t_prefill_stream_wait);
        double h2d_ms = 0.0;
        size_t h2d_bytes = 0;
        pstream.collect_h2d(buf, h2d_ms, h2d_bytes);
        if (h2d_bytes) {
          perf().t_prefill_stream_h2d.fetch_add(h2d_ms / 1000.0,
                                                std::memory_order_relaxed);
          perf().bytes_prefill_stream_h2d.fetch_add(h2d_bytes,
                                                    std::memory_order_relaxed);
        }

        // Classify first (serially -- find_expert takes cache_mutex, and at
        // ~10k calls/prefill this is ~0.07 s, far cheaper than the transfer
        // it avoids). Misses are COMPACTED into the low slots so the H2D
        // still moves one contiguous run with no holes.
        const int cbase = buf * stream_chunk;
        int n_miss = 0;
        for (int i = sc0; i < sc1; ++i) {
          const int e = used[(size_t)i];
          const ExpertLocation *rl =
              ram_mgr.vram_cache->find_expert(layer_idx, e);
          if (rl && rl->device_ptr && rl->upload_ready &&
              ram_mgr.vram_cache->hold_for_compute(layer_idx, e, rl->device_ptr,
                                                   false)) {
            res_loc[(size_t)cbase + (size_t)(i - sc0)] = rl;
            slot_of[(size_t)cbase + (size_t)(i - sc0)] = -1;
          } else {
            res_loc[(size_t)cbase + (size_t)(i - sc0)] = nullptr;
            slot_of[(size_t)cbase + (size_t)(i - sc0)] = n_miss++;
          }
        }
        perf().n_prefill_stream_vram_hit.fetch_add(
            (uint64_t)((sc1 - sc0) - n_miss), std::memory_order_relaxed);
        perf().n_prefill_stream_vram_miss.fetch_add((uint64_t)n_miss,
                                                    std::memory_order_relaxed);
        if (n_miss == 0)
          return true; // nothing to transfer for this chunk

        // The RAM tier can already contain nearly every routed expert, in
        // which case the prefetcher performs almost no real reads and cannot
        // warm the page cache for this mmap path. Give the kernel the exact
        // upcoming ranges before the OMP memcpy starts faulting them in.
        // This is advisory only; mmap_bytes() below remains the correctness
        // check and the existing staging failure path is unchanged.
        for (int i = sc0; i < sc1; ++i) {
          const int sl = slot_of[(size_t)cbase + (size_t)(i - sc0)];
          if (sl < 0)
            continue; // already resident in VRAM, no mmap bytes needed
          const int e = used[(size_t)i];
          size_t gb, ub, db;
          const GGUFTensor *gM, *uM, *dM;
          if (!engine.expert_geometry_for(layer_idx, e, gb, ub, db, gM, uM, dM))
            continue;
          const uint64_t goff =
              unpacked_layer ? gM->offset : gM->offset + (uint64_t)e * gb;
          const uint64_t uoff =
              unpacked_layer ? uM->offset : uM->offset + (uint64_t)e * ub;
          const uint64_t doff =
              unpacked_layer ? dM->offset : dM->offset + (uint64_t)e * db;
          engine.model_file.mmap_willneed(*gM, goff, gb);
          engine.model_file.mmap_willneed(*uM, uoff, ub);
          engine.model_file.mmap_willneed(*dM, doff, db);
        }

        std::atomic<bool> stage_ok{true};
        size_t staged_bytes = 0;
        // Hoisted OUT of the parallel region on purpose. pstream is
        // `static thread_local`, so calling pstream.host_slot() from an OMP
        // worker would resolve to that WORKER's own default-constructed
        // instance (h_stage=nullptr, slab_stride=0) rather than the one the
        // main thread just allocated -- a null memcpy destination. MEASURED
        // as a real SIGSEGV here 2026-09-05.
        uint8_t *const stage_base = pstream.h_stage[buf];
        const size_t stage_stride = pstream.stride();
        // Timed into t_disk_read/bytes_disk_read on purpose: this is the
        // direct replacement for load_expert_slab's pread, so the existing
        // `disk=` field in the [PREFILL] line stays an apples-to-apples
        // comparison against the pre-streaming baseline.
        PERF_START(pf_stage);
#pragma omp parallel for schedule(static) reduction(+ : staged_bytes)
        for (int i = sc0; i < sc1; ++i) {
          const int e = used[(size_t)i];
          size_t gb, ub, db;
          const GGUFTensor *gM, *uM, *dM;
          if (!engine.expert_geometry_for(layer_idx, e, gb, ub, db, gM, uM,
                                          dM)) {
            stage_ok.store(false, std::memory_order_relaxed);
            continue;
          }
          // Packed layers store every expert in one tensor at
          // base + e*stride; an unpacked layer's tensor IS the expert's own,
          // so it must NOT be scaled (getting this wrong silently reads a
          // different expert's bytes -- no error path is taken).
          const uint64_t goff =
              unpacked_layer ? gM->offset : gM->offset + (uint64_t)e * gb;
          const uint64_t uoff =
              unpacked_layer ? uM->offset : uM->offset + (uint64_t)e * ub;
          const uint64_t doff =
              unpacked_layer ? dM->offset : dM->offset + (uint64_t)e * db;
          const uint8_t *gs = engine.model_file.mmap_bytes(*gM, goff, gb);
          const uint8_t *us = engine.model_file.mmap_bytes(*uM, uoff, ub);
          const uint8_t *ds = engine.model_file.mmap_bytes(*dM, doff, db);
          if (!gs || !us || !ds || gb + ub + db > stage_stride) {
            stage_ok.store(false, std::memory_order_relaxed);
            continue;
          }
          const int sl = slot_of[(size_t)cbase + (size_t)(i - sc0)];
          if (sl < 0)
            continue; // already VRAM-resident, nothing to ship
          uint8_t *dst = stage_base + (size_t)sl * stage_stride;
          std::memcpy(dst, gs, gb);
          std::memcpy(dst + gb, us, ub);
          std::memcpy(dst + gb + ub, ds, db);
          staged_bytes += gb + ub + db;
        }
        PERF_END(pf_stage, t_disk_read);
        perf().bytes_disk_read.fetch_add(staged_bytes,
                                         std::memory_order_relaxed);
        if (!stage_ok.load(std::memory_order_relaxed) ||
            !pstream.push(buf, n_miss)) {
          LOG_ERROR("[PREFILL-STREAM] layer=%d chunk@%d: staging failed -- "
                    "falling back to the cached path for this chunk.\n",
                    layer_idx, sc0);
          return false;
        }
        return true;
      };

      // Size the activation arena ONCE for the whole layer: sum(n) over any
      // chunk can never exceed total_pairs, so this is a hard upper bound
      // and the arena cannot grow (and therefore cannot reallocate under an
      // in-flight DMA) part-way through the chunk loop.
      if (!ggml_handled && !act_arena.resize((size_t)total_pairs * (size_t)D,
                                             "moe_prefill_activation_arena")) {
        LOG_WARN("[MEM-GUARD] grouped MoE activation arena refused at layer=%d "
                 "(%zu bytes); aborting this layer before any grouped output "
                 "is committed\n",
                 layer_idx, (size_t)total_pairs * (size_t)D * sizeof(float));
        return;
      }

      // The GLM-specific path also chunks the grouped GPU work when the
      // legacy mmap streaming path is off. This bounds the number of live
      // per-expert CUDA scratch buffers and lets the accumulator synchronize
      // once per small group, while retaining GLM's normal RAM-tier source.
      if (!stream_enabled && glm5next_grouped_prefill && n_used > 0) {
        stream_chunk =
            std::min(std::max(1, moe_prefill_stream_chunk()), n_used);
      }
      const int n_chunks = (!ggml_handled && n_used > 0)
                               ? (n_used + stream_chunk - 1) / stream_chunk
                               : 0;
      std::vector<char> chunk_ok((size_t)std::max(1, n_chunks), 0);
      // With device SwiGLU and device accumulation enabled, each expert
      // bucket can keep its own device scratch until the whole streaming
      // chunk has issued. That lets the gate/up, activation, down, and
      // scatter work remain ordered on CUDA streams without a host sync
      // after every bucket. The old path remains available whenever either
      // fast path is disabled.
      const bool grouped_async_enabled = accum_gpu && expert_gpu_activation();
      if (grouped_async_enabled &&
          (int)scratch.gpu_pool.size() < std::max(1, stream_chunk))
        scratch.gpu_pool.resize(std::max(1, stream_chunk));
      struct AsyncChunkHold {
        const ExpertLocation *loc = nullptr;
        MoEExpertBlock *exp = nullptr;
        int expert_id = -1;
        bool needs_release = false;
        bool from_active_layer = false;
      };
      std::vector<AsyncChunkHold> async_holds;
      std::array<bool, 8> async_gpu_used{};
      std::array<bool, 8> async_gpu_synced{};
      // Prime the pipeline so chunk 0's transfer is already in flight before
      // the loop body runs.
      if (stream_enabled && n_chunks > 0)
        chunk_ok[0] = stage_chunk(0) ? 1 : 0;

      for (int c = 0; c < n_chunks; ++c) {
        const int c0 = c * stream_chunk;
        const int c1 = std::min(n_used, c0 + stream_chunk);
        const int buf = c % PrefillExpertStream::NBUF;
        const bool grouped_async =
            grouped_async_enabled &&
            ((stream_enabled && chunk_ok[(size_t)c] != 0) ||
             (glm5next_grouped_prefill && !stream_enabled));
        async_holds.clear();
        async_holds.reserve((size_t)(c1 - c0));
        async_gpu_used.fill(false);
        async_gpu_synced.fill(false);

        // Kick the NEXT chunk's staging + H2D off BEFORE computing this one.
        // That is the entire point of the double buffer: c+1's transfer (and
        // the CPU memcpy feeding it) overlap c's GEMMs, instead of stage /
        // H2D / compute simply ADDING the way the legacy per-expert path
        // made them.
        if (stream_enabled && c + 1 < n_chunks)
          chunk_ok[(size_t)c + 1] = stage_chunk(c + 1) ? 1 : 0;

        const bool chunk_streamed = stream_enabled && chunk_ok[(size_t)c] != 0;
        if (chunk_streamed)
          pstream.gate_compute(buf);

        // Gather every expert's token rows for this chunk in one parallel
        // pass, into per-expert slices of the pinned arena.
        act_off.assign((size_t)(c1 - c0) + 1, 0);
        for (int i = c0; i < c1; ++i) {
          const int ee = used[(size_t)i];
          act_off[(size_t)(i - c0) + 1] =
              act_off[(size_t)(i - c0)] +
              (size_t)(offs[(size_t)ee + 1] - offs[(size_t)ee]);
        }
        if (!act_arena.resize(act_off.back() * (size_t)D,
                              "moe_prefill_activation_arena")) {
          LOG_WARN("[MEM-GUARD] grouped MoE activation arena grow refused at "
                   "layer=%d chunk=%d (%zu bytes); stopping grouped dispatch\n",
                   layer_idx, c, act_off.back() * (size_t)D * sizeof(float));
          return;
        }
        // Hoisted for the same reason as stage_base above: act_arena is
        // `static thread_local`, so an OMP worker calling act_arena.data()
        // would get its OWN empty instance, not this one. Cost me a second
        // SIGSEGV in the same session -- if you add another thread_local
        // buffer here, hoist its pointer too.
        float *const arena_base = act_arena.data();
#pragma omp parallel for schedule(static)
        for (int i = c0; i < c1; ++i) {
          const int ee = used[(size_t)i];
          const int *mine = slot.data() + offs[(size_t)ee];
          const int nn = offs[(size_t)ee + 1] - offs[(size_t)ee];
          float *dst = arena_base + act_off[(size_t)(i - c0)] * (size_t)D;
          for (int j = 0; j < nn; ++j)
            std::memcpy(dst + (size_t)j * D,
                        scratch.normed.data() + (size_t)(mine[j] / topk) * D,
                        (size_t)D * sizeof(float));
        }

        for (int i = c0; i < c1; ++i) {
          const int e = used[(size_t)i];
          const int n = offs[(size_t)e + 1] - offs[(size_t)e];
          if (n <= 0)
            continue;
          const int *my = slot.data() + offs[(size_t)e];

          // exp stays null on the streaming path -- there is no RAM slot at
          // all, which is the entire point (no claim_or_wait, no LRU scan, no
          // eviction, no temp-pin). Every later use is guarded on it.
          MoEExpertBlock *exp = nullptr;
          MoEExpertBlock shadow;
          ExpertLocation sloc;
          const ExpertLocation *loc = nullptr;
          bool needs_release = false, from_active = false;
          const GGUFTensor *gmp = nullptr, *ump = nullptr, *dmp = nullptr;

          bool vram_shadowed = false;
          if (chunk_streamed) {
            // Metadata only: type + the packed slab's slab-RELATIVE offsets
            // (gate=0, up=gb, down=gb+ub). That is exactly the layout this
            // chunk slot was staged in, so expert_ffn_gpu_issue's existing
            // `base + (up_meta.offset - gate_meta.offset)` pointer math works
            // against it unchanged.
            if (!engine.populate_vram_shadow(shadow, layer_idx, e)) {
              LOG_ERROR("[PREFILL-STREAM] layer=%d expert=%d: no geometry (%d "
                        "token(s) dropped).\n",
                        layer_idx, e, n);
              continue;
            }
            gmp = &shadow.gate_proj.meta;
            ump = &shadow.up_proj.meta;
            dmp = &shadow.down_proj.meta;
            const size_t cls = (size_t)(buf * stream_chunk) + (size_t)(i - c0);
            const ExpertLocation *rl = res_loc[cls];
            if (rl) {
              // Already in VRAM -- use it directly. hold_for_compute was taken
              // at classification time and is released after this expert's
              // kernels are issued, below.
              loc = rl;
              needs_release = true;
              from_active = false;
              gmp = &shadow.gate_proj.meta;
              ump = &shadow.up_proj.meta;
              dmp = &shadow.down_proj.meta;
              goto have_loc;
            }
            sloc.gpu_id = pstream.device_id;
            sloc.device_ptr = pstream.dev_slot(buf, slot_of[cls]);
            sloc.size = gmp->size_bytes + ump->size_bytes + dmp->size_bytes;
            sloc.layer_idx = layer_idx;
            sloc.expert_id = e;
            sloc.upload_ready = true;
            loc = &sloc;
          have_loc:;
          } else {
            // GLM5NEXT's grouped path may reach this branch with an expert
            // already resident in VRAM but with its RAM-tier copy absent.  A
            // get_expert() before the VRAM check would materialize a redundant
            // host copy and can issue an SSD read just to obtain metadata that
            // the GPU path never reads.  Check the VRAM shadow first.
            //
            // The shadow contains only projection types and packed-slab-
            // relative offsets.  expert_ffn_gpu_issue reads the actual bytes
            // from loc->device_ptr.  Unsupported types intentionally continue
            // to get_expert(), because their CPU fallback needs host bytes.
            if (glm5next_grouped_prefill && ram_mgr.vram_cache) {
              const ExpertLocation *resident =
                  ram_mgr.vram_cache->find_expert(layer_idx, e);
              if (resident && resident->device_ptr &&
                  engine.populate_vram_shadow(shadow, layer_idx, e) &&
                  gpu_type_supported(shadow.gate_proj.meta.type) &&
                  gpu_type_supported(shadow.up_proj.meta.type) &&
                  gpu_type_supported(shadow.down_proj.meta.type)) {
                exp = &shadow;
                loc = resident;
                gmp = &shadow.gate_proj.meta;
                ump = &shadow.up_proj.meta;
                dmp = &shadow.down_proj.meta;
                vram_shadowed = true;
                if (glm5next_grouped_prefill)
                  perf().n_glm_grouped_vram_shadow.fetch_add(
                      1, std::memory_order_relaxed);
              }
            }

            if (!vram_shadowed) {
              if (glm5next_grouped_prefill)
                perf().n_glm_grouped_host_lookup.fetch_add(
                    1, std::memory_order_relaxed);
              exp = ram_mgr.get_expert(engine, layer_idx, e);
              if (!exp || !exp->gate_proj.host_ptr || !exp->up_proj.host_ptr ||
                  !exp->down_proj.host_ptr) {
                LOG_ERROR(
                    "[MOE-BATCH] layer=%d expert=%d: no RAM block (%d token(s) "
                    "dropped this layer).\n",
                    layer_idx, e, n);
                continue;
              }

              // Pin for the WHOLE lifetime of this expert's use, not just the
              // CPU branch. The GPU cold-batch route (upload_expert /
              // upload_expert_active) DMAs straight out of exp->slab_buffer,
              // so a prefetch worker's victim scan reclaiming or realloc'ing
              // this slot mid-upload yields "cudaMemcpy H2D failed ... invalid
              // argument" on a host_src that was valid microseconds earlier.
              // Passing n (not 1) as S makes resolve_gpu_loc take the
              // cold-batch upload route far more often than the per-token path
              // ever does, which is why only the batched path shows this.
              ram_mgr.ram_pin(*exp);
              gmp = &exp->gate_proj.meta;
              ump = &exp->up_proj.meta;
              dmp = &exp->down_proj.meta;
            }
          }

          const GGUFTensor &gm = *gmp;
          const GGUFTensor &um = *ump;
          const GGUFTensor &dm = *dmp;
          // Hoisted out of the legacy branch's ExpertGpuContext, which no
          // longer exists on the streaming path -- same value either way.
          GpuTopology *const topo = ram_mgr.vram_cache;

          // This expert's slice of the chunk-wide pinned arena, gathered above.
          const float *const my_in =
              act_arena.data() + act_off[(size_t)(i - c0)] * (size_t)D;
          batch_out.assign((size_t)n * D, 0.0f);

          if (!chunk_streamed) {
            ExpertGpuContext gpu_ctx;
            gpu_ctx.topo = ram_mgr.vram_cache;
            gpu_ctx.loc = loc;
            gpu_ctx.layer_idx = layer_idx;
            gpu_ctx.expert_id = e;
            gpu_ctx.slab_buffer = exp->slab_buffer;
            // Real total, NOT exp->slab_capacity. This mirrors the per-token
            // path's own identical fix (see dispatch_expert_k below): a RAM
            // slot's slab_capacity is its GROW-ONLY allocation, which after an
            // LRU reuse can be larger than whatever smaller expert now occupies
            // it -- and this field is the byte count every synchronous
            // cold-miss GPU upload below reads. Getting it wrong uploads the
            // wrong number of bytes with no error path taken, i.e. silent
            // garbage weights. MEASURED: this was the cause of
            // moe_prefill_batched producing incoherent output on GLM-5.3
            // (2026-09-04), whose UD-IQ2_M layers have genuinely mixed
            // per-expert sizes (IQ2_XXS/IQ2_S gate, IQ3_XXS/IQ4_XS down) so the
            // two values routinely disagree. qwen4exp's UD-Q4_K_XL is
            // near-uniform, which is why it never surfaced there.
            gpu_ctx.slab_capacity = exp->gate_proj.meta.size_bytes +
                                    exp->up_proj.meta.size_bytes +
                                    exp->down_proj.meta.size_bytes;
            gpu_ctx.exp = exp;

            loc = expert_ffn_resolve_gpu_loc(&gpu_ctx, n, gm, um, dm,
                                             &needs_release, &from_active,
                                             ram_mgr, engine);
          }

          bool did_gpu = false;
          bool gpu_accumulated_on_device = false;
          if (loc) {
            // Same three helpers the per-token path uses -- every one of them
            // already takes S, so the ONLY difference here is passing n
            // instead of 1.
            GpuExpertScratch &gs =
                scratch.gpu_pool[grouped_async ? (size_t)(i - c0) : 0];
            if (gs.ensure(n, D, moe_inter, loc->gpu_id) &&
                expert_ffn_gpu_issue(gs, loc, my_in, gm, um, n, D, moe_inter,
                                     topo)) {
              if (!scratch.gate_buf.resize((size_t)n * moe_inter,
                                           "grouped_gate_buf") ||
                  !scratch.up_buf.resize((size_t)n * moe_inter,
                                         "grouped_up_buf")) {
                LOG_WARN("[MEM-GUARD] grouped MoE expert scratch refused at "
                         "layer=%d expert=%d; stopping grouped dispatch\n",
                         layer_idx, e);
                if (needs_release && topo)
                  topo->release_hold(layer_idx, e, loc->device_ptr,
                                     from_active);
                return;
              }
              expert_ffn_gpu_finish_stage1(
                  gs, loc, scratch.gate_buf.data(), scratch.up_buf.data(), gm,
                  dm, layer_idx, e, n, D, moe_inter, clamp_exp, topo);
              const int gid = loc->gpu_id;
              // Sync ONLY when the host is about to read the result. With
              // moe_gpu_accum the accumulate kernel launches on the legacy
              // default stream, ordered behind this expert's own down-proj
              // kernel, so nothing crosses back to the host and a sync here
              // would just stall the CPU for the chunk's in-flight H2D.
              // gs's device buffers are shared across experts, but the
              // upload (compute_stream) and the kernels (default stream)
              // implicitly synchronize with each other, so reuse is ordered
              // by the streams rather than by this call.
              // Skip the sync ONLY when explicitly opted in: it is provably
              // redundant under moe_gpu_accum, but removing it measured
              // WORSE (see expert_stream_no_sync()'s own comment).
              if (grouped_async && gid >= 0 && gid < 8)
                async_gpu_used[(size_t)gid] = true;
              const bool skip_sync =
                  grouped_async ||
                  (accum_gpu && gid >= 0 && gid < 8 && expert_stream_no_sync());
              if (!skip_sync) {
                PERF_START(bsync);
                cudaStreamSynchronize(0);
                PERF_END(bsync, t_gpu_batch_sync);
                perf().n_gpu_batch_sync_calls.fetch_add(
                    1, std::memory_order_relaxed);
              }
              bool accumulated = false;
              if (accum_gpu && gid >= 0 && gid < 8) {
                if (!accum_begun[gid])
                  accum_begun[gid] = engine.gpu->expert_accum_begin(
                      gid, S, D, tok_pair.data(), w_pair.data(), total_pairs);
                if (accum_begun[gid]) {
                  accumulated = engine.gpu->expert_accum_add(
                      gid, gs.d_down, offs[(size_t)e], n, D, S);
                  if (accumulated && grouped_async) {
                    // Keep both the VRAM hold and the RAM pin until the
                    // chunk-level synchronization below confirms that this
                    // expert's down/scatter work is complete.
                    async_holds.push_back({needs_release ? loc : nullptr, exp,
                                           e, needs_release, from_active});
                  } else if (accumulated) {
                    // Non-grouped path preserves the old per-expert release
                    // behavior because its stream has already been waited on.
                    if (needs_release && topo)
                      topo->release_hold(layer_idx, e, loc->device_ptr,
                                         from_active);
                  } else {
                    perf().n_moe_gpu_accum_fallback.fetch_add(
                        1, std::memory_order_relaxed);
                  }
                }
              }
              if (!accumulated) {
                if (grouped_async) {
                  // A fallback download needs the result now; this is rare
                  // (accumulator allocation or launch failure), so isolate its
                  // synchronization rather than compromising the normal
                  // grouped path.
                  cudaSetDevice(loc->gpu_id);
                  cudaStreamSynchronize(0);
                }
                expert_ffn_gpu_finish_stage2(loc, batch_out.data(), layer_idx,
                                             e, n, D, topo, needs_release,
                                             from_active, gs);
              }
              did_gpu = true;
              gpu_accumulated_on_device = accumulated;
            } else if (needs_release && topo) {
              topo->release_hold(layer_idx, e, loc->device_ptr, from_active);
            }
          }
          if (!did_gpu) {
            // The streaming path has no host copy of the weights (that is the
            // point), so a GPU-dispatch failure has to materialise one before
            // it can fall back. Pulling the RAM slot here rather than dropping
            // the expert keeps this a slow path, not a silently-wrong one --
            // an expert whose contribution is skipped corrupts the residual
            // stream with no error the output would reveal.
            MoEExpertBlock *fb = exp;
            bool fb_pinned = false;
            if (!fb) {
              fb = ram_mgr.get_expert(engine, layer_idx, e);
              if (fb) {
                ram_mgr.ram_pin(*fb);
                fb_pinned = true;
              }
            }
            if (fb && fb->gate_proj.host_ptr && fb->up_proj.host_ptr &&
                fb->down_proj.host_ptr) {
              if (!scratch.gate_buf.resize((size_t)n * moe_inter,
                                           "grouped_cpu_gate_buf") ||
                  !scratch.up_buf.resize((size_t)n * moe_inter,
                                         "grouped_cpu_up_buf")) {
                LOG_WARN(
                    "[MEM-GUARD] grouped MoE CPU-fallback scratch refused at "
                    "layer=%d expert=%d; preserving prior output\n",
                    layer_idx, e);
                if (fb_pinned)
                  ram_mgr.ram_unpin(*fb);
                return;
              }
              expert_ffn_cpu(my_in, batch_out.data(), fb->gate_proj.host_ptr,
                             fb->gate_proj.meta, fb->up_proj.host_ptr,
                             fb->up_proj.meta, fb->down_proj.host_ptr,
                             fb->down_proj.meta, scratch.gate_buf.data(),
                             scratch.up_buf.data(), n, D, moe_inter, clamp_exp,
                             layer_idx, e);
            } else {
              LOG_ERROR(
                  "[MOE-BATCH] layer=%d expert=%d: GPU dispatch failed and "
                  "no host fallback available (%d token(s) dropped).\n",
                  layer_idx, e, n);
            }
            if (fb_pinned)
              ram_mgr.ram_unpin(*fb);
          }

          // Scatter-accumulate: output[s] += w[s,k] * expert_out[i]
          // (skipped when the GPU already did it into the device accumulator)
          for (int i2 = 0; !gpu_accumulated_on_device && i2 < n; ++i2) {
            const int flat = my[i2];
            const int s2 = flat / topk;
            const float w = scratch.tok_topk_weights[(size_t)flat];
            float *dst = output + (size_t)s2 * D;
            const float *src = batch_out.data() + (size_t)i2 * D;
            for (int d = 0; d < D; ++d)
              dst[d] += w * src[d];
          }

          const bool deferred_unpin =
              grouped_async && gpu_accumulated_on_device;
          if (exp && !deferred_unpin)
            ram_mgr.ram_unpin(*exp);
        } // per-expert

        if (grouped_async) {
          // One wait per vault GPU for the whole chunk, instead of one wait
          // per expert bucket. The default-stream kernels are already
          // ordered behind each bucket's activation/down/scatter work.
          for (int gid = 0; gid < 8; ++gid) {
            if (!async_gpu_used[(size_t)gid])
              continue;
            cudaSetDevice(gid);
            PERF_START(bsync_chunk);
            cudaStreamSynchronize(0);
            PERF_END(bsync_chunk, t_gpu_batch_sync);
            perf().n_gpu_batch_sync_calls.fetch_add(1,
                                                    std::memory_order_relaxed);
            async_gpu_synced[(size_t)gid] = true;
          }
          for (const AsyncChunkHold &hold : async_holds) {
            if (hold.needs_release && hold.loc && ram_mgr.vram_cache)
              ram_mgr.vram_cache->release_hold(layer_idx, hold.expert_id,
                                               hold.loc->device_ptr,
                                               hold.from_active_layer);
            if (hold.exp)
              ram_mgr.ram_unpin(*hold.exp);
          }
        }

        // Signal that this chunk's compute is issued, so the buffer can be
        // recycled two chunks from now. Recorded on the legacy default
        // stream, which is where every cuda/ matmul actually launched.
        if (chunk_streamed)
          pstream.mark_computed(buf);
      } // per-chunk
      // One [S,D] download per GPU for this whole layer, ADDED onto whatever
      // CPU-dispatched experts already wrote into `output`.
      for (int g = 0; g < 8; ++g)
        if (accum_begun[g] && !engine.gpu->expert_accum_end(g, output, S, D))
          perf().n_moe_gpu_accum_fallback.fetch_add(1,
                                                    std::memory_order_relaxed);

      // The shared expert is not part of the routed expert buckets above.
      // The legacy path executes it inside its per-token loop, but the
      // batched path skips that loop entirely once batched_handled becomes
      // true. Run the shared expert over all S normalized token rows here so
      // enabling moe_prefill_batched preserves the model's full FFN:
      // routed top-k contribution PLUS the always-active shared expert.
      const std::string shared_prefix =
          "blk." + std::to_string(layer_idx) + ".";
      if (!shared_device_handled && c.n_expert_shared > 0 &&
          engine.resident_layers.count(shared_prefix + "ffn_gate_shexp.weight"))
        run_qwen4exp_shared_host(engine, layer_idx, S, D, moe_inter,
                                 scratch.normed.data(), output, scratch, c);
      batched_handled = true;
    }
  }

  // Qwen4Exp decode-only grouped routed FFN. The ordinary grouped path above
  // is intentionally prefill-only because its expert-major buckets are built
  // for many source rows. For S==1, the same GGML MUL_MAT_ID graph is still
  // valuable: it collapses this token's up-to-topk expert cache lookup,
  // upload, dispatch, and host accumulation into one graph submission.
  //
  // Device-HC is preferred. If exact device-HC is disabled because its larger
  // Q8 HC copies do not fit beside the VRAM cache, use the same grouped graph
  // with the already-computed host HC activation. That costs one small input
  // upload and result readback per layer, but avoids the old per-expert GPU
  // dispatch/synchronization sequence. This remains explicitly Qwen-only;
  // other model families and the legacy Qwen path are unchanged.
  const bool qwen_decode_grouped_device =
      engine.qwen4exp_decode_ggml_moe &&
      engine.family_ops.family == ModelFamily::Qwen4Exp &&
      engine.family_ops.uses_hyper_connections && qwen4exp_hc_device_live() &&
      engine.qwen4exp_device_hc_ffn_input && moe_ggml_mul_mat_id();
  const bool qwen_decode_grouped_host =
      engine.qwen4exp_decode_ggml_moe &&
      engine.family_ops.family == ModelFamily::Qwen4Exp &&
      engine.family_ops.uses_hyper_connections && !qwen_decode_grouped_device &&
      moe_ggml_mul_mat_id();
  if (!batched_handled && S == 1 && topk > 0 && n_experts > 0 &&
      (qwen_decode_grouped_device || qwen_decode_grouped_host) &&
      clamp_exp == 0.0f && !is_cpu_affinity_layer(layer_idx) &&
      ram_mgr.vram_cache) {
    std::vector<int> used;
    used.reserve((size_t)n_experts);
    for (int e = 0; e < n_experts; ++e)
      if (scratch.expert_freq[(size_t)e] > 0)
        used.push_back(e);

    if (!used.empty()) {
      const void *grouped_device_input =
          qwen_decode_grouped_device ? engine.qwen4exp_device_hc_ffn_input
                                     : nullptr;
      const void *grouped_moe_output = nullptr;
      const bool handled = run_ggml_moe_layer(
          engine, ram_mgr, ram_mgr.vram_cache, layer_idx, S, D, moe_inter, topk,
          n_experts, used, scratch.tok_topk_ids.data(),
          scratch.tok_topk_weights.data(), scratch.normed.data(), output,
          moe_prefill_stream_chunk(), engine.layer_is_unpacked(layer_idx),
          (qwen_decode_grouped_device || qwen_decode_grouped_host)
              ? &grouped_moe_output
              : nullptr,
          grouped_device_input, nullptr, nullptr,
          /*allow_active_layer_sources=*/true, &scratch.moe_device_runner);
      if (handled) {
        bool shared_device_handled = false;
        if (qwen_decode_grouped_device || qwen_decode_grouped_host) {
          // Keep the routed runner's output separate from the optional shared
          // graph result.  In host-HC mode the shared graph can successfully
          // queue a combined device result but still fail its host readback;
          // the fallback must then recover the original routed result through
          // the routed runner, not pass the shared runner's pointer to it.
          const void *routed_device_output = grouped_moe_output;
          scratch.moe_device_output = grouped_moe_output;
          scratch.moe_device_output_ready = grouped_moe_output != nullptr;
          if (moe_ggml_shared_expert() && c.n_expert_shared > 0) {
            const void *shared_input =
                qwen_decode_grouped_device
                    ? engine.qwen4exp_device_hc_ffn_input
                    : (scratch.moe_device_runner
                           ? scratch.moe_device_runner->device_input_tensor()
                           : nullptr);
            const void *combined = nullptr;
            shared_device_handled = run_qwen4exp_shared_device(
                engine, layer_idx, S, D, moe_inter, shared_input,
                scratch.moe_device_output, c, &combined);
            if (shared_device_handled) {
              scratch.moe_device_output = combined;
              // Even when the HC pre-mix was also exported to the host for
              // routing/cache bookkeeping, keep this routed+shared result on
              // the device. The layer loop below passes it to
              // qwen4exp_hc_post_combine_device(), which can update the
              // device-authoritative HC state without a per-layer D2H wait.
              // If that device post-combine fails, the existing fallback
              // recovers the routed result and runs the host shared path.
            }
          }
          if (qwen_decode_grouped_host) {
            // A successful shared graph already copied the combined result to
            // `output`; do not fall through and interpret that shared-runner
            // pointer as a routed-runner output.  Only recover the routed
            // result when the shared graph was unavailable or its combined
            // host copy failed.
            if (!shared_device_handled) {
              if (!scratch.moe_device_runner ||
                  !scratch.moe_device_runner->copy_device_output_to_host(
                      scratch.moe_device_output, output)) {
                LOG_ERROR("[GGML-MOE] Qwen decode layer=%d: routed output "
                          "recovery failed\n",
                          layer_idx);
              }
              if (moe_ggml_shared_expert() && c.n_expert_shared > 0)
                run_qwen4exp_shared_host(engine, layer_idx, S, D, moe_inter,
                                         scratch.normed.data(), output, scratch,
                                         c);
            }
          } else if (qwen_decode_grouped_device && !shared_device_handled &&
                     c.n_expert_shared > 0) {
            // Device-HC fallback must recover the HC input before using the
            // established host shared-expert path.
            if (!GgmlHcRunner::instance().copy_pre_mix_to_host(
                    engine.qwen4exp_device_hc_ffn_input, scratch.normed.data(),
                    S)) {
              LOG_ERROR("[GGML-SHARED] decode layer=%d: HC input recovery "
                        "failed; shared host fallback is unsafe\n",
                        layer_idx);
            }
            if (!scratch.moe_device_runner ||
                !scratch.moe_device_runner->copy_device_output_to_host(
                    scratch.moe_device_output, output)) {
              LOG_ERROR("[GGML-MOE] Qwen decode layer=%d: routed output "
                        "recovery failed\n",
                        layer_idx);
            }
            run_qwen4exp_shared_host(engine, layer_idx, S, D, moe_inter,
                                     scratch.normed.data(), output, scratch, c);
          }
          // In the device-HC case the routed+shared graph result is the
          // actual input to qwen4exp_hc_post_combine_device(), which runs
          // later in the layer loop.  Do not clear the handoff here: doing
          // so silently skipped the MoE contribution to the device HC state
          // and left the next layer's router reading stale state.  The host
          // grouped path has already recovered/copy-combined its result and
          // can keep the old cleanup behavior.
          if (!(qwen_decode_grouped_device && shared_device_handled)) {
            scratch.moe_device_output = nullptr;
            scratch.moe_device_runner = nullptr;
            scratch.moe_shared_device_output = false;
            scratch.moe_device_output_ready = false;
          }
        }
        batched_handled = true;
        static bool logged_qwen_decode_graph_device = false;
        static bool logged_qwen_decode_graph_host = false;
        bool &logged = qwen_decode_grouped_device
                           ? logged_qwen_decode_graph_device
                           : logged_qwen_decode_graph_host;
        if (!logged) {
          LOG_INFO("[GGML-MOE] Qwen4Exp grouped decode graph active (S=1, "
                   "%s HC input)\n",
                   qwen_decode_grouped_device ? "device" : "host");
          logged = true;
        }
      }
    }
  }

  // Qwen CPU-affinity layers can use the same source-backed workflow as
  // Llama.cpp: the complete packed expert layer stays in the already-loaded
  // DDR arena, the CPU GGML graph receives the real global route IDs, and
  // only the [D] input and [D] result cross the CPU/GPU boundary. This branch
  // is intentionally
  // Qwen+decode+CPU-affinity gated; all other families and the existing RAM
  // slab fallback retain their established dispatch.
  if (!batched_handled && S == 1 && topk > 0 && n_experts > 0 &&
      engine.qwen4exp_cpu_ggml_moe &&
      engine.family_ops.family == ModelFamily::Qwen4Exp &&
      engine.family_ops.uses_hyper_connections &&
      is_cpu_affinity_layer(layer_idx) && clamp_exp == 0.0f) {
    QwenRamCpuMoeLayer ram_layer;
    if (qwen4exp_ram_cpu_moe_layer(engine, layer_idx, n_experts, ram_layer)) {
      PERF_START(qwen_cpu_ram);
      GgmlQwenCpuMoeRunner &cpu_runner =
          GgmlQwenCpuMoeRunner::instance();
      const bool shared_available =
          c.n_expert_shared > 0 &&
          engine.resident_layers.count("blk." + std::to_string(layer_idx) +
                                       ".ffn_gate_shexp.weight");
      // The old overlap experiment ran the shared expert on the host while a
      // worker computed the routed result.  That still paid three GPU dense
      // launches, a host SiLU/mul, and a host accumulation for every CPU
      // layer.  The useful split is narrower: keep the routed projections on
      // the CPU, move only the top-k weighting/reduction to CUDA, and let the
      // existing device graph do shared FFN + routed/shared addition.  This
      // is Qwen-only and opt-in so other model families retain their
      // established CPU path.
      const bool try_device_shared =
          engine.qwen4exp_cpu_gpu_overlap && shared_available &&
          qwen4exp_hc_device_live() && engine.qwen4exp_device_hc_ffn_input &&
          moe_ggml_shared_expert();

      // The compact A/B keeps the source in the preloaded DDR slabs but gives
      // GGML the same per-projection contiguous expert layout that Bee uses.
      // It is intentionally constructed from the exact routed IDs, so this
      // does not copy the other 502 experts in a Qwen layer.
      auto run_qwen_cpu_graph = [&](float *result) {
        if (!qwen_cpu_compact_graph_enabled()) {
          return cpu_runner.run_strided(
              scratch.normed.data(), result, ram_layer.meta,
              ram_layer.gate_plane, ram_layer.up_plane, ram_layer.down_plane,
              scratch.tok_topk_ids.data(), n_experts,
              scratch.tok_topk_weights.data(), D, moe_inter, topk,
              ram_layer.expert_stride);
        }

        std::vector<GgmlQwenCpuExpertRef> refs((size_t)topk);
        const auto &slots = engine.layers_experts[(size_t)layer_idx];
        for (int k = 0; k < topk; ++k) {
          const int expert_id = scratch.tok_topk_ids[(size_t)k];
          if (expert_id < 0 || expert_id >= n_experts ||
              (size_t)expert_id >= slots.size()) {
            return false;
          }
          const MoEExpertBlock &expert = slots[(size_t)expert_id];
          refs[(size_t)k] = {
              expert.gate_proj.host_ptr,
              expert.up_proj.host_ptr,
              expert.down_proj.host_ptr,
              expert.gate_proj.meta.size_bytes,
              expert.up_proj.meta.size_bytes,
              expert.down_proj.meta.size_bytes,
              static_cast<uint32_t>(expert.gate_proj.meta.type),
              static_cast<uint32_t>(expert.up_proj.meta.type),
              static_cast<uint32_t>(expert.down_proj.meta.type)};
        }
        return cpu_runner.run(
            scratch.normed.data(), result, refs.data(), topk,
            scratch.tok_topk_weights.data(), D, moe_inter, topk);
      };

      auto run_qwen_cpu_down_graph = [&](float *result) {
        return cpu_runner.run_strided_down(
            scratch.normed.data(), result, ram_layer.meta,
            ram_layer.gate_plane, ram_layer.up_plane, ram_layer.down_plane,
            scratch.tok_topk_ids.data(), n_experts, D, moe_inter, topk,
            ram_layer.expert_stride);
      };

      bool handled = false;
      bool device_shared_handled = false;
      if (try_device_shared) {
        if (qwen_cpu_gpu_combine_enabled() &&
            scratch.qwen_cpu_moe_out.resize(
                (size_t)D * (size_t)topk, "qwen_cpu_moe_down_out")) {
          handled = run_qwen_cpu_down_graph(scratch.qwen_cpu_moe_out.data());
          if (handled) {
            const int combine_device =
                moe_ggml_device() >= 0 ? moe_ggml_device() : 0;
            const void *device_routed = nullptr;
            auto &combine = GgmlQwenDeviceCombineRunner::instance();
            if (combine.ensure(combine_device, D, topk) &&
                combine.run(scratch.qwen_cpu_moe_out.data(),
                            scratch.tok_topk_weights.data(),
                            &device_routed) &&
                device_routed) {
              const void *combined = nullptr;
              device_shared_handled = run_qwen4exp_shared_device(
                  engine, layer_idx, S, D, moe_inter,
                  engine.qwen4exp_device_hc_ffn_input, device_routed, c,
                  &combined, nullptr);
              if (device_shared_handled && combined) {
                scratch.moe_device_output = combined;
                scratch.moe_shared_device_output = true;
                scratch.moe_device_output_ready = true;
              }
            }
          }
        }

        if (!device_shared_handled) {
          // Safe numerical fallback: retain the established CPU weighted
          // result and host/device shared-expert behavior if the new
          // post-down GPU combine cannot be built or accepted.
          if (scratch.qwen_cpu_moe_out.resize((size_t)D,
                                               "qwen_cpu_moe_out")) {
            handled = run_qwen_cpu_graph(scratch.qwen_cpu_moe_out.data());
            if (handled) {
              const void *combined = nullptr;
              device_shared_handled = run_qwen4exp_shared_device(
                  engine, layer_idx, S, D, moe_inter,
                  engine.qwen4exp_device_hc_ffn_input, nullptr, c, &combined,
                  scratch.qwen_cpu_moe_out.data());
              if (device_shared_handled && combined) {
                scratch.moe_device_output = combined;
                scratch.moe_shared_device_output = true;
                scratch.moe_device_output_ready = true;
              } else {
                std::memcpy(output, scratch.qwen_cpu_moe_out.data(),
                            (size_t)D * sizeof(float));
                run_qwen4exp_shared_host(engine, layer_idx, S, D, moe_inter,
                                         scratch.normed.data(), output, scratch,
                                         c);
              }
            }
          }
        }
      } else {
        handled = run_qwen_cpu_graph(output);
        if (handled && shared_available)
          run_qwen4exp_shared_host(engine, layer_idx, S, D, moe_inter,
                                   scratch.normed.data(), output, scratch, c);
      }
      PERF_END(qwen_cpu_ram, t_cpu_expert);
      if (handled) {
        perf().n_expert_cpu_calls.fetch_add((uint64_t)topk,
                                            std::memory_order_relaxed);
        batched_handled = true;
        static std::atomic<bool> logged_qwen_ram_cpu{false};
        if (!logged_qwen_ram_cpu.exchange(true, std::memory_order_relaxed))
          LOG_STREAM_INFO()
              << "[CPU-MOE] Qwen DDR-RAM-strided CPU graph active "
                 "(source=preloaded-DDR-arena, file-mmap=not-used, full expert "
                 "layer resident, layout="
              << (qwen_cpu_compact_graph_enabled() ? "compact-routed" :
                                                       "strided-slab")
              << ", post-down-combine="
              << (try_device_shared && qwen_cpu_gpu_combine_enabled() ? "GPU"
                                                                        : "CPU")
              << ", global route IDs, threads="
              << omp_get_max_threads() << ", shared expert remains GPU)\n";
        if (device_shared_handled) {
          static std::atomic<bool> logged_qwen_cpu_device_handoff{false};
          if (!logged_qwen_cpu_device_handoff.exchange(
                  true, std::memory_order_relaxed))
            LOG_STREAM_INFO()
                << "[CPU-MOE] Qwen CPU routed result handed to device "
                   "shared FFN (one pinned H2D, no host shared projections)\n";
        }
      }
    }
  }

  // --- Phase 3: execute each token's already-selected experts ---
  for (int s = 0; !batched_handled && s < S; ++s) {
    // Diagnostic upper bound on this whole per-token dispatch loop --
    // see t_moe_phase3_dispatch's own comment (perf_timer.hpp) for why
    // this is DELIBERATELY not a clean partition against gpu=/cpu=/
    // ram_scan= (all nested inside this same loop body); its own real
    // per-loop-iteration cost accumulates correctly across repeated
    // scope entry/exit (ScopedTimer's destructor ADDS elapsed, doesn't
    // overwrite), matching the S>1 prefill case too.
    PERF_SCOPE(t_moe_phase3_dispatch);
    // FIX: both routed and shared expert FFNs must consume the SAME
    // ffn_norm'd input the router scored (matching the reference: llama.cpp
    // builds one normed `cur` tensor and feeds it to both the router and
    // build_moe_ffn's per-expert matmuls). This previously fed the raw,
    // un-normalized hidden_states into every expert FFN — router weights
    // were correct but the FFN itself was operating on the wrong input.
    const float *token_hidden = scratch.normed.data() + s * D;
    float *token_output = output + s * D;

    // Weighted accumulation + VERBOSE trace, shared by both the CPU path
    // (dispatch_expert_k's own synchronous fallback) and the GPU path
    // (Pass B, accumulates after finish()) -- same logic either way, just
    // called at a different point in the pipeline depending on path.
    // Moved up (was previously defined between the loading block and the
    // old Pass A loop) so dispatch_expert_k below can capture it.
    auto accumulate = [&](int k, MoEExpertBlock *exp,
                          const float *down_buf_ptr) {
      PERF_SCOPE(t_accumulate_total);
      // Per-expert output norm, VERBOSE-gated: distinguishes "expert_ffn
      // itself computed all-zero" from "it computed something real but
      // the weighted accumulation below never applied it" -- check_finite
      // only catches NaN/Inf, never a suspiciously-exact zero, which is
      // exactly the symptom under investigation (moe_out|=0 every MoE
      // layer during prefill)
      if (trace_gate("TRACE-EXPERT-NORM")) {
        double ss = 0.0;
        for (int d = 0; d < D; ++d)
          ss += (double)down_buf_ptr[d] * down_buf_ptr[d];
        LOG_STREAM_VERBOSE()
            << "[TRACE l=" << layer_idx << "] expert k=" << k
            << " id=" << exp->expert_id
            << " requested_id=" << scratch.tok_topk_ids[s * topk + k]
            << " |down_buf|=" << std::sqrt(ss) << "\n";
      }

      // Weighted accumulation: output += weight * expert_output
      float w = scratch.tok_topk_weights[s * topk + k];
      if (std::isnan(w) || std::isinf(w)) {
        LOG_ERROR("[EXPERT-GPU] router weight itself is non-finite: layer=%d "
                  "token=%d k=%d "
                  "expert=%d w=%f -- expert_ffn's own output can be perfectly "
                  "clean and this "
                  "still poisons token_output; check "
                  "router_logits/normalization above.\n",
                  layer_idx, s, k, exp->expert_id, w);
      }
      __m512 v_w = _mm512_set1_ps(w);
      int i = 0;
      for (; i + 16 <= D; i += 16) {
        __m512 v_out = _mm512_loadu_ps(token_output + i);
        __m512 v_exp = _mm512_loadu_ps(down_buf_ptr + i);
        _mm512_storeu_ps(token_output + i, _mm512_fmadd_ps(v_w, v_exp, v_out));
      }
      for (; i < D; ++i)
        token_output[i] += w * down_buf_ptr[i];
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
    // EngineConfig::cpu_offload_max_experts: per-k flag for THIS token/
    // layer iteration only, read by dispatch_expert_k below (via gpu_
    // ctx.force_cpu_offload) and populated right after get_experts_
    // batch resolves this layer/token's genuine misses (further down).
    // Reset to all-false HERE, fresh every token/layer iteration, since
    // a shadowed (already VRAM-resident) k dispatches inline through
    // this same lambda BEFORE that later assignment ever runs -- and a
    // shadowed k must never be forced to CPU, it was never a disk-cold
    // miss to begin with.
    static thread_local std::vector<bool> force_cpu_offload_k;
    if ((int)force_cpu_offload_k.size() < topk)
      force_cpu_offload_k.resize(topk);
    std::fill(force_cpu_offload_k.begin(), force_cpu_offload_k.begin() + topk,
              false);

    // Decode normally routes one token through each selected expert.  Reuse
    // the same device-side scatter accumulator as the grouped prefill path so
    // the final [D] result is downloaded once per layer/GPU instead of once
    // per expert.  The route table is tiny at S==1, but keep it in a
    // thread-local grow-only vector so this does not allocate on every token.
    const bool decode_gpu_accum =
        S == 1 && topk > 0 && moe_gpu_accum() && ram_mgr.vram_cache;
    static thread_local std::vector<int> decode_accum_tok;
    static thread_local std::vector<float> decode_accum_w;
    bool decode_accum_begun[PerfCounters::PERF_MAX_GPU_SLOTS] = {};
    if (decode_gpu_accum) {
      decode_accum_tok.resize((size_t)topk);
      decode_accum_w.resize((size_t)topk);
      for (int k = 0; k < topk; ++k) {
        decode_accum_tok[(size_t)k] = 0;
        decode_accum_w[(size_t)k] = scratch.tok_topk_weights[s * topk + k];
      }
    }

    auto dispatch_expert_k = [&](int k) {
      // 2026-08-29, "Fifth round" of the phase3_loop-residual
      // investigation (see PerfCounters::t_shadow_precheck_total's
      // own comment) -- an upper bound over this WHOLE lambda body,
      // same "measure the whole thing, subtract known nested pieces"
      // discipline as t_moe_phase3_dispatch itself. Nests
      // t_resolve_gpu_loc_total/t_gpu_issue_total/t_gpu_finish_total
      // (already separately timed) plus this lambda's own genuinely
      // untimed bookkeeping (the gpu_ctx.loc fallback find_expert()
      // call below, ram_pin(), error-path branching).
      PERF_SCOPE(t_dispatch_expert_k_total);
      MoEExpertBlock *exp = scratch.expert_ptrs[k];
      // A shadow block (see the loading loop below) legitimately has
      // null host_ptr by design -- it's metadata-only, GPU dispatch
      // reads weight bytes from ExpertLocation::device_ptr, never
      // from here. Only a REAL (non-shadow) block missing host_ptr
      // is the genuine "something went wrong" case this check exists
      // to catch.
      if (!exp || (!exp->vram_shadow &&
                   (!exp->gate_proj.host_ptr || !exp->down_proj.host_ptr))) {
        // Previously a completely silent skip -- if this fires for
        // EVERY k this would exactly explain moe_out staying at
        // exactly 0.0 for a whole layer (not just NaN-poisoned,
        // which check_finite already covers) with zero prior
        // diagnostic pointing at it.
        LOG_ERROR(
            "[EXPERT-GPU] layer=%d token=%d k=%d expert_id=%d: expert_ptrs[k] "
            "is "
            "%s -- skipping this expert's contribution entirely (exp=%p).\n",
            layer_idx, s, k, scratch.tok_topk_ids[s * topk + k],
            !exp ? "NULL"
                 : (!exp->gate_proj.host_ptr ? "missing gate_proj.host_ptr"
                                             : "missing down_proj.host_ptr"),
            (void *)exp);
        return;
      }

      // EngineConfig::moe_decode_batched (default OFF, PROTOTYPE) --
      // decode-time (S==1) fast path. Skips the VRAM-cache resolve/lock/
      // GPU-pool machinery below entirely and goes straight to the same
      // CPU dequant-matmul call this lambda's own GPU-dispatch-failure
      // branch already uses further down (ram_pin -> expert_ffn_cpu ->
      // ram_unpin -> accumulate) -- see dense_dispatch.hpp's
      // moe_decode_batched() for the full writeup, including the real
      // trade-off (this expert can no longer be newly promoted into VRAM
      // from a decode-time hit while this path is taken for it). Never
      // engages for a vram_shadow block -- it has no host bytes to read
      // without the synchronous reload the code below already handles;
      // falls through unchanged for that case.
      if (S == 1 && !exp->vram_shadow && moe_decode_batched()) {
        ram_mgr.ram_pin(*exp);
        expert_ffn_cpu(token_hidden, scratch.down_buf.data(),
                       exp->gate_proj.host_ptr, exp->gate_proj.meta,
                       exp->up_proj.host_ptr, exp->up_proj.meta,
                       exp->down_proj.host_ptr, exp->down_proj.meta,
                       scratch.gate_buf.data(), scratch.up_buf.data(), 1, D,
                       moe_inter, clamp_exp, layer_idx, exp->expert_id);
        ram_mgr.ram_unpin(*exp);
        accumulate(k, exp, scratch.down_buf.data());
        return;
      }

      ExpertGpuContext gpu_ctx;
      gpu_ctx.topo = ram_mgr.vram_cache;
      gpu_ctx.layer_idx = layer_idx;
      gpu_ctx.expert_id = exp->expert_id;
      gpu_ctx.slab_buffer = exp->slab_buffer;
      // Real total, not slab_capacity -- same bug/reasoning as
      // decode_async_prewarm's own fix above (see its comment for the
      // full incident): slab_capacity is this RAM slot's grow-only
      // allocation, which can legitimately outlive whatever smaller
      // expert currently occupies it after an LRU reuse. This is the
      // field every synchronous cold-miss GPU upload below reads its
      // byte count from, so getting it right here covers all of them.
      gpu_ctx.slab_capacity = exp->gate_proj.meta.size_bytes +
                              exp->up_proj.meta.size_bytes +
                              exp->down_proj.meta.size_bytes;
      // Null for a shadow block (exp->vram_shadow) -- correct: a
      // shadow's slab_buffer is already null too, so none of the
      // synchronous-upload branches below (which all require
      // gpu_ctx->slab_buffer) can ever fire for it anyway; also_in_vram
      // marking is only relevant for a REAL RAM-tier block being
      // promoted, never a metadata-only shadow.
      gpu_ctx.exp = exp;
      gpu_ctx.force_cpu_offload =
          (k >= 0 && k < (int)force_cpu_offload_k.size()) &&
          force_cpu_offload_k[k];
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
                        : (ram_mgr.vram_cache ? ram_mgr.vram_cache->find_expert(
                                                    layer_idx, exp->expert_id)
                                              : nullptr);

      bool needs_release = false;
      bool loc_from_active_layer = false;
      const ExpertLocation *resolved = expert_ffn_resolve_gpu_loc(
          &gpu_ctx, 1, exp->gate_proj.meta, exp->up_proj.meta,
          exp->down_proj.meta, &needs_release, &loc_from_active_layer, ram_mgr,
          engine);

      bool gpu_issued = false;
      if (resolved) {
        if (scratch.pending_gpu.size() >= scratch.gpu_pool.size()) {
          LOG_STREAM_ERROR()
              << "[DIAG-OOB] gpu_pool OVERFLOW: pending_gpu.size()="
              << scratch.pending_gpu.size()
              << " gpu_pool.size()=" << scratch.gpu_pool.size()
              << " layer=" << layer_idx << " k=" << k
              << " expert=" << exp->expert_id << " topk=" << topk << "\n";
        }
        GpuExpertScratch &gs = scratch.gpu_pool[scratch.pending_gpu.size()];
        gpu_issued = expert_ffn_gpu_issue(
            gs, resolved, token_hidden, exp->gate_proj.meta, exp->up_proj.meta,
            1, D, moe_inter, gpu_ctx.topo);
        if (gpu_issued) {
          scratch.pending_gpu.push_back({k, exp, resolved, needs_release,
                                         loc_from_active_layer,
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
            gpu_ctx.topo->release_hold(layer_idx, gpu_ctx.expert_id,
                                       resolved->device_ptr,
                                       loc_from_active_layer);
          }
          LOG_ERROR("[EXPERT-GPU] layer=%d expert=%d: GPU issue failed "
                    "(scratch buffer "
                    "grow), falling back to CPU for this call\n",
                    layer_idx, gpu_ctx.expert_id);
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
        MoEExpertBlock *real =
            ram_mgr.get_expert(engine, layer_idx, exp->expert_id);
        if (real && real->gate_proj.host_ptr && real->down_proj.host_ptr) {
          perf().n_shadow_race_reload_success.fetch_add(
              1, std::memory_order_relaxed);
          exp = real;
          scratch.expert_ptrs[k] = real;
        } else {
          perf().n_shadow_race_reload_hard_fail.fetch_add(
              1, std::memory_order_relaxed);
          LOG_ERROR(
              "[EXPERT-GPU] layer=%d expert=%d: VRAM fast-path failed AND "
              "emergency "
              "RAM reload also failed -- skipping this expert's contribution "
              "entirely rather than read a null/stale pointer\n",
              layer_idx, exp->expert_id);
          return;
        }
      }
      if (!gpu_issued) {
        // 2026-08-29: real, hardware-confirmed race found via
        // bisection of a live SIGSEGV (multiple OMP worker threads
        // faulting at the identical instruction inside
        // dot_iq2_xxs_f32_avx512, reproducible at normal execution
        // speed, absent under gdb's slower scheduling -- classic
        // race signature, not a deterministic bug). This is the
        // ONLY reader of a RAM-tier block's raw host_ptr bytes in
        // this whole file that did NOT bracket its read with
        // ram_pin/ram_unpin (grepped every other call site: the
        // GPU-dispatch Pass A/B hold, decode_async_prewarm's
        // candidates, the speculative-promotion claim all do) --
        // without it, a concurrent prefetch-worker or active-layer-
        // upload-worker thread's own RAM-tier victim-selection scan
        // (get_expert/get_experts_batch/demote_from_vram, all of
        // which check ram_temp_pin_count) could reclaim/realloc
        // this exact slab WHILE cpu_quant_matmul's OMP threads are
        // still mid-read of gate/up/down_data below -- exactly
        // explaining every observed symptom (only on a longer
        // prefill with more CPU fallbacks, multiple threads
        // faulting at once, timing-sensitive). exp is guaranteed a
        // real RAM-tier block here (never the vram_shadow's null
        // host_ptr -- the branch above either reloads it via
        // ram_mgr.get_expert or returns early).
        ram_mgr.ram_pin(*exp);
        // 2026-08-30: a same-day diagnostic here (now removed) was
        // built while chasing a real, reproducible S=100 crash in
        // this exact dequant path, and briefly compared exp->
        // expert_id against scratch.tok_topk_ids[s*topk+k] right
        // after ram_pin -- but this comparison is WRONG in general,
        // not just here: a Hit result from claim_or_wait can be a
        // MOVE-promoted slot (move_on_promotion, do_promote_ram_
        // slot_to_vram's allow_move branch, ram_manager.hpp) --
        // expert_id is deliberately reset to -1 as part of that
        // promotion's own bookkeeping, while gate_proj/up_proj/
        // down_proj/slab_buffer remain the CORRECT, still-valid
        // bytes for exactly this expert (the RAM copy just became
        // "redundant, VRAM has it now," not wrong) -- see
        // get_experts_batch's own "claimed_this_batch" comment,
        // which already documents and relies on this exact
        // behavior. The diagnostic's own mismatch check fired on
        // this completely normal, expected case, and its early
        // return was silently discarding valid experts' contributions
        // (confirmed on real hardware: expert_calls=.../0CPU for a
        // whole request, cpu=0.00s -- every CPU-dispatched expert
        // that hit this false positive got dropped, not computed).
        // The REAL race (see claim_or_wait's own pin_on_hit
        // parameter comment) was a separate, genuine unlock-then-
        // relock gap between claim_or_wait's own return and a
        // caller pinning the slot in a SEPARATE, later lock
        // acquisition -- fixed by pinning atomically inside claim_
        // or_wait itself (get_experts_batch's Phase A/C, ram_
        // manager.hpp) before ever releasing its lock. Confirmed
        // fixed on real hardware: the same S=100 crashing config
        // that reliably segfaulted (or, without external
        // interruption, ran to completion with 187K+ NaN-cascade
        // errors) now completes cleanly, 0 errors, real hardware,
        // once ONLY the atomic pin_on_hit fix is applied (i.e.
        // without this now-removed, overly-strict identity check).
        expert_ffn_cpu(token_hidden, scratch.down_buf.data(),
                       exp->gate_proj.host_ptr, exp->gate_proj.meta,
                       exp->up_proj.host_ptr, exp->up_proj.meta,
                       exp->down_proj.host_ptr, exp->down_proj.meta,
                       scratch.gate_buf.data(), scratch.up_buf.data(), 1, D,
                       moe_inter, clamp_exp, layer_idx, exp->expert_id);
        ram_mgr.ram_unpin(*exp);
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
      const GGUFTensor *shadow_gM = nullptr, *shadow_uM = nullptr,
                       *shadow_dM = nullptr;
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
      bool layer_gpu_capable =
          !cpu_only_mode() && !is_cpu_affinity_layer(layer_idx) &&
          ram_mgr.vram_cache &&
          engine.expert_layer_geometry(layer_idx, shadow_gb, shadow_ub,
                                       shadow_db, shadow_gM, shadow_uM,
                                       shadow_dM) &&
          gpu_type_supported(shadow_gM->type) &&
          gpu_type_supported(shadow_uM->type) &&
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
      bool cufile_layer_eligible = engine.cuda_cufile_active &&
                                   !is_cpu_affinity_layer(layer_idx) &&
                                   ram_mgr.vram_cache;
      if (cufile_layer_eligible) {
        size_t cf_gb = 0, cf_ub = 0, cf_db = 0;
        const GGUFTensor *cf_gM = nullptr, *cf_uM = nullptr, *cf_dM = nullptr;
        cufile_layer_eligible =
            engine.expert_layer_geometry(layer_idx, cf_gb, cf_ub, cf_db, cf_gM,
                                         cf_uM, cf_dM) &&
            gpu_type_supported(cf_gM->type) &&
            gpu_type_supported(cf_uM->type) && gpu_type_supported(cf_dM->type);
      }

      static thread_local std::vector<int> ram_needed_ids;
      static thread_local std::vector<int> ram_needed_k;
      static thread_local std::vector<MoEExpertBlock *> ram_needed_ptrs;
      // std::vector<bool> is a bitset specialization with no .data()
      // -- use a fixed-size std::array instead (real per-layer topk
      // is always small in every model this engine supports; 64 is
      // generous headroom, matching kFetchHistogramBuckets/the 0-8
      // cpu_offload_max_experts range's own "small" assumption).
      static thread_local std::array<bool, 64> ram_was_real_fetch{};
      ram_needed_ids.clear();
      ram_needed_k.clear();
      ram_needed_ids.reserve(topk);
      ram_needed_k.reserve(topk);

      for (int k = 0; k < topk; ++k) {
        // 2026-08-29, same "Fifth round" investigation as
        // t_dispatch_expert_k_total above -- an upper bound over
        // this whole shadow-fast-path pre-check iteration
        // (find_expert() + populate_vram_shadow() for the shadow
        // check itself, plus dispatch_expert_k's own now-timed
        // body when this k IS shadowed and dispatched inline
        // below, before get_experts_batch's blocking disk read).
        PERF_SCOPE(t_shadow_precheck_total);
        int expert_id = scratch.tok_topk_ids[s * topk + k];
        bool shadowed = false;
        scratch.vram_loc_hint[k] =
            nullptr; // explicit every layer/token -- resize()'s
                     // one-time clear only covers the FIRST call
        if (layer_gpu_capable) {
          const ExpertLocation *loc =
              ram_mgr.vram_cache->find_expert(layer_idx, expert_id);
          if (loc && loc->device_ptr) {
            // Measurement only (expert_fetch_tracker.hpp) -- this
            // is a genuine VRAM-tier HIT via the shadow fast
            // path, resolved WITHOUT ever calling consume_staged.
            // Was this exact expert ALSO recently touched by the
            // async prefetcher?
            record_hit_if_recently_fetched(layer_idx, expert_id,
                                           /*is_vram_tier=*/true);
            MoEExpertBlock &shadow = scratch.vram_shadow_blocks[k];
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
            perf().n_cufile_skipped_ram_resident.fetch_add(
                1, std::memory_order_relaxed);
          } else {
            // Resolve THIS expert's absolute file offsets --
            // packed format needs expert_id*per_expert_size
            // added to the shared _exps tensor's base offset;
            // unpacked format's per-expert tensor offset IS
            // already absolute (mirrors build_expert_read_
            // requests' identical packed-vs-unpacked branching).
            // layer_is_unpacked, not the global flag -- see its own
            // comment in engine.hpp (2026-09, Phase 5: per-layer mixed
            // packed/unpacked models). Cached once per layer here since
            // it's consulted 4 times below, all needing the same answer.
            bool this_layer_unpacked = engine.layer_is_unpacked(layer_idx);
            size_t eg = 0, eu = 0, ed = 0;
            const GGUFTensor *egM = nullptr, *euM = nullptr, *edM = nullptr;
            bool have_geom =
                this_layer_unpacked
                    ? engine.unpacked_expert_geometry(layer_idx, expert_id, eg,
                                                      eu, ed, egM, euM, edM)
                    : engine.packed_slab_geometry(layer_idx, eg, eu, ed, egM,
                                                  euM, edM);
            if (have_geom) {
              uint64_t gate_off =
                  egM->offset +
                  (this_layer_unpacked ? 0 : (uint64_t)expert_id * eg);
              uint64_t up_off =
                  euM->offset +
                  (this_layer_unpacked ? 0 : (uint64_t)expert_id * eu);
              uint64_t down_off =
                  edM->offset +
                  (this_layer_unpacked ? 0 : (uint64_t)expert_id * ed);
              const cufile_io::FileHandle *h =
                  engine.model_file.cufile_handle_for_tensor(*egM);
              if (h && h->valid) {
                const ExpertLocation *cf_loc =
                    ram_mgr.vram_cache->upload_expert_cufile(
                        layer_idx, expert_id, *h, gate_off, eg, up_off, eu,
                        down_off, ed);
                if (cf_loc) {
                  MoEExpertBlock &shadow = scratch.vram_shadow_blocks[k];
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

      // EngineConfig::prefill_dispatch_lookahead_tokens (prefill-only,
      // additive) -- see that field's own comment (engine_config.hpp)
      // for the full motivation. Widens the combined get_experts_batch
      // call below to also opportunistically RAM-warm upcoming tokens'
      // not-yet-resident picks. A sentinel k of -1 marks a "lookahead
      // only" entry in ram_needed_k -- it has no (token,k) slot of its
      // own to write into or dispatch, so every one of the 4 real
      // consumers of ram_needed_k below (expert_ptrs write-back, the
      // cpu_offload_max_experts forcing loop, decode_async_prewarm's
      // candidate collection, and the final dispatch loop) must skip
      // it. Real risk this deliberately avoids: NOT promoting a
      // lookahead entry into the active-layer VRAM pool via decode_
      // async_prewarm -- only warming the RAM tier -- since this
      // codebase's own history (CLAUDE.md) has repeatedly found
      // over-eager speculative VRAM promotion to be a real regression
      // (churn/eviction), while RAM-tier warming alone has no such
      // downside (a claim_or_wait slot that's never consumed just sits
      // resident, exactly like an ordinary background prefetch hit).
      if (S > 1 && engine.prefill_dispatch_lookahead_tokens > 0 &&
          layer_gpu_capable) {
        int lookahead_end =
            std::min(S, s + 1 + engine.prefill_dispatch_lookahead_tokens);
        for (int s2 = s + 1; s2 < lookahead_end; ++s2) {
          for (int k2 = 0; k2 < topk; ++k2) {
            int expert_id2 = scratch.tok_topk_ids[s2 * topk + k2];
            if (ram_mgr.vram_cache->find_expert(layer_idx, expert_id2)) {
              perf().n_prefill_lookahead_skipped_vram.fetch_add(
                  1, std::memory_order_relaxed);
              continue;
            }
            if (ram_mgr.is_ram_resident(layer_idx, expert_id2)) {
              perf().n_prefill_lookahead_skipped_ram.fetch_add(
                  1, std::memory_order_relaxed);
              continue;
            }
            bool dup = std::find(ram_needed_ids.begin(), ram_needed_ids.end(),
                                 expert_id2) != ram_needed_ids.end();
            if (dup) {
              perf().n_prefill_lookahead_skipped_dup.fetch_add(
                  1, std::memory_order_relaxed);
              continue;
            }
            ram_needed_ids.push_back(expert_id2);
            ram_needed_k.push_back(
                -1); // sentinel: lookahead-only, no (token,k) to fill
            perf().n_prefill_lookahead_added.fetch_add(
                1, std::memory_order_relaxed);
          }
        }
      }

      if (!ram_needed_ids.empty()) {
        ram_needed_ptrs.resize(ram_needed_ids.size());
        // nullptr (rather than overflow the fixed-size array) on the
        // extremely-unlikely chance this layer's real per-layer topk
        // exceeds ram_was_real_fetch's capacity -- cpu_offload_max_
        // experts just can't engage for this one call, same
        // graceful-degrade discipline as every other GPU/best-effort
        // fast path in this codebase; every OTHER field this call
        // populates is unaffected.
        bool real_fetch_overflow =
            ram_needed_ids.size() > ram_was_real_fetch.size();
        // ram_needed_k.data() doubles as the caller_k parameter --
        // its own -1 sentinel (prefill_dispatch_lookahead_tokens
        // additions) is exactly what get_experts_batch needs to tag
        // MoEExpertBlock::loaded_by_lookahead correctly, no separate
        // array required.
        if (S == 1 && engine.family_ops.family == ModelFamily::Qwen4Exp)
          qwen4exp_record_pilot_fetch_start(layer_idx);
        ram_mgr.get_experts_batch(
            engine, layer_idx, ram_needed_ids.data(),
            (int)ram_needed_ids.size(), ram_needed_ptrs.data(),
            real_fetch_overflow ? nullptr : ram_was_real_fetch.data(),
            ram_needed_k.data());
        // 2026-08-30: real, hardware-confirmed bug found while
        // debugging a S=100 crash/NaN-corruption with speculative_
        // promote_to_active_layer + active_layer_upload_worker_
        // threads both enabled -- see dispatch_expert_k's own
        // RAM-IDENTITY diagnostic for the full investigation.
        // get_experts_batch's Phase A (ram_manager.hpp) now pins
        // every out[k] it resolves (Hit and Claimed) at the MOMENT
        // of resolution -- see that function's own comment for why
        // "pin right before use" (the 2026-08-29 fix, dispatch_
        // expert_k's own CPU-path pin below) is insufficient: the
        // slot can already be reclaimed before that later pin ever
        // runs, including DURING get_experts_batch's own Phase B/C
        // execution (real blocking I/O, real time for another
        // thread's victim-selection scan to run). This loop is
        // that pin's matching release:
        //   - a REAL entry (ram_needed_k[i]>=0) is held through
        //     THIS token's own full dispatch instead -- released
        //     right after dispatch_expert_k(k) returns, in the
        //     ram_needed_k dispatch loop below (covers both the CPU
        //     path's synchronous compute and the GPU path's
        //     synchronous upload_expert/upload_expert_active call).
        //   - a LOOKAHEAD-only entry (ram_needed_k[i]<0,
        //     prefill_dispatch_lookahead_tokens) belongs to a
        //     LATER token, never dispatched by this one -- once
        //     get_experts_batch returns it's fully published
        //     (Ready, not just Loading), safe to treat exactly like
        //     any other resident expert from here on (a later
        //     token's own Phase A will find+use it normally); held
        //     no longer than necessary, released right here.
        for (size_t i = 0; i < ram_needed_ids.size(); ++i) {
          if (ram_needed_k[i] < 0) {
            ram_mgr.ram_unpin(*ram_needed_ptrs[i]);
            continue;
          }
          scratch.expert_ptrs[ram_needed_k[i]] = ram_needed_ptrs[i];
        }

        // Qwen-only CPU-affinity grouping. At this point every selected
        // expert has a published RAM block and get_experts_batch has already
        // pinned it for the dispatch that follows. Consume the whole token in
        // one grouped gate/up pass when the IQ1_M geometry permits it, then
        // clear the normal per-expert work lists so the legacy active-layer
        // prewarm/upload/dispatch code becomes a no-op for this token. Other
        // families, mixed quant signatures, and non-affinity layers retain
        // the established path byte-for-byte.
        if (S == 1 && engine.qwen4exp_cpu_ggml_moe &&
            engine.family_ops.family == ModelFamily::Qwen4Exp &&
            engine.family_ops.uses_hyper_connections &&
            is_cpu_affinity_layer(layer_idx) &&
            ram_needed_k.size() == (size_t)topk) {
          std::vector<MoEExpertBlock *> grouped_experts((size_t)topk);
          bool complete = true;
          for (int k = 0; k < topk; ++k) {
            grouped_experts[(size_t)k] = scratch.expert_ptrs[k];
            if (!grouped_experts[(size_t)k] ||
                grouped_experts[(size_t)k]->vram_shadow)
              complete = false;
          }
          const float *grouped_weights =
              scratch.tok_topk_weights.data() + (size_t)s * topk;
          if (complete &&
              expert_ffn_cpu_qwen_grouped_quantized(
                  token_hidden, token_output, grouped_experts, grouped_weights,
                  D, moe_inter, clamp_exp, layer_idx)) {
            static std::atomic<bool> logged_grouped_cpu{false};
            if (!logged_grouped_cpu.exchange(true, std::memory_order_relaxed))
              LOG_STREAM_INFO()
                  << "[CPU-MOE] Qwen grouped IQ1_M CPU path active at layer "
                  << layer_idx << " (topk=" << topk
                  << ", shared activation quantization)\n";
            for (MoEExpertBlock *block : ram_needed_ptrs)
              if (block)
                ram_mgr.ram_unpin(*block);
            ram_needed_ids.clear();
            ram_needed_k.clear();
            ram_needed_ptrs.clear();
          }
        }

        // EngineConfig::cpu_offload_max_experts: of THIS token/
        // layer's genuine real-fetch misses (ram_was_real_fetch[i],
        // set by get_experts_batch's own Phase A -- the exact
        // population [[LAYER-MISS]]'s data was gathered from, see
        // that field's own comment), redirect up to the configured
        // cap to CPU compute (force_cpu_offload_k[k], read by
        // dispatch_expert_k below) instead of letting them continue
        // to a VRAM upload. Scoped OUT of cpu_affinity_layers layers
        // (already 100% CPU by a standing config, nothing to gain by
        // also computing this) and OUT of cpu_only_mode() (the
        // diagnostic kill-switch already forces everything to CPU;
        // running this loop there would just waste cycles double-
        // deciding an outcome that's already fixed). A Hit/Loading
        // entry (ram_was_real_fetch[i]==false) never touched disk
        // this call and is never eligible -- forcing it to CPU would
        // throw away a cache hit for no reason.
        if (engine.cpu_offload_max_experts > 0 && !cpu_only_mode() &&
            !is_cpu_affinity_layer(layer_idx) && !real_fetch_overflow) {
          int budget = engine.cpu_offload_max_experts;
          for (size_t i = 0; i < ram_needed_ids.size(); ++i) {
            // A lookahead-only entry (ram_needed_k[i]==-1) isn't
            // this token's own real fetch to force to CPU or
            // not -- skip before any k-indexed access.
            if (ram_needed_k[i] < 0)
              continue;
            if (!ram_was_real_fetch[i])
              continue;
            if (budget > 0) {
              force_cpu_offload_k[ram_needed_k[i]] = true;
              --budget;
              perf().n_cpu_offload_forced.fetch_add(1,
                                                    std::memory_order_relaxed);
            } else {
              perf().n_cpu_offload_left_uncapped.fetch_add(
                  1, std::memory_order_relaxed);
            }
          }
        }

        // Batch-async pre-warm this token's own RAM->VRAM active-
        // layer misses BEFORE dispatching them one at a time below --
        // mirrors prefill Phase 2's own upload_expert_active_async +
        // sync_active_layer_batch pattern (gpu_topology.cpp),
        // applied to this TOKEN's real ram_needed_k set instead of a
        // whole layer's predicted candidates. Real, measured DMA-level
        // win on both GPUs in this box via tests/bench_dma_sync_vs_
        // async.cu -- see that file's header comment.
        //
        // DEFAULT OFF (engine.decode_async_prewarm) and CHUNKED
        // (engine.decode_async_prewarm_max_batch, default 3), NOT
        // the simple "batch everything in one shot" shape this
        // started as -- that unbounded version caused a real
        // production incident (host RAM exhaustion / swap thrashing
        // / freeze) on a config running host_ram_gb close to
        // physical RAM. Processing in small chunks, each with its
        // own full pin -> issue -> sync -> unpin cycle before the
        // next chunk starts, bounds how many RAM slots are
        // simultaneously pinned and how many active-layer slots are
        // simultaneously mid-DMA at any one instant -- trading some
        // batching benefit for tokens with more misses than
        // max_batch, in exchange for a hard cap on peak simultaneous
        // footprint. See EngineConfig::decode_async_prewarm's own
        // comment (engine_config.hpp) for the full incident writeup.
        //
        // Only ever targets the ACTIVE-LAYER pool (the one pool with
        // an existing async+batch-sync primitive) -- a model without
        // one allocated (use_active_layer_area=false, e.g. DeepSeek-
        // V4) makes upload_expert_active_async a guaranteed no-op
        // nullptr return for every candidate, so this whole block
        // degrades to zero-cost no-op and dispatch_expert_k's
        // existing main-cache (upload_expert) sync path runs
        // completely unchanged for those models.
        // 2026-08-19: a dispatch-reordering fix was TRIED and
        // MEASURED here (issue dispatch_expert_k as soon as each
        // piece of data is ready -- immediately for non-prewarm
        // candidates, right after each chunk's own sync for
        // prewarmed ones -- instead of deferring every ram_needed
        // dispatch to one final loop after ALL decode_async_prewarm
        // chunks finish), on the theory that queuing GPU compute
        // earlier would give later chunks' H2D DMA real, already-
        // in-flight work to overlap with (motivated by n_active_
        // prewarm_saw_compute_stream_busy showing 0% busy at this
        // call site -- see that counter's own comment, perf_timer.hpp).
        // REVERTED after real-hardware A/B: the SAME diagnostic still
        // read compute_stream busy 0/15092 (0.0%) even with compute
        // issued earlier -- i.e. this decode-time model's per-token
        // GPU kernels are small/fast enough that they finish before
        // the CPU-side orchestration (locks, identity checks, the
        // next chunk's own pin/issue calls) loops back around to
        // check again, so there was never any slow GPU work to hide
        // DMA behind in the first place. Decode throughput was
        // unchanged (0.86 tok/s, same band as before), output stayed
        // coherent, full test suite passed -- a clean, well-
        // instrumented negative result, not a bug, but real added
        // control-flow complexity in this exact historically
        // incident-prone area for zero measured benefit, so reverted
        // rather than kept. This is now the THIRD independent real-
        // hardware confirmation (after demote_from_vram's and
        // upload_expert_active's own stream-busy findings, both
        // documented in CLAUDE.md) of the same underlying fact: this
        // hardware's decode-time dispatch loop is genuinely CPU-
        // orchestration-bound per call, not schedulable-around by
        // reordering when work gets issued. See CLAUDE.md's
        // 2026-08-19 entry for the full writeup.
        if (engine.decode_async_prewarm && !qwen4exp_decode_ggml_path(engine) &&
            !cpu_only_mode() && !is_cpu_affinity_layer(layer_idx) &&
            ram_mgr.vram_cache) {
          static thread_local std::vector<size_t> decode_async_candidates;
          decode_async_candidates.clear();
          for (size_t i = 0; i < ram_needed_ids.size(); ++i) {
            // A lookahead-only entry (ram_needed_k[i]==-1,
            // prefill_dispatch_lookahead_tokens) is deliberately
            // NEVER a candidate for active-layer VRAM pre-warm --
            // only for RAM-tier warming above. Promoting a not-
            // yet-needed expert into the scarce active-layer pool
            // this early is exactly the over-eager speculative-
            // promotion regression this codebase has repeatedly
            // found and reverted (see CLAUDE.md) -- and reading
            // force_cpu_offload_k[-1] just below would be OOB.
            if (ram_needed_k[i] < 0)
              continue;
            MoEExpertBlock *blk = ram_needed_ptrs[i];
            // Cheap pre-filter only, NOT the safety-critical check
            // (that one is re-done immediately adjacent to
            // ram_pin below, right before use -- see that
            // comment for why re-checking there, not trusting
            // this earlier read, is what actually matters).
            // Still worth filtering here too, to avoid adding a
            // definitely-invalid candidate to the list at all.
            if (!blk || !blk->slab_buffer || blk->slab_capacity == 0)
              continue;
            // Already capped to CPU compute this token (see
            // force_cpu_offload_k above) -- pre-warming its VRAM
            // slot would pay for exactly the DMA this feature
            // exists to skip, for an expert dispatch_expert_k is
            // about to force off the GPU path anyway.
            if (force_cpu_offload_k[ram_needed_k[i]])
              continue;
            int expert_id = ram_needed_ids[i];
            // Side-effect-free peek (active_layer_has, NOT
            // find_active_expert -- the real lookup would bump
            // hit_count/could promote as a side effect of a mere
            // pre-check, corrupting the exact state this batch is
            // simultaneously choosing victims from). Already-
            // resident candidates need no upload -- dispatch_
            // expert_k's own find_active_expert call will hit
            // them directly.
            if (ram_mgr.vram_cache->active_layer_has(layer_idx, expert_id))
              continue;
            bool gpu_capable = gpu_type_supported(blk->gate_proj.meta.type) &&
                               gpu_type_supported(blk->up_proj.meta.type) &&
                               gpu_type_supported(blk->down_proj.meta.type);
            if (!gpu_capable)
              continue;
            decode_async_candidates.push_back(i);
          }

          static thread_local std::vector<ExpertLocation *>
              decode_async_victims;
          static thread_local std::vector<MoEExpertBlock *>
              decode_async_pinned_blocks;
          size_t max_batch =
              (size_t)std::max(1, engine.decode_async_prewarm_max_batch);
          for (size_t chunk_start = 0;
               chunk_start < decode_async_candidates.size();
               chunk_start += max_batch) {
            size_t chunk_end = std::min(decode_async_candidates.size(),
                                        chunk_start + max_batch);
            decode_async_victims.clear();
            decode_async_pinned_blocks.clear();
            for (size_t ci = chunk_start; ci < chunk_end; ++ci) {
              size_t i = decode_async_candidates[ci];
              MoEExpertBlock *blk = ram_needed_ptrs[i];
              int expert_id = ram_needed_ids[i];
              // FIX (real bug behind a real production
              // incident's "[ERROR] [GPU] Invalid upload_
              // expert_active_async parameters" log line): the
              // FIRST version of this code checked slab_buffer
              // once in the filter pass above, then did MORE
              // lock-taking work (active_layer_has, gpu_
              // capable) before actually pinning and using it
              // -- a real TOCTOU window in which a concurrent
              // prefetch-worker thread's own ORDINARY (not
              // even the buggy "fallback to slot 0") victim-
              // selection could legitimately reclaim this
              // exact, still-unprotected RAM slot before this
              // code got around to protecting it. The fix is
              // this exact ordering: re-check freshly, pin,
              // THEN re-read slab_buffer/slab_capacity AGAIN
              // (not the filter pass's now-possibly-stale
              // values) immediately after pinning -- ram_pin
              // is what actually excludes this slot from
              // claim_or_wait's victim-selection from this
              // point on (never_pick checks ram_temp_pin_
              // count>0 under the same lock), so only a value
              // read AFTER the pin is guaranteed protected.
              // Nothing else (no lock-taking calls) sits
              // between this check and the pin below.
              if (!blk || !blk->slab_buffer || blk->slab_capacity == 0)
                continue;
              ram_mgr.ram_pin(*blk);
              // Identity check, not just non-null/non-zero:
              // if this slot was reclaimed AND already
              // repopulated for a DIFFERENT expert in the
              // window before the pin above, slab_buffer/
              // slab_capacity could both look perfectly valid
              // while actually belonging to someone else's
              // data -- a silent wrong-bytes-uploaded-under-
              // the-wrong-label bug, worse than a clean error.
              // Confirms this slot still actually holds the
              // expert we intend before trusting its bytes.
              if (blk->expert_id != expert_id) {
                ram_mgr.ram_unpin(*blk);
                perf().n_decode_async_prewarm_stale_caught.fetch_add(
                    1, std::memory_order_relaxed);
                continue;
              }
              const uint8_t *host_data = blk->slab_buffer;
              // real total, not slab_capacity -- same bug/reasoning as
              // ram_manager.hpp's promote_ram_to_active_layer (see its own
              // comment for the full incident): slab_capacity is this
              // slot's grow-only allocation, which can legitimately
              // outlive whatever smaller expert currently occupies it
              // after an LRU reuse -- uploading it copies real data
              // followed by a previous occupant's stale tail bytes.
              size_t host_bytes = blk->gate_proj.meta.size_bytes +
                                  blk->up_proj.meta.size_bytes +
                                  blk->down_proj.meta.size_bytes;
              if (!host_data || host_bytes == 0) {
                // Reclaimed (freed, not yet repopulated) in
                // the now-negligible window between the
                // check above and the pin -- extremely
                // unlikely (no lock-taking work in between
                // anymore) but cheap to guard against rather
                // than assume can't happen.
                ram_mgr.ram_unpin(*blk);
                perf().n_decode_async_prewarm_stale_caught.fetch_add(
                    1, std::memory_order_relaxed);
                continue;
              }
              // The async DMA below keeps reading host_data in
              // the background until sync_active_layer_batch's
              // cudaStreamSynchronize confirms it landed --
              // ram_pin above is what protects that window
              // (bounded further by chunking); unpinned again
              // immediately below once this CHUNK's sync
              // confirms completion.
              // 2026-08-27: real coverage measurement -- see
              // n_active_layer_upload_src_pinned's own comment
              // (perf_timer.hpp).
              if (blk->slab_pinned)
                perf().n_active_layer_upload_src_pinned.fetch_add(
                    1, std::memory_order_relaxed);
              else {
                perf().n_active_layer_upload_src_unpinned.fetch_add(
                    1, std::memory_order_relaxed);
                perf().mark_unpinned_touch(layer_idx, expert_id);
              }
              // 2026-08-28: multi-GPU routing decision -- ci
              // (this candidate's position within decode_
              // async_candidates) preserves ram_needed_ids'
              // own top-K priority order (the filter pass
              // that built decode_async_candidates only
              // ever skips entries, never reorders), so it's
              // a real, if not perfectly 1:1, priority
              // signal for the router. See route_active_
              // layer_gpu's own comment (dense_dispatch.hpp).
              // total_candidates = decode_async_candidates.
              // size() -- the REAL count for THIS transition,
              // not a fixed nominal 8 (which real hardware
              // showed made the tail unreachable, since this
              // list is typically much shorter than 8 on a
              // well-cached run).
              int routed_gpu =
                  active_layer_multi_gpu_enabled()
                      ? route_active_layer_gpu(
                            (int)ci, (int)decode_async_candidates.size())
                      : -1;
              if (routed_gpu >= 0) {
                perf()
                    .n_active_layer_routed_by_gpu
                        [routed_gpu < PerfCounters::PERF_MAX_GPU_SLOTS
                             ? routed_gpu
                             : 0]
                    .fetch_add(1, std::memory_order_relaxed);
              }
              // 2026-08-28: an UNPINNED source's inline call
              // secretly blocks this (decode) thread for
              // ~1.7ms (a driver-internal staged copy -- see
              // ActiveLayerUploadWorkers' own header comment
              // for the full evidence chain) -- route those
              // through the worker pool instead when it's
              // enabled. A pinned source's inline call is
              // already near-free, so it's left exactly as-is
              // regardless of the pool's availability.
              bool use_worker_pool = !blk->slab_pinned &&
                                     ram_mgr.upload_workers &&
                                     ram_mgr.upload_workers->enabled();
              if (use_worker_pool) {
                ExpertLocation *claimed =
                    ram_mgr.vram_cache->claim_active_layer_slot_for_worker(
                        layer_idx, expert_id, host_bytes, /*speculative=*/false,
                        routed_gpu);
                if (claimed && ram_mgr.upload_workers->try_submit(
                                   ActiveLayerUploadWorkItem{
                                       claimed, blk, host_data, host_bytes})) {
                  // Worker owns the RAM-source unpin (via
                  // ram_mark_pending_copy_and_unpin, once it
                  // genuinely dispatches) -- this thread's
                  // own ram_pin above must NOT be released
                  // here.
                  perf().n_decode_async_prewarm_issued.fetch_add(
                      1, std::memory_order_relaxed);
                } else {
                  // Either the claim itself failed (pool
                  // full) or try_submit rejected it (queue
                  // full) -- the latter needs an explicit
                  // abandon to release the claim's own
                  // hold; the former has nothing to
                  // abandon.
                  if (claimed)
                    ram_mgr.vram_cache->abandon_worker_claim(claimed);
                  ram_mgr.ram_unpin(*blk);
                  perf().n_decode_async_prewarm_no_slot.fetch_add(
                      1, std::memory_order_relaxed);
                }
              } else {
                ExpertLocation *v =
                    ram_mgr.vram_cache->upload_expert_active_async(
                        layer_idx, expert_id, host_data, host_bytes,
                        /*speculative=*/false, routed_gpu);
                if (v) {
                  if (engine.decode_async_prewarm_nonblocking) {
                    // 2026-08-28: no batch sync at all for
                    // this k -- hand the RAM-source
                    // protection off to pending_copy_event
                    // (self-healing, polled) right now
                    // instead of holding ram_temp_pin_count
                    // until a later blocking wait. The
                    // active-layer DESTINATION side needs no
                    // change: find_active_expert's own poll-
                    // before-trust check (gpu_topology.cpp)
                    // already treats "not landed yet" as a
                    // plain miss for ANY active-layer slot,
                    // not just speculative ones. See
                    // EngineConfig::decode_async_prewarm_
                    // nonblocking's own comment.
                    ram_mgr.ram_mark_pending_copy_and_unpin(*blk,
                                                            v->upload_event);
                  } else {
                    decode_async_victims.push_back(v);
                    decode_async_pinned_blocks.push_back(blk);
                  }
                  perf().n_decode_async_prewarm_issued.fetch_add(
                      1, std::memory_order_relaxed);
                } else {
                  // Pool full/alloc failed -- no async copy was
                  // actually issued, nothing to protect; this k
                  // falls through to dispatch_expert_k's
                  // existing synchronous fallback path.
                  ram_mgr.ram_unpin(*blk);
                  perf().n_decode_async_prewarm_no_slot.fetch_add(
                      1, std::memory_order_relaxed);
                }
              }
            }
            if (!decode_async_victims.empty()) {
              // Only ever populated when decode_async_prewarm_
              // nonblocking is false (see the branch above) --
              // the nonblocking path never pushes into this
              // vector at all, so this whole block is a clean
              // no-op under the new mode, not merely unused.
              ram_mgr.vram_cache->sync_active_layer_batch(decode_async_victims);
              for (auto *blk : decode_async_pinned_blocks)
                ram_mgr.ram_unpin(*blk);
              perf().n_decode_async_prewarm_synced.fetch_add(
                  decode_async_victims.size(), std::memory_order_relaxed);
              perf().n_decode_async_prewarm_chunks.fetch_add(
                  1, std::memory_order_relaxed);
            }
          }
        }

        // Dispatch whatever get_experts_batch (and the pre-warm pass
        // above, if enabled) just resolved. The shadowed experts
        // above already had their kernels issued inline, before the
        // blocking get_experts_batch call ran -- only the genuine
        // misses are dispatched here, after their data is ready. Any
        // expert pre-warmed above is now active-layer-resident with
        // upload_ready=true, so expert_ffn_resolve_gpu_loc's find_
        // active_expert call inside dispatch_expert_k will hit it
        // directly instead of paying for its own synchronous upload
        // -- everything else (not pre-warmed: feature off, active-
        // layer pool full, unsupported type, or no active-layer area
        // on this model) takes the original synchronous path exactly
        // as before.
        {
          // Same "Fifth round" investigation as shadow_precheck
          // above -- this is the OTHER (mutually exclusive)
          // half of every dispatch_expert_k call this token/layer:
          // shadow_precheck covers shadowed k's (dispatched
          // inline, before get_experts_batch); this covers the
          // genuine RAM-tier misses (dispatched after). Together
          // they should account for ~all of t_dispatch_expert_k_
          // total's own aggregate, letting phase3_loop's residual
          // be decomposed additively: shadow_precheck + ram_scan
          // (already reported, covers get_experts_batch in
          // between the two) + this + gpu_finish/accumulate
          // (Pass B, below) + shared_expert + layer_cleanup +
          // active_layer_prewarm_issue/sync should sum close to
          // phase3_loop's own total, with whatever's left being
          // the genuinely still-unexplained residual.
          PERF_SCOPE(t_ram_needed_dispatch_total);
          for (size_t i = 0; i < ram_needed_k.size(); ++i) {
            // A lookahead-only entry (-1) belongs to a later
            // token -- never dispatched by this one; it's now
            // sitting RAM-resident, ready for a cheap resolve
            // when its own token's turn comes up. (Its own pin
            // was already released right after get_experts_
            // batch returned -- see that call site's own
            // comment.)
            if (ram_needed_k[i] < 0)
              continue;
            dispatch_expert_k(ram_needed_k[i]);
            // 2026-08-30: matching release for get_experts_
            // batch's pin-at-resolution fix (ram_manager.hpp) --
            // this k's dispatch (CPU synchronous compute, or
            // GPU's synchronous upload_expert/upload_expert_
            // active at this S>=GPU_COLD_BATCH_THRESHOLD prefill
            // scale) has now fully consumed the RAM-tier bytes,
            // so it's safe to drop this reference-lifetime pin.
            // ram_needed_ptrs[i] is the exact slot get_experts_
            // batch pinned for this same i (identical index
            // space -- both arrays are built together above).
            ram_mgr.ram_unpin(*ram_needed_ptrs[i]);
          }
        }
      }
    }

    // --- Pass B, batched (2026-08-29): finish every GPU-issued expert
    // queued above, grouped by loc->gpu_id rather than synchronizing per
    // expert.  With device SwiGLU and moe_gpu_accum enabled (the normal
    // qwen4exp decode configuration), stage1 and the scatter accumulate are
    // both queued on the legacy stream and the accumulator's one final wait
    // per GPU confirms the whole chain.  A first wait remains only when a
    // host gate/up read is required; a second wait remains only for a legacy
    // per-expert host-output fallback.  This is the same cache-safety
    // invariant as the old two-round implementation, but it avoids waits
    // that no longer guard a host-visible dependency.
    //
    // gate_buf/up_buf/down_buf are SHARED host buffers (scratch's own,
    // one copy, not one per pending expert) -- stage1 is safe to call
    // back-to-back for every pending expert because each call fully
    // consumes gate_buf/up_buf (uploads them to that expert's OWN
    // per-slot d_prod) before returning, so the next stage1 call
    // overwriting them is never a race. Stage2+accumulate, by
    // contrast, MUST stay paired per-expert in the same iteration --
    // down_buf can only hold one expert's output at a time, so
    // accumulate() has to consume it before the next stage2 call
    // overwrites it. This mirrors, not changes, a constraint the
    // original single-function design already had implicitly (its own
    // per-expert loop already fully finished and accumulated one
    // expert before starting the next).
    if (!scratch.pending_gpu.empty()) {
      // With device SwiGLU there is no host read between the gate/up issue
      // and stage1: stage1 only queues more work on the same legacy stream.
      // Avoid the first batch wait in that case.  If the GPU activation
      // scratch cannot be prepared (or tracing forces the host activation
      // path), retain the old wait before stage1's blocking gate/up copies.
      const bool activation_requested =
          expert_gpu_activation() && !trace_gate("SILU-TRACE-GPU") &&
          !trace_gate("SWIGLU-TRACE-GPU") && !trace_gate("EXPERT-GPU") &&
          !trace_gate("INPUT-TRACE-GPU-PAIRED");
      bool all_device_activation = activation_requested;
      if (all_device_activation) {
        for (size_t pi = 0; pi < scratch.pending_gpu.size(); ++pi) {
          if (!scratch.gpu_pool[pi].ensure_bad_counter()) {
            all_device_activation = false;
            break;
          }
        }
      }

      bool synced[PerfCounters::PERF_MAX_GPU_SLOTS] = {};
      if (!all_device_activation) {
        PERF_START(gpu_batch_sync1);
        for (auto &p : scratch.pending_gpu) {
          int gid = (p.loc->gpu_id >= 0 &&
                     p.loc->gpu_id < PerfCounters::PERF_MAX_GPU_SLOTS)
                        ? p.loc->gpu_id
                        : 0;
          if (!synced[gid]) {
            cudaSetDevice(p.loc->gpu_id);
            cudaStreamSynchronize(0);
            synced[gid] = true;
            perf().n_gpu_batch_sync_calls.fetch_add(1,
                                                    std::memory_order_relaxed);
          }
        }
        PERF_END(gpu_batch_sync1, t_gpu_batch_sync);
      }

      // The decode route table has one row per selected expert.  Upload and
      // clear the accumulator once per GPU before stage1 starts, so each
      // expert can scatter its d_down directly into the final token result.
      if (decode_gpu_accum) {
        for (auto &p : scratch.pending_gpu) {
          const int gid = p.loc->gpu_id;
          if (gid < 0 || gid >= PerfCounters::PERF_MAX_GPU_SLOTS ||
              decode_accum_begun[gid])
            continue;
          decode_accum_begun[gid] = engine.gpu->expert_accum_begin(
              gid, 1, D, decode_accum_tok.data(), decode_accum_w.data(), topk);
        }
      }

      for (size_t pi = 0; pi < scratch.pending_gpu.size(); ++pi) {
        PendingGpuExpert &p = scratch.pending_gpu[pi];
        GpuExpertScratch &gs = scratch.gpu_pool[pi];
        expert_ffn_gpu_finish_stage1(
            gs, p.loc, scratch.gate_buf.data(), scratch.up_buf.data(),
            p.exp->gate_proj.meta, p.exp->down_proj.meta, layer_idx,
            p.expert_id, 1, D, moe_inter, clamp_exp, ram_mgr.vram_cache);
      }

      // Queue the device scatter immediately after each expert's down kernel.
      // The default stream orders down -> scatter, so a second batch wait is
      // needed only for experts that still require the legacy host download.
      bool need_sync2[PerfCounters::PERF_MAX_GPU_SLOTS] = {};
      if (decode_gpu_accum) {
        for (size_t pi = 0; pi < scratch.pending_gpu.size(); ++pi) {
          PendingGpuExpert &p = scratch.pending_gpu[pi];
          const int gid = p.loc->gpu_id;
          if (gid >= 0 && gid < PerfCounters::PERF_MAX_GPU_SLOTS &&
              decode_accum_begun[gid]) {
            p.accumulated_on_device = engine.gpu->expert_accum_add(
                gid, scratch.gpu_pool[pi].d_down, p.k, 1, D, 1);
            if (p.accumulated_on_device)
              perf().n_expert_gpu_calls.fetch_add(1, std::memory_order_relaxed);
          }
          if (!p.accumulated_on_device) {
            const int sync_gid =
                (gid >= 0 && gid < PerfCounters::PERF_MAX_GPU_SLOTS) ? gid : 0;
            need_sync2[sync_gid] = true;
          }
        }
      } else {
        for (auto &p : scratch.pending_gpu) {
          const int gid = (p.loc->gpu_id >= 0 &&
                           p.loc->gpu_id < PerfCounters::PERF_MAX_GPU_SLOTS)
                              ? p.loc->gpu_id
                              : 0;
          need_sync2[gid] = true;
        }
      }

      std::fill(std::begin(synced), std::end(synced), false);
      bool any_sync2 = false;
      for (bool needed : need_sync2)
        any_sync2 |= needed;
      if (any_sync2) {
        PERF_START(gpu_batch_sync2);
        for (auto &p : scratch.pending_gpu) {
          int gid = (p.loc->gpu_id >= 0 &&
                     p.loc->gpu_id < PerfCounters::PERF_MAX_GPU_SLOTS)
                        ? p.loc->gpu_id
                        : 0;
          if (need_sync2[gid] && !synced[gid]) {
            cudaSetDevice(p.loc->gpu_id);
            cudaStreamSynchronize(0);
            synced[gid] = true;
            perf().n_gpu_batch_sync_calls.fetch_add(1,
                                                    std::memory_order_relaxed);
          }
        }
        PERF_END(gpu_batch_sync2, t_gpu_batch_sync);
      }

      for (size_t pi = 0; pi < scratch.pending_gpu.size(); ++pi) {
        PendingGpuExpert &p = scratch.pending_gpu[pi];
        GpuExpertScratch &gs = scratch.gpu_pool[pi];
        if (p.accumulated_on_device)
          continue;
        // This sync round just confirmed EVERY kernel that reads
        // this slab (gate, up, AND down -- the last of the three)
        // has actually finished executing, not just that launches
        // completed -- so it's safe to release the temporary hold
        // now, exactly the same invariant the original single-
        // function design relied on, just satisfied by the shared
        // batch sync above instead of a per-call one.
        expert_ffn_gpu_finish_stage2(
            p.loc, scratch.down_buf.data(), layer_idx, p.expert_id, 1, D,
            ram_mgr.vram_cache, p.needs_release, p.loc_from_active_layer, gs);
        // Matches Pass A's increment -- release the RAM-tier protection
        // now that expert_ffn_gpu_finish_stage2's own release_hold()
        // call (which uses p.expert_id, captured at Pass A issue time
        // -- see PendingGpuExpert's own comment for why NOT
        // p.exp->expert_id) has already happened. Must be the LAST
        // thing done with p.exp's identity in this iteration; this
        // slot becomes a valid victim-selection candidate again
        // immediately after this decrement. Goes through
        // RAMSafetyManager::ram_unpin -- see Pass A's matching
        // ram_pin call for why this is no longer a direct field
        // decrement.
        ram_mgr.ram_unpin(*p.exp);
        accumulate(p.k, p.exp, scratch.down_buf.data());
      }

      // One [D] download per active GPU for all device-accumulated routed
      // experts in this token/layer.  A GPU that also had a legacy fallback
      // was already synchronized above, so avoid paying a duplicate wait.
      for (int gid = 0; gid < PerfCounters::PERF_MAX_GPU_SLOTS; ++gid) {
        if (!decode_accum_begun[gid])
          continue;
        const bool already_synced = need_sync2[gid] && synced[gid];
        if (!engine.gpu->expert_accum_end(gid, token_output, 1, D,
                                          already_synced))
          perf().n_moe_gpu_accum_fallback.fetch_add(1,
                                                    std::memory_order_relaxed);
      }
      for (auto &p : scratch.pending_gpu) {
        if (!p.accumulated_on_device)
          continue;
        // The accumulator end above has waited for down and scatter, so the
        // temporary VRAM hold can now be released without reopening the
        // eviction-during-compute race.  The RAM pin has the same lifetime.
        if (p.needs_release && ram_mgr.vram_cache)
          ram_mgr.vram_cache->release_hold(layer_idx, p.expert_id,
                                           p.loc->device_ptr,
                                           p.loc_from_active_layer);
        ram_mgr.ram_unpin(*p.exp);
      }
    }

    check_finite("token_output (post routed experts, pre-shared)", layer_idx,
                 -1, token_output, (size_t)D);

    // Shared expert (always-active) — GLM-5.2/DeepSeek-V3/Hunyuan MoE
    // layers can have n_expert_shared always-on experts whose combined
    // FFN (intermediate width n_expert_shared * moe_inter — NOT just
    // moe_inter, confirmed against deepseek2.cpp's shexp tensor shapes)
    // is added to every token's routed-expert output. Unscaled: only the
    // routed weights get expert_weights_scale (see build_moe_ffn — w_scale
    // is applied to the gathered top-k weights, never to the shared path).
    std::string sp = "blk." + std::to_string(layer_idx) + ".";
    if (c.n_expert_shared > 0 &&
        engine.resident_layers.count(sp + "ffn_gate_shexp.weight")) {
      PERF_SCOPE(t_shared_expert_total);
      int shared_inter = moe_inter * c.n_expert_shared;
      dense_proj(engine, sp + "ffn_gate_shexp.weight", token_hidden,
                 scratch.gate_buf.data(), 1, D, shared_inter);
      dense_proj(engine, sp + "ffn_up_shexp.weight", token_hidden,
                 scratch.up_buf.data(), 1, D, shared_inter);
      if (swiglu_clamp_fused()) {
        // GLM5Next: ggml_swiglu_clamp semantics (fused min/clamp/silu/mul).
        swiglu_clamp_fused_inplace(scratch.gate_buf.data(),
                                   scratch.up_buf.data(), shared_inter,
                                   clamp_shexp);
      } else {
        silu_inplace(scratch.gate_buf.data(), shared_inter);
        elem_mul(scratch.gate_buf.data(), scratch.gate_buf.data(),
                 scratch.up_buf.data(), shared_inter);
        swiglu_clamp_inplace(scratch.gate_buf.data(), shared_inter,
                             clamp_shexp);
      }
      dense_proj(engine, sp + "ffn_down_shexp.weight", scratch.gate_buf.data(),
                 scratch.down_buf.data(), 1, shared_inter, D);
      // Previously completely uninstrumented -- the shared-expert path
      // has no equivalent of expert_ffn's check_finite calls, so a bug
      // here (e.g. a dense-tensor GPU dispatch through the same fused
      // K-quant kernels that turned out to have the Q3_K struct-layout
      // bug) would silently poison every token's output with no log
      // ever pointing at it.
      check_finite("shexp down_buf (pre-accumulate)", layer_idx, -1,
                   scratch.down_buf.data(), (size_t)D);
      // Qwen4Exp (Qwen3.8-Flash-Next): the shared expert has its OWN
      // per-token sigmoid gate (ffn_gate_inp_shexp, a [D] vector -> one
      // scalar per token -- qwen4exp.cpp's build_layer_ffn). Gated on
      // tensor PRESENCE (the established "optional small tensor" idiom),
      // not family, so GLM/DeepSeek's gateless shexp is untouched.
      float shexp_gate = 1.0f;
      if (engine.resident_layers.count(sp + "ffn_gate_inp_shexp.weight")) {
        float g = 0.0f;
        dense_proj(engine, sp + "ffn_gate_inp_shexp.weight", token_hidden, &g,
                   1, D, 1);
        shexp_gate = 1.0f / (1.0f + std::exp(-g));
      }
      for (int i = 0; i < D; ++i)
        token_output[i] += shexp_gate * scratch.down_buf[i];
      check_finite("token_output (post shared expert)", layer_idx, -1,
                   token_output, (size_t)D);
    }
  }

  // Three-tier caching: Cleanup active layer area after processing this layer
  // This reclaims temporary workspace used for current-layer experts
  if (ram_mgr.vram_cache && ram_mgr.vram_cache->active_layer_pool) {
    PERF_SCOPE(t_layer_cleanup_total);
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
    ram_mgr.vram_cache->cleanup_active_layer_area(
        layer_idx,
        [&](int l, int eid) { ram_mgr.finish_sync_promotion(engine, l, eid); });
  }

  // prefill->decode retention handoff (2026-08-31). Placed HERE, at the real
  // end of this layer's prefill pass, and NOT next to the Phase-1 window
  // block where it was first (wrongly) wired.
  //
  // The bug that placement had, worth recording because it made the whole
  // feature nearly inert: expert_freq is final after Phase 1, so stamping
  // right after Phase 1 *looks* correct -- but at that instant most of this
  // layer's experts are NOT RESIDENT YET. They arrive during Phase 2's
  // prefetch hints and Phase 3's own dispatch, i.e. AFTER that point, so
  // they were never stamped and kept prefill_demand=0 -- precisely the
  // newly-loaded experts that most need retention. Stamping after Phase 3
  // instead catches every expert this layer actually brought in.
  //
  // Note this is genuinely "who survived", not "who was needed": a slot
  // evicted mid-pass is gone and cannot be stamped. That is the intended
  // semantics -- eviction_score can only reorder what is still resident, so
  // the stamp describes exactly the population it can influence.
  if (S > 1) {
    ram_mgr.stamp_prefill_demand(engine, layer_idx, scratch.expert_freq);
  }
}

// ============================================================
// Dense (non-MoE) FFN for leading dense layers (l < first_dense) in an
// otherwise-MoE model — plain gate_proj -> SiLU, up_proj, elementwise
// multiply, down_proj on ffn_{gate,up,down}.weight. No router, no expert
// cache. Used by both GLM-5.2's leading dense layers and Hunyuan/Hy-V3
// layers before the MoE portion of the stack begins.
// ============================================================
void dense_ffn_gguf(StreamingEngine &engine, int layer_idx,
                    const float *hidden_states, int S, float *output,
                    LayerScratch &scratch) {
  auto &c = engine.model_file.architecture_config;
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
    const float *norm_w = reinterpret_cast<float *>(norm_it->second.host_ptr);
    for (int s = 0; s < S; ++s) {
      rmsnorm_row(scratch.normed.data() + (size_t)s * D,
                  hidden_states + (size_t)s * D, norm_w, D, c.eps);
    }
  } else {
    std::memcpy(scratch.normed.data(), hidden_states,
                (size_t)S * D * sizeof(float));
  }

  for (int s = 0; s < S; ++s) {
    const float *token_hidden = scratch.normed.data() + (size_t)s * D;
    float *token_output = output + (size_t)s * D;

    dense_proj(engine, p + "ffn_gate.weight", token_hidden,
               scratch.gate_buf.data(), 1, D, inter);
    dense_proj(engine, p + "ffn_up.weight", token_hidden, scratch.up_buf.data(),
               1, D, inter);
    silu_inplace(scratch.gate_buf.data(), inter);
    elem_mul(scratch.gate_buf.data(), scratch.gate_buf.data(),
             scratch.up_buf.data(), inter);
    dense_proj(engine, p + "ffn_down.weight", scratch.gate_buf.data(),
               token_output, 1, inter, D);
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
  for (auto &gs : g_layer_scratch.gpu_pool)
    gs.shrink();
}

// Execute GLM's optional NextN block using the same host-visible [S,D]
// contract as the target layer loop.  This intentionally lives in this
// translation unit: it can reuse the private LayerScratch/MoE implementation
// without exposing cache internals to the speculative driver.  The target
// trunk remains responsible for verification; this function only proposes
// from the extra DSA+MoE block.
bool glm_mtp_forward(StreamingEngine &engine, RAMSafetyManager &ram_mgr,
                     LockFreePrefetcher &async_io_queue, const int *token_ids,
                     const float *h_input, int S, int pos_base, float *h_nextn,
                     float *logits_last) {
  const auto &c = engine.model_file.architecture_config;
  const ModelFamily fam = engine.family_ops.family;
  // hc_mult > 1 is deliberately NOT a rejection here -- see
  // MtpSpeculator::supported()'s comment (mtp_speculative.cpp) for why the
  // NextN block's plain-residual math below is correct for GLM5NEXT's
  // multi-stream trunk too: the caller's hidden state is already collapsed
  // to single-stream by hc_collapse_output before it ever reaches this
  // function, and the reference NextN block itself has no HC mixer.
  if (!engine.mtp_enabled || c.n_layer_nextn != 1 || S <= 0 || c.hidden <= 0 ||
      c.vocab <= 0 ||
      (fam != ModelFamily::GLM5NEXT && fam != ModelFamily::GLM_DSA) ||
      !engine.family_ops.attention_fn ||
      engine.layers_experts.size() <= (size_t)c.n_layers) {
    return false;
  }
  // Keep the standalone NextN invocation under the same cache-mode contract
  // as layers_forward_gguf().  MTP uses this function for both a batched
  // prompt bridge (S>1) and one-row draft/correction steps (S==1).  Without
  // these updates, a one-row draft after prefill would inherit
  // RAMSafetyManager/GpuTopology's prefill mode and apply the wrong active-
  // layer/LRU policy.
  ram_mgr.in_prefill = (S > 1);
  if (ram_mgr.vram_cache)
    ram_mgr.vram_cache->in_prefill = (S > 1);
  if (S == 1 && pos_base == 0 && gdn_full_ggml())
    GgmlGdnProjRunner::instance().reset_full_device_states();
  if (S == 1)
    RAMSafetyManager::set_current_decode_token_idx((uint64_t)pos_base);
  const int layer_idx = c.n_layers;
  if (fam == ModelFamily::GLM5NEXT && glm5next_is_kda_layer(c, layer_idx)) {
    return false; // graph_initializer rejects this, keep the runtime guard too
  }
  if (pos_base < 0 || pos_base + S > engine.max_t) {
    LOG_STREAM_WARN() << "[MTP] position range [" << pos_base << ", "
                      << (pos_base + S) << ") exceeds max_t=" << engine.max_t
                      << " -- disabling this proposal batch\n";
    return false;
  }

  const int D = c.hidden;
  const std::string block = "blk." + std::to_string(layer_idx) + ".";
  auto has_host = [&](const std::string &name) {
    auto it = engine.resident_layers.find(name);
    return it != engine.resident_layers.end() && it->second.host_ptr;
  };
  // These names were checked before activating the feature.  Recheck the
  // resident map here because dense-pool backoff must never turn a required
  // MTP weight into a silent zero-filled projection.
  const char *required[] = {
      "nextn.eh_proj.weight", "nextn.enorm.weight",    "nextn.hnorm.weight",
      "attn_norm.weight",     "attn_q_a.weight",       "attn_q_a_norm.weight",
      "attn_q_b.weight",      "attn_kv_a_mqa.weight",  "attn_kv_a_norm.weight",
      "attn_k_b.weight",      "attn_v_b.weight",       "attn_output.weight",
      "ffn_norm.weight",      "ffn_gate_inp.weight",   "ffn_gate_shexp.weight",
      "ffn_up_shexp.weight",  "ffn_down_shexp.weight",
  };
  for (const char *suffix : required) {
    if (!has_host(block + suffix)) {
      LOG_STREAM_WARN() << "[MTP] required resident tensor missing: " << block
                        << suffix << " -- proposal batch skipped\n";
      return false;
    }
  }
  auto emb_it =
      engine.resident_layers.find(block + "nextn.embed_tokens.weight");
  if (emb_it == engine.resident_layers.end() || !emb_it->second.host_ptr)
    emb_it = engine.resident_layers.find("token_embd.weight");
  if (emb_it == engine.resident_layers.end() || !emb_it->second.host_ptr)
    return false;

  auto embed_row = [&](int tok, float *dst) {
    if (tok < 0 || tok >= c.vocab)
      return false;
    const int rb = row_bytes(emb_it->second.meta.type, D);
    if (rb <= 0)
      return false;
    return dequant_row(emb_it->second.meta.type,
                       emb_it->second.host_ptr + (int64_t)tok * rb, dst, D);
  };

  std::vector<float> token_emb((size_t)S * D, 0.0f);
  for (int s = 0; s < S; ++s) {
    if (!token_ids ||
        !embed_row(token_ids[s], token_emb.data() + (size_t)s * D))
      return false;
  }

  LayerScratch &scratch = g_layer_scratch;
  if (!scratch.resize(S, D, c.n_experts, c.moe_inter, c.topk, c.n_expert_shared,
                      c.dense_inter)) {
    LOG_STREAM_WARN() << "[MTP] scratch allocation denied by host-memory "
                         "budget -- skipping proposal batch\n";
    return false;
  }
  std::vector<float> e_norm((size_t)S * D);
  std::vector<float> h_norm((size_t)S * D);
  rmsnorm_proj(engine, block + "nextn.enorm.weight", token_emb.data(),
               e_norm.data(), S, D, c.eps);
  rmsnorm_proj(engine, block + "nextn.hnorm.weight", h_input, h_norm.data(), S,
               D, c.eps);

  std::vector<float> concat((size_t)S * 2 * D);
  for (int s = 0; s < S; ++s) {
    std::memcpy(concat.data() + (size_t)s * 2 * D,
                e_norm.data() + (size_t)s * D, (size_t)D * sizeof(float));
    std::memcpy(concat.data() + (size_t)s * 2 * D + D,
                h_norm.data() + (size_t)s * D, (size_t)D * sizeof(float));
  }

  std::vector<float> cur((size_t)S * D);
  dense_proj(engine, block + "nextn.eh_proj.weight", concat.data(), cur.data(),
             S, 2 * D, D);
  if (!std::all_of(cur.begin(), cur.end(),
                   [](float v) { return std::isfinite(v); }))
    return false;

  // DIAGNOSTIC (2026-09-09): per-call L2 norms through the NextN block's own
  // residual stream, added while chasing an abnormally low MTP acceptance
  // rate. Compares directly against the real trunk's own [TRACE-HIDDEN]
  // norms (layers_forward_gguf, same file) at trace_filter="MLA,TRACE-
  // HIDDEN" -- a norm wildly out of the real layers' own range would
  // localize the bug to this block specifically rather than the draft/
  // verify math in mtp_speculative.cpp. Cheap (one sqrt-sum per call, S is
  // always tiny here) and unconditional, no trace_filter gate, so it
  // survives into the next debugging session without a rebuild.
  // Matches layers_forward_gguf's own l2norm lambda EXACTLY (raw sqrt-sum,
  // not RMS) so these numbers are directly comparable against its
  // [TRACE-HIDDEN] output, not off by a sqrt(n) scale factor.
  auto mtp_l2norm = [](const float *p, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; ++i)
      s += (double)p[i] * (double)p[i];
    return std::sqrt(s);
  };
  LOG_STREAM_INFO() << "[MTP-DIAG] layer=" << layer_idx
                    << " pos_base=" << pos_base << " S=" << S
                    << " |cur(post eh_proj)|="
                    << mtp_l2norm(cur.data(), cur.size()) << "\n";

  rmsnorm_proj(engine, block + "attn_norm.weight", cur.data(),
               scratch.normed.data(), S, D, c.eps);
  engine.family_ops.attention_fn(engine, layer_idx, scratch.normed.data(), S,
                                 pos_base, scratch.attn_out.data());
  LOG_STREAM_INFO() << "[MTP-DIAG] layer=" << layer_idx << " |attn_out|="
                    << mtp_l2norm(scratch.attn_out.data(), (size_t)S * D)
                    << "\n";
  for (size_t i = 0; i < cur.size(); ++i)
    cur[i] += scratch.attn_out[i];

  rmsnorm_proj(engine, block + "ffn_norm.weight", cur.data(),
               scratch.normed.data(), S, D, c.eps);
  moe_forward_gguf(engine, ram_mgr, async_io_queue, layer_idx,
                   scratch.normed.data(), S, scratch.moe_out.data(), scratch,
                   token_ids);
  LOG_STREAM_INFO() << "[MTP-DIAG] layer=" << layer_idx << " |moe_out|="
                    << mtp_l2norm(scratch.moe_out.data(), (size_t)S * D)
                    << "\n";
  // MTP is intentionally host-residual in this first bridge.  Consuming a
  // future GGML device result here would make the host residual stale and
  // would also couple GLM MTP to Qwen's unrelated device-HC path.
  if (scratch.moe_device_output_ready) {
    scratch.moe_device_output = nullptr;
    scratch.moe_device_output_ready = false;
    LOG_STREAM_WARN() << "[MTP] NextN MoE returned a device-only result; "
                         "proposal batch skipped safely\n";
    return false;
  }
  for (size_t i = 0; i < cur.size(); ++i)
    cur[i] += scratch.moe_out[i];

  const std::string head_norm =
      has_host(block + "nextn.shared_head_norm.weight")
          ? block + "nextn.shared_head_norm.weight"
          : "output_norm.weight";
  if (!has_host(head_norm))
    return false;
  rmsnorm_proj(engine, head_norm, cur.data(), h_nextn, S, D, c.eps);
  if (!std::all_of(h_nextn, h_nextn + (size_t)S * D,
                   [](float v) { return std::isfinite(v); }))
    return false;
  if (logits_last) {
    const std::string head = block + "nextn.shared_head_head.weight";
    const std::string output_head = has_host(head) ? head : "output.weight";
    if (!has_host(output_head))
      return false;
    dense_proj(engine, output_head, h_nextn + (size_t)(S - 1) * D, logits_last,
               1, D, c.vocab);
    if (!std::all_of(logits_last, logits_last + c.vocab,
                     [](float v) { return std::isfinite(v); }))
      return false;
  }
  return true;
}

// Declared non-static in their own .cpp files (attention_layer.cpp,
// hyper_connections.cpp respectively) specifically so this function can
// reuse them; see each declaration's own comment.
void qwen4exp_full_attention_gguf(StreamingEngine &engine, int layer_idx,
                                  const float *hidden_states, int S,
                                  int pos_base, float *out_context);
void qwen4exp_hc_mix_impl(StreamingEngine &engine, const std::string &norm_name,
                          const std::string &down_name,
                          const std::string &up_name,
                          const std::string &inject_name, int S, int D,
                          float *out_hidden, float *out_inject, int trace_layer,
                          const float *in_streams, bool force_original_quant);

// Qwen4Exp shared MTP checkpoint's own extra full-attention+MoE block.
// Verified against /home/ray/Projects/unsloth/llama.cpp's qwen4exp.cpp
// (branch mtp/qwen4exp-nextn, graph_mtp) and a real tensor dump of
// mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf. Structurally: (1) eh_proj
// bridges the token embedding and the incoming wide HC state into a fresh
// per-stream residual res_hc, (2) hc_attn/hc_ffn sub-blocks run exactly
// like a normal Qwen4Exp trunk full-attention layer's own HC pre-mix ->
// sub-block -> HC post-combine, but against res_hc (a LOCAL buffer, never
// engine.hc_stream_buf), (3) res_hc is exported as h_nextn BEFORE the
// final hc_head collapse, matching the reference's own "next draft step
// re-enters here" comment, (4) the collapsed [S,D] result is projected to
// logits via the shared LM head (nextn.shared_head_head, or the base
// model's own output.weight).
//
// The routed-expert MoE step deliberately does NOT go through
// moe_forward_gguf/engine.layers_experts -- this block's own packed
// expert tensors are resident-loaded whole from the SEPARATE
// engine.mtp_model_file (graph_initializer.cpp's qwen4exp_nextn_manifest)
// rather than routed through the disk-streaming/RAM-VRAM LRU cache that
// every other layer's routed experts use (that machinery is hardwired to
// engine.model_file throughout ram_manager.hpp/async_prefetcher.hpp). Per-
// expert slicing uses the exact same size_bytes/n_experts stride
// convention as StreamingEngine::packed_slab_geometry (engine.hpp).
bool qwen4exp_mtp_forward(StreamingEngine &engine, RAMSafetyManager &ram_mgr,
                          LockFreePrefetcher &async_io_queue,
                          const int *token_ids, const float *h_input, int S,
                          int pos_base, float *h_nextn, float *logits_last) {
  auto &c = engine.model_file.architecture_config;
  const ModelFamily fam = engine.family_ops.family;
  if (!engine.mtp_enabled || !engine.has_mtp_model || c.n_layer_nextn != 1 ||
      S <= 0 || c.hidden <= 0 || c.vocab <= 0 || c.hc_mult <= 1 ||
      fam != ModelFamily::Qwen4Exp) {
    return false;
  }
  const int layer_idx = c.n_layers;
  if (pos_base < 0 || pos_base + S > engine.max_t) {
    LOG_STREAM_WARN() << "[MTP] position range [" << pos_base << ", "
                      << (pos_base + S) << ") exceeds max_t=" << engine.max_t
                      << " -- disabling this proposal batch\n";
    return false;
  }
  if ((int)engine.Kc.size() <= layer_idx ||
      (int)engine.Vc.size() <= layer_idx || !engine.Kc[layer_idx] ||
      !engine.Vc[layer_idx]) {
    LOG_STREAM_WARN() << "[MTP] Qwen4Exp NextN KV cache (layer " << layer_idx
                      << ") not allocated -- proposal batch skipped\n";
    return false;
  }

  const int D = c.hidden;
  const int HC = c.hc_mult;
  const int hc_dim = HC * D;
  const int n_experts = c.n_experts;
  const int topk = c.topk;
  const int moe_inter = c.moe_inter;
  const std::string p = "blk." + std::to_string(layer_idx) + ".";

  auto has_host = [&](const std::string &name) {
    auto it = engine.resident_layers.find(name);
    return it != engine.resident_layers.end() && it->second.host_ptr;
  };
  // Recheck resident tensors here (defense-in-depth, same convention as
  // glm_mtp_forward): graph_initializer.cpp validated these before
  // activating MTP, but a dense-pool backoff must never turn a required
  // MTP weight into a silent zero-filled projection.
  const char *required[] = {
      "attn_q.weight",
      "attn_k.weight",
      "attn_v.weight",
      "attn_output.weight",
      "attn_q_norm.weight",
      "attn_k_norm.weight",
      "hc_attn_norm.weight",
      "hc_attn_down.weight",
      "hc_attn_up.weight",
      "hc_attn_inject.weight",
      "hc_ffn_norm.weight",
      "hc_ffn_down.weight",
      "hc_ffn_up.weight",
      "hc_ffn_inject.weight",
      "ffn_gate_inp.weight",
      "ffn_gate_exps.weight",
      "ffn_up_exps.weight",
      "ffn_down_exps.weight",
      "nextn.eh_proj.weight",
      "nextn.enorm.weight",
      "nextn.hnorm.weight",
      "nextn.hc_head_norm.weight",
      "nextn.hc_head_down.weight",
      "nextn.hc_head_up.weight",
  };
  for (const char *suffix : required) {
    if (!has_host(p + suffix)) {
      LOG_STREAM_WARN() << "[MTP] required resident tensor missing: " << p
                        << suffix << " -- proposal batch skipped\n";
      return false;
    }
  }
  auto emb_it = engine.resident_layers.find(p + "nextn.embed_tokens.weight");
  if (emb_it == engine.resident_layers.end() || !emb_it->second.host_ptr)
    emb_it = engine.resident_layers.find("token_embd.weight");
  if (emb_it == engine.resident_layers.end() || !emb_it->second.host_ptr)
    return false;

  // Never let this block's attention call pick up a stale device-resident
  // activation left behind by the target trunk's own last GPU-HC layer
  // (engine.qwen4exp_device_hc_input/_ffn_input) -- this function never
  // calls qwen4exp_hc_pre_mix (the trunk's own hook, which would normally
  // set/clear these), so a leftover non-null value here would make
  // qwen4exp_full_attention_gguf silently consume the WRONG activation
  // (the trunk's, not this block's cur_attn buffer below). Force the safe
  // host dense_proj fallback path unconditionally.
  engine.qwen4exp_device_hc_input = nullptr;
  engine.qwen4exp_device_hc_ffn_input = nullptr;

  auto embed_row = [&](int tok, float *dst) {
    if (tok < 0 || tok >= c.vocab)
      return false;
    const int rb = row_bytes(emb_it->second.meta.type, D);
    if (rb <= 0)
      return false;
    return dequant_row(emb_it->second.meta.type,
                       emb_it->second.host_ptr + (int64_t)tok * rb, dst, D);
  };
  std::vector<float> token_emb((size_t)S * D, 0.0f);
  for (int s = 0; s < S; ++s) {
    if (!token_ids ||
        !embed_row(token_ids[s], token_emb.data() + (size_t)s * D))
      return false;
  }

  // ---- eh_proj bridge: e_norm (token embedding) concat h_norm (incoming
  // wide state, grouped RMS-norm'd per stream) -> res_hc ----
  std::vector<float> e_norm((size_t)S * D);
  rmsnorm_proj(engine, p + "nextn.enorm.weight", token_emb.data(),
               e_norm.data(), S, D, c.eps);

  std::vector<float> h_norm((size_t)S * hc_dim);
  {
    const float *hnorm_w = nullptr;
    auto hn_it = engine.resident_layers.find(p + "nextn.hnorm.weight");
    hnorm_w = reinterpret_cast<float *>(hn_it->second.host_ptr);
    // Grouped RMS norm: each (token, stream) row normalized independently
    // over its own D elements, NO learned gain in the norm itself (matches
    // ggml_rms_norm's definition) -- the hc_dim-wide learned gain
    // (nextn.hnorm, one scalar per (stream,dim) pair, NOT broadcast across
    // streams -- unlike e_norm below) is applied after, over the flattened
    // [hc_dim] layout. Identical math to qwen4exp_hc_mix_impl's own steps
    // 1-2, inlined here since that function's own gate/inject math past
    // step 2 doesn't apply to this bridge step.
    for (int s = 0; s < S; ++s) {
      const float *in_s = h_input + (size_t)s * hc_dim;
      float *out_s = h_norm.data() + (size_t)s * hc_dim;
      for (int h = 0; h < HC; ++h) {
        const float *row = in_s + (size_t)h * D;
        double ss = 0.0;
        for (int d = 0; d < D; ++d)
          ss += (double)row[d] * row[d];
        float inv = 1.0f / std::sqrt((float)(ss / (double)D) + c.eps);
        float *out_row = out_s + (size_t)h * D;
        for (int d = 0; d < D; ++d)
          out_row[d] = row[d] * inv;
      }
      if (hnorm_w)
        for (int j = 0; j < hc_dim; ++j)
          out_s[j] *= hnorm_w[j];
    }
  }

  std::vector<float> concat((size_t)S * HC * 2 * D);
  for (int s = 0; s < S; ++s) {
    const float *e = e_norm.data() + (size_t)s * D;
    for (int h = 0; h < HC; ++h) {
      float *row = concat.data() + ((size_t)s * HC + h) * 2 * D;
      std::memcpy(row, e, (size_t)D * sizeof(float));
      std::memcpy(row + D, h_norm.data() + ((size_t)s * HC + h) * D,
                  (size_t)D * sizeof(float));
    }
  }

  std::vector<float> res_hc((size_t)S * hc_dim);
  dense_proj(engine, p + "nextn.eh_proj.weight", concat.data(), res_hc.data(),
             S * HC, 2 * D, D);
  if (!std::all_of(res_hc.begin(), res_hc.end(),
                   [](float v) { return std::isfinite(v); }))
    return false;

  // ---- hc_attn sub-block ----
  std::vector<float> cur_attn((size_t)S * D), inject_attn((size_t)S * HC);
  qwen4exp_hc_mix_impl(engine, p + "hc_attn_norm.weight",
                       p + "hc_attn_down.weight", p + "hc_attn_up.weight",
                       p + "hc_attn_inject.weight", S, D, cur_attn.data(),
                       inject_attn.data(), layer_idx, res_hc.data(), false);
  std::vector<float> attn_out((size_t)S * D);
  qwen4exp_full_attention_gguf(engine, layer_idx, cur_attn.data(), S, pos_base,
                               attn_out.data());
  {
    const float inv_hc = 1.0f / (float)HC;
    for (int s = 0; s < S; ++s) {
      const float *inj = inject_attn.data() + (size_t)s * HC;
      const float *sub_out = attn_out.data() + (size_t)s * D;
      float *streams = res_hc.data() + (size_t)s * hc_dim;
      for (int h = 0; h < HC; ++h) {
        float w = 2.0f / (1.0f + std::exp(-inj[h] * inv_hc));
        float *stream_h = streams + (size_t)h * D;
        for (int d = 0; d < D; ++d)
          stream_h[d] += sub_out[d] * w;
      }
    }
  }

  // ---- hc_ffn sub-block: routed MoE (v1: no active-layer/LRU cache, the
  // shared checkpoint's whole expert table is already resident) + shared
  // expert ----
  std::vector<float> cur_ffn((size_t)S * D), inject_ffn((size_t)S * HC);
  qwen4exp_hc_mix_impl(engine, p + "hc_ffn_norm.weight",
                       p + "hc_ffn_down.weight", p + "hc_ffn_up.weight",
                       p + "hc_ffn_inject.weight", S, D, cur_ffn.data(),
                       inject_ffn.data(), layer_idx, res_hc.data(), false);
  std::vector<float> moe_out((size_t)S * D, 0.0f);
  {
    auto &gate_t = engine.resident_layers[p + "ffn_gate_exps.weight"];
    auto &up_t = engine.resident_layers[p + "ffn_up_exps.weight"];
    auto &down_t = engine.resident_layers[p + "ffn_down_exps.weight"];
    if (!gate_t.host_ptr || !up_t.host_ptr || !down_t.host_ptr) {
      LOG_STREAM_WARN() << "[MTP] Qwen4Exp NextN packed expert tensors not "
                           "resident -- proposal batch skipped\n";
      return false;
    }
    const size_t gate_stride = gate_t.meta.size_bytes / (size_t)n_experts;
    const size_t up_stride = up_t.meta.size_bytes / (size_t)n_experts;
    const size_t down_stride = down_t.meta.size_bytes / (size_t)n_experts;

    std::vector<float> router_logits((size_t)S * n_experts);
    dense_proj(engine, p + "ffn_gate_inp.weight", cur_ffn.data(),
               router_logits.data(), S, D, n_experts);
    for (int s = 0; s < S; ++s)
      apply_gating_activation(c.gating_func,
                              router_logits.data() + (size_t)s * n_experts,
                              n_experts);
    if (!std::all_of(router_logits.begin(), router_logits.end(),
                     [](float v) { return std::isfinite(v); }))
      return false;

    std::vector<float> gate_buf((size_t)moe_inter), up_buf((size_t)moe_inter);
    std::vector<float> expert_out((size_t)D);
    std::vector<std::pair<float, int>> scores((size_t)n_experts);
    const float clamp_exp = (layer_idx < (int)c.swiglu_clamp_exp.size())
                                ? c.swiglu_clamp_exp[layer_idx]
                                : 0.0f;
    for (int s = 0; s < S; ++s) {
      float *token_probs = router_logits.data() + (size_t)s * n_experts;
      for (int e = 0; e < n_experts; ++e)
        scores[e] = {token_probs[e], e};
      std::partial_sort(
          scores.begin(), scores.begin() + topk, scores.end(),
          [](const auto &a, const auto &b) { return a.first > b.first; });
      float sum_w = 0.0f;
      for (int k = 0; k < topk; ++k)
        sum_w += scores[k].first;
      float inv_sum = c.expert_weights_norm
                          ? 1.0f / std::max(sum_w, 6.103515625e-5f)
                          : 1.0f;
      float *out_row = moe_out.data() + (size_t)s * D;
      for (int k = 0; k < topk; ++k) {
        const int e = scores[k].second;
        float w = scores[k].first * inv_sum;
        if (c.expert_weights_scale != 0.0f)
          w *= c.expert_weights_scale;
        GGUFTensor gm = gate_t.meta, um = up_t.meta, dm = down_t.meta;
        const uint8_t *gd = gate_t.host_ptr + (size_t)e * gate_stride;
        const uint8_t *ud = up_t.host_ptr + (size_t)e * up_stride;
        const uint8_t *dd = down_t.host_ptr + (size_t)e * down_stride;
        expert_ffn_cpu(cur_ffn.data() + (size_t)s * D, expert_out.data(), gd,
                       gm, ud, um, dd, dm, gate_buf.data(), up_buf.data(), 1, D,
                       moe_inter, clamp_exp, layer_idx, e);
        for (int d = 0; d < D; ++d)
          out_row[d] += expert_out[d] * w;
      }
    }

    // Shared expert (mirrors moe_forward_gguf's own CPU shexp block --
    // same formula, same ffn_gate_inp_shexp per-token sigmoid gate).
    if (c.n_expert_shared > 0 && has_host(p + "ffn_gate_shexp.weight")) {
      const int shared_inter = moe_inter * c.n_expert_shared;
      const float clamp_shexp = (layer_idx < (int)c.swiglu_clamp_shexp.size())
                                    ? c.swiglu_clamp_shexp[layer_idx]
                                    : 0.0f;
      std::vector<float> sg((size_t)shared_inter), su((size_t)shared_inter);
      for (int s = 0; s < S; ++s) {
        const float *tok = cur_ffn.data() + (size_t)s * D;
        dense_proj(engine, p + "ffn_gate_shexp.weight", tok, sg.data(), 1, D,
                   shared_inter);
        dense_proj(engine, p + "ffn_up_shexp.weight", tok, su.data(), 1, D,
                   shared_inter);
        silu_inplace(sg.data(), shared_inter);
        elem_mul(sg.data(), sg.data(), su.data(), shared_inter);
        swiglu_clamp_inplace(sg.data(), shared_inter, clamp_shexp);
        std::vector<float> shexp_down((size_t)D);
        dense_proj(engine, p + "ffn_down_shexp.weight", sg.data(),
                   shexp_down.data(), 1, shared_inter, D);
        float shexp_gate = 1.0f;
        if (has_host(p + "ffn_gate_inp_shexp.weight")) {
          float g = 0.0f;
          dense_proj(engine, p + "ffn_gate_inp_shexp.weight", tok, &g, 1, D, 1);
          shexp_gate = 1.0f / (1.0f + std::exp(-g));
        }
        float *out_row = moe_out.data() + (size_t)s * D;
        for (int d = 0; d < D; ++d)
          out_row[d] += shexp_down[d] * shexp_gate;
      }
    }
  }
  if (!std::all_of(moe_out.begin(), moe_out.end(),
                   [](float v) { return std::isfinite(v); }))
    return false;
  {
    const float inv_hc = 1.0f / (float)HC;
    for (int s = 0; s < S; ++s) {
      const float *inj = inject_ffn.data() + (size_t)s * HC;
      const float *sub_out = moe_out.data() + (size_t)s * D;
      float *streams = res_hc.data() + (size_t)s * hc_dim;
      for (int h = 0; h < HC; ++h) {
        float w = 2.0f / (1.0f + std::exp(-inj[h] * inv_hc));
        float *stream_h = streams + (size_t)h * D;
        for (int d = 0; d < D; ++d)
          stream_h[d] += sub_out[d] * w;
      }
    }
  }

  // Export the WIDE state before collapse -- "the next draft step
  // re-enters here" (reference's own comment on res_hc/t_h_nextn).
  std::memcpy(h_nextn, res_hc.data(), (size_t)S * hc_dim * sizeof(float));

  std::vector<float> cur_head((size_t)S * D);
  qwen4exp_hc_mix_impl(engine, p + "nextn.hc_head_norm.weight",
                       p + "nextn.hc_head_down.weight",
                       p + "nextn.hc_head_up.weight", "", S, D, cur_head.data(),
                       nullptr, layer_idx, res_hc.data(), false);
  if (!std::all_of(cur_head.begin(), cur_head.end(),
                   [](float v) { return std::isfinite(v); }))
    return false;

  if (logits_last) {
    const std::string head = p + "nextn.shared_head_head.weight";
    const std::string output_head = has_host(head) ? head : "output.weight";
    if (!has_host(output_head))
      return false;
    dense_proj(engine, output_head, cur_head.data() + (size_t)(S - 1) * D,
               logits_last, 1, D, c.vocab);
    if (!std::all_of(logits_last, logits_last + c.vocab,
                     [](float v) { return std::isfinite(v); }))
      return false;
  }
  return true;
}

// ============================================================
// Main layer-by-layer forward pass with pre-allocated scratch,
// speculative pilot prefetch, and performance instrumentation.
// ============================================================
void layers_forward_gguf(StreamingEngine &engine, RAMSafetyManager &ram_mgr,
                         LockFreePrefetcher &async_io_queue,
                         float *hidden_states, int S, int pos_base,
                         const int *token_ids) {
  auto &c = engine.model_file.architecture_config;
  int D = c.hidden;

  // S > 1 = prefill (whole prompt in one pass), S == 1 = decode (one
  // token at a time) -- see RAMSafetyManager::in_prefill's own comment
  // for why the prefetch miss path needs this split.
  ram_mgr.in_prefill = (S > 1);
  // Mirror on GpuTopology -- see GpuTopology::in_prefill's own comment
  // for why active-layer promotion gates on this (2026-08-12).
  if (ram_mgr.vram_cache)
    ram_mgr.vram_cache->in_prefill = (S > 1);
  // EngineConfig::ram_grace_period_tokens (2026-09-04): only advance the
  // grace-period clock on a real single-token DECODE step, not a prefill
  // batch -- see RAMSafetyManager::current_decode_token_idx_ref's own
  // comment for why leaving it untouched during prefill is deliberate.
  if (S == 1)
    RAMSafetyManager::set_current_decode_token_idx((uint64_t)pos_base);

  // Measurement only (expert_fetch_tracker.hpp) -- one call to this
  // function IS one token's worth of processing (prefill's single S>1
  // call, or one decode step's S=1 call), so this is the exact right
  // place to advance the token-step counter record_hit_if_recently_
  // fetched uses to distinguish a same-token race from a much later,
  // unrelated token rediscovering an earlier token's stale fetch record.
  advance_token_step();

  // The Bee-style full graph paths submit the complete trunk in one GGML
  // scheduler call.  They return true only after producing final [S,D]
  // output; otherwise the established layer loop remains the fallback.
  engine.qwen4exp_full_graph_output_ready = false;
  engine.glm5next_full_graph_output_ready = false;
  engine.deepseek_v4_full_graph_output_ready = false;
  if (S > 1 && pos_base == 0 && qwen4exp_full_graph_enabled() &&
      qwen4exp_full_graph_run(engine, hidden_states, S, pos_base, token_ids)) {
    return;
  }
  if (S > 1 && pos_base == 0 && glm5next_full_graph_enabled() &&
      !engine.mtp_enabled &&
      glm5next_full_graph_run(engine, hidden_states, S, pos_base, token_ids)) {
    return;
  }
  if (S > 1 && pos_base == 0 && deepseek_v4_full_graph_enabled() &&
      deepseek_v4_full_graph_run(engine, hidden_states, S, pos_base,
                                 token_ids)) {
    return;
  }

  LayerScratch &scratch = g_layer_scratch;
  if (!scratch.resize(S, D, c.n_experts, c.moe_inter, c.topk, c.n_expert_shared,
                      c.dense_inter)) {
    LOG_STREAM_ERROR() << "[MEM-GUARD] generation scratch allocation denied; "
                          "aborting this forward pass safely\n";
    return;
  }

  // Diagnostic: L2 norm of the LAST TOKEN's hidden state
  // the last few layers while starting within ~2x at layer 0, meaning the
  // excess accumulates gradually across many layers rather than in one bad
  // layer -- full resolution is needed to see exactly where it accelerates.
  auto l2norm = [](const float *p, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i)
      s += (double)p[i] * p[i];
    return std::sqrt(s);
  };
  int n = S * D; // hoisted above the loop -- was declared after its first use
                 // at "[L0] in" below
  bool trace = trace_gate("TRACE-HIDDEN"); // gate BEFORE paying for l2norm, not
                                           // just before printing
  for (int l = 0; l < c.n_layers; ++l) {
    if (ram_mgr.vram_cache)
      ram_mgr.vram_cache->prefill_current_layer = l;
    const float *last_tok =
        hidden_states +
        (size_t)(S - 1) * D; // hidden_states is mutated in place below, not
                             // reallocated -- always valid
    if (trace)
      LOG_STREAM_VERBOSE() << "[TRACE l=" << l
                           << "] in      |hidden|=" << l2norm(last_tok, D)
                           << "\n";
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
    auto &fo = engine.family_ops;
    // Optional per-layer input hook (FamilyOps::layer_input_fn) -- Qwen4Exp's
    // PLE n-gram hash embedding modifies the HC stream buffer BEFORE the
    // attention hc_pre collapse. nullptr for every other family.
    if (fo.layer_input_fn) {
      PERF_SCOPE(t_layer_input_hook);
      qwen4exp_hc_prepare_host_layer_input(engine, l, S);
      fo.layer_input_fn(engine, l, S, pos_base, token_ids);
      qwen4exp_hc_commit_host_layer_input(engine, l, S);
    }
    {
      PERF_SCOPE(t_hc_hooks);
      fo.hc_pre_fn(engine, l, /*sub_block=*/0, hidden_states, S, D);
    }
    fo.attention_fn(engine, l, hidden_states, S, pos_base,
                    scratch.attn_out.data());
    bool attn_device_consumed = false;
    if (engine.qwen4exp_device_attn_output_ready) {
      if (engine.qwen4exp_device_attn_output_raw) {
        attn_device_consumed = qwen4exp_hc_post_combine_device_raw(
            engine, l, /*sub_block=*/0, S, D,
            engine.qwen4exp_device_attn_output);
        if (!attn_device_consumed && engine.gpu &&
            !engine.gpu->qwen4exp_attn_gpu_copy_device_output(
                engine.qwen4exp_device_attn_output, scratch.attn_out.data(),
                (size_t)S * D * sizeof(float))) {
          LOG_ERROR("[QWEN4EXP-ATTN] failed to recover raw device attention "
                    "output at layer=%d\n",
                    l);
        }
      } else {
        attn_device_consumed =
            qwen4exp_hc_post_combine_device(engine, l, /*sub_block=*/0, S, D,
                                            engine.qwen4exp_device_attn_output);
        if (!attn_device_consumed && !qwen4exp_copy_gdn_device_output_to_host(
                                         engine.qwen4exp_device_attn_output,
                                         scratch.attn_out.data(), S)) {
          LOG_ERROR("[GGML-GDN-FULL] failed to recover device attention output "
                    "at layer=%d\n",
                    l);
        }
      }
      engine.qwen4exp_device_attn_output = nullptr;
      engine.qwen4exp_device_attn_output_ready = false;
      engine.qwen4exp_device_attn_output_raw = false;
    }
    {
      PERF_SCOPE(t_hc_hooks);
      if (attn_device_consumed) {
        // The device post-combine already updated the authoritative HC state.
      } else {
        fo.hc_post_fn(engine, l, /*sub_block=*/0, hidden_states,
                      scratch.attn_out.data(), S, D);
      }
    }

    // Per-layer, per-stage checkpoint: pinpoints the FIRST layer/stage
    // where hidden_states goes non-finite, instead of only learning
    // "somewhere in this forward pass" from the aggregate
    // after-prefill/after-decode-step checks in generation_loop.hpp.
    // check_finite() is a no-op (silent) unless it actually finds
    // NaN/Inf, so this is cheap on the (expected) all-finite path.
    check_finite("hidden_states (post attention residual)", l, -1,
                 hidden_states, (size_t)n);
    if (trace)
      LOG_STREAM_VERBOSE() << "[TRACE l=" << l << "] post-attn |attn_out|="
                           << l2norm(scratch.attn_out.data() +
                                         (size_t)(S - 1) * D,
                                     D)
                           << " |hidden|=" << l2norm(last_tok, D) << "\n";

    // A device pilot queued after layer l-1 is consumed here, after layer
    // l's attention had a chance to run and immediately before l's MoE
    // bookkeeping. This is the only point where the host needs the compact
    // IDs to make a prefetch decision; delaying the readback keeps the
    // layer-boundary CPU stall out of the critical path.
    if (S == 1 && scratch.qwen_pilot_gpu_pending &&
        scratch.qwen_pilot_gpu_pending_layer == l) {
      auto &pilot_router = GgmlRouterRunner::instance();
      const bool route_ok = pilot_router.copy_route_to_host(
          l, scratch.qwen_pilot_gpu_ids.data(),
          scratch.qwen_pilot_gpu_weights.data());
      if (route_ok) {
        const int qwen_k = std::min({engine.qwen4exp_router_pilot_k, c.topk,
                                     LayerScratch::kQwenPilotMax});
        for (int k = 0; k < qwen_k; ++k)
          scratch.qwen_pilot_pred_ids[k] = scratch.qwen_pilot_gpu_ids[k];
        qwen4exp_process_pilot_predictions(engine, ram_mgr, async_io_queue,
                                           scratch, l, qwen_k);
        perf().n_qwen_pilot_gpu_predictions.fetch_add(
            1, std::memory_order_relaxed);
      } else {
        perf().n_qwen_pilot_sync_fail.fetch_add(1, std::memory_order_relaxed);
      }
      scratch.qwen_pilot_gpu_pending = false;
      scratch.qwen_pilot_gpu_pending_layer = -1;
    }

    // --- ATTN-PILOT measurement (2026-08-25, MEASUREMENT ONLY -- not
    // wired to async_io_queue.push_hint/publish_pilot_confirmed_
    // resident, same discipline DEPTH-2 originally shipped under):
    // predicts layer l+1's routing from THIS point -- right after l's
    // own attention residual, BEFORE l's own FFN runs -- rather than
    // the existing pilot block below, which waits for l's FFN to fully
    // finish first. One new signal answers two different questions,
    // since "predict layer T from layer T-1's post-attention state" is
    // a single well-defined measurement regardless of which existing
    // baseline it's read against:
    //   - vs DEPTH-1 below (predicts l+1 from l's COMPLETE, post-FFN
    //     state): firing this early loses l's own FFN contribution --
    //     is the extra lead time (this doesn't have to wait for l's
    //     FFN, "routinely the single largest per-layer compute cost in
    //     this model" per DEPTH-1's own comment below) worth what it
    //     costs in accuracy?
    //   - vs DEPTH-2 (predicts l+2 by naively reusing l's own stale
    //     pre-l+1-attention vector): when THIS SAME mechanism fires
    //     one iteration later (at l+1's own post-attention point,
    //     predicting l+2), it uses l+1's REAL attention output --
    //     strictly more current than DEPTH-2's naive reuse, at the
    //     cost of firing later (less lead time than DEPTH-2 gets,
    //     since it has to wait for l+1's real attention to actually
    //     run first).
    // Single-scalar survival (pilot_attn_pred_layer), not a 2-slot
    // ring like DEPTH-2's Pilot2Pending -- only one intervening layer
    // boundary between prediction (fired further below, AFTER l's own
    // FFN completes -- see the real ordering bug that requires this,
    // explained there) and consumption (top of l+1's own moe_forward_
    // gguf/dense_ffn_gguf call), the same single-iteration gap DEPTH-1's
    // own pilot_pred_layer already safely uses.
    //
    // The actual dense_proj/gating prediction is DEFERRED until after
    // l's own FFN block below -- not fired here, despite this being
    // the whole point (predict from PRE-FFN state) -- because l's own
    // moe_forward_gguf call (which happens next, in the FFN block)
    // is what CONSUMES whatever prediction l-1 made for layer l. Firing
    // the l+1 prediction here, before that consumption runs, would
    // overwrite pilot_attn_pred_layer before it's ever read -- a real
    // bug caught by this mechanism's own first real-hardware run: only
    // ~41 checks total (one per TOKEN, not one per layer -- only the
    // very last layer transition survived, since there's no l+1 firing
    // after the last layer to clobber it first). Only the CHEAP part
    // (snapshotting the pre-FFN input this prediction will need) can
    // safely happen here; hidden_states itself will have moved on to
    // POST-FFN by the time the prediction actually runs.
    std::memcpy(scratch.pilot_attn_snapshot.data(), last_tok,
                (size_t)D * sizeof(float));

    // --- FFN: leading dense layers (l < first_dense) use a plain dense
    // FFN — no router, no expert cache. Applies to both GLM-5.2's leading
    // dense layers and, since Hunyuan/Hy-V3 share the same first_dense
    // convention, to any dense layers there too. dense_ffn_gguf/
    // moe_forward_gguf are unchanged by the FamilyOps refactor -- they
    // still read/write plain [S,D] hidden_states/scratch.moe_out
    // regardless of family; only the hc_pre_fn/hc_post_fn bracket around
    // them (sub_block=1, matching attention's sub_block=0 above) knows
    // about hyper-connections. ---
    {
      PERF_SCOPE(t_hc_hooks);
      fo.hc_pre_fn(engine, l, /*sub_block=*/1, hidden_states, S, D);
    }
    {
      PERF_SCOPE(t_moe_expert);
      if (l < c.first_dense) {
        dense_ffn_gguf(engine, l, hidden_states, S, scratch.moe_out.data(),
                       scratch);
      } else {
        moe_forward_gguf(engine, ram_mgr, async_io_queue, l, hidden_states, S,
                         scratch.moe_out.data(), scratch, token_ids);
      }
    }
    bool moe_device_consumed = false;
  if (scratch.moe_device_output_ready) {
      // The actual residual update is device-to-device.  Do not read the
      // routed/shared result back to the host on the normal path: the old
      // unconditional copy paid one synchronizing D2H per device MoE layer
      // even though qwen4exp_hc_post_combine_device() consumed the same
      // device tensor immediately afterwards.  Keep the readback only for
      // explicit hidden-state tracing, or as the recovery path when the
      // device HC post-combine cannot accept the producer tensor.
      bool copied_device_output = false;
      if (trace) {
        if (scratch.moe_device_runner)
          copied_device_output = scratch.moe_device_runner
                                     ->copy_device_output_to_host(
                                         scratch.moe_device_output,
                                         scratch.moe_out.data());
        else if (scratch.moe_shared_device_output)
          copied_device_output = GgmlQwenSharedRunner::instance()
                                     .copy_output_to_host(
                                         scratch.moe_device_output,
                                         scratch.moe_out.data());
        if (!copied_device_output) {
          LOG_ERROR("[GGML-MOE] Qwen4Exp device output trace copy failed at "
                    "layer=%d\n",
                    l);
        }
      }
      moe_device_consumed = qwen4exp_hc_post_combine_device(
          engine, l, /*sub_block=*/1, S, D, scratch.moe_device_output);
      if (!moe_device_consumed) {
        if (!copied_device_output && scratch.moe_device_runner)
          copied_device_output =
              scratch.moe_device_runner->copy_device_output_to_host(
                  scratch.moe_device_output, scratch.moe_out.data());
        else if (!copied_device_output && scratch.moe_shared_device_output)
          copied_device_output = GgmlQwenSharedRunner::instance()
                                     .copy_output_to_host(
                                         scratch.moe_device_output,
                                         scratch.moe_out.data());
        if (!copied_device_output)
          LOG_ERROR("[GGML-MOE] Qwen4Exp device output fallback copy failed at "
                    "layer=%d\n",
                    l);
      }
      scratch.moe_device_output = nullptr;
      scratch.moe_device_runner = nullptr;
      scratch.moe_shared_device_output = false;
      scratch.moe_device_output_ready = false;
    }
    if (!moe_device_consumed)
      fo.hc_post_fn(engine, l, /*sub_block=*/1, hidden_states,
                    scratch.moe_out.data(), S, D);

    check_finite("hidden_states (post FFN/MoE residual)", l, -1, hidden_states,
                 (size_t)n);
    if (trace)
      LOG_STREAM_VERBOSE() << "[TRACE l=" << l << "] post-ffn  |moe_out|="
                           << l2norm(scratch.moe_out.data() +
                                         (size_t)(S - 1) * D,
                                     D)
                           << " |hidden|=" << l2norm(last_tok, D) << "\n";

    // --- QWEN-PILOT (Qwen4Exp only) -------------------------------------
    // Qwen's production router consumes the HC graph's mixed activation, not
    // the conventional hidden->RMSNorm->gate input used by the generic pilot
    // below. After a device HC post-combine, engine.hc_stream_buf is stale;
    // take one explicit snapshot before projecting the next layer. The
    // stream-0 row is an intentionally conservative proxy for the mixed
    // activation and is labelled as such in the shutdown report. It never
    // changes routing or expert dispatch. Optional Qwen prefetch is kept
    // separate from the generic GLM/DeepSeek pilot: its top-K candidates are
    // either issued into the next layer's temporary active-layer area via
    // cudaMemcpyAsync (when already RAM-resident), or sent to the ordinary
    // disk/RAM hint queue (when they still need a RAM fetch).
    const bool qwen_pilot_skip_fully_cached =
        engine.qwen4exp_router_pilot_skip_fully_cached && S == 1 &&
        engine.family_ops.family == ModelFamily::Qwen4Exp &&
        ram_mgr.vram_cache && l + 1 < c.n_layers &&
        ram_mgr.vram_cache->layer_fully_resident(l + 1, c.n_experts);
    const bool qwen_pilot_skip_prefetch_excluded =
        engine.qwen4exp_router_pilot_skip_prefetch_excluded_layers && S == 1 &&
        engine.family_ops.family == ModelFamily::Qwen4Exp &&
        l + 1 < c.n_layers &&
        std::find(engine.qwen4exp_router_pilot_prefetch_exclude_layers.begin(),
                  engine.qwen4exp_router_pilot_prefetch_exclude_layers.end(),
                  l + 1) !=
            engine.qwen4exp_router_pilot_prefetch_exclude_layers.end();
    const bool qwen_pilot_skip_predict_policy =
        engine.family_ops.family == ModelFamily::Qwen4Exp &&
        l + 1 < c.n_layers &&
        !qwen4exp_router_pilot_layer_allowed(engine, l + 1);
    if (qwen_pilot_skip_fully_cached) {
      perf().n_qwen_pilot_skipped_fully_cached.fetch_add(
          1, std::memory_order_relaxed);
    }
    if (qwen_pilot_skip_prefetch_excluded) {
      perf().n_qwen_pilot_skipped_prefetch_excluded.fetch_add(
          1, std::memory_order_relaxed);
    }
    if (qwen_pilot_skip_predict_policy) {
      perf().n_qwen_pilot_skipped_predict_policy.fetch_add(
          1, std::memory_order_relaxed);
    }
    if (engine.qwen4exp_router_pilot && !qwen_pilot_skip_fully_cached &&
        !qwen_pilot_skip_prefetch_excluded && !qwen_pilot_skip_predict_policy &&
        engine.family_ops.family == ModelFamily::Qwen4Exp &&
        l >= c.first_dense && l + 1 < c.n_layers && l + 1 >= c.first_dense &&
        c.n_experts > 0 && c.hc_mult > 0) {
      scratch.qwen_pilot_pred_count = 0;
      scratch.qwen_pilot_pred_layer = -1;
      scratch.qwen_pilot_prefetch_count = 0;
      const int target_layer = l + 1;
      const int qwen_k = std::min({engine.qwen4exp_router_pilot_k, c.topk,
                                   LayerScratch::kQwenPilotMax});
      const void *gpu_pilot_ids = nullptr;
      const void *gpu_pilot_weights = nullptr;
      const bool gpu_pilot_ok = S == 1 && qwen_k > 0 &&
                                qwen4exp_router_pilot_gpu_async(
                                    engine, target_layer, D, c.n_experts,
                                    c.topk, &gpu_pilot_ids, &gpu_pilot_weights);
      (void)gpu_pilot_ids;
      (void)gpu_pilot_weights;
      if (gpu_pilot_ok) {
        scratch.qwen_pilot_gpu_pending = true;
        scratch.qwen_pilot_gpu_pending_layer = target_layer;
      } else {
        perf().n_qwen_pilot_host_fallbacks.fetch_add(1,
                                                     std::memory_order_relaxed);
      }
      if (!gpu_pilot_ok) {
        const size_t hc_need = (size_t)S * c.hc_mult * D;
        const bool host_snapshot_ok = engine.hc_stream_buf.size() >= hc_need &&
                                      qwen4exp_hc_download_streams(engine, S);
        if (!host_snapshot_ok) {
          perf().n_qwen_pilot_sync_fail.fetch_add(1, std::memory_order_relaxed);
        } else {
          const std::string qwen_router_key =
              "blk." + std::to_string(target_layer) + ".ffn_gate_inp.weight";
          auto qwen_router_it = engine.resident_layers.find(qwen_router_key);
          if (gpu_pilot_ok || (qwen_router_it != engine.resident_layers.end() &&
                               qwen_router_it->second.host_ptr)) {
            if (!gpu_pilot_ok) {
              const float *proxy =
                  engine.hc_stream_buf.data() + (size_t)(S - 1) * c.hc_mult * D;
              dense_proj(engine, qwen_router_key, proxy,
                         scratch.qwen_pilot_logits.data(), 1, D, c.n_experts);
              apply_gating_activation(
                  c.gating_func, scratch.qwen_pilot_logits.data(), c.n_experts);
              const std::string qwen_bias_key =
                  "blk." + std::to_string(target_layer) + ".exp_probs_b.bias";
              auto qwen_bias_it = engine.resident_layers.find(qwen_bias_key);
              if (qwen_bias_it != engine.resident_layers.end() &&
                  qwen_bias_it->second.host_ptr &&
                  qwen_bias_it->second.meta.type == GGUFType::F32) {
                const float *bias = reinterpret_cast<const float *>(
                    qwen_bias_it->second.host_ptr);
                for (int e = 0; e < c.n_experts; ++e)
                  scratch.qwen_pilot_logits[e] += bias[e];
              }
            }
            const int qwen_prefetch_k =
                std::min({engine.qwen4exp_router_pilot_prefetch_k > 0
                              ? engine.qwen4exp_router_pilot_prefetch_k
                              : engine.qwen4exp_router_pilot_k,
                          qwen_k, LayerScratch::kQwenPilotMax});
            const bool qwen_prefetch_layer_enabled =
                std::find(
                    engine.qwen4exp_router_pilot_prefetch_exclude_layers
                        .begin(),
                    engine.qwen4exp_router_pilot_prefetch_exclude_layers.end(),
                    target_layer) ==
                engine.qwen4exp_router_pilot_prefetch_exclude_layers.end();
            const bool qwen_prefetch_enabled =
                engine.qwen4exp_router_pilot_prefetch &&
                qwen_prefetch_layer_enabled;
            perf().n_qwen_pilot_prefetch_k_observed.store(
                (uint32_t)(engine.qwen4exp_router_pilot_prefetch
                               ? qwen_prefetch_k
                               : 0),
                std::memory_order_relaxed);
            if (engine.qwen4exp_router_pilot_prefetch &&
                !qwen_prefetch_layer_enabled) {
              perf().n_qwen_pilot_prefetch_excluded.fetch_add(
                  1, std::memory_order_relaxed);
            } else if (engine.qwen4exp_router_pilot_prefetch &&
                       qwen_prefetch_k < qwen_k) {
              perf().n_qwen_pilot_prefetch_rank_skipped.fetch_add(
                  (uint64_t)(qwen_k - qwen_prefetch_k),
                  std::memory_order_relaxed);
            }
            scratch.qwen_pilot_pred_count = qwen_k;
            scratch.qwen_pilot_pred_layer = target_layer;
            scratch.qwen_pilot_prefetch_count =
                qwen_prefetch_enabled ? qwen_prefetch_k : 0;
            perf().n_qwen_pilot_predictions.fetch_add(
                1, std::memory_order_relaxed);
            const int64_t prediction_ready_ns =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch())
                    .count();
            if (target_layer < PerfCounters::kMaxTrackedLayersForMiss) {
              perf().qwen_pilot_prediction_ready_ns[target_layer].store(
                  prediction_ready_ns, std::memory_order_release);
            }

            for (int k = 0; k < qwen_k; ++k) {
              int best = gpu_pilot_ok ? scratch.qwen_pilot_gpu_ids[k] : 0;
              if (!gpu_pilot_ok) {
                for (int e = 1; e < c.n_experts; ++e) {
                  if (scratch.qwen_pilot_logits[e] >
                      scratch.qwen_pilot_logits[best])
                    best = e;
                }
                scratch.qwen_pilot_logits[best] = -2e30f;
              }
              scratch.qwen_pilot_pred_ids[k] = best;

              // Classify without a mutating lookup. The active-layer pool is a
              // real VRAM source but is separate from the main LRU cache.
              bool already_vram =
                  ram_mgr.vram_cache &&
                  (ram_mgr.vram_cache->active_layer_has(target_layer, best) ||
                   ram_mgr.vram_cache->main_layer_has(target_layer, best));
              const bool already_ram =
                  !already_vram && ram_mgr.is_ram_resident(target_layer, best);
              const bool qwen_prefetch_this_rank =
                  qwen_prefetch_enabled && k < qwen_prefetch_k;
              if (already_vram) {
                perf().n_qwen_pilot_pred_already_vram.fetch_add(
                    1, std::memory_order_relaxed);
              } else if (already_ram) {
                perf().n_qwen_pilot_pred_already_ram.fetch_add(
                    1, std::memory_order_relaxed);
                if (qwen_prefetch_this_rank) {
                  // promote_ram_to_active_layer issues cudaMemcpyAsync on the
                  // dedicated active-layer stream and records the destination
                  // event. It also protects the RAM source until that event,
                  // so the prefetch worker cannot reclaim the slab underneath
                  // the DMA.
                  ExpertLocation *issued = ram_mgr.promote_ram_to_active_layer(
                      engine, target_layer, best, -1, prediction_ready_ns);
                  if (issued) {
                    perf().n_qwen_pilot_active_prefetch_issued.fetch_add(
                        1, std::memory_order_relaxed);
                  } else {
                    perf().n_qwen_pilot_active_prefetch_failed.fetch_add(
                        1, std::memory_order_relaxed);
                  }
                }
              } else {
                perf().n_qwen_pilot_pred_needs_fetch.fetch_add(
                    1, std::memory_order_relaxed);
                if (qwen_prefetch_this_rank) {
                  async_io_queue.push_hint(target_layer, best);
                  perf().n_qwen_pilot_hints_pushed.fetch_add(
                      1, std::memory_order_relaxed);
                }
              }
              if (qwen_prefetch_this_rank && already_vram) {
                perf().n_qwen_pilot_hints_skipped_resident.fetch_add(
                    1, std::memory_order_relaxed);
              }
            }
          }
        }
      }
    }

    // EngineConfig::pilot_widen_all_tokens -- see that field's own
    // comment (engine_config.hpp) for the full motivation. Deliberately
    // self-contained (own norm/router/bias lookups, own scratch
    // buffers) rather than folded into DEPTH-1's own block below --
    // keeps this from touching DEPTH-1's real hint-push/promotion
    // logic at all, by construction, not just by care. The real hint-
    // push (prefetch_widen_all_tokens) lives in this same block, right
    // after the widened prediction is computed -- see that inner
    // comment for the residency-check/push logic itself.
    if (engine.pilot_widen_all_tokens && l + 1 < c.n_layers &&
        c.n_experts > 0 && S > 1) {
      std::string wnorm_key =
          "blk." + std::to_string(l + 1) + ".ffn_norm.weight";
      std::string wrouter_key =
          "blk." + std::to_string(l + 1) + ".ffn_gate_inp.weight";
      auto wnorm_it = engine.resident_layers.find(wnorm_key);
      auto wrouter_it = engine.resident_layers.find(wrouter_key);
      if (wnorm_it != engine.resident_layers.end() &&
          wrouter_it != engine.resident_layers.end() &&
          wnorm_it->second.host_ptr && wrouter_it->second.host_ptr) {
        const float *wnorm_w =
            reinterpret_cast<float *>(wnorm_it->second.host_ptr);
        std::string wbias_key =
            "blk." + std::to_string(l + 1) + ".exp_probs_b.bias";
        auto wbias_it = engine.resident_layers.find(wbias_key);
        const float *wbias_row =
            (wbias_it != engine.resident_layers.end() &&
             wbias_it->second.host_ptr)
                ? reinterpret_cast<float *>(wbias_it->second.host_ptr)
                : nullptr;

        static thread_local std::vector<bool> wide_seen;
        wide_seen.assign(c.n_experts, false);
        scratch.pilot_wide_pred_count = 0;
        int per_token_k = std::min(pilot_k_for_layer(l + 1), c.topk);
        for (int ws = 0; ws < S; ++ws) {
          const float *tok_hidden = hidden_states + ((size_t)ws * D);
          rmsnorm_row(scratch.pilot_wide_nrm.data(), tok_hidden, wnorm_w, D,
                      c.eps);
          dense_proj(engine, wrouter_key, scratch.pilot_wide_nrm.data(),
                     scratch.pilot_wide_logits.data(), 1, D, c.n_experts);
          apply_gating_activation(
              c.gating_func, scratch.pilot_wide_logits.data(), c.n_experts);
          if (wbias_row) {
            for (int e = 0; e < c.n_experts; ++e)
              scratch.pilot_wide_logits[e] += wbias_row[e];
          }
          for (int k = 0; k < per_token_k; ++k) {
            int best = -1;
            for (int e = 0; e < c.n_experts; ++e) {
              if (best < 0 || scratch.pilot_wide_logits[e] >
                                  scratch.pilot_wide_logits[best])
                best = e;
            }
            if (best < 0)
              break;
            scratch.pilot_wide_logits[best] = -2e30f;
            if (!wide_seen[best] &&
                scratch.pilot_wide_pred_count <
                    (int)scratch.pilot_wide_pred_ids.size()) {
              wide_seen[best] = true;
              scratch.pilot_wide_pred_ids[scratch.pilot_wide_pred_count++] =
                  best;
            }
          }
        }
        scratch.pilot_wide_pred_layer = l + 1;

        // EngineConfig::prefetch_widen_all_tokens -- real disk-
        // prefetch hint push from the widened prediction just
        // computed above. Mirrors DEPTH-1's own already_ram/
        // already_vram residency check + async_io_queue.push_hint
        // call (below, in DEPTH-1's own k-loop) exactly, applied to
        // this block's much wider candidate set instead of DEPTH-1's
        // fixed 8 -- deliberately HINT-PUSH ONLY, no VRAM promotion
        // (see this flag's own comment, engine_config.hpp, for why).
        // PilotSource::Other (same untagged high-priority lane
        // DEPTH-1 itself uses) -- no dedicated source tag added for
        // this experiment; [PILOT-WIDE]'s own coverage/recall
        // numbers already measure this source's prediction quality
        // directly, so a separate bytes_prefetched_widen-style
        // completion counter (the treatment Depth2 gets) wasn't
        // judged worth the added instrumentation surface yet.
        if (engine.prefetch_widen_all_tokens) {
          for (int i = 0; i < scratch.pilot_wide_pred_count; ++i) {
            int wide_id = scratch.pilot_wide_pred_ids[i];
            bool w_already_ram = ram_mgr.is_ram_resident(l + 1, wide_id);
            bool w_already_vram = false;
            if (!w_already_ram && ram_mgr.vram_cache) {
              for (const auto &loc : ram_mgr.vram_cache->expert_vram_cache) {
                if (loc.layer_idx == l + 1 && loc.expert_id == wide_id) {
                  w_already_vram = true;
                  break;
                }
              }
            }
            if (!w_already_ram && !w_already_vram) {
              async_io_queue.push_hint(l + 1, wide_id);
              perf().n_pilot_wide_hints_pushed.fetch_add(
                  1, std::memory_order_relaxed);
            } else {
              perf().n_pilot_wide_hints_skipped_resident.fetch_add(
                  1, std::memory_order_relaxed);
            }
          }
        }
      }
    }

    // ATTN-PILOT's own prediction (deferred from right after l's own
    // attention -- see scratch.pilot_attn_snapshot's own comment for
    // why: consumption of l's own pending prediction, inside l's just-
    // finished moe_forward_gguf/dense_ffn_gguf call, must happen BEFORE
    // this overwrites the same field) is computed further below, INSIDE
    // DEPTH-1's own block right after this comment -- not here -- since
    // it targets the SAME layer (l+1) using the SAME norm/router/bias
    // tensors DEPTH-1 already looks up, and shares DEPTH-1's own
    // confirmed_resident_this_transition/publish_pilot_confirmed_
    // resident call for its real hints. See that site for the actual
    // firing logic.

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
      std::string norm_key =
          "blk." + std::to_string(l + 1) + ".ffn_norm.weight";
      std::string router_key =
          "blk." + std::to_string(l + 1) + ".ffn_gate_inp.weight";
      auto norm_it = engine.resident_layers.find(norm_key);
      auto router_it = engine.resident_layers.find(router_key);

      // This pilot reconstructs the conventional norm->gate router input.
      // Qwen4Exp derives its route input through the HC/GPU graph instead,
      // so never apply this approximation to Qwen; its production router
      // remains the only source of truth there. GLM/DeepSeek-style families
      // stay on this existing path.
      const bool generic_pilot_inputs_ready =
          engine.family_ops.family != ModelFamily::Qwen4Exp &&
          norm_it != engine.resident_layers.end() &&
          router_it != engine.resident_layers.end() &&
          norm_it->second.host_ptr && router_it->second.host_ptr;
      const float *last_hidden = hidden_states + ((S - 1) * D);

      if (generic_pilot_inputs_ready) {
        // Use only last token for pilot prediction (saves compute)
        rmsnorm_row(scratch.pilot_nrm.data(), last_hidden,
                    reinterpret_cast<float *>(norm_it->second.host_ptr), D,
                    c.eps);

        dense_proj(engine, router_key, scratch.pilot_nrm.data(),
                   scratch.pilot_logits.data(), 1, D, c.n_experts);

        // Rank by EXACTLY the same "selection_probs" the real router
        // uses (moe_forward_gguf above: gate_func(logits) + bias),
        // not raw dense_proj output. Missing this was a second,
        // independent misprediction source on top of the residual-
        // timing fix above --GLM-5.2/DeepSeek-V3's exp_probs_b.bias exists
        // specifically to shift WHICH experts get selected for
        // load-balancing (auxiliary-loss-free routing) -- ranking on
        // the un-biased score means the pilot's top-k has no reason
        // to match the real, bias-corrected top-k the router
        // actually dispatches, independent of how accurate the input
        // hidden_states is.
        apply_gating_activation(c.gating_func, scratch.pilot_logits.data(),
                                c.n_experts);
        {
          std::string bias_key =
              "blk." + std::to_string(l + 1) + ".exp_probs_b.bias";
          auto bias_it = engine.resident_layers.find(bias_key);
          if (bias_it != engine.resident_layers.end() &&
              bias_it->second.host_ptr) {
            const float *bias_row =
                reinterpret_cast<float *>(bias_it->second.host_ptr);
            for (int e = 0; e < c.n_experts; ++e)
              scratch.pilot_logits[e] += bias_row[e];
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
        PilotConfirmedEntry
            confirmed_resident_this_transition[PILOT_CONFIRMED_CAP];
        int confirmed_resident_count = 0;
        // Cross-pilot VRAM-promotion dedup (2026-08-25) -- DEPTH-1's
        // and ATTN-PILOT's k-loops both target l+1 from two
        // different inputs, so their top-K sets routinely overlap.
        // Without this, both loops could independently call
        // ram_mgr.promote_ram_to_vram() for the SAME (layer,expert)
        // within the SAME transition -- and since GpuTopology::
        // find_expert only reports a slot resident once its async
        // H2D copy is CONFIRMED complete (cudaEventQuery), the
        // SECOND call (issued moments after the first, DMA still in
        // flight) sees "not resident yet" and genuinely re-issues a
        // second evict_lru + upload_expert_async for the exact same
        // expert -- a real wasted VRAM slot and DMA, not a harmless
        // re-check. Scoped local to this one transition (both
        // loops run sequentially on this same thread for this same
        // l+1), NOT a persistent/global structure -- deliberately
        // separate from confirmed_resident_this_transition above,
        // which tracks a different population (already-resident
        // picks for the eviction-race measurement) and must not be
        // conflated with "a promotion was issued this transition".
        // Capacity 16 = DEPTH-1's own promotion candidates (<=8,
        // rank < ram_vram_promote_top_k) + ATTN-PILOT's own
        // (<=8, rank < attn_pilot_vram_promote_top_k) -- generous,
        // not a hard model limit (mirrors PILOT_CONFIRMED_CAP's own
        // sizing comment).
        struct VramPromoAttempt {
          int layer_idx;
          int expert_id;
        };
        constexpr int MAX_VRAM_PROMO_PER_TRANSITION = 16;
        VramPromoAttempt
            vram_promoted_this_transition[MAX_VRAM_PROMO_PER_TRANSITION];
        int vram_promoted_count = 0;
        auto try_promote_ram_to_vram_deduped = [&](int layer_idx,
                                                   int expert_id) {
          for (int i = 0; i < vram_promoted_count; ++i) {
            if (vram_promoted_this_transition[i].layer_idx == layer_idx &&
                vram_promoted_this_transition[i].expert_id == expert_id) {
              perf().n_pilot_promote_dedup_skipped.fetch_add(
                  1, std::memory_order_relaxed);
              return;
            }
          }
          ram_mgr.promote_ram_to_vram(engine, layer_idx, expert_id);
          if (vram_promoted_count < MAX_VRAM_PROMO_PER_TRANSITION) {
            vram_promoted_this_transition[vram_promoted_count++] = {layer_idx,
                                                                    expert_id};
          }
        };
        // Active-layer sibling of the dedup wrapper above
        // (EngineConfig::speculative_promote_to_active_layer,
        // 2026-08-26) -- own, separate dedup buffer, NOT shared with
        // vram_promoted_this_transition: a (layer,expert) promoted
        // into the active layer and one promoted into the main
        // cache are tracked independently, since only one of the
        // two call sites below is ever active per run (the config
        // flag routes ALL of this transition's promotion calls to
        // exactly one of the two lambdas), so no cross-population
        // collision is possible in practice -- kept separate anyway
        // for clarity, mirroring MAX_VRAM_PROMO_PER_TRANSITION's own
        // sizing rationale.
        VramPromoAttempt active_layer_promoted_this_transition
            [MAX_VRAM_PROMO_PER_TRANSITION];
        int active_layer_promoted_count = 0;
        // 2026-08-26: NO batch-sync buffer needed anymore -- the
        // async DMA promote_ram_to_active_layer issues (upload_
        // ready left false) is now safe to just let run in the
        // background. find_active_expert (gpu_topology.cpp) polls
        // the slot's own completion event on the first real lookup
        // instead of trusting upload_ready blindly, exactly
        // mirroring how the main VRAM cache's finish_expert_hit
        // already handles its own async promotions -- if the DMA
        // genuinely hasn't landed by the time l+1 dispatch reaches
        // it, that single lookup is treated as a miss (see
        // n_active_layer_poll_not_ready) rather than blocking here.
        // A real, measured ~3.1s/request cost from the FIRST
        // (fully synchronous) version of this feature, and a
        // smaller but still real bandwidth-competing cost from the
        // SECOND (batched-but-still-blocking) version, are both
        // gone with this change -- there is no unconditional wait
        // left on this path at all now, only an on-demand poll
        // that costs nothing when the copy is already done (the
        // overwhelmingly common case, given a whole layer's worth
        // of intervening work before l+1 is dispatched).
        auto try_promote_ram_to_active_layer_deduped = [&](int layer_idx,
                                                           int expert_id,
                                                           int rank) {
          for (int i = 0; i < active_layer_promoted_count; ++i) {
            if (active_layer_promoted_this_transition[i].layer_idx ==
                    layer_idx &&
                active_layer_promoted_this_transition[i].expert_id ==
                    expert_id) {
              perf().n_pilot_active_fill_dedup_skipped.fetch_add(
                  1, std::memory_order_relaxed);
              return;
            }
          }
          // 2026-08-28: multi-GPU routing decision -- see
          // route_active_layer_gpu's own comment (dense_
          // dispatch.hpp). -1 (no multi-GPU config) is the
          // correct "no restriction" value for the legacy path.
          // total_candidates = speculative_active_layer_max_k_
          // value(), NOT a fixed nominal 8 -- k here can never
          // exceed that bound (this whole call site is already
          // gated by k < speculative_active_layer_max_k_value()
          // a few lines up the call chain), so THAT is the real
          // reachable range the tail must be computed against.
          int routed_gpu =
              active_layer_multi_gpu_enabled()
                  ? route_active_layer_gpu(
                        rank, speculative_active_layer_max_k_value())
                  : -1;
          if (routed_gpu >= 0) {
            perf()
                .n_active_layer_routed_by_gpu
                    [routed_gpu < PerfCounters::PERF_MAX_GPU_SLOTS ? routed_gpu
                                                                   : 0]
                .fetch_add(1, std::memory_order_relaxed);
          }
          // 2026-08-28: attribute the promotion to its rank (used
          // by record_active_layer_prewarm_reclaim's per-rank
          // used/wasted split, gpu_topology.hpp) -- see
          // ExpertLocation::promoted_at_rank's own comment.
          ExpertLocation *promoted = nullptr;
          if (ram_mgr.upload_workers && ram_mgr.upload_workers->enabled()) {
            // See RAMSafetyManager::promote_ram_to_active_layer_
            // claim_for_worker's own comment for why this claim/
            // submit split exists (ActiveLayerUploadWorkers is
            // only forward-declared inside ram_manager.hpp, so
            // the actual try_submit call has to happen here,
            // where the full type is visible).
            auto claim = ram_mgr.promote_ram_to_active_layer_claim_for_worker(
                engine, layer_idx, expert_id, routed_gpu);
            if (claim.slot) {
              ActiveLayerUploadWorkItem item{claim.slot, claim.blk,
                                             claim.host_data, claim.host_bytes};
              if (ram_mgr.upload_workers->try_submit(item)) {
                promoted = claim.slot;
              } else {
                ram_mgr.vram_cache->abandon_worker_claim(claim.slot);
                ram_mgr.ram_unpin(*claim.blk);
              }
            }
          } else {
            promoted = ram_mgr.promote_ram_to_active_layer(
                engine, layer_idx, expert_id, routed_gpu);
          }
          if (promoted && rank >= 0 && rank < 8)
            promoted->promoted_at_rank = (int8_t)rank;
          if (active_layer_promoted_count < MAX_VRAM_PROMO_PER_TRANSITION) {
            active_layer_promoted_this_transition
                [active_layer_promoted_count++] = {layer_idx, expert_id};
          }
        };
        // Adaptive-mode contiguous-prefix cutoff state for THIS
        // transition's k-loop -- ranks are walked in increasing
        // order below, so tracking "still eligible" as we go
        // correctly implements "promote rank k only if every rank
        // < k also cleared the bar" without a second pass. Reset
        // per transition (not persisted across layers/tokens) --
        // whether rank k clears the bar is a function of the LIVE,
        // GLOBAL n_pilot_ram_only_* counters at the moment this
        // runs, not of anything specific to this one transition.
        // Unused in static top_k mode.
        bool adaptive_promote_still_eligible = true;
        const auto &promo_cfg = ram_vram_promote_config();
        // "Idea 1" prediction-based GPU1 dual-role-slice loading
        // (2026-08-16) -- see RAMSafetyManager::promote_ram_to_
        // vram_gpu1's own comment for the full design. Resolved
        // ONCE per transition, not per rank: dual_role_gpu_id()
        // takes GpuTopology::lock internally (a real, if cheap,
        // lock acquisition -- see that function's own comment), no
        // reason to pay it up to 8x when the answer can't change
        // within one transition. predict_gpu1_promotions_this_
        // transition mirrors adaptive_promote_still_eligible's own
        // per-transition-local-variable pattern directly above.
        const auto &gpu1_predict_cfg = gpu1_predict_promote_config();
        int gpu1_predict_dual_role_gpu = -1;
        if (gpu1_predict_cfg.enabled && ram_mgr.vram_cache) {
          gpu1_predict_dual_role_gpu = ram_mgr.vram_cache->dual_role_gpu_id();
        }
        int gpu1_predict_promotions_this_transition = 0;
        for (int k = 0; k < pilot_k; ++k) {
          int best_exp = 0;
          for (int e = 1; e < c.n_experts; ++e) {
            if (scratch.pilot_logits[e] > scratch.pilot_logits[best_exp])
              best_exp = e;
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
            for (const auto &loc : ram_mgr.vram_cache->expert_vram_cache) {
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
            perf().n_pilot_pred_already_ram.fetch_add(
                1, std::memory_order_relaxed);
            if (confirmed_resident_count < PILOT_CONFIRMED_CAP)
              confirmed_resident_this_transition[confirmed_resident_count++] = {
                  l + 1, best_exp};

            // Pilot-driven RAM->VRAM speculative promotion --
            // see dense_dispatch.hpp's ram_vram_promote_config
            // for config wiring and RAMSafetyManager::
            // promote_ram_to_vram for the actual upload +
            // deadlock-safety contract. This branch (already_
            // ram) is deliberately the ONLY place this fires:
            // it's the exact population the "ram-only" per-rank
            // metric measures, disjoint from is_hint_worthy's
            // needs_fetch population above and from the two
            // already_vram branches below (which need no
            // action -- already where they need to be).
            bool should_promote = false;
            if (promo_cfg.dynamic_layer) {
              // 2026-08-26: per-layer WIDTH instead of a flat
              // cutoff -- see dynamic_layer_top_k's own
              // comment (dense_dispatch.hpp). Judged against
              // DEPTH-1's own per-layer accuracy
              // (is_attn_pilot=false).
              should_promote =
                  (k < dynamic_layer_top_k(l + 1, /*is_attn_pilot=*/false));
            } else if (promo_cfg.top_k > 0) {
              should_promote = (k < promo_cfg.top_k);
            } else if (promo_cfg.adaptive && adaptive_promote_still_eligible) {
              uint64_t checks =
                  (k < 8) ? perf().n_pilot_ram_only_checks_by_rank[k].load(
                                std::memory_order_relaxed)
                          : 0;
              uint64_t rhits =
                  (k < 8) ? perf().n_pilot_ram_only_hits_by_rank[k].load(
                                std::memory_order_relaxed)
                          : 0;
              double acc = checks > 0 ? (double)rhits / (double)checks : 0.0;
              if (checks >= (uint64_t)promo_cfg.min_samples &&
                  acc >= promo_cfg.adaptive_threshold) {
                should_promote = true;
              } else {
                // Contiguous-prefix cutoff: this rank failed
                // (too few samples OR accuracy below
                // threshold) -- no later rank this
                // transition may promote either, even if a
                // deeper rank's OWN live accuracy happens to
                // look higher (matches the monotonic-decay
                // shape observed in every real capture so
                // far; a later rank clearing the bar alone
                // isn't trusted).
                adaptive_promote_still_eligible = false;
              }
            }
            // ram_vram_promote_layer_gate (2026-08-25): additional
            // per-LAYER veto on top of should_promote above -- see
            // dense_dispatch.hpp's layer_promote_allowed for the
            // full design (real [PILOT-BY-LAYER] accuracy spread
            // by layer, CLAUDE.md). is_attn_pilot=false: judged
            // against DEPTH-1's OWN per-layer counters. Split into
            // an explicit if/else (rather than folding the veto
            // into one && condition) so a real veto is directly
            // counted, not just inferred from a smaller attempted
            // total. SKIPPED when promo_cfg.dynamic_layer is
            // already active (2026-08-26) -- dynamic_layer_top_k
            // above already IS the per-layer decision (using its
            // own, possibly different, thresholds); re-applying
            // this separate flat-veto's threshold on top would
            // just double-gate against two different bars rather
            // than compose meaningfully.
            if (should_promote) {
              if (promo_cfg.dynamic_layer ||
                  layer_promote_allowed(l + 1, /*is_attn_pilot=*/false)) {
                // speculative_promote_to_active_layer
                // (2026-08-26): route into the cheap,
                // self-contained active-layer pool instead
                // of the main VRAM LRU cache -- see
                // EngineConfig::speculative_promote_to_
                // active_layer's own comment for the full
                // motivation.
                if (speculative_promote_to_active_layer_enabled()) {
                  // speculative_active_layer_max_k: a
                  // SEPARATE, narrower cutoff than
                  // should_promote's own width -- see its
                  // own comment (dense_dispatch.hpp) for
                  // the real A/B (added DMA volume
                  // competing for the same bandwidth-
                  // bound pipe decode_async_prewarm uses)
                  // that motivated decoupling this from
                  // the (now-bypassed) main-cache width.
                  if (k < speculative_active_layer_max_k_value()) {
                    try_promote_ram_to_active_layer_deduped(l + 1, best_exp, k);
                  } else {
                    // 2026-08-28: direct answer to "is
                    // max_k actually the binding
                    // constraint, or is the tier/
                    // threshold structure already
                    // stricter" -- fires only when the
                    // tier system (should_promote, just
                    // above) already approved this rank
                    // but max_k clipped it anyway. If
                    // this stays near-zero across a
                    // max_k range, raising max_k further
                    // can't be doing anything real --
                    // the tiers themselves are the
                    // actual ceiling. See perf_timer.hpp
                    // for the report line.
                    perf().n_speculative_max_k_binding_depth1.fetch_add(
                        1, std::memory_order_relaxed);
                  }
                } else {
                  try_promote_ram_to_vram_deduped(l + 1, best_exp);
                }
              } else {
                perf().n_pilot_promote_layer_gate_vetoed_depth1.fetch_add(
                    1, std::memory_order_relaxed);
              }
            }
            // "Idea 1" -- deliberately a SEPARATE decision from
            // should_promote above, not an else-branch: a rank
            // could in principle satisfy both ram_vram_promote_
            // top_k (general pool) and this rank window at
            // once, though in practice they're configured to
            // disjoint rank ranges (see doc/gpu-utilization-
            // vram-analysis.md). Same "already_ram" population,
            // same k this rank's real predicted expert id
            // (best_exp) -- no new classification needed.
            if (gpu1_predict_dual_role_gpu >= 0 &&
                k >= gpu1_predict_cfg.rank_start &&
                k <= gpu1_predict_cfg.rank_end) {
              if (gpu1_predict_promotions_this_transition <
                  gpu1_predict_cfg.cap_per_layer) {
                ram_mgr.promote_ram_to_vram_gpu1(engine, l + 1, best_exp,
                                                 gpu1_predict_dual_role_gpu);
                ++gpu1_predict_promotions_this_transition;
              } else {
                perf().n_predict_gpu1_cap_reached.fetch_add(
                    1, std::memory_order_relaxed);
              }
            }
          } else if (already_vram && vram_pinned) {
            perf().n_pilot_pred_already_vram_pinned.fetch_add(
                1, std::memory_order_relaxed);
            // Deliberately NOT added to confirmed_resident_this_transition
            // -- a pinned slot can never be an evict_lru victim, see
            // that buffer's own comment above.
          } else if (already_vram) {
            perf().n_pilot_pred_already_vram_lru.fetch_add(
                1, std::memory_order_relaxed);
            if (confirmed_resident_count < PILOT_CONFIRMED_CAP)
              confirmed_resident_this_transition[confirmed_resident_count++] = {
                  l + 1, best_exp};
          } else {
            perf().n_pilot_pred_needs_fetch.fetch_add(
                1, std::memory_order_relaxed);
          }
          scratch.pilot_pred_was_hint[k] = is_hint_worthy;
          scratch.pilot_pred_was_ram_only[k] = already_ram;
          if (is_hint_worthy) {
            async_io_queue.push_hint(l + 1, best_exp);
          } else {
            perf().n_pilot_hints_skipped.fetch_add(1,
                                                   std::memory_order_relaxed);
          }
        }

        // --- ATTN-PILOT: real hint-wiring (2026-08-25), gated
        // behind EngineConfig::prefetch_attn_pilot (default off).
        // Targets the SAME layer (l+1) as DEPTH-1 just above, using
        // the SAME norm_key/router_key tensors (reused, not
        // re-looked-up) -- the only difference is the INPUT:
        // scratch.pilot_attn_snapshot (l's post-attention/pre-FFN
        // state, captured earlier and deferred here for the
        // ordering reason explained at the snapshot site) instead
        // of DEPTH-1's live post-FFN hidden_states. Real recall:
        // 75.8% overall (vs DEPTH-1's 84.8% for this same target,
        // vs DEPTH-2's 72.1% when read as reaching 2 layers past
        // the last known-complete layer) -- see CLAUDE.md. Shares
        // DEPTH-1's own confirmed_resident_this_transition buffer
        // and the ONE publish_pilot_confirmed_resident call below
        // (not a second one) -- PILOT_CONFIRMED_CAP raised to 24 to
        // cover all three contributors now pilot_confirmed_set.hpp).
        bool attn_pilot_enabled = prefetch_attn_pilot_enabled();
        {
          rmsnorm_row(
              scratch.pilot_nrm.data(), scratch.pilot_attn_snapshot.data(),
              reinterpret_cast<float *>(norm_it->second.host_ptr), D, c.eps);
          dense_proj(engine, router_key, scratch.pilot_nrm.data(),
                     scratch.pilot_logits.data(), 1, D, c.n_experts);
          apply_gating_activation(c.gating_func, scratch.pilot_logits.data(),
                                  c.n_experts);
          {
            std::string abias_key =
                "blk." + std::to_string(l + 1) + ".exp_probs_b.bias";
            auto abias_it = engine.resident_layers.find(abias_key);
            if (abias_it != engine.resident_layers.end() &&
                abias_it->second.host_ptr) {
              const float *abias_row =
                  reinterpret_cast<float *>(abias_it->second.host_ptr);
              for (int e = 0; e < c.n_experts; ++e)
                scratch.pilot_logits[e] += abias_row[e];
            }
          }
          int attn_pilot_k = std::min(8, c.topk);
          int attn_pilot_push_cutoff =
              std::min(attn_pilot_k_for_layer(l + 1), attn_pilot_k);
          scratch.pilot_attn_pred_count = attn_pilot_k;
          scratch.pilot_attn_pred_layer = l + 1;
          for (int k = 0; k < attn_pilot_k; ++k) {
            int best_exp = 0;
            for (int e = 1; e < c.n_experts; ++e) {
              if (scratch.pilot_logits[e] > scratch.pilot_logits[best_exp])
                best_exp = e;
            }
            scratch.pilot_logits[best_exp] = -2e30f;
            scratch.pilot_attn_pred_ids[k] = best_exp;

            // Residency classification -- ALWAYS runs
            // (measurement, drives the fetch-worthy recall
            // split) regardless of whether this pick is
            // actually pushed as a real hint.
            bool already_ram_a = ram_mgr.is_ram_resident(l + 1, best_exp);
            bool already_vram_a = false, vram_pinned_a = false;
            if (!already_ram_a && ram_mgr.vram_cache) {
              for (const auto &loc : ram_mgr.vram_cache->expert_vram_cache) {
                if (loc.layer_idx == l + 1 && loc.expert_id == best_exp) {
                  already_vram_a = true;
                  vram_pinned_a = loc.pinned;
                  break;
                }
              }
            }
            bool is_hint_worthy_a = !already_ram_a && !already_vram_a;
            bool attn_pilot_active_here =
                attn_pilot_enabled && k < attn_pilot_push_cutoff;
            if (already_ram_a) {
              perf().n_pilot_attn_pred_already_ram.fetch_add(
                  1, std::memory_order_relaxed);
              if (attn_pilot_active_here &&
                  confirmed_resident_count < PILOT_CONFIRMED_CAP)
                confirmed_resident_this_transition[confirmed_resident_count++] =
                    {l + 1, best_exp};
              // Speculative RAM->VRAM promotion for ATTN-PILOT's
              // own top-ranked already-RAM picks -- independent
              // of attn_pilot_active_here/prefetch_attn_pilot
              // above (this fires for ANY already-RAM pick
              // within rank, regardless of whether ATTN-PILOT's
              // own real-hint-push put it there), gated only by
              // its own separate cutoff. Reuses the SAME
              // proven, COPY-only mechanism DEPTH-1's own pilot
              // loop already uses -- see attn_pilot_vram_
              // promote_top_k_value()'s own comment (dense_
              // dispatch.hpp) for why this is a SEPARATE
              // cutoff, not a reuse of ram_vram_promote_top_k.
              // ram_vram_promote_layer_gate (2026-08-25) /
              // ram_vram_promote_dynamic_layer (2026-08-26):
              // is_attn_pilot=true throughout -- judged
              // against ATTN-PILOT's OWN per-layer counters,
              // not DEPTH-1's (see the DEPTH-1 call site's
              // own comment for why). attn_pilot_dynamic_here
              // mirrors DEPTH-1's own dynamic-vs-flat branch:
              // when both the shared dynamic_layer mode AND
              // ATTN-PILOT's own opt-in
              // (attn_pilot_vram_promote_dynamic) are set, the
              // per-rank cutoff becomes a per-layer WIDTH
              // instead of the fixed attn_pilot_vram_promote_
              // top_k -- and the separate flat veto below is
              // skipped for the same reason DEPTH-1's is (the
              // dynamic width already IS the per-layer
              // decision).
              bool attn_pilot_dynamic_here =
                  promo_cfg.dynamic_layer &&
                  attn_pilot_vram_promote_dynamic_enabled();
              int attn_promote_cutoff =
                  attn_pilot_dynamic_here
                      ? dynamic_layer_top_k(l + 1, /*is_attn_pilot=*/true)
                      : attn_pilot_vram_promote_top_k_value();
              if (k < attn_promote_cutoff) {
                if (attn_pilot_dynamic_here ||
                    layer_promote_allowed(l + 1, /*is_attn_pilot=*/true)) {
                  // See the DEPTH-1 call site's own comment
                  // above for speculative_promote_to_
                  // active_layer -- same flag, same
                  // redirect, judged against the SAME
                  // shared per-transition dedup discipline
                  // (a separate buffer from DEPTH-1's own).
                  if (speculative_promote_to_active_layer_enabled()) {
                    // speculative_active_layer_max_k --
                    // see the DEPTH-1 call site's own
                    // comment above for the full
                    // reasoning; same cutoff, shared
                    // between both pilots.
                    if (k < speculative_active_layer_max_k_value()) {
                      try_promote_ram_to_active_layer_deduped(l + 1, best_exp,
                                                              k);
                    } else {
                      // 2026-08-28: ATTN-PILOT's own
                      // copy of the DEPTH-1 call
                      // site's max_k-binding check --
                      // see n_speculative_max_k_
                      // binding_depth1's own comment.
                      perf().n_speculative_max_k_binding_attn.fetch_add(
                          1, std::memory_order_relaxed);
                    }
                  } else {
                    try_promote_ram_to_vram_deduped(l + 1, best_exp);
                  }
                } else {
                  perf().n_pilot_promote_layer_gate_vetoed_attn.fetch_add(
                      1, std::memory_order_relaxed);
                }
              }
            } else if (already_vram_a && vram_pinned_a) {
              perf().n_pilot_attn_pred_already_vram_pinned.fetch_add(
                  1, std::memory_order_relaxed);
            } else if (already_vram_a) {
              perf().n_pilot_attn_pred_already_vram_lru.fetch_add(
                  1, std::memory_order_relaxed);
              if (attn_pilot_active_here &&
                  confirmed_resident_count < PILOT_CONFIRMED_CAP)
                confirmed_resident_this_transition[confirmed_resident_count++] =
                    {l + 1, best_exp};
            } else {
              perf().n_pilot_attn_pred_needs_fetch.fetch_add(
                  1, std::memory_order_relaxed);
            }
            scratch.pilot_attn_pred_was_hint[k] = is_hint_worthy_a;
            if (attn_pilot_active_here && is_hint_worthy_a) {
              async_io_queue.push_hint(l + 1, best_exp, PilotSource::AttnPilot);
              perf().n_pilot_attn_hints_pushed.fetch_add(
                  1, std::memory_order_relaxed);
            }
          }
        }

        // 2026-08-26: no batch-sync call needed here anymore --
        // see try_promote_ram_to_active_layer_deduped's own comment
        // above for why (find_active_expert now polls each slot's
        // own completion event on first real lookup instead of
        // requiring an unconditional wait here).

        // Publish AFTER the full k-loop (not per-pick) so a reader
        // (evict_lru/RAM-tier eviction, running on the decode
        // thread same as this loop but conceptually racy in
        // principle) always sees a complete, internally-consistent
        // set for this transition rather than a partially-filled
        // one -- see was_pilot_confirmed_resident's own comment on
        // why this is measurement-only, not a hard correctness
        // dependency either way.
        // --- DEPTH-2 measurement (approach (a), "naive reuse"),
        // now with REAL prefetch wiring gated by
        // prefetch_pilot_depth2_enabled() ---
        // Predicts l+2's routing by reusing the SAME `last_hidden`
        // vector above -- i.e. gated on a vector missing l+1's own
        // FFN contribution entirely, the exact term this block's
        // own opening comment already showed is fatal to omit at
        // depth-1. Reuses scratch.pilot_nrm/pilot_logits (l+1's
        // k-loop has already consumed everything it needed from
        // them). The MEASUREMENT (recall counters, residency
        // classification, "Pilot recall by rank" report) always
        // runs -- it's cheap (one more gate pass, same order as
        // the l+1 pilot's own query) and lets a user evaluate
        // depth-2 before ever turning it on. Only the REAL
        // async_io_queue.push_hint / publish_pilot_confirmed_
        // resident side effects are gated by
        // prefetch_pilot_depth2_enabled() (EngineConfig::
        // prefetch_pilot_depth) and further limited to the top
        // pilot_k2_for_layer() ranks (EngineConfig::
        // prefetch_pilot_k_depth2) -- see that config field's own
        // comment for why a separate, often-smaller cutoff makes
        // sense here than for l+1. (An FFN-reconstruction
        // alternative, approach "(b)", was measured statistically
        // tied with this despite costing real compute, and was
        // removed -- see CLAUDE.md.)
        //
        // confirmed_resident_this_transition/count (declared above,
        // in the l+1 block) is REUSED, not republished separately
        // -- l+1's and l+2's confirmed-resident entries are
        // published together in ONE call after this block, so a
        // reader (evict_lru/RAM-tier eviction) always sees a
        // complete, internally-consistent set for the WHOLE
        // transition, matching that publish call's own
        // single-call-after-the-full-k-loop reasoning at l+1's
        // depth. PILOT_CONFIRMED_CAP (16) already covers the
        // worst case (up to 8 from l+1 + up to 8 from l+2).
        bool depth2_enabled = prefetch_pilot_depth2_enabled();
        if (l + 2 < c.n_layers && c.n_experts > 0) {
          std::string norm_key2 =
              "blk." + std::to_string(l + 2) + ".ffn_norm.weight";
          std::string router_key2 =
              "blk." + std::to_string(l + 2) + ".ffn_gate_inp.weight";
          auto norm_it2 = engine.resident_layers.find(norm_key2);
          auto router_it2 = engine.resident_layers.find(router_key2);
          if (norm_it2 != engine.resident_layers.end() &&
              router_it2 != engine.resident_layers.end() &&
              norm_it2->second.host_ptr && router_it2->second.host_ptr) {
            rmsnorm_row(scratch.pilot_nrm.data(), last_hidden,
                        reinterpret_cast<float *>(norm_it2->second.host_ptr), D,
                        c.eps);

            dense_proj(engine, router_key2, scratch.pilot_nrm.data(),
                       scratch.pilot_logits.data(), 1, D, c.n_experts);

            apply_gating_activation(c.gating_func, scratch.pilot_logits.data(),
                                    c.n_experts);
            {
              std::string bias_key2 =
                  "blk." + std::to_string(l + 2) + ".exp_probs_b.bias";
              auto bias_it2 = engine.resident_layers.find(bias_key2);
              if (bias_it2 != engine.resident_layers.end() &&
                  bias_it2->second.host_ptr) {
                const float *bias_row2 =
                    reinterpret_cast<float *>(bias_it2->second.host_ptr);
                for (int e = 0; e < c.n_experts; ++e)
                  scratch.pilot_logits[e] += bias_row2[e];
              }
            }

            // Two-slot ring, indexed by l % 2 -- see
            // LayerScratch::pilot_pred2_slots's own comment:
            // a single shared field here would get clobbered
            // by layer l+1's own depth-2 write before layer
            // l+2 ever gets a chance to consume it.
            auto &slot2 = scratch.pilot_pred2_slots[l % 2];
            int pilot_k2 = std::min(pilot_k_for_layer(l + 2), c.topk);
            slot2.count = pilot_k2;
            slot2.layer = l + 2;
            // How many of THIS transition's l+2 picks (ranked
            // 0 = highest pilot confidence, since they're
            // extracted by repeated argmax below) actually get
            // pushed as real hints -- independent of pilot_k2
            // above, which governs how many get MEASURED. See
            // EngineConfig::prefetch_pilot_k_depth2's own
            // comment for why a separate, often-smaller cutoff
            // makes sense for l+2 than for l+1.
            int depth2_push_cutoff =
                std::min(pilot_k2_for_layer(l + 2), pilot_k2);
            for (int k = 0; k < pilot_k2; ++k) {
              int best_exp = 0;
              for (int e = 1; e < c.n_experts; ++e) {
                if (scratch.pilot_logits[e] > scratch.pilot_logits[best_exp])
                  best_exp = e;
              }
              scratch.pilot_logits[best_exp] = -2e30f;
              slot2.ids[k] = best_exp;

              // Residency classification -- ALWAYS runs
              // (measurement, drives the report below and
              // the recall-check's fetch-worthy split)
              // regardless of whether this pick is actually
              // pushed as a real hint.
              bool already_ram = ram_mgr.is_ram_resident(l + 2, best_exp);
              bool already_vram = false, vram_pinned = false;
              if (!already_ram && ram_mgr.vram_cache) {
                for (const auto &loc : ram_mgr.vram_cache->expert_vram_cache) {
                  if (loc.layer_idx == l + 2 && loc.expert_id == best_exp) {
                    already_vram = true;
                    vram_pinned = loc.pinned;
                    break;
                  }
                }
              }
              bool is_hint_worthy = !already_ram && !already_vram;
              // Only picks within the real push cutoff can
              // affect real dispatch -- gate the eviction-
              // guard entry and the real push_hint call on
              // the SAME condition, so nothing gets
              // "protected" that was never actually relied
              // on, and nothing gets a real hint pushed for
              // it beyond what depth2_push_cutoff allows.
              bool depth2_active_here =
                  depth2_enabled && k < depth2_push_cutoff;
              if (already_ram) {
                perf().n_pilot2_pred_already_ram.fetch_add(
                    1, std::memory_order_relaxed);
                if (depth2_active_here &&
                    confirmed_resident_count < PILOT_CONFIRMED_CAP)
                  confirmed_resident_this_transition
                      [confirmed_resident_count++] = {l + 2, best_exp};
              } else if (already_vram && vram_pinned) {
                perf().n_pilot2_pred_already_vram_pinned.fetch_add(
                    1, std::memory_order_relaxed);
                // Deliberately NOT added to confirmed_resident_this_transition
                // -- a pinned slot can never be an evict_lru victim, see
                // that buffer's own comment above (same reasoning as l+1).
              } else if (already_vram) {
                perf().n_pilot2_pred_already_vram_lru.fetch_add(
                    1, std::memory_order_relaxed);
                if (depth2_active_here &&
                    confirmed_resident_count < PILOT_CONFIRMED_CAP)
                  confirmed_resident_this_transition
                      [confirmed_resident_count++] = {l + 2, best_exp};
              } else {
                perf().n_pilot2_pred_needs_fetch.fetch_add(
                    1, std::memory_order_relaxed);
              }
              slot2.was_hint[k] = is_hint_worthy;
              if (depth2_active_here && is_hint_worthy) {
                // Tagged Depth2 -- see PilotSource's own
                // comment (async_prefetcher.hpp) -- so the
                // real disk-I/O cost of THIS hint is
                // attributable in the shutdown report's
                // "Prefetch real fetches (DEPTH-2 only)"
                // line, not silently folded into the
                // aggregate.
                async_io_queue.push_hint(l + 2, best_exp, PilotSource::Depth2);
                perf().n_pilot2_hints_pushed.fetch_add(
                    1, std::memory_order_relaxed);
              }
            }
          }
        }

        // Published ONCE for the whole transition (l+1's entries
        // above plus l+2's above, if depth-2 is enabled) -- see
        // this block's own opening comment for why a single
        // combined publish, not two separate calls.
        publish_pilot_confirmed_resident(confirmed_resident_this_transition,
                                         confirmed_resident_count);
      }
      PERF_END(pilot, t_prefetch);
    }
  }
}
