#include "cuda_kernel_dispatcher.hpp"

#include "dense_dispatch.hpp"
#include "logger.hpp"

#include <atomic>
#include <cstdint>

#include <cublas_v2.h>
#include <cuda_runtime.h>

extern "C" void launch_gguf_q4_0_dp4a(float *, const float *, const void *,
                                      int, int, int);
extern "C" void launch_q2_K_matmul(float *, const float *, const void *, int,
                                   int, int);
extern "C" void launch_q3_K_matmul(float *, const float *, const void *, int,
                                   int, int);
extern "C" void launch_q4_K_matmul(float *, const float *, const void *, int,
                                   int, int);
extern "C" void launch_q5_K_matmul(float *, const float *, const void *, int,
                                   int, int);
extern "C" void launch_q6_K_matmul(float *, const float *, const void *, int,
                                   int, int);
extern "C" void launch_iq2_xxs_matmul(float *, const float *, const void *, int,
                                      int, int);
extern "C" void launch_iq2_xs_matmul(float *, const float *, const void *, int,
                                     int, int);
extern "C" void launch_iq2_s_matmul(float *, const float *, const void *, int,
                                    int, int);
extern "C" void launch_iq3_xxs_matmul(float *, const float *, const void *, int,
                                      int, int);
extern "C" void launch_iq3s_matmul(float *, const float *, const void *, int,
                                   int, int);
extern "C" void launch_iq4_xs_matmul(float *, const float *, const void *, int,
                                     int, int);
extern "C" void launch_iq1s_matmul(float *, const float *, const void *, int,
                                   int, int);
extern "C" void launch_iq4_nl_matmul(float *, const float *, const void *, int,
                                     int, int);
extern "C" void launch_iq1m_matmul(float *, const float *, const void *, int,
                                   int, int);
extern "C" void launch_q8_0_matmul(float *, const float *, const void *, int,
                                   int, int);
extern "C" void launch_q5_1_matmul(float *, const float *, const void *, int,
                                   int, int);
extern "C" void launch_mxfp4_matmul(float *, const float *, const void *, int,
                                    int, int);
extern "C" bool launch_dequant_to_bf16(int, void *, const void *, int64_t);
extern "C" void launch_f32_to_bf16(void *, const float *, int64_t);

namespace {
constexpr int kMaxCudaDevices = 16;
}

bool CudaKernelDispatcher::matmul_bf16(GGUFType type, float *y,
                                        const float *x, const void *w, int S,
                                        int I, int O) {
  const int t = static_cast<int>(type);
  if (t != 7 && t != 8 && t != 12 && t != 13)
    return false;

  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 ||
      dev >= kMaxCudaDevices)
    return false;

  static cublasHandle_t handles[kMaxCudaDevices] = {};
  static void *w_bf16[kMaxCudaDevices] = {};
  static size_t w_cap[kMaxCudaDevices] = {};
  static void *x_bf16[kMaxCudaDevices] = {};
  static size_t x_cap[kMaxCudaDevices] = {};

  if (!handles[dev] && cublasCreate(&handles[dev]) != CUBLAS_STATUS_SUCCESS)
    return false;
  auto grow = [](void *&buf, size_t &cap, size_t need) {
    if (need <= cap)
      return true;
    if (buf)
      cudaFree(buf);
    buf = nullptr;
    cap = 0;
    if (cudaMalloc(&buf, need) != cudaSuccess)
      return false;
    cap = need;
    return true;
  };
  if (!grow(w_bf16[dev], w_cap[dev],
            static_cast<size_t>(O) * I * sizeof(uint16_t)) ||
      !grow(x_bf16[dev], x_cap[dev],
            static_cast<size_t>(S) * I * sizeof(uint16_t)))
    return false;
  if (!launch_dequant_to_bf16(t, w_bf16[dev], w,
                              static_cast<int64_t>(O) * I))
    return false;
  launch_f32_to_bf16(x_bf16[dev], x, static_cast<int64_t>(S) * I);

  const float alpha = 1.0f;
  const float beta = 0.0f;
  return cublasGemmEx(handles[dev], CUBLAS_OP_T, CUBLAS_OP_N, O, S, I,
                      &alpha, w_bf16[dev], CUDA_R_16BF, I, x_bf16[dev],
                      CUDA_R_16BF, I, &beta, y, CUDA_R_32F, O,
                      CUBLAS_COMPUTE_32F,
                      CUBLAS_GEMM_DEFAULT_TENSOR_OP) == CUBLAS_STATUS_SUCCESS;
}

bool CudaKernelDispatcher::matmul(GGUFType type, float *y, const float *x,
                                   const void *w, int S, int I, int O) {
  if (S > 1 && moe_expert_bf16_gemm()) {
    static std::atomic<uint64_t> ok{0};
    static std::atomic<uint64_t> fail{0};
    if (matmul_bf16(type, y, x, w, S, I, O)) {
      if ((ok.fetch_add(1) & 0xFFFFF) == 0)
        LOG_STREAM_INFO() << "[EXPERT-BF16] engaged " << ok.load()
                          << " / failed " << fail.load() << "\n";
      return true;
    }
    if ((fail.fetch_add(1) & 0xFFFFF) == 0)
      LOG_STREAM_INFO() << "[EXPERT-BF16] FAILED for type="
                        << static_cast<int>(type) << " S=" << S << " I=" << I
                        << " O=" << O << " (engaged " << ok.load() << ")\n";
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
  case GGUFType::IQ3_S:
    launch_iq3s_matmul(y, x, w, S, I, O);
    return true;
  case GGUFType::IQ1_S:
    launch_iq1s_matmul(y, x, w, S, I, O);
    return true;
  case GGUFType::IQ1_M:
    launch_iq1m_matmul(y, x, w, S, I, O);
    return true;
  case GGUFType::IQ4_NL:
    launch_iq4_nl_matmul(y, x, w, S, I, O);
    return true;
  case GGUFType::Q8_0:
    launch_q8_0_matmul(y, x, w, S, I, O);
    return true;
  case GGUFType::Q5_1:
    launch_q5_1_matmul(y, x, w, S, I, O);
    return true;
  case GGUFType::IQ4_XS:
    launch_iq4_xs_matmul(y, x, w, S, I, O);
    return true;
  case GGUFType::MXFP4:
    launch_mxfp4_matmul(y, x, w, S, I, O);
    return true;
  default:
    return false;
  }
}
