#pragma once
// Shared includes, external linkage declarations and forward declarations for the native
// test harness (split from src/main_test.cpp).
#include "graph_config.hpp"
#include "qwen4exp_gdn.hpp"   // qwen4exp_gdn_delta_chunk (CPU oracle for Test 25)
#include "ggml_gdn_proj_backend.hpp"  // GgmlGdnProjRunner (Test 29)
#include "ggml_glm5next_proj_backend.hpp"  // GLM5NEXT KDA fusion (Test 34)
#include "ggml_moe_backend.hpp"  // GgmlQwenSharedRunner (Test 35)
#include "ggml_hc_backend.hpp"   // GgmlHcRunner (Test 36)
#include "kernels_cpu.hpp"
#include "ram_manager.hpp"
#include "engine.hpp"
#include "cuda_manager.hpp"
#include "kquant_dequant.hpp"
#include "engine_config.hpp"
#include "exl3_cpu.hpp"
#include "exl3_cuda.hpp"
#include "exl3_gpu.hpp"
#include "cuda_kernel_dispatcher.hpp"
#include "exl3_format.hpp"
#include "exl3_model_loader.hpp"
#if defined(MOESTREAM_USE_GGML_CUDA)
#include "exl3_ggml_ffn.hpp"
#endif

#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <array>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <unordered_map>
#include <omp.h>
#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#endif

// Defined (non-static) in src/attention_layer.cpp specifically so Test 28
// can compare qwen4exp_conv1d_silu_gpu_host's GPU kernel directly against
// this already-proven CPU reference. No header declares it (it's an
// internal helper, not part of any public API), so it's forward-declared
// here to match.
void attn_causal_conv1d_silu(const float* x, const float* w, float* out,
                             int S, int channels, int kernel, float* history);

// ------------------------------------------------------------------
// External Links to Core Functions
// ------------------------------------------------------------------
extern void matmul(float* y, const float* x, const float* W, int S, int I, int O);
extern void rmsnorm_row(float* out, const float* x, const float* w, int D, float eps);
extern void softmax_row(float* x, int size);
extern void extract_gguf_row_q4_0(const QuantizedTensor& tensor, int row_idx, float* __restrict__ out_f32);

// Encodes real ggml-valid quantized bytes from f32 input (src/ggml_dequant.cpp).
// Returns 1 if this type has a reference quantizer wired up, 0 otherwise.
extern "C" int ggml_quantize_row_ref(uint32_t type, const float* src, void* dst, int64_t n);
extern "C" int ggml_dequant_row(uint32_t type, const void* src, float* out, int n);
extern "C" void quantize_row_q8_0(const float *x, void *y, int64_t k);
extern "C" void quantize_row_q8_K(const float *x, void *y, int64_t k);
extern "C" void ggml_vec_dot_q3_K_q8_K(
    int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy,
    size_t by, int nrc);
extern "C" void ggml_vec_dot_q2_0_q8_0(
    int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy,
    size_t by, int nrc);
extern "C" void ggml_vec_dot_iq4_nl_q8_0(
    int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy,
    size_t by, int nrc);
extern "C" void ggml_cpu_init(void);

class GGUFGraphInitializer {
public:
    static bool build_model_graph(StreamingEngine& engine, int max_ctx);
};
