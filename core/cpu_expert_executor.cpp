#include "cpu_expert_executor.hpp"

#include "dense_dispatch.hpp"
#include "expert_activation.hpp"
#include "kernels_cpu.hpp"
#include "logger.hpp"
#include "perf_timer.hpp"
#include "tensor_trace.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

void expert_ffn_cpu(const float *input, float *output,
                    const uint8_t *gate_data, const GGUFTensor &gate_meta,
                    const uint8_t *up_data, const GGUFTensor &up_meta,
                    const uint8_t *down_data, const GGUFTensor &down_meta,
                    float *gate_buf, float *up_buf, int S, int D,
                    int moe_inter, float swiglu_clamp, int layer_idx,
                    int expert_id) {
  PERF_START(cexp);
  cpu_quant_matmul(gate_meta.type, gate_data, input, gate_buf, S, D, moe_inter);
  cpu_quant_matmul(up_meta.type, up_data, input, up_buf, S, D, moe_inter);

  float pre_min = 0.0f;
  float pre_max = 0.0f;
  const bool trace_silu = trace_gate("SILU-TRACE");
  if (trace_silu) {
    pre_min = pre_max = gate_buf[0];
    for (int i = 1; i < S * moe_inter; ++i) {
      pre_min = std::min(pre_min, gate_buf[i]);
      pre_max = std::max(pre_max, gate_buf[i]);
    }
  }
  if (swiglu_clamp_fused()) {
    swiglu_clamp_fused_inplace(gate_buf, up_buf, S * moe_inter,
                               swiglu_clamp);
  } else {
    silu_inplace(gate_buf, S * moe_inter);
    if (trace_silu) {
      int n_zero = 0;
      for (int i = 0; i < S * moe_inter; ++i)
        n_zero += gate_buf[i] == 0.0f;
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
      n_zero += gate_buf[i] == 0.0f;
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

bool expert_ffn_cpu_grouped_ddr(
    const float *input, float *output,
    const std::vector<MoEExpertBlock *> &experts, const float *weights,
    int D, int moe_inter, float swiglu_clamp, int layer_idx) {
  const int n = static_cast<int>(experts.size());
  if (!input || !output || !weights || n < 2 || D <= 0 || moe_inter <= 0)
    return false;
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<const uint8_t *> gate_weights((size_t)n), up_weights((size_t)n),
      down_weights((size_t)n);
  std::vector<float *> gate((size_t)n), up((size_t)n);
  std::vector<float> gate_storage((size_t)n * moe_inter),
      up_storage((size_t)n * moe_inter), down_storage((size_t)n * D),
      result((size_t)D, 0.0f);
  const GGUFType gate_type = experts[0]->gate_proj.meta.type;
  const GGUFType up_type = experts[0]->up_proj.meta.type;
  const GGUFType down_type = experts[0]->down_proj.meta.type;
  for (int i = 0; i < n; ++i) {
    MoEExpertBlock *block = experts[(size_t)i];
    if (!block || block->vram_shadow || !block->gate_proj.host_ptr ||
        !block->up_proj.host_ptr || !block->down_proj.host_ptr ||
        block->gate_proj.meta.type != gate_type ||
        block->up_proj.meta.type != up_type ||
        block->down_proj.meta.type != down_type)
      return false;
    gate_weights[(size_t)i] = block->gate_proj.host_ptr;
    up_weights[(size_t)i] = block->up_proj.host_ptr;
    down_weights[(size_t)i] = block->down_proj.host_ptr;
    gate[(size_t)i] = gate_storage.data() + (size_t)i * moe_inter;
    up[(size_t)i] = up_storage.data() + (size_t)i * moe_inter;
  }
  const auto grouped_projection = [&](float *const *dst,
                                      const uint8_t *const *w,
                                      GGUFType type) {
    switch (type) {
    case GGUFType::IQ2_XXS:
      return matmul_iq2_xxs_ggml_q8k_grouped(dst, input, w, n, 1, D,
                                              moe_inter);
    case GGUFType::IQ3_XXS:
      return matmul_iq3_xxs_ggml_q8k_grouped(dst, input, w, n, 1, D,
                                              moe_inter);
    default:
      return false;
    }
  };
  if (!grouped_projection(gate.data(), gate_weights.data(), gate_type) ||
      !grouped_projection(up.data(), up_weights.data(), up_type))
    return false;
  for (int i = 0; i < n; ++i) {
    if (swiglu_clamp_fused()) {
      swiglu_clamp_fused_inplace(gate[(size_t)i], up[(size_t)i],
                                 moe_inter, swiglu_clamp);
    } else {
      silu_inplace(gate[(size_t)i], moe_inter);
      elem_mul(gate[(size_t)i], gate[(size_t)i], up[(size_t)i], moe_inter);
      swiglu_clamp_inplace(gate[(size_t)i], moe_inter, swiglu_clamp);
    }
    if (!cpu_quant_matmul(down_type, down_weights[(size_t)i],
                          gate[(size_t)i], down_storage.data() + (size_t)i * D,
                          1, moe_inter, D))
      return false;
    const float weight = weights[(size_t)i];
    const float *expert_out = down_storage.data() + (size_t)i * D;
    for (int d = 0; d < D; ++d)
      result[(size_t)d] += weight * expert_out[d];
  }
  for (int d = 0; d < D; ++d) output[d] += result[(size_t)d];
  perf().t_cpu_expert.fetch_add(
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
      std::memory_order_relaxed);
  perf().n_expert_cpu_calls.fetch_add((uint64_t)n,
                                      std::memory_order_relaxed);
  (void)layer_idx;
  return true;
}
