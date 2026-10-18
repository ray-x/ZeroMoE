#include "cpu_expert_executor.hpp"

#include "dense_dispatch.hpp"
#include "expert_activation.hpp"
#include "kernels_cpu.hpp"
#include "logger.hpp"
#include "perf_timer.hpp"
#include "tensor_trace.hpp"

#include <algorithm>
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
