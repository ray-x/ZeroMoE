#include "active_layer_upload_workers.hpp"
#include "async_prefetcher.hpp"
#include "cpu_expert_executor.hpp"
#include "gpu_expert_executor.hpp"
#include "dense_dispatch.hpp"
#include "engine.hpp"
#include "expert_activation.hpp"
#include "ggml_gdn_proj_backend.hpp"
#include "ggml_glm5next_graph.hpp"
#include "ggml_hc_backend.hpp"
#include "ggml_moe_backend.hpp"
#include "ggml_qwen4exp_graph.hpp"
#include "kernels_cpu.hpp"
#include "kquant_dequant.hpp"
#include "logger.hpp"
#include "mtp_speculative.hpp"
#include "moe_pipeline.hpp"
#include "moe_router.hpp"
#include "perf_timer.hpp"
#include "prefill_expert_stream.hpp"
#include "ram_manager.hpp"
#include "scratch_buffers.hpp"
#include "tensor_trace.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
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
                                            const void *device_sub_block_out,
                                            const void *producer_event);
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

#include "moe_pipeline_internal.hpp"
#include "shared_expert_executor.hpp"

namespace {

// One reader for the debug flag: the host dump used to test mere presence, so
// ZEROMOE_GGML_SHARED_DEBUG=0 still did the checksum work and logged.
bool shared_debug_enabled() {
  static const bool on = [] {
    const char *e = std::getenv("ZEROMOE_GGML_SHARED_DEBUG");
    return e && e[0] == '1';
  }();
  return on;
}

}  // namespace

// Run Qwen4Exp's always-on shared expert against an already device-resident
// routed result.  Keeping this as a small helper lets both the ordinary
// host-route path and the all-resident device-route path use exactly the same
// graph and fallback behavior.
bool run_qwen4exp_shared_device(StreamingEngine &engine, int layer_idx,
                                       int S, int D, int moe_inter,
                                       const void *device_input,
                                       const void *routed_output,
                                       const ArchitectureConfig &c,
                                       const void **combined_output,
                                       const float *host_routed,
                                       bool shared_only,
                                       bool shared_add) {
  const auto submit_start = std::chrono::steady_clock::now();
  const auto record_submit_time = [&]() {
    if (S == 1)
      moe_pipeline_internal::add_decode_qwen_shared_submit_ms(
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - submit_start)
              .count());
  };
  static const bool debug = shared_debug_enabled();
  if (combined_output)
    *combined_output = nullptr;
  static const bool force_host_decode = [] {
    const char *e = std::getenv("ZEROMOE_GGML_SHARED_FORCE_HOST_DECODE");
    return e && e[0] == '1';
  }();
  if (force_host_decode && S == 1)
    return false;
  if (!device_input ||
      (!routed_output && !host_routed && !shared_only) ||
      (shared_add && (!host_routed || routed_output)) ||
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
    const char *e = std::getenv("ZEROMOE_GGML_SHARED_NO_EXTERNAL");
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

  auto &runner = GgmlQwenSharedRunner::instance_for_layer(layer_idx);
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

  const bool run_ok = shared_only
                          ? runner.run_shared_only(layer_idx, device_input,
                                                   combined_output)
                          : shared_add
                                ? runner.run_host_routed_add(
                                      layer_idx, host_routed, combined_output)
                          : host_routed
                                ? runner.run_host_routed(
                                      layer_idx, device_input, host_routed,
                                      combined_output)
                                : runner.run(layer_idx, device_input,
                                             routed_output, combined_output);
  if (!run_ok) {
    if (debug)
      LOG_WARN("[GGML-SHARED-DEBUG] layer=%d S=%d runner %s failed input=%p "
               "routed=%p host_routed=%p\n",
               layer_idx, S,
               shared_only ? "shared-only run"
                           : (shared_add ? "host-routed add"
                                         : (host_routed ? "host-routed run"
                                                        : "run")),
               device_input, routed_output, (const void *)host_routed);
    record_submit_time();
    return false;
  }
  static bool logged = false;
  if (!logged) {
    logged = true;
    LOG_INFO("[GGML-SHARED] Qwen4Exp routed+shared device FFN active%s%s\n",
             host_routed ? " (CPU routed result handoff)" : "",
             shared_only ? " (shared-first overlap)" : "");
  }
  record_submit_time();
  return true;
}

void run_qwen4exp_shared_host(StreamingEngine &engine, int layer_idx,
                                     int S, int D, int moe_inter,
                                     const float *normed, float *output,
                                     LayerScratch &scratch,
                                     const ArchitectureConfig &c) {
  const std::string prefix = "blk." + std::to_string(layer_idx) + ".";
  if (!engine.resident_layers.count(prefix + "ffn_gate_shexp.weight"))
    return;
  PERF_SCOPE(t_shared_expert_total);
  const int shared_inter = moe_inter * c.n_expert_shared;
  // Opt-in first-token snapshot for the A/A boundary audit.  This is kept on
  // the real host-shared fallback path because that is the path used when the
  // optional device shared graph is unavailable.  Normal execution does not
  // allocate or copy anything here.
  const char *diag_base = std::getenv("ZEROMOE_GGML_HC_COMPARE_DUMP");
  static std::atomic<bool> diag_dumped{false};
  const bool diag = diag_base && layer_idx == 0 && S == 1 &&
                    !diag_dumped.load(std::memory_order_relaxed);
  auto dump_diag = [&](const char *label, const float *data, size_t n) {
    if (!diag || !data || !n) return;
    std::ofstream out(std::string(diag_base) + ".l00.s1." + label + ".f32",
                      std::ios::binary | std::ios::trunc);
    if (out)
      out.write(reinterpret_cast<const char *>(data),
                static_cast<std::streamsize>(n * sizeof(float)));
  };
  dump_diag("host_shared_input", normed, (size_t)S * D);
  if (diag) {
    auto log_dense_path = [&](const char *label, const std::string &name) {
      auto it = engine.resident_layers.find(name);
      if (it == engine.resident_layers.end()) return;
      LOG_ERROR("[DIAG-SHARED] layer=%d %s type=%u device=%d fast=%d bytes=%zu\n",
                layer_idx, label, (unsigned)it->second.meta.type,
                it->second.device_ptr ? 1 : 0,
                it->second.gpu_fast_ptr ? 1 : 0,
                it->second.meta.size_bytes);
    };
    log_dense_path("gate", prefix + "ffn_gate_shexp.weight");
    log_dense_path("up", prefix + "ffn_up_shexp.weight");
    log_dense_path("down", prefix + "ffn_down_shexp.weight");
    log_dense_path("gate_inp", prefix + "ffn_gate_inp_shexp.weight");
  }
  dump_diag("host_shared_routed_before", output, (size_t)S * D);
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
  // The Qwen shared expert reads normed three times: gate, up, and the
  // optional scalar gate-input projection.  With host-staged GPU dense
  // dispatch, each ordinary dense_proj() uploads that same activation again.
  // Keep this opt-in until the exact-output A/B is complete; the helper is
  // Qwen-only and falls back to the old self-contained call for every
  // unsupported weight/path.
  static const bool reuse_shared_input = [] {
    const char *e = std::getenv("ZEROMOE_QWEN_SHARED_INPUT_REUSE");
    return e && (e[0] == '1' || e[0] == 'y' || e[0] == 'Y' ||
                 e[0] == 't' || e[0] == 'T');
  }();
  // Do not mix this sequence with the BF16 fast-path.  That path uses a
  // separate d_x_bf16 buffer and its event-ordered upload; interleaving it
  // with an explicit d_x preload would reuse the shared copy event while a
  // previous record could still be in flight.  Requiring every normed-input
  // projection in this small group to use the original device-resident
  // dispatch keeps the stream/event ordering identical and makes a failed
  // capability check fall back to the old path before any upload is issued.
  auto original_gpu_projection = [&](const std::string &wname) {
    auto it = engine.resident_layers.find(wname);
    return it != engine.resident_layers.end() && it->second.host_ptr &&
           it->second.device_ptr && !it->second.gpu_fast_ptr &&
           it->second.meta.type != GGUFType::BF16;
  };
  const std::string gate_name = prefix + "ffn_gate_shexp.weight";
  const std::string up_name = prefix + "ffn_up_shexp.weight";
  const std::string gate_inp_name = prefix + "ffn_gate_inp_shexp.weight";
  const bool reuse_capable =
      original_gpu_projection(gate_name) &&
      original_gpu_projection(up_name) &&
      (!engine.resident_layers.count(gate_inp_name) ||
       original_gpu_projection(gate_inp_name));
  const bool reuse_this_call = reuse_shared_input && reuse_capable;
  const bool force_cpu_shared = [] {
    const char *e = std::getenv("ZEROMOE_QWEN_SHARED_CPU_ORACLE");
    return e && (e[0] == '1' || e[0] == 'y' || e[0] == 'Y' ||
                 e[0] == 't' || e[0] == 'T');
  }();
  if (reuse_this_call) {
    static bool logged_reuse = false;
    if (!logged_reuse) {
      logged_reuse = true;
      LOG_INFO("[QWEN-SHARED] reusing one uploaded normed activation for "
               "gate/up/gate-input projections\n");
    }
  }
  bool shared_input_uploaded = false;
  auto shared_dense_proj = [&](const std::string &wname, const float *x,
                               float *y, int in, int out) {
    if (force_cpu_shared) {
      auto it = engine.resident_layers.find(wname);
      if (it != engine.resident_layers.end() && it->second.host_ptr &&
          cpu_quant_matmul(it->second.meta.type, it->second.host_ptr, x, y,
                           S, in, out))
        return;
    }
    if (reuse_this_call && x == normed && in == D) {
      if (!shared_input_uploaded)
        shared_input_uploaded =
            dense_upload_shared_input(engine, normed, S, D);
      if (shared_input_uploaded &&
          dense_proj_from_shared_input(engine, wname, y, S, D, out))
        return;
    }
    dense_proj(engine, wname, x, y, S, in, out);
  };

  shared_dense_proj(gate_name, normed,
                    scratch.gate_buf.data(), D, shared_inter);
  shared_dense_proj(up_name, normed,
                    scratch.up_buf.data(), D, shared_inter);
  dump_diag("host_shared_gate_logits", scratch.gate_buf.data(),
            (size_t)S * shared_inter);
  dump_diag("host_shared_up_logits", scratch.up_buf.data(),
            (size_t)S * shared_inter);
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
  dump_diag("host_shared_activation", scratch.gate_buf.data(),
            (size_t)S * shared_inter);
  dense_proj(engine, prefix + "ffn_down_shexp.weight", scratch.gate_buf.data(),
             scratch.down_buf.data(), S, shared_inter, D);
  dump_diag("host_shared_raw", scratch.down_buf.data(), (size_t)S * D);

  std::vector<float> gate((size_t)S, 1.0f);
  if (engine.resident_layers.count(gate_inp_name)) {
    shared_dense_proj(gate_inp_name, normed,
                      gate.data(), D, 1);
    for (float &g : gate)
      g = 1.0f / (1.0f + std::exp(-g));
  }
  if (shared_input_uploaded && engine.gpu)
    engine.gpu->dense_matmul_host_release_input();
  if (diag) {
    std::vector<float> gated((size_t)S * D);
    for (int s = 0; s < S; ++s)
      for (int d = 0; d < D; ++d)
        gated[(size_t)s * D + d] =
            gate[(size_t)s] * scratch.down_buf[(size_t)s * D + d];
    dump_diag("host_shared_gated", gated.data(), gated.size());
  }
  for (int s = 0; s < S; ++s) {
    float *dst = output + (size_t)s * D;
    const float *src = scratch.down_buf.data() + (size_t)s * D;
    const float g = gate[(size_t)s];
    for (int d = 0; d < D; ++d)
      dst[d] += g * src[d];
  }
  dump_diag("host_shared_combined", output, (size_t)S * D);
  if (diag)
    diag_dumped.store(true, std::memory_order_relaxed);
  if (std::getenv("ZEROMOE_QWEN_SHARED_REUSE_DEBUG") && S == 1) {
    auto checksum = [](const float *p, size_t n) {
      double sum = 0.0;
      double abs_sum = 0.0;
      for (size_t i = 0; i < n; ++i) {
        sum += p[i];
        abs_sum += std::abs((double)p[i]);
      }
      return std::pair<double, double>{sum, abs_sum};
    };
    const auto nrm = checksum(normed, (size_t)D);
    const auto gb = checksum(scratch.gate_buf.data(), (size_t)shared_inter);
    const auto ub = checksum(scratch.up_buf.data(), (size_t)shared_inter);
    const auto db = checksum(scratch.down_buf.data(), (size_t)D);
    LOG_WARN("[QWEN-SHARED-REUSE-DEBUG] layer=%d reuse=%d normed=(%.9g,%.9g) "
             "gate=(%.9g,%.9g) up=(%.9g,%.9g) down=(%.9g,%.9g) gate0=%.9g\n",
             layer_idx, reuse_this_call ? 1 : 0,
             nrm.first, nrm.second, gb.first, gb.second, ub.first, ub.second,
             db.first, db.second, gate.empty() ? 1.0f : gate[0]);
  }
  if (shared_debug_enabled() && S == 1) {
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
