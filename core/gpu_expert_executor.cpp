#include "gpu_expert_executor.hpp"

#include "cuda_kernel_dispatcher.hpp"
#include "dense_dispatch.hpp"
#include "engine.hpp"
#include "expert_activation.hpp"
#include "gpu_topology.hpp"
#include "kernels_cpu.hpp"
#include "kquant_dequant.hpp"
#include "logger.hpp"
#include "perf_timer.hpp"
#include "ram_manager.hpp"
#include "scratch_buffers.hpp"
#include "tensor_trace.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

extern "C" void launch_expert_swiglu(float *, const float *, const float *, int,
                                     float, int, int *, cudaStream_t);

namespace {
constexpr int kGpuColdBatchThreshold = 4;

// 2026-10-22: demand-time active-layer uploads are async by default. The
// copy is issued on GpuTopology::active_layer_demand_stream (non-blocking)
// and expert_ffn_gpu_issue fences only its own compute stream on the slot's
// upload_event, so the decode thread can resolve/load the next expert while
// this expert's H2D is still draining. A/B on usage-survey prompt 061 showed
// 3/3 wins and byte-identical output (4.31 vs 4.24 output tok/s, +1.7%);
// the synchronous A/B control was retired 2026-10-27.
} // namespace

const ExpertLocation *expert_ffn_resolve_gpu_loc(
    const ExpertGpuContext *context, int S, const GGUFTensor &gate_meta,
    const GGUFTensor &up_meta, const GGUFTensor &down_meta,
    bool *needs_release, bool *location_from_active_layer,
    RAMSafetyManager &ram_mgr, StreamingEngine &engine) {
  PERF_SCOPE(t_resolve_gpu_loc_total);
  *needs_release = false;
  *location_from_active_layer = false;
  const ExpertLocation *location = context ? context->loc : nullptr;

  if (context && is_cpu_affinity_layer(context->layer_idx)) {
    perf().n_cpu_fallback_affinity_layer.fetch_add(1,
                                                   std::memory_order_relaxed);
    return nullptr;
  }
  if (context && context->force_cpu_offload)
    return nullptr;

  bool resident =
      location && location->device_ptr && location->layer_idx >= 0;
  if (cpu_only_mode())
    resident = false;
  bool from_active_layer = false;

  if (!resident && context && context->topo && context->slab_buffer &&
      context->slab_capacity > 0 && S >= kGpuColdBatchThreshold) {
    location = context->topo->upload_expert(
        context->layer_idx, context->expert_id, context->slab_buffer,
        context->slab_capacity);
    resident = location && location->device_ptr;
    if (resident && context->exp)
      ram_mgr.finish_sync_promotion(
          *context->exp, context->layer_idx, context->expert_id,
          &engine.layers_experts[context->layer_idx]);
  }

  const bool types_supported =
      gpu_type_supported(gate_meta.type) && gpu_type_supported(up_meta.type) &&
      gpu_type_supported(down_meta.type);
  if (!resident && context && context->topo && context->slab_buffer &&
      context->slab_capacity > 0 && S < kGpuColdBatchThreshold &&
      types_supported) {
    bool promoted_to_main = false;
    location = context->topo->find_active_expert(
        context->layer_idx, context->expert_id, [&](int layer, int expert_id) {
          promoted_to_main = true;
          if (context->exp)
            ram_mgr.finish_sync_promotion(*context->exp, layer, expert_id,
                                          &engine.layers_experts[layer]);
        });
    resident = location && location->device_ptr;
    if (resident) {
      from_active_layer = !promoted_to_main;
      LOG_DEBUG(
          "[MOE-VRAM]  ACTIVE-LAYER HIT layer=%d expert=%d hit_count=%u%s\n",
          context->layer_idx, context->expert_id, location->hit_count,
          promoted_to_main ? " (promoted to main cache)" : "");
    } else {
      // Demand-time async H2D: issue on the dedicated non-blocking stream,
      // then let expert_ffn_gpu_issue fence its compute stream on this
      // slot's event. This keeps the decode thread free to resolve/load the
      // next expert while the DMA drains. Fall back to the existing
      // synchronous upload if the async slot/stream is unavailable.
      if (S == 1) {
        location = context->topo->upload_expert_active_async(
            context->layer_idx, context->expert_id, context->slab_buffer,
            context->slab_capacity, /*speculative=*/false, /*target_gpu_id=*/-1,
            /*qwen_pilot_prediction_ns=*/-1, /*demand=*/true);
        if (location) {
          perf().n_active_layer_uploads_demand_async.fetch_add(
              1, std::memory_order_relaxed);
        }
      }
      if (!location) {
        location = context->topo->upload_expert_active(
            context->layer_idx, context->expert_id, context->slab_buffer,
            context->slab_capacity);
      }
      resident = location && location->device_ptr;
      if (resident) {
        from_active_layer = true;
        LOG_DEBUG("[MOE-VRAM]  ACTIVE-LAYER layer=%d expert=%d -> temporary "
                  "%s upload (no LRU eviction)\n",
                  context->layer_idx, context->expert_id,
                  (S == 1 && location->demand_async_pending)
                      ? "async-demand"
                      : "sync");
      } else {
        location = context->topo->upload_expert(
            context->layer_idx, context->expert_id, context->slab_buffer,
            context->slab_capacity);
        resident = location && location->device_ptr;
        from_active_layer = false;
        if (resident) {
          if (context->exp)
            ram_mgr.finish_sync_promotion(
                *context->exp, context->layer_idx, context->expert_id,
                &engine.layers_experts[context->layer_idx]);
          LOG_DEBUG("[MOE-VRAM]  COLD-GPU layer=%d expert=%d -> synchronous "
                    "H2D upload on decode-time miss\n",
                    context->layer_idx, context->expert_id);
        }
      }
    }
  }

  if (!resident) {
    if (!types_supported) {
      perf().n_cpu_fallback_type_unsupported.fetch_add(
          1, std::memory_order_relaxed);
    } else if (!context || !context->slab_buffer) {
      perf().n_cpu_fallback_no_host_buffer.fetch_add(1,
                                                     std::memory_order_relaxed);
    } else if (context->slab_capacity == 0) {
      perf().n_cpu_fallback_just_promoted.fetch_add(1,
                                                    std::memory_order_relaxed);
    } else {
      perf().n_cpu_fallback_upload_failed.fetch_add(1,
                                                    std::memory_order_relaxed);
    }
    return nullptr;
  }

  if (!types_supported) {
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
                context ? context->layer_idx : -1,
                context ? context->expert_id : -1, why.c_str());
    }
    perf().n_cpu_fallback_type_unsupported.fetch_add(1,
                                                     std::memory_order_relaxed);
    return nullptr;
  }

  if (context && context->topo) {
    if (!context->topo->hold_for_compute(
            context->layer_idx, context->expert_id, location->device_ptr,
            from_active_layer)) {
      perf().n_hold_for_compute_fail.fetch_add(1, std::memory_order_relaxed);
      // A demand async slot already owns an in-flight DMA and its own
      // temp_pin_count hold. This failure is rare (identity changed under
      // us), but leaving the slot unpublished would leak the hold forever;
      // finish that one transfer and let release_hold's demand branch
      // publish it. The extra host wait is acceptable on this error path.
      if (from_active_layer && location->demand_async_pending) {
        cudaSetDevice(location->gpu_id);
        if (location->upload_event)
          cudaEventSynchronize(location->upload_event);
        cudaGetLastError();
        context->topo->release_hold(context->layer_idx, context->expert_id,
                                    location->device_ptr, true);
      }
      LOG_WARN("[MOE-VRAM] layer=%d expert=%d: hold_for_compute failed; "
               "falling back to CPU\n",
               context->layer_idx, context->expert_id);
      return nullptr;
    }
    *needs_release = true;
    *location_from_active_layer = from_active_layer;
  }
  return location;
}

bool expert_ffn_gpu_issue(GpuExpertScratch &scratch,
                          const ExpertLocation *location, const float *input,
                          const GGUFTensor &gate_meta,
                          const GGUFTensor &up_meta, int S, int D,
                          int moe_inter, GpuTopology *topology,
                          const float *shared_device_input) {
  PERF_SCOPE(t_gpu_issue_total);
  cudaSetDevice(location->gpu_id);
  const uint8_t *gate_device = location->device_ptr;
  const uint8_t *up_device =
      location->device_ptr + (up_meta.offset - gate_meta.offset);

  if (trace_gate("VRAM-BYTE-PROBE")) {
    constexpr int windows = 16;
    constexpr size_t width = 64;
    const size_t up_offset = up_meta.offset - gate_meta.offset;
    const size_t gate_extent = std::min(up_offset, location->size);
    const size_t up_extent =
        location->size > up_offset ? location->size - up_offset : 0;
    uint8_t buffer[width];
    int gate_zero = 0, gate_count = 0, up_zero = 0, up_count = 0;
    auto sample = [&](const uint8_t *device, size_t extent, int &zero,
                      int &count) {
      for (int w = 0; w < windows && extent > 0; ++w) {
        const size_t offset =
            extent > width
                ? static_cast<size_t>(static_cast<double>(w) / windows *
                                      (extent - width))
                : 0;
        const size_t n = std::min(width, extent - offset);
        if (!n)
          continue;
        cudaMemcpy(buffer, device + offset, n, cudaMemcpyDeviceToHost);
        ++count;
        if (std::all_of(buffer, buffer + n,
                        [](uint8_t value) { return value == 0; }))
          ++zero;
      }
    };
    sample(gate_device, gate_extent, gate_zero, gate_count);
    sample(up_device, up_extent, up_zero, up_count);
    LOG_STREAM_VERBOSE() << "[VRAM-BYTE-PROBE l=" << location->layer_idx
                         << " e=" << location->expert_id
                         << "] device_ptr="
                         << static_cast<const void *>(location->device_ptr)
                         << " size=" << location->size
                         << " upload_ready=" << location->upload_ready
                         << " gate_extent=" << gate_extent
                         << " up_extent=" << up_extent
                         << " gate_zero_windows=" << gate_zero << "/"
                         << gate_count << " up_zero_windows=" << up_zero << "/"
                         << up_count
                         << " gate_type=" << gpu_type_name(gate_meta.type)
                         << " up_type=" << gpu_type_name(up_meta.type) << "\n";
  }

  if (trace_gate("VRAM-DSCALE-PROBE") &&
      (gate_meta.type == GGUFType::IQ2_XXS ||
       up_meta.type == GGUFType::IQ2_XXS)) {
    constexpr size_t block_bytes = 66;
    constexpr int samples = 8;
    const size_t up_offset = up_meta.offset - gate_meta.offset;
    const size_t gate_extent = std::min(up_offset, location->size);
    const size_t up_extent =
        location->size > up_offset ? location->size - up_offset : 0;
    auto sample_scales = [&](const uint8_t *device, size_t extent, int &zero,
                             int &count, float &minimum, float &maximum) {
      const size_t blocks = extent / block_bytes;
      if (!blocks)
        return;
      for (int i = 0; i < samples; ++i) {
        const size_t block =
            blocks > 1
                ? static_cast<size_t>(static_cast<double>(i) / (samples - 1) *
                                      (blocks - 1))
                : 0;
        uint16_t raw = 0;
        cudaMemcpy(&raw, device + block * block_bytes, sizeof(raw),
                   cudaMemcpyDeviceToHost);
        const float value = fp16_to_fp32(raw);
        ++count;
        zero += value == 0.0f;
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
      }
    };
    int gate_zero = 0, gate_count = 0, up_zero = 0, up_count = 0;
    float gate_min = 1e30f, gate_max = -1e30f;
    float up_min = 1e30f, up_max = -1e30f;
    if (gate_meta.type == GGUFType::IQ2_XXS)
      sample_scales(gate_device, gate_extent, gate_zero, gate_count, gate_min,
                    gate_max);
    if (up_meta.type == GGUFType::IQ2_XXS)
      sample_scales(up_device, up_extent, up_zero, up_count, up_min, up_max);
    LOG_STREAM_VERBOSE() << "[VRAM-DSCALE-PROBE l=" << location->layer_idx
                         << " e=" << location->expert_id
                         << "] gate_zero_d=" << gate_zero << "/" << gate_count
                         << " gate_d_range=[" << gate_min << "," << gate_max
                         << "] up_zero_d=" << up_zero << "/" << up_count
                         << " up_d_range=[" << up_min << "," << up_max
                         << "]\n";
  }

  {
    static std::atomic<bool> activation_dumped{false};
    if (!activation_dumped.load(std::memory_order_relaxed)) {
      const char *layer = std::getenv("ZEROMOE_DUMP_ACTIVATION_LAYER");
      const char *expert = std::getenv("ZEROMOE_DUMP_ACTIVATION_EXPERT");
      if (layer && expert && location->layer_idx == std::atoi(layer) &&
          location->expert_id == std::atoi(expert)) {
        bool expected = false;
        if (activation_dumped.compare_exchange_strong(expected, true)) {
          const char *configured =
              std::getenv("ZEROMOE_DUMP_ACTIVATION_OUT");
          const std::string path =
              configured ? configured : "/tmp/real_activation.bin";
          std::ofstream file(path, std::ios::binary);
          file.write(reinterpret_cast<const char *>(input),
                     static_cast<std::streamsize>(
                         static_cast<size_t>(S) * D * sizeof(float)));
          LOG_STREAM_VERBOSE()
              << "[ACTIVATION-DUMP l=" << location->layer_idx
              << " e=" << location->expert_id << "] wrote S=" << S
              << " D=" << D << " floats to " << path << "\n";
        }
      }
    }
  }

  if (trace_gate("INPUT-TRACE-GPU")) {
    float minimum = input[0], maximum = input[0];
    int zeros = 0;
    for (int i = 0; i < S * D; ++i) {
      minimum = std::min(minimum, input[i]);
      maximum = std::max(maximum, input[i]);
      zeros += input[i] == 0.0f;
    }
    LOG_STREAM_VERBOSE() << "[INPUT-TRACE-GPU l=" << location->layer_idx
                         << " e=" << location->expert_id << "] range=["
                         << minimum << ", " << maximum << "] zeros=" << zeros
                         << "/" << S * D << "\n";
  }

  if (!scratch.ensure(S, D, moe_inter, location->gpu_id)) {
    // No compute will consume this demand upload. The caller immediately
    // calls release_hold(), which publishes the slot; wait for the DMA here
    // so that publication cannot expose partially-written weights. Snapshot
    // allocation failure is rare, so this host wait is acceptable.
    if (location->demand_async_pending) {
      cudaSetDevice(location->gpu_id);
      if (location->upload_event)
        cudaEventSynchronize(location->upload_event);
      cudaGetLastError();
    }
    return false;
  }
  const float *device_input = shared_device_input;
  if (!device_input) {
    PERF_START(upload);
    cudaMemcpyAsync(
        scratch.d_x, input, static_cast<size_t>(S) * D * sizeof(float),
        cudaMemcpyHostToDevice,
        topology && location->gpu_id >= 0 &&
                static_cast<size_t>(location->gpu_id) < topology->devices.size()
            ? topology->devices[location->gpu_id].compute_stream
            : nullptr);
    PERF_END(upload, t_expert_gpu_upload);
    device_input = scratch.d_x;
  }
  // Demand async H2D: CudaKernelDispatcher and launch_expert_swiglu below
  // enqueue on the legacy default stream, so a device-side wait there is
  // sufficient: it orders the gate/up/down kernels after this expert's own
  // DMA without blocking the host. Later experts can keep resolving and
  // issuing their copies while this one drains.
  if (location->demand_async_pending) {
    if (!location->upload_event) {
      perf().n_active_layer_upload_fail.fetch_add(1, std::memory_order_relaxed);
      LOG_WARN("[MOE-VRAM] demand async slot has no upload_event; "
               "falling back to CPU for layer=%d expert=%d\n",
               location->layer_idx, location->expert_id);
      return false;
    }
    cudaError_t wait_err =
        cudaStreamWaitEvent(nullptr, location->upload_event, 0);
    if (wait_err != cudaSuccess) {
      LOG_WARN("[MOE-VRAM] cudaStreamWaitEvent(demand upload) failed: %s; "
               "falling back to CPU for layer=%d expert=%d\n",
               cudaGetErrorString(wait_err), location->layer_idx,
               location->expert_id);
      cudaGetLastError();
      cudaEventSynchronize(location->upload_event);
      cudaGetLastError();
      return false;
    }
  }
  CudaKernelDispatcher::matmul(gate_meta.type, scratch.d_gate, device_input,
                               gate_device, S, D, moe_inter);
  CudaKernelDispatcher::matmul(up_meta.type, scratch.d_up, device_input,
                               up_device, S, D, moe_inter);
  return true;
}

void expert_ffn_gpu_finish_stage1(
    GpuExpertScratch &scratch, const ExpertLocation *location,
    float *gate_buffer, float *up_buffer, const GGUFTensor &gate_meta,
    const GGUFTensor &down_meta, int layer_idx, int expert_id, int S, int D,
    int moe_inter, float clamp, GpuTopology *topology) {
  PERF_SCOPE(t_gpu_finish_total);
  cudaSetDevice(location->gpu_id);
  const uint8_t *down_device =
      location->device_ptr + (down_meta.offset - gate_meta.offset);
  bool activation_done = false;
  const bool use_gpu_activation =
      expert_gpu_activation() && !trace_gate("SILU-TRACE-GPU") &&
      !trace_gate("SWIGLU-TRACE-GPU") && !trace_gate("EXPERT-GPU") &&
      !trace_gate("INPUT-TRACE-GPU-PAIRED");
  if (use_gpu_activation) {
    if (int *bad_counter = scratch.ensure_bad_counter()) {
      PERF_START(activation);
      launch_expert_swiglu(scratch.d_prod, scratch.d_gate, scratch.d_up,
                           S * moe_inter, clamp,
                           swiglu_clamp_fused() ? 1 : 0, bad_counter, 0);
      static thread_local int sample = 0;
      if ((++sample & 511) == 0) {
        int bad[2] = {};
        cudaMemcpy(bad, bad_counter, sizeof(bad), cudaMemcpyDeviceToHost);
        if (bad[0] || bad[1]) {
          LOG_ERROR("[EXPERT-GPU] gate/up non-finite (device-side, "
                    "cumulative): %d NaN, %d Inf by layer=%d expert=%d\n",
                    bad[0], bad[1], layer_idx, expert_id);
          cudaMemset(bad_counter, 0, sizeof(bad));
        }
      }
      PERF_END(activation, t_expert_gpu_activation);
      perf().n_expert_gpu_activation.fetch_add(1, std::memory_order_relaxed);
      activation_done = true;
    }
  }

  if (!activation_done) {
    PERF_START(download);
    cudaMemcpy(gate_buffer, scratch.d_gate,
               static_cast<size_t>(S) * moe_inter * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(up_buffer, scratch.d_up,
               static_cast<size_t>(S) * moe_inter * sizeof(float),
               cudaMemcpyDeviceToHost);
    PERF_END(download, t_gpu_download);
    check_finite("gate_buf (post gate+up matmul)", layer_idx, expert_id,
                 gate_buffer, static_cast<size_t>(S) * moe_inter);
    check_finite("up_buf   (post gate+up matmul)", layer_idx, expert_id,
                 up_buffer, static_cast<size_t>(S) * moe_inter);

    float pre_min = 0.0f, pre_max = 0.0f;
    const bool trace_silu = trace_gate("SILU-TRACE-GPU");
    if (trace_silu) {
      pre_min = pre_max = gate_buffer[0];
      for (int i = 1; i < S * moe_inter; ++i) {
        pre_min = std::min(pre_min, gate_buffer[i]);
        pre_max = std::max(pre_max, gate_buffer[i]);
      }
    }
    if (trace_gate("INPUT-TRACE-GPU-PAIRED")) {
      std::vector<float> values(static_cast<size_t>(S) * D);
      cudaMemcpy(values.data(), scratch.d_x, values.size() * sizeof(float),
                 cudaMemcpyDeviceToHost);
      float minimum = values[0], maximum = values[0];
      int zeros = 0;
      for (float value : values) {
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
        zeros += value == 0.0f;
      }
      LOG_STREAM_VERBOSE() << "[INPUT-TRACE-GPU-PAIRED l=" << layer_idx
                           << " e=" << expert_id << "] d_x range=[" << minimum
                           << ", " << maximum << "] zeros=" << zeros << "/"
                           << values.size() << "\n";
    }

    PERF_START(silu);
    if (swiglu_clamp_fused()) {
      swiglu_clamp_fused_inplace(gate_buffer, up_buffer, S * moe_inter, clamp);
      PERF_END(silu, t_expert_silu_activation);
    } else {
      silu_inplace(gate_buffer, S * moe_inter);
      PERF_END(silu, t_expert_silu_activation);
      if (trace_silu) {
        int zeros = 0;
        for (int i = 0; i < S * moe_inter; ++i)
          zeros += gate_buffer[i] == 0.0f;
        LOG_STREAM_VERBOSE()
            << "[SILU-TRACE-GPU l=" << layer_idx << " e=" << expert_id
            << "] pre-SiLU gate range=[" << pre_min << ", " << pre_max
            << "] post-SiLU zeros=" << zeros << "/" << S * moe_inter << "\n";
      }
      PERF_START(multiply);
      elem_mul(gate_buffer, gate_buffer, up_buffer, S * moe_inter);
      swiglu_clamp_inplace(gate_buffer, S * moe_inter, clamp);
      PERF_END(multiply, t_expert_silu_activation);
    }
    if (trace_gate("SWIGLU-TRACE-GPU")) {
      int zeros = 0;
      for (int i = 0; i < S * moe_inter; ++i)
        zeros += gate_buffer[i] == 0.0f;
      LOG_STREAM_VERBOSE() << "[SWIGLU-TRACE-GPU l=" << layer_idx
                           << " e=" << expert_id << "] zeros=" << zeros << "/"
                           << S * moe_inter << "\n";
    }

    PERF_START(upload);
    cudaMemcpyAsync(
        scratch.d_prod, gate_buffer,
        static_cast<size_t>(S) * moe_inter * sizeof(float),
        cudaMemcpyHostToDevice,
        topology && location->gpu_id >= 0 &&
                static_cast<size_t>(location->gpu_id) <
                    topology->devices.size()
            ? topology->devices[location->gpu_id].compute_stream
            : nullptr);
    PERF_END(upload, t_expert_gpu_upload);
  }

  PERF_START(compute);
  CudaKernelDispatcher::matmul(down_meta.type, scratch.d_down, scratch.d_prod,
                               down_device, S, moe_inter, D);
  PERF_END(compute, t_gpu_compute);
}

void expert_ffn_gpu_finish_stage2(
    const ExpertLocation *location, float *output, int layer_idx, int expert_id,
    int S, int D, GpuTopology *topology, bool needs_release,
    bool location_from_active_layer, GpuExpertScratch &scratch) {
  cudaSetDevice(location->gpu_id);
  if (needs_release && topology)
    topology->release_hold(layer_idx, expert_id, location->device_ptr,
                           location_from_active_layer);
  PERF_START(download);
  cudaMemcpy(output, scratch.d_down,
             static_cast<size_t>(S) * D * sizeof(float),
             cudaMemcpyDeviceToHost);
  PERF_END(download, t_gpu_download);
  check_finite("output   (post down matmul)", layer_idx, expert_id, output,
               static_cast<size_t>(S) * D);
  perf().n_expert_gpu_calls.fetch_add(1, std::memory_order_relaxed);
}
