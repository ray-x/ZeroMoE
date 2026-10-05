#pragma once

// Private storage for the opt-in single-layer scheduler proof. The outer
// scheduler never allocates these tensors, so GPU scratch reuse cannot race
// with the synchronous cold graph.
#include "ggml-alloc.h"
#include "qwen4exp_stage_pool.hpp"

extern "C" int ggml_backend_sched_early_cpu_supported() __attribute__((weak));

struct Qwen4ExpEarlyReady {
  static constexpr uint64_t kMagic = 0x514541524c593133ull;
  uint64_t magic = kMagic;
  ggml_context *ctx = nullptr;
  ggml_cgraph *graph = nullptr;
  ggml_backend_t cpu = nullptr;
  ggml_backend_buffer_t scratch = nullptr, pinned_buffer = nullptr;
  ggml_backend_buffer_t device_buffer = nullptr;
  ggml_backend_buffer_t output_device_buffer = nullptr;
  ggml_backend_buffer_t tables_device_buffer = nullptr;
  ggml_backend_buffer_t prediction_buffer = nullptr;
  ggml_backend_buffer_t prediction_marker_buffer = nullptr;
  ggml_tensor *output = nullptr, *gpu_input = nullptr;
  ggml_tensor *gpu_output = nullptr, *cpu_marker = nullptr;
  ggml_tensor *gpu_tables[4] = {};
  ggml_tensor *prediction = nullptr;
  ggml_tensor *prediction_marker = nullptr;
  int prediction_layer = -1;
  size_t expert_bytes = 0;
  void *pinned = nullptr;
  float *input = nullptr, *weights = nullptr;
  void *prediction_pinned = nullptr;
  void *stage_pinned = nullptr;
  int32_t *prediction_ids = nullptr, *stage_ids = nullptr;
  unsigned long long *prediction_epoch = nullptr;
  cudaEvent_t prediction_ready = nullptr;
  Qwen4ExpStagePool *stage_source = nullptr, *stage_target = nullptr;
  ggml_backend_buffer_t stage_routes_buffer = nullptr;
  ggml_tensor *stage_gpu_ids = nullptr, *stage_gpu_mask = nullptr;
  int32_t *cpu_remap = nullptr;
  float *cpu_mask = nullptr;
  std::vector<int32_t> original_remap;
  std::vector<float> original_mask;
  uint64_t staged_routes = 0;
  int32_t *ids = nullptr, *position = nullptr;
  cudaEvent_t ready = nullptr;
  cudaStream_t producer_stream = nullptr;
  cudaStream_t output_stream = nullptr;
  cudaEvent_t output_done = nullptr;
  int device = -1, d = 0, k = 0, n_experts = 0, expected_pos = -1;
  bool completed = false;
  bool skip_all_hot = false;
  bool async_output_copy = false;
  bool resident_tables = false;
  std::vector<uint8_t> cold_mask;
  size_t reservation = 0;
  uint64_t calls = 0, early_calls = 0, gpu_pending = 0;
  uint64_t all_hot_visits = 0, all_hot_skips = 0;
  uint64_t output_uploads = 0;
  int64_t wait_us = 0, compute_us = 0;

  ~Qwen4ExpEarlyReady() {
    if (device >= 0) cudaSetDevice(device);
    if (ready) cudaEventDestroy(ready);
    if (output_stream) cudaStreamDestroy(output_stream);
    if (output_done) cudaEventDestroy(output_done);
    if (output_device_buffer) ggml_backend_buffer_free(output_device_buffer);
    if (tables_device_buffer) ggml_backend_buffer_free(tables_device_buffer);
    if (prediction_buffer) ggml_backend_buffer_free(prediction_buffer);
    if (prediction_marker_buffer) ggml_backend_buffer_free(prediction_marker_buffer);
    if (stage_routes_buffer) ggml_backend_buffer_free(stage_routes_buffer);
    if (prediction_ready) cudaEventDestroy(prediction_ready);
    if (prediction_pinned) cudaFreeHost(prediction_pinned);
    if (stage_pinned) cudaFreeHost(stage_pinned);
    if (device_buffer) ggml_backend_buffer_free(device_buffer);
    if (scratch) ggml_backend_buffer_free(scratch);
    if (pinned_buffer) ggml_backend_buffer_free(pinned_buffer);
    if (ctx) ggml_free(ctx);
    if (pinned) cudaFreeHost(pinned);
    if (reservation) MemoryGuard::release_host_bytes(reservation);
  }

  static ggml_tensor *alias(ggml_context *ctx, ggml_tensor *src) {
    auto *t = ggml_dup_tensor(ctx, src);
    std::memcpy(t->nb, src->nb, sizeof(t->nb));
    t->data = src->data;
    t->buffer = src->buffer;
    return t;
  }

  bool prepare(ggml_backend_t backend, int dev, int dim, int topk,
               ggml_tensor *gate, ggml_tensor *up, ggml_tensor *down,
               ggml_tensor *remap, ggml_tensor *mask) {
    cpu = backend; device = dev; d = dim; k = topk;
    n_experts = gate->ne[2];
    cpu_remap = static_cast<int32_t *>(remap->data);
    cpu_mask = static_cast<float *>(mask->data);
    original_remap.assign(cpu_remap, cpu_remap + n_experts);
    original_mask.assign(cpu_mask, cpu_mask + n_experts);
    expert_bytes = ggml_row_size(gate->type, gate->ne[0]) * gate->ne[1] +
                   ggml_row_size(up->type, up->ne[0]) * up->ne[1] +
                   ggml_row_size(down->type, down->ne[0]) * down->ne[1];
    const char *skip = std::getenv("MOESTREAM_QWEN_EARLY_READY_SKIP_ALL_HOT");
    skip_all_hot = skip && skip[0] == '1';
    const char *copy = std::getenv("MOESTREAM_QWEN_EARLY_READY_ASYNC_OUTPUT");
    async_output_copy = copy && copy[0] == '1';
    const char *tables = std::getenv("MOESTREAM_QWEN_EARLY_READY_RESIDENT_TABLES");
    resident_tables = tables && tables[0] == '1';
    cold_mask.resize(n_experts);
    for (int e = 0; e < n_experts; ++e)
      cold_mask[e] = static_cast<const float *>(mask->data)[e] != 0.0f;
    // Conservative admission for metadata, pins and small FFN intermediates.
    const size_t reserve = 2 * 1024 * 1024;
    if (!MemoryGuard::try_reserve_host_bytes(reserve)) return false;
    reservation = reserve;
    const size_t meta = 256 * ggml_tensor_overhead() +
                        ggml_graph_overhead_custom(256, false);
    ctx = ggml_init({meta, nullptr, true});
    if (!ctx) return false;
    const auto align = [](size_t n) { return (n + 63) & ~size_t(63); };
    const size_t id_off = align(d * sizeof(float));
    const size_t weight_off = id_off + align(k * sizeof(int32_t));
    const size_t pos_off = weight_off + align(k * sizeof(float));
    const size_t out_off = pos_off + 64;
    const size_t pin_bytes = out_off + align(d * k * sizeof(float));
    cudaSetDevice(device);
    if (cudaMallocHost(&pinned, pin_bytes) != cudaSuccess ||
        cudaEventCreateWithFlags(&ready, cudaEventDisableTiming |
                                       cudaEventBlockingSync) != cudaSuccess)
      return false;
    std::memset(pinned, 0, pin_bytes);
    auto *base = static_cast<uint8_t *>(pinned);
    input = reinterpret_cast<float *>(base);
    ids = reinterpret_cast<int32_t *>(base + id_off);
    weights = reinterpret_cast<float *>(base + weight_off);
    position = reinterpret_cast<int32_t *>(base + pos_off);
    pinned_buffer = ggml_backend_cpu_buffer_from_ptr(pinned, pin_bytes);
    if (!pinned_buffer) return false;
    auto *x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, 1, 1);
    auto *route = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, k, 1);
    auto *prob = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, k, 1);
    x->data = input; x->buffer = pinned_buffer;
    route->data = ids; route->buffer = pinned_buffer;
    prob->data = weights; prob->buffer = pinned_buffer;
    auto *cold_ids = ggml_reshape_2d(ctx,
        ggml_get_rows(ctx, alias(ctx, remap), route), k, 1);
    auto *cold_weights = ggml_mul(ctx, prob,
        ggml_get_rows(ctx, alias(ctx, mask), route));
    auto *u = ggml_mul_mat_id(ctx, alias(ctx, up), x, cold_ids);
    auto *g = ggml_mul_mat_id(ctx, alias(ctx, gate), x, cold_ids);
    auto *act = ggml_swiglu_split(ctx, g, u);
    output = ggml_mul(ctx,
        ggml_mul_mat_id(ctx, alias(ctx, down), act, cold_ids), cold_weights);
    output->data = base + out_off; output->buffer = pinned_buffer;
    graph = ggml_new_graph_custom(ctx, 256, false);
    ggml_build_forward_expand(graph, output);
    scratch = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    return scratch && meta + pin_bytes + ggml_backend_buffer_get_size(scratch)
                                     <= reservation;
  }

  bool prepare_gpu_output(ggml_backend_t backend, ggml_context *outer_ctx) {
    gpu_output = ggml_dup_tensor(outer_ctx, output);
    output_device_buffer = ggml_backend_alloc_buffer(backend, ggml_nbytes(gpu_output));
    return output_device_buffer &&
        ggml_backend_tensor_alloc(output_device_buffer, gpu_output,
            ggml_backend_buffer_get_base(output_device_buffer)) == GGML_STATUS_SUCCESS &&
        cudaStreamCreateWithFlags(&output_stream, cudaStreamNonBlocking) == cudaSuccess &&
        cudaEventCreateWithFlags(&output_done, cudaEventDisableTiming) == cudaSuccess;
  }

  bool prepare_gpu_tables(ggml_backend_t backend, ggml_context *outer_ctx,
                          ggml_tensor *remap_gpu, ggml_tensor *remap_cpu,
                          ggml_tensor *mask_gpu, ggml_tensor *mask_cpu) {
    ggml_tensor *sources[] = {remap_gpu, remap_cpu, mask_gpu, mask_cpu};
    const size_t alignment = ggml_backend_buft_get_alignment(
        ggml_backend_get_default_buffer_type(backend));
    size_t offsets[4], bytes = 0;
    for (int i = 0; i < 4; ++i) {
      offsets[i] = bytes;
      gpu_tables[i] = ggml_dup_tensor(outer_ctx, sources[i]);
      bytes += (ggml_nbytes(gpu_tables[i]) + alignment - 1) / alignment * alignment;
    }
    tables_device_buffer = ggml_backend_alloc_buffer(backend, bytes);
    if (!tables_device_buffer) return false;
    auto *base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(tables_device_buffer));
    for (int i = 0; i < 4; ++i) {
      if (ggml_backend_tensor_alloc(tables_device_buffer, gpu_tables[i],
                                   base + offsets[i]) != GGML_STATUS_SUCCESS)
        return false;
      // Immutable lookup tables are uploaded once, rather than copied as
      // user inputs before every backend split on every decoded token.
      ggml_backend_tensor_set(gpu_tables[i], sources[i]->data, 0, ggml_nbytes(sources[i]));
    }
    return true;
  }

  bool prepare_prediction(ggml_backend_t backend, ggml_context *outer_ctx,
                          ggml_tensor *router, ggml_tensor *bias, int target) {
    auto *probs = ggml_soft_max(outer_ctx, ggml_mul_mat(outer_ctx, router, gpu_input));
    auto *selection = bias ? ggml_add(outer_ctx, probs, bias) : probs;
    prediction = ggml_cont(outer_ctx, ggml_argsort_top_k(outer_ctx, selection, k));
    prediction_layer = target;
    prediction_buffer = ggml_backend_alloc_buffer(backend, ggml_nbytes(prediction));
    return prediction_buffer &&
        ggml_backend_tensor_alloc(prediction_buffer, prediction,
            ggml_backend_buffer_get_base(prediction_buffer)) == GGML_STATUS_SUCCESS;
  }

  bool prepare_prediction_readback(ggml_backend_t backend, ggml_context *outer_ctx) {
    if (cudaMallocHost(&prediction_pinned, 128 + k * sizeof(int32_t)) != cudaSuccess ||
        cudaEventCreateWithFlags(&prediction_ready, cudaEventDisableTiming |
                                cudaEventBlockingSync) != cudaSuccess) return false;
    prediction_ids = static_cast<int32_t *>(prediction_pinned);
    prediction_epoch = reinterpret_cast<unsigned long long *>(
        static_cast<uint8_t *>(prediction_pinned) + 64);
    std::memset(prediction_pinned, 0, 128 + k * sizeof(int32_t));
    auto *destination = ggml_dup_tensor(outer_ctx, prediction);
    prediction_marker_buffer = ggml_backend_alloc_buffer(backend, ggml_nbytes(destination));
    if (!prediction_marker_buffer || ggml_backend_tensor_alloc(prediction_marker_buffer,
        destination, ggml_backend_buffer_get_base(prediction_marker_buffer)) != GGML_STATUS_SUCCESS)
      return false;
    prediction_marker = ggml_cpy(outer_ctx, prediction, destination);
    prediction_marker->extra = this;
    return prediction_marker != nullptr;
  }

  bool prepare_stage_routes(ggml_backend_t backend, ggml_context *outer_ctx) {
    stage_gpu_ids = ggml_new_tensor_2d(outer_ctx, GGML_TYPE_I32, k, 1);
    stage_gpu_mask = ggml_new_tensor_3d(outer_ctx, GGML_TYPE_F32, 1, k, 1);
    stage_routes_buffer = ggml_backend_alloc_buffer(backend, 512);
    if (!stage_routes_buffer) return false;
    auto *base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(stage_routes_buffer));
    if (ggml_backend_tensor_alloc(stage_routes_buffer, stage_gpu_ids, base) != GGML_STATUS_SUCCESS ||
        ggml_backend_tensor_alloc(stage_routes_buffer, stage_gpu_mask, base + 256) != GGML_STATUS_SUCCESS)
      return false;
    // Prediction storage is otherwise unused on the target; keep the mapped
    // IDs in a separate cache line from producer prediction/epoch snapshots.
    if (cudaMallocHost(&stage_pinned, k * sizeof(int32_t)) != cudaSuccess) return false;
    stage_ids = static_cast<int32_t *>(stage_pinned);
    return true;
  }

  static cudaError_t record_external(cudaEvent_t event, cudaStream_t stream) {
    cudaStreamCaptureStatus status;
    const auto result = cudaStreamIsCapturing(stream, &status);
    if (result != cudaSuccess) return result;
    return status == cudaStreamCaptureStatusNone ? cudaEventRecord(event, stream) :
        cudaEventRecordWithFlags(event, stream, cudaEventRecordExternal);
  }

  bool enqueue(const ggml_tensor *node, cudaStream_t stream) {
    if (node->type == GGML_TYPE_I32 && stage_source) {
      GGML_ASSERT(cudaMemcpyAsync(prediction_ids, node->src[0]->data, k * sizeof(int32_t),
                    cudaMemcpyDeviceToHost, stream) == cudaSuccess);
      GGML_ASSERT(cudaMemcpyAsync(prediction_epoch, stage_source->epoch, sizeof(unsigned long long),
                    cudaMemcpyDeviceToHost, stream) == cudaSuccess);
      GGML_ASSERT(record_external(prediction_ready, stream) == cudaSuccess);
      return true;
    }
    GGML_ASSERT(node->src[0] && node->src[2] && node->src[3] && node->src[4]);
    const auto checked = [](cudaError_t status, const char *operation) {
      if (status != cudaSuccess)
        GGML_ABORT("early-ready %s: %s", operation, cudaGetErrorString(status));
    };
    if (stage_source)
      checked(moestream_qwen_stage_begin(stage_source->epoch, stream), "stage epoch");
    if (stage_target) {
      checked(moestream_qwen_stage_map(static_cast<const int *>(node->src[2]->data),
          stage_target->epoch, stage_target->slot_ids, stage_target->slot_epochs,
          static_cast<int *>(stage_gpu_ids->data), static_cast<float *>(stage_gpu_mask->data),
          k, stage_target->slots, stream), "stage route map");
      checked(cudaMemcpyAsync(stage_ids, stage_gpu_ids->data, k * sizeof(int32_t),
                               cudaMemcpyDeviceToHost, stream), "stage route readback");
    }
    checked(cudaMemcpyAsync(node->data, node->src[0]->data, d * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream), "input snapshot");
    checked(cudaMemcpyAsync(input, node->data, d * sizeof(float),
                            cudaMemcpyDeviceToHost, stream), "input readback");
    checked(cudaMemcpyAsync(ids, node->src[2]->data, k * sizeof(int32_t),
                            cudaMemcpyDeviceToHost, stream), "route readback");
    checked(cudaMemcpyAsync(weights, node->src[3]->data, k * sizeof(float),
                            cudaMemcpyDeviceToHost, stream), "weight readback");
    checked(cudaMemcpyAsync(position, node->src[4]->data, sizeof(int32_t),
                            cudaMemcpyDeviceToHost, stream), "position readback");
    cudaStreamCaptureStatus capture;
    checked(cudaStreamIsCapturing(stream, &capture), "capture query");
    // External records survive capture as host-visible graph event nodes.
    checked(capture == cudaStreamCaptureStatusNone
                ? cudaEventRecord(ready, stream)
                : cudaEventRecordWithFlags(ready, stream, cudaEventRecordExternal),
            "input event record");
    producer_stream = stream;
    return true;
  }

  void run() {
    GGML_ASSERT(!completed);
    const int64_t start = ggml_time_us();
    const cudaError_t status = cudaEventSynchronize(ready);
    if (status != cudaSuccess)
      GGML_ABORT("early-ready input event: %s", cudaGetErrorString(status));
    GGML_ASSERT(*position == expected_pos);
    if (stage_source)
      stage_source->request(calls + 1, prediction_ids, prediction_epoch, prediction_ready);
    if (stage_target) {
      std::copy(original_remap.begin(), original_remap.end(), cpu_remap);
      std::copy(original_mask.begin(), original_mask.end(), cpu_mask);
      for (int i = 0; i < k; ++i) {
        GGML_ASSERT(ids[i] >= 0 && ids[i] < n_experts);
        GGML_ASSERT(stage_ids[i] >= -1 && stage_ids[i] < stage_target->slots);
        if (stage_ids[i] >= 0) {
          GGML_ASSERT(cold_mask[ids[i]]);
          cpu_remap[ids[i]] = -1;
          cpu_mask[ids[i]] = 0.0f;
          ++staged_routes;
        }
      }
    }
    int cold_count = 0;
    for (int i = 0; i < k; ++i) {
      GGML_ASSERT(ids[i] >= 0 && ids[i] < n_experts);
      GGML_ASSERT(std::isfinite(weights[i]));
      cold_count += cpu_mask[ids[i]] != 0.0f;
    }
    const cudaError_t tail = cudaStreamQuery(producer_stream);
    GGML_ASSERT(tail == cudaSuccess || tail == cudaErrorNotReady);
    gpu_pending += tail == cudaErrorNotReady;
    const int64_t copied = ggml_time_us();
    all_hot_visits += cold_count == 0;
    if (skip_all_hot && cold_count == 0) {
      // Negative IDs produce positive zero in the original cold branch.
      // Keep the existing H2D copy and merge consuming the same output shape.
      std::memset(output->data, 0, ggml_nbytes(output));
      ++all_hot_skips;
    } else {
      GGML_ASSERT(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS);
    }
    const int64_t done = ggml_time_us();
    for (int i = 0; i < d * k; ++i)
      GGML_ASSERT(std::isfinite(static_cast<float *>(output->data)[i]));
    if (async_output_copy) {
      GGML_ASSERT(cudaMemcpyAsync(gpu_output->data, output->data, ggml_nbytes(output),
          cudaMemcpyHostToDevice, output_stream) == cudaSuccess);
      GGML_ASSERT(cudaEventRecord(output_done, output_stream) == cudaSuccess);
      GGML_ASSERT(cudaStreamWaitEvent(producer_stream, output_done, 0) == cudaSuccess);
      ++output_uploads;
    }
    wait_us += copied - start; compute_us += done - copied;
    ++calls; completed = true;
  }

  static void fallback(ggml_tensor *, const ggml_tensor *, int ith, int,
                       void *userdata) {
    // Never run a nested GGML graph inside the CPU worker team.
    GGML_ASSERT(ith == 0);
    GGML_ABORT("early-ready marker requires the scheduler bridge");
    (void)userdata;
  }

  // Host-node variant: D2H copies are already complete when this is called,
  // so it must only execute the prepared CPU graph. It performs no CUDA calls
  // and leaves the result in the owned pinned `output` buffer.
  bool compute_host_callback() {
    if (!cpu || !graph || !output ||
        ggml_backend_graph_compute(cpu, graph) != GGML_STATUS_SUCCESS)
      return false;
    const size_t n = static_cast<size_t>(ggml_nelements(output));
    const auto *values = static_cast<const float *>(output->data);
    for (size_t i = 0; i < n; ++i)
      if (!std::isfinite(values[i])) return false;
    return true;
  }

  bool compute_host_callback_reduced(float *destination, int width) {
    if (!destination || !compute_host_callback() || output->type != GGML_TYPE_F32 ||
        output->ne[0] != width || output->ne[1] <= 0)
      return false;
    const auto *values = static_cast<const float *>(output->data);
    const size_t row_stride = output->nb[1] / sizeof(float);
    for (int i = 0; i < width; ++i) {
      float sum = 0.0f;
      for (int row = 0; row < output->ne[1]; ++row)
        sum += values[static_cast<size_t>(row) * row_stride + i];
      destination[i] = sum;
    }
    return true;
  }
};

extern "C" int moestream_ggml_sched_early_cpu(
    ggml_backend_t backend, ggml_cgraph *graph,
    int64_t *wait_us, int64_t *compute_us) {
  Qwen4ExpEarlyReady *context = nullptr;
  for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
    auto *node = ggml_graph_node(graph, i);
    if (node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE ||
        node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE)
      continue;
    if (node->op != GGML_OP_MAP_CUSTOM1 || !node->extra || context) return 0;
    auto *candidate = static_cast<Qwen4ExpEarlyReady *>(node->extra);
    if (candidate->magic != Qwen4ExpEarlyReady::kMagic) return 0;
    context = candidate;
  }
  if (!context || context->cpu != backend) return 0;
  const int64_t w = context->wait_us, c = context->compute_us;
  context->run(); ++context->early_calls;
  *wait_us = context->wait_us - w; *compute_us = context->compute_us - c;
  return 1;
}
