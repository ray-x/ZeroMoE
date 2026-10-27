#include "exl3_ggml_ffn.hpp"
#include "exl3_gpu.hpp"
#include "graph_config.hpp"
#include "logger.hpp"

#if defined(ZEROMOE_USE_GGML_CUDA)

#include "exl3_cuda.hpp"
#include "exl3_cpu.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <stdexcept>

namespace {

void check(cudaError_t status, const char *operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
}

bool valid_layout(const Exl3PackedMatrixHeader (&headers)[3], int rows) {
  const int hidden = static_cast<int>(headers[0].input_features);
  const int intermediate = static_cast<int>(headers[0].output_features);
  return rows > 0 && hidden > 0 && intermediate > 0 &&
         headers[1].input_features == headers[0].input_features &&
         headers[1].output_features == headers[0].output_features &&
         headers[2].input_features == headers[0].output_features &&
         headers[2].output_features == headers[0].input_features;
}

} // namespace

bool exl3_ggml_build_routed_payload(
    const std::vector<std::vector<uint8_t>> &matrices,
    std::vector<uint8_t> &payload, Exl3GgmlRoutedPayloadLayout &layout,
    std::string &error) {
  payload.clear();
  layout = {};
  if (matrices.empty() || matrices.size() % 3 != 0) {
    error = "EXL3 routed payload needs three matrices per expert";
    return false;
  }
  layout.expert_count = static_cast<int>(matrices.size() / 3);
  size_t slot_size = 0;
  std::vector<Exl3PackedMatrixHeader> headers(matrices.size());
  for (size_t index = 0; index < matrices.size(); ++index) {
    if (matrices[index].size() < sizeof(Exl3PackedMatrixHeader)) {
      error = "EXL3 routed payload matrix is truncated";
      return false;
    }
    std::memcpy(&headers[index], matrices[index].data(),
                sizeof(Exl3PackedMatrixHeader));
    if (!exl3_valid_packed_header(headers[index], matrices[index].size())) {
      error = "EXL3 routed payload matrix header is invalid";
      return false;
    }
    slot_size = std::max(slot_size, (matrices[index].size() + 63) & ~size_t(63));
  }
  layout.hidden = static_cast<int>(headers[0].input_features);
  layout.intermediate = static_cast<int>(headers[0].output_features);
  for (int projection = 0; projection < 3; ++projection) {
    const auto &header = headers[projection];
    if ((projection < 2 &&
         (header.input_features != layout.hidden ||
          header.output_features != layout.intermediate)) ||
        (projection == 2 &&
         (header.input_features != layout.intermediate ||
          header.output_features != layout.hidden))) {
      error = "EXL3 routed payload dimensions do not match";
      return false;
    }
    layout.bits[projection] = static_cast<int>(header.bits);
  }
  for (int expert = 0; expert < layout.expert_count; ++expert) {
    for (int projection = 0; projection < 3; ++projection) {
      const auto &header = headers[expert * 3 + projection];
      if ((projection < 2 &&
           (header.input_features != layout.hidden ||
            header.output_features != layout.intermediate)) ||
          (projection == 2 &&
           (header.input_features != layout.intermediate ||
            header.output_features != layout.hidden))) {
        error = "EXL3 routed payload expert dimensions do not match";
        return false;
      }
      if (static_cast<int>(header.bits) != layout.bits[projection])
        layout.bits[projection] = 0;
    }
  }
  layout.expert_stride = slot_size * 3;
  layout.projection_offsets[0] = 0;
  layout.projection_offsets[1] = slot_size;
  layout.projection_offsets[2] = slot_size * 2;
  payload.resize(static_cast<size_t>(layout.expert_count) *
                 layout.expert_stride);
  for (int expert = 0; expert < layout.expert_count; ++expert)
    for (int projection = 0; projection < 3; ++projection)
      std::memcpy(payload.data() + expert * layout.expert_stride +
                      layout.projection_offsets[projection],
                  matrices[expert * 3 + projection].data(),
                  matrices[expert * 3 + projection].size());
  return true;
}

Exl3GgmlMatmulState::~Exl3GgmlMatmulState() {
  if (partials)
    cudaFree(partials);
}

bool Exl3GgmlMatmulState::initialize(
    const Exl3PackedMatrixHeader &matrix_header, int matrix_rows, int device) {
  if (!exl3_valid_packed_header(matrix_header, matrix_header.total_size) ||
      matrix_rows <= 0 || device < 0)
    return false;
  header = matrix_header;
  rows = matrix_rows;
  custom.compute = compute;
  custom.user_data = this;
  custom.graph_capture_safe = true;
  cudaSetDevice(device);
  const size_t partial_bytes = static_cast<size_t>(header.output_features) *
                               exl3_cuda_key_splits(header.input_features) *
                               matrix_rows * sizeof(float);
  return cudaMalloc(&partials, partial_bytes) == cudaSuccess;
}

void Exl3GgmlMatmulState::compute(ggml_backend_t cuda_backend,
                                  ggml_tensor *node, void *user_data) {
  auto *state = static_cast<Exl3GgmlMatmulState *>(user_data);
  state->stream = exl3_ggml_cuda_stream(cuda_backend);
  if (!node || !node->src[0] || !node->src[1])
    throw std::runtime_error("invalid EXL3 GGML matmul node");
  const auto *base = static_cast<const uint8_t *>(node->src[1]->data);
  const Exl3CudaMatrixView matrix{
      reinterpret_cast<const uint16_t *>(base + state->header.trellis_offset),
      reinterpret_cast<const uint16_t *>(base + state->header.suh_offset),
      reinterpret_cast<const uint16_t *>(base + state->header.svh_offset),
      (state->header.flags & kExl3PackedHasBias)
          ? reinterpret_cast<const uint16_t *>(base + state->header.bias_offset)
          : nullptr,
      static_cast<int>(state->header.input_features),
      static_cast<int>(state->header.output_features),
      static_cast<int>(state->header.bits)};
  check(launch_exl3_mul1_matmul(
            matrix, static_cast<const float *>(node->src[0]->data),
            static_cast<float *>(node->data), state->rows, state->partials,
            state->stream),
        "launch EXL3 GGML matmul");
}

ggml_tensor *exl3_ggml_map_matmul(ggml_context *ctx, ggml_tensor *input,
                                  ggml_tensor *opaque,
                                  Exl3GgmlMatmulState *state) {
  if (!ctx || !input || !opaque || !state || state->rows <= 0)
    return nullptr;
  auto *node = exl3_ggml_map_custom2(ctx, input, opaque, &state->custom);
  if (!node)
    return nullptr;
  node->ne[0] = state->header.output_features;
  node->ne[1] = input->ne[1];
  node->nb[1] = static_cast<size_t>(node->ne[0]) * sizeof(float);
  return node;
}

Exl3GgmlFfnState::~Exl3GgmlFfnState() {
  if (partials)
    cudaFree(partials);
  if (activation)
    cudaFree(activation);
}

bool Exl3GgmlFfnState::initialize(
    const Exl3PackedMatrixHeader (&matrix_headers)[3],
    const size_t (&matrix_offsets)[3], int matrix_rows, int device) {
  if (!valid_layout(matrix_headers, matrix_rows) || device < 0)
    return false;
  headers[0] = matrix_headers[0];
  headers[1] = matrix_headers[1];
  headers[2] = matrix_headers[2];
  offsets[0] = matrix_offsets[0];
  offsets[1] = matrix_offsets[1];
  offsets[2] = matrix_offsets[2];
  rows = matrix_rows;
  custom.compute = compute;
  custom.user_data = this;
  custom.graph_capture_safe = true;
  const int hidden = static_cast<int>(headers[0].input_features);
  const int intermediate = static_cast<int>(headers[0].output_features);
  const size_t gate_partials = static_cast<size_t>(intermediate) *
                               exl3_cuda_key_splits(hidden) * rows;
  const size_t down_partials = static_cast<size_t>(hidden) *
                               exl3_cuda_key_splits(intermediate) * rows;
  cudaSetDevice(device);
  check(cudaMalloc(&activation,
                   static_cast<size_t>(intermediate) * rows * sizeof(float)),
        "allocate EXL3 GGML FFN activation");
  check(cudaMalloc(&partials,
                   std::max(2 * gate_partials, down_partials) *
                       sizeof(float)),
        "allocate EXL3 GGML FFN partials");
  return true;
}

void Exl3GgmlFfnState::enqueue(const ggml_tensor *input,
                                const ggml_tensor *opaque,
                                ggml_tensor *output) {
  if (!input || !opaque || !output || input->ne[1] != rows)
    throw std::runtime_error("invalid EXL3 GGML FFN tensor shape");
  const auto *base = static_cast<const uint8_t *>(opaque->data);
  const auto view = [&](int index) {
    const auto &header = headers[index];
    return Exl3CudaMatrixView{
        reinterpret_cast<const uint16_t *>(base + offsets[index] +
                                           header.trellis_offset),
        reinterpret_cast<const uint16_t *>(base + offsets[index] +
                                           header.suh_offset),
        reinterpret_cast<const uint16_t *>(base + offsets[index] +
                                           header.svh_offset),
        (header.flags & kExl3PackedHasBias)
            ? reinterpret_cast<const uint16_t *>(base + offsets[index] +
                                                 header.bias_offset)
            : nullptr,
        static_cast<int>(header.input_features),
        static_cast<int>(header.output_features),
        static_cast<int>(header.bits)};
  };
  const int hidden = static_cast<int>(headers[0].input_features);
  const int intermediate = static_cast<int>(headers[0].output_features);
  const size_t gate_partials = static_cast<size_t>(intermediate) *
                               exl3_cuda_key_splits(hidden) * rows;
  check(cudaMemsetAsync(output->data,
                        0, static_cast<size_t>(hidden) * rows * sizeof(float),
                        stream),
        "clear EXL3 GGML FFN output");
  check(launch_exl3_mul1_gate_up(
            view(0), view(1), static_cast<const float *>(input->data),
            activation, rows, partials, partials + gate_partials, stream),
        "launch EXL3 GGML FFN gate/up");
  check(launch_exl3_mul1_matmul_accumulate(
            view(2), activation, static_cast<float *>(output->data), 1.0f,
            rows, partials, stream),
        "launch EXL3 GGML FFN down");
}

void Exl3GgmlFfnState::compute(ggml_backend_t cuda_backend, ggml_tensor *node,
                                void *user_data) {
  auto *state = static_cast<Exl3GgmlFfnState *>(user_data);
  state->stream = exl3_ggml_cuda_stream(cuda_backend);
  state->enqueue(node->src[0], node->src[1], node);
}

ggml_tensor *exl3_ggml_map_fused_ffn(ggml_context *ctx, ggml_tensor *input,
                                     ggml_tensor *opaque,
                                     Exl3GgmlFfnState *state) {
  if (!ctx || !input || !opaque || !state || state->rows <= 0)
    return nullptr;
  auto *node = exl3_ggml_map_custom2(ctx, input, opaque, &state->custom);
  if (!node)
    return nullptr;
  node->ne[0] = state->headers[0].input_features;
  node->ne[1] = input->ne[1];
  node->nb[1] = static_cast<size_t>(node->ne[0]) * sizeof(float);
  return node;
}

Exl3GgmlRoutedFfnState::~Exl3GgmlRoutedFfnState() {
  if (partials)
    cudaFree(partials);
  if (activation)
    cudaFree(activation);
}

bool Exl3GgmlRoutedFfnState::initialize(
    const std::vector<Exl3PackedMatrixHeader> &matrix_headers,
    const std::vector<size_t> &matrix_offsets, int matrix_rows, int device) {
  if (matrix_headers.empty() || matrix_headers.size() != matrix_offsets.size() ||
      matrix_headers.size() % 3 != 0 || matrix_rows != 1 || device < 0)
    return false;
  const int hidden = static_cast<int>(matrix_headers[0].input_features);
  const int intermediate = static_cast<int>(matrix_headers[0].output_features);
  if (hidden <= 0 || intermediate <= 0)
    return false;
  for (size_t expert = 0; expert < matrix_headers.size() / 3; ++expert) {
    const auto &gate = matrix_headers[expert * 3];
    const auto &up = matrix_headers[expert * 3 + 1];
    const auto &down = matrix_headers[expert * 3 + 2];
    if (gate.input_features != hidden || gate.output_features != intermediate ||
        up.input_features != hidden || up.output_features != intermediate ||
        down.input_features != intermediate || down.output_features != hidden)
      return false;
  }
  headers = matrix_headers;
  offsets = matrix_offsets;
  rows = matrix_rows;
  this->hidden = hidden;
  this->intermediate = intermediate;
  expert_count = static_cast<int>(matrix_headers.size() / 3);
  topk = 0;
  custom.compute = compute;
  custom.user_data = this;
  custom.graph_capture_safe = true;
  const size_t gate_partials = static_cast<size_t>(intermediate) *
                               exl3_cuda_key_splits(hidden);
  const size_t down_partials = static_cast<size_t>(hidden) *
                               exl3_cuda_key_splits(intermediate);
  cudaSetDevice(device);
  if (cudaMalloc(&activation, static_cast<size_t>(intermediate) *
                                  sizeof(float)) != cudaSuccess)
    return false;
  if (cudaMalloc(&partials,
                 std::max(2 * gate_partials, down_partials) * sizeof(float)) !=
      cudaSuccess) {
    cudaFree(activation);
    activation = nullptr;
    return false;
  }
  return true;
}

void Exl3GgmlRoutedFfnState::enqueue(const ggml_tensor *input,
                                     const ggml_tensor *opaque,
                                     const ggml_tensor *route_weights,
                                     ggml_tensor *output) {
  if (!input || !opaque || !route_weights || !output || input->ne[1] != rows ||
      route_weights->type != GGML_TYPE_F32 || route_weights->ne[0] !=
          static_cast<int64_t>(headers.size() / 3) || route_weights->ne[1] != 1)
    throw std::runtime_error("invalid EXL3 GGML routed FFN tensor shape");
  const auto *base = static_cast<const uint8_t *>(opaque->data);
  const int hidden = static_cast<int>(headers[0].input_features);
  const int intermediate = static_cast<int>(headers[0].output_features);
  const size_t gate_partials = static_cast<size_t>(intermediate) *
                               exl3_cuda_key_splits(hidden);
  check(cudaMemsetAsync(output->data, 0,
                        static_cast<size_t>(hidden) * sizeof(float), stream),
        "clear EXL3 GGML routed FFN output");
  for (size_t expert = 0; expert < headers.size() / 3; ++expert) {
    const auto view = [&](int projection) {
      const size_t index = expert * 3 + projection;
      const auto &header = headers[index];
      return Exl3CudaMatrixView{
          reinterpret_cast<const uint16_t *>(base + offsets[index] +
                                             header.trellis_offset),
          reinterpret_cast<const uint16_t *>(base + offsets[index] +
                                             header.suh_offset),
          reinterpret_cast<const uint16_t *>(base + offsets[index] +
                                             header.svh_offset),
          (header.flags & kExl3PackedHasBias)
              ? reinterpret_cast<const uint16_t *>(base + offsets[index] +
                                                   header.bias_offset)
              : nullptr,
          static_cast<int>(header.input_features),
          static_cast<int>(header.output_features),
          static_cast<int>(header.bits)};
    };
    check(launch_exl3_mul1_gate_up(
              view(0), view(1), static_cast<const float *>(input->data),
              activation, 1, partials, partials + gate_partials, stream),
          "launch EXL3 GGML routed FFN gate/up");
    check(launch_exl3_mul1_matmul_accumulate_device_weight(
              view(2), activation, static_cast<float *>(output->data),
              static_cast<const float *>(route_weights->data) + expert, 1,
              partials, stream),
          "launch EXL3 GGML routed FFN down");
  }
}

void Exl3GgmlRoutedFfnState::compute(ggml_backend_t cuda_backend,
                                     ggml_tensor *node, void *user_data) {
  auto *state = static_cast<Exl3GgmlRoutedFfnState *>(user_data);
  state->stream = exl3_ggml_cuda_stream(cuda_backend);
  state->enqueue(node->src[0], node->src[1], node->src[2], node);
}

ggml_tensor *exl3_ggml_map_routed_ffn(ggml_context *ctx, ggml_tensor *input,
                                      ggml_tensor *opaque,
                                      ggml_tensor *route_weights,
                                      Exl3GgmlRoutedFfnState *state) {
  if (!ctx || !input || !opaque || !route_weights || !state || state->rows <= 0)
    return nullptr;
  auto *node = exl3_ggml_map_custom3(ctx, input, opaque, route_weights,
                                     &state->custom);
  if (!node)
    return nullptr;
  node->ne[0] = state->headers[0].input_features;
  node->ne[1] = input->ne[1];
  node->nb[1] = static_cast<size_t>(node->ne[0]) * sizeof(float);
  return node;
}

Exl3GgmlCachedRoutedFfnState::~Exl3GgmlCachedRoutedFfnState() {
  if (route_readback_calls != 0 &&
      Logger::instance().would_log(LogLevel::DEBUG)) {
    LOG_DEBUG("[EXL3-GGML-ROUTE] layer=%d readbacks=%llu bytes=%zu "
             "sync_ms=%.3f avg_us=%.2f resolve_ms=%.3f "
             "enqueue_ms=%.3f row_wait_ms=%.3f\n",
             model_layer,
             static_cast<unsigned long long>(route_readback_calls),
             route_readback_bytes, route_readback_ms,
             route_readback_ms * 1000.0 /
                 static_cast<double>(route_readback_calls),
             expert_resolution_ms, ffn_enqueue_ms, row_batch_wait_ms);
  }
  if (route_host_storage)
    cudaFreeHost(route_host_storage);
}

bool Exl3GgmlCachedRoutedFfnState::initialize(
    Exl3GpuExecutor *executor,
    std::vector<std::string> matrix_expert_prefixes_by_id,
    int matrix_rows, int matrix_topk, int matrix_hidden,
    int matrix_intermediate) {
  if (!executor || matrix_expert_prefixes_by_id.empty() ||
      matrix_rows <= 0 || matrix_topk <= 0 ||
      matrix_hidden <= 0 || matrix_intermediate <= 0 ||
      matrix_hidden % 128 != 0 || matrix_intermediate % 128 != 0)
    return false;
  gpu_executor = executor;
  expert_prefixes_by_id = std::move(matrix_expert_prefixes_by_id);
  rows = matrix_rows;
  topk = matrix_topk;
  hidden = matrix_hidden;
  intermediate = matrix_intermediate;
  routes_from_host = false;
  route_tensors_on_device = false;
  const size_t route_count = static_cast<size_t>(rows) * topk;
  if (route_count >
      SIZE_MAX / (sizeof(int32_t) + sizeof(float)))
    return false;
  if (route_host_storage) {
    cudaFreeHost(route_host_storage);
    route_host_storage = nullptr;
    route_ids_host = nullptr;
    route_weights_host = nullptr;
  }
  const size_t route_storage_bytes =
      route_count * (sizeof(int32_t) + sizeof(float));
  if (cudaHostAlloc(&route_host_storage, route_storage_bytes,
                    cudaHostAllocPortable) != cudaSuccess)
    return false;
  route_ids_host = static_cast<int32_t *>(route_host_storage);
  route_weights_host = reinterpret_cast<float *>(
      static_cast<uint8_t *>(route_host_storage) +
      route_count * sizeof(int32_t));
  std::fill_n(route_ids_host, route_count, 0);
  std::fill_n(route_weights_host, route_count, 0.0f);
  external_experts.resize(static_cast<size_t>(topk));
  custom.compute = compute;
  custom.user_data = this;
  return true;
}

bool Exl3GgmlCachedRoutedFfnState::set_host_routes(
    const int32_t *route_ids, const float *route_weights) {
  if (!route_ids || !route_weights || rows <= 0 || topk <= 0)
    return false;
  const size_t route_count = static_cast<size_t>(rows) * topk;
  std::copy_n(route_ids, route_count, route_ids_host);
  std::copy_n(route_weights, route_count, route_weights_host);
  routes_from_host = true;
  route_tensors_on_device = false;
  return true;
}

void Exl3GgmlCachedRoutedFfnState::set_matrix_loader(
    std::function<std::vector<uint8_t>(const std::string &)> loader) {
  matrix_loader = std::move(loader);
}

void Exl3GgmlCachedRoutedFfnState::set_external_expert_resolver(
    ExternalExpertResolver resolver) {
  external_expert_resolver = std::move(resolver);
}

void Exl3GgmlCachedRoutedFfnState::compute(ggml_backend_t cuda_backend,
                                          ggml_tensor *node,
                                          void *user_data) {
  auto *state = static_cast<Exl3GgmlCachedRoutedFfnState *>(user_data);
  if (!state || !node || !node->src[0] || !node->src[2] || !node->data ||
      !node->src[0]->data || !node->src[2]->data ||
      node->src[0]->type != GGML_TYPE_F32 || node->type != GGML_TYPE_F32 ||
      node->src[0]->ne[0] != state->hidden ||
      node->src[0]->ne[1] != state->rows ||
      node->ne[0] != state->hidden || node->ne[1] != state->rows)
    throw std::runtime_error("invalid EXL3 cached GGML routed FFN node");
  const size_t route_count = static_cast<size_t>(state->rows) * state->topk;
  const size_t ids_bytes = route_count * sizeof(int32_t);
  const size_t weights_bytes = route_count * sizeof(float);
  const cudaStream_t stream = exl3_ggml_cuda_stream(cuda_backend);
  const float *device_route_weights = nullptr;
  if (!stream)
    throw std::runtime_error("EXL3 cached GGML op has no CUDA stream");
  if (state->routes_from_host) {
    if (!node->src[1] || !node->src[1]->data ||
        node->src[1]->type != GGML_TYPE_I32 ||
        node->src[1]->ne[0] != state->topk ||
        node->src[1]->ne[1] != state->rows ||
        node->src[2]->type != GGML_TYPE_F32 ||
        node->src[2]->ne[0] != state->topk ||
        node->src[2]->ne[1] != state->rows)
      throw std::runtime_error("invalid EXL3 host-route tensors");
    if (state->external_expert_resolver) {
      check(cudaStreamSynchronize(stream),
            "complete prior EXL3 routed work before resolving host routes");
      (void)state->gpu_executor->stats();
    }
  } else if (state->route_tensors_on_device) {
    if (!node->src[1] || !node->src[1]->data ||
        node->src[1]->type != GGML_TYPE_I32 ||
        node->src[1]->ne[0] != state->topk ||
        node->src[1]->ne[1] != state->rows ||
        node->src[2]->type != GGML_TYPE_F32 ||
        node->src[2]->ne[0] != state->topk ||
        node->src[2]->ne[1] != state->rows)
      throw std::runtime_error("invalid EXL3 device-route tensors");
    device_route_weights = static_cast<const float *>(node->src[2]->data);
    const bool measure_route_readback =
        Logger::instance().would_log(LogLevel::DEBUG);
    const auto route_readback_start =
        measure_route_readback ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};
    check(cudaMemcpyAsync(state->route_ids_host, node->src[1]->data, ids_bytes,
                          cudaMemcpyDeviceToHost, stream),
          "read EXL3 device route IDs");
    if (state->verify_cpu_reference)
      check(cudaMemcpyAsync(state->route_weights_host, node->src[2]->data,
                            weights_bytes, cudaMemcpyDeviceToHost, stream),
            "read EXL3 device route weights for CPU parity");
    check(cudaStreamSynchronize(stream),
          "complete EXL3 device route readback");
    if (measure_route_readback) {
      state->route_readback_ms +=
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - route_readback_start)
              .count();
      ++state->route_readback_calls;
      state->route_readback_bytes +=
          ids_bytes + (state->verify_cpu_reference ? weights_bytes : 0);
    }
    if (state->external_expert_resolver)
      (void)state->gpu_executor->stats();
  } else {
    const size_t required_bytes = ids_bytes + weights_bytes;
    if (node->src[2]->type != GGML_TYPE_I8 ||
        ggml_nbytes(node->src[2]) < required_bytes)
      throw std::runtime_error("invalid EXL3 cached GGML route table");
    const auto *route_bytes = static_cast<const uint8_t *>(node->src[2]->data);
    device_route_weights = reinterpret_cast<const float *>(
        route_bytes + ids_bytes);
    const bool measure_route_readback =
        Logger::instance().would_log(LogLevel::DEBUG);
    const auto route_readback_start =
        measure_route_readback ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};
    check(cudaMemcpyAsync(state->route_ids_host, route_bytes, ids_bytes,
                          cudaMemcpyDeviceToHost, stream),
          "read cached GGML expert IDs");
    if (state->verify_cpu_reference)
      check(cudaMemcpyAsync(state->route_weights_host,
                            route_bytes + ids_bytes, weights_bytes,
                            cudaMemcpyDeviceToHost, stream),
            "read cached GGML route weights for CPU parity");
    check(cudaStreamSynchronize(stream),
          "complete cached GGML route readback");
    if (measure_route_readback) {
      state->route_readback_ms +=
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - route_readback_start)
              .count();
      ++state->route_readback_calls;
      state->route_readback_bytes +=
          ids_bytes + (state->verify_cpu_reference ? weights_bytes : 0);
    }
    if (state->external_expert_resolver)
      (void)state->gpu_executor->stats();
  }
  if (state->external_expert_resolver) {
    if (!state->matrix_loader)
      throw std::runtime_error("EXL3 external GPU cache needs a matrix loader");
    const bool measure_external_timing =
        Logger::instance().would_log(LogLevel::DEBUG);
    const size_t routes_per_row = static_cast<size_t>(state->topk);
    const size_t maximum_batch_rows =
        state->external_matrix_slot_capacity >= routes_per_row * 3
            ? std::max<size_t>(
                  1, state->external_matrix_slot_capacity /
                         (routes_per_row * 3))
            : 1;
    for (int first_row = 0; first_row < state->rows;
         first_row += static_cast<int>(maximum_batch_rows)) {
      const int batch_rows = std::min<int>(
          static_cast<int>(maximum_batch_rows), state->rows - first_row);
      const size_t batch_route_count =
          static_cast<size_t>(batch_rows) * routes_per_row;
      const bool use_cpu_reference_output =
          zeromoe::graphcfg::present("ZEROMOE_EXL3_GGML_CPU_REFERENCE_OUTPUT");
      if (use_cpu_reference_output && !state->verify_cpu_reference)
        throw std::runtime_error(
            "EXL3 CPU reference output requires CPU parity checking");
      std::fill(state->external_experts.begin(), state->external_experts.end(),
                Exl3GpuExternalExpert{});
      const size_t first_route = static_cast<size_t>(first_row) * routes_per_row;
      const auto resolve_start = measure_external_timing
                                     ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};
      for (size_t route = 0; route < batch_route_count; ++route) {
        const size_t route_index = first_route + route;
        const int32_t expert_id = state->route_ids_host[route_index];
        if (expert_id < 0 ||
            static_cast<size_t>(expert_id) >=
                state->expert_prefixes_by_id.size())
          throw std::runtime_error("EXL3 routed expert ID out of range");
        state->external_experts[route] = state->external_expert_resolver(
            state->expert_prefixes_by_id[static_cast<size_t>(expert_id)],
            expert_id, state->hidden, state->intermediate,
            state->matrix_loader);
      }
      if (measure_external_timing)
        state->expert_resolution_ms +=
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - resolve_start)
                .count();
      const float *row_device_weights =
          state->routes_from_host
              ? static_cast<const float *>(node->src[2]->data) + first_route
              : device_route_weights + first_route;
      const float *row_input =
          static_cast<const float *>(node->src[0]->data) +
          static_cast<size_t>(first_row) * state->hidden;
      float *row_output = static_cast<float *>(node->data) +
                          static_cast<size_t>(first_row) * state->hidden;
      std::vector<float> host_input;
      if (state->verify_cpu_reference) {
        host_input.resize(static_cast<size_t>(batch_rows) * state->hidden);
        check(cudaMemcpy(host_input.data(), row_input,
                         host_input.size() * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "read EXL3 bridge input for CPU parity");
      }
      const auto enqueue_start = measure_external_timing
                                     ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};
      state->gpu_executor->routed_ffn_device_external(
          state->external_experts.data(),
          state->route_weights_host + first_route, row_device_weights,
          row_input, row_output, batch_rows, state->topk, state->hidden,
          state->intermediate, stream);
      if (measure_external_timing &&
          Logger::instance().would_log(LogLevel::DEBUG))
        LOG_DEBUG("[EXL3-GRAPH-TRACE] callback enqueued layer=%d row=%d\n",
                     state->model_layer, first_row);
      if (measure_external_timing)
        state->ffn_enqueue_ms +=
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - enqueue_start)
                .count();
      std::fill(state->external_experts.begin(), state->external_experts.end(),
                Exl3GpuExternalExpert{});
      if (first_row + batch_rows < state->rows) {
        const auto row_wait_start = measure_external_timing
                                        ? std::chrono::steady_clock::now()
                                        : std::chrono::steady_clock::time_point{};
        check(cudaStreamSynchronize(stream),
              "complete EXL3 row batch before reusing GPU expert slots");
        (void)state->gpu_executor->stats();
        if (measure_external_timing)
          state->row_batch_wait_ms +=
              std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - row_wait_start)
                  .count();
      }
      if (state->verify_cpu_reference) {
        if (Logger::instance().would_log(LogLevel::DEBUG))
          LOG_DEBUG("[EXL3-GRAPH-TRACE] parity wait layer=%d row=%d\n",
                       state->model_layer, first_row);
        check(cudaStreamSynchronize(stream),
              "complete EXL3 routed FFN before CPU parity readback");
        (void)state->gpu_executor->stats();
        if (Logger::instance().would_log(LogLevel::DEBUG))
          LOG_DEBUG("[EXL3-GRAPH-TRACE] parity stream complete layer=%d "
                       "row=%d\n",
                       state->model_layer, first_row);
        std::vector<float> actual(static_cast<size_t>(batch_rows) *
                                  state->hidden);
        check(cudaMemcpy(actual.data(), row_output,
                         actual.size() * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "read EXL3 bridge output for CPU parity");
        if (Logger::instance().would_log(LogLevel::DEBUG))
          LOG_DEBUG("[EXL3-GRAPH-TRACE] parity output copied layer=%d "
                       "row=%d\n",
                       state->model_layer, first_row);
        if (Logger::instance().would_log(LogLevel::DEBUG))
          LOG_DEBUG("[EXL3-GRAPH-TRACE] parity vectors count=%zu "
                       "batch_rows=%d first_route=%zu\n",
                       actual.size(), batch_rows, first_route);
        std::vector<float> expected(actual.size(), 0.0f);
        if (Logger::instance().would_log(LogLevel::DEBUG))
          LOG_DEBUG("[EXL3-GRAPH-TRACE] parity expected allocated "
                       "layer=%d\n",
                       state->model_layer);
        for (int batch_row = 0; batch_row < batch_rows; ++batch_row) {
          const size_t row_offset =
              static_cast<size_t>(batch_row) * state->hidden;
          for (int route = 0; route < state->topk; ++route) {
            const size_t route_index =
                first_route + static_cast<size_t>(batch_row) * state->topk +
                route;
            const int32_t expert_id = state->route_ids_host[route_index];
            if (Logger::instance().would_log(LogLevel::DEBUG))
              LOG_DEBUG("[EXL3-GRAPH-TRACE] parity CPU route=%d expert=%d "
                           "layer=%d\n",
                           route, expert_id, state->model_layer);
            const std::string &prefix =
                state->expert_prefixes_by_id[static_cast<size_t>(expert_id)];
            const auto gate = state->matrix_loader(prefix + "gate_proj");
            const auto up = state->matrix_loader(prefix + "up_proj");
            const auto down = state->matrix_loader(prefix + "down_proj");
            std::vector<float> gate_value(state->intermediate);
            std::vector<float> up_value(state->intermediate);
            std::vector<float> activation(state->intermediate);
            std::vector<float> down_value(state->hidden);
            const float *input_row =
                host_input.data() + row_offset;
            if (!exl3_mul1_matmul_packed(
                    gate.data(), input_row, gate_value.data(), 1,
                    state->hidden, state->intermediate) ||
                !exl3_mul1_matmul_packed(
                    up.data(), input_row, up_value.data(), 1,
                    state->hidden, state->intermediate))
              throw std::runtime_error(
                  "EXL3 bridge CPU parity gate/up projection failed");
            for (int index = 0; index < state->intermediate; ++index)
              activation[static_cast<size_t>(index)] =
                  gate_value[static_cast<size_t>(index)] /
                  (1.0f + std::exp(-gate_value[static_cast<size_t>(index)])) *
                  up_value[static_cast<size_t>(index)];
            if (!exl3_mul1_matmul_packed(
                    down.data(), activation.data(), down_value.data(), 1,
                    state->intermediate, state->hidden))
              throw std::runtime_error(
                  "EXL3 bridge CPU parity down projection failed");
            if (Logger::instance().would_log(LogLevel::DEBUG))
              LOG_DEBUG("[EXL3-GRAPH-TRACE] parity CPU projections done "
                           "expert=%d layer=%d\n",
                           expert_id, state->model_layer);
            const float route_weight =
                state->route_weights_host[route_index];
            for (int index = 0; index < state->hidden; ++index)
              expected[row_offset + static_cast<size_t>(index)] +=
                  route_weight * down_value[static_cast<size_t>(index)];
            if (Logger::instance().would_log(LogLevel::DEBUG))
              LOG_DEBUG("[EXL3-GRAPH-TRACE] parity accumulated expert=%d "
                           "weight=%g layer=%d\n",
                           expert_id, route_weight, state->model_layer);
          }
        }
        if (Logger::instance().would_log(LogLevel::DEBUG))
          LOG_DEBUG("[EXL3-GRAPH-TRACE] parity expected complete "
                       "layer=%d values=%zu\n",
                       state->model_layer, expected.size());
        float max_abs_error = 0.0f;
        float max_expected = 0.0f;
        size_t max_error_index = 0;
        const bool trace_parity =
            Logger::instance().would_log(LogLevel::DEBUG);
        if (trace_parity)
          LOG_DEBUG("[EXL3-GRAPH-TRACE] parity compare begin actual=%p "
                       "expected=%p size=%zu first=%g/%g\n",
                       static_cast<void *>(actual.data()),
                       static_cast<void *>(expected.data()), actual.size(),
                       actual.empty() ? 0.0f : actual[0],
                       expected.empty() ? 0.0f : expected[0]);
        for (size_t index = 0; index < actual.size(); ++index) {
          if (trace_parity && index % 512 == 0)
            LOG_DEBUG("[EXL3-GRAPH-TRACE] parity compare index=%zu\n",
                         index);
          const float error = std::abs(actual[index] - expected[index]);
          if (error > max_abs_error) {
            max_abs_error = error;
            max_error_index = index;
          }
          max_expected = std::max(max_expected, std::abs(expected[index]));
        }
        if (trace_parity)
          LOG_DEBUG("[EXL3-GRAPH-TRACE] parity compare loop done "
                       "max_error=%g max_expected=%g index=%zu\n",
                       max_abs_error, max_expected, max_error_index);
        // EXL3 matmuls run on upstream ExLlamaV3 kernels, which take fp16 activations (about 0.6% of the
        // output peak after the gate/up/down chain), so their tolerance is
        // wider than the fp32-activation legacy kernels'.
        float relative_tolerance = 1e-3f;
#ifdef ZEROMOE_EXL3_UPSTREAM
        relative_tolerance = 2e-2f;
#endif
        if (max_abs_error > 1e-3f + relative_tolerance * max_expected) {
          const size_t first_route =
              static_cast<size_t>(first_row) * routes_per_row;
          throw std::runtime_error(
              "EXL3 bridge topology-slot parity mismatch at layer=" +
              std::to_string(state->model_layer) + " rows=" +
              std::to_string(batch_rows) + " max_abs_error=" +
              std::to_string(max_abs_error) + " max_expected=" +
              std::to_string(max_expected) + " index=" +
              std::to_string(max_error_index) + " actual=" +
              std::to_string(actual[max_error_index]) + " expected=" +
              std::to_string(expected[max_error_index]) + " route0=" +
              std::to_string(state->route_ids_host[first_route]) + "/" +
              std::to_string(state->route_weights_host[first_route]) +
              " host_weight0=" +
              std::to_string(state->route_weights_host[first_route]));
        }
        if (Logger::instance().would_log(LogLevel::DEBUG))
          LOG_DEBUG("[EXL3-GRAPH-TRACE] parity compare complete "
                       "layer=%d max_error=%g\n",
                       state->model_layer, max_abs_error);
        LOG_INFO("[EXL3-GGML-PARITY] layer=%d rows=%d max_abs_error=%g\n",
                 state->model_layer, batch_rows, max_abs_error);
        if (use_cpu_reference_output) {
          check(cudaMemcpyAsync(row_output, expected.data(),
                                expected.size() * sizeof(float),
                                cudaMemcpyHostToDevice, stream),
                "write EXL3 CPU reference FFN output");
          check(cudaStreamSynchronize(stream),
                "complete EXL3 CPU reference FFN output");
        }
      }
    }
    return;
  }
  state->gpu_executor->routed_ffn_device(
      state->expert_prefixes_by_id, state->route_ids_host,
      state->route_weights_host,
      static_cast<const float *>(node->src[0]->data),
      static_cast<float *>(node->data), state->rows, state->topk,
      state->hidden, state->intermediate, stream,
      !state->matrix_loader, state->matrix_loader,
      state->routes_from_host
          ? static_cast<const float *>(node->src[2]->data)
          : device_route_weights);
}

ggml_tensor *exl3_ggml_map_cached_routed_ffn(
    ggml_context *ctx, ggml_tensor *input, ggml_tensor *route_table,
    Exl3GgmlCachedRoutedFfnState *state) {
  if (!ctx || !input || !route_table || !state || !state->gpu_executor ||
      input->type != GGML_TYPE_F32 || input->ne[1] != state->rows)
    return nullptr;
  auto *node = exl3_ggml_map_custom3(ctx, input, input, route_table,
                                    &state->custom);
  if (!node)
    return nullptr;
  node->ne[0] = state->hidden;
  node->ne[1] = state->rows;
  node->nb[1] = static_cast<size_t>(state->hidden) * sizeof(float);
  return node;
}

ggml_tensor *exl3_ggml_map_cached_routed_ffn_host_routes(
    ggml_context *ctx, ggml_tensor *input, ggml_tensor *route_ids,
    ggml_tensor *route_weights, Exl3GgmlCachedRoutedFfnState *state) {
  if (!ctx || !input || !route_ids || !route_weights || !state ||
      !state->gpu_executor || input->type != GGML_TYPE_F32 ||
      input->ne[1] != state->rows || route_ids->type != GGML_TYPE_I32 ||
      route_ids->ne[0] != state->topk || route_ids->ne[1] != state->rows ||
      route_weights->type != GGML_TYPE_F32 ||
      route_weights->ne[0] != state->topk ||
      route_weights->ne[1] != state->rows)
    return nullptr;
  state->route_tensors_on_device = false;
  auto *node = exl3_ggml_map_custom3(ctx, input, route_ids, route_weights,
                                    &state->custom);
  if (!node)
    return nullptr;
  node->ne[0] = state->hidden;
  node->ne[1] = state->rows;
  node->nb[1] = static_cast<size_t>(state->hidden) * sizeof(float);
  return node;
}

ggml_tensor *exl3_ggml_map_cached_routed_ffn_device_routes(
    ggml_context *ctx, ggml_tensor *input, ggml_tensor *route_ids,
    ggml_tensor *route_weights, Exl3GgmlCachedRoutedFfnState *state) {
  if (!ctx || !input || !route_ids || !route_weights || !state ||
      !state->gpu_executor || input->type != GGML_TYPE_F32 ||
      input->ne[0] != state->hidden || input->ne[1] != state->rows ||
      route_ids->type != GGML_TYPE_I32 || route_ids->ne[0] != state->topk ||
      route_ids->ne[1] != state->rows || route_weights->type != GGML_TYPE_F32 ||
      route_weights->ne[0] != state->topk ||
      route_weights->ne[1] != state->rows)
    return nullptr;
  state->routes_from_host = false;
  state->route_tensors_on_device = true;
  auto *node = exl3_ggml_map_custom3(ctx, input, route_ids, route_weights,
                                    &state->custom);
  if (!node)
    return nullptr;
  node->ne[0] = state->hidden;
  node->ne[1] = state->rows;
  node->nb[1] = static_cast<size_t>(state->hidden) * sizeof(float);
  return node;
}

bool Exl3GgmlCachedMatmulState::initialize(
    Exl3GpuExecutor *executor, int matrix_rows, int matrix_input_features,
    int matrix_output_features) {
  if (!executor || matrix_rows <= 0 || matrix_input_features <= 0 ||
      matrix_output_features <= 0)
    return false;
  gpu_executor = executor;
  rows = matrix_rows;
  input_features = matrix_input_features;
  output_features = matrix_output_features;
  custom.compute = compute;
  custom.user_data = this;
  return true;
}

void Exl3GgmlCachedMatmulState::compute(ggml_backend_t cuda_backend,
                                       ggml_tensor *node, void *user_data) {
  auto *state = static_cast<Exl3GgmlCachedMatmulState *>(user_data);
  if (!state || !node || !node->src[0] || !node->src[0]->data ||
      !node->data || node->src[0]->type != GGML_TYPE_F32 ||
      node->type != GGML_TYPE_F32 || node->src[0]->ne[0] !=
          state->input_features || node->src[0]->ne[1] != state->rows ||
      node->ne[0] != state->output_features || node->ne[1] != state->rows ||
      state->key.empty())
    throw std::runtime_error("invalid EXL3 cached GGML matmul node");
  const cudaStream_t stream = exl3_ggml_cuda_stream(cuda_backend);
  if (!stream || !state->gpu_executor->matmul_device_cached(
                     state->key,
                     static_cast<const float *>(node->src[0]->data),
                     static_cast<float *>(node->data), state->rows,
                     state->input_features, state->output_features, stream))
    throw std::runtime_error("EXL3 dense matrix is not present in GPU cache: " +
                             state->key);
}

ggml_tensor *exl3_ggml_map_cached_matmul(
    ggml_context *ctx, ggml_tensor *input,
    Exl3GgmlCachedMatmulState *state) {
  if (!ctx || !input || !state || !state->gpu_executor ||
      input->type != GGML_TYPE_F32 || input->ne[0] != state->input_features ||
      input->ne[1] != state->rows)
    return nullptr;
  auto *node = exl3_ggml_map_custom2(ctx, input, input, &state->custom);
  if (!node)
    return nullptr;
  node->ne[0] = state->output_features;
  node->ne[1] = state->rows;
  node->nb[1] = static_cast<size_t>(state->output_features) * sizeof(float);
  return node;
}

#endif
