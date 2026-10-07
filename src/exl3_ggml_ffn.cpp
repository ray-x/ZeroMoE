#include "exl3_ggml_ffn.hpp"
#include "exl3_gpu.hpp"

#if defined(MOESTREAM_USE_GGML_CUDA)

#include "exl3_cuda.hpp"

#include <algorithm>
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
  dynamic_route_table = false;
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

bool Exl3GgmlRoutedFfnState::initialize_dynamic(
    size_t matrix_expert_stride,
    const size_t (&matrix_projection_offsets)[3], int matrix_expert_count,
    int matrix_topk, int matrix_hidden, int matrix_intermediate,
    const int (&matrix_bits)[3], int matrix_rows, int device) {
  if (matrix_expert_stride == 0 || matrix_expert_count <= 0 || matrix_topk <= 0 ||
      matrix_topk > matrix_expert_count || matrix_hidden <= 0 ||
      matrix_intermediate <= 0 || matrix_rows <= 0 || device < 0 ||
      matrix_bits[0] < 0 || matrix_bits[0] > 8 || matrix_bits[1] < 0 ||
      matrix_bits[1] > 8 || matrix_bits[2] < 0 || matrix_bits[2] > 8)
    return false;
  dynamic_route_table = true;
  expert_stride = matrix_expert_stride;
  projection_offsets[0] = matrix_projection_offsets[0];
  projection_offsets[1] = matrix_projection_offsets[1];
  projection_offsets[2] = matrix_projection_offsets[2];
  expert_count = matrix_expert_count;
  topk = matrix_topk;
  hidden = matrix_hidden;
  intermediate = matrix_intermediate;
  bits[0] = matrix_bits[0];
  bits[1] = matrix_bits[1];
  bits[2] = matrix_bits[2];
  rows = matrix_rows;
  custom.compute = compute;
  custom.user_data = this;
  custom.graph_capture_safe = true;
  const size_t gate_partials = static_cast<size_t>(intermediate) *
                               exl3_cuda_key_splits(hidden);
  cudaSetDevice(device);
  if (cudaMalloc(&activation, static_cast<size_t>(intermediate) * matrix_rows *
                                  sizeof(float)) !=
      cudaSuccess)
    return false;
  if (cudaMalloc(&partials, std::max(2 * gate_partials * matrix_rows,
                                    static_cast<size_t>(hidden) *
                                        exl3_cuda_key_splits(intermediate) *
                                        matrix_rows) *
                               sizeof(float)) != cudaSuccess) {
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
  if (dynamic_route_table) {
    if (!input || !opaque || !route_weights || !output || input->ne[1] != rows ||
        route_weights->type != GGML_TYPE_I8 ||
        route_weights->ne[0] < static_cast<int64_t>(topk) *
                                  rows * (sizeof(int32_t) + sizeof(float)))
      throw std::runtime_error("invalid EXL3 GGML dynamic route table");
    const auto *route_table = static_cast<const uint8_t *>(route_weights->data);
    check(launch_exl3_routed_ffn(
              static_cast<const uint8_t *>(opaque->data), expert_stride,
              projection_offsets[0], projection_offsets[1],
              projection_offsets[2], reinterpret_cast<const int32_t *>(route_table),
              reinterpret_cast<const float *>(route_table +
                                              topk * rows * sizeof(int32_t)),
              topk, expert_count, static_cast<const float *>(input->data),
              static_cast<float *>(output->data), rows, hidden, intermediate,
              activation, partials, bits[0], bits[1], bits[2], stream),
          "launch EXL3 GGML dynamic routed FFN");
    return;
  }
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

ggml_tensor *exl3_ggml_map_dynamic_routed_ffn(
    ggml_context *ctx, ggml_tensor *input, ggml_tensor *opaque,
    ggml_tensor *route_table, Exl3GgmlRoutedFfnState *state) {
  if (!ctx || !input || !opaque || !route_table || !state ||
      !state->dynamic_route_table)
    return nullptr;
  auto *node = exl3_ggml_map_custom3(ctx, input, opaque, route_table,
                                     &state->custom);
  if (!node)
    return nullptr;
  node->ne[0] = state->hidden;
  node->ne[1] = input->ne[1];
  node->nb[1] = static_cast<size_t>(state->hidden) * sizeof(float);
  return node;
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
  route_ids_host.resize(static_cast<size_t>(rows) * topk);
  route_weights_host.resize(static_cast<size_t>(rows) * topk);
  custom.compute = compute;
  custom.user_data = this;
  return true;
}

bool Exl3GgmlCachedRoutedFfnState::set_host_routes(
    const int32_t *route_ids, const float *route_weights) {
  if (!route_ids || !route_weights || rows <= 0 || topk <= 0)
    return false;
  const size_t route_count = static_cast<size_t>(rows) * topk;
  std::copy_n(route_ids, route_count, route_ids_host.data());
  std::copy_n(route_weights, route_count, route_weights_host.data());
  routes_from_host = true;
  return true;
}

void Exl3GgmlCachedRoutedFfnState::set_matrix_loader(
    std::function<std::vector<uint8_t>(const std::string &)> loader) {
  matrix_loader = std::move(loader);
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
  } else {
    const size_t required_bytes = ids_bytes + weights_bytes;
    if (node->src[2]->type != GGML_TYPE_I8 ||
        ggml_nbytes(node->src[2]) < required_bytes)
      throw std::runtime_error("invalid EXL3 cached GGML route table");
    const auto *route_bytes = static_cast<const uint8_t *>(node->src[2]->data);
    check(cudaMemcpyAsync(state->route_ids_host.data(), route_bytes, ids_bytes,
                          cudaMemcpyDeviceToHost, stream),
          "read cached GGML expert IDs");
    check(cudaMemcpyAsync(state->route_weights_host.data(),
                          route_bytes + ids_bytes, weights_bytes,
                          cudaMemcpyDeviceToHost, stream),
          "read cached GGML route weights");
    check(cudaStreamSynchronize(stream),
          "complete cached GGML route readback");
  }
  state->gpu_executor->routed_ffn_device(
      state->expert_prefixes_by_id, state->route_ids_host.data(),
      state->route_weights_host.data(),
      static_cast<const float *>(node->src[0]->data),
      static_cast<float *>(node->data), state->rows, state->topk,
      state->hidden, state->intermediate, stream,
      !state->matrix_loader, state->matrix_loader);
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

bool Exl3GgmlRoutePackState::initialize(int route_count, int rows) {
  if (route_count <= 0 || rows <= 0)
    return false;
  topk = route_count;
  route_rows = rows;
  custom.compute = compute;
  custom.user_data = this;
  custom.graph_capture_safe = true;
  return true;
}

Exl3GgmlRoutePackState::~Exl3GgmlRoutePackState() {
  if (expert_remap_device)
    cudaFree(expert_remap_device);
}

bool Exl3GgmlRoutePackState::initialize_remap(
    int route_count, const std::vector<int32_t> &remap, int device, int rows) {
  if (!initialize(route_count, rows) || remap.empty() || device < 0)
    return false;
  cudaSetDevice(device);
  if (cudaMalloc(&expert_remap_device,
                 remap.size() * sizeof(int32_t)) != cudaSuccess)
    return false;
  if (cudaMemcpy(expert_remap_device, remap.data(),
                 remap.size() * sizeof(int32_t), cudaMemcpyHostToDevice) !=
      cudaSuccess) {
    cudaFree(expert_remap_device);
    expert_remap_device = nullptr;
    return false;
  }
  expert_remap_count = static_cast<int>(remap.size());
  return true;
}

void Exl3GgmlRoutePackState::compute(ggml_backend_t cuda_backend,
                                     ggml_tensor *node, void *user_data) {
  auto *state = static_cast<Exl3GgmlRoutePackState *>(user_data);
  const cudaStream_t stream = exl3_ggml_cuda_stream(cuda_backend);
  if (!node || !node->data || !node->src[1] || !node->src[2])
    throw std::runtime_error("invalid EXL3 GGML route pack node");
  check(launch_exl3_pack_route_table(
            static_cast<uint8_t *>(node->data),
            static_cast<const int32_t *>(node->src[1]->data),
            static_cast<const float *>(node->src[2]->data), state->topk,
            state->route_rows, state->expert_remap_device, stream),
        "pack EXL3 GGML route table");
}

ggml_tensor *exl3_ggml_map_route_table(ggml_context *ctx,
                                       ggml_tensor *route_table,
                                       ggml_tensor *route_ids,
                                       ggml_tensor *route_weights,
                                       Exl3GgmlRoutePackState *state) {
  if (!ctx || !route_table || !route_ids || !route_weights || !state ||
      state->topk <= 0)
    return nullptr;
  return exl3_ggml_map_custom3(ctx, route_table, route_ids, route_weights,
                               &state->custom);
}

#endif
