// TEST-ONLY reference: MoEStream's original fp32 EXL3 mul1 matmul kernels.
// Production uses upstream ExLlamaV3 (backends/exl3/exl3_upstream_ops.cu); this file is
// kept only as an independent implementation for tests/exl3_upstream_parity.cu.
#include "exl3_cuda.hpp"
#include "exl3_format.hpp"

#include <cuda_fp16.h>
#include <climits>

namespace {

constexpr float kHadamardScale = 0.08838834764831845f;
constexpr int kKeyTilesPerSplit = 32;

__global__ void hc_silu(float *low, int dimensions, int streams) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= dimensions) return;
  const float value = low[index] / static_cast<float>(streams);
  low[index] = value / (1.0f + expf(-value));
}

__global__ void hc_mix(const float *normed, const float *gate, float *hidden,
                        int rows, int width, int streams,
                        bool divide_each_stream) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= rows * width) return;
  const int row = index / width;
  const int column = index % width;
  float sum = 0.0f;
  for (int stream_index = 0; stream_index < streams; ++stream_index) {
    const size_t offset =
        (static_cast<size_t>(row) * streams + stream_index) * width + column;
    const float gate_value = gate[offset];
    if (!isfinite(gate_value)) {
      hidden[index] = __int_as_float(0x7fffffff);
      return;
    }
    float value = __fmul_rn(normed[offset], 1.0f / (1.0f + expf(-gate_value)));
    if (divide_each_stream) value /= static_cast<float>(streams);
    sum = __fadd_rn(sum, value);
  }
  hidden[index] = divide_each_stream ? sum : sum / static_cast<float>(streams);
}

__global__ void swiglu(float *gate, const float *up, int dimensions) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < dimensions)
    gate[index] = gate[index] / (1.0f + expf(-gate[index])) * up[index];
}

__global__ void accumulate(float *output, const float *expert, float weight,
                            int dimensions) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < dimensions) output[index] += weight * expert[index];
}

__global__ void scale_device_weight(float *output, const float *weight,
                                     int dimensions) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < dimensions)
    output[index] *= weight[0];
}

__global__ void pack_route_table(uint8_t *table, const int32_t *route_ids,
                                 const float *route_weights, int topk,
                                 int rows, const int32_t *expert_remap) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= topk * rows)
    return;
  const int32_t route_id = route_ids[index];
  reinterpret_cast<int32_t *>(table)[index] =
      expert_remap ? expert_remap[route_id] : route_id;
  reinterpret_cast<float *>(table + topk * rows * sizeof(int32_t))[index] =
      route_weights[index];
}

__device__ __forceinline__ float decode_mul1(uint16_t state) {
  const uint32_t product = static_cast<uint32_t>(state) * 0x83DCD12Du;
  const uint32_t sum = __dp4a(product, 0x01010101u, 0x6400u);
  return __half2float(__hfma(__ushort_as_half(static_cast<uint16_t>(sum)),
                            __ushort_as_half(0x1eee), __ushort_as_half(0xc931)));
}

template <int Bits>
__device__ __forceinline__ uint16_t decode_state(const uint16_t *packed,
                                                 int offset) {
  constexpr int word_count = Bits * 8;
  const int bit_begin = (offset + 257) * Bits - 16;
  const int bit_end = bit_begin + 16;
  const int last_word = (bit_end - 1) / 32;
  const int shift = (last_word + 1) * 32 - bit_end;
  const auto *words = reinterpret_cast<const uint32_t *>(packed);
  const uint64_t merged =
      (static_cast<uint64_t>(words[(bit_begin / 32) % word_count]) << 32) |
      words[last_word % word_count];
  return static_cast<uint16_t>(merged >> shift);
}

template <int Bits>
__global__ void rotated_matmul_staged(const uint16_t *trellis,
                                       const float *input,
                                       const uint16_t *suh, float *partials,
                                       int input_width, int output_width,
                                       int rows, const uint8_t *routed_payload,
                                       size_t expert_stride,
                                       size_t projection_offset,
                                       const int32_t *route_ids,
                                       int route_slot, int route_topk,
                                       int expert_count) {
  constexpr int kVectorsPerTile = 2 * Bits;
  __shared__ uint4 packed_tiles[kKeyTilesPerSplit * kVectorsPerTile];
  __shared__ float transformed_input[kKeyTilesPerSplit * 16];
  __shared__ float hadamard_scratch[256];

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x / 32;
  const int native_thread = threadIdx.x / 8;
  const int component = threadIdx.x % 8;
  const int input_row = (native_thread % 4) * 2 + (component % 2) +
                        ((component % 4) >= 2 ? 8 : 0);
  const int output_column = warp + (component >= 4 ? 8 : 0);
  const int output_tiles = output_width / 16;
  const int first_key_tile = blockIdx.y * kKeyTilesPerSplit;
  const int last_key_tile = min(first_key_tile + kKeyTilesPerSplit, input_width / 16);
  const int key_tile_count = last_key_tile - first_key_tile;
  const int input_groups = key_tile_count / 8;
  const int waves = (input_groups + 1) / 2;
  const float *input_row_data = input + static_cast<size_t>(blockIdx.z) * input_width;

  if (routed_payload) {
    const int expert = route_ids[blockIdx.z * route_topk + route_slot];
    if (expert < 0 || expert >= expert_count)
      return;
    const auto *header = reinterpret_cast<const Exl3PackedMatrixHeader *>(
        routed_payload + static_cast<size_t>(expert) * expert_stride +
        projection_offset);
    if (header->bits != Bits)
      return;
    trellis = reinterpret_cast<const uint16_t *>(
        reinterpret_cast<const uint8_t *>(header) + header->trellis_offset);
    suh = reinterpret_cast<const uint16_t *>(
        reinterpret_cast<const uint8_t *>(header) + header->suh_offset);
  }

  for (int wave = 0; wave < waves; ++wave) {
    const int part = threadIdx.x / 128;
    const int element = threadIdx.x & 127;
    const int group = wave * 2 + part;
    float value = 0.0f;
    if (group < input_groups) {
      const int index = first_key_tile * 16 + group * 128 + element;
      value = input_row_data[index] * __half2float(__ushort_as_half(suh[index]));
    }
    const int warp_lane = threadIdx.x & 31;
    for (int distance = 1; distance <= 16; distance *= 2) {
      const float other = __shfl_xor_sync(0xffffffffu, value, distance);
      value = (warp_lane & distance) ? other - value : value + other;
    }
    for (int distance = 32; distance < 128; distance *= 2) {
      hadamard_scratch[part * 128 + element] = value;
      __syncthreads();
      const float other = hadamard_scratch[part * 128 + (element ^ distance)];
      value = (element & distance) ? other - value : value + other;
      __syncthreads();
    }
    transformed_input[group * 128 + element] = value * kHadamardScale;
    __syncthreads();
  }

  const int vector_count = key_tile_count * kVectorsPerTile;
  const uint4 *trellis_vectors = reinterpret_cast<const uint4 *>(trellis);
  for (int vector_index = threadIdx.x; vector_index < vector_count;
       vector_index += blockDim.x) {
    const int key_offset = vector_index / kVectorsPerTile;
    const int vector_offset = vector_index % kVectorsPerTile;
    const size_t tile = static_cast<size_t>(first_key_tile + key_offset) * output_tiles + blockIdx.x;
    packed_tiles[vector_index] = trellis_vectors[tile * kVectorsPerTile + vector_offset];
  }
  __syncthreads();

  float sum = 0.0f;
  for (int key_tile = 0; key_tile < key_tile_count; ++key_tile) {
    const uint16_t *packed = reinterpret_cast<const uint16_t *>(
        packed_tiles + key_tile * kVectorsPerTile);
    const float transformed =
        transformed_input[key_tile * 16 + input_row];
    sum = __fmaf_rn(decode_mul1(decode_state<Bits>(packed, threadIdx.x)),
                    transformed, sum);
  }
  sum += __shfl_xor_sync(0xffffffffu, sum, 1);
  sum += __shfl_xor_sync(0xffffffffu, sum, 2);
  sum += __shfl_xor_sync(0xffffffffu, sum, 8);
  sum += __shfl_xor_sync(0xffffffffu, sum, 16);
  if (lane == 0 || lane == 4) {
    const size_t base =
        (static_cast<size_t>(blockIdx.y) * rows + blockIdx.z) * output_width;
    partials[base + blockIdx.x * 16 + output_column] = sum;
  }
}

template <int GateBits, int UpBits>
__global__ void gate_up_rotated_matmul_fused(
    Exl3CudaMatrixView gate, Exl3CudaMatrixView up, const float *input,
    float *gate_partials, float *up_partials, int rows) {
  constexpr int kGateVectorsPerTile = 2 * GateBits;
  constexpr int kUpVectorsPerTile = 2 * UpBits;
  __shared__ uint4 gate_packed_tiles[kKeyTilesPerSplit * kGateVectorsPerTile];
  __shared__ uint4 up_packed_tiles[kKeyTilesPerSplit * kUpVectorsPerTile];
  __shared__ float gate_transformed_input[kKeyTilesPerSplit * 16];
  __shared__ float up_transformed_input[kKeyTilesPerSplit * 16];
  __shared__ float gate_hadamard_scratch[256];
  __shared__ float up_hadamard_scratch[256];

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x / 32;
  const int native_thread = threadIdx.x / 8;
  const int component = threadIdx.x % 8;
  const int input_row = (native_thread % 4) * 2 + (component % 2) +
                        ((component % 4) >= 2 ? 8 : 0);
  const int output_column = warp + (component >= 4 ? 8 : 0);
  const int output_tiles = gate.output_features / 16;
  const int first_key_tile = blockIdx.y * kKeyTilesPerSplit;
  const int last_key_tile = min(first_key_tile + kKeyTilesPerSplit,
                                gate.input_features / 16);
  const int key_tile_count = last_key_tile - first_key_tile;
  const int input_groups = key_tile_count / 8;
  const int waves = (input_groups + 1) / 2;
  const float *input_row_data =
      input + static_cast<size_t>(blockIdx.z) * gate.input_features;

  for (int wave = 0; wave < waves; ++wave) {
    const int part = threadIdx.x / 128;
    const int element = threadIdx.x & 127;
    const int group = wave * 2 + part;
    float gate_value = 0.0f;
    float up_value = 0.0f;
    if (group < input_groups) {
      const int index = first_key_tile * 16 + group * 128 + element;
      const float input_value = input_row_data[index];
      gate_value = input_value *
                   __half2float(__ushort_as_half(gate.suh[index]));
      up_value = input_value *
                 __half2float(__ushort_as_half(up.suh[index]));
    }
    const int warp_lane = threadIdx.x & 31;
    for (int distance = 1; distance <= 16; distance *= 2) {
      const float gate_other =
          __shfl_xor_sync(0xffffffffu, gate_value, distance);
      const float up_other = __shfl_xor_sync(0xffffffffu, up_value, distance);
      gate_value = (warp_lane & distance) ? gate_other - gate_value
                                           : gate_value + gate_other;
      up_value = (warp_lane & distance) ? up_other - up_value
                                         : up_value + up_other;
    }
    for (int distance = 32; distance < 128; distance *= 2) {
      gate_hadamard_scratch[part * 128 + element] = gate_value;
      up_hadamard_scratch[part * 128 + element] = up_value;
      __syncthreads();
      const float gate_other =
          gate_hadamard_scratch[part * 128 + (element ^ distance)];
      const float up_other =
          up_hadamard_scratch[part * 128 + (element ^ distance)];
      gate_value = (element & distance) ? gate_other - gate_value
                                         : gate_other + gate_value;
      up_value = (element & distance) ? up_other - up_value
                                       : up_other + up_value;
      __syncthreads();
    }
    gate_transformed_input[group * 128 + element] =
        gate_value * kHadamardScale;
    up_transformed_input[group * 128 + element] = up_value * kHadamardScale;
    __syncthreads();
  }

  const uint4 *gate_vectors = reinterpret_cast<const uint4 *>(gate.trellis);
  const uint4 *up_vectors = reinterpret_cast<const uint4 *>(up.trellis);
  const int gate_vector_count = key_tile_count * kGateVectorsPerTile;
  const int up_vector_count = key_tile_count * kUpVectorsPerTile;
  for (int vector_index = threadIdx.x; vector_index < gate_vector_count;
       vector_index += blockDim.x) {
    const int key_offset = vector_index / kGateVectorsPerTile;
    const int vector_offset = vector_index % kGateVectorsPerTile;
    const size_t tile =
        static_cast<size_t>(first_key_tile + key_offset) * output_tiles +
        blockIdx.x;
    gate_packed_tiles[vector_index] =
        gate_vectors[tile * kGateVectorsPerTile + vector_offset];
  }
  for (int vector_index = threadIdx.x; vector_index < up_vector_count;
       vector_index += blockDim.x) {
    const int key_offset = vector_index / kUpVectorsPerTile;
    const int vector_offset = vector_index % kUpVectorsPerTile;
    const size_t tile =
        static_cast<size_t>(first_key_tile + key_offset) * output_tiles +
        blockIdx.x;
    up_packed_tiles[vector_index] =
        up_vectors[tile * kUpVectorsPerTile + vector_offset];
  }
  __syncthreads();

  float gate_sum = 0.0f;
  float up_sum = 0.0f;
  for (int key_tile = 0; key_tile < key_tile_count; ++key_tile) {
    const uint16_t *gate_packed = reinterpret_cast<const uint16_t *>(
        gate_packed_tiles + key_tile * kGateVectorsPerTile);
    const uint16_t *up_packed = reinterpret_cast<const uint16_t *>(
        up_packed_tiles + key_tile * kUpVectorsPerTile);
    gate_sum = __fmaf_rn(
        decode_mul1(decode_state<GateBits>(gate_packed, threadIdx.x)),
        gate_transformed_input[key_tile * 16 + input_row], gate_sum);
    up_sum = __fmaf_rn(
        decode_mul1(decode_state<UpBits>(up_packed, threadIdx.x)),
        up_transformed_input[key_tile * 16 + input_row], up_sum);
  }
  gate_sum += __shfl_xor_sync(0xffffffffu, gate_sum, 1);
  gate_sum += __shfl_xor_sync(0xffffffffu, gate_sum, 2);
  gate_sum += __shfl_xor_sync(0xffffffffu, gate_sum, 8);
  gate_sum += __shfl_xor_sync(0xffffffffu, gate_sum, 16);
  up_sum += __shfl_xor_sync(0xffffffffu, up_sum, 1);
  up_sum += __shfl_xor_sync(0xffffffffu, up_sum, 2);
  up_sum += __shfl_xor_sync(0xffffffffu, up_sum, 8);
  up_sum += __shfl_xor_sync(0xffffffffu, up_sum, 16);
  if (lane == 0 || lane == 4) {
    const size_t base =
        (static_cast<size_t>(blockIdx.y) * rows + blockIdx.z) *
        gate.output_features;
    const size_t output_index = blockIdx.x * 16 + output_column;
    gate_partials[base + output_index] = gate_sum;
    up_partials[base + output_index] = up_sum;
  }
}

__global__ void output_hadamard(const float *partials, const uint16_t *scales,
                                 const uint16_t *bias, float *output,
                                 float *accumulator, float weight,
                                 const float *device_weight, int width,
                                 int rows, int splits,
                                 const uint8_t *routed_payload,
                                 size_t expert_stride, size_t projection_offset,
                                 const int32_t *route_ids, int route_slot,
                                 int route_topk, int expert_count) {
  if (routed_payload) {
    const int expert = route_ids[blockIdx.y * route_topk + route_slot];
    if (expert < 0 || expert >= expert_count)
      return;
    const auto *header = reinterpret_cast<const Exl3PackedMatrixHeader *>(
        routed_payload + static_cast<size_t>(expert) * expert_stride +
        projection_offset);
    scales = reinterpret_cast<const uint16_t *>(
        reinterpret_cast<const uint8_t *>(header) + header->svh_offset);
    bias = (header->flags & kExl3PackedHasBias)
               ? reinterpret_cast<const uint16_t *>(
                     reinterpret_cast<const uint8_t *>(header) +
                     header->bias_offset)
               : nullptr;
  }
  __shared__ float values[128];
  const int index = threadIdx.x;
  const int column = blockIdx.x * 128 + index;
  float sum = 0.0f;
  for (int split = 0; split < splits; ++split)
    sum += partials[(static_cast<size_t>(split) * rows + blockIdx.y) * width + column];
  values[index] = sum;
  __syncthreads();
  for (int distance = 1; distance < 128; distance *= 2) {
    const float own = values[index];
    const float other = values[index ^ distance];
    __syncthreads();
    values[index] = (index & distance) ? other - own : own + other;
    __syncthreads();
  }
  float value = values[index] * kHadamardScale *
                  __half2float(__ushort_as_half(scales[column]));
  if (bias) value += __half2float(__ushort_as_half(bias[column]));
  const size_t output_index = static_cast<size_t>(blockIdx.y) * width + column;
  const float applied_weight = device_weight ? device_weight[blockIdx.y] : weight;
  if (accumulator)
    accumulator[output_index] += applied_weight * value;
  else
    output[output_index] = value;
}

__global__ void gate_up_hadamard_swiglu(
    const float *gate_partials, const float *up_partials,
    const uint16_t *gate_scales, const uint16_t *up_scales,
    const uint16_t *gate_bias, const uint16_t *up_bias,
    float *activation, int width, int rows, int splits,
    const uint8_t *routed_payload, size_t expert_stride,
    size_t gate_projection_offset, size_t up_projection_offset,
    const int32_t *route_ids, int route_slot, int route_topk,
    int expert_count) {
  if (routed_payload) {
    const int expert = route_ids[blockIdx.y * route_topk + route_slot];
    if (expert < 0 || expert >= expert_count)
      return;
    const auto *gate_header = reinterpret_cast<const Exl3PackedMatrixHeader *>(
        routed_payload + static_cast<size_t>(expert) * expert_stride +
        gate_projection_offset);
    const auto *up_header = reinterpret_cast<const Exl3PackedMatrixHeader *>(
        routed_payload + static_cast<size_t>(expert) * expert_stride +
        up_projection_offset);
    gate_scales = reinterpret_cast<const uint16_t *>(
        reinterpret_cast<const uint8_t *>(gate_header) + gate_header->svh_offset);
    up_scales = reinterpret_cast<const uint16_t *>(
        reinterpret_cast<const uint8_t *>(up_header) + up_header->svh_offset);
    gate_bias = (gate_header->flags & kExl3PackedHasBias)
                    ? reinterpret_cast<const uint16_t *>(
                          reinterpret_cast<const uint8_t *>(gate_header) +
                          gate_header->bias_offset)
                    : nullptr;
    up_bias = (up_header->flags & kExl3PackedHasBias)
                  ? reinterpret_cast<const uint16_t *>(
                        reinterpret_cast<const uint8_t *>(up_header) +
                        up_header->bias_offset)
                  : nullptr;
  }
  __shared__ float gate_values[128];
  __shared__ float up_values[128];
  const int index = threadIdx.x;
  const int column = blockIdx.x * 128 + index;
  float gate_sum = 0.0f;
  float up_sum = 0.0f;
  for (int split = 0; split < splits; ++split) {
    const size_t offset = (static_cast<size_t>(split) * rows + blockIdx.y) * width + column;
    gate_sum += gate_partials[offset];
    up_sum += up_partials[offset];
  }
  gate_values[index] = gate_sum;
  up_values[index] = up_sum;
  __syncthreads();
  for (int distance = 1; distance < 128; distance *= 2) {
    const float own_gate = gate_values[index];
    const float other_gate = gate_values[index ^ distance];
    const float own_up = up_values[index];
    const float other_up = up_values[index ^ distance];
    __syncthreads();
    gate_values[index] = (index & distance) ? other_gate - own_gate : own_gate + other_gate;
    up_values[index] = (index & distance) ? other_up - own_up : own_up + other_up;
    __syncthreads();
  }
  float gate = gate_values[index] * kHadamardScale *
      __half2float(__ushort_as_half(gate_scales[column]));
  float up = up_values[index] * kHadamardScale *
      __half2float(__ushort_as_half(up_scales[column]));
  if (gate_bias) gate += __half2float(__ushort_as_half(gate_bias[column]));
  if (up_bias) up += __half2float(__ushort_as_half(up_bias[column]));
  activation[static_cast<size_t>(blockIdx.y) * width + column] =
      gate / (1.0f + expf(-gate)) * up;
}

template <int Bits>
__global__ void reconstruct_hadamard(Exl3CudaMatrixView matrix, float *weight) {
  extern __shared__ float tile[];
  const int input_base = blockIdx.y * 128;
  const int output_base = blockIdx.x * 128;
  for (int index = threadIdx.x; index < 128 * 128; index += blockDim.x) {
    const int row = index / 128;
    const int column = index % 128;
    const int native_row = row % 16;
    const int native_column = column % 16;
    const int native_thread = (native_column % 8) * 4 + (native_row % 8) / 2;
    const int component = native_row % 2 + (native_row >= 8 ? 2 : 0) +
                            (native_column >= 8 ? 4 : 0);
    const uint16_t *packed = matrix.trellis +
        (static_cast<size_t>((input_base + row) / 16) * (matrix.output_features / 16) +
          (output_base + column) / 16) * 16 * Bits;
    tile[index] = decode_mul1(decode_state<Bits>(packed, native_thread * 8 + component));
  }
  __syncthreads();
  for (int distance = 1; distance < 128; distance *= 2) {
    for (int pair = threadIdx.x; pair < 8192; pair += blockDim.x) {
      const int column = pair % 128;
      const int pair_row = pair / 128;
      const int row = (pair_row / distance) * 2 * distance + pair_row % distance;
      const int left = row * 128 + column;
      const int right = left + distance * 128;
      const float first = tile[left];
      const float second = tile[right];
      tile[left] = first + second;
      tile[right] = first - second;
    }
    __syncthreads();
  }
  for (int index = threadIdx.x; index < 16384; index += blockDim.x)
    tile[index] *= kHadamardScale *
        __half2float(__ushort_as_half(matrix.suh[input_base + index / 128]));
  __syncthreads();
  for (int distance = 1; distance < 128; distance *= 2) {
    for (int pair = threadIdx.x; pair < 8192; pair += blockDim.x) {
      const int row = pair / 64;
      const int pair_column = pair % 64;
      const int column = (pair_column / distance) * 2 * distance + pair_column % distance;
      const int left = row * 128 + column;
      const int right = left + distance;
      const float first = tile[left];
      const float second = tile[right];
      tile[left] = first + second;
      tile[right] = first - second;
    }
    __syncthreads();
  }
  for (int index = threadIdx.x; index < 16384; index += blockDim.x) {
    const int column = output_base + index % 128;
    weight[static_cast<size_t>(input_base + index / 128) * matrix.output_features + column] =
        tile[index] * kHadamardScale * __half2float(__ushort_as_half(matrix.svh[column]));
  }
}

bool valid(const Exl3CudaMatrixView &matrix) {
  return matrix.trellis && matrix.suh && matrix.svh &&
         matrix.input_features > 0 && matrix.output_features > 0 &&
         matrix.input_features % 128 == 0 && matrix.output_features % 128 == 0 &&
         matrix.bits >= 1 && matrix.bits <= 8;
}

template <int Bits>
cudaError_t reconstruct(const Exl3CudaMatrixView &matrix, float *weight,
                         cudaStream_t stream) {
  const cudaError_t attribute = cudaFuncSetAttribute(
      reconstruct_hadamard<Bits>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
  if (attribute != cudaSuccess) return attribute;
  const dim3 grid(matrix.output_features / 128, matrix.input_features / 128);
  reconstruct_hadamard<Bits><<<grid, 256, 65536, stream>>>(matrix, weight);
  return cudaGetLastError();
}

} // namespace

int exl3_cuda_key_splits(int input_features) {
  return (input_features / 16 + kKeyTilesPerSplit - 1) / kKeyTilesPerSplit;
}

cudaError_t launch_exl3_hc_silu(float *low, int dimensions, int streams,
                                cudaStream_t stream) {
  if (!low || dimensions <= 0 || streams <= 0) return cudaErrorInvalidValue;
  hc_silu<<<(static_cast<size_t>(dimensions) + 255) / 256, 256, 0, stream>>>(low, dimensions, streams);
  return cudaGetLastError();
}

cudaError_t launch_exl3_hc_mix(const float *normed, const float *gate,
                               float *hidden, int rows, int width, int streams,
                               bool divide_each_stream, cudaStream_t stream) {
  if (!normed || !gate || !hidden || rows <= 0 || width <= 0 || streams <= 0)
    return cudaErrorInvalidValue;
  const size_t dimensions = static_cast<size_t>(rows) * width;
  if (dimensions > INT_MAX) return cudaErrorInvalidValue;
  hc_mix<<<(dimensions + 255) / 256, 256, 0, stream>>>(
      normed, gate, hidden, rows, width, streams, divide_each_stream);
  return cudaGetLastError();
}

cudaError_t launch_exl3_swiglu(float *gate, const float *up, int dimensions,
                               cudaStream_t stream) {
  if (!gate || !up || dimensions <= 0) return cudaErrorInvalidValue;
  swiglu<<<(dimensions + 255) / 256, 256, 0, stream>>>(gate, up, dimensions);
  return cudaGetLastError();
}

cudaError_t launch_exl3_accumulate(float *output, const float *expert,
                                   float weight, int dimensions,
                                   cudaStream_t stream) {
  if (!output || !expert || dimensions <= 0) return cudaErrorInvalidValue;
  accumulate<<<(dimensions + 255) / 256, 256, 0, stream>>>(output, expert, weight, dimensions);
  return cudaGetLastError();
}

cudaError_t launch_exl3_scale_device_weight(float *output, const float *weight,
                                            int dimensions,
                                            cudaStream_t stream) {
  if (!output || !weight || dimensions <= 0)
    return cudaErrorInvalidValue;
  scale_device_weight<<<(dimensions + 255) / 256, 256, 0, stream>>>(
      output, weight, dimensions);
  return cudaGetLastError();
}

cudaError_t launch_exl3_pack_route_table(uint8_t *table,
                                         const int32_t *route_ids,
                                         const float *route_weights, int topk,
                                         int rows,
                                         const int32_t *expert_remap,
                                         cudaStream_t stream) {
  if (!table || !route_ids || !route_weights || topk <= 0 || rows <= 0)
    return cudaErrorInvalidValue;
  const int elements = topk * rows;
  pack_route_table<<<(elements + 255) / 256, 256, 0, stream>>>(
      table, route_ids, route_weights, topk, rows, expert_remap);
  return cudaGetLastError();
}

static cudaError_t launch_rotated_matmul(
    const Exl3CudaMatrixView &matrix, const float *input, int rows,
    float *partials, cudaStream_t stream,
    const uint8_t *routed_payload = nullptr, size_t expert_stride = 0,
    size_t projection_offset = 0, const int32_t *route_ids = nullptr,
    int route_slot = 0, int expert_count = 0) {
  const int splits = exl3_cuda_key_splits(matrix.input_features);
  const dim3 grid(matrix.output_features / 16, splits, rows);
  switch (matrix.bits) {
#define EXL3_LAUNCH(Bits) case Bits: rotated_matmul_staged<Bits><<<grid, 256, 0, stream>>>( \
      matrix.trellis, input, matrix.suh, partials, matrix.input_features, \
      matrix.output_features, rows, routed_payload, expert_stride, \
      projection_offset, route_ids, route_slot, 0, expert_count); break
    EXL3_LAUNCH(1);
    EXL3_LAUNCH(2);
    EXL3_LAUNCH(3);
    EXL3_LAUNCH(4);
    EXL3_LAUNCH(5);
    EXL3_LAUNCH(6);
    EXL3_LAUNCH(7);
    EXL3_LAUNCH(8);
#undef EXL3_LAUNCH
  }
  return cudaGetLastError();
}

template <int GateBits, int UpBits>
static cudaError_t launch_gate_up_rotated_matmul_fused(
    const Exl3CudaMatrixView &gate, const Exl3CudaMatrixView &up,
    const float *input, float *gate_partials, float *up_partials, int rows,
    cudaStream_t stream) {
  const dim3 grid(gate.output_features / 16,
                  exl3_cuda_key_splits(gate.input_features), rows);
  gate_up_rotated_matmul_fused<GateBits, UpBits>
      <<<grid, 256, 0, stream>>>(gate, up, input, gate_partials, up_partials,
                                rows);
  return cudaGetLastError();
}

#define EXL3_UP_BITS_CASE(GateBits, UpBits)                                 \
  case UpBits:                                                              \
    return launch_gate_up_rotated_matmul_fused<GateBits, UpBits>(           \
        gate, up, input, gate_partials, up_partials, rows, stream)
#define EXL3_GATE_BITS_CASE(GateBits)                                       \
  case GateBits:                                                            \
    switch (up.bits) {                                                      \
      EXL3_UP_BITS_CASE(GateBits, 1);                                       \
      EXL3_UP_BITS_CASE(GateBits, 2);                                       \
      EXL3_UP_BITS_CASE(GateBits, 3);                                       \
      EXL3_UP_BITS_CASE(GateBits, 4);                                       \
      EXL3_UP_BITS_CASE(GateBits, 5);                                       \
      EXL3_UP_BITS_CASE(GateBits, 6);                                       \
      EXL3_UP_BITS_CASE(GateBits, 7);                                       \
      EXL3_UP_BITS_CASE(GateBits, 8);                                       \
    default:                                                                \
      return cudaErrorInvalidValue;                                         \
    }
static cudaError_t launch_gate_up_rotated_matmul_fused(
    const Exl3CudaMatrixView &gate, const Exl3CudaMatrixView &up,
    const float *input, float *gate_partials, float *up_partials, int rows,
    cudaStream_t stream) {
  switch (gate.bits) {
    EXL3_GATE_BITS_CASE(1);
    EXL3_GATE_BITS_CASE(2);
    EXL3_GATE_BITS_CASE(3);
    EXL3_GATE_BITS_CASE(4);
    EXL3_GATE_BITS_CASE(5);
    EXL3_GATE_BITS_CASE(6);
    EXL3_GATE_BITS_CASE(7);
    EXL3_GATE_BITS_CASE(8);
  default:
    return cudaErrorInvalidValue;
  }
}
#undef EXL3_GATE_BITS_CASE
#undef EXL3_UP_BITS_CASE

static cudaError_t launch_exl3_mul1_matmul_impl(
    const Exl3CudaMatrixView &matrix, const float *input, float *output,
    float *accumulator, float weight, int rows, float *partials,
    cudaStream_t stream) {
  if (!valid(matrix) || !input || (!output && !accumulator) || !partials ||
      rows <= 0 || rows > 65535)
    return cudaErrorInvalidValue;
  cudaError_t error = launch_rotated_matmul(matrix, input, rows, partials, stream);
  if (error != cudaSuccess) return error;
  const int splits = exl3_cuda_key_splits(matrix.input_features);
  output_hadamard<<<dim3(matrix.output_features / 128, rows), 128, 0, stream>>>(
      partials, matrix.svh, matrix.bias, output, accumulator, weight, nullptr,
      matrix.output_features, rows, splits, nullptr, 0, 0, nullptr, 0, 0, 0);
  return cudaGetLastError();
}

cudaError_t launch_exl3_mul1_matmul(const Exl3CudaMatrixView &matrix,
                                   const float *input, float *output, int rows,
                                   float *partials, cudaStream_t stream) {
  return launch_exl3_mul1_matmul_impl(matrix, input, output, nullptr, 0.0f,
                                      rows, partials, stream);
}

cudaError_t launch_exl3_mul1_matmul_accumulate(
    const Exl3CudaMatrixView &matrix, const float *input, float *accumulator,
    float weight, int rows, float *partials, cudaStream_t stream) {
  if (!accumulator) return cudaErrorInvalidValue;
  return launch_exl3_mul1_matmul_impl(matrix, input, nullptr, accumulator,
                                      weight, rows, partials, stream);
}

cudaError_t launch_exl3_mul1_matmul_accumulate_device_weight(
    const Exl3CudaMatrixView &matrix, const float *input, float *accumulator,
    const float *weight, int rows, float *partials, cudaStream_t stream) {
  if (!valid(matrix) || !input || !accumulator || !weight || !partials ||
      rows <= 0 || rows > 65535)
    return cudaErrorInvalidValue;
  cudaError_t error = launch_rotated_matmul(matrix, input, rows, partials, stream);
  if (error != cudaSuccess)
    return error;
  const int splits = exl3_cuda_key_splits(matrix.input_features);
  output_hadamard<<<dim3(matrix.output_features / 128, rows), 128, 0, stream>>>(
      partials, matrix.svh, matrix.bias, nullptr, accumulator, 0.0f, weight,
      matrix.output_features, rows, splits, nullptr, 0, 0, nullptr, 0, 0, 0);
  return cudaGetLastError();
}

cudaError_t launch_exl3_mul1_gate_up(
    const Exl3CudaMatrixView &gate, const Exl3CudaMatrixView &up,
    const float *input, float *activation, int rows, float *gate_partials,
    float *up_partials, cudaStream_t stream) {
  if (!valid(gate) || !valid(up) || !input || !activation ||
      !gate_partials || !up_partials || gate_partials == up_partials ||
      gate.input_features != up.input_features ||
      gate.output_features != up.output_features || rows <= 0 || rows > 65535)
    return cudaErrorInvalidValue;
  cudaError_t error = launch_gate_up_rotated_matmul_fused(
      gate, up, input, gate_partials, up_partials, rows, stream);
  if (error != cudaSuccess) return error;
  gate_up_hadamard_swiglu<<<dim3(gate.output_features / 128, rows), 128, 0, stream>>>(
      gate_partials, up_partials, gate.svh, up.svh, gate.bias, up.bias,
      activation, gate.output_features, rows,
      exl3_cuda_key_splits(gate.input_features), nullptr, 0, 0, 0, nullptr,
      0, 0, 0);
  return cudaGetLastError();
}

static cudaError_t launch_routed_rotated(
    const uint8_t *payload, size_t expert_stride, size_t projection_offset,
    const int32_t *route_ids, int route_slot, int route_topk, int expert_count,
    const float *input, int rows, int hidden, int output_width,
    float *partials, int bits_hint, cudaStream_t stream) {
  const int splits = exl3_cuda_key_splits(hidden);
  const dim3 grid(output_width / 16, splits, rows);
#define EXL3_ROUTED_LAUNCH(Bits)                                             \
  {                                                                          \
    if (bits_hint == 0 || bits_hint == Bits) {                              \
    const Exl3CudaMatrixView matrix{nullptr, nullptr, nullptr, nullptr,     \
                                    hidden, output_width, Bits};             \
    rotated_matmul_staged<Bits><<<grid, 256, 0, stream>>>(                   \
        matrix.trellis, input, matrix.suh, partials, hidden, output_width,  \
        rows, payload, expert_stride, projection_offset, route_ids,          \
        route_slot, route_topk, expert_count);                               \
    const cudaError_t launch_error = cudaGetLastError();                     \
    if (launch_error != cudaSuccess) return launch_error;                    \
    }                                                                        \
  }
  EXL3_ROUTED_LAUNCH(1);
  EXL3_ROUTED_LAUNCH(2);
  EXL3_ROUTED_LAUNCH(3);
  EXL3_ROUTED_LAUNCH(4);
  EXL3_ROUTED_LAUNCH(5);
  EXL3_ROUTED_LAUNCH(6);
  EXL3_ROUTED_LAUNCH(7);
  EXL3_ROUTED_LAUNCH(8);
#undef EXL3_ROUTED_LAUNCH
  return cudaGetLastError();
}

cudaError_t launch_exl3_routed_ffn(
    const uint8_t *payload, size_t expert_stride, size_t gate_offset,
    size_t up_offset, size_t down_offset, const int32_t *route_ids,
    const float *route_weights, int topk, int expert_count, const float *input,
    float *output, int rows, int hidden, int intermediate, float *activation,
    float *partials, int gate_bits, int up_bits, int down_bits,
    cudaStream_t stream) {
  if (!payload || !route_ids || !route_weights || !input || !output ||
      !activation || !partials || expert_stride == 0 || topk <= 0 ||
      topk > expert_count || expert_count <= 0 || rows <= 0 || hidden <= 0 ||
      intermediate <= 0)
    return cudaErrorInvalidValue;
  if (gate_bits < 0 || gate_bits > 8 || up_bits < 0 || up_bits > 8 ||
      down_bits < 0 || down_bits > 8)
    return cudaErrorInvalidValue;
  const size_t gate_partials = static_cast<size_t>(intermediate) *
                               exl3_cuda_key_splits(hidden);
  cudaError_t error = cudaMemsetAsync(
      output, 0, static_cast<size_t>(hidden) * rows * sizeof(float), stream);
  if (error != cudaSuccess)
    return error;
  for (int route = 0; route < topk; ++route) {
    error = launch_routed_rotated(
        payload, expert_stride, gate_offset, route_ids, route, topk, expert_count,
        input, rows, hidden, intermediate, partials, gate_bits, stream);
    if (error != cudaSuccess)
      return error;
    error = launch_routed_rotated(
        payload, expert_stride, up_offset, route_ids, route, topk, expert_count,
        input, rows, hidden, intermediate, partials + gate_partials, up_bits,
        stream);
    if (error != cudaSuccess)
      return error;
    gate_up_hadamard_swiglu<<<dim3(intermediate / 128, rows), 128, 0, stream>>>(
        partials, partials + gate_partials, nullptr, nullptr, nullptr, nullptr,
        activation, intermediate, rows, exl3_cuda_key_splits(hidden), payload,
        expert_stride, gate_offset, up_offset, route_ids, route, topk,
        expert_count);
    error = cudaGetLastError();
    if (error != cudaSuccess)
      return error;
    error = launch_routed_rotated(
        payload, expert_stride, down_offset, route_ids, route, topk, expert_count,
        activation, rows, intermediate, hidden, partials, down_bits, stream);
    if (error != cudaSuccess)
      return error;
    output_hadamard<<<dim3(hidden / 128, rows), 128, 0, stream>>>(
        partials, nullptr, nullptr, nullptr, output, 0.0f,
        route_weights + static_cast<size_t>(route) * rows, hidden, rows,
        exl3_cuda_key_splits(intermediate), payload, expert_stride,
        down_offset, route_ids, route, topk, expert_count);
    error = cudaGetLastError();
    if (error != cudaSuccess)
      return error;
  }
  return cudaSuccess;
}

cudaError_t launch_exl3_mul1_reconstruct(const Exl3CudaMatrixView &matrix,
                                        float *weight, cudaStream_t stream) {
  if (!valid(matrix) || !weight) return cudaErrorInvalidValue;
  switch (matrix.bits) {
#define EXL3_RECONSTRUCT(Bits) case Bits: return reconstruct<Bits>(matrix, weight, stream)
    EXL3_RECONSTRUCT(1);
    EXL3_RECONSTRUCT(2);
    EXL3_RECONSTRUCT(3);
    EXL3_RECONSTRUCT(4);
    EXL3_RECONSTRUCT(5);
    EXL3_RECONSTRUCT(6);
    EXL3_RECONSTRUCT(7);
    EXL3_RECONSTRUCT(8);
#undef EXL3_RECONSTRUCT
  }
  return cudaErrorInvalidValue;
}
