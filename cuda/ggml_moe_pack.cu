// Transfer-side unpack for the GGML MoE runner.
//
// The host/RAM representation is one expert-major [gate|up|down] record so a
// routed expert can cross PCIe with one contiguous copy. GGML's quantized
// MUL_MAT_ID tensors remain three ordinary planar tensors; this kernel is the
// device-local layout conversion between the two representations.

#include <cuda_runtime.h>
#include <stdint.h>

namespace {

__global__ void ggml_moe_unpack_kernel(
    const uint8_t * __restrict__ src,
    uint8_t * __restrict__ gate,
    uint8_t * __restrict__ up,
    uint8_t * __restrict__ down,
    int count,
    size_t src_stride,
    size_t gate_bytes,
    size_t up_bytes,
    size_t down_bytes,
    size_t gate_stride,
    size_t up_stride,
    size_t down_stride) {
    const size_t gate_units = (gate_bytes + 15) / 16;
    const size_t up_units = (up_bytes + 15) / 16;
    const size_t down_units = (down_bytes + 15) / 16;
    const size_t units_per_expert = gate_units + up_units + down_units;
    const size_t unit = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (unit >= (size_t) count * units_per_expert) return;

    const int expert = (int) (unit / units_per_expert);
    const size_t local = unit % units_per_expert;
    uint8_t *dst_base;
    const uint8_t *src_part;
    size_t part_bytes;
    size_t part_offset;

    if (local < gate_units) {
        dst_base = gate + (size_t) expert * gate_stride;
        src_part = src + (size_t) expert * src_stride;
        part_bytes = gate_bytes;
        part_offset = local * 16;
    } else if (local < gate_units + up_units) {
        const size_t u = local - gate_units;
        dst_base = up + (size_t) expert * up_stride;
        src_part = src + (size_t) expert * src_stride + gate_bytes;
        part_bytes = up_bytes;
        part_offset = u * 16;
    } else {
        const size_t d = local - gate_units - up_units;
        dst_base = down + (size_t) expert * down_stride;
        src_part = src + (size_t) expert * src_stride + gate_bytes + up_bytes;
        part_bytes = down_bytes;
        part_offset = d * 16;
    }

    const size_t remaining = part_offset < part_bytes ? part_bytes - part_offset : 0;
    const size_t n = remaining < 16 ? remaining : 16;
    for (size_t i = 0; i < n; ++i)
        dst_base[part_offset + i] = src_part[part_offset + i];
}

__global__ void ggml_moe_unpack_scatter_kernel(
    const uint8_t * __restrict__ src,
    uint8_t * __restrict__ gate,
    uint8_t * __restrict__ up,
    uint8_t * __restrict__ down,
    const int32_t * __restrict__ dst_slots,
    int count,
    size_t src_stride,
    size_t gate_bytes,
    size_t up_bytes,
    size_t down_bytes,
    size_t gate_stride,
    size_t up_stride,
    size_t down_stride) {
    const size_t gate_units = (gate_bytes + 15) / 16;
    const size_t up_units = (up_bytes + 15) / 16;
    const size_t down_units = (down_bytes + 15) / 16;
    const size_t units_per_expert = gate_units + up_units + down_units;
    const size_t unit = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (unit >= (size_t) count * units_per_expert) return;

    const int compact = (int) (unit / units_per_expert);
    const int expert = dst_slots[compact];
    if (expert < 0) return;
    const size_t local = unit % units_per_expert;
    uint8_t *dst_base;
    const uint8_t *src_part;
    size_t part_bytes;
    size_t part_offset;

    if (local < gate_units) {
        dst_base = gate + (size_t) expert * gate_stride;
        src_part = src + (size_t) compact * src_stride;
        part_bytes = gate_bytes;
        part_offset = local * 16;
    } else if (local < gate_units + up_units) {
        const size_t u = local - gate_units;
        dst_base = up + (size_t) expert * up_stride;
        src_part = src + (size_t) compact * src_stride + gate_bytes;
        part_bytes = up_bytes;
        part_offset = u * 16;
    } else {
        const size_t d = local - gate_units - up_units;
        dst_base = down + (size_t) expert * down_stride;
        src_part = src + (size_t) compact * src_stride + gate_bytes + up_bytes;
        part_bytes = down_bytes;
        part_offset = d * 16;
    }

    const size_t remaining = part_offset < part_bytes ? part_bytes - part_offset : 0;
    const size_t n = remaining < 16 ? remaining : 16;
    for (size_t i = 0; i < n; ++i)
        dst_base[part_offset + i] = src_part[part_offset + i];
}

} // namespace

extern "C" cudaError_t launch_ggml_moe_unpack(
    const void *src,
    void *gate,
    void *up,
    void *down,
    int count,
    size_t src_stride,
    size_t gate_bytes,
    size_t up_bytes,
    size_t down_bytes,
    size_t gate_stride,
    size_t up_stride,
    size_t down_stride,
    cudaStream_t stream) {
    if (!src || !gate || !up || !down || count <= 0)
        return cudaErrorInvalidValue;
    const size_t gate_units = (gate_bytes + 15) / 16;
    const size_t up_units = (up_bytes + 15) / 16;
    const size_t down_units = (down_bytes + 15) / 16;
    const size_t total = (size_t) count * (gate_units + up_units + down_units);
    constexpr int threads = 256;
    const int blocks = (int) ((total + threads - 1) / threads);
    ggml_moe_unpack_kernel<<<blocks, threads, 0, stream>>>(
        (const uint8_t *) src, (uint8_t *) gate, (uint8_t *) up,
        (uint8_t *) down, count, src_stride, gate_bytes, up_bytes, down_bytes,
        gate_stride, up_stride, down_stride);
    return cudaGetLastError();
}

extern "C" cudaError_t launch_ggml_moe_unpack_scatter(
    const void *src,
    void *gate,
    void *up,
    void *down,
    const int32_t *dst_slots,
    int count,
    size_t src_stride,
    size_t gate_bytes,
    size_t up_bytes,
    size_t down_bytes,
    size_t gate_stride,
    size_t up_stride,
    size_t down_stride,
    cudaStream_t stream) {
    if (!src || !gate || !up || !down || !dst_slots || count <= 0)
        return cudaErrorInvalidValue;
    const size_t gate_units = (gate_bytes + 15) / 16;
    const size_t up_units = (up_bytes + 15) / 16;
    const size_t down_units = (down_bytes + 15) / 16;
    const size_t total = (size_t) count * (gate_units + up_units + down_units);
    constexpr int threads = 256;
    const int blocks = (int) ((total + threads - 1) / threads);
    ggml_moe_unpack_scatter_kernel<<<blocks, threads, 0, stream>>>(
        (const uint8_t *) src, (uint8_t *) gate, (uint8_t *) up,
        (uint8_t *) down, dst_slots, count, src_stride, gate_bytes, up_bytes,
        down_bytes, gate_stride, up_stride, down_stride);
    return cudaGetLastError();
}
