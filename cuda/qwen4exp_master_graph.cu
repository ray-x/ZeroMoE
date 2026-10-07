// Predicate kernel for the Phase2 master-graph unification (CPUOffload
// branch): replaces the per-token host readback + std::unordered_set
// membership check ("is every routed id already in the GPU-resident hot
// set?") with a device-side check, so that decision can drive a real
// cudaGraphCondTypeIf node instead of a host branch between two separate
// graph launches.
//
// Must run as exactly one thread: cudaGraphSetConditional is undefined
// behavior under concurrent/racing calls on the same handle.
#include <cuda_runtime.h>

#include <cstdint>
#include <cuda/atomic>

namespace {

__global__ void qwen4exp_master_predicate_kernel(
    cudaGraphConditionalHandle handle, const int32_t *routed_ids,
    const uint8_t *gpu_hot_bitmap, int topk, int32_t *fallback_flag) {
    bool all_in_gpu_set = true;
    for (int k = 0; k < topk; ++k) {
        const int32_t id = routed_ids[k];
        if (id >= 0 && !gpu_hot_bitmap[id]) {
            all_in_gpu_set = false;
            break;
        }
    }
    cudaGraphSetConditional(handle, all_in_gpu_set ? 1u : 0u);
    // Pinned/host-visible: lets the caller know, after the one sync that
    // follows the master graph launch, whether it still needs to submit
    // the scpu fallback graph -- no separate host readback of routed_ids.
    *fallback_flag = all_in_gpu_set ? 0 : 1;
}

// Three-way route classifier for the partial-layer graph. A route set is
// eligible for a pure resident child only when every selected expert is in
// the immutable resident bitmap. Otherwise it may take the staged child only
// when every selected expert is in the published staged bitmap. Mixed sets
// deliberately select neither child; the host CPU fallback then owns every
// route exactly once. The two conditional handles are written by this one
// thread, so no concurrent cudaGraphSetConditional calls race.
__global__ void qwen4exp_master_route_mode_kernel(
    cudaGraphConditionalHandle resident_handle,
    cudaGraphConditionalHandle staged_handle, const int32_t *routed_ids,
    const uint8_t *resident_bitmap, const uint8_t *staged_bitmap, int topk,
    int n_experts, int32_t *route_mode, float *staged_slot_mask,
    int32_t *staged_dispatch_ids) {
    bool any_resident = false;
    bool any_staged = false;
    bool any_cpu = false;
    for (int k = 0; k < topk; ++k) {
        const int32_t id = routed_ids[k];
        bool resident = false;
        bool staged = false;
        if (id >= 0 && id < n_experts) {
            resident = resident_bitmap[id] != 0;
            staged = !resident && staged_bitmap[id] != 0;
        }
        any_resident |= resident;
        any_staged |= staged;
        any_cpu |= !resident && !staged;
        if (staged_slot_mask)
            staged_slot_mask[k] = staged ? 1.0f : 0.0f;
        if (staged_dispatch_ids)
            staged_dispatch_ids[k] = staged ? k : -1;
    }
    // Branch presence is independent: resident and staged work may run while
    // the sibling CPU host node computes only routes absent from both maps.
    // Bits 0..2 report resident, staged and CPU route presence respectively.
    // The original route-slot masks preserve exact-once weighted reduction.
    const int32_t mode = (any_resident ? 1 : 0) |
                         (any_staged ? 2 : 0) |
                         (any_cpu ? 4 : 0);
    cudaGraphSetConditional(resident_handle, any_resident ? 1u : 0u);
    cudaGraphSetConditional(staged_handle, any_staged ? 1u : 0u);
    *route_mode = mode;
}

__global__ void qwen4exp_zero_moe_outputs_kernel(float *resident,
                                                 float *staged, float *cpu,
                                                 int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        resident[i] = 0.0f;
        staged[i] = 0.0f;
        cpu[i] = 0.0f;
    }
}

__global__ void qwen4exp_merge_moe_outputs_kernel(const float *resident,
                                                   const float *staged,
                                                   const float *cpu,
                                                   float *merged, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        merged[i] = resident[i] + staged[i] + cpu[i];
}

// Unpack one Ready storage payload into the fixed top-k compact tensors used
// by the staged GGML child. StorageService serializes each slot as
// [gate|up|down], in router order. The kernel is deliberately device-only so
// it can live in a conditional child graph; no host callback or dynamic
// address calculation is required during replay.
__global__ void qwen4exp_unpack_staged_payload_kernel(
    const uint8_t *payload, uint8_t *gate, uint8_t *up, uint8_t *down,
    size_t gate_bytes, size_t up_bytes, size_t down_bytes,
    size_t gate_stride, size_t up_stride, size_t down_stride, int topk) {
    const int slot = static_cast<int>(blockIdx.x);
    if (slot >= topk) return;
    const size_t slot_bytes = gate_bytes + up_bytes + down_bytes;
    const uint8_t *src = payload + static_cast<size_t>(slot) * slot_bytes;
    for (size_t i = threadIdx.x; i < gate_bytes; i += blockDim.x)
        gate[static_cast<size_t>(slot) * gate_stride + i] = src[i];
    for (size_t i = threadIdx.x; i < up_bytes; i += blockDim.x)
        up[static_cast<size_t>(slot) * up_stride + i] =
            src[gate_bytes + i];
    for (size_t i = threadIdx.x; i < down_bytes; i += blockDim.x)
        down[static_cast<size_t>(slot) * down_stride + i] =
            src[gate_bytes + up_bytes + i];
}

} // namespace

// extern "C" so the .cpp caller (compiled by the host compiler, this file
// by nvcc) can declare and call it without relying on matching C++ name
// mangling across compilers -- same convention as the ggml-cuda.cu bridge
// functions in patches/ggml-cuda-stream-bridge.patch and
// patches/ggml-cuda-conditional-graph-harvest.patch. Returns the device
// function pointer for use as cudaKernelNodeParams::func when building the
// master graph's predicate node with the explicit graph API.
extern "C" void *moestream_qwen4exp_master_predicate_kernel_ptr() {
    return reinterpret_cast<void *>(&qwen4exp_master_predicate_kernel);
}

extern "C" void *moestream_qwen4exp_master_route_mode_kernel_ptr() {
    return reinterpret_cast<void *>(&qwen4exp_master_route_mode_kernel);
}

extern "C" void *moestream_qwen4exp_zero_moe_outputs_kernel_ptr() {
    return reinterpret_cast<void *>(&qwen4exp_zero_moe_outputs_kernel);
}

extern "C" void *moestream_qwen4exp_merge_moe_outputs_kernel_ptr() {
    return reinterpret_cast<void *>(&qwen4exp_merge_moe_outputs_kernel);
}

extern "C" void *moestream_qwen4exp_unpack_staged_payload_kernel_ptr() {
    return reinterpret_cast<void *>(&qwen4exp_unpack_staged_payload_kernel);
}

namespace {

__global__ void qwen_stage_begin_kernel(unsigned long long *epoch) {
    atomicAdd(epoch, 1ull);
}

__global__ void qwen_stage_publish_kernel(int *slot_ids,
    unsigned long long *slot_epochs, int slot, int expert,
    unsigned long long epoch) {
    slot_ids[slot] = expert;
    cuda::atomic_ref<unsigned long long, cuda::thread_scope_device>
        ready(slot_epochs[slot]);
    ready.store(epoch, cuda::memory_order_release);
}

__global__ void qwen_stage_map_kernel(const int *actual_ids,
    const unsigned long long *epoch, const int *slot_ids,
    unsigned long long *slot_epochs, int *dispatch_ids, int *stage_slots,
    float *stage_mask,
    int topk, int slots) {
    const int k = threadIdx.x;
    if (k >= topk) return;
    int found = -1;
    for (int slot = 0; slot < slots; ++slot) {
        cuda::atomic_ref<unsigned long long, cuda::thread_scope_device>
            ready(slot_epochs[slot]);
        if (ready.load(cuda::memory_order_acquire) == *epoch &&
            slot_ids[slot] == actual_ids[k]) {
            found = slot;
            break;
        }
    }
    stage_slots[k] = found;
    // Keep the GPU dispatch path on a valid expert even when this route is
    // not staged. The following mask multiply removes the dummy result. This
    // avoids relying on quant-kernel negative-ID behavior for an inactive
    // branch and makes the zero contribution deterministic across formats.
    dispatch_ids[k] = found >= 0 ? found : 0;
    stage_mask[k] = found >= 0 ? 1.0f : 0.0f;
}

__global__ void qwen_stage_delay_kernel(unsigned int microseconds) {
    unsigned long long remaining = static_cast<unsigned long long>(microseconds) * 1000;
    while (remaining) {
        const unsigned int chunk = remaining > 1000000 ? 1000000 : remaining;
        __nanosleep(chunk);
        remaining -= chunk;
    }
}

} // namespace

extern "C" cudaError_t moestream_qwen_stage_begin(
    unsigned long long *epoch, cudaStream_t stream) {
    qwen_stage_begin_kernel<<<1, 1, 0, stream>>>(epoch);
    return cudaPeekAtLastError();
}

extern "C" cudaError_t moestream_qwen_stage_publish(int *slot_ids,
    unsigned long long *slot_epochs, int slot, int expert,
    unsigned long long epoch, cudaStream_t stream) {
    qwen_stage_publish_kernel<<<1, 1, 0, stream>>>(slot_ids, slot_epochs,
                                                slot, expert, epoch);
    return cudaPeekAtLastError();
}

extern "C" cudaError_t moestream_qwen_stage_map(const int *actual_ids,
    const unsigned long long *epoch, const int *slot_ids,
    unsigned long long *slot_epochs, int *dispatch_ids, int *stage_slots,
    float *stage_mask,
    int topk, int slots, cudaStream_t stream) {
    qwen_stage_map_kernel<<<1, 32, 0, stream>>>(actual_ids, epoch, slot_ids,
        slot_epochs, dispatch_ids, stage_slots, stage_mask, topk, slots);
    return cudaPeekAtLastError();
}

extern "C" cudaError_t moestream_qwen_stage_delay(unsigned int microseconds,
                                                cudaStream_t stream) {
    qwen_stage_delay_kernel<<<1, 1, 0, stream>>>(microseconds);
    return cudaPeekAtLastError();
}
