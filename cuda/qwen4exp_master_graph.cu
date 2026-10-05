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
    int n_experts, int32_t *route_mode) {
    bool all_resident = true;
    bool all_staged = true;
    for (int k = 0; k < topk; ++k) {
        const int32_t id = routed_ids[k];
        if (id < 0) continue;
        if (id >= n_experts || !resident_bitmap[id]) all_resident = false;
        if (id >= n_experts || !staged_bitmap[id]) all_staged = false;
    }
    // Resident wins if a malformed publication marks both branches. The
    // host-side RouteClassification rejects overlapping maps; this tie-break
    // keeps the device predicate fail-safe during an in-flight bitmap update.
    const int32_t mode = all_resident ? 1 : (all_staged ? 2 : 0);
    cudaGraphSetConditional(resident_handle, mode == 1 ? 1u : 0u);
    cudaGraphSetConditional(staged_handle, mode == 2 ? 1u : 0u);
    *route_mode = mode;
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
