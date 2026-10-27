// Device kernels for the Qwen early-ready VRAM staging ring (epoch / publish / route map).
#include <cstdint>
#include <cuda/atomic>
#include <cuda_runtime.h>
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


} // namespace

extern "C" cudaError_t zeromoe_qwen_stage_begin(
    unsigned long long *epoch, cudaStream_t stream) {
    qwen_stage_begin_kernel<<<1, 1, 0, stream>>>(epoch);
    return cudaPeekAtLastError();
}

extern "C" cudaError_t zeromoe_qwen_stage_publish(int *slot_ids,
    unsigned long long *slot_epochs, int slot, int expert,
    unsigned long long epoch, cudaStream_t stream) {
    qwen_stage_publish_kernel<<<1, 1, 0, stream>>>(slot_ids, slot_epochs,
                                                slot, expert, epoch);
    return cudaPeekAtLastError();
}

extern "C" cudaError_t zeromoe_qwen_stage_map(const int *actual_ids,
    const unsigned long long *epoch, const int *slot_ids,
    unsigned long long *slot_epochs, int *dispatch_ids, int *stage_slots,
    float *stage_mask,
    int topk, int slots, cudaStream_t stream) {
    qwen_stage_map_kernel<<<1, 32, 0, stream>>>(actual_ids, epoch, slot_ids,
        slot_epochs, dispatch_ids, stage_slots, stage_mask, topk, slots);
    return cudaPeekAtLastError();
}
