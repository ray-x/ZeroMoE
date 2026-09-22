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
