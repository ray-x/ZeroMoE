// GDN's causal short conv + SiLU on the GPU (EngineConfig::qwen4exp_gdn_conv_gpu).
//
// CPU reference: attn_causal_conv1d_silu (src/attention_layer.cpp). That
// function is already a real, measured win over a naive channel-outer loop
// (see CLAUDE.md's "conv1d loop order: 29.4x" entry) -- but it's still
// purely CPU-DRAM-bandwidth-bound (~0.17 GFLOPS of real compute against
// ~103 MB/layer of traffic, per that same investigation), and this op has
// no sequential dependency beyond the kernel width (d_conv, typically 4):
// every (token, channel) output is independent given x and the persistent
// history. That makes it a clean fit for a GPU kernel that trades CPU DRAM
// bandwidth for GPU HBM bandwidth on the exact same memory-bound workload.
//
// Grid: one thread per (token, channel) pair -- blockIdx.y = token index s,
// (blockIdx.x, threadIdx.x) = channel c. This fully parallelises across
// S*channels threads (e.g. 504*10240 ~= 5.16M for a real qwen4exp prefill),
// unlike a one-thread-per-channel design that would still serially loop S
// times per thread. For a FIXED s, adjacent threads (adjacent c) read
// contiguous addresses at each of the `kernel` positions they touch (x is
// [S, channels] row-major, matching the CPU function's own layout exactly)
// -- so each of the up-to-4 loads per thread is itself a coalesced warp
// access, not a scatter.
extern "C" {

__global__ void gdn_conv1d_silu_kernel(
    const float* __restrict__ x,        // [S, channels], row-major
    const float* __restrict__ w,        // [channels, kernel], row-major (on-disk layout, no transpose needed)
    const float* __restrict__ history,  // [channels, kernel-1], row-major
    float* __restrict__ out,            // [S, channels], row-major
    int S, int channels, int kernel) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int s = blockIdx.y;
    if (c >= channels || s >= S) return;

    const int hist = kernel - 1;
    float acc = 0.0f;
    for (int k = 0; k < kernel; ++k) {
        // Same convention as the CPU reference: k=0 is the OLDEST tap,
        // k=kernel-1 the current sample; j is the position in the
        // conceptual [history ++ x] sequence.
        const int j = s + k;
        const float xv = (j >= hist)
            ? x[(size_t)(j - hist) * channels + c]
            : history[(size_t)c * hist + j];
        acc += xv * w[(size_t)c * kernel + k];
    }
    // SiLU on the conv output (not on the projections) -- matches the CPU
    // reference's own post-loop activation exactly.
    out[(size_t)s * channels + c] = acc / (1.0f + expf(-acc));
}

void launch_gdn_conv1d_silu(const float* x, const float* w, const float* history,
                            float* out, int S, int channels, int kernel,
                            cudaStream_t stream) {
    if (S <= 0 || channels <= 0 || kernel <= 0) return;
    const int block = 256;
    const int grid_x = (channels + block - 1) / block;
    dim3 grid(grid_x, S);
    gdn_conv1d_silu_kernel<<<grid, block, 0, stream>>>(x, w, history, out, S, channels, kernel);
}

}  // extern "C"
