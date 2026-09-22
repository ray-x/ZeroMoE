// Milestone C: synthetic branched CUDA-graph sandbox for CPUOffload.
//
// The graph has two branches after the input upload:
//
//   Branch A: GPU hot-expert work
//   Branch B: D2H -> synchronous cudaHostNode -> H2D
//   Branch C: device combine, dependent on both A and B
//
// A serialized graph with the same work is measured as the control.  The
// branched graph validates that a synchronous host node can overlap with
// independent GPU work without an external event ring or a polling kernel.

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <thread>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace {

constexpr int kDefaultIterations = 32;
constexpr int kDefaultElements = 4096;       // 16 KiB CPU activation.
constexpr int kDefaultCpuRepeats = 256;      // Make overlap measurable.
constexpr int kDefaultGpuWork = 1 << 20;     // Independent GPU branch work.
constexpr int kDefaultGpuRepeats = 64;
constexpr float kOutputTolerance = 1.0e-2f;

using Clock = std::chrono::steady_clock;

uint64_t now_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<
        std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count());
}

const char *cuda_error(cudaError_t status) {
    return cudaGetErrorString(status);
}

#define CUDA_OK(expr)                                                        \
    do {                                                                      \
        const cudaError_t _status = (expr);                                  \
        if (_status != cudaSuccess) {                                        \
            std::fprintf(stderr, "[GRAPH-SANDBOX] %s failed: %s\n",         \
                         #expr, cuda_error(_status));                       \
            return false;                                                     \
        }                                                                     \
    } while (false)

struct GraphState {
    int elements = 0;
    int cpu_repeats = 0;
    int gpu_work = 0;
    int gpu_repeats = 0;
    uint64_t sequence = 0;

    float *host_input = nullptr;
    float *host_cpu_output = nullptr;
    float *host_result = nullptr;
    float *device_input = nullptr;
    float *device_cpu_output = nullptr;
    float *device_gpu_output = nullptr;
    float *device_combined = nullptr;

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    cudaEvent_t gpu_start = nullptr;
    cudaEvent_t gpu_end = nullptr;
    cudaEvent_t graph_start = nullptr;
    cudaEvent_t graph_end = nullptr;
    cudaStream_t stream = nullptr;

    std::atomic<uint64_t> host_callback_ns{0};
};

float cpu_transform(float value, int repeats, uint64_t sequence) {
    float out = value + 0.125f + static_cast<float>(sequence) * 0.001f;
    for (int i = 0; i < repeats; ++i)
        out = out * 1.000001f + 0.000001f;
    return out;
}

float gpu_transform(float value, int repeats) {
    float out = value;
    for (int i = 0; i < repeats; ++i)
        out = out * 1.000001f + 0.0000005f;
    return out;
}

void CUDART_CB cpu_host_node(void *user) {
    auto *state = static_cast<GraphState *>(user);
    const uint64_t start = now_ns();
    for (int i = 0; i < state->elements; ++i)
        state->host_cpu_output[i] = cpu_transform(
            state->host_input[i], state->cpu_repeats, state->sequence);
#if defined(__x86_64__) || defined(__i386__)
    _mm_sfence();
#endif
    std::atomic_thread_fence(std::memory_order_release);
    state->host_callback_ns.store(now_ns() - start,
                                  std::memory_order_release);
}

__global__ void gpu_hot_branch_kernel(const float *input, float *output,
                                      int elements, int work_elements,
                                      int repeats) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= work_elements)
        return;
    float value = input[index % elements];
    for (int i = 0; i < repeats; ++i)
        value = value * 1.000001f + 0.0000005f;
    output[index] = value;
}

__global__ void combine_branches_kernel(const float *cpu_output,
                                        const float *gpu_output,
                                        float *combined, int elements) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements)
        return;
    combined[index] = cpu_output[index] + gpu_output[index];
}

struct Stats {
    double wall_ms = 0.0;
    double graph_ms = 0.0;
    double gpu_ms = 0.0;
    double cpu_ms = 0.0;
    double launch_us = 0.0;
    double hidden_ms = 0.0;
    int correct = 0;
    int checksum_failures = 0;
};

class SandboxGraph {
  public:
    SandboxGraph() = default;
    ~SandboxGraph() { reset(); }

    SandboxGraph(const SandboxGraph &) = delete;
    SandboxGraph &operator=(const SandboxGraph &) = delete;

    bool initialize(int device, int elements, int cpu_repeats, int gpu_work,
                    int gpu_repeats, bool branched) {
        reset();
        state_.elements = elements;
        state_.cpu_repeats = cpu_repeats;
        state_.gpu_work = gpu_work;
        state_.gpu_repeats = gpu_repeats;
        branched_ = branched;
        device_ = device;
        if (device < 0 || elements <= 0 || cpu_repeats < 0 || gpu_work <= 0 ||
            gpu_repeats < 0)
            return false;

        CUDA_OK(cudaSetDevice(device_));
        CUDA_OK(cudaMallocHost(
            &state_.host_input, static_cast<size_t>(elements) * sizeof(float)));
        CUDA_OK(cudaMallocHost(&state_.host_cpu_output,
                               static_cast<size_t>(elements) * sizeof(float)));
        CUDA_OK(cudaMallocHost(
            &state_.host_result, static_cast<size_t>(elements) * sizeof(float)));
        CUDA_OK(cudaMalloc(&state_.device_input,
                           static_cast<size_t>(elements) * sizeof(float)));
        CUDA_OK(cudaMalloc(&state_.device_cpu_output,
                           static_cast<size_t>(elements) * sizeof(float)));
        CUDA_OK(cudaMalloc(&state_.device_gpu_output,
                           static_cast<size_t>(gpu_work) * sizeof(float)));
        CUDA_OK(cudaMalloc(&state_.device_combined,
                           static_cast<size_t>(elements) * sizeof(float)));
        CUDA_OK(cudaStreamCreateWithFlags(&state_.stream,
                                          cudaStreamNonBlocking));
        CUDA_OK(cudaEventCreate(&state_.gpu_start));
        CUDA_OK(cudaEventCreate(&state_.gpu_end));
        CUDA_OK(cudaEventCreate(&state_.graph_start));
        CUDA_OK(cudaEventCreate(&state_.graph_end));
        return build_graph();
    }

    bool run(int iterations, Stats &stats) {
        if (!state_.exec || !state_.stream || iterations <= 0)
            return false;

        float *launch_input = nullptr;
        CUDA_OK(cudaMallocHost(
            &launch_input, static_cast<size_t>(state_.elements) * sizeof(float)));
        for (int iteration = 0; iteration < iterations; ++iteration) {
            state_.sequence = static_cast<uint64_t>(iteration + 1);
            for (int i = 0; i < state_.elements; ++i)
                launch_input[i] =
                    static_cast<float>((i % 97) - 48) * 0.01f +
                    static_cast<float>(iteration) * 0.25f;
            std::fill_n(state_.host_cpu_output, state_.elements,
                        std::numeric_limits<float>::quiet_NaN());

            const auto wall_start = Clock::now();
            CUDA_OK(cudaMemcpyAsync(
                state_.device_input, launch_input,
                static_cast<size_t>(state_.elements) * sizeof(float),
                cudaMemcpyHostToDevice, state_.stream));
            const auto launch_start = Clock::now();
            CUDA_OK(cudaEventRecord(state_.graph_start, state_.stream));
            CUDA_OK(cudaGraphLaunch(state_.exec, state_.stream));
            const auto launch_end = Clock::now();
            CUDA_OK(cudaEventRecord(state_.graph_end, state_.stream));
            CUDA_OK(cudaStreamSynchronize(state_.stream));
            const auto wall_end = Clock::now();

            float graph_ms = 0.0f;
            float gpu_ms = 0.0f;
            CUDA_OK(cudaEventElapsedTime(&graph_ms, state_.graph_start,
                                         state_.graph_end));
            CUDA_OK(cudaEventElapsedTime(&gpu_ms, state_.gpu_start,
                                         state_.gpu_end));
            const double cpu_ms = static_cast<double>(
                                      state_.host_callback_ns.load(
                                          std::memory_order_acquire)) /
                                  1.0e6;
            stats.wall_ms += std::chrono::duration<double, std::milli>(
                                  wall_end - wall_start)
                                  .count();
            stats.graph_ms += graph_ms;
            stats.gpu_ms += gpu_ms;
            stats.cpu_ms += cpu_ms;
            stats.launch_us += std::chrono::duration<double, std::micro>(
                                   launch_end - launch_start)
                                   .count();
            stats.hidden_ms += cpu_ms + static_cast<double>(gpu_ms) - graph_ms;

            CUDA_OK(cudaMemcpy(state_.host_result, state_.device_combined,
                               static_cast<size_t>(state_.elements) *
                                   sizeof(float),
                               cudaMemcpyDeviceToHost));
            bool correct = true;
            for (int i = 0; i < state_.elements; ++i) {
                const float expected = cpu_transform(
                    launch_input[i], state_.cpu_repeats, state_.sequence);
                const float gpu = gpu_transform(launch_input[i],
                                                state_.gpu_repeats);
                const float actual = state_.host_result[i];
                if (!std::isfinite(actual) ||
                    std::fabs(actual - (expected + gpu)) > kOutputTolerance) {
                    std::fprintf(stderr,
                                 "[GRAPH-SANDBOX] mismatch i=%d actual=%g "
                                 "expected=%g diff=%g\\n",
                                 i, actual, expected + gpu,
                                 std::fabs(actual - (expected + gpu)));
                    correct = false;
                    break;
                }
            }
            if (correct)
                ++stats.correct;
            else
                std::fprintf(stderr,
                             "[GRAPH-SANDBOX] stale/corrupt result at "
                             "iteration %d (%s graph)\n",
                             iteration, branched_ ? "branched" : "serial");
        }
        cudaFreeHost(launch_input);
        return true;
    }

    void reset() {
        if (device_ >= 0)
            cudaSetDevice(device_);
        if (state_.stream)
            cudaStreamSynchronize(state_.stream);
        if (state_.exec)
            cudaGraphExecDestroy(state_.exec);
        if (state_.graph)
            cudaGraphDestroy(state_.graph);
        if (state_.graph_start)
            cudaEventDestroy(state_.graph_start);
        if (state_.graph_end)
            cudaEventDestroy(state_.graph_end);
        if (state_.gpu_start)
            cudaEventDestroy(state_.gpu_start);
        if (state_.gpu_end)
            cudaEventDestroy(state_.gpu_end);
        if (state_.stream)
            cudaStreamDestroy(state_.stream);
        if (state_.device_combined)
            cudaFree(state_.device_combined);
        if (state_.device_gpu_output)
            cudaFree(state_.device_gpu_output);
        if (state_.device_cpu_output)
            cudaFree(state_.device_cpu_output);
        if (state_.device_input)
            cudaFree(state_.device_input);
        if (state_.host_result)
            cudaFreeHost(state_.host_result);
        if (state_.host_cpu_output)
            cudaFreeHost(state_.host_cpu_output);
        if (state_.host_input)
            cudaFreeHost(state_.host_input);
        state_.elements = 0;
        state_.cpu_repeats = 0;
        state_.gpu_work = 0;
        state_.gpu_repeats = 0;
        state_.sequence = 0;
        state_.host_input = nullptr;
        state_.host_cpu_output = nullptr;
        state_.host_result = nullptr;
        state_.device_input = nullptr;
        state_.device_cpu_output = nullptr;
        state_.device_gpu_output = nullptr;
        state_.device_combined = nullptr;
        state_.graph = nullptr;
        state_.exec = nullptr;
        state_.gpu_start = nullptr;
        state_.gpu_end = nullptr;
        state_.graph_start = nullptr;
        state_.graph_end = nullptr;
        state_.stream = nullptr;
        state_.host_callback_ns.store(0, std::memory_order_relaxed);
        device_ = -1;
        branched_ = false;
    }

  private:
    bool build_graph() {
        CUDA_OK(cudaGraphCreate(&state_.graph, 0));

        cudaMemcpy3DParms d2h{};
        d2h.srcPtr = make_cudaPitchedPtr(
            state_.device_input,
            static_cast<size_t>(state_.elements) * sizeof(float),
            static_cast<size_t>(state_.elements) * sizeof(float), 1);
        d2h.dstPtr = make_cudaPitchedPtr(
            state_.host_input,
            static_cast<size_t>(state_.elements) * sizeof(float),
            static_cast<size_t>(state_.elements) * sizeof(float), 1);
        d2h.extent = make_cudaExtent(
            static_cast<size_t>(state_.elements) * sizeof(float), 1, 1);
        d2h.kind = cudaMemcpyDeviceToHost;
        cudaGraphNode_t d2h_node = nullptr;
        CUDA_OK(cudaGraphAddMemcpyNode(&d2h_node, state_.graph, nullptr, 0,
                                       &d2h));

        cudaHostNodeParams host_params{};
        host_params.fn = cpu_host_node;
        host_params.userData = &state_;
        cudaGraphNode_t host_node = nullptr;
        CUDA_OK(cudaGraphAddHostNode(&host_node, state_.graph, &d2h_node, 1,
                                     &host_params));

        cudaMemcpy3DParms h2d{};
        h2d.srcPtr = make_cudaPitchedPtr(
            state_.host_cpu_output,
            static_cast<size_t>(state_.elements) * sizeof(float),
            static_cast<size_t>(state_.elements) * sizeof(float), 1);
        h2d.dstPtr = make_cudaPitchedPtr(
            state_.device_cpu_output,
            static_cast<size_t>(state_.elements) * sizeof(float),
            static_cast<size_t>(state_.elements) * sizeof(float), 1);
        h2d.extent = make_cudaExtent(
            static_cast<size_t>(state_.elements) * sizeof(float), 1, 1);
        h2d.kind = cudaMemcpyHostToDevice;
        cudaGraphNode_t h2d_node = nullptr;
        CUDA_OK(cudaGraphAddMemcpyNode(&h2d_node, state_.graph, &host_node, 1,
                                       &h2d));

        cudaGraphNode_t gpu_start_node = nullptr;
        if (branched_) {
            CUDA_OK(cudaGraphAddEventRecordNode(&gpu_start_node, state_.graph,
                                                nullptr, 0, state_.gpu_start));
        } else {
            CUDA_OK(cudaGraphAddEventRecordNode(&gpu_start_node, state_.graph,
                                                &h2d_node, 1, state_.gpu_start));
        }

        cudaKernelNodeParams gpu_kernel{};
        gpu_kernel.func = reinterpret_cast<void *>(gpu_hot_branch_kernel);
        void *gpu_args[] = {&state_.device_input, &state_.device_gpu_output,
                            &state_.elements, &state_.gpu_work,
                            &state_.gpu_repeats};
        gpu_kernel.gridDim = dim3((state_.gpu_work + 255) / 256, 1, 1);
        gpu_kernel.blockDim = dim3(256, 1, 1);
        gpu_kernel.kernelParams = gpu_args;
        cudaGraphNode_t gpu_kernel_node = nullptr;
        CUDA_OK(cudaGraphAddKernelNode(&gpu_kernel_node, state_.graph,
                                       &gpu_start_node, 1, &gpu_kernel));

        cudaGraphNode_t gpu_end_node = nullptr;
        CUDA_OK(cudaGraphAddEventRecordNode(&gpu_end_node, state_.graph,
                                            &gpu_kernel_node, 1,
                                            state_.gpu_end));

        cudaKernelNodeParams combine{};
        combine.func = reinterpret_cast<void *>(combine_branches_kernel);
        void *combine_args[] = {&state_.device_cpu_output,
                                &state_.device_gpu_output,
                                &state_.device_combined, &state_.elements};
        combine.gridDim = dim3((state_.elements + 255) / 256, 1, 1);
        combine.blockDim = dim3(256, 1, 1);
        combine.kernelParams = combine_args;
        const cudaGraphNode_t dependencies[] = {h2d_node, gpu_end_node};
        cudaGraphNode_t combine_node = nullptr;
        CUDA_OK(cudaGraphAddKernelNode(&combine_node, state_.graph,
                                       dependencies, 2, &combine));
        CUDA_OK(cudaGraphInstantiate(&state_.exec, state_.graph, 0));
        return true;
    }

    GraphState state_{};
    int device_ = -1;
    bool branched_ = false;
};

void print_stats(const char *name, const Stats &stats, int iterations) {
    const double divisor = static_cast<double>(std::max(1, iterations));
    std::printf(
        "[GRAPH-SANDBOX] %s iterations=%d correct=%d wall=%.3f ms "
        "(%.3f/replay) graph=%.3f ms (%.3f/replay) gpu=%.3f ms "
        "(%.3f/replay) cpu_host=%.3f ms (%.3f/replay) launch=%.3f us/replay "
        "hidden=%.3f ms/replay\n",
        name, iterations, stats.correct, stats.wall_ms,
        stats.wall_ms / divisor, stats.graph_ms, stats.graph_ms / divisor,
        stats.gpu_ms, stats.gpu_ms / divisor, stats.cpu_ms,
        stats.cpu_ms / divisor, stats.launch_us / divisor,
        stats.hidden_ms / divisor);
}

int parse_positive(const char *value, int fallback, int max_value) {
    if (!value || !value[0])
        return fallback;
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed < 1 || parsed > max_value)
        return fallback;
    return static_cast<int>(parsed);
}

} // namespace

int main(int argc, char **argv) {
    const int iterations = argc > 1
                               ? parse_positive(argv[1], kDefaultIterations, 10000)
                               : kDefaultIterations;
    const int elements = argc > 2
                             ? parse_positive(argv[2], kDefaultElements, 1 << 20)
                             : kDefaultElements;
    const int cpu_repeats = argc > 3
                                ? parse_positive(argv[3], kDefaultCpuRepeats,
                                                1000000)
                                : kDefaultCpuRepeats;
    const int gpu_work = argc > 4
                             ? parse_positive(argv[4], kDefaultGpuWork, 1 << 24)
                             : kDefaultGpuWork;
    const int gpu_repeats = argc > 5
                                ? parse_positive(argv[5], kDefaultGpuRepeats,
                                                1000000)
                                : kDefaultGpuRepeats;

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
        std::fprintf(stderr, "[GRAPH-SANDBOX] no CUDA device available\n");
        return 2;
    }
    if (cudaSetDevice(0) != cudaSuccess) {
        std::fprintf(stderr, "[GRAPH-SANDBOX] cudaSetDevice failed\n");
        return 1;
    }
    std::printf(
        "[GRAPH-SANDBOX] iterations=%d elements=%d cpu_repeats=%d "
        "gpu_work=%d gpu_repeats=%d\n",
        iterations, elements, cpu_repeats, gpu_work, gpu_repeats);
    std::printf("[GRAPH-SANDBOX] topology=Branch-A(GPU hot) + "
                "Branch-B(D2H -> sync host node -> H2D) -> "
                "Branch-C(combine)\n");

    SandboxGraph serial;
    if (!serial.initialize(0, elements, cpu_repeats, gpu_work, gpu_repeats,
                           false))
        return 1;
    Stats serial_stats;
    if (!serial.run(iterations, serial_stats))
        return 1;
    print_stats("serialized-control", serial_stats, iterations);

    SandboxGraph branched;
    if (!branched.initialize(0, elements, cpu_repeats, gpu_work, gpu_repeats,
                             true))
        return 1;
    Stats branched_stats;
    if (!branched.run(iterations, branched_stats))
        return 1;
    print_stats("branched-sync-host-node", branched_stats, iterations);

    const bool pass = serial_stats.correct == iterations &&
                      branched_stats.correct == iterations;
    const double serial_per_replay =
        serial_stats.graph_ms / static_cast<double>(iterations);
    const double branched_per_replay =
        branched_stats.graph_ms / static_cast<double>(iterations);
    const bool overlap_observed = branched_per_replay < serial_per_replay;
    std::printf("[GRAPH-SANDBOX] overlap_observed=%s serial_graph=%.3f "
                "branched_graph=%.3f\n",
                overlap_observed ? "yes" : "no", serial_per_replay,
                branched_per_replay);
    std::printf("[GRAPH-SANDBOX] %s\n",
                pass && overlap_observed ? "PASS" : "FAIL");
    return pass && overlap_observed ? 0 : 1;
}
