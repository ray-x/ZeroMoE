#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

[[noreturn]] void die(const char *what, cudaError_t err) {
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
    std::exit(2);
}

void check(const char *what, cudaError_t err) {
    if (err != cudaSuccess) die(what, err);
}

struct Args {
    std::size_t expert_bytes = 1774080; // Qwen log: legacy staging stride.
    int iters = 300;
    int warmup = 20;
    int max_group = 10;
};

Args parse(int argc, char **argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        auto next = [&](const char *name) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (!std::strcmp(argv[i], "--expert-bytes"))
            a.expert_bytes = std::strtoull(next(argv[i]), nullptr, 0);
        else if (!std::strcmp(argv[i], "--iters"))
            a.iters = std::atoi(next(argv[i]));
        else if (!std::strcmp(argv[i], "--max-group"))
            a.max_group = std::atoi(next(argv[i]));
        else {
            std::fprintf(stderr, "unknown option: %s\n", argv[i]);
            std::exit(2);
        }
    }
    if (!a.expert_bytes || a.iters < 1 || a.max_group < 1) std::exit(2);
    return a;
}

double elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    check("cudaEventElapsedTime", cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

void report(const char *name, std::size_t bytes, int iters, double ms) {
    const double seconds = ms / 1000.0;
    const double gb = (double)bytes * (double)iters / 1.0e9;
    std::printf("%-22s %8.3f ms/group  %8.2f GB/s  (%zu bytes, %d iterations)\n",
                name, ms / iters, gb / seconds, bytes, iters);
}

} // namespace

int main(int argc, char **argv) {
    const Args args = parse(argc, argv);
    int device = 0;
    check("cudaGetDevice", cudaGetDevice(&device));
    cudaDeviceProp prop{};
    check("cudaGetDeviceProperties", cudaGetDeviceProperties(&prop, device));
    std::printf("device=%d %s\n", device, prop.name);
    std::printf("expert_bytes=%zu max_group=%d\n", args.expert_bytes,
                args.max_group);

    std::uint8_t *host = nullptr;
    const std::size_t max_bytes = args.expert_bytes * (std::size_t)args.max_group;
    check("cudaMallocHost", cudaMallocHost((void **)&host, max_bytes));
    for (std::size_t i = 0; i < max_bytes; ++i)
        host[i] = (std::uint8_t)(i * 131u + 17u);

    std::uint8_t *d_packed = nullptr;
    std::uint8_t *d_gate = nullptr;
    std::uint8_t *d_up = nullptr;
    std::uint8_t *d_down = nullptr;
    check("cudaMalloc packed", cudaMalloc((void **)&d_packed, max_bytes));
    const std::size_t gate_bytes = args.expert_bytes / 3;
    const std::size_t up_bytes = args.expert_bytes / 3;
    const std::size_t down_bytes = args.expert_bytes - gate_bytes - up_bytes;
    check("cudaMalloc gate", cudaMalloc((void **)&d_gate,
                                         gate_bytes * args.max_group));
    check("cudaMalloc up", cudaMalloc((void **)&d_up,
                                       up_bytes * args.max_group));
    check("cudaMalloc down", cudaMalloc((void **)&d_down,
                                         down_bytes * args.max_group));

    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    check("cudaStreamCreate", cudaStreamCreateWithFlags(&stream,
                                                         cudaStreamNonBlocking));
    check("cudaEventCreate start", cudaEventCreate(&start));
    check("cudaEventCreate stop", cudaEventCreate(&stop));

    auto run = [&](const char *name, int group, bool packed) {
        const std::size_t total = args.expert_bytes * (std::size_t)group;
        for (int i = 0; i < args.warmup; ++i) {
            if (packed) {
                check("warmup packed", cudaMemcpyAsync(
                    d_packed, host, total, cudaMemcpyHostToDevice, stream));
            } else {
                for (int e = 0; e < group; ++e) {
                    const std::uint8_t *src = host + (std::size_t)e * args.expert_bytes;
                    check("warmup gate", cudaMemcpyAsync(
                        d_gate + (std::size_t)e * gate_bytes, src, gate_bytes,
                        cudaMemcpyHostToDevice, stream));
                    check("warmup up", cudaMemcpyAsync(
                        d_up + (std::size_t)e * up_bytes, src + gate_bytes,
                        up_bytes, cudaMemcpyHostToDevice, stream));
                    check("warmup down", cudaMemcpyAsync(
                        d_down + (std::size_t)e * down_bytes,
                        src + gate_bytes + up_bytes, down_bytes,
                        cudaMemcpyHostToDevice, stream));
                }
            }
        }
        check("warmup synchronize", cudaStreamSynchronize(stream));
        check("event start", cudaEventRecord(start, stream));
        for (int i = 0; i < args.iters; ++i) {
            if (packed) {
                check("packed copy", cudaMemcpyAsync(
                    d_packed, host, total, cudaMemcpyHostToDevice, stream));
            } else {
                for (int e = 0; e < group; ++e) {
                    const std::uint8_t *src = host + (std::size_t)e * args.expert_bytes;
                    check("gate copy", cudaMemcpyAsync(
                        d_gate + (std::size_t)e * gate_bytes, src, gate_bytes,
                        cudaMemcpyHostToDevice, stream));
                    check("up copy", cudaMemcpyAsync(
                        d_up + (std::size_t)e * up_bytes, src + gate_bytes,
                        up_bytes, cudaMemcpyHostToDevice, stream));
                    check("down copy", cudaMemcpyAsync(
                        d_down + (std::size_t)e * down_bytes,
                        src + gate_bytes + up_bytes, down_bytes,
                        cudaMemcpyHostToDevice, stream));
                }
            }
        }
        check("event stop", cudaEventRecord(stop, stream));
        check("stop synchronize", cudaEventSynchronize(stop));
        report(name, total, args.iters, elapsed_ms(start, stop));
    };

    for (int group : {1, 3, 5, 10}) {
        if (group > args.max_group) continue;
        char name[64];
        std::snprintf(name, sizeof(name), "planar %d expert(s)", group);
        run(name, group, false);
        std::snprintf(name, sizeof(name), "packed %d expert(s)", group);
        run(name, group, true);
    }

    check("cudaFree d_down", cudaFree(d_down));
    check("cudaFree d_up", cudaFree(d_up));
    check("cudaFree d_gate", cudaFree(d_gate));
    check("cudaFree d_packed", cudaFree(d_packed));
    check("cudaFreeHost", cudaFreeHost(host));
    check("cudaEventDestroy start", cudaEventDestroy(start));
    check("cudaEventDestroy stop", cudaEventDestroy(stop));
    check("cudaStreamDestroy", cudaStreamDestroy(stream));
    return 0;
}

