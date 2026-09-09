// bench_register_parallel.cu — standalone, isolated microbenchmark.
//
// Answers ONE empirical question, directly, before any production code is
// touched: does cudaHostRegister/cudaHostUnregister genuinely parallelize
// across multiple CPU threads registering DIFFERENT host buffers
// concurrently, or does the CUDA driver serialize these calls internally
// (e.g. via a driver-wide or context-wide lock on page-table/DMA-mapping
// bookkeeping)? This determines whether a multi-worker-thread design for
// upload_expert_active_async's register+copy+deferred-unregister path (see
// CLAUDE.md's 2026-08-28 "find where the thread is actually blocked"
// investigation) is worth building, or whether a single dedicated worker
// thread (still real, still gets the work off the decode thread, just no
// N-way speedup on the register/unregister calls themselves) is the right
// shape instead.
//
// Real numbers this test is calibrated against (this project's own live
// counters, upload_expert_async's Alt-A conversion, same call shape):
//   register:   ~0.918 ms/call
//   unregister: ~0.449 ms/call
//   (combined ~1.367 ms/call, real slab size ~15.1 MB)
//
// Four conditions, all timed via std::chrono::high_resolution_clock wall-
// clock around the relevant span (this project's own established
// convention -- no cudaEvent-based timing infra exists in this repo to
// reuse for a standalone benchmark):
//
//   A. REGISTER sequential   -- N cudaHostRegister calls, one thread, back
//                                to back. Baseline: N x per-call cost.
//   B. REGISTER concurrent   -- N threads, each registers its OWN buffer,
//                                launched via a barrier so they start at
//                                the same moment. If genuinely parallel,
//                                wall time approaches ONE call's cost, not
//                                N x. If serialized internally, wall time
//                                approaches the same N x total as A.
//   C. UNREGISTER sequential / concurrent -- same shape, for the deferred-
//                                unregister half of the pattern.
//   D. FULL PATTERN (the realistic end-to-end case) -- register -> issue
//      cudaMemcpyAsync H2D -> cudaEventRecord -> (benchmark-only)
//      cudaEventSynchronize -> cudaHostUnregister, sequential vs
//      concurrent (N worker threads, each owning one CUDA stream + one
//      buffer, targeting GPU 0 -- the real active-layer pool's GPU).
//
// Build (matches this project's own standalone-.cu convention, no CMake
// integration needed for a throwaway diagnostic):
//   nvcc -std=c++17 -O2 -arch=sm_86 tests/bench_register_parallel.cu \
//       -o /tmp/bench_register_parallel -lpthread
//   /tmp/bench_register_parallel
//
// This file is NOT wired into the build system and touches no production
// code -- delete freely once its question is answered.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <vector>
#include <atomic>
#include <barrier>

static void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(err));
        exit(1);
    }
}

using Clock = std::chrono::high_resolution_clock;
static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// Real active-layer expert slab size, confirmed from this project's own
// real allocation log ("Active layer area allocated: 256 experts (3.77487
// GB)" -> 3.77487*1024/256 MB).
static constexpr size_t SLAB_BYTES = (size_t)(15.099 * 1024 * 1024);

int main() {
    check(cudaSetDevice(0), "cudaSetDevice(0)");
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    printf("=== bench_register_parallel: GPU 0 = %s ===\n", prop.name);
    printf("slab size: %.3f MB\n\n", SLAB_BYTES / 1e6);

    const int N_VALUES[] = {2, 4, 8};

    for (int N : N_VALUES) {
        printf("---------- N = %d ----------\n", N);
        std::vector<void*> bufs(N);
        for (int i = 0; i < N; ++i) {
            bufs[i] = malloc(SLAB_BYTES);
            memset(bufs[i], 0x42, SLAB_BYTES);  // touch pages, avoid lazy-alloc skew
        }

        // ---- A. REGISTER sequential ----
        {
            auto t0 = Clock::now();
            for (int i = 0; i < N; ++i) {
                check(cudaHostRegister(bufs[i], SLAB_BYTES, cudaHostRegisterDefault), "register(seq)");
            }
            double total_ms = ms_since(t0);
            printf("A. REGISTER sequential:   %8.3f ms total  (%6.3f ms/call avg)\n",
                   total_ms, total_ms / N);
            for (int i = 0; i < N; ++i) check(cudaHostUnregister(bufs[i]), "unregister(cleanup-A)");
        }

        // ---- B. REGISTER concurrent (N threads, barrier-synchronized start) ----
        {
            std::barrier sync_point(N);
            std::vector<double> per_thread_ms(N, 0.0);
            std::vector<std::thread> threads;
            auto t0 = Clock::now();
            for (int i = 0; i < N; ++i) {
                threads.emplace_back([&, i]() {
                    sync_point.arrive_and_wait();
                    auto tstart = Clock::now();
                    cudaError_t e = cudaHostRegister(bufs[i], SLAB_BYTES, cudaHostRegisterDefault);
                    per_thread_ms[i] = ms_since(tstart);
                    if (e != cudaSuccess) {
                        fprintf(stderr, "thread %d register failed: %s\n", i, cudaGetErrorString(e));
                    }
                });
            }
            for (auto& t : threads) t.join();
            double wall_ms = ms_since(t0);
            double max_thread_ms = 0, sum_thread_ms = 0;
            for (double v : per_thread_ms) { max_thread_ms = std::max(max_thread_ms, v); sum_thread_ms += v; }
            printf("B. REGISTER concurrent:   %8.3f ms wall   (per-thread avg %6.3f ms, max %6.3f ms, "
                   "sum-if-serial-would-be %6.3f ms)\n",
                   wall_ms, sum_thread_ms / N, max_thread_ms, sum_thread_ms);
            printf("   -> speedup vs sequential-equivalent: %.2fx  (%.0f%% of ideal N-way)\n",
                   sum_thread_ms / wall_ms, 100.0 * (sum_thread_ms / wall_ms) / N);
            for (int i = 0; i < N; ++i) check(cudaHostUnregister(bufs[i]), "unregister(cleanup-B)");
        }

        // ---- C. UNREGISTER sequential vs concurrent ----
        {
            for (int i = 0; i < N; ++i) check(cudaHostRegister(bufs[i], SLAB_BYTES, cudaHostRegisterDefault), "register(pre-C-seq)");
            auto t0 = Clock::now();
            for (int i = 0; i < N; ++i) check(cudaHostUnregister(bufs[i]), "unregister(seq)");
            double total_ms = ms_since(t0);
            printf("C1. UNREGISTER sequential: %8.3f ms total  (%6.3f ms/call avg)\n",
                   total_ms, total_ms / N);
        }
        {
            for (int i = 0; i < N; ++i) check(cudaHostRegister(bufs[i], SLAB_BYTES, cudaHostRegisterDefault), "register(pre-C-conc)");
            std::barrier sync_point(N);
            std::vector<double> per_thread_ms(N, 0.0);
            std::vector<std::thread> threads;
            auto t0 = Clock::now();
            for (int i = 0; i < N; ++i) {
                threads.emplace_back([&, i]() {
                    sync_point.arrive_and_wait();
                    auto tstart = Clock::now();
                    cudaError_t e = cudaHostUnregister(bufs[i]);
                    per_thread_ms[i] = ms_since(tstart);
                    if (e != cudaSuccess) {
                        fprintf(stderr, "thread %d unregister failed: %s\n", i, cudaGetErrorString(e));
                    }
                });
            }
            for (auto& t : threads) t.join();
            double wall_ms = ms_since(t0);
            double sum_thread_ms = 0;
            for (double v : per_thread_ms) sum_thread_ms += v;
            printf("C2. UNREGISTER concurrent: %8.3f ms wall   (sum-if-serial-would-be %6.3f ms) "
                   "-> speedup %.2fx (%.0f%% of ideal N-way)\n",
                   wall_ms, sum_thread_ms, sum_thread_ms / wall_ms, 100.0 * (sum_thread_ms / wall_ms) / N);
        }

        // ---- D. FULL PATTERN: register -> memcpyAsync -> event -> sync -> unregister ----
        // Simulates the real active-layer upload shape. Uses per-thread
        // dedicated streams (mirrors production's own per-GPU copy_stream/
        // active_layer_stream discipline -- never share one stream across
        // concurrent issuers). Device buffer is a single shared allocation
        // sliced into N regions (avoids VRAM pressure from N separate
        // cudaMalloc calls skewing the timing).
        {
            void* d_base = nullptr;
            check(cudaMalloc(&d_base, SLAB_BYTES * N), "cudaMalloc(D)");

            // D1. sequential (today's real shape: register, copy, event, sync, unregister -- one at a time)
            {
                cudaStream_t stream;
                check(cudaStreamCreate(&stream), "streamCreate(D1)");
                auto t0 = Clock::now();
                for (int i = 0; i < N; ++i) {
                    check(cudaHostRegister(bufs[i], SLAB_BYTES, cudaHostRegisterDefault), "register(D1)");
                    void* dst = (char*)d_base + (size_t)i * SLAB_BYTES;
                    check(cudaMemcpyAsync(dst, bufs[i], SLAB_BYTES, cudaMemcpyHostToDevice, stream), "memcpyAsync(D1)");
                    check(cudaStreamSynchronize(stream), "sync(D1)");
                    check(cudaHostUnregister(bufs[i]), "unregister(D1)");
                }
                double total_ms = ms_since(t0);
                printf("D1. FULL sequential:       %8.3f ms total  (%6.3f ms/call avg)\n",
                       total_ms, total_ms / N);
                cudaStreamDestroy(stream);
            }

            // D2. concurrent (N worker threads, each owns a stream + does register->copy->sync->unregister for its own buffer)
            {
                std::vector<cudaStream_t> streams(N);
                for (int i = 0; i < N; ++i) check(cudaStreamCreate(&streams[i]), "streamCreate(D2)");
                std::barrier sync_point(N);
                std::vector<double> per_thread_ms(N, 0.0);
                std::vector<std::thread> threads;
                auto t0 = Clock::now();
                for (int i = 0; i < N; ++i) {
                    threads.emplace_back([&, i]() {
                        check(cudaSetDevice(0), "setDevice(D2 worker)");
                        sync_point.arrive_and_wait();
                        auto tstart = Clock::now();
                        cudaHostRegister(bufs[i], SLAB_BYTES, cudaHostRegisterDefault);
                        void* dst = (char*)d_base + (size_t)i * SLAB_BYTES;
                        cudaMemcpyAsync(dst, bufs[i], SLAB_BYTES, cudaMemcpyHostToDevice, streams[i]);
                        cudaStreamSynchronize(streams[i]);
                        cudaHostUnregister(bufs[i]);
                        per_thread_ms[i] = ms_since(tstart);
                    });
                }
                for (auto& t : threads) t.join();
                double wall_ms = ms_since(t0);
                double sum_thread_ms = 0;
                for (double v : per_thread_ms) sum_thread_ms += v;
                printf("D2. FULL concurrent:       %8.3f ms wall   (sum-if-serial-would-be %6.3f ms) "
                       "-> speedup %.2fx (%.0f%% of ideal N-way)\n",
                       wall_ms, sum_thread_ms, sum_thread_ms / wall_ms, 100.0 * (sum_thread_ms / wall_ms) / N);
                for (auto s : streams) cudaStreamDestroy(s);
            }
            cudaFree(d_base);
        }

        for (int i = 0; i < N; ++i) free(bufs[i]);
        printf("\n");
    }

    printf("Done.\n");
    return 0;
}
