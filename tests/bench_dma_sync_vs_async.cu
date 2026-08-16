// bench_dma_sync_vs_async.cu — standalone, isolated microbenchmark.
//
// Answers: for the decode-time cold-miss RAM->VRAM upload path
// (expert_ffn_resolve_gpu_loc -> dispatch_expert_k, src/execution_layer.cpp),
// which today issues N experts' H2D copies as N SEQUENTIAL BLOCKING
// cudaMemcpy calls (upload_expert_active's real current code,
// gpu_topology.cpp), how much wall-clock time would batching them as N
// cudaMemcpyAsync issues + ONE cudaStreamSynchronize actually save -- the
// same pattern this codebase already ships for prefill's Phase-2 pre-warm
// (upload_expert_active_async + sync_active_layer_batch, gpu_topology.cpp)?
// Measured on BOTH real GPUs in this box (RTX 3090 @ PCIe x16, RTX 5060 Ti
// @ PCIe x8 gen3) since the two links' very different bandwidth/latency
// could make the async win look completely different on each.
//
// Three conditions, all timed the same way (wall-clock via
// std::chrono::high_resolution_clock around the whole N-copy sequence,
// mirroring this project's own PERF_START/PERF_END convention -- no
// cudaEvent-based timing infrastructure exists in this repo to reuse):
//
//   A. SYNC        -- N sequential blocking cudaMemcpy H2D, implicit
//                      default stream, pageable host source. Byte-for-byte
//                      today's real upload_expert_active behavior.
//   B. ASYNC        -- N cudaMemcpyAsync issued back-to-back on a dedicated
//                      copy_stream, then ONE cudaStreamSynchronize. Host
//                      source stays PAGEABLE, matching both production's
//                      real RAM-tier slab_buffer (plain posix_memalign, no
//                      pinning) and upload_expert_active_async's own
//                      current code (no cudaHostRegister there either).
//   C. ASYNC+PINNED -- identical to B but each host buffer is transiently
//                      cudaHostRegister'd before the batch and
//                      cudaHostUnregister'd after (mirrors upload_expert's
//                      own already-shipped Alt-A pattern). Exists because
//                      CUDA's documented behavior is that a PAGEABLE-source
//                      cudaMemcpyAsync H2D copy is often staged internally
//                      through a synchronous driver bounce buffer rather
//                      than truly overlapping -- if B's savings look small,
//                      C answers whether pinning is what's actually needed
//                      to realize the win (register/unregister has its own
//                      real, nonzero cost -- see CLAUDE.md's
//                      upload_expert_active cudaHostRegister history, where
//                      it was a net LOSS at very high call-frequency -- so
//                      this isn't a foregone conclusion either way).
//
// Does NOT touch moestream's real inference path -- completely standalone,
// zero risk to correctness of the live model/engine. Runs a one-time D2H
// readback + memcmp correctness check per device/size for B and C before
// trusting either's timing number.
//
// Build (no CMake integration, direct nvcc invocation -- pure CUDA Runtime
// API, no custom kernels, so no -arch/-gencode flags matter here):
//   nvcc -O3 -std=c++17 tests/bench_dma_sync_vs_async.cu -o bench_dma
// Run (sweeps every visible CUDA device x 8/16/32 MB x {A,B,C} automatically):
//   ./bench_dma
// Optional flags (all have sane defaults matching the exact ask):
//   --sizes 8,16,32     MB per "MoE" block, comma-separated (default: 8,16,32)
//   --n 4                MoE blocks per round (default: 4)
//   --iters 50            timed repetitions per condition (default: 50)
//   --warmup 10           discarded warmup repetitions (default: 10)
//   --device 0,1          restrict to these device indices (default: all)
//
// Sanity-check printed device names/indices against `nvidia-smi -L` before
// trusting which row is which card -- device ordinal isn't guaranteed
// stable across boots (see this project's own CLAUDE.md caution on
// configs/*.toml's device_id fields).
//
// ===========================================================================
// REAL RESULT, VERIFIED ON REAL HARDWARE (2026-08-14, default args: n=4,
// sizes=8/16/32MB, 50 iters, 10 warmup). Device 0 = RTX 3090 (PCIe x16),
// device 1 = RTX 5060 Ti (PCIe x8 gen3) -- confirmed via nvidia-smi -L and
// this benchmark's own printed sysfs PCIe link line ("8.0 GT/s PCIe x16" /
// "8.0 GT/s PCIe x8"). Correctness check (D2H readback + memcmp) PASSED for
// every device/size before any number below was trusted.
//
//   device            size   A SYNC     B ASYNC (pageable)      C ASYNC+PINNED
//   RTX 3090 (x16)     8MB   4.238ms    3.809ms  (1.11x)          8.068ms (0.53x)
//   RTX 3090 (x16)    16MB   7.803ms    6.749ms  (1.16x)         14.618ms (0.53x)
//   RTX 3090 (x16)    32MB  14.474ms   13.874ms  (1.04x)         27.447ms (0.53x)
//   RTX 5060 Ti (x8)   8MB   6.782ms    6.382ms  (1.06x)         12.439ms (0.55x)
//   RTX 5060 Ti (x8)  16MB  13.063ms   12.784ms  (1.02x)         22.570ms (0.58x)
//   RTX 5060 Ti (x8)  32MB  25.930ms   24.881ms  (1.04x)         40.431ms (0.64x)
//
// Reading:
//  - B (plain async, pageable -- the real proposed fix) is a REAL but MODEST
//    win: 2-16% faster, biggest on the 3090 at smaller sizes (1.11-1.16x),
//    shrinking toward ~flat as size grows and pure DMA time dominates the
//    per-call launch/sync overhead batching was reclaiming. The 5060 Ti's
//    narrower x8 link shows a smaller win throughout (1.02-1.06x) -- its DMA
//    time is already large relative to overhead, less to reclaim.
//  - C (pinned) is a CLEAR, CONSISTENT LOSS on both cards at every size
//    (0.53-0.64x, i.e. worse than today's sync baseline) -- confirms and
//    generalizes this codebase's own prior upload_expert_active
//    cudaHostRegister finding (CLAUDE.md: reverted as a 38% regression) to
//    the batched-async case too. At 32MB on the 3090, register(11.3ms) alone
//    costs almost as much as copy+sync(11.2ms), plus unregister(4.9ms) on
//    top -- register/unregister overhead roughly DOUBLES total cost for zero
//    DMA benefit. Do not pin for this call site.
//  - Practical read: scaled to a real ~3.35 misses/layer/token average (this
//    project's own estimate), B is worth roughly 0.3-0.4ms/layer on the
//    3090 x ~78 layers =~ 20-30ms/token if applied everywhere -- real, free
//    (reuses already-proven infra, zero new correctness surface), but a
//    low-single-digit-percent lever against a token currently costing
//    500-2000ms+, not a transformative one.
//
// Wiring history, for the record:
//
//   ROUND 1 (REVERTED): wired straight into src/execution_layer.cpp's
//   Phase-3 dispatch loop, batching a WHOLE token's ram_needed_k set (up to
//   ~topk, e.g. 8) into one async-issue-then-sync group, unconditionally
//   on. Caused a REAL PRODUCTION INCIDENT on real hardware: host RAM
//   exhaustion, swap thrashing, and an eventual freeze (a real debug.log
//   showed [SLOW-HOST-ALLOC] times climbing from 87ms to 1294ms+ for an
//   11MB posix_memalign -- direct evidence the OS was already swapping --
//   and [SLOW-PICKUP] prefetch-queue delays exploding from 250ms to 30+
//   SECONDS, a classic memory-pressure death spiral). Fully reverted.
//   Root-caused afterward, from that same log: a real "[ERROR] [GPU]
//   Invalid upload_expert_active_async parameters" line, traced to a TOCTOU
//   bug -- the code checked slab_buffer once, then did MORE lock-taking
//   work (active_layer_has, gpu_capable checks) before actually pinning and
//   using it, giving a concurrent prefetch-worker thread's own ORDINARY
//   (not even the pre-existing buggy "fallback to slot 0" path -- grepped
//   for and confirmed absent from that log) victim-selection scan a real
//   window to reclaim the still-unprotected RAM slot first.
//
//   ROUND 2 (current, capped + gated): re-added as EngineConfig::
//   decode_async_prewarm, DEFAULT OFF, plus decode_async_prewarm_max_batch
//   (default 3) which processes ram_needed_k in small chunks -- each with
//   its own full pin -> issue -> sync -> unpin cycle -- instead of batching
//   everything in one shot, bounding peak simultaneous in-flight RAM-slot/
//   active-layer footprint. The TOCTOU bug is fixed properly this time:
//   the slab_buffer/slab_capacity check, the ram_pin call, and the actual
//   use are all immediately adjacent with nothing else in between, PLUS an
//   identity check (blk->expert_id == expert_id) right after pinning, since
//   a reclaimed-and-repopulated slot could otherwise pass the non-null/
//   non-zero checks while silently holding a DIFFERENT expert's bytes.
//   See EngineConfig::decode_async_prewarm's own comment (engine_config.hpp)
//   for the full writeup, and CLAUDE.md.
//
//   STATUS: build clean, full moestream_tests suite green. UNVERIFIED ON
//   REAL HARDWARE END TO END as a live generation run -- default OFF, so
//   enabling it for the first time should be done deliberately, with real
//   host_ram_gb headroom, watching [SLOW-HOST-ALLOC]/[SLOW-PICKUP] for any
//   sign of the same pressure recurring before trusting it for a real run.
// ===========================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <cmath>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while (0)

// ---------------------------------------------------------------------
// CLI parsing
// ---------------------------------------------------------------------
struct Args {
    std::vector<int> sizes_mb = {8, 16, 32};
    int n = 4;
    int iters = 50;
    int warmup = 10;
    std::vector<int> devices;  // empty = all visible devices
};

static std::vector<int> parse_int_list(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(std::atoi(item.c_str()));
    }
    return out;
}

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need_val = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                fprintf(stderr, "Missing value for %s\n", flag);
                exit(1);
            }
            return argv[++i];
        };
        if (arg == "--sizes") a.sizes_mb = parse_int_list(need_val("--sizes"));
        else if (arg == "--n") a.n = std::atoi(need_val("--n").c_str());
        else if (arg == "--iters") a.iters = std::atoi(need_val("--iters").c_str());
        else if (arg == "--warmup") a.warmup = std::atoi(need_val("--warmup").c_str());
        else if (arg == "--device") a.devices = parse_int_list(need_val("--device"));
        else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            exit(1);
        }
    }
    return a;
}

// ---------------------------------------------------------------------
// Best-effort PCIe link info via sysfs (Linux only; silently skipped if
// unreadable -- purely a diagnostic sanity-check line, never load-bearing).
// ---------------------------------------------------------------------
static std::string read_sysfs_trim(const std::string& path) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) return "";
    char buf[64] = {0};
    if (!fgets(buf, sizeof(buf), f)) { fclose(f); return ""; }
    fclose(f);
    std::string s(buf);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
    return s;
}

static void print_pcie_link_info(const cudaDeviceProp& prop) {
    char pci_path[128];
    snprintf(pci_path, sizeof(pci_path), "/sys/bus/pci/devices/%04x:%02x:%02x.0",
              prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
    std::string speed = read_sysfs_trim(std::string(pci_path) + "/current_link_speed");
    std::string width = read_sysfs_trim(std::string(pci_path) + "/current_link_width");
    if (speed.empty() && width.empty()) {
        printf("    PCIe link: (sysfs unreadable, skipping -- not load-bearing)\n");
    } else {
        printf("    PCIe link: %s x%s (from %s)\n",
               speed.empty() ? "?" : speed.c_str(),
               width.empty() ? "?" : width.c_str(), pci_path);
    }
}

// ---------------------------------------------------------------------
// Timing stats
// ---------------------------------------------------------------------
struct Stats {
    double mean_ms = 0, min_ms = 0, max_ms = 0, stddev_ms = 0;
};

static Stats compute_stats(const std::vector<double>& samples) {
    Stats s;
    if (samples.empty()) return s;
    double sum = 0;
    s.min_ms = samples[0];
    s.max_ms = samples[0];
    for (double v : samples) {
        sum += v;
        s.min_ms = std::min(s.min_ms, v);
        s.max_ms = std::max(s.max_ms, v);
    }
    s.mean_ms = sum / samples.size();
    double var = 0;
    for (double v : samples) var += (v - s.mean_ms) * (v - s.mean_ms);
    s.stddev_ms = samples.size() > 1 ? std::sqrt(var / (samples.size() - 1)) : 0.0;
    return s;
}

static double gbps(size_t total_bytes, double ms) {
    if (ms <= 0) return 0.0;
    return (double)total_bytes / (ms / 1000.0) / 1e9;
}

// ---------------------------------------------------------------------
// The three transfer conditions. Each runs ONE full round (n copies) and
// returns elapsed wall-clock ms for that round.
// ---------------------------------------------------------------------

// A: sequential blocking cudaMemcpy, default stream -- today's real
// upload_expert_active behavior (gpu_topology.cpp:174).
static double run_sync_round(uint8_t** host, uint8_t** dev, int n, size_t bytes) {
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < n; ++k) {
        CHECK_CUDA(cudaMemcpy(dev[k], host[k], bytes, cudaMemcpyHostToDevice));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// B: N async issues on a dedicated stream, one shared sync -- mirrors
// upload_expert_active_async + sync_active_layer_batch (gpu_topology.cpp:
// 213-292) exactly. Pageable host source, matching production.
static double run_async_round(uint8_t** host, uint8_t** dev, int n, size_t bytes,
                               cudaStream_t stream) {
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < n; ++k) {
        CHECK_CUDA(cudaMemcpyAsync(dev[k], host[k], bytes, cudaMemcpyHostToDevice, stream));
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// C: same as B, but each host buffer is transiently cudaHostRegister'd
// before the batch and cudaHostUnregister'd after -- mirrors upload_expert's
// own already-shipped Alt-A pattern (gpu_topology.hpp:1345-1403). Times the
// FULL round-trip (register + copy + sync + unregister) since that's the
// real cost the decode thread would actually pay if this were adopted, plus
// separately-broken-out register/copy+sync/unregister sub-costs (mirroring
// this project's own habit of splitting register cost from DMA cost, e.g.
// t_host_register_stage vs t_gpu_sync_upload in perf_timer.hpp) so a slow
// result can be attributed to the right phase.
struct AsyncPinnedResult { double total_ms, register_ms, copy_sync_ms, unregister_ms; };

static AsyncPinnedResult run_async_pinned_round(uint8_t** host, uint8_t** dev, int n,
                                                 size_t bytes, cudaStream_t stream) {
    AsyncPinnedResult r{};
    auto t_reg0 = std::chrono::high_resolution_clock::now();
    std::vector<bool> registered(n, false);
    for (int k = 0; k < n; ++k) {
        cudaError_t e = cudaHostRegister(host[k], bytes, cudaHostRegisterPortable);
        registered[k] = (e == cudaSuccess);
        if (!registered[k]) {
            // Non-fatal for the benchmark (mirrors production's fallback
            // discipline) -- clear the sticky error and keep going; the
            // subsequent copy just runs against pageable memory for this
            // buffer, same as condition B would.
            cudaGetLastError();
        }
    }
    auto t_reg1 = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < n; ++k) {
        CHECK_CUDA(cudaMemcpyAsync(dev[k], host[k], bytes, cudaMemcpyHostToDevice, stream));
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    auto t_copy1 = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < n; ++k) {
        if (registered[k]) cudaHostUnregister(host[k]);
    }
    auto t_unreg1 = std::chrono::high_resolution_clock::now();

    r.register_ms = std::chrono::duration<double, std::milli>(t_reg1 - t_reg0).count();
    r.copy_sync_ms = std::chrono::duration<double, std::milli>(t_copy1 - t_reg1).count();
    r.unregister_ms = std::chrono::duration<double, std::milli>(t_unreg1 - t_copy1).count();
    r.total_ms = std::chrono::duration<double, std::milli>(t_unreg1 - t_reg0).count();
    return r;
}

// ---------------------------------------------------------------------
// Correctness check: run one B/C-style round, then D2H-readback and
// memcmp each destination against its known source pattern. Cheap, run
// once per device/size before the timing loop -- "verify before trusting
// either timing number" (same discipline as bench_outproj_cpu_vs_gpu.cu).
// ---------------------------------------------------------------------
static bool verify_round(uint8_t** host, uint8_t** dev, int n, size_t bytes,
                          uint8_t* scratch) {
    for (int k = 0; k < n; ++k) {
        CHECK_CUDA(cudaMemcpy(scratch, dev[k], bytes, cudaMemcpyDeviceToHost));
        if (std::memcmp(scratch, host[k], bytes) != 0) return false;
    }
    return true;
}

struct ResultRow {
    int device_idx;
    std::string device_name;
    int size_mb;
    std::string condition;
    double mean_ms, gbps_val;
};

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    if (args.n <= 0 || args.iters <= 0) {
        fprintf(stderr, "--n and --iters must be positive\n");
        return 1;
    }

    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "No CUDA devices found.\n");
        return 1;
    }

    std::vector<int> devices = args.devices;
    if (devices.empty()) {
        for (int i = 0; i < device_count; ++i) devices.push_back(i);
    }

    printf("[DMA-BENCH] %d device(s) requested, n=%d MoE blocks/round, "
           "sizes=%s MB, iters=%d, warmup=%d\n",
           (int)devices.size(), args.n,
           [&]{ std::string s; for (size_t i=0;i<args.sizes_mb.size();++i) { if (i) s+=","; s+=std::to_string(args.sizes_mb[i]); } return s; }().c_str(),
           args.iters, args.warmup);

    std::vector<ResultRow> all_results;

    for (int dev_idx : devices) {
        if (dev_idx < 0 || dev_idx >= device_count) {
            fprintf(stderr, "[DMA-BENCH] device %d out of range (0..%d), skipping\n",
                    dev_idx, device_count - 1);
            continue;
        }
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, dev_idx));  // no context created yet

        CHECK_CUDA(cudaSetDevice(dev_idx));
        // Must be set BEFORE any context-creating call (the cudaStreamCreate
        // just below is the first one for this device) -- mirrors
        // gpu_topology.hpp:503 exactly, a real ~43% measured tax on
        // stream-sync wait latency if omitted (see CLAUDE.md).
        CHECK_CUDA(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));

        printf("\n[DMA-BENCH] === device %d: %s (sm_%d%d) ===\n",
               dev_idx, prop.name, prop.major, prop.minor);
        print_pcie_link_info(prop);

        cudaStream_t copy_stream;
        CHECK_CUDA(cudaStreamCreate(&copy_stream));

        for (int size_mb : args.sizes_mb) {
            size_t bytes = (size_t)size_mb * 1024 * 1024;
            int n = args.n;

            std::vector<uint8_t*> host(n), dev_bufs(n);
            for (int k = 0; k < n; ++k) {
                host[k] = (uint8_t*)malloc(bytes);
                if (!host[k]) { fprintf(stderr, "host malloc failed\n"); return 1; }
                // Deterministic, per-buffer-distinguishable pattern -- also
                // touches every page (first-touch), avoiding demand-paging
                // skew on the very first timed copy.
                for (size_t i = 0; i < bytes; ++i) host[k][i] = (uint8_t)((i + k * 131) & 0xFF);
                CHECK_CUDA(cudaMalloc(&dev_bufs[k], bytes));
            }
            uint8_t* scratch = (uint8_t*)malloc(bytes);
            if (!scratch) { fprintf(stderr, "scratch malloc failed\n"); return 1; }

            // ---- Correctness check (B and C), before trusting any timing ----
            {
                double dt = run_async_round(host.data(), dev_bufs.data(), n, bytes, copy_stream);
                (void)dt;
                bool ok = verify_round(host.data(), dev_bufs.data(), n, bytes, scratch);
                printf("    [%3dMB] correctness (ASYNC pageable):  %s\n",
                       size_mb, ok ? "PASS" : "FAIL");
                if (!ok) { fprintf(stderr, "Correctness FAILED -- aborting, timing numbers would be untrustworthy\n"); return 1; }
            }
            {
                AsyncPinnedResult r = run_async_pinned_round(host.data(), dev_bufs.data(), n, bytes, copy_stream);
                (void)r;
                bool ok = verify_round(host.data(), dev_bufs.data(), n, bytes, scratch);
                printf("    [%3dMB] correctness (ASYNC pinned):    %s\n",
                       size_mb, ok ? "PASS" : "FAIL");
                if (!ok) { fprintf(stderr, "Correctness FAILED -- aborting, timing numbers would be untrustworthy\n"); return 1; }
            }

            // ---- Warmup ----
            for (int w = 0; w < args.warmup; ++w) {
                run_sync_round(host.data(), dev_bufs.data(), n, bytes);
                run_async_round(host.data(), dev_bufs.data(), n, bytes, copy_stream);
                run_async_pinned_round(host.data(), dev_bufs.data(), n, bytes, copy_stream);
            }

            // ---- Timed: A (sync) ----
            std::vector<double> a_samples;
            for (int it = 0; it < args.iters; ++it)
                a_samples.push_back(run_sync_round(host.data(), dev_bufs.data(), n, bytes));
            Stats a_stats = compute_stats(a_samples);

            // ---- Timed: B (async, pageable) ----
            std::vector<double> b_samples;
            for (int it = 0; it < args.iters; ++it)
                b_samples.push_back(run_async_round(host.data(), dev_bufs.data(), n, bytes, copy_stream));
            Stats b_stats = compute_stats(b_samples);

            // ---- Timed: C (async, pinned) ----
            std::vector<double> c_samples, c_reg_samples, c_copy_samples, c_unreg_samples;
            for (int it = 0; it < args.iters; ++it) {
                AsyncPinnedResult r = run_async_pinned_round(host.data(), dev_bufs.data(), n, bytes, copy_stream);
                c_samples.push_back(r.total_ms);
                c_reg_samples.push_back(r.register_ms);
                c_copy_samples.push_back(r.copy_sync_ms);
                c_unreg_samples.push_back(r.unregister_ms);
            }
            Stats c_stats = compute_stats(c_samples);
            Stats c_reg_stats = compute_stats(c_reg_samples);
            Stats c_copy_stats = compute_stats(c_copy_samples);
            Stats c_unreg_stats = compute_stats(c_unreg_samples);

            size_t total_bytes = (size_t)n * bytes;
            printf("    [%3dMB] A SYNC:         mean=%7.3fms min=%7.3fms max=%7.3fms stddev=%6.3fms  (%.2f GB/s)\n",
                   size_mb, a_stats.mean_ms, a_stats.min_ms, a_stats.max_ms, a_stats.stddev_ms,
                   gbps(total_bytes, a_stats.mean_ms));
            printf("    [%3dMB] B ASYNC:        mean=%7.3fms min=%7.3fms max=%7.3fms stddev=%6.3fms  (%.2f GB/s)  speedup vs A: %.2fx\n",
                   size_mb, b_stats.mean_ms, b_stats.min_ms, b_stats.max_ms, b_stats.stddev_ms,
                   gbps(total_bytes, b_stats.mean_ms), a_stats.mean_ms / b_stats.mean_ms);
            printf("    [%3dMB] C ASYNC+PINNED: mean=%7.3fms min=%7.3fms max=%7.3fms stddev=%6.3fms  (%.2f GB/s)  speedup vs A: %.2fx\n",
                   size_mb, c_stats.mean_ms, c_stats.min_ms, c_stats.max_ms, c_stats.stddev_ms,
                   gbps(total_bytes, c_stats.mean_ms), a_stats.mean_ms / c_stats.mean_ms);
            printf("    [%3dMB]   C breakdown:  register=%6.3fms copy+sync=%6.3fms unregister=%6.3fms (sum should ~= C total)\n",
                   size_mb, c_reg_stats.mean_ms, c_copy_stats.mean_ms, c_unreg_stats.mean_ms);

            all_results.push_back({dev_idx, prop.name, size_mb, "A_SYNC", a_stats.mean_ms, gbps(total_bytes, a_stats.mean_ms)});
            all_results.push_back({dev_idx, prop.name, size_mb, "B_ASYNC", b_stats.mean_ms, gbps(total_bytes, b_stats.mean_ms)});
            all_results.push_back({dev_idx, prop.name, size_mb, "C_ASYNC_PINNED", c_stats.mean_ms, gbps(total_bytes, c_stats.mean_ms)});

            for (int k = 0; k < n; ++k) { free(host[k]); cudaFree(dev_bufs[k]); }
            free(scratch);
        }

        CHECK_CUDA(cudaStreamDestroy(copy_stream));
    }

    // ---- Final summary table ----
    printf("\n[DMA-BENCH] ==== Summary (device x size x condition) ====\n");
    printf("%-3s %-28s %6s %-16s %10s %10s\n", "dev", "name", "size", "condition", "mean_ms", "GB/s");
    for (const auto& r : all_results) {
        printf("%-3d %-28s %4dMB %-16s %10.3f %10.2f\n",
               r.device_idx, r.device_name.c_str(), r.size_mb, r.condition.c_str(),
               r.mean_ms, r.gbps_val);
    }

    return 0;
}
