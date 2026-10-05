// Apples-to-apples Qwen CPU MoE graph benchmark.
//
// This deliberately depends only on GGML's public CPU/backend API so the same
// source can be linked against MoEStream's and llama.cpp's ggml libraries:
//
//   bench/run_qwen_cpu_moe_compare.sh
//
// It measures the graph used by the Qwen CPU-affinity path, not the full model:
//   merged gate+up MUL_MAT_ID -> SwiGLU -> down MUL_MAT_ID
// with the real Qwen dimensions and a fixed-stride [gate|up|down] host arena.
// Only the ten selected experts are quantized into valid non-zero blocks; the
// other expert slots remain zero because MUL_MAT_ID never reads them.  This
// makes the timing representative without loading a 75 GB model or spending
// the timed region in a reference requantizer.  The production correctness
// tests cover additional quantization/layout combinations separately.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sys/resource.h>
#include <thread>
#include <vector>

namespace {

enum class Stage { GateUp, Down, Full };

struct CpuUsage {
    double user_s = 0.0;
    double system_s = 0.0;
};

CpuUsage cpu_usage() {
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    return {static_cast<double>(ru.ru_utime.tv_sec) +
                static_cast<double>(ru.ru_utime.tv_usec) * 1e-6,
            static_cast<double>(ru.ru_stime.tv_sec) +
                static_cast<double>(ru.ru_stime.tv_usec) * 1e-6};
}

double elapsed_s(std::chrono::steady_clock::time_point begin,
                 std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double>(end - begin).count();
}

int env_int(const char *name, int fallback) {
    const char *value = std::getenv(name);
    if (!value || !value[0])
        return fallback;
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0 || parsed > 100000)
        return fallback;
    return static_cast<int>(parsed);
}

ggml_type bench_gate_up_type() {
    const char *value = std::getenv("MOESTREAM_QWEN_CPU_BENCH_GATE_UP_TYPE");
    if (value && (std::strcmp(value, "IQ2_XXS") == 0 ||
                  std::strcmp(value, "iq2_xxs") == 0))
        return GGML_TYPE_IQ2_XXS;
    return GGML_TYPE_IQ1_M;
}

void fill_selected_expert_weights(uint8_t *gate_plane, uint8_t *up_plane,
                                  uint8_t *down_plane, size_t gate_stride,
                                  size_t up_stride, size_t down_stride,
                                  int n_experts, int d_model, int d_ff,
                                  int topk, ggml_type gate_up_type,
                                  int initialized_experts) {
    const size_t gate_bytes =
        ggml_row_size(gate_up_type, d_model) * static_cast<size_t>(d_ff);
    const size_t down_bytes = ggml_row_size(GGML_TYPE_IQ4_NL, d_ff) *
                              static_cast<size_t>(d_model);
    std::vector<float> gate_f(static_cast<size_t>(d_ff) * d_model);
    std::vector<float> down_f(static_cast<size_t>(d_model) * d_ff);
    std::vector<float> imatrix(static_cast<size_t>(d_model), 1.0f);
    const int n_initialized = std::clamp(initialized_experts, topk, n_experts);
    if (n_initialized <= 0)
        return;

    // Quantizing every synthetic expert makes a 512-expert benchmark spend
    // minutes in setup before it measures a single graph call.  The timing
    // does not depend on the expert values, so make one valid quantized
    // template and copy it into every routed slot.  This keeps all IQ blocks
    // valid while making route-locality experiments measure the graph rather
    // than the benchmark's initializer.
    for (size_t i = 0; i < gate_f.size(); ++i)
        gate_f[i] = 0.05f + 0.02f * static_cast<float>(i % 17);
    for (size_t i = 0; i < down_f.size(); ++i)
        down_f[i] = 0.03f + 0.015f * static_cast<float>(i % 19);
    const float * im = ggml_quantize_requires_imatrix(gate_up_type)
                           ? imatrix.data()
                           : nullptr;
    std::vector<uint8_t> gate_template(gate_bytes);
    std::vector<uint8_t> down_template(down_bytes);
    const size_t gate_q = ggml_quantize_chunk(
        gate_up_type, gate_f.data(), gate_template.data(), 0, d_ff, d_model,
        im);
    const size_t up_q = gate_q;
    const size_t down_q = ggml_quantize_chunk(
        GGML_TYPE_IQ4_NL, down_f.data(), down_template.data(), 0, d_model,
        d_ff, nullptr);
    if (gate_q != gate_bytes || up_q != gate_bytes || down_q != down_bytes)
        std::abort();
    for (int expert = 0; expert < n_initialized; ++expert) {
        std::memcpy(gate_plane + static_cast<size_t>(expert) * gate_stride,
                    gate_template.data(), gate_bytes);
        std::memcpy(up_plane + static_cast<size_t>(expert) * up_stride,
                    gate_template.data(), gate_bytes);
        std::memcpy(down_plane + static_cast<size_t>(expert) * down_stride,
                    down_template.data(), down_bytes);
    }
}

Stage env_stage() {
    const char *value = std::getenv("MOESTREAM_QWEN_CPU_BENCH_STAGE");
    if (value && (std::strcmp(value, "gate_up") == 0 ||
                  std::strcmp(value, "gateup") == 0))
        return Stage::GateUp;
    if (value && std::strcmp(value, "down") == 0)
        return Stage::Down;
    return Stage::Full;
}

const char *stage_name(Stage stage) {
    switch (stage) {
    case Stage::GateUp:
        return "gate_up_swiglu";
    case Stage::Down:
        return "down";
    case Stage::Full:
        return "full_weighted";
    }
    return "unknown";
}

struct Graph {
    ggml_backend_t backend = nullptr;
    ggml_threadpool_t threadpool = nullptr;
    ggml_context *weight_ctx = nullptr;
    ggml_context *graph_ctx = nullptr;
    ggml_backend_buffer_t gate_up_buf = nullptr;
    ggml_backend_buffer_t down_buf = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_backend_graph_plan_t plan = nullptr;
    ggml_cgraph *graph = nullptr;
    ggml_tensor *x = nullptr;
    ggml_tensor *ids = nullptr;
    ggml_tensor *weights = nullptr;
    ggml_tensor *output = nullptr;
    ggml_tensor *gate_up = nullptr;
    ggml_tensor *down = nullptr;
    int n_experts = 0;
    int d_model = 0;
    int d_ff = 0;
    int topk = 0;
    size_t gate_up_stride = 0;
    size_t down_stride = 0;

    ~Graph() { reset(); }

    void reset() {
        if (plan && backend)
            ggml_backend_graph_plan_free(backend, plan);
        plan = nullptr;
        if (allocator)
            ggml_gallocr_free(allocator);
        allocator = nullptr;
        if (graph_ctx)
            ggml_free(graph_ctx);
        graph_ctx = nullptr;
        if (gate_up_buf)
            ggml_backend_buffer_free(gate_up_buf);
        if (down_buf)
            ggml_backend_buffer_free(down_buf);
        gate_up_buf = nullptr;
        down_buf = nullptr;
        if (weight_ctx)
            ggml_free(weight_ctx);
        weight_ctx = nullptr;
        if (threadpool)
            ggml_threadpool_free(threadpool);
        threadpool = nullptr;
        if (backend)
            ggml_backend_free(backend);
        backend = nullptr;
        graph = nullptr;
        x = ids = weights = output = gate_up = down = nullptr;
    }
};

bool make_graph(Graph &result, Stage stage, int threads, int n_experts,
                int d_model, int d_ff, int topk, ggml_type gate_up_type,
                void *gate_plane, void *up_plane, void *down_plane,
                size_t gate_up_stride, size_t down_stride) {
    result.reset();
    result.backend = ggml_backend_cpu_init();
    if (!result.backend)
        return false;

    ggml_backend_cpu_set_n_threads(result.backend, threads);
    ggml_threadpool_params tp_params = ggml_threadpool_params_default(threads);
    result.threadpool = ggml_threadpool_new(&tp_params);
    if (!result.threadpool)
        return false;
    ggml_backend_cpu_set_threadpool(result.backend, result.threadpool);

    ggml_init_params wip{};
    wip.mem_size = ggml_tensor_overhead() * 8;
    wip.no_alloc = true;
    result.weight_ctx = ggml_init(wip);
    if (!result.weight_ctx)
        return false;

    result.gate_up = ggml_new_tensor_3d(
        result.weight_ctx, gate_up_type, d_model, d_ff * 2, n_experts);
    result.down = ggml_new_tensor_3d(
        result.weight_ctx, GGML_TYPE_IQ4_NL, d_ff, d_model, n_experts);
    if (!result.gate_up || !result.down)
        return false;

    const size_t gate_up_per_expert = ggml_nbytes(result.gate_up) /
                                      static_cast<size_t>(n_experts);
    const size_t down_per_expert = ggml_nbytes(result.down) /
                                   static_cast<size_t>(n_experts);
    result.gate_up->nb[2] = gate_up_stride;
    result.gate_up->nb[3] = gate_up_stride * static_cast<size_t>(n_experts);
    result.down->nb[2] = down_stride;
    result.down->nb[3] = down_stride * static_cast<size_t>(n_experts);

    const size_t gate_up_view_bytes =
        gate_up_stride * static_cast<size_t>(n_experts - 1) + gate_up_per_expert;
    const size_t down_view_bytes =
        down_stride * static_cast<size_t>(n_experts - 1) + down_per_expert;
    result.gate_up_buf = ggml_backend_cpu_buffer_from_ptr(
        gate_plane, gate_up_view_bytes);
    result.down_buf = ggml_backend_cpu_buffer_from_ptr(down_plane,
                                                       down_view_bytes);
    if (!result.gate_up_buf || !result.down_buf ||
        ggml_backend_tensor_alloc(result.gate_up_buf, result.gate_up,
                                  gate_plane) != GGML_STATUS_SUCCESS ||
        ggml_backend_tensor_alloc(result.down_buf, result.down, down_plane) !=
            GGML_STATUS_SUCCESS)
        return false;

    ggml_init_params gip{};
    gip.mem_size = 2ull * 1024ull * 1024ull + ggml_graph_overhead() +
                   ggml_tensor_overhead() * 128;
    gip.no_alloc = true;
    result.graph_ctx = ggml_init(gip);
    if (!result.graph_ctx)
        return false;
    result.graph = ggml_new_graph(result.graph_ctx);
    result.x = ggml_new_tensor_2d(result.graph_ctx, GGML_TYPE_F32, d_model, 1);
    result.ids = ggml_new_tensor_2d(result.graph_ctx, GGML_TYPE_I32, topk, 1);
    if (!result.graph || !result.x || !result.ids)
        return false;
    ggml_set_input(result.x);
    ggml_set_input(result.ids);

    ggml_tensor *route_weights = nullptr;
    if (stage == Stage::Full) {
        route_weights = ggml_new_tensor_3d(result.graph_ctx, GGML_TYPE_F32, 1,
                                           topk, 1);
        if (!route_weights)
            return false;
        ggml_set_input(route_weights);
        result.weights = route_weights;
    }

    ggml_tensor *input3 = ggml_reshape_3d(result.graph_ctx, result.x, d_model,
                                          1, 1);
    ggml_tensor *gate_up_out =
        ggml_mul_mat_id(result.graph_ctx, result.gate_up, input3, result.ids);
    if (!input3 || !gate_up_out)
        return false;
    ggml_tensor *gate_out = ggml_view_3d(
        result.graph_ctx, gate_up_out, d_ff, gate_up_out->ne[1],
        gate_up_out->ne[2], gate_up_out->nb[1], gate_up_out->nb[2], 0);
    ggml_tensor *up_out = ggml_view_3d(
        result.graph_ctx, gate_up_out, d_ff, gate_up_out->ne[1],
        gate_up_out->ne[2], gate_up_out->nb[1], gate_up_out->nb[2],
        static_cast<size_t>(d_ff) * gate_up_out->nb[0]);
    ggml_tensor *activation = ggml_swiglu_split(result.graph_ctx, gate_out,
                                                up_out);
    if (!gate_out || !up_out || !activation)
        return false;

    if (stage == Stage::GateUp) {
        result.output = activation;
    } else {
        ggml_tensor *down_out = ggml_mul_mat_id(
            result.graph_ctx, result.down, activation, result.ids);
        if (!down_out)
            return false;
        if (stage == Stage::Down) {
            result.output = down_out;
        } else {
            ggml_tensor *weighted =
                ggml_mul(result.graph_ctx, down_out, route_weights);
            if (!weighted)
                return false;
            ggml_tensor *sum = nullptr;
            for (int k = 0; k < topk; ++k) {
                ggml_tensor *expert = ggml_view_2d(
                    result.graph_ctx, weighted, d_model, 1, weighted->nb[2],
                    static_cast<size_t>(k) * weighted->nb[1]);
                sum = k == 0 ? expert : ggml_add(result.graph_ctx, sum, expert);
            }
            result.output = topk > 1 ? ggml_cont(result.graph_ctx, sum) : sum;
        }
    }
    if (!result.output)
        return false;
    ggml_set_output(result.output);
    ggml_build_forward_expand(result.graph, result.output);

    result.allocator = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    if (!result.allocator || !ggml_gallocr_alloc_graph(result.allocator,
                                                       result.graph))
        return false;
    result.plan = ggml_backend_graph_plan_create(result.backend, result.graph);
    if (!result.plan)
        return false;
    result.n_experts = n_experts;
    result.d_model = d_model;
    result.d_ff = d_ff;
    result.topk = topk;
    result.gate_up_stride = gate_up_stride;
    result.down_stride = down_stride;
    if (std::getenv("MOESTREAM_QWEN_CPU_BENCH_GRAPH_DEBUG")) {
        std::cerr << "graph stage=" << stage_name(stage)
                  << " nodes=" << ggml_graph_n_nodes(result.graph) << " output="
                  << result.output->ne[0] << "x" << result.output->ne[1]
                  << "x" << result.output->ne[2] << "\n";
        for (int i = 0; i < ggml_graph_n_nodes(result.graph); ++i)
            std::cerr << "  " << i << ": "
                      << ggml_op_name(ggml_graph_node(result.graph, i)->op)
                      << "\n";
    }
    return true;
}

bool run_graph(Graph &graph, const float *input, const int32_t *ids,
               const float *weights) {
    std::memcpy(graph.x->data, input,
                static_cast<size_t>(graph.d_model) * sizeof(float));
    std::memcpy(graph.ids->data, ids,
                static_cast<size_t>(graph.topk) * sizeof(int32_t));
    if (graph.weights)
        std::memcpy(graph.weights->data, weights,
                    static_cast<size_t>(graph.topk) * sizeof(float));
    // The engine's persistent Qwen runner uses graph_plan_compute().  Keep a
    // switch for comparison with the ordinary backend path because it also
    // exposes any stale-plan/data-pointer issue in this standalone probe.
    if (std::getenv("MOESTREAM_QWEN_CPU_BENCH_NO_PLAN"))
        return ggml_backend_graph_compute(graph.backend, graph.graph) ==
               GGML_STATUS_SUCCESS;
    return ggml_backend_graph_plan_compute(graph.backend, graph.plan) ==
           GGML_STATUS_SUCCESS;
}

bool run_stage(Stage stage, int threads, int n_experts, int d_model, int d_ff,
               int topk, ggml_type gate_up_type, int warmup, int iterations,
               void *gate_plane, void *up_plane, void *down_plane,
               size_t gate_up_stride, size_t down_stride, int route_variety,
               bool packed) {
    Graph graph;
    if (!make_graph(graph, stage, threads, n_experts, d_model, d_ff, topk,
                    gate_up_type, gate_plane, up_plane, down_plane,
                    gate_up_stride, down_stride)) {
        std::cerr << "failed to create " << stage_name(stage) << " graph\n";
        return false;
    }

    std::vector<float> input(static_cast<size_t>(d_model), 0.1f);
    std::vector<int32_t> ids(static_cast<size_t>(topk));
    std::vector<float> weights(static_cast<size_t>(topk), 1.0f / topk);
    const int route_count = route_variety > 0
                                ? std::clamp(route_variety, topk, n_experts)
                                : topk;
    auto set_ids = [&](int iteration) {
        for (int k = 0; k < topk; ++k) {
            const int base = route_variety > 0
                                 ? (iteration * 17) % route_count
                                 : 0;
            ids[static_cast<size_t>(k)] =
                route_variety > 0
                    ? (base + k * 37) % route_count
                    : k % n_experts;
        }
    };
    set_ids(0);
    for (int i = 0; i < warmup; ++i) {
        set_ids(i);
        if (!run_graph(graph, input.data(), ids.data(), weights.data()))
            return false;
    }

    const CpuUsage cpu_begin = cpu_usage();
    const auto wall_begin = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        set_ids(warmup + i);
        if (!run_graph(graph, input.data(), ids.data(), weights.data()))
            return false;
    }
    const auto wall_end = std::chrono::steady_clock::now();
    const CpuUsage cpu_end = cpu_usage();
    const double wall = elapsed_s(wall_begin, wall_end);
    const double user = cpu_end.user_s - cpu_begin.user_s;
    const double system = cpu_end.system_s - cpu_begin.system_s;
    const float *out = static_cast<const float *>(graph.output->data);
    double checksum = 0.0;
    const size_t output_values = ggml_nelements(graph.output);
    for (size_t i = 0; i < output_values; ++i)
        checksum += static_cast<double>(out[i]);

    std::cout << "stage=" << stage_name(stage) << " threads=" << threads
              << " gate_up_type=" << ggml_type_name(gate_up_type)
              << " layout=" << (packed ? "packed" : "strided")
              << " experts=" << n_experts << " topk=" << topk
              << " route_variety=" << route_variety
              << " D=" << d_model << " FF=" << d_ff << " iterations="
              << iterations << " wall_ms=" << std::fixed
              << std::setprecision(3) << (wall * 1000.0 / iterations)
              << " user_ms=" << (user * 1000.0 / iterations)
              << " system_ms=" << (system * 1000.0 / iterations)
              << " cpu_util=" << (100.0 * (user + system) / wall)
              << " checksum=" << checksum << "\n";
    if (!std::isfinite(checksum)) {
        std::cerr << "non-finite benchmark output\n";
        return false;
    }
    return true;
}

} // namespace

int main() {
    const int d_model = env_int("MOESTREAM_QWEN_CPU_BENCH_D_MODEL", 2560);
    const int d_ff = env_int("MOESTREAM_QWEN_CPU_BENCH_D_FF", 640);
    const int topk = env_int("MOESTREAM_QWEN_CPU_BENCH_TOPK", 10);
    const int n_experts = env_int("MOESTREAM_QWEN_CPU_BENCH_EXPERTS", 512);
    const int threads = env_int(
        "MOESTREAM_QWEN_CPU_BENCH_THREADS",
        env_int("OMP_NUM_THREADS",
                static_cast<int>(std::max(1u, std::thread::hardware_concurrency()))));
    const int warmup = env_int("MOESTREAM_QWEN_CPU_BENCH_WARMUP", 2);
    const int iterations = env_int("MOESTREAM_QWEN_CPU_BENCH_ITERS", 8);
    const int route_variety = env_int(
        "MOESTREAM_QWEN_CPU_BENCH_ROUTE_VARIETY", 0);
    const ggml_type gate_up_type = bench_gate_up_type();
    if (n_experts < topk) {
        std::cerr << "experts must be >= topk\n";
        return 2;
    }

    const size_t gate_bytes =
        ggml_row_size(gate_up_type, d_model) * static_cast<size_t>(d_ff);
    const size_t up_bytes = gate_bytes;
    const size_t down_bytes =
        ggml_row_size(GGML_TYPE_IQ4_NL, d_ff) * static_cast<size_t>(d_model);
    const size_t raw_stride = gate_bytes + up_bytes + down_bytes;
    const size_t stride = (raw_stride + 4095) & ~static_cast<size_t>(4095);
    const bool packed = [] {
        const char *value = std::getenv("MOESTREAM_QWEN_CPU_BENCH_PACKED");
        return value && value[0] == '1';
    }();
    const size_t gate_up_stride = packed ? gate_bytes + up_bytes : stride;
    const size_t down_stride = packed ? down_bytes : stride;
    if (gate_up_stride > SIZE_MAX / static_cast<size_t>(n_experts) ||
        down_stride > SIZE_MAX / static_cast<size_t>(n_experts)) {
        std::cerr << "arena size overflow\n";
        return 2;
    }
    const size_t gate_up_bytes =
        gate_up_stride * static_cast<size_t>(n_experts);
    const size_t down_plane_bytes =
        down_stride * static_cast<size_t>(n_experts);
    const size_t arena_bytes = packed ? gate_up_bytes + down_plane_bytes
                                      : gate_up_bytes;
    void *arena_raw = nullptr;
    if (posix_memalign(&arena_raw, 4096, arena_bytes) != 0 || !arena_raw) {
        std::cerr << "failed to allocate "
                  << (arena_bytes / (1024.0 * 1024.0 * 1024.0)) << " GiB\n";
        return 2;
    }
    std::memset(arena_raw, 0, arena_bytes);
    auto *arena = static_cast<uint8_t *>(arena_raw);
    void *gate_plane = arena;
    void *up_plane = arena + gate_bytes;
    void *down_plane = packed ? arena + gate_up_bytes
                              : arena + gate_bytes + up_bytes;
    const int initialized_experts = route_variety > 0
                                       ? std::clamp(route_variety, topk, n_experts)
                                       : topk;
    fill_selected_expert_weights(static_cast<uint8_t *>(gate_plane),
                                 static_cast<uint8_t *>(up_plane),
                                 static_cast<uint8_t *>(down_plane),
                                 gate_up_stride, gate_up_stride, down_stride,
                                 n_experts, d_model, d_ff, topk,
                                 gate_up_type, initialized_experts);

    const Stage selected = env_stage();
    std::cout << "ggml_cpu_moe_bench arena_gib=" << std::fixed
              << std::setprecision(3)
              << (arena_bytes / (1024.0 * 1024.0 * 1024.0))
              << " layout=" << (packed ? "packed" : "strided")
              << " gate_up_stride=" << gate_up_stride
              << " down_stride=" << down_stride
              << " gate_bytes=" << gate_bytes
              << " up_bytes=" << up_bytes << " down_bytes=" << down_bytes
              << " gate_up_type=" << ggml_type_name(gate_up_type)
              << " features=AVX2:" << ggml_cpu_has_avx2()
              << ",AVX512:" << ggml_cpu_has_avx512()
              << ",AVX512_VNNI:" << ggml_cpu_has_avx512_vnni() << "\n";

    bool ok = true;
    if (std::getenv("MOESTREAM_QWEN_CPU_BENCH_STAGE")) {
        ok = run_stage(selected, threads, n_experts, d_model, d_ff, topk,
                       gate_up_type, warmup, iterations, gate_plane,
                       up_plane, down_plane, gate_up_stride, down_stride,
                       route_variety, packed);
    } else {
        for (Stage stage : {Stage::GateUp, Stage::Down, Stage::Full}) {
            ok = run_stage(stage, threads, n_experts, d_model, d_ff, topk,
                           gate_up_type, warmup, iterations, gate_plane,
                           up_plane, down_plane, gate_up_stride, down_stride,
                           route_variety, packed) &&
                 ok;
        }
    }
    std::free(arena_raw);
    return ok ? 0 : 1;
}
