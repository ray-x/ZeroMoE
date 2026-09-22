// One-block IQ2_XXS diagnostic.
//
// This is intentionally smaller than the Qwen MoE benchmark.  It exercises
// one 256-value IQ2_XXS weight block through ggml_mul_mat_id on both the CPU
// and CUDA backends, while also printing the two activation quantization
// contracts involved in the production implementations:
//
//   CPU:  Q8_K, one 256-value block
//   CUDA: Q8_1, eight 32-value blocks (MMVQ)
//
// The CUDA temporary quantization buffer is internal to ggml-cuda, so the
// host-side CUDA mirror below follows quantize_row_q8_1_cuda() and
// vec_dot_iq2_xxs_q8_1() exactly.  The actual CUDA graph result is printed
// beside that mirror; this prevents a plausible-looking host calculation
// from being mistaken for a measurement of the production kernel.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "ggml-quants.h"
#include "ggml.h"

#include "ggml-cpu/quants.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

extern "C" const void * ms_grid_iq2xxs();
extern "C" const void * ms_grid_ksigns_iq2xs();
extern "C" const void * ms_grid_kmask_iq2xs();

namespace {

constexpr int kD = 256;
constexpr int kTopK = 1;
constexpr int kExperts = 1;

struct Graph {
    ggml_backend_t backend = nullptr;
    ggml_context * weight_ctx = nullptr;
    ggml_context * graph_ctx = nullptr;
    ggml_backend_buffer_t weight_buffer = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * weight = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * ids = nullptr;
    ggml_tensor * output = nullptr;

    ~Graph() { reset(); }

    void reset() {
        if (allocator) {
            ggml_gallocr_free(allocator);
            allocator = nullptr;
        }
        if (weight_buffer) {
            ggml_backend_buffer_free(weight_buffer);
            weight_buffer = nullptr;
        }
        if (graph_ctx) {
            ggml_free(graph_ctx);
            graph_ctx = nullptr;
        }
        if (weight_ctx) {
            ggml_free(weight_ctx);
            weight_ctx = nullptr;
        }
        if (backend) {
            ggml_backend_free(backend);
            backend = nullptr;
        }
        graph = nullptr;
        weight = input = ids = output = nullptr;
    }
};

void print_tensor(const char *label, const ggml_tensor *t) {
    std::cout << "tensor=" << label << " type=" << ggml_type_name(t->type)
              << " ne=" << t->ne[0] << "x" << t->ne[1] << "x" << t->ne[2]
              << "x" << t->ne[3] << " nb=" << t->nb[0] << "," << t->nb[1]
              << "," << t->nb[2] << "," << t->nb[3] << " data=" << t->data;
    if (t->buffer) {
        std::cout << " buffer=" << ggml_backend_buffer_name(t->buffer)
                  << " buft="
                  << ggml_backend_buft_name(ggml_backend_buffer_get_type(t->buffer));
    }
    std::cout << "\n";
}

bool make_graph(Graph &g, bool cuda, const uint8_t *weights, size_t weight_bytes) {
    g.reset();
    g.backend = cuda ? ggml_backend_cuda_init(0) : ggml_backend_cpu_init();
    if (!g.backend)
        return false;
    if (!cuda)
        ggml_backend_cpu_set_n_threads(g.backend, 1);

    ggml_init_params wp{};
    wp.mem_size = ggml_tensor_overhead() * 4;
    wp.no_alloc = true;
    g.weight_ctx = ggml_init(wp);
    if (!g.weight_ctx)
        return false;
    g.weight = ggml_new_tensor_3d(g.weight_ctx, GGML_TYPE_IQ2_XXS,
                                  kD, 1, kExperts);
    if (!g.weight)
        return false;

    if (cuda) {
        const auto buft = ggml_backend_cuda_buffer_type(0);
        const size_t alloc_bytes =
            std::max(ggml_nbytes(g.weight),
                     ggml_backend_buft_get_alloc_size(buft, g.weight));
        g.weight_buffer = ggml_backend_buft_alloc_buffer(
            buft, alloc_bytes);
        if (!g.weight_buffer ||
            ggml_backend_tensor_alloc(
                g.weight_buffer, g.weight,
                ggml_backend_buffer_get_base(g.weight_buffer)) !=
                GGML_STATUS_SUCCESS)
            return false;
    } else {
        g.weight_buffer = ggml_backend_cpu_buffer_from_ptr(
            const_cast<uint8_t *>(weights), weight_bytes);
        if (!g.weight_buffer ||
            ggml_backend_tensor_alloc(g.weight_buffer, g.weight,
                                      const_cast<uint8_t *>(weights)) !=
                GGML_STATUS_SUCCESS)
            return false;
    }

    ggml_init_params gp{};
    gp.mem_size = 2ull * 1024ull * 1024ull + ggml_graph_overhead() +
                  ggml_tensor_overhead() * 32;
    gp.no_alloc = true;
    g.graph_ctx = ggml_init(gp);
    if (!g.graph_ctx)
        return false;
    g.graph = ggml_new_graph(g.graph_ctx);
    g.input = ggml_new_tensor_3d(g.graph_ctx, GGML_TYPE_F32, kD, 1, 1);
    g.ids = ggml_new_tensor_2d(g.graph_ctx, GGML_TYPE_I32, kTopK, 1);
    if (!g.graph || !g.input || !g.ids)
        return false;
    ggml_set_input(g.input);
    ggml_set_input(g.ids);

    g.output = ggml_mul_mat_id(
        g.graph_ctx, g.weight, ggml_reshape_3d(g.graph_ctx, g.input, kD, 1, 1),
        g.ids);
    if (!g.output)
        return false;
    ggml_set_output(g.output);
    ggml_build_forward_expand(g.graph, g.output);

    g.allocator = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(g.backend));
    if (!g.allocator || !ggml_gallocr_alloc_graph(g.allocator, g.graph))
        return false;

    print_tensor("weight", g.weight);
    print_tensor("input", g.input);
    print_tensor("ids", g.ids);
    print_tensor("output", g.output);
    std::cout << "backend=" << (cuda ? "CUDA" : "CPU")
              << " op=" << ggml_op_name(g.output->op)
              << " nodes=" << ggml_graph_n_nodes(g.graph) << "\n";
    return true;
}

bool run_graph(Graph &g, const std::vector<float> &input,
               const uint8_t *weights, size_t weight_bytes,
               std::vector<float> &output) {
    ggml_backend_tensor_set(g.input, input.data(), 0,
                            input.size() * sizeof(float));
    const int32_t id = 0;
    ggml_backend_tensor_set(g.ids, &id, 0, sizeof(id));
    if (g.backend && ggml_backend_is_cuda(g.backend))
        ggml_backend_tensor_set(g.weight, weights, 0, weight_bytes);
    if (ggml_backend_graph_compute(g.backend, g.graph) != GGML_STATUS_SUCCESS)
        return false;
    ggml_backend_synchronize(g.backend);
    output.resize(ggml_nelements(g.output));
    ggml_backend_tensor_get(g.output, output.data(), 0,
                            output.size() * sizeof(float));
    return true;
}

void print_values(const char *label, const std::vector<float> &values) {
    std::cout << label << " first=";
    const size_t n = std::min<size_t>(8, values.size());
    for (size_t i = 0; i < n; ++i)
        std::cout << (i ? "," : "") << std::setprecision(9) << values[i];
    std::cout << "\n";
}

struct CpuDump {
    block_q8_K q8{};
    int32_t integer_sum = 0;
    float dot = 0.0f;
};

CpuDump cpu_dump(const block_iq2_xxs &w, const std::vector<float> &activation) {
    CpuDump d;
    quantize_row_q8_K_ref(activation.data(), &d.q8, kD);
    ggml_vec_dot_iq2_xxs_q8_K(kD, &d.dot, 0, &w, 0, &d.q8, 0, 1);
    float generic_dot = 0.0f;
    ggml_vec_dot_iq2_xxs_q8_K_generic(kD, &generic_dot, 0, &w, 0, &d.q8, 0,
                                      1);

    const auto *grid = static_cast<const uint64_t *>(ms_grid_iq2xxs());
    const auto *signs = static_cast<const uint8_t *>(ms_grid_ksigns_iq2xs());
    const auto *mask = static_cast<const uint8_t *>(ms_grid_kmask_iq2xs());
    const uint16_t *q2 = w.qs;
    const int8_t *q8 = d.q8.qs;
    uint32_t aux32[2]{};
    const auto *aux8 = reinterpret_cast<const uint8_t *>(aux32);
    for (int ib32 = 0; ib32 < kD / 32; ++ib32) {
        std::memcpy(aux32, q2, sizeof(aux32));
        q2 += 4;
        const int ls = 2 * (aux32[1] >> 28) + 1;
        int sumi = 0;
        for (int l = 0; l < 4; ++l) {
            const auto *g8 = reinterpret_cast<const uint8_t *>(grid + aux8[l]);
            const uint8_t sign = signs[(aux32[1] >> (7 * l)) & 127];
            for (int j = 0; j < 8; ++j)
                sumi += static_cast<int>(g8[j]) * q8[j] *
                        (sign & mask[j] ? -1 : 1);
            q8 += 8;
        }
        d.integer_sum += sumi * ls;
        std::cout << "cpu_block=" << ib32 << " scale_d=" << d.q8.d
                  << " ls=" << ls << " integer_sum=" << sumi
                  << " weighted_integer_sum=" << sumi * ls << "\n";
    }
    std::cout << "cpu_q8k_scale=" << d.q8.d << " cpu_q8k_bsums=";
    for (int i = 0; i < kD / 16; ++i)
        std::cout << (i ? "," : "") << d.q8.bsums[i];
    std::cout << " cpu_integer_sum=" << d.integer_sum
              << " cpu_weight_scale=" << ggml_fp16_to_fp32(w.d)
              << " cpu_dot=" << std::setprecision(9) << d.dot
              << " cpu_generic_dot=" << generic_dot << "\n";
    return d;
}

struct CudaMirrorDump {
    std::vector<float> scale_d;
    std::vector<float> scale_s;
    std::vector<int32_t> integer_sum;
    float dot = 0.0f;
};

CudaMirrorDump cuda_mirror_dump(const block_iq2_xxs &w,
                                const std::vector<float> &activation) {
    CudaMirrorDump d;
    const auto *grid = static_cast<const uint64_t *>(ms_grid_iq2xxs());
    const auto *signs = static_cast<const uint8_t *>(ms_grid_ksigns_iq2xs());
    const auto *mask = static_cast<const uint8_t *>(ms_grid_kmask_iq2xs());
    const uint16_t *q2 = w.qs;
    uint32_t aux32[2]{};
    const auto *aux8 = reinterpret_cast<const uint8_t *>(aux32);
    for (int ib = 0; ib < kD / 32; ++ib) {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j)
            amax = std::max(amax, std::fabs(activation[ib * 32 + j]));
        const float scale = amax / 127.0f;
        const ggml_half scale_half = ggml_fp32_to_fp16(scale);
        const float stored_scale = ggml_fp16_to_fp32(scale_half);
        int sum_q = 0;
        int8_t q[32]{};
        for (int j = 0; j < 32; ++j) {
            q[j] = static_cast<int8_t>(std::round(activation[ib * 32 + j] /
                                                   scale));
            sum_q += q[j];
        }
        std::memcpy(aux32, q2, sizeof(aux32));
        q2 += 4;
        const int ls = (aux32[1] >> 27) | 1;
        int sumi = 0;
        for (int l = 0; l < 4; ++l) {
            const auto *g8 = reinterpret_cast<const uint8_t *>(grid + aux8[l]);
            const uint8_t sign = signs[(aux32[1] >> (7 * l)) & 127];
            for (int j = 0; j < 8; ++j)
                sumi += static_cast<int>(g8[j]) * q[l * 8 + j] *
                        (sign & mask[j] ? -1 : 1);
        }
        // vec_dot_iq2_xxs_q8_1 applies integer division per 32-value
        // fragment, before multiplying by the half-precision activation
        // scale and the half-precision IQ2 weight scale.
        const int scaled = sumi * ls / 8;
        d.scale_d.push_back(stored_scale);
        d.scale_s.push_back(stored_scale * static_cast<float>(sum_q));
        d.integer_sum.push_back(scaled);
        d.dot += ggml_fp16_to_fp32(w.d) * stored_scale * scaled;
        std::cout << "cuda_mirror_block=" << ib << " scale_d_raw=" << scale
                  << " scale_d_half=" << stored_scale
                  << " scale_s_half=" << stored_scale * sum_q
                  << " sum_q=" << sum_q << " ls=" << ls
                  << " integer_sum=" << sumi
                  << " scaled_integer_sum=" << scaled << "\n";
    }
    std::cout << "cuda_mirror_weight_scale=" << ggml_fp16_to_fp32(w.d)
              << " cuda_mirror_dot=" << std::setprecision(9) << d.dot << "\n";
    return d;
}

} // namespace

int main() {
    std::vector<float> weight_f(kD);
    std::vector<float> activation(kD);
    std::vector<float> imatrix(kD, 1.0f);
    for (int i = 0; i < kD; ++i) {
        weight_f[i] = 0.13f * std::sin(0.17f * i) + 0.01f * (i % 7 - 3);
        activation[i] = 0.21f * std::cos(0.11f * i) - 0.013f * (i % 11 - 5);
    }
    const size_t weight_bytes = ggml_row_size(GGML_TYPE_IQ2_XXS, kD);
    // ggml's CPU buffer-from-pointer contract requires TENSOR_ALIGNMENT.  The
    // CUDA buffer is device allocated, but using the same aligned host copy
    // keeps the CPU and CUDA weight bytes exactly identical.
    std::vector<uint8_t> weight_storage(weight_bytes + 127);
    const uintptr_t aligned_addr =
        (reinterpret_cast<uintptr_t>(weight_storage.data()) + 127u) &
        ~static_cast<uintptr_t>(127u);
    auto *weights = reinterpret_cast<uint8_t *>(aligned_addr);
    const size_t quantized = ggml_quantize_chunk(
        GGML_TYPE_IQ2_XXS, weight_f.data(), weights, 0, 1, kD,
        ggml_quantize_requires_imatrix(GGML_TYPE_IQ2_XXS) ? imatrix.data()
                                                          : nullptr);
    if (quantized != weight_bytes) {
        std::cerr << "IQ2 quantization produced " << quantized
                  << " bytes, expected " << weight_bytes << "\n";
        return 2;
    }

    const auto *weight_block = reinterpret_cast<const block_iq2_xxs *>(weights);
    std::cout << std::fixed << std::setprecision(9)
              << "probe=IQ2_XXS one_block dimensions=" << kD
              << " weight_bytes=" << weight_bytes << "\n";
    // Initialize ggml's CPU backend before calling its exported dot entry
    // points directly.  The graph itself also does this, but doing it here
    // makes the standalone direct-call dump independent of lazy backend
    // initialization order.
    if (ggml_backend_t cpu_init = ggml_backend_cpu_init())
        ggml_backend_free(cpu_init);
    const CpuDump cpu_dump_result = cpu_dump(*weight_block, activation);
    const CudaMirrorDump cuda_dump_result =
        cuda_mirror_dump(*weight_block, activation);

    Graph cpu_graph;
    Graph cuda_graph;
    std::vector<float> cpu_output;
    std::vector<float> cuda_output;
    if (!make_graph(cpu_graph, false, weights, weight_bytes) ||
        !run_graph(cpu_graph, activation, weights, weight_bytes, cpu_output)) {
        std::cerr << "CPU ggml_mul_mat_id probe failed\n";
        return 3;
    }
    if (!make_graph(cuda_graph, true, weights, weight_bytes) ||
        !run_graph(cuda_graph, activation, weights, weight_bytes, cuda_output)) {
        std::cerr << "CUDA ggml_mul_mat_id probe failed\n";
        return 4;
    }

    print_values("cpu_graph_output", cpu_output);
    print_values("cuda_graph_output", cuda_output);
    const float cpu_graph_dot = cpu_output.empty() ? NAN : cpu_output[0];
    const float cuda_graph_dot = cuda_output.empty() ? NAN : cuda_output[0];
    std::cout << "cpu_direct_dot=" << std::setprecision(9) << cpu_dump_result.dot
              << " cpu_graph_dot=" << cpu_graph_dot
              << " cuda_mirror_dot=" << cuda_dump_result.dot
              << " cuda_graph_dot=" << cuda_graph_dot
              << " graph_delta=" << (cuda_graph_dot - cpu_graph_dot)
              << " mirror_to_cuda_delta="
              << (cuda_graph_dot - cuda_dump_result.dot) << "\n";
    return std::isfinite(cpu_graph_dot) && std::isfinite(cuda_graph_dot) ? 0 : 5;
}
