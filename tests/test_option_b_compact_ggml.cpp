// Decode-shape Q2_0 split regression: full weights vs gathered compact weights,
// negative route IDs vs dummy IDs + masks, and changing routes on graph replay.
// Build against the same integration ggml as the engine (see validation doc).
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
constexpr int D = 2560, F = 640, E = 12, K = 10;

struct Graph {
    ggml_context *ctx;
    ggml_backend_buffer_t buffer;
    ggml_cgraph *graph;
    ggml_tensor *gate, *up, *down, *input, *ids, *output;

    Graph(ggml_backend_t backend, int count) {
        ctx = ggml_init({ggml_tensor_overhead() * 32 +
                         ggml_graph_overhead_custom(32, false) + 1024,
                         nullptr, true});
        gate = ggml_new_tensor_3d(ctx, GGML_TYPE_Q2_0, D, F, count);
        up = ggml_new_tensor_3d(ctx, GGML_TYPE_Q2_0, D, F, count);
        down = ggml_new_tensor_3d(ctx, GGML_TYPE_Q2_0, F, D, count);
        input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, 1, 1);
        ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, K, 1);
        auto *g = ggml_mul_mat_id(ctx, gate, input, ids);
        auto *u = ggml_mul_mat_id(ctx, up, input, ids);
        auto *act = ggml_swiglu_split(ctx, g, u);
        output = ggml_mul_mat_id(ctx, down, act, ids);
        ggml_set_output(output);
        graph = ggml_new_graph_custom(ctx, 32, false);
        ggml_build_forward_expand(graph, output);
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buffer) throw std::runtime_error("graph allocation failed");
    }
    ~Graph() { ggml_backend_buffer_free(buffer); ggml_free(ctx); }

    void upload(const std::vector<uint8_t> &g, const std::vector<uint8_t> &u,
                const std::vector<uint8_t> &d, const std::vector<int> &hot) {
        for (size_t i = 0; i < hot.size(); ++i) {
            const size_t gu = ggml_row_size(GGML_TYPE_Q2_0, D) * F;
            const size_t dn = ggml_row_size(GGML_TYPE_Q2_0, F) * D;
            ggml_backend_tensor_set(gate, g.data() + hot[i] * gu, i * gu, gu);
            ggml_backend_tensor_set(up, u.data() + hot[i] * gu, i * gu, gu);
            ggml_backend_tensor_set(down, d.data() + hot[i] * dn, i * dn, dn);
        }
    }

    std::vector<float> run(ggml_backend_t backend, const std::vector<float> &x,
                           const std::vector<int32_t> &routes) {
        ggml_backend_tensor_set(input, x.data(), 0, x.size() * sizeof(float));
        ggml_backend_tensor_set(ids, routes.data(), 0, routes.size() * sizeof(int32_t));
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("graph compute failed");
        ggml_backend_synchronize(backend);
        std::vector<float> y(ggml_nelements(output));
        ggml_backend_tensor_get(output, y.data(), 0, y.size() * sizeof(float));
        return y;
    }
};

std::vector<uint8_t> quantized(int width, int rows, std::mt19937 &rng) {
    std::normal_distribution<float> dist(0, 0.02f);
    std::vector<float> src(static_cast<size_t>(width) * rows);
    for (float &v : src) v = dist(rng);
    std::vector<uint8_t> out(ggml_row_size(GGML_TYPE_Q2_0, width) * rows);
    ggml_quantize_chunk(GGML_TYPE_Q2_0, src.data(), out.data(), 0, rows, width, nullptr);
    return out;
}
} // namespace

int main(int argc, char **argv) {
    const bool cuda = argc > 1 && std::strcmp(argv[1], "cuda") == 0;
    ggml_cpu_init();
    ggml_backend_t backend = cuda ? ggml_backend_cuda_init(0) : ggml_backend_cpu_init();
    if (!backend) return 2;
    if (!cuda) ggml_backend_cpu_set_n_threads(backend, 2);
    std::mt19937 rng(20261003);
    auto g = quantized(D, F * E, rng);
    auto u = quantized(D, F * E, rng);
    auto d = quantized(F, D * E, rng);
    std::normal_distribution<float> dist(0, 1);
    std::vector<float> x(D);
    for (float &v : x) v = dist(rng);
    bool ok = true;
    {
        Graph full(backend, E);
        std::vector<int> all(E);
        std::iota(all.begin(), all.end(), 0);
        full.upload(g, u, d, all);
        for (const std::vector<int> &hot : {std::vector<int>{7}, {9, 2}, {11, 3, 8, 0, 7, 1, 6, 4}}) {
            Graph compact(backend, static_cast<int>(hot.size()));
            compact.upload(g, u, d, hot);
            for (int pattern = 0; pattern < 6; ++pattern) {
                std::vector<int32_t> routes(K), mapped(K), dummy(K);
                std::vector<bool> owned(K);
                for (int k = 0; k < K; ++k) {
                    // all hot, all cold, alternating, one hot at each edge/middle;
                    // repeated hot IDs are intentional in the tiny-set cases.
                    owned[k] = pattern == 0 || (pattern == 2 && k % 2 == 0) ||
                               (pattern == 3 && k == 0) ||
                               (pattern == 4 && k == K / 2) ||
                               (pattern == 5 && k == K - 1);
                    const int local = k % hot.size();
                    routes[k] = hot[local];
                    mapped[k] = owned[k] ? local : -1;
                    dummy[k] = owned[k] ? local : 0;
                }
                const auto expected = full.run(backend, x, routes);
                // Repeated runs exercise capture/replay and hot -> cold transitions.
                for (int replay = 0; replay < 3; ++replay) {
                    const auto skipped = compact.run(backend, x, mapped);
                    const auto masked = compact.run(backend, x, dummy);
                    float max_delta = 0;
                    for (size_t i = 0; i < expected.size(); ++i) {
                        const bool active = owned[i / D];
                        const float want = active ? expected[i] : 0;
                        const float masked_value = active ? masked[i] : 0;
                        max_delta = std::max(max_delta, std::abs(skipped[i] - want));
                        if (!std::isfinite(skipped[i]) || skipped[i] != masked_value ||
                            std::abs(skipped[i] - want) > 1e-6f * std::max(1.0f, std::abs(want)))
                            ok = false;
                    }
                    std::printf("%s hot=%zu pattern=%d replay=%d max_delta=%g %s\n",
                                cuda ? "CUDA" : "CPU", hot.size(), pattern, replay,
                                max_delta, ok ? "PASS" : "FAIL");
                }
            }
        }
    }
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
