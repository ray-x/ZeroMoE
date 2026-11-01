// Bit-exactness test for the fused cold-expert node against the stock GGML
// route-remap/mask, gate/up MUL_MAT_ID, SwiGLU, down MUL_MAT_ID, and weighting
// chain. Covers Q2_0/Q8_0 and IQ* / Q8_K activation-dot paths, route masks,
// token counts, and CPU thread counts.
#include "cold_ffn_fused.hpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {
struct Case {
  const char *name;
  ggml_type gate_type;
  ggml_type up_type;
  ggml_type down_type;
  int D;
  int FF;
  bool exhaustive;
  int experts = 24;
  int topk = 10;
};

bool make_quantized_weights(std::mt19937 &rng, ggml_type type, int64_t ne0,
                            int64_t rows, std::vector<char> &data) {
  ggml_quantize_init(type);
  const auto *traits = ggml_get_type_traits(type);
  if (!traits)
    return false;

  std::normal_distribution<float> distribution(0.0f, 0.4f);
  std::vector<float> row(static_cast<size_t>(ne0));
  const size_t row_bytes = ggml_row_size(type, ne0);
  data.resize(static_cast<size_t>(rows) * row_bytes);
  std::vector<float> imatrix;
  if (ggml_quantize_requires_imatrix(type))
    imatrix.assign(static_cast<size_t>(ne0), 1.0f);
  for (int64_t r = 0; r < rows; ++r) {
    for (auto &value : row)
      value = distribution(rng);
    char *packed_row = data.data() + static_cast<size_t>(r) * row_bytes;
    if (traits->from_float_ref) {
      traits->from_float_ref(row.data(), packed_row, ne0);
    } else if (ggml_quantize_chunk(
                   type, row.data(), packed_row, 0, 1, ne0,
                   imatrix.empty() ? nullptr : imatrix.data()) != row_bytes) {
      return false;
    }
  }
  return true;
}

int run_case(const Case &test, std::mt19937 &rng, bool type_smoke,
             bool expanded_matrix) {
  int failures = 0;
  int cases = 0;
  if ((type_smoke || expanded_matrix) && test.exhaustive)
    return 0;
  const std::vector<int> thread_counts = type_smoke
      ? std::vector<int>{1}
      : test.exhaustive ? std::vector<int>{1, 2, 3, 6, 10}
                        : std::vector<int>{1, 10};
  const std::vector<int> token_counts = type_smoke
      ? std::vector<int>{1}
      : test.exhaustive ? std::vector<int>{1, 2, 4}
                        : std::vector<int>{1};
  const std::vector<int> mask_patterns = type_smoke
      ? std::vector<int>{2}
      : test.exhaustive ? std::vector<int>{0, 1, 2, 3}
                        : std::vector<int>{1, 2, 3};
  std::vector<char> gate_data, up_data, down_data;
  const int64_t gate_rows = static_cast<int64_t>(test.FF) * test.experts;
  const int64_t down_rows = static_cast<int64_t>(test.D) * test.experts;
  if (!make_quantized_weights(rng, test.gate_type, test.D, gate_rows,
                              gate_data) ||
      !make_quantized_weights(rng, test.up_type, test.D, gate_rows, up_data) ||
      !make_quantized_weights(rng, test.down_type, test.FF, down_rows,
                              down_data)) {
    std::fprintf(stderr, "FAIL %s: no reference quantizer for test type\n",
                 test.name);
    return 1;
  }
  for (int threads : thread_counts) {
    for (int S : token_counts) {
      for (int pattern : mask_patterns) {
        ggml_backend_t backend = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(backend, threads);
        ggml_init_params params{
            ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false),
            nullptr,
            true};
        ggml_context *ctx = ggml_init(params);
        auto *gate = ggml_new_tensor_3d(ctx, test.gate_type, test.D, test.FF,
                                        test.experts);
        auto *up = ggml_new_tensor_3d(ctx, test.up_type, test.D, test.FF,
                                      test.experts);
        auto *down = ggml_new_tensor_3d(ctx, test.down_type, test.FF, test.D,
                                        test.experts);
        auto *x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, test.D, 1, S);
        auto *ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, test.topk, S);
        auto *weights =
            ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, test.topk, S);
        auto *remap = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, 1,
                                         test.experts, 1);
        auto *mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1,
                                        test.experts, 1);

        // Stock graph path: identify cold routes, evaluate the complete
        // expert chain, then apply the cold mask and route weights.
        auto *ids_flat = S == 1 ? ids
                                : ggml_reshape_1d(ctx, ids, int64_t(test.topk) * S);
        auto *cold_ids = ggml_reshape_2d(
            ctx, ggml_get_rows(ctx, remap, ids_flat), test.topk, S);
        auto *mask_rows = ggml_get_rows(ctx, mask, ids_flat);
        auto *cold_weights = ggml_mul(
            ctx, weights,
            S == 1 ? mask_rows
                   : ggml_reshape_3d(ctx, mask_rows, 1, test.topk, S));
        auto *up_result = ggml_mul_mat_id(ctx, up, x, cold_ids);
        auto *gate_result = ggml_mul_mat_id(ctx, gate, x, cold_ids);
        auto *activation = ggml_swiglu_split(ctx, gate_result, up_result);
        auto *down_result = ggml_mul_mat_id(ctx, down, activation, cold_ids);
        auto *stock = ggml_mul(ctx, down_result, cold_weights);

        std::shared_ptr<qwen4exp_cold::Context> owner;
        auto *fused = qwen4exp_cold::build(
            ctx, owner, x, ids, weights, remap, mask, gate, up, down, threads);
        if (!fused) {
          std::fprintf(stderr, "FAIL %s: fused op refused supported shapes\n",
                       test.name);
          return failures + 1;
        }
        auto *stock_graph = ggml_new_graph_custom(ctx, 64, false);
        auto *fused_graph = ggml_new_graph_custom(ctx, 64, false);
        ggml_build_forward_expand(stock_graph, stock);
        ggml_build_forward_expand(fused_graph, fused);
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

        if (gate_data.size() != ggml_nbytes(gate) ||
            up_data.size() != ggml_nbytes(up) ||
            down_data.size() != ggml_nbytes(down)) {
          std::fprintf(stderr, "FAIL %s: quantized weight fixture size mismatch\n",
                       test.name);
          return failures + 1;
        }
        ggml_backend_tensor_set(gate, gate_data.data(), 0, gate_data.size());
        ggml_backend_tensor_set(up, up_data.data(), 0, up_data.size());
        ggml_backend_tensor_set(down, down_data.data(), 0, down_data.size());

        std::vector<float> x_values(static_cast<size_t>(test.D) * S);
        std::vector<float> weight_values(static_cast<size_t>(test.topk) * S);
        std::normal_distribution<float> input_distribution(0.0f, 1.0f);
        for (auto &value : x_values)
          value = input_distribution(rng);
        for (auto &value : weight_values)
          value = 0.02f + 0.3f * std::abs(input_distribution(rng));

        std::vector<int32_t> id_values(static_cast<size_t>(test.topk) * S);
        std::vector<int32_t> remap_values(test.experts);
        std::vector<float> mask_values(test.experts);
        for (int e = 0; e < test.experts; ++e) {
          const bool is_cold = pattern == 0 ? false
                               : pattern == 1 ? (e % 4 == 0)
                               : pattern == 2 ? true
                                              : (e % 3 == 1);
          remap_values[e] = is_cold ? e : -1;
          mask_values[e] = is_cold ? 1.0f : 0.0f;
        }
        for (int t = 0; t < S; ++t) {
          std::vector<int> permutation(test.experts);
          for (int e = 0; e < test.experts; ++e)
            permutation[e] = e;
          std::shuffle(permutation.begin(), permutation.end(), rng);
          for (int k = 0; k < test.topk; ++k)
            id_values[static_cast<size_t>(t) * test.topk + k] =
                pattern == 3 ? (k * 2 + t) % test.experts
                : pattern == 1 ? k
                               : permutation[k];
        }

        ggml_backend_tensor_set(x, x_values.data(), 0, ggml_nbytes(x));
        ggml_backend_tensor_set(weights, weight_values.data(), 0,
                                ggml_nbytes(weights));
        ggml_backend_tensor_set(ids, id_values.data(), 0, ggml_nbytes(ids));
        ggml_backend_tensor_set(remap, remap_values.data(), 0, ggml_nbytes(remap));
        ggml_backend_tensor_set(mask, mask_values.data(), 0, ggml_nbytes(mask));

        ggml_backend_graph_compute(backend, stock_graph);
        ggml_backend_graph_compute(backend, fused_graph);
        std::vector<float> stock_values(
            static_cast<size_t>(test.D) * test.topk * S);
        std::vector<float> fused_values(stock_values.size());
        ggml_backend_tensor_get(stock, stock_values.data(), 0,
                                stock_values.size() * sizeof(float));
        ggml_backend_tensor_get(fused, fused_values.data(), 0,
                                fused_values.size() * sizeof(float));
        ++cases;

        size_t mismatches = 0;
        size_t nonzero = 0;
        for (size_t i = 0; i < stock_values.size(); ++i) {
          uint32_t expected_bits = 0;
          uint32_t actual_bits = 0;
          std::memcpy(&expected_bits, &stock_values[i], sizeof(expected_bits));
          std::memcpy(&actual_bits, &fused_values[i], sizeof(actual_bits));
          mismatches += expected_bits != actual_bits;
          nonzero += stock_values[i] != 0.0f;
        }
        if (mismatches != 0) {
          ++failures;
          std::fprintf(stderr,
                       "FAIL %s threads=%d S=%d mask=%d: %zu/%zu values differ\n",
                       test.name, threads, S, pattern, mismatches,
                       stock_values.size());
        }
        if ((pattern == 0 && nonzero != 0) || (pattern != 0 && nonzero == 0)) {
          ++failures;
          std::fprintf(stderr,
                       "FAIL %s threads=%d S=%d mask=%d: cold output check failed\n",
                       test.name, threads, S, pattern);
        }

        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
      }
    }
  }
  std::printf("%s: %d cases, %s\n", test.name, cases,
              failures == 0 ? "PASS (bit-identical)" : "FAIL");
  return failures;
}
} // namespace

int main(int argc, char **argv) {
  const bool type_smoke = argc == 2 &&
                          std::strcmp(argv[1], "--smoke-ornith-types") == 0;
  const bool expanded_matrix =
      argc >= 2 && std::strcmp(argv[1], "--ornith-expanded-matrix") == 0;
  if (argc > 1 && ((!type_smoke && !expanded_matrix) ||
                   (expanded_matrix && argc > 3))) {
    std::fprintf(stderr,
                 "usage: cold_ffn_fused_test [--smoke-ornith-types | "
                 "--ornith-expanded-matrix [case-name]]\n");
    return 2;
  }
  const char *case_filter = expanded_matrix && argc == 3 ? argv[2] : nullptr;
  std::mt19937 rng(11);
  const Case cases[] = {
      {"q2_0", GGML_TYPE_Q2_0, GGML_TYPE_Q2_0, GGML_TYPE_Q2_0, 256, 128,
       true},
      {"iq3_xxs", GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_XXS,
       GGML_TYPE_IQ3_XXS, 512, 256, true},
      {"ornith-iq3_xxs-actual-shape", GGML_TYPE_IQ3_XXS,
       GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_XXS, 3072, 1024, false, 8, 8},
      {"ornith-mixed-iq-actual-shape", GGML_TYPE_IQ2_S, GGML_TYPE_IQ2_S,
       GGML_TYPE_IQ3_XXS, 3072, 1024, false, 8, 8},
      {"iq2_s-iq3_xxs-iq4_xs", GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS,
       GGML_TYPE_IQ4_XS, 512, 256, true},
      {"iq2_xxs-iq2_xxs-iq3_xxs", GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS,
       GGML_TYPE_IQ3_XXS, 512, 256, false},
      {"iq2_s-iq2_s-iq3_xxs", GGML_TYPE_IQ2_S, GGML_TYPE_IQ2_S,
       GGML_TYPE_IQ3_XXS, 512, 256, false},
      {"iq3_xxs-iq3_xxs-iq4_xs", GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_XXS,
       GGML_TYPE_IQ4_XS, 512, 256, false},
      {"iq2_xxs-iq2_xxs-q2_k", GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS,
       GGML_TYPE_Q2_K, 512, 256, false},
      {"iq2_xxs-iq2_xxs-iq2_s", GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS,
       GGML_TYPE_IQ2_S, 512, 256, false},
      {"iq3_xxs-iq3_xxs-iq3_s", GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_XXS,
       GGML_TYPE_IQ3_S, 512, 256, false},
      {"iq3_xxs-iq3_xxs-iq2_s", GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_XXS,
       GGML_TYPE_IQ2_S, 512, 256, false},
      {"iq1_s-iq1_s-iq2_s", GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_S,
       GGML_TYPE_IQ2_S, 512, 256, false},
      {"iq3_xxs-iq3_xxs-q3_k", GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_XXS,
       GGML_TYPE_Q3_K, 512, 256, false},
      {"iq1_s-iq1_s-q2_k", GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_S,
       GGML_TYPE_Q2_K, 512, 256, false},
      {"iq3_s-iq3_s-iq4_xs", GGML_TYPE_IQ3_S, GGML_TYPE_IQ3_S,
       GGML_TYPE_IQ4_XS, 512, 256, false},
      {"iq2_s-iq2_s-iq2_s", GGML_TYPE_IQ2_S, GGML_TYPE_IQ2_S,
       GGML_TYPE_IQ2_S, 512, 256, false},
  };
  int failures = 0;
  bool selected = false;
  for (const auto &test : cases) {
    if (case_filter &&
        (std::strcmp(test.name, case_filter) != 0 || test.exhaustive))
      continue;
    selected = true;
    failures += run_case(test, rng, type_smoke, expanded_matrix);
  }
  if (case_filter && !selected) {
    std::fprintf(stderr, "no expanded Ornith case named '%s'\n", case_filter);
    return 2;
  }
  return failures == 0 ? 0 : 1;
}
