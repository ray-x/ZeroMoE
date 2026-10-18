#pragma once
// Prototypes of the native tests defined in tests/native/*.cpp.
#include "native_test_common.hpp"

// ---- tests_basic_cuda.cpp ----
void test_cpu_matrix_ops();
void test_graph_initializer();
void test_attention_extraction();
void test_cuda_execution();
bool test_cuda_fused_kernels();
// ---- tests_cpu_quant.cpp ----
bool test_avx512_dequant();
bool test_avx512_dequant_iq();
bool test_qwen_q2_0_vnni_kernel();
bool test_cpu_iq_matmul_paths();
// ---- tests_cpu_moe.cpp ----
bool test_qwen_cpu_moe_graph();
bool test_qwen_cpu_moe_iq1m_graph();
bool test_qwen_cpu_moe_mixed_strided_graph();
bool test_qwen_cpu_moe_iq2_strided_graph();
bool bench_qwen_cpu_moe();
void bench_cpu_dequant_matmul();
// ---- tests_gpu_attention.cpp ----
bool test_bf16_cublas_dense_matmul();
bool test_score_gpu_nope_only();
bool test_bf16_matched_cublas_dense_matmul();
bool test_bf16_batched_qabsorb();
bool test_score_gpu();
bool test_wuv_batched_gpu();
bool test_score_gpu_prefill();
bool test_wuv_batched_gpu_prefill();
bool test_wuv_int8_batched_gpu_prefill();
bool test_wuv_int8_batched_gpu();
void bench_qabsorb_cpu_vs_gpu();
// ---- tests_quant_proto.cpp ----
void bench_iq2xxs_vnni_prototype();
void bench_iq3xxs_avx2_prototype();
bool test_real_expert_iq2xxs_dequant();
bool test_mixed_precision_layer();
// ---- tests_expert_cache.cpp ----
bool test_unpacked_format_relaxed_check();
bool test_compute_expert_slab_bytes_max();
bool test_compact_vram_slot_planner();
bool test_ram_sync_promotion_capacity();
bool test_ram_async_source_hold();
bool test_vram_cache_mixed_sizes();
bool test_upload_async_already_pinned_src();
bool test_expert_gpu_accum();
// ---- tests_gpu_kernels.cpp ----
bool test_dense_bf16_kquant();
bool test_gdn_conv1d_gpu();
bool test_gdn_conv_fused();
bool test_qwen_shared_expert_runner();
bool test_qwen_hc_runner();
bool test_qwen_hc_runner_q8_exact();
bool test_glm5next_dense_projection_fusion();
bool test_glm5next_kda_gpu();
bool test_glm5next_router_gpu();
// ---- tests_config.cpp ----
bool test_gpu_device_allowlist();
bool test_qwen_prefill_state_handoff_config();
bool test_graph_config();

// ---- helpers shared across test files (were file-local statics in main_test.cpp) ----
bool exact_iq_matmul_reference(GGUFType type, const uint8_t *weights,
                               const float *x, float *y, int S, int I, int O);
bool compare_iq_outputs(const std::string &label, const float *a,
                        const float *b, size_t n, float atol, float rtol);
void bench_q3k_cpu_only();
void bench_q3k_thp_cpu_only();
