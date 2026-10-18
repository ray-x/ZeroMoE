#pragma once

#include <cstddef>
#include <cstdint>

#if defined(MOESTREAM_USE_GGML_CUDA)
#include "ggml.h"

namespace moe::detail {

size_t cuda_row_pad_bytes(ggml_type type, int64_t ne0);
size_t align_up(size_t value, size_t alignment);
size_t lcm_size(size_t a, size_t b);
bool type_supported(uint32_t type);
bool qwen_decode_sched_debug();
bool copy_router_tensor_to_host(int device, const ggml_tensor *tensor,
                                void *host, size_t bytes, const char *name);
bool use_backend_stream_copies();
bool use_direct_router_input();
bool use_packed_transfer();
bool trace_pure_h2d();
bool use_async_moe_graph_reuse();
bool use_direct_shared_inputs();
bool use_direct_moe_input();
bool debug_sync_route_inputs();
bool debug_device_sync();
bool debug_copy_sync();
bool trace_backend_calls();
bool debug_check_device_output();

extern bool backend_stream_copy_override;
extern bool moe_ggml_planar_copies_override;
extern bool async_moe_graph_reuse_override;

}  // namespace moe::detail
#endif
