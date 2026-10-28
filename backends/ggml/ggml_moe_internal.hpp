#pragma once

#include <cstddef>
#include <cstdint>

#if defined(ZEROMOE_USE_GGML_CUDA)
#include "ggml.h"

namespace moe::detail {

size_t cuda_row_pad_bytes(ggml_type type, int64_t ne0);
size_t align_up(size_t value, size_t alignment);
size_t lcm_size(size_t a, size_t b);
bool type_supported(uint32_t type);
bool copy_router_tensor_to_host(int device, const ggml_tensor *tensor,
                                void *host, size_t bytes, const char *name);
bool use_packed_transfer();
bool trace_backend_calls();

extern bool moe_ggml_planar_copies_override;

}  // namespace moe::detail
#endif
