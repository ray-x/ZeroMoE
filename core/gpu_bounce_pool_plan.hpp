#pragma once

#include <cstddef>
#include <vector>

namespace zeromoe::gpu {

// CUDA events are scoped to the device context in which they are created.
// Portable host memory can be used by several GPUs, but every copy stream
// needs its own completion event.
inline int bounce_event_index_for_gpu(const std::vector<int> &gpu_ids,
                                      int gpu_id) {
  int found = -1;
  for (size_t i = 0; i < gpu_ids.size(); ++i) {
    if (gpu_ids[i] != gpu_id) continue;
    if (found >= 0) return -1;
    found = static_cast<int>(i);
  }
  return found;
}

inline bool bounce_buffer_suitable(size_t copy_bytes, size_t capacity_bytes,
                                   bool target_event_ready) {
  return copy_bytes > 0 && copy_bytes <= capacity_bytes && target_event_ready;
}

} // namespace zeromoe::gpu
