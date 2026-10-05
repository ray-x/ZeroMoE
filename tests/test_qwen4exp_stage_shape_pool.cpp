// CPU-only coverage for the shared shape-class staging worker registry.
// The registry shares execution queues for identical shapes while keeping
// incompatible shapes isolated. GPU banks are intentionally not constructed
// in this test; their lifetime is validated by the CUDA graph diagnostics.
#include <cuda_runtime_api.h>

#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>

#include "memory_guard.hpp"
#include "ggml-backend.h"
#include "ggml.h"

#include "qwen4exp_stage_pool.hpp"

int main() {
  Qwen4ExpStageShapeClassPool registry;
  Qwen4ExpStageShapeClassPool::Key key;
  key.topk = 10;
  key.shape = {256, 512, 256, 512, 256, 256};
  key.type = {3, 3, 3};
  auto *same_a = registry.service_for(key);
  auto *same_b = registry.service_for(key);
  assert(same_a != nullptr && same_a == same_b);

  auto other_key = key;
  other_key.topk = 8;
  auto *different = registry.service_for(other_key);
  assert(different != nullptr && different != same_a);

  std::atomic<int> completed{0};
  for (int i = 0; i < 4; ++i)
    assert(same_a->submit([&completed] {
      completed.fetch_add(1, std::memory_order_relaxed);
    }));
  assert(different->submit([&completed] {
    completed.fetch_add(1, std::memory_order_relaxed);
  }));

  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(2);
  while (completed.load(std::memory_order_relaxed) != 5 &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  assert(completed.load(std::memory_order_relaxed) == 5);
  assert(same_a->stats().completed == 4);
  assert(different->stats().completed == 1);
  registry.shutdown();
  std::puts("PASS: shape-class staging services share compatible queues");
}
