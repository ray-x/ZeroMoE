#include "ggml_expert_copy.hpp"
#include "ggml-cpu.h"
#include <cassert>
#include <cstdio>
#include <sys/mman.h>
#include <unistd.h>

int main() {
  const size_t page = (size_t)sysconf(_SC_PAGESIZE);
  const size_t payload = 16 * sizeof(float);
  auto *source = static_cast<uint8_t *>(mmap(nullptr, 4 * page, PROT_NONE,
      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  assert(source != MAP_FAILED);
  for (int e : {0, 2}) {
    assert(mprotect(source + e * page, page, PROT_READ | PROT_WRITE) == 0);
    for (int j = 0; j < 16; ++j)
      reinterpret_cast<float *>(source + e * page)[j] = (float)(100 * e + j);
  }
  auto *output = static_cast<uint8_t *>(mmap(nullptr, 4 * page,
      PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  assert(output != MAP_FAILED);
  std::memset(output, 0xaa, 4 * page);
  alignas(64) std::array<int32_t, 4> route_values{-1, 2, 0, 2};
  ggml_context *ctx = ggml_init({1024 * 1024, nullptr, true});
  assert(ctx);
  auto *src = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 4, 4);
  auto *dst = ggml_dup_tensor(ctx, src);
  src->nb[2] = dst->nb[2] = page;
  src->nb[3] = dst->nb[3] = 4 * page;
  src->data = source;
  dst->data = output;
  src->buffer = ggml_backend_cpu_buffer_from_ptr(source, 4 * page);
  dst->buffer = ggml_backend_cpu_buffer_from_ptr(output, 4 * page);
  auto *ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 4, 1);
  ids->data = route_values.data();
  ids->buffer = ggml_backend_cpu_buffer_from_ptr(route_values.data(), sizeof(route_values));
  auto *x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 1, 1);
  auto *graph = ggml_new_graph_custom(ctx, 32, false);
  // The consumer is deliberately not the first node in its split.
  ggml_build_forward_expand(graph, ggml_add(ctx, x, x));
  ggml_build_forward_expand(graph, ggml_mul_mat_id(ctx, dst, x, ids));
  ggml_backend_t cpu = ggml_backend_cpu_init();
  assert(cpu);
  moestream::ggmlcopy::ExpertCopy copy;
  assert(copy.copy(cpu, src, dst, graph, &copy));
  assert(copy.plane_copies == 2 && copy.bytes == 2 * payload);
  for (int e : {0, 2})
    assert(std::memcmp(output + e * page, source + e * page, payload) == 0);
  // Unselected source slots remain PROT_NONE: a bulk copy would fault.
  assert(output[page + 1024] == 0xaa && output[3 * page + 1024] == 0xaa);
  for (size_t j = 0; j < 512; ++j) assert(output[payload + j] == 0);
  const std::vector<uint8_t> previous(output, output + 4 * page);
  route_values.fill(-1);
  copy.reset();
  assert(copy.copy(cpu, src, dst, graph, &copy));
  assert(copy.plane_copies == 0 && copy.bytes == 0 &&
         std::memcmp(previous.data(), output, previous.size()) == 0);
  route_values.fill(2);
  copy.reset();
  assert(copy.copy(cpu, src, dst, graph, &copy));
  assert(copy.plane_copies == 1 && copy.bytes == payload);
  ggml_backend_free(cpu);
  ggml_backend_buffer_free(ids->buffer);
  ggml_backend_buffer_free(dst->buffer);
  ggml_backend_buffer_free(src->buffer);
  ggml_free(ctx);
  munmap(source, 4 * page);
  munmap(output, 4 * page);
  std::puts("selected planes, negative routes, deduplication, padding, reset, and inaccessible unused slots: PASS");
}
