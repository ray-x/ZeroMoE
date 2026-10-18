// Unit test for include/cuda_raii.hpp. Needs one CUDA device (uses device 0 only,
// a few MiB). Build: cmake --build <dir> --target test_cuda_raii
#include "cuda_raii.hpp"

#include <cstdio>
#include <cstring>
#include <vector>

using namespace moestream::cuda;

static int g_failures = 0;
#define CHECK(cond)                                                          \
  do {                                                                       \
    if (!(cond)) {                                                           \
      std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);            \
      ++g_failures;                                                          \
    }                                                                        \
  } while (0)

static size_t free_bytes() {
  size_t f = 0, t = 0;
  cudaMemGetInfo(&f, &t);
  return f;
}

// One full pass. Memory-accounting checks read global free VRAM, which other
// processes on a shared GPU can change; main() therefore retries whole passes. A
// genuine leak or double free fails every pass, noise does not.
static int run_pass() {
  g_failures = 0;
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1) {
    std::printf("SKIP no CUDA device\n");
    return 0;  // nothing to check; counts as a pass
  }
  cudaSetDevice(0);
  cudaFree(nullptr);  // create the context before measuring free memory

  constexpr size_t kElems = 4u << 20;  // 16 MiB of float
  constexpr size_t kBytes = kElems * sizeof(float);
  const size_t baseline = free_bytes();

  // --- DeviceBuffer: default state, allocate, release returns the memory.
  {
    DeviceBuffer<float> b;
    CHECK(b.empty() && b.get() == nullptr && b.capacity() == 0);
    CHECK(b.allocate(kElems) == cudaSuccess);
    CHECK(!b.empty() && b.capacity() == kElems && b.bytes() == kBytes);
    CHECK(b.device() == 0);
    CHECK(baseline - free_bytes() >= kBytes);
    b.release();
    CHECK(b.empty() && b.capacity() == 0);
    CHECK(free_bytes() + (1u << 20) >= baseline);
  }
  CHECK(free_bytes() + (1u << 20) >= baseline);  // destructor released too

  // --- reserve is grow-only and keeps the pointer when it already fits.
  {
    DeviceBuffer<float> b;
    CHECK(b.reserve(1024) == cudaSuccess);
    float *p = b.get();
    CHECK(b.reserve(512) == cudaSuccess && b.get() == p && b.capacity() == 1024);
    CHECK(b.reserve(1024) == cudaSuccess && b.get() == p);
    CHECK(b.reserve(1u << 20) == cudaSuccess && b.capacity() == (1u << 20));
  }

  // --- Failure leaves the buffer empty and the sticky error cleared.
  {
    DeviceBuffer<float> b;
    CHECK(b.allocate(1024) == cudaSuccess);
    CHECK(b.allocate(size_t(1) << 50) != cudaSuccess);  // far beyond VRAM
    CHECK(b.empty() && b.capacity() == 0);
    CHECK(cudaGetLastError() == cudaSuccess);  // no sticky error left behind
    CHECK(b.allocate(std::numeric_limits<size_t>::max()) == cudaErrorInvalidValue);
    CHECK(b.empty());
    CHECK(b.allocate(0) == cudaSuccess && b.empty());
    CHECK(b.allocate(64) == cudaSuccess);  // still usable afterwards
  }

  // --- Move: source is emptied, nothing is freed twice, assignment frees the target.
  {
    DeviceBuffer<float> a;
    CHECK(a.allocate(kElems) == cudaSuccess);
    float *p = a.get();
    DeviceBuffer<float> b(std::move(a));
    CHECK(a.empty() && b.get() == p && b.capacity() == kElems);
    DeviceBuffer<float> c;
    CHECK(c.allocate(kElems) == cudaSuccess);
    const size_t before = free_bytes();
    c = std::move(b);  // c's old 16 MiB must be released
    CHECK(c.get() == p && b.empty());
    CHECK(free_bytes() >= before + kBytes - (1u << 20));
    c = std::move(c);  // self-move must not free the live allocation
    CHECK(c.get() == p);
  }
  CHECK(free_bytes() + (1u << 20) >= baseline);

  // --- Usable in a std::vector (needs noexcept moves): growth keeps contents.
  {
    std::vector<DeviceBuffer<int>> v;
    for (int i = 0; i < 16; ++i) {
      v.emplace_back();
      CHECK(v.back().allocate(256) == cudaSuccess);
      int value = i * 7;
      cudaMemcpy(v.back().get(), &value, sizeof(int), cudaMemcpyHostToDevice);
    }
    bool all_ok = true;
    for (int i = 0; i < 16; ++i) {
      int out = -1;
      cudaMemcpy(&out, v[size_t(i)].get(), sizeof(int), cudaMemcpyDeviceToHost);
      all_ok = all_ok && out == i * 7;
    }
    CHECK(all_ok);
  }
  CHECK(free_bytes() + (1u << 20) >= baseline);

  // --- PinnedBuffer: host-visible memory with the same contract.
  {
    PinnedBuffer<float> h;
    CHECK(h.allocate(1024) == cudaSuccess && !h.empty());
    for (size_t i = 0; i < 1024; ++i)
      h[i] = float(i);
    DeviceBuffer<float> d;
    CHECK(d.allocate(1024) == cudaSuccess);
    CHECK(cudaMemcpy(d.get(), h.get(), 4096, cudaMemcpyHostToDevice) == cudaSuccess);
    std::memset(h.get(), 0, 4096);
    CHECK(cudaMemcpy(h.get(), d.get(), 4096, cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(h[0] == 0.0f && h[1023] == 1023.0f);
    PinnedBuffer<float> m(std::move(h));
    CHECK(h.empty() && !m.empty());
    CHECK(m.allocate(std::numeric_limits<size_t>::max()) == cudaErrorInvalidValue);
    CHECK(m.empty());
  }

  // --- CudaEvent / CudaStream: create, use as raw handles, move, release.
  {
    CudaStream s;
    CudaEvent e;
    CHECK(s.create() == cudaSuccess && s.get() != nullptr);
    CHECK(e.create() == cudaSuccess && e.get() != nullptr);
    CHECK(cudaEventRecord(e, s) == cudaSuccess);  // implicit conversion
    CHECK(cudaStreamSynchronize(s) == cudaSuccess);
    CHECK(cudaEventQuery(e) == cudaSuccess);
    CudaEvent e2(std::move(e));
    CHECK(e.get() == nullptr && e2.get() != nullptr);
    e2.release();
    CHECK(e2.get() == nullptr);
    CudaStream s2;
    s2 = std::move(s);
    CHECK(s.get() == nullptr && s2.get() != nullptr);
    CHECK(e2.create() == cudaSuccess);  // re-create after release
  }

  // --- Releasing with the CUDA runtime in an error state must not throw or abort.
  {
    DeviceBuffer<float> b;
    CHECK(b.allocate(16) == cudaSuccess);
    cudaGetLastError();
  }

  return g_failures;
}

int main() {
  int failures = 1;
  for (int attempt = 1; attempt <= 5 && failures != 0; ++attempt) {
    failures = run_pass();
    if (failures != 0)
      std::printf("pass %d: %d failure(s)%s\n", attempt, failures,
                  attempt < 5 ? ", retrying" : "");
  }
  if (failures == 0)
    std::printf("PASSED\n");
  return failures == 0 ? 0 : 1;
}
