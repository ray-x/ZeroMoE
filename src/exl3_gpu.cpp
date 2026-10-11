#include "exl3_gpu.hpp"

#include "exl3_cpu.hpp"
#include "exl3_cuda.hpp"
#include "exl3_format.hpp"
#include "logger.hpp"

#include <cublas_v2.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {

constexpr uint32_t kF32Type = 0;
constexpr uint32_t kExl3Type = 0x1000;

void check(cudaError_t status, const char *operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("EXL3 CUDA ") + operation + ": " +
                             cudaGetErrorString(status));
}

void check_blas(cublasStatus_t status, const char *operation) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string("EXL3 cuBLAS ") + operation +
                             " failed: " + std::to_string(static_cast<int>(status)));
}

struct DeviceGuard {
  int previous = 0;
  explicit DeviceGuard(int device) {
    check(cudaGetDevice(&previous), "get device");
    check(cudaSetDevice(device), "set device");
  }
  ~DeviceGuard() { cudaSetDevice(previous); }
};

struct Buffer {
  void *data = nullptr;
  size_t capacity = 0;
  Buffer() = default;
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  ~Buffer() { release(); }
  void release() {
    if (data) cudaFree(data);
    data = nullptr;
    capacity = 0;
  }
  void grow(size_t bytes) {
    if (bytes <= capacity) return;
    release();
    check(cudaMalloc(&data, bytes), "allocate buffer");
    capacity = bytes;
  }
  float *floats() { return static_cast<float *>(data); }
};

Exl3CudaMatrixView device_view(const void *storage,
                               const Exl3PackedMatrixHeader &header) {
  const auto *bytes = static_cast<const uint8_t *>(storage);
  return {
      reinterpret_cast<const uint16_t *>(bytes + header.trellis_offset),
      reinterpret_cast<const uint16_t *>(bytes + header.suh_offset),
      reinterpret_cast<const uint16_t *>(bytes + header.svh_offset),
      (header.flags & kExl3PackedHasBias)
          ? reinterpret_cast<const uint16_t *>(bytes + header.bias_offset) : nullptr,
      static_cast<int>(header.input_features), static_cast<int>(header.output_features),
      static_cast<int>(header.bits)};
}

struct Entry {
  Buffer weights;
  Exl3PackedMatrixHeader header{};
  int input_features = 0;
  int output_features = 0;
  uint32_t type = 0;
  bool persistent = true;
  size_t holds = 0;
  std::list<std::string>::iterator lru;
};

} // namespace

struct Exl3GpuExecutor::Impl {
  struct PendingRoutedUse {
    cudaEvent_t begin = nullptr;
    cudaEvent_t complete = nullptr;
    std::vector<Entry *> entries;
    std::vector<std::shared_ptr<void>> external_holds;
    uint64_t graph_id = 0;
  };

  struct CapturedExternalGraph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    std::vector<uintptr_t> key;
    std::vector<uintptr_t> site_key;
    uint64_t id = 0;
    uint64_t last_used = 0;
    size_t inflight = 0;
  };

  int device;
  size_t budget;
  cudaStream_t stream = nullptr;
  cublasHandle_t blas = nullptr;
  Buffer input;
  Buffer output;
  Buffer partials;
  Buffer activation;
  Buffer moe_output;
  Buffer hc_low;
  Buffer hc_gate;
  Buffer hc_inject;
  cudaEvent_t scratch_tail = nullptr;
  cudaStream_t scratch_stream = nullptr;
  std::vector<cudaEvent_t> available_events;
  std::vector<PendingRoutedUse> pending_routed;
  std::vector<CapturedExternalGraph> external_routed_graphs;
  bool capture_external_routed_graphs = false;
  uint64_t next_external_graph_id = 1;
  uint64_t external_graph_use_clock = 0;
  static constexpr size_t kMaxExternalRoutedGraphs = 128;
  std::unordered_map<std::string, std::unique_ptr<Entry>> cache;
  std::list<std::string> lru;
  mutable std::mutex mutex;
  Exl3GpuStats counters;

  Impl(int device_id, size_t cache_budget) : device(device_id), budget(cache_budget) {
    available_events.reserve(256);
    pending_routed.reserve(256);
    external_routed_graphs.reserve(kMaxExternalRoutedGraphs);
    DeviceGuard guard(device);
    const char *schedule_env = std::getenv("MOESTREAM_CUDA_SCHEDULE");
    if (schedule_env && schedule_env[0]) {
      unsigned schedule = cudaDeviceScheduleAuto;
      const char *name = "auto";
      if (std::strcmp(schedule_env, "blocking") == 0) {
        schedule = cudaDeviceScheduleBlockingSync;
        name = "blocking";
      } else if (std::strcmp(schedule_env, "spin") == 0) {
        schedule = cudaDeviceScheduleSpin;
        name = "spin";
      }
      unsigned flags = 0;
      check(cudaGetDeviceFlags(&flags), "get scheduling flags");
      check(cudaSetDeviceFlags((flags & ~cudaDeviceScheduleMask) | schedule),
            "set scheduling flags");
      LOG_INFO("[EXL3-CUDA-SCHEDULE] device=%d mode=%s\n", device, name);
    }
    size_t free_bytes = 0, total_bytes = 0;
    check(cudaMemGetInfo(&free_bytes, &total_bytes), "query VRAM");
    constexpr size_t reserve = 256ull * 1024 * 1024;
    if (free_bytes <= reserve || budget == 0)
      throw std::runtime_error("EXL3 GPU needs a positive cache budget and free VRAM");
    budget = std::min(budget, free_bytes - reserve);
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    try {
      check_blas(cublasCreate(&blas), "create handle");
      check_blas(cublasSetStream(blas, stream), "set stream");
      check_blas(cublasSetMathMode(blas, CUBLAS_PEDANTIC_MATH), "set F32 math");
    } catch (...) {
      if (blas) cublasDestroy(blas);
      cudaStreamDestroy(stream);
      throw;
    }
  }

  ~Impl() {
    int previous = 0;
    cudaGetDevice(&previous);
    cudaSetDevice(device);
    cudaStreamSynchronize(stream);
    for (auto &pending : pending_routed) {
      cudaEventSynchronize(pending.complete);
      for (Entry *entry : pending.entries)
        --entry->holds;
      cudaEventDestroy(pending.begin);
      cudaEventDestroy(pending.complete);
    }
    pending_routed.clear();
    for (const auto &captured : external_routed_graphs) {
      if (captured.executable)
        cudaGraphExecDestroy(captured.executable);
      if (captured.graph)
        cudaGraphDestroy(captured.graph);
    }
    external_routed_graphs.clear();
    for (cudaEvent_t event : available_events)
      cudaEventDestroy(event);
    available_events.clear();
    cache.clear();
    input.release();
    output.release();
    partials.release();
    activation.release();
    moe_output.release();
    hc_low.release();
    hc_gate.release();
    hc_inject.release();
    cublasDestroy(blas);
    cudaStreamDestroy(stream);
    cudaSetDevice(previous);
  }

  cudaEvent_t acquire_event() {
    if (!available_events.empty()) {
      const cudaEvent_t event = available_events.back();
      available_events.pop_back();
      return event;
    }
    cudaEvent_t event = nullptr;
    check(cudaEventCreate(&event), "create routed completion event");
    return event;
  }

  void recycle_event(cudaEvent_t event) {
    if (!event)
      return;
    if (available_events.size() < available_events.capacity())
      available_events.push_back(event);
    else
      cudaEventDestroy(event);
  }

  CapturedExternalGraph *find_external_graph(
      const std::vector<uintptr_t> &key) {
    for (auto &captured : external_routed_graphs) {
      if (captured.key == key) {
        captured.last_used = ++external_graph_use_clock;
        ++captured.inflight;
        ++counters.routed_graph_replays;
        return &captured;
      }
    }
    return nullptr;
  }

  bool has_external_graph_for_site(
      const std::vector<uintptr_t> &site_key) const {
    return std::any_of(external_routed_graphs.begin(),
                       external_routed_graphs.end(),
                       [&](const CapturedExternalGraph &captured) {
                         return captured.site_key == site_key;
                       });
  }

  bool make_external_graph_room() {
    if (external_routed_graphs.size() < kMaxExternalRoutedGraphs)
      return true;
    auto victim = external_routed_graphs.end();
    for (auto candidate = external_routed_graphs.begin();
         candidate != external_routed_graphs.end(); ++candidate) {
      if (candidate->inflight == 0 &&
          (victim == external_routed_graphs.end() ||
           candidate->last_used < victim->last_used))
        victim = candidate;
    }
    if (victim == external_routed_graphs.end())
      return false;
    if (victim->executable)
      cudaGraphExecDestroy(victim->executable);
    if (victim->graph)
      cudaGraphDestroy(victim->graph);
    external_routed_graphs.erase(victim);
    return true;
  }

  uint64_t store_external_graph(std::vector<uintptr_t> key,
                                std::vector<uintptr_t> site_key,
                                cudaGraph_t graph,
                                cudaGraphExec_t executable) {
    CapturedExternalGraph captured;
    captured.graph = graph;
    captured.executable = executable;
    captured.key = std::move(key);
    captured.site_key = std::move(site_key);
    captured.id = next_external_graph_id++;
    captured.last_used = ++external_graph_use_clock;
    captured.inflight = 1;
    external_routed_graphs.push_back(std::move(captured));
    ++counters.routed_graph_captures;
    return external_routed_graphs.back().id;
  }

  void release_external_graph_use(uint64_t graph_id) {
    if (graph_id == 0)
      return;
    for (auto &captured : external_routed_graphs) {
      if (captured.id == graph_id) {
        if (captured.inflight > 0)
          --captured.inflight;
        return;
      }
    }
  }

  void wait_for_scratch(cudaStream_t target_stream) {
    if (scratch_tail && target_stream != scratch_stream)
      check(cudaStreamWaitEvent(target_stream, scratch_tail, 0),
            "wait for routed scratch reuse");
  }

  void prepare_scratch(size_t activation_bytes, size_t partial_bytes,
                       cudaStream_t target_stream) {
    const bool grows = activation_bytes > activation.capacity ||
                       partial_bytes > partials.capacity;
    if (grows && scratch_tail) {
      check(cudaEventSynchronize(scratch_tail),
            "wait before growing routed scratch");
      reap_completed();
    }
    activation.grow(activation_bytes);
    partials.grow(partial_bytes);
    wait_for_scratch(target_stream);
  }

  void reap_completed() {
    size_t index = 0;
    while (index < pending_routed.size()) {
      PendingRoutedUse &pending = pending_routed[index];
      const cudaError_t status = cudaEventQuery(pending.complete);
      if (status == cudaErrorNotReady) {
        ++index;
        continue;
      }
      check(status, "query routed completion event");
      float elapsed_ms = 0.0f;
      check(cudaEventElapsedTime(&elapsed_ms, pending.begin, pending.complete),
            "measure routed FFN time");
      counters.matmul_seconds += elapsed_ms * 0.001;
      counters.routed_ffn_seconds += elapsed_ms * 0.001;
      for (Entry *entry : pending.entries)
        --entry->holds;
      if (pending.graph_id != 0) {
        for (auto &captured : external_routed_graphs) {
          if (captured.id == pending.graph_id) {
            if (captured.inflight > 0)
              --captured.inflight;
            break;
          }
        }
      }
      if (scratch_tail == pending.complete) {
        scratch_tail = nullptr;
        scratch_stream = nullptr;
      }
      recycle_event(pending.begin);
      recycle_event(pending.complete);
      if (index + 1 != pending_routed.size())
        pending_routed[index] = std::move(pending_routed.back());
      pending_routed.pop_back();
    }
  }

  std::unique_ptr<Entry> make_room(size_t bytes) {
    reap_completed();
    std::unique_ptr<Entry> reusable;
    if (bytes > budget)
      throw std::runtime_error("EXL3 GPU weight exceeds configured VRAM cache budget");
    while (counters.resident_bytes > budget - bytes) {
      auto victim = lru.end();
      for (auto candidate = lru.rbegin(); candidate != lru.rend(); ++candidate) {
        if (!cache.at(*candidate)->persistent && cache.at(*candidate)->holds == 0) {
          victim = std::prev(candidate.base());
          break;
        }
      }
      if (victim == lru.end() && !pending_routed.empty()) {
        check(cudaEventSynchronize(pending_routed.front().complete),
              "wait for routed cache slot release");
        reap_completed();
        continue;
      }
      if (victim == lru.end())
        throw std::runtime_error("EXL3 GPU cache budget is occupied by dense or held weights; increase exl3_gpu_cache_gb");
      const std::string key = *victim;
      counters.resident_bytes -= cache.at(key)->weights.capacity;
      if (!reusable && cache.at(key)->weights.capacity == bytes)
        reusable = std::move(cache.at(key));
      cache.erase(key);
      lru.erase(victim);
      ++counters.evictions;
    }
    return reusable;
  }

  Entry &upload(const std::string &key, uint32_t type, const uint8_t *weights,
                  size_t bytes, int input_features, int output_features, bool persistent) {
    if (!weights || bytes == 0 || (type != kExl3Type && type != kF32Type))
      throw std::runtime_error("Unsupported EXL3 GPU weight type");
    auto entry = std::make_unique<Entry>();
    entry->type = type;
    entry->persistent = persistent;
    entry->input_features = input_features;
    entry->output_features = output_features;
    if (type == kExl3Type) {
      Exl3Mul1MatrixView view;
      if (!exl3_mul1_view_from_packed(weights, bytes, view) ||
          view.input_features != input_features || view.output_features != output_features)
        throw std::runtime_error("Invalid EXL3 GPU packed weight");
      std::memcpy(&entry->header, weights, sizeof(entry->header));
    } else if (bytes != static_cast<size_t>(input_features) * output_features * sizeof(float)) {
      throw std::runtime_error("EXL3 GPU F32 weight size mismatch");
    }
    auto reusable = make_room(bytes);
    if (reusable) {
      std::swap(entry->weights.data, reusable->weights.data);
      std::swap(entry->weights.capacity, reusable->weights.capacity);
      ++counters.buffer_reuses;
    }
    const auto upload_begin = std::chrono::steady_clock::now();
    entry->weights.grow(bytes);
    check(cudaMemcpyAsync(entry->weights.data, weights, bytes, cudaMemcpyHostToDevice, stream), "upload weights");
    check(cudaStreamSynchronize(stream), "complete weight upload");
    counters.upload_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - upload_begin).count();
    lru.push_front(key);
    entry->lru = lru.begin();
    auto inserted = cache.emplace(key, std::move(entry)).first;
    counters.resident_bytes += bytes;
    ++counters.uploads;
    return *inserted->second;
  }

  void execute(Entry &entry, const float *host_input, float *host_output,
                int rows, int input_features, int output_features) {
    const auto begin = std::chrono::steady_clock::now();
    if (!host_input || !host_output || rows <= 0 ||
        input_features != entry.input_features || output_features != entry.output_features)
      throw std::runtime_error("EXL3 GPU projection shape mismatch");
    const size_t input_bytes = static_cast<size_t>(rows) * input_features * sizeof(float);
    const size_t output_bytes = static_cast<size_t>(rows) * output_features * sizeof(float);
    input.grow(input_bytes);
    output.grow(output_bytes);
    check(cudaMemcpyAsync(input.data, host_input, input_bytes, cudaMemcpyHostToDevice, stream), "upload input");
    if (entry.type == kExl3Type) {
      prepare_scratch(0,
                      output_bytes * exl3_cuda_key_splits(input_features),
                      stream);
      check(launch_exl3_mul1_matmul(device_view(entry.weights.data, entry.header),
                                     input.floats(), output.floats(), rows,
                                     partials.floats(), stream),
             "launch matmul");
    } else {
      const float alpha = 1.0f, beta = 0.0f;
      check_blas(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N,
                              output_features, rows, input_features, &alpha,
                              static_cast<const float *>(entry.weights.data), input_features,
                              input.floats(), input_features, &beta, output.floats(), output_features),
                   "dense matmul");
    }
    check(cudaMemcpyAsync(host_output, output.data, output_bytes, cudaMemcpyDeviceToHost, stream), "download output");
    check(cudaStreamSynchronize(stream), "complete matmul");
    ++counters.matmuls;
    ++counters.standalone_matmuls;
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - begin).count();
    counters.matmul_seconds += elapsed;
    counters.standalone_matmul_seconds += elapsed;
  }
};

Exl3GpuExecutor::Exl3GpuExecutor(int device, size_t cache_budget)
    : impl_(std::make_unique<Impl>(device, cache_budget)) {}

Exl3GpuExecutor::~Exl3GpuExecutor() = default;

int Exl3GpuExecutor::device() const { return impl_->device; }

void Exl3GpuExecutor::set_routed_graph_capture(bool enabled) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  DeviceGuard guard(impl_->device);
  if (impl_->capture_external_routed_graphs == enabled)
    return;
  if (!enabled) {
    for (const auto &pending : impl_->pending_routed)
      check(cudaEventSynchronize(pending.complete),
            "wait for external routed graph before disabling capture");
    impl_->reap_completed();
    for (const auto &captured : impl_->external_routed_graphs) {
      if (captured.executable)
        cudaGraphExecDestroy(captured.executable);
      if (captured.graph)
        cudaGraphDestroy(captured.graph);
    }
    impl_->external_routed_graphs.clear();
  }
  impl_->capture_external_routed_graphs = enabled;
}

bool Exl3GpuExecutor::matmul_cached(const std::string &key, const float *input,
                                    float *output, int rows, int input_features,
                                    int output_features) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  DeviceGuard guard(impl_->device);
  impl_->reap_completed();
  auto found = impl_->cache.find(key);
  if (found == impl_->cache.end()) return false;
  impl_->lru.splice(impl_->lru.begin(), impl_->lru, found->second->lru);
  impl_->execute(*found->second, input, output, rows, input_features, output_features);
  ++impl_->counters.cache_hits;
  return true;
}

void Exl3GpuExecutor::prepare_matrix_cache(
    const std::string &key, uint32_t type, const uint8_t *weights,
    size_t bytes, int input_features, int output_features, bool persistent) {
  if (key.empty() || !weights || bytes == 0 || input_features <= 0 ||
      output_features <= 0)
    throw std::runtime_error("Invalid EXL3 matrix cache preparation");
  std::lock_guard<std::mutex> lock(impl_->mutex);
  DeviceGuard guard(impl_->device);
  impl_->reap_completed();
  auto found = impl_->cache.find(key);
  if (found != impl_->cache.end()) {
    Entry &entry = *found->second;
    if (entry.type != type || entry.weights.capacity != bytes ||
        entry.input_features != input_features ||
        entry.output_features != output_features)
      throw std::runtime_error("EXL3 GPU cache key reused with different weights: " + key);
    entry.persistent = entry.persistent || persistent;
    impl_->lru.splice(impl_->lru.begin(), impl_->lru, entry.lru);
    ++impl_->counters.cache_hits;
    return;
  }
  (void)impl_->upload(key, type, weights, bytes, input_features,
                      output_features, persistent);
}

bool Exl3GpuExecutor::matmul_device_cached(
    const std::string &key, const float *device_input, float *device_output,
    int rows, int input_features, int output_features, cudaStream_t stream) {
  if (!device_input || !device_output || rows <= 0 || input_features <= 0 ||
      output_features <= 0 || !stream)
    throw std::runtime_error("Invalid EXL3 device matmul arguments");
  std::lock_guard<std::mutex> lock(impl_->mutex);
  DeviceGuard guard(impl_->device);
  impl_->reap_completed();
  auto found = impl_->cache.find(key);
  if (found == impl_->cache.end())
    return false;
  Entry *entry = found->second.get();
  if (entry->input_features != input_features ||
      entry->output_features != output_features)
    throw std::runtime_error("EXL3 device matmul cache shape mismatch: " + key);
  impl_->lru.splice(impl_->lru.begin(), impl_->lru, entry->lru);
  ++entry->holds;
  struct HeldEntry {
    Entry *entry;
    ~HeldEntry() { --entry->holds; }
  } held{entry};

  const auto begin = std::chrono::steady_clock::now();
  const size_t output_bytes =
      static_cast<size_t>(rows) * output_features * sizeof(float);
  if (entry->type == kExl3Type) {
    impl_->prepare_scratch(
        0, output_bytes * exl3_cuda_key_splits(input_features), stream);
    check(launch_exl3_mul1_matmul(
              device_view(entry->weights.data, entry->header), device_input,
              device_output, rows, impl_->partials.floats(), stream),
          "launch device cached EXL3 matmul");
  } else if (entry->type == kF32Type) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    check_blas(cublasSetStream(impl_->blas, stream),
               "set device matmul stream");
    check_blas(cublasSgemm(
                   impl_->blas, CUBLAS_OP_T, CUBLAS_OP_N, output_features,
                   rows, input_features, &alpha,
                   static_cast<const float *>(entry->weights.data),
                   input_features, device_input, input_features, &beta,
                   device_output, output_features),
               "launch device cached dense matmul");
    check_blas(cublasSetStream(impl_->blas, impl_->stream),
               "restore executor stream");
  } else {
    throw std::runtime_error("Unsupported EXL3 device matmul cache type: " + key);
  }
  check(cudaStreamSynchronize(stream), "complete device cached matmul");
  ++impl_->counters.matmuls;
  ++impl_->counters.standalone_matmuls;
  const double elapsed = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - begin)
                             .count();
  impl_->counters.matmul_seconds += elapsed;
  impl_->counters.standalone_matmul_seconds += elapsed;
  return true;
}

void Exl3GpuExecutor::matmul(const std::string &key, uint32_t type,
                             const uint8_t *weights, size_t bytes,
                             const float *input, float *output, int rows,
                             int input_features, int output_features, bool persistent) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  DeviceGuard guard(impl_->device);
  impl_->reap_completed();
  auto found = impl_->cache.find(key);
  if (found != impl_->cache.end()) {
    if (found->second->type != type || found->second->weights.capacity != bytes)
      throw std::runtime_error("EXL3 GPU cache key reused with different weights: " + key);
    impl_->lru.splice(impl_->lru.begin(), impl_->lru, found->second->lru);
    impl_->execute(*found->second, input, output, rows, input_features, output_features);
    ++impl_->counters.cache_hits;
    return;
  }
  auto &entry = impl_->upload(key, type, weights, bytes, input_features, output_features, persistent);
  impl_->execute(entry, input, output, rows, input_features, output_features);
}

void Exl3GpuExecutor::hc_mix(
    const Exl3GpuF32Weight &down, const Exl3GpuF32Weight &up,
    const Exl3GpuF32Weight *inject, const float *normed, float *hidden,
    float *injection, int rows, int width, int streams, bool divide_each_stream) {
  const int64_t expanded_width = static_cast<int64_t>(width) * streams;
  const int rank = down.output_features;
  if (!normed || !hidden || rows <= 0 || width <= 0 || streams <= 0 ||
      expanded_width > INT_MAX || rank <= 0 ||
      static_cast<int64_t>(rows) * width > INT_MAX ||
      static_cast<int64_t>(rows) * rank > INT_MAX ||
      down.input_features != expanded_width || up.input_features != rank ||
      up.output_features != expanded_width ||
      (inject && (!injection || inject->input_features != expanded_width ||
                  inject->output_features != streams)))
    throw std::runtime_error("Invalid EXL3 GPU HyperConnection shapes");
  std::lock_guard<std::mutex> lock(impl_->mutex);
  DeviceGuard guard(impl_->device);
  impl_->reap_completed();
  struct Held {
    std::vector<Entry *> entries;
    ~Held() { for (auto *entry : entries) --entry->holds; }
  } held;
  held.entries.reserve(3);
  const auto obtain = [&](const Exl3GpuF32Weight &weight) -> Entry & {
    if (!weight.values || weight.key.empty())
      throw std::runtime_error("Missing EXL3 GPU HyperConnection weight");
    const size_t bytes = static_cast<size_t>(weight.input_features) *
                         weight.output_features * sizeof(float);
    auto found = impl_->cache.find(weight.key);
    Entry *entry = nullptr;
    if (found == impl_->cache.end()) {
      entry = &impl_->upload(weight.key, kF32Type,
                            reinterpret_cast<const uint8_t *>(weight.values), bytes,
                            weight.input_features, weight.output_features, true);
    } else {
      entry = found->second.get();
      if (entry->type != kF32Type || entry->input_features != weight.input_features ||
          entry->output_features != weight.output_features || entry->weights.capacity != bytes)
        throw std::runtime_error("EXL3 GPU HyperConnection cache shape mismatch: " + weight.key);
      entry->persistent = true;
      impl_->lru.splice(impl_->lru.begin(), impl_->lru, entry->lru);
      ++impl_->counters.cache_hits;
    }
    ++entry->holds;
    held.entries.push_back(entry);
    return *entry;
  };
  Entry &down_entry = obtain(down);
  Entry &up_entry = obtain(up);
  Entry *inject_entry = inject ? &obtain(*inject) : nullptr;
  const auto begin = std::chrono::steady_clock::now();
  const size_t input_bytes = static_cast<size_t>(rows) * expanded_width * sizeof(float);
  const size_t output_bytes = static_cast<size_t>(rows) * width * sizeof(float);
  const size_t low_bytes = static_cast<size_t>(rows) * rank * sizeof(float);
  const size_t inject_bytes = static_cast<size_t>(rows) * streams * sizeof(float);
  impl_->input.grow(input_bytes);
  impl_->output.grow(output_bytes);
  impl_->hc_low.grow(low_bytes);
  impl_->hc_gate.grow(input_bytes);
  if (inject_entry) impl_->hc_inject.grow(inject_bytes);
  check(cudaMemcpyAsync(impl_->input.data, normed, input_bytes,
                         cudaMemcpyHostToDevice, impl_->stream), "upload HC input");
  const auto project = [&](const Entry &entry, const float *source, float *destination) {
    const float alpha = 1.0f, beta = 0.0f;
    check_blas(cublasSgemm(impl_->blas, CUBLAS_OP_T, CUBLAS_OP_N,
                            entry.output_features, rows, entry.input_features, &alpha,
                            static_cast<const float *>(entry.weights.data), entry.input_features,
                            source, entry.input_features, &beta, destination, entry.output_features),
               "HC projection");
  };
  project(down_entry, impl_->input.floats(), impl_->hc_low.floats());
  check(launch_exl3_hc_silu(impl_->hc_low.floats(), rows * rank, streams, impl_->stream),
        "HC low-rank activation");
  project(up_entry, impl_->hc_low.floats(), impl_->hc_gate.floats());
  check(launch_exl3_hc_mix(impl_->input.floats(), impl_->hc_gate.floats(),
                           impl_->output.floats(), rows, width, streams,
                           divide_each_stream, impl_->stream), "HC gated mixing");
  if (inject_entry)
    project(*inject_entry, impl_->input.floats(), impl_->hc_inject.floats());
  check(cudaMemcpyAsync(hidden, impl_->output.data, output_bytes,
                         cudaMemcpyDeviceToHost, impl_->stream), "download HC output");
  if (inject_entry)
    check(cudaMemcpyAsync(injection, impl_->hc_inject.data, inject_bytes,
                           cudaMemcpyDeviceToHost, impl_->stream), "download HC injection");
  check(cudaStreamSynchronize(impl_->stream), "complete HC mix");
  const uint64_t projections = inject_entry ? 3 : 2;
  impl_->counters.matmuls += projections;
  impl_->counters.standalone_matmuls += projections;
  ++impl_->counters.hc_mixes;
  const double elapsed = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - begin).count();
  impl_->counters.matmul_seconds += elapsed;
  impl_->counters.standalone_matmul_seconds += elapsed;
}

void Exl3GpuExecutor::routed_ffn(
    const std::vector<std::string> &expert_prefixes, const float *routing_weights,
    const float *host_input, float *host_output, int hidden, int intermediate,
    const std::function<std::vector<uint8_t>(const std::string &)> &load) {
  if (expert_prefixes.empty() || !routing_weights || !host_input || !host_output ||
      hidden <= 0 || intermediate <= 0 || hidden % 128 || intermediate % 128 || !load)
    throw std::runtime_error("Invalid EXL3 routed FFN arguments");
  std::lock_guard<std::mutex> lock(impl_->mutex);
  DeviceGuard guard(impl_->device);
  impl_->reap_completed();
  struct Held {
    std::vector<Entry *> entries;
    ~Held() { for (auto *entry : entries) --entry->holds; }
  } held;
  held.entries.reserve(expert_prefixes.size() * 3);
  for (const auto &prefix : expert_prefixes) {
    for (const char *suffix : {"gate_proj", "up_proj", "down_proj"}) {
      const std::string key = prefix + suffix;
      const bool down = std::strcmp(suffix, "down_proj") == 0;
      const int input_width = down ? intermediate : hidden;
      const int output_width = down ? hidden : intermediate;
      auto found = impl_->cache.find(key);
      Entry *entry = nullptr;
      if (found == impl_->cache.end()) {
        auto packed = load(key);
        entry = &impl_->upload(key, kExl3Type, packed.data(), packed.size(),
                               input_width, output_width, false);
      } else {
        entry = found->second.get();
        if (entry->type != kExl3Type || entry->input_features != input_width || entry->output_features != output_width)
          throw std::runtime_error("EXL3 routed FFN cache shape mismatch: " + key);
        impl_->lru.splice(impl_->lru.begin(), impl_->lru, entry->lru);
        ++impl_->counters.cache_hits;
      }
      ++entry->holds;
      held.entries.push_back(entry);
    }
  }
  const auto begin = std::chrono::steady_clock::now();
  const size_t hidden_bytes = static_cast<size_t>(hidden) * sizeof(float);
  const size_t intermediate_bytes = static_cast<size_t>(intermediate) * sizeof(float);
  const size_t gate_partial_bytes =
      intermediate_bytes * exl3_cuda_key_splits(hidden);
  const size_t down_partial_bytes =
      hidden_bytes * exl3_cuda_key_splits(intermediate);
  impl_->input.grow(hidden_bytes);
  impl_->prepare_scratch(
      intermediate_bytes, std::max(2 * gate_partial_bytes, down_partial_bytes),
      impl_->stream);
  impl_->moe_output.grow(hidden_bytes);
  check(cudaMemcpyAsync(impl_->input.data, host_input, hidden_bytes, cudaMemcpyHostToDevice, impl_->stream), "upload FFN input");
  check(cudaMemsetAsync(impl_->moe_output.data, 0, hidden_bytes, impl_->stream), "clear FFN output");
  float *up_partials = impl_->partials.floats() + gate_partial_bytes / sizeof(float);
  for (size_t expert = 0; expert < expert_prefixes.size(); ++expert) {
    check(launch_exl3_mul1_gate_up(
              device_view(held.entries[expert * 3]->weights.data,
                          held.entries[expert * 3]->header),
              device_view(held.entries[expert * 3 + 1]->weights.data,
                          held.entries[expert * 3 + 1]->header),
              impl_->input.floats(), impl_->activation.floats(), 1,
              impl_->partials.floats(), up_partials, impl_->stream),
          "launch fused gate/up projection");
    check(launch_exl3_mul1_matmul_accumulate(
              device_view(held.entries[expert * 3 + 2]->weights.data,
                          held.entries[expert * 3 + 2]->header),
              impl_->activation.floats(), impl_->moe_output.floats(),
              routing_weights[expert], 1, impl_->partials.floats(), impl_->stream),
          "launch weighted down projection");
  }
  check(cudaMemcpyAsync(host_output, impl_->moe_output.data, hidden_bytes, cudaMemcpyDeviceToHost, impl_->stream), "download FFN output");
  check(cudaStreamSynchronize(impl_->stream), "complete FFN");
  impl_->counters.matmuls += 3 * expert_prefixes.size();
  ++impl_->counters.routed_ffns;
  const double elapsed = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - begin).count();
  impl_->counters.matmul_seconds += elapsed;
  impl_->counters.routed_ffn_seconds += elapsed;
}

void Exl3GpuExecutor::routed_ffn_device(
    const std::vector<std::string> &expert_prefixes_by_id,
    const int32_t *route_ids, const float *routing_weights,
    const float *device_input, float *device_output, int rows, int topk,
    int hidden, int intermediate, cudaStream_t stream,
    bool require_cache_hit,
    const std::function<std::vector<uint8_t>(const std::string &)> &load,
    const float *device_routing_weights) {
  if (rows <= 0 || topk <= 0 || expert_prefixes_by_id.empty() ||
      !route_ids || !routing_weights || !device_input || !device_output || !stream ||
      hidden <= 0 || intermediate <= 0 || hidden % 128 ||
      intermediate % 128 || (!require_cache_hit && !load))
    throw std::runtime_error("Invalid device EXL3 routed FFN arguments");

  std::lock_guard<std::mutex> lock(impl_->mutex);
  DeviceGuard guard(impl_->device);
  impl_->reap_completed();
  const size_t route_count = static_cast<size_t>(rows) * topk;
  std::vector<Entry *> entries;
  entries.reserve(route_count * 3);
  struct Held {
    std::vector<Entry *> &entries;
    bool deferred = false;
    ~Held() {
      if (!deferred)
        for (auto *entry : entries)
          --entry->holds;
    }
  } held{entries};

  for (size_t route_index = 0; route_index < route_count; ++route_index) {
    const int expert_id = route_ids[route_index];
    if (expert_id < 0 ||
        static_cast<size_t>(expert_id) >= expert_prefixes_by_id.size())
      throw std::runtime_error("EXL3 device routed expert ID out of range");
    const std::string &prefix =
        expert_prefixes_by_id[static_cast<size_t>(expert_id)];
    for (const char *suffix : {"gate_proj", "up_proj", "down_proj"}) {
      const std::string key = prefix + suffix;
      const bool down = std::strcmp(suffix, "down_proj") == 0;
      const int input_width = down ? intermediate : hidden;
      const int output_width = down ? hidden : intermediate;
      auto found = impl_->cache.find(key);
      Entry *entry = nullptr;
      if (found == impl_->cache.end()) {
        if (require_cache_hit)
          throw std::runtime_error("EXL3 GPU cache-only routed FFN miss: " +
                                   key);
        auto packed = load(key);
        entry = &impl_->upload(key, kExl3Type, packed.data(), packed.size(),
                               input_width, output_width, false);
      } else {
        entry = found->second.get();
        if (entry->type != kExl3Type ||
            entry->input_features != input_width ||
            entry->output_features != output_width)
          throw std::runtime_error("EXL3 routed FFN cache shape mismatch: " +
                                   key);
        impl_->lru.splice(impl_->lru.begin(), impl_->lru, entry->lru);
        ++impl_->counters.cache_hits;
      }
      ++entry->holds;
      entries.push_back(entry);
    }
  }

  const size_t hidden_bytes = static_cast<size_t>(hidden) * sizeof(float);
  const size_t intermediate_bytes =
      static_cast<size_t>(intermediate) * sizeof(float);
  const size_t gate_partial_bytes =
      intermediate_bytes * exl3_cuda_key_splits(hidden);
  const size_t down_partial_bytes =
      hidden_bytes * exl3_cuda_key_splits(intermediate);
  impl_->prepare_scratch(
      intermediate_bytes, std::max(2 * gate_partial_bytes, down_partial_bytes),
      stream);
  float *up_partials =
      impl_->partials.floats() + gate_partial_bytes / sizeof(float);
  if (impl_->pending_routed.size() == impl_->pending_routed.capacity())
    impl_->pending_routed.reserve(impl_->pending_routed.size() + 1);
  cudaEvent_t begin_event = nullptr;
  cudaEvent_t complete_event = nullptr;
  try {
    begin_event = impl_->acquire_event();
    complete_event = impl_->acquire_event();
    check(cudaEventRecord(begin_event, stream),
          "record routed FFN start event");
    check(cudaMemsetAsync(device_output, 0,
                          static_cast<size_t>(rows) * hidden_bytes, stream),
          "clear device routed FFN output");
    for (int row = 0; row < rows; ++row) {
      const float *row_input = device_input + static_cast<size_t>(row) * hidden;
      float *row_output = device_output + static_cast<size_t>(row) * hidden;
      for (int expert = 0; expert < topk; ++expert) {
        const size_t route_index = static_cast<size_t>(row) * topk + expert;
        Entry *gate = entries[route_index * 3];
        Entry *up = entries[route_index * 3 + 1];
        Entry *down = entries[route_index * 3 + 2];
        check(launch_exl3_mul1_gate_up(
                  device_view(gate->weights.data, gate->header),
                  device_view(up->weights.data, up->header), row_input,
                  impl_->activation.floats(), 1, impl_->partials.floats(),
                  up_partials, stream),
              "launch device fused gate/up projection");
        if (device_routing_weights) {
          check(launch_exl3_mul1_matmul_accumulate_device_weight(
                    device_view(down->weights.data, down->header),
                    impl_->activation.floats(), row_output,
                    device_routing_weights + route_index, 1,
                    impl_->partials.floats(), stream),
                "launch device weighted down projection");
        } else {
          check(launch_exl3_mul1_matmul_accumulate(
                    device_view(down->weights.data, down->header),
                    impl_->activation.floats(), row_output,
                    routing_weights[route_index], 1, impl_->partials.floats(),
                    stream),
                "launch device weighted down projection");
        }
      }
    }
    check(cudaEventRecord(complete_event, stream),
          "record routed FFN completion event");
    Impl::PendingRoutedUse pending;
    pending.begin = begin_event;
    pending.complete = complete_event;
    pending.entries = std::move(entries);
    impl_->pending_routed.push_back(std::move(pending));
    impl_->scratch_tail = complete_event;
    impl_->scratch_stream = stream;
    begin_event = nullptr;
    complete_event = nullptr;
    held.deferred = true;
  } catch (...) {
    cudaStreamSynchronize(stream);
    impl_->recycle_event(begin_event);
    impl_->recycle_event(complete_event);
    throw;
  }
  impl_->counters.matmuls += 3 * route_count;
  ++impl_->counters.routed_ffns;
}

void Exl3GpuExecutor::routed_ffn_device_external(
    const Exl3GpuExternalExpert *experts_by_route,
    const float *routing_weights, const float *device_routing_weights,
    const float *device_input, float *device_output, int rows, int topk,
    int hidden, int intermediate, cudaStream_t stream) {
  if (!experts_by_route || (!routing_weights && !device_routing_weights) ||
      !device_input || !device_output || !stream || rows <= 0 || topk <= 0 ||
      hidden <= 0 || intermediate <= 0 || hidden % 128 || intermediate % 128)
    throw std::runtime_error("Invalid external EXL3 routed FFN arguments");

  std::lock_guard<std::mutex> lock(impl_->mutex);
  DeviceGuard guard(impl_->device);
  impl_->reap_completed();
  const size_t route_count = static_cast<size_t>(rows) * topk;
  std::vector<std::shared_ptr<void>> external_holds;
  external_holds.reserve(route_count);
  for (size_t route = 0; route < route_count; ++route) {
    const auto &expert = experts_by_route[route];
    const auto valid = [&](const Exl3CudaMatrixView &matrix, int input_width,
                           int output_width) {
      return matrix.trellis && matrix.suh && matrix.svh &&
             matrix.input_features == input_width &&
             matrix.output_features == output_width && matrix.bits >= 1 &&
             matrix.bits <= 8;
    };
    if (!valid(expert.gate, hidden, intermediate) ||
        !valid(expert.up, hidden, intermediate) ||
        !valid(expert.down, intermediate, hidden))
      throw std::runtime_error("Invalid external EXL3 expert matrix view");
    if (expert.hold)
      external_holds.push_back(expert.hold);
  }

  const size_t hidden_bytes = static_cast<size_t>(hidden) * sizeof(float);
  const size_t intermediate_bytes =
      static_cast<size_t>(intermediate) * sizeof(float);
  const size_t gate_partial_bytes =
      intermediate_bytes * exl3_cuda_key_splits(hidden);
  const size_t down_partial_bytes =
      hidden_bytes * exl3_cuda_key_splits(intermediate);
  impl_->prepare_scratch(
      intermediate_bytes, std::max(2 * gate_partial_bytes, down_partial_bytes),
      stream);
  float *up_partials =
      impl_->partials.floats() + gate_partial_bytes / sizeof(float);
  if (impl_->pending_routed.size() == impl_->pending_routed.capacity())
    impl_->pending_routed.reserve(impl_->pending_routed.size() + 1);

  const auto enqueue_sequence = [&] {
#ifdef MOESTREAM_EXL3_UPSTREAM
    if (exl3_upstream_moe_coop_enabled()) {
      std::vector<const Exl3CudaMatrixView *> gates(route_count),
          ups(route_count), downs(route_count);
      for (size_t route = 0; route < route_count; ++route) {
        gates[route] = &experts_by_route[route].gate;
        ups[route] = &experts_by_route[route].up;
        downs[route] = &experts_by_route[route].down;
      }
      const cudaError_t status = exl3_upstream_routed_ffn(
          gates.data(), ups.data(), downs.data(), routing_weights,
          device_routing_weights, device_input, device_output, rows, topk,
          hidden, intermediate, stream);
      if (status == cudaSuccess)
        return;
      if (status != cudaErrorNotSupported)
        check(status, "upstream EXL3 routed FFN");
    }
#endif
    check(cudaMemsetAsync(device_output, 0,
                          static_cast<size_t>(rows) * hidden_bytes, stream),
          "clear external routed FFN output");
    for (int row = 0; row < rows; ++row) {
      const float *row_input = device_input + static_cast<size_t>(row) * hidden;
      float *row_output = device_output + static_cast<size_t>(row) * hidden;
      for (int route = 0; route < topk; ++route) {
        const size_t route_index = static_cast<size_t>(row) * topk + route;
        const auto &expert = experts_by_route[route_index];
        check(launch_exl3_mul1_gate_up(
                  expert.gate, expert.up, row_input,
                  impl_->activation.floats(), 1, impl_->partials.floats(),
                  up_partials, stream),
              "launch external device fused gate/up projection");
        if (device_routing_weights) {
          check(launch_exl3_mul1_matmul_accumulate_device_weight(
                    expert.down, impl_->activation.floats(), row_output,
                    device_routing_weights + route_index, 1,
                    impl_->partials.floats(), stream),
                "launch external device weighted down projection");
        } else {
          check(launch_exl3_mul1_matmul_accumulate(
                    expert.down, impl_->activation.floats(), row_output,
                    routing_weights[route_index], 1, impl_->partials.floats(),
                    stream),
                "launch external weighted down projection");
        }
      }
    }
  };

  std::vector<uintptr_t> graph_key;
  std::vector<uintptr_t> graph_site_key;
  if (impl_->capture_external_routed_graphs) {
    graph_key.reserve(12 + route_count * 24);
    graph_site_key.reserve(10);
    const auto add_pointer = [&](const void *pointer) {
      graph_key.push_back(reinterpret_cast<uintptr_t>(pointer));
    };
    const auto add_matrix = [&](const Exl3CudaMatrixView &matrix) {
      add_pointer(matrix.trellis);
      add_pointer(matrix.suh);
      add_pointer(matrix.svh);
      add_pointer(matrix.bias);
      graph_key.push_back(static_cast<uintptr_t>(matrix.input_features));
      graph_key.push_back(static_cast<uintptr_t>(matrix.output_features));
      graph_key.push_back(static_cast<uintptr_t>(matrix.bits));
    };
    graph_key.push_back(static_cast<uintptr_t>(rows));
    graph_key.push_back(static_cast<uintptr_t>(topk));
    graph_key.push_back(static_cast<uintptr_t>(hidden));
    graph_key.push_back(static_cast<uintptr_t>(intermediate));
    add_pointer(device_input);
    add_pointer(device_output);
    add_pointer(device_routing_weights);
    add_pointer(impl_->activation.data);
    add_pointer(impl_->partials.data);
    add_pointer(up_partials);
    graph_site_key.push_back(static_cast<uintptr_t>(rows));
    graph_site_key.push_back(static_cast<uintptr_t>(topk));
    graph_site_key.push_back(static_cast<uintptr_t>(hidden));
    graph_site_key.push_back(static_cast<uintptr_t>(intermediate));
    graph_site_key.push_back(reinterpret_cast<uintptr_t>(device_input));
    graph_site_key.push_back(reinterpret_cast<uintptr_t>(device_output));
    graph_site_key.push_back(
        reinterpret_cast<uintptr_t>(device_routing_weights));
    graph_site_key.push_back(reinterpret_cast<uintptr_t>(impl_->activation.data));
    graph_site_key.push_back(reinterpret_cast<uintptr_t>(impl_->partials.data));
    graph_site_key.push_back(reinterpret_cast<uintptr_t>(up_partials));
    for (size_t route = 0; route < route_count; ++route) {
      const auto &expert = experts_by_route[route];
      add_matrix(expert.gate);
      add_matrix(expert.up);
      add_matrix(expert.down);
      if (!device_routing_weights) {
        uint32_t weight_bits = 0;
        std::memcpy(&weight_bits, routing_weights + route, sizeof(weight_bits));
        graph_key.push_back(static_cast<uintptr_t>(weight_bits));
      }
    }
  }

  cudaEvent_t begin_event = nullptr;
  cudaEvent_t complete_event = nullptr;
  uint64_t graph_id = 0;
  bool graph_use_registered = false;
  const bool trace_graph_capture =
      std::getenv("MOESTREAM_EXL3_GRAPH_TRACE") != nullptr;
  const auto capture_and_launch = [&] {
    cudaGraph_t captured_graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    bool capture_started = false;
    try {
      if (trace_graph_capture)
        std::fprintf(stderr, "[EXL3-GRAPH-TRACE] begin capture routes=%zu\n",
                     route_count);
      check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
            "begin external routed FFN capture");
      capture_started = true;
      enqueue_sequence();
      if (trace_graph_capture)
        std::fprintf(stderr, "[EXL3-GRAPH-TRACE] enqueued routes=%zu\n",
                     route_count);
      const cudaError_t capture_status =
          cudaStreamEndCapture(stream, &captured_graph);
      capture_started = false;
      check(capture_status, "end external routed FFN capture");
      if (trace_graph_capture)
        std::fprintf(stderr, "[EXL3-GRAPH-TRACE] ended capture\n");
      check(cudaGraphInstantiate(&executable, captured_graph, nullptr, nullptr,
                                 0),
            "instantiate external routed FFN graph");
      if (trace_graph_capture)
        std::fprintf(stderr, "[EXL3-GRAPH-TRACE] instantiated graph\n");
      graph_id = impl_->store_external_graph(
          std::move(graph_key), std::move(graph_site_key), captured_graph,
          executable);
      captured_graph = nullptr;
      executable = nullptr;
      graph_use_registered = true;
      const auto stored = std::find_if(
          impl_->external_routed_graphs.begin(),
          impl_->external_routed_graphs.end(),
          [&](const Impl::CapturedExternalGraph &candidate) {
            return candidate.id == graph_id;
          });
      if (stored == impl_->external_routed_graphs.end())
        throw std::runtime_error("stored EXL3 routed graph disappeared");
      if (trace_graph_capture)
        std::fprintf(stderr, "[EXL3-GRAPH-TRACE] stored graph=%llu\n",
                     static_cast<unsigned long long>(graph_id));
      check(cudaGraphLaunch(stored->executable, stream),
            "launch captured external routed FFN graph");
      if (trace_graph_capture)
        std::fprintf(stderr, "[EXL3-GRAPH-TRACE] launched graph=%llu\n",
                     static_cast<unsigned long long>(graph_id));
    } catch (...) {
      if (capture_started) {
        cudaGraph_t aborted_graph = nullptr;
        if (cudaStreamEndCapture(stream, &aborted_graph) == cudaSuccess &&
            aborted_graph)
          cudaGraphDestroy(aborted_graph);
      }
      if (executable)
        cudaGraphExecDestroy(executable);
      if (captured_graph)
        cudaGraphDestroy(captured_graph);
      throw;
    }
  };
  try {
    begin_event = impl_->acquire_event();
    complete_event = impl_->acquire_event();
    check(cudaEventRecord(begin_event, stream),
          "record external routed FFN start event");
    if (impl_->capture_external_routed_graphs) {
      if (auto *captured = impl_->find_external_graph(graph_key)) {
        graph_id = captured->id;
        graph_use_registered = true;
        if (trace_graph_capture)
          std::fprintf(stderr, "[EXL3-GRAPH-TRACE] replay graph=%llu\n",
                       static_cast<unsigned long long>(graph_id));
        check(cudaGraphLaunch(captured->executable, stream),
              "replay external routed FFN graph");
      } else if (impl_->has_external_graph_for_site(graph_site_key)) {
        ++impl_->counters.routed_graph_eager_fallbacks;
        enqueue_sequence();
      } else if (impl_->make_external_graph_room()) {
        capture_and_launch();
      } else {
        ++impl_->counters.routed_graph_eager_fallbacks;
        enqueue_sequence();
      }
    } else {
      enqueue_sequence();
    }
    check(cudaEventRecord(complete_event, stream),
          "record external routed FFN completion event");
    Impl::PendingRoutedUse pending;
    pending.begin = begin_event;
    pending.complete = complete_event;
    pending.external_holds = std::move(external_holds);
    pending.graph_id = graph_id;
    impl_->pending_routed.push_back(std::move(pending));
    impl_->scratch_tail = complete_event;
    impl_->scratch_stream = stream;
    begin_event = nullptr;
    complete_event = nullptr;
    graph_use_registered = false;
  } catch (...) {
    cudaStreamSynchronize(stream);
    if (graph_use_registered)
      impl_->release_external_graph_use(graph_id);
    impl_->recycle_event(begin_event);
    impl_->recycle_event(complete_event);
    throw;
  }
  impl_->counters.matmuls += 3 * route_count;
  ++impl_->counters.routed_ffns;
}

void Exl3GpuExecutor::prepare_routed_cache(
    const std::vector<std::string> &expert_prefixes_by_id,
    const int32_t *route_ids, int count, int hidden, int intermediate,
    const std::function<std::vector<uint8_t>(const std::string &)> &load) {
  if (expert_prefixes_by_id.empty() || !route_ids || count <= 0 ||
      hidden <= 0 || intermediate <= 0 || hidden % 128 ||
      intermediate % 128 || !load)
    throw std::runtime_error("Invalid EXL3 routed cache preparation");

  std::lock_guard<std::mutex> lock(impl_->mutex);
  DeviceGuard guard(impl_->device);
  impl_->reap_completed();
  std::vector<Entry *> held_entries;
  held_entries.reserve(static_cast<size_t>(count) * 3);
  struct HeldEntries {
    std::vector<Entry *> &entries;
    ~HeldEntries() {
      for (Entry *entry : entries)
        --entry->holds;
    }
  } held{held_entries};
  for (int route = 0; route < count; ++route) {
    const int expert_id = route_ids[route];
    if (expert_id < 0 ||
        static_cast<size_t>(expert_id) >= expert_prefixes_by_id.size())
      throw std::runtime_error("EXL3 cache preparation expert ID out of range");
    const std::string &prefix =
        expert_prefixes_by_id[static_cast<size_t>(expert_id)];
    for (const char *suffix : {"gate_proj", "up_proj", "down_proj"}) {
      const std::string key = prefix + suffix;
      const bool down = std::strcmp(suffix, "down_proj") == 0;
      const int input_width = down ? intermediate : hidden;
      const int output_width = down ? hidden : intermediate;
      auto found = impl_->cache.find(key);
      Entry *entry = nullptr;
      if (found == impl_->cache.end()) {
        auto packed = load(key);
        entry = &impl_->upload(key, kExl3Type, packed.data(), packed.size(),
                               input_width, output_width, false);
      } else {
        entry = found->second.get();
        if (entry->type != kExl3Type ||
            entry->input_features != input_width ||
            entry->output_features != output_width)
          throw std::runtime_error("EXL3 routed FFN cache shape mismatch: " +
                                   key);
        impl_->lru.splice(impl_->lru.begin(), impl_->lru, entry->lru);
        ++impl_->counters.cache_hits;
      }
      ++entry->holds;
      held_entries.push_back(entry);
    }
  }
}

Exl3GpuStats Exl3GpuExecutor::stats() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  DeviceGuard guard(impl_->device);
  impl_->reap_completed();
  return impl_->counters;
}

bool exl3_cuda_matmul_packed(float *output, const float *input,
                            const void *packed, int rows, int input_features,
                            int output_features) {
  if (!packed || !input || !output || rows <= 0 || rows > 65535) return false;
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) return false;
  struct Scratch {
    int device;
    Buffer partials;
    explicit Scratch(int device_id) : device(device_id) {}
    ~Scratch() {
      int previous = 0;
      cudaGetDevice(&previous);
      cudaSetDevice(device);
      partials.release();
      cudaSetDevice(previous);
    }
  };
  static thread_local std::unordered_map<int, std::unique_ptr<Scratch>> scratch;
  Exl3PackedMatrixHeader header{};
  if (cudaMemcpy(&header, packed, sizeof(header), cudaMemcpyDeviceToHost) != cudaSuccess ||
      !exl3_valid_packed_header(header, header.total_size) ||
      header.input_features != static_cast<uint32_t>(input_features) ||
      header.output_features != static_cast<uint32_t>(output_features) ||
      header.bits < 1 || header.bits > 8) return false;
  try {
    auto &buffers = scratch[device];
    if (!buffers) buffers = std::make_unique<Scratch>(device);
    buffers->partials.grow(static_cast<size_t>(rows) * output_features *
                            exl3_cuda_key_splits(input_features) * sizeof(float));
    return launch_exl3_mul1_matmul(device_view(packed, header), input, output, rows,
                                     buffers->partials.floats()) == cudaSuccess;
  } catch (const std::exception &) {
    return false;
  }
}
