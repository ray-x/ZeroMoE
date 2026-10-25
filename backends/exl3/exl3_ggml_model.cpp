#include "exl3_ggml_model.hpp"

#if defined(ZEROMOE_USE_GGML_CUDA)

#include <cuda_runtime.h>
#include "ggml-cuda.h"
#include "ggml-cpu.h"
#include "exl3_gpu.hpp"
#include "ggml_moe_backend.hpp"
#include "gpu_topology.hpp"
#include "exl3_format.hpp"
#include "engine.hpp"

#include <memory>
#include <algorithm>
#include <array>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <list>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace {

struct Exl3ModelIndex {
  SafetensorsModelInfo model;
  Exl3Contract contract;
  std::unordered_map<std::string, const Exl3LinearInfo *> mul1_linears;
};

struct Exl3TopologyExpertHold {
  struct Slot {
    int expert_id = -1;
    const uint8_t *device_ptr = nullptr;
  };

  GpuTopology *topology = nullptr;
  int layer = -1;
  std::array<Slot, 3> slots{};
  size_t count = 0;

  void add(int expert_id, const uint8_t *device_ptr) {
    slots[count++] = {expert_id, device_ptr};
  }

  ~Exl3TopologyExpertHold() {
    if (!topology)
      return;
    for (size_t index = 0; index < count; ++index)
      topology->release_hold(layer, slots[index].expert_id,
                             slots[index].device_ptr, false);
  }
};

struct Exl3TopologyCacheState {
  GpuTopology *topology = nullptr;
  std::shared_ptr<class Exl3DdrArenaCache> ddr_cache;
  std::string model_path;
  int layer = -1;
  int experts = 0;
  int device = -1;
  std::vector<std::array<std::string, 3>> matrix_keys;
  std::vector<std::array<std::string, 3>> header_keys;
};

class Exl3DdrArenaCache {
public:
  struct View {
    const uint8_t *data = nullptr;
    size_t bytes = 0;
    std::shared_ptr<void> lock;
  };

  Exl3DdrArenaCache(uint8_t *base, size_t bytes, size_t slot_count)
      : base_(base), bytes_(bytes), slot_count_(slot_count),
        slot_stride_(slot_count ? bytes / slot_count : 0),
        slot_keys_(slot_count) {}

  View get(const std::string &key,
           const std::function<std::vector<uint8_t>()> &load) {
    auto held_lock = std::make_shared<std::unique_lock<std::mutex>>(mutex_);
    auto found = entries_.find(key);
    if (found != entries_.end()) {
      lru_.splice(lru_.begin(), lru_, found->second.lru);
      const size_t offset = found->second.slot * slot_stride_;
      return {base_ + offset, found->second.bytes, std::move(held_lock)};
    }

    if (!base_ || bytes_ == 0 || slot_count_ == 0 || slot_stride_ == 0 ||
        !load)
      throw std::runtime_error("EXL3 DDR expert arena is unavailable");
    std::vector<uint8_t> packed = load();
    if (packed.empty() || packed.size() > slot_stride_)
      throw std::runtime_error("EXL3 matrix does not fit an existing DDR expert slot: " +
                               key);

    size_t slot = slot_count_;
    for (size_t index = 0; index < slot_count_; ++index) {
      if (slot_keys_[index].empty()) {
        slot = index;
        break;
      }
    }
    if (slot == slot_count_) {
      const std::string victim_key = lru_.back();
      auto victim = entries_.find(victim_key);
      slot = victim->second.slot;
      entries_.erase(victim);
      lru_.pop_back();
    }

    const size_t offset = slot * slot_stride_;
    std::memcpy(base_ + offset, packed.data(), packed.size());
    slot_keys_[slot] = key;
    lru_.push_front(key);
    entries_.emplace(key, Entry{slot, packed.size(), lru_.begin()});
    return {base_ + offset, packed.size(), std::move(held_lock)};
  }

private:
  struct Entry {
    size_t slot = 0;
    size_t bytes = 0;
    std::list<std::string>::iterator lru;
  };

  uint8_t *base_ = nullptr;
  size_t bytes_ = 0;
  size_t slot_count_ = 0;
  size_t slot_stride_ = 0;
  std::vector<std::string> slot_keys_;
  std::unordered_map<std::string, Entry> entries_;
  std::list<std::string> lru_;
  std::mutex mutex_;
};

std::mutex exl3_topology_headers_mutex;
std::unordered_map<std::string, Exl3PackedMatrixHeader> exl3_topology_headers;
std::mutex exl3_ddr_caches_mutex;
std::unordered_map<std::string, std::weak_ptr<Exl3DdrArenaCache>>
    exl3_ddr_caches;

std::shared_ptr<Exl3DdrArenaCache> get_exl3_ddr_arena_cache(
    const std::string &model_path, int layer, uint8_t *base, size_t bytes,
    size_t slot_count) {
  if (!base || bytes == 0 || slot_count == 0 || bytes / slot_count == 0)
    return {};
  const std::string key = model_path + "\n" + std::to_string(layer) + "\n" +
                          std::to_string(reinterpret_cast<uintptr_t>(base));
  std::lock_guard<std::mutex> lock(exl3_ddr_caches_mutex);
  auto found = exl3_ddr_caches.find(key);
  if (found != exl3_ddr_caches.end()) {
    if (auto cache = found->second.lock())
      return cache;
  }
  auto cache = std::make_shared<Exl3DdrArenaCache>(base, bytes, slot_count);
  exl3_ddr_caches[key] = cache;
  return cache;
}

std::shared_ptr<const Exl3ModelIndex> model_index(const std::string &path) {
  static std::mutex mutex;
  static std::unordered_map<std::string, std::shared_ptr<const Exl3ModelIndex>>
      indexes;
  std::lock_guard<std::mutex> lock(mutex);
  auto found = indexes.find(path);
  if (found != indexes.end())
    return found->second;
  auto index = std::make_shared<Exl3ModelIndex>();
  if (!SafetensorsAdapter::parse_directory(path, index->model))
    return {};
  index->contract = SafetensorsAdapter::map_exl3_contract(index->model);
  if (!index->contract.complete)
    return {};
  index->mul1_linears.reserve(index->contract.linears.size());
  for (const auto &linear : index->contract.linears)
    if (linear.valid && linear.codebook == Exl3Codebook::MUL1)
      index->mul1_linears.emplace(linear.key, &linear);
  indexes.emplace(path, index);
  return index;
}

} // namespace

Exl3GgmlBridgeLayer::~Exl3GgmlBridgeLayer() {
  ffn_state.reset();
}

bool Exl3GgmlBridgeLayer::initialize(
    const std::string &model_path, ggml_context *ctx,
    ggml_backend_t cuda_backend, Exl3GpuExecutor *executor, int model_layer,
    int n_experts, int model_topk, int model_rows, int cuda_device,
    bool load_cache_misses_from_safetensors,
    std::shared_ptr<Exl3CpuMatrixCache> cpu_matrix_cache,
    GpuTopology *gpu_topology, StreamingEngine *streaming_engine) {
  const auto decline = [&](const std::string &reason) {
    LOG_WARN("[EXL3-GGML-BRIDGE] layer=%d initialization declined: %s\n",
             model_layer, reason.c_str());
    return false;
  };
  if (model_path.empty() || !ctx || !cuda_backend || !executor || model_layer < 0 ||
      n_experts <= 0 || model_topk <= 0 || model_topk > n_experts ||
      model_rows <= 0 || cuda_device < 0)
    return decline("invalid model, graph, executor, routing shape, or device");
  const auto index = model_index(model_path);
  if (!index)
    return decline("Safetensors model index or EXL3 contract unavailable");
  const Exl3Contract &contract = index->contract;
  if (contract.n_experts != n_experts)
    return decline("EXL3 expert count differs from GGUF architecture metadata");

  const auto find_mul1 = [&](const std::string &key) -> const Exl3LinearInfo * {
    const auto found = index->mul1_linears.find(key);
    return found == index->mul1_linears.end() ? nullptr : found->second;
  };
  const std::string layer_prefix =
      "model.language_model.layers." + std::to_string(model_layer) +
      ".mlp.experts.";
  const std::string first_prefix = layer_prefix + "0.";
  const Exl3LinearInfo *first_gate = find_mul1(first_prefix + "gate_proj");
  const Exl3LinearInfo *first_up = find_mul1(first_prefix + "up_proj");
  const Exl3LinearInfo *first_down = find_mul1(first_prefix + "down_proj");
  if (!first_gate || !first_up || !first_down ||
      first_gate->in_features <= 0 || first_gate->out_features <= 0 ||
      first_gate->in_features > INT_MAX || first_gate->out_features > INT_MAX)
    return decline("layer has no supported MUL1 gate/up/down projections");
  const int hidden = static_cast<int>(first_gate->in_features);
  const int intermediate = static_cast<int>(first_gate->out_features);
  if (hidden % 128 || intermediate % 128 ||
      first_up->in_features != hidden || first_up->out_features != intermediate ||
      first_down->in_features != intermediate || first_down->out_features != hidden)
    return decline("layer-0 expert projection dimensions do not match");

  std::vector<std::string> expert_prefixes(static_cast<size_t>(n_experts));
  const char *projections[] = {"gate_proj", "up_proj", "down_proj"};
  for (int expert = 0; expert < n_experts; ++expert) {
    expert_prefixes[static_cast<size_t>(expert)] =
        layer_prefix + std::to_string(expert) + ".";
    const int expected_inputs[] = {hidden, hidden, intermediate};
    const int expected_outputs[] = {intermediate, intermediate, hidden};
    for (int projection = 0; projection < 3; ++projection) {
      const Exl3LinearInfo *linear = find_mul1(
          expert_prefixes[static_cast<size_t>(expert)] +
          projections[projection]);
      if (!linear || linear->in_features != expected_inputs[projection] ||
          linear->out_features != expected_outputs[projection]) {
        return decline("expert projection missing or dimension-mismatched: " +
                       expert_prefixes[static_cast<size_t>(expert)] +
                       projections[projection]);
      }
    }
  }

  ffn_state = std::make_unique<Exl3GgmlCachedRoutedFfnState>();
  if (!ffn_state->initialize(executor, std::move(expert_prefixes), model_rows,
                             model_topk, hidden, intermediate))
    return decline("cached routed-FFN node state initialization failed");
  ffn_state->model_layer = model_layer;
  const char *verify_layer_text =
      std::getenv("ZEROMOE_EXL3_GGML_VERIFY_CPU_LAYER");
  if (verify_layer_text && verify_layer_text[0]) {
    if (std::strcmp(verify_layer_text, "all") == 0) {
      ffn_state->verify_cpu_reference = true;
    } else {
      char *end = nullptr;
      const long verify_layer = std::strtol(verify_layer_text, &end, 10);
      ffn_state->verify_cpu_reference =
          end != verify_layer_text && *end == '\0' &&
          verify_layer == model_layer;
    }
  }
  if (load_cache_misses_from_safetensors) {
    ffn_state->set_matrix_loader(
        [index, cpu_matrix_cache,
         use_external_ddr_slots = gpu_topology != nullptr](
            const std::string &key) -> std::vector<uint8_t> {
          const auto found = index->mul1_linears.find(key);
          if (found == index->mul1_linears.end())
            throw std::runtime_error("Missing EXL3 safetensors matrix: " + key);
          if (cpu_matrix_cache && !use_external_ddr_slots) {
            auto matrix = cpu_matrix_cache->get(key);
            if (!matrix) {
              Exl3Mul1MatrixData loaded;
              std::string error;
              if (!load_exl3_mul1_components(index->model, *found->second, loaded,
                                             error))
                throw std::runtime_error(error);
              matrix = cpu_matrix_cache->insert(key, std::move(loaded));
            }
            return exl3_mul1_pack_matrix(matrix->view());
          }
          std::vector<uint8_t> packed;
          std::string error;
          if (!load_exl3_mul1_matrix(index->model, *found->second, packed,
                                     error))
            throw std::runtime_error(error);
          return packed;
      });
  }
  if (gpu_topology) {
    if (!streaming_engine ||
        !streaming_engine->graph_ram_layer_ready(model_layer))
      return decline("EXL3 DDR cache requires the existing complete graph RAM "
                     "arena for this layer");
    const auto *host_arena = streaming_engine->graph_ram_arena(model_layer);
    if (!host_arena || host_arena->n_experts != n_experts)
      return decline("EXL3 DDR cache cannot reference this layer's graph RAM arena");
    auto ddr_cache = get_exl3_ddr_arena_cache(
        model_path, model_layer, host_arena->base, host_arena->bytes,
        static_cast<size_t>(n_experts));
    if (!ddr_cache)
      return decline("EXL3 DDR cache arena has no usable existing expert slots");
    const size_t device_slot_count = static_cast<size_t>(std::count_if(
        gpu_topology->expert_vram_cache.begin(),
        gpu_topology->expert_vram_cache.end(), [&](const ExpertLocation &slot) {
          return slot.gpu_id == cuda_device && slot.device_ptr &&
                 !slot.pinned && slot.temp_pin_count == 0 && slot.upload_ready &&
                 (slot.pool_layer == model_layer || slot.pool_layer < 0);
        }));
    if (device_slot_count == 0)
      return decline("no preallocated GPU expert-cache slot on the graph device "
                     "for this layer");
    if (device_slot_count < static_cast<size_t>(model_topk) * 3)
      return decline("GPU expert-cache layer region cannot hold one routed "
                     "EXL3 row");
    ffn_state->external_matrix_slot_capacity = device_slot_count;
    const size_t route_batch_rows = std::min(
        static_cast<size_t>(model_rows),
        std::max<size_t>(1, device_slot_count /
                                (static_cast<size_t>(model_topk) * 3)));
    ffn_state->external_experts.resize(
        route_batch_rows * static_cast<size_t>(model_topk));
    LOG_INFO("[EXL3-GGML-BRIDGE] layer=%d reusing DDR arena=%p bytes=%zu "
             "slot_bytes=%zu slots=%zu row_batch=%zu and GPU=%d slots=%zu\n",
             model_layer, static_cast<void *>(host_arena->base),
             host_arena->bytes, host_arena->bytes / n_experts,
             static_cast<size_t>(n_experts), route_batch_rows, cuda_device,
             device_slot_count);
    auto cache_state = std::make_shared<Exl3TopologyCacheState>();
    cache_state->topology = gpu_topology;
    cache_state->ddr_cache = std::move(ddr_cache);
    cache_state->model_path = model_path;
    cache_state->layer = model_layer;
    cache_state->experts = n_experts;
    cache_state->device = cuda_device;
    cache_state->matrix_keys.resize(static_cast<size_t>(n_experts));
    cache_state->header_keys.resize(static_cast<size_t>(n_experts));
    const char *matrix_suffixes[] = {"gate_proj", "up_proj", "down_proj"};
    for (int expert = 0; expert < n_experts; ++expert) {
      const std::string expert_prefix =
          layer_prefix + std::to_string(expert) + ".";
      for (int projection = 0; projection < 3; ++projection) {
        auto &key = cache_state->matrix_keys[static_cast<size_t>(expert)]
                                             [static_cast<size_t>(projection)];
        key = expert_prefix + matrix_suffixes[projection];
        cache_state->header_keys[static_cast<size_t>(expert)]
                                [static_cast<size_t>(projection)] =
            model_path + "\n" + key;
      }
    }
    ffn_state->set_external_expert_resolver(
        [cache_state](
            const std::string &, int expert_id, int hidden_width,
            int intermediate_width,
            const std::function<std::vector<uint8_t>(const std::string &)> &load)
            -> Exl3GpuExternalExpert {
          if (expert_id < 0 || expert_id >= cache_state->experts)
            throw std::runtime_error("EXL3 topology cache expert ID out of range");
          const int input_widths[] = {hidden_width, hidden_width,
                                      intermediate_width};
          const int output_widths[] = {intermediate_width, intermediate_width,
                                       hidden_width};
          Exl3GpuExternalExpert result;
          auto hold = std::make_shared<Exl3TopologyExpertHold>();
          hold->topology = cache_state->topology;
          hold->layer = cache_state->layer;
          for (int projection = 0; projection < 3; ++projection) {
            const std::string &key =
                cache_state->matrix_keys[static_cast<size_t>(expert_id)]
                                        [static_cast<size_t>(projection)];
            const int64_t encoded_slot =
                1ll + static_cast<int64_t>(cache_state->device) *
                          3 * cache_state->experts +
                static_cast<int64_t>(projection) * cache_state->experts +
                expert_id;
            if (encoded_slot > INT_MAX)
              throw std::runtime_error("EXL3 topology cache slot ID overflow");
            const int cache_expert_id = -static_cast<int>(encoded_slot);
            Exl3PackedMatrixHeader header{};
            ExpertLocation *slot = cache_state->topology->find_expert(
                cache_state->layer, cache_expert_id);
            const std::string &header_key =
                cache_state->header_keys[static_cast<size_t>(expert_id)]
                                        [static_cast<size_t>(projection)];
            Exl3DdrArenaCache::View host_matrix;
            if (slot) {
              std::lock_guard<std::mutex> lock(exl3_topology_headers_mutex);
              const auto found = exl3_topology_headers.find(header_key);
              if (found == exl3_topology_headers.end())
                throw std::runtime_error(
                    "EXL3 topology cache metadata missing for " + key);
              header = found->second;
            } else {
              host_matrix = cache_state->ddr_cache->get(
                  key, [&] { return load(key); });
              if (!host_matrix.data ||
                  host_matrix.bytes < sizeof(header))
                throw std::runtime_error("EXL3 packed matrix is truncated: " +
                                         key);
              std::memcpy(&header, host_matrix.data, sizeof(header));
              if (!exl3_valid_packed_header(header, host_matrix.bytes) ||
                  header.input_features != input_widths[projection] ||
                  header.output_features != output_widths[projection])
                throw std::runtime_error("Invalid EXL3 matrix for topology slot: " +
                                         key);
              {
                std::lock_guard<std::mutex> lock(exl3_topology_headers_mutex);
                exl3_topology_headers.emplace(header_key, header);
              }
              slot = cache_state->topology->upload_expert(
                  cache_state->layer, cache_expert_id, host_matrix.data,
                  host_matrix.bytes, cache_state->device);
            }
            if (!slot || !slot->device_ptr || slot->gpu_id != cache_state->device ||
                slot->size < header.total_size)
              throw std::runtime_error("EXL3 packed matrix does not fit an existing "
                                       "GPU expert-cache slot: " + key);
            if (!cache_state->topology->hold_for_compute(
                    cache_state->layer, cache_expert_id, slot->device_ptr,
                    false))
              throw std::runtime_error("Could not hold EXL3 GPU expert-cache slot: " +
                                       key);
            hold->add(cache_expert_id, slot->device_ptr);
            const auto *bytes = slot->device_ptr;
            Exl3CudaMatrixView view{
                reinterpret_cast<const uint16_t *>(bytes + header.trellis_offset),
                reinterpret_cast<const uint16_t *>(bytes + header.suh_offset),
                reinterpret_cast<const uint16_t *>(bytes + header.svh_offset),
                (header.flags & kExl3PackedHasBias)
                    ? reinterpret_cast<const uint16_t *>(bytes + header.bias_offset)
                    : nullptr,
                static_cast<int>(header.input_features),
                static_cast<int>(header.output_features),
                static_cast<int>(header.bits)};
            if (projection == 0)
              result.gate = view;
            else if (projection == 1)
              result.up = view;
            else
              result.down = view;
          }
          result.hold = std::move(hold);
          return result;
        });
  }
  gpu_executor = executor;
  layer = model_layer;
  device = cuda_device;
  rows = model_rows;
  experts = n_experts;
  topk = model_topk;
  return true;
}

bool Exl3GgmlBridgeLayer::set_layer(int model_layer) {
  if (model_layer < 0 || !ffn_state ||
      static_cast<int>(ffn_state->expert_prefixes_by_id.size()) != experts)
    return false;
  for (int expert = 0; expert < experts; ++expert) {
    ffn_state->expert_prefixes_by_id[static_cast<size_t>(expert)] =
        "model.language_model.layers." + std::to_string(model_layer) +
        ".mlp.experts." + std::to_string(expert) + ".";
  }
  layer = model_layer;
  return true;
}

ggml_tensor *Exl3GgmlBridgeLayer::build(ggml_context *ctx, ggml_tensor *input,
                                         ggml_tensor *route_ids,
                                         ggml_tensor *route_weights,
                                         ggml_backend_sched_t scheduler,
                                         ggml_backend_t cuda_backend) {
  if (!ctx || !input || !route_ids || !route_weights || !ffn_state ||
      !scheduler || !cuda_backend ||
      input->type != GGML_TYPE_F32 || input->ne[0] != ffn_state->hidden ||
      input->ne[1] != rows || route_ids->type != GGML_TYPE_I32 ||
      route_ids->ne[0] != topk || route_ids->ne[1] != rows ||
      route_weights->type != GGML_TYPE_F32 ||
      ggml_nelements(route_weights) != static_cast<int64_t>(topk) * rows) {
    LOG_WARN("[EXL3-GGML-BRIDGE] layer=%d graph node shape mismatch: "
             "input=%p type=%d shape=%lldx%lld ids=%p type=%d shape=%lldx%lld "
             "weights=%p type=%d shape=%lldx%lld expected=%dx%d topk=%d rows=%d\n",
             layer, static_cast<void *>(input), input ? input->type : -1,
             input ? static_cast<long long>(input->ne[0]) : 0,
             input ? static_cast<long long>(input->ne[1]) : 0,
             static_cast<void *>(route_ids), route_ids ? route_ids->type : -1,
             route_ids ? static_cast<long long>(route_ids->ne[0]) : 0,
             route_ids ? static_cast<long long>(route_ids->ne[1]) : 0,
             static_cast<void *>(route_weights),
             route_weights ? route_weights->type : -1,
             route_weights ? static_cast<long long>(route_weights->ne[0]) : 0,
             route_weights ? static_cast<long long>(route_weights->ne[1]) : 0,
             ffn_state ? ffn_state->hidden : 0,
             ffn_state ? ffn_state->intermediate : 0, topk, rows);
    return nullptr;
  }
  ggml_tensor *ids = ggml_cont(ctx, route_ids);
  ggml_tensor *weights_contiguous = ggml_cont(ctx, route_weights);
  ggml_tensor *weights = weights_contiguous
                             ? ggml_reshape_2d(ctx, weights_contiguous, topk,
                                               rows)
                             : nullptr;
  if (!ids || !weights) {
    LOG_WARN("[EXL3-GGML-BRIDGE] layer=%d route tensor normalization failed "
             "ids=%p weights=%p\n",
             layer, static_cast<void *>(ids), static_cast<void *>(weights));
    return nullptr;
  }
  ggml_backend_sched_set_tensor_backend(scheduler, ids, cuda_backend);
  ggml_backend_sched_set_tensor_backend(scheduler, weights, cuda_backend);
  ggml_tensor *output = exl3_ggml_map_cached_routed_ffn_device_routes(
      ctx, input, ids, weights, ffn_state.get());
  if (output)
    ggml_backend_sched_set_tensor_backend(scheduler, output, cuda_backend);
  else
    LOG_WARN("[EXL3-GGML-BRIDGE] layer=%d cached FFN custom node creation failed\n",
             layer);
  return output;
}

Exl3GgmlRoutedFfnGraph::~Exl3GgmlRoutedFfnGraph() { reset(); }

void Exl3GgmlRoutedFfnGraph::reset() {
  if (scheduler) {
    ggml_backend_sched_synchronize(scheduler);
    ggml_backend_sched_free(scheduler);
    scheduler = nullptr;
  }
  if (cpu_backend) {
    ggml_backend_free(cpu_backend);
    cpu_backend = nullptr;
  }
  bridge.reset();
  if (context) {
    ggml_free(context);
    context = nullptr;
  }
  if (cuda_adapter) {
    ggml_backend_free(cuda_adapter);
    cuda_adapter = nullptr;
  }
  graph = nullptr;
  input = nullptr;
  route_ids = nullptr;
  route_weights = nullptr;
  output = nullptr;
  cuda_raw = nullptr;
  experts = topk = rows = hidden = intermediate = 0;
}

bool Exl3GgmlRoutedFfnGraph::initialize(
    const std::string &model_path, Exl3GpuExecutor *executor, int n_experts,
    int model_topk, int model_rows, int model_hidden,
    int model_intermediate, int device) {
  reset();
  if (model_path.empty() || !executor || n_experts <= 0 || model_topk <= 0 ||
      model_topk > n_experts || model_rows <= 0 || model_hidden <= 0 ||
      model_intermediate <= 0 || device < 0)
    return false;
  cuda_raw = static_cast<ggml_backend_t>(ggml_shared_backend(device));
  if (!cuda_raw)
    return false;
  cuda_adapter = exl3_ggml_cuda_backend_wrap(cuda_raw, false);
  if (!cuda_adapter)
    return false;
  cpu_backend = ggml_backend_cpu_init();
  if (!cpu_backend)
    return false;
  ggml_backend_t backends[] = {cuda_adapter, cpu_backend};
  scheduler = ggml_backend_sched_new(backends, nullptr, 2, 256, false, false);
  if (!scheduler)
    return false;

  ggml_init_params params{};
  params.mem_size = 2 * 1024 * 1024;
  params.no_alloc = true;
  context = ggml_init(params);
  if (!context)
    return false;
  input = ggml_new_tensor_2d(context, GGML_TYPE_F32, model_hidden,
                             model_rows);
  route_ids = ggml_new_tensor_2d(context, GGML_TYPE_I32, model_topk,
                                 model_rows);
  route_weights = ggml_new_tensor_2d(context, GGML_TYPE_F32, model_topk,
                                     model_rows);
  graph = ggml_new_graph(context);
  bridge = std::make_unique<Exl3GgmlBridgeLayer>();
  if (!input || !route_ids || !route_weights || !graph ||
      !bridge->initialize(model_path, context, cuda_adapter, executor, 0,
                          n_experts, model_topk, model_rows, device) ||
      bridge->ffn_state->hidden != model_hidden ||
      bridge->ffn_state->intermediate != model_intermediate)
    return false;
  ggml_backend_sched_set_tensor_backend(scheduler, input, cuda_adapter);
  ggml_backend_sched_set_tensor_backend(scheduler, route_ids, cuda_adapter);
  ggml_backend_sched_set_tensor_backend(scheduler, route_weights,
                                        cuda_adapter);
  output = exl3_ggml_map_cached_routed_ffn_host_routes(
      context, input, route_ids, route_weights, bridge->ffn_state.get());
  if (!output)
    return false;
  ggml_backend_sched_set_tensor_backend(scheduler, output, cuda_adapter);
  ggml_build_forward_expand(graph, output);
  if (!ggml_backend_sched_alloc_graph(scheduler, graph))
    return false;
  experts = n_experts;
  topk = model_topk;
  rows = model_rows;
  hidden = model_hidden;
  intermediate = model_intermediate;
  return true;
}

bool Exl3GgmlRoutedFfnGraph::run(
    int model_layer, const int32_t *host_route_ids,
    const float *host_route_weights, const float *host_input,
    float *host_output,
    const std::function<std::vector<uint8_t>(const std::string &)> &load) {
  if (!scheduler || !bridge || !bridge->ffn_state || !input || !route_ids ||
      !route_weights || !output || !host_route_ids || !host_route_weights ||
      !host_input || !host_output || !load || !bridge->set_layer(model_layer))
    return false;
  const size_t route_count = static_cast<size_t>(rows) * topk;
  if (!bridge->ffn_state->set_host_routes(host_route_ids,
                                          host_route_weights))
    return false;
  bridge->ffn_state->set_matrix_loader(load);

  const size_t input_bytes =
      static_cast<size_t>(rows) * hidden * sizeof(float);
  const size_t ids_bytes = route_count * sizeof(int32_t);
  const size_t weights_bytes = route_count * sizeof(float);
  ggml_backend_tensor_set(input, host_input, 0, input_bytes);
  ggml_backend_tensor_set(route_ids, host_route_ids, 0, ids_bytes);
  ggml_backend_tensor_set(route_weights, host_route_weights, 0, weights_bytes);
  if (ggml_backend_sched_graph_compute_async(scheduler, graph) !=
      GGML_STATUS_SUCCESS) {
    ggml_backend_sched_synchronize(scheduler);
    return false;
  }
  ggml_backend_sched_synchronize(scheduler);
  ggml_backend_tensor_get(output, host_output, 0, input_bytes);
  return true;
}

Exl3GgmlDenseMatmulGraph::~Exl3GgmlDenseMatmulGraph() { reset(); }

void Exl3GgmlDenseMatmulGraph::reset() {
  if (scheduler) {
    ggml_backend_sched_synchronize(scheduler);
    ggml_backend_sched_free(scheduler);
    scheduler = nullptr;
  }
  if (cpu_backend) {
    ggml_backend_free(cpu_backend);
    cpu_backend = nullptr;
  }
  if (context) {
    ggml_free(context);
    context = nullptr;
  }
  if (cuda_adapter) {
    ggml_backend_free(cuda_adapter);
    cuda_adapter = nullptr;
  }
  cuda_raw = nullptr;
  graph = nullptr;
  input = nullptr;
  output = nullptr;
  executor = nullptr;
  rows = input_features = output_features = 0;
  state = {};
}

bool Exl3GgmlDenseMatmulGraph::initialize(
    Exl3GpuExecutor *matrix_executor, int matrix_rows,
    int matrix_input_features, int matrix_output_features) {
  reset();
  if (!matrix_executor || matrix_rows <= 0 || matrix_input_features <= 0 ||
      matrix_output_features <= 0)
    return false;
  executor = matrix_executor;
  rows = matrix_rows;
  input_features = matrix_input_features;
  output_features = matrix_output_features;
  cuda_raw = static_cast<ggml_backend_t>(
      ggml_shared_backend(matrix_executor->device()));
  if (!cuda_raw)
    return false;
  cuda_adapter = exl3_ggml_cuda_backend_wrap(cuda_raw, false);
  if (!cuda_adapter)
    return false;
  cpu_backend = ggml_backend_cpu_init();
  if (!cpu_backend)
    return false;
  ggml_backend_t backends[] = {cuda_adapter, cpu_backend};
  scheduler = ggml_backend_sched_new(backends, nullptr, 2, 128, false, false);
  if (!scheduler)
    return false;
  ggml_init_params params{};
  params.mem_size = 1024 * 1024;
  params.no_alloc = true;
  context = ggml_init(params);
  if (!context)
    return false;
  input = ggml_new_tensor_2d(context, GGML_TYPE_F32, input_features, rows);
  graph = ggml_new_graph(context);
  if (!input || !graph ||
      !state.initialize(matrix_executor, rows, input_features,
                        output_features))
    return false;
  ggml_backend_sched_set_tensor_backend(scheduler, input, cuda_adapter);
  output = exl3_ggml_map_cached_matmul(context, input, &state);
  if (!output)
    return false;
  ggml_backend_sched_set_tensor_backend(scheduler, output, cuda_adapter);
  ggml_build_forward_expand(graph, output);
  return ggml_backend_sched_alloc_graph(scheduler, graph);
}

bool Exl3GgmlDenseMatmulGraph::run(const std::string &key, uint32_t type,
                                  const uint8_t *weights, size_t weight_bytes,
                                  const float *host_input,
                                  float *host_output) {
  if (!executor || !scheduler || !input || !output || key.empty() ||
      !weights || weight_bytes == 0 || !host_input || !host_output)
    return false;
  state.key = key;
  try {
    executor->prepare_matrix_cache(key, type, weights, weight_bytes,
                                   input_features, output_features, true);
  } catch (const std::runtime_error &) {
    return false;
  }
  const size_t input_bytes =
      static_cast<size_t>(rows) * input_features * sizeof(float);
  const size_t output_bytes =
      static_cast<size_t>(rows) * output_features * sizeof(float);
  ggml_backend_tensor_set(input, host_input, 0, input_bytes);
  if (ggml_backend_sched_graph_compute_async(scheduler, graph) !=
      GGML_STATUS_SUCCESS) {
    ggml_backend_sched_synchronize(scheduler);
    return false;
  }
  ggml_backend_sched_synchronize(scheduler);
  ggml_backend_tensor_get(output, host_output, 0, output_bytes);
  return true;
}

Exl3GgmlCpuRoutedFfnGraph::~Exl3GgmlCpuRoutedFfnGraph() { reset(); }

void Exl3GgmlCpuRoutedFfnGraph::reset() {
  if (scheduler) {
    ggml_backend_sched_synchronize(scheduler);
    ggml_backend_sched_free(scheduler);
    scheduler = nullptr;
  }
  if (cpu_backend) {
    ggml_backend_free(cpu_backend);
    cpu_backend = nullptr;
  }
  if (context) {
    ggml_free(context);
    context = nullptr;
  }
  graph = nullptr;
  input = route_ids = route_weights = output = nullptr;
  dispatch = nullptr;
  dispatch_context = nullptr;
  rows = topk = hidden = 0;
  layer_count = 0;
  layer = -1;
}

bool Exl3GgmlCpuRoutedFfnGraph::initialize(
    RowDispatch row_dispatch, void *row_dispatch_context, int model_layer_count,
    int model_rows, int model_topk, int model_hidden) {
  reset();
  if (!row_dispatch || !row_dispatch_context || model_layer_count <= 0 ||
      model_rows <= 0 || model_topk <= 0 || model_hidden <= 0)
    return false;
  dispatch = row_dispatch;
  dispatch_context = row_dispatch_context;
  layer_count = model_layer_count;
  rows = model_rows;
  topk = model_topk;
  hidden = model_hidden;
  cpu_backend = ggml_backend_cpu_init();
  if (!cpu_backend)
    return false;
  ggml_backend_t backends[] = {cpu_backend};
  scheduler = ggml_backend_sched_new(backends, nullptr, 1, 128, false, false);
  if (!scheduler)
    return false;
  ggml_init_params params{};
  params.mem_size = 1024 * 1024;
  params.no_alloc = true;
  context = ggml_init(params);
  if (!context)
    return false;
  input = ggml_new_tensor_2d(context, GGML_TYPE_F32, hidden, rows);
  route_ids = ggml_new_tensor_2d(context, GGML_TYPE_I32, topk, rows);
  route_weights = ggml_new_tensor_2d(context, GGML_TYPE_F32, topk, rows);
  graph = ggml_new_graph(context);
  if (!input || !route_ids || !route_weights || !graph)
    return false;
  ggml_backend_sched_set_tensor_backend(scheduler, input, cpu_backend);
  ggml_backend_sched_set_tensor_backend(scheduler, route_ids, cpu_backend);
  ggml_backend_sched_set_tensor_backend(scheduler, route_weights, cpu_backend);
  output = ggml_map_custom3(context, input, route_ids, route_weights, compute,
                            1, this);
  if (!output)
    return false;
  output->ne[0] = hidden;
  output->ne[1] = rows;
  output->nb[1] = static_cast<size_t>(hidden) * sizeof(float);
  ggml_backend_sched_set_tensor_backend(scheduler, output, cpu_backend);
  ggml_build_forward_expand(graph, output);
  return ggml_backend_sched_alloc_graph(scheduler, graph);
}

void Exl3GgmlCpuRoutedFfnGraph::compute(
    ggml_tensor *node, const ggml_tensor *input_tensor,
    const ggml_tensor *route_ids_tensor,
    const ggml_tensor *route_weights_tensor, int ith, int,
    void *user_data) {
  if (ith != 0)
    return;
  auto *state = static_cast<Exl3GgmlCpuRoutedFfnGraph *>(user_data);
  if (!state || !state->dispatch || !state->dispatch_context || !node || !input_tensor ||
      !route_ids_tensor || !route_weights_tensor || !node->data ||
      !input_tensor->data || !route_ids_tensor->data ||
      !route_weights_tensor->data)
    throw std::runtime_error("invalid EXL3 CPU GGML routed FFN node");
  const auto *ids = static_cast<const int32_t *>(route_ids_tensor->data);
  const auto *weights = static_cast<const float *>(route_weights_tensor->data);
  const auto *source = static_cast<const float *>(input_tensor->data);
  auto *destination = static_cast<float *>(node->data);
  for (int row = 0; row < state->rows; ++row) {
    if (!state->dispatch(
            state->dispatch_context, state->layer,
            ids + static_cast<size_t>(row) * state->topk,
            weights + static_cast<size_t>(row) * state->topk, state->topk,
            source + static_cast<size_t>(row) * state->hidden,
            destination + static_cast<size_t>(row) * state->hidden))
      throw std::runtime_error("EXL3 CPU GGML routed FFN dispatch failed");
  }
}

bool Exl3GgmlCpuRoutedFfnGraph::run(
    int layer, const int32_t *host_route_ids, const float *host_route_weights,
    const float *host_input, float *host_output) {
  if (!dispatch || !dispatch_context || !scheduler || !input || !route_ids || !route_weights ||
      !output || !host_route_ids || !host_route_weights || !host_input ||
      !host_output || layer < 0 || layer >= layer_count)
    return false;
  this->layer = layer;
  const size_t input_bytes = static_cast<size_t>(rows) * hidden * sizeof(float);
  const size_t route_count = static_cast<size_t>(rows) * topk;
  ggml_backend_tensor_set(input, host_input, 0, input_bytes);
  ggml_backend_tensor_set(route_ids, host_route_ids, 0,
                          route_count * sizeof(int32_t));
  ggml_backend_tensor_set(route_weights, host_route_weights, 0,
                          route_count * sizeof(float));
  if (ggml_backend_sched_graph_compute_async(scheduler, graph) !=
      GGML_STATUS_SUCCESS) {
    ggml_backend_sched_synchronize(scheduler);
    return false;
  }
  ggml_backend_sched_synchronize(scheduler);
  ggml_backend_tensor_get(output, host_output, 0, input_bytes);
  return true;
}

#endif
