#pragma once
// EngineRunner: the startup / serve / shutdown sequence that used to be one 4,160-line
// function (run_engine in main.cpp). Each phase is a member function; state that crosses
// phases is a member with the same name the old local had.
#include "active_layer_upload_workers.hpp"
#include "async_prefetcher.hpp"
#include "cuda_manager.hpp"
#include "engine.hpp"
#include "full_graph_model.hpp"
#include "option_b_global_hot.hpp"
#include "engine_config.hpp"
#include "exl3_cpu.hpp"
#include "exl3_model_loader.hpp"
#include "exl3_decode.hpp"
#if defined(MOESTREAM_EXL3_CUDA)
#include "exl3_gpu.hpp"
#endif
#include "exllamav3_backend.hpp"
#include "graph_config.hpp"
#include "thread_profile.hpp"
#include "ggml_gdn_proj_backend.hpp"
#include "ggml_glm5next_graph.hpp"
#include "ggml_glm5next_proj_backend.hpp"
#include "ggml_hc_backend.hpp"
#include "ggml_moe_backend.hpp"
#include "ggml_qwen4exp_graph.hpp"
#include "gguf_loader.hpp"
#include "gguf_tokenizer.hpp"
#include "logger.hpp"
#include "perf_timer.hpp"
#include "ram_manager.hpp"
#include "safetensors_adapter.hpp"
#include "shutdown.hpp"
#include "startup_preflight.hpp"
#include "tensor_trace.hpp"
#if defined(MOESTREAM_USE_GGML)
#include "ggml.h"
#endif
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <omp.h>
#include <regex>
#include <set>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

class EngineRunner {
public:
  EngineRunner(const EngineConfig &config, bool cufile_driver_opened)
      : config(config), cufile_driver_opened(cufile_driver_opened) {}
  EngineRunner(const EngineRunner &) = delete;
  EngineRunner &operator=(const EngineRunner &) = delete;

  std::expected<void, std::string> run();

private:
  std::expected<void, std::string> apply_engine_config();
  std::expected<void, std::string> init_host_and_gpu();
  std::expected<void, std::string> load_model();
  std::expected<void, std::string> plan_expert_cache();
  std::expected<void, std::string> setup_vram_expert_cache();
  std::expected<void, std::string> offload_dense_tensors();
  std::expected<void, std::string> init_ram_tier();
  std::expected<void, std::string> preload_experts();
  std::expected<void, std::string> start_workers();
  std::expected<void, std::string> run_repl();
  std::expected<void, std::string> shutdown_and_report();
  std::vector<StreamingEngine::UsageEntry> sanitize_usage_entries(
      const std::string &path, const char *source,
      std::vector<StreamingEngine::UsageEntry> raw);

  const EngineConfig &config;
  const bool cufile_driver_opened;
  // Long-lived subsystems (formerly locals of run_engine, same declaration order).
  StreamingEngine engine;
  RAMSafetyManager ram_mgr;
  GGUFTokenizer tokenizer;
  LockFreePrefetcher prefetcher;
  CudaManager gpu;
  // Constructed in start_workers() only when active_layer_upload_worker_threads > 0.
  std::unique_ptr<ActiveLayerUploadWorkers> upload_worker_pool;

  // State computed by plan_expert_cache() and consumed by later phases.
  size_t expert_slab_bytes = 0;
  int cache_layers = 0;
  std::vector<size_t> expert_layer_slab_bytes;
  bool graph_mode = false;
  bool graph_ram_cache = false;
  bool dsv41_graph_uses_native_moe_cache = false;
  bool graph_owns_dense = false;
  bool glm_graph_aliases_vram_cache = false;
  bool qwen_cpu_ram_cache = false;
  bool ram_graph_cache = false;

  // REPL / shutdown state.
  std::chrono::high_resolution_clock::time_point session_start;
  std::string request_failure;
};
