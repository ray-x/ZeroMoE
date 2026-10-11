#include "active_layer_upload_workers.hpp"
#include "async_prefetcher.hpp"
#include "cuda_manager.hpp"
#include "engine.hpp"
#include "option_b_global_hot.hpp"
#include "engine_config.hpp"
#include "exl3_cpu.hpp"
#include "exl3_model_loader.hpp"
#include "exl3_decode.hpp"
#if defined(MOESTREAM_EXL3_CUDA)
#include "exl3_gpu.hpp"
#endif
#include "exllamav3_backend.hpp"
#include "generation_loop.hpp"
#include "thread_profile.hpp"
#include "ggml_deepseek_v4_graph.hpp"
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

class GGUFGraphInitializer {
public:
  static bool build_model_graph(StreamingEngine &engine, int max_ctx);
};

static std::expected<void, std::string>
load_exl3_model_only(const std::string &path) {
  SafetensorsModelInfo model;
  const bool parsed = std::filesystem::is_directory(path)
                          ? SafetensorsAdapter::parse_directory(path, model)
                          : SafetensorsAdapter::parse_file(path, model);
  if (!parsed)
    return std::unexpected("Failed to parse EXL3 safetensors model: " + path);
  if (!SafetensorsAdapter::is_exl3_model(model))
    return std::unexpected("No EXL3 trellis tensors found: " + path);

  Exl3Contract contract = SafetensorsAdapter::map_exl3_contract(model);
  if (!contract.complete)
    return std::unexpected("EXL3 tensor groups are incomplete or inconsistent");

  std::map<std::string, std::set<std::string>> sampled;
  std::set<int> shard_indices;
  int expected_shards = 0;
  const std::regex shard_pattern(R"(.*-([0-9]+)-of-([0-9]+)\.safetensors$)");
  for (const auto &entry : model.tensors) {
    const SafetensorsTensorInfo &info = entry.second;
    if (info.file_path.empty())
      return std::unexpected("EXL3 tensor has no backing shard: " + entry.first);

    std::smatch match;
    const std::string filename =
        std::filesystem::path(info.file_path).filename().string();
    if (std::regex_match(filename, match, shard_pattern)) {
      shard_indices.insert(std::stoi(match[1].str()));
      expected_shards = std::max(expected_shards, std::stoi(match[2].str()));
    }

    std::string kind;
    if (entry.first.ends_with(".trellis"))
      kind = "trellis";
    else if (entry.first.ends_with(".suh"))
      kind = "suh";
    else if (entry.first.ends_with(".svh"))
      kind = "svh";
    else
      continue;
    if (!sampled[info.file_path].insert(kind).second)
      continue;

    std::ifstream shard(info.file_path, std::ios::binary);
    if (!shard)
      return std::unexpected("Cannot open EXL3 shard: " + info.file_path);
    shard.seekg(static_cast<std::streamoff>(info.offset));
    const size_t bytes =
        static_cast<size_t>(std::min<std::uint64_t>(info.length, 4096));
    std::vector<char> buffer(bytes);
    shard.read(buffer.data(), static_cast<std::streamsize>(bytes));
    if (shard.gcount() != static_cast<std::streamsize>(bytes))
      return std::unexpected("Short EXL3 tensor read: " + entry.first);
  }

  if (expected_shards > 0 &&
      static_cast<int>(shard_indices.size()) != expected_shards)
    return std::unexpected("EXL3 shard set is incomplete: " +
                           std::to_string(shard_indices.size()) + "/" +
                           std::to_string(expected_shards));

  LOG_STREAM_INFO() << "[EXL3] Model catalog loaded: tensors="
                    << model.tensors.size() << " linears="
                    << contract.linears.size() << " layers="
                    << contract.n_layers << " experts=" << contract.n_experts
                    << " shards=" << shard_indices.size() << "/"
                    << expected_shards << "\n";
  LOG_STREAM_INFO() << "[EXL3] Representative trellis/suh/svh payload reads "
                       "passed for "
                    << sampled.size() << " safetensors files.\n";

  const Exl3LinearInfo *smoke_matrix = nullptr;
  for (const Exl3LinearInfo &linear : contract.linears) {
    if (!linear.valid || linear.codebook != Exl3Codebook::MUL1)
      continue;
    if (!smoke_matrix ||
        linear.in_features * linear.out_features <
            smoke_matrix->in_features * smoke_matrix->out_features)
      smoke_matrix = &linear;
  }
  if (!smoke_matrix)
    return std::unexpected("No dispatchable EXL3 mul1 matrix found");

  std::vector<uint8_t> packed_matrix;
  std::string matrix_error;
  if (!load_exl3_mul1_matrix(model, *smoke_matrix, packed_matrix,
                             matrix_error))
    return std::unexpected(matrix_error);
  std::vector<float> input(static_cast<size_t>(smoke_matrix->in_features));
  std::vector<float> output(static_cast<size_t>(smoke_matrix->out_features));
  for (size_t index = 0; index < input.size(); ++index)
    input[index] = std::sin(static_cast<float>(index) * 0.013f);
  if (!exl3_mul1_matmul_packed(
          packed_matrix.data(), input.data(), output.data(), 1,
          static_cast<int>(smoke_matrix->in_features),
          static_cast<int>(smoke_matrix->out_features)))
    return std::unexpected("EXL3 dispatch smoke matmul failed: " +
                           smoke_matrix->key);
  if (!std::all_of(output.begin(), output.end(),
                   [](float value) { return std::isfinite(value); }))
    return std::unexpected("EXL3 dispatch produced non-finite output: " +
                           smoke_matrix->key);
  LOG_STREAM_INFO() << "[EXL3] CPU dispatch smoke passed: "
                    << smoke_matrix->key << " ["
                    << smoke_matrix->in_features << ","
                    << smoke_matrix->out_features << "] "
                    << smoke_matrix->bits_per_weight << " bpw mul1.\n";
  return {};
}

#if defined(MOESTREAM_USE_GGML)
// The ggml tree used by the CUDA build is also shared with llama.cpp and emits
// CUDA-graph reuse diagnostics through GGML_LOG_DEBUG.  At MoEStream's
// normal INFO/WARN/ERROR levels those messages are not useful, but a full
// Qwen decode graph can hit the call site hundreds of times per token.  The
// default ggml callback ignores severity and prints them anyway, turning a
// diagnostic into a measurable serialization/stdio bottleneck.  Preserve
// all non-debug ggml messages and retain debug output when the user explicitly
// asks for DEBUG or VERBOSE logging.
static void moestream_ggml_log_callback(enum ggml_log_level level,
                                        const char *text, void *user_data) {
  const bool allow_debug =
      user_data && *static_cast<const bool *>(user_data);
  if (level == GGML_LOG_LEVEL_DEBUG && !allow_debug)
    return;
  std::fputs(text, stderr);
  std::fflush(stderr);
}
#endif

void print_usage() {
  std::cout
      << "Usage: ./moestream --config <config.toml>\n"
      << "       ./moestream --model <path_to_gguf> [options]\n"
      << "       ./moestream --gen-config [output.toml]\n\n"
      << "Options:\n"
      << "  --config <file>  Load engine config from TOML file\n"
      << "  --gen-config     Generate a sample config file and exit\n"
      << "  --model <path>   Path to GGUF model file or directory\n"
      << "  --ram <GB>       Strict RAM budget in GB (default: auto-detect 88% "
         "free)\n"
      << "  --ctx <N>        Maximum context size token window (default: "
         "4096)\n"
      << "  --ngen <N>       Maximum new tokens to generate per response "
         "(default: 1024)\n"
      << "  --temp <F>       Sampling temperature. 0.0 for greedy decode "
         "(default: 0.7)\n"
      << "  --top_p <F>      Nucleus sampling top-p threshold (default: 0.90)\n"
      << "  --gpu <ID>       CUDA device ID for single-GPU mode (default: 0)\n"
      << "  --drop-pages     Drop expert pages from page cache after use\n"
      << "  --direct-io      O_DIRECT expert-slab reads (Linux only; requires "
         "a model\n"
      << "                   repacked with tools/gguf_tool pack -- falls back "
         "to buffered\n"
      << "                   reads automatically if unsupported/unaligned)\n"
      << "                   (unpacked_expert_format and cuda_cufile are "
         "TOML-config-only; "
         "see --config)\n"
      << "  --io-uring       Batch the async prefetcher's per-expert reads "
         "through io_uring\n"
      << "                   (Linux only, kernel >= 5.6; independent of "
         "--direct-io -- falls\n"
      << "                   back to sequential preads automatically if "
         "unsupported)\n"
      << "  --prefetch-threads <N>  Concurrent async prefetcher worker threads "
         "(default: 2)\n"
      << "  --verbose-moe    Log every MoE host-cache hit (high volume, debug "
         "only)\n"
      << "  --promote-threshold <N>  Number of host re-hits before async H2D "
         "promote (default: 1)\n"
      << "  --pin-threshold <N>  VRAM-resident hit_count before pinning (never "
         "evicted; default: 8, 0=off)\n"
      << "  --pin-file <file>  Usage-frequency file for VRAM hot-set preload "
         "(default: .moestream_usage)\n"
      << "  --expert-fetch-threads <N>  OMP threads for parallel disk reads of "
         "a layer's topk cache misses (default: 4)\n"
      << "  --raw-prompt     Skip the chat-template wrap; tokenize stdin "
         "verbatim (needed by\n"
      << "                   api_server.py, which already applies its own "
         "formatting; also\n"
      << "                   useful for token-count-fair benchmarking against "
         "other engines)\n"
      << "  --system-prompt <text>  Prepend a system-role message ahead of the "
         "user turn\n"
      << "                   (chat-template modes only; ignored with "
         "--raw-prompt). Empty by\n"
      << "                   default -- MoEStream never invents one on its "
         "own; e.g. use this\n"
      << "                   to steer response language: --system-prompt "
         "\"Respond in English.\"\n"
      << "  --reasoning-effort <low|high|max>  GLM-5.3-Flash only: select the\n"
      << "                   Reasoning Effort system turn when thinking is "
         "enabled\n"
      << "                   (default: max; enable_thinking=false selects "
         "direct answer).\n"
      << "  --log-path <file>  Redirect logging to a file instead of stderr\n"
      << "  --log-level <lvl>  VERBOSE | DEBUG | INFO (default) | WARN | "
         "ERROR\n"
      << "  --trace-filter <tags>  Comma-separated tags (requires --log-level "
         "VERBOSE),\n"
      << "                   e.g. --trace-filter \"HC,MLA\" -- see any "
         "tensor-trace tag\n"
      << "  --trace-abort-on-nan  Abort at the first tensor a NaN/Inf checker "
         "flags\n"
      << "  --freeze-usage-stats  Don't overwrite pin_file/prefill_pin_file at "
         "shutdown\n"
      << "                   (still read at startup) -- for controlled A/B "
         "testing\n"
      << "  --collect-usage-stats  Override freeze_usage_stats and write the "
         "usage files at shutdown\n"
      << "  --force-start    Bypass the startup safety checks (concurrent "
         "moestream\n"
      << "                   instance, real RAM headroom, free VRAM) -- only "
         "use when\n"
      << "                   certain it's safe; see startup_preflight.hpp\n"
      << "  --help           Show this help interface\n";
}

// Batches up to Ring::CAPACITY/3 packed-format experts' reads into ONE
// io_uring submission, for the startup [PIN] preload pass -- real
// measurement (gguf_tool dma's own N-sweep) found that pass's original
// single-expert-at-a-time load_expert_slab loop runs at roughly the
// device's depth-1 ceiling (~2.2 GB/s) instead of its ~2.5 GB/s
// deep-queue ceiling, unlike decode-time reads this doesn't compete with
// any GPU compute for pacing, so there's nothing structurally stopping it
// from batching the same way RAMSafetyManager::get_experts_batch's own
// Phase B already does. All-or-nothing per chunk (same discipline as
// Phase B): on ANY failure (ring unavailable, batch too large, submit
// failure, short read) returns false and touches no slot's metadata --
// ensure_slab_capacity may already have grown a slab_buffer via
// build_expert_read_requests, which is harmless (grow-only, tolerated by
// a subsequent load_expert_slab call on the same slot); the caller must
// fall back to the original per-expert load_expert_slab loop for every
// item in this exact chunk.
//
// Handles BOTH packed and genuinely-unpacked (Phase 5 mixed-precision)
// layers, matching build_expert_read_requests' own layer_is_unpacked()
// branch below -- this comment previously said "packed format only,
// unpacked falls back immediately", true when build_expert_read_requests
// itself was packed-only, no longer true since it grew a real unpacked
// branch. A real bug lived here from exactly that drift: the SECOND,
// separate geometry lookup a few lines down (feeding populate_packed_slab)
// still unconditionally assumed packed, silently corrupting a mixed
// layer's smaller experts even though the real disk read right above it
// had already used the correct geometry -- see that call site's own
// comment for the full incident.
static bool preload_batch_load(
    StreamingEngine &engine, io_uring_backend::Ring &ring,
    const std::vector<std::tuple<int, int, MoEExpertBlock *>> &items) {
  if (items.empty() || !ring.available() ||
      items.size() * 3 > io_uring_backend::Ring::CAPACITY)
    return false;

  std::vector<io_uring_backend::ReadRequest> reqs;
  reqs.reserve(items.size() * 3);
  // Per-item geometry, needed to call populate_packed_slab after a
  // successful read -- build_expert_read_requests only fills the raw
  // slab_buffer bytes, not the QuantizedTensor host_ptr/meta fields (see
  // its own comment); load_expert_slab's packed-format branch calls
  // populate_packed_slab as its own last step, which this mirrors.
  struct ItemGeom {
    int layer;
    size_t gb, ub, db;
    const GGUFTensor *gM, *uM, *dM;
    int expert_id;
    MoEExpertBlock *slot;
  };
  std::vector<ItemGeom> geoms;
  geoms.reserve(items.size());

  // Stripe alternate experts across the two replicas when a mirror is
  // configured (GGUFModel::open_mirror) -- io_uring allows mixed fds in a
  // single submission, so this one submit_and_wait below is serviced by
  // BOTH devices' queues concurrently, roughly halving the wall time of
  // each chunk. Falls back to replica 0 for everything on a single-drive
  // setup (fd_for_tensor degrades internally), so no branch is needed here.
  for (auto &[layer, expert_id, slot] : items) {
    io_uring_backend::ReadRequest tmp[3];
    int cnt = 0;
    size_t total = 0;
    // Intra-expert split (see StreamingEngine::REPLICA_SPLIT): both
    // drives work on EVERY expert rather than alternating whole experts
    // between them, so even a chunk's last, otherwise-solo expert uses
    // both devices.
    int replica =
        engine.model_file.has_mirror() ? StreamingEngine::REPLICA_SPLIT : 0;
    if (!engine.build_expert_read_requests(layer, expert_id, *slot, tmp, cnt,
                                           total, replica) ||
        cnt != 3)
      return false;
    for (int j = 0; j < cnt; ++j)
      reqs.push_back(tmp[j]);
    size_t gb, ub, db;
    const GGUFTensor *gM, *uM, *dM;
    // MUST match the geometry build_expert_read_requests actually used
    // just above, not just try packed unconditionally -- a real,
    // confirmed bug (found 2026-09 chasing a real GPU-path NaN on a
    // genuinely mixed per-expert model): this function's own header
    // comment says "unpacked_expert_format/legacy per-expert-tensor
    // layouts fall back immediately (build_expert_read_requests
    // returns cnt!=3 or false for those)" -- true when this was
    // written, no longer true since build_expert_read_requests grew a
    // real layer_is_unpacked() branch (Phase 5 mixed-precision work)
    // that SUCCEEDS for an unpacked layer instead of falling back. The
    // real disk read a few lines up already correctly used the per-
    // expert unpacked geometry in that case -- but this second,
    // independent packed_slab_geometry() call still unconditionally
    // reported the layer's OLD packed baseline sizing (that tensor is
    // still physically present in a symlinked baseline shard
    // alongside the real override, same reason load_expert_slab had
    // this identical bug -- see its own comment), so a correctly-read
    // small cold expert's bytes got handed to populate_packed_slab
    // with the WRONG (larger, baseline-uniform) gate/up/down
    // boundaries, corrupting exactly where gate ends and up begins.
    if (engine.layer_is_unpacked(layer)) {
      if (!engine.unpacked_expert_geometry(layer, expert_id, gb, ub, db, gM, uM,
                                           dM))
        return false;
    } else if (!engine.packed_slab_geometry(layer, gb, ub, db, gM, uM, dM)) {
      return false;
    }
    geoms.push_back({layer, gb, ub, db, gM, uM, dM, expert_id, slot});
  }

  std::vector<ssize_t> results(reqs.size());
  // Timed and credited to the SAME t_disk_read/bytes_disk_read counters
  // load_expert_slab's own per-expert path uses -- mirrors RAMSafetyManager
  // ::get_experts_batch's Phase B convention exactly (times submit_and_wait
  // alone, charged unconditionally since a failed/short submission still
  // consumed real wall-clock time; bytes credited only on full success,
  // to avoid double-counting a chunk the caller then re-reads via the
  // load_expert_slab fallback). Without this, the shutdown report's
  // "Disk read (expert)" line would silently omit every batched chunk's
  // real disk activity -- a real gap this had before being caught by
  // comparing two real runs' own total-bytes figures.
  auto t0 = std::chrono::high_resolution_clock::now();
  bool submitted = ring.submit_and_wait(
      reqs.data(), (int)reqs.size(), results.data(), engine.drop_expert_pages);
  double batch_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - t0)
                        .count();
  PerfCounters::atomic_add(perf().t_disk_read, batch_ms / 1000.0);
  if (!submitted)
    return false;
  size_t total_bytes = 0;
  for (size_t r = 0; r < reqs.size(); ++r) {
    if (results[r] != (ssize_t)reqs[r].len)
      return false;
    total_bytes += reqs[r].len;
  }
  perf().bytes_disk_read.fetch_add(total_bytes, std::memory_order_relaxed);
  for (auto &g : geoms) {
    if (engine.graph_ram_layer_projection_compact(g.layer)) {
      if (!engine.populate_graph_ram_projection_views(g.layer, g.expert_id,
                                                      *g.slot))
        return false;
    } else {
      engine.populate_packed_slab(*g.slot, g.expert_id, g.gb, g.ub, g.db,
                                  *g.gM, *g.uM, *g.dM);
    }
  }
  return true;
}

std::expected<void, std::string> run_engine(const EngineConfig &config) {
  if (config.exl3_backend == "exllamav3")
    return std::unexpected("ExLlamaV3 must be launched through the config-mode CLI");
  if (config.model_format == "exl3") {
    if (!config.model_load_only) {
      return run_exl3_decode({
          .model_path = config.model_path,
          .metadata_model_path = config.metadata_model_path,
          .prompt = config.prompt,
          .context_size = config.context_size,
          .max_new_tokens = config.max_new_tokens,
          .threads = config.omp_max_threads > 0 ? config.omp_max_threads : 4,
          .temperature = config.temperature,
          .top_p = config.top_p,
          .raw_prompt = config.raw_prompt,
          .enable_thinking = config.enable_thinking,
          .gpu = config.exl3_gpu,
          .gpu_hc = config.exl3_gpu_hc,
          .gpu_device = config.exl3_gpu_device,
          .gpu_cache_gb = config.exl3_gpu_cache_gb,
          .gpu_expert_layer_begin = config.exl3_gpu_expert_layer_begin,
          .gpu_expert_split_layer = config.exl3_gpu_expert_split_layer,
          .gpu_split_device = config.exl3_gpu_split_device,
          .gpu_split_cache_gb = config.exl3_gpu_split_cache_gb,
          .cpu_expert_cache_gb = config.exl3_cpu_expert_cache_gb,
          .cpu_expert_int8_vnni = config.exl3_cpu_expert_int8_vnni});
    }
    auto loaded = load_exl3_model_only(config.model_path);
    if (!loaded)
      return loaded;
    return {};
  }

  // Process-wide default OpenMP team size -- MUST run before anything
  // else, since every #pragma omp parallel for without its own
  // num_threads() clause (attention_layer.cpp's dense/attention
  // matmuls, kernels_cpu.cpp's CPU expert-fallback dequant+dot path,
  // kquant_dequant.hpp's reference dequant) previously had NO cap at
  // all, defaulting to OpenMP's own runtime default (every available
  // core) with nothing to stop several of them running concurrently.
  // A real run showed the OS itself become unresponsive to other
  // commands from exactly this -- the CPU expert-fallback path alone
  // (expert_ffn_cpu -> cpu_quant_matmul x3 per call, gate/up/down) can
  // fire thousands of times per run, each burst spinning up to every
  // core. omp_set_num_threads() only changes the DEFAULT for regions
  // without their own explicit num_threads() clause -- ram_manager.hpp's
  // Phase B disk-read loop (num_threads(expert_fetch_threads)) is
  // unaffected and keeps using its own configured value regardless.
  {
    int omp_cap =
        config.omp_max_threads > 0
            ? config.omp_max_threads
            : std::max(1, (int)std::thread::hardware_concurrency() - 2);
    omp_set_num_threads(omp_cap);
    LOG_STREAM_INFO() << "[CONFIG] OMP process-wide default team size: "
                      << omp_cap
                      << (config.omp_max_threads > 0
                              ? " (explicit)"
                              : " (auto: hardware_concurrency-2)")
                      << "\n";
    // attn_score_threads (dense_dispatch.hpp): dedicated override for
    // attention_layer.cpp's causal score loop specifically -- 0 (unset)
    // means attn_score_threads() falls back to omp_get_max_threads(),
    // i.e. omp_cap above, so this is a genuine no-op when the config
    // never sets it. See EngineConfig::attn_score_threads' own comment.
    set_attn_score_threads(config.attn_score_threads);
  }

  // Declared BEFORE engine/gpu below (destructor runs in REVERSE
  // declaration order) so cuFileDriverClose() only ever fires AFTER
  // GpuTopology's own destructor has already deregistered every VRAM
  // pool/file handle it registered -- deregistering after the driver is
  // closed would be invalid. Fires on ANY exit path from run_engine
  // (normal completion or an early std::unexpected return), same
  // guarantee engine/gpu/ram_mgr already get for free as stack locals.
  struct CuFileDriverGuard {
    bool opened = false;
    ~CuFileDriverGuard() {
      if (opened)
        cufile_io::driver_close();
    }
  } cufile_driver_guard;
  if (config.cuda_cufile) {
    cufile_driver_guard.opened = cufile_io::driver_open();
    if (cufile_driver_guard.opened) {
      LOG_STREAM_INFO() << "[CUFILE] Driver opened.\n";
    } else {
      LOG_STREAM_WARN()
          << "[CUFILE] Driver open failed (GDS/nvidia-fs not installed, or "
             "cuFile not compiled in) -- cuda_cufile falls back to the "
             "RAM-mediated upload path for this entire run.\n";
    }
  }

  StreamingEngine engine;
  RAMSafetyManager ram_mgr;
  GGUFTokenizer tokenizer;
  LockFreePrefetcher prefetcher;
  CudaManager gpu;
  // Constructed further below (after gpu.topology is initialized), only
  // when config.active_layer_upload_worker_threads > 0 -- see main()'s own
  // construction site's comment for why it's placed there. Null/unset
  // means the feature is fully off (ram_mgr.upload_workers stays nullptr,
  // every call site's existing null-check keeps its current inline
  // behavior unchanged).
  std::unique_ptr<ActiveLayerUploadWorkers> upload_worker_pool;

  const std::string &model_path = config.model_path;
  int max_ctx = config.context_size;
  int max_new_tokens = config.max_new_tokens;

  engine.drop_expert_pages = config.drop_pages;
  engine.unpacked_expert_format = config.unpacked_expert_format;
  engine.prefill_warm_via_prefetcher = config.prefill_warm_via_prefetcher;
  engine.qwen_ram_preload_all_experts = config.ram_preload_all_experts;
  engine.qwen4exp_graph_expert_vram_budget_gb =
      config.qwen4exp_graph_expert_vram_budget_gb;
  engine.qwen4exp_graph_pin_file = config.pin_file;
  engine.exl3_ggml_bridge_model_path = config.exl3_ggml_bridge_model_path;
  engine.exl3_ggml_bridge_layer = config.exl3_ggml_bridge_layer;
  engine.exl3_ggml_bridge_layers = config.exl3_ggml_bridge_layers;
  engine.exl3_ggml_graph_capture = config.exl3_ggml_graph_capture;
  engine.exl3_ggml_routed_graph_capture =
      config.exl3_ggml_routed_graph_capture;
  engine.qwen4exp_option_b_layer = config.qwen4exp_option_b_layer;
  engine.qwen4exp_option_b_top_n = config.qwen4exp_option_b_top_n;
  engine.qwen4exp_option_b_min_coverage =
      config.qwen4exp_option_b_min_coverage;
  engine.qwen4exp_option_b_right_size_top_n =
      config.qwen4exp_option_b_right_size_top_n;
  engine.qwen4exp_option_b_global_top_k =
      config.qwen4exp_option_b_global_top_k;
  engine.qwen4exp_option_b_global_floor_per_layer =
      config.qwen4exp_option_b_global_floor_per_layer;
  engine.qwen4exp_option_b_max_layers = config.qwen4exp_option_b_max_layers;
  engine.qwen4exp_option_b_fill_vram = config.qwen4exp_option_b_fill_vram;
  engine.qwen4exp_option_b_concentrated_only =
      config.qwen4exp_option_b_concentrated_only;
  engine.qwen4exp_option_b_stage_expert =
      config.qwen4exp_option_b_stage_expert;
  engine.qwen4exp_option_b_stage_count =
      config.qwen4exp_option_b_stage_count;
  qwen4exp_full_graph_configure(config.qwen4exp_full_graph,
                                config.qwen4exp_full_graph_decode,
                                config.qwen4exp_prefill_state_handoff);
  ram_mgr.configured_total_experts =
      std::max(0, config.ram_cache_total_experts);
  // Must be set BEFORE the [PIN] preload block below -- that's where the
  // bulk of the RAM tier's slabs are first allocated, and each slab is
  // pinned exactly once at its own allocation (ensure_slab_capacity).
  // Setting it later would silently leave every preloaded slab pageable.
  engine.ram_pinned_budget_bytes =
      (size_t)(config.ram_pinned_budget_gb * 1024.0 * 1024.0 * 1024.0);
  if (config.ram_pinned_budget_gb > 0.0) {
    LOG_STREAM_INFO()
        << "[CONFIG] RAM slab persistent pin budget: "
        << config.ram_pinned_budget_gb
        << " GB of host_ram_gb=" << config.host_ram_gb
        << " -- slabs are cudaHostRegister'd once at "
           "allocation so upload_expert_active_async's DMA can overlap compute "
           "(see ram_pinned_budget_gb's own comment). Pinned pages are never "
           "reclaimable by the kernel; raise this deliberately, not "
           "wholesale.\n";
    LOG_STREAM_INFO()
        << "[CONFIG] RAM pin eager pre-registration (extends Tier 3's own "
           "ranking-informed preload further down its real candidate list when "
           "budget remains, single-threaded, before any request is timed -- "
           "see "
           "ram_pin_eager_preregister's own comment): "
        << (config.ram_pin_eager_preregister ? "ON" : "off") << "\n";
    LOG_STREAM_INFO()
        << "[CONFIG] RAM preload hit_count seeding (Tier 3 slots start "
           "'known warm' from their real pin_file count, clamped below "
           "promote_hit_threshold -- see ram_preload_seed_hit_count's own "
           "comment): "
        << (config.ram_preload_seed_hit_count ? "ON" : "off") << "\n";
  }
  engine.prefill_opportunistic_promote = config.prefill_opportunistic_promote;
  engine.decode_async_prewarm = config.decode_async_prewarm;
  engine.decode_async_prewarm_max_batch = config.decode_async_prewarm_max_batch;
  engine.decode_async_prewarm_nonblocking =
      config.decode_async_prewarm_nonblocking;
  engine.cpu_offload_max_experts = config.cpu_offload_max_experts;
  engine.prefill_retention_decay = config.prefill_retention_decay;
  engine.prefill_dispatch_lookahead_tokens =
      config.prefill_dispatch_lookahead_tokens;
  engine.prefill_hint_order_by_earliest_token =
      config.prefill_hint_order_by_earliest_token;
  engine.pilot_widen_all_tokens = config.pilot_widen_all_tokens;
  engine.prefetch_widen_all_tokens = config.prefetch_widen_all_tokens;
  engine.attn_qabsorb_gpu = config.attn_qabsorb_gpu;
  engine.attn_qabsorb_gpu_max_vram_gb = config.attn_qabsorb_gpu_max_vram_gb;
  engine.attn_score_gpu = config.attn_score_gpu;
  engine.qwen4exp_gdn_conv_gpu = config.qwen4exp_gdn_conv_gpu;
  engine.qwen4exp_router_pilot = config.qwen4exp_router_pilot;
  engine.qwen4exp_router_pilot_k = config.qwen4exp_router_pilot_k;
  engine.qwen4exp_router_pilot_prefetch = config.qwen4exp_router_pilot_prefetch;
  engine.qwen4exp_router_pilot_prefetch_k =
      config.qwen4exp_router_pilot_prefetch_k;
  engine.qwen4exp_router_pilot_prefetch_exclude_layers =
      config.qwen4exp_router_pilot_prefetch_exclude_layers;
  engine.qwen4exp_router_pilot_predict_layers =
      config.qwen4exp_router_pilot_predict_layers;
  engine.qwen4exp_router_pilot_skip_fully_cached =
      config.qwen4exp_router_pilot_skip_fully_cached;
  engine.qwen4exp_router_pilot_skip_prefetch_excluded_layers =
      config.qwen4exp_router_pilot_skip_prefetch_excluded_layers;
  engine.qwen4exp_decode_ggml_moe = config.qwen4exp_decode_ggml_moe;
  engine.qwen4exp_decode_scheduler = config.qwen4exp_decode_scheduler;
  engine.qwen4exp_decode_device_hc = config.qwen4exp_decode_device_hc;
  engine.qwen4exp_decode_batch_stage = config.qwen4exp_decode_batch_stage;
  engine.qwen4exp_decode_custom_hc_q8 = config.qwen4exp_decode_custom_hc_q8;
  engine.qwen4exp_decode_gpu_router = config.qwen4exp_decode_gpu_router;
  engine.qwen4exp_cpu_ggml_moe = config.qwen4exp_cpu_ggml_moe;
  engine.qwen4exp_cpu_strided_grouped =
      config.qwen4exp_cpu_strided_grouped;
  engine.qwen4exp_cpu_fused_grouped = config.qwen4exp_cpu_fused_grouped;
  engine.qwen4exp_cpu_gpu_overlap = config.qwen4exp_cpu_gpu_overlap;
  engine.qwen4exp_cpu_gpu_overlap_max_layers =
      config.qwen4exp_cpu_gpu_overlap_max_layers;
  engine.qwen4exp_cpu_shared_device = config.qwen4exp_cpu_shared_device;
  engine.qwen4exp_cpu_offload = config.qwen4exp_cpu_offload;
  engine.qwen4exp_cpu_offload_layers = config.qwen4exp_cpu_offload_layers;
  std::sort(engine.qwen4exp_cpu_offload_layers.begin(),
            engine.qwen4exp_cpu_offload_layers.end());
  engine.qwen4exp_cpu_offload_layers.erase(
      std::unique(engine.qwen4exp_cpu_offload_layers.begin(),
                  engine.qwen4exp_cpu_offload_layers.end()),
      engine.qwen4exp_cpu_offload_layers.end());
  engine.qwen4exp_cpu_offload_force_cold_count =
      config.qwen4exp_cpu_offload_force_cold_count;
  engine.qwen4exp_cpu_offload_force_expert =
      config.qwen4exp_cpu_offload_force_expert;
  engine.qwen4exp_cpu_offload_event_fence =
      config.qwen4exp_cpu_offload_event_fence;
  engine.qwen4exp_cpu_offload_ring_slots =
      config.qwen4exp_cpu_offload_ring_slots;
  engine.qwen4exp_gpu_stage_enabled = config.qwen4exp_gpu_stage_enabled;
  engine.qwen4exp_gpu_stage_layers = config.qwen4exp_gpu_stage_layers;
  std::sort(engine.qwen4exp_gpu_stage_layers.begin(),
            engine.qwen4exp_gpu_stage_layers.end());
  engine.qwen4exp_gpu_stage_layers.erase(
      std::unique(engine.qwen4exp_gpu_stage_layers.begin(),
                  engine.qwen4exp_gpu_stage_layers.end()),
      engine.qwen4exp_gpu_stage_layers.end());
  engine.qwen4exp_track_router_usage = config.qwen4exp_track_router_usage;
  engine.glm5next_kda_gpu = config.glm5next_kda_gpu;
  engine.glm5next_router_gpu = config.glm5next_router_gpu;
  engine.glm5next_moe_prefill_gpu = config.glm5next_moe_prefill_gpu;
  engine.glm5next_dense_prefill_fusion = config.glm5next_dense_prefill_fusion;
  engine.mtp_enabled = config.mtp_enabled;
  engine.mtp_draft_tokens = config.mtp_draft_tokens;
  engine.prompt_cache = config.prompt_cache;
  static MemoryGuard g_mem_guard;
  if (config.mem_guard)
    g_mem_guard.start(config.mem_guard_warn_mb, config.mem_guard_stop_mb,
                      config.mem_guard_poll_ms);
  // Graph-owned expert arenas and the legacy RAM-tier slabs are allocated by
  // different subsystems.  Configure one process-wide admission ceiling so
  // a graph source allocation cannot silently sit outside host_ram_gb.  For
  // host_ram_gb=0, resolve the same 88%-of-currently-available policy used
  // by RAMSafetyManager; this is only an admission ceiling, while the
  // background watchdog still protects the live low-water mark.
  size_t host_budget_bytes = 0;
  if (config.host_ram_gb > 0.0) {
    const double bytes = config.host_ram_gb * 1.0e9;
    host_budget_bytes = bytes >= (double)SIZE_MAX ? SIZE_MAX : (size_t)bytes;
  } else {
    const long available_mb = MemoryGuard::read_available_mb();
    if (available_mb > 0)
      host_budget_bytes =
          (size_t)((double)available_mb * 1024.0 * 1024.0 * 0.88);
  }
  MemoryGuard::configure_host_budget(host_budget_bytes);
  LOG_STREAM_INFO()
      << "[MEM-GUARD] shared host admission budget: "
      << (host_budget_bytes > 0
              ? std::to_string((double)host_budget_bytes / 1.0e9) + " GB"
              : "unbounded (automatic budget unavailable)")
      << "\n";
  engine.attn_score_gpu_max_vram_gb = config.attn_score_gpu_max_vram_gb;
  engine.attn_score_gpu_max_context = config.attn_score_gpu_max_context;
  engine.dsv4_score_gpu = config.dsv4_score_gpu;
  engine.dsv4_score_gpu_max_vram_gb = config.dsv4_score_gpu_max_vram_gb;
  ram_mgr.configured_ram_gb = config.host_ram_gb;
  ram_mgr.promote_hit_threshold = (uint32_t)config.promote_hit_threshold;
  ram_mgr.reactive_promote_gpu1_every_n = config.reactive_promote_gpu1_every_n;
  ram_mgr.pin_threshold = (uint32_t)std::max(0, config.pin_threshold);
  ram_mgr.verbose_cache_log = config.verbose_moe_log;
  ram_mgr.expert_fetch_threads = std::max(1, config.expert_fetch_threads);
  ram_mgr.move_on_promotion = config.move_on_promotion;
  ram_mgr.ram_copy_reclaim_threshold = config.ram_copy_reclaim_threshold;
  ram_mgr.disk_batch_padding = config.disk_batch_padding;
  ram_mgr.disk_batch_padding_target = config.disk_batch_padding_target;
  ram_mgr.disk_batch_padding_evict_dup = config.disk_batch_padding_evict_dup;
  gpu.topology.active_layer_promote_threshold =
      std::max(1, config.active_layer_promote_hit_threshold);
  gpu.topology.speculative_active_layer_fill =
      config.speculative_promote_to_active_layer;
  // The CUDA runtime reports all physical adapters, but the TOML's
  // [[gpus]] list is an explicit allocation allow-list. Without passing it
  // into the topology, a GPU omitted from a single-GPU profile was still
  // auto-assigned as an ExpertVault and received a large expert pool.
  gpu.topology.configured_device_ids.clear();
  for (const auto &gc : config.gpus)
    gpu.topology.configured_device_ids.push_back(gc.device_id);
  if (config.dense_gpu_idx >= 0) {
    gpu.device_id = config.gpus[config.dense_gpu_idx].device_id;
    // REAL BUG FIX: gpu.device_id above was being set correctly here,
    // then IMMEDIATELY overwritten by CudaManager::initialize() one
    // line into gpu.initialize() below, with whatever
    // GpuTopology::assign_roles()'s SM-architecture/VRAM auto-detect
    // heuristic picked -- assign_roles() took no parameters and had
    // no way to know a config preference even existed, so a user's
    // explicit role="dense" on the OLDER-architecture GPU was silently
    // discarded every time (confirmed on a real run: TOML said the
    // 3090 was dense/CoreEngine, but "[GPU] Hardware Profile" showed
    // the newer-arch 5060 Ti as CoreEngine instead). Plumbing the same
    // device_id through to forced_core_device_id lets assign_roles()
    // actually honor it.
    gpu.topology.forced_core_device_id =
        config.gpus[config.dense_gpu_idx].device_id;
  }
  if (!gpu.initialize()) {
    return std::unexpected("Failed to initialize GPU subsystem");
  }
  gpu.set_dense_async_copies(config.dense_async_copies);
  gpu.report_vram_usage("after gpu.initialize()");
  engine.gpu = &gpu; // give the forward path access to dense GPU dispatch
  // Captured separately (not read back via engine.gpu) so StreamingEngine's
  // destructor can free its own device-resident buffers (resident_layers'
  // device_ptr, d_Lc/d_Rc/d_Lc_q4/d_Rc_q4, d_hidden) correctly regardless
  // of whether `gpu` (declared before `engine` above) has already been
  // destroyed by the time engine's destructor runs.
  engine.gpu_device_id = gpu.get_core_device();
#if defined(MOESTREAM_EXL3_CUDA) && defined(MOESTREAM_USE_GGML_CUDA)
  if (!engine.exl3_ggml_bridge_model_path.empty()) {
    const size_t cache_bytes = static_cast<size_t>(
        config.exl3_gpu_cache_gb * 1024.0 * 1024.0 * 1024.0);
    try {
    engine.exl3_gpu_executor_owner = std::make_shared<Exl3GpuExecutor>(
        engine.gpu_device_id, cache_bytes);
    engine.exl3_gpu_executor_owner->set_routed_graph_capture(
        config.exl3_ggml_routed_graph_capture);
    } catch (const std::exception &error) {
      return std::unexpected(std::string("Failed to initialize EXL3 GGML "
                                         "bridge cache: ") + error.what());
    }
    engine.exl3_gpu_executor = engine.exl3_gpu_executor_owner.get();
    if (config.exl3_cpu_expert_cache_gb > 0.0) {
      const size_t host_cache_bytes = static_cast<size_t>(
          config.exl3_cpu_expert_cache_gb * 1024.0 * 1024.0 * 1024.0);
      engine.exl3_cpu_matrix_cache =
          std::make_shared<Exl3CpuMatrixCache>(host_cache_bytes);
    }
    LOG_STREAM_INFO() << "[EXL3-GGML-BRIDGE] cache executor initialized on GPU "
                      << engine.gpu_device_id << " with "
                      << config.exl3_gpu_cache_gb << " GiB cache budget\n";
  }
#endif

  if (!engine.model_file.load_header(model_path)) {
    return std::unexpected(
        "Failed to parse GGUF file or map descriptor tree structure");
  }
  // Opt-in only (config.direct_io, default false) -- see GGUFFile::
  // enable_direct_io's own comment for the alignment check + per-shard
  // test-read verification this does before trusting O_DIRECT on any
  // shard. Called right after load_header (all shard fds + KV metadata
  // are populated by then) and before anything starts reading experts.
  if (config.direct_io)
    engine.model_file.enable_direct_io();
  // Opened AFTER enable_direct_io deliberately: open_mirror matches each
  // shard's O_DIRECT state against the primary's, which isn't decided until
  // the call above has run and verified it per shard.
  if (!config.mirror_model_path.empty())
    engine.model_file.open_mirror(config.mirror_model_path);

  // Opt-in, only reachable when the driver actually opened above (config.
  // cuda_cufile alone doesn't guarantee that -- see cufile_driver_guard).
  // EngineConfig::validate() already refused cuda_cufile=true without
  // direct_io=true, so direct_io has already run by this point whenever
  // this branch is live. engine.cuda_cufile_active (the flag execution_
  // layer.cpp's Phase 3 loop actually checks) is only set true if at
  // least one shard genuinely registered -- "requested" vs "actually
  // active", same distinction direct_io/io_uring already draw.
  if (config.cuda_cufile && cufile_driver_guard.opened) {
    int n = engine.model_file.enable_cufile();
    engine.cuda_cufile_active = (n > 0);
  }

  // Optional separate dense GGUF: dense/preloaded tensors read from here,
  // routed experts still stream from model_path. Enables a
  // dense(IQ4_XS)+expert(IQ2_*) mix across two files.
  if (!config.dense_model_path.empty() &&
      config.dense_model_path != model_path) {
    if (!engine.dense_model_file.load_header(config.dense_model_path)) {
      return std::unexpected("Failed to parse dense GGUF (dense_model_path)");
    }
    engine.has_dense_model = true;
    if (config.direct_io)
      engine.dense_model_file.enable_direct_io();
  }

  // Optional standalone "shared" MTP/NextN checkpoint (Qwen4Exp convention
  // -- see EngineConfig::mtp_model_path's own comment). Loaded BEFORE
  // build_model_graph below so graph_initializer.cpp's own MTP validation
  // can see engine.has_mtp_model/mtp_model_file when deciding whether
  // Qwen4Exp MTP is actually available for this run.
  if (!config.mtp_model_path.empty()) {
    if (!engine.mtp_model_file.load_header(config.mtp_model_path)) {
      return std::unexpected("Failed to parse MTP GGUF (mtp_model_path)");
    }
    engine.has_mtp_model = true;
    if (config.direct_io)
      engine.mtp_model_file.enable_direct_io();
  }

  const std::string &gguf_path = engine.model_file.resolved_file_path;
  std::ifstream metadata_stream(gguf_path, std::ios::binary);
  // Skip GGUF header: magic(4) + version(4) + tensor_count(8) + kv_count(8) =
  // 24 bytes
  metadata_stream.seekg(4 + 4, std::ios::beg); // skip magic + version
  uint64_t tensor_count_raw = 0, kv_count_raw = 0;
  metadata_stream.read(reinterpret_cast<char *>(&tensor_count_raw), 8);
  metadata_stream.read(reinterpret_cast<char *>(&kv_count_raw), 8);

  if (!tokenizer.load_from_gguf_metadata(metadata_stream, kv_count_raw)) {
    return std::unexpected(
        "Tokenizer vocabulary table completely missing from GGUF metadata");
  }
  metadata_stream.close();

  if (!GGUFGraphInitializer::build_model_graph(engine, max_ctx)) {
    return std::unexpected(
        "Failure triggered during topological model graph initialization pass");
  }
  engine.qwen4exp_full_graph_required =
      engine.family_ops.family == ModelFamily::Qwen4Exp &&
      qwen4exp_full_graph_decode_enabled();
  if (engine.qwen4exp_full_graph_required &&
      !qwen4exp_full_graph_enabled()) {
    return std::unexpected(
        "Qwen4Exp full-graph decode is enabled but full-graph prefill is "
        "disabled; configure both graph phases or explicitly exempt this run");
  }
  if (engine.qwen4exp_full_graph_required && engine.mtp_enabled) {
    return std::unexpected(
        "Qwen4Exp MTP is not integrated with the full-graph-only trunk; "
        "disable MTP explicitly or use a graph-enabled MTP implementation");
  }
  if (engine.qwen4exp_full_graph_required) {
    LOG_INFO("[QWEN4EXP-GRAPH-POLICY] full-graph prefill/decode required; "
             "a graph build or replay failure aborts the request\n");
  }
  engine.init_route_freq();

  // Alignment diagnostic. Must come AFTER build_model_graph: it walks every
  // MoE layer, and architecture_config.n_layers is only known as of that
  // call (an earlier placement silently checked zero layers and printed
  // nothing at all). Direct-io/mirror setup has already run by here too, so
  // the report reflects the O_DIRECT state actually in force. Emits to
  // stderr as well as the log -- see the function's own comment for why.
  engine.warn_if_expert_tensors_unaligned();

  // Whole-shard mmap for the prefill expert-streaming path. Enabled here
  // rather than lazily so a failure is visible at startup and simply
  // leaves moe_prefill_stream inert (moe_forward_gguf checks
  // mmap_active()) instead of failing mid-prefill. Deliberately AFTER
  // enable_direct_io: the mapping uses its own non-O_DIRECT fd, and the
  // ordering makes that independence obvious at the call site.
  if (config.moe_prefill_stream) {
    if (!config.moe_prefill_batched) {
      LOG_WARN("[CONFIG] moe_prefill_stream=true has no effect without "
               "moe_prefill_batched=true (it replaces how that path "
               "acquires weights) -- staying on the cached path.\n");
    } else if (!engine.model_file.enable_mmap()) {
      LOG_WARN("[CONFIG] moe_prefill_stream=true but the shard mmap could "
               "not be established -- staying on the cached path.\n");
    }
  }

  // cpu_affinity_layers (config): must be set before ANYTHING below can
  // dispatch an expert or preload VRAM -- expert_ffn_resolve_gpu_loc,
  // RAMSafetyManager's reactive-hit promotion, Phase 2's prefill warm-up,
  // and the [PIN] preload block right below all consult
  // is_cpu_affinity_layer() (dense_dispatch.hpp). architecture_config.
  // n_layers is only known as of build_model_graph above, so (like the
  // unpacked-format sanity check right below) this is the earliest point
  // it can be set.
  if (!config.cpu_affinity_layers.empty()) {
    set_cpu_affinity_layers(config.cpu_affinity_layers,
                            engine.model_file.architecture_config.n_layers);
  }
  // Same set-once-before-any-dispatch timing as cpu_affinity_layers above
  // -- execution_layer.cpp's pilot loop reads pilot_k_for_layer() on
  // every layer transition starting with the very first prefill token.
  set_prefetch_pilot_k(config.prefetch_pilot_k_per_layer,
                       config.prefetch_pilot_k);
  // prefetch_pilot_k_adaptive (2026-09-13) -- narrows the pilot_k_for_layer
  // ceiling above to each layer's own live most-confident ranks. See
  // PilotKAdaptiveConfig's own comment (dense_dispatch.hpp).
  set_pilot_k_adaptive_config(config.prefetch_pilot_k_adaptive,
                              config.prefetch_pilot_k_adaptive_threshold,
                              config.prefetch_pilot_k_adaptive_min_samples);
  // Same timing/reasoning as the pilot_k call directly above, for
  // EngineConfig::prefetch_pilot_depth / prefetch_pilot_k_depth2.
  set_prefetch_pilot_depth(config.prefetch_pilot_depth,
                           config.prefetch_pilot_k_depth2);
  set_attn_pilot_enabled(config.prefetch_attn_pilot,
                         config.prefetch_attn_pilot_k);
  set_attn_pilot_vram_promote_top_k(config.attn_pilot_vram_promote_top_k);
  set_attn_pilot_vram_promote_dynamic(config.attn_pilot_vram_promote_dynamic);
  // Same timing/reasoning again, for the pilot-driven RAM->VRAM
  // speculative promotion config (ram_vram_promote_top_k/adaptive) --
  // execution_layer.cpp's pilot loop reads this on the very same l+1
  // transition as pilot_k_for_layer above.
  set_ram_vram_promote_config(config.ram_vram_promote_top_k,
                              config.ram_vram_promote_adaptive,
                              config.ram_vram_promote_adaptive_threshold,
                              config.ram_vram_promote_min_samples);
  // Per-layer accuracy veto (2026-08-25) -- layered on top of whichever
  // mode set_ram_vram_promote_config/set_attn_pilot_vram_promote_top_k
  // just configured above, see LayerPromoteGateConfig's own comment
  // (dense_dispatch.hpp).
  set_layer_promote_gate_config(config.ram_vram_promote_layer_gate,
                                config.ram_vram_promote_layer_gate_threshold,
                                config.ram_vram_promote_layer_gate_min_samples,
                                config.ram_vram_promote_layer_gate_top_fraction);
  // Per-layer dynamic WIDTH (2026-08-26) -- graduates the veto above into
  // a smooth per-layer promotion width, see RamVramPromoteConfig's own
  // dynamic_* fields (dense_dispatch.hpp) and EngineConfig::ram_vram_
  // promote_dynamic_layer's own comment.
  set_ram_vram_promote_dynamic_config(
      config.ram_vram_promote_dynamic_layer,
      config.ram_vram_promote_dynamic_threshold_4,
      config.ram_vram_promote_dynamic_threshold_3,
      config.ram_vram_promote_dynamic_threshold_2,
      config.ram_vram_promote_dynamic_threshold_1,
      config.ram_vram_promote_dynamic_k_tier4,
      config.ram_vram_promote_dynamic_k_tier3,
      config.ram_vram_promote_dynamic_k_tier2,
      config.ram_vram_promote_dynamic_k_tier1,
      config.ram_vram_promote_dynamic_k_floor,
      config.ram_vram_promote_dynamic_cold_start_k,
      config.ram_vram_promote_dynamic_min_samples);
  set_gpu1_predict_promote_config(
      config.predict_gpu1_rank_start >= 0, config.predict_gpu1_rank_start,
      config.predict_gpu1_rank_end, config.predict_gpu1_cap_per_layer);
  // Redirect pilot promotion into the active-layer pool instead of the
  // main VRAM cache (2026-08-26) -- see EngineConfig::speculative_promote_
  // to_active_layer's own comment. The dense_dispatch.hpp-side flag
  // (read by execution_layer.cpp's call sites) and the GpuTopology-side
  // flag (gpu.topology.speculative_active_layer_fill, set above --
  // consulted by cleanup_active_layer_area) are deliberately two separate
  // settings from the SAME config value: the former decides which
  // function the pilot loop calls, the latter decides whether cleanup
  // protects a one-layer-ahead entry. Both must agree for the feature to
  // behave correctly, which is why both are set here from the same
  // config.speculative_promote_to_active_layer, not independently.
  set_speculative_promote_to_active_layer(
      config.speculative_promote_to_active_layer);
  set_speculative_active_layer_max_k(config.speculative_active_layer_max_k);
  set_speculative_active_layer_promote_on_confirm(
      config.speculative_active_layer_promote_on_confirm);

  // 2026-08-28: multi-GPU active-layer routing registry -- see
  // EngineConfig::active_layer_routing_policy's own comment for the
  // full design. Built from whichever [[gpus]] entries opted in via
  // active_layer_weight > 0; route_active_layer_gpu_impl/
  // active_layer_multi_gpu_enabled (dense_dispatch.hpp) both correctly
  // degrade to "no routing" when fewer than 2 entries qualify, so this
  // is safe to call unconditionally regardless of whether the feature
  // is actually configured.
  {
    std::vector<ActiveLayerRouteGpu> al_gpus;
    for (auto &gc : config.gpus) {
      if (gc.active_layer_weight > 0)
        al_gpus.push_back({gc.device_id, gc.active_layer_weight});
    }
    set_active_layer_routing_config(config.active_layer_routing_policy,
                                    al_gpus);
  }

  // unpacked_expert_format is an explicit performance contract (single
  // combined DMA per expert) -- validate it holds for the WHOLE model
  // now, before any real generation work (or any expert data read) ever
  // starts, rather than let it fail unpredictably mid-run or silently
  // degrade. architecture_config.n_layers/n_experts is only populated as
  // of build_model_graph above, so this is the earliest point the check
  // can run. See StreamingEngine::sanity_check_unpacked_format for what
  // gets validated and why a violation is fatal rather than a fallback.
  if (engine.unpacked_expert_format && !engine.sanity_check_unpacked_format()) {
    return std::unexpected(
        "unpacked_expert_format=true but the model failed the aligned-unpacked "
        "sanity check -- see the [FATAL] line(s) above for exactly which "
        "layer/expert failed and why. Refusing to start rather than silently "
        "fall back to the slow path this flag exists to avoid.");
  }
  // The other direction of the same mismatch: unpacked_expert_format is
  // OFF, but the model itself is genuinely in the per-expert layout --
  // a real, measured incident (see CLAUDE.md): this silently disables
  // BOTH io_uring Phase B batching and the VRAM-shadow bypass fast path
  // for the entire run with zero visible error, showing up only as 100%
  // "layer-not-packed" on the (easy-to-miss) Decode-thread io_uring
  // batches shutdown line and a disk= figure that exceeds its own
  // containing expert= wall-clock window. Warn-only, not a hard refusal
  // like the block above -- the run can still complete correctly here,
  // just slowly, so there's no contract violation to refuse over.
  if (!engine.unpacked_expert_format && engine.looks_like_unpacked_format()) {
    LOG_STREAM_WARN()
        << "[WARN] This model appears to be in the per-expert (\"unpacked\") "
           "tensor "
           "format (gguf_tool verify would report 'Expert format: per-expert' "
           "for "
           "it), but unpacked_expert_format is not set -- io_uring batching "
           "and the "
           "VRAM-shadow bypass fast path will BOTH be silently disabled for "
           "this "
           "entire run, forcing every dispatch through the slow, unbatched "
           "fallback "
           "path. Set unpacked_expert_format=true in your config to use the "
           "single-DMA fast path this format is built for. See CLAUDE.md.\n";
  }

  // --- Multi-GPU memory planning (from optimize.md) ---
  auto &cfg = engine.model_file.architecture_config;

  // Allocate expert VRAM cache on vault GPU(s)
  size_t expert_slab_bytes = 0;
  // The optional GLM NextN/MTP block is a real streamed MoE layer too.
  // Use the cache topology populated by graph_initializer rather than only
  // the target trunk, so its per-layer geometry is sized consistently.
  const int cache_layers = static_cast<int>(engine.layers_experts.size());
  std::vector<size_t> expert_layer_slab_bytes((size_t)std::max(0, cache_layers),
                                              0);
  if (cfg.n_experts > 0 && cfg.moe_inter > 0) {
    // Real per-expert byte size, accounting for each projection's own quant
    // type (e.g. GLM-5.2 UD ships gate/up=Q2_K, down=Q3_K — a flat "2-bit
    // guess" undersizes the slab by ~45% and causes VRAM cache corruption /
    // OOB reads once an expert is promoted). GpuTopology uses ONE uniform
    // slot size for the entire VRAM pool across every layer, but "Dynamic"
    // GGUF quantization (e.g. Unsloth UD) can use a DIFFERENT quant type
    // per layer — so sizing from just the first MoE layer found can still
    // undersize the pool for a later, bigger-quant layer. Scan every MoE
    // layer and take the max.
    for (int l = cfg.first_dense; l < cache_layers; ++l) {
      const size_t raw_layer_bytes = engine.compute_expert_slab_bytes(l);
      // The compact pool is still laid out in byte-addressable VRAM,
      // but 4 KiB region padding keeps every following slot/page base
      // aligned for future direct GPU/file transfers. Explicit
      // full-layer prefix regions additionally need a stride that is a
      // whole number of quantization rows so GGML's MUL_MAT_ID can view
      // the resident slots without a repack. Ordinary compact regions
      // retain the smaller 4 KiB-only layout.
      constexpr size_t COMPACT_REGION_ALIGNMENT = 4096;
      size_t layer_slot_bytes = raw_layer_bytes;
      if (layer_slot_bytes > 0) {
        size_t slot_alignment = COMPACT_REGION_ALIGNMENT;
        // The GLM5NEXT full graph aliases resident slots as MUL_MAT_ID views for ANY layer with
        // hot experts, so every layer needs a row-aligned slot stride, not just the full-layer prefix.
        const bool glm_graph_rows = engine.family_ops.family == ModelFamily::GLM5NEXT &&
                                    glm5next_full_graph_enabled();
        if (config.vram_cache_full_layer_prefix > l || glm_graph_rows) {
          const size_t row_alignment =
              engine.expert_cache_layer_stride_alignment(l);
          if (row_alignment == SIZE_MAX) {
            LOG_STREAM_WARN()
                << "[GPU] Full-layer cache row alignment overflow at layer "
                << l << "; leaving the 4 KiB compact stride\n";
          } else if (row_alignment > 1) {
            // The direct-view path only needs row alignment. Do
            // not take the LCM with 4096 here: for IQ formats the
            // LCM can be tens of MiB per expert and would waste
            // most of the cache on alignment padding alone.
            slot_alignment = row_alignment;
          }
        }
        const size_t rem = layer_slot_bytes % slot_alignment;
        if (rem != 0 && layer_slot_bytes <= SIZE_MAX - (slot_alignment - rem)) {
          layer_slot_bytes += slot_alignment - rem;
        }
      }
      expert_layer_slab_bytes[(size_t)l] = layer_slot_bytes;
      expert_slab_bytes = std::max(expert_slab_bytes, raw_layer_bytes);
    }
    // The direct resident-layer GGML view uses the cache's physical slot
    // stride as the quantized tensor nb[2].  Round the model-wide maximum
    // slot to a whole number of rows for every routed projection so
    // mixed IQ formats (for example IQ2_XXS + IQ1_M) remain valid when
    // viewed without a repack.  This is a small cache-capacity tradeoff;
    // the normal compact path would not require it, but the full graph
    // zero-repack path does.
    const size_t stride_alignment = engine.expert_cache_stride_alignment();
    if (expert_slab_bytes > 0 && stride_alignment > 1 &&
        stride_alignment != SIZE_MAX) {
      const size_t rem = expert_slab_bytes % stride_alignment;
      if (rem != 0)
        expert_slab_bytes += stride_alignment - rem;
      LOG_STREAM_INFO() << "[GPU] Expert staging/legacy stride: "
                        << expert_slab_bytes
                        << " bytes (row-compatible alignment "
                        << stride_alignment
                        << "; compact per-layer VRAM regions use their own 4 "
                           "KiB-padded sizes)\n";
    } else if (stride_alignment == SIZE_MAX) {
      LOG_STREAM_WARN() << "[GPU] Expert cache row alignment overflow; "
                        << "disabling zero-repack stride adjustment\n";
    }
  }
  // GLM NextN/MTP's own target-verification pass (MtpSpeculator::
  // target_forward, mtp_speculative.cpp) always calls layers_forward_gguf
  // directly and never the full graph -- that call site's own comment
  // already anticipates "a full-graph fast path already produced final
  // output" as a real (if previously unreachable) case and guards it
  // explicitly. The full graph itself still does not build the optional
  // NextN draft block, so MTP's draft/verification loop continues to run
  // entirely on the legacy path regardless of this flag; only the MAIN
  // generation loop's own prefill is affected.
  //
  // This used to unconditionally disable the full graph whenever MTP was
  // enabled. That was a real, warranted precaution at the time: the full
  // graph's prefill-to-decode handoff (KDA/GDN recurrent state, conv
  // history) had a silent state-transposition bug (ggml-cuda's
  // gated_delta_net stores state [k-fast,v-slow]; this codebase's own
  // legacy CPU delta step expects [k-slow,v-fast] -- see build_kda's own
  // comment, ggml_glm5next_graph.cpp) that corrupted every decode token
  // after a full-graph prefill. Running MTP -- which depends on an
  // accurate continuing state to draft against -- on top of that would
  // have made an already-hard-to-diagnose bug even harder to isolate. Now
  // that the transposition is fixed (ggml_cont(ggml_transpose(...)) before
  // the state is copied out), there is no longer a known correctness
  // reason to keep the two mutually exclusive.
  const auto &graph_ops = engine.family_ops.graph_ops;
  // Graph mode owns startup-time resources when either phase is graph-backed.
  // In particular, GLM can intentionally keep batched prefill on the legacy
  // path while using the reusable full graph for single-token decode; still
  // prepare its shared-cache references and reusable host source here.
  const bool graph_mode =
      graph_ops.enabled &&
      (graph_ops.enabled(engine, ModelGraphPhase::Prefill) ||
       graph_ops.enabled(engine, ModelGraphPhase::Decode));
  // Full-graph host-resident expert layers use fixed-stride DDR arenas when
  // ram_preload_all_experts is enabled. GLM also aliases its usage-ranked
  // resident experts in the existing VRAM cache; other graph families retain
  // their own source-ownership policy behind the explicit environment flag.
  const bool graph_ram_cache =
      graph_mode &&
      (std::getenv("MOESTREAM_FULL_GRAPH_USE_RAM_CACHE") != nullptr ||
       (engine.family_ops.family == ModelFamily::GLM5NEXT &&
        config.ram_preload_all_experts) ||
       (engine.family_ops.family == ModelFamily::Qwen4Exp &&
        config.ram_preload_all_experts));
  const bool glm_graph_aliases_vram_cache =
      graph_mode && engine.family_ops.family == ModelFamily::GLM5NEXT &&
      glm5next_full_graph_enabled();
  if (graph_mode && engine.family_ops.family == ModelFamily::Qwen4Exp &&
      qwen4exp_full_graph_decode_enabled()) {
    const int n_layers =
        std::max(0, engine.model_file.architecture_config.n_layers);
    const int n_experts =
        std::max(0, engine.model_file.architecture_config.n_experts);
    const int device_layers = std::min(
        std::max(0, qwen4exp_full_graph_device_expert_layers()), n_layers);
    if (device_layers > 0) {
      engine.qwen_ram_vram_only_layers.assign((size_t)n_layers, 0);
      const bool trailing = qwen4exp_full_graph_device_expert_trailing();
      const int first_layer = trailing ? n_layers - device_layers : 0;
      size_t vram_only_bytes = 0;
      int vram_only_layer_count = 0;
      for (int layer = first_layer; layer < first_layer + device_layers;
           ++layer) {
        if (is_cpu_affinity_layer(layer))
          continue;
        engine.qwen_ram_vram_only_layers[(size_t)layer] = 1;
        ++vram_only_layer_count;
        const size_t expert_bytes = engine.compute_expert_slab_bytes(layer);
        if (n_experts > 0 &&
            expert_bytes <= SIZE_MAX / (size_t)n_experts &&
            vram_only_bytes <=
                SIZE_MAX - expert_bytes * (size_t)n_experts)
          vram_only_bytes += expert_bytes * (size_t)n_experts;
      }
      LOG_STREAM_INFO()
          << "[MOE-OWNERSHIP] full-graph device block [" << first_layer
          << ", " << (first_layer + device_layers)
          << ") marks " << vram_only_layer_count
          << " non-CPU-affinity layer(s) as VRAM-only, approximately "
          << (double)vram_only_bytes / (1024.0 * 1024.0 * 1024.0)
          << " GiB; persistent DDR preload/graph-RAM exclusion applies "
             "when those tiers are enabled\n";
    }
  }
  // GLM5NEXT's device-resident expert layers (glm5next_full_graph's own
  // device_expert_layers) were previously invisible to this arena-request
  // loop: only Qwen4Exp populated qwen_ram_vram_only_layers, so GLM always
  // requested a complete host-RAM arena for every non-dense layer even when
  // MOESTREAM_GLM5NEXT_GRAPH_PRELOAD_LAYERS asked for some of them to live
  // in VRAM instead. That silently doubled down on host RAM (the graph's
  // per-layer arena reservation is on top of the RAM tier's own preload)
  // and could exhaust the MemoryGuard ledger's watermark headroom for the
  // model's last layer(s) even though real MemAvailable had room to spare --
  // see prepare_expert_source_only's "boot host admission refused" log.
  if (graph_mode && graph_ram_cache && config.ram_preload_all_experts &&
      engine.family_ops.family == ModelFamily::GLM5NEXT &&
      glm5next_full_graph_enabled()) {
    const int n_layers =
        std::max(0, engine.model_file.architecture_config.n_layers);
    const int n_experts =
        std::max(0, engine.model_file.architecture_config.n_experts);
    const int first_dense = engine.model_file.architecture_config.first_dense;
    const int device_layers = std::min(
        std::max(0, glm5next_full_graph_device_expert_layers()),
        std::max(0, n_layers - first_dense));
    if (device_layers > 0) {
      engine.qwen_ram_vram_only_layers.assign((size_t)n_layers, 0);
      size_t vram_only_bytes = 0;
      int vram_only_layer_count = 0;
      for (int layer = first_dense; layer < first_dense + device_layers;
           ++layer) {
        if (is_cpu_affinity_layer(layer))
          continue;
        engine.qwen_ram_vram_only_layers[(size_t)layer] = 1;
        ++vram_only_layer_count;
        const size_t expert_bytes = engine.compute_expert_slab_bytes(layer);
        if (n_experts > 0 &&
            expert_bytes <= SIZE_MAX / (size_t)n_experts &&
            vram_only_bytes <=
                SIZE_MAX - expert_bytes * (size_t)n_experts)
          vram_only_bytes += expert_bytes * (size_t)n_experts;
      }
      LOG_STREAM_INFO()
          << "[MOE-OWNERSHIP] GLM5NEXT full-graph device block ["
          << first_dense << ", " << (first_dense + device_layers)
          << ") excludes " << vram_only_layer_count
          << " non-CPU-affinity layer(s) from persistent DDR preload and"
             " graph-RAM arenas (VRAM-only), approximately "
          << (double)vram_only_bytes / (1024.0 * 1024.0 * 1024.0)
          << " GiB\n";
    }
  }
  // Qwen CPU-affinity layers use the same fixed-stride host-RAM arena as
  // the full graph, but only when the configured RAM preload really owns
  // every expert in those layers. This gives the CPU MUL_MAT_ID graph a
  // stable DDR source without enabling the full-graph cache or changing
  // any other model family's cache policy.
  const bool qwen_cpu_ram_cache =
      !graph_mode && config.qwen4exp_cpu_ggml_moe &&
      engine.family_ops.family == ModelFamily::Qwen4Exp &&
      (!config.cpu_affinity_layers.empty() ||
       (config.qwen4exp_cpu_offload &&
        !config.qwen4exp_cpu_offload_layers.empty())) &&
      config.ram_preload_all_experts;
  const bool ram_graph_cache = graph_ram_cache || qwen_cpu_ram_cache;
  if (engine.family_ops.family == ModelFamily::Qwen4Exp &&
      config.ram_preload_all_experts) {
    LOG_STREAM_INFO()
        << "[CONFIG] Qwen MoE source contract: use the already-preloaded "
           "DDR RAM tier directly; file-backed GGUF expert mmap fallback "
           "is disabled. The PLE table may still be mmap-backed, but MoE "
           "expert planes are not.\n";
  }
  if (engine.mtp_enabled && engine.family_ops.family == ModelFamily::GLM5NEXT &&
      glm5next_full_graph_enabled()) {
    LOG_STREAM_INFO()
        << "[MTP] GLM NextN/MTP is enabled alongside the GLM full graph: "
           "the main generation loop's prefill uses the full graph, "
           "MTP's own draft/verification pass continues to run entirely "
           "on the legacy path (see mtp_speculative.cpp's target_forward, "
           "which never calls the full graph).\n";
  }
  if (!graph_mode) {
    gpu.setup_expert_cache(expert_slab_bytes, cfg.n_experts, config,
                           expert_layer_slab_bytes);
  } else {
    // 2026-09-09: two real, measured incidents, both from this same
    // decision point.
    //
    // Incident A (why this can never be "skip the cache entirely"): most
    // family full graphs only handle S>1 prefill. GLM5NEXT now also has a
    // reusable S=1 decode graph, but this legacy cache remains the safe
    // fallback if either graph is declined. The old code skipped the legacy
    // cache completely whenever MOESTREAM_FULL_GRAPH_USE_RAM_CACHE was unset,
    // which for
    // qwen4exp/deepseek_v4 (neither of which implement the RAM-arena
    // dedup that env var was designed to gate -- that's GLM5NEXT-only)
    // meant EVERY graph_mode run had zero decode-time GPU cache. Measured
    // on real hardware: decode collapsed to 0.13 tok/s, expert_calls=
    // 0GPU/8160CPU, "CPU fallback cause: 8160 upload-failed" -- there was
    // no cache slot to upload into at all.
    //
    // Incident B (why the cache size can't just be a flat 4GB either):
    // setting MOESTREAM_FULL_GRAPH_USE_RAM_CACHE (the old workaround for
    // Incident A) reinstated a flat 4GB fallback cache regardless of how
    // much VRAM a graph's own device-resident expert arena
    // (device_expert_layers) already committed. At device_expert_layers=
    // 20 on a 24GB card (16.11 GiB device expert arena + 2.92 GiB dense
    // graph arena), the extra flat 4GB starved the graph's own later
    // ggml_gallocr_reserve() call of its last ~1.8GB, which ggml does not
    // handle gracefully -- it aborted the whole process (SIGSEGV).
    //
    // Fix: size the expert cache against REAL current free VRAM on the
    // device the graph will use, minus this family's graph-owned device
    // expert-layer estimate (same expert_slab_bytes/n_experts this cache
    // itself is sized from) and a flat reserve for the
    // graph's dense arena + compute workspace (measured 2.92+1.80 GiB
    // above; kept generous since neither scales down with
    // device_expert_layers). Floored at 0 (never negative) -- a 0-sized
    // cache still means every decode step pays the CPU-fallback cost,
    // but at least never crashes the process fighting the graph for the
    // same VRAM.
    //
    // 2026-09-28: the original fix additionally clamped this to a flat
    // 4GB ceiling ("capped at the original 4GB ceiling, kept generous"),
    // reusing Incident B's pre-fix number as a known-safe value rather
    // than trusting the now-headroom-aware math on its own. That ceiling
    // is vestigial, not a safety requirement: Incident B's real cause was
    // an UNCONDITIONAL flat cache stacked on top of a large already-
    // committed device_expert_layers arena, and headroom_gb below already
    // subtracts that same commitment (plus a 6GB reserve) before this
    // point -- at device_expert_layers=20, the incident's own config,
    // headroom_gb is already small and no fixed ceiling is needed to keep
    // it safe. At low/zero device_expert_layers (e.g. the "dense split"
    // topology, GRAPH_PRELOAD_LAYERS=0), the flat ceiling was the ONLY
    // thing capping this cache -- confirmed on real hardware: it silently
    // discarded ~12 GiB of real, already-computed-safe headroom on a
    // 23.3 GiB-free card (only ~50% VRAM used), while the configured
    // moe_cache_gb (11.5 GiB) sat unreachable behind a stale 4GB cap.
    // Removed; headroom_gb's own subtraction is the real budget now.
    constexpr double kGraphOverheadReserveGb = 6.0;
    constexpr double kBytesPerGb = 1024.0 * 1024.0 * 1024.0;

    const bool glm_shared_expert_cache =
        engine.family_ops.family == ModelFamily::GLM5NEXT &&
        glm5next_full_graph_enabled();
    // GLM's graph aliases the existing cache, so it owns no additional
    // device-resident expert block to reserve ahead of cache sizing.
    const int dev_layers = !glm_shared_expert_cache &&
                                   graph_ops.device_expert_layers
                               ? graph_ops.device_expert_layers()
                               : 0;

    double est_device_expert_gb = 0.0;
    if (dev_layers > 0 && expert_slab_bytes > 0)
      est_device_expert_gb = (double)expert_slab_bytes * (double)cfg.n_experts *
                             (double)dev_layers / kBytesPerGb;

    // Generalized GPU staging (ggml_qwen4exp_graph.cpp's build_gpu_stage_node):
    // a fixed, topk-sized (not n_experts-sized) compact gate+up+down
    // destination per staged host-resident decode layer, bump-allocated
    // from the SAME device_expert_buffer arena the block above estimates --
    // this is a rough pre-estimate only (the real accounting, which
    // correctly excludes layers already covered by dev_layers, lives in
    // device_expert_bytes()); folded in here so this early fallback-cache
    // sizing pass doesn't leave the legacy cache too large and get
    // squeezed out later by the real graph's own allocation, the same
    // class of bug Incident B (this file's own comment above) already
    // documents.
    double est_gpu_stage_gb = 0.0;
    if (engine.qwen4exp_gpu_stage_enabled && expert_slab_bytes > 0 &&
        cfg.topk > 0 && !engine.qwen4exp_gpu_stage_layers.empty())
      est_gpu_stage_gb = (double)expert_slab_bytes * (double)cfg.topk *
                         (double)engine.qwen4exp_gpu_stage_layers.size() /
                         kBytesPerGb;

    // Option B's static hot/cold split (plan_expert_vram_split() in
    // ggml_qwen4exp_graph.cpp) commits its own device-resident expert
    // arena later, using the SAME usage ranking/right-sizing this early
    // pass replicates below -- it was never subtracted from this cache's
    // headroom before. 2026-09-29: removing this file's old flat 4GB
    // fallback-cache ceiling (see Incident B's follow-up comment above)
    // exposed that gap: with device_expert_layers=0 the legacy cache grew
    // to its full configured moe_cache_gb (11.5 GiB) instead of being
    // capped, leaving only ~5.8 GiB free where Option B's committed hot
    // set needed ~12.6 GiB -- a mid-decode cudaMalloc OOM that corrupted
    // output (ggml-cuda's own allocator error prints straight to stdout,
    // bypassing LOG_ERROR, and leaked into the token stream). Reproduced
    // identically on both IQ1_M and Q2_0, so it's this budget gap, not a
    // quant-type issue.
    //
    // Mirrors plan_expert_vram_split()'s own per-layer ranking, coverage
    // check, and qwen4exp_option_b_right_size_top_n shrink (using the same
    // qwen4exp_graph_pin_file usage snapshot, already assigned above) so
    // this estimate matches what that function will actually commit,
    // rather than a flat top_n-for-every-selected-layer upper bound (which
    // overshoots badly: several selected layers never clear
    // qwen4exp_option_b_min_coverage and commit nothing). A first attempt
    // at this fix used that flat bound and it starved the legacy cache to
    // 0 GiB, which hit a separate, pre-existing "0 GiB legacy cache" bug
    // (a CUDA illegal-memory-access abort) -- an accurate estimate avoids
    // exercising that rare path at all. Concentrated-only auto-selection
    // still can't be estimated here (the candidate layer list itself comes
    // from this same usage pass) and contributes 0, same as before this
    // fix.
    double est_option_b_gb = 0.0;
    if (engine.family_ops.family == ModelFamily::Qwen4Exp &&
        config.qwen4exp_option_b_top_n > 0 &&
        (config.qwen4exp_option_b_layer >= 0 ||
         !config.qwen4exp_option_b_selected_layers.empty())) {
      std::vector<int> option_b_layers;
      if (config.qwen4exp_option_b_layer >= 0)
        option_b_layers.push_back(config.qwen4exp_option_b_layer);
      else
        option_b_layers = config.qwen4exp_option_b_selected_layers;
      auto [usage_entries, usage_maxc] =
          StreamingEngine::load_usage(engine.qwen4exp_graph_pin_file);
      (void)usage_maxc;
      if (!usage_entries.empty() && cfg.n_experts > 0) {
        size_t option_b_bytes = 0;
        // Global pool mode: estimate from the same selection the planner will
        // commit (option_b_global::select), not from a per-layer top_n.
        const bool est_global = config.qwen4exp_option_b_global_top_k > 0;
        if (est_global) {
          std::vector<int> live_layers;
          for (int layer : option_b_layers) {
            if (layer < 0 || layer >= cfg.n_layers ||
                is_cpu_affinity_layer(layer))
              continue;
            if (dev_layers > 0) {
              const bool in_baseline_block =
                  qwen4exp_full_graph_device_expert_trailing()
                      ? layer >= cfg.n_layers - dev_layers
                      : layer < dev_layers;
              if (in_baseline_block)
                continue;
            }
            live_layers.push_back(layer);
          }
          const auto sel = option_b_global::select(
              usage_entries, live_layers, cfg.n_layers, cfg.n_experts,
              config.qwen4exp_option_b_global_top_k,
              config.qwen4exp_option_b_global_floor_per_layer,
              config.qwen4exp_option_b_top_n);
          for (const auto &[layer, ids] : sel.hot) {
            const size_t expert_bytes = engine.compute_expert_slab_bytes(layer);
            if (expert_bytes > 0 && ids.size() <= SIZE_MAX / expert_bytes)
              option_b_bytes += expert_bytes * ids.size();
          }
        }
        for (int layer : est_global ? std::vector<int>{} : option_b_layers) {
          if (layer < 0 || layer >= cfg.n_layers ||
              is_cpu_affinity_layer(layer))
            continue;
          if (dev_layers > 0) {
            const bool in_baseline_block =
                qwen4exp_full_graph_device_expert_trailing()
                    ? layer >= cfg.n_layers - dev_layers
                    : layer < dev_layers;
            if (in_baseline_block)
              continue;
          }
          std::vector<uint64_t> counts(static_cast<size_t>(cfg.n_experts), 0);
          for (const auto &e : usage_entries) {
            if (e.layer == layer && e.expert >= 0 && e.expert < cfg.n_experts)
              counts[static_cast<size_t>(e.expert)] += e.count;
          }
          uint64_t total_count = 0;
          for (uint64_t count : counts)
            total_count += count;
          if (total_count == 0)
            continue;
          std::vector<int> ranked(static_cast<size_t>(cfg.n_experts));
          for (int expert = 0; expert < cfg.n_experts; ++expert)
            ranked[static_cast<size_t>(expert)] = expert;
          std::stable_sort(ranked.begin(), ranked.end(), [&](int a, int b) {
            if (counts[static_cast<size_t>(a)] != counts[static_cast<size_t>(b)])
              return counts[static_cast<size_t>(a)] > counts[static_cast<size_t>(b)];
            return a < b;
          });
          int hot_count = std::min(config.qwen4exp_option_b_top_n, cfg.n_experts);
          if (config.qwen4exp_option_b_right_size_top_n) {
            uint64_t cum = 0;
            for (int k = 0; k < hot_count; ++k) {
              cum += counts[static_cast<size_t>(ranked[static_cast<size_t>(k)])];
              if (static_cast<double>(cum) / static_cast<double>(total_count) >=
                  config.qwen4exp_option_b_min_coverage) {
                hot_count = k + 1;
                break;
              }
            }
          }
          uint64_t hot_route_count = 0;
          for (int k = 0; k < hot_count; ++k)
            hot_route_count +=
                counts[static_cast<size_t>(ranked[static_cast<size_t>(k)])];
          const double coverage = static_cast<double>(hot_route_count) /
                                  static_cast<double>(total_count);
          if (coverage < config.qwen4exp_option_b_min_coverage)
            continue;
          const size_t expert_bytes = engine.compute_expert_slab_bytes(layer);
          if (expert_bytes > 0 &&
              (size_t)hot_count <= SIZE_MAX / expert_bytes)
            option_b_bytes += expert_bytes * (size_t)hot_count;
        }
        est_option_b_gb = (double)option_b_bytes / kBytesPerGb;
      }
    }

    // The GLM graph aliases the existing VRAM cache for expert weights; its
    // dense graph and scheduler workspace are covered by the generic reserve.
    const double graph_overhead_reserve_gb = kGraphOverheadReserveGb;

    // Ground-truth free VRAM right now, same GPU setup_expert_cache is
    // about to target. Single-GPU-vault configs only (the graphs'
    // current real deployment target) -- core_free_vram_bytes() reads
    // the "core" device, which is also the vault/graph device in every
    // config these families ship with today.
    const double free_gb = (double)gpu.core_free_vram_bytes() / kBytesPerGb;
    const double headroom_gb = free_gb - est_device_expert_gb -
                               est_option_b_gb -
                               est_gpu_stage_gb - graph_overhead_reserve_gb;
    const double fallback_gb = std::max(headroom_gb, 0.0);

    // Design policy (2026-10-05): Qwen4Exp has no layered decode. With the
    // full-graph decode enabled the graph reads only its own device arena
    // (Option-B hot experts / resident layers) and computes the rest on the
    // CPU from the DDR arenas, so the legacy VRAM expert cache (pinned + LRU
    // tiers) and the active-layer area are never filled or consulted: a run
    // measured 0 / 4650 slots filled and 0 hits while reserving 6.4 GiB.
    // Disable them whenever the Qwen4Exp decode graph is requested, except
    // when its selected-layer EXL3 custom op explicitly borrows these slots.
    // The explicit qwen4exp_option_b_disable_legacy_cache key stays honoured
    // for runs that still use the older prefill-only graph.
    const bool qwen_full_graph_decode =
        engine.family_ops.family == ModelFamily::Qwen4Exp &&
        qwen4exp_full_graph_decode_enabled();
    const bool exl3_ggml_bridge_enabled =
        !engine.exl3_ggml_bridge_model_path.empty();
    const bool option_b_disable_legacy_cache =
        (qwen_full_graph_decode && !exl3_ggml_bridge_enabled) ||
        (config.qwen4exp_option_b_disable_legacy_cache &&
         config.qwen4exp_option_b_top_n > 0 &&
         engine.family_ops.family == ModelFamily::Qwen4Exp);
    EngineConfig fallback_config = config;
    if (option_b_disable_legacy_cache) {
      for (auto &gc : fallback_config.gpus) {
        if (gc.device_id == gpu.get_core_device())
          gc.moe_cache_gb = 0.0;
      }
      fallback_config.use_active_layer_area = false;
      LOG_INFO("[GPU] Qwen4Exp full graph: legacy VRAM expert cache and "
               "active-layer area disabled (no layered decode); experts "
               "come from the static graph arena or the DDR arenas\n");
    }
    for (auto &gc : fallback_config.gpus) {
      if (gc.moe_cache_gb > fallback_gb)
        gc.moe_cache_gb = fallback_gb;
    }
    if (engine.family_ops.family == ModelFamily::GLM5NEXT &&
        glm5next_full_graph_enabled()) {
      // The graph aliases pinned entries in the main cache; the temporary
      // active-layer pool is outside that address space and competes with
      // the shared MoE cache for VRAM.
      fallback_config.use_active_layer_area = false;
      LOG_STREAM_INFO()
          << "[GPU] Active-layer temporary area disabled for the GLM graph; "
             "hot experts alias the main VRAM cache\n";
    }
    LOG_STREAM_INFO()
        << "[GPU] VRAM expert cache capacity: capped at <=" << fallback_gb
        << " GiB/GPU ("
        << (glm_shared_expert_cache ? "shared with GLM graph; " : "")
        << "sized from free="
        << free_gb
        << " GiB - graph_expert_layers(" << dev_layers << ")~="
        << est_device_expert_gb
        << " GiB - gpu_stage_layers(" << engine.qwen4exp_gpu_stage_layers.size()
        << ")~=" << est_gpu_stage_gb << " GiB - option_b~=" << est_option_b_gb
        << " GiB - "
        << graph_overhead_reserve_gb << " GiB graph overhead reserve"
        << " -- see this call site's "
           "own comment for the two real incidents this guards against)\n";
    const bool option_b_stage_enabled =
        config.qwen4exp_option_b_stage_expert >= 0;
    if (fallback_gb <= 0.0 && !option_b_stage_enabled) {
      // This graph-only fallback is allowed to have no legacy cache.  Keep
      // the temporary active-layer pool off too; otherwise it would consume
      // another ~0.2 GB after the explicit zero-cache decision and could
      // still starve the resident graph allocator.  This is local to the
      // graph fallback config; normal model/cache policy is unchanged.
      fallback_config.use_active_layer_area = false;
      LOG_WARN(
          "[GPU] Expert cache capacity is 0 GiB -- graph allocations leave "
          "no VRAM headroom after the %.1f GiB overhead reserve; graph "
          "expert references will use DDR, and legacy decode may use CPU if "
          "the reusable graph declines. Lower graph residency if that "
          "fallback matters for this "
          "run.\n",
          graph_overhead_reserve_gb);
    } else if (fallback_gb <= 0.0 && option_b_stage_enabled) {
      // The Layer-13 proof needs one slot per concurrently staged expert.
      // These are small active-layer slabs, not the legacy cache, and fit in
      // the residual headroom even when the generic graph fallback
      // intentionally clamps that cache to zero.
      fallback_config.use_active_layer_area = true;
      fallback_config.active_layer_capacity = std::max(
          1, std::max(fallback_config.active_layer_capacity,
                      config.qwen4exp_option_b_stage_count));
      LOG_INFO("[GPU] Retaining %d active-layer slot(s) for the explicit "
               "Layer-13 staging proof despite zero legacy-cache fallback\n",
               fallback_config.active_layer_capacity);
    }
    gpu.setup_expert_cache(expert_slab_bytes, cfg.n_experts, fallback_config,
                           expert_layer_slab_bytes,
                           fallback_gb <= 0.0 ||
                               option_b_disable_legacy_cache);
  }
  gpu.report_vram_usage("after expert cache setup");

  // Connect VRAM expert cache to the RAM manager for LRU integration
  ram_mgr.vram_cache = &gpu.topology;
  // NOTE: the reverse pointer (RAMSafetyManager -> LockFreePrefetcher) no
  // longer exists -- under the unified claim/publish design the
  // prefetcher calls INTO RAMSafetyManager::claim_or_wait, not the other
  // way around, so ram_mgr itself never needs to reach the prefetcher.
  // See prefetcher.ram_mgr's own assignment further down (after the Tier-3
  // RAM preload block) for the one remaining pointer, in the opposite
  // direction.

  // Offload dense tensors to core GPU VRAM — via a single pool allocation
  // (see CudaManager::init_dense_pool/offload_tensor_to_pool) rather than
  // one cudaMalloc PER tensor. With ~1182 separate tensors of wildly
  // varying sizes, individual allocations fragment a near-full device:
  // the allocator can fail to place a tensor even when total free bytes
  // are technically sufficient, because no single gap is large enough.
  // One pool + offset suballocation has no such gaps — a tensor either
  // fits in the remaining linear space or it doesn't, with zero waste
  // between tensors beyond 256-byte alignment padding.
  LOG_STREAM_INFO()
      << "[CUDA] Offloading persistent dense layers to core GPU (device "
      << gpu.get_core_device() << ")...\n";

  // Offload attention, norms, router, shared-expert, dense leading-layer
  // FFN, and embedding/output to VRAM. These are all "resident" (always-
  // used, never streamed) tensors — unlike routed experts they should
  // never live host-only. The shared-expert weights (ffn_*_shexp — one
  // always-active FFN block PER LAYER, run on every token) and the
  // leading dense layers' plain ffn_gate/up/down (l < first_dense) were
  // previously missing from this filter entirely: they matched none of
  // the substrings below, so dense_proj's GPU dispatch check
  // (`W.device_ptr`) always failed for them and every token paid the
  // full CPU dequant-matmul cost for every layer's shared expert.
  // HC/GDN GGML runners borrow their weight pointers from this shared dense
  // pool, avoiding a second private CUDA copy. Host copies remain in
  // resident_layers for registration and CPU fallback.
  auto is_dense_offload_candidate = [&](const std::string &name) {
    // attn_k_b/attn_v_b are dequantized ONCE, CPU-side, straight from
    // host_ptr into the MLA absorption cache (wuk_cache/wuv_cache,
    // attention_layer.cpp's mla_attention_absorbed_gguf) -- dense_proj
    // (and therefore device_ptr) is NEVER called for these two specific
    // tensors. A VRAM-resident copy of them would be pure waste: bytes
    // uploaded, never read from GPU by anything. Excluded explicitly
    // since they'd otherwise match the broad "attn" substring below --
    // found while investigating what the "other_attention" VRAM bucket
    // (CLAUDE.md's dense-pool-composition breakdown) actually contains.
    //
    // attn_output_a (DeepSeek-V4's grouped output projection, first
    // stage) gets EXACTLY this same treatment in dsv4_attention_gguf
    // (attention_layer.cpp): dequant_row'd ONCE into a static per-layer
    // cache (woa_cache) and consumed via raw CPU matmul() per group,
    // never through dense_proj/device_ptr -- excluded for the identical
    // reason as attn_k_b/attn_v_b. attn_output_b (the second, single
    // dense matmul stage) is NOT excluded -- it genuinely goes through
    // dense_proj, see is_attn_output below.
    if (name.find("attn_k_b.weight") != std::string::npos ||
        name.find("attn_v_b.weight") != std::string::npos ||
        name.find("attn_output_a.weight") != std::string::npos) {
      return false;
    }
    // hc_ catches DeepSeek-V4's Hyper-Connections weights: the two
    // per-layer mixing matmuls (hc_attn_fn/hc_ffn_fn -- these DO route
    // through dense_proj, hyper_connections.cpp's hc_pre_collapse) plus
    // their small bias/scale companions (hc_{attn,ffn}_{base,scale},
    // read via a raw host_ptr lookup, no GPU dispatch -- including them
    // here is harmless over-inclusion, not a correctness issue, given
    // their negligible size) and the global output_hc_{fn,base,scale}
    // (hc_collapse_output). All contain the literal substring "hc_" and
    // nothing else in this codebase's tensor vocabulary does. Previously
    // missing entirely -- hc_attn_* happened to already match the loose
    // "attn" substring below, but hc_ffn_*/output_hc_* matched NOTHING,
    // silently dropping them from the dense VRAM total on a real
    // DeepSeek-V4 model (found auditing tools/gguf_plan.cpp, which
    // mirrors this exact function, against the newly-added tensor set).
    if (name.find("hc_") != std::string::npos)
      return true;
    // indexer.proj (DeepSeek-V4.1's Lightning Indexer projection) and
    // engram_q/engram_k/engram_wkv (its Engram gate/projection weights --
    // NOT engram_embd, the giant hash-embedding table that's deliberately
    // kept mmap'd on NVMe and never reaches resident_layers at all, see
    // deepseek41_engram.cpp) are real, per-token dense_proj targets that
    // were simply missing from this allowlist. Found investigating a real
    // ~10.7s/60-token-request CPU dequant-matmul cost that expert_calls'
    // own "0 CPU" count couldn't explain -- t_cpu_dequant/t_cpu_dot are
    // shared between the MoE CPU-fallback path and dense_proj's own,
    // perf_timer.hpp's own comment on that. Individually tiny tensors, but
    // engram_wkv fires per Engram layer per token and indexer.proj per
    // CSA/HCA layer per token, so the CPU-vs-GPU gap compounds across a
    // whole decode.
    if (name.find("indexer.proj.weight") != std::string::npos ||
        name.find("engram_wkv.weight") != std::string::npos ||
        name.find("engram_q.weight") != std::string::npos ||
        name.find("engram_k.weight") != std::string::npos)
      return true;
    // ssm_/ple_ catch Qwen4Exp's GDN recurrent-layer tensors (ssm_conv1d,
    // ssm_beta/alpha, ssm_out, ssm_norm, ...) and PLE layer tensors --
    // all resident, per-token dense_proj targets, same "resident and
    // always-used" rationale as the attention set. (attn_qkv/attn_gate
    // already match "attn".)
    if (name.find("ssm_") != std::string::npos ||
        name.find("ple_") != std::string::npos)
      return true;
    return name.find("attn") != std::string::npos ||
           name.find("norm") != std::string::npos ||
           name.find("gate_inp") != std::string::npos ||
           name.find("shexp") != std::string::npos ||
           name.find("ffn_gate.weight") != std::string::npos ||
           name.find("ffn_up.weight") != std::string::npos ||
           name.find("ffn_down.weight") != std::string::npos ||
           name.find("nextn.") != std::string::npos ||
           name == "token_embd.weight" || name == "output.weight" ||
           name == "output_norm.weight";
  };

  // HC/GDN tensors are included in this same pool. Their GGML runners
  // borrow the pool's device_ptrs through an external tensor marker, so
  // excluding them would only force a second private CUDA copy. The host
  // copies remain available for registration and CPU fallback.

  // attn_output fast-path: this weight is a STATIC per-layer matrix (same
  // every token, like attn_k_b/attn_v_b) that's large enough (~69 MB in
  // Q5_K for this project's GLM-5.2 target) to hit dense_matmul_host's
  // fused per-block dequant+matmul kernel, which has no GEMV specialization
  // and is disproportionately slow at S=1 decode — measured ~623ms/token
  // for this one weight alone, ~5x its similarly-staged qproj calls (see
  // CLAUDE.md). Fix: dequantize once to BF16 and use cuBLAS (matched-type
  // BF16xBF16 GEMM, dense_matmul_bf16_matched_host) instead, same pattern
  // as the attn_k_b/attn_v_b host-side cache. Unlike those (small,
  // low-rank MLA-only matrices), attn_output is full [n_heads*v_head,
  // hidden] — even BF16 (2 bytes/elem) is ~2-3x the Q5_K original size, so
  // applying this unconditionally to every layer could still be several GB
  // across a real model's full depth. So: convert greedily, budget-checked
  // against whatever's ACTUALLY left after reserving space for every OTHER
  // dense tensor (which already works today at its original quant size and
  // must not regress) — layers that don't fit just keep using the existing
  // (slower but correct) fused Q5_K/K-quant/IQ kernel path, exactly like
  // today.
  //
  // (An F32 version of this shipped first and worked, but covered fewer
  // layers per GB of budget than BF16 does. A first BF16 attempt used
  // MISMATCHED types (BF16 weight / F32 activation) and was reverted after
  // real NaN corruption; that corruption was later root-caused to an
  // unrelated concurrency race (now fixed) and, independently, Test 9
  // (main_test.cpp) confirmed on real hardware that cuBLAS flatly rejects
  // that mismatched-type combination anyway. This is the third attempt:
  // MATCHED types (BF16 weight AND BF16 activation), verified correct in
  // isolation via Test 10 before being wired in here — see CLAUDE.md.)
  auto is_attn_output = [](const std::string &name) {
    // attn_output_b (DeepSeek-V4's grouped output projection, SECOND
    // stage -- a single genuine dense_proj matmul, unlike attn_output_a
    // above) is the direct V4 analog of attn_output.weight here: same
    // "large per-layer static matrix, same kind of dense_proj call,
    // same reason the BF16 fast-path helps" reasoning applies
    // unchanged. attn_output_a is EXCLUDED from candidacy entirely
    // (is_dense_offload_candidate above), so it never reaches this
    // check regardless.
    return name.find("attn_output.weight") != std::string::npos ||
           name.find("attn_output_b.weight") != std::string::npos;
  };

  // gate_inp (MoE router) BF16 fast-path -- converted BEFORE the regular
  // dense-pool reservation below so its savings flow through to that
  // budget, same as if it had never needed regular-pool room at all.
  // Unlike attn_output, gate_inp is ALREADY F32 in this checkpoint (not a
  // slow quantized type) -- confirmed by its measured per-tensor size
  // exactly matching hidden*n_experts*4 bytes -- so F32 activations
  // already dispatch through fast, MATCHED-type cuBLAS SGEMM today; this
  // buys VRAM (BF16 is half F32), not speed. No greedy budget math here
  // like attn_output's (gate_inp's total is small, ~0.47 GB for this
  // checkpoint) -- just try each one against real free VRAM directly.
  //
  // PRECISION NOTE, read before re-enabling if this is ever reverted:
  // gate_inp IS the MoE router's own weight -- it produces the logits
  // top-k expert selection is based on. Rounding it to BF16 doesn't just
  // add generic numerical noise the way it does for a dense OUTPUT
  // projection (e.g. attn_output); a rounded logit can change which
  // experts survive top-k at close margins -- a real, if narrow, ROUTING
  // behavior change, not just precision loss on a value nothing else
  // depends on structurally. Done here only after this tradeoff was
  // surfaced and the user explicitly asked to proceed anyway -- see
  // CLAUDE.md. If generation quality regresses in a way attn_output's
  // fast-path never did, this is the first thing to suspect and revert.
  int gate_inp_converted = 0, gate_inp_skipped = 0;
  int gate_inp_scheduler_kept = 0;
  size_t gate_inp_bf16_bytes_used = 0, gate_inp_f32_equivalent_bytes = 0;
  const bool disable_router_bf16 = [] {
    const char *value = std::getenv("MOESTREAM_DISABLE_ROUTER_BF16");
    return value && value[0] == '1';
  }();
  // llama.cpp's CPU-MoE topology keeps the router projection on CUDA and moves only
  // the routed expert planes to the CPU.  The Qwen decode scheduler consumes
  // the original device_ptr representation; it cannot use the separate
  // gpu_fast_ptr BF16 representation without changing router math/type
  // contracts.  Keep these small Qwen router tensors in the ordinary dense
  // pool when that explicitly gated scheduler is enabled.  This is deliberately
  // scoped to Qwen + CPU-affinity layers; all other models and the normal
  // Qwen path retain the existing BF16 optimization.
  auto qwen_scheduler_cpu_router = [&](const std::string &name) {
    if (!config.qwen4exp_decode_scheduler ||
        engine.family_ops.family != ModelFamily::Qwen4Exp ||
        name.rfind("blk.", 0) != 0 ||
        name.find("gate_inp.weight") == std::string::npos)
      return false;
    char *end = nullptr;
    const long layer = std::strtol(name.c_str() + 4, &end, 10);
    return end != name.c_str() + 4 && end && *end == '.' && layer >= 0 &&
           layer < engine.model_file.architecture_config.n_layers &&
           is_cpu_affinity_layer((int)layer);
  };
  for (auto &[name, tensor] : engine.resident_layers) {
    if (!tensor.host_ptr || name.find("gate_inp.weight") == std::string::npos)
      continue;
    if (qwen_scheduler_cpu_router(name)) {
      ++gate_inp_scheduler_kept;
      continue;
    }
    if (disable_router_bf16) {
      ++gate_inp_skipped;
      continue;
    }
    size_t remaining_free_vram = gpu.core_free_vram_bytes();
    if (gpu.upload_dequantized_bf16(tensor, remaining_free_vram)) {
      ++gate_inp_converted;
      gate_inp_bf16_bytes_used += (size_t)tensor.meta.shape[0] *
                                  (size_t)tensor.meta.shape[1] *
                                  sizeof(uint16_t);
      gate_inp_f32_equivalent_bytes += (size_t)tensor.meta.shape[0] *
                                       (size_t)tensor.meta.shape[1] *
                                       sizeof(float);
    } else {
      ++gate_inp_skipped;
    }
  }
  LOG_STREAM_INFO()
      << "[CUDA] gate_inp (MoE router) BF16 fast-path: " << gate_inp_converted
      << " layer(s) converted (" << gate_inp_bf16_bytes_used / 1e9 << " GB, vs "
      << gate_inp_f32_equivalent_bytes / 1e9 << " GB at original F32 -- "
      << (gate_inp_f32_equivalent_bytes - gate_inp_bf16_bytes_used) / 1e9
      << " GB saved), " << gate_inp_skipped << " layer(s) kept at original F32"
      << ", " << gate_inp_scheduler_kept
      << " Qwen scheduler router(s) deliberately kept at original F32"
      << (disable_router_bf16 ? ", router BF16 conversion disabled by env" : "")
      << " -- NOTE: rounds router logits, a routing-behavior change, not just "
         "numerical noise.\n";

  constexpr size_t POOL_ALIGN = 256;
  std::vector<QuantizedTensor *> attn_output_candidates;
  size_t non_attn_output_pool_bytes = 0;
  // Worst-case reservation: what the regular dense pool would need if NONE
  // of attn_output converts (every candidate stays at its original
  // quantized size) -- computed with the SAME aligned-accumulation as
  // non_attn_output_pool_bytes / the real pool_bytes_needed below, so this
  // reservation and the real pool sizing agree on what "fits" means.
  size_t attn_output_quantized_total = 0;
  for (auto &[name, tensor] : engine.resident_layers) {
    if (!tensor.host_ptr || !is_dense_offload_candidate(name))
      continue;
    if (tensor.gpu_fast_ptr)
      continue; // gate_inp already fast-path converted above -- needs no
                // regular-pool room
    if (is_attn_output(name)) {
      attn_output_candidates.push_back(&tensor);
      attn_output_quantized_total =
          (attn_output_quantized_total + POOL_ALIGN - 1) & ~(POOL_ALIGN - 1);
      attn_output_quantized_total += tensor.meta.size_bytes;
      continue;
    }
    non_attn_output_pool_bytes =
        (non_attn_output_pool_bytes + POOL_ALIGN - 1) & ~(POOL_ALIGN - 1);
    non_attn_output_pool_bytes += tensor.meta.size_bytes;
  }

  // BUG FIXED: this budget used to be sized against non_attn_output_pool_bytes
  // ALONE -- i.e. it implicitly assumed every attn_output layer would end up
  // converted (needing zero room in the regular pool), when in practice the
  // greedy loop below only converts as many as fit. The ones left behind
  // still need their full quantized bytes in the regular dense pool later,
  // but that need was never reserved for here -- so the regular pool
  // allocation could (and on a real run, did: "requested 13.58 GB" against
  // ~10GB actually free) come up short, spilling hundreds of tensors to the
  // CPU fallback. Fixed by reserving the WORST CASE (nothing converts) up
  // front, then spending only each conversion's NET extra cost (F32 bytes
  // minus the quantized bytes it frees from that worst-case reservation) --
  // see the loop below. This guarantees pool_bytes_needed (computed later)
  // plus every fast-path F32 allocation together can never exceed
  // free_before_fast_path - FAST_PATH_SAFETY_MARGIN, regardless of how many
  // layers convert.
  // Reserve room for a same-device dual-role CoreEngine expert-cache slice
  // (add_dual_role_expert_cache, further below) BEFORE computing the
  // attn_output BF16 budget. That slice is deliberately sized from
  // GROUND-TRUTH free VRAM measured AFTER this fast-path loop and the
  // regular dense pool have already run (see its own comment) -- which
  // means, without this reservation, BF16 (itself ground-truth-free-VRAM
  // greedy by design) claims 100% of whatever headroom exists and the
  // slice gets whatever rounding error is left over, regardless of its
  // configured moe_cache_gb value. Confirmed on a real run: 3090 as
  // dense+moe with moe_cache_gb=8.5-9.5 requested, only 0.39 GB actually
  // allocated, because BF16 converted all 78 attn_output layers (15.7 GB)
  // first. This is a target reservation, not a hard guarantee --
  // add_dual_role_expert_cache still re-checks ground-truth free VRAM
  // itself later and may end up with less if something else also grew in
  // the meantime, but BF16 can no longer be the reason it gets nothing.
  // Only reserve for the LATER add_dual_role_expert_cache path, which
  // applies solely to a CoreEngine-only device (role="dense" with a
  // supplementary moe_cache_gb slice). A genuine single-GPU role=
  // "dense+moe" device is already in vault_gpus by this point (assign_roles
  // put it there) and already had its FULL moe_cache_gb cudaMalloc'd by the
  // main allocate_expert_cache() pass before this function even started --
  // that's already subtracted out of core_free_vram_bytes() below, so
  // reserving for it AGAIN here would double-count and needlessly shrink
  // BF16's real budget.
  bool already_primary_vault_for_reservation =
      std::find(gpu.topology.vault_gpus.begin(), gpu.topology.vault_gpus.end(),
                gpu.topology.core_gpu) != gpu.topology.vault_gpus.end();
  double core_moe_cache_gb_for_reservation = 0.0;
  if (!already_primary_vault_for_reservation) {
    for (auto &gc : config.gpus) {
      if (gc.device_id == gpu.device_id) {
        core_moe_cache_gb_for_reservation = gc.moe_cache_gb;
        break;
      }
    }
  }
  size_t dual_role_slice_reservation =
      (size_t)(core_moe_cache_gb_for_reservation * 1e9);

  size_t worst_case_pool_bytes =
      non_attn_output_pool_bytes + attn_output_quantized_total;

  // Real-vs-configured dense_gb sanity check. worst_case_pool_bytes above
  // is the SAME byte-for-byte total this run's own dense pool is about to
  // request (non_attn_output_pool_bytes + attn_output at native quant
  // size) -- ground truth for THIS exact model/quant-mix, not an
  // estimate. dense_gb itself is never actually enforced as a cap
  // anywhere (confirmed by grep: only read by engine_config.hpp's
  // auto-sizing branch, which doesn't run when dense_gb is set
  // explicitly, as it is in every real config this project uses) -- so a
  // stale/wrong value doesn't break anything structurally, but it DOES
  // silently misrepresent how much of the card's VRAM budget is actually
  // spoken for, which is exactly the kind of "reasoned from a number that
  // was never true" mistake this session's own GPU1-VRAM investigation
  // kept tripping over. Matches the same fixed tools/gguf_plan.cpp
  // categorization (NextN/MTP-excluded, DSA-indexer-excluded) so a
  // pre-flight `gguf_tool plan` run and this warning agree.
  for (auto &gc : config.gpus) {
    if (gc.device_id != gpu.device_id || gc.dense_gb <= 0)
      continue;
    double real_gb = worst_case_pool_bytes / 1e9;
    double configured_gb = gc.dense_gb;
    double diff_gb = std::abs(real_gb - configured_gb);
    double diff_pct =
        (configured_gb > 0) ? 100.0 * diff_gb / configured_gb : 0.0;
    // Combined threshold, not percentage alone: on a small (16GB-class)
    // card a "modest" 5-8% miss is still ~1GB of real, unaccounted-for
    // VRAM budget -- exactly the size of gap that mattered throughout
    // this session's own GPU1 investigation. Fires on EITHER a real
    // >=5% relative miss or a >=0.5GB absolute one, so it still catches
    // a small-percentage-but-large-absolute miss on a big card too.
    if (diff_pct >= 5.0 || diff_gb >= 0.5) {
      LOG_STREAM_WARN()
          << "[CONFIG] dense_gb=" << configured_gb << " GB (device "
          << gc.device_id << ") looks stale -- this "
          << "model's REAL dense-tensor requirement (native quant, "
          << "nothing BF16-converted, gate_inp excluded since it's "
          << "budgeted/converted separately) is " << real_gb << " GB, "
          << diff_pct << "% off. This isn't a hard cap "
          << "(the pool allocates against real free VRAM regardless "
          << "of this value), but it misrepresents this card's real "
          << "budget for any headroom math you do by hand. Suggested: "
          << "dense_gb = " << (real_gb + 0.5)
          << " (real requirement + ~0.5GB margin). Cross-check with "
          << "`gguf_tool plan <model>` before a real run.\n";
    }
  }

  constexpr size_t FAST_PATH_SAFETY_MARGIN =
      (size_t)1
      << 30; // 1 GB, matches the dense-pool backoff's own margin style
  size_t free_before_fast_path = gpu.core_free_vram_bytes();
  size_t fast_path_reserved = worst_case_pool_bytes + FAST_PATH_SAFETY_MARGIN +
                              dual_role_slice_reservation;
  size_t fast_path_budget = (free_before_fast_path > fast_path_reserved)
                                ? (free_before_fast_path - fast_path_reserved)
                                : 0;

  int fast_converted = 0, fast_skipped = 0;
  int fast_skipped_budget = 0, fast_skipped_upload = 0;
  size_t fast_bytes_used =
      0; // gross BF16 bytes actually allocated (for the log line only)
  size_t budget_spent = 0; // NET budget consumed: sum of (bf16_bytes -
                           // quant_bytes) over converted layers
  // Tracks what the SAME converted layers would have cost under the old
  // F32 fast-path -- comparison-only bookkeeping, doesn't affect which
  // layers convert or how much budget is spent. Exists purely to make
  // BF16's actual saving directly visible in the log line below instead
  // of requiring hand arithmetic against a prior run's numbers (see
  // CLAUDE.md -- this was confusing enough in practice to need its own
  // explicit log line).
  size_t f32_equivalent_bytes = 0;
  for (QuantizedTensor *qt : attn_output_candidates) {
    size_t quant_bytes = qt->meta.size_bytes;
    size_t bf16_bytes_estimate = (size_t)qt->meta.shape[0] *
                                 (size_t)qt->meta.shape[1] * sizeof(uint16_t);
    size_t net_cost = (bf16_bytes_estimate > quant_bytes)
                          ? (bf16_bytes_estimate - quant_bytes)
                          : 0;
    size_t remaining_net_budget = (fast_path_budget > budget_spent)
                                      ? (fast_path_budget - budget_spent)
                                      : 0;
    if (net_cost > remaining_net_budget) {
      ++fast_skipped;
      ++fast_skipped_budget;
      continue;
    }
    // upload_dequantized_bf16 itself checks the GROSS bf16 size against
    // whatever's passed here -- pass the real remaining free VRAM (ground
    // truth), not the net-budget figure, since the two are different
    // units (net_cost already accounts for the pool-side saving that
    // ground-truth free VRAM doesn't know about).
    size_t remaining_free_vram = gpu.core_free_vram_bytes();
    if (gpu.upload_dequantized_bf16(*qt, remaining_free_vram)) {
      ++fast_converted;
      fast_bytes_used += bf16_bytes_estimate;
      budget_spent += net_cost;
      f32_equivalent_bytes +=
          (size_t)qt->meta.shape[0] * (size_t)qt->meta.shape[1] * sizeof(float);
    } else {
      ++fast_skipped;
      ++fast_skipped_upload;
    }
  }

  LOG_STREAM_INFO()
      << "[CUDA] attn_output fast-path (BF16 dequant + matched-type cuBLAS): "
      << fast_converted << " layer(s) converted (" << fast_bytes_used / 1e9
      << " GB BF16, vs " << f32_equivalent_bytes / 1e9
      << " GB if these same layers had used the old F32 fast-path -- "
      << (f32_equivalent_bytes - fast_bytes_used) / 1e9 << " GB saved), "
      << fast_skipped << " layer(s) kept on the slower fused kernel"
      << " [" << fast_skipped_budget << " skipped for lack of budget, "
      << fast_skipped_upload << " rejected by upload_dequantized_bf16 itself]"
      << " (net budget was " << fast_path_budget / 1e9
      << " GB, worst-case pool reservation " << worst_case_pool_bytes / 1e9
      << " GB).\n";
  // 2026-09-01: the two skip reasons used to be indistinguishable, which
  // is actively misleading -- the budget figure is printed either way, so a
  // run where EVERY layer was rejected by upload_dequantized_bf16 (which
  // returns false immediately for any non-F32 tensor) still reads as though
  // VRAM budget were the constraint, sending you off to tune dense_gb /
  // moe_cache_gb for nothing. On GLM-5.3 attn_output is quantized, so all
  // 78 are 'rejected', 0 are budget skips, with 15.6 GB VRAM free.
  //
  // That rejection is CORRECT, not a bug worth 'fixing': this path's
  // original win was BF16-vs-F32 (halves memory AND speeds up compute).
  // BF16-vs-quantized is the opposite trade -- 69 MB -> 201 MB per layer,
  // ~10.3 GB extra across 78 layers on a 16.65 GB card -- to accelerate a
  // ~5.7s bucket. Do not relax the F32 guard without measuring that trade.
  if (fast_skipped_upload > 0 && fast_converted == 0) {
    LOG_STREAM_INFO()
        << "[CUDA] attn_output fast-path is INACTIVE by design on this model: "
           "upload_dequantized_bf16 accepts F32 tensors only, and this model's "
           "attn_output is quantized. Not a VRAM problem -- see the note in "
           "main.cpp.\n";
  }

  // Pass 1: collect the REMAINING candidates (fast-path attn_output layers
  // excluded — they don't need a quantized VRAM copy anymore) and total the
  // pool size needed (256-byte aligned per tensor, matching
  // offload_tensor_to_pool's own alignment — must agree or the pool could
  // report "fits" here but reject tensors during the actual suballocation
  // pass below).
  // Category breakdown of what's actually going into the dense pool --
  // added after "why is this 14GB" needed real numbers instead of a
  // guess. Categorizes by tensor NAME PATTERN (most-specific check
  // first, so e.g. "attn_norm" -- which contains both "attn" and "norm"
  // -- resolves to exactly one bucket, never double-counted) and sums
  // bytes+count per category ACROSS ALL LAYERS, rather than logging one
  // line per individual layer (78 "attn_output" lines would bury the
  // signal, not reveal it) -- the category is what explains the total,
  // not which specific layer index it came from.
  struct DenseCategoryTotal {
    size_t bytes = 0;
    int count = 0;
  };
  std::map<std::string, DenseCategoryTotal> category_totals;
  auto categorize_dense_tensor = [](const std::string &name) -> const char * {
    // attn_output_a never reaches this function at all (excluded by
    // is_dense_offload_candidate above) -- only attn_output_b (V4) or
    // attn_output (V3/dense) land in this bucket.
    if (name.find("attn_output.weight") != std::string::npos ||
        name.find("attn_output_b.weight") != std::string::npos)
      return "attn_output (unconverted -- no BF16 fast-path budget left)";
    // DeepSeek-V4 Hyper-Connections weights -- see is_dense_offload_
    // candidate's own comment on why "hc_" catches all 9 (6 per-layer +
    // 3 global) of these. Checked before "attn" below since hc_attn_*
    // would otherwise fall into the misleading "other_attention"
    // bucket (they're residual-stream mixing weights, not attention
    // projections).
    if (name.find("hc_") != std::string::npos)
      return "hyper_connections (hc_*)";
    if (name.find("indexer.proj.weight") != std::string::npos ||
        name.find("engram_wkv.weight") != std::string::npos ||
        name.find("engram_q.weight") != std::string::npos ||
        name.find("engram_k.weight") != std::string::npos)
      return "indexer/engram (DeepSeek-V4.1 gate projections)";
    if (name.find("shexp") != std::string::npos)
      return "shared_expert (shexp)";
    if (name.find("ffn_gate.weight") != std::string::npos ||
        name.find("ffn_up.weight") != std::string::npos ||
        name.find("ffn_down.weight") != std::string::npos)
      return "leading_dense_ffn (pre-MoE layers)";
    if (name == "token_embd.weight" || name == "output.weight" ||
        name == "output_norm.weight")
      return "embedding + LM head";
    if (name.find("gate_inp") != std::string::npos)
      return "moe_router (gate_inp)";
    if (name.find("norm") != std::string::npos)
      return "norms";
    if (name.find("attn") != std::string::npos)
      return "other_attention (q/k/v/kv_a projections)";
    return "other";
  };

  std::vector<QuantizedTensor *> dense_candidates;
  size_t pool_bytes_needed = 0;
  for (auto &[name, tensor] : engine.resident_layers) {
    if (!tensor.host_ptr || !is_dense_offload_candidate(name))
      continue;
    if (tensor.gpu_fast_ptr)
      continue; // already fast-path converted above
    dense_candidates.push_back(&tensor);
    pool_bytes_needed =
        (pool_bytes_needed + POOL_ALIGN - 1) & ~(POOL_ALIGN - 1);
    pool_bytes_needed += tensor.meta.size_bytes;
    auto &cat = category_totals[categorize_dense_tensor(name)];
    cat.bytes += tensor.meta.size_bytes;
    cat.count += 1;
  }

  // Largest-first so the dominant contributor is obvious without having
  // to scan every line.
  std::vector<std::pair<std::string, DenseCategoryTotal>> sorted_categories(
      category_totals.begin(), category_totals.end());
  std::sort(sorted_categories.begin(), sorted_categories.end(),
            [](const auto &a, const auto &b) {
              return a.second.bytes > b.second.bytes;
            });
  LOG_STREAM_INFO() << "[VRAM-BREAKDOWN] dense pool composition (requested "
                    << pool_bytes_needed / 1e9 << " GB across "
                    << dense_candidates.size() << " tensor(s)):\n";
  for (auto &[cat_name, tot] : sorted_categories) {
    LOG_STREAM_INFO() << "  " << cat_name << ": " << tot.bytes / 1e9 << " GB ("
                      << tot.count << " tensor(s))\n";
  }

  // Largest tensors first: with a linear bump allocator, total space used
  // is the same regardless of order, but if the pool ends up smaller than
  // requested (backoff below) it's whatever's processed LAST that spills
  // to host. Prioritize keeping large, expensive-to-CPU-dequant tensors
  // (e.g. ~69 MB attn_output) on GPU over small ones (~KB norms, cheap on
  // CPU either way) by placing large tensors first in the fill order.
  std::sort(dense_candidates.begin(), dense_candidates.end(),
            [](const QuantizedTensor *a, const QuantizedTensor *b) {
              return a->meta.size_bytes > b->meta.size_bytes;
            });

  // Full-graph mode binds these weights into its own GGML CUDA arena. The
  // main pool would be a duplicate copy that the graph cannot reference and
  // would consume the headroom needed by the scheduler workspace.
  bool pool_ready = graph_mode ? true : gpu.init_dense_pool(pool_bytes_needed);
  if (graph_mode) {
    LOG_STREAM_INFO() << "[CUDA] Dense pool skipped: full GGML graph owns a "
                         "GGML device arena\n";
  }
  if (!pool_ready && pool_bytes_needed > 0) {
    // Didn't fit at full size — back off to a fraction of ACTUAL free
    // VRAM (ground truth, not our own estimate) and retry. Whatever
    // doesn't fit in the reduced pool falls back to CPU dequant-matmul,
    // same graceful degradation as before, just now fragmentation-free
    // for however much DOES fit.
    size_t backoff_bytes = (size_t)(gpu.core_free_vram_bytes() * 0.95);
    LOG_STREAM_WARN() << "[CUDA] Full dense pool (" << pool_bytes_needed / 1e9
                      << " GB) didn't fit — retrying with "
                      << backoff_bytes / 1e9
                      << " GB (95% of currently free VRAM)\n";
    pool_ready = gpu.init_dense_pool(backoff_bytes);
  }

  int offload_ok = 0, offload_failed = 0;
  size_t offload_failed_bytes = 0;
  if (pool_ready && !graph_mode) {
    for (QuantizedTensor *tensor : dense_candidates) {
      if (gpu.offload_tensor_to_pool(*tensor)) {
        ++offload_ok;
      } else {
        ++offload_failed;
        offload_failed_bytes += tensor->meta.size_bytes;
      }
    }
  } else if (!graph_mode) {
    offload_failed = (int)dense_candidates.size();
    for (QuantizedTensor *tensor : dense_candidates)
      offload_failed_bytes += tensor->meta.size_bytes;
    LOG_STREAM_ERROR()
        << "[CUDA] Dense tensor pool could not be allocated at any size — "
        << "all " << offload_failed << " tensor(s) staying host-resident.\n";
  }

  if (graph_mode) {
    LOG_STREAM_INFO() << "[CUDA] Dense offload skipped: full GGML graph owns "
                         "the device arena\n";
  } else {
    LOG_STREAM_INFO() << "[CUDA] Dense offload: " << offload_ok
                      << " tensors on VRAM\n";
  }
  if (offload_failed > 0) {
    LOG_STREAM_WARN()
        << "[CUDA] " << offload_failed << " tensor(s) ("
        << offload_failed_bytes / 1e9 << " GB) did NOT fit in VRAM — "
        << "staying host-resident (CPU dequant-matmul fallback for those). "
        << "Increase dense_gb / use a smaller-precision dense_model_path if "
           "this "
        << "matters for your workload.\n";
  }
  gpu.report_vram_usage("after dense tensor offload");
  gpu.report_vram_breakdown("after dense tensor offload", fast_bytes_used,
                            fast_converted);

  // Dual-role CoreEngine expert cache: opt-in only, via the CoreEngine
  // GPU's own [[gpus]] moe_cache_gb TOML entry (0 = disabled, the
  // default -- this NEVER activates unless explicitly configured, same
  // "0 disables" convention as pin_threshold/pin_topk_per_layer
  // elsewhere in this file). Deliberately placed HERE -- after dense
  // tensor offload AND every BF16 fast-path conversion above have
  // already claimed their real VRAM -- specifically so this sizes
  // against GROUND TRUTH remaining free VRAM (gpu.core_free_vram_bytes())
  // rather than a pre-allocation estimate: the same "trust cudaMemGetInfo
  // over any static guess" discipline this file's fast-path budgeting
  // already relies on, after this project got burned once by an estimate
  // that drifted from reality (the attn_output worst-case-reservation
  // OOM bug -- see CLAUDE.md). Must run BEFORE prefetcher.start() below:
  // add_dual_role_expert_cache's reindex is only safe before anything
  // begins actually using expert_vram_cache.
  // Skip entirely if this device is already a primary vault (role=
  // "dense+moe" on a genuine single-GPU box: assign_roles() puts it in
  // BOTH dense_gpu_idx and vault_gpus, and the main allocate_expert_cache()
  // pass already gave it its full configured moe_cache_gb, FIRST, before
  // any of the BF16/dense-offload logic above ever ran). Without this
  // guard, add_dual_role_expert_cache would attempt a SECOND, separate
  // cudaMalloc for the same configured GB on top of the pool that already
  // exists -- harmless in practice (it just fails the cap_bytes >=
  // expert_slab_bytes check below once real free VRAM is near zero and
  // logs a confusing "skipped" line right after the real pool's own
  // success line), but pointless and misleading to leave in.
  bool already_primary_vault =
      std::find(gpu.topology.vault_gpus.begin(), gpu.topology.vault_gpus.end(),
                gpu.topology.core_gpu) != gpu.topology.vault_gpus.end();
  double core_moe_cache_gb_cfg = 0.0;
  for (auto &gc : config.gpus) {
    if (gc.device_id == gpu.device_id) {
      core_moe_cache_gb_cfg = gc.moe_cache_gb;
      break;
    }
  }
  if (!already_primary_vault && core_moe_cache_gb_cfg > 0.0 &&
      expert_slab_bytes > 0) {
    // Same 1 GB margin style as FAST_PATH_SAFETY_MARGIN above -- leaves
    // room for CUDA context growth, cuBLAS workspace, and prefill-time
    // scratch buffer growth (d_x/d_y/d_x_bf16) that hasn't happened yet
    // at this point in startup.
    constexpr size_t CORE_VAULT_SAFETY_MARGIN = (size_t)1 << 30;
    size_t requested_bytes = (size_t)(core_moe_cache_gb_cfg * 1e9);
    size_t real_free = gpu.core_free_vram_bytes();
    size_t safe_cap = (real_free > CORE_VAULT_SAFETY_MARGIN)
                          ? (real_free - CORE_VAULT_SAFETY_MARGIN)
                          : 0;
    size_t cap_bytes = std::min(requested_bytes, safe_cap);
    if (cap_bytes >= expert_slab_bytes) {
      gpu.topology.add_dual_role_expert_cache(gpu.topology.core_gpu, cap_bytes,
                                              expert_slab_bytes);
      gpu.report_vram_usage("after CoreEngine dual-role expert cache");
      gpu.report_vram_breakdown("after CoreEngine dual-role expert cache",
                                fast_bytes_used, fast_converted);
    } else {
      LOG_STREAM_WARN()
          << "[GPU] CoreEngine dual-role expert cache requested ("
          << core_moe_cache_gb_cfg << " GB) but only " << safe_cap / 1e9
          << " GB safely free after reserving "
          << CORE_VAULT_SAFETY_MARGIN / 1e9
          << " GB margin -- skipped (need at least one expert slab, "
          << expert_slab_bytes / 1e9 << " GB).\n";
    }
  }

  // prefetcher.start() moved to AFTER the [PIN] preload block below (see
  // that block's own closing comment for why) -- the worker threads it
  // spawns now claim/populate/publish directly into engine.layers_experts
  // via RAMSafetyManager::claim_or_wait, and [PIN]'s Tier-3 preload writes
  // into that exact same structure; starting the threads only once that
  // preload has fully finished removes any need to reason about ordering
  // between them.
  //
  // Decode thread's OWN io_uring ring for RAMSafetyManager::get_experts_batch's
  // Phase B (the synchronous miss path -- see ram_manager.hpp's
  // decode_uring_ring comment for why this is a separate scope from the
  // prefetcher's rings). Unaffected by the prefetcher.start() move -- [PIN]
  // preload never uses this ring, only engine.load_expert_slab's plain
  // synchronous pread.
  ram_mgr.init_decode_uring(config.io_uring, config.io_uring_workers);

  // extended to a graduated 3-tier scheme (was:
  // a single top-K-pinned VRAM tier only). Read the usage file written
  // by a PREVIOUS run's dump_usage() at shutdown -- this already records
  // EVERY expert ever routed to (StreamingEngine::route_freq is a full
  // n_layers*n_experts histogram, not just a top-K sample), so all the
  // ranking data below has been available all along; what's new is
  // USING more of it. Per layer, ranked by historical selection count:
  //   ranks [0, pin_topk_per_layer)                              -> VRAM,
  //   PINNED ranks [pin_topk_per_layer, +pin_vram_lru_per_layer)         ->
  //   VRAM, unpinned (reactive LRU) ranks [..+pin_vram_lru_per_layer,
  //   +pin_ram_per_layer)       -> host RAM tier
  // First-ever run (no usage file yet) is a silent no-op for all three
  // tiers -- both caches fill reactively exactly as before this feature
  // existed. One-time startup cost (real disk reads), not a per-token one.
  //
  // ram_mgr.slots_per_layer is needed to safely bound the RAM tier
  // preload (it's the real per-layer RAM cache cap get_expert/
  // get_experts_batch scan/evict over -- writing into a slot index at or
  // beyond it would be invisible to every lookup until some UNRELATED
  // later request happens to raise the cap, which never happens once
  // set). Normally only computed per-request (generate_text calls this
  // with the real prompt's max_ctx), so call it once here too with the
  // configured context_size as a representative estimate purely for
  // preload sizing -- idempotent, generate_text's own later call simply
  // recomputes it from the real request.
  if (ram_mgr.vram_cache)
    ram_mgr.calculate_safe_cache_capacity(engine, config.context_size);

  // ram_cache_layer_adaptive: reshapes the uniform slots_per_layer budget
  // calculate_safe_cache_capacity just computed into a per-layer
  // distribution driven by real historical need (see apply_layer_
  // adaptive_capacity's own comment) -- must run AFTER calculate_safe_
  // cache_capacity (needs its slots_per_layer as the total budget to
  // redistribute) and BEFORE the [PIN] block below (Tier 3's ram_cap
  // bound needs the final per-layer caps, via ram_mgr.cap_for_layer).
  // No-op if disabled, or if there's no usage history yet -- see that
  // function's own "first-ever run is identical to disabled" guarantee.
  if (config.ram_cache_layer_adaptive) {
    ram_mgr.apply_layer_adaptive_capacity(config.pin_file, cache_layers,
                                          cfg.n_experts);
  }

  // Prefill sliding-window RAM capacity (2026-08-30, P1+P2) -- must come
  // AFTER both capacity steps above, since the window is a third override
  // tier layered on top of whatever base cap they settled (uniform, or
  // adaptive per-layer). No-op when disabled: sizes the override vector
  // all-zero, which cap_for_layer falls straight through.
  ram_mgr.init_prefill_window(engine, config.prefill_window_cap,
                              config.prefill_window_extra_gb);

  // Optional graph/RAM handoff. A stock GGML graph cannot safely refer to
  // an LRU slot whose slab may be evicted, so only layers whose complete
  // expert set fits the already-calculated RAM capacity are exposed as
  // fixed-stride graph tensors. The remaining layers use the graph's
  // source-backed fallback and retain normal lazy cache behavior.
  if (ram_graph_cache) {
    const char *layer0_diag_env =
        std::getenv("MOESTREAM_QWEN4EXP_DIAG_LAYER0_RESIDUAL");
    const bool narrow_layer0_diag =
        layer0_diag_env && layer0_diag_env[0] == '1';
    std::vector<int> graph_ram_full_layers;
    std::vector<int> graph_ram_projection_compact_layers;
    const int n_experts = engine.model_file.architecture_config.n_experts;
    for (int layer = 0; layer < (int)engine.layers_experts.size(); ++layer) {
      if (engine.qwen_ram_vram_only_layer(layer))
        continue;
      const bool qwen_cpu_layer =
          qwen_cpu_ram_cache &&
          (is_cpu_affinity_layer(layer) ||
           engine.qwen4exp_cpu_offload_layer_enabled(layer));
      // A full-graph RAM cache is intentionally complete-model wide,
      // but the Qwen CPU-affinity path only needs fixed-stride arenas
      // for its CPU layers.  `graph_ram_cache` is also true for the
      // Qwen CPU mode, so testing it directly here used to allocate
      // graph arenas for every layer.  That was wasteful and made the
      // source ownership harder to audit; non-CPU Qwen layers already
      // use the ordinary RAM cache.  In either case the bytes are
      // populated by the preload block below, never by GGUF mmap at
      // decode time.
      const bool full_graph_ram_layer =
          graph_mode && graph_ram_cache && !narrow_layer0_diag;
      const bool narrow_layer0_ram_layer =
          narrow_layer0_diag && layer == 0 &&
          engine.qwen4exp_cpu_offload_layer_enabled(layer);
      if (!full_graph_ram_layer && !qwen_cpu_layer &&
          !narrow_layer0_ram_layer)
        continue;
      if (n_experts <= 0 || engine.compute_expert_slab_bytes(layer) == 0 ||
          (int)engine.layers_experts[(size_t)layer].size() < n_experts ||
          ram_mgr.cap_for_layer(layer) < n_experts)
        continue;
      graph_ram_full_layers.push_back(layer);
      // Qwen's full graph can consume separate compact gate/up/down views on
      // CPU. GPU layers keep their fused gate/up graph by gathering the two
      // planes directly into the final VRAM tensor, so the arena itself can
      // use the same no-duplicate layout for every Qwen graph layer. Keep an
      // explicit opt-out for A/B comparisons; it changes only the DDR arena
      // layout, not layer placement or graph topology.
      const char *compact_env = std::getenv(
          "MOESTREAM_QWEN4EXP_GRAPH_PROJECTION_COMPACT");
      const bool projection_compact_enabled =
          !(compact_env && compact_env[0] == '0');
      const char *host_node_env =
          std::getenv("MOESTREAM_QWEN4EXP_HOST_NODE");
      const bool host_node_enabled = host_node_env && host_node_env[0] == '1';
      // The legacy scheduler CPUOffload path still uses the single-stride
      // runner, so retain its interleaved arena. The opt-in host-node path
      // now supports independent projection strides and can use the compact
      // planes without creating a second copy.
      if (engine.family_ops.family == ModelFamily::Qwen4Exp &&
          projection_compact_enabled &&
          (!engine.qwen4exp_cpu_offload_layer_enabled(layer) ||
           host_node_enabled))
        graph_ram_projection_compact_layers.push_back(layer);
    }
    if (!engine.prepare_graph_ram_arenas(
            graph_ram_full_layers, graph_ram_projection_compact_layers)) {
      LOG_STREAM_WARN()
          << "[GRAPH-RAM] no complete layer arena was admitted; "
             "full graph will use its bounded source-backed path\n";
    }
  }

  // prefill->decode retention handoff (2026-08-31). 0 = off (default),
  // in which case eviction_score is byte-identical to its prior form.
  RAMSafetyManager::set_prefill_retention_weight(
      (uint64_t)std::max(0, config.prefill_retention_weight));
  RAMSafetyManager::set_prefill_retention_shape(
      std::max(0, config.prefill_retention_top_n),
      std::max(1, config.prefill_retention_min_demand),
      config.prefill_retention_flat_cut, config.prefill_retention_peaky_cut);

  // RAM-tier load grace period (2026-09-04). 0 tokens = off (default), in
  // which case eviction_score is byte-identical to its prior form.
  RAMSafetyManager::set_ram_grace_period(
      (uint64_t)std::max(0, config.ram_grace_period_tokens),
      (uint64_t)std::max(0, config.ram_grace_period_bonus));

  // Usage files are learned state, not model data.  A stale file from a
  // different checkpoint can therefore contain layer/expert IDs that do
  // not exist in this model.  The preload code below indexes
  // engine.layers_experts[layer] several times, so let the bad record be
  // visible and discard it before it can turn into an opaque SIGSEGV.
  // Keep this validation here, after build_model_graph has created the real
  // per-layer slot vectors (GLM-5.3 has dense leading layers and a hybrid
  // layer count), rather than trusting only architecture metadata.
  auto sanitize_usage_entries =
      [&](const std::string &path, const char *source,
          std::vector<StreamingEngine::UsageEntry> raw) {
        const size_t model_layer_count = engine.layers_experts.size();
        const int model_expert_count =
            engine.model_file.architecture_config.n_experts;
        std::vector<StreamingEngine::UsageEntry> valid;
        valid.reserve(raw.size());
        size_t rejected = 0;
        constexpr size_t kMaxSamples = 8;
        for (size_t record = 0; record < raw.size(); ++record) {
          const auto &entry = raw[record];
          const bool layer_valid =
              entry.layer >= 0 && (size_t)entry.layer < model_layer_count;
          const size_t layer_slot_count =
              layer_valid ? engine.layers_experts[(size_t)entry.layer].size()
                          : 0;
          const bool expert_valid = entry.expert >= 0 &&
                                    entry.expert < model_expert_count &&
                                    (size_t)entry.expert < layer_slot_count;
          if (layer_valid && expert_valid) {
            valid.push_back(entry);
            continue;
          }

          ++rejected;
          if (rejected <= kMaxSamples) {
            LOG_STREAM_ERROR()
                << "[PIN-BOUNDS] Ignoring invalid " << source << " record #"
                << (record + 1) << " from " << path << ": layer=" << entry.layer
                << ", expert=" << entry.expert << ", count=" << entry.count
                << ". Model has layer vectors [0," << model_layer_count
                << ") and n_experts=" << model_expert_count
                << "; this layer has " << layer_slot_count
                << " expert slots. The usage file is stale/incompatible or "
                   "corrupt.\n";
          }
        }
        if (rejected > 0) {
          LOG_STREAM_WARN()
              << "[PIN-BOUNDS] Rejected " << rejected << "/" << raw.size()
              << " " << source << " record(s) from " << path << "; retained "
              << valid.size()
              << " model-compatible record(s). Preload will continue safely."
              << (rejected > kMaxSamples
                      ? " Additional invalid records were suppressed.\n"
                      : "\n");
        }
        return valid;
      };

  // vram_evict_demote_to_ram: wired here (after calculate_safe_cache_capacity
  // has set a real ram_mgr.slots_per_layer, and before anything -- including
  // the [PIN] preload block right below -- can trigger a real VRAM eviction)
  // so a demote never runs against an uninitialized RAM-tier cap. See
  // gpu_topology.hpp's on_vram_evict and ram_manager.hpp's demote_from_vram
  // for why this is safe to call from the decode thread only (upload_expert),
  // never the prefetcher thread (upload_expert_async, deliberately left
  // unwired). Captures engine/ram_mgr by reference -- both outlive the
  // GpuTopology instance this callback is attached to (same function scope,
  // torn down together at the end of run_engine).
  if (config.vram_evict_demote_to_ram && ram_mgr.vram_cache) {
    ram_mgr.vram_cache->on_vram_evict =
        [&engine, &ram_mgr](int layer, int expert_id, const uint8_t *device_ptr,
                            size_t bytes, int gpu_id) {
          ram_mgr.demote_from_vram(engine, layer, expert_id, device_ptr, bytes,
                                   gpu_id);
        };
  }

  // The opt-in Qwen4Exp full graph needs one complete host-side expert
  // backing set for ggml MUL_MAT_ID.  With MOESTREAM_FULL_GRAPH_USE_RAM_CACHE
  // enabled, that set is the already-preloaded fixed-stride DDR arena and
  // the graph aliases it directly; otherwise the graph owns one startup
  // pread-backed DDR copy.  Running the ordinary pin/preload pipeline as
  // well without the RAM-cache mode would allocate a second complete expert
  // set before graph preparation (about 42 GB on UD-IQ1_M), and can push
  // this box into swap/OOM.  If the graph declines and falls back, that
  // fallback is intentionally visible in the log rather than silently
  // paying this unsafe duplicate allocation.
  if ((!graph_mode || graph_ram_cache) &&
      (!config.pin_file.empty() || config.ram_preload_all_experts) &&
      ram_mgr.vram_cache &&
      (ram_graph_cache || ram_mgr.vram_cache->expert_vram_cache_cap > 0)) {
    ram_mgr.vram_cache->startup_preload_phase = true;
    auto [entries, maxc] = StreamingEngine::load_usage(config.pin_file);
    const bool usage_file_had_entries = !entries.empty();
    entries =
        sanitize_usage_entries(config.pin_file, "pin_file", std::move(entries));
    maxc = 0;
    for (const auto &entry : entries)
      maxc = std::max(maxc, entry.count);
    if (entries.empty() && !config.ram_preload_all_experts) {
      if (usage_file_had_entries) {
        LOG_STREAM_WARN()
            << "[PIN] No model-compatible records remain in " << config.pin_file
            << " -- VRAM/RAM caches will fill reactively for this run.\n";
      } else {
        LOG_STREAM_INFO()
            << "[PIN] No usage file at " << config.pin_file
            << " yet (first run) -- VRAM/RAM caches will fill reactively.\n";
      }
    } else {
      std::unordered_map<int, std::vector<StreamingEngine::UsageEntry>>
          by_layer;
      for (auto &e : entries)
        by_layer[e.layer].push_back(e);
      for (auto &[layer, v] : by_layer) {
        // stable_sort: plain std::sort has no tie-break guarantee for
        // equal-count entries (very common at the low-count end of a
        // real usage file). See the prefill_pin_file merge block
        // below for the concrete, measured case this caused -- kept
        // stable here too so this ranking is deterministic on its
        // own, not just relative to that merge.
        std::stable_sort(v.begin(), v.end(), [](const auto &a, const auto &b) {
          return a.count > b.count;
        });
      }

      // prefill_pin_file: MERGE its ranking ahead of the general
      // pin_file ranking for any layer it has data for, rather than
      // replacing that layer's ranking outright. Prefill runs before
      // any decode token, so its own experts get priority for Tier
      // 1/2/3 slot assignment (they're needed first, chronologically)
      // -- but a hard per-layer REPLACE (the previous behavior) threw
      // away pin_file's own, typically much richer, decode-time
      // ranking for that same layer entirely. On a real run this
      // starved Tier 3 (RAM) down to 0 preloaded experts across every
      // layer: prefill_pin_file had only 732 known experts vs.
      // pin_file's 3311, and since decode dominates a normal
      // request's wall-clock time (prefill is a handful of tokens),
      // discarding pin_file's knowledge for every prefill-covered
      // layer left Tier 1/2 (topk + vram_lru target) with nothing
      // left over for Tier 3 to draw from, even though pin_ram_
      // fraction=1.0 was requesting the maximum.
      //
      // Fixed by concatenating instead: prefill's own ranked list
      // stays first (unchanged priority for Tier 1/2), then pin_
      // file's ranked list for that layer is appended MINUS any
      // expert prefill already covers (an unordered_set membership
      // check per layer) -- so a given (layer,expert) pair is never
      // attempted twice by two different source rankings, and Tier 3
      // gets pin_file's full remaining knowledge instead of nothing.
      // A layer prefill_pin_file has no entries for (e.g. its very
      // first run, before any prefill has been recorded into it yet)
      // is untouched and keeps using the general pin_file ranking
      // alone, exactly as before.
      int prefill_layers_merged = 0;
      // prefill_exclusive_protected: entries that got the bounded,
      // unconditional floor slot (pin_global_floor_per_layer/layer at
      // most). prefill_exclusive_competing: prefill-exclusive entries
      // that instead went into the honest count-sorted pool alongside
      // pin_file's own data -- a low number here relative to the
      // total known-experts count is expected and correct given how
      // flat prefill's own routing tends to be (see the per-layer
      // concentration diagnostic below); it's only a problem if it's
      // near-zero for EVERY layer, which would mean the floor itself
      // is mis-sized (pin_global_floor_per_layer too small/large).
      int prefill_exclusive_protected = 0, prefill_exclusive_competing = 0;
      if (!config.prefill_pin_file.empty()) {
        auto [pf_entries, pf_maxc] =
            StreamingEngine::load_usage(config.prefill_pin_file);
        (void)pf_maxc;
        const bool prefill_file_had_entries = !pf_entries.empty();
        pf_entries = sanitize_usage_entries(
            config.prefill_pin_file, "prefill_pin_file", std::move(pf_entries));
        if (pf_entries.empty()) {
          if (prefill_file_had_entries) {
            LOG_STREAM_WARN()
                << "[PIN] prefill_pin_file (" << config.prefill_pin_file
                << ") has no model-compatible records -- every layer falls "
                   "back "
                   "to the general pin_file ranking for this run.\n";
          } else {
            LOG_STREAM_INFO()
                << "[PIN] prefill_pin_file (" << config.prefill_pin_file
                << ") has no data yet -- every layer falls back to the general "
                   "pin_file "
                   "ranking for this run.\n";
          }
        } else {
          // Per-layer concentration diagnostic (the "statistics
          // for the prefill pin" this whole redesign was
          // motivated by) -- mirrors the general pin_file's own
          // [PIN-CONCENTRATION] block below, but computed on
          // prefill's OWN data before any merge with pin_file, so
          // it answers "how concentrated is PREFILL's routing,
          // independent of decode" directly rather than through
          // the combined ranking's own lens.
          {
            std::unordered_map<int, std::vector<StreamingEngine::UsageEntry>>
                pf_raw_by_layer;
            for (auto &e : pf_entries)
              pf_raw_by_layer[e.layer].push_back(e);
            double cov_sum = 0.0;
            int cov_n = 0;
            double cov_min = 2.0, cov_max = -1.0;
            int layer_min = -1, layer_max_l = -1;
            for (auto &[layer, v] : pf_raw_by_layer) {
              uint64_t total = 0, top1 = 0;
              for (auto &e : v) {
                total += e.count;
                if (e.count > top1)
                  top1 = e.count;
              }
              if (total == 0)
                continue;
              double share = (double)top1 / (double)total;
              cov_sum += share;
              ++cov_n;
              if (share < cov_min) {
                cov_min = share;
                layer_min = layer;
              }
              if (share > cov_max) {
                cov_max = share;
                layer_max_l = layer;
              }
              LOG_STREAM_DEBUG()
                  << "[PIN-PREFILL-CONCENTRATION] layer=" << layer
                  << " top-1 share=" << (share * 100.0) << "% of " << v.size()
                  << " distinct experts / " << total << " total picks\n";
            }
            if (cov_n > 0) {
              LOG_STREAM_INFO()
                  << "[PIN] prefill_pin_file own concentration: avg top-1 "
                     "share "
                  << (cov_sum / cov_n * 100.0) << "% across " << cov_n
                  << " layer(s) -- most concentrated: layer " << layer_max_l
                  << " (" << (cov_max * 100.0) << "%), flattest: layer "
                  << layer_min << " (" << (cov_min * 100.0)
                  << "%). Low values everywhere mean "
                     "prefill's own routing rarely has a clear per-layer "
                     "winner -- "
                     "expect the bounded floor below to protect little, by "
                     "design. "
                     "Full per-layer breakdown at log_level=DEBUG "
                     "([PIN-PREFILL-CONCENTRATION] lines).\n";
            }
          }
          std::unordered_map<int, std::vector<StreamingEngine::UsageEntry>>
              pf_by_layer;
          for (auto &e : pf_entries)
            pf_by_layer[e.layer].push_back(e);
          for (auto &[layer, pf_v] : pf_by_layer) {
            std::stable_sort(
                pf_v.begin(), pf_v.end(),
                [](const auto &a, const auto &b) { return a.count > b.count; });

            // Real, measured regression (2026-08-12, GLM-5.2),
            // fixed in two stages:
            //
            // Stage 1 (count corruption): the original version
            // here kept PREFILL's entry and dropped pin_file's
            // (decode's) entry for any (layer,expert) pair known
            // to both -- silently discarding decode's real,
            // much-larger-sample count in favor of prefill's
            // tiny one (a handful of hits from one short
            // prompt). Under pin_strategy=global_topn, ranking
            // is purely by that count value, so a genuinely
            // decode-hot expert that prefill also happened to
            // touch once lost its earned global rank to
            // whatever unrelated expert decode never needed --
            // measured net cost on a 51-token decode-heavy
            // generation: decode 0.68->0.57 tok/s (+11.2s wall)
            // vs. prefill 0.61->0.81 tok/s (-4.4s wall), a net
            // +6.7s loss. Fixed by never discarding pin_file's
            // own entry -- prefill only contributes entries
            // pin_file has ZERO history for.
            //
            // Stage 2 (positional bug, found while investigating
            // why prefill's OWN throughput never improved even
            // after stage 1 -- real per-layer concentration
            // check on .moestream_usage_prefill showed EVERY
            // layer's prefill routing is close to flat, 3.4%-
            // 11.4% top-1 share, nowhere near "an obvious
            // winner"): stage 1 still put ALL prefill-exclusive
            // entries at the front of a layer's list,
            // unconditionally, regardless of how many there
            // were or how small their own counts. The NUMBER of
            // pinned slots a layer gets (layer_pin_count below)
            // is decided by real per-entry counts, but WHICH
            // entries fill those slots was still pure array
            // position -- so on a layer with, say, 8 weak
            // prefill-exclusive picks (count 1-4 each) at the
            // front, they could still bump a genuinely strong
            // decode pick (count 50+) out of that layer's real
            // pinned budget, purely by position, not merit.
            //
            // Fixed by bounding prefill's UNCONDITIONAL
            // protection to prefill's own best
            // pin_global_floor_per_layer pick(s) (mirroring the
            // floor concept the global_topn remainder ranking
            // below already uses) -- everything else, whether
            // it came from prefill or decode, is pooled and
            // re-sorted by its own real count before being
            // sliced into tiers. A layer where prefill genuinely
            // IS concentrated naturally wins more of the
            // remainder ranking (its counts are real and
            // competitive); a flat layer correctly falls back
            // to being decided by decode's data alone. This is
            // effectively global-topk applied one level deeper,
            // to prefill's own contribution specifically, not
            // just the top-level pin_file/prefill_pin_file
            // split.
            auto it = by_layer.find(layer);
            std::unordered_set<int> decode_experts;
            if (it != by_layer.end()) {
              decode_experts.reserve(it->second.size());
              for (auto &e : it->second)
                decode_experts.insert(e.expert);
            }
            std::vector<StreamingEngine::UsageEntry>
                exclusive; // pf_v already sorted by count desc
            for (auto &e : pf_v) {
              if (decode_experts.find(e.expert) == decode_experts.end())
                exclusive.push_back(e); // prefill-exclusive only -- decode has
                                        // no data for this one
            }
            size_t protected_n =
                std::min((size_t)std::max(0, config.pin_global_floor_per_layer),
                         exclusive.size());
            std::vector<StreamingEngine::UsageEntry> merged(
                exclusive.begin(), exclusive.begin() + protected_n);
            std::vector<StreamingEngine::UsageEntry> rest(
                exclusive.begin() + protected_n, exclusive.end());
            if (it != by_layer.end())
              rest.insert(rest.end(), it->second.begin(), it->second.end());
            // stable_sort: THE fix -- this is the exact resort a
            // real run showed regressing decode from 0.68->0.62
            // tok/s. rest is `exclusive`'s remainder (often EMPTY
            // -- confirmed via a real usage-file check that this
            // prompt's prefill routing is a strict subset of
            // decode's own) plus a freshly-copied `it->second`.
            // Even when exclusive truly is empty, this is a NEW
            // vector object (different memory layout than
            // by_layer[layer]'s own in-place-sorted vector), and
            // plain std::sort gives no tie-break guarantee for
            // equal-count entries -- common at the low-count end
            // of a real usage file. Two sorts of "the same"
            // count-sorted data, on two different vector objects,
            // can legally disagree on tie order; under
            // pin_strategy=global_topn's hard top-N truncation
            // (see the [PIN] block below), that's enough to swap
            // which of several equal-count experts lands just
            // above vs. below the pin cutoff -- a real, measured
            // effect, not a theoretical one. stable_sort on an
            // already-count-sorted `it->second` preserves its
            // existing relative order for ties instead of
            // reshuffling them, so a functionally-empty merge
            // (the common case here) is now ALSO a byte-for-byte
            // no-op, matching prefill_pin_file being unset.
            std::stable_sort(
                rest.begin(), rest.end(),
                [](const auto &a, const auto &b) { return a.count > b.count; });
            merged.insert(merged.end(), rest.begin(), rest.end());
            by_layer[layer] = std::move(merged);
            prefill_exclusive_protected += (int)protected_n;
            prefill_exclusive_competing +=
                (int)(exclusive.size() - protected_n);
            ++prefill_layers_merged;
          }
          LOG_STREAM_INFO()
              << "[PIN] prefill_pin_file (" << config.prefill_pin_file << ", "
              << pf_entries.size() << " known experts): merged for "
              << prefill_layers_merged << " layer(s) -- "
              << prefill_exclusive_protected
              << " entries got the unconditional bounded floor "
                 "(pin_global_floor_per_layer="
              << config.pin_global_floor_per_layer << "/layer), "
              << prefill_exclusive_competing
              << " more prefill-exclusive entries went into honest count-based "
                 "competition "
                 "against pin_file's own data (wins depend on real "
                 "concentration, see "
                 "the [PIN] prefill_pin_file own concentration line above); "
                 "remaining "
                 "layers use the general pin_file ranking alone.\n";
        }
      }

      for (int layer = 0; layer < (int)engine.layers_experts.size(); ++layer)
        if (engine.qwen_ram_vram_only_layer(layer))
          by_layer.erase(layer);

      // A usage file is a ranking signal, not a complete model
      // manifest. When DDR is large enough to hold the whole MoE, add
      // zero-count entries for every model expert so Tier 3 preloads
      // the full set instead of leaving history-missing experts to
      // fault in during the first request. Existing counts retain
      // their ranking priority; the newly added entries simply fill
      // the remaining RAM slots.
      if (config.ram_preload_all_experts) {
        // `n_layers` is the target trunk count. When the GLM
        // NextN/MTP bridge is active, the optional final block is a
        // real routed MoE layer too and must be included here;
        // otherwise its first proposal can still fault experts from
        // SSD even when the target trunk was fully preloaded into
        // host RAM.
        const int model_layers = static_cast<int>(engine.layers_experts.size());
        const int model_experts =
            engine.model_file.architecture_config.n_experts;
        size_t added = 0;
        for (int layer = 0; layer < model_layers; ++layer) {
          if (engine.compute_expert_slab_bytes(layer) == 0 ||
              model_experts <= 0)
            continue;
          auto &v = by_layer[layer];
          std::vector<unsigned char> present((size_t)model_experts, 0);
          for (const auto &entry : v) {
            if (entry.expert >= 0 && entry.expert < model_experts)
              present[(size_t)entry.expert] = 1;
          }
          for (int expert = 0; expert < model_experts; ++expert) {
            if (present[(size_t)expert])
              continue;
            v.push_back({layer, expert, 0});
            ++added;
          }
          std::stable_sort(
              v.begin(), v.end(),
              [](const auto &a, const auto &b) { return a.count > b.count; });
        }
        LOG_STREAM_INFO()
            << "[PIN] ram_preload_all_experts: added " << added
            << " history-missing model experts to the preload manifest\n";
      }

      // Option B's static hot set is prepared from the same final ranking
      // that drives startup RAM/VRAM preload.  Keep membership in an O(1)
      // device-dispatch-side mask so cold routes can be forced to the CPU
      // without another histogram lookup in the token loop.
      //
      // In concentrated-only mode, qwen4exp_option_b_layer=-1 means:
      // inspect every layer in pin_file and commit only layers whose top-N
      // experts meet the requested historical coverage.  The baseline's
      // complete GPU block is excluded; Option B is an augmentation for the
      // CPU-side layers, not a reason to replace already-resident weights.
      engine.qwen4exp_option_b_hot_mask.clear();
      engine.qwen4exp_option_b_hot_masks.clear();
      engine.qwen4exp_option_b_layers.clear();
      engine.qwen4exp_option_b_layer_coverage.clear();
      engine.qwen4exp_option_b_committed = false;
      if (config.qwen4exp_option_b_top_n > 0) {
        const int n_experts = engine.model_file.architecture_config.n_experts;
        const int n_layers = static_cast<int>(engine.layers_experts.size());
        const int baseline_gpu_layers = std::min(
            std::max(0, qwen4exp_full_graph_device_expert_layers()), n_layers);
        const bool baseline_gpu_trailing =
            qwen4exp_full_graph_device_expert_trailing();
        auto baseline_gpu_layer = [&](int layer) {
          if (baseline_gpu_layers <= 0)
            return false;
          return baseline_gpu_trailing
                     ? layer >= n_layers - baseline_gpu_layers
                     : layer < baseline_gpu_layers;
        };

        std::vector<int> option_layers;
        if (config.qwen4exp_option_b_layer >= 0) {
          option_layers.push_back(config.qwen4exp_option_b_layer);
        } else if (!config.qwen4exp_option_b_selected_layers.empty()) {
          option_layers = config.qwen4exp_option_b_selected_layers;
        } else if (config.qwen4exp_option_b_concentrated_only) {
          option_layers.reserve(by_layer.size());
          for (const auto &[layer, entries_for_layer] : by_layer) {
            (void)entries_for_layer;
            option_layers.push_back(layer);
          }
          std::sort(option_layers.begin(), option_layers.end());
        }

        const int hot_n = std::min(config.qwen4exp_option_b_top_n,
                                   n_experts);
        if (config.qwen4exp_option_b_selected_layers.empty() &&
            config.qwen4exp_option_b_concentrated_only &&
            config.qwen4exp_option_b_max_layers > 0 && hot_n > 0) {
          auto coverage_for_layer = [&](int layer) {
            std::vector<uint64_t> counts(static_cast<size_t>(n_experts), 0);
            auto it = by_layer.find(layer);
            if (it != by_layer.end()) {
              for (const auto &entry : it->second) {
                if (entry.expert >= 0 && entry.expert < n_experts)
                  counts[static_cast<size_t>(entry.expert)] += entry.count;
              }
            }
            uint64_t total = 0;
            for (uint64_t count : counts)
              total += count;
            std::sort(counts.begin(), counts.end(), std::greater<uint64_t>());
            uint64_t covered = 0;
            for (int i = 0; i < hot_n; ++i)
              covered += counts[static_cast<size_t>(i)];
            return total == 0
                       ? 0.0
                       : static_cast<double>(covered) /
                             static_cast<double>(total);
          };
          std::stable_sort(option_layers.begin(), option_layers.end(),
                           [&](int a, int b) {
                             const double ca = coverage_for_layer(a);
                             const double cb = coverage_for_layer(b);
                             if (ca != cb)
                               return ca > cb;
                             return a < b;
                           });
          if (static_cast<int>(option_layers.size()) >
              config.qwen4exp_option_b_max_layers)
            option_layers.resize(
                static_cast<size_t>(config.qwen4exp_option_b_max_layers));
          LOG_STREAM_INFO()
              << "[OPTION-B] concentrated-only layer selection ranked top-"
              << option_layers.size() << " by top-" << hot_n
              << " traffic coverage (cap="
              << config.qwen4exp_option_b_max_layers << ")\n";
        }
        // Global pool mode: rank all selected layers' experts together and keep
        // the top K of the pool (see include/option_b_global_hot.hpp). The
        // per-layer loop below is skipped; the graph planner calls the same
        // helper, so both passes agree on every layer's hot set.
        const bool option_b_global =
            config.qwen4exp_option_b_global_top_k > 0;
        if (option_b_global) {
          std::vector<int> live_layers;
          for (int layer : option_layers) {
            if (layer < 0 || layer >= n_layers || is_cpu_affinity_layer(layer))
              continue;
            if (baseline_gpu_layer(layer)) {
              LOG_STREAM_INFO()
                  << "[OPTION-B] Layer " << layer
                  << " skipped: already covered by the complete GPU block\n";
              continue;
            }
            live_layers.push_back(layer);
          }
          // The decode planner and its VRAM estimate read pin_file alone.
          // by_layer also contains prefill-exclusive records for RAM preload;
          // including them here can change membership and disagree with the
          // compact GPU tensors' actual hot set.
          const auto sel = option_b_global::select(
              entries, live_layers, n_layers, n_experts,
              config.qwen4exp_option_b_global_top_k,
              config.qwen4exp_option_b_global_floor_per_layer,
              config.qwen4exp_option_b_top_n);
          for (const auto &[layer, ids] : sel.hot) {
            std::vector<uint8_t> hot_mask(static_cast<size_t>(n_experts), 0);
            for (int expert : ids)
              hot_mask[static_cast<size_t>(expert)] = 1;
            const double coverage = sel.layer_coverage.at(layer);
            engine.qwen4exp_option_b_hot_masks[layer] = hot_mask;
            engine.qwen4exp_option_b_layers.push_back(layer);
            engine.qwen4exp_option_b_layer_coverage[layer] = coverage;
            if (layer == config.qwen4exp_option_b_layer)
              engine.qwen4exp_option_b_hot_mask = hot_mask;
            LOG_STREAM_INFO()
                << "[OPTION-B] Layer " << layer
                << " global-pool hot set: hot=" << ids.size()
                << " cold=" << (n_experts - static_cast<int>(ids.size()))
                << " history coverage=" << (coverage * 100.0) << "%\n";
          }
          LOG_STREAM_INFO()
              << "[OPTION-B] global pool: top-"
              << config.qwen4exp_option_b_global_top_k << " of "
              << live_layers.size() << " layer(s) -> " << sel.total_hot
              << " expert(s) in " << sel.hot.size()
              << " layer(s), pooled history coverage="
              << (sel.pool_coverage * 100.0) << "% (floor="
              << config.qwen4exp_option_b_global_floor_per_layer
              << "/layer, per-layer ceiling="
              << config.qwen4exp_option_b_top_n << ")\n";
        }
        for (int layer : option_b_global ? std::vector<int>{} : option_layers) {
          if (layer < 0 || layer >= n_layers || n_experts <= 0)
            continue;
          if (baseline_gpu_layer(layer)) {
            LOG_STREAM_INFO()
                << "[OPTION-B] Layer " << layer
                << " skipped: already covered by the complete GPU block\n";
            continue;
          }

          std::vector<uint64_t> counts(static_cast<size_t>(n_experts), 0);
          auto it = by_layer.find(layer);
          if (it != by_layer.end()) {
            for (const auto &entry : it->second) {
              if (entry.expert >= 0 && entry.expert < n_experts)
                counts[static_cast<size_t>(entry.expert)] += entry.count;
            }
          }
          std::vector<int> ranked_ids(static_cast<size_t>(n_experts));
          for (int expert = 0; expert < n_experts; ++expert)
            ranked_ids[static_cast<size_t>(expert)] = expert;
          std::stable_sort(ranked_ids.begin(), ranked_ids.end(),
                           [&](int a, int b) {
                             if (counts[static_cast<size_t>(a)] !=
                                 counts[static_cast<size_t>(b)])
                               return counts[static_cast<size_t>(a)] >
                                      counts[static_cast<size_t>(b)];
                             return a < b;
                           });

          uint64_t total_count = 0, hot_count = 0;
          for (uint64_t count : counts)
            total_count += count;
          std::vector<uint8_t> hot_mask(static_cast<size_t>(n_experts), 0);
          for (int i = 0; i < hot_n; ++i) {
            const int expert = ranked_ids[static_cast<size_t>(i)];
            hot_mask[static_cast<size_t>(expert)] = 1;
            hot_count += counts[static_cast<size_t>(expert)];
          }
          const double coverage =
              total_count == 0
                  ? 0.0
                  : static_cast<double>(hot_count) /
                        static_cast<double>(total_count);
          const bool meets_coverage =
              total_count > 0 &&
              coverage >= config.qwen4exp_option_b_min_coverage;
          if (!meets_coverage) {
            LOG_STREAM_INFO()
                << "[OPTION-B] Layer " << layer
                << " remains CPU/DDR: top-" << hot_n
                << " coverage=" << (coverage * 100.0)
                << "% threshold="
                << (config.qwen4exp_option_b_min_coverage * 100.0) << "%\n";
            continue;
          }

          engine.qwen4exp_option_b_hot_masks[layer] = hot_mask;
          engine.qwen4exp_option_b_layers.push_back(layer);
          engine.qwen4exp_option_b_layer_coverage[layer] = coverage;
          if (layer == config.qwen4exp_option_b_layer)
            engine.qwen4exp_option_b_hot_mask = hot_mask;
          LOG_STREAM_INFO()
              << "[OPTION-B] Layer " << layer << " static hot set: committed"
              << " hot=" << hot_n << " cold=" << (n_experts - hot_n)
              << " history coverage=" << (coverage * 100.0)
              << "% threshold="
              << (config.qwen4exp_option_b_min_coverage * 100.0) << "%\n";
        }
        std::sort(engine.qwen4exp_option_b_layers.begin(),
                  engine.qwen4exp_option_b_layers.end());
        engine.qwen4exp_option_b_committed =
            !engine.qwen4exp_option_b_layers.empty();
        if (config.qwen4exp_option_b_concentrated_only &&
            config.qwen4exp_option_b_layer < 0) {
          LOG_STREAM_INFO()
              << "[OPTION-B] concentrated-only selection committed "
              << engine.qwen4exp_option_b_layers.size()
              << " CPU-side layer(s) from " << config.pin_file << "\n";
        }
      }

      // Persist this SAME final ranking (post prefill_pin_file
      // override) for RAMSafetyManager's runtime disk_batch_padding
      // to use as its candidate source -- see that member's own
      // comment (ram_manager.hpp) for why the persisted, cross-run
      // .moestream_usage signal is used here instead of live
      // StreamingEngine::route_freq. Cheap: at most a few thousand
      // ints total, copied once at startup, never touched again.
      for (auto &[layer, v] : by_layer) {
        auto &dst = ram_mgr.route_rank_by_layer[layer];
        dst.reserve(v.size());
        for (auto &e : v)
          dst.push_back(e.expert);
      }

      // Real per-layer capacity for each tier -- used both by
      // pin_topk_per_layer's "auto" mode (unchanged) and as the
      // DENOMINATOR for the fraction-based tier 2/3 knobs below, so
      // "0.1" means the same relative thing (10% of that tier's own
      // real capacity) regardless of how much VRAM/RAM a given
      // deployment actually has.
      // CPU-affinity layers have no physical VRAM region and must not
      // consume a share of the VRAM preload budget.  The compact planner
      // redistributes their bytes to GPU-executed layers, so dividing the
      // same target by all model layers leaves those newly available slots
      // empty (for example, 8 CPU layers used to cap the remaining 40
      // layers at 206 pins each even though their regions could hold ~248).
      // Keep the all-layer count for RAM policy/reporting, but use this
      // GPU-only count for every VRAM budget calculation.
      const int moe_layers = std::max(1, (int)by_layer.size());
      int vram_moe_layers = 0;
      for (const auto &[layer, v] : by_layer) {
        if (!v.empty() && !is_cpu_affinity_layer(layer))
          ++vram_moe_layers;
      }
      vram_moe_layers = std::max(1, vram_moe_layers);
      double avg_vram_slots_per_layer =
          (double)ram_mgr.vram_cache->expert_vram_cache_cap /
          (double)vram_moe_layers;

      int topk = config.pin_topk_per_layer;
      if (topk <= 0) {
        // Auto-size from the REAL slot count (expert_vram_cache_cap)
        // -- moe_cache_gb in the TOML is advisory only,
        // allocate_expert_cache always uses (total_vram - 1GB) per
        // vault GPU regardless of that budget, so sizing off the
        // config value here could over- or under-preload relative to
        // what actually fits.
        topk = std::max(1, (int)avg_vram_slots_per_layer);
      }

      // Three-tier caching: Use explicit pinned_vram_capacity instead of
      // implicit pin_topk_per_layer calculation. This allows precise control
      // over the pinned tier (targeting 1024 experts vs. current ~600).
      int max_pinned_total = config.pinned_vram_capacity;
      bool global_topn = (config.pin_strategy == "global_topn");
      if (max_pinned_total > 0 && !global_topn) {
        // Adjust topk per layer to respect the global pinned cap
        // When auto-sizing, round up the average rather than flooring it.
        // The physical compact-layer cap below trims the few layers whose
        // region is smaller, allowing the preload loop to consume the final
        // remainder slots instead of stranding one slot per layer.
        if (config.pin_topk_per_layer <= 0)
          topk = std::max(topk, (int)std::ceil(avg_vram_slots_per_layer));
        topk = std::min(topk, max_pinned_total / vram_moe_layers);
        LOG_STREAM_INFO() << "[PIN] Three-tier: capped per-layer topk to "
                          << topk << " (global cap: " << max_pinned_total
                          << " experts across " << vram_moe_layers
                          << " GPU-executed layers; " << moe_layers
                          << " total MoE layers)\n";
      }

      int vram_lru_n =
          std::max(0, (int)std::lround(config.pin_vram_lru_fraction *
                                       avg_vram_slots_per_layer));
      int ram_n = std::max(0, (int)std::lround(config.pin_ram_fraction *
                                               ram_mgr.slots_per_layer));
      bool prefix_vram = config.vram_preload_prefix_layers > 0;

      // Compact VRAM has a separate, non-interchangeable slot region
      // for each layer. A prefix request is expressed as a GLOBAL
      // budget, but planning it from that budget alone can ask for
      // every expert in a layer even when that layer's physical region
      // is much smaller. The old loop then read those excess slabs from
      // disk and emitted one "no victim" error per rejected upload.
      // Use the already-built physical slot map to cap each requested
      // layer before the preload loop starts. This applies to prefix
      // placement AND global_topn: the latter used to rank globally
      // without knowing that compact VRAM has independent per-layer
      // regions, so a hot layer could consume more than its region and
      // lose the excess as a no-victim upload rejection.
      std::unordered_map<int, int> compact_layer_capacity;
      if (ram_mgr.vram_cache->expert_compact_layout) {
        for (const auto &loc : ram_mgr.vram_cache->expert_vram_cache) {
          if (loc.pool_layer >= 0)
            ++compact_layer_capacity[loc.pool_layer];
        }
      }

      // Graph paths that own separate expert storage deliberately avoid a
      // second VRAM expert pool. GLM is the exception: its reusable graph
      // aliases the ordinary VRAM cache, so keep the usage-ranked preload
      // alive and let the graph reference those same pinned slots.
      if (graph_ram_cache && !glm_graph_aliases_vram_cache) {
        topk = 0;
        max_pinned_total = 0;
        global_topn = false;
        vram_lru_n = 0;
        prefix_vram = false;
      }

      // pin_strategy=="global_topn": replace the uniform per-layer topk
      // above with a per-layer pin count derived from a SINGLE ranking
      // across every layer's historical counts, not just this layer's
      // own. Every layer first gets a guaranteed floor
      // (pin_global_floor_per_layer, so a flat layer never drops to
      // zero protection), then whatever's left of pinned_vram_capacity
      // is spent on the globally-hottest remaining entries, wherever
      // they happen to live. Valid because every layer sees the same
      // number of routing decisions per token in this engine (one
      // top-k selection per layer per token) -- raw counts really are
      // comparable across layers, unlike e.g. differently-shaped
      // tensor byte sizes. Each layer's own list is already sorted
      // descending (see above), so a layer's globally-ranked share is
      // always a CONTIGUOUS prefix starting right after its floor --
      // a higher-count entry in the same layer can never lose the
      // global ranking to a lower-count one from another layer while
      // that lower one wins, so this never leaves a "hole" in the
      // middle of a layer's own sorted list. That's what lets the
      // existing per-layer "first N by rank" loop below stay
      // unchanged -- only N itself becomes per-layer instead of a
      // single shared topk.
      std::unordered_map<int, int> layer_pin_count;
      if (prefix_vram) {
        const int prefix_end =
            std::min(config.vram_preload_prefix_layers,
                     (int)engine.model_file.architecture_config.n_layers);
        const int effective_cap =
            max_pinned_total > 0 ? max_pinned_total
                                 : (int)(avg_vram_slots_per_layer * vram_moe_layers);
        int assigned = 0;
        for (int layer = 0; layer < prefix_end; ++layer) {
          if (is_cpu_affinity_layer(layer))
            continue;
          auto it = by_layer.find(layer);
          if (it == by_layer.end())
            continue;
          int layer_capacity = (int)it->second.size();
          if (!compact_layer_capacity.empty()) {
            auto cap_it = compact_layer_capacity.find(layer);
            layer_capacity = cap_it == compact_layer_capacity.end()
                                 ? 0
                                 : std::min(layer_capacity, cap_it->second);
          }
          const int take =
              std::min(layer_capacity, std::max(0, effective_cap - assigned));
          layer_pin_count[layer] = take;
          assigned += take;
          if (assigned >= effective_cap)
            break;
        }
        LOG_STREAM_INFO() << "[PIN] vram_preload_prefix_layers: layers [0, "
                          << prefix_end << ") assigned " << assigned << "/"
                          << effective_cap
                          << " VRAM slots in ascending layer order\n";
      } else if (global_topn) {
        int floor_n = std::max(0, config.pin_global_floor_per_layer);
        int floor_total = 0;
        for (auto &[layer, v] : by_layer) {
          if (is_cpu_affinity_layer(layer)) {
            layer_pin_count[layer] = 0;
            continue;
          }
          int f = std::min(floor_n, (int)v.size());
          if (!compact_layer_capacity.empty()) {
            auto cap_it = compact_layer_capacity.find(layer);
            f = cap_it == compact_layer_capacity.end()
                    ? 0
                    : std::min(f, cap_it->second);
          }
          layer_pin_count[layer] = f;
          floor_total += f;
        }
        struct GlobalCand {
          int layer;
          uint64_t count;
        };
        std::vector<GlobalCand> remainder;
        for (auto &[layer, v] : by_layer) {
          if (is_cpu_affinity_layer(layer))
            continue;
          int f = layer_pin_count[layer];
          for (int i = f; i < (int)v.size(); ++i)
            remainder.push_back({layer, v[i].count});
        }
        // stable_sort for the same reason as the merge block above.
        // Residual, NOT fixed here (separate, smaller-impact issue,
        // out of scope for the regression this fix targets): `by_layer`
        // is an unordered_map, so the for-loop feeding `remainder`
        // above already iterates layers in hash-bucket order, not a
        // deterministic one -- stable_sort preserves whatever that
        // input order was for count-ties, but doesn't make the input
        // order itself reproducible run-to-run. A fully deterministic
        // global ranking would need GlobalCand to also carry
        // `expert` and use (layer,expert) as an explicit tie-break
        // key, not just rely on input order -- a real, if lower-
        // priority, follow-up.
        std::stable_sort(remainder.begin(), remainder.end(),
                         [](const GlobalCand &a, const GlobalCand &b) {
                           return a.count > b.count;
                         });
        // If pinned_vram_capacity isn't set, fall back to the REAL VRAM
        // pool size (same quantity the per_layer_topk path's own "auto"
        // topk sizing uses: avg_vram_slots_per_layer * moe_layers) --
        // NOT "take everything remaining", which could try to pin every
        // expert ever routed to and blow far past physical VRAM capacity.
        int effective_cap = (max_pinned_total > 0)
                                ? max_pinned_total
                                : (int)(avg_vram_slots_per_layer * vram_moe_layers);
        int remaining_budget = std::max(0, effective_cap - floor_total);
        int take = 0;
        for (const GlobalCand &candidate : remainder) {
          if (take >= remaining_budget)
            break;
          int layer_cap = std::numeric_limits<int>::max();
          if (!compact_layer_capacity.empty()) {
            auto cap_it = compact_layer_capacity.find(candidate.layer);
            layer_cap =
                cap_it == compact_layer_capacity.end() ? 0 : cap_it->second;
          }
          int &assigned = layer_pin_count[candidate.layer];
          if (assigned >= layer_cap)
            continue;
          ++assigned;
          ++take;
        }
        LOG_STREAM_INFO() << "[PIN] global_topn: floor " << floor_n
                          << "/layer (" << floor_total << " total) + " << take
                          << " globally-ranked -> " << (floor_total + take)
                          << "/" << effective_cap
                          << (max_pinned_total > 0
                                  ? ""
                                  : " (pinned_vram_capacity unset -- capped to "
                                    "real VRAM pool size instead)")
                          << " target pins across " << vram_moe_layers
                          << " GPU-executed layer(s) (" << moe_layers
                          << " total MoE layers)\n";
      }

      // pin_core_gpu_top_n: of the experts layer_pin_count[] just
      // decided WILL be pinned, redirect the globally-hottest N of
      // them (by real count, across every layer) to land specifically
      // on the CoreEngine/dual-role GPU instead of wherever the
      // default unrestricted scan would put them -- see this field's
      // own EngineConfig comment for the full rationale. Only
      // meaningful under global_topn, since that's the only strategy
      // that already produces a coherent cross-layer ranking here
      // (per_layer_topk's own layer_max is decided inline, order-
      // dependent on the running total_pinned -- not worth
      // reconstructing a second ranking for a strategy this project
      // isn't actually running).
      int pinned = 0, vram_lru = 0, ram_preloaded = 0, attempted = 0;
      int pin_core_gpu_redirected = 0, pin_core_gpu_redirect_fallback = 0;
      int total_pinned = 0; // Three-tier: track total pinned count
      if (global_topn && config.pin_core_gpu_top_n > 0 &&
          gpu.topology.core_gpu >= 0 &&
          std::find(gpu.topology.vault_gpus.begin(),
                    gpu.topology.vault_gpus.end(),
                    gpu.topology.core_gpu) != gpu.topology.vault_gpus.end()) {
        struct FlatCand {
          int layer;
          int expert;
          uint64_t count;
        };
        std::vector<FlatCand> flat;
        for (auto &[layer, v] : by_layer) {
          int lim = std::min(layer_pin_count[layer], (int)v.size());
          for (int i = 0; i < lim; ++i)
            flat.push_back({layer, v[i].expert, v[i].count});
        }
        std::stable_sort(flat.begin(), flat.end(),
                         [](const FlatCand &a, const FlatCand &b) {
                           return a.count > b.count;
                         });
        int core_take = std::min((int)flat.size(), config.pin_core_gpu_top_n);
        LOG_STREAM_INFO()
            << "[PIN] pin_core_gpu_top_n: redirecting " << core_take
            << " of the globally-hottest already-pinned experts to land on the "
               "CoreEngine GPU (gpu="
            << gpu.topology.core_gpu
            << ")'s own dual-role "
               "slice specifically, instead of the default (unrestricted) "
               "placement -- "
               "falls back to unrestricted per-expert if that GPU's own slots "
               "fill up "
               "before all "
            << core_take << " fit\n";
        // Do this as its own DEDICATED pass, RIGHT NOW, before the
        // main by_layer loop below (Tier1/Tier2) ever runs -- not
        // just a check inside that loop. Reason, found via a real
        // run before trusting this: the main loop processes layers
        // in unordered_map hash-bucket order and does Tier2's own
        // UNRESTRICTED uploads for every layer along the way -- by
        // the time a later-processed layer's marked candidate is
        // reached, Tier2's own (unmarked) uploads for EARLIER-
        // processed layers can already have accidentally consumed
        // the CoreEngine GPU's tiny slot count via nothing more
        // than iteration-order luck (evict_lru's unrestricted scan
        // takes the first empty slot it finds, regardless of GPU).
        // Confirmed real: a first attempt with the check only
        // inside the main loop landed just 5 of 61 requested on
        // GPU1. Running this pass FIRST, before anything else has
        // touched that GPU's slots, removes the race entirely.
        // Simple per-item load (not the chunked io_uring batching
        // the main loop uses below) -- core_take is small (bounded
        // by the CoreEngine GPU's own tiny slot count), not worth
        // the extra machinery for a few dozen one-time reads.
        for (int i = 0; i < core_take; ++i) {
          // NOT counted into attempted/pinned/total_pinned here --
          // the main by_layer loop below will reach this exact
          // (layer,expert) again in its own normal position and
          // count it there (upload_expert's find_expert() returns
          // this SAME cached location, pin_expert() is safely
          // re-callable), which is the single source of truth for
          // "how many distinct experts are pinned" against
          // pinned_vram_capacity. Counting here too would double-
          // count these 61 and make the final tally read >1000
          // pinned when only 1000 distinct experts really are.
          MoEExpertBlock s;
          engine.load_expert_slab(flat[i].layer, flat[i].expert, s);
          if (!s.slab_buffer || s.slab_capacity == 0) {
            // load_expert_slab may have persistently pinned this
            // temporary VRAM-promotion slab.  Route cleanup
            // through the engine owner so CUDA unregisters the
            // range before the allocation is freed.
            engine.release_slab(s);
            continue; // counted as a normal disk-read miss when the main loop
                      // retries it
          }
          ExpertLocation *loc = gpu.topology.upload_expert(
              flat[i].layer, flat[i].expert, s.slab_buffer, s.slab_capacity,
              gpu.topology.core_gpu);
          if (loc) {
            gpu.topology.mark_startup_preloaded(flat[i].layer, flat[i].expert);
            ++pin_core_gpu_redirected;
          } else {
            ++pin_core_gpu_redirect_fallback; // that GPU's own slots are
                                              // already full
          }
          engine.release_slab(s);
        }
      }

      // Concentration diagnostic: pinning a FIXED top-K per layer only
      // helps future (unseen) prompts to the extent that layer's
      // historical routing is actually concentrated in that top-K --
      // a layer whose traffic is spread near-uniformly across most of
      // its 256 experts gains little from pinning any fixed subset,
      // since a new prompt is about as likely to route to an
      // unpinned expert as a pinned one. Computed here (not just
      // logged as a vague "top-heavy vs flat" guess) using the SAME
      // sorted-by-count vectors the preload loop below already builds
      // -- coverage = (sum of this layer's top-`topk` counts) /
      // (sum of ALL of this layer's recorded counts), i.e. "what
      // fraction of this layer's historical traffic would a HIT
      // against the pinned set actually explain". min/max identify
      // the single most concentrated and most uniform layers; the
      // full per-layer table (DEBUG level, high volume) lets you spot
      // whether flat layers cluster in a particular part of the
      // network (e.g. early vs late layers) rather than being random.
      {
        double coverage_sum = 0.0;
        int coverage_n = 0;
        double coverage_min = 2.0, coverage_max = -1.0;
        int layer_min = -1, layer_max = -1;
        for (auto &[layer, v] : by_layer) {
          uint64_t layer_total = 0;
          for (auto &e : v)
            layer_total += e.count;
          if (layer_total == 0)
            continue;
          uint64_t topk_sum = 0;
          int k = is_cpu_affinity_layer(layer)
                      ? 0
                      : ((global_topn || prefix_vram)
                             ? std::min(layer_pin_count[layer], (int)v.size())
                             : std::min(topk, (int)v.size()));
          for (int i = 0; i < k; ++i)
            topk_sum += v[i].count;
          double coverage = (double)topk_sum / (double)layer_total;
          coverage_sum += coverage;
          ++coverage_n;
          if (coverage < coverage_min) {
            coverage_min = coverage;
            layer_min = layer;
          }
          if (coverage > coverage_max) {
            coverage_max = coverage;
            layer_max = layer;
          }
          LOG_STREAM_DEBUG() << "[PIN-CONCENTRATION] layer=" << layer << " top-"
                             << k << " covers " << (coverage * 100.0) << "% of "
                             << v.size() << " distinct experts seen / "
                             << layer_total << " total historical selections\n";
        }
        if (coverage_n > 0) {
          LOG_STREAM_INFO()
              << "[PIN] "
              << (prefix_vram
                      ? "prefix-layer preload"
                      : (global_topn
                             ? "global_topn (per-layer K varies, see above)"
                             : ("Top-" + std::to_string(topk) + "/layer")))
              << " concentration: avg " << (coverage_sum / coverage_n * 100.0)
              << "% of each layer's historical "
              << "traffic covered by its pinned set -- most concentrated: "
                 "layer "
              << layer_max << " (" << (coverage_max * 100.0)
              << "%), most uniform: layer " << layer_min << " ("
              << (coverage_min * 100.0) << "%). A layer near the "
              << "low end gains little from pinning -- its routing is close to "
                 "evenly "
              << "spread, so a new prompt is about as likely to hit an "
                 "unpinned expert "
              << "as a pinned one. Full per-layer breakdown at log_level=DEBUG "
              << "([PIN-CONCENTRATION] lines).\n";
        }
      }

      // Batched io_uring preload (see preload_batch_load's own
      // comment for why): a dedicated Ring, separate from
      // RAMSafetyManager's decode_uring_ring/LockFreePrefetcher's
      // per-worker rings -- this whole [PIN] pass runs single-
      // threaded on the main thread before those are ever touched
      // (prefetcher.start() happens after this block, same as the
      // Unified RAM-tier design's own startup-ordering rule), so no
      // shared-ownership concern, just a plain local Ring.
      constexpr int PRELOAD_URING_CHUNK =
          (int)(io_uring_backend::Ring::CAPACITY / 3); // 21
      io_uring_backend::Ring preload_ring;
      // Keep a deterministic escape hatch for diagnosing model-specific
      // preload failures.  The normal path uses batched reads, while a
      // disabled batch ring exercises the original one-expert loader
      // without changing runtime inference behavior.
      const bool disable_preload_batch =
          std::getenv("MOESTREAM_DISABLE_PRELOAD_BATCH") != nullptr;
      bool preload_ring_ok = !disable_preload_batch && preload_ring.init();
      if (preload_ring_ok) {
        preload_ring.register_iowq_workers(
            config.expert_fetch_threads > 0
                ? (unsigned)config.expert_fetch_threads
                : 8u);
      }
      int preload_uring_chunks_ok = 0, preload_uring_chunks_fallback = 0;
      LOG_STREAM_INFO()
          << "[PIN] Batched io_uring preload: "
          << (preload_ring_ok
                  ? ("ring ready, " + std::to_string(PRELOAD_URING_CHUNK) +
                     " experts/submission")
                  : std::string(
                        "ring unavailable, falling back to the original "
                        "single-expert-at-a-time load for this whole pass"))
          << "\n";

      LOG_STREAM_INFO()
          << "[PIN] Preloading from " << config.pin_file << " ("
          << entries.size() << " known experts, max count " << maxc << "): "
          << (prefix_vram ? "prefix-layer ranking (see above)"
                          : (global_topn ? "global_topn ranking (see above)"
                                         : ("top-" + std::to_string(topk) +
                                            "/layer VRAM-pinned")))
          << ", next " << (prefix_vram ? 0 : vram_lru_n)
          << "/layer VRAM-unpinned, "
          << (config.ram_preload_all_experts
                  ? "all experts within each layer's RAM cap"
                  : ("next " + std::to_string(ram_n) + "/layer RAM"))
          << "...\n";
      // Previously, a preload attempt failing anywhere in this loop
      // (disk read producing no usable slab, or upload_expert
      // rejecting it) was completely silent -- a real run lost
      // ~99% of a 3300-attempt preload with zero evidence of WHERE
      // in the pipeline it failed. These distinguish "disk read
      // gave us nothing" from "GPU upload rejected a good slab"
      // (now itself logged in detail by upload_expert -- see
      // n_upload_expert_no_victim/n_upload_expert_memcpy_fail).
      int tier1_disk_fail = 0, tier1_upload_fail = 0;
      int tier2_disk_fail = 0, tier2_upload_fail = 0;
      for (auto &[layer, v] : by_layer) {
        // Defensive second barrier: all records are sanitized above,
        // but this loop is the last point before direct indexing into
        // layers_experts and should remain safe if a future manifest
        // augmentation adds entries from another source.
        if (layer < 0 || (size_t)layer >= engine.layers_experts.size() ||
            engine.layers_experts[(size_t)layer].empty()) {
          LOG_STREAM_ERROR() << "[PIN-BOUNDS] Skipping preload layer " << layer
                             << " from " << config.pin_file
                             << " immediately before cache indexing: valid "
                                "layer vectors are [0,"
                             << engine.layers_experts.size()
                             << ") and this layer has no expert slots.\n";
          continue;
        }
        int n = (int)v.size();
        // A complete prefix layer is intentionally placed in expert
        // ID order. Its membership is unchanged (all experts in
        // the layer are still loaded and pinned), but the physical
        // VRAM slots then form the regular [expert 0, expert 1, ...]
        // layout that GGML's 3-D MUL_MAT_ID tensor can view directly.
        // evict_lru takes the first available empty slot, while
        // pin_expert removes that slot from the free-list with
        // swap-and-pop. Therefore ascending IDs are required here:
        // expert 0 must occupy the first physical slot, expert 1 the
        // next one, and so on.
        // Partial prefix layers keep the historical frequency order
        // because they cannot use the direct resident-layer view.
        const bool complete_prefix_layer =
            prefix_vram && layer_pin_count[layer] >= n && n > 0;
        if (complete_prefix_layer) {
          std::stable_sort(
              v.begin(), v.end(),
              [](const auto &a, const auto &b) { return a.expert < b.expert; });
        }
        // cpu_affinity_layers (config): never pin or pre-warm VRAM
        // for these layers -- Tier 1/2 below both call
        // gpu.topology.upload_expert, so forcing layer_max=0 and
        // this layer's own vram_lru budget to 0 skips both loop
        // bodies entirely (their bounds become empty ranges) while
        // leaving Tier 3 (host RAM preload) untouched below --
        // that's the "MoE load into host DDR instead of DMA into
        // VRAM" contract for this layer. Also starts Tier 3 at rank
        // 0 (the layer's OWN hottest experts) instead of after a
        // pinned/vram_lru prefix that never happened for this layer.
        bool cpu_affinity = is_cpu_affinity_layer(layer);
        const bool option_b_active =
            engine.qwen4exp_option_b_committed &&
            config.qwen4exp_option_b_top_n > 0;
        const bool option_b_target =
            option_b_active && engine.qwen4exp_option_b_layer_enabled(layer);
        const bool option_b_graph_target =
            option_b_target && graph_mode && graph_ram_cache;
        int layer_vram_lru_n =
            (cpu_affinity || prefix_vram || option_b_target) ? 0
                                                             : vram_lru_n;
        // Three-tier: enforce global pinned capacity per layer.
        // global_topn already computed a final per-layer count
        // (floor + its share of the global ranking) up front; the
        // default per_layer_topk path keeps its EXACT original
        // running-total cap, unchanged.
        int layer_max;
        if (cpu_affinity || (option_b_active && !option_b_target)) {
          layer_max = 0;
        } else if (option_b_target) {
          // The full graph owns the selected hot experts in its compact
          // device arena. Do not also try to upload them through the legacy
          // cache (which is normally zero-capacity in the 22/26 topology).
          layer_max = option_b_graph_target
                          ? 0
                          : std::min(config.qwen4exp_option_b_top_n, n);
        } else if (global_topn || prefix_vram) {
          layer_max = layer_pin_count[layer];
        } else {
          layer_max = topk;
          if (max_pinned_total > 0) {
            layer_max = std::min(
                {topk, (int)v.size(), max_pinned_total - total_pinned});
          }
        }
        // Compact VRAM has a separate physical region per layer.  The
        // auto-sized top-k may round up by one, so never ask a region to
        // accept more experts than the planner actually allocated for it.
        if (!compact_layer_capacity.empty()) {
          auto cap_it = compact_layer_capacity.find(layer);
          layer_max = std::min(
              layer_max,
              cap_it == compact_layer_capacity.end() ? 0 : cap_it->second);
        }
        // Tier 2/3 start right after however many THIS layer got
        // pinned -- for per_layer_topk that's always exactly topk
        // (identical to the original code), but for global_topn it
        // varies per layer, so it can't be the shared topk anymore.
        // A cpu_affinity layer's layer_max is already 0, so
        // tier2_start correctly comes out to 0 too via the
        // global_topn branch (layer_max) regardless of topk.
        int tier2_start =
            (cpu_affinity || option_b_active || global_topn || prefix_vram)
                ? layer_max
                : topk;
        for (int i = 0; i < n && i < layer_max;) {
          int chunk_end =
              std::min(n, std::min(layer_max, i + PRELOAD_URING_CHUNK));
          int chunk_n = chunk_end - i;
          std::vector<MoEExpertBlock> scratch(chunk_n);
          bool batch_ok = false;
          if (preload_ring_ok) {
            std::vector<std::tuple<int, int, MoEExpertBlock *>> items;
            items.reserve(chunk_n);
            for (int j = 0; j < chunk_n; ++j)
              items.emplace_back(v[i + j].layer, v[i + j].expert, &scratch[j]);
            batch_ok = preload_batch_load(engine, preload_ring, items);
          }
          if (batch_ok) {
            ++preload_uring_chunks_ok;
          } else {
            ++preload_uring_chunks_fallback;
            for (int j = 0; j < chunk_n; ++j)
              engine.load_expert_slab(v[i + j].layer, v[i + j].expert,
                                      scratch[j]);
          }
          for (int j = 0; j < chunk_n; ++j) {
            ++attempted;
            MoEExpertBlock &s = scratch[j];
            if (s.slab_buffer && s.slab_capacity > 0) {
              // Plain unrestricted upload -- any expert this
              // pass reaches that pin_core_gpu_top_n already
              // placed on the CoreEngine GPU (the dedicated
              // pre-pass above) is ALREADY cached, so
              // upload_expert's own find_expert() check
              // returns that same location immediately here;
              // no need to re-check pin_to_core_gpu or pass a
              // target_gpu_id a second time (that would just
              // double-count the redirect diagnostics below
              // without changing where anything lands --
              // confirmed real on a first attempt at this).
              ExpertLocation *loc = gpu.topology.upload_expert(
                  v[i + j].layer, v[i + j].expert, s.slab_buffer,
                  s.slab_capacity, -1, true);
              if (loc &&
                  gpu.topology.pin_expert(v[i + j].layer, v[i + j].expert)) {
                gpu.topology.mark_startup_preloaded(v[i + j].layer,
                                                    v[i + j].expert);
                ++pinned;
                ++total_pinned; // Three-tier: track global pinned count
              } else if (!loc) {
                ++tier1_upload_fail;
              }
            } else {
              ++tier1_disk_fail;
            }
            engine.release_slab(s);
          }
          i = chunk_end;
        }
        // Tier 2: next vram_lru_n ranks -> VRAM, but NOT pinned --
        // a normal reactively-evictable slot, just pre-warmed
        // instead of starting cold. Reuses the exact same
        // upload_expert() call as Tier 1, just skips pin_expert().
        int tier2_end = std::min(n, tier2_start + layer_vram_lru_n);
        for (int i = tier2_start; i < tier2_end;) {
          int chunk_end = std::min(tier2_end, i + PRELOAD_URING_CHUNK);
          int chunk_n = chunk_end - i;
          std::vector<MoEExpertBlock> scratch(chunk_n);
          bool batch_ok = false;
          if (preload_ring_ok) {
            std::vector<std::tuple<int, int, MoEExpertBlock *>> items;
            items.reserve(chunk_n);
            for (int j = 0; j < chunk_n; ++j)
              items.emplace_back(v[i + j].layer, v[i + j].expert, &scratch[j]);
            batch_ok = preload_batch_load(engine, preload_ring, items);
          }
          if (batch_ok) {
            ++preload_uring_chunks_ok;
          } else {
            ++preload_uring_chunks_fallback;
            for (int j = 0; j < chunk_n; ++j)
              engine.load_expert_slab(v[i + j].layer, v[i + j].expert,
                                      scratch[j]);
          }
          for (int j = 0; j < chunk_n; ++j) {
            ++attempted;
            MoEExpertBlock &s = scratch[j];
            if (s.slab_buffer && s.slab_capacity > 0) {
              if (gpu.topology.upload_expert(v[i + j].layer, v[i + j].expert,
                                             s.slab_buffer, s.slab_capacity, -1,
                                             true)) {
                gpu.topology.mark_startup_preloaded(v[i + j].layer,
                                                    v[i + j].expert);
                ++vram_lru;
              } else {
                ++tier2_upload_fail;
              }
            } else {
              ++tier2_disk_fail;
            }
            engine.release_slab(s);
          }
          i = chunk_end;
        }
        // Tier 3: the next ranked experts -> host RAM tier directly.
        // In ram_preload_all_experts mode, use the entire effective
        // per-layer cap.  The old fixed ram_n fraction silently
        // limited this path to ~130 experts/layer even when the safe
        // RAM cache cap was materially larger, which left DDR bytes
        // unused and caused avoidable SSD misses.  Both modes remain
        // bounded by the layer's byte-aware cap and the slot-array
        // size; writes start at slot 0 while the cache is empty.
        int ram_layer_cap = std::min(ram_mgr.cap_for_layer(layer),
                                     (int)engine.layers_experts[layer].size());
        int ram_cap = config.ram_preload_all_experts
                          ? ram_layer_cap
                          : std::min(ram_n, ram_layer_cap);
        if (graph_ram_cache && !engine.graph_ram_layer_allocated(layer))
          ram_cap = 0;
        // 2026-08-29 (EngineConfig::ram_pin_eager_preregister):
        // ram_n itself is a FIXED fraction of slots_per_layer
        // (pin_ram_fraction), unrelated to how much of
        // ram_pinned_budget_gb is actually still available -- in
        // practice this routinely leaves real pin budget unused
        // (a real run: ~12.6GB of a 45GB budget). Extends ram_cap
        // further down this layer's OWN real ranking (v, already
        // sorted by historical count) using the SAME load path
        // just below, spending whatever budget room remains on
        // MORE real, historically-plausible experts -- not blind
        // capacity (see the reverted eager_preregister_remaining_
        // slots entry above this block for why that distinction
        // matters: an earlier generic-slot version of this idea
        // measurably backfired). Reading ram_pinned_bytes_used
        // fresh per layer means budget spent by earlier layers is
        // correctly reflected for later ones -- no separate
        // tracking needed, the atomic already does it.
        if (config.ram_pin_eager_preregister &&
            config.ram_pinned_budget_gb > 0.0 && ram_cap < ram_layer_cap) {
          size_t per_expert_bytes = engine.compute_expert_slab_bytes(layer);
          if (per_expert_bytes > 0) {
            size_t used =
                engine.ram_pinned_bytes_used.load(std::memory_order_relaxed);
            size_t budget = engine.ram_pinned_budget_bytes;
            size_t remaining = (used < budget) ? (budget - used) : 0;
            int extra = (int)(remaining / per_expert_bytes);
            if (extra > 0) {
              ram_cap = std::min(ram_cap + extra, ram_layer_cap);
            }
          }
        }
        // 2026-09-10: was `tier2_start + layer_vram_lru_n` -- RAM
        // tier's own preload started AFTER however many ranks VRAM's
        // pinned + LRU-prewarm tiers already consumed from this
        // SAME layer's ranked list, i.e. RAM and VRAM tiers were
        // treated as necessarily DISJOINT. Real, measured, TWICE-
        // confirmed incident: raising VRAM's own per-layer share
        // (pin_vram_lru_fraction=1.0, then separately
        // vram_preload_prefix_layers=15 with a raised
        // pinned_vram_capacity) both starved RAM tier for the exact
        // layers VRAM took the most from -- one run lost 9984 RAM
        // preloads (24528->14544, cache 100%->78%, real disk I/O
        // reappeared), a later one lost 7632 (24528->16896, cache
        // 100%->74%, 3.12 GB of real NVMe reads, prefill 74.18->
        // 22.53 tok/s, a 3.3x regression) purely from this.
        // Root cause: MoEStream's own RAM and VRAM tiers serve
        // mostly DIFFERENT consumers here (this session's own
        // investigation found qwen4exp's prefill dispatch path
        // sources almost everything from the RAM tier via
        // ram_direct/ram_staged and only checks VRAM residency via
        // a narrow per-expert D2D lookup -- see execution_layer.cpp
        // around GgmlMoeRunner::copy_expert_d2d) -- an expert being
        // VRAM-resident is not a reason to deny it a RAM-tier copy
        // too, and RAM has vastly more headroom (100 GB here vs
        // 17 GB of VRAM) to just hold both. cpu_affinity_layers
        // already special-cased exactly this (tier2_start forced
        // to 0 for those layers specifically, see this loop's own
        // comment above) -- this generalizes that same, already-
        // proven-safe pattern to every layer, unconditionally,
        // instead of only the cpu_affinity case.
        int ram_start_i = 0;
        int ram_count = std::min(std::max(0, n - ram_start_i), ram_cap);
        for (int off = 0; off < ram_count;) {
          int chunk_n = std::min(PRELOAD_URING_CHUNK, ram_count - off);
          bool batch_ok = false;
          if (preload_ring_ok) {
            std::vector<std::tuple<int, int, MoEExpertBlock *>> items;
            items.reserve(chunk_n);
            for (int j = 0; j < chunk_n; ++j) {
              int i = ram_start_i + off + j;
              // A graph RAM arena has one immutable slice per
              // expert ID. The normal preload path writes the
              // ranked list densely, which is correct for the
              // LRU cache but gives a shared arena the wrong
              // identity-to-slice mapping. Use the expert ID
              // as the slot whenever an arena is allocated.
              int slot_idx = engine.graph_ram_layer_allocated(layer)
                                 ? v[i].expert
                                 : off + j;
              items.emplace_back(v[i].layer, v[i].expert,
                                 &engine.layers_experts[layer][slot_idx]);
            }
            batch_ok = preload_batch_load(engine, preload_ring, items);
          }
          if (batch_ok) {
            ++preload_uring_chunks_ok;
          } else {
            ++preload_uring_chunks_fallback;
            for (int j = 0; j < chunk_n; ++j) {
              int i = ram_start_i + off + j;
              int slot_idx = engine.graph_ram_layer_allocated(layer)
                                 ? v[i].expert
                                 : off + j;
              engine.load_expert_slab(v[i].layer, v[i].expert,
                                      engine.layers_experts[layer][slot_idx]);
            }
          }
          for (int j = 0; j < chunk_n; ++j) {
            int i = ram_start_i + off + j;
            int slot_idx =
                engine.graph_ram_layer_allocated(layer) ? v[i].expert : off + j;
            ++attempted;
            MoEExpertBlock &slot = engine.layers_experts[layer][slot_idx];
            if (slot.slab_buffer && slot.slab_capacity > 0) {
              ++ram_preloaded;
              // Goes through RAMSafetyManager::preload_ram_slot
              // (takes ram_mutex internally) rather than poking
              // last_used_clock/hit_count/ram_expert_index
              // directly -- closes a real gap: this used to be
              // safe only because LockFreePrefetcher's worker
              // threads happened to not exist yet at this exact
              // point in startup (prefetcher.start() is now called
              // AFTER this whole [PIN] block finishes, specifically
              // so that timing argument becomes unnecessary rather
              // than load-bearing). preload_ram_slot also inserts
              // the ram_expert_index entry this write needs --
              // without it, every preloaded expert's first runtime
              // lookup would look like a genuine index miss
              // (indistinguishable from "really not resident"),
              // see that function's own comment.
              uint32_t ram_seed_hit_count = 0;
              if (config.ram_preload_seed_hit_count) {
                // Clamp below promote_hit_threshold so this only
                // biases RAM-tier retention, never fires a
                // reactive VRAM promotion before the slot has
                // actually been touched this session -- see
                // ram_preload_seed_hit_count's own comment.
                uint32_t cap = ram_mgr.promote_hit_threshold > 0
                                   ? ram_mgr.promote_hit_threshold - 1
                                   : 0;
                ram_seed_hit_count =
                    (uint32_t)std::min<uint64_t>(v[i].count, cap);
              }
              ram_mgr.preload_ram_slot(engine, v[i].layer, v[i].expert,
                                       (size_t)slot_idx, ram_seed_hit_count);
            }
          }
          off += chunk_n;
        }
      }
      LOG_STREAM_INFO() << "[PIN] Batched io_uring preload: "
                        << preload_uring_chunks_ok << " chunk(s) batched / "
                        << preload_uring_chunks_fallback
                        << " chunk(s) fell back to the original "
                           "single-expert-at-a-time load (ring "
                           "unavailable, a non-packed-format layer, or a "
                           "short/failed read -- see "
                           "preload_batch_load's own comment)\n";
      LOG_STREAM_INFO() << "[PIN] Preloaded " << pinned << " VRAM-pinned + "
                        << vram_lru << " VRAM-unpinned + " << ram_preloaded
                        << " RAM (of " << attempted << " attempted).\n";
      if (pin_core_gpu_redirected || pin_core_gpu_redirect_fallback) {
        LOG_STREAM_INFO() << "[PIN] pin_core_gpu_top_n: "
                          << pin_core_gpu_redirected
                          << " expert(s) landed on the CoreEngine GPU (gpu="
                          << gpu.topology.core_gpu << ") as intended, "
                          << pin_core_gpu_redirect_fallback
                          << " fell back to the default placement (that GPU's "
                             "own slots were full "
                             "by the time they were reached)\n";
      }
      if (tier1_disk_fail || tier1_upload_fail || tier2_disk_fail ||
          tier2_upload_fail) {
        LOG_STREAM_WARN()
            << "[PIN] Preload attrition -- Tier1(pinned): " << tier1_disk_fail
            << " disk-read failures, " << tier1_upload_fail
            << " GPU-upload rejections | "
            << "Tier2(unpinned): " << tier2_disk_fail << " disk-read failures, "
            << tier2_upload_fail << " GPU-upload rejections "
            << "(see [MOE-VRAM] upload_expert error lines above for the "
               "specific "
            << "cudaMemcpy/evict_lru cause of any upload rejections).\n";
      }

      // Three-tier caching: Log final pinned count
      if (max_pinned_total > 0) {
        const int physical_vram_target = std::min(
            max_pinned_total,
            (int)std::min<size_t>(
                ram_mgr.vram_cache->expert_vram_cache_cap,
                (size_t)std::numeric_limits<int>::max()));
        LOG_STREAM_INFO() << "[PIN] Three-tier: " << total_pinned
                          << " experts pinned (target: " << physical_vram_target
                          << ", configured: " << max_pinned_total
                          << ") across " << vram_moe_layers
                          << " GPU-executed layers (" << moe_layers
                          << " total MoE layers; CPU-affinity layers excluded) ("
                          << (total_pinned >= physical_vram_target
                                  ? "target met"
                                  : "below target")
                          << ")\n";
      }
    }
    ram_mgr.vram_cache->startup_preload_phase = false;

    if (ram_graph_cache) {
      int allocated = 0;
      int ready = 0;
      const int n_graph_layers = engine.model_file.architecture_config.n_layers;
      for (int layer = 0; layer < n_graph_layers; ++layer) {
        if (engine.graph_ram_layer_allocated(layer)) {
          ++allocated;
          if (engine.graph_ram_layer_ready(layer))
            ++ready;
        }
      }
      LOG_STREAM_INFO() << "[GRAPH-RAM] post-preload: " << ready << "/"
                        << allocated
                        << " arena layer(s) have the required expert-ID"
                           " mapping and host tensor views\n";

    }
  }


  // 2026-08-29: EngineConfig::ram_pin_eager_preregister's FIRST
  // implementation lived here -- RAMSafetyManager::eager_preregister_
  // remaining_slots, which eagerly grew+registered GENERIC (identity-
  // agnostic) empty slots for whatever ram_pinned_budget_gb the
  // ranking-informed [PIN] preload above left unused. Real A/B: it made
  // things WORSE, not better (active_up 0.76-0.81s -> 1.18-1.20s,
  // Active-layer source-pin coverage 62.7% -> 47.7%, "stayed pageable"
  // 808 -> 1232 slabs). Root cause: it's still greedy pre-allocation
  // against the SAME finite, shared budget the ranking-informed preload
  // and live demand-driven registration were already using efficiently
  // -- but UNINFORMED (no signal about which specific expert would ever
  // land in a given slot), so it displaced better-targeted allocations
  // with worse ones for the same fixed spend. Function kept in
  // ram_manager.hpp (unused, default off) per this project's own
  // convention for a correctly-diagnosed negative result -- see its own
  // comment and CLAUDE.md for the full A/B. Superseded by the extended,
  // ranking-informed Tier 3 preload below (same config flag, same
  // budget, but spends it on MORE real, historically-ranked experts
  // instead of blind slots).

  // Spun up here, AFTER the [PIN] block above has fully finished writing
  // into engine.layers_experts (Tier-3 RAM preload) and gpu.topology
  // (Tier-1/Tier-2 VRAM preload) -- worker threads claim/populate/publish
  // directly into that same RAM-tier structure via RAMSafetyManager::
  // claim_or_wait (see async_prefetcher.hpp's active_prefetch), so
  // starting them only once startup preload is done removes any need to
  // reason about ordering between the two, rather than relying on "no
  // push_hint call happens until the decode loop starts" as an implicit
  // timing argument.
  LOG_STREAM_INFO()
      << "[I/O] Spinning up asynchronous lock-free SPSC prefetch threads...\n";
  prefetcher.start(&engine, &gpu.topology, config.io_uring,
                   config.prefetch_worker_threads);
  // Unified claim/publish design: the prefetcher calls INTO
  // RAMSafetyManager::claim_or_wait, so it needs a pointer back to it --
  // the reverse of the old (now-removed) RAMSafetyManager::prefetcher
  // field. Raw, non-owning: ram_mgr outlives prefetcher for the whole
  // process lifetime (both are stack-local to main(), ram_mgr declared
  // first).
  prefetcher.ram_mgr = &ram_mgr;

  // 2026-08-28: active-layer upload worker pool -- see EngineConfig::
  // active_layer_upload_worker_threads' own comment and active_layer_
  // upload_workers.hpp's header comment for the full design. Constructed
  // here (after gpu.topology's streams/pool exist, same ordering
  // constraint the prefetcher's own start() call above already respects)
  // rather than at the ram_mgr/prefetcher declaration site above, since
  // the pool's ctor immediately spawns worker threads that will touch
  // gpu.topology.devices[...] the moment real work arrives.
  if (config.active_layer_upload_worker_threads > 0) {
    upload_worker_pool = std::make_unique<ActiveLayerUploadWorkers>(
        gpu.topology, ram_mgr, config.active_layer_upload_worker_threads);
    ram_mgr.upload_workers = upload_worker_pool.get();
    LOG_STREAM_INFO() << "[GPU] Active-layer upload worker pool: "
                      << config.active_layer_upload_worker_threads
                      << " thread(s) started.\n";
  }

  // Dequant attn_k_b/attn_v_b (MLA's absorbed-attention weights) for
  // every layer NOW, once, model load fully done but before any prompt
  // is timed -- see warm_mla_absorb_cache's own comment
  // (attention_layer.cpp) for the real measurement that motivated this
  // (1.59s previously landing inside the FIRST prefill call's own
  // "absorb" sub-phase, confirmed to be a one-time, model-only cost
  // completely independent of the prompt). No-op/harmless for
  // architectures that don't use MLA-absorbed attention (DeepSeek-V4's
  // own dsv4_attention_gguf path uses a structurally separate woa_cache
  // this doesn't touch; the function itself early-returns for any model
  // with kv_lora<=0).
  extern void warm_mla_absorb_cache(StreamingEngine & engine);
  warm_mla_absorb_cache(engine);

  // The full GLM-5.3-Flash graph needs a host-side source view of the
  // routed experts, but that source is independent of prompt length.  Load
  // it now, before accepting the first prompt, so a later prompt with a
  // different token count only rebuilds its small GGML topology and never
  // rereads the ~78 GiB expert set from SSD.  The source admission is
  // bounded by host_ram_gb and the memory watchdog; failure leaves the
  // established layer path available.
  if (engine.family_ops.family == ModelFamily::GLM5NEXT)
    glm5next_full_graph_set_vram_cache(&gpu.topology);
  if (graph_mode && graph_ops.bootstrap && !graph_ops.bootstrap(engine)) {
    LOG_STREAM_WARN()
        << "[GRAPH] graph bootstrap unavailable; "
           "the graph will attempt bounded lazy preparation on the first "
           "prompt.\n";
  }

  std::cout << "----------------------------------------\n";
  std::cout << "  MoEStream engine online and ready. Enter prompt below.\n";
  std::cout << "Use ':q' to exit execution loop cleanly.\n\n";

  // Configure only after all startup worker pools have been created.  New
  // threads inherit the creator's affinity mask, so doing this at the top of
  // run_engine() would accidentally constrain prefetch/GGML workers too.
  configure_current_thread({
      .name = "MAIN-DECODE",
      // Main decode and CUDA callbacks intentionally float.  Priority and
      // the ring's wait policy are the variables under test; a fixed core
      // was measured to create a static contention point.
      .cpu_ids = {},
      .use_rt = true,
      .rt_priority = 80,
      .nice_value = -20,
      .apply_nice = false,
  });

  // Real wall-clock anchor for perf().dump_report()'s final Throughput
  // line -- spans every turn in this session (not just one request),
  // starting AFTER model load (reported separately as "Dense load
  // (init)") so it isolates generation time specifically. See
  // dump_report()'s own comment for why this replaced a broken
  // summed-perf-counter calculation that double-counted concurrent work.
  auto session_start = std::chrono::high_resolution_clock::now();

  std::string user_input;
  std::string request_failure;
  while (true) {
    if (g_shutdown_requested.load(std::memory_order_relaxed))
      break;

    std::cout << "\n  User > ";
    std::getline(std::cin, user_input);

    // Multi-line input support (2026-09): this loop reads exactly one
    // getline() per turn, so a genuinely multi-line message piped via
    // stdin (e.g. "review this code:\n```python\ndef f():\n...") was
    // silently split into N SEPARATE turns, one per embedded newline —
    // found via a real survey harness where a code-review prompt with
    // an embedded function produced 6 fragmented, wrong generations
    // instead of one coherent one. Rather than degrade callers to
    // single-line-only content, accept a literal two-character "\n"
    // escape sequence (backslash + 'n') as a caller-side convention for
    // "real newline here" and unescape it right after reading, before
    // any tokenization -- a caller that wants a true multi-line message
    // in one turn sends its text with real newlines replaced by this
    // literal sequence. Interactive human typing is unaffected (nobody
    // types a literal backslash-n expecting it to stay literal in a
    // prompt to an LLM in practice, and if they did, the model still
    // sees intelligible text either way).
    {
      std::string unescaped;
      unescaped.reserve(user_input.size());
      for (size_t i = 0; i < user_input.size(); ++i) {
        if (user_input[i] == '\\' && i + 1 < user_input.size() &&
            user_input[i + 1] == 'n') {
          unescaped.push_back('\n');
          ++i;
        } else {
          unescaped.push_back(user_input[i]);
        }
      }
      user_input = std::move(unescaped);
    }

    // Ctrl-C (SIGINT) sets the flag and, on most platforms, interrupts
    // the blocking getline with a failed/EOF stream state. Treat any of
    // those — flag set, real EOF (Ctrl-D / piped input ending), or a
    // failed read — as "time to shut down" rather than looping forever
    // on an unusable stream.
    if (g_shutdown_requested.load(std::memory_order_relaxed) ||
        std::cin.eof() || std::cin.fail()) {
      break;
    }

    if (user_input == ":q" || user_input == "exit") {
      break;
    }
    if (user_input.empty())
      continue;

    // BUG (found this session): user_input was read but never actually
    // tokenized -- prompt_tokens only ever contained BOS, so the model
    // never saw what the user typed, on every single request. That's
    // very likely the real explanation for "the same output regardless
    // of prompt": the input to the model WAS identical every time.
    std::vector<int> prompt_tokens;
    if (!config.raw_prompt &&
        engine.family_ops.family == ModelFamily::GLM5NEXT &&
        tokenizer.has_chat_template()) {
      // GLM-5.3-Flash's real template is distinct from the older GLM
      // template even though both use [gMASK], <sop>, <|user|>, and
      // <|assistant|>. It has no newline after role tags. Thinking is
      // enabled by an open <think> block; direct-answer mode is the
      // explicit empty <think></think> block used by the GLM template.
      if (tokenizer.gmask_id >= 0)
        prompt_tokens.push_back(tokenizer.gmask_id);
      if (tokenizer.sop_id >= 0)
        prompt_tokens.push_back(tokenizer.sop_id);

      if (config.enable_thinking) {
        std::string effort = config.reasoning_effort;
        std::string effort_cap;
        if (effort == "low")
          effort_cap = "Low";
        else if (effort == "high")
          effort_cap = "High";
        else
          effort_cap = "Max";
        std::vector<int> reff =
            tokenizer.encode("<|system|>Reasoning Effort: " + effort_cap);
        prompt_tokens.insert(prompt_tokens.end(), reff.begin(), reff.end());
      }

      if (!config.system_prompt.empty()) {
        std::vector<int> sys =
            tokenizer.encode("<|system|>" + config.system_prompt);
        prompt_tokens.insert(prompt_tokens.end(), sys.begin(), sys.end());
      }
      prompt_tokens.push_back(tokenizer.user_id);
      std::vector<int> body = tokenizer.encode(user_input);
      prompt_tokens.insert(prompt_tokens.end(), body.begin(), body.end());
      prompt_tokens.push_back(tokenizer.assistant_id);
      if (config.enable_thinking) {
        if (tokenizer.ds_think_start_id >= 0)
          prompt_tokens.push_back(tokenizer.ds_think_start_id);
      } else {
        // GLM's direct-answer switch is not "no think token". The
        // model expects the assistant turn to contain an explicitly
        // empty reasoning block: <think></think>.
        if (tokenizer.ds_think_start_id >= 0 &&
            tokenizer.ds_think_end_id >= 0) {
          prompt_tokens.push_back(tokenizer.ds_think_start_id);
          prompt_tokens.push_back(tokenizer.ds_think_end_id);
        }
      }
    } else if (!config.raw_prompt && tokenizer.has_chat_template()) {
      // GLM-3/4-style: [gMASK]<sop><|user|>\n{msg}<|assistant|>\n
      // (see GGUFTokenizer::detect_chat_special_tokens for the
      // reference source and the caveat that GLM-5.2 specifically
      // isn't verified against this).
      if (tokenizer.gmask_id >= 0)
        prompt_tokens.push_back(tokenizer.gmask_id);
      if (tokenizer.sop_id >= 0)
        prompt_tokens.push_back(tokenizer.sop_id);
      // Real GLM-4 template renders EVERY message (including "system")
      // as <|{role}|>\n{content} in turn order, ahead of the user
      // turn -- see llama.cpp's LLM_CHAT_TEMPLATE_CHATGLM_4. There's
      // no reserved single-token id for "<|system|>" the way
      // user_id/assistant_id have, so it's sent as literal text.
      if (!config.system_prompt.empty()) {
        std::vector<int> sys =
            tokenizer.encode("<|system|>\n" + config.system_prompt);
        prompt_tokens.insert(prompt_tokens.end(), sys.begin(), sys.end());
      }
      prompt_tokens.push_back(tokenizer.user_id);
      std::vector<int> body = tokenizer.encode("\n" + user_input);
      prompt_tokens.insert(prompt_tokens.end(), body.begin(), body.end());
      prompt_tokens.push_back(tokenizer.assistant_id);
      std::vector<int> tail = tokenizer.encode("\n");
      prompt_tokens.insert(prompt_tokens.end(), tail.begin(), tail.end());
    } else if (!config.raw_prompt && tokenizer.has_deepseek_chat_template()) {
      // DeepSeek-V3/V4-style: {bos}<｜User｜>{msg}<｜Assistant｜></think>
      // (see GGUFTokenizer::ds_user_id for the reference source --
      // this is what fixed confidently-wrong, off-topic decode output
      // for a plain unwrapped prompt on DeepSeek-V4-Flash). No literal
      // "\n" padding around the tags, unlike the GLM branch above --
      // the real jinja template concatenates directly.
      // DeepSeek's chat template explicitly emits bos_token.  This is
      // independent of tokenizer.ggml.add_bos_token: that metadata flag
      // controls the tokenizer's automatic prefix, while the rendered
      // template still contains {{ bos_token }}.  V4.1 sets add_bos_token
      // to false, so guarding this on that flag drops the required first
      // token and shifts the entire chat prompt.
      prompt_tokens.push_back(tokenizer.bos_token_id);
      // Matches the real jinja template exactly: system-role content
      // is concatenated as raw text directly after {bos_token} and
      // before <｜User｜>, with no wrapping tag at all (see
      // ns.system_prompt / {{- ns.system_prompt -}} in
      // models/templates/deepseek-ai-DeepSeek-V4-Flash-0731.jinja).
      if (!config.system_prompt.empty()) {
        std::vector<int> sys = tokenizer.encode(config.system_prompt);
        prompt_tokens.insert(prompt_tokens.end(), sys.begin(), sys.end());
      }
      prompt_tokens.push_back(tokenizer.ds_user_id);
      std::vector<int> body = tokenizer.encode(user_input);
      prompt_tokens.insert(prompt_tokens.end(), body.begin(), body.end());
      prompt_tokens.push_back(tokenizer.ds_assistant_id);
      if (tokenizer.ds_think_end_id >= 0)
        prompt_tokens.push_back(tokenizer.ds_think_end_id);
    } else if (!config.raw_prompt && tokenizer.has_chatml_template()) {
      // ChatML (Qwen3.8-Flash-Next / qwen4exp), per the
      // model's tokenizer_config.json chat_template:
      //   <|im_start|>system\n{reasoning_instructions}[\n\n{system}]<|im_end|>\n
      //   <|im_start|>user\n{msg}<|im_end|>\n
      //   <|im_start|>assistant\n<think>\n
      // enable_thinking defaults to true for reference compatibility;
      // false omits the default reasoning system instruction.
      const char *kQwen4ExpReasoningXHigh =
          "Reasoning effort is set to xhigh. Please think carefully through "
          "the task, validate key assumptions, consider plausible "
          "alternatives, and prioritize correctness, consistency, and "
          "clarity in the final answer.";
      auto push_text = [&](const std::string &text) {
        std::vector<int> ids = tokenizer.encode(text);
        prompt_tokens.insert(prompt_tokens.end(), ids.begin(), ids.end());
      };
      if (config.enable_thinking || !config.system_prompt.empty()) {
        prompt_tokens.push_back(tokenizer.im_start_id);
        if (config.enable_thinking) {
          push_text(std::string("system\n") + kQwen4ExpReasoningXHigh +
                    (config.system_prompt.empty()
                         ? ""
                         : "\n\n" + config.system_prompt));
        } else {
          push_text("system\n" + config.system_prompt);
        }
        prompt_tokens.push_back(tokenizer.im_end_id);
        push_text("\n");
      }
      prompt_tokens.push_back(tokenizer.im_start_id);
      push_text("user\n" + user_input);
      prompt_tokens.push_back(tokenizer.im_end_id);
      push_text("\n");
      prompt_tokens.push_back(tokenizer.im_start_id);
      // The Qwen template puts the thinking markers in the rendered
      // text.  They are not special-token entries in this checkpoint's
      // vocabulary, so checking ds_think_start_id alone silently
      // dropped the whole reasoning block and left the model at
      // `<|im_start|>assistant\n`.  Preserve the template literally;
      // use the special-token IDs when a future Qwen vocabulary has
      // them, otherwise let the normal BPE encoder produce the IDs.
      push_text("assistant\n");
      if (config.enable_thinking) {
        if (tokenizer.ds_think_start_id >= 0)
          prompt_tokens.push_back(tokenizer.ds_think_start_id);
        else
          push_text("<think>");
        push_text("\n");
      } else {
        if (tokenizer.ds_think_start_id >= 0)
          prompt_tokens.push_back(tokenizer.ds_think_start_id);
        else
          push_text("<think>");
        push_text("\n\n");
        if (tokenizer.ds_think_end_id >= 0)
          prompt_tokens.push_back(tokenizer.ds_think_end_id);
        else
          push_text("</think>");
        push_text("\n\n");
      }
    } else {
      // Respect tokenizer.ggml.add_bos_token -- see
      // GGUFTokenizer::add_bos_token for why unconditionally pushing
      // BOS here is a real, confirmed bug for models (e.g.
      // Qwen3.6-35B-A3B) whose GGUF sets this to False.
      if (tokenizer.add_bos_token)
        prompt_tokens.push_back(tokenizer.bos_token_id);
      std::vector<int> encoded = tokenizer.encode(user_input);
      prompt_tokens.insert(prompt_tokens.end(), encoded.begin(), encoded.end());
    }

    // Diagnostic/fairness hook: Llama.cpp's llama-bench can consume an
    // exact token-ID file. Allow its token IDs to be replayed here so the
    // two engines execute the same prompt, independent of small tokenizer
    // wrapper differences. The normal text prompt path is unchanged when
    // this variable is absent.
    if (const char *token_input =
            std::getenv("MOESTREAM_PROMPT_TOKEN_INPUT_FILE")) {
      std::ifstream input(token_input);
      std::vector<int> exact_tokens;
      long long value = 0;
      while (input >> value) {
        if (value < INT_MIN || value > INT_MAX) {
          exact_tokens.clear();
          break;
        }
        exact_tokens.push_back((int)value);
      }
      if (!input.eof() || exact_tokens.empty()) {
        LOG_STREAM_ERROR() << "[MOE-PROMPT] failed to read token input file: "
                           << token_input << "\n";
      } else {
        prompt_tokens = std::move(exact_tokens);
        LOG_STREAM_INFO() << "[MOE-PROMPT] loaded " << prompt_tokens.size()
                          << " exact token IDs from " << token_input << "\n";
      }
    }
    if (std::getenv("MOESTREAM_MOE_SET_TRACE") != nullptr) {
      uint64_t hash = 1469598103934665603ULL;
      for (int token : prompt_tokens) {
        uint32_t value = (uint32_t)token;
        for (int i = 0; i < 4; ++i) {
          hash ^= (value >> (i * 8)) & 0xffU;
          hash *= 1099511628211ULL;
        }
      }
      LOG_STREAM_INFO() << "[MOE-PROMPT] tokens=" << prompt_tokens.size()
                        << " fnv64=0x" << std::hex << hash << std::dec << "\n";
    }
    if (const char *token_path = std::getenv("MOESTREAM_PROMPT_TOKEN_FILE")) {
      std::ofstream output(token_path);
      if (!output) {
        LOG_STREAM_ERROR()
            << "[MOE-PROMPT] failed to open token file for writing: "
            << token_path << "\n";
      } else {
        for (int token : prompt_tokens)
          output << token << "\n";
        LOG_STREAM_INFO() << "[MOE-PROMPT] wrote " << prompt_tokens.size()
                          << " token IDs to " << token_path << "\n";
      }
    }

    std::cout << "\n  MoEStream * ";
    generate_text(engine, ram_mgr, tokenizer, prefetcher, prompt_tokens,
                  max_new_tokens, tokenizer.eos_token_id, config.temperature,
                  config.top_p);
    if (!engine.qwen4exp_full_graph_failure.empty()) {
      const std::string reason = engine.qwen4exp_full_graph_failure;
      engine.qwen4exp_full_graph_failure.clear();
      request_failure = "Qwen4Exp full-graph request failed: " + reason;
      std::cout << "\n";
      LOG_STREAM_ERROR() << request_failure << "\n";
      // Leave through the normal teardown path below so the prefetch worker
      // is joined and graph-owned CUDA allocations are released before the
      // request error reaches main().
      break;
    }

    std::cout << "\n";
  }

  prefetcher.stop();
  // The selected full graph owns a GGML scheduler and CUDA workspace
  // buffers. Release it through the model graph contract while CudaManager
  // is still alive; otherwise a static graph destructor can run after CUDA
  // driver teardown and cudaFree aborts.
  if (engine.family_ops.graph_ops.shutdown)
    engine.family_ops.graph_ops.shutdown();
  // Phase A write this run's routing-frequency histogram so
  // the NEXT run's startup preload (above) has real data to work from.
  // Best-effort, matches dump_usage's own contract (a failed write just
  // logs and continues -- not worth failing shutdown over).
  // freeze_usage_stats (debug/A-B aid): skip both writes entirely so
  // pin_file/prefill_pin_file stay exactly as they were at startup --
  // see EngineConfig::freeze_usage_stats's own comment for why this
  // exists (real, confirmed run-to-run drift in the accumulated
  // histogram, even under an identical prompt).
  if (!config.freeze_usage_stats) {
    engine.dump_usage(config.pin_file);
    // Prefill-only histogram -- see route_freq_prefill's own comment.
    // No-op (dump_usage_histogram's own empty-path guard) if
  if (!engine.exl3_ggml_bridge_model_path.empty() &&
      engine.exl3_gpu_executor) {
    const Exl3GpuStats exl3_stats = engine.exl3_gpu_executor->stats();
    LOG_STREAM_INFO()
        << "[EXL3-GGML-STATS] routed_ffns=" << exl3_stats.routed_ffns
        << " matmuls=" << exl3_stats.matmuls
        << " gpu_ffn_ms=" << exl3_stats.routed_ffn_seconds * 1000.0
        << " routed_graph_captures=" << exl3_stats.routed_graph_captures
        << " routed_graph_replays=" << exl3_stats.routed_graph_replays
        << " routed_graph_eager_fallbacks="
        << exl3_stats.routed_graph_eager_fallbacks
        << " cache_uploads=" << exl3_stats.uploads
        << " cache_hits=" << exl3_stats.cache_hits << "\n";
  }
    // prefill_pin_file isn't configured.
    engine.dump_usage_prefill(config.prefill_pin_file);
  } else {
    LOG_STREAM_INFO() << "[USAGE] freeze_usage_stats=true -- skipping "
                         "usage-file write(s) this "
                         "session ("
                      << config.pin_file
                      << (config.prefill_pin_file.empty()
                              ? ""
                              : (", " + config.prefill_pin_file))
                      << " left untouched).\n";
  }

  // Cache-churn summary for both tiers: which experts were swapped in/out
  // the most, and (VRAM only, since that's the only tier with a pinning
  // mechanism) which of those are still unpinned
  // Safe without locking here: prefetcher.stop() above has already joined
  // the worker thread, so nothing else touches these structures now.
  ram_mgr.report_cache_churn();
  gpu.topology.report_cache_churn();
  ram_mgr.report_slot_summary(engine);
  gpu.topology.report_slot_summary();
  ram_mgr.report_tier_map(engine);
  prefetcher.log_per_layer_report(); // [PREFETCH-LAYER]: per-layer prefetch
                                     // hint/fetch/outcome breakdown
  report_named_timers();     // attn_q_a/attn_q_b/attn_kv_a_mqa fast-path
                             // investigation, see CLAUDE.md
  report_layer_miss_stats(); // [LAYER-MISS]: per-layer decode-time real
                             // disk-fetch misses -- see CLAUDE.md CPU-offload
                             // data-gathering entry
  report_pilot_recall_by_layer(); // [PILOT-BY-LAYER]: per-layer
                                  // DEPTH-1/ATTN-PILOT/DEPTH-2 prediction
                                  // accuracy -- see CLAUDE.md per-layer
                                  // prediction accuracy entry
  report_qwen4exp_pilot_stats();  // [QWEN-PILOT]: Qwen HC-proxy prediction
                                  // accuracy and load coverage
  report_moe_bucket_stats(); // [MOE-BUCKET]: real token-sharing rate for MoE
                             // expert dispatch -- the llama.cpp -ub/mul_mat_id
                             // batching investigation
  report_prefill_lookahead_stats(); // [PREFILL-LOOKAHEAD]:
                                    // prefill_dispatch_lookahead_tokens real
                                    // added/skipped counts
  report_pilot_wide_stats();
  report_prefill_window_stats(); // [PREFILL-WINDOW]: prefill sliding-window
                                 // capacity (P1+P2)  // [PILOT-WIDE]:
                                 // pilot_widen_all_tokens real recall/coverage
                                 // measurement
  extern void deepseek41_engram_print_cache_stats();
  deepseek41_engram_print_cache_stats(); // [DSV41-ENGRAM-CACHE]: real
                                         // hit-rate for the Engram row
                                         // cache (goal item 4), never
                                         // measured before this session

  double session_elapsed_s =
      std::chrono::duration<double>(std::chrono::high_resolution_clock::now() -
                                    session_start)
          .count();
  perf().dump_report(session_elapsed_s);
  if (!request_failure.empty())
    return std::unexpected(request_failure);
  return {};
}

int main(int argc, char *argv[]) {
  install_shutdown_handlers(); // Ctrl-C/SIGTERM -> graceful teardown (see
                               // shutdown.hpp)

  std::string config_path = "";
  std::string prompt_cli;
  std::string model_path = "";
  std::string gen_config_path = "";
  double ram_budget_gb = 0.0;
  int promote_threshold_cli = -1;
  int pin_threshold_cli = -1;
  std::string pin_file_cli;
  int expert_fetch_threads_cli = -1;
  int prefetch_worker_threads_cli = -1;
  int max_ctx = 4096;
  int max_new_tokens = 1024;
  int gpu_id = 0;
  bool drop_pages = false;
  bool direct_io_cli = false;
  bool io_uring_cli = false;
  // ctx_cli_given/ngen_cli_given: real bug found via a real test --
  // --ctx (and --ngen) were parsed but only ever applied to
  // config.context_size/max_new_tokens in the legacy --model-only
  // startup branch below; the "CLI overrides always apply" block right
  // after config-file loading never included them, so --ctx silently
  // did nothing whenever --config was used (which is every TOML-based
  // run this whole session). max_ctx/max_new_tokens keep their
  // legacy-mode defaults (4096/1024) either way -- these bools track
  // ONLY whether the flag was explicitly passed, so the override below
  // doesn't clobber a --config file's own context_size/max_new_tokens
  // with an unrequested default.
  bool ctx_cli_given = false;
  bool ngen_cli_given = false;
  bool verbose_moe = false;
  bool raw_prompt_cli = false;
  std::string system_prompt_cli;
  std::string reasoning_effort_cli;
  std::string log_path_cli;
  std::string log_level_cli;
  std::string trace_filter_cli;
  bool trace_abort_on_nan_cli = false;
  bool freeze_usage_stats_cli = false;
  bool collect_usage_stats_cli = false;
  float temp_cli = -1.0f, top_p_cli = -1.0f; // -1 sentinel = "not given" (0.0
                                             // is a valid greedy-decode value)
  // 2026-09: real incident -- a second moestream instance (a separate
  // one-off smoke test) was started while this box already had another
  // moestream benchmark running; the two together drove the box into
  // swap thrashing hard enough to force a reboot. See
  // startup_preflight.hpp's own header comment. --force-start bypasses
  // the resulting startup checks for a deliberate, known-safe case
  // (e.g. two instances on disjoint GPU sets).
  bool force_start_cli = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--config" && i + 1 < argc)
      config_path = argv[++i];
    else if (arg == "--prompt" && i + 1 < argc)
      prompt_cli = argv[++i];
    else if (arg == "--gen-config") {
      gen_config_path = (i + 1 < argc && argv[i + 1][0] != '-')
                            ? argv[++i]
                            : "moestream_config.toml";
    } else if (arg == "--model" && i + 1 < argc)
      model_path = argv[++i];
    else if (arg == "--ram" && i + 1 < argc)
      ram_budget_gb = std::stod(argv[++i]);
    else if (arg == "--ctx" && i + 1 < argc) {
      max_ctx = std::stoi(argv[++i]);
      ctx_cli_given = true;
    } else if (arg == "--ngen" && i + 1 < argc) {
      max_new_tokens = std::stoi(argv[++i]);
      ngen_cli_given = true;
    } else if (arg == "--gpu" && i + 1 < argc)
      gpu_id = std::stoi(argv[++i]);
    else if (arg == "--drop-pages")
      drop_pages = true;
    else if (arg == "--direct-io")
      direct_io_cli = true;
    else if (arg == "--io-uring")
      io_uring_cli = true;
    else if (arg == "--verbose-moe")
      verbose_moe = true;
    else if (arg == "--raw-prompt")
      raw_prompt_cli = true;
    else if (arg == "--system-prompt" && i + 1 < argc)
      system_prompt_cli = argv[++i];
    else if (arg == "--reasoning-effort" && i + 1 < argc)
      reasoning_effort_cli = argv[++i];
    else if (arg == "--promote-threshold" && i + 1 < argc)
      promote_threshold_cli = std::stoi(argv[++i]);
    else if (arg == "--pin-threshold" && i + 1 < argc)
      pin_threshold_cli = std::stoi(argv[++i]);
    else if (arg == "--pin-file" && i + 1 < argc)
      pin_file_cli = argv[++i];
    else if (arg == "--expert-fetch-threads" && i + 1 < argc)
      expert_fetch_threads_cli = std::stoi(argv[++i]);
    else if (arg == "--prefetch-threads" && i + 1 < argc)
      prefetch_worker_threads_cli = std::stoi(argv[++i]);
    else if (arg == "--log-path" && i + 1 < argc)
      log_path_cli = argv[++i];
    else if (arg == "--log-level" && i + 1 < argc)
      log_level_cli = argv[++i];
    else if (arg == "--trace-filter" && i + 1 < argc)
      trace_filter_cli = argv[++i];
    else if (arg == "--trace-abort-on-nan")
      trace_abort_on_nan_cli = true;
    else if (arg == "--freeze-usage-stats")
      freeze_usage_stats_cli = true;
    else if (arg == "--collect-usage-stats")
      collect_usage_stats_cli = true;
    else if (arg == "--temp" && i + 1 < argc)
      temp_cli = std::stof(argv[++i]);
    else if (arg == "--top_p" && i + 1 < argc)
      top_p_cli = std::stof(argv[++i]);
    else if (arg == "--force-start")
      force_start_cli = true;
    else if (arg == "--help" || arg == "-h") {
      print_usage();
      return 0;
    }
  }

  // Generate sample config and exit
  if (!gen_config_path.empty()) {
    EngineConfig::write_sample(gen_config_path);
    return 0;
  }

  EngineConfig config;

  if (!config_path.empty()) {
    // Load from config file
    if (!config.load(config_path))
      return 1;
    if (!config.validate())
      return 1;
  } else if (!model_path.empty()) {
    // Build config from CLI arguments (backward compat)
    config.model_path = model_path;
    config.context_size = max_ctx;
    config.max_new_tokens = max_new_tokens;
    config.drop_pages = drop_pages;
    config.direct_io = direct_io_cli;
    config.io_uring = io_uring_cli;

    GpuConfig gc;
    gc.device_id = gpu_id;
    gc.role = "dense+moe";
    config.gpus.push_back(gc);

    if (!config.validate())
      return 1;
  } else {
    std::cerr << "  Error: Specify --config <config.json> or --model "
                 "<path_to_gguf>\n";
    print_usage();
    return 1;
  }

  // --ram / --verbose-moe always override the config file's values, if given
  if (ram_budget_gb > 0)
    config.host_ram_gb = ram_budget_gb;
  if (!prompt_cli.empty())
    config.prompt = prompt_cli;
  if (verbose_moe)
    config.verbose_moe_log = true;
  if (raw_prompt_cli)
    config.raw_prompt = true;
  if (!system_prompt_cli.empty())
    config.system_prompt = system_prompt_cli;
  if (!reasoning_effort_cli.empty())
    config.reasoning_effort = reasoning_effort_cli;
  if (drop_pages)
    config.drop_pages = true;
  if (direct_io_cli)
    config.direct_io = true;
  if (io_uring_cli)
    config.io_uring = true;
  // ctx_cli_given/ngen_cli_given: see their own declaration comment --
  // these were completely missing from this override block before,
  // meaning --ctx/--ngen had zero effect whenever --config was used.
  if (ctx_cli_given)
    config.context_size = max_ctx;
  if (ngen_cli_given)
    config.max_new_tokens = max_new_tokens;
  if (promote_threshold_cli > 0)
    config.promote_hit_threshold = promote_threshold_cli;
  if (pin_threshold_cli >= 0)
    config.pin_threshold = pin_threshold_cli;
  if (!pin_file_cli.empty())
    config.pin_file = pin_file_cli;
  if (expert_fetch_threads_cli > 0)
    config.expert_fetch_threads = expert_fetch_threads_cli;
  if (prefetch_worker_threads_cli > 0)
    config.prefetch_worker_threads = prefetch_worker_threads_cli;
  if (!log_path_cli.empty())
    config.log_path = log_path_cli;
  if (!log_level_cli.empty())
    config.log_level = log_level_cli;
  if (!trace_filter_cli.empty())
    config.trace_filter = trace_filter_cli;
  if (trace_abort_on_nan_cli)
    config.trace_abort_on_nan = true;
  if (freeze_usage_stats_cli)
    config.freeze_usage_stats = true;
  if (collect_usage_stats_cli)
    config.freeze_usage_stats = false;
  if (temp_cli >= 0.0f)
    config.temperature = temp_cli;
  if (top_p_cli >= 0.0f)
    config.top_p = top_p_cli;

  // Initialize the logger before anything else logs — empty log_path keeps
  // everything on stderr exactly as before.
  Logger::instance().init(config.log_path, config.log_level);
#if defined(MOESTREAM_USE_GGML)
  bool allow_ggml_debug = config.log_level == "DEBUG" ||
                          config.log_level == "VERBOSE";
  ggml_log_set(moestream_ggml_log_callback, &allow_ggml_debug);
#endif
  TraceFilter::instance().configure(config.trace_filter,
                                    config.trace_abort_on_nan);

  std::cout << "  MoEStream Engine initializing (C++23 mode)...\n";
  std::cout << "----------------------------------------\n";
  config.print();

  // Real, cheap startup safety checks -- BEFORE any RAM-tier allocation,
  // [PIN] preload, or GPU VRAM pool allocation begins. See
  // startup_preflight.hpp's own header comment for the real incident
  // this exists to prevent (two moestream instances driving this box
  // into swap thrashing / a hard crash). Fail-closed: refuses to start
  // rather than silently proceeding into a repeat of that incident.
  {
    std::vector<int> gpu_ids;
    for (auto &g : config.gpus)
      gpu_ids.push_back(g.device_id);
    if (!preflight::run(config.host_ram_gb, gpu_ids, force_start_cli)) {
      LOG_STREAM_ERROR()
          << "  Startup pre-flight check(s) failed -- see [PREFLIGHT] "
             "lines above. Refusing to start (pass --force-start to "
             "override once you're certain it's safe).\n";
      return 1;
    }
  }

  // cpu_only (TOML `cpu_only = true` or MOESTREAM_CPU_ONLY env var) forces all
  // compute onto the CPU reference path. Enable it BEFORE run_engine so every
  // dense_proj / rmsnorm_proj / expert_ffn dispatch sees the flag.
  if (config.exl3_backend == "exllamav3") {
    std::vector<std::string> overrides{
        "--ngen", std::to_string(config.max_new_tokens),
        "--ctx", std::to_string(config.context_size),
        "--threads", std::to_string(config.omp_max_threads > 0 ? config.omp_max_threads : 4),
        "--cuda-schedule", config.exllamav3_cuda_schedule,
        "--temperature", std::to_string(config.temperature),
        "--top-p", std::to_string(config.top_p),
        config.raw_prompt ? "--raw-prompt" : "--no-raw-prompt",
        config.enable_thinking ? "--thinking" : "--no-thinking"};
    if (!config.prompt.empty()) {
      overrides.push_back("--prompt");
      overrides.push_back(config.prompt);
    }
    std::cout << "[EXLLAMAV3] Delegating model, KV cache and decode to upstream runtime\n"
              << std::flush;
    auto result = exec_exllamav3_backend(config_path, config.exllamav3_python,
                                        config.exllamav3_runner_path, overrides);
    if (!result) std::cerr << result.error() << "\n";
    return result ? 0 : 1;
  }
  if (config.cpu_only)
    enable_cpu_only_mode();
  if (config.dsv4_outproj_gpu)
    enable_dsv4_outproj_gpu();
  if (config.dsv4_score_gpu)
    enable_dsv4_score_gpu();
  if (config.cpu_dot_ggml_avx2)
    enable_cpu_dot_ggml_avx2();
  if (config.cpu_dot_ggml_iq1m)
    enable_cpu_dot_ggml_iq1m();
  set_moe_prefill_batched(config.moe_prefill_batched);
  set_moe_decode_batched(config.moe_decode_batched);
  set_moe_prefill_stream(config.moe_prefill_stream);
  set_moe_prefill_stream_chunk(config.moe_prefill_stream_chunk);
  set_moe_expert_bf16_gemm(config.moe_expert_bf16_gemm);
  set_moe_ggml_mul_mat_id(config.moe_ggml_mul_mat_id);
  set_moe_ggml_shared_expert(config.moe_ggml_shared_expert);
  set_moe_ggml_planar_copies(config.moe_ggml_planar_copies);
  set_moe_ggml_gpu_router(config.moe_ggml_gpu_router);
  set_moe_ggml_device(config.moe_ggml_device);
  set_ggml_hc_async_graph_reuse(config.ggml_async_graph_reuse);
  set_qwen_hc_async_readback(config.qwen4exp_hc_async_readback);
  set_ggml_moe_async_graph_reuse(config.ggml_async_graph_reuse);
  LOG_STREAM_INFO() << "[CONFIG] Qwen4Exp GGML shared expert: "
                    << (config.moe_ggml_shared_expert ? "ON" : "off")
                    << " (device routed+shared FFN)\n";
  LOG_STREAM_INFO() << "[CONFIG] Qwen4Exp decode device HC handoff: "
                    << (config.qwen4exp_decode_device_hc ? "ON" : "off")
                    << " (Qwen S=1 only)\n";
  LOG_STREAM_INFO() << "[CONFIG] Qwen4Exp full graph decode scheduler: "
                    << (config.qwen4exp_decode_scheduler ? "ON" : "off")
                    << " (complete DDR arenas only; lazy layers use the "
                       "legacy path)\n";
  LOG_STREAM_INFO() << "[CONFIG] Qwen CPU routed->device shared handoff: "
                    << (config.qwen4exp_cpu_shared_device ? "ON" : "off")
                    << " (CPU-affinity layers only; Qwen-only)\n";
  LOG_STREAM_INFO() << "[CONFIG] Qwen4Exp decode batch RAM staging: "
                    << (config.qwen4exp_decode_batch_stage ? "ON" : "off")
                    << " (Qwen S=1 only; direct pinned-slab path otherwise)\n";
  LOG_STREAM_INFO() << "[CONFIG] Qwen4Exp custom exact-Q8 HC decode: "
                    << (config.qwen4exp_decode_custom_hc_q8 ? "ON" : "off")
                    << " (requires device HC; Qwen-only)\n";
  LOG_STREAM_INFO() << "[CONFIG] Qwen4Exp async HC readback: "
                    << (config.qwen4exp_hc_async_readback ? "ON" : "off")
                    << " (pinned D2H staging; Qwen CPU-affinity path only)\n";
  LOG_STREAM_INFO() << "[CONFIG] Qwen4Exp GPU router during S==1 decode: "
                    << (config.qwen4exp_decode_gpu_router ? "ON" : "off")
                    << " (batched prefill still follows moe_ggml_gpu_router; "
                       "Qwen pilot is independent)\n";
  set_hc_ggml(config.hc_ggml);
  set_gdn_proj_ggml(config.gdn_proj_ggml);
  set_gdn_full_ggml(config.gdn_full_ggml);
  LOG_STREAM_INFO() << "[CONFIG] Full Qwen4Exp GDN GGML graph: "
                    << (config.gdn_full_ggml ? "ON" : "off")
                    << " (projection+conv+delta+norm+ssm_out)\n";
  set_moe_gpu_accum(config.moe_gpu_accum);
  set_dense_bf16_kquant(config.dense_bf16_kquant);
  set_expert_gpu_activation(config.expert_gpu_activation);
  LOG_STREAM_INFO() << "[CONFIG] Expert SwiGLU on GPU: "
                    << (config.expert_gpu_activation ? "ON" : "off")
                    << " (device silu/mul/clamp, no gate/up host round trip)\n";
  set_vram_min_free_mb(config.vram_min_free_mb);
  LOG_STREAM_INFO()
      << "[CONFIG] VRAM headroom: dynamic allocations keep at least "
      << config.vram_min_free_mb << " MB free per GPU\n";
  set_dense_bf16_prefill_gemm(config.dense_bf16_prefill_gemm);
  LOG_STREAM_INFO()
      << "[CONFIG] Dense prefill GEMM via dequant->BF16 + cuBLAS: "
      << (config.dense_bf16_prefill_gemm ? "ON" : "off")
      << " (Q8_0 weights, S>1 only)\n";
  LOG_STREAM_INFO() << "[CONFIG] Dense host copies on dedicated CUDA stream: "
                    << (config.dense_async_copies ? "ON" : "off") << "\n";
  LOG_STREAM_INFO() << "[CONFIG] MoE GPU-side expert accumulation: "
                    << (config.moe_gpu_accum ? "ON" : "off")
                    << " (device scatter-add + one [S,D] download per layer)\n";
  LOG_STREAM_INFO() << "[CONFIG] Qwen4Exp GGML device router: "
                    << (config.moe_ggml_gpu_router ? "ON" : "off")
                    << " (GPU top-k; returns IDs/weights only)\n";
  std::string qwen_prefetch_excluded = "none";
  if (!config.qwen4exp_router_pilot_prefetch_exclude_layers.empty()) {
    qwen_prefetch_excluded = "[";
    for (size_t i = 0;
         i < config.qwen4exp_router_pilot_prefetch_exclude_layers.size(); ++i) {
      if (i > 0)
        qwen_prefetch_excluded += ",";
      qwen_prefetch_excluded += std::to_string(
          config.qwen4exp_router_pilot_prefetch_exclude_layers[i]);
    }
    qwen_prefetch_excluded += "]";
  }
  std::string qwen_predict_layers = "all";
  if (!config.qwen4exp_router_pilot_predict_layers.empty()) {
    qwen_predict_layers = "[";
    for (size_t i = 0; i < config.qwen4exp_router_pilot_predict_layers.size();
         ++i) {
      if (i > 0)
        qwen_predict_layers += ",";
      qwen_predict_layers +=
          std::to_string(config.qwen4exp_router_pilot_predict_layers[i]);
    }
    qwen_predict_layers += "]";
  }
  LOG_STREAM_INFO() << "[CONFIG] Qwen4Exp router pilot: "
                    << (config.qwen4exp_router_pilot ? "ON" : "off")
                    << " (HC stream-0 proxy, top-"
                    << config.qwen4exp_router_pilot_k << ", prefetch top-"
                    << (config.qwen4exp_router_pilot_prefetch_k > 0
                            ? config.qwen4exp_router_pilot_prefetch_k
                            : config.qwen4exp_router_pilot_k)
                    << ", active-layer async prefetch "
                    << (config.qwen4exp_router_pilot_prefetch ? "ON" : "off")
                    << ", excluded layers=" << qwen_prefetch_excluded
                    << ", predict layers=" << qwen_predict_layers << ")\n";
  LOG_STREAM_INFO()
      << "[CONFIG] Qwen pilot skip cache-stable decode layers: "
      << (config.qwen4exp_router_pilot_skip_fully_cached ? "ON" : "off")
      << "\n";
  LOG_STREAM_INFO()
      << "[CONFIG] Qwen pilot skip prefetch-excluded decode layers: "
      << (config.qwen4exp_router_pilot_skip_prefetch_excluded_layers ? "ON"
                                                                     : "off")
      << "\n";
  LOG_STREAM_INFO() << "[CONFIG] GGML async graph workspace reuse: "
                    << (config.ggml_async_graph_reuse ? "ON" : "off")
                    << " (shared backend stream ordering)\n";
  LOG_STREAM_INFO()
      << "[CONFIG] MoE prefill token-batching (group tokens by expert): "
      << (config.moe_prefill_batched ? "ON" : "off")
      << " -- shared path, see dense_dispatch.hpp's blast-radius note\n";
  LOG_STREAM_INFO()
      << "[CONFIG] MoE decode batched dispatch (skip VRAM-cache "
         "lock/residency machinery, CPU-direct for RAM-resident experts): "
      << (config.moe_decode_batched ? "ON" : "off")
      << " -- see dense_dispatch.hpp's own accessor for the real "
         "measured numbers\n";
  // VERIFIED regression on real hardware (dense_dispatch.hpp's own
  // moe_decode_batched() comment / memory: moe-decode-batched-
  // regression.md): 5.03/4.97 tok/s baseline decode -> 2.93 tok/s with
  // this on for qwen4exp (Qwen3.8-Flash-Next, UD-IQ1_M), ~41% SLOWER,
  // because that model's routed experts already have a fast GPU kernel
  // and forcing CPU dequant instead is strictly worse. `engine` isn't
  // constructed yet at this point in main() (this block only has
  // `config`), so this can't be gated on model family the way the
  // header comments are -- warns unconditionally whenever the flag is
  // on. Loud, not just a comment, so a config drift or copy-paste from
  // another model's tuning doesn't silently reintroduce this.
  if (config.moe_decode_batched) {
    LOG_WARN("[CONFIG] moe_decode_batched=true -- VERIFIED ~41%% DECODE "
             "REGRESSION on qwen4exp/qwen3.8-flash-next UD-IQ1_M (5.0 -> "
             "2.9 tok/s), not a win. Only a plausible win when the routed "
             "experts have NO GPU kernel at all -- most checkpoints' quant "
             "types ARE GPU-capable. Re-measure before trusting this on "
             "for any model/quant combination.\n");
  }
  LOG_STREAM_INFO()
      << "[CONFIG] GLM5NEXT grouped GPU prefill: "
      << (config.glm5next_moe_prefill_gpu ? "ON" : "off")
      << " (GLM5NEXT only; host-validated routing, grouped GPU FFN)\n";
  LOG_STREAM_INFO() << "[CONFIG] GLM5NEXT KDA dense projection fusion: "
                    << (config.glm5next_dense_prefill_fusion ? "ON" : "off")
                    << " (S>1 best-effort GGML CUDA graph, legacy fallback)\n";

  auto outcome = run_engine(config);
  if (!outcome) {
    LOG_STREAM_ERROR() << "  Runtime Critical Failure: " << outcome.error()
                       << "\n";
    return 1;
  }

  std::cout
      << "\n  Shutting down background tasks and reclaiming system handles..."
      << std::endl;
  return 0;
}
