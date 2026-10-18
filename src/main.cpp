#include "active_layer_upload_workers.hpp"
#include "async_prefetcher.hpp"
#include "cuda_manager.hpp"
#include "engine.hpp"
#include "../models/full_graph_model.hpp"
#include "option_b_global_hot.hpp"
#include "engine_config.hpp"
#include "exl3_cpu.hpp"
#include "exl3_model_loader.hpp"
#include "exl3_decode.hpp"
#if defined(MOESTREAM_EXL3_CUDA)
#include "exl3_gpu.hpp"
#endif
#include "exllamav3_backend.hpp"
#include "engine_runner.hpp"
#include "graph_config.hpp"
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

  EngineRunner runner(config, cufile_driver_guard.opened);
  return runner.run();
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
