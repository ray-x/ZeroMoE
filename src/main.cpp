#include "gguf_loader.hpp"
#include "engine.hpp"
#include "ram_manager.hpp"
#include "gguf_tokenizer.hpp"
#include "async_prefetcher.hpp"
#include "generation_loop.hpp"
#include "cuda_manager.hpp"
#include "perf_timer.hpp"
#include "engine_config.hpp"
#include "logger.hpp"
#include "shutdown.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <expected>

class GGUFGraphInitializer {
public:
    static bool build_model_graph(StreamingEngine& engine, int max_ctx);
};

void print_usage() {
    std::cout << "Usage: ./moestream --config <config.toml>\n"
              << "       ./moestream --model <path_to_gguf> [options]\n"
              << "       ./moestream --gen-config [output.toml]\n\n"
              << "Options:\n"
              << "  --config <file>  Load engine config from TOML file\n"
              << "  --gen-config     Generate a sample config file and exit\n"
              << "  --model <path>   Path to GGUF model file or directory\n"
              << "  --ram <GB>       Strict RAM budget in GB (default: auto-detect 88% free)\n"
              << "  --ctx <N>        Maximum context size token window (default: 4096)\n"
              << "  --ngen <N>       Maximum new tokens to generate per response (default: 1024)\n"
              << "  --temp <F>       Sampling temperature. 0.0 for greedy decode (default: 0.7)\n"
              << "  --top_p <F>      Nucleus sampling top-p threshold (default: 0.90)\n"
              << "  --gpu <ID>       CUDA device ID for single-GPU mode (default: 0)\n"
              << "  --drop-pages     Drop expert pages from page cache after use\n"
              << "  --verbose-moe    Log every MoE host-cache hit (high volume, debug only)\n"
              << "  --promote-threshold <N>  Number of host re-hits before async H2D promote (default: 1)\n"
              << "  --log-path <file>  Redirect logging to a file instead of stderr\n"
              << "  --log-level <lvl>  VERBOSE | DEBUG | INFO (default) | WARN | ERROR\n"
              << "  --help           Show this help interface\n";
}

std::expected<void, std::string> run_engine(const EngineConfig& config) {
    StreamingEngine engine;
    RAMSafetyManager ram_mgr;
    GGUFTokenizer tokenizer;
    LockFreePrefetcher prefetcher;
    CudaManager gpu;

    const std::string& model_path = config.model_path;
    int max_ctx = config.context_size;
    int max_new_tokens = config.max_new_tokens;

    engine.drop_expert_pages = config.drop_pages;
    ram_mgr.configured_ram_gb = config.host_ram_gb;
    ram_mgr.promote_hit_threshold = (uint32_t)config.promote_hit_threshold;
    ram_mgr.verbose_cache_log = config.verbose_moe_log;
    if (config.dense_gpu_idx >= 0)
        gpu.device_id = config.gpus[config.dense_gpu_idx].device_id;
    if (!gpu.initialize()) {
        return std::unexpected("Failed to initialize GPU subsystem");
    }
    gpu.report_vram_usage("after gpu.initialize()");
    engine.gpu = &gpu;  // give the forward path access to dense GPU dispatch
    // Captured separately (not read back via engine.gpu) so StreamingEngine's
    // destructor can free its own device-resident buffers (resident_layers'
    // device_ptr, d_Lc/d_Rc/d_Lc_q4/d_Rc_q4, d_hidden) correctly regardless
    // of whether `gpu` (declared before `engine` above) has already been
    // destroyed by the time engine's destructor runs.
    engine.gpu_device_id = gpu.get_core_device();

    if (!engine.model_file.load_header(model_path)) {
        return std::unexpected("Failed to parse GGUF file or map descriptor tree structure");
    }

    // Optional separate dense GGUF: dense/preloaded tensors read from here, routed
    // experts still stream from model_path. Enables a dense(IQ4_XS)+expert(IQ2_*)
    // mix across two files.
    if (!config.dense_model_path.empty() && config.dense_model_path != model_path) {
        if (!engine.dense_model_file.load_header(config.dense_model_path)) {
            return std::unexpected("Failed to parse dense GGUF (dense_model_path)");
        }
        engine.has_dense_model = true;
    }

    const std::string& gguf_path = engine.model_file.resolved_file_path;
    std::ifstream metadata_stream(gguf_path, std::ios::binary);
    // Skip GGUF header: magic(4) + version(4) + tensor_count(8) + kv_count(8) = 24 bytes
    metadata_stream.seekg(4 + 4, std::ios::beg);  // skip magic + version
    uint64_t tensor_count_raw = 0, kv_count_raw = 0;
    metadata_stream.read(reinterpret_cast<char*>(&tensor_count_raw), 8);
    metadata_stream.read(reinterpret_cast<char*>(&kv_count_raw), 8);

    if (!tokenizer.load_from_gguf_metadata(metadata_stream, kv_count_raw)) {
        return std::unexpected("Tokenizer vocabulary table completely missing from GGUF metadata");
    }
    metadata_stream.close();

    if (!GGUFGraphInitializer::build_model_graph(engine, max_ctx)) {
        return std::unexpected("Failure triggered during topological model graph initialization pass");
    }

    // --- Multi-GPU memory planning (from optimize.md) ---
    auto& cfg = engine.model_file.architecture_config;

    // Allocate expert VRAM cache on vault GPU(s)
    size_t expert_slab_bytes = 0;
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
        for (int l = cfg.first_dense; l < cfg.n_layers; ++l) {
            expert_slab_bytes = std::max(expert_slab_bytes, engine.compute_expert_slab_bytes(l));
        }
    }
    gpu.setup_expert_cache(expert_slab_bytes, cfg.n_experts);
    gpu.report_vram_usage("after expert cache setup");

    // Connect VRAM expert cache to the RAM manager for LRU integration
    ram_mgr.vram_cache = &gpu.topology;

    // Offload dense tensors to core GPU VRAM — via a single pool allocation
    // (see CudaManager::init_dense_pool/offload_tensor_to_pool) rather than
    // one cudaMalloc PER tensor. With ~1182 separate tensors of wildly
    // varying sizes, individual allocations fragment a near-full device:
    // the allocator can fail to place a tensor even when total free bytes
    // are technically sufficient, because no single gap is large enough.
    // One pool + offset suballocation has no such gaps — a tensor either
    // fits in the remaining linear space or it doesn't, with zero waste
    // between tensors beyond 256-byte alignment padding.
    LOG_STREAM_INFO() << "[CUDA] Offloading persistent dense layers to core GPU (device "
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
    auto is_dense_offload_candidate = [](const std::string& name) {
        return name.find("attn") != std::string::npos ||
               name.find("norm") != std::string::npos ||
               name.find("gate_inp") != std::string::npos ||
               name.find("shexp") != std::string::npos ||
               name.find("ffn_gate.weight") != std::string::npos ||
               name.find("ffn_up.weight") != std::string::npos ||
               name.find("ffn_down.weight") != std::string::npos ||
               name == "token_embd.weight" ||
               name == "output.weight" ||
               name == "output_norm.weight";
    };

    // Pass 1: collect candidates and total the pool size needed (256-byte
    // aligned per tensor, matching offload_tensor_to_pool's own alignment —
    // must agree or the pool could report "fits" here but reject tensors
    // during the actual suballocation pass below).
    constexpr size_t POOL_ALIGN = 256;
    std::vector<QuantizedTensor*> dense_candidates;
    size_t pool_bytes_needed = 0;
    for (auto& [name, tensor] : engine.resident_layers) {
        if (!tensor.host_ptr || !is_dense_offload_candidate(name)) continue;
        dense_candidates.push_back(&tensor);
        pool_bytes_needed = (pool_bytes_needed + POOL_ALIGN - 1) & ~(POOL_ALIGN - 1);
        pool_bytes_needed += tensor.meta.size_bytes;
    }

    // Largest tensors first: with a linear bump allocator, total space used
    // is the same regardless of order, but if the pool ends up smaller than
    // requested (backoff below) it's whatever's processed LAST that spills
    // to host. Prioritize keeping large, expensive-to-CPU-dequant tensors
    // (e.g. ~69 MB attn_output) on GPU over small ones (~KB norms, cheap on
    // CPU either way) by placing large tensors first in the fill order.
    std::sort(dense_candidates.begin(), dense_candidates.end(),
              [](const QuantizedTensor* a, const QuantizedTensor* b) {
                  return a->meta.size_bytes > b->meta.size_bytes;
              });

    bool pool_ready = gpu.init_dense_pool(pool_bytes_needed);
    if (!pool_ready && pool_bytes_needed > 0) {
        // Didn't fit at full size — back off to a fraction of ACTUAL free
        // VRAM (ground truth, not our own estimate) and retry. Whatever
        // doesn't fit in the reduced pool falls back to CPU dequant-matmul,
        // same graceful degradation as before, just now fragmentation-free
        // for however much DOES fit.
        size_t backoff_bytes = (size_t)(gpu.core_free_vram_bytes() * 0.95);
        LOG_STREAM_WARN() << "[CUDA] Full dense pool (" << pool_bytes_needed / 1e9
                  << " GB) didn't fit — retrying with " << backoff_bytes / 1e9
                  << " GB (95% of currently free VRAM)\n";
        pool_ready = gpu.init_dense_pool(backoff_bytes);
    }

    int offload_ok = 0, offload_failed = 0;
    size_t offload_failed_bytes = 0;
    if (pool_ready) {
        for (QuantizedTensor* tensor : dense_candidates) {
            if (gpu.offload_tensor_to_pool(*tensor)) {
                ++offload_ok;
            } else {
                ++offload_failed;
                offload_failed_bytes += tensor->meta.size_bytes;
            }
        }
    } else {
        offload_failed = (int)dense_candidates.size();
        for (QuantizedTensor* tensor : dense_candidates) offload_failed_bytes += tensor->meta.size_bytes;
        LOG_STREAM_ERROR() << "[CUDA] Dense tensor pool could not be allocated at any size — "
                  << "all " << offload_failed << " tensor(s) staying host-resident.\n";
    }

    LOG_STREAM_INFO() << "[CUDA] Dense offload: " << offload_ok << " tensors on VRAM\n";
    if (offload_failed > 0) {
        LOG_STREAM_WARN() << "[CUDA] " << offload_failed << " tensor(s) ("
                  << offload_failed_bytes / 1e9 << " GB) did NOT fit in VRAM — "
                  << "staying host-resident (CPU dequant-matmul fallback for those). "
                  << "Increase dense_gb / use a smaller-precision dense_model_path if this "
                  << "matters for your workload.\n";
    }
    gpu.report_vram_usage("after dense tensor offload");

    LOG_STREAM_INFO() << "[I/O] Spinning up asynchronous lock-free SPSC prefetch threads...\n";
    prefetcher.start(&engine, &gpu.topology);

    std::cout << "----------------------------------------\n";
    std::cout << "  MoEStream engine online and ready. Enter prompt below.\n";
    std::cout << "Use ':q' to exit execution loop cleanly.\n\n";

    std::string user_input;
    while (true) {
        if (g_shutdown_requested.load(std::memory_order_relaxed)) break;

        std::cout << "\n  User > ";
        std::getline(std::cin, user_input);

        // Ctrl-C (SIGINT) sets the flag and, on most platforms, interrupts
        // the blocking getline with a failed/EOF stream state. Treat any of
        // those — flag set, real EOF (Ctrl-D / piped input ending), or a
        // failed read — as "time to shut down" rather than looping forever
        // on an unusable stream.
        if (g_shutdown_requested.load(std::memory_order_relaxed) || std::cin.eof() || std::cin.fail()) {
            break;
        }

        if (user_input == ":q" || user_input == "exit") {
            break;
        }
        if (user_input.empty()) continue;

        // BUG (found this session): user_input was read but never actually
        // tokenized -- prompt_tokens only ever contained BOS, so the model
        // never saw what the user typed, on every single request. That's
        // very likely the real explanation for "the same output regardless
        // of prompt": the input to the model WAS identical every time.
        std::vector<int> prompt_tokens;
        if (tokenizer.has_chat_template()) {
            // GLM-3/4-style: [gMASK]<sop><|user|>\n{msg}<|assistant|>\n
            // (see GGUFTokenizer::detect_chat_special_tokens for the
            // reference source and the caveat that GLM-5.2 specifically
            // isn't verified against this).
            if (tokenizer.gmask_id >= 0) prompt_tokens.push_back(tokenizer.gmask_id);
            if (tokenizer.sop_id >= 0) prompt_tokens.push_back(tokenizer.sop_id);
            prompt_tokens.push_back(tokenizer.user_id);
            std::vector<int> body = tokenizer.encode("\n" + user_input);
            prompt_tokens.insert(prompt_tokens.end(), body.begin(), body.end());
            prompt_tokens.push_back(tokenizer.assistant_id);
            std::vector<int> tail = tokenizer.encode("\n");
            prompt_tokens.insert(prompt_tokens.end(), tail.begin(), tail.end());
        } else {
            prompt_tokens.push_back(tokenizer.bos_token_id);
            std::vector<int> encoded = tokenizer.encode(user_input);
            prompt_tokens.insert(prompt_tokens.end(), encoded.begin(), encoded.end());
        }

        std::cout << "\n  MoEStream * ";
        generate_text(engine, ram_mgr, tokenizer, prompt_tokens, max_new_tokens, tokenizer.eos_token_id,
                      config.temperature, config.top_p);

        std::cout << "\n";
    }

    prefetcher.stop();
    perf().dump_report();
    return {};
}

int main(int argc, char* argv[]) {
    install_shutdown_handlers();  // Ctrl-C/SIGTERM -> graceful teardown (see shutdown.hpp)

    std::string config_path = "";
    std::string model_path = "";
    std::string gen_config_path = "";
    double ram_budget_gb = 0.0;
    int promote_threshold_cli = -1;
    int max_ctx = 4096;
    int max_new_tokens = 1024;
    int gpu_id = 0;
    bool drop_pages = false;
    bool verbose_moe = false;
    std::string log_path_cli;
    std::string log_level_cli;
    float temp_cli = -1.0f, top_p_cli = -1.0f;  // -1 sentinel = "not given" (0.0 is a valid greedy-decode value)

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) config_path = argv[++i];
        else if (arg == "--gen-config") {
            gen_config_path = (i + 1 < argc && argv[i+1][0] != '-') ? argv[++i] : "moestream_config.toml";
        }
        else if (arg == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "--ram" && i + 1 < argc) ram_budget_gb = std::stod(argv[++i]);
        else if (arg == "--ctx" && i + 1 < argc) max_ctx = std::stoi(argv[++i]);
        else if (arg == "--ngen" && i + 1 < argc) max_new_tokens = std::stoi(argv[++i]);
        else if (arg == "--gpu" && i + 1 < argc) gpu_id = std::stoi(argv[++i]);
        else if (arg == "--drop-pages") drop_pages = true;
        else if (arg == "--verbose-moe") verbose_moe = true;
        else if (arg == "--promote-threshold" && i + 1 < argc) promote_threshold_cli = std::stoi(argv[++i]);
        else if (arg == "--log-path" && i + 1 < argc) log_path_cli = argv[++i];
        else if (arg == "--log-level" && i + 1 < argc) log_level_cli = argv[++i];
        else if (arg == "--temp" && i + 1 < argc) temp_cli = std::stof(argv[++i]);
        else if (arg == "--top_p" && i + 1 < argc) top_p_cli = std::stof(argv[++i]);
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
        if (!config.load(config_path)) return 1;
        if (!config.validate()) return 1;
    } else if (!model_path.empty()) {
        // Build config from CLI arguments (backward compat)
        config.model_path = model_path;
        config.context_size = max_ctx;
        config.max_new_tokens = max_new_tokens;
        config.drop_pages = drop_pages;

        GpuConfig gc;
        gc.device_id = gpu_id;
        gc.role = "dense+moe";
        config.gpus.push_back(gc);

        if (!config.validate()) return 1;
    } else {
        std::cerr << "  Error: Specify --config <config.json> or --model <path_to_gguf>\n";
        print_usage();
        return 1;
    }

    // --ram / --verbose-moe always override the config file's values, if given
    if (ram_budget_gb > 0) config.host_ram_gb = ram_budget_gb;
    if (verbose_moe) config.verbose_moe_log = true;
    if (promote_threshold_cli > 0) config.promote_hit_threshold = promote_threshold_cli;
    if (!log_path_cli.empty()) config.log_path = log_path_cli;
    if (!log_level_cli.empty()) config.log_level = log_level_cli;
    if (temp_cli >= 0.0f) config.temperature = temp_cli;
    if (top_p_cli >= 0.0f) config.top_p = top_p_cli;

    // Initialize the logger before anything else logs — empty log_path keeps
    // everything on stderr exactly as before.
    Logger::instance().init(config.log_path, config.log_level);

    std::cout << "  MoEStream Engine initializing (C++23 mode)...\n";
    std::cout << "----------------------------------------\n";
    config.print();

    // cpu_only (TOML `cpu_only = true` or MOESTREAM_CPU_ONLY env var) forces all
    // compute onto the CPU reference path. Enable it BEFORE run_engine so every
    // dense_proj / rmsnorm_proj / expert_ffn dispatch sees the flag.
    if (config.cpu_only) enable_cpu_only_mode();

    auto outcome = run_engine(config);
    if (!outcome) {
        LOG_STREAM_ERROR() << "  Runtime Critical Failure: " << outcome.error() << "\n";
        return 1;
    }

    std::cout << "\n  Shutting down background tasks and reclaiming system handles..." << std::endl;
    return 0;
}