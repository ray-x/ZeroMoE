// engine_config.hpp — Runtime configuration for MoEStream multi-GPU inference
// Loaded from TOML config files. Specifies model paths, GPU roles, memory layout.
#pragma once
#include "logger.hpp"
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <unordered_map>
#include <cstdlib>
#include <cuda_runtime.h>

// Per-GPU configuration
struct GpuConfig {
    int device_id = -1;              // nvidia-smi GPU index
    std::string name;                // Human label, e.g. "NVIDIA GeForce RTX 3090"
    std::string role;                // "dense", "moe", "dense+moe"
    std::string model_path;          // GGUF file/dir for this GPU (optional override)
    double vram_gb = 0.0;            // Usable VRAM budget (0 = auto-detect)
    double dense_gb = 0.0;           // GB reserved for dense layers
    double kv_cache_gb = 0.0;        // GB reserved for KV cache
    int kv_bits = 4;                 // KV cache quantization (4 or 32)
    double moe_cache_gb = 0.0;       // GB reserved for expert LRU cache
};

// Full engine configuration
struct EngineConfig {
    // Model paths
    std::string model_path;          // Primary model GGUF (experts + architecture)
    std::string dense_model_path;    // Optional: separate GGUF for dense layers (e.g. 4-bit)

    // Generation parameters
    int context_size = 4096;         // Max context window (tokens)
    int max_new_tokens = 1024;       // Max tokens to generate per response
    float temperature = 0.7f;
    float top_p = 0.90f;
    bool drop_pages = false;         // fadvise DONTNEED after expert read
    bool cpu_only = false;           // force ALL compute to CPU (no GPU dispatch)

    // Host RAM (DDR) budget for the MoE expert LRU cache, in GB.
    // 0 = auto-detect (88% of currently-available system memory).
    double host_ram_gb = 0.0;

    // Log every host RAM cache HIT for the MoE expert cache (high volume —
    // one line per routed expert per token). MISS/LOAD and VRAM PROMOTE/READY
    // events are always logged regardless of this flag since they're rare.
    bool verbose_moe_log = false;

    // How many host-cache re-hits before triggering an async H2D promotion
    // into the GPU expert cache. Lower values promote earlier (lower latency),
    // higher values reduce unnecessary PCIe traffic.
    int promote_hit_threshold = 1;

    // Phase A colibri-port: pin a VRAM-resident expert (never evict it via
    // evict_lru) once its host-cache hit_count crosses this threshold --
    // "stop evicting hot experts within a conversation", no cross-prompt
    // profile needed. 0 disables pinning entirely (pure reactive LRU).
    int pin_threshold = 8;

    // Cross-conversation VRAM residency: at startup, read this usage file
    // (written by StreamingEngine::dump_usage at the end of a previous run)
    // and preload the top pin_topk_per_layer hottest experts per layer into
    // VRAM, pinned. Empty path disables preload (a usage file is still
    // written at shutdown for a future run). pin_topk_per_layer=0 auto-sizes
    // to fill available VRAM expert-cache capacity instead of a fixed
    // per-layer count.
    std::string pin_file = ".moestream_usage";
    int pin_topk_per_layer = 0;

    // How many OMP threads RAMSafetyManager::get_experts_batch uses to
    // parallelize the disk reads for one token's genuine (non-PIPE-served)
    // host-cache misses within a single layer's topk selection. Default 4,
    // chosen from a direct empirical measurement on the target machine
    // (colibri's iobench.c microbenchmark against the real model file):
    // buffered reads scaled ~8.5x from 1->4 threads (1.53 -> 13.12 GB/s)
    // then flattened (8 threads: 12.30 GB/s, no further gain) -- 4 captures
    // essentially the full available win on this hardware's real queue
    // depth without over-provisioning threads that would only add
    // contention against the OMP-parallel CPU dequant path running
    // elsewhere. Re-run iobench.c on a different machine/drive to re-tune.
    int expert_fetch_threads = 4;

    // Logging. Empty log_path (default) keeps everything on stderr, matching
    // prior behavior. log_level filters by severity: VERBOSE shows
    // high-volume structural traces (e.g. per-layer hidden-state norms
    // across the full model depth) on top of everything below; DEBUG shows
    // every per-token/per-expert diagnostic (cache hits/misses, VRAM
    // promotions, CPU dequant fallback notices); INFO (default) hides those
    // and shows only startup info and real warnings/errors.
    std::string log_path;
    std::string log_level = "INFO";

    // GPU configuration
    std::vector<GpuConfig> gpus;

    // Derived: which GPU index serves each role (set by validate())
    int dense_gpu_idx = -1;
    std::vector<int> moe_gpu_idxs;

    bool load(const std::string& path);
    bool validate();
    static void write_sample(const std::string& path);
    void print() const;
};

// ============================================================
// Minimal TOML parser — supports the subset we need:
//   key = "string"
//   key = 123          (int)
//   key = 1.5          (float)
//   key = true/false   (bool)
//   [section]          (table header — ignored, we use flat keys)
//   [[array]]          (array of tables — each creates a new entry)
//   # comments
// ============================================================
namespace toml {

inline std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    size_t b = s.find_last_not_of(" \t\r\n");
    return (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
}

inline std::string strip_comment(const std::string& line) {
    bool in_string = false;
    for (size_t i = 0; i < line.size(); ++i) {
        if (line[i] == '"') in_string = !in_string;
        if (line[i] == '#' && !in_string) return line.substr(0, i);
    }
    return line;
}

struct Value {
    std::string str;
    double num = 0;
    bool boolean = false;
    enum Type { String, Number, Bool } type = String;
};

struct Table {
    std::unordered_map<std::string, Value> values;

    bool has(const std::string& key) const { return values.count(key) > 0; }

    std::string get_str(const std::string& key, const std::string& def = "") const {
        auto it = values.find(key);
        return (it != values.end()) ? it->second.str : def;
    }
    int get_int(const std::string& key, int def = 0) const {
        auto it = values.find(key);
        return (it != values.end() && it->second.type == Value::Number) ? (int)it->second.num : def;
    }
    double get_double(const std::string& key, double def = 0.0) const {
        auto it = values.find(key);
        return (it != values.end() && it->second.type == Value::Number) ? it->second.num : def;
    }
    float get_float(const std::string& key, float def = 0.0f) const {
        return (float)get_double(key, def);
    }
    bool get_bool(const std::string& key, bool def = false) const {
        auto it = values.find(key);
        return (it != values.end() && it->second.type == Value::Bool) ? it->second.boolean : def;
    }
};

struct Document {
    Table root;                        // Top-level key = value pairs
    std::vector<Table> gpu_tables;     // [[gpus]] array-of-tables entries
};

inline Value parse_value(const std::string& raw) {
    Value v;
    std::string s = trim(raw);
    if (s.empty()) return v;

    // Quoted string
    if (s.front() == '"' && s.back() == '"') {
        v.type = Value::String;
        v.str = s.substr(1, s.size() - 2);
        return v;
    }
    // Boolean
    if (s == "true" || s == "false") {
        v.type = Value::Bool;
        v.boolean = (s == "true");
        v.str = s;
        return v;
    }
    // Number
    try {
        v.num = std::stod(s);
        v.type = Value::Number;
        v.str = s;
    } catch (...) {
        v.type = Value::String;
        v.str = s;
    }
    return v;
}

inline Document parse(const std::string& content) {
    Document doc;
    Table* current = &doc.root;
    bool in_gpu_array = false;

    std::istringstream stream(content);
    std::string line;
    while (std::getline(stream, line)) {
        line = trim(strip_comment(line));
        if (line.empty()) continue;

        // [[gpus]] — array of tables
        if (line == "[[gpus]]") {
            in_gpu_array = true;
            doc.gpu_tables.emplace_back();
            current = &doc.gpu_tables.back();
            continue;
        }

        // [section] — named table (not gpus)
        if (line.front() == '[' && line.back() == ']') {
            in_gpu_array = false;
            current = &doc.root;  // flatten non-gpu sections into root
            continue;
        }

        // key = value
        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = trim(line.substr(0, eq));
        std::string val_str = trim(line.substr(eq + 1));
        current->values[key] = parse_value(val_str);
    }
    return doc;
}

inline Document parse_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "[CONFIG] Cannot open: " << path << "\n";
        return {};
    }
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return parse(content);
}

} // namespace toml

// ============================================================
// EngineConfig implementation
// ============================================================
inline bool EngineConfig::load(const std::string& path) {
    auto doc = toml::parse_file(path);

    model_path       = doc.root.get_str("model_path");
    dense_model_path = doc.root.get_str("dense_model_path");
    context_size     = doc.root.get_int("context_size", context_size);
    max_new_tokens   = doc.root.get_int("max_new_tokens", max_new_tokens);
    temperature      = doc.root.get_float("temperature", temperature);
    top_p            = doc.root.get_float("top_p", top_p);
    drop_pages       = doc.root.get_bool("drop_pages", drop_pages);
    cpu_only         = doc.root.get_bool("cpu_only", cpu_only);
    host_ram_gb      = doc.root.get_double("host_ram_gb", host_ram_gb);
    verbose_moe_log  = doc.root.get_bool("verbose_moe_log", verbose_moe_log);
    promote_hit_threshold = doc.root.get_int("promote_hit_threshold", promote_hit_threshold);
    pin_threshold    = doc.root.get_int("pin_threshold", pin_threshold);
    pin_file         = doc.root.get_str("pin_file", pin_file);
    pin_topk_per_layer = doc.root.get_int("pin_topk_per_layer", pin_topk_per_layer);
    expert_fetch_threads = doc.root.get_int("expert_fetch_threads", expert_fetch_threads);
    log_path         = doc.root.get_str("log_path", log_path);
    log_level        = doc.root.get_str("log_level", log_level);

    for (auto& gt : doc.gpu_tables) {
        GpuConfig gc;
        gc.device_id    = gt.get_int("device_id", -1);
        gc.name         = gt.get_str("name");
        gc.role         = gt.get_str("role", "dense+moe");
        gc.model_path   = gt.get_str("model_path");
        gc.vram_gb      = gt.get_double("vram_gb");
        gc.dense_gb     = gt.get_double("dense_gb");
        gc.kv_cache_gb  = gt.get_double("kv_cache_gb");
        gc.kv_bits      = gt.get_int("kv_bits", 4);
        gc.moe_cache_gb = gt.get_double("moe_cache_gb");
        gpus.push_back(gc);
    }

    if (model_path.empty()) {
        std::cerr << "[CONFIG] Error: model_path is required.\n";
        return false;
    }

    std::cerr << "[CONFIG] Loaded from " << path << " (" << gpus.size() << " GPU entries)\n";
    return true;
}

inline bool EngineConfig::validate() {
    int hw_count = 0;
    cudaGetDeviceCount(&hw_count);

    std::vector<GpuConfig> valid_gpus;
    for (auto& gc : gpus) {
        if (gc.device_id < 0 || gc.device_id >= hw_count) {
            std::cerr << "[CONFIG] GPU " << gc.device_id << " not present, skipping.\n";
            continue;
        }
        if (gc.vram_gb <= 0) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, gc.device_id);
            gc.vram_gb = prop.totalGlobalMem / 1e9;
            if (gc.name.empty()) gc.name = prop.name;
        }
        valid_gpus.push_back(gc);
    }
    gpus = valid_gpus;

    if (gpus.empty()) {
        std::cerr << "[CONFIG] No valid GPUs configured.\n";
        return false;
    }

    // Assign roles
    dense_gpu_idx = -1;
    moe_gpu_idxs.clear();
    for (size_t i = 0; i < gpus.size(); ++i) {
        const std::string& r = gpus[i].role;
        if (r == "dense" || r == "dense+moe") {
            if (dense_gpu_idx < 0) dense_gpu_idx = (int)i;
        }
        if (r == "moe" || r == "dense+moe") {
            moe_gpu_idxs.push_back((int)i);
        }
    }

    if (dense_gpu_idx < 0) {
        std::cerr << "[CONFIG] No GPU with role 'dense' or 'dense+moe'. "
                  << "Assigning GPU 0 as dense+moe.\n";
        gpus[0].role = "dense+moe";
        dense_gpu_idx = 0;
        if (std::find(moe_gpu_idxs.begin(), moe_gpu_idxs.end(), 0) == moe_gpu_idxs.end())
            moe_gpu_idxs.push_back(0);
    }

    if (dense_model_path.empty()) dense_model_path = model_path;
    for (auto& gc : gpus) {
        if (gc.model_path.empty()) gc.model_path = model_path;
    }

    // Auto-compute memory layout if not specified
    for (auto& gc : gpus) {
        if (gc.role == "dense" || gc.role == "dense+moe") {
            double available = gc.vram_gb - 1.0;
            if (gc.dense_gb <= 0) gc.dense_gb = std::min(8.0, available * 0.5);
            if (gc.kv_cache_gb <= 0) gc.kv_cache_gb = std::min(5.0, available * 0.3);
            if (gc.role == "dense+moe" && gc.moe_cache_gb <= 0) {
                gc.moe_cache_gb = available - gc.dense_gb - gc.kv_cache_gb;
                if (gc.moe_cache_gb < 0) gc.moe_cache_gb = 0;
            }
        }
        if (gc.role == "moe") {
            if (gc.moe_cache_gb <= 0) gc.moe_cache_gb = gc.vram_gb - 1.0;
        }
    }

    return true;
}

inline void EngineConfig::print() const {
    LOG_STREAM_INFO() << "[CONFIG] ========== Engine Configuration ==========\n";
    LOG_STREAM_INFO() << "[CONFIG] Model: " << model_path << "\n";
    if (!dense_model_path.empty() && dense_model_path != model_path)
        LOG_STREAM_INFO() << "[CONFIG] Dense model: " << dense_model_path << "\n";
    LOG_STREAM_INFO() << "[CONFIG] Context: " << context_size << " tokens | Max gen: " << max_new_tokens << "\n";
    LOG_STREAM_INFO() << "[CONFIG] Temperature: " << temperature << " | Top-p: " << top_p << "\n";
    LOG_STREAM_INFO() << "[CONFIG] Drop pages: " << (drop_pages ? "yes" : "no") << "\n";
    if (cpu_only) LOG_STREAM_INFO() << "[CONFIG] CPU-only mode: ON (all GPU dispatch disabled)\n";
    LOG_STREAM_INFO() << "[CONFIG] Host RAM (DDR) budget: "
              << (host_ram_gb > 0 ? std::to_string(host_ram_gb) + " GB (explicit)" : "auto-detect") << "\n";
    LOG_STREAM_INFO() << "[CONFIG] Promote H2D threshold: " << promote_hit_threshold << " hits\n";
    LOG_STREAM_INFO() << "[CONFIG] Pin threshold: " << pin_threshold
              << (pin_threshold > 0 ? " hits" : " (pinning disabled)") << "\n";
    LOG_STREAM_INFO() << "[CONFIG] Pin file: " << (pin_file.empty() ? "(preload disabled)" : pin_file)
              << " | top-" << (pin_topk_per_layer > 0 ? std::to_string(pin_topk_per_layer) : std::string("auto"))
              << "/layer\n";
    LOG_STREAM_INFO() << "[CONFIG] Expert fetch threads: " << expert_fetch_threads << "\n";
    LOG_STREAM_INFO() << "[CONFIG] Verbose MoE cache logging: " << (verbose_moe_log ? "on" : "off") << "\n";
    LOG_STREAM_INFO() << "[CONFIG] Log level: " << log_level
               << " | Log path: " << (log_path.empty() ? "(stderr)" : log_path) << "\n";
    for (size_t i = 0; i < gpus.size(); ++i) {
        const auto& g = gpus[i];
        LOG_STREAM_INFO() << "[CONFIG] GPU " << g.device_id << " (" << g.name << ") — " << g.role
                  << " — " << g.vram_gb << " GB total\n";
        if (g.dense_gb > 0)
            LOG_STREAM_INFO() << "[CONFIG]   Dense: " << g.dense_gb << " GB\n";
        if (g.kv_cache_gb > 0)
            LOG_STREAM_INFO() << "[CONFIG]   KV Cache: " << g.kv_cache_gb << " GB (" << g.kv_bits << "-bit)\n";
        if (g.moe_cache_gb > 0)
            LOG_STREAM_INFO() << "[CONFIG]   MoE Cache: " << g.moe_cache_gb << " GB\n";
    }
    LOG_STREAM_INFO() << "[CONFIG] =============================================\n";
}

inline void EngineConfig::write_sample(const std::string& path) {
    std::ofstream f(path);
    f << R"(# MoEStream Engine Configuration
# See configs/ for example profiles (single_gpu, dual_gpu, triple_gpu)

model_path = "/models/GLM-5.2-UD-IQ2_M/"
# dense_model_path = "/models/GLM-5.2-Q4_K/"  # optional: separate dense weights

context_size   = 131072   # max context window (tokens)
max_new_tokens = 4096
temperature    = 0.7
top_p          = 0.95
drop_pages     = false    # true = evict expert pages from OS cache after use

# Host RAM (DDR) budget for the MoE expert LRU cache, in GB.
# 0 = auto-detect (88% of currently-available system memory).
host_ram_gb    = 0
verbose_moe_log = false
promote_hit_threshold = 1

# Phase A colibri-port: VRAM expert residency, not reactive-only eviction.
# pin_threshold: once a VRAM-resident expert's host-cache hit_count crosses
# this, it's pinned (never evicted) for the rest of the process -- stops
# thrashing hot experts out within one conversation. 0 disables pinning.
pin_threshold = 8
# pin_file: usage-frequency file (written at shutdown, read at next startup)
# used to preload the measured hot set into VRAM before the first prompt.
# Empty string disables preload (a usage file is still written at shutdown).
pin_file = ".moestream_usage"
# pin_topk_per_layer: how many of each layer's hottest experts to preload.
# 0 = auto-size to fill available VRAM expert-cache capacity.
pin_topk_per_layer = 0

# expert_fetch_threads: OMP threads used to parallelize disk reads for one
# token's genuine (non-PIPE-served) cache misses within a layer's topk
# selection. Default 4 -- tune with colibri's iobench.c microbenchmark
# against your own model file/drive (buffered reads, increasing thread
# count until throughput stops scaling) rather than assuming this value.
expert_fetch_threads = 4

# Logging. Empty log_path (default) keeps everything on stderr.
# log_level: "VERBOSE" | "DEBUG" | "INFO" (default) | "WARN" | "ERROR"
# VERBOSE adds high-volume structural traces (per-layer hidden-state norms
# across the full model depth) on top of DEBUG's per-token/per-expert
# diagnostics (cache hits/misses, VRAM promotions, CPU dequant fallback
# notices) -- both are very high volume, intended for active debugging only.
# log_path = "output.log"
log_level = "INFO"

# GPU 0: Core Engine — Dense layers + KV cache
[[gpus]]
device_id    = 0
name         = "NVIDIA GeForce RTX 5060 Ti"
role         = "dense"          # "dense", "moe", or "dense+moe"
vram_gb      = 16.0             # 0 = auto-detect
dense_gb     = 7.5              # GB for dense layers (attn, norms, embed, lm_head)
kv_cache_gb  = 4.5              # GB for KV cache
kv_bits      = 4                # KV cache quantization: 4 or 32
moe_cache_gb = 0                # no expert cache on this GPU

# GPU 1: Expert Vault — MoE LRU cache
[[gpus]]
device_id    = 1
name         = "NVIDIA GeForce RTX 3090"
role         = "moe"
vram_gb      = 24.0
moe_cache_gb = 23.0

# GPU 2: Expert Vault (optional — skipped if not present)
[[gpus]]
device_id    = 2
name         = "NVIDIA GeForce RTX 3090"
role         = "moe"
vram_gb      = 24.0
moe_cache_gb = 23.0
)";
    f.close();
    std::cerr << "[CONFIG] Sample config written to " << path << "\n";
}
