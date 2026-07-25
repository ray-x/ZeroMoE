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
    // O_DIRECT expert-slab reads (Linux only). Opt-in, requires the model
    // to have been repacked with tools/gguf_tool pack (or otherwise have
    // general.alignment >= 4096) -- see GGUFFile::enable_direct_io. A
    // plain/un-repacked GGUF silently falls back to buffered reads with a
    // logged warning, never a hard failure.
    bool direct_io = false;

    // Batch the async prefetcher's per-expert reads (up to 3: gate/up/down)
    // through io_uring instead of sequential blocking preads (Linux only,
    // kernel >= 5.6 for IORING_OP_READ). Opt-in, independent of direct_io --
    // composes with it either way (see io_uring_backend.hpp). Falls back to
    // the existing sequential path with a logged warning on any failure
    // (old kernel, io_uring disabled by seccomp/container policy, etc.),
    // never a hard failure. Prefetcher-only: does not touch the synchronous
    // decode-thread miss path (StreamingEngine::load_expert_slab).
    bool io_uring = false;

    // Number of concurrent async prefetcher worker threads pulling hints
    // off the SPSC queue and servicing them (disk read + optional async
    // H2D promotion). Was hardcoded to 1 -- real-run PIPE instrumentation
    // (n_pipe_miss_still_queued) showed a real, measurable fraction of
    // otherwise-correctly-predicted misses failing to be served purely
    // because a single worker thread hadn't even STARTED that hint's read
    // yet (pilot_k=8 hints pushed per layer transition, one thread
    // servicing them strictly one at a time). Each thread gets its own
    // io_uring ring when io_uring=true (a Ring is not shareable across
    // threads); the staging pool (prefetch_staging_slots, below) is still
    // shared and mutex-protected, so this doesn't change staging pool
    // capacity, only how fast hints get started. Kept modest by default --
    // these threads mostly block on
    // I/O rather than compute, but more of them means more concurrent
    // cudaMallocHost/staging_mutex contention too.
    int prefetch_worker_threads = 2;

    // Number of round-robin pinned-host staging slots the async prefetcher
    // holds prefetched-but-not-yet-consumed expert slabs in
    // (LockFreePrefetcher::staging, async_prefetcher.hpp). Was a hardcoded
    // compile-time constant (16, then bumped to 64) because PrefetchSlot
    // holds std::atomic<bool> members that aren't move-safe for a
    // std::vector -- now a runtime-sized array allocated once in
    // LockFreePrefetcher::start(), so this is a real knob. Real-run
    // instrumentation (n_staged_evicted_unused, "Staged-unused evict" in the
    // perf report) showed a fixed pool round-robin-evicting a correctly-
    // predicted, successfully-staged expert before the corresponding cache
    // miss ever consumed it -- i.e. the pool cycling faster than misses
    // arrive, not misprediction (Pilot recall stayed ~85% the whole time).
    // First bumped 64->128; a longer real run (32k context, 50 decode
    // tokens) showed hint volume grows with workload size roughly as fast
    // as the pool does, so 128 was STILL only good enough to keep ~2% of
    // staged experts alive to be consumed (Staged-unused evict 11387 of
    // 11671 hints processed) -- essentially unchanged from 64's ratio.
    // Raised again to 256. Each slot costs one pinned-host allocation sized
    // to the largest expert slab seen so far (grows lazily, see
    // acquire_staging_slot) -- budget roughly slots x largest-slab-size of
    // page-locked host RAM (a few GB at 256 slots for this project's expert
    // sizes), separate from and additional to moe_cache_gb/host_ram_gb --
    // watch total system memory if you raise this further.
    int prefetch_staging_slots = 256;

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

    // Phase A: pin a VRAM-resident expert (never evict it via
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

    // Multi-tier startup preload, extending the single top-K-pinned tier
    // above into a graduated scheme driven by the SAME usage file (which
    // already records every expert ever routed to, not just the top-K --
    // StreamingEngine::route_freq/dump_usage cover the full histogram).
    // Per layer, ranked by historical selection count, AFTER the
    // pin_topk_per_layer-sized pinned prefix:
    //   next round(pin_vram_lru_fraction * avg_vram_slots_per_layer) ranks
    //     -> VRAM, but NOT pinned -- a regular reactively-evictable slot,
    //     just pre-warmed instead of starting cold;
    //   next round(pin_ram_fraction * ram_mgr.slots_per_layer) ranks ->
    //     host RAM tier, preloaded (a genuinely new capability -- the RAM
    //     tier previously always started completely cold, filling only
    //     reactively during real generation).
    double pin_vram_lru_fraction = 0.0;
    double pin_ram_fraction = 0.0;

    // Three-tier VRAM caching configuration
    // Tier overlap fix: MOVE instead of COPY on RAM→VRAM promotion
    bool move_on_promotion = true;
    // Maximum number of permanently pinned experts (was ~600, targeting 1024)
    int pinned_vram_capacity = 1024;
    // Active layer area: temporary workspace for current-layer experts
    int active_layer_capacity = 256;
    bool use_active_layer_area = true;
    // Hit count (while resident in the active layer area, i.e. re-accessed
    // before this layer's window closes) at which an active-layer expert
    // gets promoted into the real VRAM LRU cache (GpuTopology::
    // promote_active_to_main, a device-to-device copy) instead of just
    // being freed once its layer finishes. Previously this whole promotion
    // path didn't exist at all -- the active layer area only ever held a
    // temporary hold-and-discard buffer with no way for a proven-hot
    // expert to escape into the real cache, and no hit-counting to know
    // "proven hot" in the first place (n_active_layer_hits was declared but
    // never incremented anywhere). Default 1 (promote on any repeat access)
    // -- much lower than RAMSafetyManager::promote_hit_threshold because
    // the active layer area's window (one layer, current token) is far
    // narrower than the RAM tier's whole-conversation lifetime, so a repeat
    // hit here is already a comparatively strong signal.
    int active_layer_promote_hit_threshold = 1;

    // When a VRAM cache slot is evicted (GpuTopology::evict_lru, via
    // upload_expert), salvage the outgoing expert's bytes into the RAM
    // tier with a D2H copy instead of letting them be silently dropped.
    // Previously a VRAM-evicted expert not ALSO already RAM-resident went
    // fully cold -- its next use always paid a full disk read even though
    // the identical bytes existed in VRAM a moment earlier. A PCIe D2H
    // copy of one expert slab costs roughly a millisecond vs. a 15-50ms+
    // NVMe re-read (this project's own measured [MOE-CACHE] MISS timings),
    // so this is a real, asymmetric win on paper -- but it's new,
    // untested on real hardware, and touches the same VRAM-eviction/
    // cross-thread-reuse territory that has produced multiple real
    // silent-corruption incidents earlier in this project's history (see
    // CLAUDE.md). Scoped deliberately narrow to avoid the worst of that
    // risk: only wired into upload_expert (decode-thread-only), never
    // upload_expert_async (the prefetcher thread's own promotion path),
    // so RAMSafetyManager::layers_experts keeps its existing
    // single-writer-thread invariant with no new locking required -- see
    // gpu_topology.hpp's on_vram_evict and ram_manager.hpp's
    // demote_from_vram for the full reasoning. Default false: opt-in only,
    // A/B test against a run with this disabled before trusting it (watch
    // dump_report()'s "VRAM->RAM demote:" line and Expert cache hit
    // rate/disk-read totals for the real effect).
    bool vram_evict_demote_to_ram = false;

    // Redistributes RAMSafetyManager's fixed total RAM-tier budget
    // (uniform slots_per_layer * n_layers) across layers proportional to
    // each layer's real historical need instead of splitting it evenly.
    // Motivated by direct measurement on this project's own accumulated
    // .moestream_usage file (75 layers, 672 selections/layer after
    // several sessions): the most routing-concentrated layer needed only
    // 31 of its own top-ranked experts to cover 80% of its traffic, the
    // flattest needed 92 -- a ~3x range the uniform per-layer split
    // ignores, while the RAM tier was simultaneously observed 100% full
    // (a real, binding constraint) in every real run this session.
    // Default false: opt-in, A/B test against the uniform baseline before
    // trusting it -- see ram_manager.hpp's apply_layer_adaptive_capacity
    // for the full algorithm and its budget-safety guarantees (the
    // redistributed total can never exceed what calculate_safe_cache_
    // capacity already determined is safe, by construction). Uses the
    // SAME `pin_file` usage history as the [PIN] preload block -- no
    // separate file/config needed. First-ever run (no usage history yet)
    // is a complete no-op, identical to leaving this false, since there's
    // no real signal yet to redistribute by.
    bool ram_cache_layer_adaptive = false;

    // How many OMP threads RAMSafetyManager::get_experts_batch uses to
    // parallelize the disk reads for one token's genuine (non-PIPE-served)
    // host-cache misses within a single layer's topk selection. Default 4,
    // chosen from a direct empirical measurement on the target machine
    // buffered reads scaled ~8.5x from 1->4 threads (1.53 -> 13.12 GB/s)
    // then flattened (8 threads: 12.30 GB/s, no further gain) -- 4 captures
    // essentially the full available win on this hardware's real queue
    // depth without over-provisioning threads that would only add
    // contention against the OMP-parallel CPU dequant path running
    // elsewhere. Re-run iobench.c on a different machine/drive to re-tune.
    int expert_fetch_threads = 4;

    // When true, main.cpp's REPL feeds stdin content to the tokenizer
    // VERBATIM -- skips the GLM-3/4-style chat-template wrap
    // ([gMASK]<sop><|user|>\n{msg}<|assistant|>\n) it would otherwise apply
    // whenever the tokenizer detects chat-template tokens in the vocab.
    // Two real uses: (1) fair prefill-time benchmarking against another
    // engine on an identical token count, since the wrap alone can more
    // than double a short prompt's token count; (2) required for
    // api/api_server.py, which already builds its own <|role|>content-
    // style prompt before ever reaching this engine -- without this flag
    // that pre-formatted text gets wrapped a SECOND time, producing
    // nested/duplicated special tokens (a real correctness bug, not just
    // extra tokens -- see CLAUDE.md). Default false: the REPL's own
    // template-detection behavior is unchanged for normal interactive use.
    bool raw_prompt = false;

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
    direct_io        = doc.root.get_bool("direct_io", direct_io);
    io_uring         = doc.root.get_bool("io_uring", io_uring);
    prefetch_worker_threads = doc.root.get_int("prefetch_worker_threads", prefetch_worker_threads);
    prefetch_staging_slots = doc.root.get_int("prefetch_staging_slots", prefetch_staging_slots);
    host_ram_gb      = doc.root.get_double("host_ram_gb", host_ram_gb);
    verbose_moe_log  = doc.root.get_bool("verbose_moe_log", verbose_moe_log);
    promote_hit_threshold = doc.root.get_int("promote_hit_threshold", promote_hit_threshold);
    pin_threshold    = doc.root.get_int("pin_threshold", pin_threshold);
    pin_file         = doc.root.get_str("pin_file", pin_file);
    pin_topk_per_layer = doc.root.get_int("pin_topk_per_layer", pin_topk_per_layer);
    pin_vram_lru_fraction = doc.root.get_double("pin_vram_lru_fraction", pin_vram_lru_fraction);
    pin_ram_fraction      = doc.root.get_double("pin_ram_fraction", pin_ram_fraction);
    vram_evict_demote_to_ram = doc.root.get_bool("vram_evict_demote_to_ram", vram_evict_demote_to_ram);
    ram_cache_layer_adaptive = doc.root.get_bool("ram_cache_layer_adaptive", ram_cache_layer_adaptive);
    expert_fetch_threads = doc.root.get_int("expert_fetch_threads", expert_fetch_threads);
    // NOTE: these four were previously missing from parsing entirely --
    // every configs/*.toml's `pinned_vram_capacity`/`move_on_promotion`/
    // `active_layer_capacity`/`use_active_layer_area` lines were silently
    // no-ops, only "working" because each one happened to match its
    // compiled default (1024/true/256/true). Any run that tried to
    // actually CHANGE one of these via TOML got the default instead, with
    // no warning.
    pinned_vram_capacity = doc.root.get_int("pinned_vram_capacity", pinned_vram_capacity);
    move_on_promotion    = doc.root.get_bool("move_on_promotion", move_on_promotion);
    active_layer_capacity = doc.root.get_int("active_layer_capacity", active_layer_capacity);
    use_active_layer_area = doc.root.get_bool("use_active_layer_area", use_active_layer_area);
    active_layer_promote_hit_threshold = doc.root.get_int("active_layer_promote_hit_threshold",
                                                            active_layer_promote_hit_threshold);
    raw_prompt           = doc.root.get_bool("raw_prompt", raw_prompt);
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
    LOG_STREAM_INFO() << "[CONFIG] Direct I/O (O_DIRECT expert reads): " << (direct_io ? "requested" : "off")
              << (direct_io ? " (falls back to buffered per-shard if unsupported/unaligned -- see [DIRECT-IO] log)" : "") << "\n";
    LOG_STREAM_INFO() << "[CONFIG] io_uring (batched prefetcher reads): " << (io_uring ? "requested" : "off")
              << (io_uring ? " (falls back to sequential preads if unsupported -- see [IO-URING] log)" : "") << "\n";
    LOG_STREAM_INFO() << "[CONFIG] Prefetch worker threads: " << prefetch_worker_threads << "\n";
    LOG_STREAM_INFO() << "[CONFIG] Prefetch staging slots: " << prefetch_staging_slots << "\n";
    if (cpu_only) LOG_STREAM_INFO() << "[CONFIG] CPU-only mode: ON (all GPU dispatch disabled)\n";
    LOG_STREAM_INFO() << "[CONFIG] Host RAM (DDR) budget: "
              << (host_ram_gb > 0 ? std::to_string(host_ram_gb) + " GB (explicit)" : "auto-detect") << "\n";
    LOG_STREAM_INFO() << "[CONFIG] Promote H2D threshold: " << promote_hit_threshold << " hits\n";
    LOG_STREAM_INFO() << "[CONFIG] Pin threshold: " << pin_threshold
              << (pin_threshold > 0 ? " hits" : " (pinning disabled)") << "\n";
    LOG_STREAM_INFO() << "[CONFIG] Pin file: " << (pin_file.empty() ? "(preload disabled)" : pin_file)
              << " | top-" << (pin_topk_per_layer > 0 ? std::to_string(pin_topk_per_layer) : std::string("auto"))
              << "/layer pinned"
              << ", next " << (pin_vram_lru_fraction * 100.0) << "% of VRAM tier capacity (unpinned)"
              << ", next " << (pin_ram_fraction * 100.0) << "% of RAM tier capacity\n";
    LOG_STREAM_INFO() << "[CONFIG] VRAM evict -> RAM demote: "
              << (vram_evict_demote_to_ram ? "ON (D2H-salvage evicted experts into the RAM tier)" : "off") << "\n";
    LOG_STREAM_INFO() << "[CONFIG] RAM cache layer-adaptive: "
              << (ram_cache_layer_adaptive ? "ON (per-layer capacity by historical need, not uniform)" : "off") << "\n";
    LOG_STREAM_INFO() << "[CONFIG] Expert fetch threads: " << expert_fetch_threads << "\n";
    LOG_STREAM_INFO() << "[CONFIG] Pinned VRAM capacity: " << pinned_vram_capacity
              << " | move_on_promotion: " << (move_on_promotion ? "MOVE" : "COPY") << "\n";
    LOG_STREAM_INFO() << "[CONFIG] Active layer area: "
              << (use_active_layer_area ? std::to_string(active_layer_capacity) + " slots, promote at "
                                           + std::to_string(active_layer_promote_hit_threshold) + " hit(s)"
                                         : "disabled") << "\n";
    if (raw_prompt)
        LOG_STREAM_INFO() << "[CONFIG] Raw prompt mode: ON -- REPL will NOT apply the chat "
                              "template, stdin content is tokenized verbatim\n";
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

# direct_io: read routed-expert slabs with O_DIRECT instead of buffered
# reads (Linux only). Requires the model to have been repacked with
# tools/gguf_tool pack (general.alignment >= 4096) -- a plain/un-repacked
# GGUF logs a warning and falls back to buffered reads automatically,
# never a hard failure. See tools/README.md for the repack workflow.
direct_io      = false

# io_uring: batch the async prefetcher's per-expert reads (up to 3:
# gate/up/down) through one io_uring submission instead of sequential
# blocking preads (Linux only, kernel >= 5.6). Opt-in, independent of
# direct_io -- combine both for O_DIRECT reads batched via io_uring, or use
# either alone. Falls back to sequential preads automatically on any
# failure (old kernel, sandboxed/container environment, etc.), never a hard
# failure. Only affects the background prefetcher, not the synchronous
# decode-thread miss path.
io_uring       = false

# prefetch_worker_threads: number of concurrent worker threads servicing
# the async prefetcher's hint queue (disk read + optional async H2D
# promotion). Default 2. Each gets its own io_uring ring when io_uring=true.
# More threads means hints get STARTED sooner (fixes a real "worker hadn't
# even begun this read yet" gap seen in PIPE instrumentation), at the cost
# of more concurrent contention on the staging pool (see
# prefetch_staging_slots below).
prefetch_worker_threads = 2

# prefetch_staging_slots: number of pinned-host round-robin slots the async
# prefetcher holds prefetched expert slabs in before a cache miss consumes
# them. Default 256 (was a hardcoded compile-time 64, then 128). Too small
# and a correctly-predicted, successfully-staged expert gets round-robin-
# evicted before the matching miss arrives -- see "Staged-unused evict" in
# the perf report; a high count there (as a fraction of total hints
# processed) means this is undersized relative to how many hints are in
# flight between pilot_k per-layer predictions and worker throughput. Hint
# volume scales with workload size (context length, tokens generated), so
# re-check this ratio after any change to those and raise further if it's
# still high. Each slot lazily grows its pinned allocation to the largest
# expert slab it has held, so cost scales with slot count x max slab size --
# a few GB of page-locked host RAM at 256 slots, separate from and
# additional to moe_cache_gb/host_ram_gb; watch total system memory if you
# raise this further.
prefetch_staging_slots = 256

# Host RAM (DDR) budget for the MoE expert LRU cache, in GB.
# 0 = auto-detect (88% of currently-available system memory).
host_ram_gb    = 0
verbose_moe_log = false
promote_hit_threshold = 1

# Phase A VRAM expert residency, not reactive-only eviction.
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
# pin_vram_lru_fraction: preload the next-hottest experts (ranked just below
# pin_topk_per_layer) into VRAM but leave UNPINNED -- a normal reactively-
# evictable slot, just pre-warmed instead of starting cold. Fraction of the
# VRAM tier's own real per-layer capacity, e.g. 0.1 = 10%, 0.3 = 30%.
# 0.0 = disabled (default -- real testing found nonzero defaults here made
# a short benchmark run SLOWER, more disk read, not less; A/B test your
# own workload before enabling
pin_vram_lru_fraction = 0.0
# pin_ram_fraction: same idea, next tier down, preloaded into the host RAM
# tier (previously always started completely cold). Fraction of
# ram_mgr.slots_per_layer. 0.0 = disabled (default, same reasoning as above).
pin_ram_fraction = 0.0

# vram_evict_demote_to_ram: when a VRAM cache slot is evicted, salvage its
# bytes into the RAM tier via a D2H copy instead of dropping them (so a
# later re-request serves from RAM instead of paying a full disk read
# again). New, untested on real hardware, opt-in only -- default false.
vram_evict_demote_to_ram = false

# ram_cache_layer_adaptive: give layers with flatter/less-predictable
# routing more RAM-tier cache slots and concentrated layers fewer, instead
# of splitting the total budget evenly -- uses the same pin_file usage
# history as the [PIN] preload above. No-op until that history exists.
# New, untested on real hardware, opt-in only -- default false.
ram_cache_layer_adaptive = false

# expert_fetch_threads: OMP threads used to parallelize disk reads for one
# token's genuine (non-PIPE-served) cache misses within a layer's topk
# selection. Default 4
# against your own model file/drive (buffered reads, increasing thread
# count until throughput stops scaling) rather than assuming this value.
expert_fetch_threads = 4

# Three-tier VRAM caching. pinned_vram_capacity is a global ceiling on
# permanently-pinned experts (see pin_topk_per_layer above -- the effective
# per-layer pin count is min(pin_topk_per_layer, floor(pinned_vram_capacity
# / total_layers)), uniform across layers with no remainder distribution,
# so exact round numbers aren't always reachable). move_on_promotion frees
# the RAM slot immediately on a RAM->VRAM promotion instead of leaving the
# expert double-resident in both tiers (see [TIER-OVERLAP] in the perf
# report if this regresses).
pinned_vram_capacity = 1024
move_on_promotion = true

# Active layer area: a small dedicated VRAM buffer holding the CURRENT
# layer's decode-time cold-miss experts, so a not-yet-proven-hot expert can
# still get a GPU dispatch without evicting a real LRU resident for it.
# active_layer_capacity slots, each sized to one expert slab -- costs
# roughly active_layer_capacity x largest-expert-slab-size of VRAM,
# additional to and separate from each GPU's own moe_cache_gb; make sure
# the vault GPU has that much genuinely free or allocation silently fails
# and this whole feature falls back to plain LRU behavior (see the
# "[GPU] Active layer area allocation failed" log line -- also check
# [GPU] Active layer area allocated for the actual GB requested).
# active_layer_promote_hit_threshold: hit count (repeat access from a LATER
# token while still resident) at which an active-layer expert is promoted
# into the real VRAM LRU cache instead of being freed once its layer's
# window closes -- default 1 since this window is much narrower than the
# RAM tier's whole-conversation lifetime.
use_active_layer_area = true
active_layer_capacity = 256
active_layer_promote_hit_threshold = 1

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
