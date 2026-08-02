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
#include <chrono>
#include <map>
#include <cmath>
#include <thread>
#include <omp.h>

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
              << "  --direct-io      O_DIRECT expert-slab reads (Linux only; requires a model\n"
              << "                   repacked with tools/gguf_tool pack -- falls back to buffered\n"
              << "                   reads automatically if unsupported/unaligned)\n"
              << "                   (unpacked_expert_format and cuda_cufile are TOML-config-only; "
                 "see --config)\n"
              << "  --io-uring       Batch the async prefetcher's per-expert reads through io_uring\n"
              << "                   (Linux only, kernel >= 5.6; independent of --direct-io -- falls\n"
              << "                   back to sequential preads automatically if unsupported)\n"
              << "  --prefetch-threads <N>  Concurrent async prefetcher worker threads (default: 2)\n"
              << "  --prefetch-staging-slots <N>  Pinned-host staging slots for the async prefetcher (default: 128)\n"
              << "  --verbose-moe    Log every MoE host-cache hit (high volume, debug only)\n"
              << "  --promote-threshold <N>  Number of host re-hits before async H2D promote (default: 1)\n"
              << "  --pin-threshold <N>  VRAM-resident hit_count before pinning (never evicted; default: 8, 0=off)\n"
              << "  --pin-file <file>  Usage-frequency file for VRAM hot-set preload (default: .moestream_usage)\n"
              << "  --expert-fetch-threads <N>  OMP threads for parallel disk reads of a layer's topk cache misses (default: 4)\n"
              << "  --raw-prompt     Skip the chat-template wrap; tokenize stdin verbatim (needed by\n"
              << "                   api_server.py, which already applies its own formatting; also\n"
              << "                   useful for token-count-fair benchmarking against other engines)\n"
              << "  --log-path <file>  Redirect logging to a file instead of stderr\n"
              << "  --log-level <lvl>  VERBOSE | DEBUG | INFO (default) | WARN | ERROR\n"
              << "  --help           Show this help interface\n";
}

std::expected<void, std::string> run_engine(const EngineConfig& config) {
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
        int omp_cap = config.omp_max_threads > 0
            ? config.omp_max_threads
            : std::max(1, (int)std::thread::hardware_concurrency() - 2);
        omp_set_num_threads(omp_cap);
        LOG_STREAM_INFO() << "[CONFIG] OMP process-wide default team size: " << omp_cap
                  << (config.omp_max_threads > 0 ? " (explicit)" : " (auto: hardware_concurrency-2)")
                  << "\n";
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
        ~CuFileDriverGuard() { if (opened) cufile_io::driver_close(); }
    } cufile_driver_guard;
    if (config.cuda_cufile) {
        cufile_driver_guard.opened = cufile_io::driver_open();
        if (cufile_driver_guard.opened) {
            LOG_STREAM_INFO() << "[CUFILE] Driver opened.\n";
        } else {
            LOG_STREAM_WARN() << "[CUFILE] Driver open failed (GDS/nvidia-fs not installed, or "
                                  "cuFile not compiled in) -- cuda_cufile falls back to the "
                                  "RAM-mediated upload path for this entire run.\n";
        }
    }

    StreamingEngine engine;
    RAMSafetyManager ram_mgr;
    GGUFTokenizer tokenizer;
    LockFreePrefetcher prefetcher;
    CudaManager gpu;

    const std::string& model_path = config.model_path;
    int max_ctx = config.context_size;
    int max_new_tokens = config.max_new_tokens;

    engine.drop_expert_pages = config.drop_pages;
    engine.unpacked_expert_format = config.unpacked_expert_format;
    engine.prefill_warm_via_prefetcher = config.prefill_warm_via_prefetcher;
    ram_mgr.configured_ram_gb = config.host_ram_gb;
    ram_mgr.promote_hit_threshold = (uint32_t)config.promote_hit_threshold;
    ram_mgr.pin_threshold = (uint32_t)std::max(0, config.pin_threshold);
    ram_mgr.verbose_cache_log = config.verbose_moe_log;
    ram_mgr.expert_fetch_threads = std::max(1, config.expert_fetch_threads);
    ram_mgr.move_on_promotion = config.move_on_promotion;
    gpu.topology.active_layer_promote_threshold = std::max(1, config.active_layer_promote_hit_threshold);
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
        gpu.topology.forced_core_device_id = config.gpus[config.dense_gpu_idx].device_id;
    }
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
    // Opt-in only (config.direct_io, default false) -- see GGUFFile::
    // enable_direct_io's own comment for the alignment check + per-shard
    // test-read verification this does before trusting O_DIRECT on any
    // shard. Called right after load_header (all shard fds + KV metadata
    // are populated by then) and before anything starts reading experts.
    if (config.direct_io) engine.model_file.enable_direct_io();

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

    // Optional separate dense GGUF: dense/preloaded tensors read from here, routed
    // experts still stream from model_path. Enables a dense(IQ4_XS)+expert(IQ2_*)
    // mix across two files.
    if (!config.dense_model_path.empty() && config.dense_model_path != model_path) {
        if (!engine.dense_model_file.load_header(config.dense_model_path)) {
            return std::unexpected("Failed to parse dense GGUF (dense_model_path)");
        }
        engine.has_dense_model = true;
        if (config.direct_io) engine.dense_model_file.enable_direct_io();
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
    engine.init_route_freq();

    // cpu_affinity_layers (config): must be set before ANYTHING below can
    // dispatch an expert or preload VRAM -- expert_ffn_resolve_gpu_loc,
    // RAMSafetyManager's reactive-hit promotion, Phase 2's prefill warm-up,
    // and the [PIN] preload block right below all consult
    // is_cpu_affinity_layer() (dense_dispatch.hpp). architecture_config.
    // n_layers is only known as of build_model_graph above, so (like the
    // unpacked-format sanity check right below) this is the earliest point
    // it can be set.
    if (!config.cpu_affinity_layers.empty()) {
        set_cpu_affinity_layers(config.cpu_affinity_layers, engine.model_file.architecture_config.n_layers);
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
        return std::unexpected("unpacked_expert_format=true but the model failed the aligned-unpacked "
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
        LOG_STREAM_WARN() << "[WARN] This model appears to be in the per-expert (\"unpacked\") tensor "
                              "format (gguf_tool verify would report 'Expert format: per-expert' for "
                              "it), but unpacked_expert_format is not set -- io_uring batching and the "
                              "VRAM-shadow bypass fast path will BOTH be silently disabled for this "
                              "entire run, forcing every dispatch through the slow, unbatched fallback "
                              "path. Set unpacked_expert_format=true in your config to use the "
                              "single-DMA fast path this format is built for. See CLAUDE.md.\n";
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
    gpu.setup_expert_cache(expert_slab_bytes, cfg.n_experts, config);
    gpu.report_vram_usage("after expert cache setup");

    // Connect VRAM expert cache to the RAM manager for LRU integration
    ram_mgr.vram_cache = &gpu.topology;
    ram_mgr.prefetcher = &prefetcher;

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
        // attn_k_b/attn_v_b are dequantized ONCE, CPU-side, straight from
        // host_ptr into the MLA absorption cache (wuk_cache/wuv_cache,
        // attention_layer.cpp's mla_attention_absorbed_gguf) -- dense_proj
        // (and therefore device_ptr) is NEVER called for these two specific
        // tensors. A VRAM-resident copy of them would be pure waste: bytes
        // uploaded, never read from GPU by anything. Excluded explicitly
        // since they'd otherwise match the broad "attn" substring below --
        // found while investigating what the "other_attention" VRAM bucket
        // (CLAUDE.md's dense-pool-composition breakdown) actually contains.
        if (name.find("attn_k_b.weight") != std::string::npos ||
            name.find("attn_v_b.weight") != std::string::npos) {
            return false;
        }
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
    auto is_attn_output = [](const std::string& name) {
        return name.find("attn_output.weight") != std::string::npos;
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
    size_t gate_inp_bf16_bytes_used = 0, gate_inp_f32_equivalent_bytes = 0;
    for (auto& [name, tensor] : engine.resident_layers) {
        if (!tensor.host_ptr || name.find("gate_inp.weight") == std::string::npos) continue;
        size_t remaining_free_vram = gpu.core_free_vram_bytes();
        if (gpu.upload_dequantized_bf16(tensor, remaining_free_vram)) {
            ++gate_inp_converted;
            gate_inp_bf16_bytes_used += (size_t)tensor.meta.shape[0] * (size_t)tensor.meta.shape[1] * sizeof(uint16_t);
            gate_inp_f32_equivalent_bytes += (size_t)tensor.meta.shape[0] * (size_t)tensor.meta.shape[1] * sizeof(float);
        } else {
            ++gate_inp_skipped;
        }
    }
    LOG_STREAM_INFO() << "[CUDA] gate_inp (MoE router) BF16 fast-path: "
              << gate_inp_converted << " layer(s) converted (" << gate_inp_bf16_bytes_used / 1e9
              << " GB, vs " << gate_inp_f32_equivalent_bytes / 1e9
              << " GB at original F32 -- " << (gate_inp_f32_equivalent_bytes - gate_inp_bf16_bytes_used) / 1e9
              << " GB saved), " << gate_inp_skipped << " layer(s) kept at original F32"
              << " -- NOTE: rounds router logits, a routing-behavior change, not just numerical noise.\n";

    constexpr size_t POOL_ALIGN = 256;
    std::vector<QuantizedTensor*> attn_output_candidates;
    size_t non_attn_output_pool_bytes = 0;
    // Worst-case reservation: what the regular dense pool would need if NONE
    // of attn_output converts (every candidate stays at its original
    // quantized size) -- computed with the SAME aligned-accumulation as
    // non_attn_output_pool_bytes / the real pool_bytes_needed below, so this
    // reservation and the real pool sizing agree on what "fits" means.
    size_t attn_output_quantized_total = 0;
    for (auto& [name, tensor] : engine.resident_layers) {
        if (!tensor.host_ptr || !is_dense_offload_candidate(name)) continue;
        if (tensor.gpu_fast_ptr) continue;  // gate_inp already fast-path converted above -- needs no regular-pool room
        if (is_attn_output(name)) {
            attn_output_candidates.push_back(&tensor);
            attn_output_quantized_total = (attn_output_quantized_total + POOL_ALIGN - 1) & ~(POOL_ALIGN - 1);
            attn_output_quantized_total += tensor.meta.size_bytes;
            continue;
        }
        non_attn_output_pool_bytes = (non_attn_output_pool_bytes + POOL_ALIGN - 1) & ~(POOL_ALIGN - 1);
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
        std::find(gpu.topology.vault_gpus.begin(), gpu.topology.vault_gpus.end(), gpu.topology.core_gpu)
        != gpu.topology.vault_gpus.end();
    double core_moe_cache_gb_for_reservation = 0.0;
    if (!already_primary_vault_for_reservation) {
        for (auto& gc : config.gpus) {
            if (gc.device_id == gpu.device_id) { core_moe_cache_gb_for_reservation = gc.moe_cache_gb; break; }
        }
    }
    size_t dual_role_slice_reservation = (size_t)(core_moe_cache_gb_for_reservation * 1e9);

    size_t worst_case_pool_bytes = non_attn_output_pool_bytes + attn_output_quantized_total;
    constexpr size_t FAST_PATH_SAFETY_MARGIN = (size_t)1 << 30;  // 1 GB, matches the dense-pool backoff's own margin style
    size_t free_before_fast_path = gpu.core_free_vram_bytes();
    size_t fast_path_reserved = worst_case_pool_bytes + FAST_PATH_SAFETY_MARGIN + dual_role_slice_reservation;
    size_t fast_path_budget =
        (free_before_fast_path > fast_path_reserved)
        ? (free_before_fast_path - fast_path_reserved)
        : 0;

    int fast_converted = 0, fast_skipped = 0;
    size_t fast_bytes_used = 0;    // gross BF16 bytes actually allocated (for the log line only)
    size_t budget_spent = 0;       // NET budget consumed: sum of (bf16_bytes - quant_bytes) over converted layers
    // Tracks what the SAME converted layers would have cost under the old
    // F32 fast-path -- comparison-only bookkeeping, doesn't affect which
    // layers convert or how much budget is spent. Exists purely to make
    // BF16's actual saving directly visible in the log line below instead
    // of requiring hand arithmetic against a prior run's numbers (see
    // CLAUDE.md -- this was confusing enough in practice to need its own
    // explicit log line).
    size_t f32_equivalent_bytes = 0;
    for (QuantizedTensor* qt : attn_output_candidates) {
        size_t quant_bytes = qt->meta.size_bytes;
        size_t bf16_bytes_estimate = (size_t)qt->meta.shape[0] * (size_t)qt->meta.shape[1] * sizeof(uint16_t);
        size_t net_cost = (bf16_bytes_estimate > quant_bytes) ? (bf16_bytes_estimate - quant_bytes) : 0;
        size_t remaining_net_budget = (fast_path_budget > budget_spent) ? (fast_path_budget - budget_spent) : 0;
        if (net_cost > remaining_net_budget) { ++fast_skipped; continue; }
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
            f32_equivalent_bytes += (size_t)qt->meta.shape[0] * (size_t)qt->meta.shape[1] * sizeof(float);
        } else {
            ++fast_skipped;
        }
    }
    LOG_STREAM_INFO() << "[CUDA] attn_output fast-path (BF16 dequant + matched-type cuBLAS): "
              << fast_converted << " layer(s) converted (" << fast_bytes_used / 1e9
              << " GB BF16, vs " << f32_equivalent_bytes / 1e9
              << " GB if these same layers had used the old F32 fast-path -- "
              << (f32_equivalent_bytes - fast_bytes_used) / 1e9 << " GB saved), "
              << fast_skipped << " layer(s) kept on the slower fused kernel"
              << " (net budget was " << fast_path_budget / 1e9 << " GB, worst-case pool reservation "
              << worst_case_pool_bytes / 1e9 << " GB).\n";

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
    struct DenseCategoryTotal { size_t bytes = 0; int count = 0; };
    std::map<std::string, DenseCategoryTotal> category_totals;
    auto categorize_dense_tensor = [](const std::string& name) -> const char* {
        if (name.find("attn_output.weight") != std::string::npos)
            return "attn_output (unconverted -- no BF16 fast-path budget left)";
        if (name.find("shexp") != std::string::npos)
            return "shared_expert (shexp)";
        if (name.find("ffn_gate.weight") != std::string::npos ||
            name.find("ffn_up.weight") != std::string::npos ||
            name.find("ffn_down.weight") != std::string::npos)
            return "leading_dense_ffn (pre-MoE layers)";
        if (name == "token_embd.weight" || name == "output.weight" || name == "output_norm.weight")
            return "embedding + LM head";
        if (name.find("gate_inp") != std::string::npos)
            return "moe_router (gate_inp)";
        if (name.find("norm") != std::string::npos)
            return "norms";
        if (name.find("attn") != std::string::npos)
            return "other_attention (q/k/v/kv_a projections)";
        return "other";
    };

    std::vector<QuantizedTensor*> dense_candidates;
    size_t pool_bytes_needed = 0;
    for (auto& [name, tensor] : engine.resident_layers) {
        if (!tensor.host_ptr || !is_dense_offload_candidate(name)) continue;
        if (tensor.gpu_fast_ptr) continue;  // already fast-path converted above
        dense_candidates.push_back(&tensor);
        pool_bytes_needed = (pool_bytes_needed + POOL_ALIGN - 1) & ~(POOL_ALIGN - 1);
        pool_bytes_needed += tensor.meta.size_bytes;
        auto& cat = category_totals[categorize_dense_tensor(name)];
        cat.bytes += tensor.meta.size_bytes;
        cat.count += 1;
    }

    // Largest-first so the dominant contributor is obvious without having
    // to scan every line.
    std::vector<std::pair<std::string, DenseCategoryTotal>> sorted_categories(
        category_totals.begin(), category_totals.end());
    std::sort(sorted_categories.begin(), sorted_categories.end(),
              [](const auto& a, const auto& b) { return a.second.bytes > b.second.bytes; });
    LOG_STREAM_INFO() << "[VRAM-BREAKDOWN] dense pool composition (requested "
              << pool_bytes_needed / 1e9 << " GB across " << dense_candidates.size()
              << " tensor(s)):\n";
    for (auto& [cat_name, tot] : sorted_categories) {
        LOG_STREAM_INFO() << "  " << cat_name << ": " << tot.bytes / 1e9
                  << " GB (" << tot.count << " tensor(s))\n";
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
    gpu.report_vram_breakdown("after dense tensor offload", fast_bytes_used, fast_converted);

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
    bool already_primary_vault = std::find(gpu.topology.vault_gpus.begin(), gpu.topology.vault_gpus.end(),
                                            gpu.topology.core_gpu) != gpu.topology.vault_gpus.end();
    double core_moe_cache_gb_cfg = 0.0;
    for (auto& gc : config.gpus) {
        if (gc.device_id == gpu.device_id) { core_moe_cache_gb_cfg = gc.moe_cache_gb; break; }
    }
    if (!already_primary_vault && core_moe_cache_gb_cfg > 0.0 && expert_slab_bytes > 0) {
        // Same 1 GB margin style as FAST_PATH_SAFETY_MARGIN above -- leaves
        // room for CUDA context growth, cuBLAS workspace, and prefill-time
        // scratch buffer growth (d_x/d_y/d_x_bf16) that hasn't happened yet
        // at this point in startup.
        constexpr size_t CORE_VAULT_SAFETY_MARGIN = (size_t)1 << 30;
        size_t requested_bytes = (size_t)(core_moe_cache_gb_cfg * 1e9);
        size_t real_free = gpu.core_free_vram_bytes();
        size_t safe_cap = (real_free > CORE_VAULT_SAFETY_MARGIN) ? (real_free - CORE_VAULT_SAFETY_MARGIN) : 0;
        size_t cap_bytes = std::min(requested_bytes, safe_cap);
        if (cap_bytes >= expert_slab_bytes) {
            gpu.topology.add_dual_role_expert_cache(gpu.topology.core_gpu, cap_bytes, expert_slab_bytes);
            gpu.report_vram_usage("after CoreEngine dual-role expert cache");
            gpu.report_vram_breakdown("after CoreEngine dual-role expert cache", fast_bytes_used, fast_converted);
        } else {
            LOG_STREAM_WARN() << "[GPU] CoreEngine dual-role expert cache requested ("
                      << core_moe_cache_gb_cfg << " GB) but only " << safe_cap / 1e9
                      << " GB safely free after reserving " << CORE_VAULT_SAFETY_MARGIN / 1e9
                      << " GB margin -- skipped (need at least one expert slab, "
                      << expert_slab_bytes / 1e9 << " GB).\n";
        }
    }

    LOG_STREAM_INFO() << "[I/O] Spinning up asynchronous lock-free SPSC prefetch threads...\n";
    prefetcher.start(&engine, &gpu.topology, config.io_uring, config.prefetch_worker_threads,
                      config.prefetch_staging_slots);
    // Decode thread's OWN io_uring ring for RAMSafetyManager::get_experts_batch's
    // Phase B (the synchronous miss path -- see ram_manager.hpp's decode_uring_ring
    // comment for why this is a separate scope from the prefetcher's rings above).
    ram_mgr.init_decode_uring(config.io_uring, config.io_uring_workers);

    // extended to a graduated 3-tier scheme (was:
    // a single top-K-pinned VRAM tier only). Read the usage file written
    // by a PREVIOUS run's dump_usage() at shutdown -- this already records
    // EVERY expert ever routed to (StreamingEngine::route_freq is a full
    // n_layers*n_experts histogram, not just a top-K sample), so all the
    // ranking data below has been available all along; what's new is
    // USING more of it. Per layer, ranked by historical selection count:
    //   ranks [0, pin_topk_per_layer)                              -> VRAM, PINNED
    //   ranks [pin_topk_per_layer, +pin_vram_lru_per_layer)         -> VRAM, unpinned (reactive LRU)
    //   ranks [..+pin_vram_lru_per_layer, +pin_ram_per_layer)       -> host RAM tier
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
    if (ram_mgr.vram_cache) ram_mgr.calculate_safe_cache_capacity(engine, config.context_size);

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
        ram_mgr.apply_layer_adaptive_capacity(config.pin_file, cfg.n_layers, cfg.n_experts);
    }

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
        ram_mgr.vram_cache->on_vram_evict = [&engine, &ram_mgr](int layer, int expert_id,
                const uint8_t* device_ptr, size_t bytes, int gpu_id) {
            ram_mgr.demote_from_vram(engine, layer, expert_id, device_ptr, bytes, gpu_id);
        };
    }

    if (!config.pin_file.empty() && ram_mgr.vram_cache && ram_mgr.vram_cache->expert_vram_cache_cap > 0) {
        auto [entries, maxc] = StreamingEngine::load_usage(config.pin_file);
        if (entries.empty()) {
            LOG_STREAM_INFO() << "[PIN] No usage file at " << config.pin_file
                      << " yet (first run) -- VRAM/RAM caches will fill reactively.\n";
        } else {
            std::unordered_map<int, std::vector<StreamingEngine::UsageEntry>> by_layer;
            for (auto& e : entries) by_layer[e.layer].push_back(e);
            for (auto& [layer, v] : by_layer) {
                std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.count > b.count; });
            }

            // prefill_pin_file: prefer ITS ranking for any layer it has data
            // for, since decode's higher routing volume can otherwise
            // outrank prefill's own needs in the combined pin_file above
            // (see StreamingEngine::route_freq_prefill's own comment).
            // Per-layer, not a blanket switch -- a layer this file has no
            // entries for (e.g. its very first run, before any prefill has
            // been recorded into it yet) keeps falling back to the general
            // pin_file's ranking for that specific layer, exactly as if
            // prefill_pin_file weren't configured at all.
            int prefill_layers_overridden = 0;
            if (!config.prefill_pin_file.empty()) {
                auto [pf_entries, pf_maxc] = StreamingEngine::load_usage(config.prefill_pin_file);
                (void)pf_maxc;
                if (pf_entries.empty()) {
                    LOG_STREAM_INFO() << "[PIN] prefill_pin_file (" << config.prefill_pin_file
                              << ") has no data yet -- every layer falls back to the general pin_file "
                                 "ranking for this run.\n";
                } else {
                    std::unordered_map<int, std::vector<StreamingEngine::UsageEntry>> pf_by_layer;
                    for (auto& e : pf_entries) pf_by_layer[e.layer].push_back(e);
                    for (auto& [layer, v] : pf_by_layer) {
                        std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.count > b.count; });
                        by_layer[layer] = std::move(v);  // override this layer only
                        ++prefill_layers_overridden;
                    }
                    LOG_STREAM_INFO() << "[PIN] prefill_pin_file (" << config.prefill_pin_file << ", "
                              << pf_entries.size() << " known experts): overriding ranking for "
                              << prefill_layers_overridden << " layer(s); remaining layers use the "
                                 "general pin_file ranking.\n";
                }
            }

            // Real per-layer capacity for each tier -- used both by
            // pin_topk_per_layer's "auto" mode (unchanged) and as the
            // DENOMINATOR for the fraction-based tier 2/3 knobs below, so
            // "0.1" means the same relative thing (10% of that tier's own
            // real capacity) regardless of how much VRAM/RAM a given
            // deployment actually has.
            int moe_layers = std::max(1, (int)by_layer.size());
            double avg_vram_slots_per_layer =
                (double)ram_mgr.vram_cache->expert_vram_cache_cap / (double)moe_layers;

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
                topk = std::min(topk, max_pinned_total / moe_layers);
                LOG_STREAM_INFO() << "[PIN] Three-tier: capped per-layer topk to " << topk
                          << " (global cap: " << max_pinned_total << " experts across "
                          << moe_layers << " layers)\n";
            }

            int vram_lru_n = std::max(0, (int)std::lround(
                config.pin_vram_lru_fraction * avg_vram_slots_per_layer));
            int ram_n = std::max(0, (int)std::lround(
                config.pin_ram_fraction * ram_mgr.slots_per_layer));

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
            if (global_topn) {
                int floor_n = std::max(0, config.pin_global_floor_per_layer);
                int floor_total = 0;
                for (auto& [layer, v] : by_layer) {
                    int f = std::min(floor_n, (int)v.size());
                    layer_pin_count[layer] = f;
                    floor_total += f;
                }
                struct GlobalCand { int layer; uint64_t count; };
                std::vector<GlobalCand> remainder;
                for (auto& [layer, v] : by_layer) {
                    int f = layer_pin_count[layer];
                    for (int i = f; i < (int)v.size(); ++i)
                        remainder.push_back({layer, v[i].count});
                }
                std::sort(remainder.begin(), remainder.end(),
                          [](const GlobalCand& a, const GlobalCand& b) { return a.count > b.count; });
                // If pinned_vram_capacity isn't set, fall back to the REAL VRAM
                // pool size (same quantity the per_layer_topk path's own "auto"
                // topk sizing uses: avg_vram_slots_per_layer * moe_layers) --
                // NOT "take everything remaining", which could try to pin every
                // expert ever routed to and blow far past physical VRAM capacity.
                int effective_cap = (max_pinned_total > 0)
                    ? max_pinned_total
                    : (int)(avg_vram_slots_per_layer * moe_layers);
                int remaining_budget = std::max(0, effective_cap - floor_total);
                int take = std::min((int)remainder.size(), remaining_budget);
                for (int i = 0; i < take; ++i) layer_pin_count[remainder[i].layer]++;
                LOG_STREAM_INFO() << "[PIN] global_topn: floor " << floor_n << "/layer (" << floor_total
                          << " total) + " << take << " globally-ranked -> " << (floor_total + take)
                          << "/" << effective_cap
                          << (max_pinned_total > 0 ? "" : " (pinned_vram_capacity unset -- capped to real VRAM pool size instead)")
                          << " target pins across " << moe_layers << " layer(s)\n";
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
                for (auto& [layer, v] : by_layer) {
                    uint64_t layer_total = 0;
                    for (auto& e : v) layer_total += e.count;
                    if (layer_total == 0) continue;
                    uint64_t topk_sum = 0;
                    int k = global_topn ? std::min(layer_pin_count[layer], (int)v.size())
                                        : std::min(topk, (int)v.size());
                    for (int i = 0; i < k; ++i) topk_sum += v[i].count;
                    double coverage = (double)topk_sum / (double)layer_total;
                    coverage_sum += coverage;
                    ++coverage_n;
                    if (coverage < coverage_min) { coverage_min = coverage; layer_min = layer; }
                    if (coverage > coverage_max) { coverage_max = coverage; layer_max = layer; }
                    LOG_STREAM_DEBUG() << "[PIN-CONCENTRATION] layer=" << layer << " top-" << k
                              << " covers " << (coverage * 100.0) << "% of "
                              << v.size() << " distinct experts seen / " << layer_total
                              << " total historical selections\n";
                }
                if (coverage_n > 0) {
                    LOG_STREAM_INFO() << "[PIN] "
                              << (global_topn ? "global_topn (per-layer K varies, see above)"
                                              : ("Top-" + std::to_string(topk) + "/layer"))
                              << " concentration: avg "
                              << (coverage_sum / coverage_n * 100.0) << "% of each layer's historical "
                              << "traffic covered by its pinned set -- most concentrated: layer "
                              << layer_max << " (" << (coverage_max * 100.0) << "%), most uniform: layer "
                              << layer_min << " (" << (coverage_min * 100.0) << "%). A layer near the "
                              << "low end gains little from pinning -- its routing is close to evenly "
                              << "spread, so a new prompt is about as likely to hit an unpinned expert "
                              << "as a pinned one. Full per-layer breakdown at log_level=DEBUG "
                              << "([PIN-CONCENTRATION] lines).\n";
                }
            }

            LOG_STREAM_INFO() << "[PIN] Preloading from " << config.pin_file << " (" << entries.size()
                      << " known experts, max count " << maxc << "): "
                      << (global_topn ? "global_topn ranking (see above)"
                                      : ("top-" + std::to_string(topk) + "/layer VRAM-pinned"))
                      << ", next " << vram_lru_n << "/layer VRAM-unpinned, next "
                      << ram_n << "/layer RAM...\n";
            int pinned = 0, vram_lru = 0, ram_preloaded = 0, attempted = 0;
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
            int total_pinned = 0;  // Three-tier: track total pinned count
            for (auto& [layer, v] : by_layer) {
                int n = (int)v.size();
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
                int layer_vram_lru_n = cpu_affinity ? 0 : vram_lru_n;
                // Three-tier: enforce global pinned capacity per layer.
                // global_topn already computed a final per-layer count
                // (floor + its share of the global ranking) up front; the
                // default per_layer_topk path keeps its EXACT original
                // running-total cap, unchanged.
                int layer_max;
                if (cpu_affinity) {
                    layer_max = 0;
                } else if (global_topn) {
                    layer_max = layer_pin_count[layer];
                } else {
                    layer_max = topk;
                    if (max_pinned_total > 0) {
                        layer_max = std::min({topk, (int)v.size(),
                                             max_pinned_total - total_pinned});
                    }
                }
                // Tier 2/3 start right after however many THIS layer got
                // pinned -- for per_layer_topk that's always exactly topk
                // (identical to the original code), but for global_topn it
                // varies per layer, so it can't be the shared topk anymore.
                // A cpu_affinity layer's layer_max is already 0, so
                // tier2_start correctly comes out to 0 too via the
                // global_topn branch (layer_max) regardless of topk.
                int tier2_start = (cpu_affinity || global_topn) ? layer_max : topk;
                for (int i = 0; i < n && i < layer_max; ++i) {
                    ++attempted;
                    MoEExpertBlock scratch{};
                    engine.load_expert_slab(v[i].layer, v[i].expert, scratch);
                    if (scratch.slab_buffer && scratch.slab_capacity > 0) {
                        ExpertLocation* loc = gpu.topology.upload_expert(
                            v[i].layer, v[i].expert, scratch.slab_buffer, scratch.slab_capacity);
                        if (loc && gpu.topology.pin_expert(v[i].layer, v[i].expert)) {
                            ++pinned;
                            ++total_pinned;  // Three-tier: track global pinned count
                        } else if (!loc) {
                            ++tier1_upload_fail;
                        }
                    } else {
                        ++tier1_disk_fail;
                    }
                    if (scratch.slab_buffer) free(scratch.slab_buffer);
                }
                // Tier 2: next vram_lru_n ranks -> VRAM, but NOT pinned --
                // a normal reactively-evictable slot, just pre-warmed
                // instead of starting cold. Reuses the exact same
                // upload_expert() call as Tier 1, just skips pin_expert().
                for (int i = tier2_start; i < n && i < tier2_start + layer_vram_lru_n; ++i) {
                    ++attempted;
                    MoEExpertBlock scratch{};
                    engine.load_expert_slab(v[i].layer, v[i].expert, scratch);
                    if (scratch.slab_buffer && scratch.slab_capacity > 0) {
                        if (gpu.topology.upload_expert(v[i].layer, v[i].expert,
                                                        scratch.slab_buffer, scratch.slab_capacity)) {
                            ++vram_lru;
                        } else {
                            ++tier2_upload_fail;
                        }
                    } else {
                        ++tier2_disk_fail;
                    }
                    if (scratch.slab_buffer) free(scratch.slab_buffer);
                }
                // Tier 3: next ram_n ranks -> host RAM tier directly.
                // Bounded to ram_mgr.slots_per_layer (see comment above) and
                // to this layer's own slot-array size for safety; writes
                // into slot indices [0, count) -- safe because every slot
                // starts empty (expert_id=-1) at startup, before any
                // reactive fill has happened.
                int ram_cap = std::min({ram_n, ram_mgr.cap_for_layer(layer),
                                         (int)engine.layers_experts[layer].size()});
                int slot_idx = 0;
                for (int i = tier2_start + layer_vram_lru_n; i < n && slot_idx < ram_cap; ++i, ++slot_idx) {
                    ++attempted;
                    MoEExpertBlock& slot = engine.layers_experts[layer][slot_idx];
                    engine.load_expert_slab(v[i].layer, v[i].expert, slot);
                    if (slot.slab_buffer && slot.slab_capacity > 0) {
                        slot.last_used_clock = ++ram_mgr.global_clock;
                        slot.hit_count = 0;
                        ++ram_preloaded;
                        // Writes directly into layers_experts, bypassing
                        // RAMSafetyManager::get_expert/get_experts_batch
                        // entirely -- so ram_expert_index (added to fix
                        // get_experts_batch's O(cap) hit-check scan, see
                        // CLAUDE.md) never gets an entry for a Tier-3-
                        // preloaded expert unless this is added explicitly.
                        // Without this, EVERY preloaded expert's first
                        // runtime lookup would look like a genuine index
                        // miss, which is indistinguishable from "really not
                        // resident" once find_ram_slot stops falling back to
                        // a linear scan on absence (see that function's own
                        // comment for why the fallback-on-miss had to go).
                        ram_mgr.ram_expert_index[RAMSafetyManager::swap_key(v[i].layer, v[i].expert)]
                            = (size_t)slot_idx;
                    }
                }
            }
            LOG_STREAM_INFO() << "[PIN] Preloaded " << pinned << " VRAM-pinned + " << vram_lru
                      << " VRAM-unpinned + " << ram_preloaded << " RAM (of " << attempted
                      << " attempted).\n";
            if (tier1_disk_fail || tier1_upload_fail || tier2_disk_fail || tier2_upload_fail) {
                LOG_STREAM_WARN() << "[PIN] Preload attrition -- Tier1(pinned): " << tier1_disk_fail
                          << " disk-read failures, " << tier1_upload_fail << " GPU-upload rejections | "
                          << "Tier2(unpinned): " << tier2_disk_fail << " disk-read failures, "
                          << tier2_upload_fail << " GPU-upload rejections "
                          << "(see [MOE-VRAM] upload_expert error lines above for the specific "
                          << "cudaMemcpy/evict_lru cause of any upload rejections).\n";
            }

            // Three-tier caching: Log final pinned count
            if (max_pinned_total > 0) {
                LOG_STREAM_INFO() << "[PIN] Three-tier: " << total_pinned << " experts pinned (target: "
                          << max_pinned_total << ") across " << moe_layers << " layers ("
                          << (total_pinned >= max_pinned_total ? "target met" : "below target")
                          << ")\n";
            }
        }
    }

    std::cout << "----------------------------------------\n";
    std::cout << "  MoEStream engine online and ready. Enter prompt below.\n";
    std::cout << "Use ':q' to exit execution loop cleanly.\n\n";

    // Real wall-clock anchor for perf().dump_report()'s final Throughput
    // line -- spans every turn in this session (not just one request),
    // starting AFTER model load (reported separately as "Dense load
    // (init)") so it isolates generation time specifically. See
    // dump_report()'s own comment for why this replaced a broken
    // summed-perf-counter calculation that double-counted concurrent work.
    auto session_start = std::chrono::high_resolution_clock::now();

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
        if (!config.raw_prompt && tokenizer.has_chat_template()) {
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
        generate_text(engine, ram_mgr, tokenizer, prefetcher, prompt_tokens, max_new_tokens, tokenizer.eos_token_id,
                      config.temperature, config.top_p);

        std::cout << "\n";
    }

    prefetcher.stop();
    // Phase A write this run's routing-frequency histogram so
    // the NEXT run's startup preload (above) has real data to work from.
    // Best-effort, matches dump_usage's own contract (a failed write just
    // logs and continues -- not worth failing shutdown over).
    engine.dump_usage(config.pin_file);
    // Prefill-only histogram -- see route_freq_prefill's own comment.
    // No-op (dump_usage_histogram's own empty-path guard) if
    // prefill_pin_file isn't configured.
    engine.dump_usage_prefill(config.prefill_pin_file);

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
    report_named_timers();  // attn_q_a/attn_q_b/attn_kv_a_mqa fast-path investigation, see CLAUDE.md

    double session_elapsed_s = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - session_start).count();
    perf().dump_report(session_elapsed_s);
    return {};
}

int main(int argc, char* argv[]) {
    install_shutdown_handlers();  // Ctrl-C/SIGTERM -> graceful teardown (see shutdown.hpp)

    std::string config_path = "";
    std::string model_path = "";
    std::string gen_config_path = "";
    double ram_budget_gb = 0.0;
    int promote_threshold_cli = -1;
    int pin_threshold_cli = -1;
    std::string pin_file_cli;
    int expert_fetch_threads_cli = -1;
    int prefetch_worker_threads_cli = -1;
    int prefetch_staging_slots_cli = -1;
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
        else if (arg == "--ctx" && i + 1 < argc) { max_ctx = std::stoi(argv[++i]); ctx_cli_given = true; }
        else if (arg == "--ngen" && i + 1 < argc) { max_new_tokens = std::stoi(argv[++i]); ngen_cli_given = true; }
        else if (arg == "--gpu" && i + 1 < argc) gpu_id = std::stoi(argv[++i]);
        else if (arg == "--drop-pages") drop_pages = true;
        else if (arg == "--direct-io") direct_io_cli = true;
        else if (arg == "--io-uring") io_uring_cli = true;
        else if (arg == "--verbose-moe") verbose_moe = true;
        else if (arg == "--raw-prompt") raw_prompt_cli = true;
        else if (arg == "--promote-threshold" && i + 1 < argc) promote_threshold_cli = std::stoi(argv[++i]);
        else if (arg == "--pin-threshold" && i + 1 < argc) pin_threshold_cli = std::stoi(argv[++i]);
        else if (arg == "--pin-file" && i + 1 < argc) pin_file_cli = argv[++i];
        else if (arg == "--expert-fetch-threads" && i + 1 < argc) expert_fetch_threads_cli = std::stoi(argv[++i]);
        else if (arg == "--prefetch-threads" && i + 1 < argc) prefetch_worker_threads_cli = std::stoi(argv[++i]);
        else if (arg == "--prefetch-staging-slots" && i + 1 < argc) prefetch_staging_slots_cli = std::stoi(argv[++i]);
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
        config.direct_io = direct_io_cli;
        config.io_uring = io_uring_cli;

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
    if (raw_prompt_cli) config.raw_prompt = true;
    if (drop_pages) config.drop_pages = true;
    if (direct_io_cli) config.direct_io = true;
    if (io_uring_cli) config.io_uring = true;
    // ctx_cli_given/ngen_cli_given: see their own declaration comment --
    // these were completely missing from this override block before,
    // meaning --ctx/--ngen had zero effect whenever --config was used.
    if (ctx_cli_given) config.context_size = max_ctx;
    if (ngen_cli_given) config.max_new_tokens = max_new_tokens;
    if (promote_threshold_cli > 0) config.promote_hit_threshold = promote_threshold_cli;
    if (pin_threshold_cli >= 0) config.pin_threshold = pin_threshold_cli;
    if (!pin_file_cli.empty()) config.pin_file = pin_file_cli;
    if (expert_fetch_threads_cli > 0) config.expert_fetch_threads = expert_fetch_threads_cli;
    if (prefetch_worker_threads_cli > 0) config.prefetch_worker_threads = prefetch_worker_threads_cli;
    if (prefetch_staging_slots_cli > 0) config.prefetch_staging_slots = prefetch_staging_slots_cli;
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