#pragma once
#include "gguf_loader.hpp"
#include "perf_timer.hpp"
#include "logger.hpp"
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <thread>
#include <future>
#include <vector>
#include <memory>
#include <unordered_map>

#ifdef __APPLE__
// macOS doesn't have posix_fadvise; provide no-op stubs
#ifndef POSIX_FADV_WILLNEED
#define POSIX_FADV_WILLNEED 3
#define POSIX_FADV_DONTNEED 4
#endif
static inline int posix_fadvise(int, off_t, off_t, int) { return 0; }
#endif

class CudaManager;  // forward decl — defined in cuda_manager.hpp; engine holds a pointer

struct QuantizedTensor {
    GGUFTensor meta;
    uint8_t* device_ptr = nullptr;
    uint8_t* host_ptr = nullptr;
};

// Q4_0 KV cache block: 32 values in 18 bytes (4.5 bits/value effective)
// Used when kv_quantized=true to compress KV cache by 8x vs FP32
struct BlockQ4_KV {
    uint16_t d;       // FP16 scale (2 bytes)
    uint8_t  qs[16];  // 32 x 4-bit values packed (16 bytes)
};

struct MoEExpertBlock {
    int expert_id;
    QuantizedTensor gate_proj;
    QuantizedTensor up_proj;
    QuantizedTensor down_proj;
    uint8_t* slab_buffer = nullptr;
    size_t slab_capacity = 0;
    uint64_t last_used_clock = 0;
    uint32_t hit_count = 0;  // frequency signal for eviction scoring (power-law routing)
};

class StreamingEngine {
public:
    GGUFFile model_file;
    GGUFFile dense_model_file;      // separate dense GGUF (config.dense_model_path); optional
    bool has_dense_model = false;
    std::unordered_map<std::string, QuantizedTensor> resident_layers;
    std::vector<std::vector<MoEExpertBlock>> layers_experts; 
    std::vector<float*> Lc;   // Host KV latent cache [n_layers+1][max_t * kv_lora]
    std::vector<float*> Rc;   // Host KV rope cache [n_layers+1][max_t * qk_rope]

    // Dense KV cache for non-MLA arches (Hunyuan/Llama): [n_layers][max_t * n_head_kv * head_dim]
    std::vector<float*> Kc;
    std::vector<float*> Vc;

    // GPU-resident KV cache (allocated when VRAM available)
    std::vector<float*> d_Lc;  // Device pointers for latent cache (FP32)
    std::vector<float*> d_Rc;  // Device pointers for rope cache (FP32)
    bool kv_on_device = false; // true if KV cache is in VRAM

    // Q4-compressed KV cache (8x savings over FP32, for large contexts)
    std::vector<BlockQ4_KV*> d_Lc_q4;  // Device Q4 latent cache [n_layers][max_t * kv_lora / 32 blocks]
    std::vector<BlockQ4_KV*> d_Rc_q4;  // Device Q4 rope cache
    bool kv_quantized = false;          // true when using Q4 KV cache

    // GPU-resident hidden state (persistent, never freed mid-session)
    float* d_hidden = nullptr;
    size_t d_hidden_cap = 0;

    int max_t = 0;

    // GPU dense-matmul dispatch handle (set by main.cpp after CudaManager
    // initialization). Forward-declared above (CudaManager itself, to avoid a
    // circular include with cuda_manager.hpp) — cuda_runtime.h is included
    // directly below so this header can free its own device allocations.
    CudaManager* gpu = nullptr;

    // Core GPU device id, captured once (by main.cpp, alongside `gpu` above)
    // rather than read back via gpu->get_core_device() at destruction time —
    // `gpu` may already be destroyed by then depending on local-variable
    // declaration order in run_engine(). -1 means "no GPU was ever attached"
    // (device buffers below were never allocated, destructor skips them).
    int gpu_device_id = -1;

    // Configuration: page cache policy for expert slabs after reading
    // false = leave pages in OS page cache (free L2 for hot experts, colibri's default)
    // true = POSIX_FADV_DONTNEED after read (reduces memory pressure under tight RAM)
    bool drop_expert_pages = false;

    // Frees every buffer this class allocated directly: host-side
    // posix_memalign'd tensors (resident dense weights, KV cache, expert
    // slabs) and GPU-side cudaMalloc'd buffers (offloaded dense tensors, KV
    // cache mirrors). Previously there was no destructor at all — every one
    // of these leaked until the OS/CUDA driver reclaimed the whole process
    // at exit, which "worked" for a single run but meant Ctrl-C (or any
    // path that doesn't kill the process) never actually released anything.
    ~StreamingEngine() {
        bool have_gpu = (gpu_device_id >= 0);
        if (have_gpu) cudaSetDevice(gpu_device_id);

        for (auto& [name, qt] : resident_layers) {
            if (qt.host_ptr) free(qt.host_ptr);
            if (have_gpu && qt.device_ptr) cudaFree(qt.device_ptr);
        }
        for (auto& layer_slots : layers_experts) {
            for (auto& slot : layer_slots) {
                if (slot.slab_buffer) free(slot.slab_buffer);
            }
        }
        for (float* p : Lc) if (p) free(p);
        for (float* p : Rc) if (p) free(p);
        for (float* p : Kc) if (p) free(p);
        for (float* p : Vc) if (p) free(p);

        if (have_gpu) {
            for (float* p : d_Lc) if (p) cudaFree(p);
            for (float* p : d_Rc) if (p) cudaFree(p);
            for (BlockQ4_KV* p : d_Lc_q4) if (p) cudaFree(p);
            for (BlockQ4_KV* p : d_Rc_q4) if (p) cudaFree(p);
            if (d_hidden) cudaFree(d_hidden);
        }
    }

    void initialize_resident_memory(const std::vector<std::string>& dense_tensor_names) {
        PERF_START(dense_load);
        size_t total_loaded = 0;
        // Dense tensors load from the separate dense GGUF when one is configured
        // (dense_model_path); routed experts still stream from model_file. Both
        // files are complete GGUFs of the same model at different precisions.
        GGUFFile& src = has_dense_model ? dense_model_file : model_file;
        for (const auto& name : dense_tensor_names) {
            if (src.tensors.find(name) == src.tensors.end()) continue;

            auto& t_meta = src.tensors[name];
            QuantizedTensor qt{t_meta, nullptr, nullptr};

            posix_memalign((void**)&qt.host_ptr, 4096, t_meta.size_bytes);
            int tensor_fd = src.fd_for_tensor(t_meta);
            ssize_t bytes_read = pread(tensor_fd, qt.host_ptr, t_meta.size_bytes, t_meta.offset);
            if (bytes_read <= 0 || (uint64_t)bytes_read != t_meta.size_bytes) {
                LOG_WARN("[WARN] Short read for %s (shard %d, offset %lu): got %zd / %lu\n",
                         name.c_str(), t_meta.shard_idx, (unsigned long)t_meta.offset,
                         bytes_read, (unsigned long)t_meta.size_bytes);
                // Mark as unusable — downstream code checks host_ptr != nullptr
                free(qt.host_ptr);
                qt.host_ptr = nullptr;
            } else {
                total_loaded += t_meta.size_bytes;
            }

            // Selective mlock: only pin small latency-sensitive tensors (norms, router)
            bool is_small_critical = (qt.host_ptr &&
                                      (name.find("norm") != std::string::npos ||
                                       name.find("gate_inp") != std::string::npos));
            if (is_small_critical) {
                mlock(qt.host_ptr, t_meta.size_bytes);
            }
            
            resident_layers[name] = qt;
        }
        PERF_END(dense_load, t_dense_load);
        perf().bytes_disk_read.fetch_add(total_loaded, std::memory_order_relaxed);
        LOG_INFO("[PERF] Dense resident load: %.3f s (%.2f GB, %.2f GB/s)\n",
                 perf().t_dense_load.load(),
                 total_loaded / 1e9,
                 perf().t_dense_load > 0 ? (total_loaded / 1e9) / perf().t_dense_load.load() : 0.0);
    }

    void load_expert_slab(int layer_idx, int expert_id, MoEExpertBlock& target_slot) {
        std::string base = "blk." + std::to_string(layer_idx) + ".";
        int n_exp = model_file.architecture_config.n_experts;

        // ---- Packed expert format: ffn_{gate,up,down}_exps.weight ----
        // Each packed tensor is [a, b, n_experts] with the expert index as the
        // LAST (contiguous) dimension, so expert E occupies bytes
        // [E*per_exp, (E+1)*per_exp). ne0 is the input dim, so the slice is
        // already the [O,I] row-major weight the matmul expects — no shuffle.
        // (GLM-5.2 UD ships this way: gate/up = Q2_K, down = Q3_K, packed.)
        auto gE = model_file.tensors.find(base + "ffn_gate_exps.weight");
        auto uE = model_file.tensors.find(base + "ffn_up_exps.weight");
        auto dE = model_file.tensors.find(base + "ffn_down_exps.weight");
        if (n_exp > 0 && gE != model_file.tensors.end() &&
            uE != model_file.tensors.end() && dE != model_file.tensors.end()) {
            const size_t gb = gE->second.size_bytes / n_exp;
            const size_t ub = uE->second.size_bytes / n_exp;
            const size_t db = dE->second.size_bytes / n_exp;
            const size_t total = gb + ub + db;

            PERF_START(alloc);
            if (target_slot.slab_capacity < total) {
                if (target_slot.slab_buffer) free(target_slot.slab_buffer);
                posix_memalign((void**)&target_slot.slab_buffer, 4096, total);
                target_slot.slab_capacity = total;
            }
            PERF_END(alloc, t_host_alloc);

            uint8_t* dst_gate = target_slot.slab_buffer;
            uint8_t* dst_up   = target_slot.slab_buffer + gb;
            uint8_t* dst_down = target_slot.slab_buffer + gb + ub;

            PERF_START(disk);
            auto read_slice = [this](const GGUFTensor& m, size_t rel_off, size_t bytes, uint8_t* dst, int l, int e) {
                int fd = model_file.fd_for_tensor(m);
                ssize_t r = pread(fd, dst, bytes, m.offset + rel_off);
                if (r < 0 || (size_t)r != bytes)
                    LOG_WARN("[WARN] packed expert short read L%d E%d: %zd/%zu\n", l, e, r, bytes);
            };
            read_slice(gE->second, (size_t)expert_id * gb, gb, dst_gate, layer_idx, expert_id);
            read_slice(uE->second, (size_t)expert_id * ub, ub, dst_up,   layer_idx, expert_id);
            read_slice(dE->second, (size_t)expert_id * db, db, dst_down, layer_idx, expert_id);
            PERF_END(disk, t_disk_read);
            perf().bytes_disk_read.fetch_add(total, std::memory_order_relaxed);

            if (drop_expert_pages) {
                posix_fadvise(model_file.fd_for_tensor(gE->second),
                              gE->second.offset, gE->second.size_bytes, POSIX_FADV_DONTNEED);
            }

            // Per-projection metas: offset is slab-relative so expert_ffn's
            // (up_meta.offset - gate_meta.offset) delta resolves correctly, and
            // each projection carries its OWN type (Q2_K vs Q3_K) for dispatch.
            target_slot.gate_proj.host_ptr = dst_gate;
            target_slot.gate_proj.meta = gE->second;
            target_slot.gate_proj.meta.offset = 0;
            target_slot.gate_proj.meta.size_bytes = gb;
            target_slot.up_proj.host_ptr = dst_up;
            target_slot.up_proj.meta = uE->second;
            target_slot.up_proj.meta.offset = gb;
            target_slot.up_proj.meta.size_bytes = ub;
            target_slot.down_proj.host_ptr = dst_down;
            target_slot.down_proj.meta = dE->second;
            target_slot.down_proj.meta.offset = gb + ub;
            target_slot.down_proj.meta.size_bytes = db;
            target_slot.expert_id = expert_id;
            return;
        }

        // ---- Per-expert format: ffn_{gate,up,down}.E.weight (legacy) ----
        std::string gate_name = base + "ffn_gate." + std::to_string(expert_id) + ".weight";
        std::string up_name   = base + "ffn_up."   + std::to_string(expert_id) + ".weight";
        std::string down_name = base + "ffn_down." + std::to_string(expert_id) + ".weight";

        auto gate_it = model_file.tensors.find(gate_name);
        if (gate_it == model_file.tensors.end()) {
            target_slot.expert_id = expert_id;  // neither packed nor per-expert found
            return;
        }
        auto& gate = gate_it->second;
        auto down_it = model_file.tensors.find(down_name);
        if (down_it == model_file.tensors.end()) {
            target_slot.expert_id = expert_id;
            return;
        }
        auto& down = down_it->second;

        size_t total_chunk_size = (down.offset + down.size_bytes) - gate.offset;
        PERF_START(alloc);
        if (target_slot.slab_capacity < total_chunk_size) {
            if (target_slot.slab_buffer) free(target_slot.slab_buffer);
            posix_memalign((void**)&target_slot.slab_buffer, 4096, total_chunk_size);
            target_slot.slab_capacity = total_chunk_size;
        }
        PERF_END(alloc, t_host_alloc);

        PERF_START(disk);
        int slab_fd = model_file.fd_for_tensor(gate);
        ssize_t read_bytes = pread(slab_fd, target_slot.slab_buffer, total_chunk_size, gate.offset);
        if (read_bytes < 0 || (size_t)read_bytes != total_chunk_size) {
            LOG_WARN("[WARN] Expert slab short read L%d E%d: %zd / %zu\n",
                     layer_idx, expert_id, read_bytes, total_chunk_size);
        }
        PERF_END(disk, t_disk_read);
        perf().bytes_disk_read.fetch_add(total_chunk_size, std::memory_order_relaxed);

        if (drop_expert_pages) {
            posix_fadvise(slab_fd, gate.offset, total_chunk_size, POSIX_FADV_DONTNEED);
        }

        target_slot.gate_proj.host_ptr = target_slot.slab_buffer;
        target_slot.gate_proj.meta = gate;
        if (model_file.tensors.count(up_name)) {
            auto& up = model_file.tensors[up_name];
            target_slot.up_proj.host_ptr = target_slot.slab_buffer + (up.offset - gate.offset);
            target_slot.up_proj.meta = up;
        }
        target_slot.down_proj.host_ptr = target_slot.slab_buffer + (down.offset - gate.offset);
        target_slot.down_proj.meta = down;
        target_slot.expert_id = expert_id;
    }

    // True per-expert slab size in bytes, accounting for each projection's own
    // quant type (e.g. GLM-5.2 UD: gate/up=Q2_K, down=Q3_K — NOT uniformly
    // sized). Mirrors load_expert_slab's sizing exactly so the VRAM cache slot
    // size and the prefetcher's staging buffer size always match what
    // load_expert_slab actually allocates. Returns 0 if layer_idx has no
    // expert tensors at all (e.g. a leading dense layer).
    size_t compute_expert_slab_bytes(int layer_idx) const {
        std::string base = "blk." + std::to_string(layer_idx) + ".";
        int n_exp = model_file.architecture_config.n_experts;

        auto gE = model_file.tensors.find(base + "ffn_gate_exps.weight");
        auto uE = model_file.tensors.find(base + "ffn_up_exps.weight");
        auto dE = model_file.tensors.find(base + "ffn_down_exps.weight");
        if (n_exp > 0 && gE != model_file.tensors.end() &&
            uE != model_file.tensors.end() && dE != model_file.tensors.end()) {
            return gE->second.size_bytes / n_exp + uE->second.size_bytes / n_exp
                 + dE->second.size_bytes / n_exp;
        }

        auto git = model_file.tensors.find(base + "ffn_gate.0.weight");
        auto dit = model_file.tensors.find(base + "ffn_down.0.weight");
        if (git != model_file.tensors.end() && dit != model_file.tensors.end())
            return (dit->second.offset + dit->second.size_bytes) - git->second.offset;

        return 0;
    }

    void issue_prefetch(int layer_idx, const std::vector<int>& predicted_expert_ids) {
        PERF_SCOPE(t_prefetch);
        for (int exp_id : predicted_expert_ids) {
            std::string gate_name = "blk." + std::to_string(layer_idx) + ".ffn_gate." + std::to_string(exp_id) + ".weight";
            auto it = model_file.tensors.find(gate_name);
            if (it != model_file.tensors.end()) {
                posix_fadvise(model_file.fd_for_tensor(it->second), it->second.offset, it->second.size_bytes * 3, POSIX_FADV_WILLNEED);
            }
        }
    }
};