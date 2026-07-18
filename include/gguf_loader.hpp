#pragma once
#include "logger.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <unordered_map>
#include <cstdint>
#include <fcntl.h>
#include <unistd.h>

// GGML / GGUF tensor type IDs. These values MUST match the raw `type` field
// stored in the GGUF tensor table (see ggml.h). calculate_size() below switches
// on the integer value, so these names exist only for readable dispatch — but
// they must still carry the correct ggml numeric IDs, or `case GGUFType::X:`
// switches will silently match the wrong tensors.
enum class GGUFType : uint32_t {
    F32     = 0,
    F16     = 1,
    Q4_0    = 2,
    Q4_1    = 3,
    // 4,5 were removed legacy Q4_2/Q4_3
    Q5_0    = 6,
    Q5_1    = 7,
    Q8_0    = 8,
    Q8_1    = 9,
    Q2_K    = 10,
    Q3_K    = 11,
    Q4_K    = 12,
    Q5_K    = 13,
    Q6_K    = 14,
    Q8_K    = 15,
    IQ2_XXS = 16,
    IQ2_XS  = 17,
    IQ3_XXS = 18,
    IQ1_S   = 19,
    IQ4_NL  = 20,
    IQ3_S   = 21,
    IQ2_S   = 22,
    IQ4_XS  = 23,
    IQ1_M   = 24,
    BF16    = 25,
    // 26,27,28 = removed Q4_0_4_4 / Q4_0_4_8 / Q4_1_4_4 (ND-packed, not in the wild for our targets)
    TQ1_0   = 29,
    TQ2_0   = 30,
    IQ2_M   = 31,   // FIX: was incorrectly 16 here. 16 is IQ2_XXS; real IQ2_M is 31.
};

struct GGUFTensor {
    std::string name;
    GGUFType type;
    std::vector<uint64_t> shape;
    uint64_t offset; 
    uint64_t size_bytes;
    int shard_idx = 0;  // which shard this tensor lives in (for split GGUF)
}; 

struct ArchitectureConfig {
    int hidden = 0;
    int n_layers = 0;
    int n_heads = 0;
    int n_experts = 0;
    int topk = 0;
    int moe_inter = 0;
    int vocab = 0;
    int kv_lora = 512;
    int qk_nope = 128;
    int qk_rope = 64;
    int v_head = 128;
    int first_dense = 0;  // first N layers are dense (no MoE)
    // Dense Q/K/V attention (Hunyuan / Llama / GLM-4); unused for MLA arches.
    int n_head_kv = 0;    // GQA kv-head count (defaults to n_heads)
    int head_dim = 0;     // per-head dim (= attention.key_length)
    bool is_mla = false;  // MLA (absorbed) vs dense Q/K/V attention
    float eps = 1e-5f;

    // Dense (non-MoE) FFN intermediate width — used for leading dense layers
    // (l < first_dense) in an otherwise-MoE model, and for every layer in a
    // fully-dense model. Distinct from moe_inter (expert_feed_forward_length).
    int dense_inter = 0;

    // ---- MoE router (sigmoid/softmax gating, selection bias, weight norm
    // and scale) — see llama.cpp's build_moe_ffn for the reference algorithm.
    int n_expert_shared = 0;             // 0 = no shared expert
    float expert_weights_scale = 0.0f;   // 0 = no scale applied
    bool expert_weights_norm = true;     // normalize top-k weights to sum to 1
    bool use_sigmoid_gate = false;       // sigmoid vs softmax router activation
    // HunyuanMoE (original, pre-Hy-V3) ropes Q/K before applying QK-norm;
    // every other supported arch (Hy-V3, GLM-5.2) norms first, then ropes.
    bool rope_before_norm = false;

    // ---- YaRN long-context RoPE scaling (see ggml's rope_yarn/
    // ggml_rope_yarn_corr_dims). ext_factor==0 reduces to plain RoPE.
    bool rope_scaling_yarn = false;
    float rope_freq_scale = 1.0f;
    float rope_ext_factor = 0.0f;
    float rope_attn_factor = 1.0f;
    float rope_yarn_log_mul = 0.0f;
    int rope_n_ctx_orig = 0;
};

class GGUFFile {
public:
    std::unordered_map<std::string, GGUFTensor> tensors;
    ArchitectureConfig architecture_config; 
    uint32_t version;
    int file_fd = -1;  // primary shard fd (backward compat)
    std::string resolved_file_path;

    // Split GGUF support: one fd per shard
    std::vector<int> shard_fds;
    std::vector<std::string> shard_paths;

    // KV metadata extracted during header parsing
    std::unordered_map<std::string, uint64_t> kv_uint;
    std::unordered_map<std::string, int64_t> kv_int;
    std::unordered_map<std::string, float> kv_float;
    std::unordered_map<std::string, std::string> kv_string;

    int fd_for_tensor(const GGUFTensor& t) const {
        if (t.shard_idx < (int)shard_fds.size())
            return shard_fds[t.shard_idx];
        return file_fd;
    }

    // Extract integer config value from KV metadata, trying multiple key patterns
    int get_config_int(const std::string& suffix, int fallback = 0) const {
        // Detect architecture prefix from general.architecture KV
        std::string arch_prefix;
        auto arch_it = kv_string.find("general.architecture");
        if (arch_it != kv_string.end()) arch_prefix = arch_it->second + ".";

        // Try: detected arch prefix, then common fallbacks
        std::vector<std::string> prefixes;
        if (!arch_prefix.empty()) prefixes.push_back(arch_prefix);
        for (const auto& p : {"glm-dsa.", "glm.", "glm4.", "llama.", "deepseek2.", ""})
            prefixes.push_back(p);

        for (const auto& prefix : prefixes) {
            std::string key = prefix + suffix;
            auto it = kv_uint.find(key);
            if (it != kv_uint.end()) return (int)it->second;
            auto it2 = kv_int.find(key);
            if (it2 != kv_int.end()) return (int)it2->second;
        }
        return fallback;
    }

    float get_config_float(const std::string& suffix, float fallback = 0.0f) const {
        std::string arch_prefix;
        auto arch_it = kv_string.find("general.architecture");
        if (arch_it != kv_string.end()) arch_prefix = arch_it->second + ".";

        std::vector<std::string> prefixes;
        if (!arch_prefix.empty()) prefixes.push_back(arch_prefix);
        for (const auto& p : {"glm-dsa.", "glm.", "glm4.", "llama.", "deepseek2.", ""})
            prefixes.push_back(p);

        for (const auto& prefix : prefixes) {
            std::string key = prefix + suffix;
            auto it = kv_float.find(key);
            if (it != kv_float.end()) return it->second;
        }
        return fallback;
    }

    std::string get_config_string(const std::string& suffix, const std::string& fallback = "") const {
        std::string arch_prefix;
        auto arch_it = kv_string.find("general.architecture");
        if (arch_it != kv_string.end()) arch_prefix = arch_it->second + ".";

        std::vector<std::string> prefixes;
        if (!arch_prefix.empty()) prefixes.push_back(arch_prefix);
        for (const auto& p : {"glm-dsa.", "glm.", "glm4.", "llama.", "deepseek2.", ""})
            prefixes.push_back(p);

        for (const auto& prefix : prefixes) {
            std::string key = prefix + suffix;
            auto it = kv_string.find(key);
            if (it != kv_string.end()) return it->second;
        }
        return fallback;
    }

    bool load_header(const std::string& path) {
        std::vector<std::string> gguf_files;

        if (std::filesystem::is_directory(path)) {
            for (const auto& entry : std::filesystem::directory_iterator(path)) {
                if (entry.path().extension() == ".gguf")
                    gguf_files.push_back(entry.path().string());
            }
            if (gguf_files.empty()) {
                LOG_STREAM_ERROR() << "[GGUF] No .gguf files found in directory: " << path << "\n";
                return false;
            }
            std::sort(gguf_files.begin(), gguf_files.end());
        } else {
            gguf_files.push_back(path);
        }

        LOG_STREAM_INFO() << "[GGUF] Loading " << gguf_files.size() << " shard(s)...\n";

        for (size_t shard = 0; shard < gguf_files.size(); ++shard) {
            if (!load_shard(gguf_files[shard], (int)shard)) {
                LOG_STREAM_ERROR() << "[GGUF] Failed to load shard: " << gguf_files[shard] << "\n";
                return false;
            }
        }

        resolved_file_path = gguf_files[0];
        file_fd = shard_fds.empty() ? -1 : shard_fds[0];
        shard_paths = gguf_files;

        LOG_STREAM_INFO() << "[GGUF] Total tensors loaded: " << tensors.size() << "\n";
        return file_fd >= 0;
    }

    ~GGUFFile() {
        for (int fd : shard_fds) {
            if (fd >= 0) close(fd);
        }
    }

private:
    bool load_shard(const std::string& shard_path, int shard_idx) {
        std::ifstream f(shard_path, std::ios::binary);
        if (!f.is_open()) return false;

        char magic[4];
        f.read(magic, 4);
        if (std::string(magic, 4) != "GGUF") return false;

        uint32_t shard_version;
        f.read(reinterpret_cast<char*>(&shard_version), 4);
        if (shard_idx == 0) version = shard_version;

        uint64_t tensor_count = 0, kv_count = 0;
        f.read(reinterpret_cast<char*>(&tensor_count), 8);
        f.read(reinterpret_cast<char*>(&kv_count), 8);

        LOG_STREAM_DEBUG() << "[GGUF]   shard " << shard_idx << ": " << tensor_count
                  << " tensors, " << kv_count << " kv pairs"
                  << " (" << std::filesystem::path(shard_path).filename().string() << ")\n";

        // Read KV metadata — store values for architecture config extraction
        for (uint64_t i = 0; i < kv_count; ++i) {
            std::string key = read_string(f);
            uint32_t value_type;
            f.read(reinterpret_cast<char*>(&value_type), 4);
            read_kv_value(f, key, value_type);
        }

        // Read tensor info
        for (uint64_t i = 0; i < tensor_count; ++i) {
            GGUFTensor t;
            t.name = read_string(f);
            uint32_t n_dimensions;
            f.read(reinterpret_cast<char*>(&n_dimensions), 4);
            t.shape.resize(n_dimensions);
            f.read(reinterpret_cast<char*>(t.shape.data()), n_dimensions * 8);
            f.read(reinterpret_cast<char*>(&t.type), 4);
            f.read(reinterpret_cast<char*>(&t.offset), 8);

            uint64_t numel = 1;
            for (auto dim : t.shape) numel *= dim;
            t.size_bytes = calculate_size(numel, t.type);
            t.shard_idx = shard_idx;
            tensors[t.name] = t;
        }

        uint64_t current_pos = f.tellg();
        f.close();

        int fd = open(shard_path.c_str(), O_RDONLY);
        if (fd < 0) return false;

        // Compute data_start for this shard and adjust tensor offsets
        uint64_t alignment = 32;
        uint64_t data_start = (current_pos + alignment - 1) & ~(alignment - 1);
        for (auto& [name, t] : tensors) {
            if (t.shard_idx == shard_idx) {
                t.offset += data_start;
            }
        }

        // Grow shard_fds vector if needed
        if ((int)shard_fds.size() <= shard_idx)
            shard_fds.resize(shard_idx + 1, -1);
        shard_fds[shard_idx] = fd;

        return true;
    }

    std::string read_string(std::ifstream& f) {
        uint64_t len; f.read(reinterpret_cast<char*>(&len), 8);
        std::string s(len, '\0'); f.read(&s[0], len);
        return s;
    }

    // GGUF KV types: 0=UINT8, 1=INT8, 2=UINT16, 3=INT16, 4=UINT32, 5=INT32,
    // 6=FLOAT32, 7=BOOL, 8=STRING, 9=ARRAY, 10=UINT64, 11=INT64, 12=FLOAT64
    void read_kv_value(std::ifstream& f, const std::string& key, uint32_t type) {
        switch (type) {
            case 0: { uint8_t v; f.read(reinterpret_cast<char*>(&v), 1); kv_uint[key] = v; break; }
            case 1: { int8_t v; f.read(reinterpret_cast<char*>(&v), 1); kv_int[key] = v; break; }
            case 2: { uint16_t v; f.read(reinterpret_cast<char*>(&v), 2); kv_uint[key] = v; break; }
            case 3: { int16_t v; f.read(reinterpret_cast<char*>(&v), 2); kv_int[key] = v; break; }
            case 4: { uint32_t v; f.read(reinterpret_cast<char*>(&v), 4); kv_uint[key] = v; break; }
            case 5: { int32_t v; f.read(reinterpret_cast<char*>(&v), 4); kv_int[key] = v; break; }
            case 6: { float v; f.read(reinterpret_cast<char*>(&v), 4); kv_float[key] = v; break; }
            case 7: { uint8_t v; f.read(reinterpret_cast<char*>(&v), 1); kv_uint[key] = v; break; }
            case 8: { kv_string[key] = read_string(f); break; }
            case 9: {
                uint32_t sub_type; f.read(reinterpret_cast<char*>(&sub_type), 4);
                uint64_t arr_len; f.read(reinterpret_cast<char*>(&arr_len), 8);
                // Arrays: skip elements (too complex to store generically)
                for (uint64_t j = 0; j < arr_len; ++j) skip_kv_value(f, sub_type);
                break;
            }
            case 10: { uint64_t v; f.read(reinterpret_cast<char*>(&v), 8); kv_uint[key] = v; break; }
            case 11: { int64_t v; f.read(reinterpret_cast<char*>(&v), 8); kv_int[key] = v; break; }
            case 12: { double v; f.read(reinterpret_cast<char*>(&v), 8); kv_float[key] = (float)v; break; }
            default: break;
        }
    }

    void skip_kv_value(std::ifstream& f, uint32_t type) {
        switch (type) {
            case 0: case 1: case 7: f.seekg(1, std::ios::cur); break;
            case 2: case 3: f.seekg(2, std::ios::cur); break;
            case 4: case 5: case 6: f.seekg(4, std::ios::cur); break;
            case 10: case 11: case 12: f.seekg(8, std::ios::cur); break;
            case 8: { uint64_t len; f.read(reinterpret_cast<char*>(&len), 8); f.seekg(len, std::ios::cur); break; }
            default: break;
        }
    }

    uint64_t calculate_size(uint64_t numel, GGUFType type) {
        uint32_t t = (uint32_t)type;
        // Block size and type_size from ggml (block_size=32 for Q4/Q5/Q8_0, 256 for K-quants)
        if (t == 0) return numel * 4;                         // F32
        if (t == 1) return numel * 2;                         // F16
        if (t == 2) return (numel / 32) * 18;                 // Q4_0: 2+16
        if (t == 3) return (numel / 32) * 20;                 // Q4_1: 2+2+16
        if (t == 6) return (numel / 32) * 22;                 // Q5_0: 2+4+16
        if (t == 7) return (numel / 32) * 24;                 // Q5_1: 2+2+4+16
        if (t == 8) return (numel / 32) * 34;                 // Q8_0: 2+32
        if (t == 9) return (numel / 32) * 36;                 // Q8_1: 4+32 (with sum)
        if (t == 10) return (numel / 256) * 84;               // Q2_K
        if (t == 11) return (numel / 256) * 110;              // Q3_K
        if (t == 12) return (numel / 256) * 144;              // Q4_K
        if (t == 13) return (numel / 256) * 176;              // Q5_K
        if (t == 14) return (numel / 256) * 210;              // Q6_K
        if (t == 15) return (numel / 256) * 292;              // Q8_K
        if (t == 16) return (numel / 256) * 66;               // IQ2_XXS
        if (t == 17) return (numel / 256) * 74;               // IQ2_XS
        if (t == 18) return (numel / 256) * 98;               // IQ3_XXS
        if (t == 19) return (numel / 256) * 50;               // IQ1_S
        if (t == 20) return (numel / 32) * 18;                // IQ4_NL (block size 32, not 256 — like Q4_0;
                                                                // was (numel/256)*132, ~20% too small)
        if (t == 21) return (numel / 256) * 110;              // IQ3_S
        if (t == 22) return (numel / 256) * 82;               // IQ2_S
        if (t == 23) return (numel / 256) * 136;              // IQ4_XS
        if (t == 24) return (numel / 256) * 56;               // IQ1_M
        if (t == 25) return numel * 2;                        // BF16
        if (t == 29) return (numel / 256) * 54;               // TQ1_0
        if (t == 30) return (numel / 256) * 66;               // TQ2_0
        if (t == 31) return (numel / 256) * 84;               // IQ2_M
        return numel;  // fallback: 1 byte per element
    }
};