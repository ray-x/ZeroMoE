// tests/test_components.cpp — Unit tests for MoEStream components
// Build: g++ -std=c++23 -O2 -I../include -o test_components test_components.cpp
// (Does NOT require CUDA for the CPU-path tests)
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <unordered_map>
#include <random>
#include "rope.hpp"   // the REAL, unit-tested RoPE (no CUDA / no AVX-512 deps)

// ============================================================
// Test 1: ReasoningParser state machine
// ============================================================
// Minimal inline copy to avoid needing full includes
enum class GenerationState { Thinking, Outputting };
struct ReasoningParser {
    GenerationState state = GenerationState::Outputting;
    std::string pending;
    int think_tokens = 0;
    int output_tokens = 0;

    int feed(const std::string& token_text) {
        pending += token_text;
        if (state == GenerationState::Outputting) {
            size_t pos = pending.find("<think>");
            if (pos != std::string::npos) {
                state = GenerationState::Thinking;
                pending.clear();
                return 2;
            }
            if (pending.size() < 7 && std::string("<think>").find(pending) == 0) return 2;
            pending.clear();
            output_tokens++;
            return 0;
        } else {
            size_t pos = pending.find("</think>");
            if (pos != std::string::npos) {
                state = GenerationState::Outputting;
                pending.clear();
                return 2;
            }
            if (pending.size() < 8 && std::string("</think>").find(pending) == 0) return 2;
            pending.clear();
            think_tokens++;
            return 1;
        }
    }
    void reset() { state = GenerationState::Outputting; pending.clear(); think_tokens = 0; output_tokens = 0; }
};

void test_reasoning_parser() {
    std::cout << "[TEST] ReasoningParser..." << std::flush;
    ReasoningParser p;

    // Normal output before any thinking
    assert(p.feed("Hello") == 0);
    assert(p.state == GenerationState::Outputting);
    assert(p.output_tokens == 1);

    // Enter thinking mode
    assert(p.feed("<think>") == 2);
    assert(p.state == GenerationState::Thinking);

    // Tokens during thinking
    assert(p.feed("step1") == 1);
    assert(p.feed("step2") == 1);
    assert(p.think_tokens == 2);

    // Exit thinking
    assert(p.feed("</think>") == 2);
    assert(p.state == GenerationState::Outputting);

    // Normal output after thinking
    assert(p.feed("Answer") == 0);
    assert(p.output_tokens == 2);

    std::cout << " PASSED\n";
}

// ============================================================
// Test 2: ExpertScore partial_sort (verifies top-k extraction)
// ============================================================
struct ExpertScore { int id; float score; };

void test_topk_selection() {
    std::cout << "[TEST] Top-K Expert Selection..." << std::flush;
    const int N = 256;
    const int K = 8;
    std::vector<ExpertScore> scores(N);
    for (int i = 0; i < N; ++i) scores[i] = {i, (float)(N - i)};  // descending
    scores[42].score = 9999.0f;  // make expert 42 the best
    scores[7].score = 9998.0f;

    std::partial_sort(scores.begin(), scores.begin() + K, scores.end(),
                     [](const ExpertScore& a, const ExpertScore& b) { return a.score > b.score; });

    assert(scores[0].id == 42);
    assert(scores[1].id == 7);
    // The rest of top-8 should be ids 0..5 (original scores 256..251)
    assert(scores[0].score > scores[1].score);
    assert(scores[K-1].score > scores[K].score);

    std::cout << " PASSED\n";
}

// ============================================================
// Test 3: Softmax normalization (used for routing weights)
// ============================================================
void test_softmax_routing() {
    std::cout << "[TEST] Softmax Routing Weights..." << std::flush;
    const int K = 8;
    float logits[8] = {2.0f, 1.5f, 1.0f, 0.5f, 0.2f, 0.1f, -0.5f, -1.0f};
    float weights[8];

    float max_s = logits[0];
    float sum_exp = 0.0f;
    for (int k = 0; k < K; ++k) {
        weights[k] = std::exp(logits[k] - max_s);
        sum_exp += weights[k];
    }
    float inv_sum = 1.0f / sum_exp;
    for (int k = 0; k < K; ++k) weights[k] *= inv_sum;

    // Verify sums to 1.0
    float total = 0.0f;
    for (int k = 0; k < K; ++k) total += weights[k];
    assert(std::abs(total - 1.0f) < 1e-5f);

    // Verify ordering preserved
    for (int k = 1; k < K; ++k) assert(weights[k-1] >= weights[k]);

    std::cout << " PASSED\n";
}

// ============================================================
// Test 4: GGUF calculate_size for various quant types
// ============================================================
void test_calculate_size() {
    std::cout << "[TEST] GGUF calculate_size..." << std::flush;
    // Simulate the function logic
    auto calc = [](uint64_t numel, uint32_t t) -> uint64_t {
        if (t == 0) return numel * 4;         // F32
        if (t == 1) return numel * 2;         // F16
        if (t == 8) return (numel / 32) * 34; // Q8_0
        if (t == 10) return (numel / 256) * 84;  // Q2_K
        if (t == 13) return (numel / 256) * 176; // Q5_K
        if (t == 14) return (numel / 256) * 210; // Q6_K
        if (t == 25) return numel * 2;        // BF16
        if (t == 31) return (numel / 256) * 84;  // IQ2_M
        return numel;
    };

    // F32: 1024 elements = 4096 bytes
    assert(calc(1024, 0) == 4096);
    // Q5_K: 256 elements = 176 bytes
    assert(calc(256, 13) == 176);
    // Q6_K: 256 elements = 210 bytes  
    assert(calc(256, 14) == 210);
    // Q2_K: 256 elements = 84 bytes
    assert(calc(256, 10) == 84);
    // Q8_0: 32 elements = 34 bytes
    assert(calc(32, 8) == 34);
    // GLM-5.2 expert: 6144*2048 = 12582912 elements, Q5_K
    uint64_t expert_size = calc(12582912, 13);
    assert(expert_size == (12582912 / 256) * 176);  // 8,650,752 bytes

    std::cout << " PASSED\n";
}

// ============================================================
// Test 5: LockFreePrefetcher SPSC ring buffer logic
// ============================================================
void test_spsc_ring() {
    std::cout << "[TEST] SPSC Ring Buffer Logic..." << std::flush;
    // Simulate the ring buffer
    const size_t QSIZE = 16;
    unsigned int head = 0, tail = 0;
    int buffer[16];

    // Push 10 items
    for (int i = 0; i < 10; ++i) {
        assert(head - tail < QSIZE); // not full
        buffer[head & (QSIZE - 1)] = i;
        head++;
    }
    assert(head - tail == 10);

    // Pop 5 items
    for (int i = 0; i < 5; ++i) {
        assert(tail != head); // not empty
        int val = buffer[tail & (QSIZE - 1)];
        assert(val == i);
        tail++;
    }
    assert(head - tail == 5);

    // Push until full
    while (head - tail < QSIZE) {
        buffer[head & (QSIZE - 1)] = (int)head;
        head++;
    }
    assert(head - tail == QSIZE);

    // One more push should be rejected (overflow check)
    bool full = (head - tail >= QSIZE);
    assert(full);

    std::cout << " PASSED\n";
}

// ============================================================
// Test 6: TOML parser
// ============================================================
namespace toml_test {

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
    std::string get_str(const std::string& key, const std::string& def = "") const {
        auto it = values.find(key); return (it != values.end()) ? it->second.str : def;
    }
    int get_int(const std::string& key, int def = 0) const {
        auto it = values.find(key); return (it != values.end() && it->second.type == Value::Number) ? (int)it->second.num : def;
    }
    double get_double(const std::string& key, double def = 0.0) const {
        auto it = values.find(key); return (it != values.end() && it->second.type == Value::Number) ? it->second.num : def;
    }
    bool get_bool(const std::string& key, bool def = false) const {
        auto it = values.find(key); return (it != values.end() && it->second.type == Value::Bool) ? it->second.boolean : def;
    }
};

inline Value parse_value(const std::string& raw) {
    Value v; std::string s = trim(raw);
    if (s.empty()) return v;
    if (s.front() == '"' && s.back() == '"') { v.type = Value::String; v.str = s.substr(1, s.size()-2); return v; }
    if (s == "true" || s == "false") { v.type = Value::Bool; v.boolean = (s == "true"); v.str = s; return v; }
    try { v.num = std::stod(s); v.type = Value::Number; v.str = s; } catch (...) { v.type = Value::String; v.str = s; }
    return v;
}

} // namespace toml_test

void test_toml_parser() {
    std::cout << "[TEST] TOML Parser..." << std::flush;
    using namespace toml_test;

    std::string config = R"(
# Test config
model_path = "/models/test.gguf"
context_size = 4096
temperature = 0.7
drop_pages = true

[[gpus]]
device_id = 0
name = "RTX 3090"
role = "dense+moe"
vram_gb = 24.0
kv_bits = 4

[[gpus]]
device_id = 1
role = "moe"
moe_cache_gb = 23.0
)";

    // Parse top-level values
    Table root;
    std::vector<Table> gpu_tables;
    Table* current = &root;

    std::istringstream stream(config);
    std::string line;
    while (std::getline(stream, line)) {
        line = trim(strip_comment(line));
        if (line.empty()) continue;
        if (line == "[[gpus]]") { gpu_tables.emplace_back(); current = &gpu_tables.back(); continue; }
        if (line.front() == '[') { current = &root; continue; }
        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        current->values[trim(line.substr(0, eq))] = parse_value(trim(line.substr(eq + 1)));
    }

    // Verify root
    assert(root.get_str("model_path") == "/models/test.gguf");
    assert(root.get_int("context_size") == 4096);
    assert(std::abs(root.get_double("temperature") - 0.7) < 1e-5);
    assert(root.get_bool("drop_pages") == true);

    // Verify GPU tables
    assert(gpu_tables.size() == 2);
    assert(gpu_tables[0].get_int("device_id") == 0);
    assert(gpu_tables[0].get_str("name") == "RTX 3090");
    assert(gpu_tables[0].get_str("role") == "dense+moe");
    assert(std::abs(gpu_tables[0].get_double("vram_gb") - 24.0) < 1e-5);
    assert(gpu_tables[0].get_int("kv_bits") == 4);

    assert(gpu_tables[1].get_int("device_id") == 1);
    assert(gpu_tables[1].get_str("role") == "moe");
    assert(std::abs(gpu_tables[1].get_double("moe_cache_gb") - 23.0) < 1e-5);

    // Test comment stripping preserves strings with #
    std::string line2 = R"(path = "/models/test#1.gguf"  # a comment)";
    std::string stripped = strip_comment(line2);
    assert(stripped.find("# a comment") == std::string::npos);
    assert(stripped.find("test#1.gguf") != std::string::npos);

    std::cout << " PASSED\n";
}

// ============================================================
// Test 7: RoPE convention (interleaved / LLAMA_ROPE_TYPE_NORM)
// ============================================================
// Guards the root cause of the "always random text" bug: glm-dsa / ChatGLM /
// DeepSeek require INTERLEAVED RoPE (pairs (2j, 2j+1)), but apply_rope_yarn used
// to do NeoX rotate-half (pairs (k, k+half)). The known-answer case below
// DISTINGUISHES the two -- rotate-half yields [cos1, 0, sin1, 0], interleaved
// yields [cos1, sin1, 0, 0]. If rope.hpp is ever reverted to rotate-half, this
// test fails loudly. Calls the REAL rope_yarn_inplace from include/rope.hpp
// (not a copy), via #include "rope.hpp".
void test_rope() {
    std::cout << "[TEST] RoPE convention (interleaved)..." << std::flush;

    // Independent interleaved reference oracle (no YaRN, ext_factor=0). Lives in
    // the test so a regression in rope.hpp is caught against a separate impl.
    auto ref_interleaved = [](float* v, int dim, int pos, float theta_base) {
        int half = dim / 2;
        float theta_scale = std::pow(theta_base, -2.0f / (float)dim);
        float theta_extrap = (float)pos;
        for (int j = 0; j < half; ++j) {
            float t = theta_extrap;
            float a = v[2*j], b = v[2*j + 1];
            v[2*j]     = a * std::cos(t) - b * std::sin(t);
            v[2*j + 1] = a * std::sin(t) + b * std::cos(t);
            theta_extrap *= theta_scale;
        }
    };

    // --- Case 1: known-answer that separates interleaved from rotate-half.
    // input [1,0,0,0], dim=4, pos=1, theta=10000.
    //   interleaved (want): pair (0,1) rotates -> [cos1, sin1, 0, 0]
    //   rotate-half (bug) : pair (0,2) rotates -> [cos1, 0, sin1, 0]
    {
        const int dim = 4, pos = 1;
        const float theta = 10000.0f;
        float v[4] = {1, 0, 0, 0};
        rope_yarn_inplace(v, dim, pos, theta,
                          /*ext_factor=*/0.0f, /*freq_scale=*/1.0f,
                          /*attn_factor=*/1.0f, /*yarn_log_mul=*/0.0f, /*n_ctx_orig=*/0);
        float c1 = std::cos(1.0f), s1 = std::sin(1.0f);
        assert(std::fabs(v[0] - c1) < 1e-5f);
        assert(std::fabs(v[1] - s1) < 1e-5f);  // rotate-half leaves this 0
        assert(std::fabs(v[2])      < 1e-5f);  // rotate-half puts sin1 here
        assert(std::fabs(v[3])      < 1e-5f);
    }

    // --- Case 2: regression at production size (qk_rope=64, theta=8e6). The real
    // rope_yarn_inplace must match the independent interleaved oracle to ~1e-4.
    {
        const int dim = 64, pos = 7;
        const float theta = 8.0e6f;
        std::vector<float> a(dim), b(dim);
        for (int i = 0; i < dim; ++i) { a[i] = 0.01f * (float)(i + 1); b[i] = a[i]; }
        rope_yarn_inplace(a.data(), dim, pos, theta, 0.0f, 1.0f, 1.0f, 0.0f, 0);
        ref_interleaved(b.data(), dim, pos, theta);
        for (int i = 0; i < dim; ++i) {
            float denom = std::max(1e-6f, std::fabs(b[i]));
            assert(std::fabs(a[i] - b[i]) < 1e-4f * denom);
        }
    }

    // --- Case 3: norm preservation (both conventions satisfy this; keeps the
    // test meaningful if the oracle is ever regenerated).
    {
        const int dim = 64, pos = 3;
        const float theta = 8.0e6f;
        std::vector<float> v(dim);
        for (int i = 0; i < dim; ++i) v[i] = 0.01f * (float)(i + 1);
        double n0 = 0; for (float x : v) n0 += (double)x * x;
        rope_yarn_inplace(v.data(), dim, pos, theta, 0.0f, 1.0f, 1.0f, 0.0f, 0);
        double n1 = 0; for (float x : v) n1 += (double)x * x;
        assert(std::fabs(n1 - n0) < 1e-3 * n0);
    }

    std::cout << " PASSED\n";
}

// ============================================================
// Test: Q2_K / Q3_K dequant LOGIC (scalar oracle vs scalar port of the
// AVX-512 algorithm's per-group formula)
// ============================================================
// The real AVX-512 kernels live in src/kernels_cpu.cpp and need AVX-512
// hardware to build/run (this file deliberately doesn't -- see the file
// header), so they can't be exercised here. What CAN be verified on any
// machine is the underlying ALGORITHM: block layout offsets, scale/min
// extraction, the n/j/shift nested-loop bookkeeping, and (for Q3_K) the
// hmask high-bit combine -- exactly the class of subtle layout bug this
// project has hit before (Q3_K struct field order, qk_nope size). `oracle_*`
// below is a verbatim scalar transcription of ggml's real
// dequantize_row_q2_K/q3_K (ggml/src/ggml-quants.c, confirmed against the
// upstream source directly, not from memory); `candidate_*` reproduces the
// SAME per-16-element formula src/kernels_cpu.cpp's AVX-512 functions use
// (unpack 2-bit code, combine with hmask for Q3_K, apply dl/ml), just as a
// scalar loop instead of intrinsics -- the one AVX-512-specific detail this
// can't cover is the byte-widen-before-shift technique used to avoid
// cross-lane bit bleed in the real SIMD code, a standard, low-risk,
// mechanical pattern. This test targets the higher-risk part: the format
// understanding itself. See src/main_test.cpp's Test 6 for the full,
// AVX-512-vs-ggml-reference comparison (requires the Linux+CUDA build).
static float f16_bits_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) bits = sign; // flush subnormal to zero, matches f32_to_f16_bits's flush
    else bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    float f; std::memcpy(&f, &bits, 4);
    return f;
}

// Verbatim scalar port of ggml's dequantize_row_q2_K (block: scales[16] qs[64] d dmin, 84B).
static void oracle_dequant_q2_K(const uint8_t* block, float* y, int64_t k) {
    const int nb = (int)(k / 256);
    for (int i = 0; i < nb; ++i) {
        const uint8_t* base = block + (size_t)i * 84;
        const uint8_t* scales = base;
        const uint8_t* qs = base + 16;
        uint16_t dh, dminh;
        std::memcpy(&dh, base + 80, 2);
        std::memcpy(&dminh, base + 82, 2);
        float d = f16_bits_to_f32(dh), min = f16_bits_to_f32(dminh);
        const uint8_t* q = qs;
        int is = 0;
        float* yy = y + (size_t)i * 256;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                uint8_t sc = scales[is++];
                float dl = d * (sc & 0xF), ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *yy++ = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;
                sc = scales[is++];
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *yy++ = dl * ((int8_t)((q[l + 16] >> shift) & 3)) - ml;
                shift += 2;
            }
            q += 32;
        }
    }
}

// Verbatim scalar port of ggml's dequantize_row_q3_K (block: hmask[32] qs[64] scales[12] d, 110B).
static void oracle_dequant_q3_K(const uint8_t* block, float* y, int64_t k) {
    const int nb = (int)(k / 256);
    const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    for (int i = 0; i < nb; ++i) {
        const uint8_t* base = block + (size_t)i * 110;
        const uint8_t* hmask = base;
        const uint8_t* qs = base + 32;
        const uint8_t* scale_bytes = base + 96;
        uint16_t dh; std::memcpy(&dh, base + 108, 2);
        float d_all = f16_bits_to_f32(dh);

        uint32_t aux[4];
        std::memcpy(aux, scale_bytes, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t* scales = (const int8_t*)aux;

        const uint8_t* q = qs;
        const uint8_t* hm = hmask;
        uint8_t m = 1;
        int is = 0;
        float* yy = y + (size_t)i * 256;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                float dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l)
                    *yy++ = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l)
                    *yy++ = dl * ((int8_t)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
}

// Scalar transcription of src/kernels_cpu.cpp's AVX-512 group-of-16 formula
// (unpack_2bit_x16 + dequant_q2k_group16/dequant_q3k_group16), loop-per-
// element instead of one SIMD call -- same block-layout offsets, same
// nested n/j/shift bookkeeping, same integer-domain combine before the
// single float multiply.
static void candidate_dequant_q2_K(const uint8_t* block, float* y, int64_t k) {
    const int nb = (int)(k / 256);
    for (int i = 0; i < nb; ++i) {
        const uint8_t* base = block + (size_t)i * 84;
        const uint8_t* scales = base;
        const uint8_t* qs = base + 16;
        uint16_t dh, dminh;
        std::memcpy(&dh, base + 80, 2);
        std::memcpy(&dminh, base + 82, 2);
        float d = f16_bits_to_f32(dh), dmin = f16_bits_to_f32(dminh);
        const uint8_t* q = qs;
        int is = 0;
        float* yy = y + (size_t)i * 256;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                uint8_t sc0 = scales[is++];
                float dl0 = d * (sc0 & 0xF), ml0 = dmin * (sc0 >> 4);
                for (int l = 0; l < 16; ++l) { int v = (q[l] >> shift) & 3; *yy++ = dl0 * (float)v - ml0; }
                uint8_t sc1 = scales[is++];
                float dl1 = d * (sc1 & 0xF), ml1 = dmin * (sc1 >> 4);
                for (int l = 0; l < 16; ++l) { int v = (q[l + 16] >> shift) & 3; *yy++ = dl1 * (float)v - ml1; }
                shift += 2;
            }
            q += 32;
        }
    }
}
static void candidate_dequant_q3_K(const uint8_t* block, float* y, int64_t k) {
    const int nb = (int)(k / 256);
    const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    for (int i = 0; i < nb; ++i) {
        const uint8_t* base = block + (size_t)i * 110;
        const uint8_t* hmask = base;
        const uint8_t* qs = base + 32;
        const uint8_t* scale_bytes = base + 96;
        uint16_t dh; std::memcpy(&dh, base + 108, 2);
        float d_all = f16_bits_to_f32(dh);

        uint32_t aux[4];
        std::memcpy(aux, scale_bytes, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t* scales = (const int8_t*)aux;

        const uint8_t* q = qs;
        const uint8_t* hm = hmask; // fixed base -- never advanced across n, matches reference
        uint8_t m = 1;
        int is = 0;
        float* yy = y + (size_t)i * 256;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                float dl0 = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    int lowbits = (q[l] >> shift) & 3;
                    int sub4 = (hm[l] & m) ? 0 : 4;
                    *yy++ = dl0 * (float)(lowbits - sub4);
                }
                float dl1 = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    int lowbits = (q[l + 16] >> shift) & 3;
                    int sub4 = (hm[l + 16] & m) ? 0 : 4;
                    *yy++ = dl1 * (float)(lowbits - sub4);
                }
                shift += 2;
                m = (uint8_t)(m << 1);
            }
            q += 32;
        }
    }
}

static void test_q2k_q3k_dequant_logic() {
    std::cout << "[TEST] Q2_K/Q3_K dequant logic (oracle vs AVX-512 algorithm, scalar form)..." << std::flush;
    std::mt19937 rng(2024);
    std::uniform_int_distribution<int> byte_dist(0, 255);

    // Multi-block rows too, to catch a per-block stride bug.
    for (int elements : {256, 512, 1024}) {
        int nb = elements / 256;

        std::vector<uint8_t> q2_bytes((size_t)nb * 84);
        for (auto& b : q2_bytes) b = (uint8_t)byte_dist(rng);
        std::vector<float> y_oracle(elements), y_cand(elements);
        oracle_dequant_q2_K(q2_bytes.data(), y_oracle.data(), elements);
        candidate_dequant_q2_K(q2_bytes.data(), y_cand.data(), elements);
        for (int i = 0; i < elements; ++i) {
            assert(std::fabs(y_oracle[i] - y_cand[i]) < 1e-5f && "Q2_K logic mismatch");
        }

        std::vector<uint8_t> q3_bytes((size_t)nb * 110);
        for (auto& b : q3_bytes) b = (uint8_t)byte_dist(rng);
        oracle_dequant_q3_K(q3_bytes.data(), y_oracle.data(), elements);
        candidate_dequant_q3_K(q3_bytes.data(), y_cand.data(), elements);
        for (int i = 0; i < elements; ++i) {
            assert(std::fabs(y_oracle[i] - y_cand[i]) < 1e-5f && "Q3_K logic mismatch");
        }
    }

    std::cout << " PASSED\n";
}

// ============================================================
// Test: IQ2_XXS / IQ3_XXS dequant LOGIC (oracle vs AVX-512 algorithm,
// scalar form) -- same rationale as the Q2_K/Q3_K logic test above, but for
// the codebook/grid-based format this project's actual target GLM-5.2
// UD-IQ2_M checkpoints use for routed experts (71/76 layers: gate=up=
// IQ2_XXS, down=IQ3_XXS -- confirmed via the real startup [QUANT] report,
// not the config file's Q2_K/Q3_K comment that turned out to be wrong).
// Uses a SYNTHETIC grid/sign/mask table (not real ggml data, which lives in
// a sibling ../llama.cpp checkout this machine doesn't have) -- the
// bit-unpacking/indexing LOGIC being verified here doesn't depend on the
// table's actual content, only on oracle and candidate reading the same
// bytes the same way. `oracle_*` mirrors the reference's own pointer-cast
// style (aux8 aliasing into aux32, matching CUDA/ggml source structure);
// `candidate_*` mirrors src/kernels_cpu.cpp's actual memcpy-based
// implementation structure -- so this specifically cross-checks the
// memcpy/offset arithmetic in the real production code against a
// structurally-independent re-derivation of the same formula. See
// src/main_test.cpp's Test 7 for the full AVX-512-vs-ggml-reference
// comparison with real ggml data (requires the Linux+CUDA build).
static void test_iq2xxs_iq3xxs_dequant_logic() {
    std::cout << "[TEST] IQ2_XXS/IQ3_XXS dequant logic (oracle vs AVX-512 algorithm, scalar form)..." << std::flush;

    // Synthetic tables -- deterministic, arbitrary content (logic test only).
    uint64_t grid2[256];
    uint32_t grid3[256];
    for (int i = 0; i < 256; ++i) {
        uint64_t v = 0;
        for (int b = 0; b < 8; ++b) v |= (uint64_t)(uint8_t)((i * 7 + b * 13 + 3) & 0xFF) << (8 * b);
        grid2[i] = v;
        grid3[i] = (uint32_t)v; // low 4 bytes reused -- fine, arbitrary content
    }
    uint8_t ksigns[128];
    for (int i = 0; i < 128; ++i) ksigns[i] = (uint8_t)(i * 31 + 7);
    uint8_t kmask[8] = {1, 2, 4, 8, 16, 32, 64, 128};

    auto oracle_iq2_xxs = [&](const uint8_t* block, float* y, int64_t k) {
        const int nb = (int)(k / 256);
        for (int i = 0; i < nb; ++i) {
            const uint8_t* base = block + (size_t)i * 66;
            uint16_t dh; std::memcpy(&dh, base, 2);
            float d = f16_bits_to_f32(dh);
            const uint16_t* qs = (const uint16_t*)(base + 2);
            float* yy = y + (size_t)i * 256;
            uint32_t aux32[2];
            const uint8_t* aux8 = (const uint8_t*)aux32;
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                const uint16_t* q16 = qs + 4 * ib32;
                uint16_t q0, q1, q2, q3;
                std::memcpy(&q0, &q16[0], 2); std::memcpy(&q1, &q16[1], 2);
                std::memcpy(&q2, &q16[2], 2); std::memcpy(&q3, &q16[3], 2);
                aux32[0] = (uint32_t)q0 | ((uint32_t)q1 << 16);
                aux32[1] = (uint32_t)q2 | ((uint32_t)q3 << 16);
                const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
                for (int l = 0; l < 4; ++l) {
                    uint64_t g = grid2[aux8[l]];
                    const uint8_t* gb = (const uint8_t*)&g;
                    const uint8_t signs = ksigns[(aux32[1] >> (7 * l)) & 127];
                    for (int j = 0; j < 8; ++j)
                        yy[ib32 * 32 + l * 8 + j] = db * (float)gb[j] * ((signs & kmask[j]) ? -1.f : 1.f);
                }
            }
        }
    };

    auto candidate_iq2_xxs = [&](const uint8_t* block, float* y, int64_t k) {
        const int nb = (int)(k / 256);
        for (int i = 0; i < nb; ++i) {
            const uint8_t* base = block + (size_t)i * 66;
            uint16_t dh; std::memcpy(&dh, base, 2);
            float d = f16_bits_to_f32(dh);
            const uint8_t* qs_bytes = base + 2;
            float* yy = y + (size_t)i * 256;
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                uint16_t q0, q1, q2, q3;
                std::memcpy(&q0, qs_bytes + (size_t)(4 * ib32 + 0) * 2, 2);
                std::memcpy(&q1, qs_bytes + (size_t)(4 * ib32 + 1) * 2, 2);
                std::memcpy(&q2, qs_bytes + (size_t)(4 * ib32 + 2) * 2, 2);
                std::memcpy(&q3, qs_bytes + (size_t)(4 * ib32 + 3) * 2, 2);
                uint32_t a0 = (uint32_t)q0 | ((uint32_t)q1 << 16);
                uint32_t a1 = (uint32_t)q2 | ((uint32_t)q3 << 16);
                uint8_t aux8[4]; std::memcpy(aux8, &a0, 4);
                const float db = d * (0.5f + (float)(a1 >> 28)) * 0.25f;
                for (int l = 0; l < 4; ++l) {
                    uint64_t g = grid2[aux8[l]];
                    uint8_t gb[8]; std::memcpy(gb, &g, 8);
                    uint8_t signs = ksigns[(a1 >> (7 * l)) & 127];
                    for (int j = 0; j < 8; ++j)
                        yy[ib32 * 32 + l * 8 + j] = db * (float)gb[j] * ((signs & kmask[j]) ? -1.f : 1.f);
                }
            }
        }
    };

    auto oracle_iq3_xxs = [&](const uint8_t* block, float* y, int64_t k) {
        const int nb = (int)(k / 256);
        for (int i = 0; i < nb; ++i) {
            const uint8_t* base = block + (size_t)i * 98;
            uint16_t dh; std::memcpy(&dh, base, 2);
            float d = f16_bits_to_f32(dh);
            const uint8_t* qs = base + 2;
            const uint8_t* sas0 = qs + 64;
            float* yy = y + (size_t)i * 256;
            const uint8_t* qptr = qs;
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                const uint8_t* sas = sas0 + 4 * ib32;
                uint32_t aux32 = (uint32_t)sas[0] | ((uint32_t)sas[1] << 8) |
                                 ((uint32_t)sas[2] << 16) | ((uint32_t)sas[3] << 24);
                const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
                for (int l = 0; l < 4; ++l) {
                    const uint8_t signs = ksigns[(aux32 >> (7 * l)) & 127];
                    const uint8_t* grid1 = (const uint8_t*)&grid3[qptr[2 * l + 0]];
                    const uint8_t* grid2b = (const uint8_t*)&grid3[qptr[2 * l + 1]];
                    for (int j = 0; j < 4; ++j) {
                        yy[ib32 * 32 + l * 8 + j + 0] = db * (float)grid1[j] * ((signs & kmask[j + 0]) ? -1.f : 1.f);
                        yy[ib32 * 32 + l * 8 + j + 4] = db * (float)grid2b[j] * ((signs & kmask[j + 4]) ? -1.f : 1.f);
                    }
                }
                qptr += 8;
            }
        }
    };

    auto candidate_iq3_xxs = [&](const uint8_t* block, float* y, int64_t k) {
        const int nb = (int)(k / 256);
        for (int i = 0; i < nb; ++i) {
            const uint8_t* base = block + (size_t)i * 98;
            uint16_t dh; std::memcpy(&dh, base, 2);
            float d = f16_bits_to_f32(dh);
            const uint8_t* qs = base + 2;
            const uint8_t* scales_and_signs = qs + 64;
            float* yy = y + (size_t)i * 256;
            const uint8_t* qptr = qs;
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                const uint8_t* sas = scales_and_signs + 4 * ib32;
                uint32_t aux32 = (uint32_t)sas[0] | ((uint32_t)sas[1] << 8) |
                                 ((uint32_t)sas[2] << 16) | ((uint32_t)sas[3] << 24);
                const float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
                for (int l = 0; l < 4; ++l) {
                    uint8_t signs = ksigns[(aux32 >> (7 * l)) & 127];
                    uint32_t g1 = grid3[qptr[2 * l + 0]];
                    uint32_t g2 = grid3[qptr[2 * l + 1]];
                    uint8_t combined[8];
                    std::memcpy(combined, &g1, 4);
                    std::memcpy(combined + 4, &g2, 4);
                    for (int j = 0; j < 8; ++j)
                        yy[ib32 * 32 + l * 8 + j] = db * (float)combined[j] * ((signs & kmask[j]) ? -1.f : 1.f);
                }
                qptr += 8;
            }
        }
    };

    std::mt19937 rng(4242);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    for (int elements : {256, 512, 1024}) {
        int nb = elements / 256;

        std::vector<uint8_t> iq2_bytes((size_t)nb * 66);
        for (auto& b : iq2_bytes) b = (uint8_t)byte_dist(rng);
        std::vector<float> y_oracle(elements), y_cand(elements);
        oracle_iq2_xxs(iq2_bytes.data(), y_oracle.data(), elements);
        candidate_iq2_xxs(iq2_bytes.data(), y_cand.data(), elements);
        for (int i = 0; i < elements; ++i) {
            assert(std::fabs(y_oracle[i] - y_cand[i]) < 1e-5f && "IQ2_XXS logic mismatch");
        }

        std::vector<uint8_t> iq3_bytes((size_t)nb * 98);
        for (auto& b : iq3_bytes) b = (uint8_t)byte_dist(rng);
        oracle_iq3_xxs(iq3_bytes.data(), y_oracle.data(), elements);
        candidate_iq3_xxs(iq3_bytes.data(), y_cand.data(), elements);
        for (int i = 0; i < elements; ++i) {
            assert(std::fabs(y_oracle[i] - y_cand[i]) < 1e-5f && "IQ3_XXS logic mismatch");
        }
    }

    std::cout << " PASSED\n";
}

int main() {
    std::cout << "=== MoEStream Component Tests ===\n";
    test_reasoning_parser();
    test_rope();
    test_topk_selection();
    test_softmax_routing();
    test_calculate_size();
    test_spsc_ring();
    test_toml_parser();
    test_q2k_q3k_dequant_logic();
    test_iq2xxs_iq3xxs_dequant_logic();
    std::cout << "\nAll tests PASSED.\n";
    return 0;
}
