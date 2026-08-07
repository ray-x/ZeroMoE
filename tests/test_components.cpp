// tests/test_components.cpp — Unit tests for MoEStream components
// Build: g++ -std=c++23 -O2 -I../include -o test_components test_components.cpp
// (Does NOT require CUDA for the CPU-path tests)
#include "direct_io.hpp" // the REAL O_DIRECT alignment-window arithmetic (no CUDA deps)
#include "kernels_cpu.hpp" // CPU kernel backend (AVX-512 or NEON, selected by build script)
#include "rope.hpp" // the REAL, unit-tested RoPE (no CUDA / no AVX-512 deps)
#include "route_usage.hpp" // CUDA-free routing histogram used by CPU and GPU paths
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <unordered_map>
#include <vector>

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

  int feed(const std::string &token_text) {
    pending += token_text;
    if (state == GenerationState::Outputting) {
      size_t pos = pending.find("<think>");
      if (pos != std::string::npos) {
        state = GenerationState::Thinking;
        pending.clear();
        return 2;
      }
      if (pending.size() < 7 && std::string("<think>").find(pending) == 0)
        return 2;
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
      if (pending.size() < 8 && std::string("</think>").find(pending) == 0)
        return 2;
      pending.clear();
      think_tokens++;
      return 1;
    }
  }
  void reset() {
    state = GenerationState::Outputting;
    pending.clear();
    think_tokens = 0;
    output_tokens = 0;
  }
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
struct ExpertScore {
  int id;
  float score;
};

void test_topk_selection() {
  std::cout << "[TEST] Top-K Expert Selection..." << std::flush;
  const int N = 256;
  const int K = 8;
  std::vector<ExpertScore> scores(N);
  for (int i = 0; i < N; ++i)
    scores[i] = {i, (float)(N - i)}; // descending
  scores[42].score = 9999.0f;        // make expert 42 the best
  scores[7].score = 9998.0f;

  std::partial_sort(scores.begin(), scores.begin() + K, scores.end(),
                    [](const ExpertScore &a, const ExpertScore &b) {
                      return a.score > b.score;
                    });

  assert(scores[0].id == 42);
  assert(scores[1].id == 7);
  // The rest of top-8 should be ids 0..5 (original scores 256..251)
  assert(scores[0].score > scores[1].score);
  assert(scores[K - 1].score > scores[K].score);

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
  for (int k = 0; k < K; ++k)
    weights[k] *= inv_sum;

  // Verify sums to 1.0
  float total = 0.0f;
  for (int k = 0; k < K; ++k)
    total += weights[k];
  assert(std::abs(total - 1.0f) < 1e-5f);

  // Verify ordering preserved
  for (int k = 1; k < K; ++k)
    assert(weights[k - 1] >= weights[k]);

  std::cout << " PASSED\n";
}

// ============================================================
// Test 4: GGUF calculate_size for various quant types
// ============================================================
void test_calculate_size() {
  std::cout << "[TEST] GGUF calculate_size..." << std::flush;
  // Simulate the function logic
  auto calc = [](uint64_t numel, uint32_t t) -> uint64_t {
    if (t == 0)
      return numel * 4; // F32
    if (t == 1)
      return numel * 2; // F16
    if (t == 8)
      return (numel / 32) * 34; // Q8_0
    if (t == 10)
      return (numel / 256) * 84; // Q2_K
    if (t == 13)
      return (numel / 256) * 176; // Q5_K
    if (t == 14)
      return (numel / 256) * 210; // Q6_K
    if (t == 25)
      return numel * 2; // BF16
    if (t == 31)
      return (numel / 256) * 84; // IQ2_M
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
  assert(expert_size == (12582912 / 256) * 176); // 8,650,752 bytes

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

inline std::string trim(const std::string &s) {
  size_t a = s.find_first_not_of(" \t\r\n");
  size_t b = s.find_last_not_of(" \t\r\n");
  return (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
}

inline std::string strip_comment(const std::string &line) {
  bool in_string = false;
  for (size_t i = 0; i < line.size(); ++i) {
    if (line[i] == '"')
      in_string = !in_string;
    if (line[i] == '#' && !in_string)
      return line.substr(0, i);
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
  std::string get_str(const std::string &key,
                      const std::string &def = "") const {
    auto it = values.find(key);
    return (it != values.end()) ? it->second.str : def;
  }
  int get_int(const std::string &key, int def = 0) const {
    auto it = values.find(key);
    return (it != values.end() && it->second.type == Value::Number)
               ? (int)it->second.num
               : def;
  }
  double get_double(const std::string &key, double def = 0.0) const {
    auto it = values.find(key);
    return (it != values.end() && it->second.type == Value::Number)
               ? it->second.num
               : def;
  }
  bool get_bool(const std::string &key, bool def = false) const {
    auto it = values.find(key);
    return (it != values.end() && it->second.type == Value::Bool)
               ? it->second.boolean
               : def;
  }
};

inline Value parse_value(const std::string &raw) {
  Value v;
  std::string s = trim(raw);
  if (s.empty())
    return v;
  if (s.front() == '"' && s.back() == '"') {
    v.type = Value::String;
    v.str = s.substr(1, s.size() - 2);
    return v;
  }
  if (s == "true" || s == "false") {
    v.type = Value::Bool;
    v.boolean = (s == "true");
    v.str = s;
    return v;
  }
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
  Table *current = &root;

  std::istringstream stream(config);
  std::string line;
  while (std::getline(stream, line)) {
    line = trim(strip_comment(line));
    if (line.empty())
      continue;
    if (line == "[[gpus]]") {
      gpu_tables.emplace_back();
      current = &gpu_tables.back();
      continue;
    }
    if (line.front() == '[') {
      current = &root;
      continue;
    }
    size_t eq = line.find('=');
    if (eq == std::string::npos)
      continue;
    current->values[trim(line.substr(0, eq))] =
        parse_value(trim(line.substr(eq + 1)));
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
// DeepSeek require INTERLEAVED RoPE (pairs (2j, 2j+1)), but apply_rope_yarn
// used to do NeoX rotate-half (pairs (k, k+half)). The known-answer case below
// DISTINGUISHES the two -- rotate-half yields [cos1, 0, sin1, 0], interleaved
// yields [cos1, sin1, 0, 0]. If rope.hpp is ever reverted to rotate-half, this
// test fails loudly. Calls the REAL rope_yarn_inplace from include/rope.hpp
// (not a copy), via #include "rope.hpp".
void test_rope() {
  std::cout << "[TEST] RoPE convention (interleaved)..." << std::flush;

  // Independent interleaved reference oracle (no YaRN, ext_factor=0). Lives in
  // the test so a regression in rope.hpp is caught against a separate impl.
  auto ref_interleaved = [](float *v, int dim, int pos, float theta_base) {
    int half = dim / 2;
    float theta_scale = std::pow(theta_base, -2.0f / (float)dim);
    float theta_extrap = (float)pos;
    for (int j = 0; j < half; ++j) {
      float t = theta_extrap;
      float a = v[2 * j], b = v[2 * j + 1];
      v[2 * j] = a * std::cos(t) - b * std::sin(t);
      v[2 * j + 1] = a * std::sin(t) + b * std::cos(t);
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
                      /*attn_factor=*/1.0f, /*yarn_log_mul=*/0.0f,
                      /*n_ctx_orig=*/0);
    float c1 = std::cos(1.0f), s1 = std::sin(1.0f);
    assert(std::fabs(v[0] - c1) < 1e-5f);
    assert(std::fabs(v[1] - s1) < 1e-5f); // rotate-half leaves this 0
    assert(std::fabs(v[2]) < 1e-5f);      // rotate-half puts sin1 here
    assert(std::fabs(v[3]) < 1e-5f);
  }

  // --- Case 2: regression at production size (qk_rope=64, theta=8e6). The real
  // rope_yarn_inplace must match the independent interleaved oracle to ~1e-4.
  {
    const int dim = 64, pos = 7;
    const float theta = 8.0e6f;
    std::vector<float> a(dim), b(dim);
    for (int i = 0; i < dim; ++i) {
      a[i] = 0.01f * (float)(i + 1);
      b[i] = a[i];
    }
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
    for (int i = 0; i < dim; ++i)
      v[i] = 0.01f * (float)(i + 1);
    double n0 = 0;
    for (float x : v)
      n0 += (double)x * x;
    rope_yarn_inplace(v.data(), dim, pos, theta, 0.0f, 1.0f, 1.0f, 0.0f, 0);
    double n1 = 0;
    for (float x : v)
      n1 += (double)x * x;
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
  if (exp == 0)
    bits = sign; // flush subnormal to zero, matches f32_to_f16_bits's flush
  else
    bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

// Verbatim scalar port of ggml's dequantize_row_q2_K (block: scales[16] qs[64]
// d dmin, 84B).
static void oracle_dequant_q2_K(const uint8_t *block, float *y, int64_t k) {
  const int nb = (int)(k / 256);
  for (int i = 0; i < nb; ++i) {
    const uint8_t *base = block + (size_t)i * 84;
    const uint8_t *scales = base;
    const uint8_t *qs = base + 16;
    uint16_t dh, dminh;
    std::memcpy(&dh, base + 80, 2);
    std::memcpy(&dminh, base + 82, 2);
    float d = f16_bits_to_f32(dh), min = f16_bits_to_f32(dminh);
    const uint8_t *q = qs;
    int is = 0;
    float *yy = y + (size_t)i * 256;
    for (int n = 0; n < 256; n += 128) {
      int shift = 0;
      for (int j = 0; j < 4; ++j) {
        uint8_t sc = scales[is++];
        float dl = d * (sc & 0xF), ml = min * (sc >> 4);
        for (int l = 0; l < 16; ++l)
          *yy++ = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;
        sc = scales[is++];
        dl = d * (sc & 0xF);
        ml = min * (sc >> 4);
        for (int l = 0; l < 16; ++l)
          *yy++ = dl * ((int8_t)((q[l + 16] >> shift) & 3)) - ml;
        shift += 2;
      }
      q += 32;
    }
  }
}

// Verbatim scalar port of ggml's dequantize_row_q3_K (block: hmask[32] qs[64]
// scales[12] d, 110B).
static void oracle_dequant_q3_K(const uint8_t *block, float *y, int64_t k) {
  const int nb = (int)(k / 256);
  const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
  for (int i = 0; i < nb; ++i) {
    const uint8_t *base = block + (size_t)i * 110;
    const uint8_t *hmask = base;
    const uint8_t *qs = base + 32;
    const uint8_t *scale_bytes = base + 96;
    uint16_t dh;
    std::memcpy(&dh, base + 108, 2);
    float d_all = f16_bits_to_f32(dh);

    uint32_t aux[4];
    std::memcpy(aux, scale_bytes, 12);
    uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const int8_t *scales = (const int8_t *)aux;

    const uint8_t *q = qs;
    const uint8_t *hm = hmask;
    uint8_t m = 1;
    int is = 0;
    float *yy = y + (size_t)i * 256;
    for (int n = 0; n < 256; n += 128) {
      int shift = 0;
      for (int j = 0; j < 4; ++j) {
        float dl = d_all * (scales[is++] - 32);
        for (int l = 0; l < 16; ++l)
          *yy++ = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
        dl = d_all * (scales[is++] - 32);
        for (int l = 0; l < 16; ++l)
          *yy++ = dl * ((int8_t)((q[l + 16] >> shift) & 3) -
                        ((hm[l + 16] & m) ? 0 : 4));
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
static void candidate_dequant_q2_K(const uint8_t *block, float *y, int64_t k) {
  const int nb = (int)(k / 256);
  for (int i = 0; i < nb; ++i) {
    const uint8_t *base = block + (size_t)i * 84;
    const uint8_t *scales = base;
    const uint8_t *qs = base + 16;
    uint16_t dh, dminh;
    std::memcpy(&dh, base + 80, 2);
    std::memcpy(&dminh, base + 82, 2);
    float d = f16_bits_to_f32(dh), dmin = f16_bits_to_f32(dminh);
    const uint8_t *q = qs;
    int is = 0;
    float *yy = y + (size_t)i * 256;
    for (int n = 0; n < 256; n += 128) {
      int shift = 0;
      for (int j = 0; j < 4; ++j) {
        uint8_t sc0 = scales[is++];
        float dl0 = d * (sc0 & 0xF), ml0 = dmin * (sc0 >> 4);
        for (int l = 0; l < 16; ++l) {
          int v = (q[l] >> shift) & 3;
          *yy++ = dl0 * (float)v - ml0;
        }
        uint8_t sc1 = scales[is++];
        float dl1 = d * (sc1 & 0xF), ml1 = dmin * (sc1 >> 4);
        for (int l = 0; l < 16; ++l) {
          int v = (q[l + 16] >> shift) & 3;
          *yy++ = dl1 * (float)v - ml1;
        }
        shift += 2;
      }
      q += 32;
    }
  }
}
static void candidate_dequant_q3_K(const uint8_t *block, float *y, int64_t k) {
  const int nb = (int)(k / 256);
  const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
  for (int i = 0; i < nb; ++i) {
    const uint8_t *base = block + (size_t)i * 110;
    const uint8_t *hmask = base;
    const uint8_t *qs = base + 32;
    const uint8_t *scale_bytes = base + 96;
    uint16_t dh;
    std::memcpy(&dh, base + 108, 2);
    float d_all = f16_bits_to_f32(dh);

    uint32_t aux[4];
    std::memcpy(aux, scale_bytes, 12);
    uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const int8_t *scales = (const int8_t *)aux;

    const uint8_t *q = qs;
    const uint8_t *hm =
        hmask; // fixed base -- never advanced across n, matches reference
    uint8_t m = 1;
    int is = 0;
    float *yy = y + (size_t)i * 256;
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
  std::cout << "[TEST] Q2_K/Q3_K dequant logic (oracle vs AVX-512 algorithm, "
               "scalar form)..."
            << std::flush;
  std::mt19937 rng(2024);
  std::uniform_int_distribution<int> byte_dist(0, 255);

  // Multi-block rows too, to catch a per-block stride bug.
  for (int elements : {256, 512, 1024}) {
    int nb = elements / 256;

    std::vector<uint8_t> q2_bytes((size_t)nb * 84);
    for (auto &b : q2_bytes)
      b = (uint8_t)byte_dist(rng);
    std::vector<float> y_oracle(elements), y_cand(elements);
    oracle_dequant_q2_K(q2_bytes.data(), y_oracle.data(), elements);
    candidate_dequant_q2_K(q2_bytes.data(), y_cand.data(), elements);
    for (int i = 0; i < elements; ++i) {
      assert(std::fabs(y_oracle[i] - y_cand[i]) < 1e-5f &&
             "Q2_K logic mismatch");
    }

    std::vector<uint8_t> q3_bytes((size_t)nb * 110);
    for (auto &b : q3_bytes)
      b = (uint8_t)byte_dist(rng);
    oracle_dequant_q3_K(q3_bytes.data(), y_oracle.data(), elements);
    candidate_dequant_q3_K(q3_bytes.data(), y_cand.data(), elements);
    for (int i = 0; i < elements; ++i) {
      assert(std::fabs(y_oracle[i] - y_cand[i]) < 1e-5f &&
             "Q3_K logic mismatch");
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
  std::cout << "[TEST] IQ2_XXS/IQ3_XXS dequant logic (oracle vs AVX-512 "
               "algorithm, scalar form)..."
            << std::flush;

  // Synthetic tables -- deterministic, arbitrary content (logic test only).
  uint64_t grid2[256];
  uint32_t grid3[256];
  for (int i = 0; i < 256; ++i) {
    uint64_t v = 0;
    for (int b = 0; b < 8; ++b)
      v |= (uint64_t)(uint8_t)((i * 7 + b * 13 + 3) & 0xFF) << (8 * b);
    grid2[i] = v;
    grid3[i] = (uint32_t)v; // low 4 bytes reused -- fine, arbitrary content
  }
  uint8_t ksigns[128];
  for (int i = 0; i < 128; ++i)
    ksigns[i] = (uint8_t)(i * 31 + 7);
  uint8_t kmask[8] = {1, 2, 4, 8, 16, 32, 64, 128};

  auto oracle_iq2_xxs = [&](const uint8_t *block, float *y, int64_t k) {
    const int nb = (int)(k / 256);
    for (int i = 0; i < nb; ++i) {
      const uint8_t *base = block + (size_t)i * 66;
      uint16_t dh;
      std::memcpy(&dh, base, 2);
      float d = f16_bits_to_f32(dh);
      const uint16_t *qs = (const uint16_t *)(base + 2);
      float *yy = y + (size_t)i * 256;
      uint32_t aux32[2];
      const uint8_t *aux8 = (const uint8_t *)aux32;
      for (int ib32 = 0; ib32 < 8; ++ib32) {
        const uint16_t *q16 = qs + 4 * ib32;
        uint16_t q0, q1, q2, q3;
        std::memcpy(&q0, &q16[0], 2);
        std::memcpy(&q1, &q16[1], 2);
        std::memcpy(&q2, &q16[2], 2);
        std::memcpy(&q3, &q16[3], 2);
        aux32[0] = (uint32_t)q0 | ((uint32_t)q1 << 16);
        aux32[1] = (uint32_t)q2 | ((uint32_t)q3 << 16);
        const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
        for (int l = 0; l < 4; ++l) {
          uint64_t g = grid2[aux8[l]];
          const uint8_t *gb = (const uint8_t *)&g;
          const uint8_t signs = ksigns[(aux32[1] >> (7 * l)) & 127];
          for (int j = 0; j < 8; ++j)
            yy[ib32 * 32 + l * 8 + j] =
                db * (float)gb[j] * ((signs & kmask[j]) ? -1.f : 1.f);
        }
      }
    }
  };

  auto candidate_iq2_xxs = [&](const uint8_t *block, float *y, int64_t k) {
    const int nb = (int)(k / 256);
    for (int i = 0; i < nb; ++i) {
      const uint8_t *base = block + (size_t)i * 66;
      uint16_t dh;
      std::memcpy(&dh, base, 2);
      float d = f16_bits_to_f32(dh);
      const uint8_t *qs_bytes = base + 2;
      float *yy = y + (size_t)i * 256;
      for (int ib32 = 0; ib32 < 8; ++ib32) {
        uint16_t q0, q1, q2, q3;
        std::memcpy(&q0, qs_bytes + (size_t)(4 * ib32 + 0) * 2, 2);
        std::memcpy(&q1, qs_bytes + (size_t)(4 * ib32 + 1) * 2, 2);
        std::memcpy(&q2, qs_bytes + (size_t)(4 * ib32 + 2) * 2, 2);
        std::memcpy(&q3, qs_bytes + (size_t)(4 * ib32 + 3) * 2, 2);
        uint32_t a0 = (uint32_t)q0 | ((uint32_t)q1 << 16);
        uint32_t a1 = (uint32_t)q2 | ((uint32_t)q3 << 16);
        uint8_t aux8[4];
        std::memcpy(aux8, &a0, 4);
        const float db = d * (0.5f + (float)(a1 >> 28)) * 0.25f;
        for (int l = 0; l < 4; ++l) {
          uint64_t g = grid2[aux8[l]];
          uint8_t gb[8];
          std::memcpy(gb, &g, 8);
          uint8_t signs = ksigns[(a1 >> (7 * l)) & 127];
          for (int j = 0; j < 8; ++j)
            yy[ib32 * 32 + l * 8 + j] =
                db * (float)gb[j] * ((signs & kmask[j]) ? -1.f : 1.f);
        }
      }
    }
  };

  auto oracle_iq3_xxs = [&](const uint8_t *block, float *y, int64_t k) {
    const int nb = (int)(k / 256);
    for (int i = 0; i < nb; ++i) {
      const uint8_t *base = block + (size_t)i * 98;
      uint16_t dh;
      std::memcpy(&dh, base, 2);
      float d = f16_bits_to_f32(dh);
      const uint8_t *qs = base + 2;
      const uint8_t *sas0 = qs + 64;
      float *yy = y + (size_t)i * 256;
      const uint8_t *qptr = qs;
      for (int ib32 = 0; ib32 < 8; ++ib32) {
        const uint8_t *sas = sas0 + 4 * ib32;
        uint32_t aux32 = (uint32_t)sas[0] | ((uint32_t)sas[1] << 8) |
                         ((uint32_t)sas[2] << 16) | ((uint32_t)sas[3] << 24);
        const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
        for (int l = 0; l < 4; ++l) {
          const uint8_t signs = ksigns[(aux32 >> (7 * l)) & 127];
          const uint8_t *grid1 = (const uint8_t *)&grid3[qptr[2 * l + 0]];
          const uint8_t *grid2b = (const uint8_t *)&grid3[qptr[2 * l + 1]];
          for (int j = 0; j < 4; ++j) {
            yy[ib32 * 32 + l * 8 + j + 0] =
                db * (float)grid1[j] * ((signs & kmask[j + 0]) ? -1.f : 1.f);
            yy[ib32 * 32 + l * 8 + j + 4] =
                db * (float)grid2b[j] * ((signs & kmask[j + 4]) ? -1.f : 1.f);
          }
        }
        qptr += 8;
      }
    }
  };

  auto candidate_iq3_xxs = [&](const uint8_t *block, float *y, int64_t k) {
    const int nb = (int)(k / 256);
    for (int i = 0; i < nb; ++i) {
      const uint8_t *base = block + (size_t)i * 98;
      uint16_t dh;
      std::memcpy(&dh, base, 2);
      float d = f16_bits_to_f32(dh);
      const uint8_t *qs = base + 2;
      const uint8_t *scales_and_signs = qs + 64;
      float *yy = y + (size_t)i * 256;
      const uint8_t *qptr = qs;
      for (int ib32 = 0; ib32 < 8; ++ib32) {
        const uint8_t *sas = scales_and_signs + 4 * ib32;
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
            yy[ib32 * 32 + l * 8 + j] =
                db * (float)combined[j] * ((signs & kmask[j]) ? -1.f : 1.f);
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
    for (auto &b : iq2_bytes)
      b = (uint8_t)byte_dist(rng);
    std::vector<float> y_oracle(elements), y_cand(elements);
    oracle_iq2_xxs(iq2_bytes.data(), y_oracle.data(), elements);
    candidate_iq2_xxs(iq2_bytes.data(), y_cand.data(), elements);
    for (int i = 0; i < elements; ++i) {
      assert(std::fabs(y_oracle[i] - y_cand[i]) < 1e-5f &&
             "IQ2_XXS logic mismatch");
    }

    std::vector<uint8_t> iq3_bytes((size_t)nb * 98);
    for (auto &b : iq3_bytes)
      b = (uint8_t)byte_dist(rng);
    oracle_iq3_xxs(iq3_bytes.data(), y_oracle.data(), elements);
    candidate_iq3_xxs(iq3_bytes.data(), y_cand.data(), elements);
    for (int i = 0; i < elements; ++i) {
      assert(std::fabs(y_oracle[i] - y_cand[i]) < 1e-5f &&
             "IQ3_XXS logic mismatch");
    }
  }

  std::cout << " PASSED\n";
}

// IQ2_S: block = d(2) + qs[64] + qh[8] + scales[8] = 82B/block. Same
// methodology/caveat as test_iq2xxs_iq3xxs_dequant_logic above: synthetic
// grid table (real ggml table data isn't available on this darwin machine),
// verifies the bit-unpacking/indexing bookkeeping (grid-index assembly from
// qs+qh, per-ib32 4-bit-pair scale split, sign-byte-overlaps-back-half-of-
// qs layout) against an independently-written oracle -- not the real table
// content, and can't cover the AVX-512 intrinsics (apply_grid_group8)
// themselves, but does cover the higher-risk part.
static void test_iq2s_dequant_logic() {
  std::cout << "[TEST] IQ2_S dequant logic (oracle vs AVX-512 algorithm, "
               "scalar form)..."
            << std::flush;

  uint64_t grid[1024];
  for (int i = 0; i < 1024; ++i) {
    uint64_t v = 0;
    for (int b = 0; b < 8; ++b)
      v |= (uint64_t)(uint8_t)((i * 7 + b * 13 + 3) & 0xFF) << (8 * b);
    grid[i] = v;
  }
  uint8_t kmask[8] = {1, 2, 4, 8, 16, 32, 64, 128};

  auto oracle_iq2_s = [&](const uint8_t *block, float *y, int64_t k) {
    const int nb = (int)(k / 256);
    for (int i = 0; i < nb; ++i) {
      const uint8_t *base = block + (size_t)i * 82;
      uint16_t dh;
      std::memcpy(&dh, base, 2);
      float d = f16_bits_to_f32(dh);
      const uint8_t *qs = base + 2;
      const uint8_t *qh = base + 2 + 64;
      const uint8_t *scales = base + 2 + 64 + 8;
      const uint8_t *signs = qs + 32;
      float *yy = y + (size_t)i * 256;
      for (int ib32 = 0; ib32 < 8; ++ib32) {
        const float db0 = d * (0.5f + (scales[ib32] & 0xf)) * 0.25f;
        const float db1 = d * (0.5f + (scales[ib32] >> 4)) * 0.25f;
        for (int l = 0; l < 4; ++l) {
          const float dl = (l / 2 == 0) ? db0 : db1;
          const uint32_t idx =
              qs[4 * ib32 + l] | (((uint32_t)qh[ib32] << (8 - 2 * l)) & 0x300);
          uint64_t g = grid[idx];
          const uint8_t *gb = (const uint8_t *)&g;
          uint8_t sg = signs[4 * ib32 + l];
          for (int j = 0; j < 8; ++j)
            yy[ib32 * 32 + l * 8 + j] =
                dl * (float)gb[j] * ((sg & kmask[j]) ? -1.f : 1.f);
        }
      }
    }
  };

  auto candidate_iq2_s = [&](const uint8_t *block, float *y, int64_t k) {
    const int nb = (int)(k / 256);
    for (int i = 0; i < nb; ++i) {
      const uint8_t *base = block + (size_t)i * 82;
      uint16_t dh;
      std::memcpy(&dh, base, 2);
      float d = f16_bits_to_f32(dh);
      const uint8_t *qs = base + 2;
      const uint8_t *qh = base + 2 + 64;
      const uint8_t *scales = base + 2 + 64 + 8;
      const uint8_t *signs = qs + 32;
      float *yy = y + (size_t)i * 256;
      for (int ib32 = 0; ib32 < 8; ++ib32) {
        const float db0 = d * (0.5f + (float)(scales[ib32] & 0xf)) * 0.25f;
        const float db1 = d * (0.5f + (float)(scales[ib32] >> 4)) * 0.25f;
        const uint8_t *qs_ib = qs + 4 * ib32;
        const uint8_t *signs_ib = signs + 4 * ib32;
        for (int l = 0; l < 4; ++l) {
          const float dl = (l / 2 == 0) ? db0 : db1;
          const uint32_t idx =
              qs_ib[l] | (((uint32_t)qh[ib32] << (8 - 2 * l)) & 0x300);
          uint64_t g = grid[idx];
          uint8_t gb[8];
          std::memcpy(gb, &g, 8);
          uint8_t sg = signs_ib[l];
          for (int j = 0; j < 8; ++j)
            yy[ib32 * 32 + l * 8 + j] =
                dl * (float)gb[j] * ((sg & kmask[j]) ? -1.f : 1.f);
        }
      }
    }
  };

  std::mt19937 rng(9797);
  std::uniform_int_distribution<int> byte_dist(0, 255);
  for (int elements : {256, 512, 1024}) {
    int nb = elements / 256;
    std::vector<uint8_t> bytes((size_t)nb * 82);
    for (auto &b : bytes)
      b = (uint8_t)byte_dist(rng);
    std::vector<float> y_oracle(elements), y_cand(elements);
    oracle_iq2_s(bytes.data(), y_oracle.data(), elements);
    candidate_iq2_s(bytes.data(), y_cand.data(), elements);
    for (int i = 0; i < elements; ++i) {
      assert(std::fabs(y_oracle[i] - y_cand[i]) < 1e-5f &&
             "IQ2_S logic mismatch");
    }
  }

  std::cout << " PASSED\n";
}

// Verifies the EXPERIMENTAL 16-wide dequant functions' pairing logic
// (src/kernels_cpu.cpp's dequantize_row_iq{2_xxs,3_xxs,2_s}_avx512_wide):
// that batching two 8-element groups into one 16-wide op produces IDENTICAL
// output to doing them as two separate 8-element ops. Same methodology as
// the other dequant logic tests above (synthetic grid table, scalar
// oracle-vs-independently-written-candidate, can't cover the actual AVX-512
// intrinsics on this darwin machine) -- "narrow" here mirrors the ALREADY-
// tested 8-wide bookkeeping from the tests above; "wide" mirrors the NEW
// paired bookkeeping, scalar form. A mismatch here would mean the pairing
// itself (which l's go together, which db/signs apply to which half) is
// wrong, independent of whether the real AVX-512 intrinsics are correct.
static void test_iq2_wide_dequant_logic() {
  std::cout << "[TEST] IQ2_XXS/IQ3_XXS/IQ2_S 16-wide pairing logic (narrow vs "
               "wide, scalar form)..."
            << std::flush;

  uint64_t grid2[256], grid2s[1024];
  uint32_t grid3[256];
  for (int i = 0; i < 256; ++i) {
    uint64_t v = 0;
    for (int b = 0; b < 8; ++b)
      v |= (uint64_t)(uint8_t)((i * 7 + b * 13 + 3) & 0xFF) << (8 * b);
    grid2[i] = v;
    grid3[i] = (uint32_t)v;
  }
  for (int i = 0; i < 1024; ++i) {
    uint64_t v = 0;
    for (int b = 0; b < 8; ++b)
      v |= (uint64_t)(uint8_t)((i * 11 + b * 17 + 5) & 0xFF) << (8 * b);
    grid2s[i] = v;
  }
  uint8_t ksigns[128];
  for (int i = 0; i < 128; ++i)
    ksigns[i] = (uint8_t)(i * 31 + 7);
  uint8_t kmask[8] = {1, 2, 4, 8, 16, 32, 64, 128};

  auto apply8 = [&](uint64_t g, uint8_t signs, float db, float *out) {
    const uint8_t *gb = (const uint8_t *)&g;
    for (int j = 0; j < 8; ++j)
      out[j] = db * (float)gb[j] * ((signs & kmask[j]) ? -1.f : 1.f);
  };

  // ---- IQ2_XXS: narrow (4 separate 8-wide calls) vs wide (2 paired 16-wide
  // calls) ----
  auto narrow_iq2_xxs = [&](const uint8_t *block, float *y, int64_t k) {
    const int nb = (int)(k / 256);
    for (int i = 0; i < nb; ++i) {
      const uint8_t *base = block + (size_t)i * 66;
      uint16_t dh;
      std::memcpy(&dh, base, 2);
      float d = f16_bits_to_f32(dh);
      const uint8_t *qs_bytes = base + 2;
      float *yy = y + (size_t)i * 256;
      for (int ib32 = 0; ib32 < 8; ++ib32) {
        uint16_t q0, q1, q2, q3;
        std::memcpy(&q0, qs_bytes + (size_t)(4 * ib32 + 0) * 2, 2);
        std::memcpy(&q1, qs_bytes + (size_t)(4 * ib32 + 1) * 2, 2);
        std::memcpy(&q2, qs_bytes + (size_t)(4 * ib32 + 2) * 2, 2);
        std::memcpy(&q3, qs_bytes + (size_t)(4 * ib32 + 3) * 2, 2);
        uint32_t a0 = (uint32_t)q0 | ((uint32_t)q1 << 16),
                 a1 = (uint32_t)q2 | ((uint32_t)q3 << 16);
        uint8_t aux8[4];
        std::memcpy(aux8, &a0, 4);
        const float db = d * (0.5f + (float)(a1 >> 28)) * 0.25f;
        for (int l = 0; l < 4; ++l) {
          uint8_t signs = ksigns[(a1 >> (7 * l)) & 127];
          apply8(grid2[aux8[l]], signs, db, yy + ib32 * 32 + l * 8);
        }
      }
    }
  };
  auto wide_iq2_xxs = [&](const uint8_t *block, float *y, int64_t k) {
    const int nb = (int)(k / 256);
    for (int i = 0; i < nb; ++i) {
      const uint8_t *base = block + (size_t)i * 66;
      uint16_t dh;
      std::memcpy(&dh, base, 2);
      float d = f16_bits_to_f32(dh);
      const uint8_t *qs_bytes = base + 2;
      float *yy = y + (size_t)i * 256;
      for (int ib32 = 0; ib32 < 8; ++ib32) {
        uint16_t q0, q1, q2, q3;
        std::memcpy(&q0, qs_bytes + (size_t)(4 * ib32 + 0) * 2, 2);
        std::memcpy(&q1, qs_bytes + (size_t)(4 * ib32 + 1) * 2, 2);
        std::memcpy(&q2, qs_bytes + (size_t)(4 * ib32 + 2) * 2, 2);
        std::memcpy(&q3, qs_bytes + (size_t)(4 * ib32 + 3) * 2, 2);
        uint32_t a0 = (uint32_t)q0 | ((uint32_t)q1 << 16),
                 a1 = (uint32_t)q2 | ((uint32_t)q3 << 16);
        uint8_t aux8[4];
        std::memcpy(aux8, &a0, 4);
        const float db = d * (0.5f + (float)(a1 >> 28)) * 0.25f;
        for (int pair = 0; pair < 2; ++pair) {
          int l0 = pair * 2, l1 = pair * 2 + 1;
          uint8_t s0 = ksigns[(a1 >> (7 * l0)) & 127],
                  s1 = ksigns[(a1 >> (7 * l1)) & 127];
          apply8(grid2[aux8[l0]], s0, db, yy + ib32 * 32 + l0 * 8);
          apply8(grid2[aux8[l1]], s1, db, yy + ib32 * 32 + l1 * 8);
        }
      }
    }
  };

  // ---- IQ3_XXS ----
  auto narrow_iq3_xxs = [&](const uint8_t *block, float *y, int64_t k) {
    const int nb = (int)(k / 256);
    for (int i = 0; i < nb; ++i) {
      const uint8_t *base = block + (size_t)i * 98;
      uint16_t dh;
      std::memcpy(&dh, base, 2);
      float d = f16_bits_to_f32(dh);
      const uint8_t *qs = base + 2;
      const uint8_t *sas0 = qs + 64;
      float *yy = y + (size_t)i * 256;
      const uint8_t *qptr = qs;
      for (int ib32 = 0; ib32 < 8; ++ib32) {
        const uint8_t *sas = sas0 + 4 * ib32;
        uint32_t aux32 = (uint32_t)sas[0] | ((uint32_t)sas[1] << 8) |
                         ((uint32_t)sas[2] << 16) | ((uint32_t)sas[3] << 24);
        const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
        for (int l = 0; l < 4; ++l) {
          uint8_t signs = ksigns[(aux32 >> (7 * l)) & 127];
          uint8_t comb[8];
          uint32_t g1 = grid3[qptr[2 * l + 0]], g2 = grid3[qptr[2 * l + 1]];
          std::memcpy(comb, &g1, 4);
          std::memcpy(comb + 4, &g2, 4);
          uint64_t g;
          std::memcpy(&g, comb, 8);
          apply8(g, signs, db, yy + ib32 * 32 + l * 8);
        }
        qptr += 8;
      }
    }
  };
  auto wide_iq3_xxs = [&](const uint8_t *block, float *y, int64_t k) {
    const int nb = (int)(k / 256);
    for (int i = 0; i < nb; ++i) {
      const uint8_t *base = block + (size_t)i * 98;
      uint16_t dh;
      std::memcpy(&dh, base, 2);
      float d = f16_bits_to_f32(dh);
      const uint8_t *qs = base + 2;
      const uint8_t *sas0 = qs + 64;
      float *yy = y + (size_t)i * 256;
      const uint8_t *qptr = qs;
      for (int ib32 = 0; ib32 < 8; ++ib32) {
        const uint8_t *sas = sas0 + 4 * ib32;
        uint32_t aux32 = (uint32_t)sas[0] | ((uint32_t)sas[1] << 8) |
                         ((uint32_t)sas[2] << 16) | ((uint32_t)sas[3] << 24);
        const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
        for (int pair = 0; pair < 2; ++pair) {
          int l0 = pair * 2, l1 = pair * 2 + 1;
          uint8_t s0 = ksigns[(aux32 >> (7 * l0)) & 127],
                  s1 = ksigns[(aux32 >> (7 * l1)) & 127];
          uint8_t comb0[8], comb1[8];
          uint32_t g0a = grid3[qptr[2 * l0 + 0]], g0b = grid3[qptr[2 * l0 + 1]];
          uint32_t g1a = grid3[qptr[2 * l1 + 0]], g1b = grid3[qptr[2 * l1 + 1]];
          std::memcpy(comb0, &g0a, 4);
          std::memcpy(comb0 + 4, &g0b, 4);
          std::memcpy(comb1, &g1a, 4);
          std::memcpy(comb1 + 4, &g1b, 4);
          uint64_t gg0, gg1;
          std::memcpy(&gg0, comb0, 8);
          std::memcpy(&gg1, comb1, 8);
          apply8(gg0, s0, db, yy + ib32 * 32 + l0 * 8);
          apply8(gg1, s1, db, yy + ib32 * 32 + l1 * 8);
        }
        qptr += 8;
      }
    }
  };

  // ---- IQ2_S ----
  auto narrow_iq2_s = [&](const uint8_t *block, float *y, int64_t k) {
    const int nb = (int)(k / 256);
    for (int i = 0; i < nb; ++i) {
      const uint8_t *base = block + (size_t)i * 82;
      uint16_t dh;
      std::memcpy(&dh, base, 2);
      float d = f16_bits_to_f32(dh);
      const uint8_t *qs = base + 2, *qh = base + 2 + 64,
                    *scales = base + 2 + 64 + 8, *signs = qs + 32;
      float *yy = y + (size_t)i * 256;
      for (int ib32 = 0; ib32 < 8; ++ib32) {
        const float db0 = d * (0.5f + (scales[ib32] & 0xf)) * 0.25f;
        const float db1 = d * (0.5f + (scales[ib32] >> 4)) * 0.25f;
        for (int l = 0; l < 4; ++l) {
          const float dl = (l / 2 == 0) ? db0 : db1;
          const uint32_t idx =
              qs[4 * ib32 + l] | (((uint32_t)qh[ib32] << (8 - 2 * l)) & 0x300);
          apply8(grid2s[idx], signs[4 * ib32 + l], dl, yy + ib32 * 32 + l * 8);
        }
      }
    }
  };
  auto wide_iq2_s = [&](const uint8_t *block, float *y, int64_t k) {
    const int nb = (int)(k / 256);
    for (int i = 0; i < nb; ++i) {
      const uint8_t *base = block + (size_t)i * 82;
      uint16_t dh;
      std::memcpy(&dh, base, 2);
      float d = f16_bits_to_f32(dh);
      const uint8_t *qs = base + 2, *qh = base + 2 + 64,
                    *scales = base + 2 + 64 + 8, *signs = qs + 32;
      float *yy = y + (size_t)i * 256;
      for (int ib32 = 0; ib32 < 8; ++ib32) {
        const float db0 = d * (0.5f + (scales[ib32] & 0xf)) * 0.25f;
        const float db1 = d * (0.5f + (scales[ib32] >> 4)) * 0.25f;
        for (int pair = 0; pair < 2; ++pair) {
          int l0 = pair * 2, l1 = pair * 2 + 1;
          const float dbp = (pair == 0) ? db0 : db1;
          const uint32_t idx0 = qs[4 * ib32 + l0] |
                                (((uint32_t)qh[ib32] << (8 - 2 * l0)) & 0x300);
          const uint32_t idx1 = qs[4 * ib32 + l1] |
                                (((uint32_t)qh[ib32] << (8 - 2 * l1)) & 0x300);
          apply8(grid2s[idx0], signs[4 * ib32 + l0], dbp,
                 yy + ib32 * 32 + l0 * 8);
          apply8(grid2s[idx1], signs[4 * ib32 + l1], dbp,
                 yy + ib32 * 32 + l1 * 8);
        }
      }
    }
  };

  std::mt19937 rng(31337);
  std::uniform_int_distribution<int> byte_dist(0, 255);
  for (int elements : {256, 512, 1024}) {
    int nb = elements / 256;
    std::vector<float> y_n(elements), y_w(elements);

    std::vector<uint8_t> b2((size_t)nb * 66);
    for (auto &b : b2)
      b = (uint8_t)byte_dist(rng);
    narrow_iq2_xxs(b2.data(), y_n.data(), elements);
    wide_iq2_xxs(b2.data(), y_w.data(), elements);
    for (int i = 0; i < elements; ++i)
      assert(y_n[i] == y_w[i] && "IQ2_XXS wide pairing mismatch");

    std::vector<uint8_t> b3((size_t)nb * 98);
    for (auto &b : b3)
      b = (uint8_t)byte_dist(rng);
    narrow_iq3_xxs(b3.data(), y_n.data(), elements);
    wide_iq3_xxs(b3.data(), y_w.data(), elements);
    for (int i = 0; i < elements; ++i)
      assert(y_n[i] == y_w[i] && "IQ3_XXS wide pairing mismatch");

    std::vector<uint8_t> b2s((size_t)nb * 82);
    for (auto &b : b2s)
      b = (uint8_t)byte_dist(rng);
    narrow_iq2_s(b2s.data(), y_n.data(), elements);
    wide_iq2_s(b2s.data(), y_w.data(), elements);
    for (int i = 0; i < elements; ++i)
      assert(y_n[i] == y_w[i] && "IQ2_S wide pairing mismatch");
  }

  std::cout << " PASSED\n";
}

// ============================================================
// Test 8: fp32_to_bf16 / bf16_to_fp32 round-trip (kquant_dequant.hpp)
// ============================================================
// Minimal inline copy (kquant_dequant.hpp itself pulls in <cuda_runtime.h>,
// unavailable on this darwin dev machine -- same "re-implement the logic
// under test inline" pattern as every other test in this file). Added as
// part of a careful, test-gated retry of the attn_output BF16 fast-path
// after a real run produced NaN corruption on real hardware (see
// CLAUDE.md's incident writeup) -- the actual GPU-side suspect there was
// cublasGemmEx's mismatched-type dispatch (covered by src/main_test.cpp's
// Test 9, which needs real CUDA), but this rounding math is the one piece
// of that feature that's fully portable and verifiable right here, so no
// reason to leave it merely "reviewed, not run" when it can genuinely be
// run.
static inline float test_bf16_to_fp32(uint16_t b) {
  uint32_t u = ((uint32_t)b) << 16;
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}
static inline uint16_t test_fp32_to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  if ((u & 0x7fffffffu) > 0x7f800000u)
    return (uint16_t)((u >> 16) | 0x0040u); // NaN: keep quiet
  uint32_t rounding_bias = 0x00007fffu + ((u >> 16) & 1u);
  return (uint16_t)((u + rounding_bias) >> 16);
}

void test_bf16_roundtrip_logic() {
  std::cout << "[TEST] fp32_to_bf16/bf16_to_fp32 round-trip logic..."
            << std::flush;

  // Exact-in-bf16 values (powers of two, zero) must round-trip bit-exact.
  for (float exact : {0.0f, 1.0f, -1.0f, 2.0f, 0.5f, -0.5f, 128.0f, -128.0f}) {
    uint16_t b = test_fp32_to_bf16(exact);
    float back = test_bf16_to_fp32(b);
    assert(back == exact &&
           "exact power-of-two value did not round-trip bit-exact");
  }
  // -0.0f: sign bit must survive (distinguishable from +0.0f via memcmp,
  // even though -0.0f == 0.0f numerically).
  {
    uint16_t b = test_fp32_to_bf16(-0.0f);
    float back = test_bf16_to_fp32(b);
    uint32_t back_bits;
    std::memcpy(&back_bits, &back, sizeof(back_bits));
    assert(back_bits == 0x80000000u && "-0.0f lost its sign bit");
  }

  // Idempotence: re-encoding an already-bf16-representable value (i.e.
  // one that came FROM a bf16_to_fp32 call) must reproduce the exact same
  // bf16 bits -- this is the real correctness property upload_dequantized
  // needs (only ever encodes fresh f32 values, but if the pair weren't
  // mutually consistent, a real dequant bug could hide behind it).
  std::mt19937 rng(2024);
  std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
  for (int i = 0; i < 100000; ++i) {
    float f = dist(rng);
    uint16_t b1 = test_fp32_to_bf16(f);
    float back = test_bf16_to_fp32(b1);
    uint16_t b2 = test_fp32_to_bf16(back);
    assert(
        b1 == b2 &&
        "re-encoding a bf16-rounded value changed its bits (not idempotent)");

    // Rounding error bound: bf16 has 8 bits of significant precision
    // (1 implicit + 7 explicit mantissa), so relative error must be
    // within 2^-8 (one full mantissa ULP), generously bounded here.
    if (f != 0.0f) {
      float rel_err = std::abs((double)back - (double)f) / std::abs((double)f);
      assert(rel_err < (1.0f / 256.0f) &&
             "bf16 rounding error exceeded one mantissa ULP");
    }
  }

  // NaN in -> NaN out (quiet), never silently becomes a finite garbage value.
  {
    float nan_in = std::numeric_limits<float>::quiet_NaN();
    uint16_t b = test_fp32_to_bf16(nan_in);
    float back = test_bf16_to_fp32(b);
    assert(std::isnan(back) && "NaN input did not round-trip to NaN");
  }

  // Round-half-to-even: a value exactly halfway between two representable
  // bf16 values must round to the EVEN one, not always up or always down.
  // 1.0f's bf16 mantissa is all-zero (even); construct a value whose low
  // 16 bits are exactly 0x8000 (the halfway point) relative to 1.0f and
  // confirm it rounds back down to 1.0f (already-even), not up.
  {
    uint32_t half_bits =
        0x3f800000u |
        0x8000u; // 1.0f with the low 16 bits set to the halfway pattern
    float half;
    std::memcpy(&half, &half_bits, sizeof(half));
    uint16_t b = test_fp32_to_bf16(half);
    assert(b == 0x3f80u && "round-half-to-even did not round a halfway value "
                           "to the even mantissa");
  }

  std::cout << " PASSED\n";
}

// ============================================================
// Test: GpuTopology::expert_index (O(1) find_expert) logic
// ============================================================
// include/gpu_topology.hpp's GpuTopology can't be instantiated on this
// darwin dev machine at all (deeply CUDA-entangled: cudaEvent_t members,
// cudaSetDevice/cudaEventQuery calls) -- so this is a minimal, CUDA-free
// reimplementation of exactly the identity-resolution logic under test
// (the expert_index map + find_expert's index-hit/validate/fallback/
// self-heal path + evict_lru's index-erase-on-real-eviction + upload's
// index-insert), mirroring the real code's structure/invariants but with
// every cudaXxx call and GPU-specific field stripped out. This is the
// same "portable oracle for the CUDA-adjacent logic" pattern already
// used by test_q2k_q3k_dequant_logic / test_iq2xxs_iq3xxs_dequant_logic
// / test_bf16_roundtrip_logic elsewhere in this file.
struct TestExpertSlot {
  int layer_idx = -1;
  int expert_id = -1;
  bool has_device_ptr =
      false; // stands in for the real device_ptr != nullptr check
};
struct TestExpertVramCacheSim {
  std::vector<TestExpertSlot> slots;
  std::unordered_map<uint64_t, size_t> expert_index;
  int stale_index_hits = 0; // mirrors perf().n_stale_expert_index_hits

  static uint64_t key(int layer, int expert_id) {
    return ((uint64_t)(uint32_t)layer << 32) | (uint32_t)expert_id;
  }

  // Mirrors GpuTopology::find_expert's index-first, validate-or-fallback
  // structure exactly (minus the upload_ready/cudaEventQuery piece,
  // which is orthogonal to identity resolution).
  int find_expert(int layer, int expert_id) {
    uint64_t k = key(layer, expert_id);
    auto it = expert_index.find(k);
    if (it != expert_index.end()) {
      size_t idx = it->second;
      if (idx < slots.size()) {
        TestExpertSlot &s = slots[idx];
        if (s.layer_idx == layer && s.expert_id == expert_id &&
            s.has_device_ptr) {
          return (int)idx; // validated index hit
        }
      }
      // Stale entry -- log-and-fallback, same as the real code.
      stale_index_hits++;
      expert_index.erase(it);
    }
    for (size_t i = 0; i < slots.size(); ++i) {
      TestExpertSlot &s = slots[i];
      if (s.layer_idx == layer && s.expert_id == expert_id &&
          s.has_device_ptr) {
        expert_index[k] = i; // opportunistic self-heal
        return (int)i;
      }
    }
    return -1;
  }

  // Mirrors evict_lru's index-erase-on-real-eviction: only erase when a
  // slot with a REAL prior identity is being reclaimed, never for an
  // already-empty slot (mirrors the real code's `if (victim->layer_idx
  // != -1)` guard around both the swap-out stat and the index erase).
  void evict(size_t idx) {
    TestExpertSlot &s = slots[idx];
    if (s.layer_idx != -1) {
      expert_index.erase(key(s.layer_idx, s.expert_id));
    }
    s.layer_idx = -1;
    s.expert_id = -1;
    s.has_device_ptr = false;
  }

  // Mirrors upload_expert's index-insert-on-assign.
  void upload(size_t idx, int layer, int expert_id) {
    slots[idx].layer_idx = layer;
    slots[idx].expert_id = expert_id;
    slots[idx].has_device_ptr = true;
    expert_index[key(layer, expert_id)] = idx;
  }
};

void test_expert_vram_index_logic() {
  std::cout << "[TEST] GpuTopology expert_index (O(1) find_expert) logic..."
            << std::flush;

  TestExpertVramCacheSim sim;
  sim.slots.resize(4);

  // 1. Fresh insert is found by index lookup, at the right slot.
  sim.upload(0, /*layer=*/3, /*expert_id=*/7);
  sim.upload(1, /*layer=*/5, /*expert_id=*/12);
  assert(sim.find_expert(3, 7) == 0);
  assert(sim.find_expert(5, 12) == 1);
  assert(sim.stale_index_hits == 0);

  // 2. A never-inserted identity is a clean miss, not a false hit.
  assert(sim.find_expert(3, 8) == -1);
  assert(sim.find_expert(99, 99) == -1);

  // 3. Eviction removes the OLD identity's index entry -- a lookup for
  // the evicted (layer, expert_id) must miss cleanly, not resolve to
  // whatever the slot holds next (this is the exact bug class the
  // real evict_lru fix targets: a stale index pointing a slot's index
  // at data that's no longer there).
  sim.evict(0);
  assert(sim.find_expert(3, 7) == -1 &&
         "evicted identity must not still resolve via the index");
  assert(sim.stale_index_hits == 0 &&
         "a clean index miss after erase should not count as stale");

  // 4. Reusing slot 0 for a DIFFERENT expert must resolve the NEW
  // identity correctly, and must never accidentally still answer for
  // the OLD identity that used to live there (the core correctness
  // property: same slot, different expert over time, index always
  // reflects only the CURRENT occupant).
  sim.upload(0, /*layer=*/9, /*expert_id=*/1);
  assert(sim.find_expert(9, 1) == 0);
  assert(sim.find_expert(3, 7) == -1);
  assert(sim.find_expert(5, 12) == 1 &&
         "unrelated slot 1 must be unaffected by slot 0 churn");

  // 5. Safety net: simulate the index/array desyncing (something this
  // should never happen by construction in the real code, but the
  // fallback path exists specifically to make even a maintenance-site
  // bug non-catastrophic) -- point the index at a slot whose real
  // contents don't match, and confirm find_expert detects the mismatch
  // (logs it as stale) and still finds the CORRECT answer via the
  // linear-scan fallback rather than returning wrong data or nothing.
  sim.upload(2, /*layer=*/20, /*expert_id=*/4);
  sim.expert_index[TestExpertVramCacheSim::key(20, 4)] =
      1; // deliberately wrong index
  int before = sim.stale_index_hits;
  int result = sim.find_expert(20, 4);
  assert(result == 2 && "stale index entry must fall back to the linear scan "
                        "and find the REAL slot");
  assert(sim.stale_index_hits == before + 1 &&
         "a validated mismatch must increment the stale-hit counter");
  // Self-healed: a second lookup for the same identity should now hit
  // the index directly again (no additional stale count).
  int before2 = sim.stale_index_hits;
  assert(sim.find_expert(20, 4) == 2);
  assert(sim.stale_index_hits == before2 &&
         "a self-healed index entry must not re-trigger a stale hit");

  // 6. An empty-slot claim (no prior identity) must NOT erase any
  // index entry -- mirrors evict_lru's `if (victim->layer_idx != -1)`
  // guard. Evicting an already-empty slot 3 must leave every other
  // slot's index entries untouched.
  assert(sim.slots[3].layer_idx == -1);
  sim.evict(3); // no-op identity-wise, must not touch expert_index at all
  assert(sim.find_expert(9, 1) == 0);
  assert(sim.find_expert(5, 12) == 1);
  assert(sim.find_expert(20, 4) == 2);

  std::cout << " PASSED\n";
}

// ============================================================
// Test: direct_io::compute_aligned_window (O_DIRECT alignment arithmetic)
// ============================================================
// This is the REAL function used by include/direct_io.hpp's
// pread_direct_aware -- unlike the O_DIRECT syscall behavior itself
// (which this darwin dev machine cannot exercise at all, and which
// tests/main_test.cpp's own CUDA-dependent tests can't reach either
// since O_DIRECT isn't GPU-related), this is pure, dependency-free
// arithmetic, so it's tested directly rather than via a reimplemented
// stand-in -- there's no "portable oracle" gap to bridge here the way
// there is for CUDA-entangled code elsewhere in this file.
void test_direct_io_aligned_window_logic() {
  std::cout << "[TEST] direct_io::compute_aligned_window arithmetic..."
            << std::flush;

  auto check_invariants = [](uint64_t offset, uint64_t len,
                             uint64_t alignment) {
    auto w = direct_io::compute_aligned_window(offset, len, alignment);
    assert(w.aligned_offset <= offset &&
           "aligned_offset must not start after the real offset");
    assert(w.aligned_offset % alignment == 0 &&
           "aligned_offset must be alignment-aligned");
    assert(w.aligned_len % alignment == 0 &&
           "aligned_len must be a multiple of alignment");
    assert(w.aligned_offset + w.aligned_len >= offset + len &&
           "aligned window must fully cover [offset, offset+len)");
    assert(w.front_slack == offset - w.aligned_offset &&
           "front_slack must equal offset - aligned_offset");
    assert(w.front_slack < alignment &&
           "front_slack must be smaller than one alignment unit");
    // The window shouldn't overshoot by more than one alignment unit on
    // either side beyond what's strictly required to cover the range --
    // catches a rounding-direction bug that would silently over-read by
    // a large, wrong amount instead of the expected "at most alignment-1
    // bytes of slack on each end".
    assert(offset - w.aligned_offset < alignment);
    assert((w.aligned_offset + w.aligned_len) - (offset + len) < alignment);
  };

  // Already-aligned offset, already-aligned length: no padding needed at all.
  {
    auto w = direct_io::compute_aligned_window(4096, 8192, 4096);
    assert(w.aligned_offset == 4096 && w.aligned_len == 8192 &&
           w.front_slack == 0);
  }
  // Already-aligned offset, length NOT a multiple of alignment: only the
  // tail needs padding.
  {
    auto w = direct_io::compute_aligned_window(4096, 100, 4096);
    assert(w.aligned_offset == 4096 && w.front_slack == 0);
    assert(w.aligned_len == 4096 &&
           "100 bytes should round up to exactly one alignment unit");
  }
  // Offset 0 edge case.
  {
    auto w = direct_io::compute_aligned_window(0, 1, 4096);
    assert(w.aligned_offset == 0 && w.front_slack == 0 &&
           w.aligned_len == 4096);
  }
  // Misaligned offset (the "general case" this project's real GGUF
  // tensors should never hit after gguf_tool pack --align 4096, but the
  // code must still handle correctly rather than assume away).
  {
    auto w = direct_io::compute_aligned_window(4200, 1000, 4096);
    assert(w.aligned_offset == 4096 && w.front_slack == 104);
    // real range ends at 5200 -> next 4096 boundary is 8192
    assert(w.aligned_len == 8192 - 4096);
  }
  // A real per-expert-sized read at a realistic large file offset
  // (tens of GB in, matching the scale this feature exists for).
  {
    uint64_t offset =
        37ULL * 1000 * 1000 * 1000 + 123; // ~37GB in, deliberately unaligned
    uint64_t len =
        3 * 1024 * 1024 + 777; // ~3MB expert slab, deliberately not block-sized
    check_invariants(offset, len, 4096);
  }
  // Exhaustive-ish sweep across small offsets/lengths (covers every
  // possible front_slack value for this alignment) plus a randomized
  // sweep across the full 64-bit-realistic range.
  for (uint64_t off = 0; off < 4096 * 3; off += 37) {
    for (uint64_t len : {1ULL, 4095ULL, 4096ULL, 4097ULL, 9000ULL}) {
      check_invariants(off, len, 4096);
    }
  }
  std::mt19937_64 rng(99);
  std::uniform_int_distribution<uint64_t> big_offset(0, 200ULL * 1000 * 1000 *
                                                            1000);
  std::uniform_int_distribution<uint64_t> real_len(1, 8ULL * 1024 * 1024);
  for (int i = 0; i < 20000; ++i) {
    check_invariants(big_offset(rng), real_len(rng), 4096);
  }
  // Alignment values other than 4096 (e.g. a filesystem with a smaller
  // logical block size) must also hold.
  for (uint64_t alignment : {512ULL, 1024ULL, 4096ULL}) {
    for (int i = 0; i < 2000; ++i) {
      check_invariants(big_offset(rng), real_len(rng), alignment);
    }
  }

  std::cout << " PASSED\n";
}

// ============================================================
// Tests 15-19: CPU kernel backend (platform-agnostic f32 paths)
// ============================================================
void test_cpu_matmul_f32() {
  std::cout << "[TEST] CPU f32 matmul..." << std::flush;
  // x[2,3] @ W^T[2,3] -> y[2,2]
  const float x[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  const float W[6] = {7.0f, 8.0f, 9.0f, 1.0f, 2.0f, 3.0f};
  float y[4] = {0};
  matmul(y, x, W, 2, 3, 2);
  // row 0: dot([1,2,3], [7,8,9])=50, dot([1,2,3], [1,2,3])=14
  // row 1: dot([4,5,6], [7,8,9])=122, dot([4,5,6], [1,2,3])=32
  assert(std::fabs(y[0] - 50.0f) < 1e-4f);
  assert(std::fabs(y[1] - 14.0f) < 1e-4f);
  assert(std::fabs(y[2] - 122.0f) < 1e-4f);
  assert(std::fabs(y[3] - 32.0f) < 1e-4f);
  std::cout << " PASSED\n";
}

void test_cpu_matmul_f32_large() {
  std::cout << "[TEST] CPU f32 matmul (large, exercises Metal if enabled)..."
            << std::flush;
  // Size above the 256 KFLOP Metal threshold so a Metal-enabled build
  // actually exercises the MPS path.  On non-Metal builds this just runs
  // the NEON/scalar fallback.
  const int S = 64, I = 256, O = 256;
  std::vector<float> x(static_cast<size_t>(S) * I);
  std::vector<float> W(static_cast<size_t>(O) * I);
  std::vector<float> y(static_cast<size_t>(S) * O, 0.0f);
  for (size_t i = 0; i < x.size(); ++i)
    x[i] = static_cast<float>((i * 17 + 3) % 100) * 0.01f;
  for (size_t i = 0; i < W.size(); ++i)
    W[i] = static_cast<float>((i * 31 + 7) % 100) * 0.01f;

  matmul(y.data(), x.data(), W.data(), S, I, O);

  // Reference computation using the same convention as matmul:
  // y[s,o] = sum_i x[s,i] * W[o,i]
  for (int s = 0; s < S; ++s) {
    for (int o = 0; o < O; ++o) {
      double acc = 0.0;
      for (int i = 0; i < I; ++i)
        acc += static_cast<double>(x[s * I + i]) * W[o * I + i];
      assert(std::fabs(y[s * O + o] - static_cast<float>(acc)) < 1e-3f);
    }
  }
  std::cout << " PASSED\n";
}

void test_cpu_dot_f32() {
  std::cout << "[TEST] CPU f32 dot product..." << std::flush;
  const float a[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  const float b[8] = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
  float r = dot_f32_avx512(a, b, 8);
  assert(std::fabs(r - 120.0f) < 1e-4f);
  std::cout << " PASSED\n";
}

void test_cpu_rmsnorm() {
  std::cout << "[TEST] CPU RMSNorm..." << std::flush;
  const float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  const float w[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  float out[4];
  rmsnorm_row(out, x, w, 4, 1e-5f);
  float mean_sq = (1.0f + 4.0f + 9.0f + 16.0f) / 4.0f;
  float r = 1.0f / std::sqrt(mean_sq + 1e-5f);
  for (int i = 0; i < 4; ++i) {
    assert(std::fabs(out[i] - x[i] * r) < 1e-4f);
  }
  std::cout << " PASSED\n";
}

void test_cpu_silu() {
  std::cout << "[TEST] CPU SiLU..." << std::flush;
  const float in[5] = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f};
  float x[5];
  std::memcpy(x, in, sizeof(x));
  silu_inplace(x, 5);
  auto expected = [](float v) { return v / (1.0f + std::exp(-v)); };
  for (int i = 0; i < 5; ++i) {
    assert(std::fabs(x[i] - expected(in[i])) < 1e-4f);
  }
  std::cout << " PASSED\n";
}

void test_cpu_elem_mul() {
  std::cout << "[TEST] CPU element-wise multiply..." << std::flush;
  const float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  const float b[4] = {5.0f, 6.0f, 7.0f, 8.0f};
  float out[4];
  elem_mul(out, a, b, 4);
  assert(std::fabs(out[0] - 5.0f) < 1e-4f);
  assert(std::fabs(out[1] - 12.0f) < 1e-4f);
  assert(std::fabs(out[2] - 21.0f) < 1e-4f);
  assert(std::fabs(out[3] - 32.0f) < 1e-4f);
  std::cout << " PASSED\n";
}

// ============================================================
// Test: RouteUsage histogram (CUDA-free pin-file component)
// ============================================================
void test_route_usage() {
  std::cout << "[TEST] RouteUsage histogram load/bump/dump/pin_set..."
            << std::flush;

  RouteUsage usage;
  usage.bump(0, 5);
  usage.bump(0, 5);
  usage.bump(0, 7);
  usage.bump(1, 3);
  usage.bump(1, 3);
  usage.bump(1, 3);
  usage.bump(1, 9);

  // topk_for_layer returns experts sorted by count desc.
  auto top0 = usage.topk_for_layer(0, 2);
  assert(top0.size() == 2u);
  assert(top0[0] == 5);
  assert(top0[1] == 7);

  auto top1 = usage.topk_for_layer(1, 1);
  assert(top1.size() == 1u);
  assert(top1[0] == 3);

  // pin_set: threshold=2, topk=1 -> layer0 expert5, layer1 expert3.
  auto pins = usage.pin_set(1, 2);
  assert(pins.size() == 2u);
  assert(pins.count((0ull << 32) | 5u) == 1u);
  assert(pins.count((1ull << 32) | 3u) == 1u);

  // Dump/load round-trip with merge.
  std::string tmp = "/tmp/moestream_test_usage.txt";
  assert(usage.dump(tmp));

  RouteUsage loaded;
  assert(loaded.load(tmp));
  loaded.bump(0, 5);     // merge with existing count
  loaded.bump(2, 1, 10); // merge explicit delta

  auto top0_reload = loaded.topk_for_layer(0, 1);
  assert(top0_reload.size() == 1u);
  assert(top0_reload[0] == 5); // 2+1 = 3, still top

  auto top2 = loaded.topk_for_layer(2, 1);
  assert(top2.size() == 1u);
  assert(top2[0] == 1);

  std::remove(tmp.c_str());
  std::cout << " PASSED\n";
}

// Optional substring filtering via argv, so a subset can be exercised without
// recompiling -- the workflow this file exists for: run the CPU-only
// component tests on a box where the full CUDA harness (main_test.cpp) can't
// build at all (e.g. macOS). A test runs iff its name contains ANY argv token
// (no match logic, just std::string::find); no args = run everything (the
// original behavior, so `make test` / CI that pass no args are unaffected).
//
//   ./test_components               -> all component tests
//   ./test_components rope          -> just test_rope
//   ./test_components dequant       -> every *dequant* test
//   ./test_components rope dequant  -> rope OR any dequant test
int main(int argc, char **argv) {
  std::cout << "=== MoEStream Component Tests ===\n";
  struct TestEntry {
    const char *name;
    void (*fn)();
  };
  const TestEntry tests[] = {
      {"reasoning_parser", test_reasoning_parser},
      {"rope", test_rope},
      {"topk_selection", test_topk_selection},
      {"softmax_routing", test_softmax_routing},
      {"calculate_size", test_calculate_size},
      {"spsc_ring", test_spsc_ring},
      {"toml_parser", test_toml_parser},
      {"q2k_q3k_dequant", test_q2k_q3k_dequant_logic},
      {"iq2xxs_iq3xxs_dequant", test_iq2xxs_iq3xxs_dequant_logic},
      {"iq2s_dequant", test_iq2s_dequant_logic},
      {"iq2_wide", test_iq2_wide_dequant_logic},
      {"bf16_roundtrip", test_bf16_roundtrip_logic},
      {"expert_vram_index", test_expert_vram_index_logic},
      {"direct_io_aligned_window", test_direct_io_aligned_window_logic},
      {"cpu_matmul_f32", test_cpu_matmul_f32},
      {"cpu_matmul_f32_large", test_cpu_matmul_f32_large},
      {"cpu_dot_f32", test_cpu_dot_f32},
      {"cpu_rmsnorm", test_cpu_rmsnorm},
      {"cpu_silu", test_cpu_silu},
      {"cpu_elem_mul", test_cpu_elem_mul},
      {"route_usage", test_route_usage},
  };
  const bool filtered = argc > 1;
  auto matches = [&](const char *name) {
    if (!filtered)
      return true;
    std::string n(name);
    for (int i = 1; i < argc; ++i)
      if (n.find(argv[i]) != std::string::npos)
        return true;
    return false;
  };
  int run = 0;
  for (const auto &t : tests) {
    if (!matches(t.name))
      continue;
    if (filtered)
      std::cout << "[run] " << t.name << "\n";
    t.fn();
    ++run;
  }
  if (run == 0) {
    std::cout << "No tests matched filter:";
    for (int i = 1; i < argc; ++i)
      std::cout << " " << argv[i];
    std::cout << "\n";
    return 2;
  }
  std::cout << "\n"
            << run << " test(s) " << (filtered ? "ran (filtered), all" : "")
            << " PASSED.\n";
  return 0;
}
