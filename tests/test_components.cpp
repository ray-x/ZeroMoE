// tests/test_components.cpp — Unit tests for ZeroMoE components
// Build: g++ -std=c++23 -O2 -I../include -o test_components test_components.cpp
// (Does NOT require CUDA for the CPU-path tests)
//
// The default Makefile build type is Release, which defines NDEBUG and turns
// every assert into a no-op. That is wrong for a test binary: several checks
// (and two side effects, symlink()/unlink(), written inside assert) would be
// skipped or would never happen, so a Release test run aborts in
// exl3_ngram_reader and silently "passes" the numeric checks. Keep assertions
// live regardless of the build type.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "direct_io.hpp" // the REAL O_DIRECT alignment-window arithmetic (no CUDA deps)
#include "gguf_loader.hpp" // DeepSeek-V4 compression metadata validation helper
#include "gguf_tokenizer.hpp"
#include "dsv4_compressed_attention.hpp"
#include "kernels_cpu.hpp" // CPU kernel backend (AVX-512 or NEON, selected by build script)
#include "rope.hpp" // the REAL, unit-tested RoPE (no CUDA / no AVX-512 deps)
#include "hyper_connections.hpp" // the REAL Sinkhorn normalization (DeepSeek-V4)
#include "route_usage.hpp" // CUDA-free routing histogram used by CPU and GPU paths
#include "safetensors_adapter.hpp" // DFlash/DSpark safetensors adapter boundary
#include "exl3_cpu.hpp"
#include "exl3_model_loader.hpp"
#include "exllamav3_backend.hpp"
#include "cpu_config.hpp"
#include "tensor_trace.hpp" // TraceFilter (pure C++, no CUDA deps -- see logger.hpp)
#include "qwen4exp_gdn.hpp" // the REAL Qwen4Exp GDN delta-rule step (no CUDA deps)
#include "ple_hash.hpp" // the REAL PLE n-gram hash (no CUDA deps)
#include "moe_decode_batch.hpp" // decode-gap prototype: grouped per-token expert dispatch (no CUDA deps)
#include "memory_guard.hpp" // deterministic host-budget admission arithmetic
#include "dsv41_router_selection.hpp"
#include "option_b_global_hot.hpp" // Option-B global-pool hot-set selection (header-only)
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <vector>
#include <unistd.h>

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
// Option-B global pool: pooled top-K across layers (models/qwen/option_b_global_hot.hpp)
// ============================================================
void test_option_b_global_pool() {
  std::cout << "[TEST] Option-B Global Pool Selection..." << std::flush;
  struct U { int layer; int expert; uint64_t count; };
  // 3 layers x 4 experts. Layer 0 is concentrated (one dominant expert),
  // layer 1 is flat, layer 2 is concentrated on two experts. Every layer
  // routes the same total (100) so counts are comparable across layers.
  std::vector<U> u = {
      {0, 0, 85}, {0, 1, 5}, {0, 2, 5}, {0, 3, 5},
      {1, 0, 25}, {1, 1, 25}, {1, 2, 25}, {1, 3, 25},
      {2, 0, 45}, {2, 1, 45}, {2, 2, 5}, {2, 3, 5}};
  std::vector<int> layers = {0, 1, 2};

  // per-layer would give each layer K/3; the pool follows the counts instead.
  auto r = option_b_global::select(u, layers, 3, 4, /*top_k=*/6,
                                   /*floor=*/1, /*ceiling=*/0);
  assert(r.total_hot == 6);
  // Top 6 of the pool: 85 (L0e0), 45,45 (L2e0,e1), then the 25s (L1) fill 3.
  assert(r.hot.at(0).size() == 1 && r.hot.at(0)[0] == 0);
  assert(r.hot.at(2).size() == 2);
  assert(r.hot.at(1).size() == 3);
  // Hot list is ordered by the layer's own descending usage, ties by id.
  assert(r.hot.at(2)[0] == 0 && r.hot.at(2)[1] == 1);
  assert(std::abs(r.layer_coverage.at(0) - 0.85) < 1e-9);
  assert(std::abs(r.pool_coverage - (85 + 90 + 75) / 300.0) < 1e-9);

  // Floor: with floor=2 every layer keeps its 2 hottest regardless of rank.
  auto f = option_b_global::select(u, layers, 3, 4, 6, 2, 0);
  assert(f.hot.at(0).size() >= 2 && f.hot.at(1).size() >= 2 &&
         f.hot.at(2).size() >= 2);
  assert(f.total_hot == 6);

  // Ceiling: no layer may exceed it even if it dominates the pool.
  auto c = option_b_global::select(u, layers, 3, 4, 6, 1, 1);
  for (const auto &kv : c.hot)
    assert(kv.second.size() <= 1);
  assert(c.total_hot == 3);

  // A floor larger than top_k / layers is clamped (never exceeds top_k).
  auto g = option_b_global::select(u, layers, 3, 4, 3, 4, 0);
  assert(g.total_hot == 3);

  // Layers without usage get no hot set; invalid/duplicate ids are ignored.
  std::vector<int> odd = {0, 0, 7, -1, 2, 9};
  auto o = option_b_global::select(u, odd, 3, 4, 4, 1, 0);
  assert(o.hot.count(1) == 0 && o.hot.count(7) == 0);

  // Deterministic for any entry order.
  auto rev = u;
  std::reverse(rev.begin(), rev.end());
  auto r2 = option_b_global::select(rev, layers, 3, 4, 6, 1, 0);
  assert(r2.hot == r.hot);

  // top_k beyond the pool keeps everything.
  auto all = option_b_global::select(u, layers, 3, 4, 1000, 1, 0);
  assert(all.total_hot == 12);

  // A budget below the layer count cannot satisfy the floor; coverage must
  // still include the traffic of layers that receive no resident experts.
  auto sparse = option_b_global::select(u, layers, 3, 4, 1, 1, 0);
  assert(sparse.total_hot == 1 && sparse.hot.size() == 1);
  assert(std::abs(sparse.pool_coverage - 85.0 / 300.0) < 1e-9);
  assert(sparse.layer_coverage.at(1) == 0.0);
  assert(sparse.layer_coverage.at(2) == 0.0);

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
    if (t == 26)
      return numel * 4; // I32 -- DeepSeek-V4 ffn_gate_tid2eid.weight (raw
                        // int32 token-id->expert-id lookup table); missing
                        // this case fell through to the numel-fallback
                        // (1 byte/element, 4x too small).
    if (t == 27)
      return numel * 8; // I64 -- defensive gap-fill alongside I32 (see
                        // gguf_loader.hpp's own comment); without this case
                        // an I64 tid2eid tensor would fall through to the
                        // numel-fallback (1 byte/element, 8x too small).
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
  // I32: 8 elements = 32 bytes (4 bytes/element, no block packing).
  assert(calc(8, 26) == 32);
  // I64: 8 elements = 64 bytes (8 bytes/element, no block packing).
  assert(calc(8, 27) == 64);

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
// test fails loudly. Calls the REAL rope_yarn_inplace from models/base/rope.hpp
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

// DeepSeek-V4-specific RoPE pieces: the dsv4_mscale attn-factor formula
// (rope_yarn_inplace's dsv4_mscale=true branch) and the inverse/"de-rope"
// function (rope_yarn_inplace_inverse). See rope.hpp's own comments for the
// derivation this test is checking against.
void test_dsv4_rope() {
  std::cout << "[TEST] DeepSeek-V4 RoPE (dsv4_mscale + inverse)..." << std::flush;

  // --- Case 1: dsv4_mscale formula, known-answer against the documented
  // closed form mscale = attn_factor / (1 + 0.1*log(1/freq_scale)), computed
  // independently here (not copy-pasted from rope.hpp's own expression) so a
  // regression in the header is actually caught.
  {
    const int dim = 8, pos = 5;
    const float theta = 10000.0f, freq_scale = 0.25f, attn_factor = 1.0f;
    float factor = 1.0f / freq_scale; // 4.0
    float expected_mscale = attn_factor / (1.0f + 0.1f * std::log(factor));

    // Isolate mscale by comparing dim=2 (one pair, j=0 -> ramp_mix=1 since
    // corr_start==corr_end==0 collapses y to a divide-by-0.001f giving
    // ramp=0 -- avoid that corner by using dim=8 and reading pair j=0's
    // magnitude, then dividing out the plain-RoPE (mscale=1) case's own
    // magnitude at the same position to isolate mscale from the YaRN ramp.
    float v_yarn[dim], v_plain[dim];
    for (int i = 0; i < dim; ++i) { v_yarn[i] = (i == 0) ? 1.0f : 0.0f; v_plain[i] = v_yarn[i]; }
    rope_yarn_inplace(v_yarn, dim, pos, theta, /*ext_factor=*/1.0f, freq_scale,
                      attn_factor, /*yarn_log_mul=*/0.0f, /*n_ctx_orig=*/2048,
                      /*dsv4_mscale=*/true);
    rope_yarn_inplace(v_plain, dim, pos, theta, /*ext_factor=*/0.0f, /*freq_scale=*/1.0f,
                      attn_factor, /*yarn_log_mul=*/0.0f, /*n_ctx_orig=*/2048,
                      /*dsv4_mscale=*/true); // ext_factor=0 -> dsv4_mscale branch not even reached
    // pair j=0 ALWAYS has ramp=1 (full extrapolation, t=pos exactly) --
    // provably, not just for this case: y=(0-corr_start)/(corr_end-
    // corr_start) with corr_start>=0 by construction (std::max(0.0f, ...))
    // means y<=0, which std::max(0.0f, y) clamps to 0, giving ramp=1-0=1
    // regardless of corr_start/corr_end's actual values. The plain-RoPE
    // comparison branch also has t=freq_scale*pos=pos at j=0 (freq_scale=1
    // there) -- both rotate by the IDENTICAL angle, so v_yarn's magnitude
    // at j=0 is exactly mscale * (plain rotation magnitude), letting the
    // two known unit
    // vectors' ratio isolate mscale directly.
    float mag_yarn = std::sqrt(v_yarn[0]*v_yarn[0] + v_yarn[1]*v_yarn[1]);
    float mag_plain = std::sqrt(v_plain[0]*v_plain[0] + v_plain[1]*v_plain[1]);
    float measured_mscale = mag_yarn / mag_plain;
    assert(std::fabs(measured_mscale - expected_mscale) < 1e-4f);
  }

  // --- Case 2: plain-RoPE (ext_factor=0, mscale=1) round trip is exact
  // identity -- forward then inverse recovers the original vector.
  {
    const int dim = 16, pos = 11;
    const float theta = 10000.0f;
    std::vector<float> v(dim), orig(dim);
    for (int i = 0; i < dim; ++i) v[i] = orig[i] = 0.01f * (float)(i + 1);
    rope_yarn_inplace(v.data(), dim, pos, theta, 0.0f, 1.0f, 1.0f, 0.0f, 0);
    rope_yarn_inplace_inverse(v.data(), dim, pos, theta, 0.0f, 1.0f, 1.0f, 0.0f, 0);
    for (int i = 0; i < dim; ++i)
      assert(std::fabs(v[i] - orig[i]) < 1e-4f);
  }

  // --- Case 3: YaRN-active (mscale != 1) round trip is NOT identity -- it's
  // mscale^2 * original, matching ggml_rope_ext_back's transpose semantics
  // (documented in rope_yarn_inplace_inverse's own comment). This is the
  // regression this test exists to catch: if a future edit "fixes" the
  // inverse to divide by mscale instead of reusing it, this assertion fails
  // loudly instead of silently changing de-rope's real behavior.
  {
    const int dim = 8, pos = 5;
    const float theta = 10000.0f, freq_scale = 0.25f, attn_factor = 1.0f;
    float factor = 1.0f / freq_scale;
    float mscale = attn_factor / (1.0f + 0.1f * std::log(factor));
    std::vector<float> v(dim), orig(dim);
    for (int i = 0; i < dim; ++i) v[i] = orig[i] = 0.01f * (float)(i + 1);
    rope_yarn_inplace(v.data(), dim, pos, theta, 1.0f, freq_scale, attn_factor,
                      0.0f, 2048, /*dsv4_mscale=*/true);
    rope_yarn_inplace_inverse(v.data(), dim, pos, theta, 1.0f, freq_scale, attn_factor,
                              0.0f, 2048, /*dsv4_mscale=*/true);
    for (int i = 0; i < dim; ++i) {
      float denom = std::max(1e-6f, std::fabs(orig[i]));
      assert(std::fabs(v[i] - mscale * mscale * orig[i]) < 1e-3f * denom);
    }
  }

  std::cout << " PASSED\n";
}

// Sinkhorn normalization (DeepSeek-V4 Hyper-Connections' comb-matrix
// mixing, core/hyper_connections.hpp). Pure math, no engine dependency
// -- see that header's own comment for how tentative the REST of the
// Hyper-Connections reconstruction is; this specific primitive (the
// iterative normalization procedure itself, independent of what feeds
// into it) has a hard mathematical property this test can check directly.
void test_sinkhorn_normalize() {
  std::cout << "[TEST] Sinkhorn comb-matrix normalization..." << std::flush;

  // --- Case 1: after enough iterations, every row AND every column of a
  // random (non-trivial) matrix sums to ~1 -- the defining property of a
  // doubly-stochastic matrix, and the point of Sinkhorn iteration.
  {
    const int hc = 4;
    float comb[hc * hc] = {
      2.1f, -0.5f, 0.3f, 1.8f,
      -1.2f, 0.9f, 2.4f, -0.1f,
      0.7f, 1.5f, -0.8f, 0.2f,
      -0.3f, 0.6f, 1.1f, -1.9f,
    };
    sinkhorn_normalize_comb(comb, hc, /*iters=*/20, /*eps=*/1e-6f);
    for (int dst = 0; dst < hc; ++dst) {
      float row_sum = 0.0f;
      for (int src = 0; src < hc; ++src) row_sum += comb[dst * hc + src];
      assert(std::fabs(row_sum - 1.0f) < 1e-3f);
    }
    for (int src = 0; src < hc; ++src) {
      float col_sum = 0.0f;
      for (int dst = 0; dst < hc; ++dst) col_sum += comb[dst * hc + src];
      assert(std::fabs(col_sum - 1.0f) < 1e-3f);
    }
    // Every entry must also be non-negative (a doubly-stochastic matrix's
    // defining second property, alongside row/col sums) -- catches a
    // normalization bug that happens to preserve sums while going negative.
    for (int i = 0; i < hc * hc; ++i) assert(comb[i] >= 0.0f);
  }

  // --- Case 2: iters=0 (and iters=1, identically -- the loop only runs for
  // iters>1) still runs ONE unconditional row-normalize after Step 1, since
  // that leading call is not gated on `iters` at all -- confirmed against
  // deepseek4.cpp's build_hc_sinkhorn, where the equivalent leading
  // norm_cols() call sits before the `for i=1;i<iters` loop with no
  // conditional guard. So EVERY ROW sums to ~1 even at iters=0; columns are
  // NOT expected to (only the row-normalize ran, not the alternating
  // column+row loop). This replaces an earlier version of this test that
  // asserted the opposite (iters=0 -> only columns sum to ~1) -- that
  // matched a prior, incorrect implementation which gated the leading
  // normalize on `iters`, found and fixed 2026-08-08 via direct comparison
  // against the real llama.cpp source.
  {
    const int hc = 3;
    float comb[hc * hc] = {
      1.0f, 0.0f, -1.0f,
      0.5f, 2.0f, 0.5f,
      -0.5f, 1.0f, 1.5f,
    };
    sinkhorn_normalize_comb(comb, hc, /*iters=*/0, /*eps=*/1e-6f);
    for (int dst = 0; dst < hc; ++dst) {
      float row_sum = 0.0f;
      for (int src = 0; src < hc; ++src) row_sum += comb[dst * hc + src];
      assert(std::fabs(row_sum - 1.0f) < 1e-3f);
    }
  }

  std::cout << " PASSED\n";
}

// DeepSeek-V4's sqrtsoftplus MoE gating (execution_layer.cpp's
// apply_gating_activation, GatingFunc::SqrtSoftplus case) -- that function
// lives in an engine/CUDA-dependent .cpp this Mac-only harness can't link
// against, so this test checks the MATH independently (an "independent
// oracle" against the real implementation, same pattern as test_rope's own
// ref_interleaved) rather than calling the real function directly.
void test_sqrtsoftplus_gating() {
  std::cout << "[TEST] sqrtsoftplus gating..." << std::flush;

  // Same numerically-stable form execution_layer.cpp uses: softplus(x) =
  // x + log1p(exp(-x)) for x>0 (exp(-x)->0, never overflows), else
  // log1p(exp(x)) directly (safe since x<=0 there).
  auto sqrt_softplus = [](float x) {
    float softplus = (x > 0.0f) ? (x + std::log1p(std::exp(-x)))
                                 : std::log1p(std::exp(x));
    return std::sqrt(softplus);
  };

  // --- Case 1: known answer at x=0 -- softplus(0)=ln(2), so
  // sqrt(softplus(0)) = sqrt(ln 2).
  {
    float expected = std::sqrt(std::log(2.0f));
    assert(std::fabs(sqrt_softplus(0.0f) - expected) < 1e-5f);
  }

  // --- Case 2: monotonically increasing (a defining property of a real
  // gating activation -- higher logit must never produce a lower gate).
  {
    float prev = sqrt_softplus(-10.0f);
    for (float x = -9.0f; x <= 10.0f; x += 0.5f) {
      float cur = sqrt_softplus(x);
      assert(cur >= prev - 1e-6f);
      prev = cur;
    }
  }

  // --- Case 3: no overflow/NaN at large-magnitude logits in either
  // direction -- the exact failure mode a naive log1p(exp(x)) hits for
  // large positive x (exp(x) overflows to Inf before log1p ever runs).
  {
    for (float x : {-1000.0f, -100.0f, 100.0f, 1000.0f, 1e6f}) {
      float v = sqrt_softplus(x);
      assert(std::isfinite(v));
      assert(v >= 0.0f); // softplus is always >= 0, so is its sqrt
    }
    // For large positive x, softplus(x) ~= x, so sqrt_softplus(x) ~= sqrt(x).
    assert(std::fabs(sqrt_softplus(1e6f) - std::sqrt(1e6f)) < 1.0f);
  }

  std::cout << " PASSED\n";
}

// DeepSeek-V4 SWA ring-buffer index math (dsv4_attention_gguf,
// attention_layer.cpp): given current_pos and a window size, the windowed
// read range is past_context_len=min(current_pos+1, window) entries,
// abs_pos[t] = current_pos - past_context_len + 1 + t for t in
// [0, past_context_len), slot = abs_pos % window. Pure integer arithmetic,
// checkable without a real cache buffer or engine dependency at all.
void test_swa_ring_index() {
  std::cout << "[TEST] SWA ring-buffer index math..." << std::flush;

  auto check_one = [](int current_pos, int window) {
    int past_context_len = std::min(current_pos + 1, window);
    std::vector<int> seen_slots;
    int oldest_allowed = std::max(0, current_pos - window + 1);
    for (int t = 0; t < past_context_len; ++t) {
      int abs_pos = current_pos - past_context_len + 1 + t;
      int slot = abs_pos % window;
      // Never reads a position older than the window, and never a
      // not-yet-written future position.
      assert(abs_pos >= oldest_allowed);
      assert(abs_pos <= current_pos);
      assert(slot >= 0 && slot < window);
      seen_slots.push_back(slot);
    }
    // Never double-reads the same ring slot within one windowed read (that
    // would mean two DIFFERENT abs_pos values aliased to the same slot
    // within a single window's worth of history -- a real corruption bug,
    // not just redundant work).
    std::vector<int> sorted_slots = seen_slots;
    std::sort(sorted_slots.begin(), sorted_slots.end());
    assert(std::adjacent_find(sorted_slots.begin(), sorted_slots.end()) == sorted_slots.end());
    // Completeness: covers every position from oldest_allowed to
    // current_pos exactly once (past_context_len entries, matching the
    // real span size current_pos-oldest_allowed+1).
    assert((int)seen_slots.size() == current_pos - oldest_allowed + 1);
  };

  // --- Case 1: before the window has ever filled (current_pos < window)
  // -- past_context_len should track current_pos+1, not the full window.
  {
    int window = 16;
    for (int pos = 0; pos < window; ++pos) check_one(pos, window);
  }
  // --- Case 2: well past the window boundary -- past_context_len pinned
  // at window, oldest_allowed slides forward with current_pos.
  {
    int window = 16;
    for (int pos = window; pos < window * 5; pos += 3) check_one(pos, window);
  }
  // --- Case 3: window=1 (degenerate, every token evicts the previous one)
  // and a large window relative to a small current_pos.
  {
    check_one(0, 1);
    check_one(100, 1);
    check_one(5, 1000);
  }

  std::cout << " PASSED\n";
}

// dsv4_score_gpu (EngineConfig): the GPU dispatch (cuda_manager.hpp's
// dsv4_score_gpu_batched_host, and this same math duplicated in
// tests/bench_dsv4_score_cpu_vs_gpu.cu) compacts the raw ring buffer's
// (possibly-wrapped) [oldest_slot, oldest_slot+past_context_len) span into
// a contiguous [0,past_context_len) scratch buffer via a tail_len/head_len
// split, instead of a wrap-aware GEMM. This is pure arithmetic -- no CUDA
// needed -- so it's verified here directly against test_swa_ring_index's
// own already-trusted per-t `abs_pos % window` formula: the ring order the
// GPU path's split produces must exactly match the ring order the existing
// CPU per-t loop already visits, for every t.
void test_dsv4_gpu_ring_compact() {
  std::cout << "[TEST] DSV4 GPU score ring-buffer compaction math..." << std::flush;

  auto check_one = [](int current_pos, int window) {
    int past_context_len = std::min(current_pos + 1, window);
    int oldest_pos = current_pos - past_context_len + 1;
    int oldest_slot = ((oldest_pos % window) + window) % window;

    // GPU-path split (mirrors dsv4_score_gpu_batched_host / the bench).
    int tail_len = std::min(past_context_len, window - oldest_slot);
    int head_len = past_context_len - tail_len;
    assert(tail_len >= 0 && tail_len <= window);
    assert(head_len >= 0 && head_len < window);
    assert(tail_len + head_len == past_context_len);

    for (int t = 0; t < past_context_len; ++t) {
      // Reference: the CPU per-t ring slot (test_swa_ring_index's own
      // formula, i.e. what production's existing CPU loop reads).
      int abs_pos = current_pos - past_context_len + 1 + t;
      int expected_slot = abs_pos % window;

      // GPU-path reconstruction: t < tail_len reads compacted row t from
      // ring[oldest_slot + t]; t >= tail_len reads compacted row t from
      // ring[t - tail_len] (the wrapped-around head segment).
      int compacted_slot = (t < tail_len) ? (oldest_slot + t) : (t - tail_len);
      assert(compacted_slot == expected_slot);
    }
  };

  // Same case shapes as test_swa_ring_index: never-wrapped, steady-state
  // wrapped, and the window=1 degenerate case.
  {
    int window = 16;
    for (int pos = 0; pos < window; ++pos) check_one(pos, window);
  }
  {
    int window = 16;
    for (int pos = window; pos < window * 5; pos += 3) check_one(pos, window);
  }
  {
    check_one(0, 1);
    check_one(100, 1);
    check_one(5, 1000);
  }

  std::cout << " PASSED\n";
}

// Compression-ratio metadata distinguishes raw Flash/SWA layers from Pro's
// CSA/HCA layers.
void test_dsv4_compression_metadata() {
  std::cout << "[TEST] DeepSeek-V4 compression metadata..." << std::flush;

  ArchitectureConfig config;
  assert(!dsv4_uses_compressed_attention(config));

  config.dsv4_compress_ratios = {0, 0, 0};
  assert(!dsv4_uses_compressed_attention(config));

  config.dsv4_compress_ratios = {0, 4, 0, 128};
  assert(dsv4_uses_compressed_attention(config));

  std::cout << " PASSED\n";
}

void test_dsv4_compressed_state_helpers() {
  std::cout << "[TEST] DeepSeek-V4 compressed state helpers..." << std::flush;

  assert(dsv4_compressed_cache_capacity(1, kDsv4CsaRatio) == 1);
  assert(dsv4_compressed_cache_capacity(128, kDsv4HcaRatio) == 1);
  assert(dsv4_compressed_cache_capacity(129, kDsv4HcaRatio) == 2);
  assert(dsv4_completed_compressed_blocks(2, kDsv4CsaRatio) == 0);
  assert(dsv4_completed_compressed_blocks(3, kDsv4CsaRatio) == 1);
  assert(dsv4_compressed_block_visible(0, 3, kDsv4CsaRatio));
  assert(!dsv4_compressed_block_visible(1, 3, kDsv4CsaRatio));
  assert(dsv4_state_slot(7, 8) == 7);
  assert(dsv4_state_slot(8, 8) == 0);

  // At the first CSA boundary, previous positions are synthetic and the
  // current half maps to tokens 0..3. Later blocks retain the preceding
  // four-token window in the eight-row state ring.
  assert(dsv4_csa_overlap_position(3, 0, 0) == -4);
  assert(dsv4_csa_overlap_position(3, 0, 3) == -1);
  assert(dsv4_csa_overlap_position(3, 1, 0) == 0);
  assert(dsv4_csa_overlap_position(7, 0, 0) == 0);
  assert(dsv4_csa_overlap_position(7, 1, 3) == 7);

  const std::vector<float> scores = {2.0f, 8.0f, 8.0f, -1.0f, 4.0f};
  const std::vector<int> top = dsv4_top_k_indices(scores, 3);
  assert((top == std::vector<int>{1, 2, 4}));

  Dsv4CompressedLayerState csa;
  csa.configure(kDsv4CsaRatio, /*key_dim=*/8, /*max_tokens=*/17,
                /*indexer_key_dim=*/6);
  assert(csa.is_csa());
  assert(csa.cache_capacity == 5);
  assert(csa.state_size == 8);
  assert(csa.state_width == 16);
  assert(csa.keys.size() == 40);
  assert(csa.indexer_keys.size() == 30);
  csa.keys[0] = 3.0f;
  csa.indexer_state_score.back() = 5.0f;
  csa.reset();
  assert(csa.keys[0] == 0.0f);
  assert(csa.indexer_state_score.back() == 0.0f);

  Dsv4CompressedLayerState hca;
  hca.configure(kDsv4HcaRatio, /*key_dim=*/8, /*max_tokens=*/129);
  assert(hca.is_hca());
  assert(hca.cache_capacity == 2);
  assert(hca.state_size == 128);
  assert(hca.state_width == 8);
  assert(hca.indexer_keys.empty());

  std::cout << " PASSED\n";
}

// ============================================================
// Test: DeepSeek-V4 hash-routed MoE token-id -> expert-id lookup
// ============================================================
// The real lookup lives in core/execution_layer.cpp's moe_forward_gguf
// (CUDA-touching, can't be built here -- see this file's header) --
// mirrors that exact arithmetic (row offset = token_id * topk, the
// token-id-in-[0,vocab) bounds check, and the per-expert-id-in-
// [0,n_experts) validity check) as a standalone re-derivation, same
// pattern as test_swa_ring_index above. Verifies: (1) a well-formed row
// resolves to the right topk expert ids at the right byte offset, (2) a
// row containing an out-of-range expert id is correctly rejected (the
// "corrupt/garbage tid2eid data" defense), (3) an out-of-vocab token id
// is rejected BEFORE any read into the row buffer (never computes an
// out-of-bounds offset in the first place), (4) a tensor whose on-disk
// type/shape[0] don't match the expected I32/topk is rejected up front
// (the type/shape guard added after a real run showed EVERY hash-route
// lookup failing -- see execution_layer.cpp's own comment at that call
// site) rather than reinterpreted and read as if it were valid.
void test_tid2eid_lookup() {
  std::cout << "[TEST] DeepSeek-V4 hash-routed tid2eid lookup..." << std::flush;

  const int vocab = 4;
  const int topk = 3;
  const int n_experts = 8;
  // token 0 -> {0,1,2}; token 1 -> {3,4,5}; token 2 -> {6,7,0}; token 3
  // has a deliberately corrupt entry (99 >= n_experts) to exercise the
  // bounds-check rejection path.
  std::vector<int32_t> tid2eid = {
      0, 1, 2,
      3, 4, 5,
      6, 7, 0,
      2, 99, 1,
  };

  // Mirrors moe_forward_gguf's exact lookup: returns {valid, ids[topk]}.
  auto lookup = [&](int tok) -> std::pair<bool, std::vector<int>> {
    if (tok < 0 || tok >= vocab) return {false, {}};
    const int32_t* row = tid2eid.data() + (size_t)tok * topk;
    std::vector<int> ids(topk);
    for (int k = 0; k < topk; ++k) {
      if (row[k] < 0 || row[k] >= n_experts) return {false, {}};
      ids[k] = row[k];
    }
    return {true, ids};
  };

  // --- Case 1: well-formed rows resolve to the right ids at the right
  // offset (token_id * topk, not e.g. token_id * n_experts or unoffset).
  {
    auto [ok, ids] = lookup(0);
    assert(ok);
    assert((ids == std::vector<int>{0, 1, 2}));
  }
  {
    auto [ok, ids] = lookup(1);
    assert(ok);
    assert((ids == std::vector<int>{3, 4, 5}));
  }
  {
    auto [ok, ids] = lookup(2);
    assert(ok);
    assert((ids == std::vector<int>{6, 7, 0}));
  }
  // --- Case 2: a row with an out-of-range expert id is rejected outright
  // (all-or-nothing -- never silently uses the 2 valid entries alongside
  // one garbage one).
  {
    auto [ok, ids] = lookup(3);
    assert(!ok);
  }
  // --- Case 3: an out-of-vocab token id is rejected before computing any
  // row offset -- token=4 would read tid2eid[12..14], one topk-row past
  // this synthetic buffer's real 12-element extent.
  {
    auto [ok, ids] = lookup(4);
    assert(!ok);
    auto [ok_neg, ids_neg] = lookup(-1);
    assert(!ok_neg);
  }

  // --- Case 4: type/shape validation gate, mirroring moe_forward_gguf's
  // guard (meta.type == GGUFType::I32 && meta.shape[0] == topk) BEFORE any
  // reinterpret_cast<int32_t*> of the tensor's bytes is attempted.
  {
    const uint32_t GGUFType_I32 = 26, GGUFType_I64 = 27;
    auto type_shape_ok = [&](uint32_t on_disk_type, uint64_t shape0) {
      return on_disk_type == GGUFType_I32 && shape0 == (uint64_t)topk;
    };
    assert(type_shape_ok(GGUFType_I32, (uint64_t)topk));       // matches -- accepted
    assert(!type_shape_ok(GGUFType_I64, (uint64_t)topk));      // wrong type -- rejected
    assert(!type_shape_ok(GGUFType_I32, (uint64_t)topk + 1));  // wrong row width -- rejected
    assert(!type_shape_ok(GGUFType_I64, (uint64_t)topk + 1));  // both wrong -- rejected
  }

  std::cout << " PASSED\n";
}

// ============================================================
// Test: Q2_K / Q3_K dequant LOGIC (scalar oracle vs scalar port of the
// AVX-512 algorithm's per-group formula)
// ============================================================
// The real AVX-512 kernels live in backends/cpu/kernels_cpu.cpp and need AVX-512
// hardware to build/run (this file deliberately doesn't -- see the file
// header), so they can't be exercised here. What CAN be verified on any
// machine is the underlying ALGORITHM: block layout offsets, scale/min
// extraction, the n/j/shift nested-loop bookkeeping, and (for Q3_K) the
// hmask high-bit combine -- exactly the class of subtle layout bug this
// project has hit before (Q3_K struct field order, qk_nope size). `oracle_*`
// below is a verbatim scalar transcription of ggml's real
// dequantize_row_q2_K/q3_K (ggml/src/ggml-quants.c, confirmed against the
// upstream source directly, not from memory); `candidate_*` reproduces the
// SAME per-16-element formula backends/cpu/kernels_cpu.cpp's AVX-512 functions use
// (unpack 2-bit code, combine with hmask for Q3_K, apply dl/ml), just as a
// scalar loop instead of intrinsics -- the one AVX-512-specific detail this
// can't cover is the byte-widen-before-shift technique used to avoid
// cross-lane bit bleed in the real SIMD code, a standard, low-risk,
// mechanical pattern. This test targets the higher-risk part: the format
// understanding itself. See tests/native/main_test.cpp's Test 6 for the full,
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

// Scalar transcription of backends/cpu/kernels_cpu.cpp's AVX-512 group-of-16 formula
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
// `candidate_*` mirrors backends/cpu/kernels_cpu.cpp's actual memcpy-based
// implementation structure -- so this specifically cross-checks the
// memcpy/offset arithmetic in the real production code against a
// structurally-independent re-derivation of the same formula. See
// tests/native/main_test.cpp's Test 7 for the full AVX-512-vs-ggml-reference
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
// (backends/cpu/kernels_cpu.cpp's dequantize_row_iq{2_xxs,3_xxs,2_s}_avx512_wide):
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
// cublasGemmEx's mismatched-type dispatch (covered by tests/native/main_test.cpp's
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
// core/gpu_topology.hpp's GpuTopology can't be instantiated on this
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
// This is the REAL function used by memory/direct_io.hpp's
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
  std::string tmp = "/tmp/zeromoe_test_usage.txt";
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

void test_trace_filter() {
  std::cout << "[TEST] TraceFilter configure/matches/abort_on_nan..."
            << std::flush;

  TraceFilter& tf = TraceFilter::instance();

  // Empty pattern list (the default, unset trace_filter config) matches
  // every tag -- existing VERBOSE behavior must be unchanged unless a user
  // explicitly opts into filtering.
  tf.configure("", false);
  assert(tf.matches("HC"));
  assert(tf.matches("EXPERT-GPU"));
  assert(!tf.abort_on_nan());

  // A configured tag matches itself and any tag it's a PREFIX of (mirrors
  // llama.cpp's own "^"-anchored convention), but not an unrelated tag.
  tf.configure("HC,EXPERT", true);
  assert(tf.matches("HC"));
  assert(tf.matches("EXPERT-GPU")); // "EXPERT" is a prefix of "EXPERT-GPU"
  assert(!tf.matches("MLA"));
  assert(!tf.matches("SILU-TRACE-GPU"));
  assert(tf.abort_on_nan());

  // Re-configuring resets both the pattern list and abort_on_nan -- no
  // state leaks from a prior configure() call.
  tf.configure("MLA", false);
  assert(tf.matches("MLA"));
  assert(!tf.matches("HC"));
  assert(!tf.abort_on_nan());

  // An invalid regex token must not throw or crash configure() -- it's
  // logged and skipped, and any other valid token in the same CSV list
  // still takes effect.
  tf.configure("[invalid,MLA", false);
  assert(tf.matches("MLA"));
  assert(!tf.matches("HC"));

  std::cout << " PASSED\n";
}

void test_safetensors_adapter() {
  std::cout << "[TEST] safetensors_adapter DFlash/DSpark normalization..." << std::flush;

  static const std::string header = R"({
    "__metadata__": {"format": "dflash"},
    "model.layers.0.self_attn.q_proj.weight": {"dtype": "BF16", "shape": [4096, 2048], "data_offsets": [0, 0]},
    "model.layers.0.self_attn.k_proj.weight": {"dtype": "BF16", "shape": [1024, 2048], "data_offsets": [0, 0]},
    "model.layers.0.self_attn.v_proj.weight": {"dtype": "BF16", "shape": [1024, 2048], "data_offsets": [0, 0]},
    "model.layers.0.self_attn.o_proj.weight": {"dtype": "BF16", "shape": [2048, 4096], "data_offsets": [0, 0]},
    "model.layers.0.mlp.gate_proj.weight": {"dtype": "BF16", "shape": [6144, 2048], "data_offsets": [0, 0]},
    "model.layers.0.mlp.up_proj.weight": {"dtype": "BF16", "shape": [6144, 2048], "data_offsets": [0, 0]},
    "model.layers.0.mlp.down_proj.weight": {"dtype": "BF16", "shape": [2048, 6144], "data_offsets": [0, 0]},
    "model.layers.0.input_layernorm.weight": {"dtype": "BF16", "shape": [2048], "data_offsets": [0, 0]},
    "model.layers.0.post_attention_layernorm.weight": {"dtype": "BF16", "shape": [2048], "data_offsets": [0, 0]},
    "model.layers.1.self_attn.q_proj.weight": {"dtype": "BF16", "shape": [4096, 2048], "data_offsets": [0, 0]},
    "model.layers.1.self_attn.k_proj.weight": {"dtype": "BF16", "shape": [1024, 2048], "data_offsets": [0, 0]},
    "model.layers.1.mlp.gate_proj.weight": {"dtype": "BF16", "shape": [6144, 2048], "data_offsets": [0, 0]},
    "model.layers.1.mlp.up_proj.weight": {"dtype": "BF16", "shape": [6144, 2048], "data_offsets": [0, 0]},
    "model.layers.1.mlp.down_proj.weight": {"dtype": "BF16", "shape": [2048, 6144], "data_offsets": [0, 0]},
    "model.embed_tokens.weight": {"dtype": "BF16", "shape": [16384, 2048], "data_offsets": [0, 0]},
    "model.norm.weight": {"dtype": "BF16", "shape": [2048], "data_offsets": [0, 0]}
  })";

  SafetensorsModelInfo model;
  assert(SafetensorsAdapter::parse_header_bytes(header, model));
  assert(model.tensors.size() == 16u);
  assert(SafetensorsAdapter::detect_format(model) == "dflash");
  assert(SafetensorsAdapter::detect_family(model) == SafetensorsFamily::DenseGQA);

  const auto normalized = SafetensorsAdapter::normalize_tensors(model, "dflash");
  assert(!normalized.empty());
  assert(std::any_of(normalized.begin(), normalized.end(), [](const NormalizedTensor &t) {
    return t.logical_name.find("attn_q.weight") != std::string::npos;
  }));

  const auto dense = SafetensorsAdapter::map_dense_cpu_contract(model, "dflash");
  assert(dense.n_layers == 2);
  assert(dense.n_heads == 32);
  assert(dense.n_kv_heads == 8);
  assert(dense.head_dim == 128);
  assert(dense.hidden == 2048);
  assert(dense.intermediate == 6144);
  assert(dense.vocab == 16384);
  assert(dense.resolved_tensors.count("token_embd.weight") == 1);
  assert(dense.resolved_tensors.count("output_norm.weight") == 1);
  assert(dense.resolved_tensors.count("blk.0.attn_q.weight") == 1);

  const std::string canonical = SafetensorsAdapter::canonicalize_qwen_name(
      "model.layers.0.self_attn.q_proj.weight", "dflash");
  assert(canonical == "blk.0.attn_q.weight");

  static const std::string exl3_header = R"({
    "__metadata__": {"format": "exl3"},
    "model.layers.0.self_attn.q_proj.trellis": {"dtype": "I16", "shape": [8, 16, 32], "data_offsets": [0, 0]},
    "model.layers.0.self_attn.q_proj.suh": {"dtype": "F16", "shape": [128], "data_offsets": [0, 0]},
    "model.layers.0.self_attn.q_proj.svh": {"dtype": "F16", "shape": [256], "data_offsets": [0, 0]},
    "model.layers.0.self_attn.q_proj.mul1": {"dtype": "I32", "shape": [1], "data_offsets": [0, 0]},
    "model.layers.0.self_attn.q_proj.bias": {"dtype": "F16", "shape": [256], "data_offsets": [0, 0]},
    "model.layers.0.mlp.experts.3.up_proj.trellis": {"dtype": "I16", "shape": [16, 32, 24], "data_offsets": [0, 0]},
    "model.layers.0.mlp.experts.3.up_proj.suh": {"dtype": "F16", "shape": [256], "data_offsets": [0, 0]},
    "model.layers.0.mlp.experts.3.up_proj.svh": {"dtype": "F16", "shape": [512], "data_offsets": [0, 0]},
    "model.layers.0.mlp.experts.3.up_proj.mul1": {"dtype": "I32", "shape": [1], "data_offsets": [0, 0]}
  })";
  SafetensorsModelInfo exl3_model;
  assert(SafetensorsAdapter::parse_header_bytes(exl3_header, exl3_model));
  assert(SafetensorsAdapter::detect_format(exl3_model) == "exl3");
  assert(SafetensorsAdapter::detect_family(exl3_model) == SafetensorsFamily::EXL3);
  const Exl3Contract exl3 = SafetensorsAdapter::map_exl3_contract(exl3_model);
  assert(exl3.complete);
  assert(exl3.linears.size() == 2);
  assert(exl3.n_layers == 1);
  assert(exl3.n_experts == 4);
  assert(exl3.linears[0].bits_per_weight == 2.0 ||
         exl3.linears[1].bits_per_weight == 2.0);
  assert(exl3.linears[0].codebook == Exl3Codebook::MUL1 ||
         exl3.linears[1].codebook == Exl3Codebook::MUL1);

  std::cout << " PASSED\n";
}

void test_exl3_ngram_reader() {
  std::cout << "[TEST] EXL3 indexed ngram reads/cache/symlinks..." << std::flush;
  for (int bits = 1; bits <= 8; ++bits) {
    char path[] = "/tmp/zeromoe_exl3_ngram_XXXXXX";
    const int descriptor = mkstemp(path);
    assert(descriptor >= 0);
    close(descriptor);
    const std::string alias = std::string(path) + ".link";
    assert(symlink(path, alias.c_str()) == 0);
    const int words = 1 + bits * 10;
    std::vector<uint16_t> bias(320), packed(4 * words);
    for (size_t index = 0; index < bias.size(); ++index)
      bias[index] = fp32_to_fp16(static_cast<float>(index % 23) * 0.003f);
    for (int row = 0; row < 4; ++row) {
      packed[row * words] = fp32_to_fp16(0.5f + 0.25f * row);
      for (int word = 1; word < words; ++word)
        packed[row * words + word] = static_cast<uint16_t>(row * 12345 + word * 56789);
    }
    {
      std::ofstream file(path, std::ios::binary | std::ios::trunc);
      file.write(reinterpret_cast<const char *>(bias.data()), bias.size() * 2);
      file.write(reinterpret_cast<const char *>(packed.data()), packed.size() * 2);
      assert(file.good());
    }
    SafetensorsModelInfo model;
    const std::string prefix = "model.ngram_embedding.";
    SafetensorsTensorInfo bias_info;
    bias_info.name = prefix + "head_bias";
    bias_info.file_path = path;
    bias_info.dtype = "F16";
    bias_info.shape = {2, 160};
    bias_info.length = bias.size() * 2;
    model.tensors.emplace(bias_info.name, bias_info);
    for (int segment = 0; segment < 2; ++segment) {
      SafetensorsTensorInfo info;
      info.name = prefix + std::to_string(segment) + ".trellis";
      info.file_path = alias;
      info.dtype = "I16";
      info.shape = {2, words};
      info.offset = bias_info.length + static_cast<uint64_t>(segment) * words * 4;
      info.data_offset = info.offset;
      info.length = words * 4;
      model.tensors.emplace(info.name, info);
    }
    Exl3NgramReader reader(model);
    assert(unlink(alias.c_str()) == 0);
    assert(unlink(path) == 0);
    const auto check_row = [&](int head, int row) {
      std::array<float, 160> actual{};
      assert(reader.read_row(head, row, actual.data(), actual.size()));
      for (int index = 0; index < 160; ++index) {
        uint32_t state = 0;
        for (int bit = 0; bit < 16; ++bit) {
          const int stream = ((index - bit / bits + 160) % 160) * bits + bit % bits;
          state |= ((packed[row * words + 1 + stream / 16] >> (stream % 16)) & 1u) << bit;
        }
        const uint32_t product = state * 0x83DCD12Du;
        const int sum = (product & 255u) + ((product >> 8) & 255u) +
                        ((product >> 16) & 255u) + (product >> 24);
        const float code = fp16_to_fp32(fp32_to_fp16(std::fma(
            static_cast<float>(1024 + sum), fp16_to_fp32(0x1eee), fp16_to_fp32(0xc931))));
        const float expected = fp16_to_fp32(fp32_to_fp16(
            code * fp16_to_fp32(packed[row * words]) + fp16_to_fp32(bias[head * 160 + index])));
        assert(actual[index] == expected);
      }
    };
    check_row(0, 0);
    for (int layer = 0; layer < 48; ++layer) check_row(0, 0);
    check_row(0, 3);
    check_row(1, 3);
    check_row(0, 0);
    assert(reader.row_reads() == 4 && reader.cache_hits() == 48);
    std::array<float, 160> output{};
    assert(!reader.read_row(0, 4, output.data(), 160));
    assert(!reader.read_row(0, -1, output.data(), 160));
    assert(!reader.read_row(-1, 0, output.data(), 160));
    assert(!reader.read_row(2, 0, output.data(), 160));
    assert(!reader.read_row(0, 0, nullptr, 160));
    assert(!reader.read_row(0, 0, output.data(), 159));
  }
  std::cout << " PASSED\n";
}

void test_exllamav3_backend_launch() {
  std::cout << "[TEST] ExLlamaV3 config/argv/no-shell handoff..." << std::flush;
  const auto config_path = std::filesystem::absolute("configs/upstream.toml");
  const std::string prompt = "Spaces; $(touch never-execute) 'quoted'";
  const auto arguments = exllamav3_backend_arguments(
      config_path.string(), "../runtime/bin/python", "../tools/run_exllamav3.py",
      {"--prompt", prompt, "--ngen", "128"});
  assert(arguments.size() == 8);
  assert(arguments[0] == (config_path.parent_path().parent_path() / "runtime/bin/python").string());
  assert(arguments[1] == (config_path.parent_path().parent_path() / "tools/run_exllamav3.py").string());
  assert(arguments[3] == config_path.string());
  assert(arguments[5] == prompt);
  const auto path_python = exllamav3_backend_arguments(config_path.string(), "python3", "/runner.py", {});
  assert(path_python[0] == "python3" && path_python[1] == "/runner.py");
  assert(!exec_exllamav3_backend(config_path.string(), "/nonexistent-zeromoe-python", "/runner.py", {}));
  assert(!exec_exllamav3_backend(config_path.string(), "", "/runner.py", {}));
  char path[] = "/tmp/zeromoe_exllamav3_config_XXXXXX";
  const int descriptor = mkstemp(path);
  assert(descriptor >= 0);
  close(descriptor);
  const auto write_config = [&](const std::string &text) {
    std::ofstream file(path);
    file << text;
  };
  write_config("model_path = \"/model\"\nmodel_format = \"exl3\"\nexl3_backend = \"exllamav3\"\n"
               "exllamav3_python = \"python3\"\nexllamav3_runner_path = \"/runner.py\"\n"
               "exllamav3_cuda_schedule = \"blocking\"\n");
  CPUConfig valid;
  assert(valid.load(path) && valid.exl3_backend == "exllamav3" && valid.exllamav3_python == "python3" &&
         valid.exllamav3_cuda_schedule == "blocking");
  write_config("exl3_backend = \"typo\"\n");
  CPUConfig invalid;
  assert(!invalid.load(path));
  write_config("model_format = \"gguf\"\nexl3_backend = \"exllamav3\"\n");
  CPUConfig wrong_format;
  assert(!wrong_format.load(path));
  unlink(path);
  std::cout << " PASSED\n";
}

void test_exl3_cpu_matrix_cache() {
  std::cout << "[TEST] EXL3 CPU matrix cache/LRU/held entries..." << std::flush;
  Exl3CpuMatrixCache cache(100);
  const auto matrix = [](uint8_t value) {
    Exl3Mul1MatrixData result;
    result.trellis.resize(32, value);
    result.suh.resize(4, value);
    result.svh.resize(4, value);
    return result;
  };
  cache.insert("a", matrix(1));
  cache.insert("b", matrix(2));
  assert(cache.get("a"));
  cache.insert("c", matrix(3));
  assert(cache.get("a") && !cache.get("b") && cache.get("c"));
  {
    auto held_a = cache.get("a");
    auto held_c = cache.get("c");
    auto uncached = cache.insert("d", matrix(4));
    assert(uncached && cache.stats().resident_bytes == 80);
  }
  cache.insert("d", matrix(4));
  assert(!cache.get("a") && cache.get("c") && cache.get("d"));
  const auto stats = cache.stats();
  assert(stats.resident_bytes == 80 && stats.evictions == 2 && stats.hits == 7 &&
         stats.misses == 2);
  Exl3CpuMatrixCache disabled(0);
  auto uncached = disabled.insert("x", matrix(5));
  assert(uncached && !disabled.get("x") && disabled.stats().resident_bytes == 0);
  std::cout << " PASSED\n";
}

void test_exl3_parallel_shard_reader() {
  std::cout << "[TEST] EXL3 concurrent shard reads/symlink..." << std::flush;
  char path[] = "/tmp/zeromoe_exl3_shard_XXXXXX";
  const int descriptor = mkstemp(path);
  assert(descriptor >= 0);
  std::array<uint8_t, 256> source{};
  for (size_t index = 0; index < source.size(); ++index)
    source[index] = static_cast<uint8_t>(index * 17 + 5);
  assert(write(descriptor, source.data(), source.size()) ==
         static_cast<ssize_t>(source.size()));
  close(descriptor);
  const std::string alias = std::string(path) + ".link";
  assert(symlink(path, alias.c_str()) == 0);
  SafetensorsModelInfo model;
  SafetensorsTensorInfo info;
  info.name = "model.layers.0.proj.trellis";
  info.file_path = alias;
  info.dtype = "I16";
  info.shape = {8, 1};
  info.offset = 23;
  info.length = 16;
  model.tensors.emplace(info.name, info);
  constexpr int thread_count = 16;
  std::array<std::array<uint8_t, 16>, thread_count> actual{};
  std::array<int, thread_count> success{};
  std::vector<std::thread> threads;
  for (int thread = 0; thread < thread_count; ++thread) {
    threads.emplace_back([&, thread] {
      std::vector<uint8_t> bytes;
      std::string error;
      success[thread] = read_exl3_tensor(model, info.name, bytes, error) && bytes.size() == 16;
      if (success[thread]) std::copy(bytes.begin(), bytes.end(), actual[thread].begin());
    });
  }
  for (auto &thread : threads) thread.join();
  for (int thread = 0; thread < thread_count; ++thread) {
    assert(success[thread]);
    for (int index = 0; index < 16; ++index)
      assert(actual[thread][index] == source[23 + index]);
  }
  assert(unlink(alias.c_str()) == 0);
  assert(unlink(path) == 0);
  std::cout << " PASSED\n";
}

void test_exl3_mul1_matmul() {
  std::cout << "[TEST] EXL3 mul1 reconstruct/matmul..." << std::flush;

  for (int bits = 1; bits <= 8; ++bits) {
    const int input_features = bits >= 3 ? 640 : 128;
    const int output_features = bits >= 3 ? 384 : 128;
    constexpr int rows = 3;
    const size_t trellis_words = static_cast<size_t>(input_features / 16) *
                                  (output_features / 16) * 16 * bits;
    std::vector<uint16_t> trellis(trellis_words);
    uint32_t state = 0x12345678u + static_cast<uint32_t>(bits);
    for (uint16_t &word : trellis) {
      state = state * 1664525u + 1013904223u;
      word = static_cast<uint16_t>(state >> 16);
    }

    std::vector<uint16_t> suh(input_features);
    std::vector<uint16_t> svh(output_features);
    std::vector<uint16_t> bias(output_features);
    for (int index = 0; index < input_features; ++index)
      suh[index] = fp32_to_fp16(0.75f + 0.5f * ((index % 17) / 16.0f));
    for (int index = 0; index < output_features; ++index) {
      svh[index] = fp32_to_fp16(0.8f + 0.4f * ((index % 13) / 12.0f));
      bias[index] = fp32_to_fp16((index % 7 - 3) * 0.01f);
    }

    Exl3Mul1MatrixView matrix{
        .trellis = trellis.data(),
        .suh = suh.data(),
        .svh = svh.data(),
        .bias = bias.data(),
        .input_features = input_features,
        .output_features = output_features,
        .bits = bits,
    };
    std::vector<float> input(rows * input_features);
    for (size_t index = 0; index < input.size(); ++index)
      input[index] = std::sin(static_cast<float>(index) * 0.071f);

    std::vector<float> direct(rows * output_features);
    assert(exl3_mul1_matmul(matrix, input.data(), direct.data(), rows));
    std::vector<float> batch_output(2 * output_features);
    const Exl3Mul1BatchItem batch_items[] = {
        {matrix, input.data(), batch_output.data()},
        {matrix, input.data() + input_features, batch_output.data() + output_features}};
    assert(exl3_mul1_matmul_batch(batch_items, 2));
    float batch_error = 0.0f;
    for (int index = 0; index < 2 * output_features; ++index)
      batch_error = std::max(batch_error, std::abs(batch_output[index] - direct[index]));
    assert(batch_error < 1e-5f);
    std::fill(batch_output.begin(), batch_output.end(), 0.0f);
    assert(exl3_mul1_matmul_batch(batch_items, 2, true));
    double quantized_error = 0.0, exact_energy = 0.0;
    for (int index = 0; index < 2 * output_features; ++index) {
      const double error = batch_output[index] - direct[index];
      quantized_error += error * error;
      exact_energy += static_cast<double>(direct[index]) * direct[index];
    }
    assert(std::sqrt(quantized_error / std::max(exact_energy, 1e-20)) < 0.03);

    const auto predecoded = exl3_mul1_predecode_q8(matrix);
    assert(predecoded.size() ==
           static_cast<size_t>(input_features) * output_features);
    const Exl3Mul1Q8MatrixView q8_matrix{
        predecoded.data(), suh.data(), svh.data(), bias.data(),
        input_features, output_features};
    std::vector<float> q8_output(2 * output_features);
    const Exl3Mul1Q8BatchItem q8_items[] = {
        {q8_matrix, input.data(), q8_output.data()},
        {q8_matrix, input.data() + input_features,
         q8_output.data() + output_features}};
    assert(exl3_mul1_q8_matmul_batch(q8_items, 2));
    double predecoded_error = 0.0;
    for (int index = 0; index < 2 * output_features; ++index) {
      const double error = q8_output[index] - direct[index];
      predecoded_error += error * error;
    }
    assert(std::sqrt(predecoded_error / std::max(exact_energy, 1e-20)) < 0.04);

    std::vector<uint8_t> packed = exl3_mul1_pack_matrix(matrix);
    assert(!packed.empty());
    Exl3Mul1MatrixView packed_view;
    assert(exl3_mul1_view_from_packed(packed.data(), packed.size(),
                                      packed_view));
    std::vector<float> packed_output(rows * output_features);
    assert(exl3_mul1_matmul_packed(
        packed.data(), input.data(), packed_output.data(), rows,
        input_features, output_features));
    for (size_t index = 0; index < direct.size(); ++index)
      assert(std::abs(direct[index] - packed_output[index]) < 1e-6f);

    std::vector<float> weight(input_features * output_features);
    assert(exl3_mul1_reconstruct(matrix, weight.data()));
    std::vector<float> reference(rows * output_features);
    for (int row = 0; row < rows; ++row) {
      for (int output = 0; output < output_features; ++output) {
        float sum = fp16_to_fp32(bias[output]);
        for (int input_index = 0; input_index < input_features; ++input_index) {
          sum += input[static_cast<size_t>(row) * input_features + input_index] *
                 weight[static_cast<size_t>(input_index) * output_features +
                        output];
        }
        reference[static_cast<size_t>(row) * output_features + output] = sum;
      }
    }

    float max_error = 0.0f;
    for (size_t index = 0; index < direct.size(); ++index)
      max_error = std::max(max_error, std::abs(direct[index] - reference[index]));
    assert(max_error < 2e-3f);

    std::fill(trellis.begin(), trellis.end(), 0);
    std::fill(suh.begin(), suh.end(), fp32_to_fp16(1.0f));
    std::fill(svh.begin(), svh.end(), fp32_to_fp16(1.0f));
    std::fill(bias.begin(), bias.end(), fp32_to_fp16(0.0f));
    std::fill(input.begin(), input.end(), 1.0f);
    assert(exl3_mul1_matmul(matrix, input.data(), direct.data(), rows));
    const float expected_dc = static_cast<float>(input_features) * fp16_to_fp32(fp32_to_fp16(
        std::fma(1024.0f, fp16_to_fp32(0x1eee), fp16_to_fp32(0xc931))));
    for (int row = 0; row < rows; ++row) {
      assert(std::abs(direct[row * output_features] - expected_dc) < 1e-3f);
      for (int column = 1; column < output_features; ++column) {
        const float expected = column % 128 == 0 ? expected_dc : 0.0f;
        if (std::abs(direct[row * output_features + column] - expected) >= 1e-3f)
          std::cerr << "EXL3 zero-state mismatch: VNNI=" << exl3_cpu_vnni_available()
                    << " bits=" << bits << " row=" << row << " column=" << column
                    << " actual=" << direct[row * output_features + column]
                    << " expected=" << expected << "\n";
        assert(std::abs(direct[row * output_features + column] - expected) < 1e-3f);
      }
    }
  }

  assert(exl3_qwen_tensor_name("model.language_model.layers.1.ple.key_proj.weight") ==
         "blk.1.ple_key.weight");
  assert(exl3_qwen_tensor_name("model.language_model.layers.3.self_attn.q_proj") ==
         "blk.3.attn_q.weight");
  assert(exl3_qwen_tensor_name("model.language_model.hyper_connection_mixer.hc_norm.weight") ==
         "output_hc_norm.weight");

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
// Qwen4Exp (Qwen3.8-Flash-Next) hybrid layer-type detection
// (gguf_loader.hpp's qwen4exp_is_gdn_layer) -- the layer-type switch the
// whole hybrid dispatch hinges on.
void test_qwen4exp_layer_detection() {
  std::cout << "[TEST] qwen4exp layer detection..." << std::flush;

  // Interval fallback (no explicit array): every 4th layer (1-based) is
  // full attention, the rest GDN -- llama.cpp qwen4exp.cpp's
  // load_arch_hparams fallback (full_attention_interval, default 4).
  {
    ArchitectureConfig c;
    c.n_layers = 8;
    c.qwen4exp_full_attention_interval = 4;
    const bool want[8] = {true, true, true, false, true, true, true, false};
    for (int l = 0; l < 8; ++l)
      assert(qwen4exp_is_gdn_layer(c, l) == want[l]);
  }

  // An explicit attention.recurrent_layers array wins over the interval.
  {
    ArchitectureConfig c;
    c.n_layers = 4;
    c.qwen4exp_full_attention_interval = 2; // would say: 0,2 full
    c.qwen4exp_recurrent_layers = {1, 0, 0, 1}; // but the array says otherwise
    assert(qwen4exp_is_gdn_layer(c, 0));
    assert(!qwen4exp_is_gdn_layer(c, 1));
    assert(!qwen4exp_is_gdn_layer(c, 2));
    assert(qwen4exp_is_gdn_layer(c, 3));
  }

  // Defensive cases: out-of-range and fully-unconfigured configs fail safe
  // toward the full-attention path (real tensors to report missing), never
  // toward the recurrent path's state allocation.
  {
    ArchitectureConfig c;
    c.n_layers = 4; // neither field set at all
    assert(!qwen4exp_is_gdn_layer(c, 0));
    assert(!qwen4exp_is_gdn_layer(c, 3));
    assert(!qwen4exp_is_gdn_layer(c, -1));
    assert(!qwen4exp_is_gdn_layer(c, 4));
    c.qwen4exp_full_attention_interval = 4;
    assert(!qwen4exp_is_gdn_layer(c, 100));
  }

  // PLE layer membership
  {
    ArchitectureConfig c;
    c.ple_layers = {1, 17};
    assert(qwen4exp_is_ple_layer(c, 1));
    assert(qwen4exp_is_ple_layer(c, 17));
    assert(!qwen4exp_is_ple_layer(c, 0));
    assert(!qwen4exp_is_ple_layer(c, 2));
  }

  std::cout << " PASSED\n";
}

// Qwen4Exp GDN gated delta rule (models/qwen/qwen4exp_gdn.hpp's
// qwen4exp_gdn_delta_chunk vs qwen4exp_gdn_delta_step: the chunked form must
// reproduce the sequential recurrence EXACTLY (to float rounding), because it
// is a closed-form rewrite of the same rule, not an approximation. The
// sequential step is the oracle here -- it is itself checked against an
// independent double-precision reference by the test below, so this chains
// onto a verified base rather than asserting two unverified things agree.
//
// Covers the cases that actually distinguish a correct derivation from a
// plausible-looking wrong one: a chunk applied to a NON-zero starting state
// (the A_t*S_0 term and the S_0^T K / S_0^T Q projections), several chunks in
// sequence (inter-chunk state carry-over), a non-multiple-of-chunk tail, and
// beta/decay values in their real ranges.
void test_qwen4exp_gdn_delta_chunk() {
  std::cout << "[TEST] qwen4exp GDN chunked vs sequential..." << std::flush;

  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  double worst = 0.0, worst_state = 0.0;
  // (hd, chunk lengths) -- last chunk deliberately shorter than the others.
  const int hd_list[] = {8, 16, 32};
  const int chunks[]  = {5, 7, 3};

  for (int hi = 0; hi < 3; ++hi) {
    const int hd = hd_list[hi];
    // A non-zero starting state: zero would hide any bug in the A_t*S_0 term.
    std::vector<float> st0((size_t)hd * hd);
    for (auto &x : st0) x = 0.25f * dist(rng);

    std::vector<float> st_seq = st0, st_chunk = st0;

    for (int ci = 0; ci < 3; ++ci) {
      const int C = chunks[ci];
      std::vector<float> Q((size_t)C * hd), K((size_t)C * hd), V((size_t)C * hd);
      std::vector<float> beta((size_t)C), g_log((size_t)C);
      for (auto &x : Q) x = dist(rng);
      for (auto &x : K) x = dist(rng);
      for (auto &x : V) x = dist(rng);
      for (int t = 0; t < C; ++t) {
        beta[(size_t)t] = 1.0f / (1.0f + std::exp(-dist(rng)));   // sigmoided
        g_log[(size_t)t] = qwen4exp_gdn_decay_log(dist(rng), dist(rng),
                                                  -std::exp(dist(rng)));
      }

      std::vector<float> out_seq((size_t)C * hd, 0.0f);
      for (int t = 0; t < C; ++t)
        qwen4exp_gdn_delta_step(Q.data() + (size_t)t * hd, K.data() + (size_t)t * hd,
                                V.data() + (size_t)t * hd, beta[(size_t)t],
                                g_log[(size_t)t], st_seq.data(),
                                out_seq.data() + (size_t)t * hd, hd);

      std::vector<float> out_chunk((size_t)C * hd, 0.0f);
      qwen4exp_gdn_delta_chunk(Q.data(), K.data(), V.data(), beta.data(),
                               g_log.data(), st_chunk.data(), out_chunk.data(),
                               C, hd);

      for (size_t n = 0; n < out_seq.size(); ++n)
        worst = std::max(worst, (double)std::fabs(out_seq[n] - out_chunk[n]));
      for (size_t n = 0; n < st_seq.size(); ++n)
        worst_state = std::max(worst_state,
                               (double)std::fabs(st_seq[n] - st_chunk[n]));
    }
  }

  std::cout << " max_out_diff=" << worst << " max_state_diff=" << worst_state;
  // Same rewrite, different summation order -> float noise only, not an
  // approximation error. 1e-4 is loose enough for accumulated rounding over
  // several chunks at hd=32 and still far tighter than any real bug would be.
  assert(worst < 1e-4);
  assert(worst_state < 1e-4);
  std::cout << " PASSED\n";
}

// qwen4exp_gdn_delta_step, the REAL function attention_layer.cpp calls)
// checked against an independent naive double-precision reference written
// straight from delta-net-base.cpp's build_delta_net_autoregressive ops.
void test_qwen4exp_gdn_delta_step() {
  std::cout << "[TEST] qwen4exp GDN delta step..." << std::flush;

  const int hd = 8;
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  // decay must be in (0,1): ssm_a is -exp(A_log) < 0, softplus > 0, so
  // g_log < 0 -- use a realistic value.
  std::vector<float> state((size_t)hd * hd, 0.0f);
  std::vector<std::vector<double>> ref(hd, std::vector<double>(hd, 0.0));

  for (int tok = 0; tok < 5; ++tok) { // multiple steps: checks state carry-over
    std::vector<float> q(hd), k(hd), v(hd);
    for (auto &x : q) x = dist(rng);
    for (auto &x : k) x = dist(rng);
    for (auto &x : v) x = dist(rng);
    const float beta = 1.0f / (1.0f + std::exp(-dist(rng))); // sigmoided
    const float g_log = qwen4exp_gdn_decay_log(dist(rng), dist(rng),
                                               -1.5f); // ssm_a < 0
    assert(g_log < 0.0f); // negative log-decay, always

    std::vector<float> out(hd, 0.0f);
    qwen4exp_gdn_delta_step(q.data(), k.data(), v.data(), beta, g_log,
                            state.data(), out.data(), hd);

    // Reference: S' = decay*S; sk = S'^T k; S'' = S' + beta * k (v-sk)^T;
    // o = S''^T (q/sqrt(hd)) -- all in double.
    const double decay = std::exp((double)g_log);
    const double qs = 1.0 / std::sqrt((double)hd);
    for (int i = 0; i < hd; ++i)
      for (int j = 0; j < hd; ++j)
        ref[i][j] *= decay;
    std::vector<double> sk(hd, 0.0);
    for (int j = 0; j < hd; ++j)
      for (int i = 0; i < hd; ++i)
        sk[j] += ref[i][j] * (double)k[i];
    for (int i = 0; i < hd; ++i)
      for (int j = 0; j < hd; ++j)
        ref[i][j] += (double)k[i] * ((double)v[j] - sk[j]) * (double)beta;
    for (int j = 0; j < hd; ++j) {
      double o = 0.0;
      for (int i = 0; i < hd; ++i)
        o += ref[i][j] * (double)q[i] * qs;
      // f32-vs-f64 accumulation tolerance
      assert(std::fabs((double)out[j] - o) < 1e-4);
    }
  }

  // Sanity: zero beta freezes the state's content contribution, so a repeated
  // (k, v) pair with beta=0 and decay=1 leaves the output unchanged.
  {
    std::vector<float> st((size_t)hd * hd, 0.0f);
    std::vector<float> q(hd), k(hd), v(hd), o1(hd), o2(hd);
    for (int i = 0; i < hd; ++i) { q[i] = 0.1f * i; k[i] = 0.2f; v[i] = 0.3f; }
    qwen4exp_gdn_delta_step(q.data(), k.data(), v.data(), /*beta=*/1.0f,
                            /*g_log=*/0.0f, st.data(), o1.data(), hd);
    qwen4exp_gdn_delta_step(q.data(), k.data(), v.data(), /*beta=*/0.0f,
                            /*g_log=*/0.0f, st.data(), o2.data(), hd);
    for (int i = 0; i < hd; ++i)
      assert(std::fabs(o2[i] - o1[i]) < 1e-6f);
  }

  std::cout << " PASSED\n";
}

// Qwen4Exp GDN L2 norm (qwen4exp_l2_norm_inplace): ggml_l2_norm's exact
// formula x / max(sqrt(sum(x^2)), eps) -- NOT x / sqrt(sum + eps).
void test_qwen4exp_l2_norm() {
  std::cout << "[TEST] qwen4exp L2 norm..." << std::flush;
  {
    float v[4] = {3.0f, 4.0f, 0.0f, 0.0f}; // sum(x^2)=25, sqrt=5
    qwen4exp_l2_norm_inplace(v, 4, 1e-6f);
    assert(std::fabs(v[0] - 0.6f) < 1e-6f);
    assert(std::fabs(v[1] - 0.8f) < 1e-6f);
  }
  {
    // near-zero vector: the max(, eps) floor dominates (would be a different
    // answer under the sqrt(sum + eps) formula)
    float v[2] = {1e-8f, 0.0f};
    qwen4exp_l2_norm_inplace(v, 2, 1e-6f);
    assert(std::fabs(v[0] - 1e-8f / 1e-6f) < 1e-4f);
  }
  std::cout << " PASSED\n";
}

// Qwen4Exp PLE n-gram hash (models/qwen/ple_hash.hpp's ple_ngram_row, the REAL
// function ple_ngram.cpp calls) against a hand-computed reference,
// including a multiplier too wide for float32 (45-bit constants are the
// whole reason this hash exists in int64 -- see the header comment).
void test_qwen4exp_ple_hash() {
  std::cout << "[TEST] qwen4exp PLE n-gram hash..." << std::flush;

  // hand-computed: mixed = (c0*m0) ^ (c1*m1); row = mixed % vocab + offset
  {
    const int64_t ctx[2] = {12345, 678};
    const int64_t mult[2] = {1000003, 1000033};
    const uint64_t mixed = (uint64_t)12345 * 1000003ull ^ (uint64_t)678 * 1000033ull;
    const int64_t row = ple_ngram_row(ctx, 2, mult, /*vocab=*/1000000, /*off=*/7777);
    assert(row == (int64_t)(mixed % 1000000ull) + 7777);
  }

  // 45-bit multiplier: (c0*m0) exceeds 2^53, where double (and float32)
  // silently round -- the row must still match exact integer math.
  {
    const int64_t ctx[3] = {248044, 99991, 7};
    const int64_t mult[3] = {(1ll << 45) + 12345, (1ll << 44) + 77, 3};
    const uint64_t mixed = (uint64_t)ctx[0] * (uint64_t)mult[0] ^
                           (uint64_t)ctx[1] * (uint64_t)mult[1] ^
                           (uint64_t)ctx[2] * (uint64_t)mult[2];
    const int64_t row = ple_ngram_row(ctx, 3, mult, /*vocab=*/20000000, /*off=*/0);
    assert(row == (int64_t)(mixed % 20000000ull));
    assert(row >= 0 && row < 20000000);
  }

  // EOS-cut semantics are the CALLER's job (ple_ngram.cpp); the hash itself
  // must simply be deterministic in its inputs -- same window, same row.
  {
    const int64_t ctx[2] = {42, 42};
    const int64_t mult[2] = {5, 9};
    assert(ple_ngram_row(ctx, 2, mult, 100, 0) ==
           ple_ngram_row(ctx, 2, mult, 100, 0));
  }

  std::cout << " PASSED\n";
}

void test_host_budget_admission() {
  std::cout << "[TEST] host-memory budget admission..." << std::flush;
  assert(MemoryGuard::effective_stop_limit_mb(4096) == 20480);
  assert(MemoryGuard::effective_stop_limit_mb(32768) == 32768);
  assert(host_budget_allows(0, 1000, 1000, 1000));
  assert(host_budget_allows(1000, 400, 500, 100));
  assert(!host_budget_allows(1000, 400, 601, 0));
  assert(!host_budget_allows(1000, 400, 500, 101));
  assert(!host_budget_allows(1000, 1001, 0, 0));
  assert(!host_budget_allows(1000, 900, 200, 0));
  constexpr size_t MiB = 1024ull * 1024ull;
  constexpr size_t GiB = 1024ull * MiB;
  // Page-aligned slab allocations can remain virtual until the I/O worker
  // faults them in. Account reservations against the startup snapshot even
  // while current MemAvailable has not fallen yet.
  assert(physical_host_budget_allows(8 * GiB, 8 * GiB, 0,
                                     6 * GiB, 2 * GiB));
  assert(!physical_host_budget_allows(8 * GiB, 8 * GiB, 1 * GiB,
                                      6 * GiB, 2 * GiB));
  // Also stop when another process consumes memory after our startup check.
  assert(!physical_host_budget_allows(16 * GiB, 5 * GiB, 1 * GiB,
                                      4 * GiB, 2 * GiB));
  // An additional graph watermark is reserved on top of the global stop
  // floor, and unavailable physical-memory data fails closed.
  assert(!physical_host_budget_allows(8 * GiB, 8 * GiB, 0,
                                      5 * GiB, 2 * GiB, 2 * GiB));
  assert(!physical_host_budget_allows(0, 8 * GiB, 0,
                                      1, 2 * GiB));
  std::cout << " PASSED\n";
}

void test_dsv41_router_stable_boundary() {
  std::cout << "[TEST] DeepSeek V4.1 stable near-tie routing..."
            << std::flush;
  std::vector<Dsv41RouterScore> native = {
      {299, 12.674845695f}, {29, 12.650976181f},
      {274, 12.420443535f}, {15, 12.419219971f},
      {68, 12.405943871f}, {336, 12.390210152f},
      {311, 12.390174866f}, {83, 12.387273789f}};
  std::vector<Dsv41RouterScore> graph = native;
  graph[5].score = 12.390183449f;
  graph[6].score = 12.390187263f;
  std::vector<int> native_ids, graph_ids;
  assert(!dsv41_select_topk_stable_boundary(native, (int)native.size(), 6,
                                            2.0e-5f, native_ids));
  assert(dsv41_select_topk_stable_boundary(graph, (int)graph.size(), 6,
                                           2.0e-5f, graph_ids));
  assert(native_ids == graph_ids);
  assert(native_ids.back() == 336);

  std::vector<Dsv41RouterScore> clear_gap = {
      {7, 1.0f}, {6, 0.9f}, {5, 0.8f}, {4, 0.7f}};
  std::vector<int> clear_ids;
  assert(!dsv41_select_topk_stable_boundary(clear_gap, 4, 2,
                                            2.0e-5f, clear_ids));
  assert((clear_ids == std::vector<int>{7, 6}));
  std::cout << " PASSED\n";
}

// moe_decode_batch.hpp's grouped per-token expert dispatch (the decode-gap
// investigation's prototype, 2026-09-10) vs a reference reimplementation of
// TODAY's per-expert-call convention (execution_layer.cpp's expert_ffn_cpu:
// gate/up matmul -> silu -> elem_mul -> down matmul, called once per k with
// its own fresh buffers, weighted-accumulated by the caller). Both use the
// exact same primitives (kernels_cpu.hpp's matmul/silu_inplace/elem_mul) in
// the exact same order, so this is a structural-equivalence proof (shared
// scratch buffers + a single grouped call vs one call per expert with fresh
// buffers), not an approximation -- output must be BIT-IDENTICAL, matching
// this project's own convention for this class of "reorganize the
// dispatch, don't change the math" change (e.g. Test 26 / moe_gpu_accum,
// "max_abs=0").
void test_moe_decode_batch() {
  std::cout << "[TEST] moe_decode_batch grouped vs per-expert loop..."
            << std::flush;

  // Deliberately smaller than qwen4exp's real shape (D=2560, moe_inter=640,
  // topk=10) -- this test is about dispatch STRUCTURE, not performance, so
  // a small shape keeps it fast while still exercising every code path
  // (multiple experts, non-square gate/down matrices, real SiLU/mul math).
  const int D = 96, moe_inter = 64, n_pool = 6, k_used = 4;

  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  auto fill = [&](std::vector<float> &v) {
    for (float &x : v) x = dist(rng);
  };

  std::vector<float> hidden(D);
  fill(hidden);

  // n_pool candidate experts' weights; only the first k_used are actually
  // "selected" (mirrors a token's real topk subset of n_experts).
  std::vector<std::vector<float>> gate_w(n_pool, std::vector<float>((size_t)moe_inter * D));
  std::vector<std::vector<float>> up_w(n_pool, std::vector<float>((size_t)moe_inter * D));
  std::vector<std::vector<float>> down_w(n_pool, std::vector<float>((size_t)D * moe_inter));
  for (int e = 0; e < n_pool; ++e) {
    fill(gate_w[e]);
    fill(up_w[e]);
    fill(down_w[e]);
  }

  std::vector<const float *> expert_gate(k_used), expert_up(k_used), expert_down(k_used);
  std::vector<float> expert_weight(k_used);
  // Non-trivial, non-uniform routing weights (as a real softmax top-k would
  // produce), NOT normalized to sum 1 -- the function must not assume that.
  const float raw_weights[k_used] = {0.41f, 0.05f, 0.30f, 0.10f};
  for (int k = 0; k < k_used; ++k) {
    // out-of-order pool indices, matching a real router's non-contiguous
    // expert-id selection.
    const int pool_idx = (k * 2 + 1) % n_pool;
    expert_gate[k] = gate_w[pool_idx].data();
    expert_up[k] = up_w[pool_idx].data();
    expert_down[k] = down_w[pool_idx].data();
    expert_weight[k] = raw_weights[k];
  }

  // --- Reference: today's per-expert-call convention, fresh buffers per k ---
  std::vector<float> ref_out(D, 0.0f);
  for (int k = 0; k < k_used; ++k) {
    std::vector<float> gate_buf(moe_inter), up_buf(moe_inter), down_buf(D);
    matmul(gate_buf.data(), hidden.data(), expert_gate[k], 1, D, moe_inter);
    matmul(up_buf.data(), hidden.data(), expert_up[k], 1, D, moe_inter);
    silu_inplace(gate_buf.data(), moe_inter);
    elem_mul(gate_buf.data(), gate_buf.data(), up_buf.data(), moe_inter);
    matmul(down_buf.data(), gate_buf.data(), expert_down[k], 1, moe_inter, D);
    for (int d = 0; d < D; ++d)
      ref_out[d] += down_buf[d] * expert_weight[k];
  }

  // --- Grouped path: one call, shared/reused scratch buffers ---
  std::vector<float> gate_scratch(moe_inter), up_scratch(moe_inter), down_scratch(D);
  std::vector<float> batched_out(D, 0.0f);
  moe_decode_expert_ffn_batched(hidden.data(), batched_out.data(),
                                expert_gate.data(), expert_up.data(),
                                expert_down.data(), expert_weight.data(),
                                k_used, D, moe_inter, gate_scratch.data(),
                                up_scratch.data(), down_scratch.data());

  double max_abs_diff = 0.0;
  for (int d = 0; d < D; ++d)
    max_abs_diff = std::max(max_abs_diff, (double)std::fabs(batched_out[d] - ref_out[d]));
  assert(max_abs_diff == 0.0);

  // Accumulate convention: calling again on a buffer that already holds a
  // prior contribution (e.g. a shared-expert term added before this call,
  // matching moe_forward_gguf's own out_row[d] += ... convention) must ADD,
  // never overwrite.
  std::vector<float> preloaded(D);
  fill(preloaded);
  // Built with the SAME left-to-right accumulation order the function
  // itself uses (preloaded, then each k's contribution in turn) -- float
  // addition isn't associative, so "preloaded[d] + ref_out[d]" (which adds
  // ref_out's own already-summed total in one shot) is only approximately,
  // not bit-exactly, equal to accumulating each term in this order.
  std::vector<float> expect_sum = preloaded;
  for (int k = 0; k < k_used; ++k) {
    matmul(gate_scratch.data(), hidden.data(), expert_gate[k], 1, D, moe_inter);
    matmul(up_scratch.data(), hidden.data(), expert_up[k], 1, D, moe_inter);
    silu_inplace(gate_scratch.data(), moe_inter);
    elem_mul(gate_scratch.data(), gate_scratch.data(), up_scratch.data(), moe_inter);
    matmul(down_scratch.data(), gate_scratch.data(), expert_down[k], 1, moe_inter, D);
    for (int d = 0; d < D; ++d)
      expect_sum[d] += down_scratch[d] * expert_weight[k];
  }
  moe_decode_expert_ffn_batched(hidden.data(), preloaded.data(),
                                expert_gate.data(), expert_up.data(),
                                expert_down.data(), expert_weight.data(),
                                k_used, D, moe_inter, gate_scratch.data(),
                                up_scratch.data(), down_scratch.data());
  for (int d = 0; d < D; ++d)
    assert(preloaded[d] == expect_sum[d]);

  // k_used=1 edge case (a token that routed to only one expert this step).
  std::vector<float> single_ref(D, 0.0f), single_batched(D, 0.0f);
  matmul(gate_scratch.data(), hidden.data(), expert_gate[0], 1, D, moe_inter);
  matmul(up_scratch.data(), hidden.data(), expert_up[0], 1, D, moe_inter);
  silu_inplace(gate_scratch.data(), moe_inter);
  elem_mul(gate_scratch.data(), gate_scratch.data(), up_scratch.data(), moe_inter);
  matmul(down_scratch.data(), gate_scratch.data(), expert_down[0], 1, moe_inter, D);
  for (int d = 0; d < D; ++d)
    single_ref[d] = down_scratch[d] * expert_weight[0];
  moe_decode_expert_ffn_batched(hidden.data(), single_batched.data(),
                                expert_gate.data(), expert_up.data(),
                                expert_down.data(), expert_weight.data(),
                                /*k_used=*/1, D, moe_inter, gate_scratch.data(),
                                up_scratch.data(), down_scratch.data());
  for (int d = 0; d < D; ++d)
    assert(single_batched[d] == single_ref[d]);

  std::cout << " PASSED\n";
}

void test_tokenizer_metadata_shard() {
  char dir[] = "/tmp/zeromoe-tokenizer-shards-XXXXXX";
  assert(mkdtemp(dir));
  const std::string root(dir);
  const auto empty = root + "/00002.gguf";
  const auto metadata = root + "/repacked-00001.gguf";
  auto write = [](const std::string &path, bool vocabulary) {
    std::ofstream out(path, std::ios::binary);
    auto u32 = [&](uint32_t n) { out.write(reinterpret_cast<char *>(&n), 4); };
    auto u64 = [&](uint64_t n) { out.write(reinterpret_cast<char *>(&n), 8); };
    auto str = [&](const std::string &s) { u64(s.size()); out.write(s.data(), s.size()); };
    out.write("GGUF", 4); u32(3); u64(0); u64(vocabulary ? 2 : 1);
    str("tokenizer.ggml.eos_token_id"); u32(4); u32(vocabulary ? 2 : 99);
    if (vocabulary) {
      str("tokenizer.ggml.tokens"); u32(9); u32(8); u64(3);
      str("a"); str("b"); str("c");
    }
  };
  write(empty, false); write(metadata, true);
  GGUFFile file;
  file.resolved_file_path = empty;
  file.shard_paths = {empty, metadata};
  GGUFTokenizer tokenizer;
  assert(tokenizer.load(file));
  assert(tokenizer.id_to_token.size() == 3);
  assert(tokenizer.id_to_token[0] == "a");
  assert(tokenizer.eos_token_id == 2);
  assert(std::find(tokenizer.stop_ids.begin(), tokenizer.stop_ids.end(), 99) ==
         tokenizer.stop_ids.end()); // Failed shard metadata must not leak.
  file.shard_paths = {empty};
  GGUFTokenizer missing;
  assert(!missing.load(file));
  std::filesystem::remove_all(root);
  std::cout << "Tokenizer metadata shard fallback PASSED.\n";
}

int main(int argc, char **argv) {
  std::cout << "=== ZeroMoE Component Tests ===\n";
  struct TestEntry {
    const char *name;
    void (*fn)();
  };
  const TestEntry tests[] = {
      {"reasoning_parser", test_reasoning_parser},
      {"rope", test_rope},
      {"dsv4_rope", test_dsv4_rope},
      {"sinkhorn", test_sinkhorn_normalize},
      {"sqrtsoftplus", test_sqrtsoftplus_gating},
      {"swa_ring_index", test_swa_ring_index},
      {"dsv4_gpu_ring_compact", test_dsv4_gpu_ring_compact},
      {"dsv4_compression_metadata", test_dsv4_compression_metadata},
      {"dsv4_compressed_state", test_dsv4_compressed_state_helpers},
      {"tid2eid_lookup", test_tid2eid_lookup},
      {"topk_selection", test_topk_selection},
      {"option_b_global_pool", test_option_b_global_pool},
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
      {"trace_filter", test_trace_filter},
      // NOTE: registered BEFORE safetensors_adapter on purpose -- that test
      // currently aborts (assert) on a pre-existing, unrelated failure, and
      // registration order is execution order here.
      {"qwen4exp_layer_detection", test_qwen4exp_layer_detection},
      {"qwen4exp_gdn_delta_step", test_qwen4exp_gdn_delta_step},
      {"qwen4exp_gdn_delta_chunk", test_qwen4exp_gdn_delta_chunk},
      {"qwen4exp_l2_norm", test_qwen4exp_l2_norm},
      {"qwen4exp_ple_hash", test_qwen4exp_ple_hash},
      {"host_budget_admission", test_host_budget_admission},
      {"dsv41_router_stable_boundary", test_dsv41_router_stable_boundary},
      {"tokenizer_metadata_shard", test_tokenizer_metadata_shard},
      {"moe_decode_batch", test_moe_decode_batch},
      {"safetensors_adapter", test_safetensors_adapter},
      {"exl3_mul1_matmul", test_exl3_mul1_matmul},
      {"exl3_ngram_reader", test_exl3_ngram_reader},
      {"exl3_cpu_matrix_cache", test_exl3_cpu_matrix_cache},
      {"exl3_parallel_shard_reader", test_exl3_parallel_shard_reader},
      {"exllamav3_backend_launch", test_exllamav3_backend_launch},
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
