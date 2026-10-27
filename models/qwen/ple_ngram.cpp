// ple_ngram.cpp — Qwen4Exp (Qwen3.8-Flash-Next) PLE n-gram hash embedding.
//
// Reference: llama.cpp branch qwen4exp/qwen3.8-flash-next
// (src/models/qwen4exp.cpp's build_ple + llm_graph_input_ple::set_input,
// conversion/qwen4exp.py). PLE gives each token, at select layers, an extra
// n-gram-hash-keyed embedding contribution:
//   mixed_n = (t[p]*m[0]) ^ ... ^ (t[p-n+1]*m[n-1]);  row = mixed_n % vocab[h] + offset[h]
// (uint64 math, host-side -- ggml has no int64/xor; same here). EOS resets
// the n-gram window. The gathered rows feed a key/value pair of projections,
// a signed-sqrt sigmoid gate against the HC stream state, and a dilated
// causal conv (dilation = ngram_size) with a persistent per-layer history;
// the result is added back into the HC stream buffer BEFORE the attention
// sub-block's HC mix (wired via FamilyOps::layer_input_fn, called from
// layers_forward_gguf).
//
// The gather table (per_layer_token_embd.weight) is far too large for the
// dense-resident set -- it is NOT preloaded (graph_initializer.cpp's
// manifest deliberately omits it); rows are gathered straight out of an
// mmap of the shard with MADV_RANDOM, mirroring the llama.cpp branch's own
// "random-access mmap advice for host-resident gather tables" work.
//
// Single-sequence only (matches this engine's single-session generation
// loop): llama's per-sequence PLE history map collapses to one
// (toks, next_pos) pair on the engine.
//
// UNVERIFIED ON REAL HARDWARE (new model-family bring-up): validate against
// a real Qwen3.8-Flash-Next GGUF with a known prompt before trusting output.

#include "engine.hpp"
#include "graph_config.hpp"
#include "dense_dispatch.hpp"
#include "kquant_dequant.hpp"
#include "ple_hash.hpp" // the REAL n-gram hash (pure, unit-tested)
#include "tensor_trace.hpp"

#include <cmath>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <atomic>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <unordered_set>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

bool ple_timing_enabled() {
  static const bool enabled = [] {
    const char *v = zeromoe::graphcfg::env("ZEROMOE_QWEN4EXP_TIMING");
    return v && v[0] == '1';
  }();
  return enabled;
}

// Opt-in A/A snapshots for the host PLE implementation.  These use the same
// boundary selector as execution_layer.cpp, but remain local to this TU so
// the normal PLE path has no dependency on diagnostic plumbing.  The files
// are only written when the existing prefill boundary dump is enabled.
bool ple_diag_selected(int layer, int pos) {
  static const char *base = zeromoe::graphcfg::env("ZEROMOE_PREFILL_BOUNDARY_DUMP");
  static const int selected_layer = [] {
    const char *v = zeromoe::graphcfg::env("ZEROMOE_PREFILL_BOUNDARY_LAYER");
    return v && *v ? std::atoi(v) : 0;
  }();
  static const int first_pos = [] {
    const char *v = zeromoe::graphcfg::env("ZEROMOE_PREFILL_BOUNDARY_FIRST_POS");
    return v && *v ? std::atoi(v) : 0;
  }();
  static const int last_pos = [] {
    const char *v = zeromoe::graphcfg::env("ZEROMOE_PREFILL_BOUNDARY_LAST_POS");
    return v && *v ? std::atoi(v) : -1;
  }();
  return base && *base && (selected_layer < 0 || selected_layer == layer) &&
         pos >= first_pos && pos <= last_pos;
}

void ple_diag_dump(const char *label, int layer, int pos, const float *data,
                   size_t count) {
  if (!data || count == 0 || !ple_diag_selected(layer, pos))
    return;
  const char *base = zeromoe::graphcfg::env("ZEROMOE_PREFILL_BOUNDARY_DUMP");
  std::string path = std::string(base) + ".l" +
                     (layer < 10 ? "0" : "") + std::to_string(layer) +
                     ".p" + std::to_string(pos) + "." + label + ".f32";
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out)
    return;
  out.write(reinterpret_cast<const char *>(data),
            static_cast<std::streamsize>(count * sizeof(float)));
  LOG_INFO("[PLE-DIAG] layer=%d pos=%d label=%s elements=%zu path=%s\n",
           layer, pos, label, count, path.c_str());
}

// Lazily mmap'd PLE gather table (process-lifetime, like the KDA/GDN conv
// weight caches in attention_layer.cpp). Not unloaded at exit.
const uint8_t *g_ple_table = nullptr; // start of the tensor's rows
size_t g_ple_row_bytes = 0;
GGUFType g_ple_type = GGUFType::F32;
int g_ple_row_dim = 0;
bool g_ple_init_attempted = false;

// The PLE table is quantized row-wise and decode usually requests only 16
// rows.  Consecutive n-gram windows can revisit rows, especially for
// repetitive code/list output.  Keep a small process-local cache of the
// already dequantized rows so those hits avoid another mmap read and IQ4_NL
// decode.  This is deliberately bounded and Qwen/PLE-local; it is not part of
// the model-wide expert cache and cannot affect other model families.
constexpr size_t kPleRowCacheCapacity = 4096;
std::mutex g_ple_row_cache_mutex;
std::unordered_map<uint64_t, std::vector<float>> g_ple_row_cache;
std::atomic<uint64_t> g_ple_row_cache_hits{0};
std::atomic<uint64_t> g_ple_row_cache_misses{0};

bool ple_row_cache_enabled() {
  static const bool enabled = [] {
    const char *v = zeromoe::graphcfg::env("ZEROMOE_QWEN4EXP_PLE_ROW_CACHE");
    return !v || v[0] != '0';
  }();
  return enabled;
}

bool ple_dequant_row_cached(uint64_t row, float *out) {
  if (!ple_row_cache_enabled()) {
    const uint8_t *src =
        g_ple_table + row * static_cast<uint64_t>(g_ple_row_bytes);
    return dequant_row(g_ple_type, src, out, g_ple_row_dim);
  }
  {
    std::lock_guard<std::mutex> lock(g_ple_row_cache_mutex);
    auto it = g_ple_row_cache.find(row);
    if (it != g_ple_row_cache.end()) {
      std::memcpy(out, it->second.data(),
                  static_cast<size_t>(g_ple_row_dim) * sizeof(float));
      g_ple_row_cache_hits.fetch_add(1, std::memory_order_relaxed);
      return true;
    }
  }

  const uint8_t *src =
      g_ple_table + row * static_cast<uint64_t>(g_ple_row_bytes);
  if (!dequant_row(g_ple_type, src, out, g_ple_row_dim))
    return false;

  g_ple_row_cache_misses.fetch_add(1, std::memory_order_relaxed);
  {
    std::lock_guard<std::mutex> lock(g_ple_row_cache_mutex);
    if (g_ple_row_cache.size() < kPleRowCacheCapacity) {
      std::vector<float> copy(static_cast<size_t>(g_ple_row_dim));
      std::memcpy(copy.data(), out,
                  static_cast<size_t>(g_ple_row_dim) * sizeof(float));
      g_ple_row_cache.emplace(row, std::move(copy));
    }
  }
  return true;
}

bool ple_table_init(StreamingEngine &engine) {
  if (g_ple_init_attempted)
    return g_ple_table != nullptr;
  g_ple_init_attempted = true;

  auto it = engine.model_file.tensors.find("per_layer_token_embd.weight");
  if (it == engine.model_file.tensors.end()) {
    LOG_ERROR("[PLE] per_layer_token_embd.weight not found in GGUF -- a file "
              "with ple.layers MUST carry the n-gram table.\n");
    return false;
  }
  const GGUFTensor &t = it->second;
  if (t.shape.size() != 2) {
    LOG_ERROR("[PLE] per_layer_token_embd.weight: expected 2-D [row_dim, "
              "n_rows], got %zu-D.\n", t.shape.size());
    return false;
  }
  const int row_dim = (int)t.shape[0];
  const uint64_t n_rows = t.shape[1];
  auto &c = engine.model_file.architecture_config;
  if (row_dim != c.ple_n_embd_per_layer) {
    LOG_ERROR("[PLE] per_layer_token_embd row_dim=%d != "
              "embedding_length_per_layer_input=%d -- metadata mismatch.\n",
              row_dim, c.ple_n_embd_per_layer);
    return false;
  }
  if (t.shard_idx >= (int)engine.model_file.shard_paths.size()) {
    LOG_ERROR("[PLE] per_layer_token_embd: shard_idx=%d out of range (%zu "
              "shards).\n", t.shard_idx, engine.model_file.shard_paths.size());
    return false;
  }
  const std::string &path = engine.model_file.shard_paths[(size_t)t.shard_idx];
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    LOG_ERROR("[PLE] cannot open shard %s for mmap: %s\n", path.c_str(),
              std::strerror(errno));
    return false;
  }
  // t.offset is already the absolute in-shard offset (adjusted by the loader
  // -- gguf_loader.hpp's data_start fixup).
  const uint64_t map_align = 4096;
  const uint64_t base = t.offset & ~(map_align - 1);
  const uint64_t delta = t.offset - base;
  const uint64_t map_len = delta + t.size_bytes;
  void *map = mmap(nullptr, map_len, PROT_READ, MAP_SHARED, fd, (off_t)base);
  ::close(fd);
  if (map == MAP_FAILED) {
    LOG_ERROR("[PLE] mmap of %s (offset=%llu len=%llu) failed: %s\n",
              path.c_str(), (unsigned long long)base,
              (unsigned long long)map_len, std::strerror(errno));
    return false;
  }
  // Random-access advice: the gather touches one row per n-gram head per
  // token, no two on the same page -- exactly the access pattern the
  // llama.cpp branch advises this way too.
  madvise(map, map_len, MADV_RANDOM);

  // Row stride. This used to be an F32/F16/BF16-only hand-rolled byte
  // multiplication that hard-refused anything else ("a quantized table needs
  // a block-aware accessor, not byte multiplication") -- correct as a guard,
  // but it made a real, shipping model unusable: the Unsloth qwen4exp
  // (Qwen3.8-Flash-Next) UD-Q4_K_XL checkpoint stores per_layer_token_embd
  // as IQ4_NL (28.8 GB, 320M rows x 160), and this returned false, which the
  // caller turns into a SILENT `return` -- one ERROR line at the first
  // token, then layer 1's entire PLE contribution quietly dropped for the
  // rest of the process.
  //
  // The block-aware accessor it asks for already existed in two pieces:
  // row_bytes() (kquant_dequant.hpp, already included here, already carries
  // a correct IQ4_NL entry) for the stride, and the gather loop below
  // already calls the fully generic dequant_row() rather than casting the
  // row to float -- so nothing downstream assumed a float layout. Only this
  // switch did.
  //
  // Two guards replace the type whitelist, both cheap and both stronger than
  // it was:
  //   1. rb * n_rows == t.size_bytes. row_bytes() floor-divides (I/blk)*sz,
  //      so a row_dim that is NOT a whole number of blocks silently
  //      truncates; this equality catches exactly that, because the loader's
  //      own calculate_size() computes over the FULL element count and would
  //      then disagree. It is also an independent cross-check that our
  //      stride matches the file's real layout at all. (For this model:
  //      160/32*18 = 90 bytes/row, 90 * 320001536 == t.size_bytes exactly.)
  //   2. a trial dequant_row() of row 0 -- fails loudly HERE, at init, if
  //      ggml has no dequant for the type, instead of at the first gather
  //      deep inside the per-token loop.
  const int rb_i = row_bytes(t.type, row_dim);
  if (rb_i <= 0) {
    LOG_ERROR("[PLE] per_layer_token_embd type=%d: row_bytes() has no block "
              "size for it (see kquant_dequant.hpp).\n", (int)t.type);
    return false;
  }
  // Named row_stride, not row_bytes: row_bytes() is the helper function
  // called just above, and a local of the same name in the same scope
  // would shadow it for any future call added below here.
  const size_t row_stride = (size_t)rb_i;
  if (row_stride * n_rows != t.size_bytes) {
    LOG_ERROR("[PLE] per_layer_token_embd type=%d: computed row stride %zu B "
              "x %llu rows = %llu B, but the tensor is %llu B. Either "
              "row_dim=%d is not a whole number of %s blocks (row_bytes() "
              "floor-divides) or the file layout is not plain row-major.\n",
              (int)t.type, row_stride, (unsigned long long)n_rows,
              (unsigned long long)(row_stride * n_rows),
              (unsigned long long)t.size_bytes, row_dim,
              gpu_type_name(t.type));
    return false;
  }
  g_ple_table = (const uint8_t *)map + delta;
  g_ple_row_bytes = row_stride;
  g_ple_type = t.type;
  g_ple_row_dim = row_dim;

  // Guard 2 (see the row-stride comment above): prove ggml can actually
  // dequantize this type BEFORE the first gather, so an unsupported table
  // fails here, once, loudly -- not per-token from inside the hot loop.
  {
    std::vector<float> probe((size_t)row_dim);
    if (!dequant_row(t.type, g_ple_table, probe.data(), row_dim)) {
      LOG_ERROR("[PLE] per_layer_token_embd type=%d (%s): row_bytes() knows "
                "its stride but ggml_dequant_row has no dequant case for it "
                "(src/ggml_dequant.cpp) -- cannot gather.\n",
                (int)t.type, gpu_type_name(t.type));
      munmap(map, map_len);
      g_ple_table = nullptr;
      return false;
    }
  }
  LOG_STREAM_INFO() << "[PLE] per_layer_token_embd mmap'd: " << n_rows
                    << " rows x " << row_dim << " (" << t.size_bytes / 1e9
                    << " GB, type=" << (uint32_t)t.type
                    << ", MADV_RANDOM) -- gathered on demand, NOT resident.\n";
  return true;
}

// Dequant a small resident tensor to F32. Mirrors attention_layer.cpp's
// attn_dequant_tensor_f32 (static there, so duplicated here -- same
// "per-TU local helper" convention this codebase uses for check_finite*).
bool ple_dequant_tensor(StreamingEngine &engine, const std::string &name,
                        std::vector<float> &out, size_t expected_numel) {
  auto it = engine.resident_layers.find(name);
  if (it == engine.resident_layers.end() || !it->second.host_ptr) {
    LOG_ERROR("[PLE] missing tensor %s\n", name.c_str());
    return false;
  }
  uint64_t real = 1;
  for (auto d : it->second.meta.shape)
    real *= d;
  if (real != expected_numel) {
    LOG_ERROR("[PLE] %s: real numel=%llu != expected=%zu -- metadata "
              "mismatch, refusing to read past the tensor.\n",
              name.c_str(), (unsigned long long)real, expected_numel);
    return false;
  }
  out.resize(expected_numel);
  dequant_row(it->second.meta.type, it->second.host_ptr, out.data(),
              (int)expected_numel);
  return true;
}

// Grouped RMS norm over the [hc_dim] layout: normalize each D-wide stream
// independently (no inner gain), then multiply the flattened [hc_dim] by w.
// Mirrors qwen4exp.cpp's grouped_norm lambda inside build_ple (same math as
// the HC mix's norm step in hyper_connections.cpp).
void ple_grouped_norm(const float *x, const float *w, float *out, int S,
                      int hc_dim, int D, float eps) {
  const int HC = hc_dim / D;
#pragma omp parallel for if (S > 1) schedule(static)
  for (int s = 0; s < S; ++s) {
    const float *xs = x + (size_t)s * hc_dim;
    float *os = out + (size_t)s * hc_dim;
    for (int h = 0; h < HC; ++h) {
      double ss = 0.0;
      for (int d = 0; d < D; ++d)
        ss += (double)xs[(size_t)h * D + d] * xs[(size_t)h * D + d];
      float inv = 1.0f / std::sqrt((float)(ss / (double)D) + eps);
      for (int d = 0; d < D; ++d)
        os[(size_t)h * D + d] = xs[(size_t)h * D + d] * inv;
    }
  }
#pragma omp parallel for if (S > 1) schedule(static)
  for (int s = 0; s < S; ++s)
    for (int i = 0; i < hc_dim; ++i)
      out[(size_t)s * hc_dim + i] *= w[i];
}

} // namespace

bool qwen4exp_ple_gather_rows(StreamingEngine &engine,
                              const std::vector<int32_t> &rows,
                              std::vector<float> &out) {
  const auto start = std::chrono::steady_clock::now();
  if (rows.empty() || !ple_table_init(engine) || g_ple_row_dim <= 0)
    return false;

  const size_t row_dim = static_cast<size_t>(g_ple_row_dim);
  out.resize(rows.size() * row_dim);
  std::atomic<bool> gather_ok{true};
  auto gather_one = [&](int i, std::vector<float> &rowbuf) {
    const int64_t row = rows[static_cast<size_t>(i)];
    if (row < 0) {
      gather_ok.store(false, std::memory_order_relaxed);
      return;
    }
    float *dst = out.data() + static_cast<size_t>(i) * row_dim;
    if (!ple_dequant_row_cached(static_cast<uint64_t>(row), rowbuf.data())) {
      gather_ok.store(false, std::memory_order_relaxed);
      std::memset(dst, 0, row_dim * sizeof(float));
    } else {
      std::memcpy(dst, rowbuf.data(), row_dim * sizeof(float));
    }
  };

  // Decode supplies one token (normally 20 rows).  Starting an OpenMP team
  // for such a small gather costs more than the dequantization itself and
  // adds another host scheduling pulse to every token.  Keep the prefill
  // case parallel, where there are hundreds of independent rows.
  if (rows.size() <= 32) {
    std::vector<float> rowbuf(row_dim);
    for (int i = 0; i < static_cast<int>(rows.size()); ++i)
      gather_one(i, rowbuf);
  } else {
#pragma omp parallel
    {
      std::vector<float> rowbuf(row_dim);
#pragma omp for schedule(static)
      for (int i = 0; i < static_cast<int>(rows.size()); ++i)
        gather_one(i, rowbuf);
    }
  }
  const bool ok = gather_ok.load(std::memory_order_relaxed);
  if (ple_timing_enabled()) {
    size_t unique_rows = 0;
    std::unordered_set<int32_t> seen;
    seen.reserve(rows.size());
    for (int32_t row : rows)
      unique_rows += seen.insert(row).second ? 1u : 0u;
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - start)
                          .count();
    LOG_INFO("[PLE-TIME] rows=%zu unique=%zu gather=%.3f ms ok=%s cache_hits=%llu cache_misses=%llu\n",
             rows.size(), unique_rows, ms, ok ? "yes" : "no",
             (unsigned long long)g_ple_row_cache_hits.load(std::memory_order_relaxed),
             (unsigned long long)g_ple_row_cache_misses.load(std::memory_order_relaxed));
  }
  return ok;
}

// FamilyOps::layer_input_fn for Qwen4Exp. Runs at the top of a PLE layer,
// BEFORE the attention hc_pre mix; modifies engine.hc_stream_buf in place.
void qwen4exp_ple_layer_input(StreamingEngine &engine, int layer_idx, int S,
                              int pos_base, const int *token_ids) {
  auto &c = engine.model_file.architecture_config;
  if (!qwen4exp_is_ple_layer(c, layer_idx))
    return;

  const int D = c.hidden;
  const int HC = c.hc_mult;
  const int hc_dim = HC * D;
  const int n_gram = c.ple_ngram_size;
  const int n_heads = c.ple_n_heads;
  const int per_gram = c.ple_heads_per_ngram;
  const int head_dim = c.ple_n_embd_per_layer;
  const int64_t eos = c.ple_eos_token_id;
  // An embeddings-only batch (multimodal) has no token ids; stand in the
  // configured image token id, as the reference does, or EOS if unset.
  const int32_t img_tok = c.ple_image_token_id != 0 ? c.ple_image_token_id
                                                    : (int32_t)eos;
  const std::string p = "blk." + std::to_string(layer_idx) + ".";

  if ((int)engine.hc_stream_buf.size() < S * hc_dim) {
    LOG_ERROR("[PLE] layer=%d: hc_stream_buf too small (%zu < %d) -- HC "
              "expand must run before the layer loop.\n",
              layer_idx, engine.hc_stream_buf.size(), S * hc_dim);
    return;
  }
  if (!ple_table_init(engine))
    return;

  // ---- 1. n-gram hash -> gather-row indices (host-side; uint64 math) ----
  // Token history across calls: engine.qwen4exp_ple_tok_hist (most recent
  // LAST) + qwen4exp_ple_next_pos; a position discontinuity EOS-pads the
  // window (llama-memory-hybrid-idx's ple_hist reset).
  std::vector<int32_t> rows((size_t)S * n_heads);
  for (int i = 0; i < S; ++i) {
    const int64_t pos = pos_base + i;
    const int32_t tok = token_ids ? token_ids[i] : img_tok;

    if (engine.qwen4exp_ple_next_pos != pos) {
      // Non-contiguous with the processed history: reset the window.
      std::fill(engine.qwen4exp_ple_tok_hist.begin(),
                engine.qwen4exp_ple_tok_hist.end(), (int32_t)eos);
      engine.qwen4exp_ple_next_pos = pos;
    }

    // predecessor s (1-based) of this token, EOS past a boundary; in-batch
    // predecessors come first (positions within one call are consecutive),
    // then the cross-call history.
    auto prev = [&](int s) -> int64_t {
      const int j = i - s;
      if (j >= 0)
        return token_ids ? (int64_t)token_ids[j] : (int64_t)img_tok;
      const int64_t back = s - i; // how far before this call's first token
      const auto &h = engine.qwen4exp_ple_tok_hist;
      const int64_t k = (int64_t)h.size() - back; // most recent last
      if (back > 0 && k >= 0 && k < (int64_t)h.size())
        return (int64_t)h[(size_t)k];
      return eos;
    };

    // An EOS in the window resets everything at or before it; the token's
    // OWN eos does not cut its own context (reference behavior).
    std::vector<int64_t> ctx((size_t)n_gram);
    ctx[0] = tok;
    bool cut = false;
    for (int s = 1; s < n_gram; ++s) {
      ctx[(size_t)s] = cut ? eos : prev(s);
      if (ctx[(size_t)s] == eos)
        cut = true;
    }

    for (int n = 2; n <= n_gram; ++n) {
      const int base = (n - 2) * per_gram;
      for (int g = 0; g < per_gram; ++g) {
        const int h_i = base + g;
        rows[(size_t)i * n_heads + h_i] = (int32_t)ple_ngram_row(
            ctx.data(), n, c.ple_layer_multipliers.data(),
            c.ple_head_vocab_sizes[(size_t)h_i], c.ple_head_offsets[(size_t)h_i]);
      }
    }

    // push this token into the cross-call history
    auto &h = engine.qwen4exp_ple_tok_hist;
    h.push_back(tok);
    if ((int64_t)h.size() > n_gram - 1)
      h.erase(h.begin(), h.end() - (n_gram - 1));
    engine.qwen4exp_ple_next_pos = pos + 1;
  }

  // ---- 2. gather rows out of the mmap'd table -> emb [S, n_heads*head_dim]
  // (head-inner per token, matching the reference's get_rows + reshape) ----
  std::vector<float> emb((size_t)S * n_heads * head_dim);
  {
    // Each table row is independent.  The old serial loop made the single
    // PLE layer a long CPU barrier during prefill: at the real 548-token
    // shape this random IQ4_NL gather was 1.45 s of the 7.12 s pass.  Keep
    // the mmap and exact row selection unchanged, but give each worker its
    // own dequant scratch so dequant_row never shares mutable state.
    std::atomic<bool> gather_ok{true};
#pragma omp parallel
    {
      std::vector<float> rowbuf((size_t)head_dim);
#pragma omp for collapse(2) schedule(static)
      for (int i = 0; i < S; ++i) {
        for (int h = 0; h < n_heads; ++h) {
        const int64_t r = rows[(size_t)i * n_heads + h];
        const uint8_t *src =
            g_ple_table + (uint64_t)r * (uint64_t)g_ple_row_bytes;
        if (!ple_dequant_row_cached((uint64_t)r, rowbuf.data())) {
          gather_ok.store(false, std::memory_order_relaxed);
          std::memset(emb.data() + ((size_t)i * n_heads + h) * head_dim,
                      0, (size_t)head_dim * sizeof(float));
        } else {
          std::memcpy(emb.data() + ((size_t)i * n_heads + h) * head_dim,
                      rowbuf.data(), (size_t)head_dim * sizeof(float));
        }
        }
      }
    }
    if (!gather_ok.load(std::memory_order_relaxed)) {
      LOG_ERROR("[PLE] layer=%d: table type %d not dequantizable.\n",
                layer_idx, (int)g_ple_type);
      return;
    }
  }
  ple_diag_dump("ple_embd", layer_idx, pos_base, emb.data(), emb.size());
  // ---- 3. key/value projections ----
  // emb per token is [n_heads * head_dim] == [hidden] wide (validated at
  // load: ple_n_embd_per_layer * ple_n_heads == hidden is required for the
  // ple_key matmul's input width; checked here defensively).
  if (n_heads * head_dim != D) {
    LOG_ERROR("[PLE] layer=%d: n_heads*head_dim (%d*%d=%d) != hidden (%d) -- "
              "ple_key's input width mismatch.\n",
              layer_idx, n_heads, head_dim, n_heads * head_dim, D);
    return;
  }
  std::vector<float> key((size_t)S * hc_dim), value((size_t)S * D);
  dense_proj(engine, p + "ple_key.weight", emb.data(), key.data(), S, D, hc_dim);
  dense_proj(engine, p + "ple_value.weight", emb.data(), value.data(), S, D, D);
  ple_diag_dump("ple_key", layer_idx, pos_base, key.data(), key.size());
  ple_diag_dump("ple_value", layer_idx, pos_base, value.data(), value.size());

  // ---- 4. grouped norms ----
  std::vector<float> norm_key_w, norm_query_w, norm_conv_w;
  if (!ple_dequant_tensor(engine, p + "ple_norm_key.weight", norm_key_w,
                          (size_t)hc_dim) ||
      !ple_dequant_tensor(engine, p + "ple_norm_query.weight", norm_query_w,
                          (size_t)hc_dim) ||
      !ple_dequant_tensor(engine, p + "ple_norm_conv.weight", norm_conv_w,
                          (size_t)hc_dim)) {
    return;
  }
  ple_grouped_norm(key.data(), norm_key_w.data(), key.data(), S, hc_dim, D, c.eps);
  // query: the HC stream state itself, normed
  std::vector<float> query((size_t)S * hc_dim);
  ple_grouped_norm(engine.hc_stream_buf.data(), norm_query_w.data(),
                   query.data(), S, hc_dim, D, c.eps);
  ple_diag_dump("ple_query", layer_idx, pos_base, query.data(), query.size());

  // ---- 5. signed-sqrt sigmoid gate; value broadcast across streams ----
  const float inv_sqrt_d = 1.0f / std::sqrt((float)D);
  std::vector<float> gated((size_t)S * hc_dim);
  std::vector<float> gate_values((size_t)S * HC);
#pragma omp parallel for if (S > 1) schedule(static)
  for (int s = 0; s < S; ++s) {
    const float *ks = key.data() + (size_t)s * hc_dim;
    const float *qs = query.data() + (size_t)s * hc_dim;
    const float *vs = value.data() + (size_t)s * D;
    float *gs = gated.data() + (size_t)s * hc_dim;
    for (int h = 0; h < HC; ++h)
    {
      // llama.cpp's ggml_sum_rows reduces the [D, HC, S] key*query tensor over
      // dimension 0, producing one signed-sqrt/sigmoid gate per HC stream.
      // The old host path accidentally reduced all HC*D values into one
      // scalar gate, then broadcast that scalar across the four streams.
      double dot = 0.0;
      for (int d = 0; d < D; ++d)
        dot += (double)ks[(size_t)h * D + d] * qs[(size_t)h * D + d];
      const float sc = (float)dot * inv_sqrt_d;
      const float mag = std::sqrt(std::fmax(std::fabs(sc), 1e-6f));
      const float gate =
          1.0f / (1.0f + std::exp(-std::copysign(mag, sc)));
      gate_values[(size_t)s * HC + h] = gate;
      for (int d = 0; d < D; ++d)
        gs[(size_t)h * D + d] = vs[d] * gate;
    }
  }
  ple_diag_dump("ple_gate", layer_idx, pos_base, gate_values.data(),
                gate_values.size());
  ple_diag_dump("ple_gated_value", layer_idx, pos_base, gated.data(),
                gated.size());

  // ---- 6. dilated causal conv over the token axis (dilation = ngram_size),
  // persistent per-layer history; silu on the output ----
  const int kern = c.ple_conv_kernel;
  const int dil = n_gram;
  const int hist = (kern - 1) * dil;
  std::vector<float> &conv_hist = engine.qwen4exp_ple_conv_history[(size_t)layer_idx];
  if ((int)conv_hist.size() != hc_dim * hist)
    conv_hist.assign((size_t)hc_dim * hist, 0.0f);
  std::vector<float> conv_w;
  if (!ple_dequant_tensor(engine, p + "ple_conv1d.weight", conv_w,
                          (size_t)hc_dim * kern)) {
    return;
  }
  // normalized = grouped_norm(gated) feeds the conv (NOT gated itself).
  std::vector<float> conv_in((size_t)S * hc_dim);
  ple_grouped_norm(gated.data(), norm_conv_w.data(), conv_in.data(), S, hc_dim, D, c.eps);
  ple_diag_dump("ple_conv_normed", layer_idx, pos_base, conv_in.data(),
                conv_in.size());
  // Opt-in parity snapshot of the recurrent state immediately before this
  // call's depthwise convolution. The existing position selector ensures
  // this adds no work or file output during ordinary inference.
  ple_diag_dump("ple_conv_history", layer_idx, pos_base, conv_hist.data(),
                conv_hist.size());

  std::vector<float> conv_out((size_t)S * hc_dim);
#pragma omp parallel for schedule(static)
  for (int ch = 0; ch < hc_dim; ++ch) {
    const float *h = conv_hist.data() + (size_t)ch * hist;
    const float *wc = conv_w.data() + (size_t)ch * kern;
    for (int s = 0; s < S; ++s) {
      float acc = 0.0f;
      for (int k = 0; k < kern; ++k) {
        // tap k reads (kern-1-k)*dilation positions back
        const int64_t tpos = (int64_t)s + hist - (int64_t)(kern - 1 - k) * dil;
        const float val =
            (tpos < hist) ? h[tpos]
                          : conv_in[(size_t)(tpos - hist) * hc_dim + ch];
        acc += val * wc[k];
      }
      conv_out[(size_t)s * hc_dim + ch] = acc / (1.0f + std::exp(-acc)); // silu
    }
  }
  ple_diag_dump("ple_conv_out", layer_idx, pos_base, conv_out.data(),
                conv_out.size());
  if (hist > 0) {
    // shift the window left by S and append this call's inputs
    std::vector<float> new_hist((size_t)hc_dim * hist);
#pragma omp parallel for schedule(static)
    for (int ch = 0; ch < hc_dim; ++ch) {
      const float *h = conv_hist.data() + (size_t)ch * hist;
      for (int j = 0; j < hist; ++j) {
        const int64_t tpos = (int64_t)S + j;
        new_hist[(size_t)ch * hist + j] =
            (tpos < hist) ? h[tpos]
                          : conv_in[(size_t)(tpos - hist) * hc_dim + ch];
      }
    }
    std::memcpy(conv_hist.data(), new_hist.data(),
                (size_t)hc_dim * hist * sizeof(float));
  }

  // ---- 7. res_hc += gated + conv_out ----
  ple_diag_dump("ple_hc_before", layer_idx, pos_base,
                engine.hc_stream_buf.data(), engine.hc_stream_buf.size());
  for (size_t i = 0; i < (size_t)S * hc_dim; ++i)
    engine.hc_stream_buf[i] += gated[i] + conv_out[i];
  ple_diag_dump("ple_hc_after", layer_idx, pos_base,
                engine.hc_stream_buf.data(), engine.hc_stream_buf.size());

  if (trace_gate("PLE")) {
    const size_t off = (size_t)(S - 1) * hc_dim;
    double g2 = 0.0, co2 = 0.0;
    for (int i = 0; i < hc_dim; ++i) {
      g2 += (double)gated[off + i] * gated[off + i];
      co2 += (double)conv_out[off + i] * conv_out[off + i];
    }
    LOG_STREAM_VERBOSE() << "[PLE-TRACE l=" << layer_idx << "] last token: "
                         << "|gated|=" << std::sqrt(g2)
                         << " |conv_out|=" << std::sqrt(co2) << "\n";
  }
}
