// DeepSeek-V4.1 conditional-memory Engram layer input.
//
// The converter keeps each enormous engram_embd table quantized in its GGUF
// shard. This file intentionally maps only the table's byte range and reads
// the small set of rows needed by the current token batch; no table is copied
// into the resident dense pool.
#include <unordered_set>
#include "engine.hpp"
#include "dense_dispatch.hpp"
#include "dsv41_engram.hpp"
#include "kquant_dequant.hpp"
#include "../graph/graph_settings.hpp"
#include "startup_preflight.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <fcntl.h>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <sys/mman.h>
#include <unistd.h>
#include <unordered_map>

namespace {

struct EngramTableMap {
  void *map_base = nullptr;
  size_t map_len = 0;
  const uint8_t *rows = nullptr;
  size_t row_bytes = 0;
  uint64_t n_rows = 0;
  // No-mmap mode: retain a descriptor for buffered positional row reads.
  std::string shard_path;
  uint64_t data_offset = 0;
  std::shared_ptr<int> direct_fd;
};

std::unordered_map<std::string, EngramTableMap> g_tables;
std::mutex g_tables_mutex;

// Small hot-row cache for the Engram tables. The tables themselves stay
// mmap'd on NVMe (map_table's own comment: "no table is copied into the
// resident dense pool") -- this does NOT change that. But decode re-hits the
// SAME small set of hash rows far more than a uniform-random access pattern
// would: the hash is built from a short, overlapping n-gram of recent
// tokens (hash_batch's own tok_hist window), so consecutive decode steps
// share most of that history and frequently rehash to the same row. Caching
// the DEQUANTIZED row avoids both the page fault on a cold mmap page and the
// dequant call on a repeat hit. Fixed, modest capacity (a few thousand
// entries at head_dim floats each is a few MB) -- deliberately not a config
// knob, matching "a small cache to keep the most used one" rather than a
// tunable RAM-tier promotion policy.
struct EngramRowKey {
  int table_id;
  int64_t row;
  bool operator==(const EngramRowKey &o) const {
    return table_id == o.table_id && row == o.row;
  }
};
struct EngramRowKeyHash {
  size_t operator()(const EngramRowKey &k) const {
    return std::hash<int64_t>()(((int64_t)k.table_id << 48) ^ k.row);
  }
};

class EngramRowCache {
public:
  // Runtime capacity (rows). Configurable via
  // ZEROMOE_DSV41_ENGRAM_CACHE_ROWS or ZEROMOE_DSV41_ENGRAM_CACHE_MB.
  // Defaults to 8192 rows.
  size_t capacity(int head_dim = 0) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (capacity_rows_ == 0) {
      long cache_mb = 0;
      try { cache_mb = zeromoe::graphsettings::active_settings().tun.engram_cache_mb.value; } catch (...) { cache_mb = 0; }
      const char *rows_env = std::getenv("ZEROMOE_DSV41_ENGRAM_CACHE_ROWS");
      const char *mb_env = std::getenv("ZEROMOE_DSV41_ENGRAM_CACHE_MB");
      if (rows_env && rows_env[0]) {
        capacity_rows_ = (size_t)std::strtoull(rows_env, nullptr, 10);
      } else if (cache_mb > 0 && head_dim > 0) {
        const size_t row_bytes = (size_t)head_dim * sizeof(float);
        if (row_bytes > 0) capacity_rows_ = ((size_t)cache_mb * 1024ULL * 1024ULL) / row_bytes;
      } else if (mb_env && mb_env[0] && head_dim > 0) {
        const size_t mb = (size_t)std::strtoull(mb_env, nullptr, 10);
        const size_t row_bytes = (size_t)head_dim * sizeof(float);
        if (row_bytes > 0) capacity_rows_ = (mb * 1024ULL * 1024ULL) / row_bytes;
      }
      if (capacity_rows_ == 0) capacity_rows_ = 8192;
    }
    return capacity_rows_;
  }
  static size_t capacity_static() { return 8192; }
  bool get(int table_id, int64_t row, int head_dim, float *out) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = map_.find({table_id, row});
    if (it == map_.end()) return false;
    order_.splice(order_.begin(), order_, it->second.order_it);
    std::memcpy(out, it->second.data.data(), (size_t)head_dim * sizeof(float));
    return true;
  }
  void put(int table_id, int64_t row, const float *data, int head_dim) {
    // Resolve capacity before taking mutex_: capacity() takes the same lock.
    // Calling it from inside this critical section deadlocks the first miss.
    const size_t effective_cap = capacity(head_dim);
    std::lock_guard<std::mutex> lock(mutex_);
    const EngramRowKey key{table_id, row};
    auto it = map_.find(key);
    if (it != map_.end()) {
      order_.splice(order_.begin(), order_, it->second.order_it);
      return;
    }
    if (map_.size() >= effective_cap) {
      const EngramRowKey victim = order_.back();
      order_.pop_back();
      map_.erase(victim);
    }
    order_.push_front(key);
    Entry &e = map_[key];
    e.data.assign(data, data + head_dim);
    e.order_it = order_.begin();
  }

private:
  // TESTED (2026-09-12): raised 8192->65536 (8x) to check whether the
  // real, measured 5.5% hit rate (545/9984 lookups on the standard
  // 4-prompt test) was a capacity problem. Confirmed NOT: hit rate moved
  // 5.5%->6.0%, well within noise for an 8x capacity increase -- genuine
  // low locality (most Engram rows are touched once per request and
  // essentially never revisited), not a sizing issue. `decode` tok/s was
  // also unaffected (2.02 vs the 2.09 checkpoint, within normal noise) --
  // consistent with this cache being too small a fraction of total
  // decode cost to matter even if hit rate had improved. Reverted to
  // 8192 -- a bigger cache buys nothing here, don't re-raise this
  // without new evidence the access pattern has changed (e.g. much
  // longer real conversations with genuine n-gram repetition, untested
  // this session). See [[deepseek-v41-decode-speed-session]] for the
  // full investigation.
  size_t capacity_rows_ = 0;
  struct Entry {
    std::vector<float> data;
    std::list<EngramRowKey>::iterator order_it;
  };
  std::list<EngramRowKey> order_;
  std::unordered_map<EngramRowKey, Entry, EngramRowKeyHash> map_;
  std::mutex mutex_;
};

EngramRowCache g_row_cache;
// TEMPORARY DIAGNOSTIC (2026-09-12): this cache has run with zero real
// visibility into its own hit rate since it was built -- the goal's own
// item 4 ("engram should keep in ssd/nvme and we can use a small cache
// to keep the most used one") is exactly what this class already does,
// but nobody has ever measured whether the fixed 8192-row capacity is
// anywhere close to right for a real generation, or whether the
// same-hash-window locality this class's own header comment predicts
// actually holds on real DeepSeek-V4.1 traffic. Cheap relaxed atomics,
// negligible per-lookup cost.
std::atomic<uint64_t> g_engram_cache_hits{0};
std::atomic<uint64_t> g_engram_cache_misses{0};

const EngramTableMap *map_table(StreamingEngine &engine, const GGUFFile &source,
                                const GGUFTensor &t, int row_dim) {
  if (t.shape.size() != 2 ||
      t.shape[0] != (uint64_t)row_dim || t.shard_idx < 0 ||
      (size_t)t.shard_idx >= source.shard_paths.size())
    return nullptr;
  const std::string shard_path = source.shard_paths[(size_t)t.shard_idx];
  const std::string key = shard_path + "#" + t.name;
  std::lock_guard<std::mutex> lock(g_tables_mutex);
  auto found = g_tables.find(key);
  if (found != g_tables.end()) return &found->second;

  // The converter's staging file uses Q8_0, but downstream Q2_K/Q3_K (and
  // other row-aligned GGUF quants) must remain usable too. Every supported
  // table type here has one independent quantized row of `row_dim` values.
  const size_t rb = row_bytes(t.type, row_dim);
  const uint64_t rows = t.shape[1];
  if (rb == 0 || rows > SIZE_MAX / rb || rows * rb > t.size_bytes)
    return nullptr;
  const long page = sysconf(_SC_PAGESIZE);
  const uint64_t align = page > 0 ? (uint64_t)page : 4096;
  const uint64_t base = t.offset & ~(align - 1);
  const uint64_t delta = t.offset - base;
  const uint64_t length = delta + rows * rb;
  if (length > SIZE_MAX || base > (uint64_t)std::numeric_limits<off_t>::max())
    return nullptr;

  const int fd = ::open(shard_path.c_str(), O_RDONLY);
  if (fd < 0) {
    LOG_ERROR("[DSV41-ENGRAM] cannot open shard %s: %s\n",
              shard_path.c_str(), std::strerror(errno));
    return nullptr;
  }

  const auto &gs = zeromoe::graphsettings::active_settings();
  long storage_mode = gs.tun.engram_storage_mode.value;
  if (storage_mode == -1) {
    double mem_avail_gb = preflight::system_info().mem_available_gb;
    if (mem_avail_gb < 0.0) mem_avail_gb = preflight::mem_available_gb();
    if (mem_avail_gb < 0.0) mem_avail_gb = 0.0;
    const double engram_gb = (double)length / 1.0e9;
    if (gs.tun.engram_cache_mb.value > 0) storage_mode = 1;
    else if (mem_avail_gb >= 0.4 * engram_gb + 1.0) storage_mode = 0;
    else storage_mode = 1;
  }
  const bool direct_requested = (storage_mode == 1);
  if (direct_requested) {
    EngramTableMap entry;
    entry.row_bytes = rb;
    entry.n_rows = rows;
    entry.shard_path = shard_path;
    entry.data_offset = t.offset;
    entry.direct_fd = std::shared_ptr<int>(
        new int(fd), [](int *open_fd) {
          if (*open_fd >= 0) ::close(*open_fd);
          delete open_fd;
        });
    auto inserted = g_tables.emplace(key, entry);
    LOG_STREAM_INFO() << "[DSV41-ENGRAM] on-demand pread " << t.name << ": " << rows
                      << " rows x " << row_dim
                      << " (buffered positional reads, no mmap)\n";
    return &inserted.first->second;
  }

  void *mapped = nullptr;
  mapped = ::mmap(nullptr, (size_t)length, PROT_READ, MAP_SHARED, fd,
                  (off_t)base);
  ::close(fd);
  if (mapped == MAP_FAILED) {
    LOG_ERROR("[DSV41-ENGRAM] mmap failed for %s: %s\n",
              t.name.c_str(), std::strerror(errno));
    return nullptr;
  }
#ifdef MADV_RANDOM
  madvise(mapped, (size_t)length, MADV_RANDOM);
#endif
  EngramTableMap entry;
  entry.map_base = mapped;
  entry.map_len = (size_t)length;
  entry.rows = static_cast<const uint8_t *>(mapped) + delta;
  entry.row_bytes = rb;
  entry.n_rows = rows;
  auto inserted = g_tables.emplace(key, entry);
  LOG_STREAM_INFO() << "[DSV41-ENGRAM] mmap " << t.name << ": " << rows
                    << " rows x " << row_dim << " (type "
                    << (unsigned)t.type << ", row gather)\n";
  return &inserted.first->second;
}

bool load_small_f32(StreamingEngine &engine, const std::string &name,
                    size_t expected, std::vector<float> &out) {
  auto it = engine.resident_layers.find(name);
  if (it == engine.resident_layers.end() || !it->second.host_ptr)
    return false;
  uint64_t n = 1;
  for (uint64_t dim : it->second.meta.shape) n *= dim;
  if (n != expected) return false;
  out.resize(expected);
  if (it->second.meta.type == GGUFType::F32) {
    std::memcpy(out.data(), it->second.host_ptr, expected * sizeof(float));
  } else {
    dequant_row(it->second.meta.type, it->second.host_ptr, out.data(),
                (int)expected);
  }
  return true;
}

int64_t mapped_token(const ArchitectureConfig &c, int token) {
  if (token < 0 || (size_t)token >= c.dsv41_engram_token_map.size())
    return -1;
  return c.dsv41_engram_token_map[(size_t)token];
}

bool hash_batch(StreamingEngine &engine, int S, int pos_base,
                const int *token_ids, std::vector<int64_t> &hashes) {
  const auto &c = engine.model_file.architecture_config;
  const int max_ngram = c.dsv41_engram_max_ngram_size;
  const int n_heads = c.dsv41_engram_n_heads;
  const int n_layers = (int)c.dsv41_engram_layer_ids.size();
  const int n_hash_cols = (max_ngram - 1) * n_heads;
  if (!token_ids || max_ngram < 2 || n_heads <= 0 || n_layers <= 0 ||
      c.dsv41_engram_primes.size() <
          (size_t)n_layers * (max_ngram - 1) * n_heads ||
      c.dsv41_engram_offsets.size() <
          (size_t)n_layers * (max_ngram - 1) * n_heads ||
      c.dsv41_engram_multipliers.size() < (size_t)n_layers * max_ngram)
    return false;

  // Position zero starts a new request in ZeroMoE; do not reuse a cached
  // batch from the previous request even though its position also was zero.
  if (pos_base != 0 && engine.dsv41_engram_hash_pos_base == pos_base &&
      engine.dsv41_engram_hash_batch == S &&
      !engine.dsv41_engram_hashes.empty()) {
    hashes = engine.dsv41_engram_hashes;
    return true;
  }

  if (pos_base == 0 || engine.dsv41_engram_next_pos != pos_base) {
    engine.dsv41_engram_tok_hist.clear();
    engine.dsv41_engram_next_pos = pos_base;
  }
  hashes.assign((size_t)S * n_layers * n_hash_cols, 0);
  const int64_t pad = c.dsv41_engram_pad_id;
  for (int s = 0; s < S; ++s) {
    const int64_t current = mapped_token(c, token_ids[s]);
    std::vector<int64_t> tokens((size_t)max_ngram, pad);
    bool blocked = current < 0;
    tokens[0] = blocked ? pad : current;
    for (int shift = 1; shift < max_ngram; ++shift) {
      const size_t hist = (size_t)shift - 1;
      const int64_t source = hist < engine.dsv41_engram_tok_hist.size()
                                 ? engine.dsv41_engram_tok_hist[
                                       engine.dsv41_engram_tok_hist.size() - 1 - hist]
                                 : -1;
      // Match NgramHashState: a DEAD token blocks this and every older
      // lookback, so no n-gram can bridge an image/dead-token boundary.
      blocked = blocked || source < 0;
      tokens[(size_t)shift] = blocked ? pad : source;
    }
    for (int li = 0; li < n_layers; ++li) {
      int64_t rolling = tokens[0] *
                        c.dsv41_engram_multipliers[(size_t)li * max_ngram];
      for (int n = 2; n <= max_ngram; ++n) {
        rolling ^= tokens[(size_t)n - 1] *
                   c.dsv41_engram_multipliers[(size_t)li * max_ngram + n - 1];
        const size_t table_base =
            ((size_t)li * (max_ngram - 1) + (n - 2)) * n_heads;
        const size_t out_base =
            ((size_t)s * n_layers + li) * n_hash_cols +
            (size_t)(n - 2) * n_heads;
        for (int h = 0; h < n_heads; ++h) {
          const int64_t prime = c.dsv41_engram_primes[table_base + h];
          const int64_t offset = c.dsv41_engram_offsets[table_base + h];
          if (prime <= 0) return false;
          hashes[out_base + h] = (rolling % prime) + offset;
        }
      }
    }
    if (current >= 0)
      engine.dsv41_engram_tok_hist.push_back(current);
    else
      engine.dsv41_engram_tok_hist.push_back(-1);
    const size_t keep = (size_t)max_ngram - 1;
    if (engine.dsv41_engram_tok_hist.size() > keep)
      engine.dsv41_engram_tok_hist.erase(
          engine.dsv41_engram_tok_hist.begin(),
          engine.dsv41_engram_tok_hist.end() - (ptrdiff_t)keep);
  }
  engine.dsv41_engram_next_pos = pos_base + S;
  engine.dsv41_engram_hashes = hashes;
  engine.dsv41_engram_hash_pos_base = pos_base;
  engine.dsv41_engram_hash_batch = S;
  return true;
}

} // namespace

// TEMPORARY DIAGNOSTIC (2026-09-12) -- see g_engram_cache_hits/misses'
// own comment above. Called from the shutdown report so real Engram
// row-cache effectiveness is visible without adding a per-request line.
void deepseek41_engram_print_cache_stats() {
  uint64_t hits = g_engram_cache_hits.load(std::memory_order_relaxed);
  uint64_t misses = g_engram_cache_misses.load(std::memory_order_relaxed);
  uint64_t total = hits + misses;
  if (total == 0) return;
  LOG_INFO("[DSV41-ENGRAM-CACHE] %llu hit / %llu miss (%.1f%% hit) of %llu "
           "row lookups, fixed capacity %zu rows\n",
           (unsigned long long)hits, (unsigned long long)misses,
           100.0 * (double)hits / (double)total, (unsigned long long)total,
           g_row_cache.capacity(0));
}

void deepseek41_engram_layer_input(StreamingEngine &engine, int layer_idx,
                                   int S, int pos_base, const int *token_ids) {
  auto &c = engine.model_file.architecture_config;
  if (!dsv41_is_engram_layer(c, layer_idx)) return;
  if (S <= 0 || c.hc_mult <= 1) return;
  const auto layer_it = std::find(c.dsv41_engram_layer_ids.begin(),
                                  c.dsv41_engram_layer_ids.end(),
                                  (int64_t)layer_idx);
  if (layer_it == c.dsv41_engram_layer_ids.end()) return;
  const int table_id = (int)(layer_it - c.dsv41_engram_layer_ids.begin());
  const int max_ngram = c.dsv41_engram_max_ngram_size;
  const int n_heads = c.dsv41_engram_n_heads;
  const int head_dim = c.dsv41_engram_head_dim;
  const int n_hash_cols = (max_ngram - 1) * n_heads;
  const int D = c.hidden;
  const int HC = c.hc_mult;
  if (table_id < 0 ||
      (!c.dsv41_engram_num_embeddings.empty() &&
       (size_t)table_id >= c.dsv41_engram_num_embeddings.size()) ||
      n_hash_cols <= 0 || head_dim <= 0)
    return;

  const std::string prefix = "blk." + std::to_string(layer_idx) + ".";
  const GGUFFile &table_file = engine.has_engram_model
                                  ? engine.engram_model_file
                                  : engine.model_file;
  auto table_it = table_file.tensors.find(prefix + "engram_embd.weight");
  if (table_it == table_file.tensors.end()) {
    LOG_ERROR("[DSV41-ENGRAM] missing table for layer %d\n", layer_idx);
    return;
  }
  std::vector<float> embed;
  if (!deepseek41_engram_embedding_rows(engine, layer_idx, S, pos_base,
                                       token_ids, embed))
    return;

  std::vector<float> kv((size_t)S * (HC + 1) * D);
  dense_proj(engine, prefix + "engram_kv.weight", embed.data(), kv.data(), S,
             n_hash_cols * head_dim, (HC + 1) * D);
  std::vector<float> gate_weight;
  if (!load_small_f32(engine, prefix + "engram_gate.weight",
                      (size_t)HC * D, gate_weight)) {
    LOG_ERROR("[DSV41-ENGRAM] missing fused gate weight at layer %d\n",
              layer_idx);
    return;
  }

  const float eps = c.eps;
  const float inv_sqrt_dim = 1.0f / std::sqrt((float)D);
  for (int s = 0; s < S; ++s) {
    const float *row = kv.data() + (size_t)s * (HC + 1) * D;
    const float *value = row + (size_t)HC * D;
    for (int h = 0; h < HC; ++h) {
      float *stream = engine.hc_stream_buf.data() + ((size_t)s * HC + h) * D;
      const float *key = row + (size_t)h * D;
      double hss = 0.0, kss = 0.0, dot = 0.0;
      for (int d = 0; d < D; ++d) {
        const float hv = stream[d];
      const float weight = gate_weight[(size_t)h * D + d];
        hss += (double)hv * hv;
        kss += (double)key[d] * key[d];
        dot += (double)hv * weight * key[d];
      }
      dot = dot / std::sqrt((float)(hss / D) + eps) /
            std::sqrt((float)(kss / D) + eps) * inv_sqrt_dim;
      const float signed_sqrt = std::copysign(
          std::sqrt(std::max(std::abs((float)dot), 1e-6f)), (float)dot);
      const float gate = 1.0f / (1.0f + std::exp(-signed_sqrt));
      for (int d = 0; d < D; ++d) stream[d] += gate * value[d];
    }
  }
}

bool deepseek41_engram_embedding_rows(StreamingEngine &engine, int layer_idx,
                                      int S, int pos_base,
                                      const int *token_ids,
                                      std::vector<float> &embedding) {
  const auto &c = engine.model_file.architecture_config;
  if (!dsv41_is_engram_layer(c, layer_idx)) {
    embedding.clear();
    return true;
  }
  const auto layer_it = std::find(c.dsv41_engram_layer_ids.begin(),
                                  c.dsv41_engram_layer_ids.end(),
                                  (int64_t)layer_idx);
  if (layer_it == c.dsv41_engram_layer_ids.end() || S <= 0 || !token_ids)
    return false;
  const int table_id = (int)(layer_it - c.dsv41_engram_layer_ids.begin());
  const int max_ngram = c.dsv41_engram_max_ngram_size;
  const int n_heads = c.dsv41_engram_n_heads;
  const int head_dim = c.dsv41_engram_head_dim;
  const int n_hash_cols = (max_ngram - 1) * n_heads;
  if (table_id < 0 || n_hash_cols <= 0 || head_dim <= 0 ||
      (!c.dsv41_engram_num_embeddings.empty() &&
       (size_t)table_id >= c.dsv41_engram_num_embeddings.size()))
    return false;

  std::vector<int64_t> hashes;
  if (!hash_batch(engine, S, pos_base, token_ids, hashes)) {
    LOG_ERROR("[DSV41-ENGRAM] hash metadata or token ids unavailable at layer %d\n",
              layer_idx);
    return false;
  }
  const std::string prefix = "blk." + std::to_string(layer_idx) + ".";
  const GGUFFile &table_file = engine.has_engram_model
                                  ? engine.engram_model_file
                                  : engine.model_file;
  auto table_it = table_file.tensors.find(prefix + "engram_embd.weight");
  if (table_it == table_file.tensors.end()) {
    LOG_ERROR("[DSV41-ENGRAM] missing table for layer %d\n", layer_idx);
    return false;
  }
  const EngramTableMap *table =
      map_table(engine, table_file, table_it->second, head_dim);
  if (!table) {
    LOG_ERROR("[DSV41-ENGRAM] cannot map table for layer %d\n", layer_idx);
    return false;
  }

  embedding.resize((size_t)S * n_hash_cols * head_dim);
  // Prefill (and every decode token) looks up S * n_hash_cols rows. With on-demand positional reads
  // each miss is a random single-row read, so a serial loop pays the storage
  // latency once per row. Fetch the distinct missing rows with a bounded
  // number of concurrent reads first; the loop below then hits the row cache.
  if (!table->rows && table->direct_fd && *table->direct_fd >= 0 &&
      (size_t)S * n_hash_cols >= 8) {
    std::vector<int64_t> missing;
    {
      std::unordered_set<int64_t> seen;
      std::vector<float> probe((size_t)head_dim);
      for (int s = 0; s < S; ++s)
        for (int col = 0; col < n_hash_cols; ++col) {
          const int64_t row = hashes[((size_t)s * c.dsv41_engram_layer_ids.size() +
                                      (size_t)table_id) * n_hash_cols + col];
          if (row < 0 || (uint64_t)row >= table->n_rows) continue;
          if (!seen.insert(row).second) continue;
          if (!g_row_cache.get(table_id, row, head_dim, probe.data()))
            missing.push_back(row);
        }
    }
    if (!missing.empty()) {
      std::vector<float> fetched(missing.size() * (size_t)head_dim);
      std::vector<char> ok(missing.size(), 0);
      const int fd = *table->direct_fd;
#pragma omp parallel for schedule(dynamic, 8) num_threads(16)
      for (long long i = 0; i < (long long)missing.size(); ++i) {
        std::vector<uint8_t> buf(table->row_bytes);
        const off_t off = (off_t)(table->data_offset +
                                 (size_t)missing[(size_t)i] * table->row_bytes);
        if (::pread(fd, buf.data(), table->row_bytes, off) ==
            (ssize_t)table->row_bytes) {
          posix_fadvise(fd, off, (off_t)table->row_bytes, POSIX_FADV_DONTNEED);
          dequant_row(table_it->second.type, buf.data(),
                      fetched.data() + (size_t)i * head_dim, head_dim);
          ok[(size_t)i] = 1;
        }
      }
      for (size_t i = 0; i < missing.size(); ++i)
        if (ok[i])
          g_row_cache.put(table_id, missing[i],
                          fetched.data() + i * (size_t)head_dim, head_dim);
    }
  }
  for (int s = 0; s < S; ++s) {
    for (int col = 0; col < n_hash_cols; ++col) {
      const int64_t row = hashes[((size_t)s * c.dsv41_engram_layer_ids.size() +
                                  (size_t)table_id) * n_hash_cols + col];
      if (row < 0 || (uint64_t)row >= table->n_rows) {
        LOG_ERROR("[DSV41-ENGRAM] hash row %lld outside %llu-row table at layer %d\n",
                  (long long)row, (unsigned long long)table->n_rows, layer_idx);
        embedding.clear();
        return false;
      }
      float *dst = embedding.data() + ((size_t)s * n_hash_cols + col) * head_dim;
      if (!g_row_cache.get(table_id, row, head_dim, dst)) {
        g_engram_cache_misses.fetch_add(1, std::memory_order_relaxed);
        std::vector<uint8_t> rowbuf(table->row_bytes);
        if (table->rows) {
          std::memcpy(rowbuf.data(), table->rows + (size_t)row * table->row_bytes,
                      table->row_bytes);
        } else {
          if (!table->direct_fd || *table->direct_fd < 0) {
            LOG_ERROR("[DSV41-ENGRAM] no open read handle for %s\n",
                      table->shard_path.c_str());
            embedding.clear();
            return false;
          }
          const off_t off = (off_t)(table->data_offset +
                                   (size_t)row * table->row_bytes);
          const ssize_t r = ::pread(*table->direct_fd, rowbuf.data(),
                                    table->row_bytes, off);
          posix_fadvise(*table->direct_fd, off, (off_t)table->row_bytes,
                        POSIX_FADV_DONTNEED);
          if (r != (ssize_t)table->row_bytes) {
            LOG_ERROR("[DSV41-ENGRAM] short read row %lld for layer %d\n",
                      (long long)row, layer_idx);
            embedding.clear();
            return false;
          }
        }
        dequant_row(table_it->second.type, rowbuf.data(), dst, head_dim);
        g_row_cache.put(table_id, row, dst, head_dim);
      } else {
        g_engram_cache_hits.fetch_add(1, std::memory_order_relaxed);
      }
    }
  }
  return true;
}
