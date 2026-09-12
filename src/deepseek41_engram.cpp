// DeepSeek-V4.1 conditional-memory Engram layer input.
//
// The converter keeps each enormous engram_embd table quantized in its GGUF
// shard. This file intentionally maps only the table's byte range and reads
// the small set of rows needed by the current token batch; no table is copied
// into the resident dense pool.
#include "engine.hpp"
#include "dense_dispatch.hpp"
#include "kquant_dequant.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cerrno>
#include <fcntl.h>
#include <limits>
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
};

std::unordered_map<std::string, EngramTableMap> g_tables;
std::mutex g_tables_mutex;

const EngramTableMap *map_table(StreamingEngine &engine, const GGUFTensor &t,
                                int row_dim) {
  if (t.shape.size() != 2 ||
      t.shape[0] != (uint64_t)row_dim || t.shard_idx < 0 ||
      (size_t)t.shard_idx >= engine.model_file.shard_paths.size())
    return nullptr;
  const std::string key = engine.model_file.shard_paths[(size_t)t.shard_idx] +
                          "#" + t.name;
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

  const int fd = ::open(engine.model_file.shard_paths[(size_t)t.shard_idx].c_str(),
                         O_RDONLY);
  if (fd < 0) {
    LOG_ERROR("[DSV41-ENGRAM] cannot open shard %s: %s\n",
              engine.model_file.shard_paths[(size_t)t.shard_idx].c_str(),
              std::strerror(errno));
    return nullptr;
  }
  void *mapped = ::mmap(nullptr, (size_t)length, PROT_READ, MAP_SHARED, fd,
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

  // Position zero starts a new request in MoEStream; do not reuse a cached
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

  std::vector<int64_t> hashes;
  if (!hash_batch(engine, S, pos_base, token_ids, hashes)) {
    LOG_ERROR("[DSV41-ENGRAM] hash metadata or token ids unavailable at layer %d\n",
              layer_idx);
    return;
  }
  const std::string prefix = "blk." + std::to_string(layer_idx) + ".";
  auto table_it = engine.model_file.tensors.find(prefix + "engram_embd.weight");
  if (table_it == engine.model_file.tensors.end()) {
    LOG_ERROR("[DSV41-ENGRAM] missing table for layer %d\n", layer_idx);
    return;
  }
  const EngramTableMap *table = map_table(engine, table_it->second, head_dim);
  if (!table) {
    LOG_ERROR("[DSV41-ENGRAM] cannot map table for layer %d\n", layer_idx);
    return;
  }

  std::vector<float> embed((size_t)S * n_hash_cols * head_dim);
  for (int s = 0; s < S; ++s) {
    for (int col = 0; col < n_hash_cols; ++col) {
      const int64_t row = hashes[((size_t)s * c.dsv41_engram_layer_ids.size() +
                                  (size_t)table_id) * n_hash_cols + col];
      if (row < 0 || (uint64_t)row >= table->n_rows) {
        LOG_ERROR("[DSV41-ENGRAM] hash row %lld outside %llu-row table at layer %d\n",
                  (long long)row, (unsigned long long)table->n_rows, layer_idx);
        return;
      }
      dequant_row(table_it->second.type,
                  table->rows + (size_t)row * table->row_bytes,
                  embed.data() + ((size_t)s * n_hash_cols + col) * head_dim,
                  head_dim);
    }
  }

  std::vector<float> kv((size_t)S * (HC + 1) * D);
  dense_proj(engine, prefix + "engram_wkv.weight", embed.data(), kv.data(), S,
             n_hash_cols * head_dim, (HC + 1) * D);
  std::vector<float> q_weight, k_weight;
  if (!load_small_f32(engine, prefix + "engram_q.weight", (size_t)HC * D,
                      q_weight) ||
      !load_small_f32(engine, prefix + "engram_k.weight", (size_t)HC * D,
                      k_weight)) {
    LOG_ERROR("[DSV41-ENGRAM] missing q/k weights at layer %d\n", layer_idx);
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
        const float weight = q_weight[(size_t)h * D + d] *
                             k_weight[(size_t)h * D + d];
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
