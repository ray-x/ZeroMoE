#pragma once

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <vector>

// CPU-side cache bookkeeping shared by DeepSeek-V4 CSA/HCA attention and the
// CPU-only component tests. This implementation intentionally represents one
// contiguous sequence; llama.cpp's multi-sequence snapshot/rollback protocol
// is outside MoEStream's single-sequence generation model.
inline constexpr int kDsv4CsaRatio = 4;
inline constexpr int kDsv4HcaRatio = 128;

inline int dsv4_compressed_cache_capacity(int max_tokens, int ratio) {
    return std::max(1, (std::max(0, max_tokens) + ratio - 1) / ratio);
}

inline int dsv4_completed_compressed_blocks(int pos, int ratio) {
    return pos < 0 ? 0 : (pos + 1) / ratio;
}

inline bool dsv4_compressed_block_visible(int block, int pos, int ratio) {
    return block >= 0 && block < dsv4_completed_compressed_blocks(pos, ratio);
}

inline int dsv4_state_slot(int pos, int state_size) {
    return pos >= 0 && state_size > 0 ? pos % state_size : -1;
}

// CSA forms one key from [previous 4 tokens | current 4 tokens]. The initial
// previous half is synthetic (zero values and -inf gates).
inline int dsv4_csa_overlap_position(int block_end_pos, int half, int index) {
    const int current_start = block_end_pos + 1 - kDsv4CsaRatio;
    return current_start + (half == 0 ? -kDsv4CsaRatio : 0) + index;
}

// Deterministic CPU equivalent of selecting the largest Lightning Indexer
// logits. Tie-breaking by lower cache row keeps runs reproducible.
inline std::vector<int> dsv4_top_k_indices(const std::vector<float>& scores,
                                            int top_k) {
    std::vector<int> indices(scores.size());
    std::iota(indices.begin(), indices.end(), 0);
    top_k = std::clamp(top_k, 0, static_cast<int>(indices.size()));
    std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
                      [&scores](int a, int b) {
                          return scores[a] == scores[b] ? a < b : scores[a] > scores[b];
                      });
    indices.resize(top_k);
    return indices;
}

struct Dsv4CompressedLayerState {
    int ratio = 0;
    int key_dim = 0;
    int indexer_key_dim = 0;
    int cache_capacity = 0;
    int state_size = 0;
    int state_width = 0;

    // K-only compressed caches. CSA additionally owns the Lightning Indexer
    // cache and its independent overlap-compressor state.
    std::vector<float> keys;
    std::vector<float> state_kv;
    std::vector<float> state_score;
    std::vector<float> indexer_keys;
    std::vector<float> indexer_state_kv;
    std::vector<float> indexer_state_score;

    bool enabled() const { return ratio != 0; }
    bool is_csa() const { return ratio == kDsv4CsaRatio; }
    bool is_hca() const { return ratio == kDsv4HcaRatio; }

    void configure(int new_ratio, int new_key_dim, int max_tokens,
                   int new_indexer_key_dim = 0) {
        ratio = new_ratio;
        key_dim = new_key_dim;
        indexer_key_dim = new_indexer_key_dim;
        cache_capacity = dsv4_compressed_cache_capacity(max_tokens, ratio);
        state_size = is_csa() ? 2 * ratio : ratio;
        state_width = is_csa() ? 2 * key_dim : key_dim;

        keys.assign(static_cast<size_t>(cache_capacity) * key_dim, 0.0f);
        state_kv.assign(static_cast<size_t>(state_size) * state_width, 0.0f);
        state_score.assign(static_cast<size_t>(state_size) * state_width, 0.0f);
        if (is_csa()) {
            indexer_keys.assign(static_cast<size_t>(cache_capacity) * indexer_key_dim, 0.0f);
            indexer_state_kv.assign(
                static_cast<size_t>(state_size) * 2 * indexer_key_dim, 0.0f);
            indexer_state_score.assign(
                static_cast<size_t>(state_size) * 2 * indexer_key_dim, 0.0f);
        } else {
            indexer_keys.clear();
            indexer_state_kv.clear();
            indexer_state_score.clear();
        }
    }

    void reset() {
        std::fill(keys.begin(), keys.end(), 0.0f);
        std::fill(state_kv.begin(), state_kv.end(), 0.0f);
        std::fill(state_score.begin(), state_score.end(), 0.0f);
        std::fill(indexer_keys.begin(), indexer_keys.end(), 0.0f);
        std::fill(indexer_state_kv.begin(), indexer_state_kv.end(), 0.0f);
        std::fill(indexer_state_score.begin(), indexer_state_score.end(), 0.0f);
    }
};
