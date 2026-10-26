#pragma once
#include <cstdio>
#include "quant_table.hpp"
#include "logger.hpp"
#include "perf_timer.hpp"  // register_mirror_fd -- per-drive read-byte attribution
#include <algorithm>
#include <cstdint>
#include <fcntl.h>
#include <sys/stat.h>   // ::stat, for open_mirror's per-shard size check
#include <sys/mman.h>   // mmap/munmap -- whole-shard mapping for the prefill stream path
#include <cstring>      // strerror
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef __APPLE__
// macOS doesn't have posix_fadvise; provide a no-op stub. Guarded by the SAME
// symbol (POSIX_FADV_WILLNEED) engine.hpp's own stub uses, and the function
// definition lives INSIDE that guard (not after it, unconditionally) -- this
// file and engine.hpp always land in the same translation unit (engine.hpp
// includes this file first), so whichever is processed first fully defines
// constants+function and the other's #ifndef skips cleanly.
#ifndef POSIX_FADV_WILLNEED
#define POSIX_FADV_WILLNEED 3
#define POSIX_FADV_DONTNEED 4
#define POSIX_FADV_RANDOM 1
static inline int posix_fadvise(int, off_t, off_t, int) { return 0; }
#endif
#endif

// GGML / GGUF tensor type IDs. These values MUST match the raw `type` field
// stored in the GGUF tensor table (see ggml.h). calculate_size() below switches
// on the integer value, so these names exist only for readable dispatch — but
// they must still carry the correct ggml numeric IDs, or `case GGUFType::X:`
// switches will silently match the wrong tensors.
enum class GGUFType : uint32_t {
  F32 = 0,
  F16 = 1,
  Q4_0 = 2,
  Q4_1 = 3,
  // 4,5 were removed legacy Q4_2/Q4_3
  Q5_0 = 6,
  Q5_1 = 7,
  Q8_0 = 8,
  Q8_1 = 9,
  Q2_K = 10,
  Q3_K = 11,
  Q4_K = 12,
  Q5_K = 13,
  Q6_K = 14,
  Q8_K = 15,
  IQ2_XXS = 16,
  IQ2_XS = 17,
  IQ3_XXS = 18,
  IQ1_S = 19,
  IQ4_NL = 20,
  IQ3_S = 21,
  IQ2_S = 22,
  IQ4_XS = 23,
  // I32 (raw 32-bit integer tensor, ggml's real numbering -- confirmed
  // against build-metal/_deps/ggml-src/include/ggml.h): NOT a weight quant
  // format, never dequantized through dequant_row/ggml_dequant_row -- added
  // specifically for DeepSeek-V4's ffn_gate_tid2eid.weight (hash-routed MoE
  // layers' per-token-id -> expert-id lookup table), which is raw int32
  // index data, read directly as int32_t* rather than dequantized like
  // every other tensor type in this enum. calculate_size() below has a
  // matching case; row_bytes()/dequant_row() deliberately do NOT, since
  // this tensor is never routed through the generic dense-matmul dequant
  // path.
  I32 = 26,
  // I64 (raw 64-bit integer tensor, ggml numbering, confirmed against
  // build-metal/_deps/ggml-src/include/ggml.h: GGML_TYPE_I64=27) -- added
  // as a defensive gap-fill while investigating a real run where EVERY
  // ffn_gate_tid2eid lookup failed its bounds check (see that lookup's own
  // comment in execution_layer.cpp): if the on-disk tid2eid tensor ever
  // turns out to be I64 rather than I32 (a real, ggml-legal encoding for
  // an integer index tensor), calculate_size() would previously fall
  // through to the `return numel;` 1-byte/element fallback -- an 8x
  // undersized read that truncates the tensor and leaves its tail as
  // uninitialized posix_memalign() heap garbage, which would reinterpret
  // as exactly the kind of uniformly-out-of-range int32 noise that
  // symptom showed. Not confirmed as the actual root cause (that requires
  // the type-check log added at the tid2eid call site, real hardware
  // only) -- added regardless since it is a genuine, previously-missing
  // gap independent of whether it's THIS bug's cause.
  I64 = 27,
  // FIX (found investigating a real DeepSeek-V4-Flash run's dequant
  // failures): every value from here on was off by 5 against the ACTUAL
  // ggml_type enum currently linked into this project (confirmed directly
  // against build-metal/_deps/ggml-src/include/ggml.h, not assumed) --
  // upstream ggml inserted I8/I16/I32/I64/F64 (integer/double tensor
  // types, never used as weight quant formats in this codebase -- I32 is
  // the one exception, see below) between IQ4_XS and IQ1_M at some point
  // after this enum was originally written, silently shifting IQ1_M
  // 24->29, BF16 25->30, TQ1_0 29->34, TQ2_0 30->35. The old values meant
  // a real file's BF16 tensor (raw type tag 30) matched NEITHER this
  // enum's old BF16(25) NOR anything else meaningful -- it fell through
  // to whatever the old numbering called type 30 (TQ2_0), silently
  // dequantizing real BF16 bytes as if they were TQ2_0-quantized blocks.
  // Renumbered to match the real, current values exactly; holes left
  // where upstream ggml itself has holes (removed ND-packed/4-wide repack
  // variants) so a FUTURE upstream insertion is easier to diff against
  // ggml.h directly instead of silently drifting again.
  IQ1_M = 29,
  BF16 = 30,
  // 31,32,33 = removed Q4_0_4_4 / Q4_0_4_8 / Q4_0_8_8 (ND-packed repack
  // variants, GGUF support removed upstream)
  TQ1_0 = 34,
  TQ2_0 = 35,
  // 36,37,38 = removed IQ4_NL_4_4 / IQ4_NL_4_8 / IQ4_NL_8_8 (ditto)
  MXFP4 = 39, // MXFP4 (1 block) -- DeepSeek-V4-Flash UD-IQ2_M uses this;
              // ggml_dequant_row (ggml_dequant.cpp) previously had no case
              // for it at all ("[GGML-DEQUANT] Unsupported ggml type 39"
              // on a real run) since it postdates when that switch was
              // written, independent of the renumbering above.
  NVFP4 = 40, // NVFP4 (4 blocks, E4M3 scale) -- not observed yet, added
              // alongside MXFP4 since both are new in the same ggml
              // version and the dequant function already exists.
  // IQ2_M: confirmed (against BOTH the linked ggml AND a full check of
  // /Users/rayxu/github/llama.cpp's own source, not just this project's
  // vendored copy) that this is NOT a real per-tensor GGML_TYPE_* value
  // anywhere -- "IQ2_M" is a FILE-LEVEL label (llama.h's
  // LLAMA_FTYPE_MOSTLY_IQ2_M = 29, a `llama_ftype`, describing the
  // overall quant RECIPE a tool like Unsloth used) that llama.cpp's own
  // quantizer (llama-quant.cpp:837, `case LLAMA_FTYPE_MOSTLY_IQ2_M:
  // return GGML_TYPE_IQ2_S;`) resolves to GGML_TYPE_IQ2_S (already
  // supported here, case 22) for the predominant per-TENSOR type, mixed
  // with ordinary K-quants/other IQ types for the rest -- never one
  // monolithic "IQ2_M" wire format applied tensor-by-tensor. This
  // project's OWN kquant_dequant.hpp already documented this independently
  // (gpu_unsupported_reason's IQ2_M case: "no ggml CPU dequant export
  // exists either... unsupported on BOTH paths") -- GGUFType::IQ2_M exists
  // purely as a DIAGNOSTIC label (name string, bits-per-weight display,
  // that reason string) for a tensor that happens to carry this raw type
  // tag; it is never actually dequantized through any real code path
  // either way. Restored (removing it broke the build -- 5 real call
  // sites in kquant_dequant.hpp/graph_initializer.cpp reference it) at
  // slot 31, one of the genuinely-retired ggml holes (Q4_0_4_4, GGUF
  // support removed upstream) -- safe precisely because no real modern
  // file can ever legitimately carry raw type tag 31 for anything, so
  // this can never collide with an actual type this enum is also trying
  // to represent.
  IQ2_M = 31,
  // Q2_0: a genuine legacy-style linear 2-bit format (per-64-element fp16
  // scale + 2-bit signed-ish codes {-1,0,+1,+2}*scale, block_q2_0 in
  // ggml-common.h) -- confirmed against the currently-linked ggml.h
  // (GGML_TYPE_Q2_0 = 42), NOT to be confused with Q2_K (already present
  // above at slot 10, a totally different super-block K-quant format).
  // Added 2026-09-28 for a real ISTA DASLab "GSQ-RCO" Qwen3.8-Flash-Next
  // release that stores blk.N.ffn_down_exps.weight (the routed-expert
  // down projection, the single largest tensor bucket in the file) as
  // this type on every MoE layer -- without this case those tensors fall
  // through calculate_size()'s `return numel;` 1-byte/element fallback,
  // a ~18x undersized read.
  Q2_0 = 42,
  EXL3 = 0x1000,
};

struct GGUFTensor {
  std::string name;
  GGUFType type;
  std::vector<uint64_t> shape;
  uint64_t offset;
  uint64_t size_bytes;
  int shard_idx = 0; // which shard this tensor lives in (for split GGUF)
};

// How the per-layer attention cache is shaped/allocated, and which MoE
// gating activation a family uses by default. Defined HERE (rather than in
// models/family_ops.hpp, where the rest of the per-family dispatch table
// lives) specifically to avoid a circular include: ArchitectureConfig below
// needs these as native field types, but family_ops.hpp -> model_family.hpp
// -> gguf_loader.hpp is already the dependency direction, so gguf_loader.hpp
// cannot include family_ops.hpp back. family_ops.hpp reuses these same two
// enums via its own (pre-existing) transitive include of this header.
enum class CacheShape {
  DenseKV,     // Kc/Vc: per-layer [max_t, n_head_kv*head_dim] (Hunyuan/Llama)
  MLAAbsorbed, // Lc/Rc: per-layer [max_t, kv_lora] + [max_t, qk_rope]
               // (GLM-5.2/DeepSeek-V3, absorbed WUK/WUV)
  MLARawSWA,   // KVc: per-layer [sliding_window, n_embd_head] ring buffer,
               // no absorption matrices (DeepSeek-V4).
  GraphOwnedKV, // Family graph owns device KV; no duplicate engine-side cache.
};

enum class GatingFunc {
  Softmax,
  Sigmoid,
  SqrtSoftplus, // sqrt(softplus(x)) -- DeepSeek-V4
};

struct ArchitectureConfig {
  int hidden = 0;
  int n_layers = 0;
  // Number of llama.cpp-style NextN/MTP blocks appended after the normal
  // transformer trunk.  n_layers remains the target/trunk count so all
  // existing forward loops keep their historical bounds; callers that
  // explicitly implement MTP can use n_layers_total() for cache/manifest
  // sizing.  Zero is the common case and is a strict no-op.
  int n_layer_nextn = 0;
  int n_heads = 0;
  int n_experts = 0;
  int topk = 0;
  int moe_inter = 0;
  int vocab = 0;
  int kv_lora = 512;
  int qk_nope = 128;
  int qk_rope = 64;
  int v_head = 128;
  int first_dense = 0; // first N layers are dense (no MoE)
  // Dense Q/K/V attention (Hunyuan / Llama / GLM-4); unused for MLA arches.
  int n_head_kv = 0;   // GQA kv-head count (defaults to n_heads)
  int head_dim = 0;    // per-head dim (= attention.key_length)
  // Set once in graph_initializer.cpp from engine.family_ops.cache_shape,
  // right after family_ops itself is resolved -- replaces the old bare
  // `bool is_mla`. graph_initializer.cpp's cache-allocation code and
  // dense_preload_manifest builder switch on this; layers_forward_gguf
  // (execution_layer.cpp) no longer checks attention shape directly at all
  // (it calls through engine.family_ops.attention_fn instead).
  CacheShape cache_shape = CacheShape::DenseKV;
  float eps = 1e-5f;

  // Dense (non-MoE) FFN intermediate width — used for leading dense layers
  // (l < first_dense) in an otherwise-MoE model, and for every layer in a
  // fully-dense model. Distinct from moe_inter (expert_feed_forward_length).
  int dense_inter = 0;

  // ---- MoE router (sigmoid/softmax gating, selection bias, weight norm
  // and scale) — see llama.cpp's build_moe_ffn for the reference algorithm.
  int n_expert_shared = 0;           // 0 = no shared expert
  float expert_weights_scale = 0.0f; // 0 = no scale applied
  bool expert_weights_norm = true;   // normalize top-k weights to sum to 1
  // Router activation. Defaults to engine.family_ops.default_gating,
  // overridden by the GGUF file's own expert_gating_func key when it sets
  // one -- same precedence the old use_sigmoid_gate fallback already had.
  // Replaces that bare bool now that a third value (SqrtSoftplus,
  // DeepSeek-V4) exists.
  GatingFunc gating_func = GatingFunc::Softmax;
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

  // ---- DeepSeek-V4 (Hyper-Connections, grouped output projection,
  // attention sinks, raw SWA plus CSA/HCA compressed attention). Every field
  // defaults to "feature inactive" so populating them is a strict no-op for
  // every other family.
  int hc_mult = 1;              // hyper_connection.count; 1 = inactive
  int hc_sinkhorn_iters = 0;    // hyper_connection.sinkhorn_iterations
  float hc_eps = 1e-6f;         // hyper_connection.epsilon
  int o_group_count = 0;        // attention.output_group_count; 0 = plain wo
  int o_lora_rank = 0;          // attention.output_lora_rank
  int hash_layer_count = 0;     // hash_layer_count; 0 = no hash-routed layers
  int sliding_window = 0;       // attention.sliding_window; 0 = SWA inactive
  std::vector<int64_t> mimo2_head_count_kv;
  std::vector<int64_t> mimo2_sliding_window_pattern;
  float mimo2_rope_freq_base_swa = 10000.0f;
  float mimo2_value_scale = 0.0f;
  // Per-layer DeepSeek-V4 attention compression. A zero ratio selects raw
  // SWA; 4 selects CSA + Lightning Indexer; 128 selects HCA.
  std::vector<int64_t> dsv4_compress_ratios;
  float dsv4_compress_rope_freq_base = 0.0f;
  int dsv4_indexer_head_count = 0;
  int dsv4_indexer_key_length = 0;
  int dsv4_indexer_top_k = 0;
  // DeepSeek-V4.1 shared compressed-attention source/candidate metadata.
  std::vector<int64_t> dsv41_kv_source_layers;
  std::vector<int64_t> dsv41_index_source_layers;
  int dsv41_candidate_source_layer = -1;
  int dsv41_candidate_top_k_blocks = 0;
  int dsv41_candidate_block_size = 0;

  // DeepSeek-V4.1 Engram metadata. The very large engram_embd tables are
  // intentionally not resident; the layer hook gathers their quantized rows.
  std::vector<int64_t> dsv41_engram_layer_ids;
  std::vector<int64_t> dsv41_engram_num_embeddings;
  std::vector<int64_t> dsv41_engram_primes;
  std::vector<int64_t> dsv41_engram_offsets;
  std::vector<int64_t> dsv41_engram_multipliers;
  std::vector<int64_t> dsv41_engram_token_map;
  int dsv41_engram_max_ngram_size = 0;
  int dsv41_engram_vocab_size = 0;
  int dsv41_engram_n_heads = 0;
  int dsv41_engram_head_dim = 0;
  int dsv41_engram_pad_id = 2;
  int dsv41_engram_compressed_vocab_size = 0;
  std::vector<float> swiglu_clamp_exp;   // per-layer, size n_layers when set
  std::vector<float> swiglu_clamp_shexp; // per-layer, size n_layers when set

  // ---- GLM5Next (GLM 5.3 Flash, GGUF arch "glm5next"): hybrid recurrent
  // KDA + nope-only absorbed-MLA (DSA) layers. Every field defaults to
  // "feature inactive" so populating them is a strict no-op for every other
  // family. Reference: llama.cpp branch glm5next (src/models/glm5next.cpp,
  // conversion/glm5next.py).
  //
  // Per-layer layer-type switch, straight from the GGUF
  // attention.head_count_kv ARRAY (the converter writes 0 for KDA layers,
  // 1 for DSA layers -- the scalar c.n_head_kv read elsewhere never sees
  // this key, since it's array-typed here). Empty for every other family;
  // glm5next_is_kda_layer() below is the only correct way to read it.
  std::vector<int64_t> glm5next_head_count_kv;
  // BailingMoE3 uses the same 0=KDA / nonzero=MLA layer map, with a
  // distinct KDA projection layout and a per-head gate after MLA.
  std::vector<int64_t> bailingmoe3_head_count_kv;
  int expert_group_count = 0;
  int expert_group_used_count = 0;
  int ssm_conv_kernel = 0;        // ssm.conv_kernel (KDA short-conv width)
  int kda_head_dim = 0;           // kda.head_dim (per-head KDA dim)
  float kda_gate_lower_bound = 0.0f; // kda.gate_lower_bound (required, < 0;
                                     // scales the sigmoid decay gate -- 0
                                     // here means "never loaded", which the
                                     // glm5next validation block rejects)
  // Indexer (Lightning sparse attention on DSA layers) -- metadata loaded
  // and logged at startup, but NOT executed in v1: DSA layers run as dense
  // absorbed MLA over the full KV cache. Recorded here so the startup log
  // can state plainly what was deferred.
  int glm5next_indexer_head_count = 0;  // attention.indexer.head_count
  int glm5next_indexer_key_length = 0;  // attention.indexer.key_length
  int glm5next_indexer_top_k = 0;       // attention.indexer.top_k
  int glm5next_indexer_kpool = 0;       // attention.indexer.kpool

  // ---- Qwen4Exp (Qwen3.8-Flash-Next, GGUF arch "qwen4exp"): hybrid Gated
  // DeltaNet (recurrent) + full-attention layers, low-rank hyper-connections
  // replacing every layer norm, QSA sparse attention (v1: deferred, full
  // attention runs dense), PLE n-gram hash embedding on select layers.
  // Every field defaults to "feature inactive" so populating them is a
  // strict no-op for every other family. Reference: llama.cpp branch
  // qwen4exp/qwen3.8-flash-next (src/models/qwen4exp.cpp's
  // load_arch_hparams/load_arch_tensors, conversion/qwen4exp.py).
  //
  // Per-layer hybrid map: attention.recurrent_layers bool array (1 = GDN
  // recurrent, 0 = full attention). If the file omits it, derived from
  // full_attention_interval (every interval-th layer is full attention).
  // qwen4exp_is_gdn_layer() below is the only correct way to read it.
  std::vector<int64_t> qwen4exp_recurrent_layers;
  int qwen4exp_full_attention_interval = 0; // attention.full_attention_interval
  // GDN dimensions (ssm.* keys, shared names with GLM5Next's KDA but a
  // DIFFERENT parameterization -- see attention_layer.cpp's qwen4exp block):
  int ssm_inner_size = 0;    // ssm.inner_size  (d_inner = n_v_heads*head_v_dim)
  int ssm_state_size = 0;    // ssm.state_size  (head_k_dim == head_v_dim)
  int ssm_time_step_rank = 0; // ssm.time_step_rank (num_v_heads)
  int ssm_group_count = 0;   // ssm.group_count (num_k_heads)
  // (ssm_conv_kernel, declared for GLM5Next above, is reused: same GGUF key.)
  // Low-rank hyper-connections (qwen4exp-specific; DeepSeek-V4 leaves the
  // key absent = full rank, qwen4exp requires it):
  int hc_low_rank = 0;       // hyper_connection.low_rank
  // QSA indexer on full-attention layers -- metadata loaded and logged, NOT
  // executed in v1 (full-attention layers run dense over the whole KV cache,
  // a correct superset of the indexer's selected blocks). Same deferral
  // discipline as GLM5Next's own indexer.
  std::vector<int64_t> qwen4exp_compress_ratios; // attention.compress_ratios
  int qwen4exp_indexer_head_count = 0;  // attention.indexer.head_count
  int qwen4exp_indexer_key_length = 0;  // attention.indexer.key_length
  int qwen4exp_indexer_top_k = 0;       // attention.indexer.top_k
  // PLE n-gram hash embedding (absent key group = no PLE layers at all).
  std::vector<int64_t> ple_layers;   // ple.layers (0-based layer indices)
  int ple_ngram_size = 0;            // ple.ngram_size
  int ple_heads_per_ngram = 0;       // ple.heads_per_ngram
  int ple_conv_kernel = 0;           // ple.conv_kernel
  int ple_eos_token_id = -1;         // ple.eos_token_id
  int ple_image_token_id = 0;        // ple.image_token_id (0 = fall back to eos)
  int ple_n_embd_per_layer = 0;      // embedding_length_per_layer_input
  // ple_n_heads = (ngram_size-1)*heads_per_ngram, computed at load.
  int ple_n_heads = 0;
  // 45-bit hash constants, read as int64 arrays (uint64 in the file, but
  // every value fits int64 -- the converter itself narrows them to int32
  // row indices after a mod, and the multipliers are < 2^63).
  std::vector<int64_t> ple_layer_multipliers; // ple.layer_multipliers [ngram]
  std::vector<int64_t> ple_head_offsets;      // ple.head_offsets [ple_n_heads]
  std::vector<int64_t> ple_head_vocab_sizes;  // ple.head_vocab_sizes
  // Interleaved partial RoPE (mrope sections; text-only degenerates to plain
  // interleaved RoPE over the first rope_n_rot dims of each head):
  std::vector<int64_t> rope_sections; // rope.dimension_sections (4 entries)
  int rope_n_rot = 0;                 // rope.dimension_count (0 = full head)

  int n_layers_total() const {
    return n_layers + std::max(0, n_layer_nextn);
  }
};

// GLM5Next layer-type query: true when layer l is a recurrent KDA layer
// (head_count_kv[l] == 0), false for a full-attention DSA layer. Returns
// false for any non-GLM5Next config (empty array) and for out-of-range
// layers (defensive -- a short array fails safe toward the DSA path, which
// at least has real tensors to report missing, rather than toward the KDA
// path's state allocation).
inline bool glm5next_is_kda_layer(const ArchitectureConfig &config, int layer_idx) {
  return layer_idx >= 0 &&
         (size_t)layer_idx < config.glm5next_head_count_kv.size() &&
         config.glm5next_head_count_kv[(size_t)layer_idx] == 0;
}

inline bool bailingmoe3_is_kda_layer(const ArchitectureConfig &config, int layer_idx) {
  return layer_idx >= 0 &&
         (size_t)layer_idx < config.bailingmoe3_head_count_kv.size() &&
         config.bailingmoe3_head_count_kv[(size_t)layer_idx] == 0;
}

inline bool hybrid_is_kda_layer(const ArchitectureConfig &config, int layer_idx) {
  return config.bailingmoe3_head_count_kv.empty()
             ? glm5next_is_kda_layer(config, layer_idx)
             : bailingmoe3_is_kda_layer(config, layer_idx);
}

// Qwen4Exp layer-type query: true when layer l is a recurrent GDN layer,
// false for a full-attention layer. Prefers the GGUF's explicit
// attention.recurrent_layers array; falls back to the
// full_attention_interval rule (every interval-th layer, 1-based, is full
// attention) when the array is absent. Returns false for any non-Qwen4Exp
// config (both empty) and for out-of-range layers (fails safe toward the
// full-attention path, which at least has real tensors to report missing,
// rather than toward the GDN path's state allocation -- mirrors
// glm5next_is_kda_layer's defensive rationale).
inline bool qwen4exp_is_gdn_layer(const ArchitectureConfig &config, int layer_idx) {
  if (layer_idx < 0 || layer_idx >= config.n_layers) return false;
  if ((size_t)layer_idx < config.qwen4exp_recurrent_layers.size())
    return config.qwen4exp_recurrent_layers[(size_t)layer_idx] != 0;
  if (config.qwen4exp_full_attention_interval > 0)
    return ((layer_idx + 1) % config.qwen4exp_full_attention_interval) != 0;
  return false;
}

inline bool qwen35moe_is_gdn_layer(const ArchitectureConfig &config,
                                   int layer_idx) {
  return qwen4exp_is_gdn_layer(config, layer_idx);
}
inline bool qwen4exp_is_ple_layer(const ArchitectureConfig &config, int layer_idx) {
  return std::find(config.ple_layers.begin(), config.ple_layers.end(),
                   (int64_t)layer_idx) != config.ple_layers.end();
}

inline bool dsv4_uses_compressed_attention(const ArchitectureConfig &config) {
  return std::any_of(config.dsv4_compress_ratios.begin(),
                     config.dsv4_compress_ratios.end(),
                     [](int64_t ratio) { return ratio != 0; });
}

inline bool dsv41_is_layer_in(const std::vector<int64_t> &layers, int layer) {
  return std::find(layers.begin(), layers.end(), (int64_t)layer) != layers.end();
}

inline bool dsv41_is_engram_layer(const ArchitectureConfig &config,
                                  int layer) {
  return dsv41_is_layer_in(config.dsv41_engram_layer_ids, layer);
}

class GGUFFile {
public:
  std::unordered_map<std::string, GGUFTensor> tensors;
  ArchitectureConfig architecture_config;
  uint32_t version;
  int file_fd = -1; // primary shard fd (backward compat)
  std::string resolved_file_path;

  // Split GGUF support: one fd per shard
  std::vector<int> shard_fds;
  std::vector<std::string> shard_paths;

  // O_DIRECT state, one entry per shard_fds -- see enable_direct_io()
  // below. Empty (all false) until a caller explicitly opts in;
  // fd_for_tensor()/reads are completely unaffected until then.
  std::vector<bool> shard_direct_io;
  uint64_t direct_io_alignment = 4096;

  // OPTIONAL SECOND REPLICA of the same model on a DIFFERENT physical
  // device (EngineConfig::mirror_model_path). Index-matched to shard_fds/
  // shard_paths, so mirror_shard_fds[i] is the same shard as
  // shard_fds[i], just opened from the other drive. Empty until
  // open_mirror() succeeds; every read path stays byte-identical to the
  // single-drive behavior until a caller explicitly asks for replica 1.
  //
  // Why: expert reads are the dominant startup and decode-time cost here,
  // and a single NVMe saturates around ~2.2-2.5 GB/s for this model's real
  // access shape (3 scattered per-expert tensors -- see tools/gguf_dma.cpp).
  // Two INDEPENDENT devices have independent queues, so striping across
  // them is genuinely additive bandwidth rather than a redistribution of
  // one device's ceiling. Measured on this box: primary (Samsung 980 PRO)
  // 2.25-2.52 GB/s vs mirror (KIOXIA KXG60ZNV1T02) 1.98-2.15 GB/s -- the
  // second drive runs at ~85-88% of the first for this pattern, so the
  // combined ceiling is ~1.85x.
  //
  // io_uring makes exploiting this nearly free: a SINGLE submission may
  // mix fds pointing at different devices, and the kernel drives both
  // queues in parallel -- so a batched read just has to alternate which
  // replica each expert's requests target, with no change to batching or
  // completion handling.
  std::vector<int> mirror_shard_fds;
  std::vector<bool> mirror_shard_direct_io;

  // Sparse-mirror index (gguf_tool mirror-subset): when <mirror_dir>/mirror_subset.tsv exists, the mirror holds ONLY
  // the listed (layer, expert) pairs; every other range in it is a hole. resolve_replica() then sends each expert to
  // exactly one drive. Empty = a full mirror, and the old behaviour (including the intra-expert split) is unchanged.
  std::unordered_set<uint64_t> mirror_subset;
  bool mirror_is_subset = false;
  static uint64_t mirror_key(int layer, int expert) {
    return ((uint64_t)(uint32_t)layer << 32) | (uint32_t)expert;
  }
  // `replica` is what the caller asked for (0 primary, 1 mirror, -1 split). With a sparse mirror the answer depends
  // on where this expert actually lives.
  int resolve_replica(int layer, int expert, int replica) const {
    if (!mirror_is_subset || replica == 0) return replica;
    return mirror_subset.count(mirror_key(layer, expert)) ? 1 : 0;
  }

  // KV metadata extracted during header parsing
  std::unordered_map<std::string, uint64_t> kv_uint;
  std::unordered_map<std::string, int64_t> kv_int;
  std::unordered_map<std::string, float> kv_float;
  std::unordered_map<std::string, std::string> kv_string;
  // GGUF array-typed KV (type 9) with a numeric element sub-type -- e.g.
  // DeepSeek-V4's per-layer swiglu_clamp_exp/shexp. Added alongside the
  // scalar maps above rather than replacing read_kv_value's prior "skip
  // arrays, too complex to store generically" behavior for non-numeric
  // sub-types (string/nested-array elements), which no known hparam in this
  // codebase needs -- see read_kv_array's own comment.
  std::unordered_map<std::string, std::vector<int64_t>> kv_arr_int;
  std::unordered_map<std::string, std::vector<float>> kv_arr_float;

  int fd_for_tensor(const GGUFTensor &t) const {
    if (t.shard_idx < (int)shard_fds.size())
      return shard_fds[t.shard_idx];
    return file_fd;
  }

  bool direct_io_active_for(const GGUFTensor &t) const {
    return t.shard_idx >= 0 && (size_t)t.shard_idx < shard_direct_io.size() &&
           shard_direct_io[t.shard_idx];
  }

  // ---- Whole-shard read-only mmap (EngineConfig::moe_prefill_stream) ----
  // Motivation, MEASURED 2026-09-05 on a real 441-token qwen4exp prefill:
  // the batched prefill path pulled 31.07 GB of expert weights per pass
  // through load_expert_slab's pread + memcpy into private RAM-tier slabs
  // -- 5.08 GB/s warm, 0.77 GB/s cold -- for data each expert touches
  // EXACTLY ONCE per prefill. That run reported `cache=0% hit`, and it is
  // not a tuning failure but arithmetic: 48 layers x 512 experts is ~68 GB
  // of DISTINCT slabs against a 29.5 GB VRAM cache, and a long prompt
  // activates essentially every expert in every layer (512 tokens x top-10
  // over 512 experts is ~5120 draws, so coupon-collector says all 512), so
  // there is no reuse for a cache to capture. llama.cpp mmaps the file and
  // copies nothing. This closes that specific gap without disturbing the
  // RAM tier, which decode -- where a token touches only 10 of 512 experts
  // and reuse across tokens is real -- genuinely needs.
  //
  // MAP_SHARED/PROT_READ over the WHOLE shard, so any tensor's bytes are
  // just (base + t.offset). Deliberately opens its OWN fd instead of
  // reusing shard_fds[i]: those may carry O_DIRECT (see enable_direct_io),
  // and pairing an O_DIRECT fd's page-cache-BYPASSING reads with a
  // page-cache-BACKED mapping of the same file is a coherence hazard --
  // this project already hit the mirror image of it once (see CLAUDE.md's
  // btrfs O_DIRECT page-cache-leak entry). The mapping holds its own
  // reference to the file, so the fd is closed immediately after mmap.
  std::vector<void *> shard_maps;
  std::vector<size_t> shard_map_sizes;

  bool mmap_active() const { return !shard_maps.empty(); }

  // Idempotent. All-or-nothing: any failure unmaps whatever already
  // succeeded and reports false, so mmap_active() never describes a
  // partially-mapped model that mmap_bytes() would then answer for some
  // shards but not others.
  bool enable_mmap() {
    if (!shard_maps.empty()) return true;
    const size_t n = shard_paths.size();
    if (n == 0) {
      LOG_WARN("[MMAP] no shard paths recorded -- cannot enable mmap.\n");
      return false;
    }
    std::vector<void *> maps(n, nullptr);
    std::vector<size_t> sizes(n, 0);
    for (size_t i = 0; i < n; ++i) {
      int fd = ::open(shard_paths[i].c_str(), O_RDONLY);
      if (fd < 0) {
        LOG_ERROR("[MMAP] cannot open shard %s: %s\n", shard_paths[i].c_str(),
                  strerror(errno));
        unmap_list(maps, sizes);
        return false;
      }
      struct stat st {};
      if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
        LOG_ERROR("[MMAP] fstat failed for %s\n", shard_paths[i].c_str());
        ::close(fd);
        unmap_list(maps, sizes);
        return false;
      }
      void *m = ::mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
      ::close(fd);
      if (m == MAP_FAILED) {
        LOG_ERROR("[MMAP] mmap of %s (%lld bytes) failed: %s\n",
                  shard_paths[i].c_str(), (long long)st.st_size, strerror(errno));
        unmap_list(maps, sizes);
        return false;
      }
      maps[i] = m;
      sizes[i] = (size_t)st.st_size;
    }
    shard_maps.swap(maps);
    shard_map_sizes.swap(sizes);
    size_t total = 0;
    for (size_t b : shard_map_sizes) total += b;
    LOG_STREAM_INFO() << "[MMAP] mapped " << n << " shard(s), "
                      << (total / (1024.0 * 1024.0 * 1024.0)) << " GB of "
                      << "address space (resident pages are the OS page "
                      << "cache's business, not ours)" << std::endl;
    return true;
  }

  // Pointer to `len` bytes at ABSOLUTE file offset `off` within t's shard,
  // or nullptr if this shard isn't mapped or the range would run past the
  // end of the file. Bounds-checked rather than trusted: a truncated or
  // malformed shard would otherwise turn into a silent out-of-bounds read
  // of mapped memory (a SIGBUS at best, wrong weights at worst).
  const uint8_t *mmap_bytes(const GGUFTensor &t, uint64_t off,
                            size_t len) const {
    if (t.shard_idx < 0 || (size_t)t.shard_idx >= shard_maps.size())
      return nullptr;
    void *base = shard_maps[(size_t)t.shard_idx];
    if (!base) return nullptr;
    const size_t cap = shard_map_sizes[(size_t)t.shard_idx];
    if (off > cap || len > cap - off) return nullptr;
    return static_cast<const uint8_t *>(base) + off;
  }

  // Ask the kernel to stage a mapped expert range into the page cache before
  // the prefill stream's memcpy touches it. This is deliberately a
  // best-effort hint: the mmap path remains correct if the kernel ignores
  // it, and the caller still validates the range through mmap_bytes().
  // Per-range hints matter here because the batched route can have every
  // expert resident in ZeroMoE's RAM tier already, so the prefetcher does
  // not perform the reads that would otherwise warm these same mmap pages.
  bool mmap_willneed(const GGUFTensor &t, uint64_t off, size_t len) const {
#ifdef __linux__
    if (len == 0 || t.shard_idx < 0 ||
        (size_t)t.shard_idx >= shard_maps.size())
      return false;
    void *base = shard_maps[(size_t)t.shard_idx];
    const size_t cap = shard_map_sizes[(size_t)t.shard_idx];
    if (!base || off > cap || len > cap - off)
      return false;
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0)
      return false;
    const uintptr_t start =
        reinterpret_cast<uintptr_t>(base) + (size_t)off;
    const uintptr_t aligned =
        start & ~((uintptr_t)page_size - 1U);
    const size_t delta = (size_t)(start - aligned);
    const size_t hinted = len + delta;
    return madvise(reinterpret_cast<void *>(aligned), hinted,
                   MADV_WILLNEED) == 0;
#else
    (void)t;
    (void)off;
    (void)len;
    return false;
#endif
  }


  bool has_mirror() const { return !mirror_shard_fds.empty(); }

  // replica 0 = primary (always valid), 1 = mirror. Degrades to the primary
  // whenever the mirror isn't open or doesn't cover this shard, so callers
  // may pass a replica index unconditionally without checking has_mirror()
  // first -- a striping policy stays correct (just unbalanced) on a
  // single-drive setup instead of needing its own fallback branch.
  int fd_for_tensor(const GGUFTensor &t, int replica) const {
    if (replica == 1 && t.shard_idx >= 0 &&
        (size_t)t.shard_idx < mirror_shard_fds.size() &&
        mirror_shard_fds[t.shard_idx] >= 0)
      return mirror_shard_fds[t.shard_idx];
    return fd_for_tensor(t);
  }

  // MUST be paired with the matching fd_for_tensor(t, replica) call -- the
  // two replicas can legitimately differ here (e.g. the mirror lives on a
  // filesystem that rejects O_DIRECT while the primary accepts it), and
  // passing a primary-derived is_direct flag alongside a mirror fd would
  // mis-describe that read to direct_io/io_uring's alignment handling.
  bool direct_io_active_for(const GGUFTensor &t, int replica) const {
    if (replica == 1 && t.shard_idx >= 0 &&
        (size_t)t.shard_idx < mirror_shard_fds.size() &&
        mirror_shard_fds[t.shard_idx] >= 0)
      return (size_t)t.shard_idx < mirror_shard_direct_io.size() &&
             mirror_shard_direct_io[t.shard_idx];
    return direct_io_active_for(t);
  }

  // Opens a second copy of this model from `mirror_dir`, matching each
  // shard by BASENAME (the mirror is expected to be a plain file-level copy,
  // so filenames match even though directories differ). Best-effort and
  // all-or-nothing per shard: any shard that can't be opened, or whose size
  // doesn't match the primary's exactly, leaves that shard mirror-less
  // (reads for it silently use the primary). A size mismatch is the one
  // check that really matters -- every read uses offsets computed from the
  // PRIMARY's tensor table, so a mirror that isn't byte-identical would
  // silently return wrong weights rather than fail loudly.
  // Returns the number of shards successfully mirrored.
  int open_mirror(const std::string &mirror_dir) {
    mirror_shard_fds.assign(shard_fds.size(), -1);
    mirror_shard_direct_io.assign(shard_fds.size(), false);
#ifndef __linux__
    LOG_STREAM_WARN() << "[MIRROR] Only supported on Linux -- ignoring.\n";
    return 0;
#else
    int n_ok = 0;
    for (size_t i = 0; i < shard_paths.size(); ++i) {
      std::string base = shard_paths[i];
      size_t slash = base.find_last_of('/');
      if (slash != std::string::npos) base = base.substr(slash + 1);
      std::string mpath = mirror_dir;
      if (!mpath.empty() && mpath.back() != '/') mpath += '/';
      mpath += base;

      // Size equality against the primary -- see this function's comment on
      // why a silent mismatch would be a correctness bug, not just a perf
      // miss. Compared via stat on both, not a trusted config assertion.
      struct stat sp;
      struct stat sm;
      // ::stat -- the `struct stat sp;` declarations above bring the TYPE
      // name into scope, which hides the function of the same name.
      if (::stat(shard_paths[i].c_str(), &sp) != 0 || ::stat(mpath.c_str(), &sm) != 0) {
        LOG_STREAM_WARN() << "[MIRROR] Shard " << i << ": cannot stat " << mpath
                          << " -- this shard will read from the primary only.\n";
        continue;
      }
      if (sp.st_size != sm.st_size) {
        LOG_STREAM_ERROR() << "[MIRROR] Shard " << i << " SIZE MISMATCH (primary "
                           << sp.st_size << " vs mirror " << sm.st_size << " bytes) for "
                           << mpath << " -- refusing to stripe this shard; every read "
                              "uses the primary's tensor offsets, so a differing copy "
                              "would silently return wrong weights.\n";
        continue;
      }

      // Match the primary's O_DIRECT state for this shard where possible;
      // verify with a real pread rather than trusting open() (same
      // discipline as enable_direct_io -- some filesystems accept the flag
      // then reject the read).
      bool want_direct = (i < shard_direct_io.size() && shard_direct_io[i]);
      int fd = -1;
      bool got_direct = false;
      if (want_direct) {
        fd = open(mpath.c_str(), O_RDONLY | O_DIRECT);
        if (fd >= 0) {
          void *tb = nullptr;
          if (posix_memalign(&tb, (size_t)direct_io_alignment,
                             (size_t)direct_io_alignment) == 0 && tb) {
            got_direct = (pread(fd, tb, (size_t)direct_io_alignment, 0) >= 0);
            free(tb);
          }
          if (!got_direct) { close(fd); fd = -1; }
        }
      }
      if (fd < 0) {
        fd = open(mpath.c_str(), O_RDONLY);
        got_direct = false;
        if (fd >= 0 && want_direct) {
          LOG_STREAM_WARN() << "[MIRROR] Shard " << i << " (" << mpath
                            << ") could not use O_DIRECT (the primary does) -- "
                               "striping this shard with buffered reads.\n";
        }
      }
      if (fd < 0) {
        LOG_STREAM_WARN() << "[MIRROR] Shard " << i << ": open failed for " << mpath
                          << " -- this shard will read from the primary only.\n";
        continue;
      }
      mirror_shard_fds[i] = fd;
      mirror_shard_direct_io[i] = got_direct;
      register_mirror_fd(fd);  // perf_timer.hpp -- lets submit_and_wait attribute real read bytes per drive
      n_ok++;
    }
    mirror_subset.clear();
    mirror_is_subset = false;
    {
      std::ifstream idx(mirror_dir + (mirror_dir.back() == '/' ? "" : "/") + "mirror_subset.tsv");
      if (idx) {
        int l, e;
        while (idx >> l >> e) mirror_subset.insert(mirror_key(l, e));
        mirror_is_subset = true;
        LOG_STREAM_INFO() << "[MIRROR] Sparse mirror: " << mirror_subset.size()
                          << " expert(s) live on the mirror; all others are read from the primary.\n";
      }
    }
    LOG_STREAM_INFO() << "[MIRROR] Second replica: " << n_ok << "/" << shard_paths.size()
                      << " shard(s) opened from " << mirror_dir
                      << " -- expert reads can now stripe across both devices "
                         "(io_uring drives both queues from one submission).\n";
    return n_ok;
#endif
  }

  // Opt-in: reopen every shard's fd with O_DIRECT, one shard at a time,
  // ONLY if this platform supports O_DIRECT (Linux; guarded rather
  // than assumed). When allow_unaligned_ranges is true, GGUF tensor
  // offsets need not meet O_DIRECT's block alignment:
  // pread_direct_aware and io_uring_backend round each read window to
  // min_alignment and stage only the requested bytes.
  // Each shard gets a real test read (not just
  // a successful open()) before being trusted -- some filesystems
  // (network mounts, some overlay/container filesystems, tmpfs) accept
  // O_DIRECT at open() but reject the actual read with EINVAL; a shard
  // that fails either step keeps its ORIGINAL buffered fd untouched
  // and is simply left out of shard_direct_io, never a hard error --
  // this feature is a pure performance opt-in, never something a
  // caller should have to work around if it doesn't pan out for a
  // given filesystem. Returns the number of shards that ended up with
  // real O_DIRECT active.
  int enable_direct_io(uint64_t min_alignment = 4096,
                       bool allow_unaligned_ranges = false) {
    shard_direct_io.assign(shard_fds.size(), false);
#ifndef __linux__
    (void)min_alignment; // only referenced in the Linux branch below
    LOG_STREAM_WARN()
        << "[DIRECT-IO] Not supported on this platform (Linux-only) -- "
           "all reads stay buffered.\n";
    return 0;
#else
    auto it = kv_uint.find("general.alignment");
    uint64_t file_alignment = (it != kv_uint.end()) ? it->second : 32;
    direct_io_alignment = min_alignment;
    if (file_alignment < min_alignment && !allow_unaligned_ranges) {
      LOG_STREAM_WARN()
          << "[DIRECT-IO] GGUF tensor alignment=" << file_alignment
          << " is below the required " << min_alignment
          << " bytes for this read path; direct I/O disabled.\n";
      return 0;
    }
    if (file_alignment < min_alignment) {
      LOG_STREAM_INFO()
          << "[DIRECT-IO] GGUF tensor alignment=" << file_alignment
          << " bytes; using " << direct_io_alignment
          << "-byte aligned read windows with scratch staging for unaligned "
             "tensor ranges.\n";
    }

    void *test_buf = nullptr;
    if (posix_memalign(&test_buf, (size_t)direct_io_alignment,
                       (size_t)direct_io_alignment) != 0)
      test_buf = nullptr;

    int n_enabled = 0;
    for (size_t i = 0; i < shard_fds.size(); ++i) {
      int new_fd = open(shard_paths[i].c_str(), O_RDONLY | O_DIRECT);
      bool ok = new_fd >= 0;
      if (ok && test_buf) {
        ssize_t r = pread(new_fd, test_buf, (size_t)direct_io_alignment, 0);
        ok = (r >= 0);
      } else if (ok && !test_buf) {
        ok = false; // couldn't allocate the test buffer -- don't trust an
                    // unverified open()
      }
      if (ok) {
        close(shard_fds[i]);
        shard_fds[i] = new_fd;
        shard_direct_io[i] = true;
        n_enabled++;
      } else {
        if (new_fd >= 0)
          close(new_fd);
        LOG_STREAM_WARN()
            << "[DIRECT-IO] Shard " << i << " (" << shard_paths[i]
            << ") does not support O_DIRECT on this filesystem -- "
               "falling back to buffered reads for this shard only.\n";
      }
    }
    if (test_buf)
      free(test_buf);
    if (file_fd >= 0 && !shard_fds.empty())
      file_fd = shard_fds[0];
    LOG_STREAM_INFO() << "[DIRECT-IO] Enabled on " << n_enabled << "/"
                      << shard_fds.size()
                      << " shard(s) (alignment=" << direct_io_alignment
                      << " bytes).\n";
    return n_enabled;
#endif
  }

  // Extract integer config value from KV metadata, trying multiple key patterns
  int get_config_int(const std::string &suffix, int fallback = 0) const {
    // Detect architecture prefix from general.architecture KV
    std::string arch_prefix;
    auto arch_it = kv_string.find("general.architecture");
    if (arch_it != kv_string.end())
      arch_prefix = arch_it->second + ".";

    // Try: detected arch prefix, then common fallbacks
    std::vector<std::string> prefixes;
    if (!arch_prefix.empty())
      prefixes.push_back(arch_prefix);
    for (const auto &p :
         {"qwen35moe.", "qwen3moe.", "qwen2moe.", "qwen.", "glm-dsa.", "glm.",
          "glm4.", "llama.", "deepseek2.", ""})
      prefixes.push_back(p);

    for (const auto &prefix : prefixes) {
      std::string key = prefix + suffix;
      auto it = kv_uint.find(key);
      if (it != kv_uint.end())
        return (int)it->second;
      auto it2 = kv_int.find(key);
      if (it2 != kv_int.end())
        return (int)it2->second;
    }
    return fallback;
  }

  float get_config_float(const std::string &suffix,
                         float fallback = 0.0f) const {
    std::string arch_prefix;
    auto arch_it = kv_string.find("general.architecture");
    if (arch_it != kv_string.end())
      arch_prefix = arch_it->second + ".";

    std::vector<std::string> prefixes;
    if (!arch_prefix.empty())
      prefixes.push_back(arch_prefix);
    for (const auto &p :
         {"qwen35moe.", "qwen3moe.", "qwen2moe.", "qwen.", "glm-dsa.", "glm.",
          "glm4.", "llama.", "deepseek2.", ""})
      prefixes.push_back(p);

    for (const auto &prefix : prefixes) {
      std::string key = prefix + suffix;
      auto it = kv_float.find(key);
      if (it != kv_float.end())
        return it->second;
    }
    return fallback;
  }

  std::string get_config_string(const std::string &suffix,
                                const std::string &fallback = "") const {
    std::string arch_prefix;
    auto arch_it = kv_string.find("general.architecture");
    if (arch_it != kv_string.end())
      arch_prefix = arch_it->second + ".";

    std::vector<std::string> prefixes;
    if (!arch_prefix.empty())
      prefixes.push_back(arch_prefix);
    for (const auto &p :
         {"qwen35moe.", "qwen3moe.", "qwen2moe.", "qwen.", "glm-dsa.", "glm.",
          "glm4.", "llama.", "deepseek2.", ""})
      prefixes.push_back(p);

    for (const auto &prefix : prefixes) {
      std::string key = prefix + suffix;
      auto it = kv_string.find(key);
      if (it != kv_string.end())
        return it->second;
    }
    return fallback;
  }

  // Array-typed counterparts of get_config_int/float above, same arch-
  // prefix-then-fixed-list-then-bare-suffix search order, reading from
  // kv_arr_int/kv_arr_float (read_kv_array's storage) instead of the scalar
  // maps. Added for DeepSeek-V4's per-layer swiglu_clamp_exp/shexp; generic
  // enough for any future per-layer array hparam, not DeepSeek-V4-specific
  // plumbing. Returns an empty vector (not `fallback` -- there's no single
  // sensible scalar fallback for a missing array) when the key isn't found
  // under any prefix; callers should treat empty as "feature inactive."
  std::vector<int64_t> get_config_int_array(const std::string &suffix) const {
    std::string arch_prefix;
    auto arch_it = kv_string.find("general.architecture");
    if (arch_it != kv_string.end())
      arch_prefix = arch_it->second + ".";

    std::vector<std::string> prefixes;
    if (!arch_prefix.empty())
      prefixes.push_back(arch_prefix);
    for (const auto &p :
         {"qwen35moe.", "qwen3moe.", "qwen2moe.", "qwen.", "glm-dsa.", "glm.",
          "glm4.", "llama.", "deepseek2.", ""})
      prefixes.push_back(p);

    for (const auto &prefix : prefixes) {
      std::string key = prefix + suffix;
      auto it = kv_arr_int.find(key);
      if (it != kv_arr_int.end())
        return it->second;
    }
    return {};
  }

  std::vector<float> get_config_float_array(const std::string &suffix) const {
    std::string arch_prefix;
    auto arch_it = kv_string.find("general.architecture");
    if (arch_it != kv_string.end())
      arch_prefix = arch_it->second + ".";

    std::vector<std::string> prefixes;
    if (!arch_prefix.empty())
      prefixes.push_back(arch_prefix);
    for (const auto &p :
         {"qwen35moe.", "qwen3moe.", "qwen2moe.", "qwen.", "glm-dsa.", "glm.",
          "glm4.", "llama.", "deepseek2.", ""})
      prefixes.push_back(p);

    for (const auto &prefix : prefixes) {
      std::string key = prefix + suffix;
      auto it = kv_arr_float.find(key);
      if (it != kv_arr_float.end())
        return it->second;
    }
    return {};
  }

  // 2026-08-31: user-facing model-load diagnostics.
  //
  // These go to stderr DIRECTLY (std::fprintf), not only through Logger.
  // That is the whole point: this project's configs set `log_path`
  // (e.g. debug.log), and Logger::log resolves `FILE* out = file_ ? file_
  // : stderr` -- so once a log file is configured, every LOG_ERROR goes to
  // that FILE and NOTHING reaches the terminal. A model that cannot be
  // loaded is exactly the case where the operator is staring at the
  // terminal wondering why nothing happened, so it must be visible there
  // regardless of log routing. Emitted IN ADDITION to the normal
  // LOG_STREAM_ERROR call, so the log file still contains the full record.
  static void warn_stderr(const std::string &msg) {
    std::fprintf(stderr, "[MODEL] %s\n", msg.c_str());
    std::fflush(stderr);
  }

  // What a file actually looks like on disk, independent of its extension
  // (an extension is a naming convention, not a format -- a .gguf file can
  // be a truncated download, an HTML error page saved by a failed fetch, or
  // a safetensors file someone renamed).
  enum class FileFormat { Unreadable, Empty, GGUF, Safetensors, Unknown };

  static FileFormat detect_file_format(const std::string &p) {
    std::ifstream f(p, std::ios::binary);
    if (!f.is_open()) return FileFormat::Unreadable;
    f.seekg(0, std::ios::end);
    std::streamoff sz = f.tellg();
    if (sz <= 0) return FileFormat::Empty;
    f.seekg(0, std::ios::beg);

    char magic[4] = {0};
    f.read(magic, 4);
    if (f.gcount() == 4 && std::string(magic, 4) == "GGUF") return FileFormat::GGUF;

    // safetensors: 8-byte little-endian header length, then that many bytes
    // of JSON beginning with '{'. Validated against the real file size so a
    // random 8 bytes that happen to look like a plausible length don't get
    // misreported as safetensors.
    f.seekg(0, std::ios::beg);
    uint64_t hdr = 0;
    f.read(reinterpret_cast<char *>(&hdr), 8);
    if (f.gcount() == 8 && hdr > 0 && hdr < (uint64_t)sz && hdr < (1ull << 32)) {
      char brace = 0;
      f.read(&brace, 1);
      if (f.gcount() == 1 && brace == '{') return FileFormat::Safetensors;
    }
    return FileFormat::Unknown;
  }

  // One place that turns a failed load into an actionable stderr message,
  // so every failure path reports the SAME way instead of each site
  // inventing its own wording.
  static void report_load_failure(const std::string &p) {
    switch (detect_file_format(p)) {
      case FileFormat::Unreadable:
        warn_stderr("ERROR: cannot open model file: " + p +
                    "  (does not exist, or no read permission)");
        break;
      case FileFormat::Empty:
        warn_stderr("ERROR: model file is empty (0 bytes): " + p +
                    "  (an interrupted or failed download?)");
        break;
      case FileFormat::Safetensors:
        // NOTE: src/safetensors_adapter.cpp exists in this tree but is NOT
        // in the build (verified: no object file in build/CMakeFiles), so
        // safetensors is a supported FORMAT that this particular binary
        // cannot currently load. Say exactly that rather than "unsupported
        // format", which would wrongly send the operator off to re-convert
        // a file that is actually fine.
        warn_stderr("ERROR: model file is safetensors: " + p +
                    "  (safetensors is supported, but this binary was built "
                    "without the safetensors adapter -- use a GGUF file, or "
                    "rebuild with src/safetensors_adapter.cpp included)");
        break;
      case FileFormat::GGUF:
        // Real GGUF magic, so the format is right and something else went
        // wrong (truncated/corrupt body, unreadable metadata). Say so
        // rather than blaming the format, which would send the operator
        // down the wrong path entirely.
        warn_stderr("ERROR: model file has a valid GGUF header but could not "
                    "be loaded: " + p +
                    "  (truncated, corrupt, or unreadable metadata -- see the "
                    "log for details)");
        break;
      case FileFormat::Unknown:
      default:
        warn_stderr("ERROR: unsupported model file format: " + p +
                    "  (only GGUF and safetensors are recognized; this file "
                    "matches neither)");
        break;
    }
  }

  bool load_header(const std::string &path) {
    std::vector<std::string> gguf_files;

    if (std::filesystem::is_directory(path)) {
      std::error_code dir_ec;
      for (std::filesystem::directory_iterator it(path, dir_ec), end;
           it != end && !dir_ec; it.increment(dir_ec)) {
        const auto &entry = *it;
        if (entry.path().extension() == ".gguf")
          gguf_files.push_back(entry.path().string());
      }
      if (dir_ec) {
        LOG_STREAM_ERROR() << "[GGUF] Cannot enumerate model directory " << path
                           << ": " << dir_ec.message() << "\n";
        return false;
      }
      if (gguf_files.empty()) {
        LOG_STREAM_ERROR() << "[GGUF] No .gguf files found in directory: "
                           << path << "\n";
        // stderr too -- see warn_stderr's own comment for why a log_path'd
        // run would otherwise show the operator nothing at all.
        warn_stderr("ERROR: no .gguf files found in model directory: " + path +
                    "  (only GGUF and safetensors are supported; check the "
                    "path points at the model, and that the download "
                    "completed)");
        return false;
      }
      std::sort(gguf_files.begin(), gguf_files.end());
    } else {
      // Report a missing/unreadable path here rather than letting it fall
      // through to load_shard's generic failure -- at this point we still
      // know it was given as a FILE path, which makes the message specific.
      if (!std::filesystem::exists(path)) {
        LOG_STREAM_ERROR() << "[GGUF] Model path does not exist: " << path << "\n";
        warn_stderr("ERROR: model path does not exist: " + path);
        return false;
      }
      // A split GGUF may be passed as its metadata-only first shard. Discover
      // the remaining numbered siblings so explicit first-shard configs work
      // the same way as directory configs.
      const std::filesystem::path input(path);
      const std::string filename = input.filename().string();
      const size_t of_pos = filename.find("-of-");
      const size_t index_dash =
          of_pos == std::string::npos ? std::string::npos
                                      : filename.rfind('-', of_pos - 1);
      if (of_pos != std::string::npos && index_dash != std::string::npos) {
        const std::string prefix = filename.substr(0, index_dash);
        const size_t total_begin = of_pos + 4;
        const size_t total_end = filename.find_first_not_of("0123456789",
                                                              total_begin);
        const std::string total =
            filename.substr(total_begin, total_end - total_begin);
        std::error_code dir_ec;
        for (std::filesystem::directory_iterator it(input.parent_path(), dir_ec),
             end;
             it != end && !dir_ec; it.increment(dir_ec)) {
          const auto &entry = *it;
          if (entry.path().extension() != ".gguf")
            continue;
          const std::string candidate = entry.path().filename().string();
          if (candidate.rfind(prefix + "-", 0) != 0)
            continue;
          const size_t candidate_of = candidate.find("-of-");
          if (candidate_of == std::string::npos ||
              candidate.substr(candidate_of + 4, total.size()) != total)
            continue;
          gguf_files.push_back(entry.path().string());
        }
        if (dir_ec)
          LOG_STREAM_WARN() << "[GGUF] Could not enumerate sibling shards in "
                            << input.parent_path() << ": " << dir_ec.message()
                            << " -- trying the requested shard only.\n";
      }
      if (gguf_files.empty())
        gguf_files.push_back(path);
      else
        std::sort(gguf_files.begin(), gguf_files.end());
    }

    LOG_STREAM_INFO() << "[GGUF] Loading " << gguf_files.size()
                      << " shard(s)...\n";

    for (size_t shard = 0; shard < gguf_files.size(); ++shard) {
      if (!load_shard(gguf_files[shard], (int)shard)) {
        LOG_STREAM_ERROR() << "[GGUF] Failed to load shard: "
                           << gguf_files[shard] << "\n";
        // Classify WHY before reporting: "failed to load" alone does not
        // tell the operator whether the file is missing, empty, the wrong
        // format, or corrupt -- four problems with four different fixes.
        report_load_failure(gguf_files[shard]);
        return false;
      }
    }

    resolved_file_path = gguf_files[0];
    file_fd = shard_fds.empty() ? -1 : shard_fds[0];
    shard_paths = gguf_files;

    LOG_STREAM_INFO() << "[GGUF] Total tensors loaded: " << tensors.size()
                      << "\n";
    return file_fd >= 0;
  }

  static void unmap_list(std::vector<void *> &maps, std::vector<size_t> &sizes) {
    for (size_t i = 0; i < maps.size(); ++i)
      if (maps[i]) ::munmap(maps[i], sizes[i]);
    maps.clear();
    sizes.clear();
  }

  ~GGUFFile() {
    unmap_list(shard_maps, shard_map_sizes);
    for (int fd : shard_fds) {
      if (fd >= 0)
        close(fd);
    }
  }

private:
  bool load_shard(const std::string &shard_path, int shard_idx) {
    std::ifstream f(shard_path, std::ios::binary);
    if (!f.is_open())
      return false;

    char magic[4];
    f.read(magic, 4);
    if (std::string(magic, 4) != "GGUF")
      return false;

    uint32_t shard_version;
    f.read(reinterpret_cast<char *>(&shard_version), 4);
    if (shard_idx == 0)
      version = shard_version;

    uint64_t tensor_count = 0, kv_count = 0;
    f.read(reinterpret_cast<char *>(&tensor_count), 8);
    f.read(reinterpret_cast<char *>(&kv_count), 8);

    // 2026-08-31: reject a TRUNCATED/corrupt GGUF here, at the point the
    // damage is actually detectable, instead of letting it through.
    // Previously a file with valid "GGUF" magic but a short/garbage body
    // sailed past this function entirely -- the reads above silently failed,
    // tensor_count/kv_count kept whatever garbage they had, and the run died
    // much later with "Tokenizer vocabulary table completely missing from
    // GGUF metadata". That message sends the operator hunting for a
    // tokenizer problem when the real fault is a truncated download, which
    // is a genuinely misleading diagnostic, not just a terse one.
    //
    // Two checks: (a) the stream must still be good -- a short file makes
    // these 8-byte reads fail, which is the direct truncation signal; and
    // (b) the counts must be within sane bounds. The bounds are deliberately
    // very loose (millions), far above any real model, so they only ever
    // catch garbage, never a legitimately large file.
    constexpr uint64_t kMaxPlausibleTensors = 10'000'000ull;
    constexpr uint64_t kMaxPlausibleKV      =  1'000'000ull;
    if (!f.good() || tensor_count > kMaxPlausibleTensors || kv_count > kMaxPlausibleKV) {
      LOG_STREAM_ERROR() << "[GGUF] Corrupt or truncated shard header: " << shard_path
                         << " (tensor_count=" << tensor_count
                         << " kv_count=" << kv_count
                         << " stream_ok=" << (f.good() ? "yes" : "no") << ")\n";
      return false;
    }

    LOG_STREAM_DEBUG() << "[GGUF]   shard " << shard_idx << ": " << tensor_count
                       << " tensors, " << kv_count << " kv pairs"
                       << " ("
                       << std::filesystem::path(shard_path).filename().string()
                       << ")\n";

    // Read KV metadata — store values for architecture config extraction.
    // general.alignment is captured a SECOND time, locally, right below --
    // see that capture's own comment for why kv_uint alone (shared/never
    // reset across shards) isn't safe to read back for this one key.
    uint64_t shard_alignment = 32;      // GGUF spec default when absent
    bool shard_alignment_seen = false;
    for (uint64_t i = 0; i < kv_count; ++i) {
      std::string key = read_string(f);
      uint32_t value_type;
      f.read(reinterpret_cast<char *>(&value_type), 4);
      read_kv_value(f, key, value_type);
      // Real, confirmed bug (found 2026-09 stitching a repacked model's
      // shards, general.alignment=4096, together with a raw/un-repacked
      // shard from a different release that never sets this key at all):
      // kv_uint is a GGUFFile-wide member map that persists across every
      // shard in this loop, never reset per shard -- so once an EARLIER
      // shard defines general.alignment, that value silently survives into
      // a LATER shard's own data_start computation below even when the
      // later shard's own header never mentions the key. Confirmed via a
      // real isolated repro: the same tensor decoded 0 NaN read alone from
      // its own file, but ~4.7% NaN once loaded after a repacked shard --
      // traced to data_start computed with the wrong (leaked) alignment,
      // off by exactly the 4096-vs-32 rounding difference. kv_uint[key] is
      // still written above (read_kv_value's normal behavior, used
      // elsewhere for real per-model KV lookups where "last shard wins" is
      // fine, e.g. architecture hparams that ARE genuinely consistent
      // across every shard of the SAME model) -- but reading it back
      // immediately, right here, right after this exact call, captures
      // ONLY what THIS shard's own KV block just said, before any later
      // shard in the loop gets a chance to touch the same map entry.
      if (key == "general.alignment") {
        auto it = kv_uint.find(key);
        if (it != kv_uint.end()) { shard_alignment = it->second; shard_alignment_seen = true; }
      }
    }

    // Read tensor info
    for (uint64_t i = 0; i < tensor_count; ++i) {
      GGUFTensor t;
      t.name = read_string(f);
      uint32_t n_dimensions;
      f.read(reinterpret_cast<char *>(&n_dimensions), 4);
      t.shape.resize(n_dimensions);
      f.read(reinterpret_cast<char *>(t.shape.data()), n_dimensions * 8);
      f.read(reinterpret_cast<char *>(&t.type), 4);
      f.read(reinterpret_cast<char *>(&t.offset), 8);

      uint64_t numel = 1;
      for (auto dim : t.shape)
        numel *= dim;
      t.size_bytes = calculate_size(numel, t.type);
      t.shard_idx = shard_idx;
      tensors[t.name] = t;
    }

    uint64_t current_pos = f.tellg();
    f.close();

    int fd = open(shard_path.c_str(), O_RDONLY);
    if (fd < 0)
      return false;

    // Tell the kernel this fd's access pattern is NOT sequential: expert
    // slabs are read by jumping to whatever offset the router picked
    // this token (effectively random across a 100+GB file), which is
    // the opposite of what Linux's default readahead heuristic assumes.
    // Left as the default (no hint), the kernel can spend real I/O
    // bandwidth speculatively reading pages after each request that
    // this workload will never touch, and/or repeatedly shrink/reset
    // its readahead window on each detected "random" jump -- both pure
    // overhead for this access pattern. POSIX_FADV_RANDOM (offset=0,
    // len=0 = whole file) disables that heuristic outright. Zero
    // correctness risk (a kernel hint, not a behavior change) and zero
    // concurrency risk -- a pure candidate fix for the measured low
    // (~1.17 GB/s) aggregate disk throughput, cheaper to test than
    // re-sharding the model file.
    posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);

    // Compute data_start for this shard and adjust tensor offsets.
    // general.alignment (GGUF spec, default 32 if absent) governs this --
    // previously hardcoded to 32 unconditionally, silently ignoring the
    // KV key entirely. Harmless for every file encountered so far (32 is
    // ggml's own writer default, so virtually all real-world GGUF files
    // use it) but a real, necessary fix for a repack tool that
    // deliberately writes a LARGER alignment (4096, for O_DIRECT-eligible
    // per-expert offsets) -- without this, this loader would compute the
    // wrong data_start for such a file (assume 32-aligned when the file
    // is actually 4096-aligned) and silently misinterpret every tensor
    // offset as a result.
    //
    // Uses shard_alignment (captured locally, right above, from THIS
    // shard's own KV block only) rather than reading kv_uint back here --
    // NOT equivalent when a model mixes shards from different releases:
    // kv_uint is shared across the whole multi-shard load, so if an
    // EARLIER shard set general.alignment and this shard's own header
    // never mentions the key, a kv_uint lookup here would silently reuse
    // the earlier shard's value for this one's offsets. Real, confirmed
    // bug (see the capture site's own comment for the full incident) --
    // this per-shard local is the actual fix, not the read_kv_value call
    // above, which still (correctly) writes into the shared map for every
    // OTHER KV consumer that wants "last shard wins" semantics.
    uint64_t alignment = shard_alignment_seen ? shard_alignment : 32;
    uint64_t data_start = (current_pos + alignment - 1) & ~(alignment - 1);
    for (auto &[name, t] : tensors) {
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

  std::string read_string(std::ifstream &f) {
    uint64_t len;
    f.read(reinterpret_cast<char *>(&len), 8);
    std::string s(len, '\0');
    f.read(&s[0], len);
    return s;
  }

  // GGUF KV types: 0=UINT8, 1=INT8, 2=UINT16, 3=INT16, 4=UINT32, 5=INT32,
  // 6=FLOAT32, 7=BOOL, 8=STRING, 9=ARRAY, 10=UINT64, 11=INT64, 12=FLOAT64
  void read_kv_value(std::ifstream &f, const std::string &key, uint32_t type) {
    switch (type) {
    case 0: {
      uint8_t v;
      f.read(reinterpret_cast<char *>(&v), 1);
      kv_uint[key] = v;
      break;
    }
    case 1: {
      int8_t v;
      f.read(reinterpret_cast<char *>(&v), 1);
      kv_int[key] = v;
      break;
    }
    case 2: {
      uint16_t v;
      f.read(reinterpret_cast<char *>(&v), 2);
      kv_uint[key] = v;
      break;
    }
    case 3: {
      int16_t v;
      f.read(reinterpret_cast<char *>(&v), 2);
      kv_int[key] = v;
      break;
    }
    case 4: {
      uint32_t v;
      f.read(reinterpret_cast<char *>(&v), 4);
      kv_uint[key] = v;
      break;
    }
    case 5: {
      int32_t v;
      f.read(reinterpret_cast<char *>(&v), 4);
      kv_int[key] = v;
      break;
    }
    case 6: {
      float v;
      f.read(reinterpret_cast<char *>(&v), 4);
      kv_float[key] = v;
      break;
    }
    case 7: {
      uint8_t v;
      f.read(reinterpret_cast<char *>(&v), 1);
      kv_uint[key] = v;
      break;
    }
    case 8: {
      kv_string[key] = read_string(f);
      break;
    }
    case 9: {
      uint32_t sub_type;
      f.read(reinterpret_cast<char *>(&sub_type), 4);
      uint64_t arr_len;
      f.read(reinterpret_cast<char *>(&arr_len), 8);
      read_kv_array(f, key, sub_type, arr_len);
      break;
    }
    case 10: {
      uint64_t v;
      f.read(reinterpret_cast<char *>(&v), 8);
      kv_uint[key] = v;
      break;
    }
    case 11: {
      int64_t v;
      f.read(reinterpret_cast<char *>(&v), 8);
      kv_int[key] = v;
      break;
    }
    case 12: {
      double v;
      f.read(reinterpret_cast<char *>(&v), 8);
      kv_float[key] = (float)v;
      break;
    }
    default:
      break;
    }
  }

  // GGUF array-typed KV (case 9 above). sub_type uses the SAME numbering as
  // read_kv_value's own `type` param (0=UINT8 .. 12=FLOAT64). Only numeric
  // sub-types are stored -- into kv_arr_int (every integer/bool width,
  // sign-extended/widened to int64_t) or kv_arr_float (float32 and float64,
  // narrowed to float) -- mirroring how the scalar kv_uint/kv_int/kv_float
  // maps already collapse multiple GGUF widths into one C++ type. A
  // string-element or nested-array-element sub_type is skipped element-by-
  // element via the existing skip_kv_value, exactly matching this function's
  // prior "skip entirely" behavior for arrays -- no known hparam in this
  // codebase needs a string or nested array, so there's nothing to widen the
  // storage for yet; get_config_int_array/get_config_float_array (below)
  // are the read side of the two cases this DOES store.
  void read_kv_array(std::ifstream &f, const std::string &key,
                     uint32_t sub_type, uint64_t arr_len) {
    switch (sub_type) {
    case 0: case 1: case 7: { // uint8 / int8 / bool
      std::vector<int64_t> v;
      v.reserve(arr_len);
      for (uint64_t j = 0; j < arr_len; ++j) {
        uint8_t x;
        f.read(reinterpret_cast<char *>(&x), 1);
        v.push_back((int64_t)x);
      }
      kv_arr_int[key] = std::move(v);
      break;
    }
    case 2: case 3: { // uint16 / int16
      std::vector<int64_t> v;
      v.reserve(arr_len);
      for (uint64_t j = 0; j < arr_len; ++j) {
        int16_t x;
        f.read(reinterpret_cast<char *>(&x), 2);
        v.push_back((int64_t)x);
      }
      kv_arr_int[key] = std::move(v);
      break;
    }
    case 4: case 5: { // uint32 / int32
      std::vector<int64_t> v;
      v.reserve(arr_len);
      for (uint64_t j = 0; j < arr_len; ++j) {
        int32_t x;
        f.read(reinterpret_cast<char *>(&x), 4);
        v.push_back((int64_t)x);
      }
      kv_arr_int[key] = std::move(v);
      break;
    }
    case 6: { // float32
      std::vector<float> v;
      v.reserve(arr_len);
      for (uint64_t j = 0; j < arr_len; ++j) {
        float x;
        f.read(reinterpret_cast<char *>(&x), 4);
        v.push_back(x);
      }
      kv_arr_float[key] = std::move(v);
      break;
    }
    case 10: case 11: { // uint64 / int64
      std::vector<int64_t> v;
      v.reserve(arr_len);
      for (uint64_t j = 0; j < arr_len; ++j) {
        int64_t x;
        f.read(reinterpret_cast<char *>(&x), 8);
        v.push_back(x);
      }
      kv_arr_int[key] = std::move(v);
      break;
    }
    case 12: { // float64
      std::vector<float> v;
      v.reserve(arr_len);
      for (uint64_t j = 0; j < arr_len; ++j) {
        double x;
        f.read(reinterpret_cast<char *>(&x), 8);
        v.push_back((float)x);
      }
      kv_arr_float[key] = std::move(v);
      break;
    }
    default: // string (8) or nested array (9) elements -- not stored today
      for (uint64_t j = 0; j < arr_len; ++j)
        skip_kv_value(f, sub_type);
      break;
    }
  }

  void skip_kv_value(std::ifstream &f, uint32_t type) {
    switch (type) {
    case 0:
    case 1:
    case 7:
      f.seekg(1, std::ios::cur);
      break;
    case 2:
    case 3:
      f.seekg(2, std::ios::cur);
      break;
    case 4:
    case 5:
    case 6:
      f.seekg(4, std::ios::cur);
      break;
    case 10:
    case 11:
    case 12:
      f.seekg(8, std::ios::cur);
      break;
    case 8: {
      uint64_t len;
      f.read(reinterpret_cast<char *>(&len), 8);
      f.seekg(len, std::ios::cur);
      break;
    }
    default:
      break;
    }
  }

  // Block sizes and bytes per block live in memory/quant_table.hpp (one table for the loader, the
  // offload store and the MMQ checks). Types not in it count one byte per element, as before.
  uint64_t calculate_size(uint64_t numel, GGUFType type) {
    return zeromoe::quant::tensor_bytes(static_cast<uint32_t>(type), numel);
  }
};
