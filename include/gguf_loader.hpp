#pragma once
#include "cufile_io.hpp"
#include "logger.hpp"
#include <algorithm>
#include <cstdint>
#include <fcntl.h>
#include <sys/stat.h>   // ::stat, for open_mirror's per-shard size check
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>
#include <unordered_map>
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
};

enum class GatingFunc {
  Softmax,
  Sigmoid,
  SqrtSoftplus, // sqrt(softplus(x)) -- DeepSeek-V4
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
  // attention sinks, SWA). Phase 1 (correctness within each layer's local
  // sliding window) only -- see models/family_ops.cpp's DeepSeek_V4 table
  // entry and CLAUDE.md's DeepSeek-V4 plan writeup. Every field here
  // defaults to "feature inactive" so populating them is a strict no-op for
  // every other family.
  int hc_mult = 1;              // hyper_connection.count; 1 = inactive
  int hc_sinkhorn_iters = 0;    // hyper_connection.sinkhorn_iterations
  float hc_eps = 1e-6f;         // hyper_connection.epsilon
  int o_group_count = 0;        // attention.output_group_count; 0 = plain wo
  int o_lora_rank = 0;          // attention.output_lora_rank
  int hash_layer_count = 0;     // hash_layer_count; 0 = no hash-routed layers
  int sliding_window = 0;       // attention.sliding_window; 0 = SWA inactive
  std::vector<float> swiglu_clamp_exp;   // per-layer, size n_layers when set
  std::vector<float> swiglu_clamp_shexp; // per-layer, size n_layers when set
};

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

  // One cuFile handle per shard_fds, registered by enable_cufile() below
  // (empty/all-invalid until then). Index-matched to shard_fds/
  // shard_direct_io, same convention as everything else per-shard here.
  std::vector<cufile_io::FileHandle> shard_cufile_handles;

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
      n_ok++;
    }
    LOG_STREAM_INFO() << "[MIRROR] Second replica: " << n_ok << "/" << shard_paths.size()
                      << " shard(s) opened from " << mirror_dir
                      << " -- expert reads can now stripe across both devices "
                         "(io_uring drives both queues from one submission).\n";
    return n_ok;
#endif
  }

  // Returns a pointer to the registered cuFile handle for this tensor's
  // shard, or nullptr if cuFile was never enabled (or failed) for that
  // specific shard -- callers (GpuTopology::upload_expert_cufile) must
  // check for null AND h->valid before using it, exactly like
  // direct_io_active_for's own per-shard granularity (one shard failing
  // O_DIRECT/cuFile registration doesn't disable it for every OTHER
  // shard that succeeded).
  const cufile_io::FileHandle *
  cufile_handle_for_tensor(const GGUFTensor &t) const {
    if (t.shard_idx >= 0 && (size_t)t.shard_idx < shard_cufile_handles.size())
      return &shard_cufile_handles[t.shard_idx];
    return nullptr;
  }

  // Opt-in: reopen every shard's fd with O_DIRECT, one shard at a time,
  // ONLY if (a) this platform supports O_DIRECT at all (Linux; this
  // project's only supported build target per CLAUDE.md, but guarded
  // anyway rather than assumed) and (b) this file's own
  // general.alignment KV value (written by tools/gguf_tool pack,
  // defaults to 32 for any file that was never repacked) is at least
  // `min_alignment` -- a plain, un-repacked GGUF has 32-byte tensor
  // offsets, which can never satisfy O_DIRECT's block-alignment
  // requirement, so direct I/O is refused outright for such a file
  // rather than silently attempting misaligned reads. Even when the
  // alignment check passes, each shard gets a real test read (not just
  // a successful open()) before being trusted -- some filesystems
  // (network mounts, some overlay/container filesystems, tmpfs) accept
  // O_DIRECT at open() but reject the actual read with EINVAL; a shard
  // that fails either step keeps its ORIGINAL buffered fd untouched
  // and is simply left out of shard_direct_io, never a hard error --
  // this feature is a pure performance opt-in, never something a
  // caller should have to work around if it doesn't pan out for a
  // given filesystem. Returns the number of shards that ended up with
  // real O_DIRECT active.
  int enable_direct_io(uint64_t min_alignment = 4096) {
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
    if (file_alignment < min_alignment) {
      LOG_STREAM_WARN() << "[DIRECT-IO] This model's general.alignment="
                        << file_alignment << " is below the required "
                        << min_alignment
                        << " bytes -- this file was "
                           "never repacked with tools/gguf_tool pack (or used "
                           "a smaller --align). "
                           "Direct I/O disabled, all reads stay buffered.\n";
      return 0;
    }
    direct_io_alignment = file_alignment;

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

  // Opt-in: register every O_DIRECT-active shard's fd with the cuFile
  // driver (see cufile_io.hpp). MUST be called AFTER enable_direct_io()
  // has already run and actually enabled O_DIRECT on at least one shard
  // -- cuFile needs the SAME O_DIRECT fd, not a fresh buffered one (see
  // EngineConfig's own cuda_cufile-requires-direct_io validation, which
  // is the higher-level guard; this function additionally re-checks per
  // shard since direct_io can partially fail per-shard even when the
  // config flag is on). Per-shard granularity mirrors enable_direct_io
  // exactly: a shard that isn't O_DIRECT-active, or whose cuFile
  // registration itself fails, is simply left with an invalid handle
  // (cufile_handle_for_tensor returns valid=false for it) rather than
  // disabling the feature for every other shard. Returns the number of
  // shards that ended up with a real registered handle.
  int enable_cufile() {
    shard_cufile_handles.assign(shard_fds.size(), cufile_io::FileHandle{});
    if (!cufile_io::available()) {
      LOG_STREAM_WARN()
          << "[CUFILE] Not compiled in (cufile.h/libcufile not found at build "
             "time -- see CMakeLists.txt) or driver unavailable -- falling "
             "back "
             "to the RAM-mediated upload path for every shard.\n";
      return 0;
    }
    int n_enabled = 0;
    for (size_t i = 0; i < shard_fds.size(); ++i) {
      if (i >= shard_direct_io.size() || !shard_direct_io[i]) {
        LOG_STREAM_WARN()
            << "[CUFILE] Shard " << i << " (" << shard_paths[i]
            << ") is not O_DIRECT-active -- skipping cuFile registration for "
               "this "
               "shard (falls back to RAM-mediated upload for its tensors).\n";
        continue;
      }
      cufile_io::FileHandle h = cufile_io::register_file(shard_fds[i]);
      if (h.valid) {
        shard_cufile_handles[i] = h;
        n_enabled++;
      } else {
        LOG_STREAM_WARN() << "[CUFILE] Shard " << i << " (" << shard_paths[i]
                          << ") failed cuFile handle registration -- falling "
                             "back to RAM-mediated "
                             "upload for this shard's tensors.\n";
      }
    }
    LOG_STREAM_INFO() << "[CUFILE] Registered " << n_enabled << "/"
                      << shard_fds.size()
                      << " shard(s) with the cuFile driver.\n";
    return n_enabled;
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

  bool load_header(const std::string &path) {
    std::vector<std::string> gguf_files;

    if (std::filesystem::is_directory(path)) {
      for (const auto &entry : std::filesystem::directory_iterator(path)) {
        if (entry.path().extension() == ".gguf")
          gguf_files.push_back(entry.path().string());
      }
      if (gguf_files.empty()) {
        LOG_STREAM_ERROR() << "[GGUF] No .gguf files found in directory: "
                           << path << "\n";
        return false;
      }
      std::sort(gguf_files.begin(), gguf_files.end());
    } else {
      gguf_files.push_back(path);
    }

    LOG_STREAM_INFO() << "[GGUF] Loading " << gguf_files.size()
                      << " shard(s)...\n";

    for (size_t shard = 0; shard < gguf_files.size(); ++shard) {
      if (!load_shard(gguf_files[shard], (int)shard)) {
        LOG_STREAM_ERROR() << "[GGUF] Failed to load shard: "
                           << gguf_files[shard] << "\n";
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

  ~GGUFFile() {
    // Deregister BEFORE closing the underlying fd -- cuFile's own
    // handle owns a reference tied to that fd staying open/valid.
    for (auto &h : shard_cufile_handles) {
      cufile_io::deregister_file(h);
    }
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

    LOG_STREAM_DEBUG() << "[GGUF]   shard " << shard_idx << ": " << tensor_count
                       << " tensors, " << kv_count << " kv pairs"
                       << " ("
                       << std::filesystem::path(shard_path).filename().string()
                       << ")\n";

    // Read KV metadata — store values for architecture config extraction
    for (uint64_t i = 0; i < kv_count; ++i) {
      std::string key = read_string(f);
      uint32_t value_type;
      f.read(reinterpret_cast<char *>(&value_type), 4);
      read_kv_value(f, key, value_type);
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
    // offset as a result. Read from kv_uint first (already populated by
    // this SAME shard's own KV loop above, and/or an earlier shard's, if
    // general.alignment is only present once -- see tools/gguf_tool pack
    // for why the repack tool writes it into every shard defensively
    // rather than relying on load order).
    auto align_it = kv_uint.find("general.alignment");
    uint64_t alignment = (align_it != kv_uint.end() && align_it->second > 0)
                             ? align_it->second
                             : 32;
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

  uint64_t calculate_size(uint64_t numel, GGUFType type) {
    uint32_t t = (uint32_t)type;
    // Block size and type_size from ggml (block_size=32 for Q4/Q5/Q8_0, 256 for
    // K-quants)
    if (t == 0)
      return numel * 4; // F32
    if (t == 1)
      return numel * 2; // F16
    if (t == 2)
      return (numel / 32) * 18; // Q4_0: 2+16
    if (t == 3)
      return (numel / 32) * 20; // Q4_1: 2+2+16
    if (t == 6)
      return (numel / 32) * 22; // Q5_0: 2+4+16
    if (t == 7)
      return (numel / 32) * 24; // Q5_1: 2+2+4+16
    if (t == 8)
      return (numel / 32) * 34; // Q8_0: 2+32
    if (t == 9)
      return (numel / 32) * 36; // Q8_1: 4+32 (with sum)
    if (t == 10)
      return (numel / 256) * 84; // Q2_K
    if (t == 11)
      return (numel / 256) * 110; // Q3_K
    if (t == 12)
      return (numel / 256) * 144; // Q4_K
    if (t == 13)
      return (numel / 256) * 176; // Q5_K
    if (t == 14)
      return (numel / 256) * 210; // Q6_K
    if (t == 15)
      return (numel / 256) * 292; // Q8_K
    if (t == 16)
      return (numel / 256) * 66; // IQ2_XXS
    if (t == 17)
      return (numel / 256) * 74; // IQ2_XS
    if (t == 18)
      return (numel / 256) * 98; // IQ3_XXS
    if (t == 19)
      return (numel / 256) * 50; // IQ1_S
    if (t == 20)
      return (numel / 32) * 18; // IQ4_NL (block size 32, not 256 — like Q4_0;
                                // was (numel/256)*132, ~20% too small)
    if (t == 21)
      return (numel / 256) * 110; // IQ3_S
    if (t == 22)
      return (numel / 256) * 82; // IQ2_S
    if (t == 23)
      return (numel / 256) * 136; // IQ4_XS
    if (t == 26)
      return numel * 4; // I32 -- raw 32-bit ints, 4 bytes/element, no block
                        // packing. Needed for DeepSeek-V4's
                        // ffn_gate_tid2eid.weight (hash-routed MoE
                        // token-id->expert-id lookup table) -- without this
                        // case it fell through to the `return numel;`
                        // fallback below (1 byte/element, 4x too small),
                        // which would have both truncated this tensor's own
                        // read AND corrupted every subsequent tensor's
                        // sequential offset in the same shard.
    if (t == 27)
      return numel * 8; // I64 -- see the enum's own comment for why this
                        // case exists defensively alongside I32.
    // FIX: these 4 cases used the SAME stale numbering the GGUFType enum
    // itself just got corrected for (see that enum's own comment) -- t==24/
    // 25/29/30 matched real ggml's I8/I16/IQ1_M/BF16, not IQ1_M/BF16/
    // TQ1_0/TQ2_0. Real values now (confirmed against build-metal/_deps/
    // ggml-src/include/ggml.h directly): IQ1_M=29, BF16=30, TQ1_0=34,
    // TQ2_0=35. The per-element byte formulas themselves are unchanged --
    // only which raw type id triggers each one.
    if (t == 29)
      return (numel / 256) * 56; // IQ1_M
    if (t == 30)
      return numel * 2; // BF16
    if (t == 34)
      return (numel / 256) * 54; // TQ1_0
    if (t == 35)
      return (numel / 256) * 66; // TQ2_0
    if (t == 39)
      return (numel / 32) * 17; // MXFP4: block=32 elements, 1(e)+16(qs) bytes
    if (t == 40)
      return (numel / 64) * 36; // NVFP4: block=64 elements, 4(d)+32(qs) bytes
    // IQ2_M: no real GGML_TYPE_* slot in the currently-linked ggml at all
    // (see the enum's own comment) -- no formula to assign a number to.
    return numel;                // fallback: 1 byte per element
  }
};