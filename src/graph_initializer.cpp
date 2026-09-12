#include "engine.hpp"
#include "kquant_dequant.hpp" // gpu_type_supported
#include "dense_dispatch.hpp" // set_swiglu_clamp_fused (GLM5Next)
#include "logger.hpp"
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <tuple>
#include <vector>

class GGUFGraphInitializer {
public:
  static bool build_model_graph(StreamingEngine &engine, int max_ctx);
};

// Several call sites (execution_layer.cpp's rmsnorm_row/router-bias reads,
// attention_layer.cpp's Q/K-norm reads, dense_dispatch.hpp's rmsnorm_proj CPU
// fallback) reinterpret a norm-weight or bias tensor's host_ptr as float*
// UNCONDITIONALLY, trusting that norm/bias tensors are always stored F32 in
// the GGUF (true for virtually every real quantization scheme, but never
// actually verified against THIS file). If any one of these is quantized or
// F16 instead, every element read through that reinterpret is garbage from
// byte zero -- and since one norm weight feeds every token/every downstream
// computation for its layer, this exactly matches a real symptom hit this
// session: a layer's router weights AND every expert's FFN going NaN
// simultaneously, from a single shared input (see moe_forward_gguf's
// scratch.normed). This is a load-time, whole-model version of the runtime
// check added at that one call site -- catches EVERY layer at once, at
// startup, instead of only the specific layer a given generation run
// happens to reach.
static void verify_float_reinterpret_tensors(const StreamingEngine &engine) {
  int checked = 0, bad = 0;
  for (const auto &kv : engine.resident_layers) {
    const std::string &name = kv.first;
    bool is_norm_or_bias =
        name.find("norm") != std::string::npos ||
        (name.size() >= 5 && name.compare(name.size() - 5, 5, ".bias") == 0);
    if (!is_norm_or_bias)
      continue;
    ++checked;
    if (kv.second.meta.type != GGUFType::F32) {
      ++bad;
      LOG_ERROR(
          "[LOAD-CHECK] %s type=%u (NOT F32) -- this tensor is read via an "
          "unconditional reinterpret_cast<float*>(host_ptr) somewhere in the "
          "forward "
          "path; every value read through it will be garbage, not just "
          "imprecise.\n",
          name.c_str(), (unsigned)kv.second.meta.type);
    }
  }
  if (bad > 0) {
    LOG_STREAM_ERROR() << "[LOAD-CHECK] " << bad << "/" << checked
                       << " norm/bias tensors are NOT F32 -- see lines above. "
                          "This alone can explain "
                          "whole-layer NaN corruption with no bug in any "
                          "matmul/dequant kernel at all.\n";
  } else {
    LOG_STREAM_INFO() << "[LOAD-CHECK] All " << checked
                      << " norm/bias tensors are F32 as every "
                         "reinterpret_cast<float*> site assumes.\n";
  }
}

// Comprehensive load-time audit: every attention/router/FFN weight tensor's
// dimensions, as ArchitectureConfig derives them from GGUF metadata KEYS,
// checked against that tensor's REAL shape as loaded from the file (its own
// ne[] dims -- ground truth, independent of any metadata key
// interpretation). This is the generalized form of the check that actually
// found this session's real corruption bug: qk_nope was silently computed
// as 256 (reading attention.key_length_mla directly) when attn_k_b's real
// ne[0] was 192 (key_length_mla is nope+rope COMBINED, not nope alone --
// confirmed against llama.cpp's deepseek2.cpp). Any OTHER metadata key with
// a similar "doesn't mean what the name suggests" issue would silently
// produce the exact same class of corruption -- a dense_proj/dequant call
// reading the wrong number of elements from that tensor's real bytes, at
// every layer, every token -- so every dimension the live MLA attention/
// router path actually uses gets checked here, once, at load time, rather
// than only being discoverable by chasing bad output through a full
// generation run each time.
static void verify_tensor_dimensions(StreamingEngine &engine) {
  auto &file = engine.model_file;
  auto &c = file.architecture_config;
  if (c.cache_shape != CacheShape::MLAAbsorbed)
    return; // this tensor set (attn_k_b/attn_v_b absorption) is specific to
            // the absorbed-MLA cache shape (GLM-5.2/DeepSeek-V3) -- DeepSeek-
            // V4's MLARawSWA has no absorption tensors to check here at all.

  int mismatches = 0;
  auto check = [&](const std::string &name, uint64_t expected) {
    auto it = file.tensors.find(name);
    if (it == file.tensors.end())
      return; // absence is reported elsewhere (dense manifest coverage)
    uint64_t real = 1;
    for (auto d : it->second.shape)
      real *= d;
    if (real != expected) {
      ++mismatches;
      std::string shape_str;
      for (auto d : it->second.shape)
        shape_str += std::to_string(d) + ",";
      LOG_ERROR(
          "[DIM-CHECK] %s: real numel=%llu (shape=[%s]) != expected=%llu "
          "(derived from ArchitectureConfig) -- a dense_proj/dequant call "
          "using "
          "the expected size will read the WRONG number of elements from this "
          "tensor's real bytes.\n",
          name.c_str(), (unsigned long long)real, shape_str.c_str(),
          (unsigned long long)expected);
    }
  };

  int D = c.hidden;
  int H = c.n_heads;
  int qh = c.qk_nope + c.qk_rope;
  int kv_dim = c.kv_lora + c.qk_rope;
  int q_lora = file.get_config_int("attention.q_lora_rank", 2048);

  check("token_embd.weight", (uint64_t)c.vocab * D);
  check("output.weight", (uint64_t)c.vocab * D);
  check("output_norm.weight", (uint64_t)D);

  // Representative MLA layer (dims should be uniform across every MLA layer)
  int l = std::min(c.first_dense, std::max(0, c.n_layers - 1));
  // GLM5Next: layer `first_dense` may be a recurrent KDA layer, which has
  // NONE of the absorbed-MLA tensors below -- every check would silently
  // skip (absent-tensor branch), defeating the audit. Pick the first DSA
  // (full-attention) layer instead.
  if (detect_model_family(file) == ModelFamily::GLM5NEXT) {
    for (int i = 0; i < c.n_layers; ++i) {
      if (!glm5next_is_kda_layer(c, i)) { l = i; break; }
    }
  }
  std::string p = "blk." + std::to_string(l) + ".";
  check(p + "attn_norm.weight", (uint64_t)D);
  check(p + "attn_q_a.weight", (uint64_t)D * q_lora);
  check(p + "attn_q_a_norm.weight", (uint64_t)q_lora);
  check(p + "attn_q_b.weight", (uint64_t)q_lora * H * qh);
  check(p + "attn_kv_a_mqa.weight", (uint64_t)D * kv_dim);
  check(p + "attn_kv_a_norm.weight", (uint64_t)c.kv_lora);
  check(p + "attn_k_b.weight", (uint64_t)c.qk_nope * c.kv_lora * H);
  check(p + "attn_v_b.weight", (uint64_t)c.kv_lora * c.v_head * H);
  check(p + "attn_output.weight", (uint64_t)H * c.v_head * D);

  // ne[0] (fastest / contiguous dim) orientation check for the two MLA
  // up-projections whose absorption matmul in mla_attention_absorbed_gguf
  // assumes a specific layout:
  //   attn_k_b ne[0] == qk_nope   (query absorption: q_absorbed[kv] = sum_nope
  //   q[nope]*W[nope,kv]) attn_v_b ne[0] == kv_lora   (value projection: v[vh]
  //   = sum_kv latent[kv]*W[kv,vh])
  // The numel check() above CANNOT catch a transpose -- [a,b,c] and [b,a,c]
  // have the same product -- so a swapped orientation would silently make the
  // absorption read the transposed matrix and corrupt every MLA layer's
  // attention (finite garbage -> "random text", identical failure mode to the
  // RoPE bug). Confirmed correct vs llama.cpp deepseek2 wk_b/wv_b shapes
  // ({qk_nope,kv_lora,n_head} and {kv_lora,v_head,n_head}); this just makes a
  // future drift loud at load time instead of invisible.
  auto check_orient = [&](const std::string &name, int expected_ne0) {
    auto it = file.tensors.find(name);
    if (it == file.tensors.end() || it->second.shape.empty())
      return;
    int ne0 = (int)it->second.shape[0];
    if (ne0 != expected_ne0) {
      LOG_ERROR(
          "[ORIENT-CHECK] %s ne[0]=%d != expected %d -- the MLA absorption "
          "matmul reads this tensor assuming ne[0]=%d; a different contiguous "
          "dim means the query/value projection is silently TRANSPOSED and "
          "every "
          "MLA layer's attention is garbage.\n",
          name.c_str(), ne0, expected_ne0, expected_ne0);
    }
  };
  check_orient(p + "attn_k_b.weight", c.qk_nope);
  check_orient(p + "attn_v_b.weight", c.kv_lora);

  // Representative MoE layer's router + shared expert (first MoE layer)
  if (c.n_experts > 0 && c.first_dense < c.n_layers) {
    std::string pm = "blk." + std::to_string(c.first_dense) + ".";
    check(pm + "ffn_norm.weight", (uint64_t)D);
    check(pm + "ffn_gate_inp.weight", (uint64_t)c.n_experts * D);
    if (c.n_expert_shared > 0) {
      uint64_t shared_inter = (uint64_t)c.moe_inter * c.n_expert_shared;
      check(pm + "ffn_gate_shexp.weight", (uint64_t)D * shared_inter);
      check(pm + "ffn_up_shexp.weight", (uint64_t)D * shared_inter);
      check(pm + "ffn_down_shexp.weight", shared_inter * (uint64_t)D);
    }
  }

  // Leading dense layer's plain FFN, if this model has any (layer 0)
  if (c.first_dense > 0) {
    check("blk.0.ffn_gate.weight", (uint64_t)D * c.dense_inter);
    check("blk.0.ffn_up.weight", (uint64_t)D * c.dense_inter);
    check("blk.0.ffn_down.weight", (uint64_t)c.dense_inter * D);
  }

  if (mismatches == 0) {
    LOG_STREAM_INFO() << "[DIM-CHECK] All checked tensor dimensions match "
                         "ArchitectureConfig.\n";
  } else {
    LOG_STREAM_ERROR()
        << "[DIM-CHECK] " << mismatches
        << " tensor(s) have a real size mismatch -- see lines above.\n";
  }
}

// Step 4 (report): surface every dense resident tensor's GGUFType and the
// routed-expert type so unsupported quants are VISIBLE. Today's GPU dense
// kernels cover F32/F16/BF16/Q8_0; anything else will hit the loud CPU-fallback
// warning in dense_proj() (attention_layer.cpp) until its kernel lands. This is
// the guard against the silent "reinterpret quant buffer as float" garbage
// class of bug — make it loud at load time, not at first wrong token.
static void report_dense_quant_coverage(const StreamingEngine &engine) {
  std::map<uint32_t, int> dense_counts;
  for (const auto &kv : engine.resident_layers)
    dense_counts[(uint32_t)kv.second.meta.type]++;

  // Kept as its own list rather than reusing gpu_type_supported()
  // (kquant_dequant.hpp): that one models gpu_matmul_dispatch (the EXPERT-FFN
  // fused-kernel path, which excludes F32/F16/BF16 since those route through
  // the SEPARATE cuBLAS path in dense_matmul_host below) — a genuinely
  // different capability set from "can this dense tensor be GPU-dispatched at
  // all". Must be kept in sync with dense_matmul_host's switch
  // (cuda_manager.hpp) by hand.
  auto gpu_fused = [](uint32_t t) -> bool {
    switch ((GGUFType)t) {
    case GGUFType::F32:
    case GGUFType::F16:
    case GGUFType::BF16:
    case GGUFType::Q4_0:
    case GGUFType::Q8_0:
    case GGUFType::Q5_1:
    case GGUFType::Q2_K:
    case GGUFType::Q3_K:
    case GGUFType::Q4_K:
    case GGUFType::Q5_K:
    case GGUFType::Q6_K:
    case GGUFType::IQ2_XXS:
    case GGUFType::IQ2_XS:
    case GGUFType::IQ2_S:
    case GGUFType::IQ3_XXS:
      return true; // GPU dispatch: cuBLAS (F32/F16/BF16) or fused dequant
                   // kernel
    default:
      return false; // IQ1/IQ3_S/IQ4: CPU dequant via libggml; IQ2_M: no ggml
                    // export
    }
  };

  LOG_STREAM_INFO() << "[QUANT] Dense tensor coverage (CPU dequant via "
                       "libggml; GPU fused for "
                       "F32/F16/BF16/Q4_0/Q8_0/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/"
                       "IQ2_XXS/XS/S/IQ3_XXS):\n";
  for (const auto &tc : dense_counts) {
    const char *status = gpu_fused(tc.first) ? "GPU dispatch OK"
                         : ((GGUFType)tc.first == GGUFType::IQ2_M)
                             ? "PENDING (IQ2_M: no ggml dequant export)"
                             : "CPU (libggml) — GPU fused pending";
    LOG_STREAM_INFO() << "  GGUFType " << tc.first << " : " << tc.second
                      << " tensor(s) — " << status << "\n";
  }

  // Routed-expert GPU capability, scanned across EVERY MoE layer (not just
  // blk.0 — Unsloth-style Dynamic quantization can vary quant type by
  // layer, so a single sample can be misleading). This is the direct
  // answer to "do we need to load all this into GPU at all": expert_ffn
  // only dispatches to GPU when gate/up/down are ALL GPU-supported
  // (all-or-nothing per expert), and ram_manager.hpp / execution_layer.cpp
  // / async_prefetcher.hpp all skip VRAM promotion for a layer whose
  // packed tensors aren't GPU-capable — so a layer reported "CPU-bound"
  // here will never be promoted to VRAM at runtime regardless of how hot
  // its experts get.
  {
    auto &c = engine.model_file.architecture_config;
    int gpu_capable_layers = 0, cpu_bound_layers = 0;
    // Layer COUNT per distinct (gate,up,down) combo, not just which
    // combos exist — Unsloth-style Dynamic quantization can mix several
    // combos across one model's layers (observed: 3+ distinct combos in
    // a single GLM-5.2 UD-IQ2_M file), and knowing "73 layers use this
    // one, 2 layers use that one" directly informs which missing GPU
    // kernel is actually worth writing first.
    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, int> combo_counts;
    for (int l = c.first_dense; l < c.n_layers; ++l) {
      std::string p = "blk." + std::to_string(l) + ".";
      auto gE = engine.model_file.tensors.find(p + "ffn_gate_exps.weight");
      auto uE = engine.model_file.tensors.find(p + "ffn_up_exps.weight");
      auto dE = engine.model_file.tensors.find(p + "ffn_down_exps.weight");
      if (gE == engine.model_file.tensors.end() ||
          uE == engine.model_file.tensors.end() ||
          dE == engine.model_file.tensors.end()) {
        // Legacy per-expert format — check expert 0 as a representative sample.
        gE = engine.model_file.tensors.find(p + "ffn_gate.0.weight");
        uE = engine.model_file.tensors.find(p + "ffn_up.0.weight");
        dE = engine.model_file.tensors.find(p + "ffn_down.0.weight");
        if (gE == engine.model_file.tensors.end() ||
            uE == engine.model_file.tensors.end() ||
            dE == engine.model_file.tensors.end())
          continue;
      }
      bool capable = gpu_type_supported(gE->second.type) &&
                     gpu_type_supported(uE->second.type) &&
                     gpu_type_supported(dE->second.type);
      if (capable)
        ++gpu_capable_layers;
      else
        ++cpu_bound_layers;
      ++combo_counts[{(uint32_t)gE->second.type, (uint32_t)uE->second.type,
                      (uint32_t)dE->second.type}];
    }

    LOG_STREAM_INFO()
        << "[QUANT] Routed experts: " << gpu_capable_layers
        << " layer(s) GPU-capable, " << cpu_bound_layers
        << " layer(s) CPU-bound (no kernel for gate/up/down quant type — "
        << "VRAM promotion skipped for these, always compute on CPU)\n";
    // Sorted by layer count descending — the combo affecting the most
    // layers (i.e. the kernel most worth writing) appears first.
    std::vector<std::pair<std::tuple<uint32_t, uint32_t, uint32_t>, int>>
        sorted_combos(combo_counts.begin(), combo_counts.end());
    std::sort(sorted_combos.begin(), sorted_combos.end(),
              [](const auto &a, const auto &b) { return a.second > b.second; });
    for (const auto &[combo, count] : sorted_combos) {
      auto [g, u, d] = combo;
      GGUFType gt = (GGUFType)g, ut = (GGUFType)u, dt = (GGUFType)d;
      bool capable = gpu_type_supported(gt) && gpu_type_supported(ut) &&
                     gpu_type_supported(dt);
      LOG_STREAM_INFO() << "[QUANT]   [" << count
                        << " layer(s)] gate=" << gpu_type_name(gt) << " ("
                        << gpu_type_bits_per_weight(gt)
                        << " bits/weight)  up=" << gpu_type_name(ut) << " ("
                        << gpu_type_bits_per_weight(ut)
                        << " bits/weight)  down=" << gpu_type_name(dt) << " ("
                        << gpu_type_bits_per_weight(dt) << " bits/weight) — "
                        << (capable ? "GPU-capable" : "CPU-bound") << "\n";
      // Explain WHY, per unsupported projection — answers "is this
      // about bit-packing efficiency" directly rather than leaving the
      // reader to guess from a bare type name.
      if (!gpu_type_supported(gt))
        LOG_STREAM_INFO() << "[QUANT]     gate=" << gpu_type_name(gt) << ": "
                          << gpu_unsupported_reason(gt) << "\n";
      if (!gpu_type_supported(ut))
        LOG_STREAM_INFO() << "[QUANT]     up=" << gpu_type_name(ut) << ": "
                          << gpu_unsupported_reason(ut) << "\n";
      if (!gpu_type_supported(dt))
        LOG_STREAM_INFO() << "[QUANT]     down=" << gpu_type_name(dt) << ": "
                          << gpu_unsupported_reason(dt) << "\n";
    }
    if (cpu_bound_layers > 0 && gpu_capable_layers == 0) {
      LOG_STREAM_WARN() << "[QUANT] WARNING: NO routed-expert layer can use "
                           "GPU acceleration with this "
                           "quantization — the expert VRAM cache "
                           "(moe_cache_gb) provides zero compute benefit "
                           "for this model/quant combination today; see the "
                           "per-type reasons above for what's "
                           "missing, or lower moe_cache_gb since it's not "
                           "currently buying any speedup.\n";
    }
  }
}

// ----------------------------------------------------
// Fixed Linkage: Function defined externally to export the symbol
// ----------------------------------------------------
bool GGUFGraphInitializer::build_model_graph(StreamingEngine &engine,
                                             int max_ctx) {
  auto &file = engine.model_file;
  auto &c = file.architecture_config;

  LOG_STREAM_INFO() << "[GRAPH] Analyzing GGUF tensor topology ("
                    << file.tensors.size() << " tensors)...\n";

  // Dump first few tensor names for debugging
  {
    int shown = 0;
    for (const auto &[name, tensor] : file.tensors) {
      if (shown < 20) {
        LogStream ls(LogLevel::DEBUG);
        ls << "  [tensor] " << name << "  shape=[";
        for (size_t d = 0; d < tensor.shape.size(); ++d) {
          if (d)
            ls << ",";
          ls << tensor.shape[d];
        }
        ls << "]  type=" << (int)tensor.type << "  shard=" << tensor.shard_idx
           << "\n";
        ++shown;
      }
    }
    if (file.tensors.size() > 20)
      LOG_STREAM_DEBUG() << "  ... (" << file.tensors.size() - 20
                         << " more tensors)\n";
  }

  // Dump KV metadata for debugging
  LOG_STREAM_DEBUG() << "[GRAPH] KV metadata ("
                     << file.kv_uint.size() + file.kv_int.size() +
                            file.kv_float.size() + file.kv_string.size()
                     << " scalar entries):\n";
  for (const auto &[k, v] : file.kv_uint)
    LOG_STREAM_DEBUG() << "  " << k << " = " << v << " (uint)\n";
  for (const auto &[k, v] : file.kv_int)
    LOG_STREAM_DEBUG() << "  " << k << " = " << v << " (int)\n";
  for (const auto &[k, v] : file.kv_float)
    LOG_STREAM_DEBUG() << "  " << k << " = " << v << " (float)\n";
  for (const auto &[k, v] : file.kv_string)
    LOG_STREAM_DEBUG() << "  " << k << " = \"" << v.substr(0, 64)
                       << "\" (string)\n";

  // ---------- Extract architecture from KV metadata ----------
  // Standard GGUF keys (llama.cpp convention): {arch}.{param}
  c.hidden = file.get_config_int("embedding_length", 0);
  int n_layer_all = file.get_config_int("block_count", 0);
  int n_layer_nextn = file.get_config_int("nextn_predict_layers", 0);
  // Qwen4Exp's own "shared" MTP convention (verified against the real
  // mtp-*-shared-*.gguf file and against the correct upstream reference,
  // /home/ray/Projects/unsloth/llama.cpp branch mtp/qwen4exp-nextn,
  // qwen4exp.cpp's load_arch_hparams/graph_mtp): the extra NextN block is
  // NOT described by the base model file at all -- its own block_count KV
  // (confirmed 48, real tensor dump) has no nextn_predict_layers entry,
  // unlike GLM5NEXT/GLM_DSA where the trunk file's own block_count already
  // counts the inline NextN block. Instead, nextn_predict_layers=1 (and a
  // matching block_count=49, i.e. n_layer_all+1) is declared in the
  // SEPARATE mtp_model_path file's own KV metadata. Read it from there
  // when that file is configured; n_layer_all itself is unaffected (it
  // still describes only the base file's real trunk layers).
  // Qwen4Exp's shared-file NextN block sits ONE PAST the base file's own
  // real trunk layers (blk.<n_layer_all>.*, in the SEPARATE mtp file) --
  // unlike GLM's inline convention, it is never carved OUT of n_layer_all,
  // so the ordinary "subtract" below must not apply to it.
  const bool qwen4exp_shared_mtp =
      engine.has_mtp_model && detect_model_family(file) == ModelFamily::Qwen4Exp;
  if (qwen4exp_shared_mtp) {
    n_layer_nextn = engine.mtp_model_file.get_config_int("nextn_predict_layers", 0);
  }
  if (n_layer_nextn < 0 || (!qwen4exp_shared_mtp && n_layer_nextn >= n_layer_all)) {
    LOG_STREAM_ERROR() << "[GRAPH] Invalid nextn_predict_layers="
                       << n_layer_nextn << " for block_count=" << n_layer_all
                       << " -- expected 0 <= nextn_predict_layers < block_count.\n";
    return false;
  }
  c.n_layer_nextn = n_layer_nextn;
  c.n_layers = qwen4exp_shared_mtp ? n_layer_all : (n_layer_all - n_layer_nextn);
  if (n_layer_nextn > 0) {
    LOG_STREAM_INFO()
        << "[GRAPH] block_count=" << n_layer_all
        << " nextn_predict_layers=" << n_layer_nextn
        << " -- using n_layers=" << c.n_layers
        << (qwen4exp_shared_mtp
                ? " for the target forward pass (shared MTP file's own "
                  "extra block sits at layer index n_layers, not carved "
                  "out of the base file's own trunk count); "
                : " for the target forward pass (excludes NextN/MTP draft "
                  "block(s); ")
        << "MTP is enabled only after model-gated validation below).\n";
  }
  c.n_heads = file.get_config_int("attention.head_count", 0);
  c.vocab = file.get_config_int("vocab_size", 0);
  if (c.vocab <= 0) {
    // FIX (found investigating a real DeepSeek-V4-Flash fluent-but-garbage
    // generation run): this used to read shape[0], not shape[1] -- WRONG.
    // Confirmed directly against llama.cpp's real tensor-creation calls
    // (checked 5 different architectures' model-loading code -- apertus.cpp,
    // baichuan.cpp, afmoe.cpp, and others -- all identical):
    // `create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0)`.
    // GGUF's on-disk tensor-shape encoding is ggml's native ne[] order
    // (dimensions[0] = fastest-varying), and this file's OWN raw parser
    // above reads it byte-for-byte in that same order (t.shape[0] ==
    // ne[0]) -- also independently confirmed by this exact file's
    // check_orient() helper a few lines up, whose own comment documents
    // shape[0]=ne[0] as the contiguous dimension. So for token_embd.weight,
    // shape[0] is ALWAYS n_embd (hidden size), never n_vocab.
    // Real-world impact of the old bug: for any GGUF missing an explicit
    // vocab_size-style KV (this DeepSeek-V4-Flash file has none under any
    // prefix get_config_int tries -- common; most real GGUFs rely on
    // tokenizer.ggml.tokens's array length instead of a redundant scalar),
    // c.vocab silently became EQUAL TO c.hidden (confirmed on a real run:
    // "Vocab: 4096" logged identically to "Hidden: 4096", vs. the
    // tokenizer's own independently-loaded "Loaded 129280 tokens."). This
    // corrupted the prefill embed loop's bounds check
    // (generation_loop.hpp's `if (tok >= 0 && tok < V)`) into silently
    // skipping embed_token for any real content token id above ~4096,
    // leaving that row at hidden_states' zero-initialized default --
    // finite, not NaN, invisible to check_finite/check_hidden_finite --
    // and truncated `logits`/lm_head output to the same wrong width,
    // producing fluent-but-garbage generation. The old comment here
    // ("some models... store vocab size as the leading dimension") was
    // simply wrong; kept the "no numeric KV" fallback need but fixed the
    // index. (Worth noting for anyone revisiting Qwen3.6 support: this
    // exact same bug may have been silently affecting it too, unverified
    // either way here.)
    auto it = file.tensors.find("token_embd.weight");
    if (it == file.tensors.end())
      it = file.tensors.find("output.weight");
    if (it != file.tensors.end() && it->second.shape.size() >= 2) {
      c.vocab = static_cast<int>(it->second.shape[1]);
    }
  }
  c.n_experts = file.get_config_int("expert_count", 0);
  c.topk = file.get_config_int("expert_used_count", 8);
  c.kv_lora = file.get_config_int("attention.kv_lora_rank", 512);
  c.qk_rope =
      file.get_config_int("rope.dimension_count",
                          file.get_config_int("attention.key_length_rope", 64));
  // ROOT CAUSE (found via a real NaN-corruption investigation this
  // session): attention.key_length_mla is the COMBINED qk head dim
  // (nope+rope), NOT the nope-only dimension the previous comment here
  // claimed -- confirmed against llama.cpp's actual loader
  // (src/models/deepseek2.cpp): `n_embd_head_qk_nope = n_embd_head_k_mla -
  // n_embd_head_qk_rope`. Reading key_length_mla straight into qk_nope
  // overcounted nope by qk_rope elements (this model: 256 instead of the
  // real 192, a 25% overrun) -- every attn_k_b/attn_v_b dequant then read
  // that many elements from host_ptr, running past attn_k_b's real data
  // into whatever tensor follows it in memory. That's what made
  // hidden_states 100% NaN from layer 0 of every single generation run.
  // (attention.value_length_mla, by contrast, IS used directly with no
  // subtraction in the reference loader -- confirmed correct, left as-is.)
  // FIX (found investigating a real DeepSeek-V4-Flash NaN-corruption run):
  // DeepSeek-V4 GGUFs don't set "attention.key_length_mla" at all -- the
  // SAME "combined nope+rope width" concept V3 calls key_length_mla is
  // just called "attention.key_length" for V4 (confirmed against a real
  // checkpoint's metadata dump: deepseek4.attention.key_length=512, no
  // key_length_mla key anywhere). Without this, key_length_mla read as 0,
  // so qk_nope silently fell all the way to the hardcoded 128 default --
  // c.qk_rope correctly read 64 (rope.dimension_count), so qh = qk_nope +
  // qk_rope came out as 192 instead of the real 512 (confirmed three
  // independent ways: key_length itself, attn_kv.weight's real shape
  // [4096,512], and attn_q_b.weight's 32768/64=512) -- every width
  // dsv4_attention_gguf computes from qh (Q, KV latent, output context,
  // the grouped-output-projection's group_in) was wrong by this margin.
  // Try key_length_mla first (V3), THEN the plain combined key_length
  // (V4) minus rope, THEN an explicit key_length_nope key, THEN the
  // hardcoded 128 as an absolute last resort. Safe for dense families:
  // they never read qk_nope at all, and c.head_dim's own SEPARATE
  // "attention.key_length" read (below, dense-only) is unaffected since
  // it's a different ArchitectureConfig field entirely.
  int key_length_mla = file.get_config_int("attention.key_length_mla", 0);
  if (key_length_mla > 0) {
    c.qk_nope = key_length_mla - c.qk_rope;
  } else {
    int key_length_combined = file.get_config_int("attention.key_length", 0);
    c.qk_nope = (key_length_combined > c.qk_rope)
                    ? (key_length_combined - c.qk_rope)
                    : file.get_config_int("attention.key_length_nope", 128);
  }
  c.v_head =
      file.get_config_int("attention.value_length_mla",
                          file.get_config_int("attention.value_length", 128));
  c.eps = file.get_config_float("attention.layer_norm_rms_epsilon", 1e-5f);
  // GLM-5.2 uses "leading_dense_block_count" (not "first_k_dense_replace")
  int first_dense =
      file.get_config_int("leading_dense_block_count",
                          file.get_config_int("first_k_dense_replace", 0));
  c.moe_inter = file.get_config_int("expert_feed_forward_length", 0);
  c.dense_inter = file.get_config_int("feed_forward_length", 0);
  c.first_dense = first_dense;

  // Dense Q/K/V attention params (Hunyuan/Llama); harmless defaults for MLA.
  c.n_head_kv = file.get_config_int("attention.head_count_kv", c.n_heads);
  c.head_dim = file.get_config_int("attention.key_length",
                                   c.hidden / std::max(1, c.n_heads));
  std::string arch0 = file.kv_string.count("general.architecture")
                          ? file.kv_string["general.architecture"]
                          : "";

  // ---------- Resolve the per-family dispatch table ----------
  // Single source of truth for "which family is this file" is
  // detect_model_family() (model_family.hpp) -- NOT another arch0=="..."
  // comparison here. engine.family_ops is what layers_forward_gguf
  // (execution_layer.cpp) and the cache-allocation/preload-manifest code
  // below now read instead of the old scattered is_mla-bool checks. See
  // family_ops.hpp for the full rationale.
  ModelFamily fam = detect_model_family(file);

  // ---------- DeepSeek-V4 (Hyper-Connections, grouped output projection,
  // attention sinks, raw SWA plus CSA/HCA compressed attention) ----------
  // Every field defaults to "feature inactive" (ArchitectureConfig's own
  // defaults), so this block is a strict no-op for every other family.
  //
  // MOVED HERE 2026-08-09 (was originally right after the YaRN block,
  // several dozen lines below `resolve_family_ops`): c.hc_mult MUST be
  // populated from the real GGUF value BEFORE resolve_family_ops(fam, c)
  // runs, since that call reads c.hc_mult to decide whether to wire
  // hc_pre_fn/hc_post_fn to the real hc_pre_collapse/hc_post_combine or
  // leave them at the shared hc_pre_noop/hc_post_plain_add default
  // (family_ops.hpp). With the old ordering, resolve_family_ops always saw
  // ArchitectureConfig's default-constructed c.hc_mult (0, not >1) --
  // meaning DeepSeek-V4's per-layer Hyper-Connections hooks NEVER actually
  // ran, for every load, silently. hc_expand_embedding/hc_collapse_output
  // (generation_loop.hpp) re-read c.hc_mult fresh at generation time (long
  // after this function returns) so THEY correctly saw hc_mult=4 and ran --
  // meaning hc_stream_buf got expanded from the embedding at the start of
  // every forward pass and then collapsed back at the end, but was NEVER
  // updated by any layer in between (hc_post_plain_add just does
  // hidden_states += sub_block_out directly, never touching hc_stream_buf).
  // The result: hc_collapse_output's final hidden_states was a gated blend
  // of the STALE, untouched-since-embedding hc_stream_buf, discarding the
  // entire 78-layer transformer computation that had (correctly, via plain
  // residual add) been accumulating in hidden_states the whole time. This
  // silently replaces the real forward-pass output with something close to
  // a gated function of the raw token embedding -- a very plausible
  // explanation for coherent-looking-per-token-but-context-blind "word
  // salad" output, and identical on CPU and GPU dispatch (this is a
  // config-resolution/control-flow bug, not a kernel bug). Found by adding
  // real [HC-TRACE] telemetry to hyper_connections.cpp and observing ZERO
  // per-layer hc_pre_collapse/hc_post_combine trace lines in a real VERBOSE
  // capture despite the hooks being unconditionally called every layer in
  // execution_layer.cpp -- traced back to this ordering bug by inspection,
  // not yet re-verified on hardware after this fix.
  {
    c.hc_mult = file.get_config_int("hyper_connection.count", 1);
    c.hc_sinkhorn_iters =
        file.get_config_int("hyper_connection.sinkhorn_iterations", 0);
    c.hc_eps = file.get_config_float("hyper_connection.epsilon", 1e-6f);
    c.o_group_count = file.get_config_int("attention.output_group_count", 0);
    c.o_lora_rank = file.get_config_int("attention.output_lora_rank", 0);
    c.hash_layer_count = file.get_config_int("hash_layer_count", 0);
    c.sliding_window = file.get_config_int("attention.sliding_window", 0);
    c.dsv4_compress_ratios =
        file.get_config_int_array("attention.compress_ratios");
    c.dsv4_compress_rope_freq_base =
        file.get_config_float("attention.compress_rope_freq_base", 0.0f);
    c.dsv4_indexer_head_count =
        file.get_config_int("attention.indexer.head_count", 0);
    c.dsv4_indexer_key_length =
        file.get_config_int("attention.indexer.key_length", 0);
    c.dsv4_indexer_top_k =
        file.get_config_int("attention.indexer.top_k", 0);
    if (fam == ModelFamily::DeepSeek_V41) {
      c.dsv41_kv_source_layers =
          file.get_config_int_array("attention.kv_source_layers");
      c.dsv41_index_source_layers =
          file.get_config_int_array("attention.index_source_layers");
      c.dsv41_candidate_source_layer =
          file.get_config_int("attention.candidate_source_layer", -1);
      c.dsv41_candidate_top_k_blocks =
          file.get_config_int("attention.candidate_topk_blocks", 0);
      c.dsv41_candidate_block_size =
          file.get_config_int("attention.candidate_block_size", 0);
      c.dsv41_engram_layer_ids = file.get_config_int_array("engram.layer_ids");
      c.dsv41_engram_num_embeddings =
          file.get_config_int_array("engram.num_embeddings");
      c.dsv41_engram_primes = file.get_config_int_array("engram.primes");
      c.dsv41_engram_offsets = file.get_config_int_array("engram.offsets");
      c.dsv41_engram_multipliers =
          file.get_config_int_array("engram.multipliers");
      c.dsv41_engram_token_map =
          file.get_config_int_array("engram.token_map");
      c.dsv41_engram_max_ngram_size =
          file.get_config_int("engram.max_ngram_size", 0);
      c.dsv41_engram_vocab_size =
          file.get_config_int("engram.vocab_size", 0);
      c.dsv41_engram_n_heads = file.get_config_int("engram.head_count", 0);
      c.dsv41_engram_head_dim = file.get_config_int("engram.key_length", 0);
      c.dsv41_engram_pad_id = file.get_config_int("engram.pad_id", 2);
      c.dsv41_engram_compressed_vocab_size =
          file.get_config_int("engram.compressed_vocab_size", 0);

      // The converter writes source lists, but infer them from the tensor
      // presence as a compatibility fallback for early converted files.
      if (c.dsv41_kv_source_layers.empty() ||
          c.dsv41_index_source_layers.empty()) {
        c.dsv41_kv_source_layers.clear();
        c.dsv41_index_source_layers.clear();
        for (int l = 0; l < c.n_layers; ++l) {
          const std::string p = "blk." + std::to_string(l) + ".";
          if (file.tensors.count(p + "attn_compressor_kv.weight"))
            c.dsv41_kv_source_layers.push_back(l);
          // attn_k/k_norm belong to the KV-key owner; the query-side
          // indexer owner is identified by its proj/q_b pair.
          if (file.tensors.count(p + "indexer.proj.weight") ||
              file.tensors.count(p + "indexer.attn_q_b.weight"))
            c.dsv41_index_source_layers.push_back(l);
        }
      }
    }
    // Per-layer arrays (read_kv_array/get_config_float_array, gguf_loader.
    // hpp) -- empty vector (not a size-n_layers vector of the scalar
    // default) when the file doesn't set them; FFN call sites (Project B's
    // SwiGLU-clamp step) check .empty() rather than assuming a specific
    // size, so a file that sets these for fewer than n_layers layers fails
    // safe (out-of-range layers just don't get clamped) rather than
    // out-of-bounds reading.
    c.swiglu_clamp_exp = file.get_config_float_array("swiglu_clamp_exp");
    c.swiglu_clamp_shexp = file.get_config_float_array("swiglu_clamp_shexp");
    if (fam == ModelFamily::DeepSeek_V4) {
      LOG_STREAM_INFO() << "[DSV4] hc_mult=" << c.hc_mult
                        << " hc_sinkhorn_iters=" << c.hc_sinkhorn_iters
                        << " hc_eps=" << c.hc_eps
                        << " o_group_count=" << c.o_group_count
                        << " o_lora_rank=" << c.o_lora_rank
                        << " hash_layer_count=" << c.hash_layer_count
                        << " sliding_window=" << c.sliding_window
                        << " compress_ratios.size()="
                        << c.dsv4_compress_ratios.size()
                        << " compress_rope_freq_base="
                        << c.dsv4_compress_rope_freq_base
                        << " indexer_head_count=" << c.dsv4_indexer_head_count
                        << " indexer_key_length=" << c.dsv4_indexer_key_length
                        << " indexer_top_k=" << c.dsv4_indexer_top_k
                        << " swiglu_clamp_exp.size()=" << c.swiglu_clamp_exp.size()
                        << " swiglu_clamp_shexp.size()=" << c.swiglu_clamp_shexp.size()
                        << "\n";
    }
    if (fam == ModelFamily::DeepSeek_V41) {
      LOG_STREAM_INFO() << "[DSV41] hc_mult=" << c.hc_mult
                        << " sliding_window=" << c.sliding_window
                        << " compress_ratios.size()="
                        << c.dsv4_compress_ratios.size()
                        << " kv_sources=" << c.dsv41_kv_source_layers.size()
                        << " index_sources=" << c.dsv41_index_source_layers.size()
                        << " engram_layers=" << c.dsv41_engram_layer_ids.size()
                        << " engram_max_ngram=" << c.dsv41_engram_max_ngram_size
                        << " engram_heads=" << c.dsv41_engram_n_heads << "\n";
    }
  }

  // ---------- GLM5Next (GLM 5.3 Flash): hybrid KDA + DSA hparams ----------
  // Placed BEFORE resolve_family_ops(fam, c) below, same ordering discipline
  // as the hc_mult block above (a prior session found a real "read a
  // hyperparameter before the code that populates it has run" bug here --
  // see the comment on that block). Reference for every key/constraint:
  // llama.cpp branch glm5next (src/models/glm5next.cpp's load_arch_hparams,
  // conversion/glm5next.py's set_gguf_parameters).
  if (fam == ModelFamily::GLM5NEXT) {
    // Per-layer hybrid map: 0 = recurrent KDA, nonzero = full-attention DSA.
    // The converter writes one entry per block_count layer (INCLUDING any
    // NextN block), so validate against n_layer_all, not c.n_layers.
    c.glm5next_head_count_kv = file.get_config_int_array("attention.head_count_kv");
    if (c.glm5next_head_count_kv.size() < static_cast<size_t>(n_layer_all)) {
      LOG_STREAM_ERROR() << "[GLM5NEXT] Invalid attention.head_count_kv: expected at "
                         << "least " << n_layer_all << " entries for block_count, got "
                         << c.glm5next_head_count_kv.size()
                         << " -- glm5next needs the per-layer array (0=KDA, 1=DSA).\n";
      return false;
    }
    int n_kda = 0;
    for (int l = 0; l < c.n_layers; ++l)
      if (glm5next_is_kda_layer(c, l)) ++n_kda;
    if (n_kda == 0 || n_kda == c.n_layers) {
      LOG_STREAM_ERROR() << "[GLM5NEXT] head_count_kv marks " << n_kda << " of "
                         << c.n_layers << " layers as KDA -- glm5next is a hybrid "
                            "and needs BOTH layer types.\n";
      return false;
    }

    c.ssm_conv_kernel = file.get_config_int("ssm.conv_kernel", 0);
    c.kda_head_dim = file.get_config_int("kda.head_dim", 0);
    c.kda_gate_lower_bound = file.get_config_float("kda.gate_lower_bound", 0.0f);
    if (c.ssm_conv_kernel <= 1 || c.kda_head_dim <= 0) {
      LOG_STREAM_ERROR() << "[GLM5NEXT] ssm.conv_kernel=" << c.ssm_conv_kernel
                         << " kda.head_dim=" << c.kda_head_dim
                         << " -- both must be positive (conv_kernel > 1).\n";
      return false;
    }
    // Required, and must be negative: absent, the reference selects the
    // softplus gate branch -- a DIFFERENT function (kimi-k3 convention).
    if (!(c.kda_gate_lower_bound < 0.0f)) {
      LOG_STREAM_ERROR() << "[GLM5NEXT] kda.gate_lower_bound="
                         << c.kda_gate_lower_bound
                         << " -- required and must be < 0 (it scales the sigmoid "
                            "decay gate; 0/absent would silently select the wrong "
                            "gate function).\n";
      return false;
    }

    // nope-only MLA: the converter hard-requires qk_rope_head_dim == 0, so
    // c.qk_rope (read from rope.dimension_count above) must already be 0.
    // mla_attention_absorbed_gguf handles rope==0 correctly (RoPE loops are
    // no-ops, the rope score term vanishes, the GPU score path declines and
    // falls back to CPU) -- but only if the value really is 0; a nonzero
    // read here would mean a metadata surprise worth failing loudly on.
    if (c.qk_rope != 0) {
      LOG_STREAM_ERROR() << "[GLM5NEXT] rope.dimension_count=" << c.qk_rope
                         << " -- glm5next's text tower is nope-only (n_rot == 0); "
                            "a nonzero rope width is a converter/metadata bug.\n";
      return false;
    }

    // Indexer metadata: loaded + logged, NOT executed in v1 (DSA layers run
    // as dense absorbed MLA over the full KV cache -- a correct superset of
    // the indexer's selected positions, not bit-matching the trained sparse
    // behavior). The reference converter REJECTS non-indexer variants, so
    // these keys are always present on a real checkpoint.
    c.glm5next_indexer_head_count = file.get_config_int("attention.indexer.head_count", 0);
    c.glm5next_indexer_key_length = file.get_config_int("attention.indexer.key_length", 0);
    c.glm5next_indexer_top_k = file.get_config_int("attention.indexer.top_k", 0);
    c.glm5next_indexer_kpool = file.get_config_int("attention.indexer.kpool", 0);

    // GLM5Next's SwiGLU clamp uses ggml_swiglu_clamp's fused semantics
    // (min(gate,limit), clamp(up,+/-limit), silu(gate)*up), NOT this
    // project's default silu→mul→clamp-product order -- see
    // dense_dispatch.hpp's swiglu_clamp_fused_flag comment. Set once here
    // (model load), read on the expert-FFN hot path.
    set_swiglu_clamp_fused(true);

    LOG_STREAM_INFO() << "[GLM5NEXT] hybrid: " << n_kda << " KDA (recurrent) layer(s), "
                      << (c.n_layers - n_kda) << " DSA (absorbed-MLA) layer(s) | "
                      << "kda_head_dim=" << c.kda_head_dim
                      << " ssm_conv_kernel=" << c.ssm_conv_kernel
                      << " kda_gate_lower_bound=" << c.kda_gate_lower_bound
                      << " | indexer (DEFERRED, dense attention in v1): heads="
                      << c.glm5next_indexer_head_count
                      << " key_length=" << c.glm5next_indexer_key_length
                      << " top_k=" << c.glm5next_indexer_top_k
                      << " kpool=" << c.glm5next_indexer_kpool << "\n";
  }

  // ---------- Qwen4Exp (Qwen3.8-Flash-Next): hybrid GDN + full-attention
  // ---------- hparams ----------
  // Placed BEFORE resolve_family_ops(fam, c) below, same ordering discipline
  // as the hc_mult/GLM5NEXT blocks above (a prior session found a real "read
  // a hyperparameter before the code that populates it has run" bug here --
  // resolve_family_ops reads c.hc_mult/c.hc_low_rank to wire the HC hooks).
  // Reference for every key/constraint: llama.cpp branch
  // qwen4exp/qwen3.8-flash-next (src/models/qwen4exp.cpp's
  // load_arch_hparams, conversion/qwen4exp.py's set_gguf_parameters).
  if (fam == ModelFamily::Qwen4Exp) {
    // Per-layer hybrid map. The converter does NOT write
    // attention.recurrent_layers today -- qwen4exp.cpp itself falls back to
    // the full_attention_interval rule (default 4) in that case, and so do
    // we (qwen4exp_is_gdn_layer, gguf_loader.hpp).
    c.qwen4exp_recurrent_layers =
        file.get_config_int_array("attention.recurrent_layers");
    c.qwen4exp_full_attention_interval =
        file.get_config_int("full_attention_interval", 4);
    if (!c.qwen4exp_recurrent_layers.empty() &&
        c.qwen4exp_recurrent_layers.size() < static_cast<size_t>(n_layer_all)) {
      LOG_STREAM_ERROR() << "[QWEN4EXP] Invalid attention.recurrent_layers: expected at "
                         << "least " << n_layer_all << " entries for block_count, got "
                         << c.qwen4exp_recurrent_layers.size() << ".\n";
      return false;
    }
    int n_gdn = 0;
    for (int l = 0; l < c.n_layers; ++l)
      if (qwen4exp_is_gdn_layer(c, l)) ++n_gdn;
    if (n_gdn == 0 || n_gdn == c.n_layers) {
      LOG_STREAM_ERROR() << "[QWEN4EXP] layer map marks " << n_gdn << " of "
                         << c.n_layers << " layers as recurrent GDN -- qwen4exp is a "
                            "hybrid and needs BOTH layer types.\n";
      return false;
    }

    // GDN dimensions (ssm.* keys, written by the Qwen3Next converter base):
    // state_size = per-head key/value dim; group_count = key/query head
    // count; time_step_rank = VALUE head count (num_v_heads -- despite the
    // name); inner_size = n_v_heads * head_v_dim.
    c.ssm_conv_kernel = file.get_config_int("ssm.conv_kernel", 0);
    c.ssm_inner_size = file.get_config_int("ssm.inner_size", 0);
    c.ssm_state_size = file.get_config_int("ssm.state_size", 0);
    c.ssm_time_step_rank = file.get_config_int("ssm.time_step_rank", 0);
    c.ssm_group_count = file.get_config_int("ssm.group_count", 0);
    if (c.ssm_conv_kernel <= 1 || c.ssm_inner_size <= 0 ||
        c.ssm_state_size <= 0 || c.ssm_time_step_rank <= 0 ||
        c.ssm_group_count <= 0) {
      LOG_STREAM_ERROR() << "[QWEN4EXP] ssm.{conv_kernel=" << c.ssm_conv_kernel
                         << ",inner_size=" << c.ssm_inner_size
                         << ",state_size=" << c.ssm_state_size
                         << ",time_step_rank=" << c.ssm_time_step_rank
                         << ",group_count=" << c.ssm_group_count
                         << "} -- all must be positive (conv_kernel > 1).\n";
      return false;
    }
    if (c.ssm_inner_size != c.ssm_state_size * c.ssm_time_step_rank ||
        c.ssm_time_step_rank % c.ssm_group_count != 0) {
      LOG_STREAM_ERROR() << "[QWEN4EXP] inconsistent ssm dims: inner_size ("
                         << c.ssm_inner_size << ") != state_size ("
                         << c.ssm_state_size << ") * time_step_rank ("
                         << c.ssm_time_step_rank << "), or time_step_rank not "
                         << "divisible by group_count (" << c.ssm_group_count
                         << ").\n";
      return false;
    }

    // Low-rank hyper-connections: qwen4exp REQUIRES the low-rank key
    // (qwen4exp.cpp GGML_ASSERTs on it) -- 0 here means a metadata bug, not
    // a full-rank variant (that's DeepSeek-V4's convention, not this file's).
    c.hc_low_rank = file.get_config_int("hyper_connection.low_rank", 0);
    if (c.hc_mult <= 1 || c.hc_low_rank <= 0) {
      LOG_STREAM_ERROR() << "[QWEN4EXP] hyper_connection.count=" << c.hc_mult
                         << " hyper_connection.low_rank=" << c.hc_low_rank
                         << " -- qwen4exp needs BOTH (>1 and >0); it has no "
                            "plain-residual or full-rank-HC form.\n";
      return false;
    }

    // QSA indexer metadata: loaded + logged, NOT executed in v1 (full-
    // attention layers run dense over the whole KV cache -- a correct
    // superset of the indexer's selected blocks, not bit-matching the
    // trained sparse behavior). Same deferral discipline as GLM5Next's own
    // indexer. Note the GGUF key (attention.compress_ratios) is shared with
    // DeepSeek-V4's CSA/HCA array but means something different here
    // (per-full-attention-layer block pool ratio), hence the separate field.
    c.qwen4exp_compress_ratios =
        file.get_config_int_array("attention.compress_ratios");
    c.qwen4exp_indexer_head_count =
        file.get_config_int("attention.indexer.head_count", 0);
    c.qwen4exp_indexer_key_length =
        file.get_config_int("attention.indexer.key_length", 0);
    c.qwen4exp_indexer_top_k = file.get_config_int("attention.indexer.top_k", 0);

    // Interleaved partial RoPE: rope.dimension_count is the ROTARY width
    // (partial_rotary_factor * head_dim, e.g. 64 of 256), read into the
    // qwen4exp-only rope_n_rot -- NOT c.qk_rope, whose dense-path read of
    // the same key (above) would silently misinform the MLA-only fields.
    c.rope_sections = file.get_config_int_array("rope.dimension_sections");
    c.rope_n_rot = file.get_config_int("rope.dimension_count", 0);
    if (c.rope_n_rot <= 0 || c.rope_n_rot > c.head_dim ||
        (c.rope_n_rot % 2) != 0) {
      LOG_STREAM_ERROR() << "[QWEN4EXP] rope.dimension_count=" << c.rope_n_rot
                         << " head_dim=" << c.head_dim
                         << " -- need 0 < n_rot <= head_dim, n_rot even.\n";
      return false;
    }

    // PLE n-gram hash embedding: absent key group = no PLE layers at all
    // (valid -- the converter emits no PLE keys when ple_layer_ids is empty).
    c.ple_layers = file.get_config_int_array("ple.layers");
    if (!c.ple_layers.empty()) {
      c.ple_ngram_size = file.get_config_int("ple.ngram_size", 0);
      c.ple_heads_per_ngram = file.get_config_int("ple.heads_per_ngram", 0);
      c.ple_conv_kernel = file.get_config_int("ple.conv_kernel", 0);
      c.ple_eos_token_id = file.get_config_int("ple.eos_token_id", -1);
      c.ple_image_token_id = file.get_config_int("ple.image_token_id", 0);
      c.ple_n_embd_per_layer =
          file.get_config_int("embedding_length_per_layer_input", 0);
      c.ple_layer_multipliers = file.get_config_int_array("ple.layer_multipliers");
      c.ple_head_offsets = file.get_config_int_array("ple.head_offsets");
      c.ple_head_vocab_sizes = file.get_config_int_array("ple.head_vocab_sizes");
      c.ple_n_heads =
          (c.ple_ngram_size - 1) * c.ple_heads_per_ngram;
      bool ple_ok = c.ple_ngram_size >= 2 && c.ple_heads_per_ngram > 0 &&
                    c.ple_conv_kernel > 0 && c.ple_eos_token_id >= 0 &&
                    c.ple_n_embd_per_layer > 0 && c.ple_n_heads > 0 &&
                    (int)c.ple_layer_multipliers.size() >= c.ple_ngram_size &&
                    (int)c.ple_head_offsets.size() >= c.ple_n_heads &&
                    (int)c.ple_head_vocab_sizes.size() >= c.ple_n_heads;
      for (int64_t l : c.ple_layers)
        if (l < 0 || l >= c.n_layers) ple_ok = false;
      if (!ple_ok) {
        LOG_STREAM_ERROR() << "[QWEN4EXP] inconsistent PLE metadata: ngram_size="
                           << c.ple_ngram_size << " heads_per_ngram="
                           << c.ple_heads_per_ngram << " conv_kernel="
                           << c.ple_conv_kernel << " eos=" << c.ple_eos_token_id
                           << " embd_per_layer=" << c.ple_n_embd_per_layer
                           << " n_heads=" << c.ple_n_heads
                           << " multipliers=" << c.ple_layer_multipliers.size()
                           << " offsets=" << c.ple_head_offsets.size()
                           << " vocab_sizes=" << c.ple_head_vocab_sizes.size()
                           << " layers=" << c.ple_layers.size() << ".\n";
        return false;
      }
    }

    LOG_STREAM_INFO() << "[QWEN4EXP] hybrid: " << n_gdn << " GDN (recurrent) layer(s), "
                      << (c.n_layers - n_gdn) << " full-attention layer(s) | "
                      << "ssm: state=" << c.ssm_state_size
                      << " k_heads=" << c.ssm_group_count
                      << " v_heads=" << c.ssm_time_step_rank
                      << " inner=" << c.ssm_inner_size
                      << " conv_kernel=" << c.ssm_conv_kernel
                      << " | HC: mult=" << c.hc_mult
                      << " low_rank=" << c.hc_low_rank
                      << " | rope: n_rot=" << c.rope_n_rot
                      << " sections=" << c.rope_sections.size()
                      << " | indexer (DEFERRED, dense attention in v1): heads="
                      << c.qwen4exp_indexer_head_count
                      << " key_length=" << c.qwen4exp_indexer_key_length
                      << " top_k=" << c.qwen4exp_indexer_top_k
                      << " | PLE: layers=" << c.ple_layers.size()
                      << " n_heads=" << c.ple_n_heads << "\n";
  }

  // Flash uses only ratio=0 raw sliding-window attention. Pro checkpoints
  // additionally select CSA (ratio=4) or HCA (ratio=128). Validate the
  // metadata and every compressed layer's load contract before allocating
  // cache state; attention_layer.cpp implements these paths below.
  if (fam == ModelFamily::DeepSeek_V4) {
    if (!c.dsv4_compress_ratios.empty() &&
        c.dsv4_compress_ratios.size() < static_cast<size_t>(n_layer_all)) {
      LOG_STREAM_ERROR() << "[DSV4] Invalid attention.compress_ratios: expected at "
                         << "least " << n_layer_all << " entries for block_count, got "
                         << c.dsv4_compress_ratios.size() << ".\n";
      return false;
    }
    bool has_csa = false;
    bool has_compressed = false;
    for (int l = 0; l < n_layer_all; ++l) {
      const int64_t ratio = c.dsv4_compress_ratios.empty()
          ? 0 : c.dsv4_compress_ratios[(size_t)l];
      if (ratio != 0 && ratio != kDsv4CsaRatio && ratio != kDsv4HcaRatio) {
        LOG_STREAM_ERROR() << "[DSV4] Invalid attention.compress_ratios[" << l
                           << "]=" << ratio << "; supported values are 0, "
                              "4 (CSA), and 128 (HCA).\n";
        return false;
      }
      has_compressed |= ratio != 0;
      has_csa |= ratio == kDsv4CsaRatio;
    }
    if (has_compressed && c.dsv4_compress_rope_freq_base <= 0.0f) {
      LOG_STREAM_ERROR() << "[DSV4] Compressed attention requires a positive "
                            "attention.compress_rope_freq_base.\n";
      return false;
    }
    if (has_csa && (c.dsv4_indexer_head_count <= 0 ||
                    c.dsv4_indexer_key_length < c.qk_rope ||
                    c.dsv4_indexer_top_k <= 0)) {
      LOG_STREAM_ERROR() << "[DSV4] CSA requires positive "
                            "attention.indexer_head_count/indexer_top_k and "
                            "attention.indexer_key_length >= rope.dimension_count.\n";
      return false;
    }

    const int qh = c.qk_nope + c.qk_rope;
    const int q_lora = file.get_config_int("attention.q_lora_rank", 0);
    auto numel = [&file](const std::string &name) -> uint64_t {
      auto it = file.tensors.find(name);
      if (it == file.tensors.end()) return 0;
      uint64_t n = 1;
      for (uint64_t dim : it->second.shape) n *= dim;
      return n;
    };
    auto require = [&numel](const std::string &name, uint64_t expected) {
      return numel(name) == expected;
    };
    for (int l = 0; l < c.n_layers; ++l) {
      const int64_t ratio = c.dsv4_compress_ratios.empty()
          ? 0 : c.dsv4_compress_ratios[(size_t)l];
      if (ratio == 0) continue;
      const std::string p = "blk." + std::to_string(l) + ".";
      const int coff = ratio == kDsv4CsaRatio ? 2 : 1;
      const bool compressor_ok =
          require(p + "attn_compressor_kv.weight", (uint64_t)c.hidden * coff * qh) &&
          require(p + "attn_compressor_gate.weight", (uint64_t)c.hidden * coff * qh) &&
          require(p + "attn_compressor_ape.weight", (uint64_t)coff * qh * ratio) &&
          require(p + "attn_compressor_norm.weight", qh);
      const bool indexer_ok = ratio != kDsv4CsaRatio ||
          (q_lora > 0 &&
           require(p + "indexer.proj.weight",
                   (uint64_t)c.hidden * c.dsv4_indexer_head_count) &&
           require(p + "indexer.attn_q_b.weight",
                   (uint64_t)q_lora * c.dsv4_indexer_head_count * c.dsv4_indexer_key_length) &&
           require(p + "indexer_compressor_kv.weight",
                   (uint64_t)c.hidden * 2 * c.dsv4_indexer_key_length) &&
           require(p + "indexer_compressor_gate.weight",
                   (uint64_t)c.hidden * 2 * c.dsv4_indexer_key_length) &&
           require(p + "indexer_compressor_ape.weight",
                   (uint64_t)2 * c.dsv4_indexer_key_length * kDsv4CsaRatio) &&
           require(p + "indexer_compressor_norm.weight",
                   c.dsv4_indexer_key_length));
      if (!compressor_ok || !indexer_ok) {
        LOG_STREAM_ERROR() << "[DSV4] Layer " << l
                           << " lacks a required CSA/HCA compressor or has "
                              "an incompatible tensor size; refusing to run "
                              "compressed attention with partial state.\n";
        return false;
      }
    }
  }

  if (fam == ModelFamily::DeepSeek_V41) {
    if (c.dsv4_compress_ratios.size() < static_cast<size_t>(c.n_layers)) {
      LOG_STREAM_ERROR() << "[DSV41] Invalid attention.compress_ratios: expected "
                         << c.n_layers << " trunk entries, got "
                         << c.dsv4_compress_ratios.size() << ".\n";
      return false;
    }
    bool has_compressed = false;
    for (int l = 0; l < c.n_layers; ++l) {
      const int64_t ratio = c.dsv4_compress_ratios[(size_t)l];
      if (ratio != 0 && ratio != 1 && ratio != 2) {
        LOG_STREAM_ERROR() << "[DSV41] Invalid attention.compress_ratios["
                           << l << "]=" << ratio
                           << "; supported values are 0, 1, and 2.\n";
        return false;
      }
      has_compressed |= ratio != 0;
    }
    if (has_compressed && c.dsv4_compress_rope_freq_base <= 0.0f) {
      LOG_STREAM_ERROR() << "[DSV41] Compressed attention requires a positive "
                            "attention.compress_rope_freq_base.\n";
      return false;
    }
    if (has_compressed &&
        (c.dsv4_indexer_head_count <= 0 || c.dsv4_indexer_key_length <= 0 ||
         c.dsv4_indexer_top_k <= 0)) {
      LOG_STREAM_ERROR() << "[DSV41] Compressed attention requires positive "
                            "indexer head_count/key_length/top_k metadata.\n";
      return false;
    }

    auto numel = [&file](const std::string &name) -> uint64_t {
      auto it = file.tensors.find(name);
      if (it == file.tensors.end()) return 0;
      uint64_t n = 1;
      for (uint64_t dim : it->second.shape) n *= dim;
      return n;
    };
    auto source_for = [](const std::vector<int64_t> &sources, int l) {
      return std::find(sources.begin(), sources.end(), (int64_t)l) !=
             sources.end();
    };
    const int qh = c.qk_nope + c.qk_rope;
    const int q_lora = file.get_config_int("attention.q_lora_rank", 0);
    for (int l = 0; l < c.n_layers; ++l) {
      const int ratio = (int)c.dsv4_compress_ratios[(size_t)l];
      if (!ratio) continue;
      const std::string p = "blk." + std::to_string(l) + ".";
      const bool kv_source = source_for(c.dsv41_kv_source_layers, l);
      const bool index_source = source_for(c.dsv41_index_source_layers, l);
      const auto has_compatible_source = [l, &c](
          const std::vector<int64_t> &sources) {
        return std::any_of(sources.begin(), sources.end(), [l, &c](int64_t s) {
          return s <= l && s >= 0 && s < c.n_layers &&
                 c.dsv4_compress_ratios[(size_t)s] ==
                     c.dsv4_compress_ratios[(size_t)l];
        });
      };
      if (!has_compatible_source(c.dsv41_kv_source_layers)) {
        LOG_STREAM_ERROR() << "[DSV41] Layer " << l
                           << " has no compatible compressed-KV source.\n";
        return false;
      }
      if (!has_compatible_source(c.dsv41_index_source_layers)) {
        LOG_STREAM_ERROR() << "[DSV41] Layer " << l
                           << " has no compatible indexer source.\n";
        return false;
      }
      if (kv_source) {
        if (numel(p + "attn_compressor_kv.weight") !=
                (uint64_t)c.hidden * qh ||
            numel(p + "attn_compressor_norm.weight") != (uint64_t)qh ||
            (ratio == 2 && numel(p + "attn_compressor_gate.weight") !=
                                (uint64_t)c.hidden * qh)) {
          LOG_STREAM_ERROR() << "[DSV41] Layer " << l
                             << " has incompatible compressor tensor shapes.\n";
          return false;
        }
        if (numel(p + "indexer.attn_k.weight") !=
                (uint64_t)qh * c.dsv4_indexer_key_length ||
            numel(p + "indexer.k_norm.weight") !=
                (uint64_t)c.dsv4_indexer_key_length) {
          LOG_STREAM_ERROR() << "[DSV41] Layer " << l
                             << " has incompatible indexer key tensors.\n";
          return false;
        }
      }
      if (index_source &&
          (q_lora <= 0 || numel(p + "indexer.proj.weight") !=
                              (uint64_t)c.hidden * c.dsv4_indexer_head_count ||
           numel(p + "indexer.attn_q_b.weight") !=
               (uint64_t)q_lora * c.dsv4_indexer_head_count *
                   c.dsv4_indexer_key_length)) {
        LOG_STREAM_ERROR() << "[DSV41] Layer " << l
                           << " has incompatible indexer tensor shapes.\n";
        return false;
      }
    }

    if (c.dsv41_engram_layer_ids.empty()) {
      LOG_STREAM_ERROR() << "[DSV41] Engram metadata is absent; refusing to "
                            "run a V4.1 file without its conditional memory.\n";
      return false;
    }
    {
      const size_t n_engram = c.dsv41_engram_layer_ids.size();
      const int n_hash_cols =
          std::max(0, c.dsv41_engram_max_ngram_size - 1) *
          c.dsv41_engram_n_heads;
      if (c.dsv41_engram_max_ngram_size < 2 || c.dsv41_engram_n_heads <= 0 ||
          c.dsv41_engram_head_dim <= 0 ||
          c.dsv41_engram_primes.size() !=
              n_engram * (size_t)n_hash_cols ||
          c.dsv41_engram_offsets.size() !=
              n_engram * (size_t)n_hash_cols ||
          c.dsv41_engram_multipliers.size() !=
              n_engram * (size_t)c.dsv41_engram_max_ngram_size ||
          c.dsv41_engram_token_map.empty()) {
        LOG_STREAM_ERROR() << "[DSV41] Incomplete Engram metadata; refusing "
                             "to run without the converter's hash layout.\n";
        return false;
      }
      for (size_t i = 0; i < n_engram; ++i) {
        const int l = (int)c.dsv41_engram_layer_ids[i];
        if (l < 0 || l >= c.n_layers) {
          LOG_STREAM_ERROR() << "[DSV41] Engram layer id out of range: " << l
                             << ".\n";
          return false;
        }
        const std::string p = "blk." + std::to_string(l) + ".";
        auto table = file.tensors.find(p + "engram_embd.weight");
        if (table == file.tensors.end() || table->second.shape.size() != 2 ||
            table->second.shape[0] != (uint64_t)c.dsv41_engram_head_dim ||
            (!c.dsv41_engram_num_embeddings.empty() &&
             table->second.shape[1] !=
                 (uint64_t)c.dsv41_engram_num_embeddings[i]) ||
            row_bytes(table->second.type, c.dsv41_engram_head_dim) == 0 ||
            numel(p + "engram_q.weight") != (uint64_t)c.hc_mult * c.hidden ||
            numel(p + "engram_k.weight") != (uint64_t)c.hc_mult * c.hidden ||
            numel(p + "engram_wkv.weight") !=
                (uint64_t)(c.hc_mult + 1) * c.hidden * n_hash_cols *
                    c.dsv41_engram_head_dim) {
          LOG_STREAM_ERROR() << "[DSV41] Engram tensors for layer " << l
                             << " are missing or have incompatible shapes.\n";
          return false;
        }
      }
    }
  }

  engine.family_ops = resolve_family_ops(fam, c);
  c.cache_shape = engine.family_ops.cache_shape;

  // ---------- MoE router config: gating function, selection bias/scale/norm
  // ---------- See llama.cpp's build_moe_ffn (llama-graph.cpp) for the
  // reference algorithm.
  c.n_expert_shared = file.get_config_int("expert_shared_count", 0);
  if (fam == ModelFamily::Qwen4Exp && c.n_expert_shared == 0) {
    // qwen4exp's converter (conversion/qwen4exp.py -> qwen.py's Qwen3Next
    // base) writes expert_shared_feed_forward_length but NOT
    // expert_shared_count; qwen4exp.cpp's load_arch_tensors creates the
    // shexp tensors unconditionally. Derive the count from the widths so
    // the shared expert isn't silently skipped (execution_layer.cpp gates
    // the whole shexp path on n_expert_shared > 0).
    int shexp_len = file.get_config_int("expert_shared_feed_forward_length", 0);
    if (shexp_len > 0 && c.moe_inter > 0) {
      c.n_expert_shared = std::max(1, shexp_len / c.moe_inter);
      LOG_STREAM_INFO() << "[QWEN4EXP] expert_shared_count absent; derived "
                        << c.n_expert_shared << " from shared width "
                        << shexp_len << " / expert width " << c.moe_inter
                        << "\n";
    }
  }
  c.expert_weights_scale = file.get_config_float("expert_weights_scale", 0.0f);
  c.expert_weights_norm = file.get_config_int("expert_weights_norm", 1) != 0;
  {
    // GGUF encoding (llama.cpp llama_expert_gating_func_type):
    // 0 = none/unset, 1 = softmax, 2 = sigmoid, 4 = sqrtsoftplus
    // (DeepSeek-V4; see llama-hparams.h's LLAMA_EXPERT_GATING_FUNC_TYPE_
    // SQRT_SOFTPLUS). A value the file doesn't set (0) falls back to this
    // family's own default (engine.family_ops.default_gating) rather than
    // the old hardcoded arch0=="glm-dsa"||"hy_v3" fallback -- note that old
    // fallback compared against "hy_v3" (missing the hyphen; the real
    // family string everywhere else in this codebase is "hy-v3"), so it
    // silently never actually fired for real Hy-V3 files. family_ops.cpp's
    // resolve_family_ops carries that ACTUAL prior behavior forward
    // unchanged (Hunyuan's default_gating is Softmax, not Sigmoid) --
    // fixing the typo is a separate decision from this refactor.
    int gating = file.get_config_int("expert_gating_func", 0);
    if (gating == 2)
      c.gating_func = GatingFunc::Sigmoid;
    else if (gating == 1)
      c.gating_func = GatingFunc::Softmax;
    else if (gating == 4)
      c.gating_func = GatingFunc::SqrtSoftplus;
    else
      c.gating_func = engine.family_ops.default_gating;
  }
  // Original HunyuanMoE ropes Q/K before QK-norm; Hy-V3/GLM-5.2 norm first.
  c.rope_before_norm = (arch0 == "hunyuan-moe");

  // ---------- YaRN long-context RoPE scaling ----------
  // See ggml's rope_yarn/ggml_rope_yarn_corr_dims for the reference formulas.
  // ext_factor == 0 (the default, when rope.scaling.type != "yarn") reduces
  // every formula below to plain RoPE — no behavior change for models that
  // don't set YaRN metadata.
  {
    std::string scaling_type =
        file.get_config_string("rope.scaling.type", "none");
    c.rope_scaling_yarn = (scaling_type == "yarn");
    c.rope_ext_factor = c.rope_scaling_yarn ? 1.0f : 0.0f;
    float scale_factor = file.get_config_float("rope.scaling.factor", 1.0f);
    c.rope_freq_scale = scale_factor > 0.0f ? 1.0f / scale_factor : 1.0f;
    c.rope_attn_factor =
        file.get_config_float("rope.scaling.attn_factor", 1.0f);
    // GLM-DSA's own hparams loader never sets yarn_log_multiplier (only
    // DeepSeek2/DeepSeek32/Mistral3 do upstream) so this is 0.0 for
    // GLM-5.2 in practice; read generically for forward-compat.
    c.rope_yarn_log_mul =
        file.get_config_float("rope.scaling.yarn_log_multiplier", 0.0f);
    c.rope_n_ctx_orig =
        file.get_config_int("rope.scaling.original_context_length",
                            file.get_config_int("context_length", max_ctx));
    // DeepSeek-V4 drops V3's rope_yarn_log_mul mscale-adjustment term from
    // its YaRN attn-factor formula entirely (confirmed against llama.cpp's
    // deepseek4.cpp: dsv4_rope_attn_factor() has no log_mul term at all,
    // unlike deepseek2.cpp's attn_factor_org*(1+0.1*rope_yarn_log_mul*
    // log(1/freq_scale))) -- forced to 0 here regardless of what the file
    // itself sets, rather than trusting a key DeepSeek-V4 checkpoints
    // shouldn't be setting in the first place. rope.hpp's own formula
    // already multiplies this coefficient through to exactly zero
    // contribution when it's 0, so no new RoPE code is needed for this --
    // only this one override.
    if (fam == ModelFamily::DeepSeek_V4 || fam == ModelFamily::DeepSeek_V41)
      c.rope_yarn_log_mul = 0.0f;
    // GLM5Next's text tower has NO rotary at all (n_rot == 0, nope-only MLA
    // -- the converter hard-rejects anything else), and its DSA kq_scale is
    // the plain 1/sqrt(qk_head_dim) with no YaRN/mscale term (glm5next.cpp's
    // build_dsa_layer). Force the no-YaRN reduction regardless of any
    // rope.scaling keys the file might carry, so yarn_kq_scale reduces to
    // exactly that.
    if (fam == ModelFamily::GLM5NEXT) {
      c.rope_scaling_yarn = false;
      c.rope_ext_factor = 0.0f;
    }

    LOG_STREAM_INFO() << "[ROPE] scaling.type=\"" << scaling_type
                      << "\" ext_factor=" << c.rope_ext_factor
                      << " freq_scale=" << c.rope_freq_scale
                      << " attn_factor=" << c.rope_attn_factor
                      << " yarn_log_mul=" << c.rope_yarn_log_mul
                      << " n_ctx_orig=" << c.rope_n_ctx_orig
                      << " -- ext_factor=0 means plain RoPE, attn_scale "
                         "reduces to 1/sqrt(qk_head) "
                         "nonzero means the YaRN branch is live.\n";
  }

  // (DeepSeek-V4 Hyper-Connections/SWA/grouped-output-projection hparam
  // block MOVED above, right after `ModelFamily fam = detect_model_family`
  // and before `resolve_family_ops(fam, c)` -- see the comment there for
  // why the original position here was a real ordering bug.)

  // Fallback: detect from tensor shapes if KV didn't provide values
  if (c.hidden == 0 || c.n_layers == 0) {
    uint64_t max_layer_found = 0;
    std::regex layer_regex(R"(blk\.([0-9]+)\.)");
    for (const auto &[name, tensor] : file.tensors) {
      std::smatch match;
      if (std::regex_search(name, match, layer_regex)) {
        uint64_t layer_idx = std::stoull(match[1].str());
        if (layer_idx > max_layer_found)
          max_layer_found = layer_idx;
      }
      // FIX: same real bug as the vocab_size-KV-absent fallback above (see
      // that comment for the full derivation against llama.cpp's real
      // {n_embd, n_vocab} create_tensor convention) -- this was reading
      // shape[0]/shape[1] exactly TRANSPOSED (vocab from shape[0], hidden
      // from shape[1], when shape[0]=n_embd/shape[1]=n_vocab is the real
      // order). Only reachable when embedding_length is ALSO absent from
      // KV (narrower gate than the vocab-only fallback above, so this
      // specific block wasn't what corrupted the real DeepSeek-V4-Flash
      // run -- c.hidden was correctly read from KV there -- but it's the
      // identical class of bug and would have silently hit any file
      // missing both KVs). Added a size() guard: this loop iterates every
      // tensor in the file, not just 2D weight matrices, so indexing
      // shape[1] unconditionally would be undefined behavior the moment a
      // 1D tensor happened to be named "token_embd.weight".
      if (c.hidden == 0 && name == "token_embd.weight" &&
          tensor.shape.size() >= 2) {
        c.vocab = tensor.shape[1];
        c.hidden = tensor.shape[0];
      }
      // Detect hidden from any attn weight shape
      if (c.hidden == 0 &&
          name.find("attn_output.weight") != std::string::npos) {
        c.hidden = tensor.shape[0]; // [H*vh, hidden] → shape[0] might be H*vh
      }
    }
    if (c.n_layers == 0)
      c.n_layers = max_layer_found + 1;
  }

  // Loud, UNGATED sanity check on the final c.vocab/c.hidden values against
  // token_embd.weight's real numel -- added directly alongside the two
  // shape[0]/shape[1] bugs just fixed above, specifically because THIS
  // exact drift (c.vocab silently == c.hidden) went completely undetected
  // until a real generation run produced garbage output and a multi-step
  // investigation traced it back. verify_tensor_dimensions()'s own
  // check()/check_orient() machinery (this same file, near the top) could
  // have caught it via the same numel-mismatch technique, but that
  // function EARLY-RETURNS for any cache_shape other than MLAAbsorbed --
  // i.e. it never runs at all for DeepSeek-V4 (MLARawSWA), which is
  // exactly the family this bug was found on. This check is architecture-
  // agnostic (token_embd.weight's numel must equal vocab*hidden regardless
  // of attention/cache family) so it runs for every family, unconditionally.
  {
    auto it = file.tensors.find("token_embd.weight");
    if (it != file.tensors.end() && !it->second.shape.empty()) {
      uint64_t real_numel = 1;
      for (auto d : it->second.shape) real_numel *= d;
      uint64_t expected_numel = (uint64_t)c.vocab * (uint64_t)c.hidden;
      if (real_numel != expected_numel) {
        std::string shape_str;
        for (auto d : it->second.shape) shape_str += std::to_string(d) + ",";
        LOG_ERROR(
            "[VOCAB-CHECK] token_embd.weight real numel=%llu (shape=[%s]) != "
            "vocab(%d)*hidden(%d)=%llu -- ArchitectureConfig::vocab is WRONG. "
            "The prefill embed loop's bounds check (generation_loop.hpp) will "
            "silently skip embedding any real token id >= vocab, leaving that "
            "row zero and producing fluent-but-garbage generation with no "
            "NaN/Inf anywhere to catch it. Check vocab_size-style KV keys and "
            "the shape[0]/shape[1] fallback logic just above in this file.\n",
            (unsigned long long)real_numel, shape_str.c_str(), c.vocab,
            c.hidden, (unsigned long long)expected_numel);
      }
    }
  }

  // Detect experts from packed expert tensors
  if (c.n_experts == 0) {
    for (const auto &[name, tensor] : file.tensors) {
      if (name.find("ffn_gate_exps.weight") != std::string::npos &&
          tensor.shape.size() >= 3) {
        c.n_experts = tensor.shape[2]; // shape=[hidden, inter, n_experts]
        if (c.moe_inter == 0)
          c.moe_inter = tensor.shape[1];
        break;
      }
    }
  }

  if (c.n_heads == 0 && c.hidden > 0 && c.v_head > 0) {
    // For MLA: attn_output is [n_heads*v_head, hidden], so n_heads =
    // shape[0]/v_head
    for (const auto &[name, tensor] : file.tensors) {
      if (name.find("blk.0.attn_output.weight") != std::string::npos &&
          tensor.shape.size() >= 2) {
        c.n_heads = tensor.shape[0] / c.v_head;
        break;
      }
    }
  }

  // Both prefill fast paths (moe_prefill_batched, dense_bf16_prefill_gemm)
  // live in family-agnostic code -- moe_forward_gguf and dense_matmul_host --
  // so they are live for GLM and DeepSeek too, but were tuned and verified on
  // qwen4exp. Engine defaults are FALSE; a config opts in per model. Warn here,
  // where the family is actually known.
  //
  // MEASURED on GLM-5.3 (UD-IQ2_M) 2026-09-04, each flag isolated, same prompt:
  //  - moe_prefill_batched ALONE  -> GARBAGE output ("RW Federation Coalition
  //    federation foundedclaimer thing..."). Its own comment in
  //    execution_layer.cpp calls it a "pure loop restructure, same arithmetic"
  //    -- that is WRONG, or at least not true for GLM. Suspect it drops or
  //    misattributes expert contributions: the same run's expert= bucket
  //    HALVED (0.98s -> 0.58s), which is what skipping work looks like, not
  //    what reordering it looks like. NOT ROOT-CAUSED YET.
  //  - dense_bf16_prefill_gemm ALONE -> COHERENT, correct output. It is a real
  //    numerical change (weights dequantised to BF16, 8-bit mantissa, cuBLAS
  //    with F32 accumulate, replacing the fused kernel's F32 decode) and it
  //    genuinely engages here -- GLM-5.3's dense pool has 313 Q8_0 tensors and
  //    the gate is (Q8_0 && S>1). It is simply accurate enough for this model.
  //
  // So the risk ranking is the OPPOSITE of what this comment claimed before
  // any of it was measured: the "safe loop restructure" is the one that breaks
  // GLM, and the "real numerical change" is the one that is fine.
  {
    const ModelFamily wfam = detect_model_family(file);
    const bool shared_fam =
        (wfam == ModelFamily::GLM_DSA || wfam == ModelFamily::GLM4_MOE ||
         wfam == ModelFamily::GLM5NEXT || wfam == ModelFamily::DeepSeek_V3 ||
         wfam == ModelFamily::DeepSeek_V4 ||
         wfam == ModelFamily::DeepSeek_V41);
    if (shared_fam && moe_prefill_batched()) {
      LOG_STREAM_WARN()
          << "[CONFIG] GLM/DeepSeek model detected with moe_prefill_batched "
             "ENABLED. This flag is verified on qwen4exp ONLY and is MEASURED "
             "to produce GARBAGE OUTPUT on GLM-5.3 (2026-09-04, isolated A/B). "
             "Set moe_prefill_batched = false in this config unless you have "
             "re-verified generated text on THIS model.\n";
    }
    if (shared_fam && dense_bf16_prefill_gemm()) {
      LOG_STREAM_WARN()
          << "[CONFIG] GLM/DeepSeek model detected with "
             "dense_bf16_prefill_gemm ENABLED (BF16 weights for Q8_0 dense at "
             "S>1 -- a real numerical change). Verified COHERENT on GLM-5.3 "
             "(2026-09-04), but unverified on DeepSeek; compare generated text "
             "against a known-good run before trusting it there.\n";
    }
  }

  LOG_STREAM_INFO() << "[GRAPH] Architecture discovered:\n";
  LOG_STREAM_INFO() << "  -> Hidden: " << c.hidden
                    << " | Layers: " << c.n_layers << " | Heads: " << c.n_heads
                    << " | Vocab: " << c.vocab << "\n";
  LOG_STREAM_INFO() << "  -> Experts: " << c.n_experts << " (top-" << c.topk
                    << ", inter=" << c.moe_inter << ")\n";
  LOG_STREAM_INFO() << "  -> MLA: kv_lora=" << c.kv_lora
                    << " qk_nope=" << c.qk_nope << " qk_rope=" << c.qk_rope
                    << " v_head=" << c.v_head << "\n";
  LOG_STREAM_INFO() << "  -> First dense layers: " << first_dense << "\n";

  // ---------- Allocate KV cache (host + attempt GPU) ----------
  // Delegates to engine.family_ops.alloc_cache_fn -- each CacheShape's own
  // allocator (family_ops.cpp) now owns the field sizing AND its own log
  // line, rather than this function switching on is_mla and inlining both
  // bodies. A new CacheShape (e.g. DeepSeek-V4's MLARawSWA) is a new
  // alloc_*_cache function pointed to from resolve_family_ops's table, not a
  // new branch here.
  engine.max_t = max_ctx;
  engine.family_ops.alloc_cache_fn(engine, c, max_ctx);
  if (engine.host_memory_allocation_failed) {
    LOG_ERROR("[MEM-GUARD] cache initialization exceeded the process host budget; "
              "aborting model initialization before any generation\n");
    return false;
  }

  // ---------- Build dense preload manifest with correct GGUF tensor names
  // ----------
  std::vector<std::string> dense_preload_manifest;
  // MTP is deliberately a narrow, opt-in bridge, restricted to the GLM
  // family (glm_family below) -- Qwen4Exp is out of scope entirely (not
  // included in glm_family), since its NextN block consumes its wide HC
  // stream and is not representable by this engine's [S,D] forward
  // interface yet. Disable unsupported combinations here, before any cache
  // vectors or manifests are allocated, so they fall back to ordinary target
  // generation rather than partially loading a draft graph.
  //
  // GLM5NEXT's own multi-stream Hyper-Connections (hc_mult > 1) is NOT
  // disabled here (previously was) -- verified against the real
  // glm5next.cpp reference: the NextN/MTP block is built with
  // `GGML_ASSERT(!layer.hc_attn_fn && "the NextN block has no mHC mixer")`,
  // an ordinary plain-residual DSA layer regardless of the trunk's own
  // hc_mult. See MtpSpeculator::supported()'s own comment (mtp_speculative.
  // cpp) for the full reasoning and how the caller-side hc_expand_embedding/
  // hc_collapse_output bracket is handled for GLM5NEXT's target-trunk
  // verification path.
  bool mtp_active = false;
  if (engine.mtp_enabled) {
    const bool glm_family = fam == ModelFamily::GLM5NEXT ||
                            fam == ModelFamily::GLM_DSA;
    const bool qwen4exp_shared_mtp_family =
        fam == ModelFamily::Qwen4Exp && engine.has_mtp_model;
    if (!glm_family && !qwen4exp_shared_mtp_family) {
      LOG_STREAM_WARN() << "[MTP] requested but this model family is not yet "
                           "supported (Qwen4Exp requires mtp_model_path to be "
                           "configured); using ordinary generation\n";
    } else if (n_layer_nextn != 1) {
      LOG_STREAM_WARN() << "[MTP] requested but nextn_predict_layers="
                        << n_layer_nextn
                        << " (the GLM/Qwen4Exp bridge requires exactly one "
                           "extra block); using ordinary generation\n";
    } else if (glm_family) {
      const int mtp_layer = c.n_layers;
      // GLM5Next's extra block is always the ordinary DSA block, even though
      // its target trunk is hybrid KDA+DSA.  Reusing the KDA interpretation
      // here would call a recurrent-state path with no MTP state allocation.
      if (fam == ModelFamily::GLM5NEXT &&
          glm5next_is_kda_layer(c, mtp_layer)) {
        LOG_STREAM_WARN() << "[MTP] GLM5NEXT NextN block is marked KDA, but "
                             "the reference MTP block is DSA; using ordinary "
                             "generation\n";
      } else {
        const std::string p = "blk." + std::to_string(mtp_layer) + ".";
        const char *required[] = {
            "nextn.eh_proj.weight", "nextn.enorm.weight", "nextn.hnorm.weight",
            "attn_norm.weight",
            "attn_q_a.weight", "attn_q_a_norm.weight", "attn_q_b.weight",
            "attn_kv_a_mqa.weight", "attn_kv_a_norm.weight",
            "attn_k_b.weight", "attn_v_b.weight", "attn_output.weight",
            "ffn_norm.weight", "ffn_gate_inp.weight",
            "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
            "ffn_down_shexp.weight",
        };
        std::string missing;
        for (const char *suffix : required) {
          if (file.tensors.find(p + suffix) == file.tensors.end()) {
            if (!missing.empty()) missing += ", ";
            missing += suffix;
          }
        }
        auto has = [&](const std::string &name) {
          return file.tensors.find(name) != file.tensors.end();
        };
        if (!missing.empty() || !has(p + "ffn_gate_exps.weight") ||
            !has(p + "ffn_up_exps.weight") ||
            !has(p + "ffn_down_exps.weight")) {
          if (missing.empty()) missing = "packed routed expert tensors";
          else missing += ", packed routed expert tensors";
          LOG_STREAM_WARN() << "[MTP] requested but NextN tensors are missing: "
                            << missing << "; using ordinary generation\n";
        } else {
          mtp_active = true;
          LOG_STREAM_INFO() << "[MTP] GLM NextN validation passed: one extra "
                               "DSA/MLA+MoE block at layer "
                            << mtp_layer << "\n";
        }
      }
    } else {
      // Qwen4Exp "shared" MTP checkpoint: the extra block lives at layer
      // index c.n_layers (one PAST the base file's own real trunk layers,
      // never carved out of them -- see the n_layer_nextn computation
      // above), and its tensors live in the SEPARATE mtp_model_file, not
      // `file`. Verified against the real mtp-*-shared-*.gguf file (32
      // tensors, all under blk.48.*) and against
      // /home/ray/Projects/unsloth/llama.cpp's qwen4exp.cpp (branch
      // mtp/qwen4exp-nextn): layer 48 is architecturally an ordinary
      // full-attention trunk layer (split attn_q/k/v, not the recurrent
      // attn_qkv-combined GDN layout) plus the nextn.* bridge/head
      // tensors. indexer.* tensors are present but intentionally not
      // required here -- v1 attends densely, same as the base trunk's own
      // qwen4exp_full_attention_gguf.
      const int mtp_layer = c.n_layers;
      const std::string p = "blk." + std::to_string(mtp_layer) + ".";
      GGUFFile &mf = engine.mtp_model_file;
      const char *required[] = {
          "attn_q.weight", "attn_k.weight", "attn_v.weight",
          "attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight",
          "hc_attn_norm.weight", "hc_attn_down.weight", "hc_attn_up.weight",
          "hc_attn_inject.weight",
          "hc_ffn_norm.weight", "hc_ffn_down.weight", "hc_ffn_up.weight",
          "hc_ffn_inject.weight",
          "ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight",
          "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
          "ffn_down_shexp.weight",
          "nextn.eh_proj.weight", "nextn.enorm.weight", "nextn.hnorm.weight",
          "nextn.hc_head_norm.weight", "nextn.hc_head_down.weight",
          "nextn.hc_head_up.weight",
      };
      std::string missing;
      for (const char *suffix : required) {
        if (mf.tensors.find(p + suffix) == mf.tensors.end()) {
          if (!missing.empty()) missing += ", ";
          missing += suffix;
        }
      }
      auto has_mtp = [&](const std::string &name) {
        return mf.tensors.find(name) != mf.tensors.end();
      };
      if (!missing.empty() || !has_mtp(p + "ffn_gate_exps.weight") ||
          !has_mtp(p + "ffn_up_exps.weight") ||
          !has_mtp(p + "ffn_down_exps.weight")) {
        if (missing.empty()) missing = "packed routed expert tensors";
        else missing += ", packed routed expert tensors";
        LOG_STREAM_WARN() << "[MTP] requested but Qwen4Exp NextN tensors are "
                             "missing from mtp_model_path: " << missing
                          << "; using ordinary generation\n";
      } else {
        mtp_active = true;
        LOG_STREAM_INFO() << "[MTP] Qwen4Exp NextN validation passed: one "
                             "extra full-attention+MoE block at layer "
                          << mtp_layer << " (shared checkpoint: "
                          << mf.resolved_file_path << ")\n";
      }
    }
  }
  engine.mtp_enabled = mtp_active;
  const int cache_layers = c.n_layers + (mtp_active ? n_layer_nextn : 0);
  engine.layers_experts.resize(cache_layers);

  // GLM5NEXT carries six additional indexer tensors on each DSA layer. The
  // v1 implementation intentionally attends densely and does not consume
  // them, but count them here so a real GGUF load proves that omission is
  // understood rather than silently hiding a naming mismatch.
  int glm5next_dsa_layers = 0;
  int glm5next_indexer_tensors_present = 0;
  static constexpr const char * kGlm5nextIndexerSuffixes[] = {
      "indexer.k_norm.weight",
      "indexer.proj.weight",
      "indexer.attn_k.weight",
      "indexer.attn_q_b.weight",
      "indexer_compressor_gate.weight",
      "indexer_compressor_ape.weight",
  };

  LOG_STREAM_INFO() << "[GRAPH] Attention type: "
                    << (fam == ModelFamily::GLM5NEXT
                            ? "HYBRID KDA (recurrent) + MLA (absorbed, DSA layers)"
                        : fam == ModelFamily::DeepSeek_V41
                            ? "MLA (raw SWA + shared compressed KV/indexer, Engram)"
                        : c.cache_shape == CacheShape::MLAAbsorbed
                            ? "MLA (absorbed)"
                        : c.cache_shape == CacheShape::MLARawSWA
                            ? "MLA (raw, SWA)"
                            : "dense Q/K/V")
                    << "  (arch=\"" << arch0 << "\")\n";

  for (int l = 0; l < c.n_layers; ++l) {
    std::string p = "blk." + std::to_string(l) + ".";

    // Norms (always present -- EXCEPT qwen4exp, whose low-rank
    // hyper-connections replace every layer norm; its per-layer norm weights
    // live inside the hc_* tensors pushed by its own branch below).
    if (fam != ModelFamily::Qwen4Exp) {
      dense_preload_manifest.push_back(p + "attn_norm.weight");
      dense_preload_manifest.push_back(p + "ffn_norm.weight");
    }

    if (fam == ModelFamily::Qwen4Exp) {
      // Qwen3.8-Flash-Next hybrid: per-layer tensor set depends on the
      // layer type (attention.recurrent_layers / full_attention_interval).
      // Names confirmed against llama.cpp's qwen4exp.cpp load_arch_tensors
      // + llama-arch.cpp tensor table. Low-rank HC tensors (every layer,
      // both sub-blocks) replace the fn/base/scale set DeepSeek-V4 uses.
      dense_preload_manifest.push_back(p + "hc_attn_norm.weight");
      dense_preload_manifest.push_back(p + "hc_attn_down.weight");
      dense_preload_manifest.push_back(p + "hc_attn_up.weight");
      dense_preload_manifest.push_back(p + "hc_attn_inject.weight");
      dense_preload_manifest.push_back(p + "hc_ffn_norm.weight");
      dense_preload_manifest.push_back(p + "hc_ffn_down.weight");
      dense_preload_manifest.push_back(p + "hc_ffn_up.weight");
      dense_preload_manifest.push_back(p + "hc_ffn_inject.weight");
      if (qwen4exp_is_gdn_layer(c, l)) {
        // Recurrent GDN layer: fused QKV projection + z-gate projection +
        // one fused short-conv weight + per-value-head decay/beta
        // projections + gated output norm.
        dense_preload_manifest.push_back(p + "attn_qkv.weight");
        dense_preload_manifest.push_back(p + "attn_gate.weight");
        dense_preload_manifest.push_back(p + "ssm_conv1d.weight");
        dense_preload_manifest.push_back(p + "ssm_dt.bias");
        dense_preload_manifest.push_back(p + "ssm_a"); // no .weight suffix
        dense_preload_manifest.push_back(p + "ssm_beta.weight");
        dense_preload_manifest.push_back(p + "ssm_alpha.weight");
        dense_preload_manifest.push_back(p + "ssm_norm.weight");
        dense_preload_manifest.push_back(p + "ssm_out.weight");
      } else {
        // Full-attention layer: fused Q|gate projection (interleaved per
        // head), per-head QK norms, plain K/V, output projection. Indexer
        // tensors (indexer.*) deliberately NOT preloaded -- QSA is deferred;
        // these layers run dense over the full KV cache in v1.
        dense_preload_manifest.push_back(p + "attn_q.weight");
        dense_preload_manifest.push_back(p + "attn_k.weight");
        dense_preload_manifest.push_back(p + "attn_v.weight");
        dense_preload_manifest.push_back(p + "attn_q_norm.weight");
        dense_preload_manifest.push_back(p + "attn_k_norm.weight");
        dense_preload_manifest.push_back(p + "attn_output.weight");
      }
      // PLE n-gram hash embedding (select layers only). The big gather table
      // (per_layer_token_embd.weight) is deliberately NOT preloaded -- it is
      // far too large for the dense-resident set (potentially tens of GB);
      // rows are gathered on demand (see src/ple_ngram.cpp).
      if (qwen4exp_is_ple_layer(c, l)) {
        dense_preload_manifest.push_back(p + "ple_key.weight");
        dense_preload_manifest.push_back(p + "ple_value.weight");
        dense_preload_manifest.push_back(p + "ple_norm_key.weight");
        dense_preload_manifest.push_back(p + "ple_norm_query.weight");
        dense_preload_manifest.push_back(p + "ple_norm_conv.weight");
        dense_preload_manifest.push_back(p + "ple_conv1d.weight");
      }
    } else if (fam == ModelFamily::GLM5NEXT) {
      // GLM 5.3 Flash hybrid: per-layer tensor set depends on the layer
      // type (head_count_kv array). Names confirmed against llama.cpp's
      // glm5next conversion + llama-arch.cpp tensor table.
      if (glm5next_is_kda_layer(c, l)) {
        // Recurrent KDA layer: separate Q/K/V projections + separate
        // per-channel short-conv weights + low-rank decay/output gates.
        dense_preload_manifest.push_back(p + "attn_q.weight");
        dense_preload_manifest.push_back(p + "attn_k.weight");
        dense_preload_manifest.push_back(p + "attn_v.weight");
        dense_preload_manifest.push_back(p + "ssm_conv1d_q.weight");
        dense_preload_manifest.push_back(p + "ssm_conv1d_k.weight");
        dense_preload_manifest.push_back(p + "ssm_conv1d_v.weight");
        dense_preload_manifest.push_back(p + "ssm_f_a.weight");
        dense_preload_manifest.push_back(p + "ssm_f_b.weight");
        dense_preload_manifest.push_back(p + "ssm_g_a.weight");
        dense_preload_manifest.push_back(p + "ssm_g_b.weight");
        dense_preload_manifest.push_back(p + "ssm_beta.weight");
        dense_preload_manifest.push_back(p + "ssm_a");        // no .weight suffix
        dense_preload_manifest.push_back(p + "ssm_dt.bias");
        dense_preload_manifest.push_back(p + "ssm_norm.weight");
        dense_preload_manifest.push_back(p + "attn_output.weight");
      } else {
        // Full-attention DSA layer: nope-only absorbed MLA, same tensor
        // names as GLM-5.2's absorbed path (which v1 reuses verbatim).
        // Indexer tensors (indexer.*) deliberately NOT preloaded -- the
        // sparse path is deferred; DSA runs dense over the full KV cache.
        dense_preload_manifest.push_back(p + "attn_q_a.weight");
        dense_preload_manifest.push_back(p + "attn_q_a_norm.weight");
        dense_preload_manifest.push_back(p + "attn_q_b.weight");
        dense_preload_manifest.push_back(p + "attn_kv_a_mqa.weight");
        dense_preload_manifest.push_back(p + "attn_kv_a_norm.weight");
        dense_preload_manifest.push_back(p + "attn_k_b.weight");
        dense_preload_manifest.push_back(p + "attn_v_b.weight");
        dense_preload_manifest.push_back(p + "attn_output.weight");

        ++glm5next_dsa_layers;
        for (const char *suffix : kGlm5nextIndexerSuffixes) {
          if (file.tensors.find(p + suffix) != file.tensors.end())
            ++glm5next_indexer_tensors_present;
        }
      }
    } else if (c.cache_shape == CacheShape::MLAAbsorbed) {
      // MLA absorbed attention (GLM-5.2 / DeepSeek-V3)
      dense_preload_manifest.push_back(p + "attn_q_a.weight");
      dense_preload_manifest.push_back(p + "attn_q_a_norm.weight");
      dense_preload_manifest.push_back(p + "attn_q_b.weight");
      dense_preload_manifest.push_back(p + "attn_kv_a_mqa.weight");
      dense_preload_manifest.push_back(p + "attn_kv_a_norm.weight");
      dense_preload_manifest.push_back(p + "attn_k_b.weight");
      dense_preload_manifest.push_back(p + "attn_v_b.weight");
      dense_preload_manifest.push_back(p + "attn_output.weight");
    } else if (fam == ModelFamily::DeepSeek_V41) {
      // DeepSeek-V4.1 uses the V4 raw/SWA tensor names, but its compressed
      // tensors are ratio-1/2 source-layer weights and its indexer derives
      // keys directly from the pooled latent. The huge engram table is never
      // added here; it is mmap-gathered by deepseek41_engram.cpp.
      dense_preload_manifest.push_back(p + "attn_q_a.weight");
      dense_preload_manifest.push_back(p + "attn_q_a_norm.weight");
      dense_preload_manifest.push_back(p + "attn_q_b.weight");
      dense_preload_manifest.push_back(p + "attn_kv.weight");
      dense_preload_manifest.push_back(p + "attn_kv_a_norm.weight");
      dense_preload_manifest.push_back(p + "attn_sinks.weight");
      dense_preload_manifest.push_back(p + "attn_output_a.weight");
      dense_preload_manifest.push_back(p + "attn_output_b.weight");

      const int64_t ratio = c.dsv4_compress_ratios[(size_t)l];
      const bool kv_source = dsv41_is_layer_in(c.dsv41_kv_source_layers, l);
      const bool index_source =
          dsv41_is_layer_in(c.dsv41_index_source_layers, l);
      if (ratio != 0 && kv_source) {
        dense_preload_manifest.push_back(p + "attn_compressor_kv.weight");
        if (ratio > 1)
          dense_preload_manifest.push_back(p + "attn_compressor_gate.weight");
        dense_preload_manifest.push_back(p + "attn_compressor_norm.weight");
      }
      if (ratio != 0 && index_source) {
        dense_preload_manifest.push_back(p + "indexer.proj.weight");
        dense_preload_manifest.push_back(p + "indexer.attn_q_b.weight");
      }
      if (ratio != 0 && kv_source) {
        dense_preload_manifest.push_back(p + "indexer.attn_k.weight");
        dense_preload_manifest.push_back(p + "indexer.k_norm.weight");
      }
      if (dsv41_is_engram_layer(c, l)) {
        dense_preload_manifest.push_back(p + "engram_q.weight");
        dense_preload_manifest.push_back(p + "engram_k.weight");
        dense_preload_manifest.push_back(p + "engram_wkv.weight");
      }
    } else if (c.cache_shape == CacheShape::MLARawSWA) {
      // MLA raw/SWA attention (DeepSeek-V4) -- no absorption tensors
      // (attn_k_b/attn_v_b/kv_lora-sized attn_kv_a_mqa): wkv already
      // produces a per-head-sized latent, attended to directly as MQA. See
      // dsv4_attention_gguf (attention_layer.cpp, Project B).
      dense_preload_manifest.push_back(p + "attn_q_a.weight");
      dense_preload_manifest.push_back(p + "attn_q_a_norm.weight");
      dense_preload_manifest.push_back(p + "attn_q_b.weight");
      dense_preload_manifest.push_back(p + "attn_kv.weight");
      // Real name is attn_kv_a_norm (reused from V3's naming), not
      // attn_kv_norm -- see attention_layer.cpp's matching fix comment.
      dense_preload_manifest.push_back(p + "attn_kv_a_norm.weight");
      dense_preload_manifest.push_back(p + "attn_sinks.weight");
      // Grouped low-rank output projection replaces the single attn_output.
      dense_preload_manifest.push_back(p + "attn_output_a.weight");
      dense_preload_manifest.push_back(p + "attn_output_b.weight");

      const int64_t ratio = c.dsv4_compress_ratios.empty()
          ? 0 : c.dsv4_compress_ratios[(size_t)l];
      if (ratio != 0) {
        dense_preload_manifest.push_back(p + "attn_compressor_kv.weight");
        dense_preload_manifest.push_back(p + "attn_compressor_gate.weight");
        dense_preload_manifest.push_back(p + "attn_compressor_ape.weight");
        dense_preload_manifest.push_back(p + "attn_compressor_norm.weight");
        if (ratio == kDsv4CsaRatio) {
          dense_preload_manifest.push_back(p + "indexer.proj.weight");
          dense_preload_manifest.push_back(p + "indexer.attn_q_b.weight");
          dense_preload_manifest.push_back(p + "indexer_compressor_kv.weight");
          dense_preload_manifest.push_back(p + "indexer_compressor_gate.weight");
          dense_preload_manifest.push_back(p + "indexer_compressor_ape.weight");
          dense_preload_manifest.push_back(p + "indexer_compressor_norm.weight");
        }
      }
    } else {
      // Dense Q/K/V attention (Hunyuan / Llama / GLM-4)
      dense_preload_manifest.push_back(p + "attn_q.weight");
      dense_preload_manifest.push_back(p + "attn_k.weight");
      dense_preload_manifest.push_back(p + "attn_v.weight");
      dense_preload_manifest.push_back(p +
                                       "attn_q_norm.weight"); // HY-V3 QK-norm
      dense_preload_manifest.push_back(p + "attn_k_norm.weight");
      dense_preload_manifest.push_back(p + "attn_output.weight");
    }

    // Hyper-Connections (DeepSeek-V4): one set of mixing-coefficient
    // tensors per sub-block (attention, FFN/MoE), every layer. Inactive
    // (c.hc_mult<=1, the default for every other family) means these
    // tensor names are simply never added to the manifest. Qwen4Exp is
    // explicitly excluded: its low-rank HC tensors (hc_*_norm/down/up/
    // inject) are pushed by its own branch above instead.
    if (c.hc_mult > 1 && fam != ModelFamily::Qwen4Exp) {
      dense_preload_manifest.push_back(p + "hc_attn_fn.weight");
      dense_preload_manifest.push_back(p + "hc_attn_base.weight");
      dense_preload_manifest.push_back(p + "hc_attn_scale.weight");
      dense_preload_manifest.push_back(p + "hc_ffn_fn.weight");
      dense_preload_manifest.push_back(p + "hc_ffn_base.weight");
      dense_preload_manifest.push_back(p + "hc_ffn_scale.weight");
    }

    bool is_moe = (c.n_experts > 0 && l >= first_dense);
    if (is_moe) {
      // Hash-routed layers (DeepSeek-V4, first c.hash_layer_count layers):
      // expert SELECTION is a direct token-id -> expert-id lookup via
      // ffn_gate_tid2eid instead of top-k over router probs. FIX: this used
      // to ALSO skip ffn_gate_inp.weight entirely for these layers, under
      // the assumption that hash-routed layers need no learned router at
      // all -- WRONG, confirmed against llama.cpp's real DeepSeek-V4
      // implementation (src/models/deepseek4.cpp + llama-graph.cpp's
      // build_moe_ffn): the router's logits/probs are computed
      // UNCONDITIONALLY for every layer, hash-routed or not -- they still
      // supply the WEIGHTS for whichever experts end up selected. Only the
      // SELECTION step (which experts) is overridden for hash-routed
      // layers; the bias tensor (exp_probs_b) genuinely doesn't apply there
      // (selection isn't argsort-over-biased-probs any more), so that part
      // stays conditional. Omitting ffn_gate_inp.weight here meant
      // moe_forward_gguf's own router lookup (execution_layer.cpp) could
      // never find it for these layers and SILENTLY ZERO-FILLED the entire
      // MoE output for every hash-routed layer, on every token -- finite
      // (not NaN), so invisible to check_finite. This was the actual root
      // cause of a real fluent-but-garbage generation run, not "falls back
      // to the wrong router" as the old comment here (and
      // execution_layer.cpp's warning) claimed.
      bool hash_routed = (l < c.hash_layer_count);
      // Router: needed by EVERY MoE layer, hash-routed or not (see above).
      dense_preload_manifest.push_back(p + "ffn_gate_inp.weight");
      if (hash_routed) {
        dense_preload_manifest.push_back(p + "ffn_gate_tid2eid.weight");
      } else {
        // Selection-bias tensor (DeepSeek-V3/GLM-5.2 style). GGUF name is
        // "blk.N.exp_probs_b.bias" — llama.cpp's tn(LLM_TENSOR_FFN_EXP_PROBS_B,
        // "bias", N) resolves LLM_TENSOR_FFN_EXP_PROBS_B's pattern
        // "blk.%d.exp_probs_b" then appends ".bias" (confirmed against
        // llama-arch.cpp:407 + LLM_TN_IMPL::str()). This used to look for
        // "ffn_gate_inp.e_score_correction_bias" — a name that doesn't
        // exist in any real GGUF, so this tensor was NEVER actually
        // loaded (explains the "1106/1182 tensors found" gap — 76
        // missing = exactly the MoE layer count). The router code
        // already handles the bias tensor being absent gracefully (falls
        // back to unbiased selection), so this wasn't a crash, just
        // silently wrong/incomplete routing for models that do ship it.
        // Not loaded for hash-routed layers -- selection bias doesn't
        // apply when selection is a direct lookup, not argsort.
        dense_preload_manifest.push_back(p + "exp_probs_b.bias");
      }
      // Shared expert
      dense_preload_manifest.push_back(p + "ffn_gate_shexp.weight");
      dense_preload_manifest.push_back(p + "ffn_up_shexp.weight");
      dense_preload_manifest.push_back(p + "ffn_down_shexp.weight");
      if (fam == ModelFamily::Qwen4Exp) {
        // Qwen4Exp's shared expert has its OWN per-token sigmoid gate
        // (ffn_gate_inp_shexp, a [n_embd] vector -- qwen4exp.cpp's
        // build_layer_ffn), unlike GLM/DeepSeek's ungated shexp.
        dense_preload_manifest.push_back(p + "ffn_gate_inp_shexp.weight");
      }
      // Packed routed experts (streamed from disk, NOT preloaded here)
      // ffn_gate_exps, ffn_up_exps, ffn_down_exps are loaded on-demand

      engine.layers_experts[l].resize(std::max(1, c.n_experts));
      for (int e = 0; e < c.n_experts; ++e) {
        engine.layers_experts[l][e].expert_id = -1;
        engine.layers_experts[l][e].slab_buffer = nullptr;
        engine.layers_experts[l][e].slab_capacity = 0;
      }
    } else {
      // Dense FFN
      dense_preload_manifest.push_back(p + "ffn_gate.weight");
      dense_preload_manifest.push_back(p + "ffn_up.weight");
      dense_preload_manifest.push_back(p + "ffn_down.weight");
    }
  }

  if (mtp_active && fam != ModelFamily::Qwen4Exp) {
    const int l = c.n_layers;
    const std::string p = "blk." + std::to_string(l) + ".";
    // The GLM reference MTP block reuses the target token embedding/output
    // head when the optional nextn copies are absent.  Keep that fallback so
    // enabling MTP does not duplicate a vocabulary-sized tensor in VRAM.
    dense_preload_manifest.push_back(p + "nextn.eh_proj.weight");
    dense_preload_manifest.push_back(p + "nextn.enorm.weight");
    dense_preload_manifest.push_back(p + "nextn.hnorm.weight");
    if (file.tensors.find(p + "nextn.embed_tokens.weight") != file.tensors.end())
      dense_preload_manifest.push_back(p + "nextn.embed_tokens.weight");
    if (file.tensors.find(p + "nextn.shared_head_norm.weight") != file.tensors.end())
      dense_preload_manifest.push_back(p + "nextn.shared_head_norm.weight");
    if (file.tensors.find(p + "nextn.shared_head_head.weight") != file.tensors.end())
      dense_preload_manifest.push_back(p + "nextn.shared_head_head.weight");
    dense_preload_manifest.push_back(p + "attn_norm.weight");
    dense_preload_manifest.push_back(p + "attn_q_a.weight");
    dense_preload_manifest.push_back(p + "attn_q_a_norm.weight");
    dense_preload_manifest.push_back(p + "attn_q_b.weight");
    dense_preload_manifest.push_back(p + "attn_kv_a_mqa.weight");
    dense_preload_manifest.push_back(p + "attn_kv_a_norm.weight");
    dense_preload_manifest.push_back(p + "attn_k_b.weight");
    dense_preload_manifest.push_back(p + "attn_v_b.weight");
    dense_preload_manifest.push_back(p + "attn_output.weight");
    dense_preload_manifest.push_back(p + "ffn_norm.weight");
    dense_preload_manifest.push_back(p + "ffn_gate_inp.weight");
    dense_preload_manifest.push_back(p + "exp_probs_b.bias");
    dense_preload_manifest.push_back(p + "ffn_gate_shexp.weight");
    dense_preload_manifest.push_back(p + "ffn_up_shexp.weight");
    dense_preload_manifest.push_back(p + "ffn_down_shexp.weight");
    engine.layers_experts[l].resize(std::max(1, c.n_experts));
    for (int e = 0; e < c.n_experts; ++e) {
      engine.layers_experts[l][e].expert_id = -1;
      engine.layers_experts[l][e].slab_buffer = nullptr;
      engine.layers_experts[l][e].slab_capacity = 0;
    }
  }

  if (mtp_active && fam == ModelFamily::Qwen4Exp) {
    // Qwen4Exp's shared MTP checkpoint is small and self-contained (32
    // tensors, ~1.9 GB at Q4_K_M) by design -- resident-load ALL of it,
    // including the packed routed-expert tensors, from the SEPARATE
    // engine.mtp_model_file rather than routing layer 48 through the
    // ordinary disk-streaming/RAM-VRAM LRU cache machinery that every
    // other layer uses (that machinery is hardwired to engine.model_file
    // throughout ram_manager.hpp/async_prefetcher.hpp/engine.hpp -- see
    // this session's own investigation notes). qwen4exp_mtp_forward
    // (execution_layer.cpp) reads the packed experts directly out of
    // engine.resident_layers by name, slicing per-expert offsets with the
    // SAME size_bytes/n_experts convention as packed_slab_geometry()
    // (engine.hpp) -- verified against a real tensor dump (per-expert
    // gate/up/down byte counts match the dense shexp sibling tensors'
    // own byte counts exactly, confirming no inter-expert padding).
    const int l = c.n_layers;
    const std::string p = "blk." + std::to_string(l) + ".";
    std::vector<std::string> qwen4exp_nextn_manifest = {
        p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight",
        p + "attn_output.weight", p + "attn_q_norm.weight",
        p + "attn_k_norm.weight",
        p + "hc_attn_norm.weight", p + "hc_attn_down.weight",
        p + "hc_attn_up.weight", p + "hc_attn_inject.weight",
        p + "hc_ffn_norm.weight", p + "hc_ffn_down.weight",
        p + "hc_ffn_up.weight", p + "hc_ffn_inject.weight",
        p + "ffn_gate_inp.weight", p + "ffn_gate_inp_shexp.weight",
        p + "ffn_gate_shexp.weight", p + "ffn_up_shexp.weight",
        p + "ffn_down_shexp.weight",
        p + "ffn_gate_exps.weight", p + "ffn_up_exps.weight",
        p + "ffn_down_exps.weight",
        p + "nextn.eh_proj.weight", p + "nextn.enorm.weight",
        p + "nextn.hnorm.weight", p + "nextn.hc_head_norm.weight",
        p + "nextn.hc_head_down.weight", p + "nextn.hc_head_up.weight",
    };
    // Optional per the reference (falls back to the base model's own
    // token_embd.weight/output.weight -- both already resident from
    // dense_preload_manifest below).
    GGUFFile &mf = engine.mtp_model_file;
    if (mf.tensors.find(p + "nextn.embed_tokens.weight") != mf.tensors.end())
      qwen4exp_nextn_manifest.push_back(p + "nextn.embed_tokens.weight");
    if (mf.tensors.find(p + "nextn.shared_head_head.weight") != mf.tensors.end())
      qwen4exp_nextn_manifest.push_back(p + "nextn.shared_head_head.weight");
    engine.initialize_resident_memory(qwen4exp_nextn_manifest, &engine.mtp_model_file);
    // Not resized/used for streaming (no disk fetch ever targets layer
    // c.n_layers for this family -- see qwen4exp_mtp_forward), but kept at
    // size 1 rather than left default-constructed so any generic
    // "iterate every layer in [0, cache_layers)" loop elsewhere still
    // finds a valid (if unused) slot instead of an out-of-bounds access.
    engine.layers_experts[l].resize(1);
    engine.layers_experts[l][0].expert_id = -1;
    engine.layers_experts[l][0].slab_buffer = nullptr;
    engine.layers_experts[l][0].slab_capacity = 0;
  }

  if (fam == ModelFamily::GLM5NEXT && glm5next_dsa_layers > 0) {
    const int indexer_tensor_count = static_cast<int>(
        sizeof(kGlm5nextIndexerSuffixes) / sizeof(kGlm5nextIndexerSuffixes[0]));
    const int expected = glm5next_dsa_layers * indexer_tensor_count;
    const int dense_until = c.glm5next_indexer_top_k > 0 &&
                                    c.glm5next_indexer_kpool > 0
                                ? c.glm5next_indexer_top_k +
                                      c.glm5next_indexer_kpool - 1
                                : 0;
    LOG_STREAM_INFO()
        << "[GLM5NEXT] Lightning indexer tensors present: "
        << glm5next_indexer_tensors_present << "/" << expected << " across "
        << glm5next_dsa_layers
        << " DSA layer(s); not preloaded because v1 uses dense DSA attention"
        << (dense_until > 0
                ? " (exact for context <= " + std::to_string(dense_until) +
                      " tokens)"
                : "")
        << "\n";
  }

  // Global tensors
  dense_preload_manifest.push_back("token_embd.weight");
  // qwen4exp has NO output_norm -- the final low-rank HC mixer
  // (output_hc_norm/down/up) replaces it (qwen4exp.cpp's graph tail).
  if (fam != ModelFamily::Qwen4Exp)
    dense_preload_manifest.push_back("output_norm.weight");
  dense_preload_manifest.push_back("output.weight");
  // Hyper-Connections (DeepSeek-V4): the top-level "collapse hc_mult
  // streams back to one" weights, used once at model output (hc_collapse_
  // output, hyper_connections.cpp) -- distinct from the per-layer
  // hc_attn_*/hc_ffn_* tensors pushed above. Inactive (c.hc_mult<=1) means
  // these names are simply never added. GLM5Next is explicitly excluded:
  // its final collapse is an unweighted hc_mean (glm5next.cpp's graph
  // constructor), so these tensors don't exist in a glm5next file.
  // Qwen4Exp likewise has its OWN low-rank output mixer instead.
  if (c.hc_mult > 1 && fam != ModelFamily::GLM5NEXT &&
      fam != ModelFamily::DeepSeek_V41 &&
      fam != ModelFamily::Qwen4Exp) {
    dense_preload_manifest.push_back("output_hc_fn.weight");
    dense_preload_manifest.push_back("output_hc_base.weight");
    dense_preload_manifest.push_back("output_hc_scale.weight");
  }
  if (fam == ModelFamily::Qwen4Exp) {
    dense_preload_manifest.push_back("output_hc_norm.weight");
    dense_preload_manifest.push_back("output_hc_down.weight");
    dense_preload_manifest.push_back("output_hc_up.weight");
  }

  // Report and load
  int found = 0;
  for (const auto &name : dense_preload_manifest) {
    if (file.tensors.find(name) != file.tensors.end())
      ++found;
  }
  LOG_STREAM_INFO() << "[GRAPH] Dense manifest: " << found << " / "
                    << dense_preload_manifest.size()
                    << " tensors found in GGUF\n";
  if (found == 0) {
    LOG_STREAM_WARN()
        << "[GRAPH] WARNING: No manifest tensors matched. First 5 expected:\n";
    for (size_t i = 0; i < std::min((size_t)5, dense_preload_manifest.size());
         ++i)
      LOG_STREAM_WARN() << "    " << dense_preload_manifest[i] << "\n";
  }

  engine.initialize_resident_memory(dense_preload_manifest);

  verify_float_reinterpret_tensors(engine);
  verify_tensor_dimensions(engine);
  report_dense_quant_coverage(engine);
  return true;
}
