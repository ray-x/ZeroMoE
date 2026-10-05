// tools/gguf_verify.cpp
//
// gguf_tool verify: pre-flight completeness + (optional) byte-comparison
// check for a GGUF model. This is the tool that would have caught the
// "incomplete repack" failure mode (a shard not repacked/present ->
// attention tensors like blk.63.attn_k_b missing -> the engine zeroes
// attention -> garbage output): io_bench only looks at expert tensors, and
// repack_gguf's own verify checks tensors per-shard-it-processed, so NEITHER
// asserts the full model directory is complete and engine-loadable. This does.
//
// Completeness check (no architecture KV parsing needed -- pure tensor-name
// consistency): for every per-layer tensor-name suffix, the suffix must appear
// under EVERY blk.<L>. Attention + norm tensors are hard-required in every
// layer (a layer missing attn_k_b is always a real error -> non-zero exit);
// FFN expert tensors are reported (packed *_exps vs per-expert, n_experts,
// per-layer expert counts) but a layer legitimately lacking experts (a leading
// dense layer) is noted, not failed.
//
// --against <other_model>: exhaustive byte-compare of every tensor common to
// both models (the cross-check io_bench's single-expert spot-check never was).
//
// Build: see gguf_common.hpp (compiled together with the other gguf_*.cpp).
#include "gguf_common.hpp"
#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <unistd.h>
#include <vector>

namespace gguf_tool {

// "blk.<L>.<suffix>" -> (L, suffix=everything after "blk.<L>.").
// Uses the shared parse_blk_layer for the leading "blk.<L>." part.
static bool parse_blk_suffix(const std::string &name, int &L,
                             std::string &suffix) {
  if (!parse_blk_layer(name, L))
    return false;
  size_t p = 4; // skip "blk."
  while (p < name.size() && name[p] >= '0' && name[p] <= '9')
    ++p;
  if (p < name.size() && name[p] == '.')
    ++p; // skip the '.' after the layer number
  suffix = name.substr(p);
  return true;
}

static bool is_attention_or_norm(const std::string &suffix) {
  // hc_ catches DeepSeek-V4's per-layer Hyper-Connections weights
  // (hc_attn_fn/base/scale, hc_ffn_fn/base/scale) -- hc_attn_* already
  // matched the "attn" substring below by accident, but hc_ffn_* matched
  // neither "attn" nor "norm" and was silently NOT hard-required per
  // layer, exactly the class of gap this check exists to catch (a shard
  // that dropped one of these residual-stream-critical weights on some
  // layer would have passed verify clean). See src/main.cpp's
  // is_dense_offload_candidate for the same "hc_" rationale.
  return suffix.find("attn") != std::string::npos ||
         suffix.find("norm") != std::string::npos ||
         suffix.find("hc_") != std::string::npos;
}

static bool is_qwen4exp_mixed_attention_suffix(const std::string &suffix) {
  return suffix == "attn_gate.weight" ||
         suffix == "attn_k.weight" ||
         suffix == "attn_k_norm.weight" ||
         suffix == "attn_output.weight" ||
         suffix == "attn_q.weight" ||
         suffix == "attn_q_norm.weight" ||
         suffix == "attn_qkv.weight" ||
         suffix == "attn_v.weight" ||
         suffix == "indexer.k_norm.weight" ||
         suffix == "indexer.q_norm.weight" ||
         suffix == "ssm_norm.weight";
}

static bool qwen4exp_has_tensor(const GGUFFile &M, int layer,
                                const char *suffix) {
  return M.tensors.find("blk." + std::to_string(layer) + "." + suffix) !=
         M.tensors.end();
}

static int verify_qwen4exp_mixed_attention(const GGUFFile &M, int min_layer,
                                           int max_layer) {
  int missing = 0;
  int gdn_layers = 0;
  int full_attention_layers = 0;
  for (int layer = min_layer; layer <= max_layer; ++layer) {
    const bool has_qkv = qwen4exp_has_tensor(M, layer, "attn_qkv.weight");
    const bool has_q = qwen4exp_has_tensor(M, layer, "attn_q.weight");
    const bool has_k = qwen4exp_has_tensor(M, layer, "attn_k.weight");
    const bool has_v = qwen4exp_has_tensor(M, layer, "attn_v.weight");
    const bool has_separate_qkv = has_q && has_k && has_v;
    const std::string prefix = "blk." + std::to_string(layer) + ".";

    if (has_qkv && has_separate_qkv) {
      printf("[verify] INVALID qwen4exp layer %d: both attn_qkv and "
             "separate attn_q/k/v tensors are present\n",
             layer);
      ++missing;
      continue;
    }

    if (has_qkv) {
      ++gdn_layers;
      for (const char *suffix : {"attn_gate.weight", "ssm_norm.weight"}) {
        if (!qwen4exp_has_tensor(M, layer, suffix)) {
          printf("[verify] MISSING %s%s on qwen4exp GDN layer %d\n",
                 prefix.c_str(), suffix, layer);
          ++missing;
        }
      }
      continue;
    }

    if (has_separate_qkv) {
      ++full_attention_layers;
      for (const char *suffix : {"attn_k_norm.weight",
                                 "attn_output.weight",
                                 "attn_q_norm.weight",
                                 "indexer.k_norm.weight",
                                 "indexer.q_norm.weight"}) {
        if (!qwen4exp_has_tensor(M, layer, suffix)) {
          printf("[verify] MISSING %s%s on qwen4exp full-attention layer %d\n",
                 prefix.c_str(), suffix, layer);
          ++missing;
        }
      }
      continue;
    }

    printf("[verify] INVALID qwen4exp layer %d: neither attn_qkv nor a "
           "complete attn_q/k/v set is present\n",
           layer);
    ++missing;
  }

  printf("[verify] qwen4exp mixed attention layout: %d GDN layer(s), "
         "%d full-attention layer(s)\n",
         gdn_layers, full_attention_layers);
  return missing;
}

static bool is_deepseek41_sparse_tensor(const std::string &suffix) {
  // DeepSeek-V4.1 publishes compressed-KV and indexer weights only on the
  // source layers named by attention.kv_source_layers and
  // attention.index_source_layers. They are deliberately not repeated for
  // every transformer block, so the generic all-layers check must not treat
  // their absence as a truncated shard.
  return suffix.rfind("attn_compressor_", 0) == 0 ||
         suffix.rfind("indexer.", 0) == 0;
}

static bool verify_deepseek41_engram(const GGUFFile &M) {
  const auto arch_it = M.kv_string.find("general.architecture");
  if (arch_it == M.kv_string.end() || arch_it->second != "deepseek41")
    return true;

  const std::string prefix = "deepseek41.engram.";
  auto array_size = [&M, &prefix](const char *name) -> size_t {
    auto it = M.kv_arr_int.find(prefix + name);
    return it == M.kv_arr_int.end() ? 0 : it->second.size();
  };
  const int max_ngram = M.get_config_int("engram.max_ngram_size", 0);
  const int heads = M.get_config_int("engram.head_count", 0);
  const int key_length = M.get_config_int("engram.key_length", 0);
  const size_t n_layers = array_size("layer_ids");
  const size_t n_hash_cols = max_ngram >= 2 && heads > 0
                                 ? (size_t)(max_ngram - 1) * heads
                                 : 0;
  const bool metadata_ok =
      n_layers > 0 && max_ngram >= 2 && heads > 0 && key_length > 0 &&
      (array_size("num_embeddings") == 0 ||
       array_size("num_embeddings") == n_layers) &&
      array_size("primes") == n_layers * n_hash_cols &&
      array_size("offsets") == n_layers * n_hash_cols &&
      array_size("multipliers") == n_layers * (size_t)max_ngram &&
      array_size("token_map") > 0;
  if (!metadata_ok) {
    printf("[verify] MISSING/INCOMPLETE deepseek41 Engram metadata "
           "(conditional memory hash layout)\n");
    return false;
  }

  const auto &layer_ids = M.kv_arr_int.at(prefix + "layer_ids");
  const auto num_embeddings_it = M.kv_arr_int.find(prefix + "num_embeddings");
  bool tensors_ok = true;
  for (size_t i = 0; i < n_layers; ++i) {
    const int layer = (int)layer_ids[i];
    const std::string base = "blk." + std::to_string(layer) + ".";
    auto table = M.tensors.find(base + "engram_embd.weight");
    auto q = M.tensors.find(base + "engram_q.weight");
    auto k = M.tensors.find(base + "engram_k.weight");
    auto wkv = M.tensors.find(base + "engram_wkv.weight");
    if (table == M.tensors.end() || q == M.tensors.end() ||
        k == M.tensors.end() || wkv == M.tensors.end() ||
        table->second.shape.size() != 2 ||
        table->second.shape[0] != (uint64_t)key_length ||
        (num_embeddings_it != M.kv_arr_int.end() &&
         table->second.shape[1] !=
             (uint64_t)num_embeddings_it->second[i])) {
      printf("[verify] MISSING/INVALID Engram tensors at blk.%d\n", layer);
      tensors_ok = false;
    }
  }
  if (!tensors_ok)
    return false;
  printf("[verify] DeepSeek-V4.1 Engram metadata/tensors complete "
         "(%zu layer tables, %d n-gram limit, %d heads)\n",
         n_layers, max_ngram, heads);
  return true;
}

int cmd_verify(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr,
            "Usage: %s <model_file_or_dir> [--against <other_model>]\n"
            "\n"
            "Checks a GGUF model is complete and engine-loadable:\n"
            "  - every per-layer attention/norm tensor must be present in ALL\n"
            "    layers (a layer missing e.g. attn_k_b zeroes attention in the\n"
            "    engine -> garbage output). Missing -> non-zero exit.\n"
            "    qwen4exp mixed GDN/full-attention layouts are checked per layer.\n"
            "  - FFN experts reported (packed *_exps vs per-expert, n_experts,\n"
            "    per-layer counts); a layer legitimately lacking experts (a\n"
            "    leading dense layer) is noted, not failed.\n"
            "  - --against <other>: exhaustive byte-compare of every tensor\n"
            "    common to both models.\n",
            argv[0]);
    return 2;
  }
  std::string model = argv[1];
  std::string against;
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--against" && i + 1 < argc)
      against = argv[++i];
    else {
      fprintf(stderr, "[verify] unknown arg: %s\n", a.c_str());
      return 2;
    }
  }

  GGUFFile M;
  if (!M.load_header(model)) {
    fprintf(stderr, "[verify] FAIL: cannot load model %s\n", model.c_str());
    return 1;
  }
  printf("[verify] Loaded %zu tensor(s) across %zu shard(s) from %s\n",
         M.tensors.size(), M.shard_paths.size(), model.c_str());

  bool incomplete = false;

  if (!verify_deepseek41_engram(M))
    incomplete = true;

  // ---- per-suffix layer-consistency completeness check ----
  std::map<std::string, std::set<int>> suffix_layers; // suffix -> layers present
  int min_layer = INT_MAX, max_layer = -1;
  bool has_packed = false, has_per_expert = false;
  int n_experts_packed = -1;
  std::map<int, int> per_expert_gate_count; // layer -> # of ffn_gate.<E>.weight
  for (auto &kv : M.tensors) {
    const std::string &name = kv.first;
    const GGUFTensor &t = kv.second;
    int L;
    std::string suf;
    if (!parse_blk_suffix(name, L, suf))
      continue;
    suffix_layers[suf].insert(L);
    min_layer = std::min(min_layer, L);
    max_layer = std::max(max_layer, L);
    if (suf == "ffn_gate_exps.weight") {
      has_packed = true;
      if (!t.shape.empty())
        n_experts_packed = (int)t.shape.back();
    }
    // per-expert gate tensor: "ffn_gate.<E>.weight" (not the packed _exps)
    if (suf.rfind("ffn_gate.", 0) == 0 && suf != "ffn_gate_exps.weight" &&
        suf.size() > 7 && suf.compare(suf.size() - 7, 7, ".weight") == 0) {
      has_per_expert = true;
      per_expert_gate_count[L]++;
    }
  }

  if (max_layer < 0) {
    printf("[verify] No blk.* tensors found -- not a (transformer) GGUF, or "
           "wrong file/dir.\n");
  } else {
    int layers_in_range = max_layer - min_layer + 1;
    std::string fmt = has_packed     ? "packed *_exps"
                      : has_per_expert ? "per-expert"
                                       : "(no expert tensors found)";
    if (n_experts_packed > 0)
      fmt += " (n_experts=" + std::to_string(n_experts_packed) + ")";
    printf("[verify] Layers blk.%d..blk.%d (%d total). Expert format: %s\n",
           min_layer, max_layer, layers_in_range, fmt.c_str());

    // (1) Attention + norm tensors: HARD-required in every layer. A layer
    // missing one is always a real error (the engine zeroes attention /
    // skips). Qwen4Exp is the exception to the one-schema rule: its GDN and
    // full-attention layers intentionally use different tensor families.
    int hard_missing = 0;
    const auto arch_it = M.kv_string.find("general.architecture");
    const bool is_qwen4exp =
        arch_it != M.kv_string.end() && arch_it->second == "qwen4exp";
    for (auto &kv : suffix_layers) {
      const std::string &suf = kv.first;
      const std::set<int> &layers = kv.second;
      if (!is_attention_or_norm(suf))
        continue;
      if (arch_it != M.kv_string.end() && arch_it->second == "deepseek41" &&
          is_deepseek41_sparse_tensor(suf))
        continue;
      if (is_qwen4exp && is_qwen4exp_mixed_attention_suffix(suf))
        continue;
      if ((int)layers.size() < 2)
        continue; // not a per-layer tensor
      std::vector<int> missing;
      for (int L = min_layer; L <= max_layer; ++L)
        if (!layers.count(L))
          missing.push_back(L);
      if (!missing.empty()) {
        hard_missing += (int)missing.size();
        printf("[verify] MISSING blk.<L>.%s in %zu/%d layer(s):", suf.c_str(),
               missing.size(), layers_in_range);
        for (int L : missing)
          printf(" %d", L);
        printf("  (present in %d/%d)\n", (int)layers.size(), layers_in_range);
      }
    }
    if (is_qwen4exp)
      hard_missing +=
          verify_qwen4exp_mixed_attention(M, min_layer, max_layer);

    // (2) FFN expert reporting (informational; dense layers legitimately
    // lack experts, so not hard-failed).
    if (has_per_expert && !has_packed) {
      int inferred = 0;
      for (auto &kv : per_expert_gate_count)
        inferred = std::max(inferred, kv.second);
      if (inferred > 0) {
        for (auto &kv : per_expert_gate_count) {
          if (kv.second != inferred)
            printf("[verify] NOTE blk.%d has %d per-expert gate tensors "
                   "(other layers have %d) -- short expert set\n",
                   kv.first, kv.second, inferred);
        }
      }
    }

    if (hard_missing == 0) {
      printf("[verify] ATTENTION/NORM COMPLETE -- every per-layer attention + "
             "norm tensor is present in all %d layers.\n",
             layers_in_range);
    } else {
      printf("[verify] ATTENTION/NORM INCOMPLETE -- %d missing tensor(s) above "
             "(engine would zero attention / skip on these layers).\n",
             hard_missing);
      incomplete = true;
    }
  }

  // ---- optional exhaustive byte-compare against another model ----
  if (!against.empty()) {
    GGUFFile A;
    if (!A.load_header(against)) {
      fprintf(stderr, "[verify] FAIL: cannot load --against model %s\n",
              against.c_str());
      return 1;
    }
    printf("[verify] Byte-comparing every common tensor against %s ...\n",
           against.c_str());
    size_t max_size = 0;
    for (auto &kv : M.tensors)
      max_size = std::max(max_size, (size_t)kv.second.size_bytes);
    for (auto &kv : A.tensors)
      max_size = std::max(max_size, (size_t)kv.second.size_bytes);
    std::vector<uint8_t> b1(max_size), b2(max_size);
    int common = 0, only_m = 0, only_a = 0, size_mismatch = 0, byte_mismatch = 0;
    for (auto &kv : M.tensors) {
      const std::string &n = kv.first;
      const GGUFTensor &t = kv.second;
      auto it = A.tensors.find(n);
      if (it == A.tensors.end()) {
        only_m++;
        continue;
      }
      const GGUFTensor &ta = it->second;
      common++;
      if (t.size_bytes != ta.size_bytes) {
        size_mismatch++;
        if (size_mismatch <= 10)
          printf("[verify] SIZE mismatch %s: %llu vs %llu\n", n.c_str(),
                 (unsigned long long)t.size_bytes,
                 (unsigned long long)ta.size_bytes);
        continue;
      }
      ssize_t r1 = pread(M.fd_for_tensor(t), b1.data(), t.size_bytes,
                         (off_t)t.offset);
      ssize_t r2 = pread(A.fd_for_tensor(ta), b2.data(), ta.size_bytes,
                         (off_t)ta.offset);
      if (r1 != (ssize_t)t.size_bytes || r2 != (ssize_t)ta.size_bytes) {
        byte_mismatch++;
        continue;
      }
      if (std::memcmp(b1.data(), b2.data(), t.size_bytes) != 0) {
        byte_mismatch++;
        if (byte_mismatch <= 10)
          printf("[verify] BYTE mismatch %s (%llu bytes)\n", n.c_str(),
                 (unsigned long long)t.size_bytes);
      }
    }
    for (auto &kv : A.tensors)
      if (!M.tensors.count(kv.first))
        only_a++;
    printf("[verify] --against: %d common tensors, %d byte-identical, %d "
           "size-mismatch, %d byte-mismatch; %d only-in-model, %d only-in-against\n",
           common, common - size_mismatch - byte_mismatch, size_mismatch,
           byte_mismatch, only_m, only_a);
    if (size_mismatch > 0 || byte_mismatch > 0)
      incomplete = true;
  }

  if (incomplete) {
    printf("[verify] RESULT: FAIL (model incomplete or bytes differ).\n");
    return 1;
  }
  printf("[verify] RESULT: OK\n");
  return 0;
}

} // namespace gguf_tool
