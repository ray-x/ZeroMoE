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

int cmd_verify(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr,
            "Usage: %s <model_file_or_dir> [--against <other_model>]\n"
            "\n"
            "Checks a GGUF model is complete and engine-loadable:\n"
            "  - every per-layer attention/norm tensor must be present in ALL\n"
            "    layers (a layer missing e.g. attn_k_b zeroes attention in the\n"
            "    engine -> garbage output). Missing -> non-zero exit.\n"
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
    // skips). This is exactly the "blk.63.attn_k_b missing" failure mode.
    int hard_missing = 0;
    for (auto &kv : suffix_layers) {
      const std::string &suf = kv.first;
      const std::set<int> &layers = kv.second;
      if (!is_attention_or_norm(suf))
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
