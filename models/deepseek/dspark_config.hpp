// dspark_config.hpp -- metadata and topology of a DSpark (DFlash) speculative-decoding
// draft GGUF for DeepSeek-V4.1 Flash. Parsing and validation only: no allocation of
// weights. See docs/design/dspark-support.md. Reference: llama.cpp
// src/models/dflash.cpp (graph_dsv4, build_dspark_markov_head).
#pragma once
#include "gguf_loader.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace zeromoe::dspark {

struct Config {
  int n_layer = 0;            // dflash.block_count (draft stages)
  int n_embd = 0;             // hidden width (== target hidden width)
  int n_head = 0, n_head_kv = 0;
  int key_len = 0, value_len = 0, rope_dim = 0;
  float rope_base = 0.0f, rms_eps = 0.0f;
  int q_lora = 0, swa = 0;    // q low-rank width, sliding-window (draft ring) length
  int n_expert = 0, n_expert_used = 0, n_expert_shared = 0, ffn_exp = 0;
  int gating_func = 0;        // 4 == sqrt-softplus
  float expert_scale = 1.0f;
  bool expert_norm = false;
  int o_groups = 0, o_rank = 0;
  int hc_count = 0, hc_sinkhorn = 0;
  float hc_eps = 0.0f;
  bool hc_single_pass = false;
  int block_size = 0;         // tokens drafted per step (anchor + predictions)
  bool sample_from_anchor = false, has_confidence_head = false;
  std::vector<int> target_layers;  // target layers whose hidden states feed the draft
  // YaRN rope scaling (present in the Lucebox draft; absent in the first BF16 file => plain rope).
  bool rope_yarn = false;
  float rope_yarn_factor = 1.0f, rope_yarn_beta_fast = 32.0f, rope_yarn_beta_slow = 1.0f;
  int rope_orig_ctx = 0;
  bool dsv41_semantics = false;  // dflash.dsv41_semantics (Lucebox draft)
  int markov_rank = 0;        // rank of markov_w1/w2
  int n_vocab = 0;            // vocabulary rows of the Markov tables (== target vocab)
};

namespace detail {
inline bool get_num(const GGUFFile &f, const std::string &key, double &out) {
  if (auto it = f.kv_uint.find(key); it != f.kv_uint.end()) { out = (double)it->second; return true; }
  if (auto it = f.kv_int.find(key); it != f.kv_int.end()) { out = (double)it->second; return true; }
  if (auto it = f.kv_float.find(key); it != f.kv_float.end()) { out = (double)it->second; return true; }
  return false;
}
}  // namespace detail

// Parses dflash.* metadata. Returns false and fills `err` on a missing or invalid key.
inline bool parse_config(const GGUFFile &f, Config &c, std::string &err) {
  auto arch = f.kv_string.find("general.architecture");
  if (arch == f.kv_string.end() || arch->second != "dflash") {
    err = "general.architecture is not 'dflash'";
    return false;
  }
  auto need = [&](const char *suffix, double &v) {
    if (detail::get_num(f, std::string("dflash.") + suffix, v)) return true;
    err = std::string("missing dflash.") + suffix;
    return false;
  };
  double v = 0;
#define DSPARK_INT(field, key) \
  do { if (!need(key, v)) return false; c.field = (int)v; } while (0)
#define DSPARK_FLOAT(field, key) \
  do { if (!need(key, v)) return false; c.field = (float)v; } while (0)
#define DSPARK_BOOL(field, key) \
  do { if (!need(key, v)) return false; c.field = v != 0.0; } while (0)
  DSPARK_INT(n_layer, "block_count");
  DSPARK_INT(n_embd, "embedding_length");
  DSPARK_INT(n_head, "attention.head_count");
  DSPARK_INT(n_head_kv, "attention.head_count_kv");
  DSPARK_INT(key_len, "attention.key_length");
  DSPARK_INT(value_len, "attention.value_length");
  DSPARK_INT(rope_dim, "rope.dimension_count");
  DSPARK_FLOAT(rope_base, "rope.freq_base");
  DSPARK_FLOAT(rms_eps, "attention.layer_norm_rms_epsilon");
  DSPARK_INT(q_lora, "attention.q_lora_rank");
  DSPARK_INT(swa, "attention.sliding_window");
  DSPARK_INT(n_expert, "expert_count");
  DSPARK_INT(n_expert_used, "expert_used_count");
  DSPARK_INT(n_expert_shared, "expert_shared_count");
  DSPARK_INT(ffn_exp, "expert_feed_forward_length");
  DSPARK_INT(gating_func, "expert_gating_func");
  DSPARK_FLOAT(expert_scale, "expert_weights_scale");
  DSPARK_BOOL(expert_norm, "expert_weights_norm");
  DSPARK_INT(o_groups, "attention.output_group_count");
  DSPARK_INT(o_rank, "attention.output_lora_rank");
  DSPARK_INT(hc_count, "hyper_connection.count");
  DSPARK_INT(hc_sinkhorn, "hyper_connection.sinkhorn_iterations");
  DSPARK_FLOAT(hc_eps, "hyper_connection.epsilon");
  DSPARK_BOOL(hc_single_pass, "hyper_connection.single_pass");
  DSPARK_INT(block_size, "block_size");
  // Optional keys: the llama.cpp reference defaults sample_from_anchor to true when absent,
  // and the confidence head is present exactly when conf_proj.weight is (checked below).
  c.sample_from_anchor = true;
  if (detail::get_num(f, "dflash.sample_from_anchor", v)) c.sample_from_anchor = v != 0.0;
  c.has_confidence_head = f.tensors.find("conf_proj.weight") != f.tensors.end();
  if (detail::get_num(f, "dflash.has_confidence_head", v)) c.has_confidence_head = v != 0.0;
  if (detail::get_num(f, "dflash.dsv41_semantics", v)) c.dsv41_semantics = v != 0.0;
  if (auto it = f.kv_string.find("dflash.rope.scaling.type"); it != f.kv_string.end() && it->second == "yarn") {
    c.rope_yarn = true;
    if (detail::get_num(f, "dflash.rope.scaling.factor", v)) c.rope_yarn_factor = (float)v;
    if (detail::get_num(f, "dflash.rope.scaling.yarn_beta_fast", v)) c.rope_yarn_beta_fast = (float)v;
    if (detail::get_num(f, "dflash.rope.scaling.yarn_beta_slow", v)) c.rope_yarn_beta_slow = (float)v;
    if (detail::get_num(f, "dflash.rope.scaling.original_context_length", v)) c.rope_orig_ctx = (int)v;
  }
#undef DSPARK_INT
#undef DSPARK_FLOAT
#undef DSPARK_BOOL
  auto layers = f.kv_arr_int.find("dflash.target_layers");
  if (layers == f.kv_arr_int.end() || layers->second.empty()) {
    err = "missing dflash.target_layers";
    return false;
  }
  c.target_layers.assign(layers->second.begin(), layers->second.end());
  auto ratios = f.kv_arr_int.find("dflash.attention.compress_ratios");
  if (ratios != f.kv_arr_int.end())
    for (int64_t r : ratios->second)
      if (r != 0) { err = "DSpark draft expects uncompressed attention on all stages"; return false; }
  if (c.gating_func != 4) { err = "DSpark draft expects sqrt-softplus MoE scoring (gating func 4)"; return false; }
  if (c.n_layer <= 0 || c.n_head <= 0 || c.n_expert <= 0 || c.block_size < 2 || c.swa <= 0 ||
      c.hc_count <= 0 || c.o_groups <= 0 || c.n_head % c.o_groups != 0 || c.rope_dim >= c.key_len) {
    err = "inconsistent dflash hyper-parameters";
    return false;
  }
  return true;
}

// Checks that every tensor the draft graph needs exists with the expected shape and a
// supported type, and fills the Markov rank / vocabulary from `markov_w1.weight`.
// GGUF dims are listed innermost first, as stored in GGUFTensor::shape.
inline bool validate_tensors(const GGUFFile &f, Config &c, std::string &err) {
  auto find = [&](const std::string &name, const std::vector<uint64_t> &shape,
                  const GGUFTensor *&t) {
    auto it = f.tensors.find(name);
    if (it == f.tensors.end()) { err = "missing tensor " + name; return false; }
    t = &it->second;
    if (t->shape != shape) {
      err = "tensor " + name + " has an unexpected shape";
      return false;
    }
    return true;
  };
  const uint64_t D = (uint64_t)c.n_embd, E = (uint64_t)c.n_expert, F = (uint64_t)c.ffn_exp;
  const uint64_t qh = (uint64_t)c.key_len, H = (uint64_t)c.n_head;
  const uint64_t og = (uint64_t)c.o_groups, orank = (uint64_t)c.o_rank;
  const uint64_t hc = (uint64_t)c.hc_count;
  const GGUFTensor *t = nullptr;
  for (int il = 0; il < c.n_layer; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    for (const char *n : {"ffn_gate_exps.weight", "ffn_up_exps.weight"})
      if (!find(p + n, {D, F, E}, t)) return false;
    if (!find(p + "ffn_down_exps.weight", {F, D, E}, t)) return false;
    if (!find(p + "attn_q_a.weight", {D, (uint64_t)c.q_lora}, t) ||
        !find(p + "attn_q_b.weight", {(uint64_t)c.q_lora, H * qh}, t) ||
        !find(p + "attn_kv.weight", {D, qh}, t) ||
        !find(p + "attn_output_a.weight", {H * qh / og, og * orank}, t) ||
        !find(p + "attn_output_b.weight", {og * orank, D}, t) ||
        !find(p + "attn_norm.weight", {D}, t) || !find(p + "ffn_norm.weight", {D}, t) ||
        !find(p + "attn_q_a_norm.weight", {(uint64_t)c.q_lora}, t) ||
        !find(p + "attn_kv_a_norm.weight", {qh}, t) || !find(p + "attn_sinks.weight", {H}, t) ||
        !find(p + "ffn_gate_inp.weight", {D, E}, t) || !find(p + "exp_probs_b.bias", {E}, t) ||
        !find(p + "ffn_gate_shexp.weight", {D, F}, t) || !find(p + "ffn_up_shexp.weight", {D, F}, t) ||
        !find(p + "ffn_down_shexp.weight", {F, D}, t))
      return false;
    for (const char *n : {"hc_attn_fn.weight", "hc_ffn_fn.weight"})
      if (!find(p + n, {hc * D, hc * (2 + hc)}, t)) return false;  // mix = pre(hc) + post(hc) + comb(hc*hc)
    for (const char *n : {"hc_attn_base.weight", "hc_ffn_base.weight"})
      if (!find(p + n, {hc * (2 + hc)}, t)) return false;
    for (const char *n : {"hc_attn_scale.weight", "hc_ffn_scale.weight"})
      if (!find(p + n, {3}, t)) return false;
  }
  const uint64_t nt = c.target_layers.size();
  if (!find("fc.weight", {nt * D, D}, t) || !find("enc.output_norm.weight", {D}, t) ||
      !find("output_norm.weight", {D}, t))
    return false;
  auto m1 = f.tensors.find("markov_w1.weight");
  if (m1 == f.tensors.end() || m1->second.shape.size() != 2) {
    err = "missing markov_w1.weight";
    return false;
  }
  c.markov_rank = (int)m1->second.shape[0];
  c.n_vocab = (int)m1->second.shape[1];
  if (!find("markov_w2.weight", {(uint64_t)c.markov_rank, (uint64_t)c.n_vocab}, t)) return false;
  if (c.has_confidence_head &&
      !find("conf_proj.weight", {D + (uint64_t)c.markov_rank, 1}, t))
    return false;
  return true;
}

}  // namespace zeromoe::dspark
