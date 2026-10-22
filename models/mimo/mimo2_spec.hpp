#pragma once

#include "gguf_loader.hpp"
#include <cmath>
#include <string>
#include <vector>

inline bool mimo2_attention_visible(int query, int key, int window, bool sliding) {
  return key >= 0 && key <= query && (!sliding || query - key < window);
}

inline int64_t mimo2_kv_row(int head, int capacity, int position) {
  return int64_t(head) * capacity + position;
}

// Load and validate the attention/expert geometry before allocating any caches.
// MiMo versions using this layout are metadata in one family, not new backends.
inline bool configure_mimo2(const GGUFFile &file, ArchitectureConfig &c,
                            std::string &reason) {
  reason.clear();
  auto fail = [&](const std::string &s) { reason = s; return false; };
  if (c.hidden <= 0 || c.n_layers <= 0 || c.n_heads <= 0 || c.head_dim <= 0 ||
      c.v_head <= 0 || c.n_experts <= 0 || c.topk <= 0 || c.topk > c.n_experts || c.moe_inter <= 0)
    return fail("invalid model dimensions or expert count");
  if (c.n_layer_nextn != 0)
    return fail("target-trunk integration requires a GGUF without NextN blocks");
  c.mimo2_head_count_kv = file.get_config_int_array("attention.head_count_kv");
  if (c.mimo2_head_count_kv.empty())
    c.mimo2_head_count_kv.assign(c.n_layers, c.n_head_kv);
  c.mimo2_sliding_window_pattern = file.get_config_int_array("attention.sliding_window_pattern");
  if (c.mimo2_head_count_kv.size() != size_t(c.n_layers) ||
      c.mimo2_sliding_window_pattern.size() != size_t(c.n_layers))
    return fail("KV-head and sliding-window arrays must cover every trunk layer");
  c.sliding_window = file.get_config_int("attention.sliding_window", 0);
  c.mimo2_rope_freq_base_swa = file.get_config_float("rope.freq_base_swa", 10000.0f);
  c.mimo2_value_scale = file.get_config_float("attention.value_scale", 0.0f);
  const int n_rot = file.get_config_int("rope.dimension_count", c.head_dim);
  if (n_rot <= 0 || n_rot > c.head_dim || n_rot % 2 ||
      !std::isfinite(c.mimo2_value_scale) || c.mimo2_value_scale < 0 ||
      !std::isfinite(c.mimo2_rope_freq_base_swa) || c.mimo2_rope_freq_base_swa <= 0)
    return fail("invalid partial RoPE or attention-value scaling");
  auto has = [&](const std::string &name) { return file.tensors.count(name) != 0; };
  auto shape = [&](const std::string &name, std::vector<uint64_t> expected) {
    const auto it = file.tensors.find(name);
    if (it == file.tensors.end() || it->second.shape != expected) {
      reason = "missing or incorrectly shaped tensor: " + name;
      return false;
    }
    return true;
  };
  bool saw_moe = false;
  c.first_dense = 0;
  for (int l = 0; l < c.n_layers; ++l) {
    const std::string p = "blk." + std::to_string(l) + ".";
    const int64_t h = c.mimo2_head_count_kv[l];
    const int64_t swa = c.mimo2_sliding_window_pattern[l];
    if (h <= 0 || h > c.n_heads || c.n_heads % h ||
        (swa != 0 && swa != 1) || (swa && c.sliding_window <= 0))
      return fail("invalid attention configuration at layer " + std::to_string(l));
    if (!shape(p + "attn_norm.weight", {uint64_t(c.hidden)}) ||
        !shape(p + "ffn_norm.weight", {uint64_t(c.hidden)}) ||
        !shape(p + "attn_output.weight", {uint64_t(c.n_heads) * c.v_head, uint64_t(c.hidden)}))
      return false;
    if (has(p + "attn_qkv.weight")) {
      if (!shape(p + "attn_qkv.weight", {uint64_t(c.hidden),
          uint64_t(c.n_heads) * c.head_dim + uint64_t(h) * (c.head_dim + c.v_head)})) return false;
    } else if (!shape(p + "attn_q.weight", {uint64_t(c.hidden), uint64_t(c.n_heads) * c.head_dim}) ||
               !shape(p + "attn_k.weight", {uint64_t(c.hidden), uint64_t(h) * c.head_dim}) ||
               !shape(p + "attn_v.weight", {uint64_t(c.hidden), uint64_t(h) * c.v_head})) return false;
    if (has(p + "attn_sinks.weight") &&
        !shape(p + "attn_sinks.weight", {uint64_t(c.n_heads)})) return false;
    if (has(p + "ffn_gate_inp.weight")) {
      saw_moe = true;
      if (!shape(p + "ffn_gate_inp.weight", {uint64_t(c.hidden), uint64_t(c.n_experts)}) ||
          !shape(p + "ffn_gate_exps.weight", {uint64_t(c.hidden), uint64_t(c.moe_inter), uint64_t(c.n_experts)}) ||
          !shape(p + "ffn_up_exps.weight", {uint64_t(c.hidden), uint64_t(c.moe_inter), uint64_t(c.n_experts)}) ||
          !shape(p + "ffn_down_exps.weight", {uint64_t(c.moe_inter), uint64_t(c.hidden), uint64_t(c.n_experts)})) return false;
      if (has(p + "exp_probs_b.bias") &&
          !shape(p + "exp_probs_b.bias", {uint64_t(c.n_experts)})) return false;
    } else {
      if (saw_moe) return fail("only leading dense FFN layers are supported");
      ++c.first_dense;
      if (c.dense_inter <= 0 ||
          !shape(p + "ffn_gate.weight", {uint64_t(c.hidden), uint64_t(c.dense_inter)}) ||
          !shape(p + "ffn_up.weight", {uint64_t(c.hidden), uint64_t(c.dense_inter)}) ||
          !shape(p + "ffn_down.weight", {uint64_t(c.dense_inter), uint64_t(c.hidden)})) return false;
    }
  }
  if (!saw_moe) return fail("no routed MoE layer found");
  c.n_head_kv = int(c.mimo2_head_count_kv.front());
  return true;
}
