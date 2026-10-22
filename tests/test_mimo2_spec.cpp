#include "mimo2_spec.hpp"
#include "model_family.hpp"
#include <cstdio>

static int failures = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); ++failures; } } while (0)

int main(int argc, char **argv) {
  GGUFFile file;
  file.kv_string["general.architecture"] = "mimo2";
  file.kv_arr_int["mimo2.attention.head_count_kv"] = {1, 2};
  file.kv_arr_int["mimo2.attention.sliding_window_pattern"] = {0, 1};
  file.kv_uint["mimo2.attention.sliding_window"] = 3;
  file.kv_uint["mimo2.rope.dimension_count"] = 2;
  auto tensor = [&](const std::string &name, std::vector<uint64_t> shape) {
    GGUFTensor t; t.name = name; t.type = GGUFType::F32; t.shape = shape;
    file.tensors.emplace(name, t);
  };
  for (int l = 0; l < 2; ++l) {
    const std::string p = "blk." + std::to_string(l) + ".";
    tensor(p + "attn_norm.weight", {8}); tensor(p + "ffn_norm.weight", {8});
    tensor(p + "attn_output.weight", {8, 8});
    tensor(p + "attn_qkv.weight", {8, uint64_t(16 + (l + 1) * 6)});
    if (!l) {
      tensor(p + "ffn_gate.weight", {8, 16}); tensor(p + "ffn_up.weight", {8, 16});
      tensor(p + "ffn_down.weight", {16, 8});
    } else {
      tensor(p + "ffn_gate_inp.weight", {8, 4});
      tensor(p + "ffn_gate_exps.weight", {8, 8, 4});
      tensor(p + "ffn_up_exps.weight", {8, 8, 4});
      tensor(p + "ffn_down_exps.weight", {8, 8, 4});
      tensor(p + "attn_sinks.weight", {4});
    }
  }
  ArchitectureConfig c;
  c.hidden = 8; c.n_layers = 2; c.n_heads = 4; c.head_dim = 4; c.v_head = 2;
  c.n_experts = 4; c.topk = 2; c.moe_inter = 8; c.dense_inter = 16;
  std::string reason;
  CHECK(detect_model_family(file) == ModelFamily::MiMo2);
  CHECK(configure_mimo2(file, c, reason));
  CHECK(c.first_dense == 1 && c.n_head_kv == 1);
  CHECK(c.mimo2_head_count_kv[1] == 2);
  c.v_head = 4; CHECK(!configure_mimo2(file, c, reason)); c.v_head = 2;
  file.kv_arr_int["mimo2.attention.sliding_window_pattern"] = {0};
  CHECK(!configure_mimo2(file, c, reason));
  file.kv_arr_int["mimo2.attention.sliding_window_pattern"] = {0, 2};
  CHECK(!configure_mimo2(file, c, reason));
  file.kv_arr_int["mimo2.attention.sliding_window_pattern"] = {0, 1};
  CHECK(configure_mimo2(file, c, reason));
  CHECK(mimo2_attention_visible(128, 1, 128, true));
  CHECK(!mimo2_attention_visible(128, 0, 128, true));
  CHECK(mimo2_attention_visible(128, 0, 128, false));
  CHECK(!mimo2_attention_visible(128, 129, 128, false));
  CHECK(mimo2_kv_row(1, 256, 128) == 384);
  CHECK(mimo2_kv_row(0, 256, 128) == 128);
  if (argc > 1) {
    GGUFFile actual;
    CHECK(actual.load_header(argv[1]));
    auto &a = actual.architecture_config;
    a.hidden = actual.get_config_int("embedding_length");
    a.n_layers = actual.get_config_int("block_count");
    a.n_heads = actual.get_config_int("attention.head_count");
    a.head_dim = actual.get_config_int("attention.key_length");
    a.v_head = actual.get_config_int("attention.value_length");
    a.n_experts = actual.get_config_int("expert_count");
    a.topk = actual.get_config_int("expert_used_count");
    a.moe_inter = actual.get_config_int("expert_feed_forward_length");
    a.dense_inter = actual.get_config_int("feed_forward_length");
    CHECK(configure_mimo2(actual, a, reason));
    std::printf("actual GGUF: layers=%d first_dense=%d experts=%d topk=%d reason=%s\n",
                  a.n_layers, a.first_dense, a.n_experts, a.topk, reason.c_str());
  }
  if (!failures) std::puts("PASS: MiMo geometry, mixed-head KV layout and SWA boundary");
  return failures ? 1 : 0;
}
