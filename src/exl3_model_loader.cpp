#include "exl3_model_loader.hpp"

#include "exl3_cpu.hpp"

#include <cmath>
#include <cstring>
#include <fstream>

namespace {

bool read_tensor(const SafetensorsModelInfo &model, const std::string &name,
                 std::vector<uint8_t> &bytes, std::string &error) {
  auto tensor = model.tensors.find(name);
  if (tensor == model.tensors.end()) {
    error = "Missing EXL3 tensor: " + name;
    return false;
  }
  const SafetensorsTensorInfo &info = tensor->second;
  if (info.file_path.empty() || info.length == 0) {
    error = "Invalid EXL3 tensor location: " + name;
    return false;
  }
  std::ifstream file(info.file_path, std::ios::binary);
  if (!file) {
    error = "Cannot open EXL3 shard: " + info.file_path;
    return false;
  }
  bytes.resize(static_cast<size_t>(info.length));
  file.seekg(static_cast<std::streamoff>(info.offset));
  file.read(reinterpret_cast<char *>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
  if (file.gcount() != static_cast<std::streamsize>(bytes.size())) {
    error = "Short read for EXL3 tensor: " + name;
    return false;
  }
  return true;
}

bool f16_tensor(const SafetensorsModelInfo &model, const std::string &name,
                 int expected_elements, std::vector<uint8_t> &bytes,
                 std::string &error) {
  auto tensor = model.tensors.find(name);
  if (tensor == model.tensors.end() || tensor->second.dtype != "F16" ||
      tensor->second.length !=
          static_cast<uint64_t>(expected_elements) * sizeof(uint16_t)) {
    error = "EXL3 tensor must be F16 with expected shape: " + name;
    return false;
  }
  return read_tensor(model, name, bytes, error);
}

} // namespace

bool read_exl3_tensor(const SafetensorsModelInfo &model, const std::string &name,
                      std::vector<uint8_t> &bytes, std::string &error) {
  return read_tensor(model, name, bytes, error);
}

std::string exl3_qwen_tensor_name(const std::string &raw) {
  std::string name = raw;
  if (name.starts_with("model.language_model."))
    name.erase(0, 21);
  else if (name.starts_with("model."))
    name.erase(0, 6);
  if (name == "embed_tokens.weight") return "token_embd.weight";
  if (name == "lm_head" || name == "lm_head.weight" || name == "language_model.lm_head")
    return "output.weight";
  if (name.starts_with("hyper_connection_mixer.")) {
    const std::string suffix = name.substr(23);
    if (suffix == "hc_norm.weight") return "output_hc_norm.weight";
    if (suffix == "input_mix_weight_down.weight") return "output_hc_down.weight";
    if (suffix == "input_mix_weight_up.weight") return "output_hc_up.weight";
  }
  if (!name.starts_with("layers.")) return {};
  const size_t end = name.find('.', 7);
  if (end == std::string::npos) return {};
  const std::string prefix = "blk." + name.substr(7, end - 7) + ".";
  std::string suffix = name.substr(end + 1);
  const std::pair<const char *, const char *> mappings[] = {
      {"linear_attn.in_proj_qkv", "attn_qkv.weight"},
      {"linear_attn.in_proj_z", "attn_gate.weight"},
      {"linear_attn.in_proj_a.weight", "ssm_alpha.weight"},
      {"linear_attn.in_proj_b.weight", "ssm_beta.weight"},
      {"linear_attn.out_proj", "ssm_out.weight"},
      {"linear_attn.conv1d.weight", "ssm_conv1d.weight"},
      {"linear_attn.A_log", "ssm_a"},
      {"linear_attn.dt_bias", "ssm_dt.bias"},
      {"linear_attn.norm.weight", "ssm_norm.weight"},
      {"self_attn.q_proj", "attn_q.weight"},
      {"self_attn.k_proj", "attn_k.weight"},
      {"self_attn.v_proj", "attn_v.weight"},
      {"self_attn.o_proj", "attn_output.weight"},
      {"self_attn.q_norm.weight", "attn_q_norm.weight"},
      {"self_attn.k_norm.weight", "attn_k_norm.weight"},
      {"mlp.shared_expert.gate_proj", "ffn_gate_shexp.weight"},
      {"mlp.shared_expert.up_proj", "ffn_up_shexp.weight"},
      {"mlp.shared_expert.down_proj", "ffn_down_shexp.weight"},
      {"mlp.shared_expert_gate.weight", "ffn_gate_inp_shexp.weight"},
      {"mlp.gate.weight", "ffn_gate_inp.weight"},
      {"ple.key_proj", "ple_key.weight"},
      {"ple.value_proj", "ple_value.weight"},
      {"ple.norm_key.weight", "ple_norm_key.weight"},
      {"ple.norm_query.weight", "ple_norm_query.weight"},
      {"ple.norm_conv.weight", "ple_norm_conv.weight"},
      {"ple.conv1d.weight", "ple_conv1d.weight"},
  };
  for (const auto &[source, destination] : mappings)
    if (suffix == source || suffix == std::string(source) + ".weight")
      return prefix + destination;
  for (const auto &[source, destination] : {
           std::pair{"attn_hyper_connection.", "hc_attn_"},
           std::pair{"mlp_hyper_connection.", "hc_ffn_"}}) {
    if (suffix.starts_with(source)) {
      suffix.erase(0, std::strlen(source));
      const std::pair<const char *, const char *> fields[] = {
          {"hc_norm.weight", "norm.weight"},
          {"input_mix_weight_down.weight", "down.weight"},
          {"input_mix_weight_up.weight", "up.weight"},
          {"block_inject_weight.weight", "inject.weight"}};
      for (const auto &[field, target] : fields)
        if (suffix == field) return prefix + destination + target;
    }
  }
  return {};
}

bool load_exl3_mul1_matrix(const SafetensorsModelInfo &model,
                           const Exl3LinearInfo &linear,
                           std::vector<uint8_t> &packed,
                           std::string &error) {
  packed.clear();
  if (!linear.valid || linear.codebook != Exl3Codebook::MUL1 ||
      linear.suh_name.empty() || linear.svh_name.empty() ||
      linear.bits_per_weight < 1.0 || linear.bits_per_weight > 8.0 ||
      std::floor(linear.bits_per_weight) != linear.bits_per_weight) {
    error = "Unsupported EXL3 matrix contract: " + linear.key;
    return false;
  }

  std::vector<uint8_t> trellis;
  std::vector<uint8_t> suh;
  std::vector<uint8_t> svh;
  std::vector<uint8_t> bias;
  if (!read_tensor(model, linear.trellis_name, trellis, error) ||
      !f16_tensor(model, linear.suh_name,
                  static_cast<int>(linear.in_features), suh, error) ||
      !f16_tensor(model, linear.svh_name,
                  static_cast<int>(linear.out_features), svh, error))
    return false;
  if (!linear.bias_name.empty() &&
      !f16_tensor(model, linear.bias_name,
                  static_cast<int>(linear.out_features), bias, error))
    return false;

  const size_t expected_trellis =
      static_cast<size_t>(linear.in_features / 16) *
      (linear.out_features / 16) * 16 *
      static_cast<int>(linear.bits_per_weight) * sizeof(uint16_t);
  if (trellis.size() != expected_trellis) {
    error = "EXL3 trellis byte size mismatch: " + linear.trellis_name;
    return false;
  }

  Exl3Mul1MatrixView matrix{
      .trellis = reinterpret_cast<const uint16_t *>(trellis.data()),
      .suh = reinterpret_cast<const uint16_t *>(suh.data()),
      .svh = reinterpret_cast<const uint16_t *>(svh.data()),
      .bias = bias.empty() ? nullptr
                           : reinterpret_cast<const uint16_t *>(bias.data()),
      .input_features = static_cast<int>(linear.in_features),
      .output_features = static_cast<int>(linear.out_features),
      .bits = static_cast<int>(linear.bits_per_weight),
  };
  packed = exl3_mul1_pack_matrix(matrix);
  if (packed.empty()) {
    error = "Failed to pack EXL3 matrix: " + linear.key;
    return false;
  }
  return true;
}
