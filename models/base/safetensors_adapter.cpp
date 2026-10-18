#include "safetensors_adapter.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace {

enum class JsonType { Null, Number, String, Array, Object };

struct JsonValue {
  JsonType type = JsonType::Null;
  std::string string_value;
  double number_value = 0.0;
  std::vector<JsonValue> array_value;
  std::unordered_map<std::string, JsonValue> object_value;
};

std::string trim(const std::string &s) {
  const std::string ws = " \t\r\n";
  size_t first = s.find_first_not_of(ws);
  if (first == std::string::npos)
    return "";
  size_t last = s.find_last_not_of(ws);
  return s.substr(first, last - first + 1);
}

std::string unescape_json_string(const std::string &in) {
  std::string out;
  out.reserve(in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    if (in[i] == '\\' && i + 1 < in.size()) {
      char c = in[++i];
      switch (c) {
      case '"':
        out.push_back('"');
        break;
      case '\\':
        out.push_back('\\');
        break;
      case '/':
        out.push_back('/');
        break;
      case 'b':
        out.push_back('\b');
        break;
      case 'f':
        out.push_back('\f');
        break;
      case 'n':
        out.push_back('\n');
        break;
      case 'r':
        out.push_back('\r');
        break;
      case 't':
        out.push_back('\t');
        break;
      case 'u': {
        if (i + 4 < in.size()) {
          std::string hex = in.substr(i + 1, 4);
          try {
            auto code = static_cast<int>(std::stoul(hex, nullptr, 16));
            if (code <= 0x7F) {
              out.push_back(static_cast<char>(code));
            } else if (code <= 0x7FF) {
              out.push_back(static_cast<char>(0xC0 | ((code >> 6) & 0x1F)));
              out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
            } else {
              out.push_back(static_cast<char>(0xE0 | ((code >> 12) & 0x0F)));
              out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
              out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
            }
            i += 4;
          } catch (...) {
            out.push_back('u');
          }
        }
        break;
      }
      default:
        out.push_back(c);
        break;
      }
    } else {
      out.push_back(in[i]);
    }
  }
  return out;
}

class JsonParser {
 public:
  explicit JsonParser(const std::string &input) : input_(input), pos_(0) {}

  JsonValue parse() {
    JsonValue v = parse_value();
    skip_ws();
    if (pos_ != input_.size()) {
      throw std::runtime_error("Unexpected trailing JSON data");
    }
    return v;
  }

 private:
  char peek() const { return pos_ < input_.size() ? input_[pos_] : '\0'; }
  void skip_ws() {
    while (pos_ < input_.size() && std::isspace(static_cast<unsigned char>(input_[pos_])))
      ++pos_;
  }
  bool consume(char expected) {
    skip_ws();
    if (peek() == expected) {
      ++pos_;
      return true;
    }
    return false;
  }
  std::string parse_string() {
    if (!consume('"'))
      throw std::runtime_error("Expected string");
    std::string out;
    while (pos_ < input_.size()) {
      char c = input_[pos_++];
      if (c == '"')
        return unescape_json_string(out);
      if (c == '\\') {
        if (pos_ >= input_.size())
          throw std::runtime_error("Unterminated escape");
        out.push_back(input_[pos_++]);
      } else if (static_cast<unsigned char>(c) < 0x20) {
        throw std::runtime_error("Unescaped control character in string");
      } else {
        out.push_back(c);
      }
    }
    throw std::runtime_error("Unterminated string");
  }

  JsonValue parse_number() {
    size_t start = pos_;
    if (peek() == '-')
      ++pos_;
    while (pos_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[pos_])))
      ++pos_;
    if (peek() == '.') {
      ++pos_;
      while (pos_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[pos_])))
        ++pos_;
    }
    if (peek() == 'e' || peek() == 'E') {
      ++pos_;
      if (peek() == '+' || peek() == '-')
        ++pos_;
      while (pos_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[pos_])))
        ++pos_;
    }
    std::string literal = input_.substr(start, pos_ - start);
    JsonValue v;
    v.type = JsonType::Number;
    v.number_value = std::stod(literal);
    return v;
  }

  JsonValue parse_array() {
    JsonValue v;
    v.type = JsonType::Array;
    if (!consume('['))
      throw std::runtime_error("Expected [");
    skip_ws();
    if (consume(']'))
      return v;
    while (true) {
      v.array_value.push_back(parse_value());
      skip_ws();
      if (consume(',')) {
        continue;
      }
      if (consume(']'))
        break;
      throw std::runtime_error("Expected , or ] in array");
    }
    return v;
  }

  JsonValue parse_object() {
    JsonValue v;
    v.type = JsonType::Object;
    if (!consume('{'))
      throw std::runtime_error("Expected {");
    skip_ws();
    if (consume('}'))
      return v;
    while (true) {
      skip_ws();
      std::string key = parse_string();
      skip_ws();
      if (!consume(':'))
        throw std::runtime_error("Expected : after object key");
      JsonValue value = parse_value();
      v.object_value.emplace(std::move(key), std::move(value));
      skip_ws();
      if (consume(','))
        continue;
      if (consume('}'))
        break;
      throw std::runtime_error("Expected , or } in object");
    }
    return v;
  }

  JsonValue parse_value() {
    skip_ws();
    char c = peek();
    if (c == '{')
      return parse_object();
    if (c == '[')
      return parse_array();
    if (c == '"') {
      JsonValue v;
      v.type = JsonType::String;
      v.string_value = parse_string();
      return v;
    }
    if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) {
      return parse_number();
    }
    if (input_.compare(pos_, 4, "true") == 0) {
      pos_ += 4;
      JsonValue v;
      v.type = JsonType::String;
      v.string_value = "true";
      return v;
    }
    if (input_.compare(pos_, 5, "false") == 0) {
      pos_ += 5;
      JsonValue v;
      v.type = JsonType::String;
      v.string_value = "false";
      return v;
    }
    if (input_.compare(pos_, 4, "null") == 0) {
      pos_ += 4;
      JsonValue v;
      v.type = JsonType::Null;
      return v;
    }
    throw std::runtime_error("Unsupported JSON value");
  }

  const std::string input_;
  size_t pos_;
};

bool parse_shape_from_json(const JsonValue &object_entry, SafetensorsTensorInfo &info) {
  auto it = object_entry.object_value.find("shape");
  if (it == object_entry.object_value.end() || it->second.type != JsonType::Array)
    return false;
  for (const auto &entry : it->second.array_value) {
    if (entry.type != JsonType::Number)
      return false;
    info.shape.push_back(static_cast<int64_t>(entry.number_value));
  }
  return true;
}

bool parse_dtype_from_json(const JsonValue &object_entry, SafetensorsTensorInfo &info) {
  auto it = object_entry.object_value.find("dtype");
  if (it == object_entry.object_value.end() || it->second.type != JsonType::String)
    return false;
  info.dtype = it->second.string_value;
  return true;
}

bool parse_data_offsets_from_json(const JsonValue &object_entry,
                                  SafetensorsTensorInfo &info) {
  auto it = object_entry.object_value.find("data_offsets");
  if (it == object_entry.object_value.end() || it->second.type != JsonType::Array ||
      it->second.array_value.size() < 2)
    return false;
  if (it->second.array_value[0].type != JsonType::Number ||
      it->second.array_value[1].type != JsonType::Number)
    return false;
  info.data_offset = static_cast<std::uint64_t>(it->second.array_value[0].number_value);
  info.data_end = static_cast<std::uint64_t>(it->second.array_value[1].number_value);
  return true;
}

std::string strip_prefixes(std::string name) {
  while (!name.empty() && (name.rfind("model.", 0) == 0 || name.rfind("transformer.", 0) == 0 ||
                           name.rfind("decoder.", 0) == 0 || name.rfind("module.", 0) == 0)) {
    if (name.rfind("model.", 0) == 0)
      name.erase(0, 6);
    else if (name.rfind("transformer.", 0) == 0)
      name.erase(0, 12);
    else if (name.rfind("decoder.", 0) == 0)
      name.erase(0, 8);
    else if (name.rfind("module.", 0) == 0)
      name.erase(0, 7);
  }
  return name;
}

int infer_layer_index(const std::string &name) {
  auto try_extract = [&](const std::string &prefix) -> int {
    size_t pos = name.find(prefix);
    if (pos == std::string::npos)
      return -1;
    size_t start = pos + prefix.size();
    size_t end = name.find('.', start);
    if (end == std::string::npos)
      return -1;
    try {
      return std::stoi(name.substr(start, end - start));
    } catch (...) {
      return -1;
    }
  };
  if (int v = try_extract("layers."); v >= 0)
    return v;
  if (int v = try_extract("blk."); v >= 0)
    return v;
  return -1;
}

std::uint64_t dtype_size_bytes(const std::string &dtype) {
  static const std::unordered_map<std::string, std::uint64_t> sizes = {
      {"F32", 4}, {"F16", 2}, {"BF16", 2}, {"I32", 4}, {"I64", 8},
      {"I16", 2}, {"U16", 2}, {"U32", 4}, {"U8", 1},  {"I8", 1}, {"BOOL", 1},
      {"F64", 8}, {"F8_E4M3", 1}, {"F8_E5M2", 1}, {"F8_E8M0", 1}};
  auto it = sizes.find(dtype);
  return it != sizes.end() ? it->second : 1;
}

bool is_known_dtype(const std::string &dtype) {
  static const std::unordered_set<std::string> known = {
      "F32", "F16", "BF16", "I32", "I64", "I16", "U16", "U32", "U8",
      "I8", "BOOL", "F64", "F8_E4M3", "F8_E5M2", "F8_E8M0"};
  return known.find(dtype) != known.end();
}

bool has_suffix(const std::string &value, const std::string &suffix) {
  return value.size() >= suffix.size() &&
         value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string exl3_base_name(const std::string &name) {
  static const std::array<const char *, 7> suffixes = {
      ".trellis", ".suh", ".svh", ".su", ".sv", ".bias", ".mcg"};
  for (const char *suffix : suffixes) {
    if (has_suffix(name, suffix))
      return name.substr(0, name.size() - std::strlen(suffix));
  }
  if (has_suffix(name, ".mul1"))
    return name.substr(0, name.size() - 5);
  return {};
}

bool valid_exl3_marker(const SafetensorsTensorInfo &info) {
  return (info.shape.empty() || (info.shape.size() == 1 && info.shape[0] == 1)) &&
         (info.dtype == "I32" || info.dtype == "U32");
}

bool valid_exl3_scale(const SafetensorsTensorInfo &info, int64_t elements,
                      bool packed) {
  if (packed)
    return info.shape.size() == 1 && info.shape[0] == elements / 16 &&
           (info.dtype == "I16" || info.dtype == "U16");
  return info.shape.size() == 1 && info.shape[0] == elements &&
         (info.dtype == "F16" || info.dtype == "BF16" || info.dtype == "F32");
}

bool infer_exl3_rate(int64_t words, double &rate) {
  if (words >= 16 && words <= 128 && words % 16 == 0) {
    rate = static_cast<double>(words) / 16.0;
    return true;
  }
  if (words == 24 || words == 40 || words == 56) {
    rate = static_cast<double>(words) / 16.0;
    return true;
  }
  return false;
}

bool is_regular_model_file(const std::filesystem::directory_entry &entry,
                           std::error_code &ec) {
  ec.clear();
  if (entry.is_regular_file(ec))
    return true;
  if (ec || !entry.is_symlink(ec))
    return false;
  ec.clear();
  return std::filesystem::is_regular_file(entry.status(ec));
}

std::string canonical_qwen_generic(const std::string &raw_name) {
  std::string name = strip_prefixes(raw_name);
  std::string out = name;
  // Directional alias normalization: keep the logical Qwen trait names stable.
  if (out.find("self_attn.q_proj") != std::string::npos)
    out = "layers." + std::to_string(infer_layer_index(out)) + ".self_attn.q_proj.weight";
  else if (out.find("self_attn.k_proj") != std::string::npos)
    out = "layers." + std::to_string(infer_layer_index(out)) + ".self_attn.k_proj.weight";
  else if (out.find("self_attn.v_proj") != std::string::npos)
    out = "layers." + std::to_string(infer_layer_index(out)) + ".self_attn.v_proj.weight";
  else if (out.find("self_attn.o_proj") != std::string::npos)
    out = "layers." + std::to_string(infer_layer_index(out)) + ".self_attn.o_proj.weight";
  else if (out.find("mlp.gate_proj") != std::string::npos)
    out = "layers." + std::to_string(infer_layer_index(out)) + ".mlp.gate_proj.weight";
  else if (out.find("mlp.up_proj") != std::string::npos)
    out = "layers." + std::to_string(infer_layer_index(out)) + ".mlp.up_proj.weight";
  else if (out.find("mlp.down_proj") != std::string::npos)
    out = "layers." + std::to_string(infer_layer_index(out)) + ".mlp.down_proj.weight";
  else if (out.find("input_layernorm") != std::string::npos)
    out = "layers." + std::to_string(infer_layer_index(out)) + ".input_layernorm.weight";
  else if (out.find("post_attention_layernorm") != std::string::npos)
    out = "layers." + std::to_string(infer_layer_index(out)) + ".post_attention_layernorm.weight";
  else if (out.find("q_norm") != std::string::npos)
    out = "layers." + std::to_string(infer_layer_index(out)) + ".self_attn.q_norm.weight";
  else if (out.find("k_norm") != std::string::npos)
    out = "layers." + std::to_string(infer_layer_index(out)) + ".self_attn.k_norm.weight";
  else if (out.find("embed_tokens") != std::string::npos || out.find("tok_embeddings") != std::string::npos)
    out = "tok_embeddings.weight";
  else if (out.find("lm_head") != std::string::npos || out.find("output.weight") != std::string::npos)
    out = "lm_head.weight";
  else if (out.find("final_norm") != std::string::npos || out.find("norm.weight") != std::string::npos)
    out = "final_norm.weight";

  // If a canonical path was not inferred from the name pattern, keep the
  // stripped, generic name so tests and future expanders can still reason about
  // it deterministically.
  return out;
}

} // namespace

bool SafetensorsAdapter::parse_file(const std::string &path,
                                   SafetensorsModelInfo &out) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    return false;

  in.seekg(0, std::ios::end);
  const std::streamoff file_size = in.tellg();
  if (file_size < 8)
    return false;
  in.seekg(0, std::ios::beg);
  std::uint64_t header_len = 0;
  in.read(reinterpret_cast<char *>(&header_len), sizeof(header_len));
  if (!in || header_len == 0 || header_len > 100 * 1024 * 1024 ||
      header_len > static_cast<std::uint64_t>(file_size - 8))
    return false;

  std::string header_json(static_cast<size_t>(header_len), '\0');
  in.read(header_json.data(), static_cast<std::streamsize>(header_len));
  if (!in)
    return false;

  if (!parse_header_bytes(header_json, out))
    return false;
  const std::uint64_t data_base = 8 + header_len;
  for (auto &entry : out.tensors) {
    auto &info = entry.second;
    if (info.data_end < info.data_offset ||
        info.data_end - info.data_offset != info.length ||
        data_base > static_cast<std::uint64_t>(file_size) ||
        info.data_end > static_cast<std::uint64_t>(file_size) - data_base)
      return false;
    info.offset = data_base + info.data_offset;
    info.file_path = path;
  }
  return true;
}

bool SafetensorsAdapter::parse_directory(const std::string &path,
                                         SafetensorsModelInfo &out) {
  namespace fs = std::filesystem;
  std::error_code ec;
  if (!fs::is_directory(path, ec))
    return false;

  std::vector<fs::path> files;
  for (const auto &entry : fs::directory_iterator(path, ec)) {
    if (ec)
      return false;
    if (is_regular_model_file(entry, ec) &&
        entry.path().extension() == ".safetensors")
      files.push_back(entry.path());
  }
  if (ec || files.empty())
    return false;
  std::sort(files.begin(), files.end());

  out = SafetensorsModelInfo{};
  for (const fs::path &file : files) {
    SafetensorsModelInfo shard;
    if (!parse_file(file.string(), shard))
      return false;
    if (out.format == "generic" && shard.format != "generic")
      out.format = shard.format;
    for (const auto &metadata : shard.metadata)
      out.metadata.emplace(metadata.first, metadata.second);
    for (auto &tensor : shard.tensors) {
      if (out.tensors.find(tensor.first) != out.tensors.end())
        return false;
      out.tensors.emplace(tensor.first, std::move(tensor.second));
    }
  }
  return !out.tensors.empty();
}

bool SafetensorsAdapter::parse_header_bytes(const std::string &header_json,
                                          SafetensorsModelInfo &out) {
  out = SafetensorsModelInfo{};
  try {
    JsonParser parser(header_json);
    const JsonValue root = parser.parse();
    if (root.type != JsonType::Object)
      return false;

    const auto metadata_it = root.object_value.find("__metadata__");
    if (metadata_it != root.object_value.end() && metadata_it->second.type == JsonType::Object) {
      for (const auto &kv : metadata_it->second.object_value) {
        if (kv.second.type == JsonType::String)
          out.metadata[kv.first] = kv.second.string_value;
        else if (kv.second.type == JsonType::Number)
          out.metadata[kv.first] = std::to_string(kv.second.number_value);
      }
      const auto format_it = out.metadata.find("format");
      if (format_it != out.metadata.end())
        out.format = format_it->second;
    }

    for (const auto &kv : root.object_value) {
      if (kv.first == "__metadata__")
        continue;
      if (kv.second.type != JsonType::Object)
        continue;
      SafetensorsTensorInfo info;
      info.name = kv.first;
      if (!parse_shape_from_json(kv.second, info) ||
          !parse_dtype_from_json(kv.second, info) ||
          !parse_data_offsets_from_json(kv.second, info))
        return false;
      if (!is_known_dtype(info.dtype) || info.data_end < info.data_offset)
        return false;
      info.length = 1;
      for (int64_t dim : info.shape) {
        if (dim < 0 || static_cast<std::uint64_t>(dim) >
                            std::numeric_limits<std::uint64_t>::max() / info.length)
          return false;
        info.length *= static_cast<std::uint64_t>(dim);
      }
      if (info.shape.empty())
        info.length = 1;
      if (info.length > std::numeric_limits<std::uint64_t>::max() /
                            dtype_size_bytes(info.dtype))
        return false;
      info.length *= dtype_size_bytes(info.dtype);
      out.tensors[info.name] = std::move(info);
    }
    return !out.tensors.empty();
  } catch (...) {
    return false;
  }
}

SafetensorsFamily SafetensorsAdapter::detect_family(const SafetensorsModelInfo &model) {
  if (is_exl3_model(model))
    return SafetensorsFamily::EXL3;

  bool has_qwen_hybrid_marker = false;
  bool has_dense_gqa_marker = false;
  bool has_moe_experts = false;

  for (const auto &kv : model.tensors) {
    std::string name = kv.first;
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    if (name.find("attn_qkv") != std::string::npos ||
        name.find("attn_gate") != std::string::npos ||
        name.find("ssm_a") != std::string::npos ||
        name.find("ssm_conv1d") != std::string::npos) {
      has_qwen_hybrid_marker = true;
    }
    if (name.find("self_attn.q_proj") != std::string::npos ||
        name.find("self_attn.k_proj") != std::string::npos ||
        name.find("self_attn.v_proj") != std::string::npos ||
        name.find("self_attn.o_proj") != std::string::npos) {
      has_dense_gqa_marker = true;
    }
    if (name.find("ffn_gate_exps") != std::string::npos ||
        name.find("mlp.gate_proj") != std::string::npos) {
      has_moe_experts = true;
    }
  }

  if (has_qwen_hybrid_marker)
    return SafetensorsFamily::QwenHybridMoE;
  if (has_dense_gqa_marker)
    return SafetensorsFamily::DenseGQA;
  // Fallback: if there are gate/up/down projections but no self_attn markers,
  // conservatively treat as dense (a future mapper can refine).
  if (has_moe_experts)
    return SafetensorsFamily::DenseGQA;
  return SafetensorsFamily::Unknown;
}

std::string SafetensorsAdapter::detect_format(const SafetensorsModelInfo &model) {
  const std::string lower = [&]() {
    std::string s = model.format;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    return s;
  }();

  if (lower.find("dflash") != std::string::npos)
    return "dflash";
  if (lower.find("dspark") != std::string::npos)
    return "dspark";

  if (is_exl3_model(model))
    return "exl3";

  for (const auto &kv : model.tensors) {
    std::string name = kv.first;
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    if (name.find("dflash") != std::string::npos || name.find("df") != std::string::npos)
      return "dflash";
    if (name.find("dspark") != std::string::npos)
      return "dspark";
  }

  return model.tensors.empty() ? "generic" : "generic";
}

bool SafetensorsAdapter::is_exl3_model(const SafetensorsModelInfo &model) {
  for (const auto &entry : model.tensors) {
    if (has_suffix(entry.first, ".trellis"))
      return true;
  }
  return false;
}

Exl3Contract SafetensorsAdapter::map_exl3_contract(
    const SafetensorsModelInfo &model) {
  Exl3Contract contract;
  std::vector<std::string> bases;
  for (const auto &entry : model.tensors) {
    if (has_suffix(entry.first, ".trellis"))
      bases.push_back(exl3_base_name(entry.first));
  }
  std::sort(bases.begin(), bases.end());
  bases.erase(std::unique(bases.begin(), bases.end()), bases.end());
  contract.complete = !bases.empty();

  for (const std::string &base : bases) {
    if (base.find("ngram_embedding") != std::string::npos)
      continue;
    Exl3LinearInfo linear;
    linear.key = base;
    linear.trellis_name = base + ".trellis";
    auto trellis_it = model.tensors.find(linear.trellis_name);
    if (trellis_it == model.tensors.end())
      continue;
    const SafetensorsTensorInfo &trellis = trellis_it->second;
    linear.trellis_shape = trellis.shape;
    bool valid = trellis.dtype == "I16" && trellis.shape.size() == 3 &&
                 trellis.shape[0] > 0 && trellis.shape[1] > 0 &&
                 trellis.shape[2] > 0;
    if (valid) {
      linear.in_features = trellis.shape[0] * 16;
      linear.out_features = trellis.shape[1] * 16;
      linear.trellis_words = trellis.shape[2];
      valid = infer_exl3_rate(linear.trellis_words, linear.bits_per_weight);
    }

    const std::string suh_name = base + ".suh";
    const std::string su_name = base + ".su";
    const std::string svh_name = base + ".svh";
    const std::string sv_name = base + ".sv";
    auto suh_it = model.tensors.find(suh_name);
    auto su_it = model.tensors.find(su_name);
    auto svh_it = model.tensors.find(svh_name);
    auto sv_it = model.tensors.find(sv_name);
    const bool has_suh = suh_it != model.tensors.end();
    const bool has_su = su_it != model.tensors.end();
    const bool has_svh = svh_it != model.tensors.end();
    const bool has_sv = sv_it != model.tensors.end();
    valid = valid && (has_suh || has_su) && (has_svh || has_sv) &&
            !(has_suh && has_su) && !(has_svh && has_sv);
    if (valid) {
      if (has_suh) {
        linear.suh_name = suh_name;
        valid = valid && valid_exl3_scale(suh_it->second, linear.in_features, false);
      } else {
        linear.su_name = su_name;
        valid = valid && valid_exl3_scale(su_it->second, linear.in_features, true);
      }
      if (has_svh) {
        linear.svh_name = svh_name;
        valid = valid && valid_exl3_scale(svh_it->second, linear.out_features, false);
      } else {
        linear.sv_name = sv_name;
        valid = valid && valid_exl3_scale(sv_it->second, linear.out_features, true);
      }
    }

    const std::string bias_name = base + ".bias";
    const std::string mcg_name = base + ".mcg";
    const std::string mul1_name = base + ".mul1";
    auto bias_it = model.tensors.find(bias_name);
    auto mcg_it = model.tensors.find(mcg_name);
    auto mul1_it = model.tensors.find(mul1_name);
    if (bias_it != model.tensors.end()) {
      linear.bias_name = bias_name;
      valid = valid && bias_it->second.shape.size() == 1 &&
              bias_it->second.shape[0] == linear.out_features &&
              (bias_it->second.dtype == "F16" || bias_it->second.dtype == "BF16" ||
               bias_it->second.dtype == "F32");
    }
    valid = valid && !(mcg_it != model.tensors.end() && mul1_it != model.tensors.end());
    if (mcg_it != model.tensors.end()) {
      valid = valid && valid_exl3_marker(mcg_it->second);
      linear.mcg_name = mcg_name;
      linear.codebook = Exl3Codebook::MCG;
    } else if (mul1_it != model.tensors.end()) {
      valid = valid && valid_exl3_marker(mul1_it->second);
      linear.mul1_name = mul1_name;
      linear.codebook = Exl3Codebook::MUL1;
    }

    linear.valid = valid;
    contract.linears.push_back(std::move(linear));
    if (!valid)
      contract.complete = false;
    const int layer = infer_layer_index(base);
    if (layer >= 0)
      contract.n_layers = std::max(contract.n_layers, layer + 1);
    const size_t expert_pos = base.find("experts.");
    if (expert_pos != std::string::npos) {
      const size_t start = expert_pos + 8;
      const size_t end = base.find('.', start);
      if (end != std::string::npos) {
        try {
          contract.n_experts = std::max(contract.n_experts,
                                       std::stoi(base.substr(start, end - start)) + 1);
        } catch (...) {
          contract.complete = false;
        }
      }
    }
    if (linear.in_features > 0)
      contract.hidden = std::max(contract.hidden, static_cast<int>(linear.in_features));
    if (linear.out_features > 0)
      contract.intermediate = std::max(contract.intermediate,
                                       static_cast<int>(linear.out_features));
  }

  contract.complete = !contract.linears.empty() && contract.complete;
  if (contract.linears.empty())
    contract.complete = false;
  return contract;
}

std::vector<NormalizedTensor>
SafetensorsAdapter::normalize_tensors(const SafetensorsModelInfo &model,
                                     const std::string &format_hint) {
  std::vector<NormalizedTensor> normalized;
  normalized.reserve(model.tensors.size());
  std::string hint = format_hint.empty() ? detect_format(model) : format_hint;

  for (const auto &kv : model.tensors) {
    NormalizedTensor nt;
    nt.canonical_name = canonicalize_qwen_name(kv.first, hint);
    nt.logical_name = nt.canonical_name;
    nt.shape = kv.second.shape;
    nt.dtype = kv.second.dtype;
    normalized.push_back(nt);
  }

  std::sort(normalized.begin(), normalized.end(),
            [](const NormalizedTensor &a, const NormalizedTensor &b) {
              return a.logical_name < b.logical_name;
            });
  return normalized;
}

QwenCpuContract SafetensorsAdapter::map_qwen_cpu_contract(const SafetensorsModelInfo &model,
                                                          const std::string &format_hint) {
  QwenCpuContract contract;
  const std::string hint = format_hint.empty() ? detect_format(model) : format_hint;

  int max_layer = -1;
  for (const auto &kv : model.tensors) {
    const std::string canonical = canonicalize_qwen_name(kv.first, hint);
    const int layer = infer_layer_index(canonical);
    if (layer >= 0)
      max_layer = std::max(max_layer, layer);

    const auto &shape = kv.second.shape;
    if (canonical.find("self_attn.q_proj.weight") != std::string::npos && !shape.empty()) {
      contract.n_heads = std::max(1, static_cast<int>(shape[0] / 256));
      contract.head_dim = shape[0] / std::max(1, contract.n_heads);
    } else if (canonical.find("self_attn.k_proj.weight") != std::string::npos && !shape.empty()) {
      const int kv_heads = std::max(1, static_cast<int>(shape[0] / 256));
      if (contract.n_kv_heads == 0 || kv_heads > contract.n_kv_heads)
        contract.n_kv_heads = kv_heads;
    } else if (canonical.find("mlp.gate_proj.weight") != std::string::npos && !shape.empty()) {
      const int gate_rows = static_cast<int>(shape[0]);
      if (gate_rows > 0)
        contract.moe_inter = std::max(contract.moe_inter, gate_rows);
    } else if (canonical.find("mlp.down_proj.weight") != std::string::npos && !shape.empty()) {
      const int down_rows = static_cast<int>(shape[0]);
      if (down_rows > 0)
        contract.moe_inter = std::max(contract.moe_inter, down_rows);
    }

    if (canonical.find("lm_head.weight") != std::string::npos) {
      contract.resolved_tensors["lm_head.weight"] = kv.first;
    } else if (canonical.find("tok_embeddings.weight") != std::string::npos) {
      contract.resolved_tensors["tok_embeddings.weight"] = kv.first;
    } else {
      contract.resolved_tensors[canonical] = kv.first;
    }
  }

  contract.n_layers = std::max(0, max_layer + 1);
  if (contract.n_heads <= 0)
    contract.n_heads = 16;
  if (contract.n_kv_heads <= 0)
    contract.n_kv_heads = std::max(1, contract.n_heads / 8);
  if (contract.head_dim <= 0)
    contract.head_dim = 256;
  if (contract.moe_inter <= 0)
    contract.moe_inter = 6144;
  if (contract.moe_inter_shared <= 0)
    contract.moe_inter_shared = contract.moe_inter;

  const auto max_qwen_rows = [&]() {
    int max_rows = 0;
    for (const auto &kv : model.tensors) {
      const std::string canonical = canonicalize_qwen_name(kv.first, hint);
      if (canonical.find("mlp.gate_proj.weight") != std::string::npos && !kv.second.shape.empty())
        max_rows = std::max(max_rows, static_cast<int>(kv.second.shape[0]));
    }
    return max_rows;
  }();
  if (max_qwen_rows > 0)
    contract.n_experts = std::max(1, max_qwen_rows / std::max(1, contract.moe_inter));
  contract.topk = std::max(1, std::min(8, contract.n_experts));

  return contract;
}

DenseCpuContract SafetensorsAdapter::map_dense_cpu_contract(const SafetensorsModelInfo &model,
                                                          const std::string &format_hint) {
  DenseCpuContract contract;
  const std::string hint = format_hint.empty() ? detect_format(model) : format_hint;

  // First pass: gather shape facts without assuming head_dim is known.
  int max_layer = -1;
  for (const auto &kv : model.tensors) {
    const std::string canonical = canonicalize_name(kv.first, hint);
    const int layer = infer_layer_index(canonical);
    if (layer >= 0)
      max_layer = std::max(max_layer, layer);

    const auto &shape = kv.second.shape;
    if (shape.empty())
      continue;

    if (canonical.find("attn_q.weight") != std::string::npos && shape.size() >= 2) {
      contract.n_heads = static_cast<int>(shape[0]); // temporary raw out dim
      if (contract.hidden <= 0)
        contract.hidden = static_cast<int>(shape[1]); // in_features
    } else if (canonical.find("attn_k.weight") != std::string::npos && shape.size() >= 2) {
      contract.n_kv_heads = static_cast<int>(shape[0]); // temporary raw out dim
      if (contract.hidden <= 0)
        contract.hidden = static_cast<int>(shape[1]);
    } else if (canonical.find("ffn_gate.weight") != std::string::npos && shape.size() >= 2) {
      contract.intermediate = std::max(contract.intermediate, static_cast<int>(shape[0]));
      if (contract.hidden <= 0)
        contract.hidden = static_cast<int>(shape[1]);
    } else if (canonical.find("token_embd.weight") != std::string::npos && shape.size() >= 2) {
      contract.vocab = static_cast<int>(shape[0]);
      if (contract.hidden <= 0)
        contract.hidden = static_cast<int>(shape[1]);
    } else if (canonical.find("output.weight") != std::string::npos && shape.size() >= 2) {
      // For tied embeddings / LM heads, the larger dimension is usually vocab.
      const int dim0 = static_cast<int>(shape[0]);
      const int dim1 = static_cast<int>(shape[1]);
      contract.vocab = std::max(contract.vocab, std::max(dim0, dim1));
      if (contract.hidden <= 0)
        contract.hidden = std::min(dim0, dim1);
    } else if ((canonical.find("attn_norm.weight") != std::string::npos ||
                canonical.find("ffn_norm.weight") != std::string::npos ||
                canonical.find("output_norm.weight") != std::string::npos ||
                canonical.find("hidden_norm.weight") != std::string::npos) &&
               shape.size() == 1) {
      // Layer-norm shape is exactly the model hidden size.
      contract.hidden = static_cast<int>(shape[0]);
    } else if ((canonical.find("attn_q_norm.weight") != std::string::npos ||
                canonical.find("attn_k_norm.weight") != std::string::npos) &&
               shape.size() == 1) {
      if (contract.head_dim <= 0)
        contract.head_dim = static_cast<int>(shape[0]);
    }

    contract.resolved_tensors[canonical] = kv.first;
  }

  contract.n_layers = std::max(0, max_layer + 1);

  // If no per-head norm told us head_dim, infer it from hidden / n_heads trial.
  if (contract.head_dim <= 0 && contract.hidden > 0 && contract.n_heads > 0) {
    if (contract.hidden % contract.n_heads == 0)
      contract.head_dim = contract.hidden / contract.n_heads;
  }

  // Convert the raw q/k output dims into real head counts now that head_dim is
  // settled. If head_dim is still unknown, use common GQA divisors.
  auto heads_from_out = [&](int raw_out, int fallback_heads) -> int {
    if (contract.head_dim > 0 && raw_out % contract.head_dim == 0)
      return raw_out / contract.head_dim;
    // For an HF-style GQA export, the Q projection's output width is the
    // reliable source for head_dim.  The hidden width is the model input
    // width and need not equal the Q output width (e.g. hidden=2048,
    // q_out=4096, heads=32 => head_dim=128).  Using hidden/fallback_heads
    // here would infer 64 and consequently turn k_out=1024 into 16 KV heads
    // instead of the intended 8.
    if (raw_out > 0 && raw_out % fallback_heads == 0) {
      contract.head_dim = raw_out / fallback_heads;
      return fallback_heads;
    }
    if (contract.hidden > 0 && contract.hidden % fallback_heads == 0) {
      contract.head_dim = contract.hidden / fallback_heads;
      return fallback_heads;
    }
    return fallback_heads;
  };

  int q_out = contract.n_heads;   // was stored as raw output dim
  int k_out = contract.n_kv_heads; // was stored as raw output dim
  contract.n_heads = heads_from_out(q_out, 32);
  contract.n_kv_heads = heads_from_out(k_out, contract.n_heads);

  if (contract.head_dim <= 0 && contract.hidden > 0 && contract.n_heads > 0)
    contract.head_dim = contract.hidden / contract.n_heads;
  if (contract.n_kv_heads <= 0)
    contract.n_kv_heads = contract.n_heads;
  if (contract.intermediate <= 0)
    contract.intermediate = contract.hidden * 4;
  if (contract.vocab <= 0)
    contract.vocab = 32000;

  return contract;
}

static int json_int_or(const JsonValue &v, int fallback) {
  if (v.type == JsonType::Number)
    return static_cast<int>(v.number_value);
  if (v.type == JsonType::String) {
    try {
      return std::stoi(v.string_value);
    } catch (...) {
      return fallback;
    }
  }
  return fallback;
}

static float json_float_or(const JsonValue &v, float fallback) {
  if (v.type == JsonType::Number)
    return static_cast<float>(v.number_value);
  if (v.type == JsonType::String) {
    try {
      return std::stof(v.string_value);
    } catch (...) {
      return fallback;
    }
  }
  return fallback;
}

static std::vector<std::string> json_string_array(const JsonValue &v) {
  std::vector<std::string> out;
  if (v.type == JsonType::Array) {
    for (const auto &e : v.array_value) {
      if (e.type == JsonType::String)
        out.push_back(e.string_value);
    }
  }
  return out;
}

DenseCpuConfigJson SafetensorsAdapter::parse_dense_config_json(const std::string &path) {
  DenseCpuConfigJson cfg;
  std::ifstream f(path);
  if (!f)
    return cfg;
  std::string json((std::istreambuf_iterator<char>(f)),
                   std::istreambuf_iterator<char>());
  try {
    JsonParser parser(json);
    JsonValue root = parser.parse();
    if (root.type != JsonType::Object)
      return cfg;

    cfg.present = true;
    auto it = root.object_value.find("num_hidden_layers");
    if (it != root.object_value.end())
      cfg.n_layers = json_int_or(it->second, cfg.n_layers);
    it = root.object_value.find("num_attention_heads");
    if (it != root.object_value.end())
      cfg.n_heads = json_int_or(it->second, cfg.n_heads);
    it = root.object_value.find("num_key_value_heads");
    if (it != root.object_value.end())
      cfg.n_kv_heads = json_int_or(it->second, cfg.n_kv_heads);
    it = root.object_value.find("head_dim");
    if (it != root.object_value.end())
      cfg.head_dim = json_int_or(it->second, cfg.head_dim);
    it = root.object_value.find("hidden_size");
    if (it != root.object_value.end())
      cfg.hidden = json_int_or(it->second, cfg.hidden);
    it = root.object_value.find("intermediate_size");
    if (it != root.object_value.end())
      cfg.intermediate = json_int_or(it->second, cfg.intermediate);
    it = root.object_value.find("vocab_size");
    if (it != root.object_value.end())
      cfg.vocab = json_int_or(it->second, cfg.vocab);
    it = root.object_value.find("sliding_window");
    if (it != root.object_value.end())
      cfg.sliding_window = json_int_or(it->second, cfg.sliding_window);
    it = root.object_value.find("rms_norm_eps");
    if (it != root.object_value.end())
      cfg.eps = json_float_or(it->second, cfg.eps);
    it = root.object_value.find("rope_parameters");
    if (it != root.object_value.end() && it->second.type == JsonType::Object) {
      auto rope_theta_it = it->second.object_value.find("rope_theta");
      if (rope_theta_it != it->second.object_value.end())
        cfg.rope_theta = json_float_or(rope_theta_it->second, cfg.rope_theta);
    }
    it = root.object_value.find("layer_types");
    if (it != root.object_value.end())
      cfg.layer_types = json_string_array(it->second);
  } catch (...) {
    cfg.present = false;
  }
  return cfg;
}

DenseCpuContract SafetensorsAdapter::apply_dense_config_json(const DenseCpuContract &contract,
                                                             const DenseCpuConfigJson &cfg) {
  if (!cfg.present)
    return contract;
  DenseCpuContract out = contract;
  if (cfg.n_layers > 0)
    out.n_layers = cfg.n_layers;
  if (cfg.n_heads > 0)
    out.n_heads = cfg.n_heads;
  if (cfg.n_kv_heads > 0)
    out.n_kv_heads = cfg.n_kv_heads;
  if (cfg.head_dim > 0)
    out.head_dim = cfg.head_dim;
  if (cfg.hidden > 0)
    out.hidden = cfg.hidden;
  if (cfg.intermediate > 0)
    out.intermediate = cfg.intermediate;
  if (cfg.vocab > 0)
    out.vocab = cfg.vocab;
  if (cfg.sliding_window > 0)
    out.sliding_window = cfg.sliding_window;
  if (cfg.eps > 0.0f)
    out.eps = cfg.eps;
  if (cfg.rope_theta > 0.0f)
    out.rope_theta = cfg.rope_theta;
  return out;
}

std::string SafetensorsAdapter::canonicalize_name(const std::string &raw_name,
                                                  const std::string &format_hint) {
  (void)format_hint;
  std::string name = strip_prefixes(raw_name);
  std::string lower = name;
  std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });

  const int layer = infer_layer_index(name);
  if (layer < 0) {
    // Top-level tensors: map to GGUF-style canonical names.
    if (lower.find("lm_head") != std::string::npos || lower == "output.weight" ||
        lower == "fc.weight")
      return "output.weight";
    if (lower.find("embed_tokens") != std::string::npos ||
        lower.find("tok_embeddings") != std::string::npos ||
        lower == "token_embd.weight")
      return "token_embd.weight";
    if (lower.find("norm.weight") != std::string::npos ||
        lower.find("final_norm") != std::string::npos)
      return "output_norm.weight";
    if (lower.find("hidden_norm") != std::string::npos)
      return "hidden_norm.weight";
    return name;
  }

  const std::string prefix = "blk." + std::to_string(layer) + ".";

  // Qwen3.6 hybrid names are already GGUF-style; keep them intact.
  if (lower.find("attn_qkv") != std::string::npos)
    return prefix + "attn_qkv.weight";
  if (lower.find("attn_gate") != std::string::npos)
    return prefix + "attn_gate.weight";
  if (lower.find("ssm_conv1d") != std::string::npos)
    return prefix + "ssm_conv1d.weight";
  if (lower.find("ssm_a") != std::string::npos)
    return prefix + "ssm_a";
  if (lower.find("ssm_beta") != std::string::npos)
    return prefix + "ssm_beta";
  if (lower.find("ssm_alpha") != std::string::npos)
    return prefix + "ssm_alpha";
  if (lower.find("ffn_gate_exps") != std::string::npos)
    return prefix + "ffn_gate_exps.weight";
  if (lower.find("ffn_up_exps") != std::string::npos)
    return prefix + "ffn_up_exps.weight";
  if (lower.find("ffn_down_exps") != std::string::npos)
    return prefix + "ffn_down_exps.weight";
  if (lower.find("ffn_up_shexp") != std::string::npos)
    return prefix + "ffn_up_shexp.weight";
  if (lower.find("ffn_down_shexp") != std::string::npos)
    return prefix + "ffn_down_shexp.weight";

  // HF-style dense layer names -> GGUF-style canonical names.
  if (lower.find("input_layernorm") != std::string::npos)
    return prefix + "attn_norm.weight";
  if (lower.find("post_attention_layernorm") != std::string::npos)
    return prefix + "ffn_norm.weight";
  if (lower.find("self_attn") != std::string::npos && lower.find("q_proj") != std::string::npos)
    return prefix + "attn_q.weight";
  if (lower.find("self_attn") != std::string::npos && lower.find("k_proj") != std::string::npos)
    return prefix + "attn_k.weight";
  if (lower.find("self_attn") != std::string::npos && lower.find("v_proj") != std::string::npos)
    return prefix + "attn_v.weight";
  if (lower.find("self_attn") != std::string::npos && lower.find("o_proj") != std::string::npos)
    return prefix + "attn_output.weight";
  if (lower.find("self_attn") != std::string::npos && lower.find("q_norm") != std::string::npos)
    return prefix + "attn_q_norm.weight";
  if (lower.find("self_attn") != std::string::npos && lower.find("k_norm") != std::string::npos)
    return prefix + "attn_k_norm.weight";
  if (lower.find("mlp.gate_proj") != std::string::npos)
    return prefix + "ffn_gate.weight";
  if (lower.find("mlp.up_proj") != std::string::npos)
    return prefix + "ffn_up.weight";
  if (lower.find("mlp.down_proj") != std::string::npos)
    return prefix + "ffn_down.weight";

  // Unknown layer tensor: keep stripped name but record it under this layer.
  return name;
}

std::string SafetensorsAdapter::canonicalize_qwen_name(const std::string &raw_name,
                                                      const std::string &format_hint) {
  // The canonical name space is now unified: GGUF-style names for both the
  // dense backend and the Qwen hybrid backend. Callers that need the older
  // layers.N.* logical names can use normalize_tensors() instead.
  return canonicalize_name(raw_name, format_hint);
}
