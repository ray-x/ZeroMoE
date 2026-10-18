#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Generic safetensors adapter boundary.
// The runtime consumes a normalized logical tensor map; the file-format
// specifics (safetensors parsing, alias normalization, DFlash/DSpark naming
// drift) stay isolated in this module so the same abstraction can later be used
// by the GPU path without duplicating format handling.

struct SafetensorsTensorInfo {
  std::string name;
  std::string file_path;
  std::vector<int64_t> shape;
  std::string dtype;
  // File offset and length of the raw tensor bytes.
  std::uint64_t offset = 0;
  std::uint64_t length = 0;
  // data_offsets from the safetensors header, relative to the start of the
  // tensor data section (i.e. after the 8-byte length prefix + header JSON).
  std::uint64_t data_offset = 0;
  std::uint64_t data_end = 0;
};

struct SafetensorsModelInfo {
  std::string format = "generic";
  std::unordered_map<std::string, SafetensorsTensorInfo> tensors;
  std::unordered_map<std::string, std::string> metadata;
};

struct NormalizedTensor {
  std::string logical_name;
  std::string canonical_name;
  std::vector<int64_t> shape;
  std::string dtype;
};

enum class SafetensorsFamily {
  Unknown,
  QwenHybridMoE, // Qwen3.6-A3B style hybrid linear-attention + dense MoE
  DenseGQA,      // Standard HF-style dense transformer with GQA (the real file)
  EXL3,          // ExLlamav3 trellis/codebook safetensors export
};

enum class Exl3Codebook {
  Default,
  MCG,
  MUL1,
};

struct Exl3LinearInfo {
  std::string key;
  std::string trellis_name;
  std::string suh_name;
  std::string svh_name;
  std::string su_name;
  std::string sv_name;
  std::string bias_name;
  std::string mcg_name;
  std::string mul1_name;
  std::vector<int64_t> trellis_shape;
  int64_t in_features = 0;
  int64_t out_features = 0;
  int64_t trellis_words = 0;
  double bits_per_weight = 0.0;
  Exl3Codebook codebook = Exl3Codebook::Default;
  bool valid = false;
};

struct Exl3Contract {
  std::vector<Exl3LinearInfo> linears;
  int n_layers = 0;
  int n_experts = 0;
  int hidden = 0;
  int intermediate = 0;
  bool complete = false;
};

struct QwenCpuContract {
  int n_layers = 0;
  int n_heads = 0;
  int n_kv_heads = 0;
  int head_dim = 0;
  int n_experts = 0;
  int topk = 0;
  int moe_inter = 0;
  int moe_inter_shared = 0;
  std::unordered_map<std::string, std::string> resolved_tensors;
};

struct DenseCpuContract {
  int n_layers = 0;
  int n_heads = 0;
  int n_kv_heads = 0;
  int head_dim = 0;
  int hidden = 0;
  int intermediate = 0;
  int vocab = 0;
  int sliding_window = 0;
  float eps = 1e-6f;
  float rope_theta = 10000.0f;
  float partial_rotary_factor = 1.0f;
  // Canonical-name -> raw-name map for the CPU loader.
  std::unordered_map<std::string, std::string> resolved_tensors;
};

// Optional companion JSON config for HF-style exports (e.g. DFlash
// config.json). Populated separately from the safetensors file and merged into
// the dense contract so the adapter can use ground-truth values instead of
// inferring them from tensor shapes.
struct DenseCpuConfigJson {
  bool present = false;
  int n_layers = 0;
  int n_heads = 0;
  int n_kv_heads = 0;
  int head_dim = 0;
  int hidden = 0;
  int intermediate = 0;
  int vocab = 0;
  int sliding_window = 0;
  float eps = 1e-6f;
  float rope_theta = 10000.0f;
  std::vector<std::string> layer_types;
};

class SafetensorsAdapter {
 public:
  // Parse a full safetensors file on disk.
  static bool parse_file(const std::string &path, SafetensorsModelInfo &out);

  // Parse all *.safetensors files in a model directory. Tensor names must be
  // unique across files; this mirrors ExLlamav3's collection loader.
  static bool parse_directory(const std::string &path, SafetensorsModelInfo &out);

  // Parse the file header bytes directly. The header is the JSON metadata blob
  // that precedes the raw tensor payload in a safetensors export.
  static bool parse_header_bytes(const std::string &header_json,
                                SafetensorsModelInfo &out);

  // Detect the likely family export (DFlash, DSpark, or generic)
  static std::string detect_format(const SafetensorsModelInfo &model);

  // Detect whether the tensor layout is a Qwen hybrid MoE or a dense HF model.
  static SafetensorsFamily detect_family(const SafetensorsModelInfo &model);

  // Build and validate the EXL3 tensor groups. An empty result means that no
  // EXL3 trellis tensors were present; incomplete groups are reported through
  // complete=false rather than silently treated as ordinary tensors.
  static Exl3Contract map_exl3_contract(const SafetensorsModelInfo &model);

  static bool is_exl3_model(const SafetensorsModelInfo &model);

  // Return all tensor names normalized to a generic canonical form. This is the
  // seam for DFlash/DSpark alias resolution.
  static std::vector<NormalizedTensor>
  normalize_tensors(const SafetensorsModelInfo &model,
                    const std::string &format_hint = "generic");

  // Convert the normalized tensor map into the Qwen CPU model contract used by
  // the host-only backend. This intentionally stays format-agnostic and only
  // maps to the logical Qwen contract.
  static QwenCpuContract map_qwen_cpu_contract(const SafetensorsModelInfo &model,
                                              const std::string &format_hint = "generic");

  // Convert the normalized tensor map into a dense CPU model contract.
  static DenseCpuContract map_dense_cpu_contract(const SafetensorsModelInfo &model,
                                                const std::string &format_hint = "generic");

  // Parse a DFlash/HF-style config.json file and return the populated view.
  static DenseCpuConfigJson parse_dense_config_json(const std::string &path);

  // Merge config.json ground-truth values into a dense contract.
  static DenseCpuContract apply_dense_config_json(const DenseCpuContract &contract,
                                                  const DenseCpuConfigJson &cfg);

  static std::string canonicalize_name(const std::string &raw_name,
                                       const std::string &format_hint = "generic");

  // Backward-compatible alias for the original Qwen-only API.
  static std::string canonicalize_qwen_name(const std::string &raw_name,
                                            const std::string &format_hint = "generic");
};
