#include "exl3_model_loader.hpp"

#include "exl3_cpu.hpp"
#include "kernels_cpu.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>
#include <memory>
#include <unordered_map>

namespace {

class SafetensorsShardHandles {
  struct File {
    int descriptor;
    uint64_t size;
    File(int descriptor_value, uint64_t size_value)
        : descriptor(descriptor_value), size(size_value) {}
    ~File() { if (descriptor >= 0) ::close(descriptor); }
  };
  std::mutex mutex_;
  std::unordered_map<std::string, std::shared_ptr<File>> files_;

  std::shared_ptr<File> open_file(const std::string &path) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto found = files_.find(path);
    if (found != files_.end()) return found->second;
    const int descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    struct stat metadata{};
    if (descriptor < 0 || fstat(descriptor, &metadata) != 0 || metadata.st_size < 0) {
      if (descriptor >= 0) ::close(descriptor);
      return {};
    }
    auto file = std::make_shared<File>(descriptor, static_cast<uint64_t>(metadata.st_size));
    return files_.emplace(path, std::move(file)).first->second;
  }

public:
  bool read(const std::string &path, uint64_t offset, uint8_t *destination,
            size_t bytes) {
    auto file = open_file(path);
    if (!file || offset > file->size || bytes > file->size - offset ||
        offset > static_cast<uint64_t>(std::numeric_limits<off_t>::max())) return false;
    while (bytes) {
      const ssize_t count = pread(file->descriptor, destination, bytes,
                                  static_cast<off_t>(offset));
      if (count < 0 && errno == EINTR) continue;
      if (count <= 0) return false;
      offset += static_cast<uint64_t>(count);
      destination += count;
      bytes -= static_cast<size_t>(count);
    }
    return true;
  }
};

SafetensorsShardHandles &shard_handles() {
  static SafetensorsShardHandles handles;
  return handles;
}

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
  bytes.resize(static_cast<size_t>(info.length));
  if (!shard_handles().read(info.file_path, info.offset, bytes.data(), bytes.size())) {
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

bool load_exl3_mul1_components(const SafetensorsModelInfo &model,
                               const Exl3LinearInfo &linear,
                               Exl3Mul1MatrixData &matrix,
                               std::string &error) {
  matrix = {};
  if (!linear.valid || linear.codebook != Exl3Codebook::MUL1 ||
      linear.suh_name.empty() || linear.svh_name.empty() ||
      linear.bits_per_weight < 1.0 || linear.bits_per_weight > 8.0 ||
      std::floor(linear.bits_per_weight) != linear.bits_per_weight) {
    error = "Unsupported EXL3 matrix contract: " + linear.key;
    return false;
  }

  if (!read_tensor(model, linear.trellis_name, matrix.trellis, error) ||
      !f16_tensor(model, linear.suh_name,
                  static_cast<int>(linear.in_features), matrix.suh, error) ||
      !f16_tensor(model, linear.svh_name,
                  static_cast<int>(linear.out_features), matrix.svh, error))
    return false;
  if (!linear.bias_name.empty() &&
      !f16_tensor(model, linear.bias_name,
                  static_cast<int>(linear.out_features), matrix.bias, error))
    return false;

  const size_t expected_trellis =
      static_cast<size_t>(linear.in_features / 16) *
      (linear.out_features / 16) * 16 *
      static_cast<int>(linear.bits_per_weight) * sizeof(uint16_t);
  if (matrix.trellis.size() != expected_trellis) {
    error = "EXL3 trellis byte size mismatch: " + linear.trellis_name;
    return false;
  }
  matrix.input_features = static_cast<int>(linear.in_features);
  matrix.output_features = static_cast<int>(linear.out_features);
  matrix.bits = static_cast<int>(linear.bits_per_weight);
  return true;
}

bool load_exl3_mul1_matrix(const SafetensorsModelInfo &model,
                           const Exl3LinearInfo &linear,
                           std::vector<uint8_t> &packed,
                           std::string &error) {
  packed.clear();
  Exl3Mul1MatrixData data;
  if (!load_exl3_mul1_components(model, linear, data, error))
    return false;
  const Exl3Mul1MatrixView matrix = data.view();
  packed = exl3_mul1_pack_matrix(matrix);
  if (packed.empty()) {
    error = "Failed to pack EXL3 matrix: " + linear.key;
    return false;
  }
  return true;
}

struct Exl3CpuMatrixCache::Impl {
  struct Entry {
    std::shared_ptr<const Exl3Mul1MatrixData> matrix;
    std::list<std::string>::iterator lru;
  };
  size_t budget;
  size_t resident_bytes = 0;
  uint64_t hits = 0;
  uint64_t misses = 0;
  uint64_t evictions = 0;
  std::unordered_map<std::string, Entry> matrices;
  std::list<std::string> lru;
  mutable std::mutex mutex;

  explicit Impl(size_t budget_bytes) : budget(budget_bytes) {}
};

Exl3CpuMatrixCache::Exl3CpuMatrixCache(size_t budget_bytes)
    : impl_(std::make_unique<Impl>(budget_bytes)) {}
Exl3CpuMatrixCache::~Exl3CpuMatrixCache() = default;

std::shared_ptr<const Exl3Mul1MatrixData> Exl3CpuMatrixCache::get(
    const std::string &key) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  auto found = impl_->matrices.find(key);
  if (found == impl_->matrices.end()) {
    ++impl_->misses;
    return {};
  }
  ++impl_->hits;
  impl_->lru.splice(impl_->lru.begin(), impl_->lru, found->second.lru);
  return found->second.matrix;
}

std::shared_ptr<const Exl3Mul1MatrixData> Exl3CpuMatrixCache::insert(
    const std::string &key, Exl3Mul1MatrixData matrix) {
  auto result = std::make_shared<Exl3Mul1MatrixData>(std::move(matrix));
  const size_t bytes = result->byte_size();
  std::lock_guard<std::mutex> lock(impl_->mutex);
  auto found = impl_->matrices.find(key);
  if (found != impl_->matrices.end()) {
    ++impl_->hits;
    impl_->lru.splice(impl_->lru.begin(), impl_->lru, found->second.lru);
    return found->second.matrix;
  }
  if (bytes == 0 || bytes > impl_->budget) return result;
  while (impl_->resident_bytes > impl_->budget - bytes) {
    auto victim = impl_->lru.end();
    for (auto candidate = impl_->lru.rbegin(); candidate != impl_->lru.rend(); ++candidate) {
      const auto entry = impl_->matrices.find(*candidate);
      if (entry != impl_->matrices.end() && entry->second.matrix.use_count() == 1) {
        victim = std::prev(candidate.base());
        break;
      }
    }
    if (victim == impl_->lru.end()) return result;
    const std::string old_key = *victim;
    impl_->resident_bytes -= impl_->matrices.at(old_key).matrix->byte_size();
    impl_->matrices.erase(old_key);
    impl_->lru.erase(victim);
    ++impl_->evictions;
  }
  impl_->lru.push_front(key);
  impl_->matrices.emplace(key, Impl::Entry{result, impl_->lru.begin()});
  impl_->resident_bytes += bytes;
  return result;
}

Exl3CpuMatrixCacheStats Exl3CpuMatrixCache::stats() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return {impl_->hits, impl_->misses, impl_->evictions, impl_->resident_bytes};
}

struct Exl3NgramReader::Impl {
  struct File {
    int descriptor = -1;
    uint64_t bytes = 0;
    explicit File(const std::string &path) {
      descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
      struct stat metadata{};
      if (descriptor < 0 || fstat(descriptor, &metadata) != 0 || metadata.st_size < 0) {
        if (descriptor >= 0) ::close(descriptor);
        throw std::runtime_error("Cannot open EXL3 ngram shard: " + path);
      }
      bytes = static_cast<uint64_t>(metadata.st_size);
    }
    ~File() { if (descriptor >= 0) ::close(descriptor); }
    bool read(uint64_t offset, void *output, size_t count) const {
      if (offset > bytes || count > bytes - offset ||
          offset > static_cast<uint64_t>(std::numeric_limits<off_t>::max())) return false;
      auto *destination = static_cast<uint8_t *>(output);
      while (count > 0) {
        const ssize_t result = pread(descriptor, destination, count, static_cast<off_t>(offset));
        if (result < 0 && errno == EINTR) continue;
        if (result <= 0) return false;
        destination += result;
        offset += static_cast<uint64_t>(result);
        count -= static_cast<size_t>(result);
      }
      return true;
    }
  };
  struct Segment {
    SafetensorsTensorInfo tensor;
    File *file = nullptr;
  };
  std::unordered_map<std::string, std::unique_ptr<File>> files;
  std::vector<Segment> segments;
  std::vector<uint16_t> bias;
  std::vector<int64_t> cached_rows;
  std::vector<std::array<float, 160>> decoded_rows;
  uint64_t reads = 0;
  uint64_t hits = 0;
  mutable std::mutex mutex;

  File &file(const SafetensorsTensorInfo &tensor) {
    auto found = files.find(tensor.file_path);
    if (found == files.end())
      found = files.emplace(tensor.file_path, std::make_unique<File>(tensor.file_path)).first;
    if (tensor.offset > found->second->bytes || tensor.length > found->second->bytes - tensor.offset)
      throw std::runtime_error("EXL3 ngram tensor exceeds shard: " + tensor.name);
    return *found->second;
  }

  explicit Impl(const SafetensorsModelInfo &model) {
    const SafetensorsTensorInfo *bias_tensor = nullptr;
    for (const auto &[name, tensor] : model.tensors) {
      if (name.find("ngram_embedding") == std::string::npos) continue;
      if (name.ends_with(".head_bias")) {
        if (bias_tensor) throw std::runtime_error("Duplicate EXL3 ngram head bias");
        bias_tensor = &tensor;
      }
      if (name.ends_with(".trellis")) {
        if (tensor.dtype != "I16" || tensor.shape.size() != 2 || tensor.shape[0] <= 0 ||
            tensor.shape[1] < 11 || tensor.shape[1] > 81 || (tensor.shape[1] - 1) % 10 ||
            static_cast<uint64_t>(tensor.shape[0]) > std::numeric_limits<uint64_t>::max() /
                (static_cast<uint64_t>(tensor.shape[1]) * 2) ||
            tensor.length != static_cast<uint64_t>(tensor.shape[0]) * tensor.shape[1] * 2)
          throw std::runtime_error("Invalid EXL3 ngram trellis: " + name);
        segments.push_back({tensor, &file(tensor)});
      }
    }
    if (segments.empty() || !bias_tensor || bias_tensor->dtype != "F16" ||
        bias_tensor->shape.size() != 2 || bias_tensor->shape[0] <= 0 ||
        bias_tensor->shape[1] != 160 ||
        static_cast<uint64_t>(bias_tensor->shape[0]) > std::numeric_limits<size_t>::max() / 320 ||
        bias_tensor->length != static_cast<uint64_t>(bias_tensor->shape[0]) * 320)
      throw std::runtime_error("Missing or invalid EXL3 ngram tables/bias");
    std::sort(segments.begin(), segments.end(), [](const Segment &left, const Segment &right) {
      if (left.tensor.data_offset != right.tensor.data_offset)
        return left.tensor.data_offset < right.tensor.data_offset;
      return left.tensor.name < right.tensor.name;
    });
    bias.resize(static_cast<size_t>(bias_tensor->length / 2));
    if (!file(*bias_tensor).read(bias_tensor->offset, bias.data(), bias_tensor->length))
      throw std::runtime_error("Short EXL3 ngram bias read");
    cached_rows.assign(static_cast<size_t>(bias_tensor->shape[0]), -1);
    decoded_rows.resize(cached_rows.size());
  }
};

Exl3NgramReader::Exl3NgramReader(const SafetensorsModelInfo &model)
    : impl_(std::make_unique<Impl>(model)) {}
Exl3NgramReader::~Exl3NgramReader() = default;

bool Exl3NgramReader::read_row(int head, int64_t row, float *output, int dimensions) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!output || row < 0 || head < 0 || static_cast<size_t>(head) >= impl_->cached_rows.size() ||
      dimensions != 160) return false;
  auto &decoded = impl_->decoded_rows[head];
  if (impl_->cached_rows[head] == row) {
    std::copy(decoded.begin(), decoded.end(), output);
    ++impl_->hits;
    return true;
  }
  int64_t local_row = row;
  const Impl::Segment *segment = nullptr;
  for (const auto &candidate : impl_->segments) {
    if (local_row < candidate.tensor.shape[0]) { segment = &candidate; break; }
    local_row -= candidate.tensor.shape[0];
  }
  if (!segment) return false;
  const int words = static_cast<int>(segment->tensor.shape[1]);
  const int bits = (words - 1) / 10;
  std::array<uint16_t, 81> packed{};
  if (!segment->file->read(segment->tensor.offset + static_cast<uint64_t>(local_row) * words * 2,
                            packed.data(), words * 2)) return false;
  const float scale = fp16_to_fp32(packed[0]);
  for (int index = 0; index < dimensions; ++index) {
    uint32_t state = 0;
    for (int bit = 0; bit < 16; ++bit) {
      const int position = (index - bit / bits + dimensions) % dimensions;
      const int stream_bit = position * bits + bit % bits;
      state |= ((packed[1 + stream_bit / 16] >> (stream_bit % 16)) & 1u) << bit;
    }
    const uint32_t product = state * 0x83DCD12Du;
    const int sum = (product & 255u) + ((product >> 8) & 255u) +
                    ((product >> 16) & 255u) + (product >> 24);
    const float code = fp16_to_fp32(fp32_to_fp16(std::fma(
        static_cast<float>(1024 + sum), fp16_to_fp32(0x1eee), fp16_to_fp32(0xc931))));
    decoded[index] = fp16_to_fp32(fp32_to_fp16(code * scale +
        fp16_to_fp32(impl_->bias[static_cast<size_t>(head) * dimensions + index])));
  }
  impl_->cached_rows[head] = row;
  std::copy(decoded.begin(), decoded.end(), output);
  ++impl_->reads;
  return true;
}

uint64_t Exl3NgramReader::row_reads() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->reads;
}
uint64_t Exl3NgramReader::cache_hits() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->hits;
}
