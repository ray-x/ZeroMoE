// src/cpu_qwen4exp.cpp — CPU-only Qwen3.8-Flash-Next forward pass.
#include "cpu_qwen4exp.hpp"

#include "qwen4exp_gdn.hpp"
#include "ple_hash.hpp"
#include "rope.hpp"
#include "exl3_cpu.hpp"
#ifdef ZEROMOE_EXL3_CUDA
#include "exl3_gpu.hpp"
#endif
#if defined(ZEROMOE_USE_GGML_CUDA)
#include "exl3_ggml_model.hpp"
#endif
#include <stdexcept>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <fcntl.h>
#include <cstring>
#include <numeric>
#include <sys/mman.h>
#include <unistd.h>

extern "C" int ggml_dequant_row(uint32_t type, const void *src, float *out,
                                int n);

namespace {

static inline float bf16_to_float(uint16_t v) {
  uint32_t bits = static_cast<uint32_t>(v) << 16;
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

bool try_exl3_gpu_hc_mix(const CPUEngine &engine,
                          const std::string &down_name, const std::string &up_name,
                          const std::string &inject_name, const float *normed,
                          float *hidden, float *injection, int rows, int width,
                          int streams, int rank, bool divide_each_stream) {
#ifdef ZEROMOE_EXL3_CUDA
  if (!engine.exl3_gpu || !engine.config.exl3_gpu_hc) return false;
  const auto *down = engine.find_layer(down_name);
  const auto *up = engine.find_layer(up_name);
  const auto *inject = inject_name.empty() ? nullptr : engine.find_layer(inject_name);
  const auto matches = [](const CPUQuantizedTensor *weight, int input, int output) {
    return weight && weight->host_ptr && weight->meta.type == GGUFType::F32 &&
           weight->meta.shape.size() == 2 &&
           weight->meta.shape[0] == static_cast<uint64_t>(input) &&
           weight->meta.shape[1] == static_cast<uint64_t>(output) &&
           weight->host_size == static_cast<size_t>(input) * output * sizeof(float);
  };
  const int expanded_width = streams * width;
  if (!matches(down, expanded_width, rank) || !matches(up, rank, expanded_width) ||
      (!inject_name.empty() && !matches(inject, expanded_width, streams)))
    return false;
  const Exl3GpuF32Weight down_weight{
      down_name, reinterpret_cast<const float *>(down->host_ptr), expanded_width, rank};
  const Exl3GpuF32Weight up_weight{
      up_name, reinterpret_cast<const float *>(up->host_ptr), rank, expanded_width};
  Exl3GpuF32Weight inject_weight;
  if (inject)
    inject_weight = {inject_name, reinterpret_cast<const float *>(inject->host_ptr),
                     expanded_width, streams};
  engine.exl3_gpu->hc_mix(down_weight, up_weight, inject ? &inject_weight : nullptr,
                          normed, hidden, injection, rows, width, streams,
                          divide_each_stream);
  if (!std::all_of(hidden, hidden + static_cast<size_t>(rows) * width,
                    [](float value) { return std::isfinite(value); }) ||
      (inject && !std::all_of(injection, injection + static_cast<size_t>(rows) * streams,
                               [](float value) { return std::isfinite(value); })))
    throw std::runtime_error("Non-finite EXL3 GPU HyperConnection output: " + down_name);
  return true;
#else
  return false;
#endif
}

} // namespace

CPUQwen4ExpBackend::~CPUQwen4ExpBackend() {
  if (ple_mapping_)
    munmap(ple_mapping_, ple_map_length_);
}

CPULayerType CPUQwen4ExpBackend::layer_type(int layer_idx) const {
  if (layer_idx < 0 || layer_idx >= static_cast<int>(layer_types.size()))
    return CPULayerType::DenseQKV;
  return layer_types[static_cast<size_t>(layer_idx)];
}

const float *CPUQwen4ExpBackend::tensor(const CPUEngine &engine,
                                        const std::string &name) {
  const auto *t = engine.find_layer(name);
  return t && t->host_ptr && t->meta.type == GGUFType::F32
             ? reinterpret_cast<const float *>(t->host_ptr)
             : nullptr;
}

bool CPUQwen4ExpBackend::dequant_row(const CPUQuantizedTensor &t, int row,
                                     float *out, int n) {
  if (!t.host_ptr || row < 0 || n <= 0)
    return false;
  int rows = 1;
  if (t.meta.shape.size() >= 2)
    rows = static_cast<int>(t.meta.shape[1]);
  if (rows <= 0 || static_cast<size_t>(row) >= static_cast<size_t>(rows))
    return false;
  size_t row_bytes = t.host_size / static_cast<size_t>(rows);
  const uint8_t *src = t.host_ptr + static_cast<size_t>(row) * row_bytes;
  if (t.meta.type == GGUFType::F32) {
    std::memcpy(out, src, static_cast<size_t>(n) * sizeof(float));
    return true;
  }

  if (t.meta.type == GGUFType::F16) {
    const auto *p = reinterpret_cast<const uint16_t *>(src);
    for (int i = 0; i < n; ++i)
      out[i] = fp16_to_fp32(p[i]);
    return true;
  }
  if (t.meta.type == GGUFType::BF16) {
    const auto *p = reinterpret_cast<const uint16_t *>(src);
    for (int i = 0; i < n; ++i)
      out[i] = bf16_to_float(p[i]);
    return true;
  }
  return ggml_dequant_row(static_cast<uint32_t>(t.meta.type), src, out, n) !=
         0;
}

bool CPUQwen4ExpBackend::dequant_tensor(const CPUQuantizedTensor &t, float *out,
                                        size_t n) {
  if (!t.host_ptr || n == 0)
    return false;
  if (t.meta.type == GGUFType::F32) {
    std::memcpy(out, t.host_ptr, n * sizeof(float));
    return true;
  }
  int rows = t.meta.shape.size() >= 2 ? static_cast<int>(t.meta.shape[1]) : 1;
  int row_n = rows > 0 ? static_cast<int>(n / static_cast<size_t>(rows)) : 0;
  if (row_n <= 0 || static_cast<size_t>(row_n) * rows != n)
    return false;
  for (int r = 0; r < rows; ++r)
    if (!dequant_row(t, r, out + static_cast<size_t>(r) * row_n, row_n))
      return false;
  return true;
}

bool CPUQwen4ExpBackend::dense_matmul(CPUEngine &engine,
                                      const std::string &name, const float *x,
                                      float *y, int S, int I, int O) {
  const auto *t = engine.find_layer(name);
  if (!t || !t->host_ptr || S <= 0 || I <= 0 || O <= 0) {
    if (!engine.exl3_model.tensors.empty())
      throw std::runtime_error("Missing EXL3 projection: " + name);
    return false;
  }
  if (!engine.exl3_model.tensors.empty() &&
      (t->meta.shape.size() != 2 || t->meta.shape[0] != static_cast<uint64_t>(I) ||
       t->meta.shape[1] != static_cast<uint64_t>(O)))
    throw std::runtime_error("EXL3 projection shape mismatch: " + name);
#ifdef ZEROMOE_EXL3_CUDA
  if (engine.exl3_gpu) {
#if defined(ZEROMOE_USE_GGML_CUDA)
    const std::string graph_key = std::to_string(S) + ":" +
                                  std::to_string(I) + ":" +
                                  std::to_string(O);
    auto &graph = engine.exl3_ggml_dense_graphs[graph_key];
    if (!graph) {
      auto candidate = std::make_shared<Exl3GgmlDenseMatmulGraph>();
      if (candidate->initialize(engine.exl3_gpu.get(), S, I, O))
        graph = std::move(candidate);
    }
    if (graph && graph->run(name, static_cast<uint32_t>(t->meta.type),
                            t->host_ptr, t->host_size, x, y)) {
      for (size_t index = 0; index < static_cast<size_t>(S) * O; ++index)
        if (!std::isfinite(y[index]))
          throw std::runtime_error("Non-finite EXL3 GGML projection: " + name);
      return true;
    }
#endif
    engine.exl3_gpu->matmul(name, static_cast<uint32_t>(t->meta.type), t->host_ptr,
                            t->host_size, x, y, S, I, O);
    for (size_t index = 0; index < static_cast<size_t>(S) * O; ++index)
      if (!std::isfinite(y[index]))
        throw std::runtime_error("Non-finite EXL3 GPU projection: " + name);
    return true;
  }
#endif
  if (t->meta.type == GGUFType::EXL3) {
    const bool success = exl3_mul1_matmul_packed(t->host_ptr, x, y, S, I, O);
    if (!success) throw std::runtime_error("EXL3 dense matmul failed: " + name);
    for (size_t index = 0; index < static_cast<size_t>(S) * O; ++index)
      if (!std::isfinite(y[index]))
        throw std::runtime_error("Non-finite EXL3 projection: " + name);
    return true;
  }
  if (t->meta.type == GGUFType::F32) {
    matmul(y, x, reinterpret_cast<const float *>(t->host_ptr), S, I, O);
    for (int i = 0; i < S * O; ++i)
      if (!std::isfinite(y[i])) {
        if (!engine.exl3_model.tensors.empty())
          throw std::runtime_error("Non-finite EXL3 dense projection: " + name);
        y[i] = 0.0f;
      }
    return true;
  }
  std::vector<float> row(static_cast<size_t>(I));
  const int rows = t->meta.shape.size() >= 2
                       ? static_cast<int>(t->meta.shape[1])
                       : 1;
  if (rows != O)
    return false;
  std::fill(y, y + static_cast<size_t>(S) * O, 0.0f);
  for (int o = 0; o < O; ++o) {
    if (!dequant_row(*t, o, row.data(), I))
      return false;
    for (int s = 0; s < S; ++s) {
      const float *xs = x + static_cast<size_t>(s) * I;
      float acc = 0.0f;
      for (int i = 0; i < I; ++i)
        acc += xs[i] * row[i];
      y[static_cast<size_t>(s) * O + o] = acc;
    }
  }
  for (int i = 0; i < S * O; ++i)
    if (!std::isfinite(y[i]))
      y[i] = 0.0f;
  return true;
}

void CPUQwen4ExpBackend::rmsnorm(float *out, const float *x, const float *w,
                                 int n, float eps) {
  double ss = 0.0;
  for (int i = 0; i < n; ++i)
    ss += static_cast<double>(x[i]) * x[i];
  const float inv = 1.0f / std::sqrt(static_cast<float>(ss / n) + eps);
  for (int i = 0; i < n; ++i)
    out[i] = x[i] * inv * (w ? w[i] : 1.0f);
}

void CPUQwen4ExpBackend::l2norm(float *x, int n, float eps) {
  double ss = 0.0;
  for (int i = 0; i < n; ++i)
    ss += static_cast<double>(x[i]) * x[i];
  const float inv = 1.0f / std::max(static_cast<float>(std::sqrt(ss)), eps);
  for (int i = 0; i < n; ++i)
    x[i] *= inv;
}

void CPUQwen4ExpBackend::silu(float *x, int n) {
  for (int i = 0; i < n; ++i)
    x[i] /= 1.0f + std::exp(-x[i]);
}

void CPUQwen4ExpBackend::sigmoid(float *x, int n) {
  for (int i = 0; i < n; ++i)
    x[i] = 1.0f / (1.0f + std::exp(-x[i]));
}

// Qwen4Exp's upstream rope_type is IMROPE, whose cos/sin pairing is
// NEOX-ordered (rotate-half: pairs (j, j+half)), not the adjacent-pair
// (2j,2j+1) ordering this function originally implemented under its old
// name (rope_interleaved). For text-only decode every M-RoPE section
// carries the same scalar position, so IMROPE's per-section theta
// selection is provably identical to plain single-axis NEOX rope (see the
// matching fix + comment in build_full_attention() in
// ggml_qwen4exp_graph.cpp for the full derivation).
void CPUQwen4ExpBackend::rope_neox(float *x, int n_rot, int pos,
                                   float theta) {
  const int half = n_rot / 2;
  const float scale = std::pow(theta, -2.0f / n_rot);
  float angle = static_cast<float>(pos);
  for (int j = 0; j < half; ++j) {
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    const float a = x[j];
    const float b = x[j + half];
    x[j] = a * c - b * s;
    x[j + half] = a * s + b * c;
    angle *= scale;
  }
}

void CPUQwen4ExpBackend::softmax(float *x, int n) {
  const float mx = *std::max_element(x, x + n);
  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    x[i] = std::exp(x[i] - mx);
    sum += x[i];
  }
  if (sum > 0.0)
    for (int i = 0; i < n; ++i)
      x[i] = static_cast<float>(x[i] / sum);
}

bool CPUQwen4ExpBackend::compatible(const GGUFFile &model) const {
  const auto &c = model.architecture_config;
  return c.n_layers > 0 && c.hidden > 0 && c.n_experts > 0 && c.topk > 0 &&
         c.hc_mult > 1 && c.hc_low_rank > 0 && c.ssm_inner_size > 0 &&
         c.ssm_state_size > 0 && c.ssm_group_count > 0 &&
         c.ssm_time_step_rank > 0;
}

void CPUQwen4ExpBackend::hc_mix(CPUEngine &engine, int layer_idx,
                                int sub_block, float *hidden_states, int S) {
  const auto &c = engine.model_file.architecture_config;
  const int D = c.hidden;
  const int HC = c.hc_mult;
  const int HCD = HC * D;
  const std::string p = "blk." + std::to_string(layer_idx) + ".";
  const std::string base = sub_block == 0 ? "hc_attn_" : "hc_ffn_";

  std::vector<float> normed(static_cast<size_t>(S) * HCD);
  const float *norm_w = tensor(engine, p + base + "norm.weight");
  for (int s = 0; s < S; ++s)
    for (int h = 0; h < HC; ++h)
      rmsnorm(normed.data() + (static_cast<size_t>(s) * HC + h) * D,
              engine.qwen4exp.hc_stream_buf.data() +
                  (static_cast<size_t>(s) * HC + h) * D,
              norm_w ? norm_w + h * D : nullptr, D, c.eps);

  engine.qwen4exp.hc_inject.resize(static_cast<size_t>(S) * HC);
  if (try_exl3_gpu_hc_mix(engine, p + base + "down.weight", p + base + "up.weight",
                           p + base + "inject.weight", normed.data(), hidden_states,
                           engine.qwen4exp.hc_inject.data(), S, D, HC,
                           c.hc_low_rank, true))
    return;

  std::vector<float> low(static_cast<size_t>(S) * c.hc_low_rank);
  std::vector<float> gate(static_cast<size_t>(S) * HCD);
  if (!dense_matmul(engine, p + base + "down.weight", normed.data(),
                    low.data(), S, HCD, c.hc_low_rank)) {
    LOG_ERROR("[CPU-Q4E] Missing or invalid HC tensors at layer %d\n",
              layer_idx);
    return;
  }
  for (float &v : low)
    v /= static_cast<float>(HC);
  silu(low.data(), static_cast<int>(low.size()));
  if (!dense_matmul(engine, p + base + "up.weight", low.data(), gate.data(), S,
                    c.hc_low_rank, HCD)) {
    LOG_ERROR("[CPU-Q4E] Missing or invalid HC tensors at layer %d\n",
              layer_idx);
    return;
  }
  sigmoid(gate.data(), static_cast<int>(gate.size()));

  for (int s = 0; s < S; ++s) {
    float *dst = hidden_states + static_cast<size_t>(s) * D;
    std::fill(dst, dst + D, 0.0f);
    for (int h = 0; h < HC; ++h)
      for (int d = 0; d < D; ++d)
        dst[d] += normed[(static_cast<size_t>(s) * HC + h) * D + d] *
                  gate[(static_cast<size_t>(s) * HC + h) * D + d] /
                  static_cast<float>(HC);
  }
  dense_matmul(engine, p + base + "inject.weight", normed.data(),
               engine.qwen4exp.hc_inject.data(), S, HCD, HC);
}

void CPUQwen4ExpBackend::hc_combine(CPUEngine &engine, float *hidden_states,
                                     const float *sub_block_out, int S) {
  const auto &c = engine.model_file.architecture_config;
  const int D = c.hidden;
  const int HC = c.hc_mult;
  for (int s = 0; s < S; ++s) {
    const float *inject =
        engine.qwen4exp.hc_inject.data() + static_cast<size_t>(s) * HC;
    for (int h = 0; h < HC; ++h) {
      const float weight =
          2.0f / (1.0f + std::exp(-inject[h] / static_cast<float>(HC)));
      float *stream = engine.qwen4exp.hc_stream_buf.data() +
                      (static_cast<size_t>(s) * HC + h) * D;
      const float *out = sub_block_out + static_cast<size_t>(s) * D;
      for (int d = 0; d < D; ++d)
        stream[d] += weight * out[d];
    }
    std::memcpy(hidden_states + static_cast<size_t>(s) * D,
                engine.qwen4exp.hc_stream_buf.data() +
                    static_cast<size_t>(s) * HC * D,
                static_cast<size_t>(D) * sizeof(float));
  }
}

bool CPUQwen4ExpBackend::ple_layer_input(CPUEngine &engine, int layer_idx,
                                             int S, int pos_base) {
      const auto &c = engine.model_file.architecture_config;
      if (!qwen4exp_is_ple_layer(c, layer_idx))
        return true;
      const int D = c.hidden;
      const int HC = c.hc_mult;
      const int n_heads = c.ple_n_heads;
      const int head_dim = c.ple_n_embd_per_layer;
      if (n_heads <= 0 || head_dim <= 0 || n_heads * head_dim != D ||
          c.ple_ngram_size < 2)
        return false;

      if (!ple_table_ && engine.exl3_model.tensors.empty()) {
        auto it = engine.model_file.tensors.find("per_layer_token_embd.weight");
        if (it == engine.model_file.tensors.end() ||
            it->second.shard_idx >=
                static_cast<int>(engine.model_file.shard_paths.size()))
          return false;
        const auto &t = it->second;
        if (t.shape.size() != 2)
          return false;
        ple_row_dim_ = static_cast<int>(t.shape[0]);
        const uint64_t n_rows = t.shape[1];
        if (ple_row_dim_ != head_dim)
          return false;
        if (t.type == GGUFType::F32)
          ple_row_bytes_ = static_cast<size_t>(head_dim) * sizeof(float);
        else if (t.type == GGUFType::F16 || t.type == GGUFType::BF16)
          ple_row_bytes_ = static_cast<size_t>(head_dim) * sizeof(uint16_t);
        else
          return false;
        const uint64_t align = 4096;
        const uint64_t base = t.offset & ~(align - 1);
        const uint64_t delta = t.offset - base;
        ple_map_length_ = static_cast<size_t>(delta + t.size_bytes);
        const int fd = ::open(
            engine.model_file.shard_paths[static_cast<size_t>(t.shard_idx)].c_str(),
            O_RDONLY);
        if (fd < 0)
          return false;
        ple_mapping_ =
            mmap(nullptr, ple_map_length_, PROT_READ, MAP_SHARED, fd, base);
        ::close(fd);
        if (ple_mapping_ == MAP_FAILED) {
          ple_mapping_ = nullptr;
          return false;
        }
        madvise(ple_mapping_, ple_map_length_, MADV_RANDOM);
        ple_table_ = static_cast<const uint8_t *>(ple_mapping_) + delta;
        ple_type_ = t.type;
        ple_rows_ = n_rows;
      }

      if (engine.qwen4exp.hc_stream_buf.size() !=
          static_cast<size_t>(S) * HC * D)
        return false;
      const std::string p = "blk." + std::to_string(layer_idx) + ".";
      std::vector<float> emb(static_cast<size_t>(S) * D);
      std::vector<float> row(static_cast<size_t>(head_dim));
      const int eos = c.ple_eos_token_id;
      for (int s = 0; s < S; ++s) {
        const int abs_pos = pos_base + s;
        const int current =
            abs_pos >= 0 &&
                    abs_pos < static_cast<int>(engine.qwen4exp.ple_token_history.size())
                ? engine.qwen4exp.ple_token_history[static_cast<size_t>(abs_pos)]
                : (c.ple_image_token_id != 0 ? c.ple_image_token_id : eos);
        std::vector<int64_t> ctx(static_cast<size_t>(c.ple_ngram_size), eos);
        ctx[0] = current;
        for (int back = 1; back < c.ple_ngram_size; ++back) {
          const int prior = abs_pos - back;
          if (prior >= 0 &&
              prior < static_cast<int>(engine.qwen4exp.ple_token_history.size()))
            ctx[static_cast<size_t>(back)] =
                engine.qwen4exp.ple_token_history[static_cast<size_t>(prior)];
        }
        for (int n = 2; n <= c.ple_ngram_size; ++n) {
          for (int g = 0; g < c.ple_heads_per_ngram; ++g) {
            const int h = (n - 2) * c.ple_heads_per_ngram + g;
            int64_t mixed = ple_ngram_row(
                ctx.data(), n, c.ple_layer_multipliers.data(),
                c.ple_head_vocab_sizes[static_cast<size_t>(h)],
                c.ple_head_offsets[static_cast<size_t>(h)]);
        if (!engine.exl3_model.tensors.empty()) {
          if (!engine.exl3_ngram_row(h, mixed, row.data(), head_dim))
            throw std::runtime_error("EXL3 PLE embedding read failed");
        } else {
        if (mixed < 0 || static_cast<uint64_t>(mixed) >= ple_rows_)
          return false;
            const uint8_t *src =
                ple_table_ + static_cast<size_t>(mixed) * ple_row_bytes_;
            if (ple_type_ == GGUFType::F32)
              std::memcpy(row.data(), src, row.size() * sizeof(float));
            else if (ple_type_ == GGUFType::F16) {
              const auto *v = reinterpret_cast<const uint16_t *>(src);
              for (int i = 0; i < head_dim; ++i)
                row[static_cast<size_t>(i)] = fp16_to_fp32(v[i]);
            } else {
              const auto *v = reinterpret_cast<const uint16_t *>(src);
              for (int i = 0; i < head_dim; ++i)
                row[static_cast<size_t>(i)] = bf16_to_float(v[i]);
            }
        }
            std::memcpy(emb.data() + static_cast<size_t>(s) * D + h * head_dim,
                        row.data(), row.size() * sizeof(float));
          }
        }
      }
      std::vector<float> key(static_cast<size_t>(S) * HC * D);
      std::vector<float> value(static_cast<size_t>(S) * D);
      if (!dense_matmul(engine, p + "ple_key.weight", emb.data(), key.data(), S,
                        D, HC * D) ||
          !dense_matmul(engine, p + "ple_value.weight", emb.data(), value.data(),
                        S, D, D))
        return false;
      auto grouped_norm = [&](const float *in, const float *w, float *out) {
        for (int s = 0; s < S; ++s)
          for (int h = 0; h < HC; ++h)
            rmsnorm(out + (static_cast<size_t>(s) * HC + h) * D,
                    in + (static_cast<size_t>(s) * HC + h) * D,
                    w ? w + h * D : nullptr, D, c.eps);
      };
      std::vector<float> nk(static_cast<size_t>(HC) * D);
      std::vector<float> nq(nk.size()), nc(nk.size());
      const auto *nkt = engine.find_layer(p + "ple_norm_key.weight");
      const auto *nqt = engine.find_layer(p + "ple_norm_query.weight");
      const auto *nct = engine.find_layer(p + "ple_norm_conv.weight");
      if (!nkt || !nqt || !nct || !dequant_tensor(*nkt, nk.data(), nk.size()) ||
          !dequant_tensor(*nqt, nq.data(), nq.size()) ||
          !dequant_tensor(*nct, nc.data(), nc.size()))
        return false;
      std::vector<float> kn(key.size()), query(key.size());
      grouped_norm(key.data(), nk.data(), kn.data());
      grouped_norm(engine.qwen4exp.hc_stream_buf.data(), nq.data(), query.data());
      std::vector<float> gated(key.size());
      for (int s = 0; s < S; ++s) {
        for (int h = 0; h < HC; ++h) {
        double dot = 0.0;
        const size_t base = (static_cast<size_t>(s) * HC + h) * D;
        for (int i = 0; i < D; ++i)
          dot += static_cast<double>(kn[base + i]) * query[base + i];
        const float score = static_cast<float>(dot) / std::sqrt(static_cast<float>(D));
        const float mag = std::sqrt(std::max(std::fabs(score), 1e-6f));
        const float g = 1.0f / (1.0f + std::exp(-std::copysign(mag, score)));
          for (int d = 0; d < D; ++d)
            gated[(static_cast<size_t>(s) * HC + h) * D + d] =
                value[static_cast<size_t>(s) * D + d] * g;
        }
      }
      const auto *conv_t = engine.find_layer(p + "ple_conv1d.weight");
      if (!conv_t)
        return false;
      const int kernel = c.ple_conv_kernel;
      std::vector<float> conv_w(static_cast<size_t>(HC * D) * kernel);
      if (!dequant_tensor(*conv_t, conv_w.data(), conv_w.size()))
        return false;
      std::vector<float> conv_in(gated.size());
      grouped_norm(gated.data(), nc.data(), conv_in.data());
      const int hist_len = (kernel - 1) * c.ple_ngram_size;
      if (engine.qwen4exp.ple_conv_history.size() <
          static_cast<size_t>(c.n_layers))
        engine.qwen4exp.ple_conv_history.resize(static_cast<size_t>(c.n_layers));
      auto &hist = engine.qwen4exp.ple_conv_history[static_cast<size_t>(layer_idx)];
      if (hist.size() != static_cast<size_t>(HC * D) * hist_len)
        hist.assign(static_cast<size_t>(HC * D) * hist_len, 0.0f);
      std::vector<float> conv_out(gated.size());
      for (int ch = 0; ch < HC * D; ++ch)
        for (int s = 0; s < S; ++s) {
          float acc = 0.0f;
          for (int k = 0; k < kernel; ++k) {
            const int64_t tpos = static_cast<int64_t>(s) -
                                 static_cast<int64_t>(kernel - 1 - k) *
                                     c.ple_ngram_size;
            const float v = tpos < 0
                                ? hist[static_cast<size_t>(ch) * hist_len + tpos +
                                       hist_len]
                                : conv_in[static_cast<size_t>(tpos) * HC * D + ch];
            acc += v * conv_w[static_cast<size_t>(ch) * kernel + k];
          }
          conv_out[static_cast<size_t>(s) * HC * D + ch] =
              acc / (1.0f + std::exp(-acc));
        }
      for (int ch = 0; ch < HC * D; ++ch)
        for (int j = 0; j < hist_len; ++j) {
          const int pos = S + j;
          hist[static_cast<size_t>(ch) * hist_len + j] =
              pos < hist_len ? hist[static_cast<size_t>(ch) * hist_len + pos]
                             : conv_in[static_cast<size_t>(pos - hist_len) * HC * D +
                                       ch];
        }
      for (size_t i = 0; i < gated.size(); ++i)
        engine.qwen4exp.hc_stream_buf[i] += gated[i] + conv_out[i];
      return true;
}

void CPUQwen4ExpBackend::forward_gdn(CPUEngine &engine, int layer_idx,
                                     float *hidden_states, int S) {
  const auto &c = engine.model_file.architecture_config;
  const int D = c.hidden;
  const int Hk = c.ssm_group_count;
  const int Hv = c.ssm_time_step_rank;
  const int hd = c.ssm_state_size;
  const int key_dim = Hk * hd;
  const int value_dim = c.ssm_inner_size;
  const int conv_dim = key_dim * 2 + value_dim;
  const int kernel = c.ssm_conv_kernel;
  const std::string p = "blk." + std::to_string(layer_idx) + ".";
  if (engine.qwen4exp.conv_history.size() <
      static_cast<size_t>(c.n_layers))
    engine.qwen4exp.conv_history.resize(static_cast<size_t>(c.n_layers));
  if (engine.qwen4exp.recur_state.size() <
      static_cast<size_t>(c.n_layers))
    engine.qwen4exp.recur_state.resize(static_cast<size_t>(c.n_layers));

  std::vector<float> qkv(static_cast<size_t>(S) * conv_dim);
  std::vector<float> z(static_cast<size_t>(S) * value_dim);
  std::vector<float> alpha(static_cast<size_t>(S) * Hv);
  std::vector<float> beta(static_cast<size_t>(S) * Hv);
  if (!dense_matmul(engine, p + "attn_qkv.weight", hidden_states, qkv.data(),
                    S, D, conv_dim) ||
      !dense_matmul(engine, p + "attn_gate.weight", hidden_states, z.data(), S,
                    D, value_dim) ||
      !dense_matmul(engine, p + "ssm_alpha.weight", hidden_states, alpha.data(),
                    S, D, Hv) ||
      !dense_matmul(engine, p + "ssm_beta.weight", hidden_states, beta.data(),
                    S, D, Hv))
    return;

  std::vector<float> conv_w(static_cast<size_t>(conv_dim) * kernel);
  const auto *conv = engine.find_layer(p + "ssm_conv1d.weight");
  if (!conv || !dequant_tensor(*conv, conv_w.data(), conv_w.size()))
    return;
  auto &hist = engine.qwen4exp.conv_history[static_cast<size_t>(layer_idx)];
  if (hist.size() != static_cast<size_t>(conv_dim) * (kernel - 1))
    hist.assign(static_cast<size_t>(conv_dim) * (kernel - 1), 0.0f);
  std::vector<float> conv_out(qkv.size());
  for (int ch = 0; ch < conv_dim; ++ch) {
    for (int s = 0; s < S; ++s) {
      float acc = 0.0f;
      for (int k = 0; k < kernel; ++k) {
        const int j = s + k;
        const float v = j < kernel - 1
                            ? hist[static_cast<size_t>(ch) * (kernel - 1) + j]
                            : qkv[static_cast<size_t>(j - (kernel - 1)) *
                                      conv_dim +
                                  ch];
        acc += v * conv_w[static_cast<size_t>(ch) * kernel + k];
      }
      conv_out[static_cast<size_t>(s) * conv_dim + ch] =
          acc / (1.0f + std::exp(-acc));
    }
  }
  for (int ch = 0; ch < conv_dim; ++ch)
    for (int j = 0; j < kernel - 1; ++j) {
      const int pos = S + j;
      hist[static_cast<size_t>(ch) * (kernel - 1) + j] =
          pos < kernel - 1
              ? hist[static_cast<size_t>(ch) * (kernel - 1) + pos]
              : qkv[static_cast<size_t>(pos - (kernel - 1)) * conv_dim + ch];
    }

  const size_t state_size = static_cast<size_t>(Hv) * hd * hd;
  auto &persistent = engine.qwen4exp.recur_state[static_cast<size_t>(layer_idx)];
  if (persistent.size() != state_size)
    persistent.assign(state_size, 0.0f);
  std::vector<float> out(static_cast<size_t>(S) * value_dim);
  const auto *a_t = engine.find_layer(p + "ssm_a");
  const auto *dt_t = engine.find_layer(p + "ssm_dt.bias");
  std::vector<float> av(Hv), dt(Hv);
  if (!a_t || !dt_t || !dequant_row(*a_t, 0, av.data(), Hv) ||
      !dequant_row(*dt_t, 0, dt.data(), Hv))
    return;
  for (int s = 0; s < S; ++s) {
    for (int hk = 0; hk < Hk; ++hk) {
      float *query = conv_out.data() + static_cast<size_t>(s) * conv_dim + hk * hd;
      l2norm(query, hd, c.eps);
      l2norm(query + key_dim, hd, c.eps);
    }
    #pragma omp parallel for schedule(static) if(Hv >= 8)
    for (int h = 0; h < Hv; ++h) {
      const int hk = engine.exl3_model.tensors.empty() ? h % Hk : h / (Hv / Hk);
      float *st = persistent.data() + static_cast<size_t>(h) * hd * hd;
      float *q = conv_out.data() + (static_cast<size_t>(s) * conv_dim) +
                 hk * hd;
      float *k = q + key_dim;
      const float *v = conv_out.data() + static_cast<size_t>(s) * conv_dim +
                       key_dim * 2 + h * hd;
      const float decay_log = qwen4exp_gdn_decay_log(
          alpha[static_cast<size_t>(s) * Hv + h], dt[h], av[h]);
      const float b =
          1.0f / (1.0f + std::exp(-beta[static_cast<size_t>(s) * Hv + h]));
      float *dst = out.data() + (static_cast<size_t>(s) * Hv + h) * hd;
      qwen4exp_gdn_delta_step(q, k, v, b, decay_log, st, dst, hd);
    }
  }
  std::vector<float> norm_w(hd);
  const auto *nw = engine.find_layer(p + "ssm_norm.weight");
  if (!nw || !dequant_row(*nw, 0, norm_w.data(), hd))
    return;
  for (int s = 0; s < S; ++s)
    for (int h = 0; h < Hv; ++h) {
      float *dst = out.data() + (static_cast<size_t>(s) * Hv + h) * hd;
      const float *zz = z.data() + (static_cast<size_t>(s) * Hv + h) * hd;
      rmsnorm(dst, dst, norm_w.data(), hd, c.eps);
      for (int i = 0; i < hd; ++i)
        dst[i] /= 1.0f + std::exp(-zz[i]);
    }
  dense_matmul(engine, p + "ssm_out.weight", out.data(), hidden_states, S,
               value_dim, D);
}

void CPUQwen4ExpBackend::forward_attention(CPUEngine &engine, int layer_idx,
                                           float *hidden_states, int S,
                                           int pos_base) {
  const auto &c = engine.model_file.architecture_config;
  const int D = c.hidden;
  const int H = c.n_heads;
  const int Hkv = std::max(1, c.n_head_kv);
  const int hd = c.head_dim;
  const int nrot = c.rope_n_rot > 0 ? c.rope_n_rot : hd;
  const int group = std::max(1, H / Hkv);
  const std::string p = "blk." + std::to_string(layer_idx) + ".";
  std::vector<float> qg(static_cast<size_t>(S) * H * 2 * hd);
  std::vector<float> K(static_cast<size_t>(S) * Hkv * hd);
  std::vector<float> V(K.size());
  if (!dense_matmul(engine, p + "attn_q.weight", hidden_states, qg.data(), S,
                    D, H * 2 * hd) ||
      !dense_matmul(engine, p + "attn_k.weight", hidden_states, K.data(), S,
                    D, Hkv * hd) ||
      !dense_matmul(engine, p + "attn_v.weight", hidden_states, V.data(), S,
                    D, Hkv * hd))
    return;
  std::vector<float> q(static_cast<size_t>(S) * H * hd);
  std::vector<float> gate(q.size());
  for (int s = 0; s < S; ++s)
    for (int h = 0; h < H; ++h) {
      const float *src = qg.data() + (static_cast<size_t>(s) * H + h) * 2 * hd;
      std::memcpy(q.data() + (static_cast<size_t>(s) * H + h) * hd, src,
                  static_cast<size_t>(hd) * sizeof(float));
      std::memcpy(gate.data() + (static_cast<size_t>(s) * H + h) * hd,
                  src + hd, static_cast<size_t>(hd) * sizeof(float));
    }
  std::vector<float> qw(hd), kw(hd);
  const auto *qwt = engine.find_layer(p + "attn_q_norm.weight");
  const auto *kwt = engine.find_layer(p + "attn_k_norm.weight");
  if (!qwt || !kwt || !dequant_row(*qwt, 0, qw.data(), hd) ||
      !dequant_row(*kwt, 0, kw.data(), hd))
    return;
  const float theta =
      engine.model_file.get_config_float("rope.freq_base", 10000.0f);
  for (int s = 0; s < S; ++s) {
    for (int h = 0; h < H; ++h) {
      rmsnorm(q.data() + (static_cast<size_t>(s) * H + h) * hd,
              q.data() + (static_cast<size_t>(s) * H + h) * hd, qw.data(), hd,
              c.eps);
      rope_neox(q.data() + (static_cast<size_t>(s) * H + h) * hd, nrot,
                       pos_base + s, theta);
    }
    for (int h = 0; h < Hkv; ++h) {
      rmsnorm(K.data() + (static_cast<size_t>(s) * Hkv + h) * hd,
              K.data() + (static_cast<size_t>(s) * Hkv + h) * hd, kw.data(), hd,
              c.eps);
      rope_neox(K.data() + (static_cast<size_t>(s) * Hkv + h) * hd, nrot,
                       pos_base + s, theta);
      if (pos_base + s < engine.max_t) {
        std::memcpy(engine.kv_cache.k_ptr(layer_idx, pos_base + s) + h * hd,
                    K.data() + (static_cast<size_t>(s) * Hkv + h) * hd,
                    static_cast<size_t>(hd) * sizeof(float));
        std::memcpy(engine.kv_cache.v_ptr(layer_idx, pos_base + s) + h * hd,
                    V.data() + (static_cast<size_t>(s) * Hkv + h) * hd,
                    static_cast<size_t>(hd) * sizeof(float));
      }
    }
  }
  std::vector<float> ctx(static_cast<size_t>(S) * H * hd, 0.0f);
  for (int s = 0; s < S; ++s)
    for (int h = 0; h < H; ++h) {
      const int len = std::min(engine.max_t, pos_base + s + 1);
      const int kvh = h / group;
      std::vector<float> scores(len);
      for (int t = 0; t < len; ++t) {
        const float *kt = engine.kv_cache.k_ptr(layer_idx, t) + kvh * hd;
        const float *qh = q.data() + (static_cast<size_t>(s) * H + h) * hd;
        for (int i = 0; i < hd; ++i)
          scores[t] += qh[i] * kt[i];
        scores[t] /= std::sqrt(static_cast<float>(hd));
      }
      softmax(scores.data(), len);
      float *dst = ctx.data() + (static_cast<size_t>(s) * H + h) * hd;
      for (int t = 0; t < len; ++t) {
        const float *vt = engine.kv_cache.v_ptr(layer_idx, t) + kvh * hd;
        for (int i = 0; i < hd; ++i)
          dst[i] += scores[t] * vt[i];
      }
    }
  sigmoid(gate.data(), static_cast<int>(gate.size()));
  for (size_t i = 0; i < ctx.size(); ++i)
    ctx[i] *= gate[i];
  dense_matmul(engine, p + "attn_output.weight", ctx.data(), hidden_states, S,
               H * hd, D);
}

void CPUQwen4ExpBackend::forward_moe(CPUEngine &engine, int layer_idx,
                                     float *hidden_states, int S) {
  const auto &c = engine.model_file.architecture_config;
  const int D = c.hidden;
  const int topk = std::min(c.topk, c.n_experts);
  const int inter = c.moe_inter;
  const std::string p = "blk." + std::to_string(layer_idx) + ".";
  const auto *router = engine.find_layer(p + "ffn_gate_inp.weight");
  if (!router || !router->host_ptr)
    return;
  std::vector<float> logits(static_cast<size_t>(S) * c.n_experts);
  if (!dense_matmul(engine, p + "ffn_gate_inp.weight", hidden_states,
                    logits.data(), S, D, c.n_experts))
    return;
  std::vector<float> out(static_cast<size_t>(S) * D, 0.0f);
  if (!engine.exl3_model.tensors.empty()) {
    const size_t route_count = static_cast<size_t>(S) * topk;
    std::vector<int32_t> route_ids(route_count);
    std::vector<float> route_weights(route_count);
    for (int s = 0; s < S; ++s) {
      float *ls = logits.data() + static_cast<size_t>(s) * c.n_experts;
      for (int e = 0; e < c.n_experts; ++e)
        if (!std::isfinite(ls[e]))
          ls[e] = -1e30f;
      softmax(ls, c.n_experts);
      std::vector<int> order(c.n_experts);
      std::iota(order.begin(), order.end(), 0);
      std::partial_sort(order.begin(), order.begin() + topk, order.end(),
                        [&](int a, int b) { return ls[a] > ls[b]; });
      float sum = 0.0f;
      for (int k = 0; k < topk; ++k)
        sum += ls[order[k]];
      for (int k = 0; k < topk; ++k) {
        const size_t index = static_cast<size_t>(s) * topk + k;
        route_ids[index] = order[k];
        route_weights[index] = c.expert_weights_norm && sum > 0.0f
                                   ? ls[order[k]] / sum
                                   : ls[order[k]];
      }
    }
    if (!engine.exl3_routed_ffn_batch(
            layer_idx, route_ids.data(), route_weights.data(), S,
            hidden_states, out.data()) &&
        !engine.exl3_cpu_routed_ffn_batch(
            layer_idx, route_ids.data(), route_weights.data(), S,
            hidden_states, out.data())) {
      for (int s = 0; s < S; ++s) {
        const int32_t *ids = route_ids.data() + static_cast<size_t>(s) * topk;
        const float *weights =
            route_weights.data() + static_cast<size_t>(s) * topk;
        if (engine.exl3_cpu_routed_ffn(
                layer_idx, ids, weights, topk,
                hidden_states + static_cast<size_t>(s) * D,
                out.data() + static_cast<size_t>(s) * D))
          continue;
        for (int k = 0; k < topk; ++k) {
          std::vector<float> expert_output(D);
          engine.exl3_expert(
              layer_idx, ids[k], hidden_states + static_cast<size_t>(s) * D,
              expert_output.data());
          for (int index = 0; index < D; ++index)
            out[static_cast<size_t>(s) * D + index] +=
                weights[k] * expert_output[index];
        }
      }
    }
  } else {
  for (int s = 0; s < S; ++s) {
    float *ls = logits.data() + static_cast<size_t>(s) * c.n_experts;
    for (int e = 0; e < c.n_experts; ++e)
      if (!std::isfinite(ls[e]))
        ls[e] = -1e30f;
    softmax(ls, c.n_experts);
    std::vector<int> order(c.n_experts);
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + topk, order.end(),
                      [&](int a, int b) { return ls[a] > ls[b]; });
    float sum = 0.0f;
    for (int k = 0; k < topk; ++k)
      sum += ls[order[k]];
    if (!engine.exl3_model.tensors.empty()) {
      std::vector<float> routing_weights(topk);
      for (int index = 0; index < topk; ++index)
        routing_weights[index] = c.expert_weights_norm && sum > 0.0f
            ? ls[order[index]] / sum : ls[order[index]];
      if (engine.exl3_routed_ffn(layer_idx, order.data(), routing_weights.data(), topk,
                                 hidden_states + static_cast<size_t>(s) * D,
                                 out.data() + static_cast<size_t>(s) * D))
        continue;
      if (engine.exl3_cpu_routed_ffn(layer_idx, order.data(), routing_weights.data(), topk,
                                     hidden_states + static_cast<size_t>(s) * D,
                                     out.data() + static_cast<size_t>(s) * D))
        continue;
    }
    for (int k = 0; k < topk; ++k) {
      const int e = order[k];
      if (!engine.exl3_model.tensors.empty()) {
        std::vector<float> expert_output(D);
        engine.exl3_expert(layer_idx, e, hidden_states + static_cast<size_t>(s) * D,
                           expert_output.data());
        const float weight = c.expert_weights_norm && sum > 0.0f ? ls[e] / sum : ls[e];
        for (int index = 0; index < D; ++index)
          out[static_cast<size_t>(s) * D + index] += weight * expert_output[index];
        continue;
      }
      CPUExpertSlot *slot = engine.expert_cache.get(layer_idx, e);
      if (!slot)
        continue;
      const float weight =
          c.expert_weights_norm && sum > 0.0f ? ls[e] / sum : ls[e];
      std::vector<float> gp(inter), up(inter), dp(D);
      // The cache stores the three raw tensors contiguously. Dequantize rows
      // directly from the slab, avoiding a second dense copy of the expert.
      // Decode rows directly from the non-owning slab. Do not wrap these
      // pointers in CPUQuantizedTensor: its destructor owns and frees host_ptr.
      auto dequant_slab_row = [&](GGUFType type, const uint8_t *base,
                                  size_t bytes, int row, int row_count,
                                  float *dst, int n) {
        const size_t row_bytes = bytes / static_cast<size_t>(row_count);
        const uint8_t *src = base + static_cast<size_t>(row) * row_bytes;
        if (type == GGUFType::F32) {
          std::memcpy(dst, src, static_cast<size_t>(n) * sizeof(float));
          return true;
        }
        if (type == GGUFType::F16) {
          const auto *p = reinterpret_cast<const uint16_t *>(src);
          for (int i = 0; i < n; ++i) dst[i] = fp16_to_fp32(p[i]);
          return true;
        }
        if (type == GGUFType::BF16) {
          const auto *p = reinterpret_cast<const uint16_t *>(src);
          for (int i = 0; i < n; ++i) dst[i] = bf16_to_float(p[i]);
          return true;
        }
        return ggml_dequant_row(static_cast<uint32_t>(type), src, dst, n) != 0;
      };
      const float *x = hidden_states + static_cast<size_t>(s) * D;
      for (int o = 0; o < inter; ++o) {
        std::vector<float> row(D);
        dequant_slab_row(slot->gate_type, slot->slab, slot->gate_size, o,
                         inter, row.data(), D);
        for (int i = 0; i < D; ++i) gp[o] += x[i] * row[i];
        dequant_slab_row(slot->up_type, slot->slab + slot->gate_size,
                         slot->up_size, o, inter, row.data(), D);
        for (int i = 0; i < D; ++i) up[o] += x[i] * row[i];
      }
      silu(gp.data(), inter);
      for (int i = 0; i < inter; ++i) gp[i] *= up[i];
      for (int o = 0; o < D; ++o) {
        std::vector<float> row(inter);
        dequant_slab_row(slot->down_type,
                         slot->slab + slot->gate_size + slot->up_size,
                         slot->down_size, o, D, row.data(), inter);
        for (int i = 0; i < inter; ++i)
          dp[o] += gp[i] * row[i];
        out[static_cast<size_t>(s) * D + o] += weight * dp[o];
      }
    }
  }
  }
  const auto *sg = engine.find_layer(p + "ffn_gate_shexp.weight");
  const auto *su = engine.find_layer(p + "ffn_up_shexp.weight");
  const auto *sd = engine.find_layer(p + "ffn_down_shexp.weight");
  const auto *sr = engine.find_layer(p + "ffn_gate_inp_shexp.weight");
  if (sg && su && sd && sr) {
    int si = inter;
    if (su->meta.shape.size() >= 2)
      si = static_cast<int>(su->meta.shape[1]);
    std::vector<float> gp(static_cast<size_t>(S) * si);
    std::vector<float> up(gp.size());
    std::vector<float> dp(static_cast<size_t>(S) * D);
    if (dense_matmul(engine, p + "ffn_gate_shexp.weight", hidden_states,
                     gp.data(), S, D, si) &&
        dense_matmul(engine, p + "ffn_up_shexp.weight", hidden_states, up.data(),
                     S, D, si)) {
      silu(gp.data(), static_cast<int>(gp.size()));
      for (size_t i = 0; i < gp.size(); ++i)
        gp[i] *= up[i];
      dense_matmul(engine, p + "ffn_down_shexp.weight", gp.data(), dp.data(),
                   S, si, D);
      std::vector<float> scalar(S);
      dense_matmul(engine, p + "ffn_gate_inp_shexp.weight", hidden_states,
                   scalar.data(), S, D, 1);
      sigmoid(scalar.data(), S);
      for (int s = 0; s < S; ++s)
        for (int d = 0; d < D; ++d)
          out[static_cast<size_t>(s) * D + d] +=
              scalar[s] * dp[static_cast<size_t>(s) * D + d];
    }
  }
  std::memcpy(hidden_states, out.data(),
              out.size() * sizeof(float));
}

void CPUQwen4ExpBackend::forward_layer(CPUEngine &engine, int layer_idx,
                                       float *hidden_states, int S,
                                       int pos_base) {
  const int D = engine.model_file.architecture_config.hidden;
  const int HC = engine.model_file.architecture_config.hc_mult;
  if (layer_idx == 0 || engine.qwen4exp.hc_stream_buf.size() !=
                             static_cast<size_t>(S) * HC * D) {
    engine.qwen4exp.hc_stream_buf.resize(static_cast<size_t>(S) * HC * D);
    for (int s = 0; s < S; ++s)
      for (int h = 0; h < HC; ++h)
        std::memcpy(engine.qwen4exp.hc_stream_buf.data() +
                        (static_cast<size_t>(s) * HC + h) * D,
                    hidden_states + static_cast<size_t>(s) * D,
                    static_cast<size_t>(D) * sizeof(float));
  }
  if (!ple_layer_input(engine, layer_idx, S, pos_base) && !engine.exl3_model.tensors.empty())
    throw std::runtime_error("EXL3 PLE layer failed: " + std::to_string(layer_idx));
  hc_mix(engine, layer_idx, 0, hidden_states, S);
  std::vector<float> attn(static_cast<size_t>(S) *
                          engine.model_file.architecture_config.hidden);
  if (qwen4exp_is_gdn_layer(engine.model_file.architecture_config, layer_idx))
    forward_gdn(engine, layer_idx, hidden_states, S);
  else
    forward_attention(engine, layer_idx, hidden_states, S, pos_base);
  std::memcpy(attn.data(), hidden_states,
              attn.size() * sizeof(float));
  hc_combine(engine, hidden_states, attn.data(), S);
  hc_mix(engine, layer_idx, 1, hidden_states, S);
  forward_moe(engine, layer_idx, hidden_states, S);
  std::vector<float> ffn(attn.size());
  std::memcpy(ffn.data(), hidden_states, ffn.size() * sizeof(float));
  hc_combine(engine, hidden_states, ffn.data(), S);
}

bool CPUQwen4ExpBackend::lm_head(CPUEngine &engine, const float *hidden,
                                 float *logits, int S) {
  const auto &c = engine.model_file.architecture_config;
  const int D = c.hidden;
  const int V = c.vocab;
  std::vector<float> tmp(static_cast<size_t>(S) * D);
  if (c.hc_mult > 1 && engine.qwen4exp.hc_stream_buf.size() ==
                           static_cast<size_t>(S) * c.hc_mult * D) {
    const int HC = c.hc_mult;
    const int HCD = HC * D;
    std::vector<float> normed(static_cast<size_t>(S) * HCD);
    const float *w = tensor(engine, "output_hc_norm.weight");
    for (int s = 0; s < S; ++s)
      for (int h = 0; h < HC; ++h)
        rmsnorm(normed.data() + (static_cast<size_t>(s) * HC + h) * D,
                engine.qwen4exp.hc_stream_buf.data() +
                    (static_cast<size_t>(s) * HC + h) * D,
                w ? w + h * D : nullptr, D, c.eps);
    if (!try_exl3_gpu_hc_mix(engine, "output_hc_down.weight", "output_hc_up.weight",
                              "", normed.data(), tmp.data(), nullptr, S, D, HC,
                              c.hc_low_rank, false)) {
      std::vector<float> low(static_cast<size_t>(S) * c.hc_low_rank);
      std::vector<float> gate(static_cast<size_t>(S) * HCD);
      if (!dense_matmul(engine, "output_hc_down.weight", normed.data(),
                        low.data(), S, HCD, c.hc_low_rank))
        return false;
      for (float &v : low)
        v /= static_cast<float>(HC);
      silu(low.data(), static_cast<int>(low.size()));
      if (!dense_matmul(engine, "output_hc_up.weight", low.data(), gate.data(),
                        S, c.hc_low_rank, HCD))
        return false;
      sigmoid(gate.data(), static_cast<int>(gate.size()));
      for (int s = 0; s < S; ++s)
        for (int d = 0; d < D; ++d) {
          float v = 0.0f;
          for (int h = 0; h < HC; ++h)
            v += normed[(static_cast<size_t>(s) * HC + h) * D + d] *
                 gate[(static_cast<size_t>(s) * HC + h) * D + d];
          tmp[static_cast<size_t>(s) * D + d] = v / static_cast<float>(HC);
        }
    }
  } else {
    std::memcpy(tmp.data(), hidden, tmp.size() * sizeof(float));
  }
  if (dense_matmul(engine, "output.weight", tmp.data(), logits, S, D, V))
    return true;
  const auto *emb = engine.find_layer("token_embd.weight");
  if (!emb || !emb->host_ptr)
    return false;
  const int rows = emb->meta.shape.size() >= 2
                       ? static_cast<int>(emb->meta.shape[1])
                       : static_cast<int>(emb->meta.shape[0]);
  if (rows != V)
    return false;
  std::vector<float> row(static_cast<size_t>(D));
  for (int s = 0; s < S; ++s) {
    const float *x = tmp.data() + static_cast<size_t>(s) * D;
    float *dst = logits + static_cast<size_t>(s) * V;
    for (int v = 0; v < V; ++v) {
      if (!dequant_row(*emb, v, row.data(), D))
        return false;
      float acc = 0.0f;
      for (int d = 0; d < D; ++d)
        acc += x[d] * row[d];
      dst[v] = acc;
    }
  }
  return true;
}
