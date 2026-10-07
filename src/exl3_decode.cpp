#include "exl3_decode.hpp"
#include "exl3_cpu.hpp"
#include "cpu_model_backend.hpp"
#include "exl3_model_loader.hpp"
#ifdef MOESTREAM_EXL3_CUDA
#include "exl3_gpu.hpp"
#endif

#include <cmath>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

std::expected<void, std::string> run_exl3_decode(const Exl3DecodeOptions &options) {
  try {
#ifdef _OPENMP
    omp_set_num_threads(std::max(1, options.threads));
#endif
    CPUEngine engine;
    engine.config.model_path = options.model_path;
    engine.config.model_format = "exl3";
    engine.config.metadata_model_path = options.metadata_model_path;
    engine.exl3_gpu_expert_layer_begin = options.gpu_expert_layer_begin;
    engine.config.exl3_cpu_expert_cache_gb = options.cpu_expert_cache_gb;
    engine.config.exl3_cpu_expert_int8_vnni = options.cpu_expert_int8_vnni;
    engine.config.exl3_gpu_hc = options.gpu_hc;
    if (options.gpu_hc && !options.gpu)
      return std::unexpected("exl3_gpu_hc requires exl3_gpu");
    if (options.cpu_expert_int8_vnni && !exl3_cpu_vnni_available())
      return std::unexpected("exl3_cpu_expert_int8_vnni requires AVX-512 VNNI");
    if (options.gpu) {
#ifdef MOESTREAM_EXL3_CUDA
      if (!std::isfinite(options.gpu_cache_gb) || options.gpu_cache_gb <= 0.0 ||
          options.gpu_cache_gb > 1024.0)
        return std::unexpected("exl3_gpu_cache_gb must be in (0, 1024]");
      if (options.gpu_expert_split_layer >= 0 &&
          (!std::isfinite(options.gpu_split_cache_gb) || options.gpu_split_cache_gb <= 0.0 ||
           options.gpu_split_cache_gb > 1024.0 || options.gpu_split_device == options.gpu_device))
        return std::unexpected("EXL3 GPU layer split needs a distinct device and positive cache");
      engine.exl3_gpu = std::make_shared<Exl3GpuExecutor>(
          options.gpu_device, static_cast<size_t>(options.gpu_cache_gb * 1024.0 * 1024.0 * 1024.0));
#else
      return std::unexpected("EXL3 GPU dispatch requires a CUDA-enabled MoEStream build");
#endif
    }
    if (!engine.build_from_exl3(options.model_path, options.metadata_model_path,
                                options.context_size))
      return std::unexpected("EXL3 Qwen backend initialization failed");
    const int layer_count = engine.model_file.architecture_config.n_layers;
    if (options.gpu_expert_layer_begin < 0 || options.gpu_expert_layer_begin > layer_count)
      return std::unexpected("exl3_gpu_expert_layer_begin must be in [0, layer_count]");
    if (options.gpu_expert_split_layer >= 0) {
      if (!engine.exl3_gpu || options.gpu_expert_split_layer <= options.gpu_expert_layer_begin ||
          options.gpu_expert_split_layer >= layer_count)
        return std::unexpected("EXL3 GPU split layer must divide the GPU expert range");
#ifdef MOESTREAM_EXL3_CUDA
      engine.exl3_gpu_secondary = std::make_shared<Exl3GpuExecutor>(
          options.gpu_split_device,
          static_cast<size_t>(options.gpu_split_cache_gb * 1024.0 * 1024.0 * 1024.0));
      engine.exl3_gpu_expert_split_layer = options.gpu_expert_split_layer;
#endif
    }
#ifdef MOESTREAM_EXL3_CUDA
    if (engine.exl3_gpu) {
      LOG_STREAM_INFO() << "[EXL3-GPU] HC device chain: "
                        << (options.gpu_hc ? "enabled" : "disabled") << "\n";
      LOG_STREAM_INFO() << "[EXL3-GPU] CUDA matrix dispatch enabled on device "
                        << options.gpu_device << "; cache budget " << options.gpu_cache_gb
                        << " GiB; routed experts on GPU layers "
                        << options.gpu_expert_layer_begin << "–"
                        << (engine.exl3_gpu_secondary
                                ? options.gpu_expert_split_layer - 1 : layer_count - 1)
                        << (engine.exl3_gpu_secondary ? " (primary)" : "") << "\n";
      if (engine.exl3_gpu_secondary)
        LOG_STREAM_INFO() << "[EXL3-GPU] Routed experts on layers "
                          << options.gpu_expert_split_layer << "–" << layer_count - 1
                          << " on device " << options.gpu_split_device << "; cache budget "
                          << options.gpu_split_cache_gb << " GiB\n";
    }
#endif
    std::string prompt = options.prompt;
    if (prompt.empty()) {
      std::cout << "Prompt: " << std::flush;
      if (!std::getline(std::cin, prompt) || prompt.empty())
        return std::unexpected("EXL3 decode needs a non-empty prompt");
    }
    std::vector<int> tokens;
    const auto append_text = [&](const std::string &text) {
      const auto encoded = engine.tokenizer.encode(text);
      tokens.insert(tokens.end(), encoded.begin(), encoded.end());
    };
    if (!options.raw_prompt) {
      if (!engine.tokenizer.has_chatml_template())
        return std::unexpected("EXL3 Qwen tokenizer has no ChatML tokens");
      tokens.push_back(engine.tokenizer.im_start_id);
      append_text("user\n" + prompt);
      tokens.push_back(engine.tokenizer.im_end_id);
      append_text("\n");
      tokens.push_back(engine.tokenizer.im_start_id);
      append_text("assistant\n");
      if (engine.tokenizer.ds_think_start_id < 0 || engine.tokenizer.ds_think_end_id < 0)
        return std::unexpected("EXL3 Qwen tokenizer lacks thinking markers");
      tokens.push_back(engine.tokenizer.ds_think_start_id);
      append_text(options.enable_thinking ? "\n" : "\n\n");
      if (!options.enable_thinking) {
        tokens.push_back(engine.tokenizer.ds_think_end_id);
        append_text("\n\n");
      }
    } else append_text(prompt);
    if (tokens.empty() || options.max_new_tokens < 0 ||
        tokens.size() + static_cast<size_t>(options.max_new_tokens) >
            static_cast<size_t>(options.context_size))
      return std::unexpected("EXL3 prompt and decode exceed context capacity");
    const auto &cfg = engine.model_file.architecture_config;
    std::vector<float> hidden(cfg.hidden), logits(cfg.vocab);
    std::vector<double> layer_seconds(static_cast<size_t>(cfg.n_layers), 0.0);
    const auto forward = [&](int token, int position) {
      engine.qwen4exp_ple_token_history.push_back(token);
      if (!cpu_embed_token(engine, token, hidden.data(), cfg.hidden))
        throw std::runtime_error("EXL3 token embedding failed");
      for (int layer = 0; layer < cfg.n_layers; ++layer) {
        const auto layer_begin = std::chrono::steady_clock::now();
        if (!cpu_forward_layer(engine, layer, hidden.data(), 1, position))
          throw std::runtime_error("EXL3 forward failed at layer " + std::to_string(layer));
        layer_seconds[static_cast<size_t>(layer)] +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - layer_begin).count();
        for (float value : engine.qwen4exp_hc_stream_buf)
          if (!std::isfinite(value))
            throw std::runtime_error("Non-finite EXL3 layer " + std::to_string(layer));
        if (position == 0 && layer % 8 == 0)
          LOG_STREAM_INFO() << "[EXL3] Initial token layer " << layer << " complete\n";
      }
      engine.qwen4exp_ple_next_pos = position + 1;
    };
    const auto start = std::chrono::steady_clock::now();
    LOG_STREAM_INFO() << "[EXL3] Prefill " << tokens.size() << " tokens\n";
    const int prefill_rows = static_cast<int>(tokens.size());
    std::vector<float> prefill_hidden(
        static_cast<size_t>(prefill_rows) * cfg.hidden);
    engine.qwen4exp_ple_token_history.clear();
    engine.qwen4exp_ple_token_history.reserve(tokens.size());
    for (int position = 0; position < prefill_rows; ++position) {
      engine.qwen4exp_ple_token_history.push_back(
          tokens[static_cast<size_t>(position)]);
      if (!cpu_embed_token(
              engine, tokens[static_cast<size_t>(position)],
              prefill_hidden.data() + static_cast<size_t>(position) * cfg.hidden,
              cfg.hidden))
        throw std::runtime_error("EXL3 prefill embedding failed");
    }
    for (int layer = 0; layer < cfg.n_layers; ++layer) {
      const auto layer_begin = std::chrono::steady_clock::now();
      if (!cpu_forward_layer(engine, layer, prefill_hidden.data(),
                             prefill_rows, 0))
        throw std::runtime_error("EXL3 prefill failed at layer " +
                                 std::to_string(layer));
      layer_seconds[static_cast<size_t>(layer)] +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        layer_begin)
              .count();
      for (float value : engine.qwen4exp_hc_stream_buf)
        if (!std::isfinite(value))
          throw std::runtime_error("Non-finite EXL3 prefill layer " +
                                   std::to_string(layer));
      if (layer % 8 == 0)
        LOG_STREAM_INFO() << "[EXL3] Prefill layer " << layer
                          << " complete for " << prefill_rows << " rows\n";
    }
    std::memcpy(hidden.data(),
                prefill_hidden.data() +
                    static_cast<size_t>(prefill_rows - 1) * cfg.hidden,
                static_cast<size_t>(cfg.hidden) * sizeof(float));
    engine.qwen4exp_ple_next_pos = prefill_rows;
    CPUSampler sampler;
    const double prefill_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    int output_tokens = 0;
    auto first_output = std::chrono::steady_clock::now();
    auto last_output = first_output;
#ifdef MOESTREAM_EXL3_CUDA
    Exl3GpuStats first_output_stats;
    Exl3GpuStats last_output_stats;
    const auto gpu_profile = [&] {
      Exl3GpuStats combined;
      const auto add = [&](const std::shared_ptr<Exl3GpuExecutor> &executor) {
        if (!executor) return;
        const auto stats = executor->stats();
        combined.matmuls += stats.matmuls;
        combined.cache_hits += stats.cache_hits;
        combined.uploads += stats.uploads;
        combined.evictions += stats.evictions;
        combined.resident_bytes += stats.resident_bytes;
        combined.routed_ffns += stats.routed_ffns;
        combined.buffer_reuses += stats.buffer_reuses;
        combined.hc_mixes += stats.hc_mixes;
        combined.matmul_seconds += stats.matmul_seconds;
        combined.standalone_matmuls += stats.standalone_matmuls;
        combined.standalone_matmul_seconds += stats.standalone_matmul_seconds;
        combined.routed_ffn_seconds += stats.routed_ffn_seconds;
        combined.upload_seconds += stats.upload_seconds;
      };
      add(engine.exl3_gpu);
      add(engine.exl3_gpu_secondary);
      return combined;
    };
#endif
    uint64_t first_cpu_matmuls = 0;
    uint64_t last_cpu_matmuls = 0;
    double first_cpu_matmul_seconds = 0.0;
    double last_cpu_matmul_seconds = 0.0;
    double first_cpu_weight_load_seconds = 0.0;
    double last_cpu_weight_load_seconds = 0.0;
    std::vector<double> first_layer_seconds(layer_seconds.size());
    std::vector<double> last_layer_seconds(layer_seconds.size());
    std::cout << "\n";
    for (int step = 0; step < options.max_new_tokens; ++step) {
      if (!cpu_lm_head(engine, hidden.data(), logits.data(), 1))
        throw std::runtime_error("EXL3 output head failed");
      if (!std::all_of(logits.begin(), logits.end(), [](float value) { return std::isfinite(value); }))
        throw std::runtime_error("Non-finite EXL3 logits");
      const int token = sampler.sample(logits, options.temperature, options.top_p);
      LOG_STREAM_INFO() << "[EXL3] Decode step=" << step << " token=" << token
                        << " logit=" << logits[token] << "\n";
      if (engine.tokenizer.is_stop_token(token)) break;
      last_output = std::chrono::steady_clock::now();
      last_cpu_matmuls = engine.exl3_cpu_matmuls;
      last_cpu_matmul_seconds = engine.exl3_cpu_matmul_seconds;
      last_cpu_weight_load_seconds = engine.exl3_cpu_weight_load_seconds;
      last_layer_seconds = layer_seconds;
#ifdef MOESTREAM_EXL3_CUDA
      if (engine.exl3_gpu) last_output_stats = gpu_profile();
#endif
      if (output_tokens++ == 0) {
        first_output = last_output;
        first_cpu_matmuls = last_cpu_matmuls;
        first_cpu_matmul_seconds = last_cpu_matmul_seconds;
        first_cpu_weight_load_seconds = last_cpu_weight_load_seconds;
        first_layer_seconds = last_layer_seconds;
#ifdef MOESTREAM_EXL3_CUDA
        first_output_stats = last_output_stats;
#endif
      }
      std::cout << engine.tokenizer.decode(token) << std::flush;
      if (step + 1 < options.max_new_tokens)
        forward(token, static_cast<int>(tokens.size()) + step);
    }
    std::cout << "\n";
    const int intervals = std::max(0, output_tokens - 1);
    const double decode_seconds = std::chrono::duration<double>(last_output - first_output).count();
    LOG_STREAM_INFO() << "[EXL3-PERF] output_tokens=" << output_tokens << " intervals=" << intervals
                      << " wall_s=" << decode_seconds
                      << " decode_tps=" << (intervals && decode_seconds > 0.0 ? intervals / decode_seconds : 0.0)
                      << " prefill_s=" << prefill_seconds << "\n";
    if (engine.exl3_ngram)
      LOG_STREAM_INFO() << "[EXL3-PLE] row_reads=" << engine.exl3_ngram->row_reads()
                        << " cache_hits=" << engine.exl3_ngram->cache_hits() << "\n";
    if (engine.exl3_cpu_matrix_cache) {
      const auto cache = engine.exl3_cpu_matrix_cache->stats();
      LOG_STREAM_INFO() << "[EXL3-CPU-CACHE] hits=" << cache.hits
                        << " misses=" << cache.misses << " evictions=" << cache.evictions
                        << " resident_GiB=" << cache.resident_bytes / (1024.0 * 1024.0 * 1024.0)
                        << "\n";
    }
    if (intervals > 0 && last_cpu_matmuls > first_cpu_matmuls)
      LOG_STREAM_INFO() << "[EXL3-PERF] cpu_matmuls_per_interval="
                        << static_cast<double>(last_cpu_matmuls - first_cpu_matmuls) / intervals
                        << " cpu_matmul_ms_per_interval="
                        << (last_cpu_matmul_seconds - first_cpu_matmul_seconds) * 1000.0 / intervals
                        << " cpu_weight_load_ms_per_interval="
                        << (last_cpu_weight_load_seconds - first_cpu_weight_load_seconds) * 1000.0 / intervals
                        << "\n";
    if (intervals > 0) {
      std::vector<std::pair<int, double>> layer_times;
      layer_times.reserve(layer_seconds.size());
      for (size_t layer = 0; layer < layer_seconds.size(); ++layer)
        layer_times.emplace_back(static_cast<int>(layer),
            (last_layer_seconds[layer] - first_layer_seconds[layer]) * 1000.0 / intervals);
      const size_t count = std::min<size_t>(8, layer_times.size());
      std::partial_sort(layer_times.begin(), layer_times.begin() + count, layer_times.end(),
          [](const auto &left, const auto &right) { return left.second > right.second; });
      for (size_t index = 0; index < count; ++index)
        LOG_STREAM_INFO() << "[EXL3-LAYER-PERF] layer=" << layer_times[index].first
                          << " ms_per_interval=" << layer_times[index].second << "\n";
    }
    LOG_STREAM_INFO() << "[EXL3-CPU] SIMD="
                      << (exl3_cpu_vnni_available() ? "AVX512-VNNI" : "scalar")
                      << " int8_experts=" << (options.cpu_expert_int8_vnni ? "on" : "off")
                      << " threads=" << options.threads
                      << " omp_wait_policy="
                      << (std::getenv("OMP_WAIT_POLICY") ? std::getenv("OMP_WAIT_POLICY") : "default")
                      << " gomp_spincount="
                      << (std::getenv("GOMP_SPINCOUNT") ? std::getenv("GOMP_SPINCOUNT") : "default")
                      << "\n";
    LOG_STREAM_INFO() << "[EXL3] Decode completed in "
                      << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
                      << " s; resident tensors=" << engine.resident_layers.size() << "\n";
#ifdef MOESTREAM_EXL3_CUDA
    if (engine.exl3_gpu) {
      const auto stats = gpu_profile();
      if (intervals > 0) {
        const double matmul_seconds = last_output_stats.matmul_seconds - first_output_stats.matmul_seconds;
        const double standalone_seconds = last_output_stats.standalone_matmul_seconds -
                                          first_output_stats.standalone_matmul_seconds;
        const double routed_seconds = last_output_stats.routed_ffn_seconds -
                                      first_output_stats.routed_ffn_seconds;
        const double upload_seconds = last_output_stats.upload_seconds - first_output_stats.upload_seconds;
        const double cpu_matmul_seconds = last_cpu_matmul_seconds - first_cpu_matmul_seconds;
        const double cpu_load_seconds = last_cpu_weight_load_seconds - first_cpu_weight_load_seconds;
        LOG_STREAM_INFO() << "[EXL3-PERF] gpu_matmul_ms_per_interval=" << matmul_seconds * 1000.0 / intervals
                          << " gpu_standalone_ms_per_interval=" << standalone_seconds * 1000.0 / intervals
                          << " gpu_routed_ffn_ms_per_interval=" << routed_seconds * 1000.0 / intervals
                          << " gpu_standalone_calls_per_interval="
                          << static_cast<double>(last_output_stats.standalone_matmuls -
                                                 first_output_stats.standalone_matmuls) / intervals
                          << " gpu_upload_ms_per_interval=" << upload_seconds * 1000.0 / intervals
                          << " cpu_matmul_ms_per_interval=" << cpu_matmul_seconds * 1000.0 / intervals
                          << " cpu_weight_load_ms_per_interval=" << cpu_load_seconds * 1000.0 / intervals
                          << " other_ms_per_interval="
                          << (decode_seconds - matmul_seconds - upload_seconds -
                              cpu_matmul_seconds - cpu_load_seconds) * 1000.0 / intervals
                          << "\n";
      }
      LOG_STREAM_INFO() << "[EXL3-GPU] matmuls=" << stats.matmuls
                        << " cache_hits=" << stats.cache_hits << " uploads=" << stats.uploads
                        << " evictions=" << stats.evictions
                        << " routed_ffns=" << stats.routed_ffns
                        << " hc_mixes=" << stats.hc_mixes
                        << " buffer_reuses=" << stats.buffer_reuses
                        << " resident_GiB=" << stats.resident_bytes / (1024.0 * 1024.0 * 1024.0) << "\n";
    }
#endif
    return {};
  } catch (const std::exception &error) {
    return std::unexpected(error.what());
  }
}
