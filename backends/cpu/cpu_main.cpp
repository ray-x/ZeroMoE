// src/cpu_main.cpp — Host-only inference entry point for zeromoe_cpu.
#include "cpu_engine.hpp"
#include "cpu_model_backend.hpp"
#include "exl3_decode.hpp"
#include "exllamav3_backend.hpp"
#include "logger.hpp"
#include "model_family.hpp"
#include "cpu_api.hpp"
#include "version.hpp"
#include <csignal>
static volatile std::sig_atomic_t cpu_api_stop = 0;
static void stop_cpu_api(int) { cpu_api_stop = 1; }

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <initializer_list>
#include <string>
#include <vector>

static void print_usage(const char *argv0) {
  std::cerr << "Usage: " << argv0
            << " --config <path.toml> [--prompt \"<text>\"] [--ctx-size N] [--predict N] [--threads N]\n"
            << "  --serve [--host 127.0.0.1] [--port 8080] [--alias zeromoe]\n"
            << "       --api-key KEY (or ZEROMOE_API_KEY)\n"
            << "  --temp F --top-p F  Sampling parameters\n"
            << "  --api-cache-mib N  Opt-in CPU prompt checkpoints (default: 0, disabled)\n";
}

static std::string parse_arg(int argc, char **argv, const std::string &flag,
                             const std::string &def) {
  for (int i = 1; i < argc - 1; ++i) {
    if (flag == argv[i])
      return argv[i + 1];
  }

  return def;
}

static int parse_int_arg(int argc, char **argv, const std::string &flag,
                         int def) {
  const std::string value = parse_arg(argc, argv, flag, "");
  if (value.empty())
    return def;
  try {
    return std::stoi(value);
  } catch (...) {
    std::cerr << "[CPU-MAIN] Invalid integer for " << flag << ": " << value
              << "\n";
    return def;
  }
}

static int parse_int_arg_any(int argc, char **argv,
                             std::initializer_list<const char *> flags,
                             int def) {
  for (const char *flag : flags) {
    const std::string value = parse_arg(argc, argv, flag, "");
    if (!value.empty()) return parse_int_arg(argc, argv, flag, def);
  }
  return def;
}

static float parse_float_arg(int argc, char **argv, const char *flag,
                             float def) {
  const std::string value = parse_arg(argc, argv, flag, "");
  if (value.empty()) return def;
  try {
    size_t end = 0;
    const float parsed = std::stof(value, &end);
    if (end != value.size()) throw std::invalid_argument("trailing characters");
    return parsed;
  } catch (const std::exception &) {
    std::cerr << "[CPU-MAIN] Invalid number for " << flag << ": " << value << "\n";
    return def;
  }
}

int main(int argc, char **argv) {
  if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
    print_usage(argv[0]);
    return 0;
  }
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--version" || std::string(argv[i]) == "-V") {
      zeromoe_print_version();
      return 0;
    }
  }
  bool serve = false;
  for (int i = 1; i < argc; ++i) if (std::string(argv[i]) == "--serve") serve = true;
  zeromoe::server::Options api_options;
  std::string api_model = "zeromoe";
  size_t api_cache_bytes = 0;
  if (serve) {
    try {
      auto value = [&](const char *flag, const std::string &fallback) {
        for (int i = 1; i < argc; ++i) {
          if (std::string(argv[i]) != flag) continue;
          if (i + 1 == argc || std::string(argv[i + 1]).starts_with("--"))
            throw std::invalid_argument(std::string(flag) + " requires a value");
          return std::string(argv[i + 1]);
        }
        return fallback;
      };
      auto integer = [&](const char *flag, int fallback, int lo, int hi) {
        const auto s = value(flag, std::to_string(fallback));
        int n = 0;
        const auto [end, error] = std::from_chars(s.data(), s.data() + s.size(), n);
        if (error != std::errc{} || end != s.data() + s.size() || n < lo || n > hi)
          throw std::invalid_argument(std::string("invalid ") + flag);
        return n;
      };
      api_options.host = value("--host", "127.0.0.1");
      api_options.port = integer("--port", 8080, 1, 65535);
      api_model = value("--alias", "zeromoe");
      api_cache_bytes = static_cast<size_t>(integer("--api-cache-mib", 0, 0, 4096)) * 1024 * 1024;
      if (api_options.host.empty() || api_model.empty())
        throw std::invalid_argument("--host and --alias cannot be empty");
      const char *env_key = std::getenv("ZEROMOE_API_KEY");
      api_options.api_key = value("--api-key", env_key ? env_key : "");
      if (api_options.host != "127.0.0.1" && api_options.host != "::1" &&
          api_options.host != "localhost" && api_options.api_key.empty())
        throw std::invalid_argument("non-loopback binding requires --api-key or ZEROMOE_API_KEY");
    } catch (const std::exception &e) {
      std::cerr << "[CPU-API] " << e.what() << "\n";
      return 1;
    }
  }
  std::string config_path = parse_arg(argc, argv, "--config", "");
  if (config_path.empty()) {
    print_usage(argv[0]);
    return 1;
  }

  CPUConfig cfg;
  if (!cfg.load(config_path)) {
    std::cerr << "[CPU-MAIN] Failed to load config: " << config_path << "\n";
    return 1;
  }
  cfg.context_size = parse_int_arg_any(argc, argv, {"--ctx-size", "-c"},
                                       cfg.context_size);
  cfg.max_new_tokens = parse_int_arg_any(
      argc, argv, {"--predict", "--n-predict", "-n"}, cfg.max_new_tokens);
  cfg.omp_max_threads = parse_int_arg_any(argc, argv, {"--threads", "-t"},
                                          cfg.omp_max_threads);
  cfg.temperature = parse_float_arg(argc, argv, "--temp", cfg.temperature);
  cfg.top_p = parse_float_arg(argc, argv, "--top-p", cfg.top_p);

  Logger::instance().init(cfg.log_path, cfg.log_level);
  cfg.print();

  if (cfg.model_format == "exl3") {
    if (serve) {
      std::cerr << "Error: CPU API serving currently requires a GGUF Qwen MoE model\n";
      return 1;
    }
    if (cfg.exl3_backend == "exllamav3") {
      std::vector<std::string> overrides{
          "--predict", std::to_string(cfg.max_new_tokens),
          "--ctx-size", std::to_string(cfg.context_size),
          "--threads", std::to_string(cfg.omp_max_threads > 0 ? cfg.omp_max_threads : 4),
          "--cuda-schedule", cfg.exllamav3_cuda_schedule,
          "--temp", std::to_string(cfg.temperature),
          "--top-p", std::to_string(cfg.top_p),
          cfg.raw_prompt ? "--raw-prompt" : "--no-raw-prompt",
          cfg.enable_thinking ? "--thinking" : "--no-thinking"};
      const auto prompt = parse_arg(argc, argv, "--prompt", "");
      if (!prompt.empty()) {
        overrides.push_back("--prompt");
        overrides.push_back(prompt);
      }
      std::cout << "[EXLLAMAV3] Delegating to upstream CUDA/CPU runtime\n" << std::flush;
      auto result = exec_exllamav3_backend(config_path, cfg.exllamav3_python,
                                          cfg.exllamav3_runner_path, overrides);
      if (!result) std::cerr << result.error() << "\n";
      return result ? 0 : 1;
    }
    auto result = run_exl3_decode({
        .model_path = cfg.model_path,
        .metadata_model_path = cfg.metadata_model_path,
        .prompt = parse_arg(argc, argv, "--prompt", ""),
        .context_size = cfg.context_size,
        .max_new_tokens = cfg.max_new_tokens,
        .threads = cfg.omp_max_threads > 0 ? cfg.omp_max_threads : 4,
        .temperature = cfg.temperature,
        .top_p = cfg.top_p,
        .raw_prompt = cfg.raw_prompt,
        .enable_thinking = cfg.enable_thinking,
        .gpu = cfg.exl3_gpu,
        .gpu_hc = cfg.exl3_gpu_hc,
        .gpu_device = cfg.exl3_gpu_device,
        .gpu_cache_gb = cfg.exl3_gpu_cache_gb,
        .gpu_expert_layer_begin = cfg.exl3_gpu_expert_layer_begin,
        .gpu_expert_split_layer = cfg.exl3_gpu_expert_split_layer,
        .gpu_split_device = cfg.exl3_gpu_split_device,
        .gpu_split_cache_gb = cfg.exl3_gpu_split_cache_gb,
        .cpu_expert_cache_gb = cfg.exl3_cpu_expert_cache_gb,
        .cpu_expert_int8_vnni = cfg.exl3_cpu_expert_int8_vnni});
    if (!result) std::cerr << "[EXL3] " << result.error() << "\n";
    return result ? 0 : 1;
  }

  CPUEngine engine;
  engine.config = std::move(cfg);

  const bool is_safetensors =
      engine.config.model_path.size() >= 12 &&
      engine.config.model_path.compare(engine.config.model_path.size() - 12, 12,
                                       ".safetensors") == 0;

  if (is_safetensors) {
    if (serve) {
      std::cerr << "Error: CPU API serving requires a GGUF with embedded tokenizer metadata\n";
      return 1;
    }
    if (!engine.build_from_safetensors(engine.config.model_path,
                                       engine.config.context_size)) {
      std::cerr << "[CPU-MAIN] Safetensors engine build failed.\n";
      return 1;
    }
    std::cerr << "[CPU-MAIN] Safetensors model loaded successfully: "
              << engine.config.model_path << "\n";
    if (!engine.tokenizer.is_loaded()) {
      std::cerr << "[CPU-MAIN] No tokenizer available for this safetensors "
                   "model; generation skipped. Provide a GGUF tokenizer or "
                   "use this path as a load smoke check.\n";
      return 0;
    }
  } else {
    if (!engine.model_file.load_header(engine.config.model_path)) {
      std::cerr << "[CPU-MAIN] Failed to load GGUF: " << engine.config.model_path
                << "\n";
      return 1;
    }

    if (!engine.build(engine.config.context_size)) {
      std::cerr << "[CPU-MAIN] Engine build failed.\n";
      return 1;
    }
  }

  // Prompt handling.
  std::string prompt = parse_arg(argc, argv, "--prompt", "");
  if (prompt.empty())
    prompt = "Hello, my name is";

  std::string system_prompt_cli =
      parse_arg(argc, argv, "--system-prompt", "");
  if (!system_prompt_cli.empty())
    engine.config.system_prompt = system_prompt_cli;
  const std::string reasoning_effort_cli =
      parse_arg(argc, argv, "--reasoning-effort", "");
  if (!reasoning_effort_cli.empty())
    engine.config.reasoning_effort = reasoning_effort_cli;

  if (serve) {
    using namespace zeromoe::server;
    try {
      if (engine.config.raw_prompt || !engine.tokenizer.has_chatml_template())
        throw ApiError(400, "CPU serving requires the Qwen ChatML template, not raw prompts");
      ModelInfo model;
      model.id = api_model;
      model.context_size = engine.max_t;
      model.default_max_tokens = engine.config.max_new_tokens;
      model.max_output_tokens = engine.config.max_new_tokens;
      model.temperature = engine.config.temperature;
      model.top_p = engine.config.top_p;
      model.tools = true;
      model.starts_thinking = engine.config.enable_thinking;
      CPUBackend backend(engine, api_cache_bytes,
                         [] { return cpu_api_stop != 0; });
      Server server(backend, model, api_options);
      const auto old_int = std::signal(SIGINT, stop_cpu_api);
      const auto old_term = std::signal(SIGTERM, stop_cpu_api);
      server.run([] { return cpu_api_stop != 0; });
      std::signal(SIGINT, old_int);
      std::signal(SIGTERM, old_term);
      return 0;
    } catch (const std::exception &e) {
      std::cerr << "[CPU-API] " << e.what() << "\n";
      return 1;
    }
  }

  // Startup/model loading is complete here. In CLI mode the prompt argument
  // is already available, so this is the closest equivalent to receiving the
  // first input token; prompt encoding and all prefill work are timed below.
  const auto prefill_start = std::chrono::steady_clock::now();

  // Mirror main.cpp's own prompt-encoding convention: a chat-tuned model
  // gets its detected special-token wrapper, everything else gets a raw
  // BOS + text encoding. Sending neither (as this file previously did)
  // gives the model a token sequence it never saw start a request during
  // training.
  std::vector<int> input_ids;
  if (!engine.config.raw_prompt &&
      detect_model_family(engine.model_file) == ModelFamily::GLM5NEXT &&
      engine.tokenizer.has_chat_template()) {
    if (engine.tokenizer.gmask_id >= 0)
      input_ids.push_back(engine.tokenizer.gmask_id);
    if (engine.tokenizer.sop_id >= 0)
      input_ids.push_back(engine.tokenizer.sop_id);
    if (engine.config.enable_thinking) {
      std::string effort = engine.config.reasoning_effort;
      std::string effort_cap = effort == "low" ? "Low"
                             : effort == "high" ? "High" : "Max";
      std::vector<int> reff = engine.tokenizer.encode(
          "<|system|>Reasoning Effort: " + effort_cap);
      input_ids.insert(input_ids.end(), reff.begin(), reff.end());
    }
    if (!engine.config.system_prompt.empty()) {
      std::vector<int> sys =
          engine.tokenizer.encode("<|system|>" + engine.config.system_prompt);
      input_ids.insert(input_ids.end(), sys.begin(), sys.end());
    }
    input_ids.push_back(engine.tokenizer.user_id);
    std::vector<int> body = engine.tokenizer.encode(prompt);
    input_ids.insert(input_ids.end(), body.begin(), body.end());
    input_ids.push_back(engine.tokenizer.assistant_id);
    if (engine.config.enable_thinking) {
      if (engine.tokenizer.ds_think_start_id >= 0)
        input_ids.push_back(engine.tokenizer.ds_think_start_id);
    } else if (engine.tokenizer.ds_think_start_id >= 0 &&
               engine.tokenizer.ds_think_end_id >= 0) {
      // GLM direct-answer mode requires an explicit empty reasoning block.
      input_ids.push_back(engine.tokenizer.ds_think_start_id);
      input_ids.push_back(engine.tokenizer.ds_think_end_id);
    }
  } else if (!engine.config.raw_prompt && engine.tokenizer.has_chat_template()) {
    if (engine.tokenizer.gmask_id >= 0)
      input_ids.push_back(engine.tokenizer.gmask_id);
    if (engine.tokenizer.sop_id >= 0)
      input_ids.push_back(engine.tokenizer.sop_id);
    // See main.cpp's equivalent branch: real GLM-4 template renders every
    // message (including "system") as <|{role}|>\n{content} ahead of the
    // user turn; there's no reserved single-token id for it, so it's sent
    // as literal text.
    if (!engine.config.system_prompt.empty()) {
      std::vector<int> sys =
          engine.tokenizer.encode("<|system|>\n" + engine.config.system_prompt);
      input_ids.insert(input_ids.end(), sys.begin(), sys.end());
    }
    input_ids.push_back(engine.tokenizer.user_id);
    std::vector<int> body = engine.tokenizer.encode("\n" + prompt);
    input_ids.insert(input_ids.end(), body.begin(), body.end());
    input_ids.push_back(engine.tokenizer.assistant_id);
    std::vector<int> tail = engine.tokenizer.encode("\n");
    input_ids.insert(input_ids.end(), tail.begin(), tail.end());
  } else if (!engine.config.raw_prompt &&
             engine.tokenizer.has_deepseek_chat_template()) {
    // DeepSeek-V3/V4-style: {bos}<｜User｜>{msg}<｜Assistant｜></think>
    // -- mirrors main.cpp's equivalent branch; see
    // GGUFTokenizer::ds_user_id for the reference source.
    if (engine.tokenizer.add_bos_token)
      input_ids.push_back(engine.tokenizer.bos_token_id);
    // Matches the real jinja template exactly: system-role content sits
    // as raw text directly after {bos_token} and before <｜User｜>, with no
    // wrapping tag -- see main.cpp's equivalent branch.
    if (!engine.config.system_prompt.empty()) {
      std::vector<int> sys = engine.tokenizer.encode(engine.config.system_prompt);
      input_ids.insert(input_ids.end(), sys.begin(), sys.end());
    }
    input_ids.push_back(engine.tokenizer.ds_user_id);
    std::vector<int> body = engine.tokenizer.encode(prompt);
    input_ids.insert(input_ids.end(), body.begin(), body.end());
    input_ids.push_back(engine.tokenizer.ds_assistant_id);
    if (engine.tokenizer.ds_think_end_id >= 0)
      input_ids.push_back(engine.tokenizer.ds_think_end_id);
  } else if (!engine.config.raw_prompt &&
             engine.tokenizer.has_chatml_template()) {
    // Qwen3.8-Flash-Next uses ChatML. Thinking is configurable; disabling it
    // omits the long default xhigh reasoning instruction from short prompts.
    const char *reasoning =
        "Reasoning effort is set to xhigh. Please think carefully through "
        "the task, validate key assumptions, consider plausible "
        "alternatives, and prioritize correctness, consistency, and "
        "clarity in the final answer.";
    auto push_text = [&](const std::string &text) {
      std::vector<int> ids = engine.tokenizer.encode(text);
      input_ids.insert(input_ids.end(), ids.begin(), ids.end());
    };
    if (engine.config.enable_thinking || !engine.config.system_prompt.empty()) {
      input_ids.push_back(engine.tokenizer.im_start_id);
      if (engine.config.enable_thinking) {
        push_text(std::string("system\n") + reasoning +
                  (engine.config.system_prompt.empty()
                       ? ""
                       : "\n\n" + engine.config.system_prompt));
      } else {
        push_text("system\n" + engine.config.system_prompt);
      }
      input_ids.push_back(engine.tokenizer.im_end_id);
      push_text("\n");
    }
    input_ids.push_back(engine.tokenizer.im_start_id);
    push_text("user\n" + prompt);
    input_ids.push_back(engine.tokenizer.im_end_id);
    push_text("\n");
    input_ids.push_back(engine.tokenizer.im_start_id);
    push_text("assistant\n");
    if (engine.config.enable_thinking) {
      if (engine.tokenizer.ds_think_start_id >= 0)
        input_ids.push_back(engine.tokenizer.ds_think_start_id);
      else
        push_text("<think>");
      push_text("\n");
    } else {
      if (engine.tokenizer.ds_think_start_id >= 0)
        input_ids.push_back(engine.tokenizer.ds_think_start_id);
      else
        push_text("<think>");
      push_text("\n\n");
      if (engine.tokenizer.ds_think_end_id >= 0)
        input_ids.push_back(engine.tokenizer.ds_think_end_id);
      else
        push_text("</think>");
      push_text("\n\n");
    }
  } else {
    // Respect the model's own tokenizer.ggml.add_bos_token metadata --
    // unconditionally prepending BOS regardless of this key fed
    // Qwen3.6-35B-A3B (add_bos_token=False in its real GGUF) a phantom
    // extra leading token llama.cpp never sends it, corrupting the
    // causal_conv1d receptive field from the first forward pass on. See
    // GGUFTokenizer::add_bos_token for the full incident writeup.
    if (engine.tokenizer.add_bos_token)
      input_ids.push_back(engine.tokenizer.bos_token_id);
    std::vector<int> encoded = engine.tokenizer.encode(prompt);
    input_ids.insert(input_ids.end(), encoded.begin(), encoded.end());
  }
  if (input_ids.empty()) {
    std::cerr << "[CPU-MAIN] Tokenizer produced no tokens.\n";
    return 1;
  }
  std::cerr << "[CPU-MAIN] Prompt tokens: " << input_ids.size() << "\n";

  if (engine.config.max_new_tokens == 0) {
    std::cerr << "[CPU-MAIN] Model load smoke test completed; generation skipped.\n";
    return 0;
  }

  const ArchitectureConfig &c = engine.model_file.architecture_config;
  int D = c.hidden;
  int V = c.vocab;

  // Prefill: process all prompt tokens at once.
  std::vector<float> hidden((size_t)input_ids.size() * D);
  engine.qwen4exp.ple_token_history.assign(input_ids.begin(), input_ids.end());
  for (size_t i = 0; i < input_ids.size(); ++i) {
    if (!cpu_embed_token(engine, input_ids[i], hidden.data() + i * D, D)) {
      std::cerr << "[CPU-MAIN] Embedding lookup failed.\n";
      return 1;
    }
  }

  for (int l = 0; l < c.n_layers; ++l) {
    if (!cpu_forward_layer(engine, l, hidden.data(), (int)input_ids.size(),
                           0)) {
      std::cerr << "[CPU-MAIN] Forward layer " << l << " failed.\n";
      return 1;
    }
  }

  // Take last token hidden state and run LM head.
  std::vector<float> logits(V);
  if (!cpu_lm_head(engine, hidden.data() + (input_ids.size() - 1) * D,
                   logits.data(), 1)) {
    std::cerr << "[CPU-MAIN] LM head failed.\n";
    return 1;
  }
  const double prefill_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                    prefill_start)
          .count();
  std::cerr << "[CPU-PERF] Prefill: " << input_ids.size() << " token(s) in "
            << prefill_seconds << " s ("
            << (input_ids.empty() ? 0.0
                                  : input_ids.size() / prefill_seconds)
            << " token/s)\n";

  // DEBUG: print top-5 logits after prefill
  {
    bool has_nan = false;
    for (float v : logits)
      if (std::isnan(v) || std::isinf(v))
        has_nan = true;
    std::vector<int> idx(V);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    std::cerr << "[CPU-MAIN-DEBUG] Prefill logits NaN/Inf=" << has_nan
              << " top-5:\n";
    for (int k = 0; k < 5; ++k)
      std::cerr << "  " << idx[k] << " '" << engine.tokenizer.decode(idx[k])
                << "' " << logits[idx[k]] << "\n";
  }

  CPUSampler sampler;
  int next_tok =
      sampler.sample(logits, engine.config.temperature, engine.config.top_p);

  std::vector<int> generated;
  generated.reserve(engine.config.max_new_tokens);

  int pos = (int)input_ids.size();
  const auto decode_start = std::chrono::steady_clock::now();
  for (int i = 0; i < engine.config.max_new_tokens; ++i) {
    generated.push_back(next_tok);
    engine.qwen4exp.ple_token_history.push_back(next_tok);

    // Decode one token.
    std::vector<float> single_hidden((size_t)D);
    if (!cpu_embed_token(engine, next_tok, single_hidden.data(), D)) {
      std::cerr << "[CPU-MAIN] Embedding lookup failed.\n";
      return 1;
    }

    for (int l = 0; l < c.n_layers; ++l) {
      if (!cpu_forward_layer(engine, l, single_hidden.data(), 1, pos)) {
        std::cerr << "[CPU-MAIN] Decode layer " << l << " failed.\n";
        return 1;
      }
    }

    if (!cpu_lm_head(engine, single_hidden.data(), logits.data(), 1)) {
      std::cerr << "[CPU-MAIN] LM head failed.\n";
      return 1;
    }

    // DEBUG: per-decode-step top-5, mirroring the prefill block above --
    // added to localize a bug downstream of the (fixed) BOS/prompt-length
    // issue: prefill logits are now provably plausible, but the final
    // generated text still isn't coherent, so something in the step-to-step
    // recurrent state handoff (conv history / KV position / delta-net
    // state) is suspected. Printed BEFORE sampling next_tok so the tag's
    // "step=i" lines up with the token that was just appended to
    // `generated` above (i.e. "step=0" = the state used to predict the
    // token AFTER the first generated one).
    {
      bool has_nan = false;
      for (float v : logits)
        if (std::isnan(v) || std::isinf(v))
          has_nan = true;
      std::vector<int> idx(V);
      std::iota(idx.begin(), idx.end(), 0);
      std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                        [&](int a, int b) { return logits[a] > logits[b]; });
      std::cerr << "[CPU-MAIN-DEBUG] Decode step=" << i << " pos=" << pos
                << " NaN/Inf=" << has_nan << " top-5:\n";
      for (int k = 0; k < 5; ++k)
        std::cerr << "  " << idx[k] << " '" << engine.tokenizer.decode(idx[k])
                  << "' " << logits[idx[k]] << "\n";
    }

    next_tok =
        sampler.sample(logits, engine.config.temperature, engine.config.top_p);
    ++pos;

    if (engine.tokenizer.is_stop_token(next_tok))
      break;
  }
  std::string out_text;
  for (int id : generated)
    out_text += engine.tokenizer.decode(id);
  std::cout << prompt << out_text << "\n" << std::flush;

  // Stop after the final generated token has been written to the console.
  const double decode_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                    decode_start)
          .count();
  std::cerr << "[CPU-PERF] Decode: " << generated.size() << " token(s) in "
            << decode_seconds << " s ("
            << (generated.empty() ? 0.0 : generated.size() / decode_seconds)
            << " token/s)\n";

  // Persist routing histogram for the next run's pin preload.
  if (!engine.config.pin_file.empty() &&
      !engine.route_usage.dump(engine.config.pin_file)) {
    LOG_STREAM_WARN() << "[CPU-MAIN] Failed to write usage file.\n";
  }
  return 0;
}
