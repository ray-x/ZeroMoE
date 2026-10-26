// EngineRunner phases: start_workers, run_repl, shutdown_and_report.
// Split from the former run_engine() in src/main.cpp; bodies are unchanged except where
// a former local became an EngineRunner member of the same name.
#include "ggml_mimo2_graph.hpp"
#include "engine_runner.hpp"
#include "generation_loop.hpp"
#include "bench_mode.hpp"

std::expected<void, std::string> EngineRunner::start_workers() {
  const FullGraphModel *graph_model = engine.family_ops.graph_model;
  // Spun up here, AFTER the [PIN] block above has fully finished writing
  // into engine.layers_experts (Tier-3 RAM preload) and gpu.topology
  // (Tier-1/Tier-2 VRAM preload) -- worker threads claim/populate/publish
  // directly into that same RAM-tier structure via RAMSafetyManager::
  // claim_or_wait (see async_prefetcher.hpp's active_prefetch), so
  // starting them only once startup preload is done removes any need to
  // reason about ordering between the two, rather than relying on "no
  // push_hint call happens until the decode loop starts" as an implicit
  // timing argument.
  LOG_STREAM_INFO()
      << "[I/O] Spinning up asynchronous lock-free SPSC prefetch threads...\n";
  prefetcher.start(&engine, &gpu.topology, config.io_uring,
                   config.prefetch_worker_threads);
  // Unified claim/publish design: the prefetcher calls INTO
  // RAMSafetyManager::claim_or_wait, so it needs a pointer back to it --
  // the reverse of the old (now-removed) RAMSafetyManager::prefetcher
  // field. Raw, non-owning: ram_mgr outlives prefetcher for the whole
  // process lifetime (both are stack-local to main(), ram_mgr declared
  // first).
  prefetcher.ram_mgr = &ram_mgr;

  // 2026-08-28: active-layer upload worker pool -- see EngineConfig::
  // active_layer_upload_worker_threads' own comment and active_layer_
  // upload_workers.hpp's header comment for the full design. Constructed
  // here (after gpu.topology's streams/pool exist, same ordering
  // constraint the prefetcher's own start() call above already respects)
  // rather than at the ram_mgr/prefetcher declaration site above, since
  // the pool's ctor immediately spawns worker threads that will touch
  // gpu.topology.devices[...] the moment real work arrives.
  if (config.active_layer_upload_worker_threads > 0) {
    upload_worker_pool = std::make_unique<ActiveLayerUploadWorkers>(
        gpu.topology, ram_mgr, config.active_layer_upload_worker_threads);
    ram_mgr.upload_workers = upload_worker_pool.get();
    LOG_STREAM_INFO() << "[GPU] Active-layer upload worker pool: "
                      << config.active_layer_upload_worker_threads
                      << " thread(s) started.\n";
  }

  // Dequant attn_k_b/attn_v_b (MLA's absorbed-attention weights) for
  // every layer NOW, once, model load fully done but before any prompt
  // is timed -- see warm_mla_absorb_cache's own comment
  // (attention_layer.cpp) for the real measurement that motivated this
  // (1.59s previously landing inside the FIRST prefill call's own
  // "absorb" sub-phase, confirmed to be a one-time, model-only cost
  // completely independent of the prompt). No-op/harmless for
  // architectures that don't use MLA-absorbed attention (DeepSeek-V4's
  // own dsv4_attention_gguf path uses a structurally separate woa_cache
  // this doesn't touch; the function itself early-returns for any model
  // with kv_lora<=0).
  extern void warm_mla_absorb_cache(StreamingEngine & engine);
  warm_mla_absorb_cache(engine);

  // The full GLM-5.3-Flash graph needs a host-side source view of the
  // routed experts, but that source is independent of prompt length.  Load
  // it now, before accepting the first prompt, so a later prompt with a
  // different token count only rebuilds its small GGML topology and never
  // rereads the ~78 GiB expert set from SSD.  The source admission is
  // bounded by host_ram_gb and the memory watchdog; failure leaves the
  // established layer path available.
  if (graph_model) {
    graph_model->bind_vram_cache(&gpu.topology);
    log_graph_capabilities(*graph_model, engine);
  }
  if (graph_mode && !graph_model->bootstrap(engine)) {
    LOG_STREAM_WARN()
        << "[GRAPH] graph bootstrap unavailable; "
           "the graph will attempt bounded lazy preparation on the first "
           "prompt.\n";
  }

  std::cout << "----------------------------------------\n";
  std::cout << "  ZeroMoE engine online and ready. Enter prompt below.\n";
  std::cout << "Use ':q' to exit execution loop cleanly.\n\n";

  // Configure only after all startup worker pools have been created.  New
  // threads inherit the creator's affinity mask, so doing this at the top of
  // run_engine() would accidentally constrain prefetch/GGML workers too.
  configure_current_thread({
      .name = "MAIN-DECODE",
      // Main decode and CUDA callbacks intentionally float.  Priority and
      // the ring's wait policy are the variables under test; a fixed core
      // was measured to create a static contention point.
      .cpu_ids = {},
      .use_rt = true,
      .rt_priority = 80,
      .nice_value = -20,
      .apply_nice = false,
  });

  return {};
}

std::expected<void, std::string> EngineRunner::run_repl() {
  int max_new_tokens = config.max_new_tokens;
  // Real wall-clock anchor for perf().dump_report()'s final Throughput
  // line -- spans every turn in this session (not just one request),
  // starting AFTER model load (reported separately as "Dense load
  // (init)") so it isolates generation time specifically. See
  // dump_report()'s own comment for why this replaced a broken
  // summed-perf-counter calculation that double-counted concurrent work.
  session_start = std::chrono::high_resolution_clock::now();

  std::string user_input;
  // --bench: the turns come from a scripted queue instead of stdin; everything after the turn is read is the
  // normal request path (see core/bench_mode.hpp).
  zeromoe::bench::Session bench(zeromoe::bench::plan());
  if (bench.enabled()) {
    const std::string err = bench.prepare(engine.prompt_cache, engine.model_file.architecture_config.vocab);
    if (!err.empty())
      return std::unexpected(err);
  }
  while (true) {
    if (g_shutdown_requested.load(std::memory_order_relaxed))
      break;

    if (bench.enabled()) {
      if (!bench.next(user_input, max_new_tokens))
        break;
    } else {
      std::cout << "\n  User > ";
      std::getline(std::cin, user_input);
    }

    // Multi-line input support (2026-09): this loop reads exactly one
    // getline() per turn, so a genuinely multi-line message piped via
    // stdin (e.g. "review this code:\n```python\ndef f():\n...") was
    // silently split into N SEPARATE turns, one per embedded newline —
    // found via a real survey harness where a code-review prompt with
    // an embedded function produced 6 fragmented, wrong generations
    // instead of one coherent one. Rather than degrade callers to
    // single-line-only content, accept a literal two-character "\n"
    // escape sequence (backslash + 'n') as a caller-side convention for
    // "real newline here" and unescape it right after reading, before
    // any tokenization -- a caller that wants a true multi-line message
    // in one turn sends its text with real newlines replaced by this
    // literal sequence. Interactive human typing is unaffected (nobody
    // types a literal backslash-n expecting it to stay literal in a
    // prompt to an LLM in practice, and if they did, the model still
    // sees intelligible text either way).
    {
      std::string unescaped;
      unescaped.reserve(user_input.size());
      for (size_t i = 0; i < user_input.size(); ++i) {
        if (user_input[i] == '\\' && i + 1 < user_input.size() &&
            user_input[i + 1] == 'n') {
          unescaped.push_back('\n');
          ++i;
        } else {
          unescaped.push_back(user_input[i]);
        }
      }
      user_input = std::move(unescaped);
    }

    // Ctrl-C (SIGINT) sets the flag and, on most platforms, interrupts
    // the blocking getline with a failed/EOF stream state. Treat any of
    // those — flag set, real EOF (Ctrl-D / piped input ending), or a
    // failed read — as "time to shut down" rather than looping forever
    // on an unusable stream.
    if (g_shutdown_requested.load(std::memory_order_relaxed) ||
        (!bench.enabled() && (std::cin.eof() || std::cin.fail()))) {
      break;
    }

    if (user_input == ":q" || user_input == "exit") {
      break;
    }
    if (user_input.empty())
      continue;

    // BUG (found this session): user_input was read but never actually
    // tokenized -- prompt_tokens only ever contained BOS, so the model
    // never saw what the user typed, on every single request. That's
    // very likely the real explanation for "the same output regardless
    // of prompt": the input to the model WAS identical every time.
    std::vector<int> prompt_tokens;
  if (!config.raw_prompt &&
      engine.family_ops.family == ModelFamily::BailingMoE3 &&
      tokenizer.has_bailing_template()) {
    auto append_text = [&](const std::string &text) {
      auto ids = tokenizer.encode(text);
      prompt_tokens.insert(prompt_tokens.end(), ids.begin(), ids.end());
    };
    append_text("<role>SYSTEM</role>");
    if (!config.system_prompt.empty())
      append_text(config.system_prompt + "\n");
    append_text(config.enable_thinking ? "detailed thinking on"
                                       : "detailed thinking off");
    prompt_tokens.push_back(tokenizer.bailing_role_end_id);
    append_text("<role>HUMAN</role>" + user_input);
    prompt_tokens.push_back(tokenizer.bailing_role_end_id);
    append_text("<role>ASSISTANT</role>\n");
    if (tokenizer.ds_think_start_id >= 0)
      prompt_tokens.push_back(tokenizer.ds_think_start_id);
    else
      append_text("<think>");
    if (!config.enable_thinking) {
      if (tokenizer.ds_think_end_id >= 0)
        prompt_tokens.push_back(tokenizer.ds_think_end_id);
      else
        append_text("</think>");
    }
  } else if (!config.raw_prompt &&
             engine.family_ops.family == ModelFamily::GLM5NEXT &&
             tokenizer.has_chat_template()) {
      // GLM-5.3-Flash's real template is distinct from the older GLM
      // template even though both use [gMASK], <sop>, <|user|>, and
      // <|assistant|>. It has no newline after role tags. Thinking is
      // enabled by an open <think> block; direct-answer mode is the
      // explicit empty <think></think> block used by the GLM template.
      if (tokenizer.gmask_id >= 0)
        prompt_tokens.push_back(tokenizer.gmask_id);
      if (tokenizer.sop_id >= 0)
        prompt_tokens.push_back(tokenizer.sop_id);

      if (config.enable_thinking) {
        std::string effort = config.reasoning_effort;
        std::string effort_cap;
        if (effort == "low")
          effort_cap = "Low";
        else if (effort == "high")
          effort_cap = "High";
        else
          effort_cap = "Max";
        std::vector<int> reff =
            tokenizer.encode("<|system|>Reasoning Effort: " + effort_cap);
        prompt_tokens.insert(prompt_tokens.end(), reff.begin(), reff.end());
      }

      if (!config.system_prompt.empty()) {
        std::vector<int> sys =
            tokenizer.encode("<|system|>" + config.system_prompt);
        prompt_tokens.insert(prompt_tokens.end(), sys.begin(), sys.end());
      }
      prompt_tokens.push_back(tokenizer.user_id);
      std::vector<int> body = tokenizer.encode(user_input);
      prompt_tokens.insert(prompt_tokens.end(), body.begin(), body.end());
      prompt_tokens.push_back(tokenizer.assistant_id);
      if (config.enable_thinking) {
        if (tokenizer.ds_think_start_id >= 0)
          prompt_tokens.push_back(tokenizer.ds_think_start_id);
      } else {
        // GLM's direct-answer switch is not "no think token". The
        // model expects the assistant turn to contain an explicitly
        // empty reasoning block: <think></think>.
        if (tokenizer.ds_think_start_id >= 0 &&
            tokenizer.ds_think_end_id >= 0) {
          prompt_tokens.push_back(tokenizer.ds_think_start_id);
          prompt_tokens.push_back(tokenizer.ds_think_end_id);
        }
      }
    } else if (!config.raw_prompt && tokenizer.has_chat_template()) {
      // GLM-3/4-style: [gMASK]<sop><|user|>\n{msg}<|assistant|>\n
      // (see GGUFTokenizer::detect_chat_special_tokens for the
      // reference source and the caveat that GLM-5.2 specifically
      // isn't verified against this).
      if (tokenizer.gmask_id >= 0)
        prompt_tokens.push_back(tokenizer.gmask_id);
      if (tokenizer.sop_id >= 0)
        prompt_tokens.push_back(tokenizer.sop_id);
      // Real GLM-4 template renders EVERY message (including "system")
      // as <|{role}|>\n{content} in turn order, ahead of the user
      // turn -- see llama.cpp's LLM_CHAT_TEMPLATE_CHATGLM_4. There's
      // no reserved single-token id for "<|system|>" the way
      // user_id/assistant_id have, so it's sent as literal text.
      if (!config.system_prompt.empty()) {
        std::vector<int> sys =
            tokenizer.encode("<|system|>\n" + config.system_prompt);
        prompt_tokens.insert(prompt_tokens.end(), sys.begin(), sys.end());
      }
      prompt_tokens.push_back(tokenizer.user_id);
      std::vector<int> body = tokenizer.encode("\n" + user_input);
      prompt_tokens.insert(prompt_tokens.end(), body.begin(), body.end());
      prompt_tokens.push_back(tokenizer.assistant_id);
      std::vector<int> tail = tokenizer.encode("\n");
      prompt_tokens.insert(prompt_tokens.end(), tail.begin(), tail.end());
    } else if (!config.raw_prompt && tokenizer.has_deepseek_chat_template()) {
      // DeepSeek-V3/V4-style: {bos}<｜User｜>{msg}<｜Assistant｜></think>
      // (see GGUFTokenizer::ds_user_id for the reference source --
      // this is what fixed confidently-wrong, off-topic decode output
      // for a plain unwrapped prompt on DeepSeek-V4-Flash). No literal
      // "\n" padding around the tags, unlike the GLM branch above --
      // the real jinja template concatenates directly.
      // DeepSeek's chat template explicitly emits bos_token.  This is
      // independent of tokenizer.ggml.add_bos_token: that metadata flag
      // controls the tokenizer's automatic prefix, while the rendered
      // template still contains {{ bos_token }}.  V4.1 sets add_bos_token
      // to false, so guarding this on that flag drops the required first
      // token and shifts the entire chat prompt.
      prompt_tokens.push_back(tokenizer.bos_token_id);
      // Matches the real jinja template exactly: system-role content
      // is concatenated as raw text directly after {bos_token} and
      // before <｜User｜>, with no wrapping tag at all (see
      // ns.system_prompt / {{- ns.system_prompt -}} in
      // models/templates/deepseek-ai-DeepSeek-V4-Flash-0731.jinja).
      if (!config.system_prompt.empty()) {
        std::vector<int> sys = tokenizer.encode(config.system_prompt);
        prompt_tokens.insert(prompt_tokens.end(), sys.begin(), sys.end());
      }
      prompt_tokens.push_back(tokenizer.ds_user_id);
      std::vector<int> body = tokenizer.encode(user_input);
      prompt_tokens.insert(prompt_tokens.end(), body.begin(), body.end());
      prompt_tokens.push_back(tokenizer.ds_assistant_id);
      if (tokenizer.ds_think_end_id >= 0)
        prompt_tokens.push_back(tokenizer.ds_think_end_id);
    } else if (!config.raw_prompt &&
               engine.family_ops.family == ModelFamily::MiMo2 &&
               tokenizer.has_chatml_template()) {
      // MiMo's embedded text template has no newline between messages and
      // uses exactly <think></think> to request a direct answer. In particular,
      // Qwen's reasoning instruction and whitespace are not interchangeable.
      auto push_text = [&](const std::string &text) {
        auto ids = tokenizer.encode(text);
        prompt_tokens.insert(prompt_tokens.end(), ids.begin(), ids.end());
      };
      auto message = [&](const std::string &role, const std::string &body) {
        prompt_tokens.push_back(tokenizer.im_start_id);
        push_text(role + "\n" + body);
        prompt_tokens.push_back(tokenizer.im_end_id);
      };
      if (!config.system_prompt.empty()) message("system", config.system_prompt);
      message("user", user_input);
      prompt_tokens.push_back(tokenizer.im_start_id);
      push_text("assistant\n");
      if (!config.enable_thinking) {
        if (tokenizer.ds_think_start_id >= 0)
          prompt_tokens.push_back(tokenizer.ds_think_start_id);
        else
          push_text("<think>");
        if (tokenizer.ds_think_end_id >= 0)
          prompt_tokens.push_back(tokenizer.ds_think_end_id);
        else
          push_text("</think>");
      }
    } else if (!config.raw_prompt && tokenizer.has_chatml_template()) {
      // ChatML (Qwen3.8-Flash-Next / qwen4exp), per the
      // model's tokenizer_config.json chat_template:
      //   <|im_start|>system\n{reasoning_instructions}[\n\n{system}]<|im_end|>\n
      //   <|im_start|>user\n{msg}<|im_end|>\n
      //   <|im_start|>assistant\n<think>\n
      // enable_thinking defaults to true for reference compatibility;
      // false omits the default reasoning system instruction.
      const char *kQwen4ExpReasoningXHigh =
          "Reasoning effort is set to xhigh. Please think carefully through "
          "the task, validate key assumptions, consider plausible "
          "alternatives, and prioritize correctness, consistency, and "
          "clarity in the final answer.";
      auto push_text = [&](const std::string &text) {
        std::vector<int> ids = tokenizer.encode(text);
        prompt_tokens.insert(prompt_tokens.end(), ids.begin(), ids.end());
      };
      if (config.enable_thinking || !config.system_prompt.empty()) {
        prompt_tokens.push_back(tokenizer.im_start_id);
        if (config.enable_thinking) {
          push_text(std::string("system\n") + kQwen4ExpReasoningXHigh +
                    (config.system_prompt.empty()
                         ? ""
                         : "\n\n" + config.system_prompt));
        } else {
          push_text("system\n" + config.system_prompt);
        }
        prompt_tokens.push_back(tokenizer.im_end_id);
        push_text("\n");
      }
      prompt_tokens.push_back(tokenizer.im_start_id);
      push_text("user\n" + user_input);
      prompt_tokens.push_back(tokenizer.im_end_id);
      push_text("\n");
      prompt_tokens.push_back(tokenizer.im_start_id);
      // The Qwen template puts the thinking markers in the rendered
      // text.  They are not special-token entries in this checkpoint's
      // vocabulary, so checking ds_think_start_id alone silently
      // dropped the whole reasoning block and left the model at
      // `<|im_start|>assistant\n`.  Preserve the template literally;
      // use the special-token IDs when a future Qwen vocabulary has
      // them, otherwise let the normal BPE encoder produce the IDs.
      push_text("assistant\n");
      if (config.enable_thinking) {
        if (tokenizer.ds_think_start_id >= 0)
          prompt_tokens.push_back(tokenizer.ds_think_start_id);
        else
          push_text("<think>");
        push_text("\n");
      } else {
        if (tokenizer.ds_think_start_id >= 0)
          prompt_tokens.push_back(tokenizer.ds_think_start_id);
        else
          push_text("<think>");
        push_text("\n\n");
        if (tokenizer.ds_think_end_id >= 0)
          prompt_tokens.push_back(tokenizer.ds_think_end_id);
        else
          push_text("</think>");
        push_text("\n\n");
      }
    } else {
      // Respect tokenizer.ggml.add_bos_token -- see
      // GGUFTokenizer::add_bos_token for why unconditionally pushing
      // BOS here is a real, confirmed bug for models (e.g.
      // Qwen3.6-35B-A3B) whose GGUF sets this to False.
      if (tokenizer.add_bos_token)
        prompt_tokens.push_back(tokenizer.bos_token_id);
      std::vector<int> encoded = tokenizer.encode(user_input);
      prompt_tokens.insert(prompt_tokens.end(), encoded.begin(), encoded.end());
    }

    // Diagnostic/fairness hook: Llama.cpp's llama-bench can consume an
    // exact token-ID file. Allow its token IDs to be replayed here so the
    // two engines execute the same prompt, independent of small tokenizer
    // wrapper differences. The normal text prompt path is unchanged when
    // this variable is absent.
    if (const char *token_input =
            std::getenv("ZEROMOE_PROMPT_TOKEN_INPUT_FILE")) {
      std::ifstream input(token_input);
      std::vector<int> exact_tokens;
      long long value = 0;
      while (input >> value) {
        if (value < INT_MIN || value > INT_MAX) {
          exact_tokens.clear();
          break;
        }
        exact_tokens.push_back((int)value);
      }
      if (!input.eof() || exact_tokens.empty()) {
        LOG_STREAM_ERROR() << "[MOE-PROMPT] failed to read token input file: "
                           << token_input << "\n";
      } else {
        prompt_tokens = std::move(exact_tokens);
        LOG_STREAM_INFO() << "[MOE-PROMPT] loaded " << prompt_tokens.size()
                          << " exact token IDs from " << token_input << "\n";
      }
    }
    if (std::getenv("ZEROMOE_MOE_SET_TRACE") != nullptr) {
      uint64_t hash = 1469598103934665603ULL;
      for (int token : prompt_tokens) {
        uint32_t value = (uint32_t)token;
        for (int i = 0; i < 4; ++i) {
          hash ^= (value >> (i * 8)) & 0xffU;
          hash *= 1099511628211ULL;
        }
      }
      LOG_STREAM_INFO() << "[MOE-PROMPT] tokens=" << prompt_tokens.size()
                        << " fnv64=0x" << std::hex << hash << std::dec << "\n";
    }
    if (const char *token_path = std::getenv("ZEROMOE_PROMPT_TOKEN_FILE")) {
      std::ofstream output(token_path);
      if (!output) {
        LOG_STREAM_ERROR()
            << "[MOE-PROMPT] failed to open token file for writing: "
            << token_path << "\n";
      } else {
        for (int token : prompt_tokens)
          output << token << "\n";
        LOG_STREAM_INFO() << "[MOE-PROMPT] wrote " << prompt_tokens.size()
                          << " token IDs to " << token_path << "\n";
      }
    }

    std::cout << "\n  ZeroMoE * ";
    generate_text(engine, ram_mgr, tokenizer, prefetcher, prompt_tokens,
                  max_new_tokens, tokenizer.eos_token_id, config.temperature,
                  config.top_p);
    if (bench.enabled())
      bench.record();
    if (!engine.qwen4exp_full_graph_failure.empty()) {
      const std::string reason = engine.qwen4exp_full_graph_failure;
      engine.qwen4exp_full_graph_failure.clear();
      const FullGraphModel *graph_model = engine.family_ops.graph_model;
      request_failure =
          std::string(graph_model ? graph_model->name() : "model") +
          " full-graph request failed: " + reason;
      std::cout << "\n";
      LOG_STREAM_ERROR() << request_failure << "\n";
      // Leave through the normal teardown path below so the prefetch worker
      // is joined and graph-owned CUDA allocations are released before the
      // request error reaches main().
      break;
    }

    std::cout << "\n";
  }

  if (bench.enabled())
    bench.finish();
  return {};
}

std::expected<void, std::string> EngineRunner::shutdown_and_report() {
  const FullGraphModel *graph_model = engine.family_ops.graph_model;
  std::string graph_gate_failure;
  if (engine.cuda_moe_graph_policy_applies) {
    const double graph_decode_run_ms =
        static_cast<double>(engine.full_graph_decode_run_ns) / 1.0e6;
    const double graph_decode_tps = engine.full_graph_decode_run_ns > 0
        ? static_cast<double>(engine.full_graph_decode_runs) * 1.0e9 /
              static_cast<double>(engine.full_graph_decode_run_ns)
        : 0.0;
    LOG_INFO("[MOE-GRAPH-GATE] family=%s policy=%s memory_plan=%s graph_decode_runs=%llu graph_decode_run_ms=%.3f steady_decode_tps=%.2f layered_s1_forwards=%llu legacy_s1_expert_dispatches=%llu\n",
             graph_model ? graph_model->name() : "missing",
             engine.cuda_moe_full_graph_decode_required ? "required"
                 : "memory-fallback",
             engine.cuda_moe_graph_memory_plan.c_str(),
             (unsigned long long)engine.full_graph_decode_runs,
             graph_decode_run_ms, graph_decode_tps,
             (unsigned long long)engine.layered_moe_s1_forwards,
             (unsigned long long)engine.legacy_moe_s1_expert_dispatches);
    if (!engine.cuda_moe_layered_fallback_reason.empty())
      LOG_INFO("[MOE-GRAPH-FALLBACK] planner_reason=%s\n",
               engine.cuda_moe_layered_fallback_reason.c_str());
    if (engine.cuda_moe_full_graph_decode_required &&
        (engine.layered_moe_s1_forwards != 0 ||
         engine.legacy_moe_s1_expert_dispatches != 0)) {
      graph_gate_failure =
          "full-graph-required CUDA MoE run entered the legacy layered expert path";
    }
  }
  prefetcher.stop();
  // The selected full graph owns a GGML scheduler and CUDA workspace
  // buffers. Release it through the model graph contract while CudaManager
  // is still alive; otherwise a static graph destructor can run after CUDA
  // driver teardown and cudaFree aborts.
  if (engine.family_ops.graph_model)
    engine.family_ops.graph_model->shutdown();
  // Phase A write this run's routing-frequency histogram so
  // the NEXT run's startup preload (above) has real data to work from.
  // Best-effort, matches dump_usage's own contract (a failed write just
  // logs and continues -- not worth failing shutdown over).
  // freeze_usage_stats (debug/A-B aid): skip both writes entirely so
  // pin_file/prefill_pin_file stay exactly as they were at startup --
  // see EngineConfig::freeze_usage_stats's own comment for why this
  // exists (real, confirmed run-to-run drift in the accumulated
  // histogram, even under an identical prompt).
  if (!config.freeze_usage_stats) {
    engine.dump_usage(config.pin_file);
    // Prefill-only histogram -- see route_freq_prefill's own comment.
    // No-op (dump_usage_histogram's own empty-path guard) if
  if (!engine.exl3_ggml_bridge_model_path.empty() &&
      engine.exl3_gpu_executor) {
    const Exl3GpuStats exl3_stats = engine.exl3_gpu_executor->stats();
    LOG_STREAM_INFO()
        << "[EXL3-GGML-STATS] routed_ffns=" << exl3_stats.routed_ffns
        << " matmuls=" << exl3_stats.matmuls
        << " gpu_ffn_ms=" << exl3_stats.routed_ffn_seconds * 1000.0
        << " routed_graph_captures=" << exl3_stats.routed_graph_captures
        << " routed_graph_replays=" << exl3_stats.routed_graph_replays
        << " routed_graph_eager_fallbacks="
        << exl3_stats.routed_graph_eager_fallbacks
        << " cache_uploads=" << exl3_stats.uploads
        << " cache_hits=" << exl3_stats.cache_hits << "\n";
  }
    // prefill_pin_file isn't configured.
    engine.dump_usage_prefill(config.prefill_pin_file);
  } else {
    LOG_STREAM_INFO() << "[USAGE] freeze_usage_stats=true -- skipping "
                         "usage-file write(s) this "
                         "session ("
                      << config.pin_file
                      << (config.prefill_pin_file.empty()
                              ? ""
                              : (", " + config.prefill_pin_file))
                      << " left untouched).\n";
  }

  // Cache-churn summary for both tiers: which experts were swapped in/out
  // the most, and (VRAM only, since that's the only tier with a pinning
  // mechanism) which of those are still unpinned
  // Safe without locking here: prefetcher.stop() above has already joined
  // the worker thread, so nothing else touches these structures now.
  ram_mgr.report_cache_churn();
  gpu.topology.report_cache_churn();
  ram_mgr.report_slot_summary(engine);
  gpu.topology.report_slot_summary();
  ram_mgr.report_tier_map(engine);
  prefetcher.log_per_layer_report(); // [PREFETCH-LAYER]: per-layer prefetch
                                     // hint/fetch/outcome breakdown
  report_named_timers();     // attn_q_a/attn_q_b/attn_kv_a_mqa fast-path
                             // investigation, see CLAUDE.md
  report_layer_miss_stats(); // [LAYER-MISS]: per-layer decode-time real
                             // disk-fetch misses -- see CLAUDE.md CPU-offload
                             // data-gathering entry
  report_pilot_recall_by_layer(); // [PILOT-BY-LAYER]: per-layer
                                  // DEPTH-1/ATTN-PILOT/DEPTH-2 prediction
                                  // accuracy -- see CLAUDE.md per-layer
                                  // prediction accuracy entry
  report_qwen4exp_pilot_stats();  // [QWEN-PILOT]: Qwen HC-proxy prediction
                                  // accuracy and load coverage
  report_moe_bucket_stats(); // [MOE-BUCKET]: real token-sharing rate for MoE
                             // expert dispatch -- the llama.cpp -ub/mul_mat_id
                             // batching investigation
  report_prefill_lookahead_stats(); // [PREFILL-LOOKAHEAD]:
                                    // prefill_dispatch_lookahead_tokens real
                                    // added/skipped counts
  report_pilot_wide_stats();
  report_prefill_window_stats(); // [PREFILL-WINDOW]: prefill sliding-window
                                 // capacity (P1+P2)  // [PILOT-WIDE]:
                                 // pilot_widen_all_tokens real recall/coverage
                                 // measurement
  extern void deepseek41_engram_print_cache_stats();
  deepseek41_engram_print_cache_stats(); // [DSV41-ENGRAM-CACHE]: real
                                         // hit-rate for the Engram row
                                         // cache (goal item 4), never
                                         // measured before this session

  double session_elapsed_s =
      std::chrono::duration<double>(std::chrono::high_resolution_clock::now() -
                                    session_start)
          .count();
  perf().dump_report(session_elapsed_s);
  if (!request_failure.empty())
    return std::unexpected(request_failure);
  if (!graph_gate_failure.empty())
    return std::unexpected(graph_gate_failure);
  return {};
}
