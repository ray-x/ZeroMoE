#include "engine_runner.hpp"

#include "generation_loop.hpp"
#include "server/server.hpp"
#include "server/chat_template.hpp"

namespace {
using namespace zeromoe::server;

class NativeBackend final : public Backend {
public:
    NativeBackend(StreamingEngine &engine, RAMSafetyManager &ram, GGUFTokenizer &tokenizer,
                  LockFreePrefetcher &prefetcher, const EngineConfig &config, TemplateKind kind)
        : engine(engine), ram(ram), tokenizer(tokenizer), prefetcher(prefetcher), config(config), kind(kind) {}

    Prepared prepare(const Request &request) override {
        Prepared p;
        const auto segments = render_chat(request, kind, config.enable_thinking,
                                          config.system_prompt, config.reasoning_effort);
        for (const auto &s : segments) {
            if (s.marker == Marker::Text) {
                const auto tokens = tokenizer.encode(s.text);
                p.tokens.insert(p.tokens.end(), tokens.begin(), tokens.end());
                continue;
            }
            int token = -1;
            switch (s.marker) {
            case Marker::Bos: token = tokenizer.bos_token_id; break;
            case Marker::Eos: token = tokenizer.eos_token_id; break;
            case Marker::ImStart: token = tokenizer.im_start_id; break;
            case Marker::ImEnd: token = tokenizer.im_end_id; break;
            case Marker::Gmask: token = tokenizer.gmask_id; break;
            case Marker::Sop: token = tokenizer.sop_id; break;
            case Marker::User:
                token = kind == TemplateKind::DeepSeek ? tokenizer.ds_user_id : tokenizer.user_id;
                break;
            case Marker::Assistant:
                token = kind == TemplateKind::DeepSeek ? tokenizer.ds_assistant_id : tokenizer.assistant_id;
                break;
            case Marker::RoleEnd: token = tokenizer.bailing_role_end_id; break;
            case Marker::ThinkStart: token = tokenizer.ds_think_start_id; break;
            case Marker::ThinkEnd: token = tokenizer.ds_think_end_id; break;
            case Marker::Text: break;
            }
            if (token < 0 && (s.marker == Marker::ThinkStart || s.marker == Marker::ThinkEnd)) {
                const auto tokens = tokenizer.encode(s.marker == Marker::ThinkStart ? "<think>" : "</think>");
                p.tokens.insert(p.tokens.end(), tokens.begin(), tokens.end());
                continue;
            }
            if (token < 0) throw ApiError(400, "loaded tokenizer is missing a required chat marker");
            p.tokens.push_back(token);
        }
        return p;
    }

    Result generate(const Request &request, const Prepared &prepared, const std::atomic_bool &cancel,
                    const std::function<bool(const std::string &)> &emit) override {
        GenerationControl control;
        control.cancelled = [&] { return cancel.load(std::memory_order_relaxed); };
        control.emit = emit;
        generate_text(engine, ram, tokenizer, prefetcher, prepared.tokens, request.max_tokens,
                      tokenizer.eos_token_id, request.temperature, request.top_p, &control);
        if (!engine.qwen4exp_full_graph_failure.empty())
            throw std::runtime_error("full-graph generation failed: " + engine.qwen4exp_full_graph_failure);
        if (!control.completed && !control.interrupted)
            throw std::runtime_error("generation ended before producing a complete result");
        Result result;
        result.usage = {static_cast<int>(prepared.tokens.size()), control.cached_tokens, control.generated_tokens};
        result.finish = control.interrupted ? Finish::Cancelled : control.stopped ? Finish::Stop : Finish::Length;
        return result;
    }
private:
    StreamingEngine &engine;
    RAMSafetyManager &ram;
    GGUFTokenizer &tokenizer;
    LockFreePrefetcher &prefetcher;
    const EngineConfig &config;
    TemplateKind kind;
};
}

std::expected<void, std::string> EngineRunner::run_api() {
    using namespace zeromoe::server;
    session_start = std::chrono::high_resolution_clock::now();
    TemplateKind kind;
    if (engine.family_ops.family == ModelFamily::BailingMoE3 && tokenizer.has_bailing_template())
        kind = TemplateKind::Bailing;
    else if (engine.family_ops.family == ModelFamily::GLM5NEXT && tokenizer.has_chat_template())
        kind = TemplateKind::GlmFlash;
    else if ((engine.family_ops.family == ModelFamily::Qwen4Exp ||
              engine.family_ops.family == ModelFamily::Qwen) &&
             tokenizer.has_chatml_template())
        kind = TemplateKind::Qwen;
    else if (tokenizer.has_chat_template())
        kind = TemplateKind::Glm;
    else if (tokenizer.has_deepseek_chat_template())
        kind = TemplateKind::DeepSeek;
    else if (engine.family_ops.family == ModelFamily::MiMo2 && tokenizer.has_chatml_template())
        kind = TemplateKind::MiMo;
    else return std::unexpected("API serving has no chat template for this model");

    // The REPL cache borrows position-indexed KV. It is not a portable,
    // graph-aware session checkpoint and must not be advertised as one.
    if (engine.prompt_cache)
        LOG_WARN("[API] prompt_cache disabled: complete graph-state checkpoint restore is not validated for serving\n");
    engine.prompt_cache = false;
    engine.prompt_state_cache = {};
    ModelInfo model;
    model.id = config.api_model;
    model.context_size = config.context_size;
    model.default_max_tokens = config.max_new_tokens;
    model.max_output_tokens = config.max_new_tokens;
    model.temperature = config.temperature;
    model.top_p = config.top_p;
    model.tools = kind == TemplateKind::Qwen;
    model.starts_thinking = config.enable_thinking && kind != TemplateKind::DeepSeek &&
                            kind != TemplateKind::Glm && kind != TemplateKind::MiMo;
    Options options;
    options.host = config.api_host;
    options.port = config.api_port;
    if (!config.api_key.empty()) options.api_key = config.api_key;
    else if (const char *key = std::getenv("ZEROMOE_API_KEY")) options.api_key = key;
    NativeBackend backend(engine, ram_mgr, tokenizer, prefetcher, config, kind);
    try {
        Server server(backend, model, options);
        server.run([] { return g_shutdown_requested.load(std::memory_order_relaxed); });
    } catch (const std::exception &e) {
        LOG_ERROR("[API] server failed: %s\n", e.what());
        return std::unexpected(e.what());
    }
    return {};
}
