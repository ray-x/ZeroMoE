#include "cpu_api.hpp"
#include "cpu_engine.hpp"
#include "cpu_model_backend.hpp"
#include "server/chat_template.hpp"

#include <cmath>
#include <cstdio>
#include <list>

namespace zeromoe::server {
struct CPUBackend::Impl {
    CPUEngine &engine;
    size_t budget;
    std::function<bool()> cancelled;
    size_t used = 0;
    struct Snapshot {
        std::vector<int> tokens;
        std::unique_ptr<CPUBackendState> recurrent;
        std::vector<int> layers;
        std::vector<float> k, v, logits;
        int64_t t_gen = 0;
        size_t bytes = 0;
    };
    std::list<Snapshot> cache;
    Impl(CPUEngine &engine, size_t budget, std::function<bool()> cancelled)
        : engine(engine), budget(budget), cancelled(std::move(cancelled)) {
        if (!engine.backend || engine.backend->family() != ModelFamily::Qwen ||
            !engine.backend->supports_sequence_state())
            throw ApiError(400, "CPU API currently requires the Qwen MoE sequence-state backend");
    }
    void check_cancel(const std::atomic_bool &cancel) const {
        if (cancel || cancelled()) throw ApiError(408, "generation cancelled");
    }
    std::vector<float> forward(int token, int pos, const std::atomic_bool &cancel) {
        check_cancel(cancel);
        const auto &c = engine.model_file.architecture_config;
        if (token < 0 || token >= c.vocab || pos < 0 || pos >= engine.max_t)
            throw ApiError(400, "token or position outside model limits");
        std::vector<float> hidden(c.hidden), logits(c.vocab);
        if (!cpu_embed_token(engine, token, hidden.data(), c.hidden))
            throw std::runtime_error("CPU embedding lookup failed");
        for (int l = 0; l < c.n_layers; ++l) {
            check_cancel(cancel);
            if (!cpu_forward_layer(engine, l, hidden.data(), 1, pos))
                throw std::runtime_error("CPU forward failed at layer " + std::to_string(l));
            if (!std::all_of(hidden.begin(), hidden.end(), [](float x) { return std::isfinite(x); }))
                throw std::runtime_error("non-finite CPU hidden state");
        }
        if (!cpu_lm_head(engine, hidden.data(), logits.data(), 1) ||
            !std::all_of(logits.begin(), logits.end(), [](float x) { return std::isfinite(x); }))
            throw std::runtime_error("CPU LM head failed or produced non-finite logits");
        engine.t_gen = pos + 1;
        return logits;
    }
    int restore(const std::vector<int> &tokens, std::vector<float> &logits, bool reuse) {
        auto best = cache.end();
        if (reuse)
            for (auto it = cache.begin(); it != cache.end(); ++it)
                if (it->tokens.size() <= tokens.size() &&
                    (best == cache.end() || it->tokens.size() > best->tokens.size()) &&
                    std::equal(it->tokens.begin(), it->tokens.end(), tokens.begin()))
                    best = it;
        // No request may inherit the rolling state of a different conversation.
        engine.backend->reset_sequence();
        engine.qwen4exp = {};
        engine.t_gen = 0;
        if (best == cache.end()) return 0;
        if (!engine.backend->restore_sequence_state(*best->recurrent))
            throw std::runtime_error("CPU sequence-state restore failed");
        const size_t row = static_cast<size_t>(engine.kv_cache.n_kv_heads) * engine.kv_cache.head_dim;
        const size_t count = best->tokens.size() * row;
        size_t offset = 0;
        for (int l : best->layers) {
            std::copy_n(best->k.data() + offset, count, engine.kv_cache.k_ptr(l, 0));
            std::copy_n(best->v.data() + offset, count, engine.kv_cache.v_ptr(l, 0));
            offset += count;
        }
        logits = best->logits;
        engine.t_gen = best->t_gen;
        const int n = static_cast<int>(best->tokens.size());
        cache.splice(cache.begin(), cache, best);
        return n;
    }
    void save(const std::vector<int> &tokens, const std::vector<float> &logits) {
        if (!budget || tokens.empty()) return;
        std::vector<int> layers;
        for (int l = 0; l < engine.model_file.architecture_config.n_layers; ++l)
            if (engine.backend->layer_type(l) == CPULayerType::DenseQKV) layers.push_back(l);
        const size_t row = static_cast<size_t>(engine.kv_cache.n_kv_heads) * engine.kv_cache.head_dim;
        const size_t count = tokens.size() * row * layers.size();
        const size_t bytes = engine.backend->sequence_state_bytes() + (2 * count + logits.size()) * sizeof(float) +
                             tokens.size() * sizeof(int) + layers.size() * sizeof(int) + sizeof(Snapshot);
        if (bytes > budget) {
            std::fprintf(stderr, "[API-CACHE] checkpoint skipped: %zu bytes exceeds %zu-byte budget\n", bytes, budget);
            return;
        }
        for (auto it = cache.begin(); it != cache.end();) {
            if (it->tokens == tokens) { used -= it->bytes; it = cache.erase(it); }
            else ++it;
        }
        while (!cache.empty() && (used + bytes > budget || cache.size() >= 16)) {
            used -= cache.back().bytes;
            cache.pop_back();
        }
        Snapshot s;
        s.bytes = bytes;
        s.tokens = tokens;
        s.layers = std::move(layers);
        s.logits = logits;
        s.t_gen = engine.t_gen;
        s.recurrent = engine.backend->save_sequence_state();
        if (!s.recurrent) throw std::runtime_error("CPU sequence-state save failed");
        s.k.resize(count);
        s.v.resize(count);
        size_t offset = 0;
        const size_t per_layer = tokens.size() * row;
        for (int l : s.layers) {
            std::copy_n(engine.kv_cache.k_ptr(l, 0), per_layer, s.k.data() + offset);
            std::copy_n(engine.kv_cache.v_ptr(l, 0), per_layer, s.v.data() + offset);
            offset += per_layer;
        }
        cache.push_front(std::move(s));
        used += bytes;
    }
    std::vector<float> prefill(const Prepared &p, bool reuse, int &cached, const std::atomic_bool &cancel) {
        if (p.tokens.empty() || p.tokens.size() > static_cast<size_t>(engine.max_t))
            throw ApiError(400, "prompt outside CPU context limits");
        std::vector<float> logits;
        cached = restore(p.tokens, logits, reuse);
        for (size_t pos = cached; pos < p.tokens.size(); ++pos) {
            logits = forward(p.tokens[pos], static_cast<int>(pos), cancel);
            const size_t end = pos + 1;
            if (reuse && (end == p.tokens.size() ||
                std::find(p.checkpoints.begin(), p.checkpoints.end(), end) != p.checkpoints.end()))
                save(std::vector<int>(p.tokens.begin(), p.tokens.begin() + end), logits);
        }
        return logits;
    }
};

CPUBackend::CPUBackend(CPUEngine &engine, size_t bytes, std::function<bool()> cancelled)
    : impl_(std::make_unique<Impl>(engine, bytes, std::move(cancelled))) {}
CPUBackend::~CPUBackend() = default;
void CPUBackend::clear_cache() { impl_->cache.clear(); impl_->used = 0; }
size_t CPUBackend::cache_bytes() const { return impl_->used; }
size_t CPUBackend::cache_entries() const { return impl_->cache.size(); }
Prepared CPUBackend::prepare(const Request &r) {
    auto &engine = impl_->engine;
    Prepared p;
    for (const auto &s : render_chat(r, TemplateKind::Qwen, engine.config.enable_thinking,
                                    engine.config.system_prompt, engine.config.reasoning_effort)) {
        if (s.marker == Marker::ImStart) p.tokens.push_back(engine.tokenizer.im_start_id);
        else if (s.marker == Marker::ImEnd) {
            p.tokens.push_back(engine.tokenizer.im_end_id);
            p.checkpoints.push_back(p.tokens.size());
        } else if (s.marker == Marker::ThinkStart && engine.tokenizer.ds_think_start_id >= 0)
            p.tokens.push_back(engine.tokenizer.ds_think_start_id);
        else if (s.marker == Marker::ThinkEnd && engine.tokenizer.ds_think_end_id >= 0)
            p.tokens.push_back(engine.tokenizer.ds_think_end_id);
        else {
            const auto tokens = engine.tokenizer.encode(s.marker == Marker::ThinkStart ? "<think>" :
                s.marker == Marker::ThinkEnd ? "</think>" : s.text);
            p.tokens.insert(p.tokens.end(), tokens.begin(), tokens.end());
        }
    }
    return p;
}
std::vector<float> CPUBackend::evaluate(const std::vector<int> &tokens, bool reuse, int &cached) {
    const std::atomic_bool cancel{false};
    return impl_->prefill({tokens, {}}, reuse, cached, cancel);
}
Result CPUBackend::generate(const Request &r, const Prepared &p, const std::atomic_bool &cancel,
                            const std::function<bool(const std::string &)> &emit) {
    int cached = 0;
    auto logits = impl_->prefill(p, true, cached, cancel);
    Result result;
    result.usage = {static_cast<int>(p.tokens.size()), cached, 0};
    CPUSampler sampler;
    std::vector<int> sequence = p.tokens;
    auto finish = [&] {
        if (impl_->budget && !cancel) impl_->save(sequence, logits);
        return result;
    };
    for (int i = 0; i < r.max_tokens; ++i) {
        impl_->check_cancel(cancel);
        const int token = sampler.sample(logits, r.temperature, r.top_p);
        ++result.usage.output_tokens;
        if (token == impl_->engine.tokenizer.eos_token_id || impl_->engine.tokenizer.is_stop_token(token))
            return finish();
        if (!emit(impl_->engine.tokenizer.decode(token))) return finish();
        // Checkpoints cover only tokens actually forwarded, not a final sampled
        // token whose recurrent/KV update has not happened.
        if (i + 1 == r.max_tokens) { result.finish = Finish::Length; break; }
        sequence.push_back(token);
        logits = impl_->forward(token, static_cast<int>(sequence.size() - 1), cancel);
    }
    return finish();
}
}
