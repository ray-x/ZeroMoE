#include "cpu_api.hpp"
#include "cpu_engine.hpp"
#include "cpu_qwen_moe.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>

using namespace zeromoe::server;
namespace {
void check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
void tensor(CPUEngine &e, const std::string &name, int input, int output, float constant = 0) {
    CPUQuantizedTensor t;
    t.meta.type = GGUFType::F32;
    t.meta.shape = {static_cast<uint64_t>(input), static_cast<uint64_t>(output)};
    t.host_size = input * output * sizeof(float);
    t.host_ptr = static_cast<uint8_t *>(std::malloc(t.host_size));
    check(t.host_ptr != nullptr, "tensor allocation");
    for (int i = 0; i < input * output; ++i)
        reinterpret_cast<float *>(t.host_ptr)[i] = constant == 0 ? std::sin(float(i + name.size())) * 0.2f : constant;
    e.resident_layers.emplace(name, std::move(t));
}
void model(CPUEngine &e) {
    auto &c = e.model_file.architecture_config;
    c.hidden = 4; c.vocab = 16; c.n_layers = 2; c.n_heads = 1; c.n_head_kv = 1; c.head_dim = 2;
    c.n_experts = 0; c.topk = 0;
    e.max_t = 64;
    check(e.kv_cache.init(64, 2, 1, 2), "KV allocation");
    auto b = std::make_unique<CPUQwenMoEBackend>();
    b->traits.layer_types = {CPULayerType::LinearAttention, CPULayerType::DenseQKV};
    b->traits.linear_num_key_heads = 1; b->traits.linear_key_head_dim = 2;
    b->traits.linear_num_value_heads = 1; b->traits.linear_value_head_dim = 2;
    b->traits.linear_conv_kernel = 2;
    b->traits.partial_rotary_factor = 1;
    e.backend = std::move(b);
    tensor(e, "token_embd.weight", 4, 16);
    tensor(e, "output.weight", 4, 16);
    tensor(e, "blk.0.attn_qkv.weight", 4, 6);
    tensor(e, "blk.0.attn_gate.weight", 4, 2);
    tensor(e, "blk.0.ssm_alpha.weight", 4, 1);
    tensor(e, "blk.0.ssm_beta.weight", 4, 1);
    tensor(e, "blk.0.ssm_conv1d.weight", 2, 6);
    tensor(e, "blk.0.ssm_a", 1, 1, -1);
    tensor(e, "blk.0.ssm_dt.bias", 1, 1, 0.1f);
    tensor(e, "blk.0.ssm_norm.weight", 2, 1, 1);
    tensor(e, "blk.0.ssm_out.weight", 2, 4);
    tensor(e, "blk.1.attn_q.weight", 4, 4);
    tensor(e, "blk.1.attn_k.weight", 4, 2);
    tensor(e, "blk.1.attn_v.weight", 4, 2);
    tensor(e, "blk.1.attn_output.weight", 2, 4);
}
void equal(const std::vector<float> &a, const std::vector<float> &b) {
    check(a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0,
          "cached and uncached per-frame logits differ");
}
void run() {
    CPUEngine engine;
    model(engine);
    CPUBackend backend(engine, 4096);
    std::vector<int> prefix{1, 2, 3};
    int hit = -1;
    auto initial = backend.evaluate(prefix, true, hit);
    check(hit == 0 && backend.cache_entries() == 1, "initial checkpoint");
    auto again = backend.evaluate(prefix, true, hit);
    check(hit == 3, "exact-repeat hit");
    equal(initial, again);
    int ignored;
    for (int token : {4, 5, 6, 7, 8}) {
        const size_t n = prefix.size();
        prefix.push_back(token);
        auto cached = backend.evaluate(prefix, true, hit);
        check(hit == static_cast<int>(n), "extended prefix hit");
        auto reference = backend.evaluate(prefix, false, ignored);
        equal(cached, reference);
        check(backend.cache_bytes() <= 4096, "cache budget exceeded");
    }
    // An unrelated conversation overwrites the live KV and recurrent state.
    backend.evaluate({9, 10, 11, 12}, true, ignored);
    auto resumed = backend.evaluate(prefix, true, hit);
    check(hit > 0, "interleaved session lost checkpoint");
    equal(resumed, backend.evaluate(prefix, false, ignored));
    auto branch = std::vector<int>{1, 2, 3, 13};
    auto forked = backend.evaluate(branch, true, hit);
    equal(forked, backend.evaluate(branch, false, ignored));
    for (int i = 0; i < 30; ++i) {
        backend.evaluate({i % 15, (i / 15) + 1, 14}, true, ignored);
        check(backend.cache_bytes() <= 4096 && backend.cache_entries() <= 16, "eviction bounds");
    }
    backend.clear_cache();
    check(backend.cache_bytes() == 0 && backend.cache_entries() == 0, "cache clear");
    auto fresh = backend.evaluate(prefix, true, hit);
    check(hit == 0, "cache clear retained state");
    equal(fresh, backend.evaluate(prefix, false, ignored));
    CPUBackend uncached(engine, 0);
    equal(uncached.evaluate(prefix, true, hit), backend.evaluate(prefix, false, ignored));
    check(hit == 0 && uncached.cache_entries() == 0, "disabled cache created entries");
    const std::atomic_bool cancelled{true};
    bool stopped = false;
    try { uncached.generate(Request{}, Prepared{{1, 2, 3}, {}}, cancelled, [](const auto &) { return true; }); }
    catch (const ApiError &e) { stopped = e.status == 408; }
    check(stopped, "CPU cancellation was ignored");
    auto missing = engine.resident_layers.extract("blk.0.ssm_out.weight");
    bool failed = false;
    try { uncached.evaluate({1, 2, 3}, false, ignored); }
    catch (const std::runtime_error &) { failed = true; }
    check(failed, "missing model tensor returned successful logits");
    engine.resident_layers.insert(std::move(missing));
    equal(uncached.evaluate(prefix, false, ignored), fresh);
}
}
int main() {
    try { run(); std::puts("CPU API checkpoint tests passed"); return 0; }
    catch (const std::exception &e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}
