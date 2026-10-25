// LayerPlan: the same blocks under two residual structures, run on ggml's CPU backend.
//  * PreNormResidualPlan: cur += attn(cur); cur += ffn(cur)
//  * a hyper-connection style plan (several residual streams [D, HC, S]): the blocks see one mixed row and
//    their outputs are added back to every stream. It stands in for DeepSeek-V4.1's structure to show the
//    block interfaces do not have to change; it is not that model's math.
#include "graph_route_survey.hpp"
#include "layer_plan.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <vector>

using namespace zeromoe::graph;
static int failures = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); ++failures; } } while (0)

namespace {
struct ScaleAttn final : AttentionBlock {
  GraphCore &g; float k; ScaleAttn(GraphCore &g_, float k_) : g(g_), k(k_) {}
  bool declare_inputs() override { return true; }
  void set_inputs(int) override {}
  ggml_tensor *build(ggml_tensor *x, int) override { return ggml_scale(g.ctx, x, k); }
};
struct ScaleFfn final : FfnBlock {
  GraphCore &g; float k; ScaleFfn(GraphCore &g_, float k_) : g(g_), k(k_) {}
  ggml_tensor *build(ggml_tensor *cur, int) override { return ggml_scale(g.ctx, cur, k); }
};
// streams [D, HC, S] -> mean over HC -> blocks -> added to each stream.
struct MeanStreamPlan final : LayerPlan {
  GraphCore &g; AttentionBlock &a; FfnBlock &f;
  MeanStreamPlan(GraphCore &g_, AttentionBlock &a_, FfnBlock &f_) : g(g_), a(a_), f(f_) {}
  ggml_tensor *mix(ggml_tensor *streams) {
    // [D, HC, S] -> [D, S]: permute HC to the outer dim, sum rows, scale by 1/HC
    ggml_tensor *p = ggml_cont(g.ctx, ggml_permute(g.ctx, streams, 1, 0, 2, 3)); // [HC, D, S]
    ggml_tensor *sum = ggml_sum_rows(g.ctx, p);                                  // [1, D, S]
    return ggml_scale(g.ctx, ggml_reshape_2d(g.ctx, sum, streams->ne[0], streams->ne[2]), 1.0f / float(streams->ne[1]));
  }
  ggml_tensor *build_layer(ggml_tensor *streams, int layer) override {
    ggml_tensor *x = mix(streams);
    ggml_tensor *ao = a.build(x, layer);
    streams = ggml_add(g.ctx, streams, ggml_repeat(g.ctx, ggml_reshape_3d(g.ctx, ao, ao->ne[0], 1, ao->ne[1]), streams));
    ggml_tensor *fo = f.build(mix(streams), layer);
    return ggml_add(g.ctx, streams, ggml_repeat(g.ctx, ggml_reshape_3d(g.ctx, fo, fo->ne[0], 1, fo->ne[1]), streams));
  }
};
} // namespace

int main() {
  GraphCore g;
  g.S = 2; g.D = 3;
  ggml_init_params ip{64ull << 20, nullptr, true};
  g.ctx = ggml_init(ip);
  g.graph = ggml_new_graph(g.ctx);
  ggml_backend_t cpu = ggml_backend_cpu_init();

  ScaleAttn attn(g, 0.5f);
  ScaleFfn ffn(g, 2.0f);
  // 1. pre-norm residual, 2 layers: per layer cur -> 1.5 cur -> 1.5 cur + 2 (1.5 cur) = 4.5 cur; two layers: 20.25
  ggml_tensor *in = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, 3, 2);
  PreNormResidualPlan plan(g, attn, [&](int) -> FfnBlock & { return ffn; });
  ggml_tensor *out = build_layer_stack(in, plan, 2);
  // 2. the stream plan: 2 streams, 1 layer. streams s0=1, s1=3 -> mean 2 -> attn 1 -> streams 2, 4 -> mean 3 -> ffn 6 -> 8, 10
  ggml_tensor *st = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, 3, 2, 2);
  MeanStreamPlan hc(g, attn, ffn);
  ggml_tensor *hout = build_layer_stack(st, hc, 1);
  ggml_build_forward_expand(g.graph, out);
  ggml_build_forward_expand(g.graph, hout);
  ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(g.ctx, cpu);
  std::vector<float> a(6, 1.0f), s(12);
  for (int t = 0; t < 2; ++t) for (int h = 0; h < 2; ++h) for (int d = 0; d < 3; ++d) s[size_t(t * 6 + h * 3 + d)] = h == 0 ? 1.0f : 3.0f;
  ggml_backend_tensor_set(in, a.data(), 0, a.size() * 4);
  ggml_backend_tensor_set(st, s.data(), 0, s.size() * 4);
  ggml_backend_graph_compute(cpu, g.graph);
  std::vector<float> o(6), h(12);
  ggml_backend_tensor_get(out, o.data(), 0, o.size() * 4);
  ggml_backend_tensor_get(hout, h.data(), 0, h.size() * 4);
  for (float v : o) CHECK(std::fabs(v - 20.25f) < 1e-4f);
  for (int t = 0; t < 2; ++t) for (int hh = 0; hh < 2; ++hh) for (int d = 0; d < 3; ++d)
    CHECK(std::fabs(h[size_t(t * 6 + hh * 3 + d)] - (hh == 0 ? 8.0f : 10.0f)) < 1e-5f);
  // 3. State as data: a KDA-style history + matrix state and a DSA-style indexer cache, zeroed, usable in a graph.
  {
    ggml_context *mc = ggml_init({4ull << 20, nullptr, true});
    std::vector<StateSpec> specs = {{"hist", GGML_TYPE_F32, {3, 8, 1}, 0}, {"state", GGML_TYPE_F32, {4, 4, 2, 1}, 0},
                                    {"idx", GGML_TYPE_F16, {16, 32}, 3}};
    StateSet ss;
    CHECK(allocate_state_set(mc, 1 << 20, cpu, specs, &ss));
    CHECK(ss.tensors.size() == 3 && ss.buffer);
    CHECK(ss.tensors[1]->ne[0] == 4 && ss.tensors[1]->ne[2] == 2 && ss.tensors[2]->type == GGML_TYPE_F16);
    std::vector<float> z(32, -1.0f);
    ggml_backend_tensor_get(ss.tensors[1], z.data(), 0, z.size() * 4);
    for (float v : z) CHECK(v == 0.0f);
    float one[3] = {1, 2, 3};
    ggml_backend_tensor_set(ss.tensors[0], one, 0, sizeof one);
    float back[3] = {};
    ggml_backend_tensor_get(ss.tensors[0], back, 0, sizeof back);
    CHECK(back[2] == 3.0f);
    ggml_backend_buffer_free(ss.buffer);
    ggml_free(mc);
  }
  // 4. Route survey: a side tap counts decode and prefill routes separately and leaves the ids' own consumer alone.
  {
    const std::string path = "/tmp/claude-1000/route_survey_test.txt";
    std::remove(path.c_str()); std::remove((path + ".prefill").c_str());
    auto &rs = GraphRouteSurvey::get();
    CHECK(!rs.enabled());
    rs.configure(path, 3, 1, 8);                       // layers 1..2 of 3, 8 experts
    CHECK(rs.enabled());
    ggml_context *c2 = ggml_init({8ull << 20, nullptr, true});
    ggml_cgraph *g2 = ggml_new_graph(c2);
    ggml_backend_sched_t sched = ggml_backend_sched_new(&cpu, nullptr, 1, 256, false, false);
    ggml_tensor *ids = ggml_new_tensor_2d(c2, GGML_TYPE_I32, 2, 3);        // topk 2, 3 tokens
    ggml_tensor *fids = ggml_new_tensor_2d(c2, GGML_TYPE_F32, 2, 3);
    ggml_tensor *user = ggml_scale(c2, ggml_cast(c2, ids, GGML_TYPE_F32), 2.0f);   // the real consumer of the ids
    ggml_tensor *bad = ggml_scale(c2, fids, 1.0f);
    CHECK(rs.tap(c2, g2, sched, cpu, ids, 1, true));   // decode table, layer 1
    CHECK(rs.tap(c2, g2, sched, cpu, ids, 2, false));  // prefill table, layer 2
    CHECK(!rs.tap(c2, g2, sched, cpu, fids, 1, true)); // not I32: refused
    ggml_build_forward_expand(g2, user);
    (void)bad;
    CHECK(ggml_backend_sched_alloc_graph(sched, g2));
    int32_t v[6] = {3, 5, 3, -1, 7, 99};               // -1 and 99 are ignored
    ggml_backend_tensor_set(ids, v, 0, sizeof v);
    CHECK(ggml_backend_sched_graph_compute(sched, g2) == GGML_STATUS_SUCCESS);
    float out[6]; ggml_backend_tensor_get(user, out, 0, sizeof out);
    CHECK(out[0] == 6.0f && out[3] == -2.0f && out[5] == 198.0f);   // the real path computed on the unchanged ids
    rs.flush();
    auto slurp = [](const std::string &f) { std::ifstream in(f); std::stringstream ss; ss << in.rdbuf(); return ss.str(); };
    CHECK(slurp(path) == "1 3 2\n1 5 1\n1 7 1\n");           // decode: layer 1 only
    CHECK(slurp(path + ".prefill") == "2 3 2\n2 5 1\n2 7 1\n");  // prefill: layer 2 only
    rs.release_sites();
    ggml_backend_sched_free(sched);
    ggml_free(c2);
  }
  if (failures) return 1;
  std::printf("layer_plan_test: OK\n");
  ggml_backend_buffer_free(buf); ggml_backend_free(cpu); ggml_free(g.ctx);
  return 0;
}
