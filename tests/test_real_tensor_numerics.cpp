// Real Ling weights + real activations through CPU, CUDA and float64, to see where the CPU-vs-CUDA gap comes from.
//   real_tensor_numerics_test <ling-model-dir> <LLAMA_NODE_DUMP file from a CPU run> [layer=63]
// Inputs: norm-<L> (the MoE input) and ffn_moe_probs_masked-<L> (top-8 = the routed experts) from the dump.
// Compares the routed-expert chain gate/up -> swiglu -> down on real weights (approximately real activations, see NOTE).
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "ggml.h"
#include "gguf.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

struct Plane { ggml_type type; int64_t K, N; std::vector<uint8_t> bytes; };  // bytes: `slots` experts packed

static bool load_experts(const std::string &dir, const std::string &name, const std::vector<int> &experts, Plane &out) {
  for (auto &e : std::filesystem::directory_iterator(dir)) {
    if (e.path().extension() != ".gguf") continue;
    ggml_context *meta = nullptr;
    gguf_context *g = gguf_init_from_file(e.path().c_str(), {true, &meta});
    if (!g) continue;
    const int64_t id = gguf_find_tensor(g, name.c_str());
    if (id < 0) { gguf_free(g); ggml_free(meta); continue; }
    ggml_tensor *t = ggml_get_tensor(meta, name.c_str());
    out.type = t->type; out.K = t->ne[0]; out.N = t->ne[1];
    const size_t per = ggml_row_size(t->type, t->ne[0]) * size_t(t->ne[1]);
    const size_t base = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, id);
    out.bytes.resize(per * experts.size());
    FILE *f = std::fopen(e.path().c_str(), "rb");
    for (size_t i = 0; i < experts.size(); ++i) {
      std::fseek(f, long(base + per * size_t(experts[i])), SEEK_SET);
      if (std::fread(out.bytes.data() + per * i, 1, per, f) != per) return false;
    }
    std::fclose(f); gguf_free(g); ggml_free(meta);
    return true;
  }
  return false;
}

static std::map<std::string, std::vector<float>> read_dump(const char *path, const std::vector<std::string> &want) {
  std::map<std::string, std::vector<float>> out;
  FILE *f = std::fopen(path, "rb");
  uint32_t n;
  while (std::fread(&n, 4, 1, f) == 1) {
    std::string name(n, 0); std::fread(name.data(), 1, n, f);
    std::fread(&n, 4, 1, f); std::fseek(f, n, SEEK_CUR);
    std::fread(&n, 4, 1, f);
    if (std::find(want.begin(), want.end(), name) != want.end() && !out.count(name)) {
      out[name].resize(n); std::fread(out[name].data(), 4, n, f);
    } else std::fseek(f, long(n) * 4, SEEK_CUR);
  }
  std::fclose(f);
  return out;
}

static double rel(const std::vector<float> &a, const std::vector<float> &b) {
  double d = 0, n = 0;
  for (size_t i = 0; i < a.size(); ++i) { d += double(a[i] - b[i]) * (a[i] - b[i]); n += double(b[i]) * b[i]; }
  return std::sqrt(d / (n > 0 ? n : 1));
}

// out[N, used, T] = W[slot of ids[used,T]] . b  (b is [K,1,T] or [K,used,T])
static std::vector<float> mmid(ggml_backend_t be, const Plane &w, int64_t slots, const std::vector<float> &b, int64_t b_used,
                               const std::vector<int32_t> &ids, int64_t used, int64_t T) {
  ggml_context *ctx = ggml_init({64ull << 20, nullptr, true});
  ggml_tensor *wt = ggml_new_tensor_3d(ctx, w.type, w.K, w.N, slots);
  ggml_tensor *bt = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, w.K, b_used, T);
  ggml_tensor *it = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, used, T);
  ggml_tensor *o = ggml_mul_mat_id(ctx, wt, bt, it);
  ggml_cgraph *gr = ggml_new_graph(ctx);
  ggml_build_forward_expand(gr, o);
  ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
  ggml_backend_tensor_set(wt, w.bytes.data(), 0, w.bytes.size());
  ggml_backend_tensor_set(bt, b.data(), 0, b.size() * 4);
  ggml_backend_tensor_set(it, ids.data(), 0, ids.size() * 4);
  ggml_backend_graph_compute(be, gr);
  std::vector<float> out(size_t(w.N * used * T));
  ggml_backend_tensor_get(o, out.data(), 0, out.size() * 4);
  ggml_backend_buffer_free(buf); ggml_free(ctx);
  return out;
}

static std::vector<float> reference(const Plane &w, const std::vector<float> &b, int64_t b_used,
                                    const std::vector<int32_t> &ids, int64_t used, int64_t T) {
  const auto *tr = ggml_get_type_traits(w.type);
  const size_t rb = ggml_row_size(w.type, w.K), per = rb * size_t(w.N);
  std::vector<float> row(size_t(w.K)), out(size_t(w.N * used * T));
  for (int64_t t = 0; t < T; ++t)
    for (int64_t j = 0; j < used; ++j) {
      const float *x = b.data() + (size_t(t) * size_t(b_used) + size_t(b_used == 1 ? 0 : j)) * size_t(w.K);
      const uint8_t *base = w.bytes.data() + per * size_t(ids[size_t(t * used + j)]);
      for (int64_t n = 0; n < w.N; ++n) {
        tr->to_float(base + size_t(n) * rb, row.data(), w.K);
        double acc = 0;
        for (int64_t k = 0; k < w.K; ++k) acc += double(row[size_t(k)]) * x[k];
        out[size_t(t * used + j) * size_t(w.N) + size_t(n)] = float(acc);
      }
    }
  return out;
}

static std::vector<float> swiglu(const std::vector<float> &g, const std::vector<float> &u) {
  std::vector<float> h(g.size());
  for (size_t i = 0; i < g.size(); ++i) h[i] = float(double(g[i]) / (1.0 + std::exp(-double(g[i]))) * u[i]);
  return h;
}

// Round activations like the two backends do, then evaluate with exact math: isolates the activation-rounding error.
static std::vector<float> round_blocks(const std::vector<float> &v, int block, bool q8k) {
  std::vector<float> o(v.size());
  for (size_t b = 0; b + size_t(block) <= v.size(); b += size_t(block)) {
    float amax = 0, mx = 0;
    for (int i = 0; i < block; ++i) if (std::fabs(v[b + size_t(i)]) > amax) { amax = std::fabs(v[b + size_t(i)]); mx = v[b + size_t(i)]; }
    if (amax == 0) { for (int i = 0; i < block; ++i) o[b + size_t(i)] = 0; continue; }
    const float d = q8k ? -mx / 127.0f * -1.0f : amax / 127.0f;   // Q8_K uses the signed max: scale = |max|/127 in magnitude
    const float scale = q8k ? std::fabs(mx) / 127.0f : d;
    for (int i = 0; i < block; ++i) {
      const float q = std::max(-127.0f, std::min(127.0f, std::nearbyint(v[b + size_t(i)] / scale)));
      o[b + size_t(i)] = q * scale;
    }
  }
  return o;
}

int main(int argc, char **argv) {
  if (argc < 3) return 2;
  const int L = argc > 3 ? std::atoi(argv[3]) : 63, used = 8;
  const std::string sL = std::to_string(L);
  auto dump = read_dump(argv[2], {"norm-" + sL, "ffn_moe_probs_masked-" + sL, "ffn_moe_gate-" + sL, "ffn_moe_up-" + sL,
                                   "ffn_moe_swiglu-" + sL, "ffn_moe_down-" + sL});
  const auto &x = dump.at("norm-" + sL), &probs = dump.at("ffn_moe_probs_masked-" + sL);
  const int64_t K = 4096, E = 512, T = int64_t(x.size()) / K;
  std::vector<int32_t> ids_full(size_t(T) * used);
  std::vector<int> uniq;
  for (int64_t t = 0; t < T; ++t) {
    std::vector<int> order(E); for (int i = 0; i < E; ++i) order[size_t(i)] = i;
    std::partial_sort(order.begin(), order.begin() + used, order.end(), [&](int a, int b) { return probs[size_t(t * E + a)] > probs[size_t(t * E + b)]; });
    for (int j = 0; j < used; ++j) { ids_full[size_t(t * used + j)] = order[size_t(j)]; uniq.push_back(order[size_t(j)]); }
  }
  std::sort(uniq.begin(), uniq.end()); uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
  std::vector<int32_t> ids(ids_full.size());
  for (size_t i = 0; i < ids.size(); ++i) ids[i] = int32_t(std::lower_bound(uniq.begin(), uniq.end(), ids_full[i]) - uniq.begin());
  Plane gate, up, down;
  if (!load_experts(argv[1], "blk." + sL + ".ffn_gate_exps.weight", uniq, gate) || !load_experts(argv[1], "blk." + sL + ".ffn_up_exps.weight", uniq, up) ||
      !load_experts(argv[1], "blk." + sL + ".ffn_down_exps.weight", uniq, down)) { std::fprintf(stderr, "cannot load layer %d experts\n", L); return 1; }
  std::printf("layer %d: %lld tokens, %zu distinct experts; gate/up %s %lldx%lld, down %s %lldx%lld\n", L, (long long)T, uniq.size(),
              ggml_type_name(gate.type), (long long)gate.K, (long long)gate.N, ggml_type_name(down.type), (long long)down.K, (long long)down.N);
  ggml_backend_t cuda = ggml_backend_cuda_init(0), cpu = ggml_backend_cpu_init();
  ggml_backend_cpu_set_n_threads(cpu, std::getenv("CPU_THREADS") ? std::atoi(std::getenv("CPU_THREADS")) : 8);
  const int64_t S = int64_t(uniq.size());
  auto run = [&](ggml_backend_t be, const Plane &w, const std::vector<float> &b, int64_t bu) { return mmid(be, w, S, b, bu, ids, used, T); };

  {
    double xm = 0; size_t nonfinite = 0;
    for (float v : x) { if (!std::isfinite(v)) ++nonfinite; else xm = std::max(xm, double(std::fabs(v))); }
    std::printf("activation: max |x| %.3g, non-finite %zu of %zu\n", xm, nonfinite, x.size());
  }
  const auto g_cpu = run(cpu, gate, x, 1), g_gpu = run(cuda, gate, x, 1), g_ref = reference(gate, x, 1, ids, used, T);
  const auto u_cpu = run(cpu, up, x, 1), u_gpu = run(cuda, up, x, 1), u_ref = reference(up, x, 1, ids, used, T);
  // NOTE: not an exact replay of the graph. `norm-<L>` may lack the fused ffn_norm weight and the top-8 of
  // probs_masked need not equal the graph's routing, so the dump's own gate output is NOT reproduced (rel ~1.8). The
  // inputs are real activation statistics and the weights are real expert weights of that layer.
  {
    size_t bad = 0, first = SIZE_MAX;
    for (size_t i = 0; i < g_cpu.size(); ++i) if (!std::isfinite(g_cpu[i])) { if (first == SIZE_MAX) first = i; ++bad; }
    std::printf("CPU gate non-finite outputs: %zu (first at %zd = token %zd slot %zd row %zd)\n", bad, (ssize_t)first,
                first == SIZE_MAX ? -1 : (ssize_t)(first / (1280 * 8)), first == SIZE_MAX ? -1 : (ssize_t)((first / 1280) % 8), first == SIZE_MAX ? -1 : (ssize_t)(first % 1280));
  }
  if (!g_cpu.empty()) for (size_t i = 0; i < g_cpu.size(); ++i) if (!std::isfinite(g_cpu[i])) {
    const size_t t = i / (1280 * 8), j = (i / 1280) % 8, r = i % 1280;
    const auto *tr = ggml_get_type_traits(gate.type);
    std::vector<float> row(4096);
    const size_t rb = ggml_row_size(gate.type, 4096), per = rb * 1280;
    tr->to_float(gate.bytes.data() + per * size_t(ids[t * 8 + j]) + r * rb, row.data(), 4096);
    double mx = 0; for (float v : row) mx = std::max(mx, double(std::fabs(v)));
    double xm = 0; for (int k = 0; k < 4096; ++k) xm = std::max(xm, double(std::fabs(x[t * 4096 + size_t(k)])));
    std::printf("first bad: token %zu slot %zu row %zu expert-slot %d: cpu=%g cuda=%g f64=%g max|w| %.3g max|x| %.3g\n", t, j, r, ids[t * 8 + j], g_cpu[i], g_gpu[i], g_ref[i], mx, xm);
    break;
  }
  std::printf("%-34s %-10s %-10s %-10s\n", "op (rel L2)", "CPU/f64", "CUDA/f64", "CPU/CUDA");
  std::printf("%-34s %-10.2e %-10.2e %-10.2e\n", "gate IQ1_S (real x)", rel(g_cpu, g_ref), rel(g_gpu, g_ref), rel(g_cpu, g_gpu));
  std::printf("%-34s %-10.2e %-10.2e %-10.2e\n", "up   IQ1_S (real x)", rel(u_cpu, u_ref), rel(u_gpu, u_ref), rel(u_cpu, u_gpu));
  const auto h_ref = swiglu(g_ref, u_ref), h_cpu = swiglu(g_cpu, u_cpu), h_gpu = swiglu(g_gpu, u_gpu);
  std::printf("%-34s %-10.2e %-10.2e %-10.2e\n", "swiglu (each backend's own)", rel(h_cpu, h_ref), rel(h_gpu, h_ref), rel(h_cpu, h_gpu));
  const auto d_ref = reference(down, h_ref, used, ids, used, T);
  const auto d1_cpu = run(cpu, down, h_ref, used), d1_gpu = run(cuda, down, h_ref, used);
  std::printf("%-34s %-10.2e %-10.2e %-10.2e\n", "down IQ2_XXS, SAME exact input", rel(d1_cpu, d_ref), rel(d1_gpu, d_ref), rel(d1_cpu, d1_gpu));
  {
    const auto dk = reference(down, round_blocks(h_ref, 256, true), used, ids, used, T);
    const auto d1 = reference(down, round_blocks(h_ref, 32, false), used, ids, used, T);
    std::printf("%-34s %-10.2e (256-value blocks, Q8_K-style)\n", "down: exact math on rounded input", rel(dk, d_ref));
    std::printf("%-34s %-10.2e (32-value blocks, Q8_1-style)\n", "down: exact math on rounded input", rel(d1, d_ref));
    double mx_over_rms = 0; size_t nb = 0;
    for (size_t b = 0; b + 256 <= h_ref.size(); b += 256) {
      double amax = 0, ss = 0;
      for (int i = 0; i < 256; ++i) { amax = std::max(amax, double(std::fabs(h_ref[b + size_t(i)]))); ss += double(h_ref[b + size_t(i)]) * h_ref[b + size_t(i)]; }
      if (ss > 0) { mx_over_rms += amax / std::sqrt(ss / 256); ++nb; }
    }
    std::printf("swiglu output: mean(block max / block rms) over 256-value blocks = %.1f (Gaussian: about 3)\n", mx_over_rms / double(nb));
  }
  const auto d2_cpu = run(cpu, down, h_cpu, used), d2_gpu = run(cuda, down, h_gpu, used);
  std::printf("%-34s %-10.2e %-10.2e %-10.2e\n", "down, each backend's own chain", rel(d2_cpu, d_ref), rel(d2_gpu, d_ref), rel(d2_cpu, d2_gpu));
  std::printf("dump (llama.cpp CPU vs CPU+GPU) gate/swiglu/down for comparison: see node_dump_diff (0.93%% / 1.4%% / 2.3%%)\n");
  return 0;
}
