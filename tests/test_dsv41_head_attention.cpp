// Accuracy of the GPU DeepSeek-V4.1 single-token head attention kernel against
// a float64 reference of the same math (scores over window + selected
// compressed keys, softmax including the sink logit, weighted sum with K == V),
// next to a float32 host computation with the CPU path's dot (AVX-512) and a
// serial weighted sum. The GPU error must stay within 2x of the float32 host
// error plus a small floor.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

extern float dot_f32_avx512(const float*, const float*, int);
extern "C" void launch_dsv41_head_attention(
    const float* q, const float* win, const float* comp, const int* sel, int nsel,
    const float* sinks, int pos, int past, int window, int qh, int H, float scale,
    float* out, cudaStream_t stream);

static void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }

int main() {
  const int H = 64, qh = 512, window = 128, comp_blocks = 600;
  int fails = 0;
  for (int nsel : {0, 64, 512}) {
    for (int pos : {200, 1500}) {
      const int past = std::min(pos + 1, window);
      std::mt19937 rng(11 + nsel + pos);
      std::normal_distribution<float> nd(0.f, 1.f);
      std::vector<float> q((size_t)H * qh), win((size_t)window * qh), comp((size_t)comp_blocks * qh), sinks(H);
      for (auto& v : q) v = nd(rng) * 0.2f;
      for (auto& v : win) v = nd(rng);
      for (auto& v : comp) v = nd(rng);
      for (auto& v : sinks) v = nd(rng);
      std::vector<int> sel(std::max(nsel, 1));
      for (int i = 0; i < nsel; ++i) sel[i] = (i * 7 + 3) % comp_blocks;
      const float scale = 1.0f / std::sqrt((float)qh);
      const int n = past + nsel;
      auto key_of = [&](int i) -> const float* {
        if (i < past) return win.data() + (size_t)((pos - past + 1 + i) % window) * qh;
        return comp.data() + (size_t)sel[i - past] * qh;
      };
      std::vector<double> ref((size_t)H * qh);
      std::vector<float> host32((size_t)H * qh);
      for (int h = 0; h < H; ++h) {
        const float* hq = q.data() + (size_t)h * qh;
        std::vector<double> sc(n + 1); std::vector<float> sc32(n + 1);
        for (int i = 0; i < n; ++i) {
          long double a = 0; const float* k = key_of(i);
          for (int d = 0; d < qh; ++d) a += (long double)hq[d] * k[d];
          sc[i] = (double)a * scale;
          sc32[i] = dot_f32_avx512(hq, k, qh) * scale;
        }
        sc[n] = sinks[h]; sc32[n] = sinks[h];
        double m = *std::max_element(sc.begin(), sc.end()); double den = 0;
        for (auto& v : sc) { v = std::exp(v - m); den += v; }
        float m32 = *std::max_element(sc32.begin(), sc32.end()); float den32 = 0;
        for (auto& v : sc32) { v = std::exp(v - m32); den32 += v; }
        for (int d = 0; d < qh; ++d) {
          long double a = 0; float a32 = 0;
          for (int i = 0; i < n; ++i) { a += (long double)(sc[i] / den) * key_of(i)[d]; a32 += (sc32[i] / den32) * key_of(i)[d]; }
          ref[(size_t)h * qh + d] = (double)a; host32[(size_t)h * qh + d] = a32;
        }
      }
      float *dq, *dwin, *dcomp, *dsink, *dout; int* dsel;
      check(cudaMalloc(&dq, q.size() * 4)); check(cudaMalloc(&dwin, win.size() * 4));
      check(cudaMalloc(&dcomp, comp.size() * 4)); check(cudaMalloc(&dsink, H * 4));
      check(cudaMalloc(&dout, q.size() * 4)); check(cudaMalloc(&dsel, sel.size() * 4));
      check(cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
      check(cudaMemcpy(dwin, win.data(), win.size() * 4, cudaMemcpyHostToDevice));
      check(cudaMemcpy(dcomp, comp.data(), comp.size() * 4, cudaMemcpyHostToDevice));
      check(cudaMemcpy(dsink, sinks.data(), H * 4, cudaMemcpyHostToDevice));
      check(cudaMemcpy(dsel, sel.data(), sel.size() * 4, cudaMemcpyHostToDevice));
      launch_dsv41_head_attention(dq, dwin, dcomp, dsel, nsel, dsink, pos, past, window, qh, H, scale, dout, nullptr);
      check(cudaDeviceSynchronize());
      {  // kernel timing (median of 200 launches)
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        std::vector<float> ts;
        for (int it = 0; it < 220; ++it) {
          cudaEventRecord(e0);
          launch_dsv41_head_attention(dq, dwin, dcomp, dsel, nsel, dsink, pos, past, window, qh, H, scale, dout, nullptr);
          cudaEventRecord(e1); cudaEventSynchronize(e1);
          float ms = 0; cudaEventElapsedTime(&ms, e0, e1); if (it >= 20) ts.push_back(ms * 1000.f);
        }
        std::sort(ts.begin(), ts.end());
        std::printf("  kernel median %.1f us (n=%d keys)\n", ts[ts.size() / 2], n);
      }
      std::vector<float> gpu(q.size());
      check(cudaMemcpy(gpu.data(), dout, gpu.size() * 4, cudaMemcpyDeviceToHost));
      {  // bitwise comparison with the output saved from the previous kernel version
        char fn[256]; std::snprintf(fn, sizeof fn, "/tmp/claude-1000/ha_ref_%d_%d.bin", nsel, pos);
        FILE* f = std::fopen(fn, "rb");
        if (!f) { f = std::fopen(fn, "wb"); std::fwrite(gpu.data(), 4, gpu.size(), f); std::printf("  saved reference output %s\n", fn); }
        else { std::vector<float> old(gpu.size()); size_t r = std::fread(old.data(), 4, old.size(), f);
               int diff = 0; for (size_t i = 0; i < gpu.size(); ++i) diff += std::memcmp(&old[i], &gpu[i], 4) != 0;
               std::printf("  vs saved output: %s (%d differing values)\n", (r == old.size() && !diff) ? "bit-identical" : "DIFFERENT", diff); }
        std::fclose(f);
      }
      double eg = 0, eh = 0, mx = 0;
      for (size_t i = 0; i < gpu.size(); ++i) {
        eg = std::max(eg, std::fabs(gpu[i] - ref[i])); eh = std::max(eh, std::fabs(host32[i] - ref[i]));
        mx = std::max(mx, std::fabs(ref[i]));
      }
      const bool ok = eg <= 2.0 * eh + 1e-6 * std::max(mx, 1.0);
      std::printf("nsel=%d pos=%d past=%d  max|gpu-f64|=%.3e  max|host_f32-f64|=%.3e  out_max=%.3e  %s\n",
                  nsel, pos, past, eg, eh, mx, ok ? "ok" : "WORSE THAN HOST");
      fails += !ok;
      cudaFree(dq); cudaFree(dwin); cudaFree(dcomp); cudaFree(dsink); cudaFree(dout); cudaFree(dsel);
    }
  }
  std::printf(fails ? "FAIL\n" : "PASS\n");
  return fails ? 1 : 0;
}
