// Bitwise check: GPU indexer block scores == the host loop in
// dsv41_attention_impl (compiled with the project's CPU flags).
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

extern "C" bool launch_dsv41_indexer_score(const float*, const float*, const float*, int, int, int, float, float*, cudaStream_t);

int main() {
  const int heads = 64, klen = 128;
  int fails = 0;
  for (int visible : {1, 7, 64, 260, 1000}) {
    std::mt19937 rng(visible);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> iq((size_t)heads * klen), iw(heads), keys((size_t)visible * klen);
    for (auto& v : iq) v = nd(rng);
    for (auto& v : iw) v = nd(rng) * 0.1f;
    for (auto& v : keys) v = nd(rng);
    const float scale = 1.0f / std::sqrt((float)(klen * heads));
    std::vector<float> ref(visible), got(visible);
    for (int block = 0; block < visible; ++block) {  // verbatim host loop
      const float* key = keys.data() + (size_t)block * klen;
      float score = 0.0f;
      for (int h = 0; h < heads; ++h) {
        const float* q = iq.data() + (size_t)h * klen;
        float dot = 0.0f;
        for (int d = 0; d < klen; ++d) dot += q[d] * key[d];
        score += std::max(0.0f, dot) * iw[(size_t)h] * scale;
      }
      ref[block] = score;
    }
    float *dq, *dw, *dk, *dout;
    cudaMalloc(&dq, iq.size() * 4); cudaMalloc(&dw, iw.size() * 4);
    cudaMalloc(&dk, keys.size() * 4); cudaMalloc(&dout, visible * 4);
    cudaMemcpy(dq, iq.data(), iq.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dw, iw.data(), iw.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dk, keys.data(), keys.size() * 4, cudaMemcpyHostToDevice);
    if (!launch_dsv41_indexer_score(dq, dw, dk, visible, heads, klen, scale, dout, nullptr)) return 2;
    cudaMemcpy(got.data(), dout, visible * 4, cudaMemcpyDeviceToHost);
    int bad = 0;
    for (int b = 0; b < visible; ++b)
      if (std::memcmp(&ref[b], &got[b], 4) != 0) ++bad;
    std::printf("visible=%d mismatching bit patterns=%d\n", visible, bad);
    fails += bad;
    cudaFree(dq); cudaFree(dw); cudaFree(dk); cudaFree(dout);
  }
  std::printf(fails ? "FAIL\n" : "PASS (bit-identical)\n");
  return fails ? 1 : 0;
}
