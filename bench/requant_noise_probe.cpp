// Re-quantisation noise of real GLM expert weights: dequantise an expert's gate/up/down (as stored), quantise to a
// candidate type, dequantise again and report RMS error relative to the stored (dequantised) weights.
//   requant_noise_probe <gguf> <layer> <expert>
#include "ggml.h"
#include "gguf.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
int main(int argc, char **argv) {
  if (argc < 4) return 1;
  gguf_init_params ip = {true, nullptr};
  gguf_context *g = gguf_init_from_file(argv[1], ip);
  if (!g) { fprintf(stderr, "open failed\n"); return 1; }
  const int layer = atoi(argv[2]), expert = atoi(argv[3]);
  const size_t data_off = gguf_get_data_offset(g);
  FILE *f = fopen(argv[1], "rb");
  ggml_type cands[] = {GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q4_0};
  int64_t F = 0;
  for (const char *stem : {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"}) {
    std::string name = "blk." + std::to_string(layer) + "." + stem + ".weight";
    int64_t id = gguf_find_tensor(g, name.c_str());
    if (id < 0) { printf("%s missing\n", name.c_str()); continue; }
    ggml_type t = gguf_get_tensor_type(g, id);
    size_t off = data_off + gguf_get_tensor_offset(g, id);
    // shape not exposed by gguf API directly: use size / n_experts and assume 256 experts; ne0 from row size search.
    size_t tot = gguf_get_tensor_size(g, id);
    const int NE = 256;
    size_t per = tot / NE;
    const ggml_type_traits *tr = ggml_get_type_traits(t);
    // ne0 (row length): GLM hidden 4096 for gate/up, 2048 for down
    int64_t ne0 = (std::strcmp(stem, "ffn_down_exps") == 0) ? F : 4096;
    size_t rowb = ggml_row_size(t, ne0);
    int64_t rows = per / rowb;
    if (!F) F = rows;
    std::vector<uint8_t> raw(per);
    fseek(f, (long)(off + per * (size_t)expert), SEEK_SET);
    if (fread(raw.data(), 1, per, f) != per) { printf("read fail\n"); continue; }
    std::vector<float> w((size_t)(ne0 * rows));
    for (int64_t r = 0; r < rows; ++r) tr->to_float(raw.data() + r * rowb, w.data() + r * ne0, ne0);
    double ss = 0;
    for (float v : w) ss += (double)v * v;
    printf("%s type=%s rows=%lld ne0=%lld rms=%.5f\n", name.c_str(), ggml_type_name(t), (long long)rows, (long long)ne0, std::sqrt(ss / w.size()));
    for (ggml_type c : cands) {
      if (ne0 % ggml_blck_size(c)) continue;
      size_t cb = ggml_row_size(c, ne0);
      std::vector<uint8_t> q(cb * rows);
      ggml_quantize_chunk(c, w.data(), q.data(), 0, rows, ne0, nullptr);
      std::vector<float> d((size_t)ne0);
      double se = 0;
      const ggml_type_traits *ct = ggml_get_type_traits(c);
      for (int64_t r = 0; r < rows; ++r) {
        ct->to_float(q.data() + r * cb, d.data(), ne0);
        for (int64_t k = 0; k < ne0; ++k) { double e = d[k] - w[r * ne0 + k]; se += e * e; }
      }
      printf("   -> %-8s %5.2f bpw  rel RMS err vs stored = %.3f\n", ggml_type_name(c), cb * 8.0 / ne0, std::sqrt(se / ss));
    }
  }
}
