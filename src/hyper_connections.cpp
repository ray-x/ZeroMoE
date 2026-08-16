// hyper_connections.cpp — engine-facing DeepSeek-V4 Hyper-Connections glue:
// hc_pre_collapse/hc_post_combine (per-layer, per-sub-block FamilyOps
// hooks) and hc_expand_embedding/hc_collapse_output (model-level, called
// from generation_loop.hpp around the whole forward pass). The actual
// Sinkhorn-normalization math lives in include/hyper_connections.hpp
// (pure, unit-tested); this file is the part that touches StreamingEngine,
// resident tensors, and dense_proj/GPU dispatch.
//
// UNVERIFIED ON REAL HARDWARE -- see hyper_connections.hpp's own header
// comment for the honesty caveat on how confident this reconstruction of
// the real DeepSeek-V4 Hyper-Connections algorithm is (best-effort from
// reading llama.cpp's deepseek4.cpp, not a byte-exact transcription).
//
// Key design property (family_ops.hpp's own comment): the multi-stream
// shape NEVER leaves this file + engine.hc_stream_buf. Every other
// function in the forward pass (dsv4_attention_gguf, dense_ffn_gguf,
// moe_forward_gguf, LayerScratch) keeps reading/writing plain [S,D]
// buffers regardless of family -- only the two per-layer hooks here
// (hc_pre_collapse/hc_post_combine) and the two model-level functions
// below ever touch hc_stream_buf.
#include "engine.hpp"
#include "dense_dispatch.hpp"
#include "hyper_connections.hpp"
#include "tensor_trace.hpp"
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace {

// FIX: this file had ZERO finite-checks anywhere (confirmed via a real
// audit) despite being brand-new, UNVERIFIED-ON-REAL-HARDWARE code (see
// this file's own header comment) sitting directly in the residual-stream
// path for every DeepSeek-V4 layer -- a NaN introduced here was only ever
// catchable (if at all) by the generic post-residual check two frames
// later in execution_layer.cpp, which reads the COLLAPSED single-stream
// view and would miss anything finite-but-wrong (e.g. a bad sinkhorn
// normalization) entirely. Mirrors execution_layer.cpp's check_finite /
// attention_layer.cpp's check_finite_attn -- same per-translation-unit
// local-helper pattern already established in both of those files rather
// than a shared header (this project's own convention).
// row_width mirrors attention_layer.cpp's check_finite_attn: when >0 and
// VERBOSE logging is enabled, also emits the L2 norm of the LAST token's
// row (n - row_width .. n) so real magnitude/scale data is visible even
// when nothing is NaN/Inf -- this file had NO such visibility before
// 2026-08-09 (only the NaN/Inf branch ever logged anything), which is
// exactly why a real garbled-output run showed zero [HC] lines: silence
// here means "no NaN/Inf", not "HC didn't run" or "HC is producing sane
// values" -- those are two different, previously indistinguishable claims.
static bool check_finite_hc(const char* label, int layer_idx, const float* data, size_t n,
                            int row_width = 0) {
    int nan_count = 0, inf_count = 0;
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(data[i])) ++nan_count;
        else if (std::isinf(data[i])) ++inf_count;
    }
    if (nan_count > 0 || inf_count > 0) {
        LOG_ERROR("[HC] %s layer=%d: %d/%zu NaN, %d/%zu Inf.\n", label, layer_idx, nan_count, n, inf_count, n);
        trace_abort_if_nan("HC", true);
        return false;
    }
    if (row_width > 0 && trace_gate("HC")) {
        trace_row("HC", layer_idx, label, data, n, row_width);
    }
    return true;
}

// Shared lookup for a small resident bias/scale vector (NOT a matmul
// weight -- read directly, not via dense_proj). Returns nullptr (caller
// treats as "all zero"/"all one" per its own default) if the tensor isn't
// resident, mirroring execution_layer.cpp's own exp_probs_b.bias lookup
// pattern (the established "optional small tensor" idiom in this
// codebase).
const float *find_resident_float(StreamingEngine &engine,
                                  const std::string &key) {
  auto it = engine.resident_layers.find(key);
  if (it == engine.resident_layers.end() || !it->second.host_ptr)
    return nullptr;
  return reinterpret_cast<float *>(it->second.host_ptr);
}

// Flatten engine.hc_stream_buf (layout: token-major, then stream, then
// dim -- (s*HC+h)*D+d, "stream-major with dim innermost per stream",
// matching ggml's conventional fastest-varying-dim-first tensor flatten
// order for the hc_{attn,ffn,head}_fn weight's expected input layout; see
// hyper_connections.hpp's own header comment on this being an unverified
// interpretation) into a plain-RMS-normalized (no learned gain -- no
// dedicated hc-norm weight tensor found in the researched tensor list)
// [S, HC*D] buffer, ready for dense_proj.
void hc_flatten_and_norm(StreamingEngine &engine, int S, int HC, int D,
                         std::vector<float> &out_normed) {
  int width = HC * D;
  out_normed.resize((size_t)S * width);
  for (int s = 0; s < S; ++s) {
    const float *row = engine.hc_stream_buf.data() + (size_t)s * width;
    float *out = out_normed.data() + (size_t)s * width;
    double ss = 0.0;
    for (int i = 0; i < width; ++i) ss += (double)row[i] * row[i];
    float rms = (float)std::sqrt(ss / (double)width + 1e-6);
    float inv = 1.0f / rms;
    for (int i = 0; i < width; ++i) out[i] = row[i] * inv;
  }
}

} // namespace

// --- Per-layer, per-sub-block hooks (FamilyOps::hc_pre_fn/hc_post_fn) ---

void hc_pre_collapse(StreamingEngine &engine, int layer_idx, int sub_block,
                     float *hidden_states, int S, int D) {
  auto &c = engine.model_file.architecture_config;
  int HC = c.hc_mult;
  int mix_dim = (2 + HC) * HC;
  std::string p = "blk." + std::to_string(layer_idx) + ".";
  std::string prefix = (sub_block == 0) ? "hc_attn_" : "hc_ffn_";

  engine.hc_post_gate_scratch.resize((size_t)S * HC);
  engine.hc_comb_scratch.resize((size_t)S * HC * HC);

  std::vector<float> normed;
  hc_flatten_and_norm(engine, S, HC, D, normed);

  std::vector<float> mix((size_t)S * mix_dim);
  dense_proj(engine, p + prefix + "fn.weight", normed.data(), mix.data(), S,
             HC * D, mix_dim);

  const float *base = find_resident_float(engine, p + prefix + "base.weight");
  const float *scale = find_resident_float(engine, p + prefix + "scale.weight");
  // scale is {3}: [0]=pre-gate group, [1]=post-gate group, [2]=comb group.
  // Missing tensor (shouldn't happen for a real DeepSeek-V4 checkpoint)
  // falls back to 1.0 for every group -- a strict no-op on the raw matmul
  // output, not a silent zero.
  float scale_pre = scale ? scale[0] : 1.0f;
  float scale_post = scale ? scale[1] : 1.0f;
  float scale_comb = scale ? scale[2] : 1.0f;

  for (int s = 0; s < S; ++s) {
    const float *row = mix.data() + (size_t)s * mix_dim;
    float *pg = engine.hc_post_gate_scratch.data() + (size_t)s * HC;
    float *cb = engine.hc_comb_scratch.data() + (size_t)s * HC * HC;

    // pre-gate: mix[0..HC), sigmoid, +eps. The +eps AFTER sigmoid is
    // confirmed against deepseek4.cpp's build_hc_pre (ggml_scale_bias(pre,
    // 1.0f, hc_eps) right after ggml_sigmoid) -- found missing here
    // 2026-08-08 during the post-[SILU-TRACE-GPU]-fix garbled-output
    // re-investigation; post-gate does NOT get this (confirmed, no
    // ggml_scale_bias call on that branch in the real source).
    std::vector<float> pre_gate(HC);
    for (int h = 0; h < HC; ++h) {
      float b = base ? base[h] : 0.0f;
      pre_gate[h] = hc_sigmoid(row[h] * scale_pre + b) + c.hc_eps;
    }
    // post-gate: mix[HC..2HC), sigmoid*2.
    for (int h = 0; h < HC; ++h) {
      float b = base ? base[HC + h] : 0.0f;
      pg[h] = 2.0f * hc_sigmoid(row[HC + h] * scale_post + b);
    }
    // comb: mix[2HC..2HC+HC*HC), raw logits -- sinkhorn_normalize_comb
    // applies its own softmax as step 1, so no activation here.
    for (int i = 0; i < HC * HC; ++i) {
      float b = base ? base[2 * HC + i] : 0.0f;
      cb[i] = row[2 * HC + i] * scale_comb + b;
    }
    sinkhorn_normalize_comb(cb, HC, c.hc_sinkhorn_iters, c.hc_eps);

    // Real telemetry for the LAST token only (added 2026-08-09 -- this file
    // previously had zero visibility into pre_gate/post_gate/comb beyond
    // "not NaN"). pre_gate/post_gate are expected roughly in (eps,1+eps)/
    // (0,2); a doubly-stochastic comb should show every row AND column
    // summing to ~1 regardless of hc_sinkhorn_iters -- if either sum is
    // wildly off from 1 here, the Sinkhorn normalization (or the
    // dst/src indexing feeding it) is still wrong despite passing the
    // pure-math unit test, since the unit test can't catch a
    // caller-side transposition of a matrix that's ALREADY symmetric-ish.
    if (s == S - 1 && trace_gate("HC")) {
      auto &&os = LOG_STREAM_VERBOSE();
      os << "[HC-TRACE l=" << layer_idx << "] " << prefix << "pre_gate=[";
      for (int h = 0; h < HC; ++h) os << (h ? "," : "") << pre_gate[h];
      os << "] post_gate=[";
      for (int h = 0; h < HC; ++h) os << (h ? "," : "") << pg[h];
      os << "] comb_row_sums=[";
      for (int dst = 0; dst < HC; ++dst) {
        float rs = 0.0f;
        for (int src = 0; src < HC; ++src) rs += cb[dst * HC + src];
        os << (dst ? "," : "") << rs;
      }
      os << "] comb_col_sums=[";
      for (int src = 0; src < HC; ++src) {
        float cs = 0.0f;
        for (int dst = 0; dst < HC; ++dst) cs += cb[dst * HC + src];
        os << (src ? "," : "") << cs;
      }
      os << "]\n";
    }

    // Collapse: hidden_states[s][d] = sum_h pre_gate[h] * stream[s][h][d].
    float *out = hidden_states + (size_t)s * D;
    const float *streams = engine.hc_stream_buf.data() + (size_t)s * HC * D;
    std::memset(out, 0, (size_t)D * sizeof(float));
    for (int h = 0; h < HC; ++h) {
      const float *stream_h = streams + (size_t)h * D;
      float w = pre_gate[h];
      for (int d = 0; d < D; ++d) out[d] += w * stream_h[d];
    }
  }
  check_finite_hc("hidden_states (post hc_pre_collapse)", layer_idx, hidden_states, (size_t)S * D, D);
}

void hc_post_combine(StreamingEngine &engine, int layer_idx, int /*sub_block*/,
                     float *hidden_states, const float *sub_block_out, int S,
                     int D) {
  auto &c = engine.model_file.architecture_config;
  int HC = c.hc_mult;
  // new_stream[dst] = sub_block_out * post_gate[dst]
  //                 + sum_src comb[dst][src] * old_stream[src]
  // Reads engine.hc_post_gate_scratch/hc_comb_scratch, populated by the
  // hc_pre_collapse call for this SAME layer/sub_block just before the
  // sub-block ran (family_ops.hpp's hc_pre_fn/hc_post_fn bracket
  // invariant: always paired, same layer_idx/sub_block, nothing else runs
  // an hc_pre_collapse for a DIFFERENT layer/sub_block in between).
  std::vector<float> new_streams((size_t)S * HC * D);
  for (int s = 0; s < S; ++s) {
    const float *pg = engine.hc_post_gate_scratch.data() + (size_t)s * HC;
    const float *cb = engine.hc_comb_scratch.data() + (size_t)s * HC * HC;
    const float *old_streams =
        engine.hc_stream_buf.data() + (size_t)s * HC * D;
    const float *sub_out = sub_block_out + (size_t)s * D;
    float *out_streams = new_streams.data() + (size_t)s * HC * D;

    for (int dst = 0; dst < HC; ++dst) {
      float *out = out_streams + (size_t)dst * D;
      float post = pg[dst];
      for (int d = 0; d < D; ++d) out[d] = post * sub_out[d];
      for (int src = 0; src < HC; ++src) {
        float w = cb[dst * HC + src];
        const float *old = old_streams + (size_t)src * D;
        for (int d = 0; d < D; ++d) out[d] += w * old[d];
      }
    }
  }
  check_finite_hc("new_streams (post hc_post_combine, pre-move into hc_stream_buf)",
                  layer_idx, new_streams.data(), (size_t)S * HC * D, HC * D);
  engine.hc_stream_buf = std::move(new_streams);

  // hidden_states is left holding the collapsed single-stream value the
  // NEXT sub_block's own hc_pre_collapse will overwrite anyway -- but for
  // the LAST sub_block of the LAST layer, layers_forward_gguf's caller
  // (generation_loop.hpp) reads hidden_states directly before
  // hc_collapse_output runs its own, separate top-level collapse, so this
  // must still hold a well-defined value. Use stream 0 as the "primary"
  // stream, matching hc_pre_collapse's own convention of writing the
  // collapsed value into hidden_states (both are "some single-stream view
  // of the current state", consistency of WHICH view matters less than
  // "always well-defined" here since every real consumer of hidden_states
  // between sub-blocks is another hc_pre_collapse call that overwrites it
  // immediately).
  for (int s = 0; s < S; ++s) {
    std::memcpy(hidden_states + (size_t)s * D,
                engine.hc_stream_buf.data() + (size_t)s * HC * D,
                (size_t)D * sizeof(float));
  }
}

// --- Model-level hooks (generation_loop.hpp, around the whole forward pass) ---

void hc_expand_embedding(StreamingEngine &engine, const float *hidden_states,
                         int S, int D) {
  auto &c = engine.model_file.architecture_config;
  int HC = c.hc_mult;
  engine.hc_stream_buf.resize((size_t)S * HC * D);
  // Structural repeat, no weights -- every one of the HC streams starts as
  // an identical copy of the token embedding (ggml_repeat_4d in
  // deepseek4.cpp's model-input handling).
  for (int s = 0; s < S; ++s) {
    const float *emb = hidden_states + (size_t)s * D;
    for (int h = 0; h < HC; ++h) {
      std::memcpy(engine.hc_stream_buf.data() + ((size_t)s * HC + h) * D,
                  emb, (size_t)D * sizeof(float));
    }
  }
  check_finite_hc("hc_stream_buf (post hc_expand_embedding)", -1,
                  engine.hc_stream_buf.data(), (size_t)S * HC * D, HC * D);
}

void hc_collapse_output(StreamingEngine &engine, float *hidden_states, int S,
                        int D) {
  auto &c = engine.model_file.architecture_config;
  int HC = c.hc_mult;

  std::vector<float> normed;
  hc_flatten_and_norm(engine, S, HC, D, normed);

  // output_hc_fn: {HC*D, HC} -- simpler than the per-layer hc_{attn,ffn}_fn
  // (no post-gate/comb groups, this is the FINAL collapse with nothing
  // downstream to write multiple streams back into).
  std::vector<float> gate_logits((size_t)S * HC);
  dense_proj(engine, "output_hc_fn.weight", normed.data(), gate_logits.data(),
             S, HC * D, HC);

  const float *base = find_resident_float(engine, "output_hc_base.weight");
  const float *scale = find_resident_float(engine, "output_hc_scale.weight");
  float scale_v = scale ? scale[0] : 1.0f; // output_hc_scale is {1}, one scalar

  for (int s = 0; s < S; ++s) {
    const float *row = gate_logits.data() + (size_t)s * HC;
    const float *streams = engine.hc_stream_buf.data() + (size_t)s * HC * D;
    float *out = hidden_states + (size_t)s * D;
    std::memset(out, 0, (size_t)D * sizeof(float));
    for (int h = 0; h < HC; ++h) {
      float b = base ? base[h] : 0.0f;
      // +eps after sigmoid, same as hc_pre_collapse's pre-gate -- confirmed
      // against deepseek4.cpp's build_hc_head (identical ggml_scale_bias
      // call after ggml_sigmoid). Found missing here 2026-08-08 alongside
      // the same bug in hc_pre_collapse.
      float w = hc_sigmoid(row[h] * scale_v + b) + c.hc_eps;
      const float *stream_h = streams + (size_t)h * D;
      for (int d = 0; d < D; ++d) out[d] += w * stream_h[d];
    }
  }
  check_finite_hc("hidden_states (post hc_collapse_output)", -1, hidden_states, (size_t)S * D, D);
  if (trace_gate("HC")) {
    const float *row = gate_logits.data() + (size_t)(S - 1) * HC;
    auto &&os = LOG_STREAM_VERBOSE();
    os << "[HC-TRACE] output_hc gate_logits(last token)=[";
    for (int h = 0; h < HC; ++h) os << (h ? "," : "") << row[h];
    os << "]\n";
  }
}
