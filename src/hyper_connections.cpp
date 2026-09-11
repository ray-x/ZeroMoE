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
#include "ggml_hc_backend.hpp"
#include "ggml_gdn_proj_backend.hpp"
#include "ggml_moe_backend.hpp"
#include "hyper_connections.hpp"
#include "kquant_dequant.hpp" // dequant_row (find_resident_float_anytype)
#include "tensor_trace.hpp"
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// Diagnostic A/B only.  Keep the HC graph and its persistent streams active,
// but return the GDN pre-mix result through the existing host tensor_get.  This
// isolates the HC pre-mix device output from the post-combine graph.
static bool force_hc_gdn_pre_host() {
    static const bool enabled = [] {
        const char *v = std::getenv("MOESTREAM_GGML_FORCE_HC_GDN_PRE_HOST");
        return v && v[0] == '1';
    }();
    return enabled;
}

static bool dual_hc_gdn_pre_output() {
    static const bool enabled = [] {
        const char *v = std::getenv("MOESTREAM_GGML_HC_DUAL_GDN_PRE_OUTPUT");
        return v && v[0] == '1';
    }();
    return enabled;
}

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

// Type-tolerant variant for qwen4exp's HC/PLE norm weights: F32 tensors get
// the direct pointer (zero copy); anything else is dequantized ONCE into a
// static cache (they're static model weights, identical on every call --
// same pattern as attention_layer.cpp's conv-weight caches). The float-cast
// above would silently reinterpret BF16 bytes as F32 otherwise (a real risk
// here: the qwen4exp converter does not promise F32 norms the way the
// GLM/DeepSeek checkpoints' F32 norms let the DSV4 code above get away with
// it).
const float *find_resident_float_anytype(StreamingEngine &engine,
                                          const std::string &key, size_t n) {
  auto it = engine.resident_layers.find(key);
  if (it == engine.resident_layers.end() || !it->second.host_ptr)
    return nullptr;
  if (it->second.meta.type == GGUFType::F32)
    return reinterpret_cast<float *>(it->second.host_ptr);
  static std::unordered_map<std::string, std::vector<float>> cache;
  auto cit = cache.find(key);
  if (cit == cache.end()) {
    std::vector<float> buf(n);
    dequant_row(it->second.meta.type, it->second.host_ptr, buf.data(), (int)n);
    cit = cache.emplace(key, std::move(buf)).first;
  }
  return cit->second.data();
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

// ---------------------------------------------------------------------------
// Qwen4Exp (Qwen3.8-Flash-Next) low-rank hyper-connections -- a DIFFERENT HC
// variant from DeepSeek-V4's Sinkhorn-comb form above (no fn/base/scale MLP,
// no comb matrix). Reference: llama.cpp branch qwen4exp/qwen3.8-flash-next
// (src/models/qwen4exp.cpp's build_hc_mix / build_hc_combine), ported
// step-for-step:
//   mix:    per-stream RMS norm (no inner gain) -> flatten [hc*D] -> *w_norm
//           -> lo = silu(w_down(xn) / hc) -> gate = sigmoid(w_up(lo))
//           -> gated = xn * gate -> collapse = MEAN over the hc streams
//           -> inject = w_inject(xn)  [hc] per token (stashed for combine)
//   combine: stream[c] += block_out * (2 * sigmoid(inject[c] / hc))
// There is no output_norm in a qwen4exp file: the model-level collapse
// (output_hc_norm/down/up) IS the final norm -- see hc_collapse_output's
// Qwen4Exp branch below.
// ---------------------------------------------------------------------------

// Shared mix implementation for both the per-layer hooks (which also compute
// and stash the inject vector) and the model-level output mixer (which has
// no inject tensor). norm/down/up are required; inject may be null.
// out_hidden receives the mean-collapsed [S,D]; out_inject (when non-null)
// receives [S,HC].
// `in_streams`: the [S, HC, D] multi-stream state to mix, stream-major per
// token (same layout as engine.hc_stream_buf). Defaults to nullptr, meaning
// "use engine.hc_stream_buf.data()" -- the trunk's own live, shared
// residual state, exactly as before this parameter was added (both
// pre-existing call sites omit it, so their behavior is byte-for-byte
// unchanged). qwen4exp_mtp_forward (execution_layer.cpp) passes its own
// local NextN-block res_hc buffer instead -- that state must never alias
// engine.hc_stream_buf, which belongs to the target trunk's own, entirely
// separate forward pass.
// NOT static: qwen4exp_mtp_forward (execution_layer.cpp) calls this
// directly (via its own extern declaration at the call site) to run the
// shared MTP checkpoint's own hc_attn/hc_ffn/hc_head mixers against a
// LOCAL res_hc buffer, deliberately bypassing qwen4exp_hc_pre_mix/
// qwen4exp_hc_post_combine below (which are hardwired to engine.
// hc_stream_buf and the trunk's own ggml-HC/device-router bookkeeping --
// wrong state entirely for the NextN block's own independent residual).
void qwen4exp_hc_mix_impl(StreamingEngine &engine,
                                  const std::string &norm_name,
                                  const std::string &down_name,
                                  const std::string &up_name,
                                  const std::string &inject_name, int S, int D,
                                  float *out_hidden, float *out_inject,
                                  int trace_layer,
                                  const float *in_streams = nullptr) {
  auto &c = engine.model_file.architecture_config;
  const int HC = c.hc_mult;
  const int hc_dim = HC * D;
  const int hc_lr = c.hc_low_rank;
  const float *const streams_base =
      in_streams ? in_streams : engine.hc_stream_buf.data();

  // 1. grouped RMS norm: ggml_rms_norm over x [n_embd, hc, T] normalizes
  // over ne0 = n_embd PER (stream, token) with NO learned gain (the gain is
  // the separate w_norm multiply after the flatten). Engine layout of
  // hc_stream_buf is (s*HC + h)*D + d -- stream-major per token, matching
  // ggml's [n_embd, hc, T] flatten exactly.
  // Grow-only scratch instead of three fresh std::vectors per call. At
  // qwen4exp's real prefill shape (S=441, HC=4, D=2560) xn and gate are
  // ~18 MB EACH, so the old code malloc'd + value-initialised ~36 MB on
  // every one of the ~192 calls a single prefill makes -- pure overhead
  // before a single useful flop. thread_local: layers_forward_gguf is
  // single-threaded, and this keeps it safe if that ever changes.
  static thread_local std::vector<float> xn_buf, lo_buf, gate_buf;
  auto grow = [](std::vector<float> &v, size_t n) -> float * {
    if (v.size() < n) v.resize(n);
    return v.data();
  };
  float *const xn = grow(xn_buf, (size_t)S * hc_dim);

  // Every loop below is independent per token, so parallelising over s
  // leaves the arithmetic (and its summation order) BIT-IDENTICAL -- which
  // is exactly why the output can be diffed against the serial version as
  // a correctness check rather than merely eyeballed.
  // MEASURED 2026-09-05: hc_pre/hc_post together were 7.502 s of a 20.05 s
  // prefill (37%), entirely serial scalar code, and completely untimed
  // until this session added t_hc_hooks.
#pragma omp parallel for schedule(static)
  for (int s = 0; s < S; ++s) {
    const float *streams = streams_base + (size_t)s * HC * D;
    float *out = xn + (size_t)s * hc_dim;
    for (int h = 0; h < HC; ++h) {
      const float *row = streams + (size_t)h * D;
      double ss = 0.0;
      for (int d = 0; d < D; ++d) ss += (double)row[d] * row[d];
      // ggml_rms_norm: x / sqrt(mean(x^2) + eps) -- NOT the l2-norm max()
      // form the KDA path uses.
      float inv = 1.0f / std::sqrt((float)(ss / (double)D) + c.eps);
      for (int d = 0; d < D; ++d) out[(size_t)h * D + d] = row[d] * inv;
    }
  }

  // 2. learned gain over the whole [hc_dim] flattened layout (the converter
  // already folded the (1 + w) shift in -- multiply directly).
  const float *norm_w = find_resident_float_anytype(engine, norm_name,
                                                      (size_t)hc_dim);
  if (norm_w) {
    // Restructured from a flat `xn[i] *= norm_w[i % hc_dim]`: that modulo
    // ran once per element, i.e. ~4.5M integer divisions per call at this
    // shape. i == s*hc_dim + j makes i % hc_dim == j exactly, so this is
    // the same multiply in the same order, just without the division.
#pragma omp parallel for schedule(static)
    for (int s = 0; s < S; ++s) {
      float *row = xn + (size_t)s * hc_dim;
      for (int j = 0; j < hc_dim; ++j) row[j] *= norm_w[j];
    }
  } else {
    LOG_ERROR("[HC-Q4E] missing %s -- HC gate/norm silently unweighted.\n",
              norm_name.c_str());
  }

  // 3. low-rank gate: lo = silu(down(xn) / hc); gate = sigmoid(up(lo)).
  float *const lo = grow(lo_buf, (size_t)S * hc_lr);
  dense_proj(engine, down_name, xn, lo, S, hc_dim, hc_lr);
  const float inv_hc = 1.0f / (float)HC;
  const size_t lo_n = (size_t)S * hc_lr;
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < lo_n; ++i) {
    float x = lo[i] * inv_hc;
    lo[i] = x / (1.0f + std::exp(-x)); // silu
  }
  float *const gate = grow(gate_buf, (size_t)S * hc_dim);
  dense_proj(engine, up_name, lo, gate, S, hc_lr, hc_dim);
  // ~4.5M std::exp calls per call at this shape -- the single hottest loop
  // in the whole HC block, and it was serial.
  const size_t gate_n = (size_t)S * hc_dim;
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < gate_n; ++i)
    gate[i] = 1.0f / (1.0f + std::exp(-gate[i]));

  // 4. gated = xn * gate; mixed = mean over the HC streams.
#pragma omp parallel for schedule(static)
  for (int s = 0; s < S; ++s) {
    const float *xs = xn + (size_t)s * hc_dim;
    const float *gs = gate + (size_t)s * hc_dim;
    float *out = out_hidden + (size_t)s * D;
    std::memset(out, 0, (size_t)D * sizeof(float));
    for (int h = 0; h < HC; ++h)
      for (int d = 0; d < D; ++d)
        out[d] += xs[(size_t)h * D + d] * gs[(size_t)h * D + d];
    for (int d = 0; d < D; ++d) out[d] *= inv_hc;
  }

  // 5. inject = w_inject(xn) -> [S, HC], consumed by the matching
  // hc_post_combine (per-layer hooks only).
  if (out_inject) {
    dense_proj(engine, inject_name, xn, out_inject, S, hc_dim, HC);
    if (trace_gate("HC")) {
      const float *row = out_inject + (size_t)(S - 1) * HC;
      auto &&os = LOG_STREAM_VERBOSE();
      os << "[HC-TRACE l=" << trace_layer << "] inject(last token)=[";
      for (int h = 0; h < HC; ++h) os << (h ? "," : "") << row[h];
      os << "]\n";
    }
  }
}


// ---------------------------------------------------------------------------
// ggml GPU path for Qwen4Exp hyper-connections (opt-in, hc_ggml).
//
// ALL-OR-NOTHING PER FORWARD PASS, deliberately. hc_stream_buf exists in two
// places while this is on -- the host vector and the runner's device tensor --
// and the device copy is authoritative between expand and collapse. Switching
// to the CPU path for a single bracket mid-pass would read a stale host copy
// and silently corrupt the residual stream, so any failure downloads the
// device state back to the host and disables the path for the REST of this
// pass (g_hc_ggml_live), rather than falling back per call.
// ---------------------------------------------------------------------------
static bool g_hc_ggml_live = false;
static void hc_ggml_abort(StreamingEngine &engine, int S, const char *why);

// Narrow bridge used by execution_layer.cpp. Keeping the live-state check in
// this translation unit prevents the generic MoE loop from depending on HC
// implementation details, and makes the fast handoff impossible for other
// model families.
bool qwen4exp_hc_device_live() { return g_hc_ggml_live; }

// PLE is a legacy host-side layer-input hook. Before it edits the host copy,
// make that copy current; after it returns, publish the edit back to the
// device-authoritative HC state. This is normally one pair per prefill (the
// checkpoint has one PLE layer), and avoids silently dropping the PLE update
// while the GGML HC path is live.
bool qwen4exp_hc_prepare_host_layer_input(StreamingEngine &engine, int S) {
  if (!g_hc_ggml_live) return true;
  if (GgmlHcRunner::instance().download_streams(engine.hc_stream_buf.data(), S))
    return true;
  hc_ggml_abort(engine, S, "layer-input host sync failed");
  return false;
}

// Measurement-only bridge for Qwen's router pilot. Unlike the PLE bridge
// above, a failed snapshot must not abort the forward pass: the pilot is
// advisory and the production router remains authoritative. The host copy is
// already current on the legacy HC path; only the GGML-device path needs the
// one explicit device->host snapshot.
bool qwen4exp_hc_download_streams(StreamingEngine &engine, int S) {
  if (!g_hc_ggml_live) return true;
  return GgmlHcRunner::instance().download_streams(
      engine.hc_stream_buf.data(), S);
}

bool qwen4exp_hc_commit_host_layer_input(StreamingEngine &engine, int S) {
  if (!g_hc_ggml_live) return true;
  if (GgmlHcRunner::instance().upload_streams(engine.hc_stream_buf.data(), S))
    return true;
  hc_ggml_abort(engine, S, "layer-input device sync failed");
  return false;
}

static bool hc_ggml_usable(StreamingEngine &engine, int D) {
  if (!hc_ggml() || !GgmlHcRunner::compiled_in()) return false;
  auto &c = engine.model_file.architecture_config;
  if (c.hc_mult <= 1 || c.hc_low_rank <= 0) return false;
  const int dev = moe_ggml_device() >= 0 ? moe_ggml_device() : 0;
  return GgmlHcRunner::instance().ensure(dev, D, c.hc_mult, c.hc_low_rank,
                                         c.eps);
}

// Pull one bracket's four weights out of the resident dense pool and hand
// them to the runner. Kept here rather than in the runner so that header
// needs nothing from engine.hpp.
static bool hc_ggml_register(StreamingEngine &engine, int layer, int sub,
                             int D) {
  GgmlHcRunner &r = GgmlHcRunner::instance();
  if (r.weights_ready(layer, sub)) return true;
  const std::string p = "blk." + std::to_string(layer) + ".";
  const std::string pre = (sub == 0) ? "hc_attn_" : "hc_ffn_";
  auto &c = engine.model_file.architecture_config;
  const int hc_dim = c.hc_mult * D;

  const float *norm_w =
      find_resident_float_anytype(engine, p + pre + "norm.weight", (size_t)hc_dim);
  if (!norm_w) return false;
  GgmlHcWeightRef nref;
  nref.host_data = norm_w;
  nref.type = (uint32_t)GGUFType::F32;
  nref.ne0 = hc_dim;
  nref.ne1 = 1;
  nref.nbytes = (size_t)hc_dim * sizeof(float);
  // find_resident_float_anytype may return a dequantized host cache for a
  // non-F32 source.  Borrow the dense-pool pointer only for an actual F32
  // resident tensor, since the runner's norm tensor is F32.
  auto norm_it = engine.resident_layers.find(p + pre + "norm.weight");
  if (norm_it != engine.resident_layers.end() && norm_it->second.device_ptr &&
      norm_it->second.meta.type == GGUFType::F32) {
    nref.device_data = norm_it->second.device_ptr;
    nref.device_bytes = norm_it->second.meta.size_bytes;
  }

  auto ref = [&](const std::string &name, GgmlHcWeightRef &out) -> bool {
    auto it = engine.resident_layers.find(name);
    if (it == engine.resident_layers.end() || !it->second.host_ptr) return false;
    const auto &W = it->second;
    if (W.meta.shape.size() < 2) return false;
    out.host_data = W.host_ptr;
    out.type = (uint32_t)W.meta.type;
    out.ne0 = (int64_t)W.meta.shape[0];
    out.ne1 = (int64_t)W.meta.shape[1];
    out.nbytes = W.meta.size_bytes;
    out.device_data = W.device_ptr;
    out.device_bytes = W.device_ptr ? W.meta.size_bytes : 0;
    return true;
  };
  GgmlHcWeightRef dref, uref, iref;
  if (!ref(p + pre + "down.weight", dref) || !ref(p + pre + "up.weight", uref) ||
      !ref(p + pre + "inject.weight", iref))
    return false;
  return r.ensure_weights(layer, sub, nref, dref, uref, iref);
}

// Drop out of the GPU path for the rest of this pass, leaving the host copy
// of hc_stream_buf authoritative again.
static void hc_ggml_abort(StreamingEngine &engine, int S, const char *why) {
  if (!g_hc_ggml_live) return;
  auto &c = engine.model_file.architecture_config;
  GgmlHcRunner::instance().download_streams(engine.hc_stream_buf.data(), S);
  // Device HC post-combine normally consumes t_inject_ directly.  The host
  // copy is only needed if this forward pass is being abandoned and a later
  // fallback hook must finish the current bracket on CPU.  Keeping it lazy
  // removes one synchronising tensor-get from every successful device
  // pre_mix call.
  if (engine.qwen4exp_hc_inject.size() >=
      (size_t)S * (size_t)c.hc_mult) {
    GgmlHcRunner::instance().download_inject(
        engine.qwen4exp_hc_inject.data(), S);
  }
  g_hc_ggml_live = false;
  LOG_ERROR("[GGML-HC] %s -- reverting to the CPU path for the rest of this "
            "forward pass (hc_stream_buf synced back to host).\n", why);
  (void)c;
}

void qwen4exp_hc_pre_mix(StreamingEngine &engine, int layer_idx,
                           int sub_block, float *hidden_states, int S, int D) {
  auto &c = engine.model_file.architecture_config;
  const int HC = c.hc_mult;
  std::string p = "blk." + std::to_string(layer_idx) + ".";
  const std::string prefix = (sub_block == 0) ? "hc_attn_" : "hc_ffn_";

  engine.qwen4exp_device_hc_input = nullptr;
  engine.qwen4exp_device_hc_ffn_input = nullptr;
  if (sub_block == 0) {
    engine.qwen4exp_device_attn_output = nullptr;
    engine.qwen4exp_device_attn_output_ready = false;
    engine.qwen4exp_device_attn_output_raw = false;
  }

  engine.qwen4exp_hc_inject.resize((size_t)S * HC);
  if (g_hc_ggml_live) {
    const bool device_gdn_input =
        gdn_full_ggml() && sub_block == 0 &&
        qwen4exp_is_gdn_layer(c, layer_idx) &&
        !force_hc_gdn_pre_host();
    const bool dual_gdn_pre_output =
        device_gdn_input && dual_hc_gdn_pre_output();
    const bool device_full_attn_input =
        engine.qwen4exp_attn_gpu && sub_block == 0 && S > 0 &&
        !qwen4exp_is_gdn_layer(c, layer_idx);
    const bool device_moe_input =
        moe_ggml_mul_mat_id() && sub_block == 1 &&
        layer_idx >= c.first_dense && S > 1;
    const bool need_device_input =
        device_gdn_input || device_full_attn_input || device_moe_input;
    const bool keep_moe_input_on_device =
        // GPU routing consumes the mixed activation directly and returns only
        // the compact route table to the host.  The shared-expert flag used
        // to be included here, which accidentally forced a full [S,D] D2H
        // copy for the ordinary GPU-router path as well.  That copy is one
        // of the per-layer boundaries this path is intended to remove.
        // Without GPU routing the host router still needs hidden_states, so
        // its copy remains mandatory.
        device_moe_input && moe_ggml_gpu_router();
    const void *device_input = nullptr;
    if (hc_ggml_register(engine, layer_idx, sub_block, D) &&
        GgmlHcRunner::instance().pre_mix(layer_idx, sub_block, S,
                                         dual_gdn_pre_output
                                             ? hidden_states
                                             : ((device_gdn_input ||
                                                 device_full_attn_input ||
                                                 keep_moe_input_on_device)
                                                    ? nullptr
                                                    : hidden_states),
                                         need_device_input ? &device_input
                                                            : nullptr)) {
      GgmlHcRunner::instance().queue_pre_mix_dependency();
      // The normal device path consumes the injection vector from the
      // persistent GGML tensor in post_combine.  Do not download it here:
      // tensor_get synchronizes the backend stream once per bracket.  If a
      // later device operation fails, hc_ggml_abort() downloads the current
      // injection tensor before switching the pass back to the host path.
      if (device_gdn_input)
        engine.qwen4exp_device_hc_input = device_input;
      if (device_full_attn_input)
        engine.qwen4exp_device_hc_input = device_input;
      if (device_moe_input)
        engine.qwen4exp_device_hc_ffn_input = device_input;
      check_finite_hc("hidden_states (post qwen4exp hc_pre_mix, ggml)",
                      layer_idx, hidden_states, (size_t)S * D, D);
      return;
    }
    hc_ggml_abort(engine, S, "pre_mix failed");
  }
  qwen4exp_hc_mix_impl(engine, p + prefix + "norm.weight",
                         p + prefix + "down.weight", p + prefix + "up.weight",
                         p + prefix + "inject.weight", S, D, hidden_states,
                         engine.qwen4exp_hc_inject.data(), layer_idx);
  check_finite_hc("hidden_states (post qwen4exp hc_pre_mix)", layer_idx,
                  hidden_states, (size_t)S * D, D);
}

void qwen4exp_hc_post_combine(StreamingEngine &engine, int layer_idx,
                                int /*sub_block*/, float *hidden_states,
                                const float *sub_block_out, int S, int D) {
  auto &c = engine.model_file.architecture_config;
  const int HC = c.hc_mult;
  const float inv_hc = 1.0f / (float)HC;
  if (g_hc_ggml_live) {
    if (GgmlHcRunner::instance().post_combine(layer_idx, 0, S, sub_block_out))
      return;
    hc_ggml_abort(engine, S, "post_combine failed");
  }
  // Reads engine.qwen4exp_hc_inject, populated by the qwen4exp_hc_pre_mix
  // call for this SAME layer/sub_block just before the sub-block ran (same
  // paired-bracket invariant as hc_pre_collapse/hc_post_combine above).
  // Per-token independent (each s touches only its own HC*D stream slice),
  // so this is bit-identical to the serial form -- same reasoning as the
  // matching loops in qwen4exp_hc_pre_mix.
#pragma omp parallel for schedule(static)
  for (int s = 0; s < S; ++s) {
    const float *inj = engine.qwen4exp_hc_inject.data() + (size_t)s * HC;
    float *streams = engine.hc_stream_buf.data() + (size_t)s * HC * D;
    const float *sub_out = sub_block_out + (size_t)s * D;
    for (int h = 0; h < HC; ++h) {
      // 2*sigmoid centres the scatter weight on 1, so a zero injection is a
      // plain residual add (qwen4exp.cpp's build_hc_combine).
      float w = 2.0f / (1.0f + std::exp(-inj[h] * inv_hc));
      float *stream_h = streams + (size_t)h * D;
      for (int d = 0; d < D; ++d) stream_h[d] += sub_out[d] * w;
    }
  }
  check_finite_hc("hc_stream_buf (post qwen4exp hc_post_combine)", layer_idx,
                  engine.hc_stream_buf.data(), (size_t)S * HC * D, HC * D);

  // Keep hidden_states well-defined between sub-blocks (stream 0), same
  // convention as hc_post_combine above.
  for (int s = 0; s < S; ++s) {
    std::memcpy(hidden_states + (size_t)s * D,
                engine.hc_stream_buf.data() + (size_t)s * HC * D,
                (size_t)D * sizeof(float));
  }
}

bool qwen4exp_hc_post_combine_device(StreamingEngine &engine, int layer_idx,
                                      int sub_block, int S, int D,
                                      const void *device_sub_block_out) {
  if (!g_hc_ggml_live || !device_sub_block_out ||
      (sub_block != 0 && sub_block != 1))
    return false;
  if (sub_block == 1)
    GgmlMoeRunner::active_instance().wait_device_output();
  if (GgmlHcRunner::instance().post_combine_device(
          layer_idx, /*runner_sub_block=*/0, S, device_sub_block_out))
    return true;
  // Preserve the existing all-or-nothing HC invariant. The caller will copy
  // the MoE result back and invoke the ordinary host hook after this abort.
  hc_ggml_abort(engine, S, "post_combine_device failed");
  (void)D;
  return false;
}

bool qwen4exp_hc_post_combine_device_raw(StreamingEngine &engine,
                                          int layer_idx, int sub_block, int S,
                                          int D, const void *device_data) {
  if (!g_hc_ggml_live || !device_data || (sub_block != 0 && sub_block != 1))
    return false;
  if (GgmlHcRunner::instance().post_combine_device_raw(
          layer_idx, /*runner_sub_block=*/0, S, device_data,
          (size_t)S * D * sizeof(float)))
    return true;
  hc_ggml_abort(engine, S, "post_combine_device_raw failed");
  return false;
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

  // Hand the freshly expanded streams to the GPU and make the device copy
  // authoritative for the rest of this forward pass. This is the ONLY upload:
  // at [D, HC, S] = 18 MB, a per-bracket round trip would cost more than the
  // CPU path it replaces, which is the whole reason the runner keeps state.
  g_hc_ggml_live = false;
  if (hc_ggml_usable(engine, D) &&
      GgmlHcRunner::instance().upload_streams(engine.hc_stream_buf.data(), S))
    g_hc_ggml_live = true;
}

void hc_collapse_output(StreamingEngine &engine, float *hidden_states, int S,
                        int D) {
  auto &c = engine.model_file.architecture_config;
  int HC = c.hc_mult;
  const auto drain_start = std::chrono::steady_clock::now();

  // Backend-stream MoE copies are intentionally left in flight by the
  // per-layer path so H2D can overlap later host orchestration and graph
  // execution. This is the one forward-level drain.
  GgmlMoeRunner::finish_all_h2d();
  const auto after_h2d = std::chrono::steady_clock::now();
  GgmlMoeTransferStats moe_copy;
  GgmlMoeRunner::collect_all_transfer_stats(moe_copy);
  if (moe_copy.h2d_bytes || moe_copy.d2d_bytes) {
    perf().t_ggml_moe_d2d.fetch_add(moe_copy.d2d_seconds,
                                    std::memory_order_relaxed);
    perf().t_ggml_moe_h2d.fetch_add(moe_copy.h2d_seconds,
                                    std::memory_order_relaxed);
    perf().t_ggml_moe_h2d_dma.fetch_add(moe_copy.h2d_dma_seconds,
                                        std::memory_order_relaxed);
    perf().bytes_ggml_moe_d2d.fetch_add(moe_copy.d2d_bytes,
                                        std::memory_order_relaxed);
    perf().bytes_ggml_moe_h2d.fetch_add(moe_copy.h2d_bytes,
                                        std::memory_order_relaxed);
    if (moe_copy.h2d_bytes) {
      perf().t_prefill_stream_h2d.fetch_add(moe_copy.h2d_seconds,
                                            std::memory_order_relaxed);
      perf().bytes_prefill_stream_h2d.fetch_add(
          moe_copy.h2d_bytes, std::memory_order_relaxed);
    }
  }

  // Full device GDN keeps each layer's recurrent state and convolution tail
  // on the shared GGML backend while the 48-layer forward pass is queued.
  // Publish those small persistent updates to the engine's host state once,
  // before any subsequent request can observe them.
  // Prefill keeps the existing host state contract for prompt-cache and
  // fallback paths.  Decode consumes the persistent GDN state on the GGML
  // device and must not perform a state D2H/D2H round trip every token.
  GgmlGdnProjRunner::instance().finish_full_device_updates(S > 1);
  const auto after_gdn = std::chrono::steady_clock::now();

  // Every layer's brackets are done; bring the device state back so the
  // model-level collapse below (and anything else that reads hc_stream_buf)
  // sees the real streams. One download per forward pass.
  if (g_hc_ggml_live) {
    GgmlHcRunner::instance().download_streams(engine.hc_stream_buf.data(), S);
    g_hc_ggml_live = false;
  }
  const auto after_hc = std::chrono::steady_clock::now();
  LOG_INFO("[QWEN4EXP-DRAIN] H2D=%.3f ms GDN=%.3f ms HC=%.3f ms total=%.3f ms\n",
           std::chrono::duration<double, std::milli>(after_h2d - drain_start).count(),
           std::chrono::duration<double, std::milli>(after_gdn - after_h2d).count(),
           std::chrono::duration<double, std::milli>(after_hc - after_gdn).count(),
           std::chrono::duration<double, std::milli>(after_hc - drain_start).count());

  // GLM5Next (GLM 5.3 Flash): the final collapse is an UNWEIGHTED mean over
  // the HC streams (glm5next.cpp's graph constructor: `cur = build_hc_mean(
  // ctx0, inpL)` -- "unweighted mean, not DeepSeek-V4's learned gated head"),
  // and a glm5next file has no output_hc_* tensors at all. Every other
  // HC family (DeepSeek-V4 today) uses the learned gated head below.
  if (detect_model_family(engine.model_file) == ModelFamily::GLM5NEXT) {
    const float inv_hc = 1.0f / (float)HC;
    for (int s = 0; s < S; ++s) {
      const float *streams = engine.hc_stream_buf.data() + (size_t)s * HC * D;
      float *out = hidden_states + (size_t)s * D;
      std::memcpy(out, streams, (size_t)D * sizeof(float));
      for (int h = 1; h < HC; ++h) {
        const float *stream_h = streams + (size_t)h * D;
        for (int d = 0; d < D; ++d) out[d] += stream_h[d];
      }
      for (int d = 0; d < D; ++d) out[d] *= inv_hc;
    }
    check_finite_hc("hidden_states (post hc_collapse_output, hc_mean)", -1,
                    hidden_states, (size_t)S * D, D);
    return;
  }

  // Qwen4Exp (Qwen3.8-Flash-Next): the final mixer IS the output norm --
  // the file has no output_norm tensor at all. Same low-rank mix as the
  // per-layer hooks (output_hc_norm/down/up), but with NO inject (nothing
  // downstream to combine into). qwen4exp.cpp's graph tail: `build_hc_mix(
  // res_hc, hc_head_norm, hc_head_down, hc_head_up, nullptr, nullptr, -1)`.
  if (detect_model_family(engine.model_file) == ModelFamily::Qwen4Exp) {
    qwen4exp_hc_mix_impl(engine, "output_hc_norm.weight",
                           "output_hc_down.weight", "output_hc_up.weight",
                           /*inject_name=*/"", S, D, hidden_states,
                           /*out_inject=*/nullptr, /*trace_layer=*/-1);
    check_finite_hc("hidden_states (post qwen4exp hc_collapse_output)", -1,
                    hidden_states, (size_t)S * D, D);
    return;
  }

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
