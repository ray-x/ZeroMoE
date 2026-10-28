#include "active_layer_upload_workers.hpp"
#include "graph_config.hpp"
#include "async_prefetcher.hpp"
#include "cpu_expert_executor.hpp"
#include "gpu_expert_executor.hpp"
#include "dense_dispatch.hpp"
#include "engine.hpp"
#include "full_graph_model.hpp"
#include "expert_activation.hpp"
#include "ggml_bailingmoe3_graph.hpp"
#include "ggml_deepseek_v41_graph.hpp"
#include "dsv41_trace.hpp"
#include "ggml_gdn_proj_backend.hpp"
#include "ggml_glm5next_graph.hpp"
#include "ggml_hc_backend.hpp"
#include "ggml_moe_backend.hpp"
#include "ggml_qwen4exp_graph.hpp"
#include "qwen4exp_graph_mtp.hpp"
#include "kernels_cpu.hpp"
#include "kquant_dequant.hpp"
#include "logger.hpp"
#include "mtp_speculative.hpp"
#include "moe_pipeline.hpp"
#include "moe_router.hpp"
#include "perf_timer.hpp"
#include "prefill_expert_stream.hpp"
#include "ram_manager.hpp"
#include "scratch_buffers.hpp"
#include "tensor_trace.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <immintrin.h>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

extern void mla_attention_absorbed_gguf(StreamingEngine &engine, int layer_idx,
                                        const float *Q_roped, int S,
                                        int pos_base, float *out);
extern void dense_attention_gguf(StreamingEngine &engine, int layer_idx,
                                 const float *hidden_states, int S,
                                 int pos_base, float *out);
extern bool qwen4exp_hc_device_live();
extern bool qwen4exp_hc_post_combine_device(StreamingEngine &engine,
                                            int layer_idx, int sub_block, int S,
                                            int D,
                                            const void *device_sub_block_out,
                                            const void *producer_event);
extern bool qwen4exp_hc_post_combine_device_raw(StreamingEngine &engine,
                                                int layer_idx, int sub_block,
                                                int S, int D,
                                                const void *device_data);
extern bool qwen4exp_hc_prepare_host_layer_input(StreamingEngine &engine,
                                                 int layer_idx, int S);
extern bool qwen4exp_hc_download_streams(StreamingEngine &engine, int S);
extern bool qwen4exp_hc_commit_host_layer_input(StreamingEngine &engine,
                                                int layer_idx, int S);
extern bool qwen4exp_copy_gdn_device_output_to_host(const void *device_output,
                                                    float *host, int S);

// ============================================================
// Dense (non-MoE) FFN for leading dense layers (l < first_dense) in an
// otherwise-MoE model — plain gate_proj -> SiLU, up_proj, elementwise
// multiply, down_proj on ffn_{gate,up,down}.weight. No router, no expert
// cache. Used by both GLM-5.2's leading dense layers and Hunyuan/Hy-V3
// layers before the MoE portion of the stack begins.
// ============================================================
void dense_ffn_gguf(StreamingEngine &engine, int layer_idx,
                    const float *hidden_states, int S, float *output,
                    LayerScratch &scratch) {
  auto &c = engine.model_file.architecture_config;
  int D = c.hidden;
  int inter = c.dense_inter;
  if (inter <= 0) {
    std::fill(output, output + S * D, 0.0f);
    return;
  }

  std::string p = "blk." + std::to_string(layer_idx) + ".";
  auto norm_it = engine.resident_layers.find(p + "ffn_norm.weight");
  if (norm_it != engine.resident_layers.end() && norm_it->second.host_ptr) {
    // Same bug as moe_forward_gguf's router norm (see its comment):
    // rmsnorm_row only normalizes ONE row -- must loop over S or every
    // token past s=0 reads stale scratch.normed during prefill.
    const float *norm_w = reinterpret_cast<float *>(norm_it->second.host_ptr);
    for (int s = 0; s < S; ++s) {
      rmsnorm_row(scratch.normed.data() + (size_t)s * D,
                  hidden_states + (size_t)s * D, norm_w, D, c.eps);
    }
  } else {
    std::memcpy(scratch.normed.data(), hidden_states,
                (size_t)S * D * sizeof(float));
  }

  for (int s = 0; s < S; ++s) {
    const float *token_hidden = scratch.normed.data() + (size_t)s * D;
    float *token_output = output + (size_t)s * D;

    dense_proj(engine, p + "ffn_gate.weight", token_hidden,
               scratch.gate_buf.data(), 1, D, inter);
    dense_proj(engine, p + "ffn_up.weight", token_hidden, scratch.up_buf.data(),
               1, D, inter);
    silu_inplace(scratch.gate_buf.data(), inter);
    elem_mul(scratch.gate_buf.data(), scratch.gate_buf.data(),
             scratch.up_buf.data(), inter);
    dense_proj(engine, p + "ffn_down.weight", scratch.gate_buf.data(),
               token_output, 1, inter, D);
  }
}

// Pre-allocated once, reused every layer/call (avoids per-layer heap
// churn) -- promoted from a function-local static to file scope so
// shrink_moe_gpu_scratch() below can reach the SAME instance
// layers_forward_gguf uses, without changing either function's own
// behavior (still exactly one instance per thread, still only ever
// touched from the single main decode thread).
static thread_local LayerScratch g_layer_scratch;

// Frees every GpuExpertScratch slot's device buffers back to empty.
// Exposed (declared extern in generation_loop.hpp) so generate_text can
// call this right after a request's single prefill call completes, before
// settling into the decode loop. In practice this reclaims very little --
// expert FFN dispatch is always called with S=1 even during prefill (see
// the per-token loop in moe_forward_gguf), so these buffers never grow
// much regardless of prompt length -- this exists mainly for consistency
// with CudaManager::shrink_dense_scratch (the one that actually matters,
// see CLAUDE.md) and as defense-in-depth. Safe to call between requests:
// the next expert dispatch regrows via GpuExpertScratch::ensure's existing
// grow-only logic.
void shrink_moe_gpu_scratch() {
  for (auto &gs : g_layer_scratch.gpu_pool)
    gs.shrink();
}

// Execute GLM's optional NextN block using the same host-visible [S,D]
// contract as the target layer loop.  This intentionally lives in this
// translation unit: it can reuse the private LayerScratch/MoE implementation
// without exposing cache internals to the speculative driver.  The target
// trunk remains responsible for verification; this function only proposes
// from the extra DSA+MoE block.
bool glm_mtp_forward(StreamingEngine &engine, RAMSafetyManager &ram_mgr,
                     LockFreePrefetcher &async_io_queue, const int *token_ids,
                     const float *h_input, int S, int pos_base, float *h_nextn,
                     float *logits_last) {
  const auto &c = engine.model_file.architecture_config;
  const ModelFamily fam = engine.family_ops.family;
  // hc_mult > 1 is deliberately NOT a rejection here -- see
  // MtpSpeculator::supported()'s comment (mtp_speculative.cpp) for why the
  // NextN block's plain-residual math below is correct for GLM5NEXT's
  // multi-stream trunk too: the caller's hidden state is already collapsed
  // to single-stream by hc_collapse_output before it ever reaches this
  // function, and the reference NextN block itself has no HC mixer.
  if (!engine.mtp_enabled || c.n_layer_nextn != 1 || S <= 0 || c.hidden <= 0 ||
      c.vocab <= 0 ||
      (fam != ModelFamily::GLM5NEXT && fam != ModelFamily::GLM_DSA) ||
      !engine.family_ops.attention_fn ||
      engine.layers_experts.size() <= (size_t)c.n_layers) {
    return false;
  }
  // Keep the standalone NextN invocation under the same cache-mode contract
  // as layers_forward_gguf().  MTP uses this function for both a batched
  // prompt bridge (S>1) and one-row draft/correction steps (S==1).  Without
  // these updates, a one-row draft after prefill would inherit
  // RAMSafetyManager/GpuTopology's prefill mode and apply the wrong active-
  // layer/LRU policy.
  ram_mgr.in_prefill = (S > 1);
  if (ram_mgr.vram_cache)
    ram_mgr.vram_cache->in_prefill = (S > 1);
  if (S == 1 && pos_base == 0 && gdn_full_ggml())
    GgmlGdnProjRunner::instance().reset_full_device_states();
  const int layer_idx = c.n_layers;
  if (fam == ModelFamily::GLM5NEXT && glm5next_is_kda_layer(c, layer_idx)) {
    return false; // graph_initializer rejects this, keep the runtime guard too
  }
  if (pos_base < 0 || pos_base + S > engine.max_t) {
    LOG_STREAM_WARN() << "[MTP] position range [" << pos_base << ", "
                      << (pos_base + S) << ") exceeds max_t=" << engine.max_t
                      << " -- disabling this proposal batch\n";
    return false;
  }

  const int D = c.hidden;
  const std::string block = "blk." + std::to_string(layer_idx) + ".";
  auto has_host = [&](const std::string &name) {
    auto it = engine.resident_layers.find(name);
    return it != engine.resident_layers.end() && it->second.host_ptr;
  };
  // These names were checked before activating the feature.  Recheck the
  // resident map here because dense-pool backoff must never turn a required
  // MTP weight into a silent zero-filled projection.
  const char *required[] = {
      "nextn.eh_proj.weight", "nextn.enorm.weight",    "nextn.hnorm.weight",
      "attn_norm.weight",     "attn_q_a.weight",       "attn_q_a_norm.weight",
      "attn_q_b.weight",      "attn_kv_a_mqa.weight",  "attn_kv_a_norm.weight",
      "attn_k_b.weight",      "attn_v_b.weight",       "attn_output.weight",
      "ffn_norm.weight",      "ffn_gate_inp.weight",   "ffn_gate_shexp.weight",
      "ffn_up_shexp.weight",  "ffn_down_shexp.weight",
  };
  for (const char *suffix : required) {
    if (!has_host(block + suffix)) {
      LOG_STREAM_WARN() << "[MTP] required resident tensor missing: " << block
                        << suffix << " -- proposal batch skipped\n";
      return false;
    }
  }
  auto emb_it =
      engine.resident_layers.find(block + "nextn.embed_tokens.weight");
  if (emb_it == engine.resident_layers.end() || !emb_it->second.host_ptr)
    emb_it = engine.resident_layers.find("token_embd.weight");
  if (emb_it == engine.resident_layers.end() || !emb_it->second.host_ptr)
    return false;

  auto embed_row = [&](int tok, float *dst) {
    if (tok < 0 || tok >= c.vocab)
      return false;
    const int rb = row_bytes(emb_it->second.meta.type, D);
    if (rb <= 0)
      return false;
    return dequant_row(emb_it->second.meta.type,
                       emb_it->second.host_ptr + (int64_t)tok * rb, dst, D);
  };

  std::vector<float> token_emb((size_t)S * D, 0.0f);
  for (int s = 0; s < S; ++s) {
    if (!token_ids ||
        !embed_row(token_ids[s], token_emb.data() + (size_t)s * D))
      return false;
  }

  LayerScratch &scratch = g_layer_scratch;
  if (!scratch.resize(S, D, c.n_experts, c.moe_inter, c.topk, c.n_expert_shared,
                      c.dense_inter)) {
    LOG_STREAM_WARN() << "[MTP] scratch allocation denied by host-memory "
                         "budget -- skipping proposal batch\n";
    return false;
  }
  std::vector<float> e_norm((size_t)S * D);
  std::vector<float> h_norm((size_t)S * D);
  rmsnorm_proj(engine, block + "nextn.enorm.weight", token_emb.data(),
               e_norm.data(), S, D, c.eps);
  rmsnorm_proj(engine, block + "nextn.hnorm.weight", h_input, h_norm.data(), S,
               D, c.eps);

  std::vector<float> concat((size_t)S * 2 * D);
  for (int s = 0; s < S; ++s) {
    std::memcpy(concat.data() + (size_t)s * 2 * D,
                e_norm.data() + (size_t)s * D, (size_t)D * sizeof(float));
    std::memcpy(concat.data() + (size_t)s * 2 * D + D,
                h_norm.data() + (size_t)s * D, (size_t)D * sizeof(float));
  }

  std::vector<float> cur((size_t)S * D);
  dense_proj(engine, block + "nextn.eh_proj.weight", concat.data(), cur.data(),
             S, 2 * D, D);
  if (!std::all_of(cur.begin(), cur.end(),
                   [](float v) { return std::isfinite(v); }))
    return false;

  // DIAGNOSTIC (2026-09-09): per-call L2 norms through the NextN block's own
  // residual stream, added while chasing an abnormally low MTP acceptance
  // rate. Compares directly against the real trunk's own [TRACE-HIDDEN]
  // norms (layers_forward_gguf, same file) at verbose log level.
  // HIDDEN" -- a norm wildly out of the real layers' own range would
  // localize the bug to this block specifically rather than the draft/
  // verify math in mtp_speculative.cpp. Cheap (one sqrt-sum per call, S is
  // always tiny here) and unconditional, so it
  // survives into the next debugging session without a rebuild.
  // Matches layers_forward_gguf's own l2norm lambda EXACTLY (raw sqrt-sum,
  // not RMS) so these numbers are directly comparable against its
  // [TRACE-HIDDEN] output, not off by a sqrt(n) scale factor.
  auto mtp_l2norm = [](const float *p, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; ++i)
      s += (double)p[i] * (double)p[i];
    return std::sqrt(s);
  };
  LOG_STREAM_INFO() << "[MTP-DIAG] layer=" << layer_idx
                    << " pos_base=" << pos_base << " S=" << S
                    << " |cur(post eh_proj)|="
                    << mtp_l2norm(cur.data(), cur.size()) << "\n";

  // The family attention callback receives the raw residual input, just like
  // layers_forward_gguf().  In particular, GLM's MLA implementation performs
  // blk.<layer>.attn_norm internally before q/kv projection.  Applying the
  // norm here as well would normalize the GLM NextN attention input twice,
  // making the draft distribution diverge from the target distribution and
  // causing rejected drafts/corrupted long-form output.  The reference MTP
  // graph applies this norm exactly once inside its attention subgraph.
  engine.family_ops.attention_fn(engine, layer_idx, cur.data(), S, pos_base,
                                 scratch.attn_out.data());
  LOG_STREAM_INFO() << "[MTP-DIAG] layer=" << layer_idx << " |attn_out|="
                    << mtp_l2norm(scratch.attn_out.data(), (size_t)S * D)
                    << "\n";
  for (size_t i = 0; i < cur.size(); ++i)
    cur[i] += scratch.attn_out[i];

  rmsnorm_proj(engine, block + "ffn_norm.weight", cur.data(),
               scratch.normed.data(), S, D, c.eps);
  moe_forward_gguf(engine, ram_mgr, async_io_queue, layer_idx,
                   scratch.normed.data(), S, scratch.moe_out.data(), scratch,
                   token_ids);
  LOG_STREAM_INFO() << "[MTP-DIAG] layer=" << layer_idx << " |moe_out|="
                    << mtp_l2norm(scratch.moe_out.data(), (size_t)S * D)
                    << "\n";
  // MTP is intentionally host-residual in this first bridge.  Consuming a
  // future GGML device result here would make the host residual stale and
  // would also couple GLM MTP to Qwen's unrelated device-HC path.
  if (scratch.moe_device_output_ready) {
    scratch.moe_device_output = nullptr;
    scratch.moe_device_output_ready = false;
    LOG_STREAM_WARN() << "[MTP] NextN MoE returned a device-only result; "
                         "proposal batch skipped safely\n";
    return false;
  }
  for (size_t i = 0; i < cur.size(); ++i)
    cur[i] += scratch.moe_out[i];

  const std::string head_norm =
      has_host(block + "nextn.shared_head_norm.weight")
          ? block + "nextn.shared_head_norm.weight"
          : "output_norm.weight";
  if (!has_host(head_norm))
    return false;
  rmsnorm_proj(engine, head_norm, cur.data(), h_nextn, S, D, c.eps);
  if (!std::all_of(h_nextn, h_nextn + (size_t)S * D,
                   [](float v) { return std::isfinite(v); }))
    return false;
  if (logits_last) {
    const std::string head = block + "nextn.shared_head_head.weight";
    const std::string output_head = has_host(head) ? head : "output.weight";
    if (!has_host(output_head))
      return false;
    dense_proj(engine, output_head, h_nextn + (size_t)(S - 1) * D, logits_last,
               1, D, c.vocab);
    if (!std::all_of(logits_last, logits_last + c.vocab,
                     [](float v) { return std::isfinite(v); }))
      return false;
  }
  return true;
}

// Declared non-static in their own .cpp files (attention_layer.cpp,
// hyper_connections.cpp respectively) specifically so this function can
// reuse them; see each declaration's own comment.
void qwen4exp_full_attention_gguf(StreamingEngine &engine, int layer_idx,
                                  const float *hidden_states, int S,
                                  int pos_base, float *out_context);
void qwen4exp_hc_mix_impl(StreamingEngine &engine, const std::string &norm_name,
                          const std::string &down_name,
                          const std::string &up_name,
                          const std::string &inject_name, int S, int D,
                          float *out_hidden, float *out_inject, int trace_layer,
                          const float *in_streams, bool force_original_quant);

// Qwen4Exp shared MTP checkpoint's own extra full-attention+MoE block.
// Verified against /path/to/unsloth/llama.cpp's qwen4exp.cpp
// (branch mtp/qwen4exp-nextn, graph_mtp) and a real tensor dump of
// mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf. Structurally: (1) eh_proj
// bridges the token embedding and the incoming wide HC state into a fresh
// per-stream residual res_hc, (2) hc_attn/hc_ffn sub-blocks run exactly
// like a normal Qwen4Exp trunk full-attention layer's own HC pre-mix ->
// sub-block -> HC post-combine, but against res_hc (a LOCAL buffer, never
// engine.hc_stream_buf), (3) res_hc is exported as h_nextn BEFORE the
// final hc_head collapse, matching the reference's own "next draft step
// re-enters here" comment, (4) the collapsed [S,D] result is projected to
// logits via the shared LM head (nextn.shared_head_head, or the base
// model's own output.weight).
//
// The routed-expert MoE step deliberately does NOT go through
// moe_forward_gguf/engine.layers_experts -- this block's own packed
// expert tensors are resident-loaded whole from the SEPARATE
// engine.mtp_model_file (graph_initializer.cpp's qwen4exp_nextn_manifest)
// rather than routed through the disk-streaming/RAM-VRAM LRU cache that
// every other layer's routed experts use (that machinery is hardwired to
// engine.model_file throughout ram_manager.hpp/async_prefetcher.hpp). Per-
// expert slicing uses the exact same size_bytes/n_experts stride
// convention as StreamingEngine::packed_slab_geometry (engine.hpp).
bool qwen4exp_mtp_forward(StreamingEngine &engine, RAMSafetyManager &ram_mgr,
                          LockFreePrefetcher &async_io_queue,
                          const int *token_ids, const float *h_input, int S,
                          int pos_base, float *h_nextn, float *logits_last) {
  auto &c = engine.model_file.architecture_config;
  const ModelFamily fam = engine.family_ops.family;
  if (!engine.mtp_enabled || !engine.has_mtp_model || c.n_layer_nextn != 1 ||
      S <= 0 || c.hidden <= 0 || c.vocab <= 0 || c.hc_mult <= 1 ||
      fam != ModelFamily::Qwen4Exp) {
    return false;
  }
  const int layer_idx = c.n_layers;
  if (pos_base < 0 || pos_base + S > engine.max_t) {
    LOG_STREAM_WARN() << "[MTP] position range [" << pos_base << ", "
                      << (pos_base + S) << ") exceeds max_t=" << engine.max_t
                      << " -- disabling this proposal batch\n";
    return false;
  }
  if ((int)engine.Kc.size() <= layer_idx ||
      (int)engine.Vc.size() <= layer_idx || !engine.Kc[layer_idx] ||
      !engine.Vc[layer_idx]) {
    LOG_STREAM_WARN() << "[MTP] Qwen4Exp NextN KV cache (layer " << layer_idx
                      << ") not allocated -- proposal batch skipped\n";
    return false;
  }

  const int D = c.hidden;
  const int HC = c.hc_mult;
  const int hc_dim = HC * D;
  const int n_experts = c.n_experts;
  const int topk = c.topk;
  const int moe_inter = c.moe_inter;
  const std::string p = "blk." + std::to_string(layer_idx) + ".";

  auto has_host = [&](const std::string &name) {
    auto it = engine.resident_layers.find(name);
    return it != engine.resident_layers.end() && it->second.host_ptr;
  };
  // Recheck resident tensors here (defense-in-depth, same convention as
  // glm_mtp_forward): graph_initializer.cpp validated these before
  // activating MTP, but a dense-pool backoff must never turn a required
  // MTP weight into a silent zero-filled projection.
  const char *required[] = {
      "attn_q.weight",
      "attn_k.weight",
      "attn_v.weight",
      "attn_output.weight",
      "attn_q_norm.weight",
      "attn_k_norm.weight",
      "hc_attn_norm.weight",
      "hc_attn_down.weight",
      "hc_attn_up.weight",
      "hc_attn_inject.weight",
      "hc_ffn_norm.weight",
      "hc_ffn_down.weight",
      "hc_ffn_up.weight",
      "hc_ffn_inject.weight",
      "ffn_gate_inp.weight",
      "ffn_gate_exps.weight",
      "ffn_up_exps.weight",
      "ffn_down_exps.weight",
      "nextn.eh_proj.weight",
      "nextn.enorm.weight",
      "nextn.hnorm.weight",
      "nextn.hc_head_norm.weight",
      "nextn.hc_head_down.weight",
      "nextn.hc_head_up.weight",
  };
  for (const char *suffix : required) {
    if (!has_host(p + suffix)) {
      LOG_STREAM_WARN() << "[MTP] required resident tensor missing: " << p
                        << suffix << " -- proposal batch skipped\n";
      return false;
    }
  }
  auto emb_it = engine.resident_layers.find(p + "nextn.embed_tokens.weight");
  if (emb_it == engine.resident_layers.end() || !emb_it->second.host_ptr)
    emb_it = engine.resident_layers.find("token_embd.weight");
  if (emb_it == engine.resident_layers.end() || !emb_it->second.host_ptr)
    return false;

  // Never let this block's attention call pick up a stale device-resident
  // activation left behind by the target trunk's own last GPU-HC layer
  // (engine.qwen4exp_device_hc_input/_ffn_input) -- this function never
  // calls qwen4exp_hc_pre_mix (the trunk's own hook, which would normally
  // set/clear these), so a leftover non-null value here would make
  // qwen4exp_full_attention_gguf silently consume the WRONG activation
  // (the trunk's, not this block's cur_attn buffer below). Force the safe
  // host dense_proj fallback path unconditionally.
  engine.qwen4exp_device_hc_input = nullptr;
  engine.qwen4exp_device_hc_ffn_input = nullptr;

  auto embed_row = [&](int tok, float *dst) {
    if (tok < 0 || tok >= c.vocab)
      return false;
    const int rb = row_bytes(emb_it->second.meta.type, D);
    if (rb <= 0)
      return false;
    return dequant_row(emb_it->second.meta.type,
                       emb_it->second.host_ptr + (int64_t)tok * rb, dst, D);
  };
  std::vector<float> token_emb((size_t)S * D, 0.0f);
  for (int s = 0; s < S; ++s) {
    if (!token_ids ||
        !embed_row(token_ids[s], token_emb.data() + (size_t)s * D))
      return false;
  }

  // ---- eh_proj bridge: e_norm (token embedding) concat h_norm (incoming
  // wide state, grouped RMS-norm'd per stream) -> res_hc ----
  std::vector<float> e_norm((size_t)S * D);
  rmsnorm_proj(engine, p + "nextn.enorm.weight", token_emb.data(),
               e_norm.data(), S, D, c.eps);

  std::vector<float> h_norm((size_t)S * hc_dim);
  {
    const float *hnorm_w = nullptr;
    auto hn_it = engine.resident_layers.find(p + "nextn.hnorm.weight");
    hnorm_w = reinterpret_cast<float *>(hn_it->second.host_ptr);
    // Grouped RMS norm: each (token, stream) row normalized independently
    // over its own D elements, NO learned gain in the norm itself (matches
    // ggml_rms_norm's definition) -- the hc_dim-wide learned gain
    // (nextn.hnorm, one scalar per (stream,dim) pair, NOT broadcast across
    // streams -- unlike e_norm below) is applied after, over the flattened
    // [hc_dim] layout. Identical math to qwen4exp_hc_mix_impl's own steps
    // 1-2, inlined here since that function's own gate/inject math past
    // step 2 doesn't apply to this bridge step.
    for (int s = 0; s < S; ++s) {
      const float *in_s = h_input + (size_t)s * hc_dim;
      float *out_s = h_norm.data() + (size_t)s * hc_dim;
      for (int h = 0; h < HC; ++h) {
        const float *row = in_s + (size_t)h * D;
        double ss = 0.0;
        for (int d = 0; d < D; ++d)
          ss += (double)row[d] * row[d];
        float inv = 1.0f / std::sqrt((float)(ss / (double)D) + c.eps);
        float *out_row = out_s + (size_t)h * D;
        for (int d = 0; d < D; ++d)
          out_row[d] = row[d] * inv;
      }
      if (hnorm_w)
        for (int j = 0; j < hc_dim; ++j)
          out_s[j] *= hnorm_w[j];
    }
  }

  std::vector<float> concat((size_t)S * HC * 2 * D);
  for (int s = 0; s < S; ++s) {
    const float *e = e_norm.data() + (size_t)s * D;
    for (int h = 0; h < HC; ++h) {
      float *row = concat.data() + ((size_t)s * HC + h) * 2 * D;
      std::memcpy(row, e, (size_t)D * sizeof(float));
      std::memcpy(row + D, h_norm.data() + ((size_t)s * HC + h) * D,
                  (size_t)D * sizeof(float));
    }
  }

  std::vector<float> res_hc((size_t)S * hc_dim);
  dense_proj(engine, p + "nextn.eh_proj.weight", concat.data(), res_hc.data(),
             S * HC, 2 * D, D);
  if (!std::all_of(res_hc.begin(), res_hc.end(),
                   [](float v) { return std::isfinite(v); }))
    return false;

  // ---- hc_attn sub-block ----
  std::vector<float> cur_attn((size_t)S * D), inject_attn((size_t)S * HC);
  qwen4exp_hc_mix_impl(engine, p + "hc_attn_norm.weight",
                       p + "hc_attn_down.weight", p + "hc_attn_up.weight",
                       p + "hc_attn_inject.weight", S, D, cur_attn.data(),
                       inject_attn.data(), layer_idx, res_hc.data(), false);
  std::vector<float> attn_out((size_t)S * D);
  qwen4exp_full_attention_gguf(engine, layer_idx, cur_attn.data(), S, pos_base,
                               attn_out.data());
  {
    const float inv_hc = 1.0f / (float)HC;
    for (int s = 0; s < S; ++s) {
      const float *inj = inject_attn.data() + (size_t)s * HC;
      const float *sub_out = attn_out.data() + (size_t)s * D;
      float *streams = res_hc.data() + (size_t)s * hc_dim;
      for (int h = 0; h < HC; ++h) {
        float w = 2.0f / (1.0f + std::exp(-inj[h] * inv_hc));
        float *stream_h = streams + (size_t)h * D;
        for (int d = 0; d < D; ++d)
          stream_h[d] += sub_out[d] * w;
      }
    }
  }

  // ---- hc_ffn sub-block: routed MoE (v1: no active-layer/LRU cache, the
  // shared checkpoint's whole expert table is already resident) + shared
  // expert ----
  std::vector<float> cur_ffn((size_t)S * D), inject_ffn((size_t)S * HC);
  qwen4exp_hc_mix_impl(engine, p + "hc_ffn_norm.weight",
                       p + "hc_ffn_down.weight", p + "hc_ffn_up.weight",
                       p + "hc_ffn_inject.weight", S, D, cur_ffn.data(),
                       inject_ffn.data(), layer_idx, res_hc.data(), false);
  std::vector<float> moe_out((size_t)S * D, 0.0f);
  {
    auto &gate_t = engine.resident_layers[p + "ffn_gate_exps.weight"];
    auto &up_t = engine.resident_layers[p + "ffn_up_exps.weight"];
    auto &down_t = engine.resident_layers[p + "ffn_down_exps.weight"];
    if (!gate_t.host_ptr || !up_t.host_ptr || !down_t.host_ptr) {
      LOG_STREAM_WARN() << "[MTP] Qwen4Exp NextN packed expert tensors not "
                           "resident -- proposal batch skipped\n";
      return false;
    }
    const size_t gate_stride = gate_t.meta.size_bytes / (size_t)n_experts;
    const size_t up_stride = up_t.meta.size_bytes / (size_t)n_experts;
    const size_t down_stride = down_t.meta.size_bytes / (size_t)n_experts;

    std::vector<float> router_logits((size_t)S * n_experts);
    dense_proj(engine, p + "ffn_gate_inp.weight", cur_ffn.data(),
               router_logits.data(), S, D, n_experts);
    for (int s = 0; s < S; ++s)
      apply_gating_activation(c.gating_func,
                              router_logits.data() + (size_t)s * n_experts,
                              n_experts);
    if (!std::all_of(router_logits.begin(), router_logits.end(),
                     [](float v) { return std::isfinite(v); }))
      return false;

    std::vector<float> gate_buf((size_t)moe_inter), up_buf((size_t)moe_inter);
    std::vector<float> expert_out((size_t)D);
    std::vector<std::pair<float, int>> scores((size_t)n_experts);
    const float clamp_exp = (layer_idx < (int)c.swiglu_clamp_exp.size())
                                ? c.swiglu_clamp_exp[layer_idx]
                                : 0.0f;
    for (int s = 0; s < S; ++s) {
      float *token_probs = router_logits.data() + (size_t)s * n_experts;
      for (int e = 0; e < n_experts; ++e)
        scores[e] = {token_probs[e], e};
      std::partial_sort(
          scores.begin(), scores.begin() + topk, scores.end(),
          [](const auto &a, const auto &b) { return a.first > b.first; });
      float sum_w = 0.0f;
      for (int k = 0; k < topk; ++k)
        sum_w += scores[k].first;
      float inv_sum = c.expert_weights_norm
                          ? 1.0f / std::max(sum_w, 6.103515625e-5f)
                          : 1.0f;
      float *out_row = moe_out.data() + (size_t)s * D;
      for (int k = 0; k < topk; ++k) {
        const int e = scores[k].second;
        float w = scores[k].first * inv_sum;
        if (c.expert_weights_scale != 0.0f)
          w *= c.expert_weights_scale;
        GGUFTensor gm = gate_t.meta, um = up_t.meta, dm = down_t.meta;
        const uint8_t *gd = gate_t.host_ptr + (size_t)e * gate_stride;
        const uint8_t *ud = up_t.host_ptr + (size_t)e * up_stride;
        const uint8_t *dd = down_t.host_ptr + (size_t)e * down_stride;
        expert_ffn_cpu(
            cur_ffn.data() + (size_t)s * D, expert_out.data(), gd, gm, ud, um,
            dd, dm, gate_buf.data(), up_buf.data(), 1, D, moe_inter, clamp_exp,
            layer_idx, e);
        for (int d = 0; d < D; ++d)
          out_row[d] += expert_out[d] * w;
      }
    }

    // Shared expert (mirrors moe_forward_gguf's own CPU shexp block --
    // same formula, same ffn_gate_inp_shexp per-token sigmoid gate).
    if (c.n_expert_shared > 0 && has_host(p + "ffn_gate_shexp.weight")) {
      const int shared_inter = moe_inter * c.n_expert_shared;
      const float clamp_shexp = (layer_idx < (int)c.swiglu_clamp_shexp.size())
                                    ? c.swiglu_clamp_shexp[layer_idx]
                                    : 0.0f;
      std::vector<float> sg((size_t)shared_inter), su((size_t)shared_inter);
      for (int s = 0; s < S; ++s) {
        const float *tok = cur_ffn.data() + (size_t)s * D;
        dense_proj(engine, p + "ffn_gate_shexp.weight", tok, sg.data(), 1, D,
                   shared_inter);
        dense_proj(engine, p + "ffn_up_shexp.weight", tok, su.data(), 1, D,
                   shared_inter);
        silu_inplace(sg.data(), shared_inter);
        elem_mul(sg.data(), sg.data(), su.data(), shared_inter);
        swiglu_clamp_inplace(sg.data(), shared_inter, clamp_shexp);
        std::vector<float> shexp_down((size_t)D);
        dense_proj(engine, p + "ffn_down_shexp.weight", sg.data(),
                   shexp_down.data(), 1, shared_inter, D);
        float shexp_gate = 1.0f;
        if (has_host(p + "ffn_gate_inp_shexp.weight")) {
          float g = 0.0f;
          dense_proj(engine, p + "ffn_gate_inp_shexp.weight", tok, &g, 1, D, 1);
          shexp_gate = 1.0f / (1.0f + std::exp(-g));
        }
        float *out_row = moe_out.data() + (size_t)s * D;
        for (int d = 0; d < D; ++d)
          out_row[d] += shexp_down[d] * shexp_gate;
      }
    }
  }
  if (!std::all_of(moe_out.begin(), moe_out.end(),
                   [](float v) { return std::isfinite(v); }))
    return false;
  {
    const float inv_hc = 1.0f / (float)HC;
    for (int s = 0; s < S; ++s) {
      const float *inj = inject_ffn.data() + (size_t)s * HC;
      const float *sub_out = moe_out.data() + (size_t)s * D;
      float *streams = res_hc.data() + (size_t)s * hc_dim;
      for (int h = 0; h < HC; ++h) {
        float w = 2.0f / (1.0f + std::exp(-inj[h] * inv_hc));
        float *stream_h = streams + (size_t)h * D;
        for (int d = 0; d < D; ++d)
          stream_h[d] += sub_out[d] * w;
      }
    }
  }

  // Export the WIDE state before collapse -- "the next draft step
  // re-enters here" (reference's own comment on res_hc/t_h_nextn).
  std::memcpy(h_nextn, res_hc.data(), (size_t)S * hc_dim * sizeof(float));

  std::vector<float> cur_head((size_t)S * D);
  qwen4exp_hc_mix_impl(engine, p + "nextn.hc_head_norm.weight",
                       p + "nextn.hc_head_down.weight",
                       p + "nextn.hc_head_up.weight", "", S, D, cur_head.data(),
                       nullptr, layer_idx, res_hc.data(), false);
  if (!std::all_of(cur_head.begin(), cur_head.end(),
                   [](float v) { return std::isfinite(v); }))
    return false;

  if (logits_last) {
    const std::string head = p + "nextn.shared_head_head.weight";
    const std::string output_head = has_host(head) ? head : "output.weight";
    if (!has_host(output_head))
      return false;
    dense_proj(engine, output_head, cur_head.data() + (size_t)(S - 1) * D,
               logits_last, 1, D, c.vocab);
    if (!std::all_of(logits_last, logits_last + c.vocab,
                     [](float v) { return std::isfinite(v); }))
      return false;
  }
  return true;
}

// TEMPORARY DIAGNOSTIC (2026-09-12) -- investigating the confirmed ~35-50%
// gap between the [REQUEST] line's own wall_ms and the sum of its 4 named
// top-level buckets (attn=/route=/expert=/dense_gpu=). Every named
// PERF_SCOPE/PERF_START in layers_forward_gguf's per-layer loop has already
// been checked; this accumulates the loop body's OWN full wall-clock time
// (wrapping one entire iteration, all 40ish layers, for the whole forward
// call) so it can be compared directly against that gap. Single-writer
// (the one thread running generate_text's prefill/decode calls) -- plain
// double, no atomic needed, same as diag_lmhead_ms/diag_vramlog_ms in
// generation_loop.hpp. Reset once per request from generate_text via
// diag_reset_layer_loop_total(); read via diag_get_layer_loop_total_ms()
// right before print_summary("REQUEST", ...). Left in place after this
// investigation for a later cleanup/productionize pass -- do not remove.
static double g_diag_layer_loop_total_ms = 0.0;
// Wall-clock around the two suspected async-dispatch-return-early call
// sites (fo.attention_fn and the dense_ffn_gguf/moe_forward_gguf block) --
// see each call site's own comment. Compared against t_attn/t_moe_expert's
// own PERF_SCOPE-measured durations for the SAME calls: if GPU work is
// queued but not actually awaited before PERF_SCOPE's RAII destructor
// fires, the named counter under-reports relative to this wall-clock sum.
static double g_diag_attn_call_ms = 0.0;
static double g_diag_ffn_call_ms = 0.0;
static double g_diag_decode_layer_loop_ms = 0.0;
static double g_diag_decode_attn_call_ms = 0.0;
static double g_diag_decode_ffn_call_ms = 0.0;
static double g_diag_decode_pilot_ms = 0.0;
static double g_diag_decode_hc_ms = 0.0;
static double g_diag_decode_hc_pre_ms = 0.0;
static double g_diag_decode_hc_attn_pre_ms = 0.0;
static double g_diag_decode_hc_ffn_pre_ms = 0.0;
static double g_diag_decode_hc_post_ms = 0.0;
static double g_diag_decode_layer_input_ms = 0.0;

// Opt-in raw boundary capture for the ordinary S==1 path. Norms cannot tell
// us which side of the HC/attention/FFN boundary first diverges, so this
// writes exact F32 rows for one selected layer and prompt position. It is
// disabled by default and intended only for A/A diagnosis.
static const char *prefill_boundary_dump_base() {
  static const char *value = zeromoe::graphcfg::env("ZEROMOE_PREFILL_BOUNDARY_DUMP");
  return value && *value ? value : nullptr;
}

static int prefill_boundary_dump_last_pos() {
  static const int value = [] {
    const char *env = zeromoe::graphcfg::env("ZEROMOE_PREFILL_BOUNDARY_LAST_POS");
    return env && *env ? std::atoi(env) : -1;
  }();
  return value;
}

static int prefill_boundary_dump_first_pos() {
  static const int value = [] {
    const char *env = zeromoe::graphcfg::env("ZEROMOE_PREFILL_BOUNDARY_FIRST_POS");
    return env && *env ? std::atoi(env) : 0;
  }();
  return value;
}

static int prefill_boundary_dump_layer() {
  static const int value = [] {
    const char *env = zeromoe::graphcfg::env("ZEROMOE_PREFILL_BOUNDARY_LAYER");
    return env && *env ? std::atoi(env) : 0;
  }();
  return value;
}

static bool prefill_boundary_dump_layer_selected(int layer) {
  const int selected = prefill_boundary_dump_layer();
  // -1 means all layers.  This is useful for a same-token teacher-forced
  // sweep: capture one exact position at every layer without changing the
  // execution path or allocating a second graph.
  return selected < 0 || layer == selected;
}

static uint64_t prefill_boundary_fnv64(const float *data, size_t count) {
  constexpr uint64_t offset = 1469598103934665603ULL;
  constexpr uint64_t prime = 1099511628211ULL;
  uint64_t hash = offset;
  const auto *bytes = reinterpret_cast<const uint8_t *>(data);
  for (size_t i = 0; i < count * sizeof(float); ++i) {
    hash ^= bytes[i];
    hash *= prime;
  }
  return hash;
}

static void dump_prefill_boundary(const char *label, int layer, int pos,
                                  const float *data, int elements) {
  const char *base = prefill_boundary_dump_base();
  if (!base || !data || elements <= 0 ||
      !prefill_boundary_dump_layer_selected(layer) ||
      pos < prefill_boundary_dump_first_pos() ||
      pos > prefill_boundary_dump_last_pos())
    return;

  const std::string path = std::string(base) + ".l" +
                           (layer < 10 ? "0" : "") +
                           std::to_string(layer) + ".p" + std::to_string(pos) +
                           "." + label + ".f32";
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output)
    return;
  output.write(reinterpret_cast<const char *>(data),
               static_cast<std::streamsize>(elements * sizeof(float)));
  output.close();
  LOG_STREAM_INFO() << "[PREFILL-BOUNDARY] layer=" << layer
                    << " pos=" << pos << " label=" << label
                    << " elements=" << elements << " fnv64=0x" << std::hex
                    << prefill_boundary_fnv64(data, elements) << std::dec
                    << " path=" << path << "\n";
}

static void dump_prefill_boundary_i32(const char *label, int layer, int pos,
                                      const int *data, int elements) {
  const char *base = prefill_boundary_dump_base();
  if (!base || !data || elements <= 0 ||
      !prefill_boundary_dump_layer_selected(layer) ||
      pos < prefill_boundary_dump_first_pos() ||
      pos > prefill_boundary_dump_last_pos())
    return;

  const std::string path = std::string(base) + ".l" +
                           (layer < 10 ? "0" : "") +
                           std::to_string(layer) + ".p" + std::to_string(pos) +
                           "." + label + ".i32";
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output)
    return;
  output.write(reinterpret_cast<const char *>(data),
               static_cast<std::streamsize>(elements * sizeof(int)));
  output.close();
  LOG_STREAM_INFO() << "[PREFILL-BOUNDARY] layer=" << layer
                    << " pos=" << pos << " label=" << label
                    << " elements=" << elements << " path=" << path << "\n";
}

static void dump_prefill_device_boundary(const char *label, int layer, int pos,
                                         const void *device_data,
                                         int elements) {
  const char *base = prefill_boundary_dump_base();
  if (!base || !device_data || elements <= 0 ||
      !prefill_boundary_dump_layer_selected(layer) ||
      pos < prefill_boundary_dump_first_pos() ||
      pos > prefill_boundary_dump_last_pos())
    return;
  std::vector<float> host(static_cast<size_t>(elements));
  if (!GgmlHcRunner::instance().copy_pre_mix_to_host(device_data, host.data(),
                                                      /*S=*/1)) {
    LOG_STREAM_WARN() << "[PREFILL-BOUNDARY] failed to copy device label="
                      << label << " layer=" << layer << " pos=" << pos
                      << "\n";
    return;
  }
  dump_prefill_boundary(label, layer, pos, host.data(), elements);
}

static void dump_prefill_hc_streams(const char *label, int layer,
                                    int pos_base, int S, int hc_mult,
                                    int d_model) {
  const char *base = prefill_boundary_dump_base();
  if (!base || hc_mult <= 0 || d_model <= 0 || S <= 0 ||
      !prefill_boundary_dump_layer_selected(layer))
    return;
  bool any_in_range = false;
  for (int i = 0; i < S; ++i) {
    const int pos = pos_base + i;
    if (pos >= prefill_boundary_dump_first_pos() &&
        pos <= prefill_boundary_dump_last_pos()) {
      any_in_range = true;
      break;
    }
  }
  if (!any_in_range) return;
  const size_t per_token = static_cast<size_t>(hc_mult) * d_model;
  std::vector<float> host((size_t)S * per_token);
  if (!GgmlHcRunner::instance().download_streams(host.data(), S)) {
    LOG_STREAM_WARN() << "[PREFILL-BOUNDARY] failed to copy HC streams label="
                      << label << " layer=" << layer
                      << " pos_base=" << pos_base << "\n";
    return;
  }
  for (int i = 0; i < S; ++i) {
    const int pos = pos_base + i;
    dump_prefill_boundary(label, layer, pos, host.data() + (size_t)i * per_token,
                          static_cast<int>(per_token));
  }
}

static void dump_prefill_attention_output(int layer, int pos_base, int S,
                                          const void *device_output,
                                          int d_model) {
  const char *base = prefill_boundary_dump_base();
  if (!base || !device_output || d_model <= 0 || S <= 0 ||
      !prefill_boundary_dump_layer_selected(layer))
    return;
  // Any requested absolute position inside [pos_base, pos_base+S) is
  // servable from one full-batch download -- batched (S>1) prefill calls
  // this once per layer with the whole chunk, not once per token, so the
  // old S=1-only version could never answer a batched-mode boundary dump.
  bool any_in_range = false;
  for (int i = 0; i < S; ++i) {
    const int pos = pos_base + i;
    if (pos >= prefill_boundary_dump_first_pos() &&
        pos <= prefill_boundary_dump_last_pos()) {
      any_in_range = true;
      break;
    }
  }
  if (!any_in_range) return;
  std::vector<float> host((size_t)S * d_model);
  if (!qwen4exp_copy_gdn_device_output_to_host(device_output, host.data(),
                                                S)) {
    LOG_STREAM_WARN() << "[PREFILL-BOUNDARY] failed to copy attention output "
                      << "layer=" << layer << " pos_base=" << pos_base << "\n";
    return;
  }
  for (int i = 0; i < S; ++i) {
    const int pos = pos_base + i;
    dump_prefill_boundary("attn_out", layer, pos,
                          host.data() + (size_t)i * d_model, d_model);
  }
}

void diag_reset_layer_loop_total() {
  g_diag_layer_loop_total_ms = 0.0;
  g_diag_attn_call_ms = 0.0;
  g_diag_ffn_call_ms = 0.0;
  g_diag_decode_layer_loop_ms = 0.0;
  g_diag_decode_attn_call_ms = 0.0;
  g_diag_decode_ffn_call_ms = 0.0;
  g_diag_decode_pilot_ms = 0.0;
  g_diag_decode_hc_ms = 0.0;
  g_diag_decode_hc_pre_ms = 0.0;
  g_diag_decode_hc_attn_pre_ms = 0.0;
  g_diag_decode_hc_ffn_pre_ms = 0.0;
  g_diag_decode_hc_post_ms = 0.0;
  g_diag_decode_layer_input_ms = 0.0;
  moe_pipeline_detail::reset_diagnostics();
}
double diag_get_layer_loop_total_ms() { return g_diag_layer_loop_total_ms; }
double diag_get_attn_call_ms() { return g_diag_attn_call_ms; }
double diag_get_ffn_call_ms() { return g_diag_ffn_call_ms; }
double diag_get_decode_layer_loop_ms() { return g_diag_decode_layer_loop_ms; }
double diag_get_decode_attn_call_ms() { return g_diag_decode_attn_call_ms; }
double diag_get_decode_ffn_call_ms() { return g_diag_decode_ffn_call_ms; }
double diag_get_decode_pilot_ms() { return g_diag_decode_pilot_ms; }
double diag_get_decode_hc_ms() { return g_diag_decode_hc_ms; }
double diag_get_decode_hc_pre_ms() { return g_diag_decode_hc_pre_ms; }
double diag_get_decode_hc_attn_pre_ms() { return g_diag_decode_hc_attn_pre_ms; }
double diag_get_decode_hc_ffn_pre_ms() { return g_diag_decode_hc_ffn_pre_ms; }
double diag_get_decode_hc_post_ms() { return g_diag_decode_hc_post_ms; }
double diag_get_decode_layer_input_ms() {
  return g_diag_decode_layer_input_ms;
}
// ============================================================
// Main layer-by-layer forward pass with pre-allocated scratch,
// speculative pilot prefetch, and performance instrumentation.
// ============================================================
void layers_forward_gguf(StreamingEngine &engine, RAMSafetyManager &ram_mgr,
                         LockFreePrefetcher &async_io_queue,
                         float *hidden_states, int S, int pos_base,
                         const int *token_ids) {
  auto &c = engine.model_file.architecture_config;
  int D = c.hidden;

  // S > 1 = prefill (whole prompt in one pass), S == 1 = decode (one
  // token at a time) -- see RAMSafetyManager::in_prefill's own comment
  // for why the prefetch miss path needs this split.
  ram_mgr.in_prefill = (S > 1);
  // Mirror on GpuTopology -- see GpuTopology::in_prefill's own comment
  // for why active-layer promotion gates on this (2026-08-12).
  if (ram_mgr.vram_cache)
    ram_mgr.vram_cache->in_prefill = (S > 1);

  // Measurement only (expert_fetch_tracker.hpp) -- one call to this
  // function IS one token's worth of processing (prefill's single S>1
  // call, or one decode step's S=1 call), so this is the exact right
  // place to advance the token-step counter record_hit_if_recently_
  // fetched uses to distinguish a same-token race from a much later,
  // unrelated token rediscovering an earlier token's stale fetch record.
  advance_token_step();

  // Full graph paths submit the complete trunk in one GGML scheduler call.
  // Families with the full-graph-only policy turn build/run failure into a
  // request error below; other families retain their explicitly supported
  // layered path during migration.
  engine.qwen4exp_full_graph_output_ready = false;
  engine.qwen4exp_full_graph_logits_ready = false;
  engine.glm5next_full_graph_output_ready = false;
  engine.deepseek_v4_full_graph_output_ready = false;
  const FullGraphModel *graph_model = engine.family_ops.graph_model;
  engine.full_graph_output_ready = false;
  if (graph_model && graph_model->uses_native_moe_cache()) {
    if (!g_layer_scratch.resize(S, D, c.n_experts, c.moe_inter, c.topk,
                                c.n_expert_shared, c.dense_inter)) {
      engine.qwen4exp_full_graph_failure = "native graph runtime scratch allocation failed";
      return;
    }
    graph_model->bind_runtime(engine, ram_mgr, async_io_queue, g_layer_scratch);
  }
  if (S > 1 && pos_base == 0 &&
      engine.family_ops.family == ModelFamily::DeepSeek_V41 &&
      graph_model && graph_model->enabled(engine, ModelGraphPhase::Prefill)) {
    // Prefill and decode graphs each need a dense arena. They run in separate
    // phases, so release the previous decode graph before building prefill;
    // the DDR/VRAM MoE caches remain owned by the engine and are untouched.
    deepseek_v41_full_graph_decode_shutdown();
    if (!g_layer_scratch.resize(S, D, c.n_experts, c.moe_inter, c.topk,
                                c.n_expert_shared, c.dense_inter))
      return;
    deepseek_v41_bind_graph_runtime(engine, ram_mgr, async_io_queue,
                                    g_layer_scratch);
  }
  if (S == 1 && engine.family_ops.family == ModelFamily::DeepSeek_V41 &&
      deepseek_v41_full_graph_decode_enabled()) {
    if (!g_layer_scratch.resize(1, D, c.n_experts, c.moe_inter, c.topk,
                                c.n_expert_shared, c.dense_inter))
      return;
    deepseek_v41_bind_graph_runtime(engine, ram_mgr, async_io_queue,
                                    g_layer_scratch);
  }
  if (engine.family_ops.family == ModelFamily::BailingMoE3 &&
      bailingmoe3_native_graph_enabled()) {
    if (!g_layer_scratch.resize(S, D, c.n_experts, c.moe_inter, c.topk,
                                c.n_expert_shared, c.dense_inter))
      return;
    bailingmoe3_native_graph_bind_runtime(engine, ram_mgr, async_io_queue,
                                          g_layer_scratch);
  }
  const ModelGraphRequest prefill_graph_request{
      ModelGraphPhase::Prefill, S, pos_base, token_ids, hidden_states};
  const bool require_full_graph_prefill =
      engine.qwen4exp_full_graph_required ||
      (engine.cuda_moe_full_graph_decode_required && graph_model &&
       graph_model->requires_full_graph_prefill(engine));
  if (S > 1 && require_full_graph_prefill) {
    auto fail_graph = [&engine](const char *reason) {
      engine.qwen4exp_full_graph_failure = reason;
      LOG_ERROR("[MOE-GRAPH-POLICY] %s; layered execution is disabled\n",
                reason);
    };
    if ((pos_base != 0 &&
         (!graph_model ||
          !graph_model->supports_prefill_continuation(engine))) ||
        !graph_model ||
        !graph_model->enabled(engine, ModelGraphPhase::Prefill)) {
      fail_graph("required full-graph prefill adapter is unavailable for this request shape");
      return;
    }
    if (!graph_model->build(engine, prefill_graph_request)) {
      fail_graph("required full-graph prefill construction failed");
      return;
    }
    qwen4exp_graph_mtp_prefill_begin(engine);
    if (!graph_model->run(engine, prefill_graph_request)) {
      fail_graph("required full-graph prefill execution failed");
      return;
    }
    ++engine.full_graph_prefill_runs;
    return;
  }
  if (S > 1 && graph_model &&
      (pos_base == 0 || graph_model->supports_prefill_continuation(engine)) &&
      graph_model->enabled(engine, ModelGraphPhase::Prefill) &&
      graph_model->build(engine, prefill_graph_request) &&
      graph_model->run(engine, prefill_graph_request)) {
    ++engine.full_graph_prefill_runs;
    return;
  }
  if (!engine.qwen4exp_full_graph_failure.empty())
    return;
  const ModelGraphRequest decode_graph_request{
      ModelGraphPhase::Decode,
      S,
      pos_base,
      token_ids,
      hidden_states,
      engine.qwen4exp_full_graph_logits_target,
      engine.qwen4exp_full_graph_logits_target_size};
  const bool require_decode_graph =
      engine.cuda_moe_full_graph_decode_required ||
      (engine.qwen4exp_full_graph_required &&
       engine.cuda_moe_layered_fallback_reason.empty());
  if (S == 1 && require_decode_graph) {
    auto fail_graph = [&engine](const char *reason) {
      engine.qwen4exp_full_graph_failure = reason;
      LOG_ERROR("[MOE-GRAPH-POLICY] %s; layered execution is disabled\n",
                reason);
    };
    if (S == 1 && pos_base >= 0) {
      if (!graph_model ||
          !graph_model->enabled(engine, ModelGraphPhase::Decode)) {
        fail_graph("required full-graph decode adapter is unavailable");
        return;
      }
      if (!graph_model->build(engine, decode_graph_request)) {
        fail_graph("required full-graph decode construction failed");
        return;
      }
      const auto graph_run_started = std::chrono::steady_clock::now();
      if (!graph_model->run(engine, decode_graph_request)) {
        fail_graph("required full-graph decode execution failed");
        return;
      }
      engine.full_graph_decode_run_ns += static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - graph_run_started)
              .count());
      ++engine.full_graph_decode_runs;
      return;
    }
    fail_graph("required full graph received an unsupported token shape");
    return;
  }
  if (S == 1 && pos_base >= 0 &&
      engine.cuda_moe_layered_fallback_reason.empty() && graph_model &&
      graph_model->enabled(engine, ModelGraphPhase::Decode) &&
      graph_model->build(engine, decode_graph_request)) {
    const auto graph_run_started = std::chrono::steady_clock::now();
    if (graph_model->run(engine, decode_graph_request)) {
      engine.full_graph_decode_run_ns += static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - graph_run_started)
              .count());
      ++engine.full_graph_decode_runs;
      return;
    }
  }

  if (graph_model && graph_model->graph_only()) {
    if (engine.qwen4exp_full_graph_failure.empty())
      engine.qwen4exp_full_graph_failure =
          std::string(graph_model->name()) + " graph build/run failed; layered fallback is unsupported";
    return;
  }
  if (S == 1 && c.n_experts > 0) {
    if (engine.cuda_moe_full_graph_decode_required) {
      engine.qwen4exp_full_graph_failure =
          "required CUDA MoE graph reached the layered execution path";
      LOG_ERROR("[MOE-GRAPH-POLICY] %s; aborting request\n",
                engine.qwen4exp_full_graph_failure.c_str());
      return;
    }
    if (engine.cuda_moe_graph_policy_applies &&
        !engine.cuda_moe_layered_fallback_reason.empty()) {
      ++engine.layered_moe_s1_forwards;
    }
  }
  if (S == 1 && engine.family_ops.family == ModelFamily::Qwen4Exp &&
      !qwen4exp_full_graph_sync_state_to_host(engine)) {
    LOG_STREAM_ERROR() << "[GGML-QWEN-FULL] cannot materialize device state "
                         "for layered fallback\n";
    return;
  }

  LayerScratch &scratch = g_layer_scratch;
  if (!scratch.resize(S, D, c.n_experts, c.moe_inter, c.topk, c.n_expert_shared,
                      c.dense_inter)) {
    LOG_STREAM_ERROR() << "[MEM-GUARD] generation scratch allocation denied; "
                          "aborting this forward pass safely\n";
    return;
  }

  // Diagnostic: L2 norm of the LAST TOKEN's hidden state
  // the last few layers while starting within ~2x at layer 0, meaning the
  // excess accumulates gradually across many layers rather than in one bad
  // layer -- full resolution is needed to see exactly where it accelerates.
  auto l2norm = [](const float *p, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i)
      s += (double)p[i] * p[i];
    return std::sqrt(s);
  };
  int n = S * D; // hoisted above the loop -- was declared after its first use
                 // at "[L0] in" below
  bool trace = trace_gate("TRACE-HIDDEN"); // gate BEFORE paying for l2norm, not
                                           // just before printing
  for (int l = 0; l < c.n_layers; ++l) {
    // TEMPORARY DIAGNOSTIC (2026-09-12) -- see g_diag_layer_loop_total_ms's
    // own comment above. Wraps this ENTIRE iteration (attention, hc hooks,
    // FFN/MoE dispatch, and every pilot/prefetch block below, timed or not)
    // so its accumulated total across all layers can be compared directly
    // against the [REQUEST] line's own wall_ms and against the sum of the
    // named PERF_SCOPE buckets that already feed that line.
    auto diag_layer_loop_start = std::chrono::high_resolution_clock::now();
    if (ram_mgr.vram_cache)
      ram_mgr.vram_cache->prefill_current_layer = l;
    const float *last_tok =
        hidden_states +
        (size_t)(S - 1) * D; // hidden_states is mutated in place below, not
                             // reallocated -- always valid
    if (S == 1)
      dump_prefill_boundary("input", l, pos_base, last_tok, D);
    if (engine.family_ops.family == ModelFamily::DeepSeek_V41)
      dsv41_trace::save("input", l, pos_base, last_tok, D);
    if (trace)
      LOG_STREAM_VERBOSE() << "[TRACE l=" << l
                           << "] in      |hidden|=" << l2norm(last_tok, D)
                           << "\n";
    // --- Attention: dispatched through engine.family_ops.attention_fn
    // (models/family_ops.hpp) -- MLA absorbed (GLM-5.2/DeepSeek-V3), MLA
    // raw/SWA (DeepSeek-V4), or dense GQA causal (Hunyuan/Hy-V3/Llama),
    // depending on which family this file resolved to at load time. The
    // hc_pre_fn/hc_post_fn bracket is the hyper-connection pre-collapse/
    // post-combine hook (DeepSeek-V4); for every other family it's the
    // shared no-op / plain residual-add (hc_pre_noop/hc_post_plain_add,
    // family_ops.cpp) -- bit-for-bit the same two operations this loop
    // used to inline directly, just reachable through one function
    // pointer instead of a bool check with no third option. ---
    auto &fo = engine.family_ops;
    // Optional per-layer input hook (FamilyOps::layer_input_fn) -- Qwen4Exp's
    // PLE n-gram hash embedding modifies the HC stream buffer BEFORE the
    // attention hc_pre collapse. nullptr for every other family.
    if (fo.layer_input_fn) {
      const auto diag_layer_input_start =
          std::chrono::high_resolution_clock::now();
      PERF_SCOPE(t_layer_input_hook);
      qwen4exp_hc_prepare_host_layer_input(engine, l, S);
      fo.layer_input_fn(engine, l, S, pos_base, token_ids);
      qwen4exp_hc_commit_host_layer_input(engine, l, S);
      if (engine.family_ops.family == ModelFamily::DeepSeek_V41)
        dsv41_trace::save(
            "post-input-hook", l, pos_base,
            engine.hc_stream_buf.data() + (size_t)(S - 1) * c.hc_mult * D,
            D);
      if (engine.family_ops.family == ModelFamily::DeepSeek_V41 && S > 1)
        dsv41_trace::save("post-input-hook-seq", l, pos_base,
                          engine.hc_stream_buf.data(),
                          (size_t)S * c.hc_mult * D);
      // A/A diagnostic: after a sparse Qwen PLE layer has modified the
      // host HC copy and committed it back to the device-authoritative
      // runner, capture the actual wide state before hc_pre.  This is the
      // exact boundary that must equal llama.cpp's previous l_last plus its PLE
      // gated-value and causal-convolution outputs.  The helper is fully
      // gated by ZEROMOE_PREFILL_BOUNDARY_DUMP and is a no-op normally.
      dump_prefill_hc_streams("hc_streams_post_layer_input", l, pos_base, S,
                              c.hc_mult, D);
      if (S == 1)
        g_diag_decode_layer_input_ms +=
            std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() -
                diag_layer_input_start)
                .count();
    }
    {
      const auto diag_hc_start = std::chrono::high_resolution_clock::now();
      PERF_SCOPE(t_hc_hooks);
      fo.hc_pre_fn(engine, l, /*sub_block=*/0, hidden_states, S, D);
      if (S == 1) {
        const double elapsed = std::chrono::duration<double, std::milli>(
                                   std::chrono::high_resolution_clock::now() -
                                   diag_hc_start)
                                   .count();
        g_diag_decode_hc_ms +=
            elapsed;
        g_diag_decode_hc_pre_ms += elapsed;
        g_diag_decode_hc_attn_pre_ms += elapsed;
      }
    }
    if (S == 1)
      dump_prefill_device_boundary("hc_attn_pre", l, pos_base,
                                   engine.qwen4exp_device_hc_input, D);
    // TEMPORARY DIAGNOSTIC (2026-09-12) -- wall-clock around this exact call,
    // to compare against t_attn's OWN reported duration for the same call.
    // If attention_fn issues async GPU work and returns before it lands
    // (with the real wait happening implicitly at a later blocking call
    // this loop doesn't separately time), t_attn would under-report this
    // call's real cost relative to this wall-clock measurement. See
    // g_diag_attn_call_ms's own comment above the function.
    auto diag_attn_call_start = std::chrono::high_resolution_clock::now();
    fo.attention_fn(engine, l, hidden_states, S, pos_base,
                    scratch.attn_out.data());
    if (engine.family_ops.family == ModelFamily::DeepSeek_V41)
      dsv41_trace::save("attn-out", l, pos_base,
                        scratch.attn_out.data() + (size_t)(S - 1) * D, D);
    g_diag_attn_call_ms += std::chrono::duration<double, std::milli>(
                               std::chrono::high_resolution_clock::now() -
                               diag_attn_call_start)
                               .count();
    if (S == 1)
      g_diag_decode_attn_call_ms +=
          std::chrono::duration<double, std::milli>(
              std::chrono::high_resolution_clock::now() -
              diag_attn_call_start)
              .count();
    if (engine.qwen4exp_device_attn_output_ready)
      dump_prefill_attention_output(l, pos_base, S,
                                    engine.qwen4exp_device_attn_output, D);
    if (!engine.qwen4exp_device_attn_output_ready) {
      for (int i = 0; i < S; ++i)
        dump_prefill_boundary("attn_out", l, pos_base + i,
                              scratch.attn_out.data() + (size_t)i * D, D);
    }
    bool attn_device_consumed = false;
    if (engine.qwen4exp_device_attn_output_ready) {
      {
        attn_device_consumed =
            qwen4exp_hc_post_combine_device(engine, l, /*sub_block=*/0, S, D,
                                            engine.qwen4exp_device_attn_output,
                                            nullptr);
        if (!attn_device_consumed && !qwen4exp_copy_gdn_device_output_to_host(
                                         engine.qwen4exp_device_attn_output,
                                         scratch.attn_out.data(), S)) {
          LOG_ERROR("[GGML-GDN-FULL] failed to recover device attention output "
                    "at layer=%d\n",
                    l);
        }
      }
      engine.qwen4exp_device_attn_output = nullptr;
      engine.qwen4exp_device_attn_output_ready = false;
      engine.qwen4exp_device_attn_output_raw = false;
    }
    {
      const auto diag_hc_start = std::chrono::high_resolution_clock::now();
      PERF_SCOPE(t_hc_hooks);
      if (attn_device_consumed) {
        // The device post-combine already updated the authoritative HC state.
      } else {
        fo.hc_post_fn(engine, l, /*sub_block=*/0, hidden_states,
                      scratch.attn_out.data(), S, D);
      }
      if (S == 1)
      {
        const double elapsed = std::chrono::duration<double, std::milli>(
                                   std::chrono::high_resolution_clock::now() -
                                   diag_hc_start)
                                   .count();
        g_diag_decode_hc_ms += elapsed;
        g_diag_decode_hc_post_ms += elapsed;
      }
    }
    dump_prefill_hc_streams("hc_streams_post_attn", l, pos_base, S,
                            c.hc_mult, D);

    // Per-layer, per-stage checkpoint: pinpoints the FIRST layer/stage
    // where hidden_states goes non-finite, instead of only learning
    // "somewhere in this forward pass" from the aggregate
    // after-prefill/after-decode-step checks in generation_loop.hpp.
    // check_finite() is a no-op (silent) unless it actually finds
    // NaN/Inf, so this is cheap on the (expected) all-finite path.
    check_finite("hidden_states (post attention residual)", l, -1,
                 hidden_states, (size_t)n);
    for (int i = 0; i < S; ++i)
      dump_prefill_boundary("post_attn", l, pos_base + i,
                            hidden_states + (size_t)i * D, D);
    if (engine.family_ops.family == ModelFamily::DeepSeek_V41)
      dsv41_trace::save("post-attn", l, pos_base,
                        hidden_states + (size_t)(S - 1) * D, D);
    if (engine.family_ops.family == ModelFamily::DeepSeek_V41 &&
        engine.hc_stream_buf.size() >= (size_t)S * c.hc_mult * D)
      dsv41_trace::save(
          "post-attn-streams", l, pos_base,
          engine.hc_stream_buf.data() + (size_t)(S - 1) * c.hc_mult * D,
          (size_t)c.hc_mult * D);
    if (trace)
      LOG_STREAM_VERBOSE() << "[TRACE l=" << l << "] post-attn |attn_out|="
                           << l2norm(scratch.attn_out.data() +
                                         (size_t)(S - 1) * D,
                                     D)
                           << " |hidden|=" << l2norm(last_tok, D) << "\n";

    // A device pilot queued after layer l-1 is consumed here, after layer
    // l's attention had a chance to run and immediately before l's MoE
    // bookkeeping. This is the only point where the host needs the compact
    // IDs to make a prefetch decision; delaying the readback keeps the
    // layer-boundary CPU stall out of the critical path.
    if (S == 1 && scratch.qwen_pilot_gpu_pending &&
        scratch.qwen_pilot_gpu_pending_layer == l) {
      auto &pilot_router = GgmlRouterRunner::instance();
      const bool route_ok = pilot_router.copy_route_to_host(
          l, scratch.qwen_pilot_gpu_ids.data(),
          scratch.qwen_pilot_gpu_weights.data());
      if (route_ok) {
        const int qwen_k = std::min({engine.qwen4exp_router_pilot_k, c.topk,
                                     LayerScratch::kQwenPilotMax});
        for (int k = 0; k < qwen_k; ++k)
          scratch.qwen_pilot_pred_ids[k] = scratch.qwen_pilot_gpu_ids[k];
        moe_pipeline_detail::qwen4exp_process_pilot_predictions(
            engine, ram_mgr, async_io_queue, scratch, l, qwen_k);
        perf().n_qwen_pilot_gpu_predictions.fetch_add(
            1, std::memory_order_relaxed);
      } else {
        perf().n_qwen_pilot_sync_fail.fetch_add(1, std::memory_order_relaxed);
      }
      scratch.qwen_pilot_gpu_pending = false;
      scratch.qwen_pilot_gpu_pending_layer = -1;
    }

    // --- ATTN-PILOT measurement (2026-08-25, MEASUREMENT ONLY -- not
    // wired to async_io_queue.push_hint/publish_pilot_confirmed_
    // resident, same discipline DEPTH-2 originally shipped under):
    // predicts layer l+1's routing from THIS point -- right after l's
    // own attention residual, BEFORE l's own FFN runs -- rather than
    // the existing pilot block below, which waits for l's FFN to fully
    // finish first. One new signal answers two different questions,
    // since "predict layer T from layer T-1's post-attention state" is
    // a single well-defined measurement regardless of which existing
    // baseline it's read against:
    //   - vs DEPTH-1 below (predicts l+1 from l's COMPLETE, post-FFN
    //     state): firing this early loses l's own FFN contribution --
    //     is the extra lead time (this doesn't have to wait for l's
    //     FFN, "routinely the single largest per-layer compute cost in
    //     this model" per DEPTH-1's own comment below) worth what it
    //     costs in accuracy?
    //   - vs DEPTH-2 (predicts l+2 by naively reusing l's own stale
    //     pre-l+1-attention vector): when THIS SAME mechanism fires
    //     one iteration later (at l+1's own post-attention point,
    //     predicting l+2), it uses l+1's REAL attention output --
    //     strictly more current than DEPTH-2's naive reuse, at the
    //     cost of firing later (less lead time than DEPTH-2 gets,
    //     since it has to wait for l+1's real attention to actually
    //     run first).
    // Single-scalar survival (pilot_attn_pred_layer), not a 2-slot
    // ring like DEPTH-2's Pilot2Pending -- only one intervening layer
    // boundary between prediction (fired further below, AFTER l's own
    // FFN completes -- see the real ordering bug that requires this,
    // explained there) and consumption (top of l+1's own moe_forward_
    // gguf/dense_ffn_gguf call), the same single-iteration gap DEPTH-1's
    // own pilot_pred_layer already safely uses.
    //
    // The actual dense_proj/gating prediction is DEFERRED until after
    // l's own FFN block below -- not fired here, despite this being
    // the whole point (predict from PRE-FFN state) -- because l's own
    // moe_forward_gguf call (which happens next, in the FFN block)
    // is what CONSUMES whatever prediction l-1 made for layer l. Firing
    // the l+1 prediction here, before that consumption runs, would
    // overwrite pilot_attn_pred_layer before it's ever read -- a real
    // bug caught by this mechanism's own first real-hardware run: only
    // ~41 checks total (one per TOKEN, not one per layer -- only the
    // very last layer transition survived, since there's no l+1 firing
    // after the last layer to clobber it first). Only the CHEAP part
    // (snapshotting the pre-FFN input this prediction will need) can
    // safely happen here; hidden_states itself will have moved on to
    // POST-FFN by the time the prediction actually runs.
    std::memcpy(scratch.pilot_attn_snapshot.data(), last_tok,
                (size_t)D * sizeof(float));

    // --- FFN: leading dense layers (l < first_dense) use a plain dense
    // FFN — no router, no expert cache. Applies to both GLM-5.2's leading
    // dense layers and, since Hunyuan/Hy-V3 share the same first_dense
    // convention, to any dense layers there too. dense_ffn_gguf/
    // moe_forward_gguf are unchanged by the FamilyOps refactor -- they
    // still read/write plain [S,D] hidden_states/scratch.moe_out
    // regardless of family; only the hc_pre_fn/hc_post_fn bracket around
    // them (sub_block=1, matching attention's sub_block=0 above) knows
    // about hyper-connections. ---
    {
      const auto diag_hc_start = std::chrono::high_resolution_clock::now();
      PERF_SCOPE(t_hc_hooks);
      fo.hc_pre_fn(engine, l, /*sub_block=*/1, hidden_states, S, D);
      if (S == 1) {
        const double elapsed = std::chrono::duration<double, std::milli>(
                                   std::chrono::high_resolution_clock::now() -
                                   diag_hc_start)
                                   .count();
        g_diag_decode_hc_ms +=
            elapsed;
        g_diag_decode_hc_pre_ms += elapsed;
        g_diag_decode_hc_ffn_pre_ms += elapsed;
      }
    }
    if (engine.family_ops.family == ModelFamily::DeepSeek_V41)
      dsv41_trace::save("hc-ffn-pre", l, pos_base,
                        hidden_states + (size_t)(S - 1) * D, D);
    if (S == 1)
      dump_prefill_device_boundary("hc_ffn_pre", l, pos_base,
                                   engine.qwen4exp_device_hc_ffn_input, D);
    // TEMPORARY DIAGNOSTIC (2026-09-12) -- wall-clock around this exact
    // block, same cross-check rationale as g_diag_attn_call_ms above (async
    // GPU dispatch inside dense_ffn_gguf/moe_forward_gguf could return
    // before the work is really done, making t_moe_expert under-report
    // vs. this wall-clock measurement of the same call).
    // ZEROMOE_PROBE_DUMP=<dir>: training data for a learned next-layer expert predictor (diagnostic only). For every
    // MoE layer l, append the post-attention residual rows x_l (S x D, f32) and layer l's REAL selection scores s_l
    // (S x n_experts, f32: gate(logits)+bias, what the router ranks) so x_l can be paired with s_{l+1} offline.
    // Re-take the ATTN-PILOT snapshot right before the FFN. The copy taken as soon as attention returns (above) precedes the point
    // where the post-attention residual is final in `hidden_states`, so it predicted from a stale vector: GLM-5.3-Flash ATTN-PILOT
    // recall was 37.2% (rank 0: 71.6%) with the early copy and 70.6% (rank 0: 98.9%) with this one.
    if (S >= 1 && !scratch.pilot_attn_snapshot.empty())
      std::memcpy(scratch.pilot_attn_snapshot.data(), hidden_states + (size_t)(S - 1) * D, (size_t)D * sizeof(float));
    std::vector<float> probe_last_scores; // layer l's score row of the last token (route-agreement check)
    if (static const char *probe_dir = zeromoe::graphcfg::env("ZEROMOE_PROBE_DUMP");
        probe_dir && probe_dir[0] && l >= c.first_dense && c.n_experts > 0 &&
        engine.family_ops.family != ModelFamily::Qwen4Exp) {
      const std::string norm_key = "blk." + std::to_string(l) + ".ffn_norm.weight";
      const std::string router_key = "blk." + std::to_string(l) + ".ffn_gate_inp.weight";
      const std::string bias_key = "blk." + std::to_string(l) + ".exp_probs_b.bias";
      auto n_it = engine.resident_layers.find(norm_key);
      auto r_it = engine.resident_layers.find(router_key);
      if (n_it != engine.resident_layers.end() && r_it != engine.resident_layers.end() &&
          n_it->second.host_ptr && r_it->second.host_ptr) {
        std::vector<float> nrm((size_t)S * D), sc((size_t)S * c.n_experts);
        for (int t = 0; t < S; ++t)
          rmsnorm_row(nrm.data() + (size_t)t * D, hidden_states + (size_t)t * D,
                      reinterpret_cast<float *>(n_it->second.host_ptr), D, c.eps);
        dense_proj(engine, router_key, nrm.data(), sc.data(), S, D, c.n_experts);
        for (int t = 0; t < S; ++t) {
          float *row = sc.data() + (size_t)t * c.n_experts;
          apply_gating_activation(c.gating_func, row, c.n_experts);
          auto b_it = engine.resident_layers.find(bias_key);
          if (b_it != engine.resident_layers.end() && b_it->second.host_ptr) {
            const float *bias = reinterpret_cast<float *>(b_it->second.host_ptr);
            for (int e = 0; e < c.n_experts; ++e) row[e] += bias[e];
          }
        }
        probe_last_scores.assign(sc.data() + (size_t)(S - 1) * c.n_experts, sc.data() + (size_t)S * c.n_experts);
        const std::string base = std::string(probe_dir) + "/L" + std::to_string(l);
        std::ofstream xo(base + ".x.bin", std::ios::binary | std::ios::app);
        xo.write(reinterpret_cast<const char *>(hidden_states), (std::streamsize)((size_t)S * D * sizeof(float)));
        std::ofstream so(base + ".s.bin", std::ios::binary | std::ios::app);
        so.write(reinterpret_cast<const char *>(sc.data()), (std::streamsize)(sc.size() * sizeof(float)));
      }
    }
    auto diag_ffn_call_start = std::chrono::high_resolution_clock::now();
    {
      PERF_SCOPE(t_moe_expert);
      if (l < c.first_dense) {
        dense_ffn_gguf(engine, l, hidden_states, S, scratch.moe_out.data(),
                       scratch);
      } else {
        moe_forward_gguf(engine, ram_mgr, async_io_queue, l, hidden_states, S,
                         scratch.moe_out.data(), scratch, token_ids);
      }
    }
    if (engine.family_ops.family == ModelFamily::DeepSeek_V41)
      dsv41_trace::save("moe-out", l, pos_base,
                        scratch.moe_out.data() + (size_t)(S - 1) * D, D);
    if (engine.family_ops.family == ModelFamily::DeepSeek_V41 && S > 1)
      dsv41_trace::save("moe-out-seq", l, pos_base, scratch.moe_out.data(),
                        (size_t)S * D);
    if (!probe_last_scores.empty() && S == 1 && (int)scratch.tok_topk_ids.size() >= c.topk) {
      static std::atomic<uint64_t> agree{0}, total{0};
      std::vector<int> idx((size_t)c.n_experts);
      for (int e = 0; e < c.n_experts; ++e) idx[(size_t)e] = e;
      std::partial_sort(idx.begin(), idx.begin() + c.topk, idx.end(),
                        [&](int a, int b) { return probe_last_scores[(size_t)a] > probe_last_scores[(size_t)b]; });
      for (int k = 0; k < c.topk; ++k)
        for (int j = 0; j < c.topk; ++j)
          if (idx[(size_t)k] == scratch.tok_topk_ids[(size_t)j]) { ++agree; break; }
      total += (uint64_t)c.topk;
      if (l == c.n_layers - 1)
        LOG_INFO("[PROBE-ROUTE-CHECK] hook top-%d vs engine routed ids: %llu/%llu agree\n", c.topk,
                 (unsigned long long)agree.load(), (unsigned long long)total.load());
    }
    g_diag_ffn_call_ms += std::chrono::duration<double, std::milli>(
                              std::chrono::high_resolution_clock::now() -
                              diag_ffn_call_start)
                              .count();
    if (S == 1)
      g_diag_decode_ffn_call_ms +=
          std::chrono::duration<double, std::milli>(
              std::chrono::high_resolution_clock::now() -
              diag_ffn_call_start)
              .count();
    // A/A prefill diagnostic: capture the plain routed FFN result before HC
      // post-combine.  llama.cpp's qwen4exp graph exposes the corresponding
    // ffn_moe_out tensor, so this separates CPU-MoE math from the subsequent
    // HC reduction.  Opt-in through ZEROMOE_PREFILL_BOUNDARY_DUMP; the
    // normal path does no dump or extra device copy.
    if (S == 1)
      dump_prefill_boundary("moe_out_host", l, pos_base,
                            scratch.moe_out.data(), D);
    if (S == 1) {
      dump_prefill_boundary("route_weights", l, pos_base,
                            scratch.tok_topk_weights.data(), c.topk);
      dump_prefill_boundary_i32("route_ids", l, pos_base,
                                scratch.tok_topk_ids.data(), c.topk);
    }
    bool moe_device_consumed = false;
  if (scratch.moe_device_output_ready) {
      // The actual residual update is device-to-device.  Do not read the
      // routed/shared result back to the host on the normal path: the old
      // unconditional copy paid one synchronizing D2H per device MoE layer
      // even though qwen4exp_hc_post_combine_device() consumed the same
      // device tensor immediately afterwards.  Keep the readback only for
      // explicit hidden-state tracing, or as the recovery path when the
      // device HC post-combine cannot accept the producer tensor.
      bool copied_device_output = false;
      // A boundary dump is itself an explicit request for the device result.
      // Do not require TRACE-HIDDEN as well: otherwise the pre-combine dump
      // below observes the stale host scratch.moe_out while the real Qwen
      // scheduler result is still device-resident.
      const bool boundary_requested = prefill_boundary_dump_base() != nullptr;
      if (trace || boundary_requested) {
        if (scratch.moe_shared_device_output)
          copied_device_output = GgmlQwenSharedRunner::instance_for_layer(l)
                                     .copy_output_to_host(
                                         scratch.moe_device_output,
                                         scratch.moe_out.data());
        else if (scratch.moe_device_runner)
          copied_device_output = scratch.moe_device_runner
                                     ->copy_device_output_to_host(
                                         scratch.moe_device_output,
                                         scratch.moe_out.data());
        if (!copied_device_output) {
          LOG_ERROR("[GGML-MOE] Qwen4Exp device output trace copy failed at "
                    "layer=%d\n",
                    l);
        }
      }
      if (S == 1 && boundary_requested && copied_device_output)
        dump_prefill_boundary("moe_out_device", l, pos_base,
                              scratch.moe_out.data(), D);
      moe_device_consumed = qwen4exp_hc_post_combine_device(
          engine, l, /*sub_block=*/1, S, D, scratch.moe_device_output,
          scratch.moe_shared_device_output
              ? GgmlQwenSharedRunner::instance_for_layer(l).output_done_event()
              : nullptr);
      if (!moe_device_consumed) {
        if (!copied_device_output && scratch.moe_shared_device_output)
          copied_device_output = GgmlQwenSharedRunner::instance_for_layer(l)
                                     .copy_output_to_host(
                                         scratch.moe_device_output,
                                         scratch.moe_out.data());
        else if (!copied_device_output && scratch.moe_device_runner)
          copied_device_output =
              scratch.moe_device_runner->copy_device_output_to_host(
                  scratch.moe_device_output, scratch.moe_out.data());
        if (!copied_device_output)
          LOG_ERROR("[GGML-MOE] Qwen4Exp device output fallback copy failed at "
                    "layer=%d\n",
                    l);
      }
      scratch.moe_device_output = nullptr;
      scratch.moe_device_runner = nullptr;
      scratch.moe_shared_device_output = false;
      scratch.moe_device_output_ready = false;
    }
    if (!moe_device_consumed) {
      // Missing timer found investigating a large gap between [REQUEST]'s
      // own wall_ms and the sum of its named buckets: every OTHER hc_pre_fn/
      // hc_post_fn call site in this function (sub_block=0 x2, sub_block=1's
      // own hc_pre_fn) is wrapped in PERF_SCOPE(t_hc_hooks) -- this one, the
      // FFN-side post-combine, was not. Same counter as its siblings so the
      // existing "Per-layer hooks:" report line picks it up automatically.
      const auto diag_hc_start = std::chrono::high_resolution_clock::now();
      {
        PERF_SCOPE(t_hc_hooks);
        fo.hc_post_fn(engine, l, /*sub_block=*/1, hidden_states,
                      scratch.moe_out.data(), S, D);
      }
      if (S == 1)
      {
        const double elapsed = std::chrono::duration<double, std::milli>(
                                   std::chrono::high_resolution_clock::now() -
                                   diag_hc_start)
                                   .count();
        g_diag_decode_hc_ms += elapsed;
        g_diag_decode_hc_post_ms += elapsed;
      }
    }
    if (engine.family_ops.family == ModelFamily::DeepSeek_V41) {
      dsv41_trace::save("post-ffn", l, pos_base,
                        hidden_states + (size_t)(S - 1) * D, D);
      if (engine.hc_stream_buf.size() >= (size_t)S * c.hc_mult * D)
        dsv41_trace::save(
            "post-ffn-streams", l, pos_base,
            engine.hc_stream_buf.data() + (size_t)(S - 1) * c.hc_mult * D,
            (size_t)c.hc_mult * D);
      if (dsv41_trace::enabled(l, pos_base) &&
          (int)scratch.tok_topk_ids.size() >= c.topk &&
          (int)scratch.tok_topk_weights.size() >= c.topk) {
        std::vector<float> ids((size_t)c.topk);
        for (int k = 0; k < c.topk; ++k)
          ids[(size_t)k] =
              static_cast<float>(scratch.tok_topk_ids[
                  (size_t)(S - 1) * c.topk + k]);
        dsv41_trace::save("route-ids", l, pos_base, ids.data(), ids.size());
        dsv41_trace::save("route-weights", l, pos_base,
                          scratch.tok_topk_weights.data() +
                              (size_t)(S - 1) * c.topk,
                          c.topk);
      }
      if (S > 1) {
        std::vector<float> ids((size_t)S * c.topk);
        for (size_t i = 0; i < ids.size(); ++i)
          ids[i] = static_cast<float>(scratch.tok_topk_ids[i]);
        dsv41_trace::save("route-ids-seq", l, pos_base, ids.data(),
                          ids.size());
        dsv41_trace::save("route-weights-seq", l, pos_base,
                          scratch.tok_topk_weights.data(), ids.size());
      }
    }
    dump_prefill_hc_streams("hc_streams_post_ffn", l, pos_base, S,
                            c.hc_mult, D);

    check_finite("hidden_states (post FFN/MoE residual)", l, -1, hidden_states,
                 (size_t)n);
    for (int i = 0; i < S; ++i)
      dump_prefill_boundary("post_ffn", l, pos_base + i,
                            hidden_states + (size_t)i * D, D);
    if (trace)
      LOG_STREAM_VERBOSE() << "[TRACE l=" << l << "] post-ffn  |moe_out|="
                           << l2norm(scratch.moe_out.data() +
                                         (size_t)(S - 1) * D,
                                     D)
                           << " |hidden|=" << l2norm(last_tok, D) << "\n";

    // --- QWEN-PILOT (Qwen4Exp only) -------------------------------------
    // Qwen's production router consumes the HC graph's mixed activation, not
    // the conventional hidden->RMSNorm->gate input used by the generic pilot
    // below. After a device HC post-combine, engine.hc_stream_buf is stale;
    // take one explicit snapshot before projecting the next layer. The
    // stream-0 row is an intentionally conservative proxy for the mixed
    // activation and is labelled as such in the shutdown report. It never
    // changes routing or expert dispatch. Optional Qwen prefetch is kept
    // separate from the generic GLM/DeepSeek pilot: its top-K candidates are
    // either issued into the next layer's temporary active-layer area via
    // cudaMemcpyAsync (when already RAM-resident), or sent to the ordinary
    // disk/RAM hint queue (when they still need a RAM fetch).
    const bool qwen_pilot_skip_fully_cached =
        engine.qwen4exp_router_pilot_skip_fully_cached && S == 1 &&
        engine.family_ops.family == ModelFamily::Qwen4Exp &&
        ram_mgr.vram_cache && l + 1 < c.n_layers &&
        ram_mgr.vram_cache->layer_fully_resident(l + 1, c.n_experts);
    const bool qwen_pilot_skip_hot_set =
        engine.qwen4exp_router_pilot_skip_fully_cached && S == 1 &&
        engine.family_ops.family == ModelFamily::Qwen4Exp &&
        ram_mgr.vram_cache && l + 1 < c.n_layers &&
        moe_pipeline_detail::qwen4exp_router_pilot_layer_allowed(engine,
                                                                 l + 1) &&
        moe_pipeline_detail::qwen4exp_decode_hot_set_stable(l + 1);
    const bool qwen_pilot_skip_any_resident =
        qwen_pilot_skip_fully_cached || qwen_pilot_skip_hot_set;
    const bool qwen_pilot_skip_prefetch_excluded =
        engine.qwen4exp_router_pilot_skip_prefetch_excluded_layers && S == 1 &&
        engine.family_ops.family == ModelFamily::Qwen4Exp &&
        l + 1 < c.n_layers &&
        std::find(engine.qwen4exp_router_pilot_prefetch_exclude_layers.begin(),
                  engine.qwen4exp_router_pilot_prefetch_exclude_layers.end(),
                  l + 1) !=
            engine.qwen4exp_router_pilot_prefetch_exclude_layers.end();
    const bool qwen_pilot_skip_predict_policy =
        engine.family_ops.family == ModelFamily::Qwen4Exp &&
        l + 1 < c.n_layers &&
        !moe_pipeline_detail::qwen4exp_router_pilot_layer_allowed(engine,
                                                                  l + 1);
    if (qwen_pilot_skip_any_resident) {
      perf().n_qwen_pilot_skipped_fully_cached.fetch_add(
          1, std::memory_order_relaxed);
    }
    if (qwen_pilot_skip_hot_set) {
      perf().n_qwen_pilot_adaptive_skips.fetch_add(1,
                                                   std::memory_order_relaxed);
    }
    if (qwen_pilot_skip_prefetch_excluded) {
      perf().n_qwen_pilot_skipped_prefetch_excluded.fetch_add(
          1, std::memory_order_relaxed);
    }
    if (qwen_pilot_skip_predict_policy) {
      perf().n_qwen_pilot_skipped_predict_policy.fetch_add(
          1, std::memory_order_relaxed);
    }
    if (engine.qwen4exp_router_pilot && !qwen_pilot_skip_any_resident &&
        !qwen_pilot_skip_prefetch_excluded && !qwen_pilot_skip_predict_policy &&
        engine.family_ops.family == ModelFamily::Qwen4Exp &&
        l >= c.first_dense && l + 1 < c.n_layers && l + 1 >= c.first_dense &&
        c.n_experts > 0 && c.hc_mult > 0) {
      scratch.qwen_pilot_pred_count = 0;
      scratch.qwen_pilot_pred_layer = -1;
      scratch.qwen_pilot_prefetch_count = 0;
      const int target_layer = l + 1;
      const int qwen_k = std::min({engine.qwen4exp_router_pilot_k, c.topk,
                                   LayerScratch::kQwenPilotMax});
      const void *gpu_pilot_ids = nullptr;
      const void *gpu_pilot_weights = nullptr;
      const bool gpu_pilot_ok =
          S == 1 && qwen_k > 0 &&
          moe_pipeline_detail::qwen4exp_router_pilot_gpu_async(
              engine, target_layer, D, c.n_experts, c.topk, &gpu_pilot_ids,
              &gpu_pilot_weights);
      (void)gpu_pilot_ids;
      (void)gpu_pilot_weights;
      if (gpu_pilot_ok) {
        scratch.qwen_pilot_gpu_pending = true;
        scratch.qwen_pilot_gpu_pending_layer = target_layer;
      } else {
        perf().n_qwen_pilot_host_fallbacks.fetch_add(1,
                                                     std::memory_order_relaxed);
      }
      if (!gpu_pilot_ok) {
        const size_t hc_need = (size_t)S * c.hc_mult * D;
        const bool host_snapshot_ok = engine.hc_stream_buf.size() >= hc_need &&
                                      qwen4exp_hc_download_streams(engine, S);
        if (!host_snapshot_ok) {
          perf().n_qwen_pilot_sync_fail.fetch_add(1, std::memory_order_relaxed);
        } else {
          const std::string qwen_router_key =
              "blk." + std::to_string(target_layer) + ".ffn_gate_inp.weight";
          auto qwen_router_it = engine.resident_layers.find(qwen_router_key);
          if (gpu_pilot_ok || (qwen_router_it != engine.resident_layers.end() &&
                               qwen_router_it->second.host_ptr)) {
            if (!gpu_pilot_ok) {
              const float *proxy =
                  engine.hc_stream_buf.data() + (size_t)(S - 1) * c.hc_mult * D;
              dense_proj(engine, qwen_router_key, proxy,
                         scratch.qwen_pilot_logits.data(), 1, D, c.n_experts);
              apply_gating_activation(
                  c.gating_func, scratch.qwen_pilot_logits.data(), c.n_experts);
              const std::string qwen_bias_key =
                  "blk." + std::to_string(target_layer) + ".exp_probs_b.bias";
              auto qwen_bias_it = engine.resident_layers.find(qwen_bias_key);
              if (qwen_bias_it != engine.resident_layers.end() &&
                  qwen_bias_it->second.host_ptr &&
                  qwen_bias_it->second.meta.type == GGUFType::F32) {
                const float *bias = reinterpret_cast<const float *>(
                    qwen_bias_it->second.host_ptr);
                for (int e = 0; e < c.n_experts; ++e)
                  scratch.qwen_pilot_logits[e] += bias[e];
              }
            }
            const int qwen_prefetch_k =
                std::min({engine.qwen4exp_router_pilot_prefetch_k > 0
                              ? engine.qwen4exp_router_pilot_prefetch_k
                              : engine.qwen4exp_router_pilot_k,
                          qwen_k, LayerScratch::kQwenPilotMax});
            const bool qwen_prefetch_layer_enabled =
                std::find(
                    engine.qwen4exp_router_pilot_prefetch_exclude_layers
                        .begin(),
                    engine.qwen4exp_router_pilot_prefetch_exclude_layers.end(),
                    target_layer) ==
                engine.qwen4exp_router_pilot_prefetch_exclude_layers.end();
            const bool qwen_prefetch_enabled =
                engine.qwen4exp_router_pilot_prefetch &&
                qwen_prefetch_layer_enabled;
            perf().n_qwen_pilot_prefetch_k_observed.store(
                (uint32_t)(engine.qwen4exp_router_pilot_prefetch
                               ? qwen_prefetch_k
                               : 0),
                std::memory_order_relaxed);
            if (engine.qwen4exp_router_pilot_prefetch &&
                !qwen_prefetch_layer_enabled) {
              perf().n_qwen_pilot_prefetch_excluded.fetch_add(
                  1, std::memory_order_relaxed);
            } else if (engine.qwen4exp_router_pilot_prefetch &&
                       qwen_prefetch_k < qwen_k) {
              perf().n_qwen_pilot_prefetch_rank_skipped.fetch_add(
                  (uint64_t)(qwen_k - qwen_prefetch_k),
                  std::memory_order_relaxed);
            }
            scratch.qwen_pilot_pred_count = qwen_k;
            scratch.qwen_pilot_pred_layer = target_layer;
            scratch.qwen_pilot_prefetch_count =
                qwen_prefetch_enabled ? qwen_prefetch_k : 0;
            perf().n_qwen_pilot_predictions.fetch_add(
                1, std::memory_order_relaxed);
            const int64_t prediction_ready_ns =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch())
                    .count();
            if (target_layer < PerfCounters::kMaxTrackedLayersForMiss) {
              perf().qwen_pilot_prediction_ready_ns[target_layer].store(
                  prediction_ready_ns, std::memory_order_release);
            }

            for (int k = 0; k < qwen_k; ++k) {
              int best = gpu_pilot_ok ? scratch.qwen_pilot_gpu_ids[k] : 0;
              if (!gpu_pilot_ok) {
                for (int e = 1; e < c.n_experts; ++e) {
                  if (scratch.qwen_pilot_logits[e] >
                      scratch.qwen_pilot_logits[best])
                    best = e;
                }
                scratch.qwen_pilot_logits[best] = -2e30f;
              }
              scratch.qwen_pilot_pred_ids[k] = best;

              // Classify without a mutating lookup. The active-layer pool is a
              // real VRAM source but is separate from the main LRU cache.
              bool already_vram =
                  ram_mgr.vram_cache &&
                  (ram_mgr.vram_cache->active_layer_has(target_layer, best) ||
                   ram_mgr.vram_cache->main_layer_has(target_layer, best));
              const bool already_ram =
                  !already_vram && ram_mgr.is_ram_resident(target_layer, best);
              const bool qwen_prefetch_this_rank =
                  qwen_prefetch_enabled && k < qwen_prefetch_k;
              if (already_vram) {
                perf().n_qwen_pilot_pred_already_vram.fetch_add(
                    1, std::memory_order_relaxed);
              } else if (already_ram) {
                perf().n_qwen_pilot_pred_already_ram.fetch_add(
                    1, std::memory_order_relaxed);
                if (qwen_prefetch_this_rank) {
                  // promote_ram_to_active_layer issues cudaMemcpyAsync on the
                  // dedicated active-layer stream and records the destination
                  // event. It also protects the RAM source until that event,
                  // so the prefetch worker cannot reclaim the slab underneath
                  // the DMA.
                  ExpertLocation *issued = ram_mgr.promote_ram_to_active_layer(
                      engine, target_layer, best, -1, prediction_ready_ns);
                  if (issued) {
                    perf().n_qwen_pilot_active_prefetch_issued.fetch_add(
                        1, std::memory_order_relaxed);
                  } else {
                    perf().n_qwen_pilot_active_prefetch_failed.fetch_add(
                        1, std::memory_order_relaxed);
                  }
                }
              } else {
                perf().n_qwen_pilot_pred_needs_fetch.fetch_add(
                    1, std::memory_order_relaxed);
                if (qwen_prefetch_this_rank) {
                  async_io_queue.push_hint(target_layer, best);
                  perf().n_qwen_pilot_hints_pushed.fetch_add(
                      1, std::memory_order_relaxed);
                }
              }
              if (qwen_prefetch_this_rank && already_vram) {
                perf().n_qwen_pilot_hints_skipped_resident.fetch_add(
                    1, std::memory_order_relaxed);
              }
            }
          }
        }
      }
    }

    // EngineConfig::pilot_widen_all_tokens -- see that field's own
    // comment (engine_config.hpp) for the full motivation. Deliberately
    // self-contained (own norm/router/bias lookups, own scratch
    // buffers) rather than folded into DEPTH-1's own block below --
    // keeps this from touching DEPTH-1's real hint-push/promotion
    // logic at all, by construction, not just by care. The real hint-
    // push (prefetch_widen_all_tokens) lives in this same block, right
    // after the widened prediction is computed -- see that inner
    // comment for the residency-check/push logic itself.
    if (engine.pilot_widen_all_tokens && l + 1 < c.n_layers &&
        c.n_experts > 0 && S > 1) {
      std::string wnorm_key =
          "blk." + std::to_string(l + 1) + ".ffn_norm.weight";
      std::string wrouter_key =
          "blk." + std::to_string(l + 1) + ".ffn_gate_inp.weight";
      auto wnorm_it = engine.resident_layers.find(wnorm_key);
      auto wrouter_it = engine.resident_layers.find(wrouter_key);
      if (wnorm_it != engine.resident_layers.end() &&
          wrouter_it != engine.resident_layers.end() &&
          wnorm_it->second.host_ptr && wrouter_it->second.host_ptr) {
        const float *wnorm_w =
            reinterpret_cast<float *>(wnorm_it->second.host_ptr);
        std::string wbias_key =
            "blk." + std::to_string(l + 1) + ".exp_probs_b.bias";
        auto wbias_it = engine.resident_layers.find(wbias_key);
        const float *wbias_row =
            (wbias_it != engine.resident_layers.end() &&
             wbias_it->second.host_ptr)
                ? reinterpret_cast<float *>(wbias_it->second.host_ptr)
                : nullptr;

        static thread_local std::vector<bool> wide_seen;
        wide_seen.assign(c.n_experts, false);
        scratch.pilot_wide_pred_count = 0;
        int per_token_k = std::min(pilot_k_for_layer(l + 1), c.topk);
        for (int ws = 0; ws < S; ++ws) {
          const float *tok_hidden = hidden_states + ((size_t)ws * D);
          rmsnorm_row(scratch.pilot_wide_nrm.data(), tok_hidden, wnorm_w, D,
                      c.eps);
          dense_proj(engine, wrouter_key, scratch.pilot_wide_nrm.data(),
                     scratch.pilot_wide_logits.data(), 1, D, c.n_experts);
          apply_gating_activation(
              c.gating_func, scratch.pilot_wide_logits.data(), c.n_experts);
          if (wbias_row) {
            for (int e = 0; e < c.n_experts; ++e)
              scratch.pilot_wide_logits[e] += wbias_row[e];
          }
          for (int k = 0; k < per_token_k; ++k) {
            int best = -1;
            for (int e = 0; e < c.n_experts; ++e) {
              if (best < 0 || scratch.pilot_wide_logits[e] >
                                  scratch.pilot_wide_logits[best])
                best = e;
            }
            if (best < 0)
              break;
            scratch.pilot_wide_logits[best] = -2e30f;
            if (!wide_seen[best] &&
                scratch.pilot_wide_pred_count <
                    (int)scratch.pilot_wide_pred_ids.size()) {
              wide_seen[best] = true;
              scratch.pilot_wide_pred_ids[scratch.pilot_wide_pred_count++] =
                  best;
            }
          }
        }
        scratch.pilot_wide_pred_layer = l + 1;

        // EngineConfig::prefetch_widen_all_tokens -- real disk-
        // prefetch hint push from the widened prediction just
        // computed above. Mirrors DEPTH-1's own already_ram/
        // already_vram residency check + async_io_queue.push_hint
        // call (below, in DEPTH-1's own k-loop) exactly, applied to
        // this block's much wider candidate set instead of DEPTH-1's
        // fixed 8 -- deliberately HINT-PUSH ONLY, no VRAM promotion
        // (see this flag's own comment, engine_config.hpp, for why).
        // PilotSource::Other (same untagged high-priority lane
        // DEPTH-1 itself uses) -- no dedicated source tag added for
        // this experiment; [PILOT-WIDE]'s own coverage/recall
        // numbers already measure this source's prediction quality
        // directly, so a separate bytes_prefetched_widen-style
        // completion counter (the treatment Depth2 gets) wasn't
        // judged worth the added instrumentation surface yet.
        if (engine.prefetch_widen_all_tokens) {
          for (int i = 0; i < scratch.pilot_wide_pred_count; ++i) {
            int wide_id = scratch.pilot_wide_pred_ids[i];
            bool w_already_ram = ram_mgr.is_ram_resident(l + 1, wide_id);
            bool w_already_vram = false;
            if (!w_already_ram && ram_mgr.vram_cache) {
              for (const auto &loc : ram_mgr.vram_cache->expert_vram_cache) {
                if (loc.layer_idx == l + 1 && loc.expert_id == wide_id) {
                  w_already_vram = true;
                  break;
                }
              }
            }
            if (!w_already_ram && !w_already_vram) {
              async_io_queue.push_hint(l + 1, wide_id);
              perf().n_pilot_wide_hints_pushed.fetch_add(
                  1, std::memory_order_relaxed);
            } else {
              perf().n_pilot_wide_hints_skipped_resident.fetch_add(
                  1, std::memory_order_relaxed);
            }
          }
        }
      }
    }

    // ATTN-PILOT's own prediction (deferred from right after l's own
    // attention -- see scratch.pilot_attn_snapshot's own comment for
    // why: consumption of l's own pending prediction, inside l's just-
    // finished moe_forward_gguf/dense_ffn_gguf call, must happen BEFORE
    // this overwrites the same field) is computed further below, INSIDE
    // DEPTH-1's own block right after this comment -- not here -- since
    // it targets the SAME layer (l+1) using the SAME norm/router/bias
    // tensors DEPTH-1 already looks up, and shares DEPTH-1's own
    // confirmed_resident_this_transition/publish_pilot_confirmed_
    // resident call for its real hints. See that site for the actual
    // firing logic.

    // --- Speculative pilot prefetch for next layer's experts ---
    // Deliberately placed HERE -- AFTER this layer's own FFN/MoE
    // residual has been added -- not right after the attention
    // residual. hidden_states at this point is EXACTLY what layer
    // l+1's real router will see when the loop reaches it next
    // iteration. Predicting from the pre-FFN hidden_states (the old
    // position, right after the attention residual above) meant the
    // pilot's router pass ran on a vector missing this layer's own
    // MoE FFN contribution entirely -- routinely the single largest
    // per-layer compute cost in this model -- so its guessed top-8
    // experts had no reason to match what the real router at l+1
    // would actually select. Real-run prefetch instrumentation
    // (n_prefetch_hits_prefill/decode, n_prefetch_miss_still_queued) showed
    // 0% served in BOTH prefill and decode with 0/N misses even
    // finding a leftover trace of a hint -- exactly what stale-input
    // misprediction looks like (a timing/pool-size problem would
    // still show occasional lucky matches, not a clean zero).
    if (l + 1 < c.n_layers && c.n_experts > 0) {
      const auto diag_pilot_start = std::chrono::high_resolution_clock::now();
      PERF_START(pilot);
      std::string norm_key =
          "blk." + std::to_string(l + 1) + ".ffn_norm.weight";
      std::string router_key =
          "blk." + std::to_string(l + 1) + ".ffn_gate_inp.weight";
      auto norm_it = engine.resident_layers.find(norm_key);
      auto router_it = engine.resident_layers.find(router_key);

      // This pilot reconstructs the conventional norm->gate router input.
      // Qwen4Exp derives its route input through the HC/GPU graph instead,
      // so never apply this approximation to Qwen; its production router
      // remains the only source of truth there. GLM/DeepSeek-style families
      // stay on this existing path.
      const bool generic_pilot_inputs_ready =
          engine.family_ops.family != ModelFamily::Qwen4Exp &&
          norm_it != engine.resident_layers.end() &&
          router_it != engine.resident_layers.end() &&
          norm_it->second.host_ptr && router_it->second.host_ptr;
      const float *last_hidden = hidden_states + ((S - 1) * D);

      if (generic_pilot_inputs_ready) {
        // Use only last token for pilot prediction (saves compute)
        rmsnorm_row(scratch.pilot_nrm.data(), last_hidden,
                    reinterpret_cast<float *>(norm_it->second.host_ptr), D,
                    c.eps);

        dense_proj(engine, router_key, scratch.pilot_nrm.data(),
                   scratch.pilot_logits.data(), 1, D, c.n_experts);

        // Rank by EXACTLY the same "selection_probs" the real router
        // uses (moe_forward_gguf above: gate_func(logits) + bias),
        // not raw dense_proj output. Missing this was a second,
        // independent misprediction source on top of the residual-
        // timing fix above --GLM-5.2/DeepSeek-V3's exp_probs_b.bias exists
        // specifically to shift WHICH experts get selected for
        // load-balancing (auxiliary-loss-free routing) -- ranking on
        // the un-biased score means the pilot's top-k has no reason
        // to match the real, bias-corrected top-k the router
        // actually dispatches, independent of how accurate the input
        // hidden_states is.
        apply_gating_activation(c.gating_func, scratch.pilot_logits.data(),
                                c.n_experts);
        {
          std::string bias_key =
              "blk." + std::to_string(l + 1) + ".exp_probs_b.bias";
          auto bias_it = engine.resident_layers.find(bias_key);
          if (bias_it != engine.resident_layers.end() &&
              bias_it->second.host_ptr) {
            const float *bias_row =
                reinterpret_cast<float *>(bias_it->second.host_ptr);
            for (int e = 0; e < c.n_experts; ++e)
              scratch.pilot_logits[e] += bias_row[e];
          }
        }

        int pilot_k = std::min(pilot_k_for_layer(l + 1), c.topk);
        // prefetch_pilot_k_adaptive (2026-09-13): narrows the above
        // static ceiling to THIS layer's own live, most-confident ranks
        // only -- see adaptive_pilot_k_for_layer's own comment
        // (dense_dispatch.hpp). No-op (returns pilot_k unchanged) unless
        // explicitly enabled in config.
        pilot_k = adaptive_pilot_k_for_layer(l + 1, pilot_k);
        // Recall instrumentation: capture the full predicted set
        // BEFORE the already-resident skip below, so this measures
        // pure prediction accuracy against the real router's later
        // selection -- see LayerScratch::pilot_pred_ids's own
        // comment. moe_forward_gguf compares and resets this the
        // next time layer l+1 actually runs.
        scratch.pilot_pred_count = pilot_k;
        scratch.pilot_pred_layer = l + 1;
        // Collected below, published once after the k-loop --
        // feeds was_pilot_confirmed_resident's eviction-race
        // measurement (dense_dispatch.hpp). Only RAM-tier and
        // VRAM-UNPINNED-LRU picks are worth tracking here: a
        // pinned VRAM slot can never be chosen as an evict_lru
        // victim in the first place (see that function's pinned
        // check), so including it would just waste the fixed-size
        // buffer on something that could never match.
        PilotConfirmedEntry
            confirmed_resident_this_transition[PILOT_CONFIRMED_CAP];
        int confirmed_resident_count = 0;
        // Cross-pilot VRAM-promotion dedup (2026-08-25) -- DEPTH-1's
        // and ATTN-PILOT's k-loops both target l+1 from two
        // different inputs, so their top-K sets routinely overlap.
        // Without this, both loops could independently call
        // ram_mgr.promote_ram_to_vram() for the SAME (layer,expert)
        // within the SAME transition -- and since GpuTopology::
        // find_expert only reports a slot resident once its async
        // H2D copy is CONFIRMED complete (cudaEventQuery), the
        // SECOND call (issued moments after the first, DMA still in
        // flight) sees "not resident yet" and genuinely re-issues a
        // second evict_lru + upload_expert_async for the exact same
        // expert -- a real wasted VRAM slot and DMA, not a harmless
        // re-check. Scoped local to this one transition (both
        // loops run sequentially on this same thread for this same
        // l+1), NOT a persistent/global structure -- deliberately
        // separate from confirmed_resident_this_transition above,
        // which tracks a different population (already-resident
        // picks for the eviction-race measurement) and must not be
        // conflated with "a promotion was issued this transition".
        // Capacity 16 = DEPTH-1's own promotion candidates (<=8,
        // rank < ram_vram_promote_top_k) + ATTN-PILOT's own
        // (<=8, rank < attn_pilot_vram_promote_top_k) -- generous,
        // not a hard model limit (mirrors PILOT_CONFIRMED_CAP's own
        // sizing comment).
        struct VramPromoAttempt {
          int layer_idx;
          int expert_id;
        };
        constexpr int MAX_VRAM_PROMO_PER_TRANSITION = 16;
        VramPromoAttempt
            vram_promoted_this_transition[MAX_VRAM_PROMO_PER_TRANSITION];
        int vram_promoted_count = 0;
        auto try_promote_ram_to_vram_deduped = [&](int layer_idx,
                                                   int expert_id) {
          for (int i = 0; i < vram_promoted_count; ++i) {
            if (vram_promoted_this_transition[i].layer_idx == layer_idx &&
                vram_promoted_this_transition[i].expert_id == expert_id) {
              perf().n_pilot_promote_dedup_skipped.fetch_add(
                  1, std::memory_order_relaxed);
              return;
            }
          }
          ram_mgr.promote_ram_to_vram(engine, layer_idx, expert_id);
          if (vram_promoted_count < MAX_VRAM_PROMO_PER_TRANSITION) {
            vram_promoted_this_transition[vram_promoted_count++] = {layer_idx,
                                                                    expert_id};
          }
        };
        // Active-layer sibling of the dedup wrapper above
        // (EngineConfig::speculative_promote_to_active_layer,
        // 2026-08-26) -- own, separate dedup buffer, NOT shared with
        // vram_promoted_this_transition: a (layer,expert) promoted
        // into the active layer and one promoted into the main
        // cache are tracked independently, since only one of the
        // two call sites below is ever active per run (the config
        // flag routes ALL of this transition's promotion calls to
        // exactly one of the two lambdas), so no cross-population
        // collision is possible in practice -- kept separate anyway
        // for clarity, mirroring MAX_VRAM_PROMO_PER_TRANSITION's own
        // sizing rationale.
        VramPromoAttempt active_layer_promoted_this_transition
            [MAX_VRAM_PROMO_PER_TRANSITION];
        int active_layer_promoted_count = 0;
        // 2026-08-26: NO batch-sync buffer needed anymore -- the
        // async DMA promote_ram_to_active_layer issues (upload_
        // ready left false) is now safe to just let run in the
        // background. find_active_expert (gpu_topology.cpp) polls
        // the slot's own completion event on the first real lookup
        // instead of trusting upload_ready blindly, exactly
        // mirroring how the main VRAM cache's finish_expert_hit
        // already handles its own async promotions -- if the DMA
        // genuinely hasn't landed by the time l+1 dispatch reaches
        // it, that single lookup is treated as a miss (see
        // n_active_layer_poll_not_ready) rather than blocking here.
        // A real, measured ~3.1s/request cost from the FIRST
        // (fully synchronous) version of this feature, and a
        // smaller but still real bandwidth-competing cost from the
        // SECOND (batched-but-still-blocking) version, are both
        // gone with this change -- there is no unconditional wait
        // left on this path at all now, only an on-demand poll
        // that costs nothing when the copy is already done (the
        // overwhelmingly common case, given a whole layer's worth
        // of intervening work before l+1 is dispatched).
        auto try_promote_ram_to_active_layer_deduped = [&](int layer_idx,
                                                           int expert_id,
                                                           int rank) {
          for (int i = 0; i < active_layer_promoted_count; ++i) {
            if (active_layer_promoted_this_transition[i].layer_idx ==
                    layer_idx &&
                active_layer_promoted_this_transition[i].expert_id ==
                    expert_id) {
              perf().n_pilot_active_fill_dedup_skipped.fetch_add(
                  1, std::memory_order_relaxed);
              return;
            }
          }
          // 2026-08-28: multi-GPU routing decision -- see
          // route_active_layer_gpu's own comment (dense_
          // dispatch.hpp). -1 (no multi-GPU config) is the
          // correct "no restriction" value for the legacy path.
          // total_candidates = speculative_active_layer_max_k_
          // value(), NOT a fixed nominal 8 -- k here can never
          // exceed that bound (this whole call site is already
          // gated by k < speculative_active_layer_max_k_value()
          // a few lines up the call chain), so THAT is the real
          // reachable range the tail must be computed against.
          int routed_gpu =
              active_layer_multi_gpu_enabled()
                  ? route_active_layer_gpu(
                        rank, speculative_active_layer_max_k_value())
                  : -1;
          if (routed_gpu >= 0) {
            perf()
                .n_active_layer_routed_by_gpu
                    [routed_gpu < PerfCounters::PERF_MAX_GPU_SLOTS ? routed_gpu
                                                                   : 0]
                .fetch_add(1, std::memory_order_relaxed);
          }
          // 2026-08-28: attribute the promotion to its rank (used
          // by record_active_layer_prewarm_reclaim's per-rank
          // used/wasted split, gpu_topology.hpp) -- see
          // ExpertLocation::promoted_at_rank's own comment.
          ExpertLocation *promoted = nullptr;
          if (ram_mgr.upload_workers && ram_mgr.upload_workers->enabled()) {
            // See RAMSafetyManager::promote_ram_to_active_layer_
            // claim_for_worker's own comment for why this claim/
            // submit split exists (ActiveLayerUploadWorkers is
            // only forward-declared inside ram_manager.hpp, so
            // the actual try_submit call has to happen here,
            // where the full type is visible).
            auto claim = ram_mgr.promote_ram_to_active_layer_claim_for_worker(
                engine, layer_idx, expert_id, routed_gpu);
            if (claim.slot) {
              ActiveLayerUploadWorkItem item{claim.slot, claim.blk,
                                             claim.host_data, claim.host_bytes};
              if (ram_mgr.upload_workers->try_submit(item)) {
                promoted = claim.slot;
              } else {
                ram_mgr.vram_cache->abandon_worker_claim(claim.slot);
                ram_mgr.ram_unpin(*claim.blk);
              }
            }
          } else {
            promoted = ram_mgr.promote_ram_to_active_layer(
                engine, layer_idx, expert_id, routed_gpu);
          }
          if (promoted && rank >= 0 && rank < 8)
            promoted->promoted_at_rank = (int8_t)rank;
          if (active_layer_promoted_count < MAX_VRAM_PROMO_PER_TRANSITION) {
            active_layer_promoted_this_transition
                [active_layer_promoted_count++] = {layer_idx, expert_id};
          }
        };
        // Adaptive-mode contiguous-prefix cutoff state for THIS
        // transition's k-loop -- ranks are walked in increasing
        // order below, so tracking "still eligible" as we go
        // correctly implements "promote rank k only if every rank
        // < k also cleared the bar" without a second pass. Reset
        // per transition (not persisted across layers/tokens) --
        // whether rank k clears the bar is a function of the LIVE,
        // GLOBAL n_pilot_ram_only_* counters at the moment this
        // runs, not of anything specific to this one transition.
        // Unused in static top_k mode.
        bool adaptive_promote_still_eligible = true;
        const auto &promo_cfg = ram_vram_promote_config();
        // "Idea 1" prediction-based GPU1 dual-role-slice loading
        // (2026-08-16) -- see RAMSafetyManager::promote_ram_to_
        // vram_gpu1's own comment for the full design. Resolved
        // ONCE per transition, not per rank: dual_role_gpu_id()
        // takes GpuTopology::lock internally (a real, if cheap,
        // lock acquisition -- see that function's own comment), no
        // reason to pay it up to 8x when the answer can't change
        // within one transition. predict_gpu1_promotions_this_
        // transition mirrors adaptive_promote_still_eligible's own
        // per-transition-local-variable pattern directly above.
        const auto &gpu1_predict_cfg = gpu1_predict_promote_config();
        int gpu1_predict_dual_role_gpu = -1;
        if (gpu1_predict_cfg.enabled && ram_mgr.vram_cache) {
          gpu1_predict_dual_role_gpu = ram_mgr.vram_cache->dual_role_gpu_id();
        }
        int gpu1_predict_promotions_this_transition = 0;
        for (int k = 0; k < pilot_k; ++k) {
          int best_exp = 0;
          for (int e = 1; e < c.n_experts; ++e) {
            if (scratch.pilot_logits[e] > scratch.pilot_logits[best_exp])
              best_exp = e;
          }
          scratch.pilot_logits[best_exp] = -2e30f;
          scratch.pilot_pred_ids[k] = best_exp;

          // Checked so a hint is never pushed for an expert that's
          // ALREADY resident -- a guaranteed instant hit needing no
          // read at all once actually reached, so speculatively
          // prefetching it anyway only steals scarce worker-thread
          // I/O bandwidth from hints that genuinely need a disk
          // read. Routed through RAMSafetyManager::is_ram_resident
          // (locked internally) rather than a raw scan of
          // engine.layers_experts[l+1] -- a LockFreePrefetcher
          // worker thread can now concurrently claim/evict slots
          // in that exact array via claim_or_wait, so an unlocked
          // read of expert_id here would be a real data race, not
          // just an occasionally-stale peek.
          bool already_ram = ram_mgr.is_ram_resident(l + 1, best_exp);
          // FIX: this used to only check the RAM tier. Before the
          // [TIER-OVERLAP] fix, a VRAM-resident expert was almost
          // always ALSO RAM-resident (moe_forward_gguf loaded
          // everything into RAM unconditionally), so this check
          // incidentally caught most VRAM-resident cases too. Now
          // that VRAM-resident-but-NOT-RAM-resident is the common
          // case (the fast path skips the RAM load entirely), this
          // check was missing it -- a real run showed prefetch-served
          // misses collapse from ~24-25% to 0.5% as a direct
          // result: hints kept getting pushed for experts that,
          // once reached, never touch the staging pool at all
          // (the shadow fast path bypasses get_expert/
          // consume_staged entirely), so the staged bytes just sat
          // unused. Direct read of expert_vram_cache's public
          // fields (NOT vram_cache->find_expert, which mutates
          // hit_count/last_used_clock as a side effect -- this is
          // a speculative "is it worth hinting" check on the NEXT
          // layer, not a genuine use yet, same non-mutating style
          // the RAM-tier check above already uses).
          bool already_vram = false, vram_pinned = false;
          if (!already_ram && ram_mgr.vram_cache) {
            for (const auto &loc : ram_mgr.vram_cache->expert_vram_cache) {
              if (loc.layer_idx == l + 1 && loc.expert_id == best_exp) {
                already_vram = true;
                vram_pinned = loc.pinned;
                break;
              }
            }
          }
          // Classify into exactly one bucket for the "Pilot
          // prediction coverage" diagnostic (perf_timer.hpp) --
          // pure bookkeeping alongside the pre-existing
          // skip/push decision below, doesn't change which
          // branch fires.
          bool is_hint_worthy = !already_ram && !already_vram;
          if (already_ram) {
            perf().n_pilot_pred_already_ram.fetch_add(
                1, std::memory_order_relaxed);
            if (confirmed_resident_count < PILOT_CONFIRMED_CAP)
              confirmed_resident_this_transition[confirmed_resident_count++] = {
                  l + 1, best_exp};

            // Pilot-driven RAM->VRAM speculative promotion --
            // see dense_dispatch.hpp's ram_vram_promote_config
            // for config wiring and RAMSafetyManager::
            // promote_ram_to_vram for the actual upload +
            // deadlock-safety contract. This branch (already_
            // ram) is deliberately the ONLY place this fires:
            // it's the exact population the "ram-only" per-rank
            // metric measures, disjoint from is_hint_worthy's
            // needs_fetch population above and from the two
            // already_vram branches below (which need no
            // action -- already where they need to be).
            bool should_promote = false;
            if (promo_cfg.dynamic_layer) {
              // 2026-08-26: per-layer WIDTH instead of a flat
              // cutoff -- see dynamic_layer_top_k's own
              // comment (dense_dispatch.hpp). Judged against
              // DEPTH-1's own per-layer accuracy
              // (is_attn_pilot=false).
              should_promote =
                  (k < dynamic_layer_top_k(l + 1, /*is_attn_pilot=*/false));
            } else if (promo_cfg.top_k > 0) {
              should_promote = (k < promo_cfg.top_k);
            } else if (promo_cfg.adaptive && adaptive_promote_still_eligible) {
              uint64_t checks =
                  (k < 8) ? perf().n_pilot_ram_only_checks_by_rank[k].load(
                                std::memory_order_relaxed)
                          : 0;
              uint64_t rhits =
                  (k < 8) ? perf().n_pilot_ram_only_hits_by_rank[k].load(
                                std::memory_order_relaxed)
                          : 0;
              double acc = checks > 0 ? (double)rhits / (double)checks : 0.0;
              if (checks >= (uint64_t)promo_cfg.min_samples &&
                  acc >= promo_cfg.adaptive_threshold) {
                should_promote = true;
              } else {
                // Contiguous-prefix cutoff: this rank failed
                // (too few samples OR accuracy below
                // threshold) -- no later rank this
                // transition may promote either, even if a
                // deeper rank's OWN live accuracy happens to
                // look higher (matches the monotonic-decay
                // shape observed in every real capture so
                // far; a later rank clearing the bar alone
                // isn't trusted).
                adaptive_promote_still_eligible = false;
              }
            }
            // ram_vram_promote_layer_gate (2026-08-25): additional
            // per-LAYER veto on top of should_promote above -- see
            // dense_dispatch.hpp's layer_promote_allowed for the
            // full design (real [PILOT-BY-LAYER] accuracy spread
            // by layer, CLAUDE.md). is_attn_pilot=false: judged
            // against DEPTH-1's OWN per-layer counters. Split into
            // an explicit if/else (rather than folding the veto
            // into one && condition) so a real veto is directly
            // counted, not just inferred from a smaller attempted
            // total. SKIPPED when promo_cfg.dynamic_layer is
            // already active (2026-08-26) -- dynamic_layer_top_k
            // above already IS the per-layer decision (using its
            // own, possibly different, thresholds); re-applying
            // this separate flat-veto's threshold on top would
            // just double-gate against two different bars rather
            // than compose meaningfully.
            if (should_promote) {
              if (promo_cfg.dynamic_layer ||
                  layer_promote_allowed(l + 1, /*is_attn_pilot=*/false)) {
                // speculative_promote_to_active_layer
                // (2026-08-26): route into the cheap,
                // self-contained active-layer pool instead
                // of the main VRAM LRU cache -- see
                // EngineConfig::speculative_promote_to_
                // active_layer's own comment for the full
                // motivation.
                if (speculative_promote_to_active_layer_enabled()) {
                  // speculative_active_layer_max_k: a
                  // SEPARATE, narrower cutoff than
                  // should_promote's own width -- see its
                  // own comment (dense_dispatch.hpp) for
                  // the real A/B (added DMA volume
                  // competing for the same bandwidth-
                  // bound pipe decode_async_prewarm uses)
                  // that motivated decoupling this from
                  // the (now-bypassed) main-cache width.
                  if (k < speculative_active_layer_max_k_value()) {
                    try_promote_ram_to_active_layer_deduped(l + 1, best_exp, k);
                  } else {
                    // 2026-08-28: direct answer to "is
                    // max_k actually the binding
                    // constraint, or is the tier/
                    // threshold structure already
                    // stricter" -- fires only when the
                    // tier system (should_promote, just
                    // above) already approved this rank
                    // but max_k clipped it anyway. If
                    // this stays near-zero across a
                    // max_k range, raising max_k further
                    // can't be doing anything real --
                    // the tiers themselves are the
                    // actual ceiling. See perf_timer.hpp
                    // for the report line.
                    perf().n_speculative_max_k_binding_depth1.fetch_add(
                        1, std::memory_order_relaxed);
                  }
                } else {
                  try_promote_ram_to_vram_deduped(l + 1, best_exp);
                }
              } else {
                perf().n_pilot_promote_layer_gate_vetoed_depth1.fetch_add(
                    1, std::memory_order_relaxed);
              }
            }
            // "Idea 1" -- deliberately a SEPARATE decision from
            // should_promote above, not an else-branch: a rank
            // could in principle satisfy both ram_vram_promote_
            // top_k (general pool) and this rank window at
            // once, though in practice they're configured to
            // disjoint rank ranges (see doc/gpu-utilization-
            // vram-analysis.md). Same "already_ram" population,
            // same k this rank's real predicted expert id
            // (best_exp) -- no new classification needed.
            if (gpu1_predict_dual_role_gpu >= 0 &&
                k >= gpu1_predict_cfg.rank_start &&
                k <= gpu1_predict_cfg.rank_end) {
              if (gpu1_predict_promotions_this_transition <
                  gpu1_predict_cfg.cap_per_layer) {
                ram_mgr.promote_ram_to_vram_gpu1(engine, l + 1, best_exp,
                                                 gpu1_predict_dual_role_gpu);
                ++gpu1_predict_promotions_this_transition;
              } else {
                perf().n_predict_gpu1_cap_reached.fetch_add(
                    1, std::memory_order_relaxed);
              }
            }
          } else if (already_vram && vram_pinned) {
            perf().n_pilot_pred_already_vram_pinned.fetch_add(
                1, std::memory_order_relaxed);
            // Deliberately NOT added to confirmed_resident_this_transition
            // -- a pinned slot can never be an evict_lru victim, see
            // that buffer's own comment above.
          } else if (already_vram) {
            perf().n_pilot_pred_already_vram_lru.fetch_add(
                1, std::memory_order_relaxed);
            if (confirmed_resident_count < PILOT_CONFIRMED_CAP)
              confirmed_resident_this_transition[confirmed_resident_count++] = {
                  l + 1, best_exp};
          } else {
            perf().n_pilot_pred_needs_fetch.fetch_add(
                1, std::memory_order_relaxed);
          }
          scratch.pilot_pred_was_hint[k] = is_hint_worthy;
          scratch.pilot_pred_was_ram_only[k] = already_ram;
          if (is_hint_worthy) {
            async_io_queue.push_hint(l + 1, best_exp);
          } else {
            perf().n_pilot_hints_skipped.fetch_add(1,
                                                   std::memory_order_relaxed);
          }
        }

        // --- ATTN-PILOT: real hint-wiring (2026-08-25), gated
        // behind EngineConfig::prefetch_attn_pilot (default off).
        // Targets the SAME layer (l+1) as DEPTH-1 just above, using
        // the SAME norm_key/router_key tensors (reused, not
        // re-looked-up) -- the only difference is the INPUT:
        // scratch.pilot_attn_snapshot (l's post-attention/pre-FFN
        // state, captured earlier and deferred here for the
        // ordering reason explained at the snapshot site) instead
        // of DEPTH-1's live post-FFN hidden_states. Real recall:
        // 75.8% overall (vs DEPTH-1's 84.8% for this same target,
        // vs DEPTH-2's 72.1% when read as reaching 2 layers past
        // the last known-complete layer) -- see CLAUDE.md. Shares
        // DEPTH-1's own confirmed_resident_this_transition buffer
        // and the ONE publish_pilot_confirmed_resident call below
        // (not a second one) -- PILOT_CONFIRMED_CAP raised to 24 to
        // cover all three contributors now pilot_confirmed_set.hpp).
        bool attn_pilot_enabled = prefetch_attn_pilot_enabled();
        {
          rmsnorm_row(
              scratch.pilot_nrm.data(), scratch.pilot_attn_snapshot.data(),
              reinterpret_cast<float *>(norm_it->second.host_ptr), D, c.eps);
          dense_proj(engine, router_key, scratch.pilot_nrm.data(),
                     scratch.pilot_logits.data(), 1, D, c.n_experts);
          apply_gating_activation(c.gating_func, scratch.pilot_logits.data(),
                                  c.n_experts);
          {
            std::string abias_key =
                "blk." + std::to_string(l + 1) + ".exp_probs_b.bias";
            auto abias_it = engine.resident_layers.find(abias_key);
            if (abias_it != engine.resident_layers.end() &&
                abias_it->second.host_ptr) {
              const float *abias_row =
                  reinterpret_cast<float *>(abias_it->second.host_ptr);
              for (int e = 0; e < c.n_experts; ++e)
                scratch.pilot_logits[e] += abias_row[e];
            }
          }
          int attn_pilot_k = std::min(8, c.topk);
          int attn_pilot_push_cutoff =
              std::min(attn_pilot_k_for_layer(l + 1), attn_pilot_k);
          scratch.pilot_attn_pred_count = attn_pilot_k;
          scratch.pilot_attn_pred_layer = l + 1;
          for (int k = 0; k < attn_pilot_k; ++k) {
            int best_exp = 0;
            for (int e = 1; e < c.n_experts; ++e) {
              if (scratch.pilot_logits[e] > scratch.pilot_logits[best_exp])
                best_exp = e;
            }
            scratch.pilot_logits[best_exp] = -2e30f;
            scratch.pilot_attn_pred_ids[k] = best_exp;

            // Residency classification -- ALWAYS runs
            // (measurement, drives the fetch-worthy recall
            // split) regardless of whether this pick is
            // actually pushed as a real hint.
            bool already_ram_a = ram_mgr.is_ram_resident(l + 1, best_exp);
            bool already_vram_a = false, vram_pinned_a = false;
            if (!already_ram_a && ram_mgr.vram_cache) {
              for (const auto &loc : ram_mgr.vram_cache->expert_vram_cache) {
                if (loc.layer_idx == l + 1 && loc.expert_id == best_exp) {
                  already_vram_a = true;
                  vram_pinned_a = loc.pinned;
                  break;
                }
              }
            }
            bool is_hint_worthy_a = !already_ram_a && !already_vram_a;
            bool attn_pilot_active_here =
                attn_pilot_enabled && k < attn_pilot_push_cutoff;
            if (already_ram_a) {
              perf().n_pilot_attn_pred_already_ram.fetch_add(
                  1, std::memory_order_relaxed);
              if (attn_pilot_active_here &&
                  confirmed_resident_count < PILOT_CONFIRMED_CAP)
                confirmed_resident_this_transition[confirmed_resident_count++] =
                    {l + 1, best_exp};
              // Speculative RAM->VRAM promotion for ATTN-PILOT's
              // own top-ranked already-RAM picks -- independent
              // of attn_pilot_active_here/prefetch_attn_pilot
              // above (this fires for ANY already-RAM pick
              // within rank, regardless of whether ATTN-PILOT's
              // own real-hint-push put it there), gated only by
              // its own separate cutoff. Reuses the SAME
              // proven, COPY-only mechanism DEPTH-1's own pilot
              // loop already uses -- see attn_pilot_vram_
              // promote_top_k_value()'s own comment (dense_
              // dispatch.hpp) for why this is a SEPARATE
              // cutoff, not a reuse of ram_vram_promote_top_k.
              // ram_vram_promote_layer_gate (2026-08-25) /
              // ram_vram_promote_dynamic_layer (2026-08-26):
              // is_attn_pilot=true throughout -- judged
              // against ATTN-PILOT's OWN per-layer counters,
              // not DEPTH-1's (see the DEPTH-1 call site's
              // own comment for why). attn_pilot_dynamic_here
              // mirrors DEPTH-1's own dynamic-vs-flat branch:
              // when both the shared dynamic_layer mode AND
              // ATTN-PILOT's own opt-in
              // (attn_pilot_vram_promote_dynamic) are set, the
              // per-rank cutoff becomes a per-layer WIDTH
              // instead of the fixed attn_pilot_vram_promote_
              // top_k -- and the separate flat veto below is
              // skipped for the same reason DEPTH-1's is (the
              // dynamic width already IS the per-layer
              // decision).
              bool attn_pilot_dynamic_here =
                  promo_cfg.dynamic_layer &&
                  attn_pilot_vram_promote_dynamic_enabled();
              int attn_promote_cutoff =
                  attn_pilot_dynamic_here
                      ? dynamic_layer_top_k(l + 1, /*is_attn_pilot=*/true)
                      : attn_pilot_vram_promote_top_k_value();
              if (k < attn_promote_cutoff) {
                if (attn_pilot_dynamic_here ||
                    layer_promote_allowed(l + 1, /*is_attn_pilot=*/true)) {
                  // See the DEPTH-1 call site's own comment
                  // above for speculative_promote_to_
                  // active_layer -- same flag, same
                  // redirect, judged against the SAME
                  // shared per-transition dedup discipline
                  // (a separate buffer from DEPTH-1's own).
                  if (speculative_promote_to_active_layer_enabled()) {
                    // speculative_active_layer_max_k --
                    // see the DEPTH-1 call site's own
                    // comment above for the full
                    // reasoning; same cutoff, shared
                    // between both pilots.
                    if (k < speculative_active_layer_max_k_value()) {
                      try_promote_ram_to_active_layer_deduped(l + 1, best_exp,
                                                              k);
                    } else {
                      // 2026-08-28: ATTN-PILOT's own
                      // copy of the DEPTH-1 call
                      // site's max_k-binding check --
                      // see n_speculative_max_k_
                      // binding_depth1's own comment.
                      perf().n_speculative_max_k_binding_attn.fetch_add(
                          1, std::memory_order_relaxed);
                    }
                  } else {
                    try_promote_ram_to_vram_deduped(l + 1, best_exp);
                  }
                } else {
                  perf().n_pilot_promote_layer_gate_vetoed_attn.fetch_add(
                      1, std::memory_order_relaxed);
                }
              }
            } else if (already_vram_a && vram_pinned_a) {
              perf().n_pilot_attn_pred_already_vram_pinned.fetch_add(
                  1, std::memory_order_relaxed);
            } else if (already_vram_a) {
              perf().n_pilot_attn_pred_already_vram_lru.fetch_add(
                  1, std::memory_order_relaxed);
              if (attn_pilot_active_here &&
                  confirmed_resident_count < PILOT_CONFIRMED_CAP)
                confirmed_resident_this_transition[confirmed_resident_count++] =
                    {l + 1, best_exp};
            } else {
              perf().n_pilot_attn_pred_needs_fetch.fetch_add(
                  1, std::memory_order_relaxed);
            }
            scratch.pilot_attn_pred_was_hint[k] = is_hint_worthy_a;
            if (attn_pilot_active_here && is_hint_worthy_a) {
              async_io_queue.push_hint(l + 1, best_exp, PilotSource::AttnPilot);
              perf().n_pilot_attn_hints_pushed.fetch_add(
                  1, std::memory_order_relaxed);
            }
          }
        }

        // 2026-08-26: no batch-sync call needed here anymore --
        // see try_promote_ram_to_active_layer_deduped's own comment
        // above for why (find_active_expert now polls each slot's
        // own completion event on first real lookup instead of
        // requiring an unconditional wait here).

        // Publish AFTER the full k-loop (not per-pick) so a reader
        // (evict_lru/RAM-tier eviction, running on the decode
        // thread same as this loop but conceptually racy in
        // principle) always sees a complete, internally-consistent
        // set for this transition rather than a partially-filled
        // one -- see was_pilot_confirmed_resident's own comment on
        // why this is measurement-only, not a hard correctness
        // dependency either way.
        // --- DEPTH-2 measurement (approach (a), "naive reuse"),
        // now with REAL prefetch wiring gated by
        // prefetch_pilot_depth2_enabled() ---
        // Predicts l+2's routing by reusing the SAME `last_hidden`
        // vector above -- i.e. gated on a vector missing l+1's own
        // FFN contribution entirely, the exact term this block's
        // own opening comment already showed is fatal to omit at
        // depth-1. Reuses scratch.pilot_nrm/pilot_logits (l+1's
        // k-loop has already consumed everything it needed from
        // them). The MEASUREMENT (recall counters, residency
        // classification, "Pilot recall by rank" report) always
        // runs -- it's cheap (one more gate pass, same order as
        // the l+1 pilot's own query) and lets a user evaluate
        // depth-2 before ever turning it on. Only the REAL
        // async_io_queue.push_hint / publish_pilot_confirmed_
        // resident side effects are gated by
        // prefetch_pilot_depth2_enabled() (EngineConfig::
        // prefetch_pilot_depth) and further limited to the top
        // pilot_k2_for_layer() ranks (EngineConfig::
        // prefetch_pilot_k_depth2) -- see that config field's own
        // comment for why a separate, often-smaller cutoff makes
        // sense here than for l+1. (An FFN-reconstruction
        // alternative, approach "(b)", was measured statistically
        // tied with this despite costing real compute, and was
        // removed -- see CLAUDE.md.)
        //
        // confirmed_resident_this_transition/count (declared above,
        // in the l+1 block) is REUSED, not republished separately
        // -- l+1's and l+2's confirmed-resident entries are
        // published together in ONE call after this block, so a
        // reader (evict_lru/RAM-tier eviction) always sees a
        // complete, internally-consistent set for the WHOLE
        // transition, matching that publish call's own
        // single-call-after-the-full-k-loop reasoning at l+1's
        // depth. PILOT_CONFIRMED_CAP (16) already covers the
        // worst case (up to 8 from l+1 + up to 8 from l+2).
        bool depth2_enabled = prefetch_pilot_depth2_enabled();
        if (l + 2 < c.n_layers && c.n_experts > 0) {
          std::string norm_key2 =
              "blk." + std::to_string(l + 2) + ".ffn_norm.weight";
          std::string router_key2 =
              "blk." + std::to_string(l + 2) + ".ffn_gate_inp.weight";
          auto norm_it2 = engine.resident_layers.find(norm_key2);
          auto router_it2 = engine.resident_layers.find(router_key2);
          if (norm_it2 != engine.resident_layers.end() &&
              router_it2 != engine.resident_layers.end() &&
              norm_it2->second.host_ptr && router_it2->second.host_ptr) {
            rmsnorm_row(scratch.pilot_nrm.data(), last_hidden,
                        reinterpret_cast<float *>(norm_it2->second.host_ptr), D,
                        c.eps);

            dense_proj(engine, router_key2, scratch.pilot_nrm.data(),
                       scratch.pilot_logits.data(), 1, D, c.n_experts);

            apply_gating_activation(c.gating_func, scratch.pilot_logits.data(),
                                    c.n_experts);
            {
              std::string bias_key2 =
                  "blk." + std::to_string(l + 2) + ".exp_probs_b.bias";
              auto bias_it2 = engine.resident_layers.find(bias_key2);
              if (bias_it2 != engine.resident_layers.end() &&
                  bias_it2->second.host_ptr) {
                const float *bias_row2 =
                    reinterpret_cast<float *>(bias_it2->second.host_ptr);
                for (int e = 0; e < c.n_experts; ++e)
                  scratch.pilot_logits[e] += bias_row2[e];
              }
            }

            // Two-slot ring, indexed by l % 2 -- see
            // LayerScratch::pilot_pred2_slots's own comment:
            // a single shared field here would get clobbered
            // by layer l+1's own depth-2 write before layer
            // l+2 ever gets a chance to consume it.
            auto &slot2 = scratch.pilot_pred2_slots[l % 2];
            int pilot_k2 = std::min(pilot_k_for_layer(l + 2), c.topk);
            slot2.count = pilot_k2;
            slot2.layer = l + 2;
            // How many of THIS transition's l+2 picks (ranked
            // 0 = highest pilot confidence, since they're
            // extracted by repeated argmax below) actually get
            // pushed as real hints -- independent of pilot_k2
            // above, which governs how many get MEASURED. See
            // EngineConfig::prefetch_pilot_k_depth2's own
            // comment for why a separate, often-smaller cutoff
            // makes sense for l+2 than for l+1.
            int depth2_push_cutoff =
                std::min(pilot_k2_for_layer(l + 2), pilot_k2);
            for (int k = 0; k < pilot_k2; ++k) {
              int best_exp = 0;
              for (int e = 1; e < c.n_experts; ++e) {
                if (scratch.pilot_logits[e] > scratch.pilot_logits[best_exp])
                  best_exp = e;
              }
              scratch.pilot_logits[best_exp] = -2e30f;
              slot2.ids[k] = best_exp;

              // Residency classification -- ALWAYS runs
              // (measurement, drives the report below and
              // the recall-check's fetch-worthy split)
              // regardless of whether this pick is actually
              // pushed as a real hint.
              bool already_ram = ram_mgr.is_ram_resident(l + 2, best_exp);
              bool already_vram = false, vram_pinned = false;
              if (!already_ram && ram_mgr.vram_cache) {
                for (const auto &loc : ram_mgr.vram_cache->expert_vram_cache) {
                  if (loc.layer_idx == l + 2 && loc.expert_id == best_exp) {
                    already_vram = true;
                    vram_pinned = loc.pinned;
                    break;
                  }
                }
              }
              bool is_hint_worthy = !already_ram && !already_vram;
              // Only picks within the real push cutoff can
              // affect real dispatch -- gate the eviction-
              // guard entry and the real push_hint call on
              // the SAME condition, so nothing gets
              // "protected" that was never actually relied
              // on, and nothing gets a real hint pushed for
              // it beyond what depth2_push_cutoff allows.
              bool depth2_active_here =
                  depth2_enabled && k < depth2_push_cutoff;
              if (already_ram) {
                perf().n_pilot2_pred_already_ram.fetch_add(
                    1, std::memory_order_relaxed);
                if (depth2_active_here &&
                    confirmed_resident_count < PILOT_CONFIRMED_CAP)
                  confirmed_resident_this_transition
                      [confirmed_resident_count++] = {l + 2, best_exp};
              } else if (already_vram && vram_pinned) {
                perf().n_pilot2_pred_already_vram_pinned.fetch_add(
                    1, std::memory_order_relaxed);
                // Deliberately NOT added to confirmed_resident_this_transition
                // -- a pinned slot can never be an evict_lru victim, see
                // that buffer's own comment above (same reasoning as l+1).
              } else if (already_vram) {
                perf().n_pilot2_pred_already_vram_lru.fetch_add(
                    1, std::memory_order_relaxed);
                if (depth2_active_here &&
                    confirmed_resident_count < PILOT_CONFIRMED_CAP)
                  confirmed_resident_this_transition
                      [confirmed_resident_count++] = {l + 2, best_exp};
              } else {
                perf().n_pilot2_pred_needs_fetch.fetch_add(
                    1, std::memory_order_relaxed);
              }
              slot2.was_hint[k] = is_hint_worthy;
              if (depth2_active_here && is_hint_worthy) {
                // Tagged Depth2 -- see PilotSource's own
                // comment (async_prefetcher.hpp) -- so the
                // real disk-I/O cost of THIS hint is
                // attributable in the shutdown report's
                // "Prefetch real fetches (DEPTH-2 only)"
                // line, not silently folded into the
                // aggregate.
                async_io_queue.push_hint(l + 2, best_exp, PilotSource::Depth2);
                perf().n_pilot2_hints_pushed.fetch_add(
                    1, std::memory_order_relaxed);
              }
            }
          }
        }

        // Published ONCE for the whole transition (l+1's entries
        // above plus l+2's above, if depth-2 is enabled) -- see
        // this block's own opening comment for why a single
        // combined publish, not two separate calls.
        publish_pilot_confirmed_resident(confirmed_resident_this_transition,
                                         confirmed_resident_count);
      }
      PERF_END(pilot, t_prefetch);
      if (S == 1)
        g_diag_decode_pilot_ms +=
            std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - diag_pilot_start)
                .count();
    }
    // TEMPORARY DIAGNOSTIC (2026-09-12) -- see g_diag_layer_loop_total_ms's
    // own comment above the function. No continue/break/return exits this
    // loop early (verified) so this always runs once per layer iteration.
    g_diag_layer_loop_total_ms +=
        std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - diag_layer_loop_start)
            .count();
    if (S == 1)
      g_diag_decode_layer_loop_ms +=
          std::chrono::duration<double, std::milli>(
              std::chrono::high_resolution_clock::now() - diag_layer_loop_start)
              .count();
  }
}
