// Qwen4Exp full graph: Whole-model graph construction and prefill->decode state hand-off.
//
// Split out of the former single-file ggml_qwen4exp_graph.cpp; shared
// declarations live in ggml_qwen4exp_graph_internal.hpp.

#include <algorithm>
#include "ggml_qwen4exp_graph_internal.hpp"

#if defined(ZEROMOE_USE_GGML_CUDA)

using namespace qwen4exp_graph;

namespace qwen4exp_graph {

bool build_graph(FullGraph &g, const std::vector<float> *ple_embedding) {
  g_qwen4exp_build_graph_retry_smaller_expert_layers = false;
  // Timing split (2026-09-17): full-graph decode's FIRST call pays a real
  // ~5.3s one-time cost, invisible in any --ngen 400 average (6000/400=
  // 15ms/token) but a genuine multi-second latency cliff for interactive
  // use (every new conversation/context reset pays it). See memory:
  // full-graph-decode-graph-construction-5s-latency.md. This split isolates
  // OUR host-side node-construction C++ (this function's own body) from
  // ggml's own ggml_backend_sched_alloc_graph() call below, since the fix
  // differs completely depending on which one dominates.
  const auto t_build_graph_start = std::chrono::steady_clock::now();
  g.dense_upload_ms_total = 0.0;
  g.expert_upload_ms_total = 0.0;
  g.expert_upload_calls = 0;
  g.expert_weight_meta_ms_total = 0.0;
  g.expert_weight_source_ms_total = 0.0;
  g.expert_weight_host_bind_ms_total = 0.0;
  g.expert_weight_device_bind_ms_total = 0.0;
  g.expert_weight_load_ms_total = 0.0;
  g.expert_weight_split_ms_total = 0.0;
  g.expert_vram_stage_alloc_ms_total = 0.0;
  g.expert_vram_file_read_ms_total = 0.0;
  g.expert_vram_file_read_bytes = 0;
  g.expert_vram_file_read_calls = 0;
  g.expert_weight_host_bind_calls = 0;
  g.expert_weight_device_bind_calls = 0;
  g.attn_construction_ms_total = 0.0;
  g.moe_construction_ms_total = 0.0;
  g.moe_weight_lookup_ms_total = 0.0;
  const auto &c = g.engine->model_file.architecture_config;
  ggml_init_params ip = {};
  ip.mem_size = kTensorContextBytes;
  ip.no_alloc = true;
  if (!MemoryGuard::try_reserve_host_bytes(kTensorContextBytes)) {
    LOG_WARN("[GGML-QWEN-FULL-MEM] host graph context refused: %.2f MiB "
             "(reserved=%.2f GiB, budget=%.2f GB)\n",
             (double)kTensorContextBytes / (1024.0 * 1024.0),
             (double)MemoryGuard::reserved_host_bytes() / 1.0e9,
             (double)MemoryGuard::host_budget_bytes() / 1.0e9);
    return false;
  }
  g.context_reservation_bytes = kTensorContextBytes;
  g.ctx = ggml_init(ip);
  if (!g.ctx) {
    MemoryGuard::release_host_bytes(g.context_reservation_bytes);
    g.context_reservation_bytes = 0;
    return false;
  }
  g.graph = ggml_new_graph_custom(g.ctx, kGraphSize, false);
  if (!g.graph)
    return false;

  g.input = host_input(g, g.ctx, GGML_TYPE_F32, {g.D, g.HC, g.S},
                       static_cast<size_t>(g.D) * g.HC * g.S * sizeof(float),
                       "qwen_full_input");
  g.positions = host_input(g, g.ctx, GGML_TYPE_I32, {g.S},
                           static_cast<size_t>(g.S) * sizeof(int32_t),
                           "qwen_full_positions");
  if (g.decode_mode) {
    g.attn_mask = host_input(
        g, g.ctx, GGML_TYPE_F16, {g.kv_capacity, 1, 1, 1},
        static_cast<size_t>(g.kv_capacity) * sizeof(ggml_fp16_t),
        "qwen_full_decode_mask");
    g.decode_indices = host_input(
        g, g.ctx, GGML_TYPE_I32, {g.Hkv},
        static_cast<size_t>(g.Hkv) * sizeof(int32_t),
        "qwen_full_decode_kv_indices");
  } else {
    const int64_t kv_len = (int64_t)g.pos_base + g.S;
    g.attn_mask = host_input(
        g, g.ctx, GGML_TYPE_F16, {kv_len, g.S},
        static_cast<size_t>(kv_len) * g.S * sizeof(ggml_fp16_t),
        "qwen_full_causal_mask");
  }
  if (!g.input || !g.positions || !g.attn_mask ||
      (g.decode_mode && !g.decode_indices))
    return false;
  auto *pos = static_cast<int32_t *>(g.positions->data);
  auto *mask = static_cast<ggml_fp16_t *>(g.attn_mask->data);
  for (int i = 0; i < g.S; ++i)
    pos[i] = g.decode_mode ? i : g.pos_base + i;
  if (g.decode_mode) {
    for (int k = 0; k < g.kv_capacity; ++k)
      mask[k] = ggml_fp32_to_fp16(0.0f);
    auto *indices = static_cast<int32_t *>(g.decode_indices->data);
    // k_cache/v_cache are [hd, kv_capacity * Hkv]; the later
    // ggml_reshape_4d(..., hd, kv_capacity, Hkv, 1) makes kv_capacity the
    // FASTER-varying of the two combined dims, so a flat row index r
    // decomposes as (head = r / kv_capacity, pos = r % kv_capacity) --
    // head is the outer index, position the inner one. Must write
    // h * kv_capacity + pos, not pos * Hkv + h (this is the pos_base=0
    // case of that same formula).
    for (int h = 0; h < g.Hkv; ++h)
      indices[h] = h * g.kv_capacity;
  } else {
    const int kv_len = g.pos_base + g.S;
    for (int q = 0; q < g.S; ++q)
      for (int k = 0; k < kv_len; ++k)
        mask[static_cast<size_t>(q) * kv_len + k] =
            ggml_fp32_to_fp16(k <= g.pos_base + q ? 0.0f : -INFINITY);
  }

  ggml_tensor *res = g.input;
  if (graph_match_llama_cpp_topology(g)) {
    // The scheduler normally starts directly on the first GPU-weighted
    // operation because g.input is already a CPU buffer.  An explicit CPU
    // CONT makes the initial CPU->GPU handoff visible as the same leading
    // CPU/GPU split that llama.cpp has around model.input_embed/hc_init.
    res = ggml_cont(g.ctx, res);
    if (!res)
      return false;
    ggml_set_name(res, "qwen_full_llama_cpp_cpu_prologue");
    g.topology_cpu_prologue = res;
    if (g.sched) {
      // In host-node mode every CPU MoE boundary is represented by a
      // CUDA-owned CPY marker.  Keep the initial contiguous handoff on the
      // same CUDA split as those markers; otherwise this otherwise harmless
      // topology node leaves a second scheduler split in the graph.  The
      // default path remains llama.cpp's original CPU prologue assignment.
      ggml_backend_sched_set_tensor_backend(
          g.sched, res,
          env_qwen4exp_host_node_enabled() &&
                  !env_host_node_keep_cpu_prologue()
              ? g.cuda
              : g.cpu);
    }
  }
  // Decode-only router-usage tap (see Qwen4ExpUsageTrackContext's own
  // comment). Every layer that goes through build_moe() below appends its
  // real routed ids into g.usage_track_ids via an ordinary ggml_concat node;
  // a single flush marker after the loop reads the whole buffer once.
  const bool usage_track_enabled =
      g.decode_mode && g.engine->qwen4exp_track_router_usage;
  for (int l = 0; l < g.n_layers; ++l) {
    const std::string p = "blk." + std::to_string(l) + ".";
    if (qwen4exp_is_ple_layer(c, l)) {
      if (!ple_embedding)
        return false;
      g.ple_embedding =
          host_input(g, g.ctx, GGML_TYPE_F32,
                     {c.ple_n_embd_per_layer * c.ple_n_heads, g.S},
                     static_cast<size_t>(c.ple_n_embd_per_layer) *
                         c.ple_n_heads * g.S * sizeof(float),
                     "qwen_full_ple_embedding");
      if (!g.ple_embedding)
        return false;
      std::memcpy(g.ple_embedding->data, ple_embedding->data(),
                  ple_embedding->size() * sizeof(float));
      g.ple_layer = l;
      // PLE graph is constructed below with the same equations as llama.cpp's
      // build_ple.  The input embedding is host gathered, while its learned
      // projections and the rest of the operation stay in this graph.
      ggml_tensor *ple_key = resident_tensor(g, g.ctx, p + "ple_key.weight");
      ggml_tensor *ple_value =
          resident_tensor(g, g.ctx, p + "ple_value.weight");
      ggml_tensor *ple_nk =
          resident_tensor(g, g.ctx, p + "ple_norm_key.weight");
      ggml_tensor *ple_nq =
          resident_tensor(g, g.ctx, p + "ple_norm_query.weight");
      ggml_tensor *ple_nc =
          resident_tensor(g, g.ctx, p + "ple_norm_conv.weight");
      ggml_tensor *ple_conv =
          resident_tensor(g, g.ctx, p + "ple_conv1d.weight");
      if (!ple_key || !ple_value || !ple_nk || !ple_nq || !ple_nc || !ple_conv)
        return false;
      ggml_tensor *pk = ggml_mul_mat(g.ctx, ple_key, g.ple_embedding);
      ggml_tensor *pv = ggml_mul_mat(g.ctx, ple_value, g.ple_embedding);
      auto group_norm = [&](ggml_tensor *x, ggml_tensor *w) {
        x = ggml_reshape_3d(g.ctx, x, g.D, g.HC, g.S);
        x = ggml_rms_norm(g.ctx, x, c.eps);
        x = ggml_reshape_2d(g.ctx, x, g.D * g.HC, g.S);
        x = ggml_mul(g.ctx, x, w);
        return ggml_reshape_3d(g.ctx, x, g.D, g.HC, g.S);
      };
      pk = group_norm(pk, ple_nk);
      ggml_tensor *pq = group_norm(res, ple_nq);
      ggml_tensor *score =
          ggml_scale(g.ctx, ggml_sum_rows(g.ctx, ggml_mul(g.ctx, pk, pq)),
                     1.0f / std::sqrt(static_cast<float>(g.D)));
      ggml_tensor *gate = ggml_sigmoid(
          g.ctx,
          ggml_mul(g.ctx, ggml_sgn(g.ctx, score),
                   ggml_sqrt(g.ctx, ggml_clamp(g.ctx, ggml_abs(g.ctx, score),
                                               1e-6f, INFINITY))));
      ggml_tensor *v3 = ggml_repeat_4d(
          g.ctx, ggml_reshape_3d(g.ctx, pv, g.D, 1, g.S), g.D, g.HC, g.S, 1);
      ggml_tensor *gated = ggml_mul(g.ctx, v3, gate);
      ggml_tensor *ple_conv_grouped =
          ggml_reshape_3d(g.ctx, gated, g.D, g.HC, g.S);
      ggml_tensor *ple_conv_rms =
          ggml_rms_norm(g.ctx, ple_conv_grouped, c.eps);
      ple_conv_rms =
          ggml_reshape_2d(g.ctx, ple_conv_rms, g.D * g.HC, g.S);
      ggml_tensor *normed = ggml_mul(g.ctx, ple_conv_rms, ple_nc);

      const int hist = (c.ple_conv_kernel - 1) * c.ple_ngram_size;
      ggml_tensor *ph = nullptr;
      if (g.decode_mode) {
        ph = new_device_state_tensor(
            g, GGML_TYPE_F32, {hist, g.D * g.HC, 1},
            "qwen_full_decode_ple_history");
        g.ple_hist_storage = ph;
      } else {
        ph = host_input(
            g, g.ctx, GGML_TYPE_F32, {hist, g.D * g.HC, 1},
            static_cast<size_t>(hist) * g.D * g.HC * sizeof(float),
            "qwen_full_ple_history");
      }
      if (!ph)
        return false;
      if (!g.decode_mode)
        g.ple_hist_input = static_cast<float *>(ph->data);
      if (!g.decode_mode &&
          l < static_cast<int>(g.engine->qwen4exp_ple_conv_history.size())) {
        auto &h = g.engine->qwen4exp_ple_conv_history[l];
        if (h.size() == static_cast<size_t>(hist) * g.D * g.HC)
          std::memcpy(ph->data, h.data(), h.size() * sizeof(float));
      }
      ggml_tensor *padded =
          ggml_concat(g.ctx, ph, ggml_transpose(g.ctx, normed), 0);
      ggml_tensor *conv_out = nullptr;
      for (int k = 0; k < c.ple_conv_kernel; ++k) {
        const int start = hist - (c.ple_conv_kernel - 1 - k) * c.ple_ngram_size;
        ggml_tensor *shifted = ggml_cont(
            g.ctx,
            ggml_transpose(g.ctx,
                           ggml_view_3d(g.ctx, padded, g.S, g.D * g.HC, 1,
                                        padded->nb[1], padded->nb[2],
                                        ggml_row_size(padded->type, start))));
        ggml_tensor *wk = ggml_cont(
            g.ctx, ggml_view_2d(g.ctx, ple_conv, 1, g.D * g.HC, ple_conv->nb[1],
                                static_cast<size_t>(k) * ple_conv->nb[0]));
        wk = ggml_reshape_1d(g.ctx, wk, g.D * g.HC);
        if (wk->type != GGML_TYPE_F32)
          wk = ggml_cast(g.ctx, wk, GGML_TYPE_F32);
        ggml_tensor *term = ggml_mul(g.ctx, shifted, wk);
        conv_out = conv_out ? ggml_add(g.ctx, conv_out, term) : term;
      }
      ggml_tensor *ple_conv_raw = conv_out;
      conv_out =
          ggml_reshape_3d(g.ctx, ggml_silu(g.ctx, conv_out), g.D, g.HC, g.S);
      res = ggml_add(g.ctx, res, ggml_add(g.ctx, gated, conv_out));
      g.ple_hist_out = ggml_cont(
          g.ctx, ggml_view_3d(g.ctx, padded, hist, g.D * g.HC, 1, padded->nb[1],
                              padded->nb[2], ggml_row_size(padded->type, g.S)));
      if (g.decode_mode) {
        g.ple_hist_update =
            ggml_cpy(g.ctx, g.ple_hist_out, g.ple_hist_storage);
      }
    }

    const auto t_attn_start = std::chrono::steady_clock::now();
    ggml_tensor *inject = nullptr;
    // Declared here (not inside the guarded block below) because the FFN
    // section further down reassigns these same four names to its own
    // hc_ffn_* weights -- must stay in scope across the
    // phase2_suffix_skip_attn guard.
    ggml_tensor *hn = nullptr, *hdn = nullptr, *hup = nullptr, *hinj = nullptr;
  hn = resident_tensor(g, g.ctx, p + "hc_attn_norm.weight");
  hdn = resident_tensor(g, g.ctx, p + "hc_attn_down.weight");
  hup = resident_tensor(g, g.ctx, p + "hc_attn_up.weight");
  hinj = resident_tensor(g, g.ctx, p + "hc_attn_inject.weight");
  if (!hn || !hdn || !hup || !hinj)
    return false;
  ggml_tensor *attn_in =
      hc_mix(g, g.ctx, res, hn, hdn, hup, hinj, &inject);
  ggml_tensor *attn_out = nullptr;
  if (qwen4exp_is_gdn_layer(c, l)) {
    ggml_tensor *qkv = resident_tensor(g, g.ctx, p + "attn_qkv.weight");
    ggml_tensor *zg = resident_tensor(g, g.ctx, p + "attn_gate.weight");
    ggml_tensor *al = resident_tensor(g, g.ctx, p + "ssm_alpha.weight");
    ggml_tensor *be = resident_tensor(g, g.ctx, p + "ssm_beta.weight");
    ggml_tensor *dt = resident_tensor(g, g.ctx, p + "ssm_dt.bias");
    ggml_tensor *aa = resident_tensor(g, g.ctx, p + "ssm_a");
    ggml_tensor *cv = resident_tensor(g, g.ctx, p + "ssm_conv1d.weight");
    ggml_tensor *nm = resident_tensor(g, g.ctx, p + "ssm_norm.weight");
    ggml_tensor *oo = resident_tensor(g, g.ctx, p + "ssm_out.weight");
    if (!qkv || !zg || !al || !be || !dt || !aa || !cv || !nm || !oo)
      return false;
    const int hist = g.d_conv - 1;
    ggml_tensor *hi = nullptr;
    ggml_tensor *si = nullptr;
    if (g.decode_mode || g.prefill_device_state_inputs) {
      hi = new_device_state_tensor(g, GGML_TYPE_F32, {hist, g.conv_dim, 1},
                                    "qwen_full_decode_gdn_history");
      si = new_device_state_tensor(
          g, GGML_TYPE_F32, {g.gdn_hd, g.gdn_hd, g.Hv_gdn, 1},
          "qwen_full_decode_gdn_state");
      g.gdn_hist_storage.push_back(hi);
      g.gdn_state_storage.push_back(si);
    } else {
      hi = host_input(
          g, g.ctx, GGML_TYPE_F32, {hist, g.conv_dim, 1},
          static_cast<size_t>(hist) * g.conv_dim * sizeof(float),
          "qwen_full_gdn_history");
      si = host_input(
          g, g.ctx, GGML_TYPE_F32, {g.gdn_hd, g.gdn_hd, g.Hv_gdn, 1},
          static_cast<size_t>(g.gdn_hd) * g.gdn_hd * g.Hv_gdn *
              sizeof(float),
          "qwen_full_gdn_state");
    }
    if (!hi || !si)
      return false;
    if (g.decode_mode) {
      g.gdn_hist_updates.push_back(nullptr);
      g.gdn_state_updates.push_back(nullptr);
    } else if (!g.prefill_device_state_inputs) {
      g.gdn_hist_inputs.push_back(static_cast<float *>(hi->data));
      g.gdn_state_inputs.push_back(static_cast<float *>(si->data));
      if (l < static_cast<int>(g.engine->qwen4exp_conv_history.size()) &&
          g.engine->qwen4exp_conv_history[l].size() ==
              static_cast<size_t>(hist) * g.conv_dim)
        std::memcpy(hi->data, g.engine->qwen4exp_conv_history[l].data(),
                    static_cast<size_t>(hist) * g.conv_dim * sizeof(float));
      if (l < static_cast<int>(g.engine->qwen4exp_recur_state.size()) &&
          g.engine->qwen4exp_recur_state[l].size() ==
              static_cast<size_t>(g.gdn_hd) * g.gdn_hd * g.Hv_gdn)
        // qwen4exp_recur_state is host-legacy layout; si->data feeds the
        // ggml op directly and needs native layout. See
        // gdn_state_transpose_host.
        gdn_state_transpose_host(static_cast<float *>(si->data),
                                 g.engine->qwen4exp_recur_state[l].data(),
                                 g.gdn_hd, g.Hv_gdn);
    }
    ggml_tensor *so = nullptr, *ho = nullptr;
    attn_out = build_gdn(g, g.ctx, attn_in, l, qkv, zg, al, be, dt, aa, cv,
                         nm, oo, hi, si, &so, &ho);
    if (g.decode_mode) {
      g.gdn_state_updates.back() =
          ggml_cpy(g.ctx, so, g.gdn_state_storage.back());
      g.gdn_hist_updates.back() =
          ggml_cpy(g.ctx, ho, g.gdn_hist_storage.back());
    } else {
      g.gdn_state_out.push_back(so);
      g.gdn_hist_out.push_back(ho);
    }
    g.gdn_layers.push_back(l);
  } else {
    ggml_tensor *wq = dense_weight(g, g.ctx, l, "attn_q.weight");
    ggml_tensor *wk = dense_weight(g, g.ctx, l, "attn_k.weight");
    ggml_tensor *wv = dense_weight(g, g.ctx, l, "attn_v.weight");
    ggml_tensor *wqn = dense_weight(g, g.ctx, l, "attn_q_norm.weight");
    ggml_tensor *wkn = dense_weight(g, g.ctx, l, "attn_k_norm.weight");
    ggml_tensor *wo = dense_weight(g, g.ctx, l, "attn_output.weight");
    if (!wq || !wk || !wv || !wqn || !wkn || !wo)
      return false;
    if (g.decode_mode) {
      ggml_tensor *k_cache = new_device_state_tensor(
          g, GGML_TYPE_F32, {g.hd, g.kv_capacity * g.Hkv},
          "qwen_full_decode_kv_k");
      ggml_tensor *v_cache = new_device_state_tensor(
          g, GGML_TYPE_F32, {g.hd, g.kv_capacity * g.Hkv},
          "qwen_full_decode_kv_v");
      if (!k_cache || !v_cache)
        return false;
      g.kv_state_k.push_back(k_cache);
      g.kv_state_v.push_back(v_cache);
      attn_out = build_full_attention_decode(
          g, g.ctx, attn_in, l, wq, wk, wv, wqn, wkn, wo, k_cache, v_cache);
    } else {
      ggml_tensor *k_cache = nullptr;
      ggml_tensor *v_cache = nullptr;
      attn_out = build_full_attention(g, g.ctx, attn_in, l, wq, wk, wv, wqn,
                                      wkn, wo, &k_cache, &v_cache);
      if (!k_cache || !v_cache)
        return false;
      g.kv_k.push_back(k_cache);
      g.kv_v.push_back(v_cache);
    }
    g.kv_layers.push_back(l);
  }
  res = hc_combine(g, g.ctx, res, attn_out, inject);
  g.attn_construction_ms_total += std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t_attn_start).count();

    const auto t_moe_start = std::chrono::steady_clock::now();
    inject = nullptr;
    hn = resident_tensor(g, g.ctx, p + "hc_ffn_norm.weight");
    hdn = resident_tensor(g, g.ctx, p + "hc_ffn_down.weight");
    hup = resident_tensor(g, g.ctx, p + "hc_ffn_up.weight");
    hinj = resident_tensor(g, g.ctx, p + "hc_ffn_inject.weight");
    if (!hn || !hdn || !hup || !hinj)
      return false;
    ggml_tensor *ffn_in =
        hc_mix(g, g.ctx, res, hn, hdn, hup, hinj, &inject);
    if (!ffn_in)
      return false;
    ggml_tensor *router = resident_tensor(g, g.ctx, p + "ffn_gate_inp.weight");
    ggml_tensor *bias = resident_tensor(g, g.ctx, p + "exp_probs_b.bias");
    if (!router)
      return false;
    if (zeromoe::graphcfg::env("ZEROMOE_QWEN4EXP_PROBE_DUMP")) {
      ggml_tensor *tap_x = ggml_cont(g.ctx, ffn_in);
      ggml_tensor *tap_p = ggml_soft_max(g.ctx, ggml_mul_mat(g.ctx, router, ffn_in));
      ggml_set_output(tap_x);
      ggml_set_output(tap_p);
      ggml_build_forward_expand(g.graph, tap_x);
      ggml_build_forward_expand(g.graph, tap_p);
      g.probe_taps.emplace_back(l, tap_x, tap_p);
    }
    if (!g.decode_mode && l == g.engine->qwen4exp_partial_layer && bias &&
        partial_prefill_capture_enabled())
      g.partial_layer_bias_out = bias;

    // The per-expert split path still needs separate gate/up tensors because
    // its two branches have different expert-axis remaps.  For ordinary
    // whole-layer CPU/GPU placement, prefer the adjacent gate+up view: on
    // graph-RAM layers this is zero-copy and reduces three routed matmuls to
    // two, matching llama.cpp's reference graph.
    const auto split_it = g.split_gpu_expert_ids.find(l);
    const bool split_planned =
        g.S == 1 && split_it != g.split_gpu_expert_ids.end() &&
        !split_it->second.empty();
    ggml_tensor *moe = nullptr;

    // (Phase 1 static partial-layer GPU commit and the Phase 2 dynamic
    // dispatch were removed together with the conditional-node graphs.)

    // Prefill-only: capture this layer's routing `ids` for a real per-prompt
    // readback in qwen4exp_full_graph_run, so decode can decide whether to
    // commit to the path above. Never set outside prefill.
    const bool partial_capture_here =
        !g.decode_mode && l == g.engine->qwen4exp_partial_layer &&
        partial_prefill_capture_enabled();
    ggml_tensor **partial_ids_out =
        partial_capture_here ? &g.partial_layer_ids_out : nullptr;
    ggml_tensor **partial_probs_out =
        partial_capture_here ? &g.partial_layer_probs_out : nullptr;

    // Decode-only router-usage tap: captures this layer's real routed ids
    // via build_moe's own ids_out param. Safe to fold into partial_ids_out's
    // slot -- usage_track_enabled implies g.decode_mode, and partial_ids_out
    // above is only ever non-null when !g.decode_mode, so the two never
    // collide.
    ggml_tensor *layer_ids_for_usage = nullptr;
    ggml_tensor **ids_out_param =
        usage_track_enabled ? &layer_ids_for_usage : partial_ids_out;
    if (!moe && !split_planned && env_merged_gate_up_enabled()) {
      const auto t_wlookup_start = std::chrono::steady_clock::now();
      const bool gate_up_device =
          graph_expert_projection_resident(g, l, "ffn_gate_exps.weight");
      ggml_tensor *gate_up =
          expert_gate_up_weight(g, g.ctx, l, gate_up_device);
      ggml_tensor *down = expert_weight(g, g.ctx, l,
                                        "ffn_down_exps.weight");
      g.moe_weight_lookup_ms_total +=
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - t_wlookup_start)
              .count();
      if (gate_up && down) {
        moe = build_moe(g, g.ctx, ffn_in, l, router, bias, nullptr, nullptr,
                        down, nullptr, nullptr, nullptr, nullptr, gate_up,
                        ids_out_param, partial_probs_out, false,
                        nullptr, nullptr);
      }
    }

    if (!moe) {
      if (g.exl3_bridge_failed)
        return false;
      const auto t_wlookup_start = std::chrono::steady_clock::now();
      SplitExpertWeight ge =
          expert_weight_split(g, g.ctx, l, "ffn_gate_exps.weight");
      SplitExpertWeight ue =
          expert_weight_split(g, g.ctx, l, "ffn_up_exps.weight");
      SplitExpertWeight de =
          expert_weight_split(g, g.ctx, l, "ffn_down_exps.weight");
      g.moe_weight_lookup_ms_total +=
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - t_wlookup_start)
              .count();
      if (!ge.cpu || !ue.cpu || !de.cpu) {
        LOG_ERROR("[GGML-QWEN-FULL-ARENA-BIND-FAIL] expert-weight "
                  "layer=%d gate=%p up=%p down=%p\n",
                  l, static_cast<void *>(ge.cpu), static_cast<void *>(ue.cpu),
                  static_cast<void *>(de.cpu));
        return false;
      }
      // v1 scope: uniform M across gate/up/down for a layer -- only treat
      // this as a real split if ALL THREE projections' gpu tensor succeeded
      // (expert_weight_split() already falls back to null on any failure).
      const bool split_ok = ge.gpu && ue.gpu && de.gpu;
      const std::vector<int> *split_ids =
          split_ok ? &g.split_gpu_expert_ids.at(l) : nullptr;
      moe = build_moe(g, g.ctx, ffn_in, l, router, bias, ge.cpu, ue.cpu,
                      de.cpu, split_ok ? ge.gpu : nullptr,
                      split_ok ? ue.gpu : nullptr,
                      split_ok ? de.gpu : nullptr, split_ids, nullptr,
                      ids_out_param, partial_probs_out, false,
                      nullptr, nullptr);
    }
    if (g.exl3_bridge_failed)
      return false;
    if (usage_track_enabled && layer_ids_for_usage) {
      // Ordinary device-side accumulation -- ggml_concat inherits CUDA
      // placement from layer_ids_for_usage's argsort_top_k ancestor, no
      // manual backend pin needed (unlike the host-node CPY markers, this
      // is a totally normal graph op). ggml_cont matches the same
      // defensive-robustness convention build_qwen4exp_host_node's
      // route_ids already uses, even though decode S==1 rows never hit the
      // argsort-view-stride pitfall that convention guards against.
      ggml_tensor *ids_cont = ggml_cont(g.ctx, layer_ids_for_usage);
      g.usage_track_ids = g.usage_track_ids
                              ? ggml_concat(g.ctx, g.usage_track_ids, ids_cont, 0)
                              : ids_cont;
    }
    ggml_tensor *sg =
        resident_tensor(g, g.ctx, p + "ffn_gate_inp_shexp.weight");
    ggml_tensor *sge = resident_tensor(g, g.ctx, p + "ffn_gate_shexp.weight");
    ggml_tensor *sup = resident_tensor(g, g.ctx, p + "ffn_up_shexp.weight");
    ggml_tensor *sdo = resident_tensor(g, g.ctx, p + "ffn_down_shexp.weight");
    if (sge && sup && sdo) {
      const auto early = g.early_ready.find(l);
      ggml_tensor *shared =
          build_shared(g, g.ctx,
                       early != g.early_ready.end() ? early->second->gpu_input : ffn_in,
                       l, sge, sup, sdo, sg);
      if (early != g.early_ready.end())
        ggml_build_forward_expand(g.graph, shared);
      moe = ggml_add(g.ctx, moe, shared);
    }
    const auto early_output = g.early_ready.find(l);
    if (early_output != g.early_ready.end() && early_output->second->cpu_marker) {
      // The private GPU leaf is filled by the CPU hook. Preserve the explicit
      // fork -> hot/shared -> CPU -> merge order; its upload joins this stream.
      ggml_build_forward_expand(g.graph, early_output->second->cpu_marker);
    }
    res = hc_combine(g, g.ctx, res, moe, inject);
    g.moe_construction_ms_total += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_moe_start).count();
  }

  // Decode-only router-usage tap: one flush marker for the whole graph,
  // built once here after every layer has appended into g.usage_track_ids
  // (see the per-layer ggml_concat block above and
  // Qwen4ExpUsageTrackContext's own comment). A construction failure here
  // is non-fatal -- it just means this session's usage histogram stays
  // incomplete, never that decode itself should fail.
  if (usage_track_enabled && g.usage_track_ids) {
    const int total_ids = c.topk * g.n_layers;
    auto usage_ctx = std::make_unique<Qwen4ExpUsageTrackContext>();
    usage_ctx->engine = g.engine;
    usage_ctx->n_layers = g.n_layers;
    usage_ctx->n_experts = c.n_experts;
    usage_ctx->topk = c.topk;
    usage_ctx->ids_bytes = static_cast<size_t>(total_ids) * sizeof(int32_t);
    usage_ctx->hot_mask.assign(g.n_layers, {});
    for (const auto &[layer, ids] : g.split_gpu_expert_ids) {
      if (layer < 0 || layer >= g.n_layers)
        continue;
      usage_ctx->hot_mask[layer].assign(c.n_experts, 0);
      for (int e : ids)
        if (e >= 0 && e < c.n_experts)
          usage_ctx->hot_mask[layer][e] = 1;
    }
    usage_ctx->host_ids = static_cast<int32_t *>(
        alloc_registered_host(g, usage_ctx->ids_bytes));
    if (usage_ctx->host_ids) {
      std::fill_n(usage_ctx->host_ids, total_ids, 0);
      ggml_tensor *usage_dummy_dst =
          ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, total_ids);
      ggml_tensor *usage_marker =
          usage_dummy_dst ? ggml_cpy(g.ctx, g.usage_track_ids, usage_dummy_dst)
                          : nullptr;
      if (usage_marker) {
        usage_marker->extra = usage_ctx.get();
        ggml_set_name(usage_marker, "qwen_usage_track_flush_marker");
        if (g.sched) {
          ggml_backend_sched_set_tensor_backend(g.sched, usage_dummy_dst,
                                                g.cuda);
          ggml_backend_sched_set_tensor_backend(g.sched, usage_marker, g.cuda);
        }
        ggml_set_output(usage_marker);
        ggml_build_forward_expand(g.graph, usage_marker);
        g.usage_track_context = std::move(usage_ctx);
        LOG_INFO("[QWEN4EXP-USAGE-TRACK] flush marker captured: n_layers=%d "
                 "topk=%d total_ids=%d\n",
                 g.n_layers, c.topk, total_ids);
      } else {
        LOG_WARN("[QWEN4EXP-USAGE-TRACK] marker construction failed; usage "
                 "tracking disabled for this graph\n");
      }
    } else {
      LOG_WARN("[QWEN4EXP-USAGE-TRACK] pinned host buffer allocation "
               "failed; usage tracking disabled for this graph\n");
    }
  }

  {
  ggml_tensor *on = resident_tensor(g, g.ctx, "output_hc_norm.weight");
  ggml_tensor *od = resident_tensor(g, g.ctx, "output_hc_down.weight");
  ggml_tensor *ou = resident_tensor(g, g.ctx, "output_hc_up.weight");
  if (!on || !od || !ou)
    return false;
  g.final_streams = res;
  g.output = hc_mix(g, g.ctx, res, on, od, ou, nullptr, nullptr);
  if (!g.decode_mode) {
    ggml_set_output(g.final_streams);
    ggml_build_forward_expand(g.graph, g.final_streams);
  }
  ggml_set_output(g.output);
  ggml_build_forward_expand(g.graph, g.output);
  if (g.decode_mode || qwen4exp_full_graph_decode_enabled()) {
    ggml_tensor *lm_in = g.decode_mode
                            ? g.output
                            : ggml_view_2d(g.ctx, g.output, g.D, 1,
                                           g.output->nb[1],
                                           (g.S - 1) * g.output->nb[1]);
    ggml_tensor *out_norm =
        resident_tensor(g, g.ctx, "output_norm.weight");
    if (out_norm) {
      lm_in = ggml_mul(
          g.ctx,
          ggml_rms_norm(g.ctx, lm_in,
                        g.engine->model_file.architecture_config.eps),
          out_norm);
    }
    ggml_tensor *out_weight =
        resident_tensor(g, g.ctx, "output.weight");
    if (!out_weight)
      return false;
    g.logits = ggml_mul_mat(g.ctx, out_weight, lm_in);
    ggml_set_output(g.logits);
    ggml_build_forward_expand(g.graph, g.logits);
  }
  }
  if (!g.decode_mode) {
    for (ggml_tensor *t : g.kv_k) {
      ggml_set_output(t);
      ggml_build_forward_expand(g.graph, t);
    }
    for (ggml_tensor *t : g.kv_v) {
      ggml_set_output(t);
      ggml_build_forward_expand(g.graph, t);
    }
    for (ggml_tensor *t : g.gdn_state_out) {
      ggml_set_output(t);
      ggml_build_forward_expand(g.graph, t);
    }
    for (ggml_tensor *t : g.gdn_hist_out) {
      ggml_set_output(t);
      ggml_build_forward_expand(g.graph, t);
    }
    if (g.ple_hist_out) {
      ggml_set_output(g.ple_hist_out);
      ggml_build_forward_expand(g.graph, g.ple_hist_out);
    }
    if (g.partial_layer_ids_out) {
      ggml_set_output(g.partial_layer_ids_out);
      ggml_build_forward_expand(g.graph, g.partial_layer_ids_out);
    }
    if (g.partial_layer_probs_out) {
      ggml_set_output(g.partial_layer_probs_out);
      ggml_build_forward_expand(g.graph, g.partial_layer_probs_out);
    }
    if (g.partial_layer_bias_out) {
      ggml_set_output(g.partial_layer_bias_out);
      ggml_build_forward_expand(g.graph, g.partial_layer_bias_out);
    }
  } else {
    // State copies are graph nodes, but are intentionally not graph outputs:
    // decode keeps GDN/KV/PLE state on the CUDA backend between submissions.
    for (size_t i = 0; i < g.gdn_state_updates.size(); ++i) {
      ggml_tensor *t = g.gdn_state_updates[i];
      if (t) {
        ggml_build_forward_expand(g.graph, t);
      }
    }
    for (size_t i = 0; i < g.gdn_hist_updates.size(); ++i) {
      ggml_tensor *t = g.gdn_hist_updates[i];
      if (t) {
        ggml_build_forward_expand(g.graph, t);
      }
    }
    if (g.ple_hist_update)
      ggml_build_forward_expand(g.graph, g.ple_hist_update);
  }
  // Real, measured incident (2026-09-09): ggml_backend_sched_alloc_graph's
  // internal ggml_gallocr_reserve_n_impl did not fail gracefully on a real
  // cudaMalloc OOM on llama.cpp's older ggml fork -- it logged "cudaMalloc
  // failed: out of memory" and then segfaulted the whole process (exit 139)
  // instead of returning a checkable failure, at the exact point where
  // main.cpp's own fallback VRAM cache reservation (a fixed 4GB, unaware of
  // device_expert_layers' own VRAM use) left the scheduler's own reserve()
  // short by ~1.8GB. main.cpp now sizes that fallback cache against real
  // free VRAM and this family's own device_expert_layers estimate -- THAT
  // was the actual root cause and is already fixed there. This check stays
  // as a second, independent line of defense: refuse to even attempt the
  // allocator call when real free VRAM is below a floor, and decline to the
  // legacy path instead of risking the same crash again.
  //
  // 2026-09-15: tried twice to replace this flat constant with something
  // smarter -- both reverted, see below -- then settled on this smaller,
  // still-flat, but now EVIDENCE-BACKED value instead of an arbitrary one.
  // (1) Lowering 2 GiB -> 512 MiB: crashed at device_expert_layers=24
  // (needed >=650.24 MiB there -- the exact size of the single buffer
  // ggml_gallocr_reserve_n_impl tried and failed to cudaMalloc). (2)
  // Querying the REAL requirement via ggml_backend_sched_reserve_size() (a
  // safe, no_alloc=true measure pass that cannot itself OOM) reported
  // 0.640 GiB at N=22 -- consistent with N=24's 0.635 GiB failure point
  // above, i.e. the real requirement is roughly STABLE around ~0.64 GiB,
  // not scaling dramatically with layer count as first assumed from just
  // those two data points. But reusing that measure call's split_graph
  // pass on the SAME g.graph object that a real
  // ggml_backend_sched_alloc_graph() call re-splits moments later leaves
  // something behind that a plain ggml_backend_sched_reset() in between
  // does NOT fully clear: real-hardware repro, twice, with and without that
  // reset -- a hard `GGML_ASSERT(src0->type == src1->type)` abort inside
  // ggml_cuda_op_concat on the FIRST real graph compute afterward (prefill,
  // S=7). Do not retry live-measuring this way without either a genuinely
  // separate measure-graph object (llama.cpp's own convention -- never
  // reuse/re-split the exact cgraph that will actually execute) or a
  // from-scratch ggml_backend_sched for the real run after measuring on a
  // throwaway one.
  //
  // Given the two real measurements above both land at ~0.63-0.64 GiB,
  // 1 GiB keeps a real ~1.5-1.6x margin over observed need (vs 2 GiB's
  // ~3.1x, which cost every device_expert_layers config ~1 GiB of pure
  // waste) while recovering enough headroom for device_expert_layers=22
  // to fit (was declining with only 1.83 GiB free). Re-widen if a real
  // OOM/crash resurfaces at some untested N; don't lower below ~0.8 GiB
  // without a fresh measurement -- these two points aren't enough to prove
  // it never goes higher for a different N or model.
  {
    // The scheduler allocator is not a safe place to discover that a static
    // model weight lost its external binding: on older ggml-cuda revisions it
    // will reinterpret the whole unbound expert subtree as a CUDA workspace
    // and attempt an allocation roughly equal to the complete MoE payload
    // (~86 GiB for this model), then may crash instead of returning failure.
    // This can surface only after repeated prompt graph rebuilds, when a
    // stale backend assignment or buffer view is carried across teardown.
    int unbound_static_weights = 0;
    int misplaced_host_weights = 0;
    size_t unbound_static_bytes = 0;
    // GGML may wrap a resident weight in more than one view (for example
    // view -> contiguous -> reshaped for PLE convolution).  The outer view
    // does not necessarily carry the backend buffer itself; walk the full
    // chain before classifying a static tensor as unbound.  A one-hop check
    // falsely rejected these valid weights and sent every build to fallback.
    auto view_buffer = [](ggml_tensor *tensor) -> ggml_backend_buffer_t {
      for (int depth = 0; tensor && depth < 32; ++depth) {
        if (tensor->buffer)
          return tensor->buffer;
        tensor = tensor->view_src;
      }
      return nullptr;
    };
    for (int i = 0; i < ggml_graph_n_nodes(g.graph); ++i) {
      ggml_tensor *node = ggml_graph_node(g.graph, i);
      if (!node)
        continue;
      for (int j = 0; j < GGML_MAX_SRC; ++j) {
        ggml_tensor *src = node->src[j];
        if (!src)
          continue;
        const char *name = ggml_get_name(src);
        const std::string tensor_name = name ? name : "";
        // Operation results can inherit a source weight's display name
        // (the PLE path produces names such as "...weight (cont)" and
        // "...weight (reshaped)").  They are activation/intermediate
        // tensors, not static model-storage leaves, and are expected to be
        // allocated by the graph allocator.  Only GGML_OP_NONE tensors can
        // represent a missing pre-bound model weight here.
        if (src->op != GGML_OP_NONE)
          continue;
        const bool named_static_weight =
            tensor_name.find(".weight") != std::string::npos ||
            tensor_name.find(".bias") != std::string::npos ||
            tensor_name.find("_exps") != std::string::npos;
        if (!named_static_weight)
          continue;
        ggml_backend_buffer_t buffer = view_buffer(src);
        if (!buffer) {
          ++unbound_static_weights;
          if (unbound_static_bytes <= SIZE_MAX - ggml_nbytes(src))
            unbound_static_bytes += ggml_nbytes(src);
          if (unbound_static_weights <= 8)
            LOG_ERROR(
                "[GGML-QWEN-FULL] refusing graph allocation: unbound static "
                "weight name=%s bytes=%zu\n",
                tensor_name.c_str(), ggml_nbytes(src));
          continue;
        }
        if (ggml_backend_buffer_is_host(buffer) &&
            ggml_backend_sched_get_tensor_backend(g.sched, src) == g.cuda &&
            tensor_name.find("_exps") != std::string::npos) {
          ++misplaced_host_weights;
          if (misplaced_host_weights <= 8)
            LOG_ERROR(
                "[GGML-QWEN-FULL] refusing graph allocation: host expert "
                "weight assigned to CUDA name=%s\n",
                tensor_name.c_str());
        }
      }
    }
    if (unbound_static_weights > 0 || misplaced_host_weights > 0) {
      LOG_ERROR(
          "[GGML-QWEN-FULL] static binding validation failed: unbound=%d "
          "(%.2f GiB), misplaced_host_experts=%d; skipping unsafe allocator "
          "reserve\n",
          unbound_static_weights,
          static_cast<double>(unbound_static_bytes) /
              (1024.0 * 1024.0 * 1024.0),
          misplaced_host_weights);
      return false;
    }

    size_t free_b = 0, total_b = 0;
    int prev_device = -1;
    cudaGetDevice(&prev_device);
    cudaSetDevice(g.device);
    const cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
    if (prev_device >= 0)
      cudaSetDevice(prev_device);
    size_t kMinGraphAllocHeadroomBytes = 1ull * 1024ull * 1024ull * 1024ull;
    // Opt-in override of the admission floor (MiB, 64..4096). The default is
    // unchanged. The measured S=1 decode workspace is ~0.13 GiB, so a large
    // Option-B arena that leaves 0.5-1 GiB free can still build; the caller
    // owns the risk of the allocator reserve itself running out of memory.
    if (const char *floor_env =
            zeromoe::graphcfg::env("ZEROMOE_QWEN4EXP_GRAPH_ALLOC_FLOOR_MIB")) {
      char *end = nullptr;
      const long mib = std::strtol(floor_env, &end, 10);
      if (end != floor_env && *end == '\0' && mib >= 64 && mib <= 4096) {
        kMinGraphAllocHeadroomBytes =
            static_cast<size_t>(mib) * 1024ull * 1024ull;
        LOG_WARN("[GGML-QWEN-FULL] graph allocator admission floor overridden "
                 "to %ld MiB by ZEROMOE_QWEN4EXP_GRAPH_ALLOC_FLOOR_MIB\n",
                 mib);
      }
    }
    // Size check against the batch: the workspace is requested in one piece and scales with the prompt (no-allocation
    // sizing pass: 0.13 GiB at S=1, 12.38 GiB at S=1611, i.e. ~7.8 MiB per token; pinned host buffer ~1.1 MiB per token).
    // A request the device cannot satisfy must never reach cudaMalloc (an allocator failure here has crashed the box
    // before, see above), so it is refused with the batch size that would fit, which the caller retries with.
    if (!g.decode_mode && g.S > 1 && err == cudaSuccess) {
      constexpr double kGiB = 1024.0 * 1024.0 * 1024.0, kMiB = 1024.0 * 1024.0;
      // The decode graph is built right after prefill; its workspace is ~0.12 GiB, and the allocator admission floor
      // above is 1 GiB, so that much stays out of what the prefill batch may use. (6 GiB here refused even a 16-token
      // batch on configs that keep many expert layers resident.)
      // Device workspace: see estimate_prefill_workspace_bytes (shared with manage_decode_arena_for_prefill).
      const bool oo = g_qwen4exp_prefill_op_offload;
      const double fixed = prefill_fixed_bytes(oo), per_token = prefill_per_token_bytes(oo);
      const double need = (double)estimate_prefill_workspace_bytes(g.S, oo);
      const double host_avail =
          std::max(0.0, (double)MemoryGuard::read_available_mb() * kMiB);
      const double host_need = 20.0 * kGiB + 1.2 * kMiB * g.S;
      int fit = g.S;
      if (need > (double)free_b)
        fit = (int)std::max(0.0, ((double)free_b / 1.05 - fixed) / per_token);
      if (host_avail > 0 && host_avail < host_need)
        fit = 0;
      if (fit < g.S) {
        fit = (int)(fit * 0.9) / 8 * 8; // headroom: later batches see slightly less free memory than this one
        g_qwen4exp_suggested_batch = fit >= 8 ? fit : 0;
        LOG_ERROR("[GGML-QWEN-FULL] refusing graph allocator reserve before any allocation: a batch of %d tokens "
                  "needs ~%.2f GiB of device workspace and %.1f GiB of host memory; %.2f GiB VRAM free, %.1f GiB "
                  "host available%s\n",
                  g.S, need / kGiB, host_need / kGiB, (double)free_b / kGiB, host_avail / kGiB,
                  g_qwen4exp_suggested_batch > 0
                      ? ("; a batch of " + std::to_string(g_qwen4exp_suggested_batch) + " tokens fits").c_str()
                      : "");
        return false;
      }
      g_qwen4exp_suggested_batch = 0;
    }
    if (err != cudaSuccess || free_b < kMinGraphAllocHeadroomBytes) {
      LOG_ERROR(
          "[GGML-QWEN-FULL] refusing graph allocator reserve: device %d "
          "free=%.2f GiB "
          "< %.2f GiB safety floor (device expert arena + dense arena already "
          "committed %.2f GiB) -- falling back to the legacy path\n",
          g.device,
          err == cudaSuccess
              ? static_cast<double>(free_b) / (1024.0 * 1024.0 * 1024.0)
              : -1.0,
          static_cast<double>(kMinGraphAllocHeadroomBytes) /
              (1024.0 * 1024.0 * 1024.0),
          static_cast<double>(g.expert.capacity +
                              g.dense.capacity) /
              (1024.0 * 1024.0 * 1024.0));
      // Signal to a caller that wants one: THIS specific failure (scheduler
      // admission floor, not some other construction error) may be fixed by
      // shrinking device_expert_layers and rebuilding from scratch -- see
      // this flag's own comment for why init_device_expert_arena's existing
      // allow_layer_clamp retry doesn't already cover it (that check runs
      // BEFORE the scheduler's own workspace is known to be needed).
      if (err == cudaSuccess && g.device_expert_layers > 0)
        g_qwen4exp_build_graph_retry_smaller_expert_layers = true;
      return false;
    }
  }
  // Safe allocator-size probe for diagnosing unexpectedly large graph
  // workspaces. reserve_size() performs GGML's no_alloc sizing pass, but it
  // also rebuilds the scheduler's internal split graph. Do not reuse this
  // FullGraph for a real allocation/compute after probing; decline it and let
  // the owning prepare path destroy it. This is intentionally opt-in because
  // it disables the full-graph instance for the diagnostic run.
  const char *reserve_diag =
      zeromoe::graphcfg::env("ZEROMOE_QWEN4EXP_GRAPH_RESERVE_DIAG");
  const bool reserve_diag_enabled =
      reserve_diag &&
      ((std::strcmp(reserve_diag, "1") == 0 ||
        std::strcmp(reserve_diag, "all") == 0) ||
       (std::strcmp(reserve_diag, "decode") == 0 && g.S == 1));
  if (reserve_diag_enabled) {
    const int n_backends = ggml_backend_sched_get_n_backends(g.sched);
    std::vector<size_t> backend_sizes(static_cast<size_t>(n_backends), 0);
    ggml_backend_sched_reserve_size(g.sched, g.graph, backend_sizes.data());
    LOG_WARN("[GGML-QWEN-FULL-RESERVE-DIAG] no-allocation sizing only; "
             "declining this graph instance (S=%d splits=%d copies=%d)\n",
             g.S, ggml_backend_sched_get_n_splits(g.sched),
             ggml_backend_sched_get_n_copies(g.sched));
    {
      // Largest intermediate tensors of the graph, to see what the workspace is made of.
      std::vector<std::pair<size_t, int>> big;
      for (int i = 0; i < ggml_graph_n_nodes(g.graph); ++i) {
        ggml_tensor *n = ggml_graph_node(g.graph, i);
        if (n && n->op != GGML_OP_NONE && n->op != GGML_OP_VIEW && n->op != GGML_OP_RESHAPE &&
            n->op != GGML_OP_PERMUTE && n->op != GGML_OP_TRANSPOSE)
          big.push_back({ggml_nbytes(n), i});
      }
      std::sort(big.rbegin(), big.rend());
      double total = 0;
      for (auto &b : big) total += (double)b.first;
      LOG_WARN("[GGML-QWEN-FULL-RESERVE-DIAG] %zu non-view nodes, sum of their sizes %.2f GiB (S=%d)\n", big.size(),
               total / (1024.0 * 1024.0 * 1024.0), g.S);
      for (size_t k = 0; k < big.size() && k < 14; ++k) {
        ggml_tensor *n = ggml_graph_node(g.graph, big[k].second);
        LOG_WARN("[GGML-QWEN-FULL-RESERVE-DIAG] top%zu node %d %.3f GiB op=%s name=%s type=%s ne=[%lld,%lld,%lld,%lld] src0=%s\n", k,
                 big[k].second, (double)big[k].first / (1024.0 * 1024.0 * 1024.0), ggml_op_name(n->op), n->name,
                 ggml_type_name(n->type), (long long)n->ne[0], (long long)n->ne[1], (long long)n->ne[2],
                 (long long)n->ne[3], n->src[0] ? n->src[0]->name : "-");
      }
    }
    for (int i = 0; i < n_backends; ++i) {
      ggml_backend_t backend = ggml_backend_sched_get_backend(g.sched, i);
      ggml_backend_buffer_type_t buft =
          ggml_backend_sched_get_buffer_type(g.sched, backend);
      const double gib = static_cast<double>(backend_sizes[i]) /
                         (1024.0 * 1024.0 * 1024.0);
      LOG_WARN("[GGML-QWEN-FULL-RESERVE-DIAG] backend=%s buffer=%s "
               "required=%.3f GiB (%zu bytes)\n",
               ggml_backend_name(backend), ggml_backend_buft_name(buft), gib,
               backend_sizes[i]);
    }
    return false;
  }
  const double t_construction_ms = std::chrono::duration<double, std::milli>(
                                       std::chrono::steady_clock::now() -
                                       t_build_graph_start)
                                       .count();
  const auto t_alloc_start = std::chrono::steady_clock::now();
  for (const auto &entry : g.early_ready) {
    if (!entry.second->stage_target) continue;
    const auto producer = g.early_ready.find(entry.first - 1);
    GGML_ASSERT(producer != g.early_ready.end() && producer->second->prediction_marker &&
                producer->second->prediction_layer == entry.first);
    producer->second->stage_source = entry.second->stage_target;
  }
  if (!ggml_backend_sched_alloc_graph(g.sched, g.graph))
    return false;
  const double t_alloc_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() -
                                t_alloc_start)
                                .count();
  LOG_INFO("[GGML-QWEN-FULL-BUILD-TIME] node_construction=%.1fms "
           "(of which dense_cudaMemcpy=%.1fms/%d tensors, "
           "expert_cudaMemcpy=%.1fms/%ld calls, attn_block=%.1fms, "
           "moe_block=%.1fms [weight_lookup=%.1fms]) sched_alloc_graph=%.1fms\n",
           t_construction_ms, g.dense_upload_ms_total,
           g.dense.tensors, g.expert_upload_ms_total,
           g.expert_upload_calls, g.attn_construction_ms_total,
           g.moe_construction_ms_total, g.moe_weight_lookup_ms_total,
           t_alloc_ms);
  LOG_INFO("[GGML-QWEN-FULL-BUILD-WEIGHTS] split_calls=%.1fms "
           "load_full_weight=%.1fms metadata=%.1fms "
           "ram_source=%.1fms host_bind=%.1fms/%ld tensors "
           "device_bind=%.1fms/%ld tensors (H2D is reported above) "
           "vram_stage_alloc=%.1fms vram_file_read=%.1fms/%ld calls/%.2f GiB\n",
           g.expert_weight_split_ms_total,
           g.expert_weight_load_ms_total,
           g.expert_weight_meta_ms_total, g.expert_weight_source_ms_total,
           g.expert_weight_host_bind_ms_total,
           g.expert_weight_host_bind_calls,
           g.expert_weight_device_bind_ms_total,
           g.expert_weight_device_bind_calls,
           g.expert_vram_stage_alloc_ms_total,
           g.expert_vram_file_read_ms_total,
           g.expert_vram_file_read_calls,
           static_cast<double>(g.expert_vram_file_read_bytes) / (1024.0 * 1024.0 * 1024.0));
  g.graph_bytes = ggml_backend_sched_get_buffer_size(g.sched, g.cuda);
  dump_qwen_moe_layout(g);

  // The graph is only useful for performance if the scheduler keeps most of
  // the dependency chain on CUDA.  A high split count can otherwise hide a
  // CPU fallback for an op or a view/copy boundary, which recreates the
  // per-layer host orchestration that this graph is meant to remove.
  int cuda_nodes = 0;
  int cpu_nodes = 0;
  int other_nodes = 0;
  int backend_transitions = 0;
  ggml_backend_t previous_backend = nullptr;
  std::unordered_map<std::string, int> non_cuda_ops;
  for (int i = 0; i < ggml_graph_n_nodes(g.graph); ++i) {
    ggml_tensor *node = ggml_graph_node(g.graph, i);
    ggml_backend_t backend =
        ggml_backend_sched_get_tensor_backend(g.sched, node);
    if (backend == g.cuda) {
      ++cuda_nodes;
    } else if (backend == g.cpu) {
      ++cpu_nodes;
      ++non_cuda_ops[ggml_op_name(node->op)];
    } else {
      ++other_nodes;
      ++non_cuda_ops[backend ? ggml_backend_name(backend) : "NULL"];
    }
    if (previous_backend && backend != previous_backend)
      ++backend_transitions;
    previous_backend = backend;
  }
  LOG_INFO("[GGML-QWEN-FULL] scheduler nodes cuda=%d cpu=%d other=%d "
           "transitions=%d\n",
           cuda_nodes, cpu_nodes, other_nodes, backend_transitions);
  if (!non_cuda_ops.empty()) {
    std::string detail;
    for (const auto &entry : non_cuda_ops) {
      if (!detail.empty())
        detail += ", ";
      detail += entry.first + "=" + std::to_string(entry.second);
    }
    LOG_INFO("[GGML-QWEN-FULL] non-CUDA node ops: %s\n", detail.c_str());
  }
  LOG_INFO("[GGML-QWEN-FULL] graph built layers=%d nodes=%d splits=%d "
           "workspace=%.2f GiB\n",
           g.n_layers, ggml_graph_n_nodes(g.graph),
           ggml_backend_sched_get_n_splits(g.sched),
           static_cast<double>(g.graph_bytes) / (1024.0 * 1024.0 * 1024.0));
  const FullGraph &dense_arena_owner =
      g.dense_source_graph ? *g.dense_source_graph : g;
  LOG_INFO(
      "[GGML-QWEN-FULL] dense device arena used=%.2f/%.2f GiB; %d tensors\n",
      static_cast<double>(dense_arena_owner.dense.used) /
          (1024.0 * 1024.0 * 1024.0),
      static_cast<double>(dense_arena_owner.dense.capacity) /
          (1024.0 * 1024.0 * 1024.0),
      g.dense.tensors);
  if (g.device_expert_layers > 0) {
    const FullGraph &expert_arena_owner =
        g.expert_source_graph ? *g.expert_source_graph : g;
    const int resident_layers = device_expert_resident_layer_count(g);
    const int complete_records = resident_layers * c.n_experts;
    LOG_INFO(
        "[GGML-QWEN-FULL] device expert arena used=%.2f/%.2f GiB; "
        "%d complete layer(s), %d expert records, all 3 projections resident\n",
        static_cast<double>(expert_arena_owner.expert.used) /
            (1024.0 * 1024.0 * 1024.0),
        static_cast<double>(expert_arena_owner.expert.capacity) /
            (1024.0 * 1024.0 * 1024.0),
        resident_layers, complete_records);
  }
  LOG_INFO(
      "[GGML-QWEN-FULL] host expert source=%.2f GiB graph-owned DDR copy; "
      "%.2f GiB aliased from preloaded DDR arena (%zu projection views)\n",
      static_cast<double>(g.host_weight_bytes) / (1024.0 * 1024.0 * 1024.0),
      static_cast<double>(g.host_ram_alias_bytes) / (1024.0 * 1024.0 * 1024.0),
      g.host_ram_alias_tensors);
  if (g.host_weight_bytes > 0) {
    LOG_INFO("[GGML-QWEN-FULL] host expert source pinned=%.2f GiB; "
             "pin failures=%zu\n",
             static_cast<double>(g.host_pinned_bytes) /
                 (1024.0 * 1024.0 * 1024.0),
             g.host_pin_failures);
  }
  LOG_INFO("[GGML-QWEN-FULL-MEM] host ledger=%.2f/%.2f GB; graph source="
           "%.2f GiB + context=%.2f MiB + owned-extra=%.2f MiB; device "
           "workspace=%.2f GiB (GPU, outside host_ram_gb)\n",
           (double)MemoryGuard::reserved_host_bytes() / 1.0e9,
           (double)MemoryGuard::host_budget_bytes() / 1.0e9,
           (double)g.host_weight_bytes / (1024.0 * 1024.0 * 1024.0),
           (double)g.context_reservation_bytes / (1024.0 * 1024.0),
           (double)(g.host_weight_bytes > g.host_reservation_bytes
                        ? g.host_weight_bytes - g.host_reservation_bytes
                        : 0) /
               (1024.0 * 1024.0),
           (double)g.graph_bytes / (1024.0 * 1024.0 * 1024.0));
  g.ready = true;
  return true;
}

bool sync_prefill_recurrent_state_to_host(FullGraph &g) {
  if (g.decode_mode || !g.ready || !g.engine)
    return false;
  StreamingEngine &engine = *g.engine;
  const size_t state_n = static_cast<size_t>(g.gdn_hd) * g.gdn_hd * g.Hv_gdn;
  const size_t hist_n = static_cast<size_t>(g.d_conv - 1) * g.conv_dim;
  std::vector<float> native_state(state_n);
  for (size_t index = 0; index < g.gdn_layers.size(); ++index) {
    const int layer = g.gdn_layers[index];
    if (index >= g.gdn_state_out.size() || index >= g.gdn_hist_out.size() ||
        layer < 0 ||
        layer >= static_cast<int>(engine.qwen4exp_recur_state.size()) ||
        layer >= static_cast<int>(engine.qwen4exp_conv_history.size()) ||
        engine.qwen4exp_recur_state[layer].size() != state_n ||
        engine.qwen4exp_conv_history[layer].size() != hist_n)
      return false;
    ggml_backend_tensor_get(g.gdn_state_out[index], native_state.data(), 0,
                            state_n * sizeof(float));
    gdn_state_transpose_host(engine.qwen4exp_recur_state[layer].data(),
                             native_state.data(), g.gdn_hd, g.Hv_gdn);
    ggml_backend_tensor_get(g.gdn_hist_out[index],
                            engine.qwen4exp_conv_history[layer].data(), 0,
                            hist_n * sizeof(float));
  }
  g.prefill_host_state_deferred = false;
  return true;
}

bool sync_decode_state_to_host(FullGraph &g) {
  if (!g.decode_mode || !g.ready || !g.engine)
    return true;
  StreamingEngine &engine = *g.engine;
  const size_t state_n = static_cast<size_t>(g.gdn_hd) * g.gdn_hd * g.Hv_gdn;
  const size_t hist_n = static_cast<size_t>(g.d_conv - 1) * g.conv_dim;
  for (size_t i = 0; i < g.gdn_layers.size(); ++i) {
    const int l = g.gdn_layers[i];
    if (i >= g.gdn_state_storage.size() || i >= g.gdn_hist_storage.size() ||
        l < 0 || l >= static_cast<int>(engine.qwen4exp_recur_state.size()) ||
        l >= static_cast<int>(engine.qwen4exp_conv_history.size()) ||
        engine.qwen4exp_recur_state[l].size() != state_n ||
        engine.qwen4exp_conv_history[l].size() != hist_n)
      return false;
    // gdn_state_storage is native layout (it feeds the decode self-loop
    // directly); qwen4exp_recur_state is host-legacy layout for legacy CPU
    // consumption. See gdn_state_transpose_host.
    std::vector<float> native_state(state_n);
    ggml_backend_tensor_get(g.gdn_state_storage[i], native_state.data(), 0,
                            state_n * sizeof(float));
    gdn_state_transpose_host(engine.qwen4exp_recur_state[l].data(),
                             native_state.data(), g.gdn_hd, g.Hv_gdn);
    ggml_backend_tensor_get(g.gdn_hist_storage[i],
                            engine.qwen4exp_conv_history[l].data(), 0,
                            hist_n * sizeof(float));
  }
  const size_t row = static_cast<size_t>(g.Hkv) * g.hd;
  const int rows = std::min(std::max(0, g.decode_last_pos + 1), g.kv_capacity);
  for (size_t i = 0; i < g.kv_layers.size(); ++i) {
    const int l = g.kv_layers[i];
    if (i >= g.kv_state_k.size() || i >= g.kv_state_v.size() || l < 0 ||
        l >= static_cast<int>(engine.Kc.size()) ||
        l >= static_cast<int>(engine.Vc.size()) || !engine.Kc[l] ||
        !engine.Vc[l])
      return false;
    const size_t bytes = static_cast<size_t>(rows) * row * sizeof(float);
    if (bytes > 0) {
      ggml_backend_tensor_get(g.kv_state_k[i], engine.Kc[l], 0, bytes);
      ggml_backend_tensor_get(g.kv_state_v[i], engine.Vc[l], 0, bytes);
    }
  }
  if (g.ple_layer >= 0 && g.ple_hist_storage) {
    const auto &c = engine.model_file.architecture_config;
    const size_t ple_n = static_cast<size_t>(c.ple_conv_kernel - 1) *
                         c.ple_ngram_size * g.HC * g.D;
    if (g.ple_layer >= static_cast<int>(engine.qwen4exp_ple_conv_history.size()) ||
        engine.qwen4exp_ple_conv_history[g.ple_layer].size() != ple_n)
      return false;
    ggml_backend_tensor_get(g.ple_hist_storage,
                            engine.qwen4exp_ple_conv_history[g.ple_layer].data(),
                            0, ple_n * sizeof(float));
  }
  g.decode_host_state_dirty = false;
  return true;
}

// Build the state-address table used when the next decode graph follows a
// completed full-graph prefill.  GDN and PLE recurrent outputs have identical
// shapes in the prefill and decode graphs, so their already-computed CUDA
// storage can be bound directly.  Prefill KV outputs cover only S rows while
// decode allocates the full context capacity; reserve null entries for those
// tensors so new_device_state_tensor() allocates them normally.
bool prepare_prefill_state_handoff(FullGraph &source) {
  source.state_handoff_ready = false;
  source.state_handoff_addrs.clear();
  source.state_handoff_buffers.clear();
  source.state_handoff_cursor = 0;
  source.state_handoff_tensors.clear();
  if (source.decode_mode || !source.sched || !source.graph ||
      source.gdn_layers.empty())
    return false;

  const auto &c = source.engine->model_file.architecture_config;
  size_t gdn_index = 0;
  size_t kv_index = 0;
  size_t compatible = 0;
  size_t expected = 0;
  auto add = [&](ggml_tensor *t, bool compatible_shape) {
    ggml_backend_buffer_t buffer = nullptr;
    for (ggml_tensor *storage = t; storage; storage = storage->view_src) {
      if (storage->buffer) {
        buffer = storage->buffer;
        break;
      }
    }
    if (compatible_shape)
      ++expected;
    if (compatible_shape && t && t->data && buffer &&
        ggml_is_contiguous(t) && !ggml_backend_buffer_is_host(buffer)) {
      source.state_handoff_addrs.push_back(t->data);
      source.state_handoff_buffers.push_back(buffer);
      ++compatible;
    } else {
      source.state_handoff_addrs.push_back(nullptr);
      source.state_handoff_buffers.push_back(nullptr);
    }
  };

  for (int layer = 0; layer < source.n_layers; ++layer) {
    if (layer == source.ple_layer)
      add(source.ple_hist_out, true);
    if (qwen4exp_is_gdn_layer(c, layer)) {
      if (gdn_index >= source.gdn_state_out.size() ||
          gdn_index >= source.gdn_hist_out.size())
        return false;
      add(source.gdn_hist_out[gdn_index], true);
      add(source.gdn_state_out[gdn_index], true);
      ++gdn_index;
    } else {
      // Decode KV tensors are larger than prefill KV tensors.  Keep the
      // source position in the table, but force private decode allocation.
      if (kv_index >= source.kv_k.size() || kv_index >= source.kv_v.size())
        return false;
      add(nullptr, false);
      add(nullptr, false);
      ++kv_index;
    }
  }
  if (compatible == 0 || compatible != expected ||
      source.state_handoff_addrs.size() !=
                              source.state_handoff_buffers.size()) {
    source.state_handoff_addrs.clear();
    source.state_handoff_buffers.clear();
    LOG_WARN("[GGML-QWEN-FULL-STATE-HANDOFF] unavailable: %zu/%zu "
             "recurrent/PLE tensors have contiguous device storage; "
             "using host readback\n", compatible, expected);
    return false;
  }
  source.state_handoff_ready = true;
  LOG_INFO("[GGML-QWEN-FULL-DECODE] prefill state handoff: %zu/%zu "
           "recurrent/PLE tensors alias the completed prefill graph; KV "
           "state remains decode-owned\n",
           compatible, source.state_handoff_addrs.size());
  return true;
}

} // namespace qwen4exp_graph

#endif // ZEROMOE_USE_GGML_CUDA
