// PerfCounters::dump_report sections (split from perf_timer.hpp).
#include "perf_timer.hpp"

void PerfCounters::dump_report_part4(double session_elapsed_s) const {
    (void)session_elapsed_s;
    {
        uint64_t used = n_active_layer_prewarm_used.load(), wasted = n_active_layer_prewarm_wasted.load();
        uint64_t total_pw = used + wasted;
        LOG_INFO("Active-layer prewarm waste: %llu/%llu (%.1f%%) speculatively pre-warmed slot(s) "
                 "(decode_async_prewarm + prefill's Phase-2 batch warm-up) reclaimed WITHOUT any real "
                 "dispatch ever touching them first -- %.2f GB moved for nothing. Answers whether "
                 "narrowing this feature's own eagerness (fewer, higher-confidence candidates) is worth "
                 "pursuing before any further transfer-mechanism tuning -- see t_active_layer_upload_"
                 "async's own comment for why two such mechanism-level fixes were already tried and "
                 "reverted here.\n",
                 (unsigned long long)wasted, (unsigned long long)total_pw,
                 total_pw > 0 ? 100.0 * wasted / total_pw : 0.0,
                 bytes_active_layer_prewarm_wasted.load() / 1e9);
    }
    {
        uint64_t attempted = n_pilot_active_fill_attempted.load();
        uint64_t used = n_pilot_active_fill_used.load(), wasted = n_pilot_active_fill_wasted.load();
        uint64_t total_pf = used + wasted;
        if (attempted > 0 || n_pilot_active_fill_already_present.load() > 0
            || n_pilot_active_fill_already_main_cache.load() > 0) {
            LOG_INFO("Pilot speculative active-layer fill (speculative_promote_to_active_layer): "
                     "%lu attempted / %lu already active-layer-resident (skipped) / %lu already main-"
                     "cache-resident (skipped) / %lu pool-full / %lu cross-pilot dedup skipped -- of "
                     "%lu reclaimed, %lu/%lu (%.1f%%) were NEVER touched by a real dispatch before "
                     "reclaim (%.2f GB moved for nothing). This is the direct measurement of whether the "
                     "predicted expert was actually used, for THIS mechanism specifically -- unlike main-"
                     "cache promotion, a wrong guess here costs nothing beyond this cheap, self-contained "
                     "active-layer pool (never evicts a real main-cache resident).\n",
                     (unsigned long)attempted,
                     (unsigned long)n_pilot_active_fill_already_present.load(),
                     (unsigned long)n_pilot_active_fill_already_main_cache.load(),
                     (unsigned long)n_pilot_active_fill_pool_full.load(),
                     (unsigned long)n_pilot_active_fill_dedup_skipped.load(),
                     (unsigned long)total_pf, (unsigned long)wasted, (unsigned long)total_pf,
                     total_pf > 0 ? 100.0 * wasted / total_pf : 0.0,
                     bytes_pilot_active_fill_wasted.load() / 1e9);
        }
        {
            // Direct answer to "of the MoEs promoted at rank K
            // specifically, what fraction were actually used" -- NOT
            // the same question as the pilot's own per-rank prediction-
            // accuracy metric (n_pilot_ram_only_hits_by_rank) above,
            // which is measured regardless of whether promotion or use
            // ever happened. See ExpertLocation::promoted_at_rank's own
            // comment.
            std::string rank_used_line = "Pilot active-layer fill used-rate by rank: ";
            bool any_rank = false;
            for (int r = 0; r < 8; ++r) {
                uint64_t ru = n_pilot_active_fill_used_by_rank[r].load();
                uint64_t rw = n_pilot_active_fill_wasted_by_rank[r].load();
                uint64_t rt = ru + rw;
                if (rt == 0) continue;
                any_rank = true;
                char buf[64];
                snprintf(buf, sizeof(buf), "k%d=%.1f%%(n=%lu) ", r, 100.0 * ru / rt, (unsigned long)rt);
                rank_used_line += buf;
            }
            if (any_rank) LOG_INFO("%s\n", rank_used_line.c_str());
        }
        uint64_t mkb_d1 = n_speculative_max_k_binding_depth1.load();
        uint64_t mkb_at = n_speculative_max_k_binding_attn.load();
        uint64_t mkb_total = mkb_d1 + mkb_at;
        if (attempted > 0 || mkb_total > 0) {
            LOG_INFO("speculative_active_layer_max_k binding check: %lu rank(s) approved by the tier/"
                     "threshold structure (dynamic_layer_top_k) but CLIPPED by max_k (%lu DEPTH-1, %lu "
                     "ATTN-PILOT) vs %lu actually attempted -- %.1f%% of tier-approved ranks were blocked "
                     "by max_k specifically. Near-0%% means max_k ISN'T the real binding constraint at its "
                     "current value (the tier/threshold structure already caps most candidates first) -- "
                     "raising max_k further wouldn't be expected to promote meaningfully more.\n",
                     (unsigned long)mkb_total, (unsigned long)mkb_d1, (unsigned long)mkb_at,
                     (unsigned long)attempted,
                     (attempted + mkb_total) > 0 ? 100.0 * mkb_total / (attempted + mkb_total) : 0.0);
        }
    }
    LOG_INFO("Active-layer upload (H2D): %8.3f s  (%6.2f GB/s, %.2f GB total) -- upload_expert_active's "
             "own cudaHostRegister conversion, same Alt-A treatment as 'Host register stage' above but a "
             "SEPARATE call site (the active-layer pool, not the main VRAM cache) -- %llu registration "
             "failure(s), %llu stream-0 fallback(s), %llu cudaMemcpy failure(s) (all should be ~0)\n",
             t_active_layer_host_register.load(),
             t_active_layer_host_register > 0
                 ? (bytes_active_layer_host_registered / 1e9) / t_active_layer_host_register : 0.0,
             bytes_active_layer_host_registered / 1e9,
             (unsigned long long)n_active_layer_host_register_fail.load(),
             (unsigned long long)n_active_layer_stream_unavailable.load(),
             (unsigned long long)n_active_layer_upload_fail.load());
    {
        uint64_t ai = n_active_layer_uploads_async_issued.load();
        uint64_t as = n_active_layer_uploads_async_synced.load();
        uint64_t anr = n_active_layer_prewarm_not_ram_ready.load();
        if (ai > 0 || as > 0 || anr > 0) {
            LOG_INFO("Active-layer BATCH pre-warm (async): %llu issued / %llu synced, issue=%.3fs "
                     "sync=%.3fs (should sum to LESS than an equivalent population's plain "
                     "'Active-layer upload (H2D)'-style sync cost if batching is actually amortizing "
                     "per-call overhead) / %llu skipped (not yet RAM-resident, Phase 3's reactive path "
                     "handled them as before)\n",
                     (unsigned long long)ai, (unsigned long long)as,
                     t_active_layer_upload_async.load(), t_active_layer_batch_sync.load(),
                     (unsigned long long)anr);
        }
    }
    {
        uint64_t qc = n_qabsorb_gpu_calls.load();
        uint64_t qf = n_qabsorb_gpu_fallback_cpu.load();
        if (qc > 0 || qf > 0) {
            double up = t_qabsorb_gpu_up.load(), cmp = t_qabsorb_gpu_compute.load(), dn = t_qabsorb_gpu_down.load();
            LOG_INFO("MLA qabsorb BF16 batched GPU path (EngineConfig::attn_qabsorb_gpu): %llu call(s), "
                     "up=%.3fs cmp=%.3fs dn=%.3fs (%.3fms/call avg) / %llu CPU fallback(s) (this layer's "
                     "WUK BF16 upload never succeeded -- should track toward 0 once warm_mla_absorb_cache "
                     "finishes at startup)\n",
                     (unsigned long long)qc, up, cmp, dn,
                     qc > 0 ? ((up + cmp + dn) * 1000.0 / qc) : 0.0,
                     (unsigned long long)qf);
        }
    }
    {
        uint64_t sc = n_score_gpu_calls.load();
        uint64_t sf = n_score_gpu_fallback_cpu.load();
        if (sc > 0 || sf > 0) {
            double up = t_score_gpu_up.load(), cmp = t_score_gpu_compute.load(), dn = t_score_gpu_down.load();
            double wuv = t_score_gpu_wuv.load();
            LOG_INFO("MLA score GPU path (EngineConfig::attn_score_gpu): %llu call(s), "
                     "up=%.3fs cmp=%.3fs dn=%.3fs wuv=%.3fs (gpu %.3fms/call avg, wuv %.3fms/call avg) / "
                     "%llu CPU fallback(s) (S>1 prefill, or this layer's Lc/Rc GPU mirror never succeeded -- "
                     "should track toward 0 for decode once warm_mla_absorb_cache finishes at startup)\n",
                     (unsigned long long)sc, up, cmp, dn, wuv,
                     sc > 0 ? ((up + cmp + dn) * 1000.0 / sc) : 0.0,
                     sc > 0 ? (wuv * 1000.0 / sc) : 0.0,
                     (unsigned long long)sf);
        }
    }
    {
        uint64_t qc = n_qwen4exp_attn_gpu_calls.load();
        uint64_t qf = n_qwen4exp_attn_gpu_fallback_cpu.load();
        if (qc || qf) {
            double up = t_qwen4exp_attn_gpu_up.load();
            double cmp = t_qwen4exp_attn_gpu_cmp.load();
            double dn = t_qwen4exp_attn_gpu_dn.load();
            LOG_INFO("Qwen4Exp dense-attention GPU path (EngineConfig::qwen4exp_attn_gpu): "
                     "%llu call(s), up=%.3fs cmp=%.3fs dn=%.3fs (%.3fms/call avg) / "
                     "%llu CPU fallback(s) (a fallback means an alloc or cuBLAS call failed "
                     "and that layer used the byte-identical CPU loop -- should be 0)\n",
                     (unsigned long long)qc, up, cmp, dn,
                     qc ? (up + cmp + dn) * 1000.0 / (double)qc : 0.0,
                     (unsigned long long)qf);
        }
    }
    {
        uint64_t ga = n_expert_gpu_activation.load();
        if (ga) LOG_INFO("Expert SwiGLU on GPU: %llu call(s), %.3fs "
                         "(replaces the gate/up download + CPU silu/mul + product upload)\n",
                         (unsigned long long)ga, t_expert_gpu_activation.load());
        uint64_t wm = n_vram_watermark_refused.load();
        if (wm) LOG_INFO("VRAM watermark: %llu dynamic allocation(s) refused to stay "
                         "under the configured fraction of device memory\n",
                         (unsigned long long)wm);
        uint64_t sk = n_dense_bf16_skipped_vram.load();
        if (sk) LOG_INFO("Dense BF16 GEMM: %llu call(s) skipped for lack of VRAM "
                         "(fused kernel used instead)\n", (unsigned long long)sk);
    }
    {
        uint64_t mg = n_mem_guard_alloc_refused.load();
        long minmb = MemoryGuard::min_seen_mb();
        if (mg || MemoryGuard::block_events())
            LOG_INFO("Memory guard: %llu allocation(s) refused, %llu block event(s), "
                     "MemAvailable low-water %ld MB -- host_ram_gb is too large for this "
                     "model on this machine\n",
                     (unsigned long long)mg,
                     (unsigned long long)MemoryGuard::block_events(), minmb);
    }
    {
        uint64_t al = n_moe_gpu_accum_layers.load();
        uint64_t af = n_moe_gpu_accum_fallback.load();
        if (al || af)
            LOG_INFO("MoE GPU-side accumulation: %llu layer-download(s), dn=%.3fs "
                     "(replaces per-expert D2H) / %llu fallback(s) to CPU accumulate\n",
                     (unsigned long long)al, t_moe_accum_dn.load(),
                     (unsigned long long)af);
    }
    {
        uint64_t ph = n_prompt_cache_hits.load(), pm = n_prompt_cache_misses.load();
        if (ph || pm)
            LOG_INFO("Prompt cache: %llu hit(s) / %llu miss(es), %llu prompt token(s) "
                     "reused (prefix of a previous prompt, so not re-prefilled)\n",
                     (unsigned long long)ph, (unsigned long long)pm,
                     (unsigned long long)n_prompt_cache_tokens_reused.load());
    }
    {
        double a1=t_gdn_qkv_proj.load(), a2=t_gdn_conv.load(), a3=t_gdn_l2.load(),
               a4=t_gdn_gates.load(), a5=t_gdn_delta.load(), a6=t_gdn_outnorm.load(),
               a7=t_gdn_outproj.load();
        double af = t_gdn_full.load();
        if (a1+a2+a3+a4+a5+a6+a7 > 0.0)
            LOG_INFO("Qwen4Exp GDN layer breakdown: qkv+gate_proj=%.2fs conv=%.2fs "
                     "l2norm=%.2fs gates=%.2fs delta_rule=%.2fs out_norm=%.2fs "
                     "out_proj=%.2fs (sum=%.2fs; nested inside the GDN attention bucket)\n",
                     a1, a2, a3, a4, a5, a6, a7, a1+a2+a3+a4+a5+a6+a7);
        if (af > 0.0)
            LOG_INFO("Qwen4Exp full GDN GGML graph: %.2fs (complete device "
                     "projection+conv+delta+norm+ssm_out; nested inside GDN attention)\n",
                     af);
        const uint64_t gd = n_gdn_full_device_success.load();
        const uint64_t gh = n_gdn_full_host_success.load();
        const uint64_t gdf = n_gdn_full_device_failure.load();
        const uint64_t gs = n_gdn_full_split_fallback.load();
        if (gd || gh || gdf || gs)
            LOG_INFO("Qwen4Exp full GDN path: %llu device success / "
                     "%llu host success / %llu device failure / "
                     "%llu split fallbacks\n",
                     (unsigned long long)gd, (unsigned long long)gh,
                     (unsigned long long)gdf, (unsigned long long)gs);
    }
    {
        uint64_t gc = n_qwen4exp_gdn_gpu_calls.load();
        uint64_t gf = n_qwen4exp_gdn_gpu_fallback_cpu.load();
        if (gc || gf) {
            double up = t_qwen4exp_gdn_gpu_up.load();
            double cmp = t_qwen4exp_gdn_gpu_cmp.load();
            double dn = t_qwen4exp_gdn_gpu_dn.load();
            LOG_INFO("Qwen4Exp chunkwise GDN GPU path (EngineConfig::qwen4exp_gdn_gpu): "
                     "%llu call(s), up=%.3fs cmp=%.3fs dn=%.3fs (%.3fms/call avg) / "
                     "%llu CPU fallback(s)\n",
                     (unsigned long long)gc, up, cmp, dn,
                     gc ? (up + cmp + dn) * 1000.0 / (double)gc : 0.0,
                     (unsigned long long)gf);
        }
    }
    {
        uint64_t cc = n_gdn_conv1d_gpu_calls.load();
        uint64_t cf = n_gdn_conv1d_gpu_fallback_cpu.load();
        if (cc || cf) {
            double up = t_gdn_conv1d_gpu_up.load();
            double cmp = t_gdn_conv1d_gpu_cmp.load();
            double dn = t_gdn_conv1d_gpu_dn.load();
            LOG_INFO("GDN causal conv1d+SiLU GPU path (EngineConfig::qwen4exp_gdn_conv_gpu): "
                     "%llu call(s), up=%.3fs cmp=%.3fs dn=%.3fs (%.3fms/call avg) / "
                     "%llu CPU fallback(s)\n",
                     (unsigned long long)cc, up, cmp, dn,
                     cc ? (up + cmp + dn) * 1000.0 / (double)cc : 0.0,
                     (unsigned long long)cf);
        }
    }
    {
        uint64_t kc = n_glm5next_kda_gpu_calls.load();
        uint64_t kf = n_glm5next_kda_gpu_fallback_cpu.load();
        if (kc || kf) {
            double up = t_glm5next_kda_gpu_up.load();
            double cmp = t_glm5next_kda_gpu_cmp.load();
            double dn = t_glm5next_kda_gpu_dn.load();
            LOG_INFO("GLM5NEXT KDA delta-rule GPU path (EngineConfig::glm5next_kda_gpu): "
                     "%llu call(s), up=%.3fs cmp=%.3fs dn=%.3fs (%.3fms/call avg) / "
                     "%llu CPU fallback(s)\n",
                     (unsigned long long)kc, up, cmp, dn,
                     kc ? ((up + cmp + dn) * 1000.0 / (double)kc) : 0.0,
                     (unsigned long long)kf);
        }
    }
    {
        uint64_t rc = n_glm5next_router_gpu_calls.load();
        uint64_t rf = n_glm5next_router_gpu_fallback_cpu.load();
        if (rc || rf) {
            double up = t_glm5next_router_gpu_up.load();
            double cmp = t_glm5next_router_gpu_cmp.load();
            double dn = t_glm5next_router_gpu_dn.load();
            LOG_INFO("GLM5NEXT router GPU path (EngineConfig::glm5next_router_gpu): "
                     "%llu call(s), up=%.3fs cmp=%.3fs dn=%.3fs (%.3fms/call avg) / "
                     "%llu CPU fallback(s)\n",
                     (unsigned long long)rc, up, cmp, dn,
                     rc ? ((up + cmp + dn) * 1000.0 / (double)rc) : 0.0,
                     (unsigned long long)rf);
        }
    }
    {
        uint64_t dc = n_dsv4_score_gpu_calls.load();
        uint64_t df = n_dsv4_score_gpu_fallback_cpu.load();
        if (dc > 0 || df > 0) {
            double up = t_dsv4_score_gpu_up.load(), cmp = t_dsv4_score_gpu_compute.load(), dn = t_dsv4_score_gpu_down.load();
            LOG_INFO("DSV4 raw-SWA/HCA score GPU path (EngineConfig::dsv4_score_gpu): %llu call(s), "
                     "up=%.3fs cmp=%.3fs dn=%.3fs (%.3fms/call avg) / %llu CPU fallback(s) (S>1 prefill, "
                     "a CSA/ratio==4 layer -- always CPU by design, or this layer's KVc/HCA GPU mirror "
                     "never succeeded)\n",
                     (unsigned long long)dc, up, cmp, dn,
                     dc > 0 ? ((up + cmp + dn) * 1000.0 / dc) : 0.0,
                     (unsigned long long)df);
        }
    }
    {
        // 2026-08-29 RELABELED: this block reports n_wuv_gpu_calls /
        // n_wuv_gpu_fallback_cpu / t_wuv_gpu_up/compute/down -- these
        // are the LIVE counters incremented by wuv_bf16_batched_host
        // (cuda_manager.hpp), the proven-safe path actually wired into
        // production (attention_layer.cpp). The label used to say
        // "int8 batched... int8-persistent... never succeeded", left
        // over from the same-day int8-persistent WUV format experiment
        // that was tried and reverted (see CLAUDE.md). 2026-09-02
        // UPDATE: that path is no longer dead -- it is reachable via the
        // opt-in ZEROMOE_WUV_INT8=1 toggle (wuv_int8_enabled(),
        // attention_layer.cpp), and when enabled it increments the very
        // counters printed below, since both formats share one call
        // site. The int8-only counter (t_wuv_int8_dequant) is broken out
        // into its own line just underneath, so the two formats stay
        // distinguishable in a report rather than silently merging. The
        // stale label was actively misleading: it read as "0 of 78
        // layers covered, an int8 upload problem" when the real,
        // nonzero call/fallback counts below are the CURRENT BF16
        // path's own coverage gap (see the "score=" GPU coverage
        // investigation in CLAUDE.md for that gap's own history).
        uint64_t wc = n_wuv_gpu_calls.load();
        uint64_t wf = n_wuv_gpu_fallback_cpu.load();
        if (wc > 0 || wf > 0) {
            double up = t_wuv_gpu_up.load(), cmp = t_wuv_gpu_compute.load(), dn = t_wuv_gpu_down.load();
            LOG_INFO("WUV BF16 batched GPU path (score GPU path's down-projection, wuv_bf16_batched_host): "
                     "%llu call(s), up=%.3fs cmp=%.3fs dn=%.3fs (%.3fms/call avg) / %llu CPU fallback(s) "
                     "(this layer's WUV BF16 upload never succeeded -- VRAM budget/coverage gap, not an "
                     "error; should track toward 0 once warm_mla_absorb_cache finishes at startup and "
                     "budget allows; CPU-fallback cost is included in t_score_gpu_wuv above)\n",
                     (unsigned long long)wc, up, cmp, dn,
                     wc > 0 ? ((up + cmp + dn) * 1000.0 / wc) : 0.0,
                     (unsigned long long)wf);
            // 2026-09-02: the int8-persistent WUV variant shares every
            // counter above EXCEPT this one -- its per-call dequant
            // kernel, which re-expands the whole [H,kvl,vh] weight set
            // back to BF16 on EVERY call (the price of storing it at
            // half size). Printed separately because that cost is the
            // whole question when deciding whether int8's extra layer
            // coverage actually pays: it is charged to the GPU, while
            // what the extra coverage saves is charged to the CPU
            // fallback (t_score_gpu_wuv), so the two never net out
            // inside a single bucket. Stays silent on the BF16 path,
            // where the timer is genuinely zero.
            double i8dq = t_wuv_int8_dequant.load();
            if (i8dq > 0.0) {
                LOG_INFO("  ^ int8-persistent WUV variant active: per-call dequant kernel "
                         "%.3fs total (%.3fms/call avg over the %llu GPU call(s) above) -- "
                         "this is the GPU-side cost paid to buy the extra layer coverage that "
                         "reduces the CPU fallback count\n",
                         i8dq, wc > 0 ? (i8dq * 1000.0 / wc) : 0.0,
                         (unsigned long long)wc);
            }
        }
    }
}

void PerfCounters::dump_report_part5(double session_elapsed_s) const {
    (void)session_elapsed_s;
    uint64_t hits = n_expert_cache_hit, misses = n_expert_cache_miss;
    {
        uint64_t di = n_decode_async_prewarm_issued.load();
        uint64_t ds = n_decode_async_prewarm_synced.load();
        uint64_t dc = n_decode_async_prewarm_chunks.load();
        uint64_t dn = n_decode_async_prewarm_no_slot.load();
        uint64_t dst = n_decode_async_prewarm_stale_caught.load();
        if (di > 0 || ds > 0 || dc > 0 || dn > 0 || dst > 0) {
            LOG_INFO("Decode-time BATCH pre-warm (async, EngineConfig::decode_async_prewarm): "
                     "%llu issued / %llu synced in %llu chunk(s) (avg %.2f/chunk, capped at "
                     "decode_async_prewarm_max_batch), %llu skipped (active-layer pool full/alloc "
                     "failed -- fell through to dispatch_expert_k's existing synchronous path), "
                     "%llu stale/reclaimed slot(s) caught by the post-pin identity check (should be "
                     "~0 -- see that code's own comment; SEPARATE from the combined 'Active-layer "
                     "BATCH pre-warm' counters above, which this also feeds into but can't "
                     "distinguish from prefill's own contribution alone)\n",
                     (unsigned long long)di, (unsigned long long)ds, (unsigned long long)dc,
                     dc > 0 ? (double)di / (double)dc : 0.0,
                     (unsigned long long)dn, (unsigned long long)dst);
        }
    }
    LOG_INFO("Pilot RAM->VRAM promotion: %lu attempted / %lu found slot already gone / "
             "%lu skipped by cross-pilot dedup (DEPTH-1 and ATTN-PILOT both targeting the "
             "same l+1 expert this transition) / %lu (DEPTH-1) + %lu (ATTN-PILOT) vetoed by "
             "per-layer accuracy gate (ram_vram_promote_layer_gate) "
             "(ram_vram_promote_top_k/adaptive -- MOVE/COPY "
             "counts above include this path's successes too, shared with the reactive "
             "hit-count promotion)\n",
             (unsigned long)n_pilot_promote_attempted.load(),
             (unsigned long)n_pilot_promote_slot_gone.load(),
             (unsigned long)n_pilot_promote_dedup_skipped.load(),
             (unsigned long)n_pilot_promote_layer_gate_vetoed_depth1.load(),
             (unsigned long)n_pilot_promote_layer_gate_vetoed_attn.load());
    if (n_reactive_promote_routed_gpu1.load() > 0) {
        LOG_INFO("Reactive->GPU1 routing (Direction 1): %lu of the reactive (proven-hot) "
                 "promotions above were routed to the dual-role slice specifically, real "
                 "demand only, no speculation\n",
                 (unsigned long)n_reactive_promote_routed_gpu1.load());
    }
    if (n_predict_gpu1_promote_attempted.load() > 0 || n_predict_gpu1_cap_reached.load() > 0) {
        LOG_INFO("Predict->GPU1 promotion (Idea 1, predict_gpu1_rank_start/end): %lu attempted / "
                 "%lu found slot already gone / %lu held back by predict_gpu1_cap_per_layer "
                 "-- always COPY (never MOVE), always the dual-role slice specifically, see "
                 "RAMSafetyManager::promote_ram_to_vram_gpu1\n",
                 (unsigned long)n_predict_gpu1_promote_attempted.load(),
                 (unsigned long)n_predict_gpu1_promote_slot_gone.load(),
                 (unsigned long)n_predict_gpu1_cap_reached.load());
        LOG_INFO("Predict->GPU1 churn: %lu already-resident (no transfer) / %lu genuinely fresh "
                 "H2D upload -- fresh-upload count against the dual-role slice's own (small) slot "
                 "count is the direct measure of repeat-upload churn. TEMPORARY diagnostic, see "
                 "doc/gpu-utilization-vram-analysis.md.\n",
                 (unsigned long)n_predict_gpu1_already_resident.load(),
                 (unsigned long)n_predict_gpu1_fresh_upload.load());
    }
    LOG_INFO("Prefetch pickup:     avg %.2fms over %lu hints, peak queue backlog %lu "
             "(large delay + small backlog => worker CPU-starved; growing backlog => worker throughput-bound)\n",
             n_prefetch_hints_processed > 0
                 ? 1000.0 * t_prefetch_pickup_delay.load() / n_prefetch_hints_processed.load() : 0.0,
             (unsigned long)n_prefetch_hints_processed.load(),
             (unsigned long)n_prefetch_backlog_peak.load());
    {
        // Dropped hints are the unambiguous throughput-ceiling signal --
        // see n_prefetch_hints_dropped_queue_full's own comment. Printed
        // unconditionally (including the healthy 0) so its absence from a
        // report never has to be read as "the counter did not exist yet".
        uint64_t dropped = n_prefetch_hints_dropped_queue_full.load();
        uint64_t pushed  = n_prefetch_hints_processed.load() + dropped;
        LOG_INFO("Prefetch drops:      %lu hint(s) dropped, queue full (%.2f%% of %lu pushed) -- "
                 "each one is a guaranteed future cache miss; 0 is the healthy value, nonzero "
                 "means raise prefetch_worker_threads or lower prefetch_pilot_k\n",
                 (unsigned long)dropped,
                 pushed > 0 ? (100.0 * dropped / pushed) : 0.0,
                 (unsigned long)pushed);
    }
    {
        uint64_t hi = n_prefetch_hi_lane_processed.load(), lo = n_prefetch_lo_lane_processed.load();
        uint64_t starved = n_prefetch_lo_lane_starved_ticks.load();
        if (hi + lo > 0) {
            LOG_INFO("Prefetch lanes:      %lu HIGH (l+1/prefill) / %lu LOW (l+2) processed, "
                     "%lu tick(s) LOW was ready but skipped for HIGH (strict priority -- expected "
                     "to be nonzero whenever both lanes are active; watch for l+2 starvation if this "
                     "is a large fraction of lo-lane pushes)\n",
                     (unsigned long)hi, (unsigned long)lo, (unsigned long)starved);
        }
    }
    {
        uint64_t uok = n_uring_batch_ok.load(), ufb = n_uring_batch_fallback.load();
        if (uok + ufb > 0) {
            LOG_INFO("io_uring batches:    %lu batched / %lu fell back to sequential preads (%.1f%% batched)\n",
                     (unsigned long)uok, (unsigned long)ufb,
                     100.0 * uok / (uok + ufb));
        }
    }
    {
        uint64_t dok = n_decode_uring_ok.load();
        uint64_t dna = n_decode_uring_not_available.load();
        uint64_t dnp = n_decode_uring_not_packed.load();
        uint64_t dtl = n_decode_uring_too_large.load();
        uint64_t dsf = n_decode_uring_submit_failed.load();
        uint64_t dsr = n_decode_uring_short_read.load();
        uint64_t dtotal = dok + dna + dnp + dtl + dsf + dsr;
        if (dtotal > 0) {
            LOG_INFO("Decode-thread io_uring batches: %lu/%lu batched (%.1f%%) -- fallback cause: "
                     "%lu ring-not-available / %lu layer-not-packed (check gguf_tool verify!) / "
                     "%lu batch-too-large / %lu submit-failed / %lu short-read\n",
                     (unsigned long)dok, (unsigned long)dtotal,
                     100.0 * dok / dtotal,
                     (unsigned long)dna, (unsigned long)dnp, (unsigned long)dtl,
                     (unsigned long)dsf, (unsigned long)dsr);
        }
    }
    {
        // Gated on ANY of these being nonzero, not just issued>0 -- a
        // run where padding is enabled but never issues a single read
        // is exactly the "why didn't this help" case these diagnostics
        // exist for; gating only on issued>0 would hide every counter
        // that explains why, precisely when they're most needed.
        uint64_t pi = n_pad_reads_issued.load();
        uint64_t pns = n_pad_no_free_slot.load();
        uint64_t pge = n_pad_gate_entered.load();
        uint64_t pnr = n_pad_no_ranking_for_layer.load();
        uint64_t pnc = n_pad_no_candidates_after_filter.load();
        uint64_t pev = n_pad_evict_dup_used.load();
        uint64_t pev_un = n_pad_evict_dup_unavailable.load();
        if (pi > 0 || pns > 0 || pge > 0 || pnr > 0 || pnc > 0 || pev > 0 || pev_un > 0) {
            LOG_INFO("Disk-batch padding:  %lu issued (%lu ok / %lu failed), %lu later genuinely "
                     "hit (%.1f%% payoff), %.2f GB speculative bytes read, %lu no-free-slot skips "
                     "(%lu of which were filled anyway by evicting an also-VRAM-resident RAM slot, "
                     "disk_batch_padding_evict_dup; %lu still had none available even then)\n",
                     (unsigned long)pi, (unsigned long)n_pad_reads_ok.load(),
                     (unsigned long)n_pad_reads_failed.load(), (unsigned long)n_pad_reads_hit_later.load(),
                     pi > 0 ? 100.0 * n_pad_reads_hit_later.load() / pi : 0.0,
                     bytes_disk_read_speculative.load() / 1e9,
                     (unsigned long)pns, (unsigned long)pev, (unsigned long)pev_un);
            LOG_INFO("Disk-batch padding diag: %lu gate-entered, %lu layer-had-no-ranking, "
                     "%lu ranking-had-zero-survivors-after-filter\n",
                     (unsigned long)pge, (unsigned long)pnr, (unsigned long)pnc);
        }
        uint64_t pdc = n_pad_dbg_calls.load();
        if (pdc > 0) {
            LOG_INFO("Disk-batch padding gate terms (of %lu real Phase B calls): "
                     "%lu disk_batch_padding=false, %lu !all_packed, %lu reqs.empty(), "
                     "%lu in_prefill, %lu n_pending>=target\n",
                     (unsigned long)pdc,
                     (unsigned long)n_pad_dbg_padding_false.load(),
                     (unsigned long)n_pad_dbg_not_all_packed.load(),
                     (unsigned long)n_pad_dbg_reqs_empty.load(),
                     (unsigned long)n_pad_dbg_in_prefill.load(),
                     (unsigned long)n_pad_dbg_target_not_reached.load());
        }
    }
    LOG_INFO("Pinned residents:    %lu current / %lu desync-detected "
             "(evict_lru no longer scans pinned slots at all -- this should read 0; "
             "nonzero means unpinned_slot_indices desynced from the real pinned flags)\n",
             (unsigned long)n_pinned_resident.load(),
             (unsigned long)n_pinned_eviction_skips.load());
    LOG_INFO("Temp-pin skips:      %lu (decode-time cold-GPU-dispatch slots protected mid-compute)\n",
             (unsigned long)n_temp_pin_eviction_skips.load());
    LOG_INFO("Upload-pending skips: %lu genuinely-in-flight (confirmed via re-poll) / "
             "%lu stale-flag reclaimed (a speculative promotion's copy had actually "
             "finished but was never re-confirmed -- see gpu_topology.hpp's evict_lru fix)\n",
             (unsigned long)n_upload_pending_eviction_skips.load(),
             (unsigned long)n_stale_upload_ready_reclaimed.load());
    LOG_INFO("Stale index hits:    %lu (find_expert's O(1) expert_index disagreed with "
             "expert_vram_cache's real contents -- should always be 0; nonzero means the "
             "index/array desynced somewhere, see gpu_topology.hpp)\n",
             (unsigned long)n_stale_expert_index_hits.load());
    if (n_upload_expert_no_victim.load() || n_upload_expert_memcpy_fail.load()) {
        LOG_INFO("upload_expert fails: %lu no-victim (evict_lru found nothing evictable) / %lu "
                 "cudaMemcpy H2D failures -- both should normally be 0; nonzero during startup "
                 "preload means pinning/pre-warming silently lost most of its attempts (see "
                 "[MOE-VRAM] upload_expert error lines for the specific cause per attempt)\n",
                 (unsigned long)n_upload_expert_no_victim.load(),
                 (unsigned long)n_upload_expert_memcpy_fail.load());
    }
    LOG_INFO("upload_expert stream fallback: %lu (fell back to legacy stream-0 cudaMemcpy "
             "instead of the dedicated copy_stream; should be 0 -- nonzero means the "
             "dedicated-stream fix silently didn't engage for those calls)\n",
             (unsigned long)n_upload_expert_stream_unavailable.load());
    LOG_INFO("Hold-for-compute fails: %lu (an apparently-VRAM-resident expert's identity changed "
             "between the residency check and hold_for_compute -- forces a safety fallback to CPU "
             "for that call rather than trust a possibly-stale pointer. Correctness-safe either way; "
             "high/rising values across runs with a shrinking unpinned/reactive VRAM pool point at "
             "eviction churn as the driver of any CPU-fallback-call increase, not a bug)\n",
             (unsigned long)n_hold_for_compute_fail.load());
    LOG_INFO("CPU fallback cause:  %lu type-unsupported (fixed, model-intrinsic) / %lu no-host-buffer "
             "(shadow-fast-path likely raced and lost VRAM residency -- see shadow-race line below) / "
             "%lu just-promoted (expected/benign -- RAM slot was just MOVE-promoted to VRAM, needs one "
             "more reactive hit before it's GPU-usable) / %lu upload-failed (a genuine cudaMemcpy/"
             "evict_lru attempt failed -- see upload_expert fails line above) / %lu cpu_affinity_layers "
             "(expected -- config explicitly pinned this layer to CPU, never even attempted a GPU "
             "residency check) of the total CPU-dispatched calls\n",
             (unsigned long)n_cpu_fallback_type_unsupported.load(),
             (unsigned long)n_cpu_fallback_no_host_buffer.load(),
             (unsigned long)n_cpu_fallback_just_promoted.load(),
             (unsigned long)n_cpu_fallback_upload_failed.load(),
             (unsigned long)n_cpu_fallback_affinity_layer.load());
    if (n_shadow_race_reload_success.load() || n_shadow_race_reload_hard_fail.load()) {
        LOG_INFO("Shadow-race reload:  %lu succeeded (expert still computed, via CPU instead of the "
                 "originally-expected GPU path) / %lu hard-failed (expert's contribution skipped "
                 "ENTIRELY for that token -- a real quality gap, not just perf) -- previously assumed "
                 "\"rare\" but never actually counted\n",
                 (unsigned long)n_shadow_race_reload_success.load(),
                 (unsigned long)n_shadow_race_reload_hard_fail.load());
    }
    if (n_bf16_overflow_clamped.load()) {
        LOG_WARN("BF16 overflow clamped: %lu (a finite F32 weight or activation rounded up into "
                 "BF16's Inf/NaN exponent pattern and was clamped to the largest finite BF16 "
                 "magnitude instead of propagating -- should be essentially unreachable for real "
                 "LLM values; nonzero here means something upstream produced a value within a "
                 "fraction of a percent of FLT_MAX, worth investigating directly)\n",
                 (unsigned long)n_bf16_overflow_clamped.load());
    }
    LOG_INFO("Staging mutex wait:  %8.3f s  (staging slot regrow: %lu events, %.3f s -- "
             "cudaMallocHost/cudaFreeHost held under the SAME lock, see [SLOW-STAGING-REGROW])\n",
             t_staging_mutex_wait.load(), (unsigned long)n_staging_regrow.load(), t_staging_regrow.load());
    // REMOVED: "Staged-unused evict" and "Stale-slot reclaim" report
    // lines used to live here (n_staged_evicted_unused/n_staging_stale_
    // reclaimed/n_staging_cross_token_stale_reclaimed/n_staging_
    // fallback_events) -- both measured the old separate staging pool's
    // own round-robin eviction/reclaim behavior, which no longer exists
    // (LockFreePrefetcher writes directly into the RAM tier now, see
    // ram_manager.hpp's claim_or_wait). The replacement signal is
    // "Unified RAM waste" further down (n_ram_prefetch_loaded_evicted_
    // unused), measured against the real cache's real LRU eviction
    // instead of an artificial round-robin pool.
    {
        uint64_t vram_race = n_evicted_was_pilot_confirmed_vram.load();
        uint64_t ram_race = n_evicted_was_pilot_confirmed_ram.load();
        LOG_INFO("Pilot-confirm evict (raced past protection): %lu VRAM / %lu RAM -- experts the pilot's "
                 "most recent skip-fetch decision had JUST confirmed already-resident (RAM or VRAM-unpinned), "
                 "then ordinary LRU eviction picked as a victim anyway before decode's real dispatch reached "
                 "them (see was_pilot_confirmed_resident, pilot_confirmed_set.hpp). Should now be near-zero -- "
                 "2026-08-17: victim-selection scans actively exclude a pilot-confirmed candidate instead of "
                 "just counting the race after the fact; see 'Pilot-confirm protected' below for how often "
                 "that exclusion engages. Nonzero here means a candidate was STILL picked despite the "
                 "exclusion -- only possible if literally every OTHER slot in that scan was also ineligible "
                 "(pinned/temp-pinned/in-flight), the same rare last-resort condition the pre-existing "
                 "[RAM-CLAIM] slot-0 fallback warning covers.\n",
                 (unsigned long)vram_race, (unsigned long)ram_race);
        uint64_t vram_protected = n_pilot_protected_eviction_skips_vram.load();
        uint64_t ram_protected = n_pilot_protected_eviction_skips_ram.load();
        LOG_INFO("Pilot-confirm protected: %lu VRAM / %lu RAM candidate(s) excluded from eviction because "
                 "the pilot's most recent skip-fetch decision had just confirmed them resident -- each one is "
                 "a real re-read/re-promotion this run avoided paying for (compare against the 'raced past "
                 "protection' counts above, which is what this used to cost before the exclusion existed).\n",
                 (unsigned long)vram_protected, (unsigned long)ram_protected);
    }
    {
        uint64_t v_if = n_vram_hit_had_inflight_fetch.load(), v_st = n_vram_hit_had_staged_fetch.load();
        uint64_t v_ev = n_vram_hit_had_evicted_fetch.load();
        uint64_t r_if = n_ram_hit_had_inflight_fetch.load(), r_st = n_ram_hit_had_staged_fetch.load();
        uint64_t r_ev = n_ram_hit_had_evicted_fetch.load();
        uint64_t total = v_if + v_st + v_ev + r_if + r_st + r_ev;
        if (total > 0) {
            double v_st_avg_ms = v_st > 0 ? t_vram_hit_staged_fetch_elapsed_us_sum.load() / 1000.0 / (double)v_st : 0.0;
            double r_st_avg_ms = r_st > 0 ? t_ram_hit_staged_fetch_elapsed_us_sum.load() / 1000.0 / (double)r_st : 0.0;
            LOG_INFO("Cache-hit-beat-fetch: %lu total -- of every INDEPENDENT cache hit (bypassed "
                     "consume_staged entirely) whose expert had ALSO been recently touched by the "
                     "prefetcher: VRAM tier: %lu in-flight / %lu still-staged-unconsumed (avg %.1fms old, "
                     "the CLEANEST redundancy signal -- the fetch fully succeeded and was still sitting "
                     "there) / %lu already-evicted | RAM tier: %lu in-flight / %lu still-staged-unconsumed "
                     "(avg %.1fms old) / %lu already-evicted -- this is the direct measurement of why "
                     "correctly-predicted fetches go unused: an independent cache hit resolved the "
                     "dispatch before the prefetched copy was ever consulted.\n",
                     (unsigned long)total, (unsigned long)v_if, (unsigned long)v_st, v_st_avg_ms,
                     (unsigned long)v_ev, (unsigned long)r_if, (unsigned long)r_st, r_st_avg_ms,
                     (unsigned long)r_ev);

            // Hypothesis test: was the above a SAME-TOKEN race, or an
            // earlier token's abandoned fetch being rediscovered by a
            // later, unrelated token's own hit? step_delta==0 across
            // (nearly) all matches would mean same-token race; a large
            // average step_delta with same_step near 0 would confirm
            // the stale-record hypothesis instead.
            uint64_t v_st_same = n_vram_staged_fetch_same_step.load(), v_ev_same = n_vram_evicted_fetch_same_step.load();
            uint64_t r_st_same = n_ram_staged_fetch_same_step.load(), r_ev_same = n_ram_evicted_fetch_same_step.load();
            double v_st_step_avg = v_st > 0 ? t_vram_staged_fetch_step_delta_sum.load() / (double)v_st : 0.0;
            double v_ev_step_avg = v_ev > 0 ? t_vram_evicted_fetch_step_delta_sum.load() / (double)v_ev : 0.0;
            double r_st_step_avg = r_st > 0 ? t_ram_staged_fetch_step_delta_sum.load() / (double)r_st : 0.0;
            double r_ev_step_avg = r_ev > 0 ? t_ram_evicted_fetch_step_delta_sum.load() / (double)r_ev : 0.0;
            LOG_INFO("Cache-hit-beat-fetch token-step test: same-token-step (race within one layer "
                     "transition) vs cross-token-step (an earlier token's fetch, never consumed, "
                     "rediscovered later by an unrelated token's own hit) -- VRAM staged: %lu/%lu "
                     "same-step, avg %.1f steps back | VRAM evicted: %lu/%lu same-step, avg %.1f steps "
                     "back | RAM staged: %lu/%lu same-step, avg %.1f steps back | RAM evicted: %lu/%lu "
                     "same-step, avg %.1f steps back -- near-zero same-step counts + large avg steps-back "
                     "confirms the stale-record hypothesis; same-step counts close to the total would "
                     "instead mean a genuine same-transition race.\n",
                     (unsigned long)v_st_same, (unsigned long)v_st, v_st_step_avg,
                     (unsigned long)v_ev_same, (unsigned long)v_ev, v_ev_step_avg,
                     (unsigned long)r_st_same, (unsigned long)r_st, r_st_step_avg,
                     (unsigned long)r_ev_same, (unsigned long)r_ev, r_ev_step_avg);
        }
    }
    // REMOVED: "RAM-load-vs-prefetch" and "Phase-A/Phase-B gap" report
    // blocks used to live here, fed by n_ram_load_had_prefetch_*/
    // n_phase_gap_staged_*/t_phase_gap_ms_sum -- both measured races
    // against the old separate staging pool that no longer exists. See
    // "Unified RAM claim"/"Unified RAM wait" right below for the
    // replacement instrumentation (RAMSafetyManager::claim_or_wait,
    // ram_manager.hpp).
    {
        uint64_t hit = n_ram_claim_hit.load();
        uint64_t by_main_thread = n_ram_claim_by_main_thread.load();
        uint64_t by_prefetch = n_ram_claim_by_prefetch.load();
        uint64_t already_resident = n_ram_claim_already_resident_prefetch.load();
        uint64_t already_inflight = n_ram_claim_already_inflight_prefetch.load();
        uint64_t waited = n_ram_wait_inflight_main_thread.load();
        uint64_t evicted_unused = n_ram_prefetch_loaded_evicted_unused.load();
        uint64_t total_claims = hit + by_main_thread + by_prefetch;
        if (total_claims > 0 || already_resident > 0 || already_inflight > 0) {
            // WHOLE-SESSION totals (both prefill and decode combined) --
            // "by-main-thread" is labeled by CALLING THREAD (the main
            // generation-loop thread's own synchronous claim), not by
            // generation phase; it fires from the same per-token
            // dispatch loop during both prefill and decode, so this
            // line alone can't tell you which phase dominates it -- see
            // RamClaimCaller's own comment (engine.hpp). For a prefill-
            // only breakdown, read the [PREFILL]-scoped "RAM claim"
            // line (generation_loop.hpp) instead.
            LOG_INFO("Unified RAM claim:   %lu hit, %lu claimed-by-main-thread, %lu claimed-by-prefetch "
                     "(of %lu total claim_or_wait calls) -- prefetch's own claim step consults the "
                     "SAME index a main-thread hit-check uses, so it structurally cannot re-fetch an "
                     "already-resident expert: %lu prefetch claim attempts found the expert already "
                     "Ready (this is the direct, by-construction elimination of the old "
                     "Cache-hit-beat-fetch waste category), %lu found it already Loading (a no-op "
                     "dedup against whoever else is fetching it).\n",
                     (unsigned long)hit, (unsigned long)by_main_thread, (unsigned long)by_prefetch,
                     (unsigned long)total_claims, (unsigned long)already_resident,
                     (unsigned long)already_inflight);
            if (waited > 0) {
                LOG_INFO("Unified RAM wait:    %lu main-thread miss(es) found the expert already "
                         "Loading (claimed by a prefetch worker) and waited for that exact in-flight "
                         "read instead of racing past it -- avg wait %.2fms, total %.3fs. This is the "
                         "direct, by-construction elimination of the old Phase-A/Phase-B ordering gap: "
                         "no more check-once-then-race, the main thread now waits for the SAME read "
                         "the prefetcher already started.\n",
                         (unsigned long)waited, t_ram_wait_inflight_ms_sum.load() / (double)waited,
                         t_ram_wait_inflight_ms_sum.load() / 1000.0);
            }
            if (by_prefetch > 0) {
                LOG_INFO("Unified RAM waste:   %lu/%lu (%.1f%%) prefetch-loaded slot(s) were reclaimed "
                         "by normal LRU eviction before any real hit ever touched them -- the "
                         "unified design's replacement for the old staging-pool-based wasted-prefetch "
                         "measurement, now grounded in the real cache's real LRU pressure instead of "
                         "an artificial round-robin pool.\n",
                         (unsigned long)evicted_unused, (unsigned long)by_prefetch,
                         by_prefetch > 0 ? 100.0 * evicted_unused / by_prefetch : 0.0);
                // 2026-08-30: the line above is AMBIGUOUS on its own --
                // see MoEExpertBlock::consumed_by_vram_upload (engine.hpp).
                // This splits it into the two structurally different
                // causes, which need completely different fixes.
                uint64_t ev_consumed = n_ram_prefetch_evicted_consumed_by_vram.load();
                uint64_t ev_untouched = n_ram_prefetch_evicted_untouched.load();
                uint64_t ev_split_sum = ev_consumed + ev_untouched;
                LOG_INFO("  ^ waste SPLIT:     %lu (%.1f%% of prefetches) BENIGN -- bytes DID reach VRAM "
                         "(active-layer pre-warm / promotion) and were used from there; only the RAM "
                         "tier's own hit_count is blind to it, nothing to fix. %lu (%.1f%%) REAL WASTE "
                         "-- fetched, never read by anyone, reclaimed under capacity pressure; THIS is "
                         "the number that justifies (or doesn't) prefill capacity work.%s\n",
                         (unsigned long)ev_consumed,
                         by_prefetch > 0 ? 100.0 * ev_consumed / by_prefetch : 0.0,
                         (unsigned long)ev_untouched,
                         by_prefetch > 0 ? 100.0 * ev_untouched / by_prefetch : 0.0,
                         ev_split_sum == evicted_unused ? ""
                             : "  [!! SPLIT DOES NOT SUM TO THE TOTAL ABOVE -- a real accounting bug, "
                               "not a tuning signal; do not trust these two numbers until fixed]");
            }
        }
    }
}

void PerfCounters::dump_report_part6(double session_elapsed_s) const {
    (void)session_elapsed_s;
    LOG_INFO("RAM mutex wait:      %8.3f s  (RAMSafetyManager::ram_mutex, shared between the decode "
             "thread's own miss-handling and the prefetcher worker thread's claims)\n",
             t_ram_mutex_wait.load());
    LOG_INFO("Cache mutex wait:    %8.3f s  (GpuTopology::cache_mutex, shared between the decode "
             "thread's hit-triggered promotions and the prefetcher worker thread)\n",
             t_cache_mutex_wait.load());
    LOG_INFO("Tokens generated:    %lu\n", (unsigned long)n_tokens_generated.load());
    {
        const unsigned long cycles = (unsigned long)n_mtp_cycles.load();
        const unsigned long proposed = (unsigned long)n_mtp_drafts_proposed.load();
        const unsigned long accepted = (unsigned long)n_mtp_drafts_accepted.load();
        const unsigned long emitted = (unsigned long)n_mtp_tokens_emitted.load();
        if (cycles > 0 || proposed > 0) {
            LOG_INFO("MTP speculation:     %lu cycle(s), %lu draft(s) proposed / %lu accepted "
                     "(%.1f%% acceptance), %lu token(s) emitted via MTP (%.2f tokens/cycle "
                     "including the always-emitted correction token -- >1.0 means MTP is "
                     "genuinely faster than one-token-at-a-time target decoding, not just "
                     "correct)\n",
                     cycles, proposed, accepted,
                     proposed > 0 ? 100.0 * (double)accepted / (double)proposed : 0.0,
                     emitted, cycles > 0 ? (double)emitted / (double)cycles : 0.0);
        }
    }
    double active_s = t_active_processing.load();
    if (n_tokens_generated > 0 && active_s > 0) {
        LOG_INFO("Throughput:          %.2f tok/s  (%.2fs active processing across %lu request(s) -- "
                 "sum of each [REQUEST] line's own wall time, EXCLUDING any idle time between "
                 "prompts/turns -- this is the real generation-speed number, mixes prefill+decode; "
                 "see the per-request [REQUEST] line for a single turn's own rate, or per-token "
                 "[TOKEN N] DEBUG lines excluding token 0 for a decode-only rate)\n",
                 n_tokens_generated.load() / active_s, active_s, (unsigned long)n_requests.load());
    }
    // Explicit overall/decode/prefill split -- the Throughput line above
    // mixes prefill+decode into one denominator (see its own comment),
    // which understates real steady-state decode speed and gives no way
    // to directly compare prefill-only cost across runs/config changes
    // without manually subtracting t_prefill_total by hand each time.
    // decode = generated tokens over wall time EXCLUDING prefill;
    // prefill = prompt tokens over prefill's own wall time (both derived
    // from counters that already existed independently -- n_prompt_
    // tokens is the only new one, added specifically to complete this
    // line).
    {
        double decode_s = active_s - t_prefill_total.load();
        double overall_rate = active_s > 0 ? n_tokens_generated.load() / active_s : 0.0;
        double decode_rate = decode_s > 0 ? n_tokens_generated.load() / decode_s : 0.0;
        double prefill_rate = t_prefill_total > 0 ? n_prompt_tokens.load() / t_prefill_total.load() : 0.0;
        if (n_tokens_generated > 0 && active_s > 0) {
            LOG_INFO("Rates:               overall=%.2f tok/s | decode=%.2f tok/s (%lu tokens / %.2fs) | "
                     "prefill=%.2f tok/s (%lu tokens / %.2fs)\n",
                     overall_rate, decode_rate, (unsigned long)n_tokens_generated.load(), decode_s,
                     prefill_rate, (unsigned long)n_prompt_tokens.load(), t_prefill_total.load());
        }
    }
    if (n_tokens_generated > 0 && session_elapsed_s > 0) {
        LOG_INFO("Session wall-clock:  %.2f tok/s  (%.2fs engine-ready to shutdown -- INCLUDES any "
                 "idle time between turns, e.g. a REPL waiting on the next prompt; use the "
                 "Throughput line above for real generation speed. A large gap between these two "
                 "numbers means the process sat idle, not that generation was slow.)\n",
                 n_tokens_generated.load() / session_elapsed_s, session_elapsed_s);
    }
    LOG_INFO("RSS:                 %.2f GB\n", rss_gb());
    LOG_INFO("===================================================\n\n");
}
