// Per-subsystem end-of-run report_* functions (split from perf_timer.hpp).
#include "perf_timer.hpp"

// Call once at shutdown (mirrors report_cache_churn()'s placement/style).
// Largest-total-first so the dominant contributor is obvious at a glance.
void report_named_timers() {
    auto& stats = named_timer_stats();
    if (stats.empty()) return;
    std::vector<std::pair<std::string, NamedTimerStats>> sorted(stats.begin(), stats.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.second.total_ms > b.second.total_ms; });
    LOG_INFO("[NAMED-TIMING] Per-tensor-name dense_proj cost (diagnostic -- "
             "added to evaluate BF16 fast-path candidates):\n");
    for (auto& [name, s] : sorted) {
        LOG_INFO("  %-20s total=%.2f ms  calls=%llu  avg=%.4f ms/call\n",
                 name.c_str(), s.total_ms, (unsigned long long)s.calls,
                 s.calls ? s.total_ms / (double)s.calls : 0.0);
    }
}

// Real per-layer disk-fetch-miss breakdown (decode-only) -- see
// PerfCounters::n_decode_real_fetch_by_layer's own comment for why this
// exists: measuring the real per-layer cold-miss population BEFORE
// designing a CPU-offload cap (Fiddler-style CPU/GPU split, arxiv
// 2402.07033), not guessing at it. One line per layer that saw at least one
// decode-time get_experts_batch call; a layer never visited (e.g. fully
// covered by the shadow VRAM-hit fast path, or cpu_affinity_layers) is
// skipped rather than padding the log with an all-zero row -- same
// convention as LockFreePrefetcher::log_per_layer_report().
void report_layer_miss_stats() {
    auto& p = perf();
    uint64_t total_calls = 0, total_fetch = 0;
    int layers_seen = 0;
    LOG_INFO("[LAYER-MISS] per-layer decode-time real disk-fetch misses (of a layer's topk-routed "
             "experts, how many were neither RAM- nor VRAM-resident and needed a real SSD read; "
             "gathered to size a CPU-offload cap, not yet acted on):\n");
    for (int l = 0; l < PerfCounters::kMaxTrackedLayersForMiss; ++l) {
        uint32_t calls = p.n_decode_calls_by_layer[l].load(std::memory_order_relaxed);
        if (calls == 0) continue;
        uint32_t fetch = p.n_decode_real_fetch_by_layer[l].load(std::memory_order_relaxed);
        uint32_t layer_max = p.n_decode_max_fetch_by_layer[l].load(std::memory_order_relaxed);
        total_calls += calls;
        total_fetch += fetch;
        ++layers_seen;
        LOG_INFO("  layer=%-3d calls=%-6u real_fetch=%-6u avg=%.3f/token  max_in_one_call=%u\n",
                 l, calls, fetch, calls ? (double)fetch / (double)calls : 0.0, layer_max);
    }
    if (layers_seen > 0) {
        LOG_INFO("[LAYER-MISS] %d layer(s) tracked, %llu total call(s), %llu total real fetch(es), "
                 "overall avg=%.3f real-fetch expert(s)/token/layer\n",
                 layers_seen, (unsigned long long)total_calls, (unsigned long long)total_fetch,
                 total_calls ? (double)total_fetch / (double)total_calls : 0.0);
    }

    // Pooled histogram: of every decode-time get_experts_batch call (any
    // layer), how many saw exactly N real-fetch misses -- this is the
    // number that actually decides cap=1 vs cap=2 vs higher, not the mean.
    uint64_t hist_total = 0;
    for (int b = 0; b < PerfCounters::kFetchHistogramBuckets; ++b) {
        hist_total += p.n_decode_fetch_count_histogram[b].load(std::memory_order_relaxed);
    }
    if (hist_total > 0) {
        LOG_INFO("[LAYER-MISS] pooled distribution (of %llu decode-time call(s), any layer, how many "
                 "real-fetch misses landed in ONE call -- this is what actually sizes a CPU-offload "
                 "cap, not the mean):\n", (unsigned long long)hist_total);
        uint64_t cum = 0;
        for (int b = 0; b < PerfCounters::kFetchHistogramBuckets; ++b) {
            uint32_t n = p.n_decode_fetch_count_histogram[b].load(std::memory_order_relaxed);
            cum += n;
            LOG_INFO("  misses=%s  count=%-7u  %5.1f%%   cumulative<=%s: %5.1f%%\n",
                     (b == PerfCounters::kFetchHistogramBuckets - 1) ? "8+" : std::to_string(b).c_str(),
                     n, 100.0 * (double)n / (double)hist_total,
                     (b == PerfCounters::kFetchHistogramBuckets - 1) ? "8+" : std::to_string(b).c_str(),
                     100.0 * (double)cum / (double)hist_total);
        }
    }

    // EngineConfig::cpu_offload_max_experts's own real effect this run --
    // only meaningful once that cap is nonzero (both counters stay 0 at the
    // default off setting, same convention as the other gated-feature report
    // lines in this file).
    uint64_t forced = p.n_cpu_offload_forced.load(std::memory_order_relaxed);
    uint64_t left_uncapped = p.n_cpu_offload_left_uncapped.load(std::memory_order_relaxed);
    if (forced > 0 || left_uncapped > 0) {
        uint64_t total_real_miss = forced + left_uncapped;
        LOG_INFO("[CPU-OFFLOAD] %llu/%llu (%.1f%%) real disk-cold miss(es) redirected to CPU compute "
                 "instead of a VRAM upload (cpu_offload_max_experts cap); %llu left on the normal "
                 "VRAM-upload path because that token/layer's own cap was already spent\n",
                 (unsigned long long)forced, (unsigned long long)total_real_miss,
                 total_real_miss ? 100.0 * (double)forced / (double)total_real_miss : 0.0,
                 (unsigned long long)left_uncapped);
    }
}

// Real per-layer prediction accuracy for all three pilots (2026-08-25) --
// answers "is layer X more/less predictable than layer Y", the question
// this project's own global recall numbers (n_pilot*_recall_hits/checks,
// one number for the WHOLE model) can't answer. Motivation: if a layer is
// reliably >X% predictable, a config could afford to speculate more
// aggressively there (wider promotion/hint cutoffs) than on a layer whose
// prediction is closer to noise -- but that's a real, separate design
// question this report only gathers the DATA for; it doesn't act on it.
// One line per layer that saw at least one check from ANY of the three
// pilots -- a layer never checked by any of them (e.g. cpu_affinity_layers,
// or simply the LAST layer, which has no l+1/l+2 to predict) is skipped
// rather than padding the log with an all-zero row, same convention as
// report_layer_miss_stats() above. Each pilot's own accuracy is printed
// as "--" (not 0.0%) when that pilot has zero checks for this layer --
// DEPTH-1/ATTN-PILOT are only ever off if the whole prefetch pilot is
// disabled at the config level (rare), but DEPTH-2 is off by default
// (prefetch_pilot_depth=1), so most real runs will show "--" for its
// column unless depth=2 was explicitly enabled this run -- printing 0.0%
// there would misleadingly read as "DEPTH-2 predicts this layer badly"
// instead of "DEPTH-2 wasn't even running".
// See EngineConfig::prefill_dispatch_lookahead_tokens's own comment for the
// full mechanism. Prints only if the feature ever actually ran (added>0 or
// skipped>0) -- silent (matching report_moe_bucket_stats's own convention)
// when the feature is off or no prefill call happened this run.
// EngineConfig::pilot_widen_all_tokens -- see that field's own comment,
// and each n_pilot_wide_* counter's own comment above, for the full
// mechanism and the 2026-08-30 recall-mislabeling fix. Silent when the
// feature never ran this session (no checks recorded).
void report_pilot_wide_stats() {
    auto& p = perf();
    uint64_t checks = p.n_pilot_wide_recall_checks.load(std::memory_order_relaxed);
    if (checks == 0) return;
    uint64_t recall_hits = p.n_pilot_wide_recall_hits_last.load(std::memory_order_relaxed);
    uint64_t topk_sum = p.n_pilot_wide_recall_topk_sum.load(std::memory_order_relaxed);
    uint64_t pred_size_sum = p.n_pilot_wide_pred_size_sum.load(std::memory_order_relaxed);
    uint64_t coverage_hits = p.n_pilot_wide_coverage_hits.load(std::memory_order_relaxed);
    uint64_t real_unique_sum = p.n_pilot_wide_real_unique_sum.load(std::memory_order_relaxed);
    uint64_t hints_pushed = p.n_pilot_wide_hints_pushed.load(std::memory_order_relaxed);
    uint64_t hints_skipped = p.n_pilot_wide_hints_skipped_resident.load(std::memory_order_relaxed);
    std::string hint_suffix;
    if (hints_pushed + hints_skipped > 0) {
        hint_suffix = " | real hint push: " + std::to_string((unsigned long long)hints_pushed) +
                      " pushed / " + std::to_string((unsigned long long)hints_skipped) +
                      " skipped (already resident)";
    }
    LOG_INFO("[PILOT-WIDE] %lu check(s), avg predicted-set size %.1f (vs DEPTH-1's fixed 8): "
             "recall (accuracy vs the last token's own real picks, apples-to-apples with DEPTH-1's "
             "own hits/checks*topk convention) %lu/%lu (%.1f%%); precision (of the widened set's own "
             "size, how much was a hit for that same one token) %lu/%lu (%.1f%%); coverage (of the "
             "real WHOLE-layer unique-expert demand, avg %.1f/check, how much did the widened "
             "prediction actually address) %lu/%lu (%.1f%%)%s.\n",
             (unsigned long)checks, pred_size_sum > 0 ? (double)pred_size_sum / (double)checks : 0.0,
             (unsigned long)recall_hits, (unsigned long)topk_sum,
             topk_sum > 0 ? 100.0 * (double)recall_hits / (double)topk_sum : 0.0,
             (unsigned long)recall_hits, (unsigned long)pred_size_sum,
             pred_size_sum > 0 ? 100.0 * (double)recall_hits / (double)pred_size_sum : 0.0,
             real_unique_sum > 0 ? (double)real_unique_sum / (double)checks : 0.0,
             (unsigned long)coverage_hits, (unsigned long)real_unique_sum,
             real_unique_sum > 0 ? 100.0 * (double)coverage_hits / (double)real_unique_sum : 0.0,
             hint_suffix.c_str());
}

void report_prefill_lookahead_stats() {
    auto& p = perf();
    uint64_t added = p.n_prefill_lookahead_added.load(std::memory_order_relaxed);
    uint64_t sk_vram = p.n_prefill_lookahead_skipped_vram.load(std::memory_order_relaxed);
    uint64_t sk_ram = p.n_prefill_lookahead_skipped_ram.load(std::memory_order_relaxed);
    uint64_t sk_dup = p.n_prefill_lookahead_skipped_dup.load(std::memory_order_relaxed);
    uint64_t total = added + sk_vram + sk_ram + sk_dup;
    if (total == 0) return;
    LOG_INFO("[PREFILL-LOOKAHEAD] %llu candidate(s) scanned across upcoming tokens: %llu (%.1f%%) added "
             "as genuine extra RAM-warm fetches, %llu already VRAM-resident, %llu already RAM-resident, "
             "%llu already in the same combined call -- added ones get real extra lead time before "
             "their own token's turn, never dispatched by the token that triggered the scan.\n",
             (unsigned long long)total, (unsigned long long)added,
             total > 0 ? 100.0 * (double)added / (double)total : 0.0,
             (unsigned long long)sk_vram, (unsigned long long)sk_ram, (unsigned long long)sk_dup);

    // The "was it actually used" half -- see MoEExpertBlock::loaded_by_
    // lookahead's own comment. NOT a full partition of `added` (a slot
    // still resident, un-evicted, at shutdown falls into neither bucket) --
    // read this as a lower bound on the used/wasted split, and the gap
    // between (used+wasted) and `added` as "still resident, outcome not
    // yet known" rather than assuming it's accounted for either way.
    uint64_t used = p.n_prefill_lookahead_first_hit.load(std::memory_order_relaxed);
    uint64_t wasted = p.n_prefill_lookahead_loaded_evicted_unused.load(std::memory_order_relaxed);
    uint64_t known_outcome = used + wasted;
    LOG_INFO("[PREFILL-LOOKAHEAD] Of %llu added: %llu (%.1f%% of added) later confirmed USED (the future "
             "token that needed it found it still resident), %llu (%.1f%% of added) confirmed WASTED "
             "(evicted again before anyone ever claimed it a second time -- pure lost disk I/O), %llu "
             "still resident/outcome unknown at shutdown.\n",
             (unsigned long long)added, (unsigned long long)used,
             added > 0 ? 100.0 * (double)used / (double)added : 0.0,
             (unsigned long long)wasted,
             added > 0 ? 100.0 * (double)wasted / (double)added : 0.0,
             (unsigned long long)(added > known_outcome ? added - known_outcome : 0));
}

// MEASUREMENT ONLY -- see PerfCounters::n_moe_bucket_unique_experts's own
// comment for the full motivation (the "how would llama.cpp's -ub/mul_mat_id
// batching apply here" investigation). Real per-layer lines are printed live
// (LOG_INFO, [MOE-BUCKET] tag, execution_layer.cpp's Phase 1) during any
// prefill call -- this is just the whole-run aggregate, printed once at
// shutdown alongside every other summary report.
// EngineConfig::prefill_window_cap (2026-08-30, P1+P2) -- see
// RAMSafetyManager::prefill_window_cap_by_layer's own comment for the
// design. Silent when the feature never ran. The per-layer table is the
// point of this line: without P3's trim the extra-slot budget only covers
// the first handful of layers, so the SESSION-WIDE waste % can barely move
// even if the mechanism is perfect -- comparing windowed vs un-windowed
// layers WITHIN one run is what actually settles whether P3 is worth
// building.
void report_prefill_window_stats() {
    auto& p = perf();
    uint64_t granted = p.n_prefill_window_granted.load(std::memory_order_relaxed);
    uint64_t denied  = p.n_prefill_window_denied_budget.load(std::memory_order_relaxed);
    uint64_t partial = p.n_prefill_window_partial.load(std::memory_order_relaxed);
    uint64_t noneed  = p.n_prefill_window_not_needed.load(std::memory_order_relaxed);
    uint64_t extra   = p.n_prefill_window_extra_slots_granted.load(std::memory_order_relaxed);
    if (granted == 0 && denied == 0 && noneed == 0) return;
    LOG_INFO("[PREFILL-WINDOW] %lu layer-grant(s) (%lu of them CLIPPED by the extra-slot budget), "
             "%lu denied (budget fully spent -- kept base cap), %lu not needed (demand already <= cap, "
             "e.g. a short prompt). %lu extra slot(s) granted in total.\n",
             (unsigned long)granted, (unsigned long)partial, (unsigned long)denied,
             (unsigned long)noneed, (unsigned long)extra);
    uint64_t sdef = p.n_ram_slab_free_deferred.load(std::memory_order_relaxed);
    uint64_t sfre = p.n_ram_slab_deferred_freed.load(std::memory_order_relaxed);
    uint64_t sinv = p.n_active_layer_worker_slot_invalidated.load(std::memory_order_relaxed);
    if (sdef > 0 || sfre > 0 || sinv > 0) {
      LOG_INFO("[SLAB-SAFETY] regrow-under-live-reader caught %lu time(s), %lu deferred buffer(s) later "
               "freed (a persistent gap means some block's guards never clear); %lu active-layer slot(s) "
               "invalidated after a failed H2D copy (nonzero = the upload bug still fires but is contained, "
               "not corrupting output).\n",
               (unsigned long)sdef, (unsigned long)sfre, (unsigned long)sinv);
    }

    uint64_t stamped = p.n_prefill_retention_stamped.load(std::memory_order_relaxed);
    if (stamped > 0) {
      LOG_INFO("[PREFILL-RETENTION] %lu RAM slot(s) stamped with this request's own prefill "
               "demand -- eviction_score keeps these ahead of merely-recently-touched slots, "
               "so the experts prefill proved decode will want survive the handoff.\n",
               (unsigned long)stamped);
    }

    uint64_t trimmed = p.n_prefill_window_trimmed_layers.load(std::memory_order_relaxed);
    uint64_t sfreed  = p.n_prefill_window_slots_freed.load(std::memory_order_relaxed);
    uint64_t sskip   = p.n_prefill_window_slots_skipped_inuse.load(std::memory_order_relaxed);
    if (trimmed > 0 || sfreed > 0) {
        LOG_INFO("[PREFILL-WINDOW] P3 trim: %lu layer-trim(s), %lu slot(s) freed and returned to the "
                 "budget, %lu slot(s) SKIPPED as still-in-use (ram_loading / temp-pinned / async-copy "
                 "pending -- skipping is the SAFE outcome: that memory is simply held one layer longer "
                 "rather than freed out from under a live reader).\n",
                 (unsigned long)trimmed, (unsigned long)sfreed, (unsigned long)sskip);
    }

    // Per-layer untouched-waste rate: the controlled comparison.
    uint64_t win_load = 0, win_waste = 0, nowin_load = 0, nowin_waste = 0;
    bool any = false;
    for (int l = 0; l < PerfCounters::kMaxTrackedLayersForMiss; ++l) {
        uint64_t loaded = p.n_ram_prefetch_loaded_by_layer[l].load(std::memory_order_relaxed);
        if (loaded == 0) continue;
        any = true;
        uint64_t wasted = p.n_ram_prefetch_evicted_untouched_by_layer[l].load(std::memory_order_relaxed);
        if (p.prefill_window_layer_granted[l].load(std::memory_order_relaxed)) {
            win_load += loaded; win_waste += wasted;
        } else {
            nowin_load += loaded; nowin_waste += wasted;
        }
    }
    if (!any) return;
    LOG_INFO("[PREFILL-WINDOW] untouched-waste by layer group -- WINDOWED layer(s): %lu/%lu (%.1f%%) "
             "prefetch-loaded then evicted unread | UN-WINDOWED layer(s): %lu/%lu (%.1f%%). "
             "A large gap here IS the mechanism working (same run, same prompt, same hardware -- no "
             "cross-run noise); a small gap means capacity was never the real constraint.\n",
             (unsigned long)win_waste, (unsigned long)win_load,
             win_load > 0 ? 100.0 * (double)win_waste / (double)win_load : 0.0,
             (unsigned long)nowin_waste, (unsigned long)nowin_load,
             nowin_load > 0 ? 100.0 * (double)nowin_waste / (double)nowin_load : 0.0);
}

void report_moe_bucket_stats() {
    auto& p = perf();
    uint64_t unique = p.n_moe_bucket_unique_experts.load(std::memory_order_relaxed);
    uint64_t total = p.n_moe_bucket_total_dispatches.load(std::memory_order_relaxed);
    uint64_t layers = p.n_moe_bucket_layers_sampled.load(std::memory_order_relaxed);
    if (layers == 0) return;  // no prefill (S>1) call happened this run -- nothing to report
    double reduction = unique > 0 ? (double)total / (double)unique : 0.0;
    LOG_INFO("[MOE-BUCKET] Real token-sharing rate for MoE expert dispatch, %llu prefill layer-instance(s) "
             "sampled: %llu total (token,k) dispatches -> %llu unique (layer,expert) buckets "
             "(%.2fx real call-count reduction IF batched per-expert, per llama.cpp/ggml's own "
             "mul_mat_id design -- see [MOE-BUCKET] per-layer lines above for the per-layer breakdown; "
             "measurement only, does not change dispatch behavior).\n",
             (unsigned long long)layers, (unsigned long long)total, (unsigned long long)unique, reduction);
}

void report_pilot_recall_by_layer() {
    auto& p = perf();
    int layers_seen = 0;
    LOG_INFO("[PILOT-BY-LAYER] per-layer prediction accuracy (hits/topk-slots checked, all decode+"
             "prefill tokens pooled) for DEPTH-1 (l+1 from post-FFN state), ATTN-PILOT (l+1 from "
             "post-attention state), DEPTH-2 (l+2, naive reuse) -- \"--\" means that pilot had zero "
             "checks for this layer (commonly DEPTH-2, off by default):\n");
    for (int l = 0; l < PerfCounters::kMaxTrackedLayersForMiss; ++l) {
        uint32_t d1_checks = p.n_pilot_recall_checks_by_layer[l].load(std::memory_order_relaxed);
        uint32_t attn_checks = p.n_pilot_attn_recall_checks_by_layer[l].load(std::memory_order_relaxed);
        uint32_t d2_checks = p.n_pilot2_recall_checks_by_layer[l].load(std::memory_order_relaxed);
        if (d1_checks == 0 && attn_checks == 0 && d2_checks == 0) continue;
        ++layers_seen;

        char d1_buf[16], attn_buf[16], d2_buf[16];
        auto fmt_acc = [](char* buf, size_t buflen, uint32_t checks, uint32_t hits, uint32_t topk_sum) {
            if (checks == 0) { std::snprintf(buf, buflen, "--"); return; }
            double acc = topk_sum > 0 ? 100.0 * (double)hits / (double)topk_sum : 0.0;
            std::snprintf(buf, buflen, "%5.1f%%(n=%u)", acc, checks);
        };
        fmt_acc(d1_buf, sizeof(d1_buf), d1_checks,
                p.n_pilot_recall_hits_by_layer[l].load(std::memory_order_relaxed),
                p.n_pilot_recall_topk_sum_by_layer[l].load(std::memory_order_relaxed));
        fmt_acc(attn_buf, sizeof(attn_buf), attn_checks,
                p.n_pilot_attn_recall_hits_by_layer[l].load(std::memory_order_relaxed),
                p.n_pilot_attn_recall_topk_sum_by_layer[l].load(std::memory_order_relaxed));
        fmt_acc(d2_buf, sizeof(d2_buf), d2_checks,
                p.n_pilot2_recall_hits_by_layer[l].load(std::memory_order_relaxed),
                p.n_pilot2_recall_topk_sum_by_layer[l].load(std::memory_order_relaxed));

        LOG_INFO("  layer=%-3d DEPTH-1=%-14s ATTN-PILOT=%-14s DEPTH-2=%-14s\n",
                 l, d1_buf, attn_buf, d2_buf);
    }
    if (layers_seen > 0) {
        LOG_INFO("[PILOT-BY-LAYER] %d layer(s) tracked. Accuracy = hits / (checks * topk) per layer, "
                 "n=checks (higher n = more reliable estimate for that layer) -- NOT yet acted on by "
                 "any config, gathered purely to inform a future per-layer speculation policy.\n",
                 layers_seen);
    }
    // TEMPORARY DIAGNOSTIC (2026-09-13): the real [layer][rank] cross
    // table for DEPTH-1, requested directly -- see n_pilot_recall_hits_
    // by_layer_rank's own comment (PerfCounters) for why the two existing
    // marginal breakdowns above (by rank alone, by layer alone) can't
    // answer "for layer L specifically, is rank 0 more reliable than
    // rank 3." rank_i here means the i-th highest-confidence prediction
    // DEPTH-1 made for that layer (0 = most confident).
    {
        int lr_layers_seen = 0;
        for (int l = 0; l < PerfCounters::kMaxTrackedLayersForMiss; ++l) {
            uint32_t row_checks = 0;
            for (int rk = 0; rk < 8; ++rk)
                row_checks += p.n_pilot_recall_checks_by_layer_rank[l][rk].load(std::memory_order_relaxed);
            if (row_checks == 0) continue;
            if (lr_layers_seen == 0) {
                LOG_INFO("[PILOT-BY-LAYER-RANK] DEPTH-1 accuracy for layer L at rank k (k=0 is the "
                         "single most-confident prediction for that layer, k=7 the least) -- the real "
                         "cross table, not a marginal:\n");
            }
            ++lr_layers_seen;
            std::string row = "  layer=" + std::to_string(l) + " ";
            char cell[24];
            for (int rk = 0; rk < 8; ++rk) {
                uint32_t hits = p.n_pilot_recall_hits_by_layer_rank[l][rk].load(std::memory_order_relaxed);
                uint32_t checks = p.n_pilot_recall_checks_by_layer_rank[l][rk].load(std::memory_order_relaxed);
                if (checks == 0) { std::snprintf(cell, sizeof(cell), "k%d=--  ", rk); }
                else {
                    std::snprintf(cell, sizeof(cell), "k%d=%4.1f%%(%u) ", rk,
                                  100.0 * (double)hits / (double)checks, checks);
                }
                row += cell;
            }
            row += "\n";
            LOG_STREAM_INFO() << row;
        }
        if (lr_layers_seen > 0) {
            LOG_INFO("[PILOT-BY-LAYER-RANK] %d layer(s) tracked, real cross table (not a marginal).\n",
                     lr_layers_seen);
        }
    }
}

void report_qwen4exp_pilot_stats() {
    auto &p = perf();
    const uint64_t predictions =
        p.n_qwen_pilot_predictions.load(std::memory_order_relaxed);
    const uint64_t checks =
        p.n_qwen_pilot_recall_checks.load(std::memory_order_relaxed);
    const uint64_t sync_fail =
        p.n_qwen_pilot_sync_fail.load(std::memory_order_relaxed);
    const uint64_t skipped_fully_cached =
        p.n_qwen_pilot_skipped_fully_cached.load(std::memory_order_relaxed);
    const uint64_t skipped_prefetch_excluded =
        p.n_qwen_pilot_skipped_prefetch_excluded.load(
            std::memory_order_relaxed);
    const uint64_t skipped_predict_policy =
        p.n_qwen_pilot_skipped_predict_policy.load(
            std::memory_order_relaxed);
    if (predictions == 0 && checks == 0 && sync_fail == 0 &&
        skipped_fully_cached == 0 && skipped_prefetch_excluded == 0 &&
        skipped_predict_policy == 0 &&
        p.n_qwen_pilot_adaptive_residency_checks.load(
            std::memory_order_relaxed) == 0)
        return;

    const uint64_t hits =
        p.n_qwen_pilot_recall_hits.load(std::memory_order_relaxed);
    const uint64_t pred_slots =
        p.n_qwen_pilot_recall_pred_size_sum.load(std::memory_order_relaxed);
    const double accuracy = pred_slots > 0
                                ? 100.0 * (double)hits / (double)pred_slots
                                : 0.0;
    LOG_INFO("[QWEN-PILOT] HC stream-0 proxy: predictions=%llu gpu=%llu "
             "host_fallback=%llu checks=%llu "
             "overlap=%llu/%llu (%.1f%% predicted-slot accuracy), sync_fail=%llu "
             "skipped_fully_cached=%llu skipped_prefetch_excluded=%llu\n",
             (unsigned long long)predictions,
             (unsigned long long)p.n_qwen_pilot_gpu_predictions.load(
                 std::memory_order_relaxed),
             (unsigned long long)p.n_qwen_pilot_host_fallbacks.load(
                 std::memory_order_relaxed),
             (unsigned long long)checks,
             (unsigned long long)hits, (unsigned long long)pred_slots,
             accuracy, (unsigned long long)sync_fail,
             (unsigned long long)p.n_qwen_pilot_skipped_fully_cached.load(
                 std::memory_order_relaxed),
             (unsigned long long)skipped_prefetch_excluded);
    LOG_INFO("[QWEN-PILOT] prediction policy skips: allow-list=%llu\n",
             (unsigned long long)skipped_predict_policy);
    LOG_INFO("[QWEN-PILOT] adaptive hot-set: checks=%llu misses=%llu "
             "stable_skips=%llu (two consecutive all-resident routes)\n",
             (unsigned long long)p.n_qwen_pilot_adaptive_residency_checks.load(
                 std::memory_order_relaxed),
             (unsigned long long)p.n_qwen_pilot_adaptive_residency_misses.load(
                 std::memory_order_relaxed),
             (unsigned long long)p.n_qwen_pilot_adaptive_skips.load(
                 std::memory_order_relaxed));
    LOG_INFO("[QWEN-PILOT] load coverage: already_ram=%llu already_vram=%llu "
             "needs_fetch=%llu hints_pushed=%llu hints_skipped_resident=%llu\n",
             (unsigned long long)p.n_qwen_pilot_pred_already_ram.load(
                 std::memory_order_relaxed),
             (unsigned long long)p.n_qwen_pilot_pred_already_vram.load(
                 std::memory_order_relaxed),
             (unsigned long long)p.n_qwen_pilot_pred_needs_fetch.load(
                 std::memory_order_relaxed),
             (unsigned long long)p.n_qwen_pilot_hints_pushed.load(
                 std::memory_order_relaxed),
             (unsigned long long)p.n_qwen_pilot_hints_skipped_resident.load(
                 std::memory_order_relaxed));
    LOG_INFO("[QWEN-PILOT] active-layer prefetch: issued=%llu failed=%llu "
             "excluded_layer=%llu rank_skipped=%llu\n",
             (unsigned long long)p.n_qwen_pilot_active_prefetch_issued.load(
                 std::memory_order_relaxed),
             (unsigned long long)p.n_qwen_pilot_active_prefetch_failed.load(
                 std::memory_order_relaxed),
             (unsigned long long)p.n_qwen_pilot_prefetch_excluded.load(
                 std::memory_order_relaxed),
             (unsigned long long)p.n_qwen_pilot_prefetch_rank_skipped.load(
                 std::memory_order_relaxed));

    const uint64_t q_queue =
        p.n_qwen_pilot_queue_to_ready.load(std::memory_order_relaxed);
    const uint64_t q_router =
        p.n_qwen_pilot_ready_to_router.load(std::memory_order_relaxed);
    const uint64_t q_fetch =
        p.n_qwen_pilot_ready_to_fetch.load(std::memory_order_relaxed);
    const uint64_t q_router_fetch =
        p.n_qwen_pilot_router_to_fetch.load(std::memory_order_relaxed);
    if (q_queue || q_router || q_fetch || q_router_fetch) {
      LOG_INFO("[QWEN-PILOT] timeline: GPU prediction queue->host IDs %.3f ms avg "
               "(n=%llu) | IDs-ready->exact-router-ready %.3f ms avg (n=%llu) | "
               "exact-router-ready->DDR fetch start %.3f ms avg (n=%llu) | "
               "IDs-ready->DDR fetch start %.3f ms avg (n=%llu)\n",
               q_queue ? 1000.0 * p.t_qwen_pilot_queue_to_ready / q_queue : 0.0,
               (unsigned long long)q_queue,
               q_router ? 1000.0 * p.t_qwen_pilot_ready_to_router / q_router : 0.0,
               (unsigned long long)q_router,
               q_router_fetch ? 1000.0 * p.t_qwen_pilot_router_to_fetch /
                                      q_router_fetch
                              : 0.0,
               (unsigned long long)q_router_fetch,
               q_fetch ? 1000.0 * p.t_qwen_pilot_ready_to_fetch / q_fetch : 0.0,
               (unsigned long long)q_fetch);
    }
    const uint64_t q_issue = p.n_qwen_pilot_active_ready_to_issue.load(
        std::memory_order_relaxed);
    const uint64_t q_dma =
        p.n_qwen_pilot_active_dma.load(std::memory_order_relaxed);
    if (q_issue || q_dma) {
      const int64_t q_bytes =
          p.bytes_qwen_pilot_active_dma.load(std::memory_order_relaxed);
      LOG_INFO("[QWEN-PILOT] predicted active H2D: ready->cudaMemcpyAsync issue "
               "%.3f ms/copy (n=%llu) | CUDA-event DMA service %.3f ms/copy "
               "(n=%llu, %.2f GB, %.2f GB/s)\n",
               q_issue ? 1000.0 * p.t_qwen_pilot_active_ready_to_issue / q_issue
                       : 0.0,
               (unsigned long long)q_issue,
               q_dma ? 1000.0 * p.t_qwen_pilot_active_dma / q_dma : 0.0,
               (unsigned long long)q_dma, q_bytes / 1e9,
               p.t_qwen_pilot_active_dma > 0.0
                   ? (q_bytes / 1e9) / p.t_qwen_pilot_active_dma
                   : 0.0);
    }

    const uint32_t prefetch_k =
        p.n_qwen_pilot_prefetch_k_observed.load(std::memory_order_relaxed);
    const uint64_t prefetch_hits =
        p.n_qwen_pilot_prefetch_recall_hits.load(std::memory_order_relaxed);
    const uint64_t prefetch_checks =
        p.n_qwen_pilot_prefetch_recall_checks.load(std::memory_order_relaxed);
    const uint64_t prefetch_slots = p.n_qwen_pilot_prefetch_recall_pred_size_sum
                                        .load(std::memory_order_relaxed);
    const double prefetch_accuracy =
        prefetch_slots > 0
            ? 100.0 * (double)prefetch_hits / (double)prefetch_slots
            : 0.0;
    if (prefetch_k > 0) {
      LOG_INFO(
          "[QWEN-PILOT] prefetch accuracy: top-%u on eligible layers "
          "overlap=%llu/%llu (%.1f%%), checks=%llu -- actionable metric "
          "for the active-layer prefetch policy\n",
          prefetch_k, (unsigned long long)prefetch_hits,
          (unsigned long long)prefetch_slots, prefetch_accuracy,
          (unsigned long long)prefetch_checks);
      std::string prefetch_rank_line;
      for (uint32_t r = 0; r < prefetch_k &&
                           r < (uint32_t)PerfCounters::kQwenPilotMaxRanks;
           ++r) {
        const uint64_t rc = p.n_qwen_pilot_prefetch_rank_checks[r].load(
            std::memory_order_relaxed);
        if (rc == 0)
          continue;
        const uint64_t rh = p.n_qwen_pilot_prefetch_rank_hits[r].load(
            std::memory_order_relaxed);
        char part[64];
        std::snprintf(part, sizeof(part), " r%u=%.1f%%(%llu/%llu)", r,
                      100.0 * (double)rh / (double)rc,
                      (unsigned long long)rh, (unsigned long long)rc);
        prefetch_rank_line += part;
      }
      LOG_INFO("[QWEN-PILOT] prefetch rank membership (eligible layers):%s\n",
               prefetch_rank_line.empty() ? " none"
                                          : prefetch_rank_line.c_str());
    } else {
      LOG_INFO("[QWEN-PILOT] prefetch accuracy: disabled for this run\n");
    }

    std::string overall_rank_line;
    for (int r = 0; r < PerfCounters::kQwenPilotMaxRanks; ++r) {
        const uint64_t rc =
            p.n_qwen_pilot_rank_checks[r].load(std::memory_order_relaxed);
        if (rc == 0)
            continue;
        const uint64_t rh =
            p.n_qwen_pilot_rank_hits[r].load(std::memory_order_relaxed);
        char part[64];
        std::snprintf(part, sizeof(part), " r%d=%.1f%%(%llu/%llu)", r,
                      100.0 * (double)rh / (double)rc,
                      (unsigned long long)rh, (unsigned long long)rc);
        overall_rank_line += part;
    }
    LOG_INFO("[QWEN-PILOT] rank membership accuracy overall:%s\n",
             overall_rank_line.empty() ? " none" : overall_rank_line.c_str());

    int layers_seen = 0;
    int prefetch_enabled_layers = 0;
    int prefetch_excluded_layers = 0;
    for (int l = 0; l < PerfCounters::kMaxTrackedLayersForMiss; ++l) {
        const uint32_t lc = p.n_qwen_pilot_recall_checks_by_layer[l].load(
            std::memory_order_relaxed);
        if (lc == 0)
            continue;
        ++layers_seen;
        const uint32_t lh = p.n_qwen_pilot_recall_hits_by_layer[l].load(
            std::memory_order_relaxed);
        const uint32_t lp =
            p.n_qwen_pilot_recall_pred_size_sum_by_layer[l].load(
                std::memory_order_relaxed);
        const double la = lp > 0 ? 100.0 * (double)lh / (double)lp : 0.0;
        // Performance runs may predict only the active prefetch K (currently
        // five), while diagnostic runs can use the full top-10. Derive the
        // label from the observed candidate count instead of hard-coding it.
        const uint32_t predicted_k =
            lc > 0 ? (lp + lc / 2) / lc : 0;
        const uint32_t plc =
            p.n_qwen_pilot_prefetch_recall_checks_by_layer[l].load(
                std::memory_order_relaxed);
        const uint32_t plh =
            p.n_qwen_pilot_prefetch_recall_hits_by_layer[l].load(
                std::memory_order_relaxed);
        const uint32_t plp =
            p.n_qwen_pilot_prefetch_recall_pred_size_sum_by_layer[l].load(
                std::memory_order_relaxed);
        const double pla =
            plp > 0 ? 100.0 * (double)plh / (double)plp : 0.0;
        char prefetch_part[128];
        if (prefetch_k > 0 && plc > 0) {
          ++prefetch_enabled_layers;
          std::snprintf(prefetch_part, sizeof(prefetch_part),
                        " prefetch=ON top-%u=%u/%u (%.1f%%)", prefetch_k,
                        plh, plp, pla);
        } else if (prefetch_k > 0) {
          ++prefetch_excluded_layers;
          std::snprintf(prefetch_part, sizeof(prefetch_part),
                        " prefetch=EXCLUDED");
        } else {
          std::snprintf(prefetch_part, sizeof(prefetch_part),
                        " prefetch=OFF");
        }
        std::string layer_rank_line;
        for (int r = 0; r < PerfCounters::kQwenPilotMaxRanks; ++r) {
            const uint32_t rc =
                p.n_qwen_pilot_rank_checks_by_layer[l][r].load(
                    std::memory_order_relaxed);
            if (rc == 0)
                continue;
            const uint32_t rh =
                p.n_qwen_pilot_rank_hits_by_layer[l][r].load(
                    std::memory_order_relaxed);
            char part[48];
            std::snprintf(part, sizeof(part), " r%d=%.1f%%", r,
                          100.0 * (double)rh / (double)rc);
            layer_rank_line += part;
        }
        LOG_INFO("[QWEN-PILOT] layer=%-3d top-%u=%u/%u (%.1f%%) checks=%u%s "
                 "ranks:%s\n",
                 l, predicted_k, lh, lp, la, lc, prefetch_part,
                 layer_rank_line.empty() ? " none" : layer_rank_line.c_str());
    }
    {
      const uint64_t ppp_tokens =
          p.n_ppp_tokens_checked.load(std::memory_order_relaxed);
      if (ppp_tokens > 0) {
        static const char *kScenarioNames[PerfCounters::kPPPStagingScenarios] = {
            "K=0 (static pin only, no pilot)",
            "K=prefetch_k (today's real config)",
            "K=5 (full diagnostic top-5 staged)"};
        LOG_INFO("[QWEN-PILOT-PPP] Predictive Partial Pinning empirical "
                 "filter: %llu real decode token(s) checked on the "
                 "configured partial layer. M = uncovered routed experts "
                 "after (static pin set) UNION (staged set).\n",
                 (unsigned long long)ppp_tokens);
        for (int sc = 0; sc < PerfCounters::kPPPStagingScenarios; ++sc) {
          const uint64_t m0 = p.n_ppp_m0[sc].load(std::memory_order_relaxed);
          const uint64_t m1 = p.n_ppp_m1[sc].load(std::memory_order_relaxed);
          const uint64_t m2 =
              p.n_ppp_m2plus[sc].load(std::memory_order_relaxed);
          const uint64_t total = m0 + m1 + m2;
          const uint64_t m2_sum =
              p.n_ppp_m2plus_sum[sc].load(std::memory_order_relaxed);
          LOG_INFO("[QWEN-PILOT-PPP]   %s: M=0 %llu (%.1f%%) M=1 %llu "
                   "(%.1f%%) M>=2 %llu (%.1f%%, avg M=%.2f)\n",
                   kScenarioNames[sc], (unsigned long long)m0,
                   total > 0 ? 100.0 * m0 / total : 0.0,
                   (unsigned long long)m1,
                   total > 0 ? 100.0 * m1 / total : 0.0,
                   (unsigned long long)m2,
                   total > 0 ? 100.0 * m2 / total : 0.0,
                   m2 > 0 ? (double)m2_sum / m2 : 0.0);
        }
      }
    }
    if (layers_seen > 0) {
        LOG_INFO("[QWEN-PILOT] %d target layer(s) tracked; accuracy is "
                 "predicted IDs overlapping the real top-k divided by "
                 "predicted IDs checked.\n",
                 layers_seen);
      if (prefetch_k > 0)
        LOG_INFO("[QWEN-PILOT] prefetch policy: %d eligible layer(s), %d "
                 "excluded layer(s); per-layer top-%u values above are the "
                 "prefetch-relevant measurements.\n",
                 prefetch_enabled_layers, prefetch_excluded_layers,
                 prefetch_k);
    }
}
