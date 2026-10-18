#pragma once
#include "memory_guard.hpp"
#include "logger.hpp"
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <atomic>
#include <map>
#include <string>
#include <sstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#ifdef __APPLE__
#include <mach/mach.h>
#include <sys/resource.h>
#else
#include <sys/resource.h>
#endif
#include "perf/counters_08_pad.hpp"

struct PerfCounters : PerfCountersPart08 {
    // Plain-value point-in-time copy of the cumulative counters above, so
    // callers can diff two snapshots to get a per-token or per-request
    // breakdown without disturbing the global running totals.
    struct Snapshot {
        double t_disk_read=0, t_host_alloc=0, t_gpu_upload=0, t_expert_gpu_upload=0, t_gpu_download=0, t_gpu_compute=0, t_expert_silu_activation=0, t_cpu_expert=0;
        double t_cpu_dequant=0, t_cpu_dot=0;
        double t_qwen_cpu_grouped_gate_up=0, t_qwen_cpu_grouped_swiglu=0;
        double t_qwen_cpu_grouped_down=0, t_qwen_cpu_grouped_combine=0;
        double t_attn=0;
        double t_attn_preamble=0;
        double t_attn_norm=0, t_attn_qproj=0, t_attn_kvproj=0, t_attn_absorb=0, t_attn_score=0, t_attn_outproj=0;
        double t_attn_outproj_dequant=0;
        double t_dense_gpu_up=0, t_dense_gpu_compute=0, t_dense_gpu_down=0;
        double t_moe_route=0, t_moe_expert=0, t_prefetch=0, t_dense_load=0, t_consume_staged=0;
        double t_staging_regrow=0, t_staging_mutex_wait=0, t_cache_mutex_wait=0;
        double t_active_layer_upload=0, t_active_layer_promote_copy=0, t_vram_demote_copy=0;
        double t_hold_for_compute=0, t_release_hold=0, t_find_active_expert=0;
        double t_find_expert=0, t_evict_lru=0, t_upload_expert_async=0;
        double t_ram_get_expert=0, t_ram_get_experts_batch=0;
        double t_ram_batch_phase_a=0, t_ram_batch_phase_b=0, t_ram_batch_phase_c=0;
        double t_bounce_stage_memcpy=0, t_host_register_stage=0, t_bounce_buffer_wait=0;
        double t_ggml_moe_d2d=0, t_ggml_moe_h2d=0, t_ggml_moe_h2d_dma=0;
        uint64_t n_expert_cache_hit=0, n_expert_cache_miss=0, n_expert_promote=0, n_tokens_generated=0;
        uint64_t n_expert_gpu_calls=0, n_expert_cpu_calls=0, n_dense_gpu_calls=0;
        int64_t bytes_disk_read=0, bytes_gpu_uploaded=0, bytes_bounce_staged=0, bytes_host_registered=0;
        int64_t bytes_ggml_moe_d2d=0, bytes_ggml_moe_h2d=0;
        // Added alongside print_summary's own "disk=" throughput figures --
        // see dump_report's "Disk read (expert)" line and bytes_disk_read_
        // speculative's own comment (PerfCounters) for why t_disk_read
        // already includes padding's wall-clock and a real-only GB/s
        // figure alone under-reports true device throughput.
        int64_t bytes_disk_read_speculative=0;
        uint64_t n_prefetch_hits=0, n_pinned_eviction_skips=0, n_pinned_resident=0, n_staging_regrow=0;
        // Added to answer "is phase_b's remaining prefill cost a coverage
        // gap (never hinted) or a worker-capacity gap (hinted, but no
        // prefetch worker got to it before the decode thread did)" -- see
        // claim_or_wait's own comment for what Hit/by_main_thread/by_prefetch
        // mean -- "main_thread" is the CALLING THREAD (runs during both
        // prefill and decode), not a generation phase; read this line
        // against WHICH report line it appears in ([PREFILL]-scoped vs.
        // the whole-session shutdown line) to know which phase it's from.
        // Scoped via the SAME Snapshot diff as everything else here
        // (e.g. via the new [PREFILL] line, generation_loop.hpp), not just
        // the whole-program-lifetime dump_report() totals these already
        // fed before this addition.
        uint64_t n_ram_claim_hit=0, n_ram_claim_by_main_thread=0, n_ram_claim_by_prefetch=0;
        uint64_t n_ram_wait_inflight_main_thread=0;
        double t_ram_wait_inflight_ms_sum=0;
        double t_attn_qabsorb_batch=0;
        double t_moe_topk_select=0, t_moe_phase3_dispatch=0;
        double t_resolve_gpu_loc_total=0, t_gpu_issue_total=0, t_gpu_finish_total=0;
        double t_accumulate_total=0, t_shared_expert_total=0, t_layer_cleanup_total=0;
        // Fourth round, same investigation: these two ALREADY EXISTED
        // (feeding the shutdown-only "Active-layer BATCH pre-warm ...
        // issue=Xs sync=Ys" line) but were never threaded into the
        // per-request Snapshot/[REQUEST] line at all -- a real, live shot
        // shutdown-report figure (issue=18.308s on one real run) was
        // invisible in every [REQUEST] line to date despite explaining most
        // of the ~15-18s phase3_loop residual left after ruling out
        // resolve_gpu_loc/gpu_issue/gpu_finish/accumulate/shared_expert/
        // layer_cleanup. t_active_layer_upload_async's own comment expected
        // this to be "small/cheap, no wait" -- worth confirming or
        // disproving directly against the real per-request number, not just
        // the whole-session total.
        double t_active_layer_upload_async=0, t_active_layer_batch_sync=0;
        double t_shadow_precheck_total=0, t_dispatch_expert_k_total=0;
        double t_ram_needed_dispatch_total=0;

        Snapshot operator-(const Snapshot& b) const {
            return Snapshot{
                t_disk_read - b.t_disk_read, t_host_alloc - b.t_host_alloc,
                t_gpu_upload - b.t_gpu_upload, t_expert_gpu_upload - b.t_expert_gpu_upload,
                t_gpu_download - b.t_gpu_download,
                t_gpu_compute - b.t_gpu_compute,
                t_expert_silu_activation - b.t_expert_silu_activation,
                t_cpu_expert - b.t_cpu_expert,
                t_cpu_dequant - b.t_cpu_dequant, t_cpu_dot - b.t_cpu_dot,
                t_qwen_cpu_grouped_gate_up - b.t_qwen_cpu_grouped_gate_up,
                t_qwen_cpu_grouped_swiglu - b.t_qwen_cpu_grouped_swiglu,
                t_qwen_cpu_grouped_down - b.t_qwen_cpu_grouped_down,
                t_qwen_cpu_grouped_combine - b.t_qwen_cpu_grouped_combine,
                t_attn - b.t_attn,
                t_attn_preamble - b.t_attn_preamble,
                t_attn_norm - b.t_attn_norm, t_attn_qproj - b.t_attn_qproj,
                t_attn_kvproj - b.t_attn_kvproj, t_attn_absorb - b.t_attn_absorb,
                t_attn_score - b.t_attn_score, t_attn_outproj - b.t_attn_outproj,
                t_attn_outproj_dequant - b.t_attn_outproj_dequant,
                t_dense_gpu_up - b.t_dense_gpu_up, t_dense_gpu_compute - b.t_dense_gpu_compute,
                t_dense_gpu_down - b.t_dense_gpu_down,
                t_moe_route - b.t_moe_route,
                t_moe_expert - b.t_moe_expert, t_prefetch - b.t_prefetch,
                t_dense_load - b.t_dense_load, t_consume_staged - b.t_consume_staged,
                t_staging_regrow - b.t_staging_regrow,
                t_staging_mutex_wait - b.t_staging_mutex_wait,
                t_cache_mutex_wait - b.t_cache_mutex_wait,
                t_active_layer_upload - b.t_active_layer_upload,
                t_active_layer_promote_copy - b.t_active_layer_promote_copy,
                t_vram_demote_copy - b.t_vram_demote_copy,
                t_hold_for_compute - b.t_hold_for_compute,
                t_release_hold - b.t_release_hold,
                t_find_active_expert - b.t_find_active_expert,
                t_find_expert - b.t_find_expert,
                t_evict_lru - b.t_evict_lru,
                t_upload_expert_async - b.t_upload_expert_async,
                t_ram_get_expert - b.t_ram_get_expert,
                t_ram_get_experts_batch - b.t_ram_get_experts_batch,
                t_ram_batch_phase_a - b.t_ram_batch_phase_a,
                t_ram_batch_phase_b - b.t_ram_batch_phase_b,
                t_ram_batch_phase_c - b.t_ram_batch_phase_c,
                t_bounce_stage_memcpy - b.t_bounce_stage_memcpy,
                t_host_register_stage - b.t_host_register_stage,
                t_bounce_buffer_wait - b.t_bounce_buffer_wait,
                t_ggml_moe_d2d - b.t_ggml_moe_d2d,
                t_ggml_moe_h2d - b.t_ggml_moe_h2d,
                t_ggml_moe_h2d_dma - b.t_ggml_moe_h2d_dma,
                    n_expert_cache_hit - b.n_expert_cache_hit,
                    n_expert_cache_miss - b.n_expert_cache_miss,
                    n_expert_promote - b.n_expert_promote,
                    n_tokens_generated - b.n_tokens_generated,
                    n_expert_gpu_calls - b.n_expert_gpu_calls,
                    n_expert_cpu_calls - b.n_expert_cpu_calls,
                    n_dense_gpu_calls - b.n_dense_gpu_calls,
                bytes_disk_read - b.bytes_disk_read,
                bytes_gpu_uploaded - b.bytes_gpu_uploaded,
                bytes_bounce_staged - b.bytes_bounce_staged,
                bytes_host_registered - b.bytes_host_registered,
                bytes_ggml_moe_d2d - b.bytes_ggml_moe_d2d,
                bytes_ggml_moe_h2d - b.bytes_ggml_moe_h2d,
                    bytes_disk_read_speculative - b.bytes_disk_read_speculative,
                    n_prefetch_hits - b.n_prefetch_hits,
                    n_pinned_eviction_skips - b.n_pinned_eviction_skips,
                    n_pinned_resident - b.n_pinned_resident,
                    n_staging_regrow - b.n_staging_regrow,
                    n_ram_claim_hit - b.n_ram_claim_hit,
                    n_ram_claim_by_main_thread - b.n_ram_claim_by_main_thread,
                    n_ram_claim_by_prefetch - b.n_ram_claim_by_prefetch,
                    n_ram_wait_inflight_main_thread - b.n_ram_wait_inflight_main_thread,
                    t_ram_wait_inflight_ms_sum - b.t_ram_wait_inflight_ms_sum,
                    t_attn_qabsorb_batch - b.t_attn_qabsorb_batch,
                    t_moe_topk_select - b.t_moe_topk_select,
                    t_moe_phase3_dispatch - b.t_moe_phase3_dispatch,
                    t_resolve_gpu_loc_total - b.t_resolve_gpu_loc_total,
                    t_gpu_issue_total - b.t_gpu_issue_total,
                    t_gpu_finish_total - b.t_gpu_finish_total,
                    t_accumulate_total - b.t_accumulate_total,
                    t_shared_expert_total - b.t_shared_expert_total,
                    t_layer_cleanup_total - b.t_layer_cleanup_total,
                    t_active_layer_upload_async - b.t_active_layer_upload_async,
                    t_active_layer_batch_sync - b.t_active_layer_batch_sync,
                    t_shadow_precheck_total - b.t_shadow_precheck_total,
                    t_dispatch_expert_k_total - b.t_dispatch_expert_k_total,
                    t_ram_needed_dispatch_total - b.t_ram_needed_dispatch_total
            };
        }
    };

    Snapshot snapshot() const {
        return Snapshot{
            t_disk_read.load(), t_host_alloc.load(), t_gpu_upload.load(), t_expert_gpu_upload.load(),
            t_gpu_download.load(),
            t_gpu_compute.load(),
            t_expert_silu_activation.load(),
            t_cpu_expert.load(),
            t_cpu_dequant.load(), t_cpu_dot.load(),
            t_qwen_cpu_grouped_gate_up.load(), t_qwen_cpu_grouped_swiglu.load(),
            t_qwen_cpu_grouped_down.load(), t_qwen_cpu_grouped_combine.load(),
            t_attn.load(),
            t_attn_preamble.load(),
            t_attn_norm.load(), t_attn_qproj.load(), t_attn_kvproj.load(),
            t_attn_absorb.load(), t_attn_score.load(), t_attn_outproj.load(),
            t_attn_outproj_dequant.load(),
            t_dense_gpu_up.load(), t_dense_gpu_compute.load(), t_dense_gpu_down.load(),
            t_moe_route.load(), t_moe_expert.load(), t_prefetch.load(), t_dense_load.load(), t_consume_staged.load(),
            t_staging_regrow.load(), t_staging_mutex_wait.load(), t_cache_mutex_wait.load(),
            t_active_layer_upload.load(), t_active_layer_promote_copy.load(), t_vram_demote_copy.load(),
            t_hold_for_compute.load(), t_release_hold.load(), t_find_active_expert.load(),
            t_find_expert.load(), t_evict_lru.load(), t_upload_expert_async.load(),
            t_ram_get_expert.load(), t_ram_get_experts_batch.load(),
            t_ram_batch_phase_a.load(), t_ram_batch_phase_b.load(), t_ram_batch_phase_c.load(),
            t_bounce_stage_memcpy.load(), t_host_register_stage.load(), t_bounce_buffer_wait.load(),
            t_ggml_moe_d2d.load(), t_ggml_moe_h2d.load(), t_ggml_moe_h2d_dma.load(),
            n_expert_cache_hit.load(), n_expert_cache_miss.load(), n_expert_promote.load(), n_tokens_generated.load(),
            n_expert_gpu_calls.load(), n_expert_cpu_calls.load(), n_dense_gpu_calls.load(),
            bytes_disk_read.load(), bytes_gpu_uploaded.load(), bytes_bounce_staged.load(), bytes_host_registered.load(),
            (int64_t)bytes_ggml_moe_d2d.load(),
            (int64_t)bytes_ggml_moe_h2d.load(),
            bytes_disk_read_speculative.load(),
            n_prefetch_hits.load(), n_pinned_eviction_skips.load(), n_pinned_resident.load(),
            n_staging_regrow.load(),
            n_ram_claim_hit.load(), n_ram_claim_by_main_thread.load(), n_ram_claim_by_prefetch.load(),
            n_ram_wait_inflight_main_thread.load(), t_ram_wait_inflight_ms_sum.load(),
            t_attn_qabsorb_batch.load(),
            t_moe_topk_select.load(), t_moe_phase3_dispatch.load(),
            t_resolve_gpu_loc_total.load(), t_gpu_issue_total.load(), t_gpu_finish_total.load(),
            t_accumulate_total.load(), t_shared_expert_total.load(), t_layer_cleanup_total.load(),
            t_active_layer_upload_async.load(), t_active_layer_batch_sync.load(),
            t_shadow_precheck_total.load(), t_dispatch_expert_k_total.load(),
            t_ram_needed_dispatch_total.load()
        };
    }

    // CAS-loop add for atomic<double> (no native fetch_add for double until
    // C++20's atomic<double>::fetch_add, and this project targets a mix of
    // GCC versions -- keep the same pattern ScopedTimer/PERF_END already use).
    static void atomic_add(std::atomic<double>& target, double v) {
        double old = target.load(std::memory_order_relaxed);
        while (!target.compare_exchange_weak(old, old + v,
                    std::memory_order_relaxed, std::memory_order_relaxed)) {}
    }

    // CAS-loop max update for atomic<uint64_t> -- tracks a high-water-mark
    // (e.g. n_prefetch_backlog_peak) rather than a running total. Only
    // writes if v is actually greater than the current value.
    static void atomic_max(std::atomic<uint64_t>& target, uint64_t v) {
        uint64_t old = target.load(std::memory_order_relaxed);
        while (v > old && !target.compare_exchange_weak(old, v,
                    std::memory_order_relaxed, std::memory_order_relaxed)) {}
    }

    static void print_delta(const char* label, double wall_ms, const Snapshot& d);
    static void print_summary(const char* label, double wall_ms, int n_tokens, const Snapshot& d);
    void dump_report(double session_elapsed_s = 0.0) const;
    void dump_report_part1(double session_elapsed_s) const;
    void dump_report_part2(double session_elapsed_s) const;
    void dump_report_part3(double session_elapsed_s) const;
    void dump_report_part4(double session_elapsed_s) const;
    void dump_report_part5(double session_elapsed_s) const;
    void dump_report_part6(double session_elapsed_s) const;
    static double rss_gb();
};
