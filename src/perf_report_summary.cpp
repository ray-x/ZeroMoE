// PerfCounters per-token / per-request summary lines and RSS helper
// (split from perf_timer.hpp).
#ifdef __APPLE__
#include <mach/mach.h>
#endif
#include <sys/resource.h>

#include "perf_timer.hpp"

// Prints a single-line latency breakdown for a diffed window. Called once
// PER GENERATED TOKEN (see generation_loop.hpp) -- DEBUG level, since at
// INFO this would print one line per token for the entire response
// (hundreds of lines for a normal reply) with no way to turn it off.
// gpu_up/gpu_dn/gpu_cmp/cpu_exp show whether GPU round-trip overhead or
// CPU fallback dominates THIS token; use `log_level = "DEBUG"` to see
// the per-token pattern (useful for "is this latency- or throughput-
// bound", which an aggregate summary averages away).
void PerfCounters::print_delta(const char* label, double wall_ms, const Snapshot& d) {
    uint64_t hits = d.n_expert_cache_hit, misses = d.n_expert_cache_miss;
    LOG_DEBUG(
        "[%s] wall=%8.2fms  attn=%7.2fms [preamble=%6.2f norm=%6.2f qproj=%6.2f kvproj=%6.2f absorb=%6.2f "
        "score=%6.2f outproj=%6.2f] route=%6.2fms expert=%7.2fms "
        "gpu_up=%6.2fms gpu_dn=%6.2fms gpu_cmp=%6.2fms cpu_exp=%7.2fms (dequant=%6.2fms dot=%6.2fms) "
        "dense_gpu[up=%6.2f cmp=%6.2f dn=%6.2f calls=%llu] "
        "disk=%6.2fms  cache=%lluH/%lluM (%.0f%%) gpu_calls=%llu cpu_calls=%llu "
        "pipe=%llu pin_skip=%llu pin=%llu\n",
        label, wall_ms,
        d.t_attn * 1000.0,
        d.t_attn_preamble * 1000.0,
        d.t_attn_norm * 1000.0, d.t_attn_qproj * 1000.0, d.t_attn_kvproj * 1000.0,
        d.t_attn_absorb * 1000.0, d.t_attn_score * 1000.0, d.t_attn_outproj * 1000.0,
        d.t_moe_route * 1000.0, d.t_moe_expert * 1000.0,
        d.t_gpu_upload * 1000.0, d.t_gpu_download * 1000.0, d.t_gpu_compute * 1000.0,
        d.t_cpu_expert * 1000.0, d.t_cpu_dequant * 1000.0, d.t_cpu_dot * 1000.0,
        d.t_dense_gpu_up * 1000.0, d.t_dense_gpu_compute * 1000.0, d.t_dense_gpu_down * 1000.0,
        (unsigned long long)d.n_dense_gpu_calls,
        d.t_disk_read * 1000.0,
        (unsigned long long)hits, (unsigned long long)misses,
        (hits + misses) > 0 ? 100.0 * hits / (hits + misses) : 0.0,
        (unsigned long long)d.n_expert_gpu_calls, (unsigned long long)d.n_expert_cpu_calls,
        (unsigned long long)d.n_prefetch_hits, (unsigned long long)d.n_pinned_eviction_skips,
        (unsigned long long)d.n_pinned_resident);
    // Do not print this for the per-token DEBUG/INFO snapshots.  The
    // counters are intended to explain a whole request; emitting a line
    // on every token would itself perturb a short decode benchmark.
    if (label && label[0] != 'T' &&
        (d.t_qwen_cpu_grouped_gate_up > 0.0 ||
        d.t_qwen_cpu_grouped_swiglu > 0.0 ||
        d.t_qwen_cpu_grouped_down > 0.0 ||
        d.t_qwen_cpu_grouped_combine > 0.0)) {
        LOG_INFO("[%s] Qwen direct CPU stages: gate/up=%.3fs SwiGLU=%.3fs "
                 "down=%.3fs combine=%.3fs (opt-in wall-clock timing)\n",
                 label, d.t_qwen_cpu_grouped_gate_up,
                 d.t_qwen_cpu_grouped_swiglu,
                 d.t_qwen_cpu_grouped_down,
                 d.t_qwen_cpu_grouped_combine);
    }
    if (d.bytes_ggml_moe_d2d > 0 || d.bytes_ggml_moe_h2d > 0) {
        const double d2d_gb = d.bytes_ggml_moe_d2d / 1e9;
        const double h2d_gb = d.bytes_ggml_moe_h2d / 1e9;
        LOG_INFO("[%s] GGML MoE transfer window: D2D %.3f s (%.2f GB, %.2f GB/s) | "
                 "H2D %.3f s (%.2f GB, %.2f GB/s; pure DMA %.3f s, %.2f GB/s)\n",
                 label, d.t_ggml_moe_d2d, d2d_gb,
                 d.t_ggml_moe_d2d > 0.0 ? d2d_gb / d.t_ggml_moe_d2d : 0.0,
                 d.t_ggml_moe_h2d, h2d_gb,
                 d.t_ggml_moe_h2d > 0.0 ? h2d_gb / d.t_ggml_moe_h2d : 0.0,
                 d.t_ggml_moe_h2d_dma,
                 d.t_ggml_moe_h2d_dma > 0.0
                     ? h2d_gb / d.t_ggml_moe_h2d_dma
                     : 0.0);
    }
}

// Concise INFO-level summary for one finished request (prompt + all
// generated tokens) -- unlike print_delta (DEBUG, one line per token),
// this is the always-visible answer to "how fast was that and where did
// the time go", printed once when generation completes rather than only
// at process exit (dump_report, which a killed/long-running process may
// never reach, and which mixes every request together over the whole
// program lifetime rather than isolating this one).
void PerfCounters::print_summary(const char* label, double wall_ms, int n_tokens, const Snapshot& d) {
    // t_expert_gpu_upload (NOT t_gpu_upload -- see that counter's own
    // comment) so this "gpu=... [up=...]" breakdown reflects
    // expert_ffn's own per-call GPU round trip, not cache-promotion
    // slab uploads bleeding in from a shared bucket.
    double gpu_round_trip = d.t_expert_gpu_upload + d.t_gpu_compute + d.t_gpu_download;
    // Diagnostic addition (see t_active_layer_upload's own comment): these
    // three previously had ZERO instrumentation, and were the leading
    // hypothesis for why expert= vastly exceeded gpu=+cpu= in every prior
    // [REQUEST] line. lock_wait (t_cache_mutex_wait) is separate from
    // "untimed" -- it's SEPARATELY instrumented already, just never
    // printed here before; surfaced alongside so a large "untimed" total
    // caused by lock contention (as opposed to genuinely necessary copy
    // time) is visible in the SAME line instead of requiring a
    // shutdown-only dump_report() correlation.
    double untimed_total = d.t_active_layer_upload + d.t_active_layer_promote_copy + d.t_vram_demote_copy;
    // hold_for_compute/release_hold/find_active_expert's own work-under-
    // lock cost -- see t_hold_for_compute's own comment in PerfCounters
    // for why this was added (TAKE 2's fix to the first two produced no
    // measurable change in expert=/untimed=, so rather than guess again,
    // these are now measured directly instead of inferred from the
    // residual gap).
    double scan_total = d.t_hold_for_compute + d.t_release_hold + d.t_find_active_expert;
    // t_gpu_upload (NOT t_expert_gpu_upload, which is already in
    // gpu_round_trip above) -- the SYNCHRONOUS cache-promotion H2D slab
    // upload inside upload_expert() (gpu_topology.hpp), called from the
    // decode thread's cold-miss GPU dispatch path
    // (execution_layer.cpp's expert_ffn_resolve_gpu_loc, both the S>=4
    // cold-batch branch and the decode-time active-layer-failed
    // fallback branch). Was already being accumulated correctly but had
    // NEVER been printed anywhere except the per-token DEBUG line
    // (print_delta) and the shutdown-only dump_report() -- invisible in
    // every [REQUEST] line to date. Added as the next concrete
    // hypothesis after scan= (hold_for_compute/release_hold/
    // find_active_expert) was directly measured and found negligible
    // (0.65s out of a 104.92s expert= total) -- unlike that dead end,
    // this is a real, substantial cudaMemcpy (whole ~12-15MB expert
    // slabs, not small activation vectors) on a call path confirmed
    // to run on the decode thread, and it was never even a candidate
    // in prior investigation because it was invisible in the one line
    // being compared against gpu=+cpu=.
    double promote_up = d.t_gpu_upload;
    // Companion to promote_up: the CPU-side bounce-buffer staging memcpy,
    // now upload_expert_async-only (see t_bounce_stage_memcpy's own
    // comment) -- printed alongside promote_up so a real run can
    // directly compare "DMA time" vs "CPU staging time" instead of only
    // seeing their sum attributed to one bucket.
    double stage_cpy = d.t_bounce_stage_memcpy;
    // upload_expert's own replacement mechanism (see
    // t_host_register_stage's own comment) -- the number that should
    // now dominate promote_up's companion cost on the sync decode-hot
    // path, in place of what stage_cpy used to measure there.
    double reg_stage = d.t_host_register_stage;
    // Direct test of the "was this the long-mysterious untimed gap"
    // hypothesis -- see t_bounce_buffer_wait's own comment. Now
    // upload_expert_async-only; should read small if removing
    // upload_expert's competition for the same 4-slot pool was really
    // what closed the gap.
    double bb_wait = d.t_bounce_buffer_wait;
    // Second round: find_expert/evict_lru/upload_expert_async
    // (GpuTopology) and get_expert/get_experts_batch (RAMSafetyManager)
    // were the last remaining decode-time-hot-path functions with no
    // timer of their own -- see t_find_expert's comment in PerfCounters
    // for the full reasoning. Deliberately NOT mutually exclusive with
    // each other (upload_expert_async calls find_expert+evict_lru
    // internally, so their scopes nest/overlap) -- same intentional
    // pattern as t_prefill_total vs t_attn elsewhere in this file; this
    // sum is a diagnostic upper bound on "RAM/VRAM lookup+eviction
    // machinery", not a clean partition.
    double ram_scan_total = d.t_find_expert + d.t_evict_lru + d.t_upload_expert_async
                           + d.t_ram_get_expert + d.t_ram_get_experts_batch;
    uint64_t hits = d.n_expert_cache_hit, misses = d.n_expert_cache_miss;
    uint64_t gcalls = d.n_expert_gpu_calls, ccalls = d.n_expert_cpu_calls;
    double tok_s = wall_ms > 0 ? n_tokens / (wall_ms / 1000.0) : 0.0;
    double dense_gpu_total = d.t_dense_gpu_up + d.t_dense_gpu_compute + d.t_dense_gpu_down;
    // Same real-only vs incl.-padding split as dump_report's "Disk read
    // (expert)" line -- see bytes_disk_read_speculative's own comment
    // for why t_disk_read already bills padding's wall-clock, making
    // the real-only figure alone under-report true device throughput
    // whenever disk_batch_padding is active this request.
    double disk_real_gb = d.bytes_disk_read / 1e9;
    double disk_pad_gb = d.bytes_disk_read_speculative / 1e9;
    double disk_gbps_real = d.t_disk_read > 0 ? disk_real_gb / d.t_disk_read : 0.0;
    double disk_gbps_combined = d.t_disk_read > 0 ? (disk_real_gb + disk_pad_gb) / d.t_disk_read : 0.0;
    LOG_INFO(
        "[%s] %d tokens in %.2fs (%.2f tok/s)  attn=%.2fs [preamble=%.2f norm=%.2f qproj=%.2f kvproj=%.2f "
        "absorb=%.2f qabsorb=%.2f score=%.2f outproj=%.2f [outproj_dequant=%.2f]] route=%.2fs "
        "expert=%.2fs (topk_select=%.2fs gpu=%.2fs [up=%.2fs cmp=%.2fs dn=%.2fs] silu=%.2fs cpu=%.2fs [dequant=%.2fs dot=%.2fs] "
        "untimed=%.2fs [active_up=%.2fs promote_copy=%.2fs vram_demote=%.2fs] "
        "scan=%.2fs [hold=%.2fs release=%.2fs find_active=%.2fs] promote_up=%.2fs stage_cpy=%.2fs reg_stage=%.2fs bb_wait=%.2fs "
        "ram_scan=%.2fs [find_expert=%.2fs evict_lru=%.2fs upload_async=%.2fs get_expert=%.2fs "
        "get_batch=%.2fs [phase_a=%.2fs phase_b=%.2fs phase_c=%.2fs]] "
        "lock_wait=%.2fs phase3_loop=%.2fs(upper-bound,overlaps gpu/cpu/ram_scan) "
        "[resolve_loc=%.2fs(upper-bound,overlaps scan) gpu_issue=%.2fs(upper-bound,overlaps gpu_up) "
        "gpu_finish=%.2fs(upper-bound,overlaps gpu_cmp+gpu_dn) accumulate=%.2fs "
        "shared_expert=%.2fs(overlaps dense_gpu) layer_cleanup=%.2fs "
        "active_layer_prewarm_issue=%.2fs active_layer_prewarm_sync=%.2fs "
        "shadow_precheck=%.2fs(upper-bound,includes dispatch_expert_k for shadowed k's) "
        "ram_needed_dispatch=%.2fs(upper-bound,includes dispatch_expert_k for the rest) "
        "dispatch_k=%.2fs(upper-bound,overlaps resolve_loc+gpu_issue+gpu_finish)]) "
        "dense_gpu=%.2fs [up=%.2fs cmp=%.2fs dn=%.2fs calls=%llu] "
        "disk=%.2fs (%.2f GB/s real-only / %.2f GB/s incl. padding, %.2f GB real + %.2f GB padding)  "
        "cache=%.0f%% hit  expert_calls=%lluGPU/%lluCPU "
        "pipe=%llu(pin_skip=%llu pin=%llu)\n",
        label, n_tokens, wall_ms / 1000.0, tok_s,
        d.t_attn,
        d.t_attn_preamble,
        d.t_attn_norm, d.t_attn_qproj, d.t_attn_kvproj, d.t_attn_absorb, d.t_attn_qabsorb_batch,
        d.t_attn_score, d.t_attn_outproj,
        d.t_attn_outproj_dequant,
        d.t_moe_route, d.t_moe_expert,
        d.t_moe_topk_select,
        gpu_round_trip, d.t_expert_gpu_upload, d.t_gpu_compute, d.t_gpu_download,
        d.t_expert_silu_activation,
        d.t_cpu_expert,
        d.t_cpu_dequant, d.t_cpu_dot,
        untimed_total, d.t_active_layer_upload, d.t_active_layer_promote_copy, d.t_vram_demote_copy,
        scan_total, d.t_hold_for_compute, d.t_release_hold, d.t_find_active_expert, promote_up, stage_cpy, reg_stage, bb_wait,
        ram_scan_total, d.t_find_expert, d.t_evict_lru, d.t_upload_expert_async,
        d.t_ram_get_expert, d.t_ram_get_experts_batch,
        d.t_ram_batch_phase_a, d.t_ram_batch_phase_b, d.t_ram_batch_phase_c,
        d.t_cache_mutex_wait, d.t_moe_phase3_dispatch,
        d.t_resolve_gpu_loc_total, d.t_gpu_issue_total, d.t_gpu_finish_total,
        d.t_accumulate_total, d.t_shared_expert_total, d.t_layer_cleanup_total,
        d.t_active_layer_upload_async, d.t_active_layer_batch_sync,
        d.t_shadow_precheck_total, d.t_ram_needed_dispatch_total, d.t_dispatch_expert_k_total,
        dense_gpu_total, d.t_dense_gpu_up, d.t_dense_gpu_compute, d.t_dense_gpu_down,
        (unsigned long long)d.n_dense_gpu_calls,
        d.t_disk_read, disk_gbps_real, disk_gbps_combined, disk_real_gb, disk_pad_gb,
        (hits + misses) > 0 ? 100.0 * hits / (hits + misses) : 0.0,
        (unsigned long long)gcalls, (unsigned long long)ccalls,
        (unsigned long long)d.n_prefetch_hits, (unsigned long long)d.n_pinned_eviction_skips,
        (unsigned long long)d.n_pinned_resident);
    if (d.bytes_ggml_moe_d2d > 0 || d.bytes_ggml_moe_h2d > 0) {
        const double d2d_gb = d.bytes_ggml_moe_d2d / 1e9;
        const double h2d_gb = d.bytes_ggml_moe_h2d / 1e9;
        LOG_INFO("[%s] GGML MoE transfer window: D2D %.3f s (%.2f GB, %.2f GB/s) | "
                 "H2D %.3f s (%.2f GB, %.2f GB/s; pure DMA %.3f s, %.2f GB/s)\n",
                 label, d.t_ggml_moe_d2d, d2d_gb,
                 d.t_ggml_moe_d2d > 0.0 ? d2d_gb / d.t_ggml_moe_d2d : 0.0,
                 d.t_ggml_moe_h2d, h2d_gb,
                 d.t_ggml_moe_h2d > 0.0 ? h2d_gb / d.t_ggml_moe_h2d : 0.0,
                 d.t_ggml_moe_h2d_dma,
                 d.t_ggml_moe_h2d_dma > 0.0
                     ? h2d_gb / d.t_ggml_moe_h2d_dma
                     : 0.0);
    }
}

double PerfCounters::rss_gb() {
#ifdef __APPLE__
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count) == KERN_SUCCESS)
        return info.resident_size / 1e9;
    return 0;
#else
    struct rusage r;
    getrusage(RUSAGE_SELF, &r);
    return r.ru_maxrss / (1024.0 * 1024.0); // Linux: KB
#endif
}

// session_elapsed_s: REAL wall-clock seconds from "engine ready to
// accept prompts" to shutdown (see main.cpp's call site) -- used ONLY
// for the final Throughput line. Deliberately NOT derived from summing
// individual perf counters (t_disk_read + t_gpu_upload + ... etc.):
// that was this function's ORIGINAL approach, and it quietly became
// wrong as concurrency was added to this codebase (4-thread parallel
// expert fetch, GPU/CPU Pass A/B overlap, async prefetch all
// running concurrently with compute) -- summing durations that
// genuinely overlap in real time systematically OVER-counts elapsed
// time (confirmed on a real run: the old summed "total" came out to
// 244.8s for a request whose ACTUAL wall-clock duration, per the
// `[REQUEST]` line's own wall_ms-based calculation, was 118.08s -- more
// than 2x inflated), which silently made every "Throughput: X tok/s"
// line this session under-report real speed by a growing margin as
// more overlap was added. 0 is a valid default (e.g. a test harness
// that never calls generate_text) -- Tokens/Throughput lines simply
// don't print in that case, same as the old total>0 guard did.
void PerfCounters::dump_report(double session_elapsed_s) const {
    dump_report_part1(session_elapsed_s);
    dump_report_part2(session_elapsed_s);
    dump_report_part3(session_elapsed_s);
    dump_report_part4(session_elapsed_s);
    dump_report_part5(session_elapsed_s);
    dump_report_part6(session_elapsed_s);
}
