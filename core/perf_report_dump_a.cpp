// PerfCounters::dump_report sections (split from perf_timer.hpp).
#include "perf_timer.hpp"

void PerfCounters::dump_report_part1(double session_elapsed_s) const {
    (void)session_elapsed_s;
    // t_expert_gpu_upload, NOT t_gpu_upload -- see t_expert_gpu_upload's
    // own comment for the real conflation bug this split fixes (cache-
    // promotion slab uploads were bleeding into this "expert_ffn only"
    // total, inflating the reported per-call GPU upload figure).
    double gpu_round_trip = t_expert_gpu_upload + t_gpu_compute + t_gpu_download;
    LOG_INFO("\n========== MoEStream Performance Report ==========\n");
    LOG_INFO("Dense load (init):   %8.3f s\n", t_dense_load.load());
    // 2026-08-18: the plain "bytes_disk_read / t_disk_read" figure below
    // is real-demand-only bandwidth, kept exactly as it always was (see
    // bytes_disk_read_speculative's own comment for why real and
    // speculative bytes are deliberately counted separately). But
    // t_disk_read itself was NEVER real-only to begin with -- Phase B's
    // disk_batch_padding rides its speculative reads on the SAME
    // io_uring submission as the real misses (ram_manager.hpp), so
    // batch_ms (the one wall-clock this whole submission costs) already
    // includes time spent on padding bytes, added to t_disk_read
    // unconditionally. Dividing real-only bytes by a real+padding time
    // systematically UNDER-reports true device throughput whenever
    // padding is active and doing real work -- not a rounding error: a
    // real session (100.55 GB real + 12.14 GB padding) showed a "4.30
    // GB/s" real-only figure that was actually ~4.82 GB/s once padding's
    // own bytes are credited against the same shared time, matching this
    // hardware's previously-established ~4.77 GB/s ceiling almost
    // exactly (see CLAUDE.md/memory) -- i.e. disk was already near its
    // known ceiling the whole time, the metric just couldn't show it.
    // The second figure below answers "how fast did the device actually
    // move bytes" (both real and speculative, real time); the first
    // continues to answer "how much real demand did we satisfy per
    // second of blocking wait" -- deliberately keeping both rather than
    // replacing one with the other, since they answer different
    // questions and a padding-tuning A/B needs the first to not look
    // like a regression just because padding is doing more speculative
    // work in the same call.
    double disk_bytes_combined = bytes_disk_read.load() + bytes_disk_read_speculative.load();
    LOG_INFO("Disk read (expert):  %8.3f s  (%6.2f GB/s real-only / %6.2f GB/s incl. padding, "
             "%.2f GB real + %.2f GB padding)\n",
             t_disk_read.load(),
             t_disk_read > 0 ? (bytes_disk_read / 1e9) / t_disk_read : 0.0,
             t_disk_read > 0 ? (disk_bytes_combined / 1e9) / t_disk_read : 0.0,
             bytes_disk_read / 1e9,
             bytes_disk_read_speculative / 1e9);
    {
        // Real per-drive balance for the dual-SSD intra-expert split
        // (gate+up->primary, down->mirror, see engine.hpp's REPLICA_
        // SPLIT) -- io_uring path only (see bytes_disk_read_primary/
        // _mirror's own comment for why this may not sum to the exact
        // combined total above on a run with any sequential-pread
        // fallback).
        double bp = bytes_disk_read_primary.load() / 1e9;
        double bm = bytes_disk_read_mirror.load() / 1e9;
        double bsum = bp + bm;
        LOG_INFO("Disk read by drive:  primary=%.2f GB (%.1f%%) / mirror=%.2f GB (%.1f%%)"
                 " (io_uring path only; see 'Disk read (expert)' above for the combined total)\n",
                 bp, bsum > 0 ? 100.0 * bp / bsum : 0.0,
                 bm, bsum > 0 ? 100.0 * bm / bsum : 0.0);
    }
    LOG_INFO("Host alloc/setup:    %8.3f s\n", t_host_alloc.load());
    LOG_INFO("GPU upload (H2D):    %8.3f s  (%6.2f GB/s, %.2f GB total) -- cache PROMOTION "
             "slab uploads only (RAM->VRAM); expert_ffn's own small per-call activation "
             "upload is now tracked separately, see 'GPU round-trip total' below\n",
             t_gpu_upload.load(),
             t_gpu_upload > 0 ? (bytes_gpu_uploaded / 1e9) / t_gpu_upload : 0.0,
             bytes_gpu_uploaded / 1e9);
    // Qwen4Exp hybrid attention split -- see t_attn_qwen_gdn's comment.
    // Prints only when nonzero, so every other family is unaffected.
    {
        double g = t_attn_qwen_gdn.load(), d = t_attn_qwen_dense.load();
        if (g > 0.0 || d > 0.0) {
            LOG_INFO("Qwen4Exp attention split: GDN (recurrent) %8.3f s | dense full-attention %8.3f s "
                     "(%.1f%% of the two) -- GDN is a sequential O(S) recurrence, dense is O(S^2); "
                     "they need completely different fixes, which is why this split exists.\n",
                     g, d, (g + d) > 0.0 ? 100.0 * d / (g + d) : 0.0);
        }
    }
    LOG_INFO("Bounce stage memcpy: %8.3f s  (%6.2f GB/s, %.2f GB total) -- CPU-side memcpy into "
             "the pinned bounce buffer, BEFORE the DMA above -- upload_expert_async's FALLBACK path "
             "only now (2026-08-13: its common-case path converted to the same cudaHostRegister "
             "approach as upload_expert, see 'Upload-async host register' below; this line should "
             "collapse toward ~0 on a healthy run, nonzero mainly from cudaHostRegister failures). "
             "Compare this against 'GPU upload (H2D)' above: if this still dominates, the register "
             "conversion isn't engaging for some reason worth checking (see that line's own "
             "failure count)\n",
             t_bounce_stage_memcpy.load(),
             t_bounce_stage_memcpy > 0 ? (bytes_bounce_staged / 1e9) / t_bounce_stage_memcpy : 0.0,
             bytes_bounce_staged / 1e9);
    LOG_INFO("Host register stage: %8.3f s  (%6.2f GB/s, %.2f GB total) -- upload_expert's "
             "replacement for the bounce-buffer memcpy above: transient cudaHostRegister/"
             "cudaHostUnregister of the RAM tier's own slab_buffer bytes directly (Alt A of the "
             "cudaHostRegister investigation) -- %llu registration failure(s) (should be ~0; "
             "falls back to the direct pageable-source copy on failure, same as bounce-pool "
             "exhaustion previously did)\n",
             t_host_register_stage.load(),
             t_host_register_stage > 0 ? (bytes_host_registered / 1e9) / t_host_register_stage : 0.0,
             bytes_host_registered / 1e9,
             (unsigned long long)n_host_register_fail.load());
    LOG_INFO("Upload-async host register: %8.3f s register + %8.3f s deferred-unregister  "
             "(%6.2f GB/s, %.2f GB total, %llu reaped) -- upload_expert_async's OWN cudaHostRegister "
             "conversion (2026-08-13), same Alt-A mechanism as 'Host register stage' above but a "
             "SEPARATE call site: the register call is synchronous (paid here, before the async DMA "
             "is issued), but the matching cudaHostUnregister can't run until the DMA -- still in "
             "flight when upload_expert_async returns -- actually completes, so it's deferred and "
             "reaped opportunistically (GpuTopology::pending_host_unregisters); this replaces the "
             "'Bounce stage memcpy' line above for upload_expert_async's common case, not just adds "
             "to it -- %llu registration failure(s) (should be ~0; falls back to the old bounce-"
             "buffer staging on failure)\n",
             t_upload_async_host_register.load(), t_upload_async_host_unregister.load(),
             t_upload_async_host_register > 0 ? (bytes_upload_async_host_registered / 1e9) / t_upload_async_host_register : 0.0,
             bytes_upload_async_host_registered / 1e9,
             (unsigned long long)n_upload_async_deferred_unregister_reaped.load(),
             (unsigned long long)n_upload_async_host_register_fail.load());
    {
        // 2026-08-27: real coverage for upload_expert_async's own
        // "already pinned" fast path (main-cache path, used by the
        // dynamic-layer pilot promotion when speculative_promote_to_
        // active_layer=false) -- direct comparison point for the
        // active-layer path's own coverage report just below/above
        // (n_active_layer_upload_src_pinned).
        uint64_t already = n_upload_async_src_already_pinned.load();
        uint64_t total_promote = n_expert_promote.load();
        if (total_promote > 0) {
            LOG_INFO("Main-cache upload_expert_async source-pin state: %lu/%lu (%.1f%%) calls found the "
                     "source ALREADY persistently pinned (skip transient register); the rest pay a real "
                     "but cheap transient cudaHostRegister/deferred-unregister (Alt-A, see 'Upload-async "
                     "host register' above) -- either way this path's DMA is always genuinely async, "
                     "unlike upload_expert_active_async's own coverage-dependent path (see 'Active-layer "
                     "upload source-pin coverage' below).\n",
                     (unsigned long)already, (unsigned long)total_promote,
                     total_promote > 0 ? 100.0 * already / total_promote : 0.0);
        }
    }
    if (n_ram_slab_pinned.load() > 0 || n_ram_slab_pin_fail.load() > 0
        || n_ram_slab_pin_over_budget.load() > 0) {
        LOG_INFO("RAM slab persistent pin: %8.3f s to pin %llu slab(s) / %.2f GB "
                 "(%.2f GB/s) -- ONE-TIME per allocation (EngineConfig::ram_pinned_budget_gb), "
                 "not per transfer: lets upload_expert_active_async's cudaMemcpyAsync run "
                 "genuinely async (overlapping compute) instead of the driver staging it "
                 "synchronously through its own internal bounce buffer. %llu slab(s) stayed "
                 "pageable (budget exhausted -- the direct measure of what fraction of the RAM "
                 "tier this budget covers), %llu registration failure(s) (should be ~0), %llu "
                 "unregister+repin(s) from a slab regrow (should be ~0 after warmup)\n",
                 t_ram_slab_pin.load(), (unsigned long long)n_ram_slab_pinned.load(),
                 bytes_ram_slab_pinned.load() / 1e9,
                 t_ram_slab_pin > 0 ? (bytes_ram_slab_pinned / 1e9) / t_ram_slab_pin : 0.0,
                 (unsigned long long)n_ram_slab_pin_over_budget.load(),
                 (unsigned long long)n_ram_slab_pin_fail.load(),
                 (unsigned long long)n_ram_slab_unpinned_on_regrow.load());
    }
    if (n_ram_eager_preregistered.load() > 0) {
        LOG_INFO("RAM slab EAGER pre-registration (startup, single-threaded, before any request is "
                 "timed): %8.3f s to pre-register %llu empty slot(s) / %.2f GB (%.2f GB/s) -- "
                 "generic, not-yet-assigned slots sized to their own layer's real expert size, "
                 "filling the REST of ram_pinned_budget_gb beyond what pin_file/prefill_pin_file's "
                 "historically-ranked preload already covered. A real future load into one of these "
                 "slots finds slab_capacity already sufficient and skips both the realloc and the "
                 "register call -- see n_ram_eager_preregistered's own comment (perf_timer.hpp) for "
                 "the real-hardware finding (5x driver contention under 7 concurrent threads during "
                 "live prefill) this exists to avoid entirely.\n",
                 t_ram_eager_preregister.load(), (unsigned long long)n_ram_eager_preregistered.load(),
                 bytes_ram_eager_preregistered.load() / 1e9,
                 t_ram_eager_preregister > 0 ? (bytes_ram_eager_preregistered / 1e9) / t_ram_eager_preregister : 0.0);
    }
    {
        uint64_t un = n_uring_direct_unstaged.load(), st = n_uring_direct_staged.load();
        if (un + st > 0) {
            LOG_INFO("io_uring direct staging: %llu unstaged / %llu staged (%.1f%% read straight into "
                     "the caller's buffer) -- %.2f GB of scratch->dst memcpy avoided. Staging is only "
                     "needed when a request's offset/length/destination aren't all alignment-clean; "
                     "for this model's expert reads they always are, so a high unstaged share is the "
                     "expected healthy state (a low one means something upstream lost its alignment).\n",
                     (unsigned long long)un, (unsigned long long)st,
                     (un + st) > 0 ? 100.0 * un / (un + st) : 0.0,
                     bytes_uring_stage_memcpy_saved.load() / 1e9);
        }
    }
    LOG_INFO("Promotions (H2D):    %lu uploads / %.2f GB total\n",
        (unsigned long)n_expert_promote.load(),
        bytes_gpu_uploaded.load() / 1e9);
    {
        const uint64_t h = n_prefill_stream_vram_hit.load(), m = n_prefill_stream_vram_miss.load();
        if (h + m > 0)
            LOG_INFO("Prefill stream VRAM reuse: %llu / %llu (%.1f%%) routed experts were ALREADY resident "
                     "in the VRAM cache and needed no transfer at all; %llu had to be streamed.\n",
                     (unsigned long long)h, (unsigned long long)(h + m), 100.0 * h / (double)(h + m),
                     (unsigned long long)m);
    }
    {
        const uint64_t sh = n_glm_grouped_vram_shadow.load();
        const uint64_t hl = n_glm_grouped_host_lookup.load();
        if (sh + hl > 0)
            LOG_INFO("GLM grouped residency: %llu VRAM shadows bypassed get_expert() / %llu host-tier lookups "
                     "(%.1f%% of grouped expert buckets avoided a RAM/SSD materialization)\n",
                     (unsigned long long)sh, (unsigned long long)hl,
                     100.0 * sh / (double)(sh + hl));
    }
    {
        const double li = t_layer_input_hook.load();
        const double hc = t_hc_hooks.load();
        if (li > 0.0 || hc > 0.0)
            LOG_INFO("Per-layer hooks:     %8.3f s layer_input (Qwen4Exp PLE n-gram gather out of the "
                     "28.8 GB MADV_RANDOM mmap) | %8.3f s hyper-connections (hc_pre/hc_post, which "
                     "REPLACE every layer norm in Qwen4Exp -- 2 brackets per layer)\n", li, hc);
    }
    {
        const double sw = t_prefill_stream_wait.load();
        const double sh = t_prefill_stream_h2d.load();
        const uint64_t sb = bytes_prefill_stream_h2d.load();
        if (sw > 0.0 || sb > 0) {
            LOG_INFO("Prefill expert stream:  %8.3f s CPU blocked on the double-buffer handoff"
                     " | real on-GPU H2D %8.3f s (%.2f GB, %.2f GB/s) -- any file-backed"
                     " mmap->pinned staging is folded into 'disk=' in the [PREFILL] line;"
                     " resident DDR sources are reported separately as pinned/pageable-DDR."
                     " If the H2D figure is"
                     " far below this box's measured link ceiling (3090 12.30 GB/s, 5060 Ti 6.81),"
                     " the transfer is the lever; if the blocked figure greatly exceeds the H2D"
                     " figure, the GEMMs (not the transfer) are what the CPU is waiting on.\n",
                     sw, sh, sb / 1e9, sh > 0 ? (sb / 1e9) / sh : 0.0);
        }
    }
    {
        const double d2d = t_ggml_moe_d2d.load();
        const double h2d = t_ggml_moe_h2d.load();
        const double h2d_dma = t_ggml_moe_h2d_dma.load();
        const uint64_t db = bytes_ggml_moe_d2d.load();
        const uint64_t hb = bytes_ggml_moe_h2d.load();
        if (db > 0 || hb > 0) {
            LOG_INFO("GGML MoE copy stream: D2D %8.3f s (%.2f GB, %.2f GB/s) | "
                     "H2D %8.3f s (%.2f GB, %.2f GB/s) -- CUDA-event service "
                     "intervals; H2D is the PCIe-relevant number, not request-total "
                     "bytes / request-total time\n",
                     d2d, db / 1e9, d2d > 0 ? (db / 1e9) / d2d : 0.0,
                     h2d, hb / 1e9, h2d > 0 ? (hb / 1e9) / h2d : 0.0);
            if (h2d_dma > 0.0) {
                LOG_INFO("GGML MoE pure H2D DMA: %.3f s (%.2f GB, %.2f GB/s) -- "
                         "opt-in per-copy CUDA events; excludes device unpack kernels\n",
                         h2d_dma, hb / 1e9, (hb / 1e9) / h2d_dma);
            }
            LOG_INFO("GGML MoE source split: D2D=%llu experts (active-layer=%llu), "
                     "pinned-DDR-direct=%llu, pageable-DDR-staged=%llu, "
                     "file-mmap=%llu\n",
                     (unsigned long long)n_ggml_moe_d2d_experts.load(),
                     (unsigned long long)n_ggml_moe_active_d2d_experts.load(),
                     (unsigned long long)n_ggml_moe_ram_direct_experts.load(),
                     (unsigned long long)n_ggml_moe_ram_staged_experts.load(),
                     (unsigned long long)n_ggml_moe_mmap_experts.load());
        }
    }
    LOG_INFO("Expert activation upload (H2D): %8.3f s  (expert_ffn's own per-call upload, "
             "isolated from cache-promotion uploads above -- %llu calls, %.4f ms/call avg)\n",
             t_expert_gpu_upload.load(), (unsigned long long)n_expert_gpu_calls.load(),
             n_expert_gpu_calls > 0 ? 1000.0 * t_expert_gpu_upload / n_expert_gpu_calls : 0.0);
    LOG_INFO("GPU compute:         %8.3f s\n", t_gpu_compute.load());
    LOG_INFO("GPU batch sync:      %8.3f s  (%llu call(s), %.4f ms/call avg -- shared cudaStreamSynchronize "
             "for a whole Pass-B batch of pending experts on the same vault GPU, replacing what used to be up "
             "to 2 sync calls PER EXPERT; compare n_gpu_batch_sync_calls against 2x n_expert_gpu_calls to see "
             "the real reduction achieved this run)\n",
             t_gpu_batch_sync.load(), (unsigned long long)n_gpu_batch_sync_calls.load(),
             n_gpu_batch_sync_calls > 0 ? 1000.0 * t_gpu_batch_sync / n_gpu_batch_sync_calls : 0.0);
    LOG_INFO("GPU download (D2H):  %8.3f s\n", t_gpu_download.load());
    LOG_INFO("GPU round-trip total:%8.3f s  (upload+compute+download, expert_ffn only)\n", gpu_round_trip);
    LOG_INFO("Expert SiLU/mul/clamp (CPU): %8.3f s  (silu_inplace+elem_mul+swiglu_clamp_inplace on the "
             "downloaded gate/up buffers, between GPU download and the down_proj re-upload -- %llu calls, "
             "%.4f ms/call avg; previously untimed, see t_expert_silu_activation's own comment)\n",
             t_expert_silu_activation.load(), (unsigned long long)n_expert_gpu_calls.load(),
             n_expert_gpu_calls > 0 ? 1000.0 * t_expert_silu_activation / n_expert_gpu_calls : 0.0);
    LOG_INFO("CPU expert (fallback):%7.3f s  (dequant=%.3fs dot=%.3fs)\n",
             t_cpu_expert.load(), t_cpu_dequant.load(), t_cpu_dot.load());
    if (t_qwen_cpu_grouped_gate_up.load() > 0.0 ||
        t_qwen_cpu_grouped_swiglu.load() > 0.0 ||
        t_qwen_cpu_grouped_down.load() > 0.0 ||
        t_qwen_cpu_grouped_combine.load() > 0.0) {
        LOG_INFO("Qwen direct CPU stages: gate/up=%.3fs SwiGLU=%.3fs "
                 "down=%.3fs combine=%.3fs (opt-in wall-clock timing)\n",
                 t_qwen_cpu_grouped_gate_up.load(),
                 t_qwen_cpu_grouped_swiglu.load(),
                 t_qwen_cpu_grouped_down.load(),
                 t_qwen_cpu_grouped_combine.load());
    }
    LOG_INFO("Attention (MLA):     %8.3f s  (preamble=%.3f norm=%.3f qproj=%.3f kvproj=%.3f absorb=%.3f score=%.3f "
             "outproj=%.3f [outproj_dequant=%.3f])\n",
             t_attn.load(), t_attn_preamble.load(), t_attn_norm.load(), t_attn_qproj.load(), t_attn_kvproj.load(),
             t_attn_absorb.load(), t_attn_score.load(), t_attn_outproj.load(), t_attn_outproj_dequant.load());
    LOG_INFO("Prefill (whole pass):%8.3f s  (ALL layers' attention+MoE routing+MoE FFN combined, one call per "
             "request -- was previously double-counted into Attention (MLA) above; see t_prefill_total's own "
             "comment. If Attention (MLA) still doesn't reconcile against its own sub-phases after subtracting "
             "prefill's own attention share, that's a genuinely separate gap worth investigating.)\n",
             t_prefill_total.load());
    LOG_INFO("Dense GPU dispatch:  %8.3f s  (up=%.3f cmp=%.3f dn=%.3f, %lu calls, %.3fms/call avg)\n",
             t_dense_gpu_up.load() + t_dense_gpu_compute.load() + t_dense_gpu_down.load(),
             t_dense_gpu_up.load(), t_dense_gpu_compute.load(), t_dense_gpu_down.load(),
             (unsigned long)n_dense_gpu_calls.load(),
             n_dense_gpu_calls > 0
                 ? 1000.0 * (t_dense_gpu_up.load() + t_dense_gpu_compute.load() + t_dense_gpu_down.load()) / n_dense_gpu_calls.load()
                 : 0.0);
    LOG_INFO("MoE routing:         %8.3f s\n", t_moe_route.load());
    LOG_INFO("GGML router host:    %8.3f s total (%8.3f s sync, %8.3f s readback, %llu calls); device submissions %llu\n",
             t_ggml_router_host_call.load(),
             t_ggml_router_host_sync.load(),
             t_ggml_router_host_get.load(),
             (unsigned long long)n_ggml_router_host_calls.load(),
             (unsigned long long)n_ggml_router_device_calls.load());
    LOG_INFO("MoE expert FFN:      %8.3f s\n", t_moe_expert.load());
    {
        double qwen_cpu_total = 0.0;
        double qwen_gpu_total = 0.0;
        uint64_t qwen_calls = 0;
        for (int l = 0; l < kMaxTrackedLayersForMiss; ++l) {
            qwen_cpu_total += t_qwen_decode_cpu_layer[l].load();
            qwen_gpu_total += t_qwen_decode_gpu_layer[l].load();
            qwen_calls += n_qwen_decode_layer[l].load();
        }
        if (qwen_calls > 0) {
            const double qwen_total = qwen_cpu_total + qwen_gpu_total;
            LOG_INFO("Qwen decode MoE split: CPU-affinity %8.3f s | GPU %8.3f s | "
                     "%.3f s total (%llu layer calls)\n",
                     qwen_cpu_total, qwen_gpu_total, qwen_total,
                     (unsigned long long)qwen_calls);
            LOG_INFO("Qwen decode layer timing (wall, includes route/cache/expert):\n");
            for (int l = 0; l < kMaxTrackedLayersForMiss; ++l) {
                const uint64_t calls = n_qwen_decode_layer[l].load();
                if (calls == 0)
                    continue;
                const double total = t_qwen_decode_layer[l].load();
                const double cpu = t_qwen_decode_cpu_layer[l].load();
                const double gpu = t_qwen_decode_gpu_layer[l].load();
                LOG_INFO("  layer=%d path=%s calls=%llu total=%.3f s avg=%.3f ms "
                         "share=%.1f%%\n",
                         l, cpu > gpu ? "CPU-affinity" : "GPU",
                         (unsigned long long)calls, total,
                         1000.0 * total / (double)calls,
                         qwen_total > 0.0 ? 100.0 * total / qwen_total : 0.0);
            }
        }
    }
}

void PerfCounters::dump_report_part2(double session_elapsed_s) const {
    (void)session_elapsed_s;
    double gpu_round_trip = t_expert_gpu_upload + t_gpu_compute + t_gpu_download;
    LOG_INFO("Prefetch overhead:   %8.3f s\n", t_prefetch.load());
    LOG_INFO("--------------------------------------------------\n");
    uint64_t gcalls = n_expert_gpu_calls, ccalls = n_expert_cpu_calls;
    LOG_INFO("Expert path split:   %lu GPU calls / %lu CPU calls (%.1f%% GPU)\n",
             (unsigned long)gcalls, (unsigned long)ccalls,
             (gcalls + ccalls) > 0 ? 100.0 * gcalls / (gcalls + ccalls) : 0.0);
    if (gcalls > 0)
        LOG_INFO("  avg GPU round-trip: %6.3f ms/call (up=%.3f cmp=%.3f dn=%.3f)\n",
                 1000.0 * gpu_round_trip / gcalls,
                 1000.0 * t_expert_gpu_upload / gcalls, 1000.0 * t_gpu_compute / gcalls,
                 1000.0 * t_gpu_download / gcalls);
    if (ccalls > 0)
        LOG_INFO("  avg CPU expert:     %6.3f ms/call\n", 1000.0 * t_cpu_expert / ccalls);
    uint64_t hits = n_expert_cache_hit, misses = n_expert_cache_miss;
    LOG_INFO("Expert cache:        %lu hit / %lu miss (%.1f%% hit rate)\n",
             (unsigned long)hits, (unsigned long)misses,
             (hits + misses) > 0 ? 100.0 * hits / (hits + misses) : 0.0);
    // REMOVED: "Prefetch" report block used to live here
    // (n_prefetch_hits/n_prefetch_miss_*/n_prefetch_hits_prefill/decode/
    // t_prefetch_not_ready_elapsed_ms_sum/t_prefetch_still_queued_elapsed_ms_sum)
    // -- all measured the old separate staging pool's own consume/miss
    // bookkeeping, which no longer exists under the unified claim/
    // publish design (RAMSafetyManager::claim_or_wait, ram_manager.hpp).
    // See "Unified RAM claim"/"Unified RAM wait" further down for the
    // replacement instrumentation. n_prefetch_staged_later_promoted is still
    // real (see MoEExpertBlock::loaded_by_prefetch's own comment) but no
    // longer has a meaningful "of how many prefetch hits" denominator to
    // report a percentage against, so it's not printed as its own line
    // here anymore -- read it directly from a raw counter dump if
    // needed.
    LOG_INFO("Pilot hints skipped: %lu (already host-cache-resident -- avoided wasting prefetch bandwidth)\n",
             (unsigned long)n_pilot_hints_skipped.load());
    {
        uint64_t pw_att = n_prefill_warm_attempted.load();
        uint64_t pw_ok = n_prefill_warm_succeeded.load();
        uint64_t pw_fail = n_prefill_warm_failed.load();
        uint64_t pw_hint = n_prefill_hint_pushed.load();
        if (pw_att || pw_hint) {
            LOG_INFO("Prefill warm-up:     %lu attempted / %lu succeeded / %lu failed, %.3f s "
                     "(Phase 2's cold-start VRAM fill -- synchronous path; only fires when "
                     "prefill_warm_via_prefetcher=false) / %lu hint(s) pushed to the prefetcher "
                     "instead (prefill_warm_via_prefetcher=true -- served/miss visible via the "
                     "Pilot recall/prefetch lines above, since these share the same queue)\n",
                     (unsigned long)pw_att, (unsigned long)pw_ok, (unsigned long)pw_fail,
                     t_prefill_warm_time.load(), (unsigned long)pw_hint);
        }
        uint64_t pp_promo = n_prefill_pending_promoted.load();
        uint64_t pp_load = n_prefill_pending_still_loading.load();
        uint64_t pp_age = n_prefill_pending_aged_out.load();
        uint64_t pp_slot = n_prefill_pending_no_free_slot.load();
        uint64_t pp_res = n_prefill_pending_already_resident.load();
        uint64_t pp_cap = n_prefill_pending_cap_dropped.load();
        if (pp_promo || pp_load || pp_age || pp_slot || pp_res || pp_cap) {
            LOG_INFO("Prefill opportunistic VRAM promotion: %lu promoted / %lu already resident by the "
                     "time checked (something else, e.g. Phase 3's own dispatch, beat this pass to it) "
                     "/ %lu still loading (checked again next layer) / %lu aged out (never landed in "
                     "time, Phase 3's own synchronous dispatch handled it as usual) / %lu found no free "
                     "VRAM slot left / %lu never tracked at all (pending_prefill_promotions' 32-entry "
                     "cap was already full at push time -- high here means the cap, not fetch speed, "
                     "is the binding constraint) -- cross-layer follow-up for prefill_warm_via_"
                     "prefetcher's own pushed hints, not a speculative bet (every entry here is a "
                     "confirmed real router pick for this exact prefill batch). Should sum to "
                     "~n_prefill_hint_pushed above.\n",
                     (unsigned long)pp_promo, (unsigned long)pp_res, (unsigned long)pp_load,
                     (unsigned long)pp_age, (unsigned long)pp_slot, (unsigned long)pp_cap);
        }
    }
    {
        uint64_t rhits = n_pilot_recall_hits.load(), rchecks = n_pilot_recall_checks.load();
        uint64_t rdenom = n_pilot_recall_topk_sum.load();
        if (rchecks > 0) {
            LOG_INFO("Pilot recall:        %lu/%lu (%.1f%%) predicted-top-K overlap with the REAL "
                     "router's top-K, over %lu (layer, last-token) checks -- independent of worker "
                     "speed/staging pool size, measures prediction accuracy alone\n",
                     (unsigned long)rhits, (unsigned long)rdenom,
                     rdenom > 0 ? 100.0 * rhits / rdenom : 0.0, (unsigned long)rchecks);
            uint64_t already_ram = n_pilot_pred_already_ram.load();
            uint64_t already_pin = n_pilot_pred_already_vram_pinned.load();
            uint64_t already_lru = n_pilot_pred_already_vram_lru.load();
            uint64_t needs_fetch = n_pilot_pred_needs_fetch.load();
            uint64_t total_pred = already_ram + already_pin + already_lru + needs_fetch;
            LOG_INFO("Pilot prediction coverage: %lu/%lu top-K picks needed a fetch (already cached: "
                     "%lu vram-pin + %lu vram-lru + %lu ram)\n",
                     (unsigned long)needs_fetch, (unsigned long)total_pred,
                     (unsigned long)already_pin, (unsigned long)already_lru, (unsigned long)already_ram);
            uint64_t hhits = n_pilot_recall_hint_hits.load(), hdenom = n_pilot_recall_hint_checks_denom.load();
            LOG_INFO("Pilot accuracy (fetch-worthy only): %lu/%lu (%.1f%%) -- the number that actually "
                     "determines whether enabling prefetch can help this model\n",
                     (unsigned long)hhits, (unsigned long)hdenom,
                     hdenom > 0 ? 100.0 * hhits / hdenom : 0.0);

            // Per-RANK breakdown for l+1 -- see n_pilot_recall_hits_by_
            // rank's own comment (PerfCounters) for why this was added:
            // every prior "Pilot recall by rank" line was DEPTH-2 only.
            // Same two-line format (overall / fetch-worthy), one hop
            // closer where the pilot's prediction is structurally more
            // accurate than DEPTH-2's naive-reuse approximation.
            std::string rank_line_overall1 = "Pilot recall by rank (DEPTH-1, overall):     ";
            std::string rank_line_hint1    = "Pilot recall by rank (DEPTH-1, fetch-worthy): ";
            // Third line: accuracy restricted to the already_ram bucket
            // specifically -- see n_pilot_ram_only_hits_by_rank's own
            // comment (PerfCounters) for why this is a genuinely
            // different population from both lines above. THIS is the
            // number that should calibrate a RAM->VRAM promotion cutoff,
            // not the other two.
            std::string rank_line_ramonly1 = "Pilot recall by rank (DEPTH-1, ram-only):     ";
            char buf1[64];
            for (int rk = 0; rk < 8; ++rk) {
                uint64_t rh1 = n_pilot_recall_hits_by_rank[rk].load();
                uint64_t rc1 = n_pilot_recall_checks_by_rank[rk].load();
                snprintf(buf1, sizeof(buf1), "k%d=%.1f%%(n=%lu) ", rk,
                         rc1 > 0 ? 100.0 * rh1 / rc1 : 0.0, (unsigned long)rc1);
                rank_line_overall1 += buf1;
                uint64_t hh1 = n_pilot_recall_hint_hits_by_rank[rk].load();
                uint64_t hc1 = n_pilot_recall_hint_checks_by_rank[rk].load();
                snprintf(buf1, sizeof(buf1), "k%d=%.1f%%(n=%lu) ", rk,
                         hc1 > 0 ? 100.0 * hh1 / hc1 : 0.0, (unsigned long)hc1);
                rank_line_hint1 += buf1;
                uint64_t rmh1 = n_pilot_ram_only_hits_by_rank[rk].load();
                uint64_t rmc1 = n_pilot_ram_only_checks_by_rank[rk].load();
                snprintf(buf1, sizeof(buf1), "k%d=%.1f%%(n=%lu) ", rk,
                         rmc1 > 0 ? 100.0 * rmh1 / rmc1 : 0.0, (unsigned long)rmc1);
                rank_line_ramonly1 += buf1;
            }
            LOG_INFO("%s\n", rank_line_overall1.c_str());
            LOG_INFO("%s\n", rank_line_hint1.c_str());
            LOG_INFO("%s\n", rank_line_ramonly1.c_str());
        }
    }
    {
        // ATTN-PILOT report (2026-08-25, MEASUREMENT ONLY) -- predicts
        // a layer from the PREVIOUS layer's post-attention (pre-FFN)
        // state. Read this TWO ways: (1) against "Pilot recall" above
        // (DEPTH-1) -- same target-layer population, less input (missing
        // the firing layer's own FFN), but fires earlier/more lead time;
        // (2) against "Pilot recall (DEPTH-2...)" below -- both predict
        // 2 layers past this mechanism's own OWN firing layer's
        // predecessor, but this one uses the intervening layer's REAL
        // attention output instead of DEPTH-2's naive stale reuse, at
        // the cost of firing later (after that attention actually
        // runs). See execution_layer.cpp's firing site for the full
        // design. Real hints gated by EngineConfig::prefetch_attn_pilot
        // (default off) -- see n_pilot_attn_hints_pushed below.
        uint64_t rhitsA = n_pilot_attn_recall_hits.load(), rchecksA = n_pilot_attn_recall_checks.load();
        uint64_t rdenomA = n_pilot_attn_recall_topk_sum.load();
        if (rchecksA > 0) {
            LOG_INFO("Pilot recall (ATTN-PILOT): %lu/%lu (%.1f%%) predicted-top-K "
                     "overlap with the REAL router's top-K, over %lu (layer, last-token) checks -- predicts "
                     "from the PREVIOUS layer's post-attention/pre-FFN state (see CLAUDE.md for how to read "
                     "this against DEPTH-1 and DEPTH-2)\n",
                     (unsigned long)rhitsA, (unsigned long)rdenomA,
                     rdenomA > 0 ? 100.0 * rhitsA / rdenomA : 0.0, (unsigned long)rchecksA);
            uint64_t already_ram_a = n_pilot_attn_pred_already_ram.load();
            uint64_t already_pin_a = n_pilot_attn_pred_already_vram_pinned.load();
            uint64_t already_lru_a = n_pilot_attn_pred_already_vram_lru.load();
            uint64_t needs_fetch_a = n_pilot_attn_pred_needs_fetch.load();
            uint64_t total_pred_a = already_ram_a + already_pin_a + already_lru_a + needs_fetch_a;
            LOG_INFO("Pilot prediction coverage (ATTN-PILOT): %lu/%lu top-K picks needed a fetch "
                     "(already cached: %lu vram-pin + %lu vram-lru + %lu ram)\n",
                     (unsigned long)needs_fetch_a, (unsigned long)total_pred_a,
                     (unsigned long)already_pin_a, (unsigned long)already_lru_a, (unsigned long)already_ram_a);
            uint64_t hhitsA = n_pilot_attn_recall_hint_hits.load(), hdenomA = n_pilot_attn_recall_hint_checks_denom.load();
            LOG_INFO("Pilot accuracy (ATTN-PILOT, fetch-worthy only): %lu/%lu (%.1f%%)\n",
                     (unsigned long)hhitsA, (unsigned long)hdenomA,
                     hdenomA > 0 ? 100.0 * hhitsA / hdenomA : 0.0);
            std::string rank_lineA = "Pilot recall by rank (ATTN-PILOT, overall):     ";
            std::string rank_lineHintA = "Pilot recall by rank (ATTN-PILOT, fetch-worthy): ";
            char bufA[64];
            for (int rk = 0; rk < 8; ++rk) {
                uint64_t rhA = n_pilot_attn_recall_hits_by_rank[rk].load();
                uint64_t rcA = n_pilot_attn_recall_checks_by_rank[rk].load();
                snprintf(bufA, sizeof(bufA), "k%d=%.1f%%(n=%lu) ", rk,
                         rcA > 0 ? 100.0 * rhA / rcA : 0.0, (unsigned long)rcA);
                rank_lineA += bufA;
                uint64_t hhA = n_pilot_attn_recall_hint_hits_by_rank[rk].load();
                uint64_t hcA = n_pilot_attn_recall_hint_checks_by_rank[rk].load();
                snprintf(bufA, sizeof(bufA), "k%d=%.1f%%(n=%lu) ", rk,
                         hcA > 0 ? 100.0 * hhA / hcA : 0.0, (unsigned long)hcA);
                rank_lineHintA += bufA;
            }
            LOG_INFO("%s\n", rank_lineA.c_str());
            LOG_INFO("%s\n", rank_lineHintA.c_str());
            LOG_INFO("Pilot real hints pushed (ATTN-PILOT): %lu (0 when prefetch_attn_pilot=false -- compare "
                     "against %lu fetch-worthy picks above to see how much prefetch_attn_pilot_k's rank "
                     "cutoff is holding back)\n",
                     (unsigned long)n_pilot_attn_hints_pushed.load(), (unsigned long)needs_fetch_a);
        }
    }
    {
        // DEPTH-2 report -- approach (a), "naive reuse", measurement
        // only (see n_pilot2_recall_hits's own comment). Not wired to
        // any real prefetch path; exists to answer whether depth-2 is
        // worth building at all.
        uint64_t rhits2 = n_pilot2_recall_hits.load(), rchecks2 = n_pilot2_recall_checks.load();
        uint64_t rdenom2 = n_pilot2_recall_topk_sum.load();
        if (rchecks2 > 0) {
            LOG_INFO("Pilot recall (DEPTH-2, EXPERIMENTAL, measurement-only): %lu/%lu (%.1f%%) predicted-top-K "
                     "overlap with the REAL router's top-K two layers out, over %lu (layer, last-token) "
                     "checks -- naive reuse of the SAME vector the depth-1 pilot uses, missing l+1's own FFN "
                     "contribution entirely\n",
                     (unsigned long)rhits2, (unsigned long)rdenom2,
                     rdenom2 > 0 ? 100.0 * rhits2 / rdenom2 : 0.0, (unsigned long)rchecks2);
            uint64_t already_ram2 = n_pilot2_pred_already_ram.load();
            uint64_t already_pin2 = n_pilot2_pred_already_vram_pinned.load();
            uint64_t already_lru2 = n_pilot2_pred_already_vram_lru.load();
            uint64_t needs_fetch2 = n_pilot2_pred_needs_fetch.load();
            uint64_t total_pred2 = already_ram2 + already_pin2 + already_lru2 + needs_fetch2;
            LOG_INFO("Pilot prediction coverage (DEPTH-2): %lu/%lu top-K picks needed a fetch (already cached: "
                     "%lu vram-pin + %lu vram-lru + %lu ram)\n",
                     (unsigned long)needs_fetch2, (unsigned long)total_pred2,
                     (unsigned long)already_pin2, (unsigned long)already_lru2, (unsigned long)already_ram2);
            uint64_t hhits2 = n_pilot2_recall_hint_hits.load(), hdenom2 = n_pilot2_recall_hint_checks_denom.load();
            LOG_INFO("Pilot accuracy (DEPTH-2, fetch-worthy only): %lu/%lu (%.1f%%) -- the number that "
                     "determines whether depth-2 prefetch (prefetch_pilot_depth=2) can help\n",
                     (unsigned long)hhits2, (unsigned long)hdenom2,
                     hdenom2 > 0 ? 100.0 * hhits2 / hdenom2 : 0.0);
            uint64_t pushed2 = n_pilot2_hints_pushed.load();
            LOG_INFO("Pilot DEPTH-2 real hints pushed: %lu (nonzero only when prefetch_pilot_depth=2 -- "
                     "compare against %lu fetch-worthy picks above to see how much prefetch_pilot_k_depth2's "
                     "rank cutoff is holding back)\n",
                     (unsigned long)pushed2, (unsigned long)hdenom2);

            // Per-RANK breakdown -- see n_pilot2_recall_hits_by_rank's
            // own comment. Answers "does accuracy concentrate in the
            // pilot's own highest-confidence ranks" -- the question
            // that decides whether truncating to fewer fetched ranks
            // is a sound idea, as opposed to naively scaling K by the
            // aggregate fetch-worthy percentage above (which says
            // nothing about the SHAPE of the accuracy-by-rank curve).
            // Two lines (overall / fetch-worthy) mirroring the two
            // aggregate numbers just printed, one %% per rank instead
            // of one %% total.
            std::string rank_line_overall = "Pilot recall by rank (DEPTH-2, overall):     ";
            std::string rank_line_hint    = "Pilot recall by rank (DEPTH-2, fetch-worthy): ";
            char buf[64];
            for (int rk = 0; rk < 8; ++rk) {
                uint64_t rh = n_pilot2_recall_hits_by_rank[rk].load();
                uint64_t rc = n_pilot2_recall_checks_by_rank[rk].load();
                snprintf(buf, sizeof(buf), "k%d=%.1f%%(n=%lu) ", rk,
                         rc > 0 ? 100.0 * rh / rc : 0.0, (unsigned long)rc);
                rank_line_overall += buf;
                uint64_t hh = n_pilot2_recall_hint_hits_by_rank[rk].load();
                uint64_t hc = n_pilot2_recall_hint_checks_by_rank[rk].load();
                snprintf(buf, sizeof(buf), "k%d=%.1f%%(n=%lu) ", rk,
                         hc > 0 ? 100.0 * hh / hc : 0.0, (unsigned long)hc);
                rank_line_hint += buf;
            }
            LOG_INFO("%s\n", rank_line_overall.c_str());
            LOG_INFO("%s\n", rank_line_hint.c_str());
        }
    }
    {
        // UNION coverage (2026-08-25) -- real answer to "if DEPTH-1,
        // DEPTH-2, and ATTN-PILOT are ALL enabled, is there still a
        // real disk-cold miss when we reach a layer?" See n_pilot_
        // union_recall_hits's own comment for the exact accounting.
        // Non-zero regardless of which subset of the three is actually
        // enabled for REAL hints -- this measures PREDICTION coverage
        // (would a fetch have been avoidable), not real-hint behavior,
        // same "measurement always runs" convention as DEPTH-2/ATTN-
        // PILOT's own recall counters.
        uint64_t uhits = n_pilot_union_recall_hits.load(), uchecks = n_pilot_union_recall_checks.load();
        uint64_t udenom = n_pilot_union_recall_topk_sum.load();
        if (uchecks > 0) {
            double avg_union_size = (double)n_pilot_union_pred_size_sum.load() / (double)uchecks;
            LOG_INFO("Pilot UNION coverage (DEPTH-1 + DEPTH-2 + ATTN-PILOT, whichever are pending): "
                     "%lu/%lu (%.1f%%) of the real router's top-K was predicted by AT LEAST ONE "
                     "mechanism, over %lu (layer, last-token) checks, avg %.2f distinct experts/check "
                     "in the union (before intersecting with the real top-K) -- the real remaining "
                     "gap (100%% minus this) is what STILL needs a real SSD read even with everything "
                     "enabled\n",
                     (unsigned long)uhits, (unsigned long)udenom,
                     udenom > 0 ? 100.0 * uhits / udenom : 0.0, (unsigned long)uchecks, avg_union_size);
        }
    }
    {
        uint64_t issued = n_prefetch_issued.load();
        if (issued > 0) {
            uint64_t completed = n_prefetch_completed.load();
            uint64_t failed = n_prefetch_failed.load();
            double mb = bytes_prefetched.load() / 1e6;
            double avg_ms = completed > 0 ? t_prefetch_read_ms_sum.load() / completed : 0.0;
            // NOTE: this used to also report a "later actually used"
            // fraction here, computed from n_prefetch_hits -- that counter
            // fed the old separate staging pool's own consume-tracking
            // and is now permanently 0 (nothing increments it under the
            // unified claim/publish design), which would make this line
            // print a misleading, always-0.0% "later actually used"
            // forever rather than a real measurement. The genuinely
            // correct modern version of that same question -- was a
            // prefetch worker's real fetch later actually used before
            // eviction -- is already answered by "Unified RAM waste"
            // below (its complement: n_ram_claim_by_prefetch minus
            // n_ram_prefetch_loaded_evicted_unused), so it's not
            // recomputed redundantly here.
            LOG_INFO("Prefetch real fetches: %lu issued, %lu completed / %lu failed | %.2f MB read, "
                     "avg %.2f ms/fetch\n",
                     (unsigned long)issued, (unsigned long)completed, (unsigned long)failed, mb, avg_ms);

            // Concurrency-bucketed latency -- see n_prefetch_fetch_by_
            // concurrency's own comment for the full rationale. Read
            // this against the isolated single-fetch benchmark
            // (tools/gguf_tool dma --experts 1, real request shape):
            // bucket "0 others" should be the most directly comparable
            // number. If it's still elevated vs the isolated figure,
            // concurrency isn't the (whole) explanation; if it matches
            // and higher buckets climb, concurrent contention is
            // confirmed as a real, measured cause.
            {
                std::ostringstream oss;
                oss << "Prefetch fetch latency by concurrency (N other prefetch-worker fetches "
                       "in flight at submit time): ";
                bool any = false;
                for (int b = 0; b < PREFETCH_CONCURRENCY_BUCKETS; ++b) {
                    uint64_t c = n_prefetch_fetch_by_concurrency[b].load();
                    if (c == 0) continue;
                    any = true;
                    double avg_b = t_prefetch_fetch_ms_by_concurrency[b].load() / (double)c;
                    oss << (b == PREFETCH_CONCURRENCY_BUCKETS - 1 ? (std::to_string(b) + "+") : std::to_string(b))
                        << "=" << std::fixed << std::setprecision(2) << avg_b << "ms(n=" << c << ") ";
                }
                if (any) LOG_INFO("%s\n", oss.str().c_str());
            }

            // DEPTH-2's marginal share of the above -- see
            // n_prefetch_issued_depth2's own comment for why this is a
            // subset, not a separate total, and why "depth-1's share"
            // isn't computed here as a literal subtraction (the
            // aggregate also includes Phase 2 prefill-warm hints).
            // Only printed once depth-2 has actually issued at least
            // one real fetch -- silent (not a misleading 0-line) when
            // prefetch_pilot_depth=1, the default.
            uint64_t issued2 = n_prefetch_issued_depth2.load();
            if (issued2 > 0) {
                uint64_t completed2 = n_prefetch_completed_depth2.load();
                uint64_t failed2 = n_prefetch_failed_depth2.load();
                uint64_t beaten2 = n_prefetch_claims_beaten_depth2.load();
                double mb2 = bytes_prefetched_depth2.load() / 1e6;
                double avg_ms2 = completed2 > 0 ? t_prefetch_read_ms_sum_depth2.load() / completed2 : 0.0;
                LOG_INFO("Prefetch real fetches (DEPTH-2 only): %lu issued (%.1f%% of all real fetches), "
                         "%lu completed / %lu failed / %lu claim-beaten (already resident/loading by the "
                         "time a worker got to it) | %.2f MB read (%.1f%% of all prefetched bytes), "
                         "avg %.2f ms/fetch\n",
                         (unsigned long)issued2, issued > 0 ? 100.0 * issued2 / issued : 0.0,
                         (unsigned long)completed2, (unsigned long)failed2, (unsigned long)beaten2,
                         mb2, mb > 0 ? 100.0 * mb2 / mb : 0.0, avg_ms2);
            }
        }
    }
}

void PerfCounters::dump_report_part3(double session_elapsed_s) const {
    (void)session_elapsed_s;
    uint64_t hits = n_expert_cache_hit, misses = n_expert_cache_miss;
    LOG_INFO("VRAM->RAM demote:    %lu salvaged / %lu skipped (benign) / %lu failed "
             "(vram_evict_demote_to_ram: instead of dropping an evicted expert's bytes, D2H "
             "copy them back into the RAM tier before the VRAM slot is overwritten) -- "
             "%lu fell back to legacy stream-0 cudaMemcpy (should be 0; nonzero means the "
             "dedicated-stream fix silently didn't engage for those calls)\n",
             (unsigned long)n_vram_evict_demoted.load(),
             (unsigned long)n_vram_evict_demote_skipped.load(),
             (unsigned long)n_vram_evict_demote_failed.load(),
             (unsigned long)n_vram_demote_stream_unavailable.load());
    {
        uint64_t cb = n_demote_saw_compute_stream_busy.load(), ci = n_demote_saw_compute_stream_idle.load();
        uint64_t pb = n_demote_saw_copy_stream_busy.load(), pi = n_demote_saw_copy_stream_idle.load();
        uint64_t ct = cb + ci, pt = pb + pi;
        LOG_INFO("Demote stream-busy check: compute_stream busy %lu/%lu (%.1f%%) / copy_stream busy %lu/%lu (%.1f%%) "
                 "at the moment demote_from_vram's D2H fired -- direct test of whether there was genuinely "
                 "concurrent other-stream work to avoid waiting for (see n_demote_saw_* comment)\n",
                 (unsigned long)cb, (unsigned long)ct, ct > 0 ? 100.0 * cb / ct : 0.0,
                 (unsigned long)pb, (unsigned long)pt, pt > 0 ? 100.0 * pb / pt : 0.0);
    }
    {
        uint64_t acb = n_active_upload_saw_compute_stream_busy.load();
        uint64_t aci = n_active_upload_saw_compute_stream_idle.load();
        uint64_t apb = n_active_upload_saw_copy_stream_busy.load();
        uint64_t api = n_active_upload_saw_copy_stream_idle.load();
        uint64_t act = acb + aci, apt = apb + api;
        if (act > 0 || apt > 0) {
            LOG_INFO("Active-layer upload stream-busy check: compute_stream busy %lu/%lu (%.1f%%) / "
                     "copy_stream busy %lu/%lu (%.1f%%) at the moment upload_expert_active's H2D fired "
                     "-- same diagnostic as the Demote check above, different call site (GLM's dominant "
                     "dispatch path). A real busy%% here, unlike demote_from_vram's near-0%%, would "
                     "justify re-trying the async conversion already reverted for this call site\n",
                     (unsigned long)acb, (unsigned long)act, act > 0 ? 100.0 * acb / act : 0.0,
                     (unsigned long)apb, (unsigned long)apt, apt > 0 ? 100.0 * apb / apt : 0.0);
        }
    }
    {
        uint64_t alb = n_active_upload_saw_active_layer_stream_busy.load();
        uint64_t ali = n_active_upload_saw_active_layer_stream_idle.load();
        uint64_t spb = n_active_upload_saw_speculative_stream_busy.load();
        uint64_t spi = n_active_upload_saw_speculative_stream_idle.load();
        uint64_t alt = alb + ali, spt = spb + spi;
        if (alt > 0 || spt > 0) {
            LOG_INFO("Active-layer upload PRIORITY-INVERSION check: active_layer_stream busy %lu/%lu "
                     "(%.1f%%) / speculative_prefetch_stream(s) busy %lu/%lu (%.1f%%) at the moment "
                     "upload_expert_active's cudaMemcpyAsync fired. 2026-08-29: this call was moved off "
                     "the legacy default (NULL) stream onto its own dedicated "
                     "active_layer_sync_fallback_stream (see that field's own comment) specifically "
                     "because the default stream implicitly waits for ALL other blocking-type streams to "
                     "drain before it can even start -- so these busy%% figures are now background-load "
                     "measurements only, retained for before/after comparison, not a live prediction of "
                     "this call's own wait time the way they were pre-fix.\n",
                     (unsigned long)alb, (unsigned long)alt, alt > 0 ? 100.0 * alb / alt : 0.0,
                     (unsigned long)spb, (unsigned long)spt, spt > 0 ? 100.0 * spb / spt : 0.0);
        }
        uint64_t dup = n_active_layer_sync_duplicate_transfer.load();
        int64_t dup_bytes = bytes_active_layer_sync_duplicate_transfer.load();
        if (dup > 0) {
            LOG_INFO("Active-layer SYNC-FALLBACK duplicate transfer: %lu time(s) (%.2f GB) upload_expert_"
                     "active's fallback re-fetched a (layer,expert) that was ALREADY present elsewhere in "
                     "active_layer_slots (an in-flight decode_async_prewarm or speculative async issue) -- "
                     "direct answer to 'do we double-queue the same MoE'. Should be ~0 now that the wait-"
                     "on-inflight fix (below) intercepts this case before it reaches victim-selection; a "
                     "nonzero figure here post-fix means the identity changed between the check and the "
                     "wait (see that fix's own fallback-path comment).\n",
                     (unsigned long)dup, dup_bytes / 1e9);
        }
        uint64_t wi = n_active_layer_wait_on_inflight.load();
        uint64_t wir = n_active_layer_wait_on_inflight_ready.load();
        double wit = t_active_layer_wait_on_inflight.load();
        if (wi > 0 || wir > 0) {
            LOG_INFO("Active-layer wait-on-inflight (duplicate-transfer fix): %lu waited (%.3fs total, "
                     "%.3fms/call avg) + %lu found already-landed (no wait needed) -- these replace what "
                     "used to become a wasted duplicate DDR->VRAM transfer plus a legacy-default-stream "
                     "priority-inversion stall.\n",
                     (unsigned long)wi, wit, wi > 0 ? (wit * 1000.0 / wi) : 0.0, (unsigned long)wir);
        }
    }
    {
        uint64_t pcb = n_active_prewarm_saw_compute_stream_busy.load();
        uint64_t pci = n_active_prewarm_saw_compute_stream_idle.load();
        uint64_t ppb = n_active_prewarm_saw_copy_stream_busy.load();
        uint64_t ppi = n_active_prewarm_saw_copy_stream_idle.load();
        uint64_t pct = pcb + pci, ppt = ppb + ppi;
        if (pct > 0 || ppt > 0) {
            LOG_INFO("Active-layer BATCH prewarm stream-busy check: compute_stream busy %lu/%lu (%.1f%%) / "
                     "copy_stream busy %lu/%lu (%.1f%%) at the moment upload_expert_active_async's H2D "
                     "fired -- same diagnostic again, THIRD call site (the batch-prewarm path -- see "
                     "n_active_prewarm_saw_*'s own comment for why the two checks above don't cover this "
                     "one). Near-0%% would confirm active_layer_prewarm_issue is genuinely bandwidth-bound "
                     "(no scheduling fix available); a real busy%% would mean there's other decode-thread "
                     "GPU work this DMA could be overlapped with instead of serializing behind\n",
                     (unsigned long)pcb, (unsigned long)pct, pct > 0 ? 100.0 * pcb / pct : 0.0,
                     (unsigned long)ppb, (unsigned long)ppt, ppt > 0 ? 100.0 * ppb / ppt : 0.0);
        }
    }
    {
        // Direct test of the two leading candidates for demote_from_vram's
        // remaining ~23% unexplained residual (see CLAUDE.md): per-GPU
        // ms/call + GB/s (multi-GPU PCIe asymmetry -- this box's 3090 is
        // x16, 5060 Ti is x8) and overall avg slab size for the
        // specifically-EVICTED population (vs the ~8.26MB corpus-wide
        // figure used to benchmark).
        int64_t total_bytes = 0; uint64_t total_calls = 0;
        for (int g = 0; g < PERF_MAX_GPU_SLOTS; g++) {
            uint64_t n = n_vram_demote_by_gpu[g].load();
            if (n == 0) continue;
            double t = t_vram_demote_by_gpu[g].load();
            int64_t b = bytes_vram_demote_by_gpu[g].load();
            total_bytes += b; total_calls += n;
            LOG_INFO("  demote gpu=%d: %lu calls, %.3f ms/call, %.2f GB/s, avg slab %.2f MB\n",
                     g, (unsigned long)n, t * 1000.0 / n,
                     t > 0 ? (b / 1e9) / t : 0.0, (b / 1e6) / (double)n);
        }
        if (total_calls > 0) {
            LOG_INFO("Demote per-GPU/size breakdown: %lu total calls, avg slab %.2f MB "
                     "(compare against the ~8.26MB corpus-wide figure used to benchmark)\n",
                     (unsigned long)total_calls, (total_bytes / 1e6) / (double)total_calls);
        }
    }
    LOG_INFO("Three-tier caching:  %lu active-layer uploads / %lu active-layer hits / %lu cleanups / "
             "%lu promoted-to-main (MOVE vs COPY: %lu promotions freed RAM slots vs %lu left occupied, "
             "%lu of the freed ones were COPY-mode reclaimed under ram_copy_reclaim_threshold pressure)\n",
             (unsigned long)n_active_layer_uploads.load(),
             (unsigned long)n_active_layer_hits.load(),
             (unsigned long)n_active_layer_cleanup.load(),
             (unsigned long)n_active_layer_promoted.load(),
             (unsigned long)n_move_on_promotion.load(),
             (unsigned long)n_copy_on_promotion.load(),
             (unsigned long)n_copy_reclaimed_threshold.load());
    if (uint64_t demand_async = n_active_layer_uploads_demand_async.load()) {
        LOG_INFO("Active-layer demand async uploads: %llu (fenced on the "
                 "consumer stream)\n",
                 (unsigned long long)demand_async);
    }
    {
        uint64_t samples = active_layer_decode_occupancy_samples.load();
        if (samples > 0) {
            uint64_t max_occ = active_layer_decode_occupancy_max.load();
            double avg_occ = (double)active_layer_decode_occupancy_sum.load() / (double)samples;
            LOG_INFO("[ACTIVE-LAYER-OCCUPANCY] decode-only: max=%lu avg=%.2f slots occupied "
                     "(of %lu samples, one per cleanup_active_layer_area call while !in_prefill) "
                     "-- compare against active_layer_capacity_slots to judge real headroom vs waste\n",
                     (unsigned long)max_occ, avg_occ, (unsigned long)samples);
        }
        uint64_t psamples = active_layer_prefill_occupancy_samples.load();
        if (psamples > 0) {
            uint64_t pmax = active_layer_prefill_occupancy_max.load();
            double pavg = (double)active_layer_prefill_occupancy_sum.load() / (double)psamples;
            LOG_INFO("[ACTIVE-LAYER-OCCUPANCY] prefill-only: max=%lu avg=%.2f slots occupied "
                     "(of %lu samples)\n", (unsigned long)pmax, pavg, (unsigned long)psamples);
        }
    }
    {
        uint64_t not_ready = n_active_layer_poll_not_ready.load();
        uint64_t hits = n_active_layer_hits.load();
        uint64_t stale = n_active_layer_stale_hold_released.load();
        if (not_ready > 0 || hits > 0 || stale > 0) {
            LOG_INFO("Active-layer poll-not-ready: %lu/%lu (%.1f%%) lookups found the identity already "
                     "resident but its async DMA still in flight -- treated as a miss (poll, don't block), "
                     "see find_active_expert's own comment. Healthy at a low rate; high relative to hits "
                     "means speculative fills routinely don't have enough real slack time before real "
                     "dispatch reaches them. %lu stale hold(s) released by cleanup's own opportunistic "
                     "poll (a completed DMA nothing ever consulted before its layer window closed) -- "
                     "should track close to the 'never touched by a real dispatch' waste figure above.\n",
                     (unsigned long)not_ready, (unsigned long)(not_ready + hits),
                     (not_ready + hits) > 0 ? 100.0 * not_ready / (not_ready + hits) : 0.0,
                     (unsigned long)stale);
        }
        {
            uint64_t nr_dp = n_active_layer_poll_not_ready_decode_prewarm.load();
            uint64_t nr_sp = n_active_layer_poll_not_ready_speculative.load();
            if (nr_dp > 0 || nr_sp > 0) {
                LOG_INFO("Active-layer poll-not-ready by source: decode_async_prewarm %lu, "
                         "speculative pilot fill %lu -- decode_async_prewarm issues right before its OWN "
                         "token's dispatch (near-zero natural lead time), so a high figure here specifically "
                         "means decode_async_prewarm_nonblocking's transfers are routinely aborting (poll "
                         "miss) and falling through to a fresh synchronous fetch anyway, on top of the "
                         "abandoned async DMA's own copy-engine cost -- see that flag's own comment.\n",
                         (unsigned long)nr_dp, (unsigned long)nr_sp);
            }
        }
        {
            double au_t = t_active_layer_upload_async.load();
            int64_t au_b = bytes_active_layer_upload_async.load();
            if (au_t > 0 && au_b > 0) {
                // 2026-08-28 CORRECTION: this "achieved bandwidth" figure
                // was previously (wrongly) read as evidence GPU0's PCIe
                // link was near its wire ceiling -- direct nvidia-smi
                // dmon telemetry during a real run disproved that (RX
                // averaged ~2.2-2.6 GB/s, peak ~9-13 GB/s touched
                // rarely, nowhere near saturated). The real reason this
                // ratio looks high: it blends near-zero issue time for
                // PINNED sources (genuinely async) with real, large
                // issue time for UNPINNED sources (a driver-internal
                // staged copy that's actually CPU/host-memory-bandwidth
                // -bound, not PCIe-bound, hiding inside a call that
                // looks async) -- see the pinned-vs-unpinned split just
                // below for the real, decisive per-call breakdown. Kept
                // as a raw bytes/time figure for reference, not as
                // evidence of anything being bandwidth-bound.
                LOG_INFO("Active-layer async upload bytes/time (NOT a clean bandwidth measurement -- "
                         "see pinned-vs-unpinned split below): %.2f GB moved in %.2fs (%.2f GB/s blended) "
                         "via upload_expert_active_async (decode_async_prewarm + speculative pilot fill "
                         "combined -- they share this call site/timer).\n",
                         au_b / 1e9, au_t, (au_b / 1e9) / au_t);
            }
        }
        {
            double tp = t_active_layer_upload_async_pinned.load();
            double tu = t_active_layer_upload_async_unpinned.load();
            uint64_t np = n_active_layer_upload_async_pinned_calls.load();
            uint64_t nu = n_active_layer_upload_async_unpinned_calls.load();
            if (np > 0 || nu > 0) {
                LOG_INFO("Active-layer async upload PINNED-vs-UNPINNED per-call cost (the decisive "
                         "check -- PCIe confirmed NOT saturated via real nvidia-smi telemetry, so a "
                         "large gap here means the cost is a CPU-bound driver-internal staged copy for "
                         "unpinned sources, not a wire-speed transfer cost): pinned %lu calls, "
                         "%.2fs total, %.3fms/call avg  |  unpinned %lu calls, %.2fs total, %.3fms/call "
                         "avg  |  unpinned is %.1fx the pinned per-call cost.\n",
                         (unsigned long)np, tp, np > 0 ? (tp * 1000.0 / np) : 0.0,
                         (unsigned long)nu, tu, nu > 0 ? (tu * 1000.0 / nu) : 0.0,
                         (np > 0 && nu > 0 && tp > 0) ? ((tu / nu) / (tp / np)) : 0.0);
            }
        }
        {
            // Unlike the legacy issue-time ratio above, these four
            // buckets are CUDA-event measurements of the actual H2D
            // interval on the selected stream.  They are the numbers to
            // compare with the RTX 3090 PCIe Gen3 x16 ceiling (~15.75
            // GB/s), while the CPU issue and batch-wait timers explain
            // host-side staging/queueing separately.
            auto log_dma = [](const char* label, double seconds,
                              int64_t bytes, uint64_t calls) {
                if (calls == 0 || seconds <= 0.0 || bytes <= 0) return;
                LOG_INFO("Active-layer H2D CUDA-event %-18s: %lu calls, %.2f GB in %.3fs = %.2f GB/s\n",
                         label, (unsigned long)calls, bytes / 1e9, seconds,
                         (bytes / 1e9) / seconds);
            };
            log_dma("async+pinned", t_active_layer_dma_async_pinned.load(),
                    bytes_active_layer_dma_async_pinned.load(),
                    n_active_layer_dma_async_pinned.load());
            log_dma("async+pageable", t_active_layer_dma_async_unpinned.load(),
                    bytes_active_layer_dma_async_unpinned.load(),
                    n_active_layer_dma_async_unpinned.load());
            log_dma("sync+pinned", t_active_layer_dma_sync_pinned.load(),
                    bytes_active_layer_dma_sync_pinned.load(),
                    n_active_layer_dma_sync_pinned.load());
            log_dma("sync+pageable", t_active_layer_dma_sync_unpinned.load(),
                    bytes_active_layer_dma_sync_unpinned.load(),
                    n_active_layer_dma_sync_unpinned.load());
        }
        {
            uint64_t sub = n_active_layer_worker_submitted.load();
            uint64_t qf = n_active_layer_worker_queue_full.load();
            uint64_t proc = n_active_layer_worker_processed.load();
            if (sub > 0 || qf > 0) {
                // 2026-08-28: no register/unregister stats printed here
                // any more -- this pool no longer calls cudaHostRegister
                // at all (see ActiveLayerUploadWorkers::process's own
                // comment for the real-hardware crash this removed).
                // t_active_layer_worker_register/n_*_register_calls/
                // n_*_register_fail/t_*_unregister/n_*_unregister_calls
                // are kept declared (permanently 0 now) rather than
                // removed, purely to avoid a wider ripple through
                // anything that might reference them.
                LOG_INFO("Active-layer upload worker pool: %lu submitted / %lu queue-full "
                         "(fell back) / %lu claim-failed (fell back) / %lu processed (plain "
                         "unpinned cudaMemcpyAsync only -- no cudaHostRegister, see this "
                         "pool's own comment).\n",
                         (unsigned long)sub, (unsigned long)qf,
                         (unsigned long)n_active_layer_worker_claim_failed.load(),
                         (unsigned long)proc);
                uint64_t bad_item = n_active_layer_worker_invalid_item.load();
                uint64_t copy_fail = n_active_layer_worker_copy_fail.load();
                if (bad_item > 0 || copy_fail > 0) {
                    LOG_INFO("Active-layer upload worker pool DEFENSIVE-CHECK HITS (should both be 0 -- "
                             "see ActiveLayerUploadWorkers::process's own comment, added while chasing a "
                             "real NVRM Xid 31 hardware crash): %lu invalid work item(s), %lu async-copy "
                             "failure(s) caught.\n",
                             (unsigned long)bad_item, (unsigned long)copy_fail);
                }
            }
        }
        {
            // Direct answer to "do all data transfers go to one GPU" --
            // combined across all four H2D upload call sites.
            std::string line = "H2D upload bytes by GPU (all 4 upload call sites combined): ";
            int64_t total_b = 0;
            bool any = false;
            for (int g = 0; g < PERF_MAX_GPU_SLOTS; ++g) {
                int64_t b = bytes_h2d_upload_by_gpu[g].load();
                uint64_t n = n_h2d_upload_by_gpu[g].load();
                if (n == 0) continue;
                any = true;
                total_b += b;
                char buf[128];
                snprintf(buf, sizeof(buf), "gpu%d=%.2fGB/%lucalls ", g, b / 1e9, (unsigned long)n);
                line += buf;
            }
            if (any) {
                LOG_INFO("%s(%.2f GB total)\n", line.c_str(), total_b / 1e9);
            }
        }
        {
            // Multi-GPU active-layer ROUTING decisions specifically
            // (issued before the actual upload call -- compare against
            // the H2D-by-GPU line above, which shows what actually
            // landed, to see how often a routed candidate later missed
            // its target, e.g. pool-full on that specific GPU).
            std::string line = "Active-layer routing decisions by GPU: ";
            uint64_t total_n = 0;
            bool any = false;
            for (int g = 0; g < 8; ++g) {
                uint64_t n = n_active_layer_routed_by_gpu[g].load();
                if (n == 0) continue;
                any = true;
                total_n += n;
                char buf[64];
                snprintf(buf, sizeof(buf), "gpu%d=%lu ", g, (unsigned long)n);
                line += buf;
            }
            if (any) {
                LOG_INFO("%s(%lu total)\n", line.c_str(), (unsigned long)total_n);
            }
        }
        {
            // Direct check on whether a routed GPU's slice has a
            // different real hit-rate than the others' -- the actual
            // empirical validation the multi-GPU design was built to
            // allow, not an assumption.
            std::string line = "Active-layer used-rate by GPU: ";
            bool any = false;
            for (int g = 0; g < 8; ++g) {
                uint64_t u = n_active_layer_used_by_gpu[g].load();
                uint64_t w = n_active_layer_wasted_by_gpu[g].load();
                uint64_t t = u + w;
                if (t == 0) continue;
                any = true;
                char buf[64];
                snprintf(buf, sizeof(buf), "gpu%d=%.1f%%(n=%lu) ", g, 100.0 * u / t, (unsigned long)t);
                line += buf;
            }
            if (any) LOG_INFO("%s\n", line.c_str());
        }
        uint64_t alb = n_speculative_issue_saw_active_layer_stream_busy.load();
        uint64_t ali = n_speculative_issue_saw_active_layer_stream_idle.load();
        uint64_t osb = n_speculative_issue_saw_own_stream_busy.load();
        uint64_t osi = n_speculative_issue_saw_own_stream_idle.load();
        if (alb + ali > 0) {
            LOG_INFO("Speculative issue-time stream contention: active_layer_stream busy at issue time "
                     "%lu/%lu (%.1f%%) -- if near-zero, the dedicated speculative_prefetch_stream split "
                     "isn't avoiding real cross-mechanism contention (decode_async_prewarm's own traffic "
                     "genuinely wasn't in the way), so that isn't what explains active_layer_prewarm_"
                     "issue's own growth. own stream (speculative_prefetch_stream) busy at issue time "
                     "%lu/%lu (%.1f%%) -- nonzero here means back-to-back speculative issues are queuing "
                     "behind each other (self-contention), a different mechanism from cross-stream "
                     "contention.\n",
                     (unsigned long)alb, (unsigned long)(alb + ali),
                     (alb + ali) > 0 ? 100.0 * alb / (alb + ali) : 0.0,
                     (unsigned long)osb, (unsigned long)(osb + osi),
                     (osb + osi) > 0 ? 100.0 * osb / (osb + osi) : 0.0);
        }
        uint64_t alp = n_active_layer_upload_src_pinned.load();
        uint64_t alu = n_active_layer_upload_src_unpinned.load();
        uint64_t sfp = n_speculative_fill_src_pinned.load();
        uint64_t sfu = n_speculative_fill_src_unpinned.load();
        if (alp + alu > 0) {
            LOG_INFO("Active-layer upload source-pin coverage: %lu/%lu (%.1f%%) of upload_expert_active_"
                     "async calls (ALL callers: prefill Phase-2, decode_async_prewarm, speculative fill "
                     "combined) had a source already covered by ram_pinned_budget_gb's persistent pin -- "
                     "the rest paid the driver's secretly-synchronous pageable-memory staging cost, since "
                     "this function (unlike upload_expert_async, the main-cache path) does no pin-checking "
                     "or transient cudaHostRegister of its own. Speculative-fill-only: %lu/%lu (%.1f%%) -- "
                     "compare against upload_expert_async's own 'already pinned' rate for the SAME "
                     "candidate population when speculative_promote_to_active_layer=false, to see whether "
                     "the active-layer path's lack of pin-awareness is really costing coverage the main-"
                     "cache path wouldn't have lost.\n",
                     (unsigned long)alp, (unsigned long)(alp + alu),
                     (alp + alu) > 0 ? 100.0 * alp / (alp + alu) : 0.0,
                     (unsigned long)sfp, (unsigned long)(sfp + sfu),
                     (sfp + sfu) > 0 ? 100.0 * sfp / (sfp + sfu) : 0.0);
        }
        uint64_t udist = n_unpinned_touch_distinct.load();
        uint64_t utot = n_unpinned_touch_total.load();
        if (utot > 0) {
            LOG_INFO("Unpinned active-layer touch concentration: %lu distinct (layer,expert) key(s) "
                     "account for %lu unpinned upload_expert_active_async touches (%.1fx repeat rate) -- "
                     "probe for whether a dynamic pin-swap (track live hotness, keep the budget pointed "
                     "at CURRENTLY-hot slabs instead of whichever loaded first) would meaningfully help. "
                     "A repeat rate near 1.0x means the unpinned traffic is genuinely spread across the "
                     "whole RAM-tier working set (each miss is a different expert, rarely revisited) -- "
                     "dynamic pinning would have to churn constantly for little gain. A repeat rate well "
                     "above 1.0x means a small hot subset keeps missing pin coverage -- dynamic pinning "
                     "could plausibly close much of the gap cheaply by keeping just that subset pinned.\n",
                     (unsigned long)udist, (unsigned long)utot,
                     udist > 0 ? (double)utot / (double)udist : 0.0);
        }
    }
}
