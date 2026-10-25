#pragma once
// Counter slice 1 of 8 split mechanically from perf_timer.hpp. Slices form a
// linear inheritance chain so constants declared in an earlier slice stay visible
// in later ones; `PerfCounters` derives from the last slice.
#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

struct PerfCountersPart01 {
    // Cumulative timings (seconds)
    std::atomic<double> t_disk_read{0};     // Expert slab pread from disk
    // --- moe_prefill_stream (include/prefill_expert_stream.hpp) ---
    // t_prefill_stream_wait is the CPU thread blocked on the GPU: the
    // double-buffer handoff (cudaEventSynchronize on the previous chunk's
    // compute), which transitively covers that chunk's H2D. Separated from
    // the staging memcpy (folded into t_disk_read, so the [PREFILL] line's
    // `disk=` stays comparable against the pre-streaming baseline) because
    // the two answer different questions: staging is CPU/page-cache bound,
    // this is PCIe/compute bound, and knowing WHICH dominates decides
    // whether the next lever is chunk size or transfer bandwidth.
    std::atomic<double> t_prefill_stream_wait{0};
    // Per-layer hooks outside the attn/expert/dense buckets. qwen4exp runs
    // low-rank hyper-connections in place of EVERY layer norm (2 brackets
    // per layer) and a PLE n-gram gather out of a 28.8 GB MADV_RANDOM mmap,
    // neither of which had a timer -- together they were the bulk of a
    // ~5.7 s unattributed slice of a 20 s prefill.
    std::atomic<uint64_t> n_prefill_stream_vram_hit{0};
    std::atomic<uint64_t> n_prefill_stream_vram_miss{0};
    // GLM5NEXT grouped-prefill residency decision. A shadow is a metadata-
    // only block used when the routed expert is already in VRAM, so this
    // count measures grouped expert buckets that bypassed get_expert() and
    // therefore could not trigger a redundant SSD/RAM materialization.
    // host_lookup counts the remaining grouped buckets that did consult the
    // RAM tier (a real miss or a GPU-unsupported type).
    std::atomic<uint64_t> n_glm_grouped_vram_shadow{0};
    std::atomic<uint64_t> n_glm_grouped_host_lookup{0};
    std::atomic<double> t_layer_input_hook{0};
    std::atomic<double> t_hc_hooks{0};
    std::atomic<double> t_prefill_stream_h2d{0};  // real on-GPU H2D, CUDA-event timed
    std::atomic<uint64_t> bytes_prefill_stream_h2d{0};
    // Source-separated timing for the GGML MUL_MAT_ID MoE runner. These are
    // copied from CUDA events at the forward-level drain, so they describe
    // copy-stream service time rather than request wall time. Keep them
    // separate from the older generic prefill-stream bucket because that
    // bucket also includes the legacy prefill expert stream on some models.
    std::atomic<double> t_ggml_moe_d2d{0};
    std::atomic<double> t_ggml_moe_h2d{0};
    std::atomic<double> t_ggml_moe_h2d_dma{0};
    std::atomic<uint64_t> bytes_ggml_moe_d2d{0};
    std::atomic<uint64_t> bytes_ggml_moe_h2d{0};
    // Source classification for the routed expert records submitted to the
    // GGML MoE runner. These are expert counts, not bytes: the byte counters
    // above are the authoritative transfer-volume measurements, while this
    // split explains why a given H2D volume came from VRAM reuse, pinned DDR,
    // pageable-DDR staging, or the file-backed fallback.
    std::atomic<uint64_t> n_ggml_moe_d2d_experts{0};
    std::atomic<uint64_t> n_ggml_moe_active_d2d_experts{0};
    std::atomic<uint64_t> n_ggml_moe_ram_direct_experts{0};
    std::atomic<uint64_t> n_ggml_moe_ram_staged_experts{0};
    std::atomic<uint64_t> n_ggml_moe_mmap_experts{0};
    std::atomic<double> t_host_alloc{0};    // posix_memalign + setup
    std::atomic<double> t_gpu_upload{0};    // cudaMemcpy H2D -- shared by expert cache
                                             // PROMOTION uploads (GpuTopology::upload_expert,
                                             // whole ~12-15MB slabs) AND dense_proj's own
                                             // one-time startup dense-tensor-offload uploads.
                                             // Deliberately NOT touched by expert_ffn's own
                                             // per-call activation upload anymore -- see
                                             // t_expert_gpu_upload below for why that was split
                                             // out.
    std::atomic<double> t_expert_gpu_upload{0};  // cudaMemcpy H2D of the small activation
                                             // vector before an expert's gate/up/down kernels
                                             // (execution_layer.cpp's expert_ffn_gpu_issue) --
                                             // split OUT of t_gpu_upload after a real
                                             // investigation found "avg GPU round-trip: up=X ms/call" in
                                             // dump_report()/the [REQUEST] line's "gpu=...
                                             // [up=...]" breakdown was computing
                                             // t_gpu_upload/gcalls where t_gpu_upload ALSO
                                             // included cache-PROMOTION slab uploads (12553 of
                                             // them in one real run, ~2ms each) sharing the same
                                             // counter as expert_ffn's own much-smaller per-call
                                             // activation upload (a few KB) -- i.e. the reported
                                             // per-call "up" figure was silently dominated by an
                                             // unrelated bucket, not a genuine measurement of
                                             // expert_ffn's own GPU dispatch overhead. This is
                                             // the SAME class of attribution bug as the earlier
                                             // Attention-vs-Prefill and summed-Throughput fixes
                                             // in this project's history -- fix the counter
                                             // before trusting conclusions drawn from it.
    std::atomic<double> t_cufile_read{0};   // cuFileRead (GpuTopology::upload_expert_cufile) --
                                             // deliberately its OWN counter, not folded into
                                             // t_gpu_upload (the cudaMemcpy H2D one) -- see that
                                             // function's own comment for why conflating two
                                             // genuinely different transfer mechanisms into one
                                             // counter is exactly the mistake this project's own
                                             // disk=/expert= investigation was burned by once
                                             // already.
    std::atomic<double> t_gpu_download{0};  // cudaMemcpy D2H -- was bundled into
                                             // whatever counter happened to wrap the
                                             // next call (usually none, since
                                             // expert_ffn's GPU path had ZERO
                                             // PERF_START/PERF_END until this was
                                             // added); tracked separately from
                                             // t_gpu_upload since expert_ffn does 3
                                             // D2H copies per expert vs 2 H2D.
    std::atomic<double> t_gpu_compute{0};   // Kernel execution time (real device
                                             // time, via an explicit sync point right
                                             // after each kernel launch -- launches
                                             // are async, so timing the launch call
                                             // alone would only capture negligible
                                             // CPU-side dispatch overhead, not actual
                                             // GPU execution time).
    // 2026-08-29: Pass B batching (expert_ffn_gpu_finish split into stage1/
    // stage2, dispatch_expert_k's own caller loop) -- real nsys profiling
    // of a whole request found cudaStreamSynchronize alone at 99,791 calls/
    // 21.5s/26.4% of all CUDA API time, with "0% stream busy" already
    // independently confirmed 4 separate times (this file's own history),
    // meaning each sync call was mostly waiting on NOTHING -- pure fixed
    // per-call overhead paid once per expert (up to 2x/expert: one before
    // gate/up download, one after down-proj) instead of once per BATCH of
    // pending experts sharing the same vault GPU. This counter measures
    // ONLY the new shared, once-per-distinct-GPU batched sync calls --
    // t_gpu_compute above now measures per-expert kernel ISSUE time only
    // (no sync inside it anymore), so the two together still reconcile to
    // the same real GPU-side work, just attributed differently: real
    // compute+wait time moved from N per-expert entries into this new,
    // far-fewer-calls bucket.
    std::atomic<double> t_gpu_batch_sync{0};
    std::atomic<uint64_t> n_gpu_batch_sync_calls{0};
    std::atomic<double> t_expert_silu_activation{0};  // expert_ffn_gpu_finish's CPU-side
                                             // silu_inplace/elem_mul/swiglu_clamp_inplace
                                             // on the downloaded gate_buf/up_buf, between
                                             // the gate/up download and the down_proj
                                             // re-upload -- previously had ZERO timer of
                                             // its own (the surrounding comment called it
                                             // "cheap", but that was never actually
                                             // measured). Added as the next concrete
                                             // hypothesis after three prior rounds of
                                             // instrumenting GpuTopology/RAMSafetyManager
                                             // functions (t_hold_for_compute/t_release_
                                             // hold/t_find_active_expert, then t_find_
                                             // expert/t_evict_lru/t_upload_expert_async/
                                             // t_ram_get_expert/t_ram_get_experts_batch)
                                             // all came back near-zero without closing the
                                             // real gap between expert= and every prior
                                             // sub-timer -- this is a genuinely different
                                             // function (execution_layer.cpp's own GPU
                                             // dispatch path, not the cache/eviction layer)
                                             // that none of those rounds touched.
    std::atomic<double> t_cpu_expert{0};    // expert_ffn's CPU dequant-matmul
                                             // fallback path (cpu_quant_matmul x3),
                                             // for direct comparison against the GPU
                                             // path's total (upload+compute+download).
    std::atomic<double> t_cpu_dequant{0};   // Time spent in dequant_row (K-quant/
                                             // IQ bit-unpacking) inside
                                             // cpu_quant_matmul, separate from the
                                             // dot product -- answers whether the
                                             // CPU expert cost is dominated by
                                             // unpacking ggml's reference
                                             // (unvectorized) dequant or by the
                                             // actual matmul, since only the dot
                                             // product was AVX-512 vectorized so
                                             // far. NOTE: summed across every OMP
                                             // worker thread (aggregate CPU-core-
                                             // seconds), NOT wall-clock -- with N
                                             // threads this can read up to ~N x
                                             // larger than the wall-clock portion
                                             // of t_cpu_expert it's part of. Only
                                             // the RATIO of this to t_cpu_dot is
                                             // meaningful, not the absolute value
                                             // against t_cpu_expert.
    std::atomic<double> t_cpu_dot{0};       // Same caveat as t_cpu_dequant: time
                                             // in the post-dequant AVX-512 dot
                                             // product (dot_f32_avx512), summed
                                             // across OMP worker threads.
    // Opt-in wall-clock stages for Qwen4Exp's direct DDR-strided grouped
    // CPU path.  These are deliberately separate from t_cpu_dequant/t_cpu_dot:
    // the grouped GGML kernels do not expose their internal per-worker split,
    // and the useful question here is which whole stage delays the decode
    // thread (gate/up, SwiGLU, down, or weighted combine).  The counters stay
    // zero unless ZEROMOE_QWEN_CPU_STAGE_TIMING=1 is set, so the normal
    // path pays no extra clock reads.
    std::atomic<double> t_qwen_cpu_grouped_gate_up{0};
    std::atomic<double> t_qwen_cpu_grouped_swiglu{0};
    std::atomic<double> t_qwen_cpu_grouped_down{0};
    std::atomic<double> t_qwen_cpu_grouped_combine{0};
    std::atomic<double> t_attn{0};          // MLA attention forward (total)
    // Untimed setup before the first named sub-phase (arch-config reads,
    // rope_theta lookup via get_config_float, string prefix build) -- added
    // after the six phases below plus t_attn itself failed to fully
    // reconcile (sub-phases summed to only ~35-40% of t_attn across
    // multiple real runs, even with WUK/WUV's one-time cost -- see absorb's
    // own comment -- confirmed via a same-process two-request test to be
    // warm on the second call). Static review of every line in this
    // preamble suggested it should be microseconds, not the tens of
    // seconds needed to explain the gap -- this timer stayed near-zero on
    // every real run, correctly ruling this out as the explanation (see
    // t_prefill_total below for the real one).
    // Qwen4Exp hybrid attention split (2026-09-04). qwen4exp_attention_gguf
    // dispatches per layer to EITHER a recurrent GDN layer or a dense
    // full-attention layer, and NONE of the MLA sub-timers below fire for
    // either -- so a real 3,189-token prefill reported attn=179.00s with
    // every sub-timer at 0.00, i.e. 53% of prefill entirely unattributed.
    // These two split that: 36 GDN layers vs 12 dense-attention layers on
    // this checkpoint. The distinction matters because the two have
    // completely different fixes -- GDN is a sequential O(S) recurrence
    // (wants a chunked/parallel formulation), dense attention is O(S^2)
    // (wants AVX-512/GPU/flash-attention). Zero for every other family.
    std::atomic<double> t_attn_qwen_gdn{0};
    std::atomic<double> t_attn_qwen_dense{0};
    std::atomic<double> t_attn_preamble{0};
    // Per-phase breakdown of t_attn, added to root-cause an intermittent
    // severe stall (~90s in one token) that showed the CPU idle/cool at the
    // time (ruled out: thermal throttling, memory pressure, other
    // processes, disk I/O) -- meaning something inside attention is
    // BLOCKING without burning CPU, most likely a GPU call. attn_q_a/q_b,
    // attn_kv_a_mqa, and attn_output all go through dense_proj, which can
    // dispatch to GPU if the weight is VRAM-resident -- a blocking
    // cudaMemcpy/cudaStreamSynchronize stalling there (e.g. contention with
    // the async expert-promotion traffic on the same or a P2P-linked GPU)
    // would show up as elevated wall-clock time here with the CPU idle,
    // exactly matching what was observed.
    std::atomic<double> t_attn_norm{0};     // attn_norm rmsnorm
    std::atomic<double> t_attn_qproj{0};    // attn_q_a -> q_a_norm -> attn_q_b + RoPE
    std::atomic<double> t_attn_kvproj{0};   // attn_kv_a_mqa + norm/RoPE + Lc/Rc cache write
    std::atomic<double> t_attn_absorb{0};   // attn_k_b/attn_v_b cache lookup-or-fill (should be ~0 after layer's first call)
    // warm_mla_absorb_cache's own dedicated timer (attention_layer.cpp,
    // 2026-08-12) -- DELIBERATELY separate from t_attn_absorb above: the
    // whole point is moving this one-time, model-only dequant cost OUT of
    // the first prefill call's own timing window, so it must NOT feed the
    // same counter prefill's [PREFILL] breakdown reads from. Not part of
    // Snapshot/operator-/snapshot() (same convention as t_dense_load) --
    // this is a startup-only, once-per-process cost, not something a
    // per-request delta needs to report.
    std::atomic<double> t_mla_absorb_warmup{0};
    std::atomic<double> t_attn_score{0};    // the absorbed score/softmax/value-aggregation loop
    // Batched q_absorbed GEMM (attention_layer.cpp, 2026-08-12) -- ONE real
    // matmul per head covering all S tokens at once, moved OUT of the
    // per-(s,h) loop t_attn_score measures (was previously invisible,
    // folded into attn_score's own total as S separate per-token GEMVs).
    // Given its own counter specifically so a real run can see this
    // sub-step's cost split out from the causal score/softmax/accumulate
    // loop that still follows it. Part of Snapshot (unlike t_mla_absorb_
    // warmup above) -- this is a real per-request cost, not a one-time
    // startup cost.
    std::atomic<double> t_attn_qabsorb_batch{0};
    std::atomic<double> t_attn_outproj_dequant{0};  // dsv4_attention_gguf's
                                             // one-time-per-layer WOA
                                             // (attn_output_a) dequant_row
                                             // call, gated by WOA.empty() --
                                             // split out from t_attn_outproj
                                             // (which now measures only the
                                             // recurring per-call work) to
                                             // distinguish a one-time-43x
                                             // cost from a per-call-1720x
                                             // one after outproj came back
                                             // at 11.96s/14.67s (82% of all
                                             // attention) on a real run --
                                             // see t_attn_outproj's own
                                             // comment for the leading
                                             // hypothesis this is meant to
                                             // confirm/refute.
    std::atomic<double> t_attn_outproj{0};  // attn_output dense_proj
                                             // PLUS dsv4_attention_gguf's
                                             // recurring grouped matmul()
                                             // loop (attn_output_a per o_group
                                             // -- see t_attn_outproj_dequant's
                                             // comment). Leading hypothesis
                                             // for why outproj dominates: that
                                             // loop calls the AVX-512+OMP
                                             // matmul() kernel o_groups times
                                             // PER TOKEN PER LAYER, every one
                                             // with S=1 (a single decode
                                             // token) -- spinning up a full
                                             // OpenMP thread team for a
                                             // single tiny matmul, up to
                                             // ~1720 times across a 40-token
                                             // run, is a classic overhead-
                                             // dominates-the-work anti-
                                             // pattern. This split isolates
                                             // that loop from the one-time
                                             // dequant to test that
                                             // hypothesis directly.

    // Phase split for CudaManager::dense_matmul_host -- ALL dense_proj GPU
    // calls share these (not just attention's), added after confirming
    // attn_output's dense_proj call *does* successfully GPU-dispatch every
    // time (zero "[DENSE] ... CPU dequant-matmul fallback" lines logged)
    // yet still costs ~623ms/token, dwarfing the qproj dense_proj calls
    // (~120ms combined for two calls) -- this splits WHERE inside that
    // successful GPU call the time actually goes, same method that found
    // the real issue in the expert-FFN GPU path earlier this session.
    std::atomic<double> t_dense_gpu_up{0};
    std::atomic<double> t_dense_gpu_compute{0};
    std::atomic<double> t_dense_gpu_down{0};
    std::atomic<uint64_t> n_dense_gpu_calls{0};

    // MLA qabsorb BF16 batched fast-path (EngineConfig::attn_qabsorb_gpu,
    // CudaManager::qabsorb_bf16_batched_host, cuda_manager.hpp).
    // Shutdown-report-only, NOT threaded through the live per-request
    // Snapshot (matching the precedent set by upload_expert_active's own
    // separate counters, and t_active_layer_upload's -- a structurally
    // distinct call site's instrumentation shouldn't share buckets with
    // dense_proj's t_dense_gpu_* above, which would hide whether qabsorb is
    // engaging at all behind that already-large, unrelated total).
    std::atomic<double> t_qabsorb_gpu_up{0};
    std::atomic<double> t_qabsorb_gpu_compute{0};
    std::atomic<double> t_qabsorb_gpu_down{0};
    std::atomic<uint64_t> n_qabsorb_gpu_calls{0};
    // Incremented by the CALLER (attention_layer.cpp) whenever
    // qabsorb_bf16_batched_host returns false and it falls back to the CPU
    // per-head matmul() loop for that layer/call -- e.g. attn_qabsorb_gpu is
    // on but this layer's WUK BF16 upload never succeeded (VRAM budget,
    // missing weights). Should track close to 0 once warm_mla_absorb_cache
    // has finished uploading every eligible layer; a steady nonzero rate
    // across a whole run means some layer(s) never got a working GPU
    // fast-path and are silently paying the CPU cost every call.
    std::atomic<uint64_t> n_qabsorb_gpu_fallback_cpu{0};

    // MLA score GPU fast-path (EngineConfig::attn_score_gpu) -- same
    // shutdown-report-only discipline as the qabsorb counters above.
    std::atomic<double> t_score_gpu_up{0};
    std::atomic<double> t_score_gpu_compute{0};
    std::atomic<double> t_score_gpu_down{0};
    std::atomic<uint64_t> n_score_gpu_calls{0};

    // qwen4exp dense full-attention GPU path (EngineConfig::qwen4exp_attn_gpu).
    // Kept separate from t_score_gpu_* above: that is GLM's MLA score phase, a
    // structurally different call site, and sharing buckets would hide which
    // one dominates -- the same discipline upload_expert_active's own counters
    // follow. t_attn_qwen_dense remains the WHOLE dense-layer cost (projections
    // + norms + RoPE + attention + gate + wo), so these three are a strict
    // subset of it and the difference is what stays on the CPU.
    std::atomic<double> t_qwen4exp_attn_gpu_up{0};
    std::atomic<double> t_qwen4exp_attn_gpu_cmp{0};
    std::atomic<double> t_qwen4exp_attn_gpu_dn{0};
    std::atomic<uint64_t> n_qwen4exp_attn_gpu_calls{0};
    std::atomic<uint64_t> n_qwen4exp_attn_gpu_fallback_cpu{0};

    // qwen4exp chunkwise GDN GPU path (EngineConfig::qwen4exp_gdn_gpu).
    std::atomic<double> t_qwen4exp_gdn_gpu_up{0};
    std::atomic<double> t_qwen4exp_gdn_gpu_cmp{0};
    std::atomic<double> t_qwen4exp_gdn_gpu_dn{0};
    std::atomic<uint64_t> n_qwen4exp_gdn_gpu_calls{0};
    std::atomic<uint64_t> n_qwen4exp_gdn_gpu_fallback_cpu{0};

    // GDN causal conv1d+SiLU GPU path (EngineConfig::qwen4exp_gdn_conv_gpu).
    std::atomic<double> t_gdn_conv1d_gpu_up{0};
    std::atomic<double> t_gdn_conv1d_gpu_cmp{0};
    std::atomic<double> t_gdn_conv1d_gpu_dn{0};
    std::atomic<uint64_t> n_gdn_conv1d_gpu_calls{0};
    std::atomic<uint64_t> n_gdn_conv1d_gpu_fallback_cpu{0};

    // GLM5NEXT KDA recurrent delta-rule GPU path. This is separate from the
    // Qwen4Exp GDN counters because the recurrence and its state layout are
    // different (GLM decay is per key channel).
    std::atomic<double> t_glm5next_kda_gpu_up{0};
    std::atomic<double> t_glm5next_kda_gpu_cmp{0};
    std::atomic<double> t_glm5next_kda_gpu_dn{0};
    std::atomic<uint64_t> n_glm5next_kda_gpu_calls{0};
    std::atomic<uint64_t> n_glm5next_kda_gpu_fallback_cpu{0};

    // Prompt (prefix) state cache (EngineConfig::prompt_cache).
    std::atomic<uint64_t> n_prompt_cache_hits{0};
    std::atomic<uint64_t> n_prompt_cache_misses{0};
    std::atomic<uint64_t> n_prompt_cache_tokens_reused{0};

    // GPU-side routed-expert accumulation (EngineConfig::moe_gpu_accum).
    // t_moe_accum_dn is the ONE [S,D] per-layer download that replaces the
    // per-expert downloads previously counted in t_gpu_download.
    std::atomic<double> t_moe_accum_dn{0};
    std::atomic<uint64_t> n_moe_gpu_accum_layers{0};
    std::atomic<uint64_t> n_moe_gpu_accum_fallback{0};

    // Memory guard: host allocations refused because MemAvailable was below
    // the stop threshold. Non-zero means the run was memory-constrained and
    // the RAM tier re-read experts instead of growing -- expect a lower cache
    // hit rate, but a finished run rather than a thrashed one.
    std::atomic<uint64_t> n_mem_guard_alloc_refused{0};
    // Dense BF16 GEMM skipped because the weight's BF16 copy would not fit in
    // VRAM. Non-zero is EXPECTED on a VRAM-tight model (GLM): those tensors
    // just use the fused kernel. Large values mean the BF16 path is mostly
    // inactive and its win will not materialise on this config.
    std::atomic<uint64_t> n_dense_bf16_skipped_vram{0};
    // Dynamic VRAM grows refused by the watermark (EngineConfig::vram_max_fraction).
    std::atomic<uint64_t> n_vram_watermark_refused{0};

    // Routed-expert SwiGLU computed on the DEVICE (expert_gpu_activation),
    // replacing the download / CPU silu+mul+clamp / upload round trip.
    std::atomic<double> t_expert_gpu_activation{0};
};
