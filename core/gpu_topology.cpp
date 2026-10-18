// gpu_topology.cpp - Active layer area implementation for three-tier VRAM caching
#include "gpu_topology.hpp"
#include "perf_timer.hpp"
#include "logger.hpp"
#include "dense_dispatch.hpp"  // speculative_active_layer_promote_on_confirm_enabled()
#include <cuda_runtime.h>
#include <algorithm>

bool GpuTopology::allocate_active_layer_area(size_t expert_slab_bytes, int max_experts_per_layer) {
    if (expert_slab_bytes == 0 || max_experts_per_layer <= 0) {
        LOG_STREAM_ERROR() << "[GPU] Invalid active layer area parameters: slab_bytes="
                           << expert_slab_bytes << " max_experts=" << max_experts_per_layer << "\n";
        return false;
    }

    // Allocate on primary vault GPU (or core GPU if no vault)
    int target_gpu = vault_gpus.empty() ? core_gpu : vault_gpus[0];
    if (target_gpu < 0 || target_gpu >= (int)devices.size()) {
        LOG_STREAM_ERROR() << "[GPU] Invalid target GPU for active layer area: " << target_gpu << "\n";
        return false;
    }

    auto& dev = devices[target_gpu];
    cudaSetDevice(target_gpu);

    size_t total_bytes = expert_slab_bytes * max_experts_per_layer;
    cudaError_t err = cudaMalloc((void**)&active_layer_pool, total_bytes);
    if (err != cudaSuccess) {
        LOG_STREAM_ERROR() << "[GPU] Failed to allocate active layer area: "
                           << cudaGetErrorString(err) << " (requested "
                           << total_bytes / 1e9 << " GB)\n";
        return false;
    }

    active_layer_capacity = max_experts_per_layer;

    // Pre-allocate slot metadata
    active_layer_slots.clear();
    active_layer_slots.reserve(max_experts_per_layer);
    for (int i = 0; i < max_experts_per_layer; ++i) {
        ExpertLocation slot;
        slot.gpu_id = target_gpu;
        slot.device_ptr = active_layer_pool + (i * expert_slab_bytes);
        slot.size = expert_slab_bytes;
        slot.offset = i * expert_slab_bytes;
        slot.layer_idx = -1;
        slot.expert_id = -1;
        slot.last_used_clock = 0;
        slot.hit_count = 0;
        slot.upload_ready = true;
        slot.pinned = false;
        slot.temp_pin_count = 0;
        // 2026-08-26: real per-slot completion event, mirroring
        // allocate_expert_cache's own main-cache init exactly -- see
        // find_active_expert's own new poll-before-trust branch for why
        // this is needed. Previously always nullptr: the active-layer
        // subsystem had ZERO completion-tracking infrastructure of its
        // own, relying entirely on callers blocking via sync_active_layer_
        // batch (a real, unconditional per-transition wait cost) instead
        // of the main cache's already-proven cudaEventQuery poll pattern.
        cudaSetDevice(target_gpu);
        cudaEventCreate(&slot.upload_event);
        cudaEventCreate(&slot.upload_start_event);
        slot.prewarm_skip_next_promote = false;
        active_layer_slots.push_back(slot);
    }

    LOG_STREAM_INFO() << "[GPU] Active layer area allocated: "
                      << max_experts_per_layer << " experts ("
                      << total_bytes / 1e9 << " GB) on GPU " << target_gpu << "\n";
    return true;
}

// 2026-08-28: multi-GPU active-layer pool allocation -- see GpuTopology::
// ActiveLayerGpuSlice's own comment (gpu_topology.hpp) for the design.
// Real motivation: the active-layer pool carries the dominant share of
// DDR->VRAM transfer volume (confirmed via real per-GPU byte counters,
// ~90%+ of total H2D upload bytes), and it was hard-coded to a single GPU
// whose own PCIe link is the measured bottleneck (~13.5-14.1 GB/s
// achieved, consistent with its real PCIe Gen3 x16 ceiling) while a
// second GPU sat almost entirely idle for this traffic (99.5%/0.5% split,
// measured). Since expert-FFN compute dispatch is already fully GPU-
// agnostic per-expert (activation/output round-trip through HOST memory
// regardless of which GPU holds the weights -- confirmed by reading
// expert_ffn_gpu_issue/_finish, no cross-GPU P2P copy needed), giving a
// second GPU its own small pool adds a genuinely separate, non-contending
// PCIe link's worth of bandwidth, not just relocated cost.
bool GpuTopology::allocate_active_layer_area_multi(size_t expert_slab_bytes, int legacy_max_experts_per_layer,
                                                     const std::vector<ActiveLayerGpuRequest>& requests) {
    // Count how many requests genuinely opt in (weight > 0, valid gpu_id) --
    // fewer than 2 means multi-GPU routing has nothing to route between,
    // so fall through to the plain single-GPU path UNCHANGED. This is what
    // makes the whole feature strictly opt-in: a config with 0 or 1
    // active_layer_weight entries behaves byte-identically to before this
    // change existed.
    std::vector<ActiveLayerGpuRequest> qualifying;
    for (auto& r : requests) {
        if (r.weight > 0 && r.gpu_id >= 0 && r.gpu_id < (int)devices.size()) qualifying.push_back(r);
    }
    if (qualifying.size() < 2) {
        return allocate_active_layer_area(expert_slab_bytes, legacy_max_experts_per_layer);
    }
    if (expert_slab_bytes == 0) {
        LOG_STREAM_ERROR() << "[GPU] Invalid multi-GPU active layer area parameters: slab_bytes=0\n";
        return false;
    }

    active_layer_slots.clear();
    active_layer_gpu_slices.clear();
    size_t running_index = 0;
    bool any_alloc_ok = false;
    for (auto& r : qualifying) {
        int capacity = r.capacity_slots > 0 ? r.capacity_slots : legacy_max_experts_per_layer;
        if (capacity <= 0) {
            LOG_STREAM_WARN() << "[GPU] Multi-GPU active layer: GPU " << r.gpu_id
                               << " has no valid capacity (active_layer_capacity_slots="
                               << r.capacity_slots << ") -- skipped\n";
            continue;
        }
        cudaSetDevice(r.gpu_id);
        size_t total_bytes = expert_slab_bytes * (size_t)capacity;
        uint8_t* pool_base = nullptr;
        cudaError_t err = cudaMalloc((void**)&pool_base, total_bytes);
        if (err != cudaSuccess) {
            LOG_STREAM_ERROR() << "[GPU] Multi-GPU active layer: failed to allocate on GPU " << r.gpu_id
                               << ": " << cudaGetErrorString(err) << " (requested "
                               << total_bytes / 1e9 << " GB) -- this GPU will not participate\n";
            continue;
        }
        ActiveLayerGpuSlice slice;
        slice.gpu_id = r.gpu_id;
        slice.weight = r.weight;
        slice.slot_start = running_index;
        slice.slot_count = (size_t)capacity;
        slice.pool_base = pool_base;
        active_layer_gpu_slices.push_back(slice);

        for (int i = 0; i < capacity; ++i) {
            ExpertLocation slot;
            slot.gpu_id = r.gpu_id;
            slot.device_ptr = pool_base + ((size_t)i * expert_slab_bytes);
            slot.size = expert_slab_bytes;
            slot.offset = (size_t)i * expert_slab_bytes;
            slot.layer_idx = -1;
            slot.expert_id = -1;
            slot.last_used_clock = 0;
            slot.hit_count = 0;
            slot.upload_ready = true;
            slot.pinned = false;
            slot.temp_pin_count = 0;
            cudaEventCreate(&slot.upload_event);
            cudaEventCreate(&slot.upload_start_event);
            slot.prewarm_skip_next_promote = false;
            active_layer_slots.push_back(slot);
        }
        running_index += (size_t)capacity;
        any_alloc_ok = true;
        LOG_STREAM_INFO() << "[GPU] Multi-GPU active layer: GPU " << r.gpu_id << " weight=" << r.weight
                          << " allocated " << capacity << " experts (" << total_bytes / 1e9 << " GB)\n";
    }

    if (!any_alloc_ok) {
        LOG_STREAM_ERROR() << "[GPU] Multi-GPU active layer: no GPU allocation succeeded -- "
                               "falling back to legacy single-GPU path\n";
        active_layer_slots.clear();
        active_layer_gpu_slices.clear();
        return allocate_active_layer_area(expert_slab_bytes, legacy_max_experts_per_layer);
    }
    // Keep active_layer_pool/active_layer_capacity in sync with the FIRST
    // participating GPU's own allocation -- legacy-compat only (nothing in
    // the multi-GPU-aware code paths reads these two fields anymore, but
    // kept populated in case anything not yet updated still does).
    active_layer_pool = active_layer_gpu_slices[0].pool_base;
    active_layer_capacity = active_layer_slots.size();
    LOG_STREAM_INFO() << "[GPU] Multi-GPU active layer: " << active_layer_gpu_slices.size()
                      << " GPU(s) participating, " << active_layer_slots.size() << " total slots\n";
    return true;
}

ExpertLocation* GpuTopology::upload_expert_active(int layer, int expert_id,
                                                 const uint8_t* host_data, size_t bytes,
                                                 bool speculative, int target_gpu_id) {
    if (active_layer_pool == nullptr || active_layer_capacity == 0) {
        return nullptr;  // Active layer area not allocated
    }

    if (!host_data || bytes == 0) {
        LOG_STREAM_ERROR() << "[GPU] Invalid upload_expert_active parameters\n";
        return nullptr;
    }

    // 2026-08-28: FIX for a real, confirmed double-transfer + priority-
    // inversion bug (see n_active_layer_sync_duplicate_transfer's own
    // comment, perf_timer.hpp). This function is reached specifically when
    // find_active_expert's poll-before-trust check found this exact
    // identity already resident but its async DMA still in flight, and
    // correctly returned a miss rather than blocking THERE -- but this
    // function used to react to that "miss" by picking a totally different
    // victim slot and re-copying the SAME bytes a second time (wasting real
    // DDR->VRAM bandwidth) while ALSO, via the legacy default stream's
    // implicit whole-device-drain semantics, forcing this genuinely urgent
    // transfer to wait for active_layer_stream to empty anyway before its
    // own (redundant) copy could even start. Confirmed on real hardware:
    // 494/494 (100%) of these fallback calls found active_layer_stream
    // busy at that moment. Fixed by checking for this exact identity FIRST,
    // before any victim-selection: if found already landed, return it
    // directly (no wait); if found still in flight, wait on THAT transfer's
    // OWN event (a bounded, targeted cudaEventSynchronize) instead of
    // starting a second one -- mirrors the RAM tier's own claim_or_wait
    // "WaitOnOther" pattern (ram_manager.hpp), applied here to the active-
    // layer VRAM tier's sync fallback path specifically.
    //
    // Two-phase, same discipline sync_active_layer_batch's own design
    // already established: capture what's needed under a short lock, THEN
    // release it before the blocking cudaEventSynchronize (never hold
    // cache_mutex across a CUDA wait), then re-acquire a fresh lock to
    // finalize state. This function is decode-thread-only (its one call
    // site is dispatch_expert_k's own resolve path, execution_layer.cpp --
    // grepped, confirmed) and the captured slot's temp_pin_count stays >0
    // throughout (excluded from every OTHER victim-selection loop, this
    // one's own and upload_expert_active_async's), so the identity cannot
    // legitimately change between release and re-acquire -- the re-check
    // below is defense-in-depth, not a load-bearing assumption.
    ExpertLocation* inflight_slot = nullptr;
    cudaEvent_t inflight_event = nullptr;
    int inflight_gpu = -1;
    {
        GPU_TOPO_TIMED_LOCK(precheck_lock);
        for (auto& slot : active_layer_slots) {
            if (slot.layer_idx != layer || slot.expert_id != expert_id) continue;
            if (slot.upload_ready) {
                // Landed in the narrow window between find_active_expert's
                // poll and this call -- cheap fast path, no wait needed.
                slot.last_used_clock = ++cache_clock;
                slot.prewarm_touched = true;
                perf().n_active_layer_wait_on_inflight_ready.fetch_add(1, std::memory_order_relaxed);
                return &slot;
            }
            // 2026-08-28 REAL-HARDWARE BUG FOUND AND FIXED: this pre-check
            // predates ExpertLocation::copy_dispatched (added for
            // ActiveLayerUploadWorkers' claim_active_layer_slot_for_worker,
            // which claims a slot -- setting layer_idx/expert_id and
            // temp_pin_count -- BEFORE any cudaMemcpyAsync/cudaEventRecord
            // has actually run). Without this check, an identity match here
            // whose worker hasn't dispatched yet still has slot.upload_event
            // non-null (the event OBJECT always exists, pre-allocated at
            // pool creation) but pointing at whatever the SLOT'S PREVIOUS
            // occupant last recorded (or nothing, if never used) -- NOT a
            // real, pending recording for THIS claim. cudaEventSynchronize
            // on that stale/unrecorded event returns IMMEDIATELY (nothing
            // to wait for), spuriously marking upload_ready=true and
            // handing the caller a slot whose device_ptr still holds the
            // PREVIOUS occupant's stale bytes -- garbage weights feeding
            // the gate/up matmul. CONFIRMED as the real cause of a severe,
            // real-hardware NaN cascade (gate_buf/up_buf/output NaN
            // starting at an early layer, eventually poisoning router
            // weights) that persisted even after fixing a DIFFERENT,
            // genuine-but-not-causal hazard (pending_host_unregisters
            // starvation, see ActiveLayerUploadWorkers' own header comment)
            // -- isolated via bisection: the corruption reproduced with
            // ONLY decode_async_prewarm's worker routing active (which
            // reaches this exact fallback on a real miss, since dispatch_
            // expert_k runs for the SAME candidates immediately after
            // submitting them to the worker pool) and was absent with only
            // the speculative-promotion worker path active (which is never
            // followed by an immediate same-token dispatch, so this
            // precheck's pre-existing bug was never exercised by it).
            if (!slot.copy_dispatched) break;
            if (slot.upload_event) {
                inflight_slot = &slot;
                inflight_event = slot.upload_event;
                inflight_gpu = slot.gpu_id;
            }
            break;
        }
    }  // precheck_lock released here

    if (inflight_slot && inflight_event) {
        perf().n_active_layer_wait_on_inflight.fetch_add(1, std::memory_order_relaxed);
        PERF_START(active_layer_wait_on_inflight);
        if (inflight_gpu >= 0 && (size_t)inflight_gpu < devices.size()) cudaSetDevice(inflight_gpu);
        cudaEventSynchronize(inflight_event);
        PERF_END(active_layer_wait_on_inflight, t_active_layer_wait_on_inflight);

        GPU_TOPO_TIMED_LOCK(postwait_lock);
        if (inflight_slot->layer_idx == layer && inflight_slot->expert_id == expert_id) {
            inflight_slot->upload_ready = true;
            if (inflight_slot->temp_pin_count > 0) inflight_slot->temp_pin_count--;
            inflight_slot->last_used_clock = ++cache_clock;
            inflight_slot->prewarm_touched = true;
            return inflight_slot;
        }
        // Identity changed underneath us -- believed unreachable per the
        // analysis above, but if it ever happens, fall through to the
        // normal victim-selection path below as a safe fallback (matches
        // the pre-fix behavior exactly, just for this one rare case).
    }

    GPU_TOPO_TIMED_LOCK(lock);

    // Find empty slot or lowest-eviction_score victim within active layer
    // area -- eviction_score (recency + hit_count, same blend the main
    // cache uses) rather than plain oldest-clock, so a slot that's already
    // taken one hit (but not yet crossed active_layer_promote_threshold)
    // isn't evicted ahead of a colder one just for being slightly older.
    // temp_pin_count > 0 slots are SKIPPED ENTIRELY (never selected), same
    // discipline evict_lru already uses for the main cache -- a real
    // in-flight GPU kernel may still be reading this slot's device_ptr
    // (hold_for_compute'd by expert_ffn_resolve_gpu_loc, not yet released
    // by expert_ffn_gpu_finish's cudaStreamSynchronize); overwriting it via
    // the cudaMemcpy below would be the exact "evicted mid-compute"
    // corruption this project has already been burned by once for the main
    // cache (see CLAUDE.md) -- this pool had no such guard until now.
    ExpertLocation* victim = nullptr;

    for (auto& slot : active_layer_slots) {
        // 2026-08-28: multi-GPU routing restriction -- when target_gpu_id
        // is set (a router decided this candidate belongs on a SPECIFIC
        // GPU, not "wherever's free"), only consider slots on that GPU.
        // -1 (legacy default, and every single-GPU-mode call) scans
        // everything, unchanged.
        if (target_gpu_id >= 0 && slot.gpu_id != target_gpu_id) continue;
        if (slot.temp_pin_count > 0) continue;
        if (slot.layer_idx == -1) {
            victim = &slot;  // Prefer empty slot
            break;
        }
        // 2026-08-26: an unconfirmed speculative fill always outranks a
        // real/confirmed slot as a victim, regardless of recency -- see
        // ExpertLocation::preferred_victim_over's own comment.
        if (!victim || slot.preferred_victim_over(*victim)) {
            victim = &slot;
        }
    }

    if (!victim) {
        LOG_STREAM_WARN() << "[GPU] Active layer area full - cannot upload expert"
                          << (target_gpu_id >= 0 ? (" on GPU " + std::to_string(target_gpu_id)) : std::string())
                          << "\n";
        return nullptr;
    }

    // 2026-08-28: duplicate-transfer detection (instrumentation only, see
    // n_active_layer_sync_duplicate_transfer's own comment) -- runs BEFORE
    // the victim's identity is overwritten below, so this genuinely checks
    // "does some OTHER slot already hold this identity", not the victim
    // itself. `victim` was chosen specifically excluding temp_pin_count>0
    // slots, so an in-flight async issue can never BE the victim -- this
    // scan is what would have caught it as a pre-existing duplicate instead.
    for (const auto& slot : active_layer_slots) {
        if (&slot == victim) continue;
        if (slot.layer_idx == layer && slot.expert_id == expert_id) {
            perf().n_active_layer_sync_duplicate_transfer.fetch_add(1, std::memory_order_relaxed);
            perf().bytes_active_layer_sync_duplicate_transfer.fetch_add((int64_t)std::min(bytes, victim->size),
                                                                         std::memory_order_relaxed);
            break;
        }
    }

    // Synchronous upload (decode-thread-only, no async needed). Previously
    // completely untimed -- see t_active_layer_upload's own comment in
    // perf_timer.hpp for why this specific call was the leading suspect for
    // the expert=/gpu=+cpu= gap. This runs while cache_mutex (held by the
    // GPU_TOPO_TIMED_LOCK above) is locked, so a slow copy here also directly
    // extends every OTHER thread's t_cache_mutex_wait, not just this one.
    //
    // cudaHostRegister/cudaHostUnregister was TRIED here (mirroring
    // upload_expert's own Alt-A conversion, see CLAUDE.md) and REVERTED
    // after a real-hardware A/B showed it's a net regression for THIS call
    // site specifically, not the win it was for upload_expert. Measured on
    // real GLM-5.2 hardware: register+unregister overhead alone
    // (t_active_layer_host_register) was 17.61s for 16758 calls (~1.05ms/
    // call fixed OS/driver cost -- page-locking, TLB invalidation), while
    // the DMA itself (t_active_layer_upload) dropped from the old plain-
    // memcpy's 27.31s to 17.06s as pinning predicted -- but 17.61+17.06 =
    // 34.67s combined is WORSE than the original single unpinned memcpy's
    // 27.31s (1.502ms/call old vs 2.069ms/call new, normalized -- a 38%
    // regression, not an improvement). Root cause of the discrepancy with
    // upload_expert's validated win: THAT investigation compared
    // register+DMA against a bounce-buffer memcpy baseline (CPU-side stage
    // into pinned memory, then DMA) -- a slower baseline than this
    // function ever had. upload_expert_active's baseline was always a raw
    // pageable cudaMemcpy of an ~11.6MB slab, already reasonably efficient
    // at that size, and at this call site's much higher call volume
    // (~16700/decode here vs upload_expert's 1477 promotions the same run)
    // the register/unregister fixed-per-call tax outweighs what pinning
    // saves on the DMA side.
    //
    // The dedicated-copy_stream conversion (cudaMemcpyAsync+
    // cudaStreamSynchronize instead of plain cudaMemcpy) was ALSO reverted,
    // in a later pass -- it had been kept on the assumption it was harmless
    // ("0% cross-stream contention"), but that finding came from a
    // DIFFERENT, much-lower-frequency call site (demote_from_vram) and was
    // never actually re-measured at THIS call site's own volume
    // (15000-17000+ calls/run, GLM's dominant dispatch path) -- exactly the
    // kind of carried-over assumption this project's own house style warns
    // against. Reverted to a plain synchronous cudaMemcpy on the implicit
    // default stream, byte-for-byte matching this function's original,
    // pre-this-session behavior, while chasing a real GLM throughput gap
    // against a known-good week-old reference build that still used the
    // plain form here.
    if (bytes > victim->size) {
        LOG_STREAM_ERROR() << "[DIAG-CLAMP] active-layer upload TRUNCATED: layer=" << layer
                           << " expert=" << expert_id << " requested_bytes=" << bytes
                           << " slot_capacity=" << victim->size
                           << " (prev occupant layer=" << victim->layer_idx
                           << " expert=" << victim->expert_id << ")\n";
    }
    size_t copy_bytes = std::min(bytes, victim->size);
    cudaSetDevice(victim->gpu_id);
    // A completed nonblocking prewarm can be selected as a victim before any
    // later lookup consumes it. Consume its old event pair before reusing the
    // persistent events, otherwise the next upload overwrites the timing and
    // the CUDA-event report undercounts real DMA time.
    if (victim->upload_timing_pending && victim->upload_event &&
        cudaEventQuery(victim->upload_event) == cudaSuccess) {
        record_active_layer_upload_event_timing(*victim, false);
    }
    cudaGetLastError();
    // Diagnostic ONLY, mirrors demote_from_vram's own cudaStreamQuery check
    // (ram_manager.hpp) exactly, at THIS call site instead -- see
    // n_active_upload_saw_*'s own comment in perf_timer.hpp for why this
    // call site's own re-check matters despite the general "0% cross-
    // stream contention" finding elsewhere: this fires during prefill's
    // own hint-serving burst, genuinely different conditions from where
    // that finding was measured. Non-blocking; cudaSuccess = idle,
    // cudaErrorNotReady = real in-flight work on that stream right now.
    if (victim->gpu_id >= 0 && (size_t)victim->gpu_id < devices.size()) {
        cudaError_t cq = cudaStreamQuery(devices[victim->gpu_id].compute_stream);
        cudaError_t pq = cudaStreamQuery(devices[victim->gpu_id].copy_stream);
        if (cq == cudaErrorNotReady) perf().n_active_upload_saw_compute_stream_busy.fetch_add(1, std::memory_order_relaxed);
        else perf().n_active_upload_saw_compute_stream_idle.fetch_add(1, std::memory_order_relaxed);
        if (pq == cudaErrorNotReady) perf().n_active_upload_saw_copy_stream_busy.fetch_add(1, std::memory_order_relaxed);
        else perf().n_active_upload_saw_copy_stream_idle.fetch_add(1, std::memory_order_relaxed);
        // 2026-08-28: the actual streams decode_async_prewarm/speculative
        // fills queue their own async DMAs onto -- see n_active_upload_saw_
        // active_layer_stream_busy's own comment (perf_timer.hpp) for why
        // checking THESE, not just compute_stream/copy_stream, is what
        // actually tests the legacy-default-stream priority-inversion risk
        // this plain cudaMemcpy call is structurally exposed to.
        cudaError_t alq = cudaStreamQuery(devices[victim->gpu_id].active_layer_stream);
        cudaError_t sq1 = cudaStreamQuery(devices[victim->gpu_id].speculative_prefetch_stream);
        cudaError_t sq2 = cudaStreamQuery(devices[victim->gpu_id].speculative_prefetch_stream2);
        bool spec_busy = (sq1 == cudaErrorNotReady) || (sq2 == cudaErrorNotReady);
        if (alq == cudaErrorNotReady) perf().n_active_upload_saw_active_layer_stream_busy.fetch_add(1, std::memory_order_relaxed);
        else perf().n_active_upload_saw_active_layer_stream_idle.fetch_add(1, std::memory_order_relaxed);
        if (spec_busy) perf().n_active_upload_saw_speculative_stream_busy.fetch_add(1, std::memory_order_relaxed);
        else perf().n_active_upload_saw_speculative_stream_idle.fetch_add(1, std::memory_order_relaxed);
        // cudaStreamQuery can leave a spurious error in the per-thread
        // last-error slot on some driver versions when it returns
        // cudaErrorNotReady -- clear it so it's never misattributed to the
        // real cudaMemcpy a few lines below (same precaution
        // demote_from_vram's own copy of this check already takes).
        cudaGetLastError();
    }
    // Record source coverage without changing the active-layer policy.  This
    // path intentionally does not register/unregister pageable slabs per
    // call; the query tells the report whether a persistent RAM pin was
    // available for this transfer.
    cudaPointerAttributes sync_attr{};
    cudaError_t sync_attr_err = cudaPointerGetAttributes(&sync_attr, host_data);
    victim->upload_source_pinned =
        sync_attr_err == cudaSuccess && sync_attr.type == cudaMemoryTypeHost;
    cudaGetLastError();
    victim->last_upload_bytes = copy_bytes;
    victim->upload_timing_pending = false;
    victim->upload_timing_recorded = false;
    victim->upload_timing_async = false;

    PERF_START(active_layer_upload);
    // 2026-08-29: cudaMemcpyAsync on a dedicated stream + an explicit
    // cudaStreamSynchronize, instead of a plain cudaMemcpy on the legacy
    // default (NULL) stream -- see active_layer_sync_fallback_stream's own
    // comment (gpu_topology.hpp) for why: the NULL stream implicitly waits
    // for every other blocking-type stream on the device to drain first, a
    // real, measured 22.5%-of-calls cost for this specific fallback. This
    // keeps the exact same synchronous contract this function has always
    // had -- the calling thread still blocks here until the copy is done,
    // only WHICH stream it queues behind (its own dedicated one, not
    // whatever active_layer_stream/copy_stream/etc. happen to have queued)
    // changes.
    cudaStream_t sync_stream = devices[victim->gpu_id].active_layer_sync_fallback_stream;
    if (victim->upload_start_event) cudaEventRecord(victim->upload_start_event, sync_stream);
    cudaError_t err = cudaMemcpyAsync(victim->device_ptr, host_data, copy_bytes,
                                cudaMemcpyHostToDevice, sync_stream);
    if (err == cudaSuccess) {
        if (victim->upload_event) cudaEventRecord(victim->upload_event, sync_stream);
        victim->upload_timing_pending = victim->upload_start_event && victim->upload_event;
        err = cudaStreamSynchronize(sync_stream);
        if (err == cudaSuccess && victim->upload_timing_pending) {
            record_active_layer_upload_event_timing(*victim, true);
        }
    }
    PERF_END(active_layer_upload, t_active_layer_upload);
    if (err != cudaSuccess) {
        LOG_STREAM_ERROR() << "[GPU] Failed to upload expert to active layer: "
                           << cudaGetErrorString(err) << "\n";
        return nullptr;
    }
    if (victim->gpu_id >= 0 && (size_t)victim->gpu_id < PerfCounters::PERF_MAX_GPU_SLOTS) {
        perf().bytes_h2d_upload_by_gpu[victim->gpu_id].fetch_add((int64_t)copy_bytes, std::memory_order_relaxed);
        perf().n_h2d_upload_by_gpu[victim->gpu_id].fetch_add(1, std::memory_order_relaxed);
    }

    // Waste accounting for whatever this victim slot held BEFORE this call
    // overwrites it -- see record_active_layer_prewarm_reclaim's own
    // comment. This function itself never sets was_async_prewarmed (a
    // reactive/confirmed-need upload, not a speculative one) UNLESS
    // `speculative` is set below, so no corresponding set call after in the
    // common case -- only the reclaim of the OUTGOING content matters here.
    record_active_layer_prewarm_reclaim(*victim);

    victim->layer_idx = layer;
    victim->expert_id = expert_id;
    victim->last_used_clock = ++cache_clock;
    victim->hit_count = 0;  // fresh identity in this slot -- find_active_expert accumulates from here
    victim->upload_ready = true;
    if (speculative) {
        // Pilot one-layer-ahead prediction (RAMSafetyManager::promote_ram_
        // to_active_layer), NOT a confirmed real need yet -- mirrors
        // upload_expert_active_async's own treatment, including its
        // 2026-08-27 speculative_active_layer_promote_on_confirm opt-in
        // (see that flag's own comment, dense_dispatch.hpp, for the real-
        // hardware data). Default (false) preserves the original skip.
        victim->pilot_speculative_fill = true;
        victim->prewarm_skip_next_promote = !speculative_active_layer_promote_on_confirm_enabled();
    } else {
        // Explicit, not just "already false from a fresh/reused slot" --
        // this IS the reactive cold-miss path (a real find_active_expert MISS
        // just happened, or this is prefill's own legacy synchronous
        // pre-warm branch), never proactive -- its own first future hit
        // should count toward promotion normally, unlike upload_expert_active_
        // async's batch pre-warm. See prewarm_skip_next_promote's own comment.
        victim->prewarm_skip_next_promote = false;
    }

    LOG_DEBUG("[GPU] Active layer upload: layer=%d expert=%d (%.2f MB)%s\n",
              layer, expert_id, copy_bytes / 1e6, speculative ? " [speculative]" : "");

    // Track three-tier metrics -- n_active_layer_uploads stays scoped to
    // real (non-speculative) uploads only, matching every existing report
    // line that reads it (e.g. the "X active-layer uploads vs Y promoted"
    // figure); speculative fills get their own n_pilot_active_fill_attempted
    // counter instead so the two populations are never silently blended.
    if (speculative) {
        perf().n_pilot_active_fill_attempted.fetch_add(1, std::memory_order_relaxed);
    } else {
        perf().n_active_layer_uploads.fetch_add(1, std::memory_order_relaxed);
    }

    return victim;
}

// See this function's own declaration in gpu_topology.hpp for the full
// correctness contract (upload_ready left false, temp_pin_count bumped,
// caller MUST batch-sync before any find_active_expert lookup can reach
// this slot). Victim-selection logic mirrors upload_expert_active exactly
// -- same eviction_score blend, same temp_pin_count skip -- the only
// differences are cudaMemcpyAsync instead of cudaMemcpy, and NOT marking
// the slot ready/unpinned here (sync_active_layer_batch does that, once,
// for the whole batch).
ExpertLocation* GpuTopology::upload_expert_active_async(int layer, int expert_id,
                                                          const uint8_t* host_data, size_t bytes,
                                                          bool speculative, int target_gpu_id,
                                                          int64_t qwen_pilot_prediction_ns) {
    if (active_layer_pool == nullptr || active_layer_capacity == 0) {
        return nullptr;
    }
    if (!host_data || bytes == 0) {
        LOG_STREAM_ERROR() << "[GPU] Invalid upload_expert_active_async parameters\n";
        return nullptr;
    }

    GPU_TOPO_TIMED_LOCK(lock);

    ExpertLocation* victim = nullptr;
    for (auto& slot : active_layer_slots) {
        // 2026-08-28: multi-GPU routing restriction -- see upload_expert_
        // active's own copy of this check for the full comment.
        if (target_gpu_id >= 0 && slot.gpu_id != target_gpu_id) continue;
        if (slot.temp_pin_count > 0) continue;  // in flight (this batch or a prior one) -- never re-target
        if (slot.layer_idx == -1) {
            victim = &slot;
            break;
        }
        // 2026-08-26: an unconfirmed speculative fill always outranks a
        // real/confirmed slot as a victim -- see ExpertLocation::
        // preferred_victim_over's own comment.
        if (!victim || slot.preferred_victim_over(*victim)) {
            victim = &slot;
        }
    }

    if (!victim) {
        LOG_STREAM_WARN() << "[GPU] Active layer area full - cannot async-upload expert"
                          << (target_gpu_id >= 0 ? (" on GPU " + std::to_string(target_gpu_id)) : std::string())
                          << "\n";
        return nullptr;
    }

    if (bytes > victim->size) {
        LOG_STREAM_ERROR() << "[DIAG-CLAMP] active-layer upload TRUNCATED: layer=" << layer
                           << " expert=" << expert_id << " requested_bytes=" << bytes
                           << " slot_capacity=" << victim->size
                           << " (prev occupant layer=" << victim->layer_idx
                           << " expert=" << victim->expert_id << ")\n";
    }
    size_t copy_bytes = std::min(bytes, victim->size);
    cudaSetDevice(victim->gpu_id);
    auto& dev = devices[victim->gpu_id];
    // Same accounting guard for the async sibling: do not re-record a slot's
    // persistent start/end events until the previous copy has been charged.
    if (victim->upload_timing_pending && victim->upload_event &&
        cudaEventQuery(victim->upload_event) == cudaSuccess) {
        record_active_layer_upload_event_timing(*victim, false);
    }
    cudaGetLastError();

    // TWO fixes tried at this call site (2026-08-15), BOTH REVERTED after
    // real, confirmed hardware regressions -- kept as a documented dead end
    // so a future session doesn't re-attempt either from scratch.
    //
    // Attempt 1: cudaHostRegister/cudaHostUnregister per call (Alt A, same
    // mechanism as upload_expert/upload_expert_async), after root-causing
    // active_layer_prewarm_issue=18.90s/13937 calls to this cudaMemcpyAsync
    // silently falling back to a synchronous internal-bounce-buffer copy
    // (host_data is unpinned posix_memalign memory). Issue cost dropped to
    // ~0.15s as predicted, but register+unregister alone cost as much as
    // the problem it replaced (~19.1-19.4s vs 18.90s) AND
    // sync_active_layer_batch's own cudaStreamSynchronize wait ballooned
    // ~10x (0.52s->5.25s). Net: expert= +~6s, tok/s 0.57->0.52-0.53.
    //
    // Attempt 2: route through a dedicated, pre-pinned bounce buffer pool
    // instead (active_layer_bounce_pool, pinned ONCE at startup via
    // cudaHostAlloc, zero per-call register/unregister) -- ruled OUT the
    // cudaHostRegister-driver-serialization theory for attempt 1's sync
    // regression: this version makes ZERO register/unregister calls yet
    // sync_active_layer_batch's wait STILL ballooned to the same ~5.25s.
    // Total upload-side cost (bounce_stage+issue+bounce_wait+sync) came to
    // ~22.5-22.6s across two real runs, WORSE than baseline's 19.42s
    // (though better than attempt 1's ~24.5-24.8s). Real explanation:
    // ~13,937 calls x ~14.7MB =~205GB/request moved by active-layer
    // pre-warm alone -- at realistic contended H2D bandwidth this is
    // ~19-20s of genuinely necessary transfer time, matching baseline's
    // total almost exactly. The original "issue" cost wasn't pure overhead
    // to eliminate -- the secretly-synchronous cudaMemcpyAsync was
    // ACCIDENTALLY providing real backpressure that paced demand on
    // copy_stream; both fixes removed that pacing, letting DMAs pile up
    // faster than the copy engine drains them, and the SAME real cost
    // resurfaced as sync= instead of issue=, plus each fix's own added
    // overhead (register/unregister, or the bounce-stage memcpy) on top.
    // Neither fix reduces the real bytes moved -- see t_active_layer_
    // upload_async's own comment for the actual next lever (reducing HOW
    // MUCH gets pre-warmed, not how it's transferred).
    // Diagnostic ONLY, same pattern as upload_expert_active's own check
    // above (mirrors demote_from_vram's cudaStreamQuery approach) -- see
    // n_active_prewarm_saw_*'s own comment (perf_timer.hpp) for why THIS
    // call site needs its own re-check rather than reusing either of the
    // two existing findings. Non-blocking; cudaSuccess = idle,
    // cudaErrorNotReady = real in-flight work on that stream right now.
    if (victim->gpu_id >= 0 && (size_t)victim->gpu_id < devices.size()) {
        cudaError_t cq = cudaStreamQuery(dev.compute_stream);
        cudaError_t pq = cudaStreamQuery(dev.copy_stream);
        if (cq == cudaErrorNotReady) perf().n_active_prewarm_saw_compute_stream_busy.fetch_add(1, std::memory_order_relaxed);
        else perf().n_active_prewarm_saw_compute_stream_idle.fetch_add(1, std::memory_order_relaxed);
        if (pq == cudaErrorNotReady) perf().n_active_prewarm_saw_copy_stream_busy.fetch_add(1, std::memory_order_relaxed);
        else perf().n_active_prewarm_saw_copy_stream_idle.fetch_add(1, std::memory_order_relaxed);
        // cudaStreamQuery can leave a spurious error in the per-thread
        // last-error slot on some driver versions when it returns
        // cudaErrorNotReady -- clear it so it's never misattributed to the
        // real cudaMemcpyAsync a few lines below (same precaution
        // upload_expert_active's own copy of this check already takes).
        cudaGetLastError();
    }
    // 2026-08-26: speculative pilot fills get their OWN dedicated stream,
    // separate from active_layer_stream's own, more time-critical
    // consumers (decode_async_prewarm's real-need pre-warm, prefill's
    // Phase-2 batch warm-up, reactive synchronous upload_expert_active) --
    // see GpuDevice::speculative_prefetch_stream's own comment for the
    // full reasoning and the prior, analogous real win (active_layer_
    // stream's own creation) this mirrors. A speculative l+1 guess should
    // never be able to queue ahead of and delay a confirmed real need on a
    // shared stream.
    // 2026-08-26 follow-up: REVERTED back to single-stream after a real,
    // controlled A/B disproved the two-stream hypothesis -- see
    // speculative_prefetch_stream2's own comment (gpu_topology.hpp) for
    // the full mechanism trace. Same max_k=8, same prompt, single-stream
    // vs two-stream (round-robin via speculative_stream_rr): self-
    // contention (n_speculative_issue_saw_own_stream_busy) dropped exactly
    // as designed, 31.6%->2.1%, but active_layer_prewarm_issue itself got
    // WORSE (15.95s->16.46s) and decode went the wrong way (0.84->0.83).
    // cudaStreamQuery's busy/idle signal doesn't correlate with real cost
    // here -- fixing the metric made it look better while the two-stream
    // split's own small, real per-stream driver/API overhead outweighed
    // whatever queuing time it saved. Combined with active_layer_stream
    // busy staying 0.0% in every test (cross-mechanism contention already
    // ruled out), this closes off BOTH stream-level hypotheses -- the
    // remaining ~2s/request cost is most likely genuine physical copy-
    // engine bandwidth saturation (this hardware's 2 real DMA engines,
    // shared by every logical stream regardless of software organization),
    // which no stream reorganization can route around. speculative_
    // prefetch_stream2/speculative_stream_rr are left in place (harmless,
    // unused) rather than deleted, in case a future revisit wants them.
    cudaStream_t issue_stream = speculative ? dev.speculative_prefetch_stream : dev.active_layer_stream;
    // 2026-08-26: diagnostic ONLY, speculative path specifically -- directly
    // tests the leading hypothesis for why active_layer_prewarm_issue grew
    // more than the ~5% wrong-guess rate alone would predict: even a
    // CORRECTLY-predicted speculative fill substitutes for (not duplicates)
    // decode_async_prewarm's own later call for the same expert (confirmed
    // via active_layer_has's own dedup check above this function's
    // caller), but substituting WHEN a transfer is issued can still cost
    // more real issue-time if the moment it's issued is more contended.
    // active_layer_stream: is decode_async_prewarm's own, more time-
    // critical traffic active on the stream THIS call was moved off of --
    // if genuinely idle, the dedicated-stream split bought nothing to
    // avoid. speculative_prefetch_stream: is THIS call backing up behind
    // an earlier speculative issue that hasn't drained yet (self-
    // contention, distinct from cross-mechanism contention). Same
    // cudaStreamQuery pattern as the pre-existing n_active_prewarm_saw_*
    // diagnostic just above -- non-blocking, cudaSuccess = idle,
    // cudaErrorNotReady = real in-flight work right now.
    if (speculative && victim->gpu_id >= 0 && (size_t)victim->gpu_id < devices.size()) {
        cudaError_t alq = cudaStreamQuery(dev.active_layer_stream);
        // Query whichever of the two speculative streams THIS call actually
        // picked above (issue_stream), not always the first -- with the
        // round-robin split, "own stream busy" now means "the specific
        // stream this call landed on has a prior, still-draining issue",
        // which is exactly the self-contention question, correctly scoped
        // per-stream instead of conflating both.
        cudaError_t spq = cudaStreamQuery(issue_stream);
        if (alq == cudaErrorNotReady) perf().n_speculative_issue_saw_active_layer_stream_busy.fetch_add(1, std::memory_order_relaxed);
        else perf().n_speculative_issue_saw_active_layer_stream_idle.fetch_add(1, std::memory_order_relaxed);
        if (spq == cudaErrorNotReady) perf().n_speculative_issue_saw_own_stream_busy.fetch_add(1, std::memory_order_relaxed);
        else perf().n_speculative_issue_saw_own_stream_idle.fetch_add(1, std::memory_order_relaxed);
        // Same spurious-last-error precaution as the pre-existing check.
        cudaGetLastError();
    }
    // 2026-08-28: is this source ALREADY pinned right now? Queried here,
    // OUTSIDE the timed section below, specifically to test whether this
    // bucket's real cost is PCIe transfer time (ruled out separately via
    // real nvidia-smi telemetry showing GPU0's PCIe RX nowhere near
    // saturated) or a CPU-bound driver-internal staged copy for unpinned
    // memory hiding inside a call that looks async. cudaPointerGetAttributes
    // is a cheap driver query (microseconds), not the multi-millisecond
    // cost this is trying to explain -- querying it here rather than
    // threading a parameter through every caller keeps this self-contained.
    cudaPointerAttributes attr{};
    cudaError_t attr_err = cudaPointerGetAttributes(&attr, host_data);
    bool src_pinned = (attr_err == cudaSuccess && attr.type == cudaMemoryTypeHost);
    cudaGetLastError();  // clear any spurious error the query itself leaves behind

    victim->last_upload_bytes = copy_bytes;
    victim->qwen_pilot_prediction_ns = qwen_pilot_prediction_ns;
    victim->qwen_pilot_issue_ns =
        qwen_pilot_prediction_ns >= 0
            ? std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now().time_since_epoch())
                  .count()
            : -1;
    victim->upload_timing_pending = false;
    victim->upload_timing_recorded = false;
    victim->upload_timing_async = true;
    victim->upload_source_pinned = src_pinned;
    if (victim->upload_start_event) cudaEventRecord(victim->upload_start_event, issue_stream);
    PERF_START(active_layer_upload_async);
    cudaError_t err = cudaMemcpyAsync(victim->device_ptr, host_data, copy_bytes,
                                       cudaMemcpyHostToDevice, issue_stream);
    PERF_END(active_layer_upload_async, t_active_layer_upload_async);
    {
        double call_s = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - _perf_active_layer_upload_async).count();
        if (src_pinned) {
            PerfCounters::atomic_add(perf().t_active_layer_upload_async_pinned, call_s);
            perf().n_active_layer_upload_async_pinned_calls.fetch_add(1, std::memory_order_relaxed);
        } else {
            PerfCounters::atomic_add(perf().t_active_layer_upload_async_unpinned, call_s);
            perf().n_active_layer_upload_async_unpinned_calls.fetch_add(1, std::memory_order_relaxed);
        }
    }
    perf().bytes_active_layer_upload_async.fetch_add((int64_t)copy_bytes, std::memory_order_relaxed);
    if (victim->gpu_id >= 0 && (size_t)victim->gpu_id < PerfCounters::PERF_MAX_GPU_SLOTS) {
        perf().bytes_h2d_upload_by_gpu[victim->gpu_id].fetch_add((int64_t)copy_bytes, std::memory_order_relaxed);
        perf().n_h2d_upload_by_gpu[victim->gpu_id].fetch_add(1, std::memory_order_relaxed);
    }
    if (err != cudaSuccess) {
        LOG_STREAM_ERROR() << "[GPU] Failed to async-upload expert to active layer: "
                           << cudaGetErrorString(err) << "\n";
        return nullptr;
    }
    // 2026-08-26: record completion on the SAME stream the copy above was
    // issued on -- CUDA streams execute enqueued work in issue order, so
    // this event firing (cudaEventQuery == cudaSuccess) is a reliable proof
    // the DMA above has genuinely finished. See find_active_expert's own
    // new poll-before-trust branch (mirrors GpuTopology::finish_expert_
    // hit's already-proven pattern for the main VRAM cache) -- this is what
    // makes it safe to skip the caller's blocking sync_active_layer_batch
    // entirely for the speculative pilot-fill path.
    if (victim->upload_event) cudaEventRecord(victim->upload_event, issue_stream);
    victim->upload_timing_pending = victim->upload_start_event && victim->upload_event;

    // Waste accounting for whatever this victim slot held BEFORE this call
    // overwrites it -- must run before the identity-overwrite below, since
    // was_async_prewarmed/prewarm_touched describe the OUTGOING content.
    // Safe/no-op for an empty slot (was_async_prewarmed already false).
    record_active_layer_prewarm_reclaim(*victim);

    // Claim identity + protection NOW (so a second call in this same batch
    // can't pick this slot again), but leave upload_ready false -- the
    // caller's batch sync flips it true only once the DMA is confirmed
    // complete for the WHOLE batch.
    victim->temp_pin_count++;
    victim->layer_idx = layer;
    victim->expert_id = expert_id;
    victim->last_used_clock = ++cache_clock;
    victim->hit_count = 0;
    victim->upload_ready = false;
    if (speculative) {
        // Pilot one-layer-ahead prediction -- own flag/counter pair, see
        // upload_expert_active's own (sync) speculative branch and
        // ExpertLocation::pilot_speculative_fill's comment for why this is
        // kept separate from was_async_prewarmed.
        victim->pilot_speculative_fill = true;
    } else {
        victim->was_async_prewarmed = true;
    }
    // The whole point of this flag -- see its own comment in
    // gpu_topology.hpp for the real-hardware regression this closes.
    // Applies to the non-speculative (decode_async_prewarm) source
    // unconditionally -- its first touch is a GUARANTEED same-token
    // dispatch, carrying zero reuse signal.
    //
    // 2026-08-27: for the speculative source specifically, this default
    // is now opt-out via speculative_active_layer_promote_on_confirm
    // (EngineConfig, dense_dispatch.hpp -- see its own comment for the
    // real-hardware data motivating this). A speculative fill's first
    // touch is NOT guaranteed (~94-95% correct, not 100%) -- it's a
    // genuine confirmation of the pilot's guess, and skipping it (the
    // default, unchanged behavior) makes promotion to the persistent
    // main cache effectively impossible for this path, since the active-
    // layer window for this layer index is gone before a later decode
    // token could ever supply the SECOND touch active_layer_promote_
    // hit_threshold needs. Default false preserves the exact prior
    // behavior; when true, skip this speculative fill's confirmation.
    victim->prewarm_skip_next_promote =
        speculative ? !speculative_active_layer_promote_on_confirm_enabled() : true;

    if (speculative) {
        perf().n_pilot_active_fill_attempted.fetch_add(1, std::memory_order_relaxed);
    } else {
        perf().n_active_layer_uploads_async_issued.fetch_add(1, std::memory_order_relaxed);
    }
    return victim;
}

ExpertLocation* GpuTopology::claim_active_layer_slot_for_worker(int layer, int expert_id, size_t bytes,
                                                                   bool speculative, int target_gpu_id) {
    if (active_layer_pool == nullptr || active_layer_capacity == 0) {
        return nullptr;
    }
    if (bytes == 0) {
        LOG_STREAM_ERROR() << "[GPU] Invalid claim_active_layer_slot_for_worker parameters\n";
        return nullptr;
    }

    GPU_TOPO_TIMED_LOCK(lock);

    // Victim selection: byte-for-byte the same policy as upload_expert_
    // active_async's own loop above (same target-GPU filter, same temp_pin_
    // count skip, same unconfirmed-speculative-outranks-real preference) --
    // deliberately kept in lockstep so a future change to one doesn't
    // silently diverge from the other. See this function's own declaration
    // comment (gpu_topology.hpp) for why victim selection is split out from
    // the actual copy here.
    ExpertLocation* victim = nullptr;
    for (auto& slot : active_layer_slots) {
        if (target_gpu_id >= 0 && slot.gpu_id != target_gpu_id) continue;
        if (slot.temp_pin_count > 0) continue;
        if (slot.layer_idx == -1) {
            victim = &slot;
            break;
        }
        if (!victim || slot.preferred_victim_over(*victim)) {
            victim = &slot;
        }
    }

    if (!victim) {
        LOG_STREAM_WARN() << "[GPU] Active layer area full - cannot claim slot for worker upload"
                          << (target_gpu_id >= 0 ? (" on GPU " + std::to_string(target_gpu_id)) : std::string())
                          << "\n";
        return nullptr;
    }

    // Waste accounting for whatever this victim slot held BEFORE this call
    // overwrites it -- must run before the identity-overwrite below, same
    // as upload_expert_active_async's own copy of this call.
    record_active_layer_prewarm_reclaim(*victim);

    // Claim identity + protection now, exactly as upload_expert_active_
    // async does -- EXCEPT no copy has been issued yet (no cudaMemcpyAsync,
    // no cudaEventRecord). The caller (ActiveLayerUploadWorkers) is
    // responsible for actually performing the register+copy+event sequence
    // on a background thread and, once cudaEventRecord genuinely runs,
    // flipping copy_dispatched back to true under this same lock -- until
    // then, this slot must be treated as a miss by any reader (see
    // find_active_expert's and cleanup_active_layer_area's own
    // copy_dispatched checks).
    (void)bytes;  // not needed here -- copy_bytes = min(bytes, victim->size) is computed by the worker
    victim->temp_pin_count++;
    victim->layer_idx = layer;
    victim->expert_id = expert_id;
    victim->last_used_clock = ++cache_clock;
    victim->hit_count = 0;
    victim->upload_ready = false;
    victim->copy_dispatched = false;
    if (speculative) {
        victim->pilot_speculative_fill = true;
    } else {
        victim->was_async_prewarmed = true;
    }
    victim->prewarm_skip_next_promote =
        speculative ? !speculative_active_layer_promote_on_confirm_enabled() : true;

    if (speculative) {
        perf().n_pilot_active_fill_attempted.fetch_add(1, std::memory_order_relaxed);
    } else {
        perf().n_active_layer_uploads_async_issued.fetch_add(1, std::memory_order_relaxed);
    }
    return victim;
}

ExpertLocation *GpuTopology::upload_expert_active_projections_async(
    int layer, int expert_id, const uint8_t *gate_data, size_t gate_bytes,
    const uint8_t *up_data, size_t up_bytes, const uint8_t *down_data,
    size_t down_bytes, int target_gpu_id) {
    if (!active_layer_pool || active_layer_capacity == 0 || !gate_data ||
        !up_data || !down_data || gate_bytes == 0 || up_bytes == 0 ||
        down_bytes == 0 || gate_bytes > SIZE_MAX - up_bytes ||
        gate_bytes + up_bytes > SIZE_MAX - down_bytes)
        return nullptr;

    const size_t total_bytes = gate_bytes + up_bytes + down_bytes;
    GPU_TOPO_TIMED_LOCK(lock);

    ExpertLocation *victim = nullptr;
    for (auto &slot : active_layer_slots) {
        if (target_gpu_id >= 0 && slot.gpu_id != target_gpu_id)
            continue;
        if (slot.temp_pin_count > 0)
            continue;
        if (slot.layer_idx == -1) {
            victim = &slot;
            break;
        }
        if (!victim || slot.preferred_victim_over(*victim))
            victim = &slot;
    }
    if (!victim || total_bytes > victim->size) {
        if (victim) {
            LOG_STREAM_ERROR()
                << "[OPTION-B-STAGE] Layer " << layer << " expert "
                << expert_id << " needs " << total_bytes
                << " bytes but active slot capacity is " << victim->size
                << "\n";
        }
        return nullptr;
    }

    cudaSetDevice(victim->gpu_id);
    auto &dev = devices[victim->gpu_id];
    if (victim->upload_timing_pending && victim->upload_event &&
        cudaEventQuery(victim->upload_event) == cudaSuccess)
        record_active_layer_upload_event_timing(*victim, false);
    cudaGetLastError();

    cudaPointerAttributes attr{};
    cudaError_t attr_err = cudaPointerGetAttributes(&attr, gate_data);
    const bool source_pinned =
        attr_err == cudaSuccess && attr.type == cudaMemoryTypeHost;
    cudaGetLastError();

    victim->last_upload_bytes = total_bytes;
    victim->qwen_pilot_prediction_ns = -1;
    victim->qwen_pilot_issue_ns = -1;
    victim->upload_timing_pending = false;
    victim->upload_timing_recorded = false;
    victim->upload_timing_async = true;
    victim->upload_source_pinned = source_pinned;
    victim->copy_dispatched = true;
    const cudaStream_t issue_stream = dev.active_layer_stream;
    if (!issue_stream)
        return nullptr;
    if (victim->upload_start_event)
        cudaEventRecord(victim->upload_start_event, issue_stream);
    cudaError_t err = cudaMemcpyAsync(victim->device_ptr, gate_data,
                                      gate_bytes, cudaMemcpyHostToDevice,
                                      issue_stream);
    if (err == cudaSuccess)
        err = cudaMemcpyAsync(victim->device_ptr + gate_bytes, up_data,
                              up_bytes, cudaMemcpyHostToDevice, issue_stream);
    if (err == cudaSuccess)
        err = cudaMemcpyAsync(victim->device_ptr + gate_bytes + up_bytes,
                              down_data, down_bytes, cudaMemcpyHostToDevice,
                              issue_stream);
    if (err != cudaSuccess) {
        LOG_STREAM_ERROR() << "[OPTION-B-STAGE] async projection upload failed: "
                           << cudaGetErrorString(err) << "\n";
        cudaGetLastError();
        return nullptr;
    }
    if (victim->upload_event)
        cudaEventRecord(victim->upload_event, issue_stream);
    victim->upload_timing_pending = victim->upload_start_event &&
                                    victim->upload_event;

    record_active_layer_prewarm_reclaim(*victim);
    victim->temp_pin_count++;
    victim->layer_idx = layer;
    victim->expert_id = expert_id;
    victim->last_used_clock = ++cache_clock;
    victim->hit_count = 0;
    victim->upload_ready = false;
    victim->was_async_prewarmed = false;
    victim->pilot_speculative_fill = false;
    victim->prewarm_skip_next_promote = true;

    perf().bytes_active_layer_upload_async.fetch_add(
        (int64_t)total_bytes, std::memory_order_relaxed);
    if (victim->gpu_id >= 0 &&
        (size_t)victim->gpu_id < PerfCounters::PERF_MAX_GPU_SLOTS) {
        perf().bytes_h2d_upload_by_gpu[victim->gpu_id].fetch_add(
            (int64_t)total_bytes, std::memory_order_relaxed);
        perf().n_h2d_upload_by_gpu[victim->gpu_id].fetch_add(
            1, std::memory_order_relaxed);
    }
    LOG_INFO("[OPTION-B-STAGE] queued Layer %d expert %d projections: "
             "%.3f MiB source=%s gpu=%d\n",
             layer, expert_id, (double)total_bytes / (1024.0 * 1024.0),
             source_pinned ? "pinned" : "pageable", victim->gpu_id);
    return victim;
}

void GpuTopology::complete_active_layer_upload_after_compute(
    ExpertLocation *slot) {
    if (!slot)
        return;
    cudaSetDevice(slot->gpu_id);
    if (slot->upload_timing_pending && slot->upload_event &&
        cudaEventQuery(slot->upload_event) == cudaSuccess)
        record_active_layer_upload_event_timing(*slot, false);
    cudaGetLastError();
    GPU_TOPO_TIMED_LOCK(lock);
    slot->upload_ready = true;
    if (slot->temp_pin_count > 0)
        slot->temp_pin_count--;
}

void GpuTopology::sync_active_layer_batch(const std::vector<ExpertLocation*>& victims) {
    if (victims.empty()) return;
    // 2026-08-28: with multi-GPU active-layer routing, a single batch CAN
    // now contain victims spread across more than one GPU (e.g. decode_
    // async_prewarm's own chunk, if the router split that chunk's
    // candidates across GPUs) -- the original single-gpu_id assumption
    // ("the active layer area lives on exactly one GPU by construction")
    // no longer holds once active_layer_gpu_slices has 2+ entries. Fixed
    // by syncing each DISTINCT gpu_id present in the batch exactly once
    // (order doesn't matter -- each sync only needs to complete before
    // this function marks THAT gpu's own victims ready, and all syncs
    // happen before ANY victim is marked ready, so there's no window
    // where a not-yet-synced GPU's victim looks ready). Legacy single-GPU
    // case (the overwhelming common case) still does exactly one
    // cudaSetDevice+cudaStreamSynchronize, identical cost to before.
    static thread_local std::vector<int> distinct_gpus;
    distinct_gpus.clear();
    for (auto* v : victims) {
        if (std::find(distinct_gpus.begin(), distinct_gpus.end(), v->gpu_id) == distinct_gpus.end()) {
            distinct_gpus.push_back(v->gpu_id);
        }
    }
    PERF_START(active_layer_batch_sync);
    for (int gpu_id : distinct_gpus) {
        cudaSetDevice(gpu_id);
        cudaStreamSynchronize(devices[gpu_id].active_layer_stream);
    }
    PERF_END(active_layer_batch_sync, t_active_layer_batch_sync);

    // The stream waits above prove every completion event in this batch is
    // ready. Read each persistent event pair before a slot can be reused.
    // This is intentionally separate from t_active_layer_batch_sync: that
    // CPU wait includes queueing behind earlier work, while this reports the
    // actual copy-engine interval.
    for (auto* v : victims) {
        if (v && v->upload_timing_pending) {
            cudaSetDevice(v->gpu_id);
            record_active_layer_upload_event_timing(*v, false);
        }
    }

    GPU_TOPO_TIMED_LOCK(lock);
    for (auto* v : victims) {
        v->upload_ready = true;
        if (v->temp_pin_count > 0) v->temp_pin_count--;
    }
    perf().n_active_layer_uploads_async_synced.fetch_add((uint64_t)victims.size(), std::memory_order_relaxed);
}

void GpuTopology::record_active_layer_upload_event_timing(ExpertLocation& slot,
                                                           bool sync_path) {
    if (!slot.upload_timing_pending || slot.upload_timing_recorded ||
        !slot.upload_start_event || !slot.upload_event) {
        return;
    }
    cudaSetDevice(slot.gpu_id);
    float elapsed_ms = 0.0f;
    cudaError_t err = cudaEventElapsedTime(&elapsed_ms, slot.upload_start_event,
                                           slot.upload_event);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return;
    }

    const double seconds = static_cast<double>(elapsed_ms) * 1e-3;
    const int64_t bytes = static_cast<int64_t>(slot.last_upload_bytes);
    const bool pinned = slot.upload_source_pinned;
    slot.upload_timing_recorded = true;
    slot.upload_timing_pending = false;

    // The CUDA event pair is the real copy-engine service interval for this
    // prediction fill.  It is intentionally separate from the host-observed
    // time at which a later poll noticed completion, which would include
    // unrelated CPU work and make a good DMA look slow.
    if (slot.qwen_pilot_prediction_ns >= 0) {
        if (slot.qwen_pilot_issue_ns >= slot.qwen_pilot_prediction_ns) {
            const double ready_to_issue =
                (double)(slot.qwen_pilot_issue_ns -
                         slot.qwen_pilot_prediction_ns) * 1e-9;
            PerfCounters::atomic_add(perf().t_qwen_pilot_active_ready_to_issue,
                                     ready_to_issue);
            perf().n_qwen_pilot_active_ready_to_issue.fetch_add(
                1, std::memory_order_relaxed);
        }
        PerfCounters::atomic_add(perf().t_qwen_pilot_active_dma, seconds);
        perf().bytes_qwen_pilot_active_dma.fetch_add(bytes,
                                                     std::memory_order_relaxed);
        perf().n_qwen_pilot_active_dma.fetch_add(1, std::memory_order_relaxed);
        slot.qwen_pilot_prediction_ns = -1;
        slot.qwen_pilot_issue_ns = -1;
    }

    auto add = [&](std::atomic<double>& t, std::atomic<int64_t>& b,
                   std::atomic<uint64_t>& n) {
        PerfCounters::atomic_add(t, seconds);
        b.fetch_add(bytes, std::memory_order_relaxed);
        n.fetch_add(1, std::memory_order_relaxed);
    };
    if (sync_path) {
        if (pinned) add(perf().t_active_layer_dma_sync_pinned,
                        perf().bytes_active_layer_dma_sync_pinned,
                        perf().n_active_layer_dma_sync_pinned);
        else add(perf().t_active_layer_dma_sync_unpinned,
                 perf().bytes_active_layer_dma_sync_unpinned,
                 perf().n_active_layer_dma_sync_unpinned);
    } else {
        if (pinned) add(perf().t_active_layer_dma_async_pinned,
                        perf().bytes_active_layer_dma_async_pinned,
                        perf().n_active_layer_dma_async_pinned);
        else add(perf().t_active_layer_dma_async_unpinned,
                 perf().bytes_active_layer_dma_async_unpinned,
                 perf().n_active_layer_dma_async_unpinned);
    }
}

void GpuTopology::cleanup_active_layer_area(int current_layer_idx,
                                             const std::function<void(int, int)>& on_promoted) {
    if (active_layer_pool == nullptr) {
        return;  // Not allocated, nothing to cleanup
    }

    GPU_TOPO_TIMED_LOCK(lock);

    // Decode-only occupancy sample -- see PerfCounters::active_layer_
    // decode_occupancy_max's own comment. Taken BEFORE any freeing below,
    // so it reflects real simultaneous usage (current layer's real misses
    // + any protected next-layer speculative fills) at the moment this
    // layer's own window closes, not the steady-state-after-cleanup count.
    {
        uint64_t occ = 0;
        for (const auto& slot : active_layer_slots) {
            if (slot.layer_idx != -1) ++occ;
        }
        auto& occ_sum = in_prefill ? perf().active_layer_prefill_occupancy_sum
                                    : perf().active_layer_decode_occupancy_sum;
        auto& occ_samples = in_prefill ? perf().active_layer_prefill_occupancy_samples
                                        : perf().active_layer_decode_occupancy_samples;
        auto& occ_max = in_prefill ? perf().active_layer_prefill_occupancy_max
                                    : perf().active_layer_decode_occupancy_max;
        occ_sum.fetch_add(occ, std::memory_order_relaxed);
        occ_samples.fetch_add(1, std::memory_order_relaxed);
        uint64_t prev_max = occ_max.load(std::memory_order_relaxed);
        while (occ > prev_max &&
               !occ_max.compare_exchange_weak(prev_max, occ, std::memory_order_relaxed)) {}
    }

    int slots_freed = 0;
    for (auto& slot : active_layer_slots) {
        // Protect a speculative one-layer-ahead pre-fill (RAMSafetyManager::
        // promote_ram_to_active_layer, tagged current_layer_idx+1, inserted
        // DURING the layer that just finished's own processing) from being
        // wiped out by THIS SAME cleanup call, before the layer it predicts
        // has even started dispatching -- see GpuTopology::speculative_
        // active_layer_fill's own comment for the full reasoning. A plain
        // "+1" match here is safe unconditionally (not itself re-checking
        // the flag) because nothing is ever tagged current_layer_idx+1 in
        // the first place unless that flag is on.
        bool is_next_layer_speculative =
            (speculative_active_layer_fill || slot.pilot_speculative_fill) &&
            slot.layer_idx == current_layer_idx + 1;
        if (slot.layer_idx != current_layer_idx && !is_next_layer_speculative && slot.layer_idx != -1) {
            // Never touch a slot a GPU kernel may still be reading (same
            // discipline as evict_lru/upload_expert_active's victim picker
            // above) -- skip it this round; a LATER cleanup call (once
            // expert_ffn_gpu_finish's release_hold drops this to 0) will
            // catch it. Leaves this slot's stale data resident one round
            // longer than ideal, never unsafe.
            if (slot.temp_pin_count > 0) {
                // 2026-08-26: opportunistic poll-and-release for a stale
                // async-issue hold whose DMA has since completed but that
                // NOTHING ever polled via find_active_expert (e.g. a
                // speculative pilot fill real dispatch never happened to
                // consult before its layer window closed -- real dispatch
                // found the expert via a fresh duplicate fetch instead).
                // Without this, temp_pin_count would stay stuck at 1
                // forever (nothing else will ever poll THIS exact identity
                // again), permanently leaking the slot out of every future
                // reclaim pass. Safe to attempt unconditionally: if
                // upload_ready is already true, this was already released
                // (by find_active_expert's own poll, or the old batch-sync
                // path) -- skip. If false and the event hasn't fired yet,
                // this is a genuine still-in-flight DMA OR a live GPU
                // kernel read (hold_for_compute's own, separate +1) --
                // cudaEventQuery only ever confirms the DMA specifically,
                // so a kernel-read hold with the DMA already long done
                // correctly still leaves temp_pin_count > 0 here (the
                // event query only releases the async-issue's own +1, and
                // a kernel hold can only exist on a slot find_active_expert
                // already returned, which means upload_ready is already
                // true by then -- so this branch is unreachable for that
                // case, this decrement is exclusively for the leak
                // scenario above).
                // 2026-08-28: same copy_dispatched guard as find_active_
                // expert's own poll (see ExpertLocation::copy_dispatched's
                // comment) -- a slot claimed via claim_active_layer_slot_
                // for_worker but not yet dispatched has upload_event either
                // null or still referring to whatever the PREVIOUS occupant
                // last recorded on it; either way cudaEventQuery on it can
                // spuriously report cudaSuccess (an event nobody has
                // re-recorded yet, or one that already fired long ago, both
                // read back as "complete"). Without this check, this
                // opportunistic release could fire BEFORE the worker thread
                // has even started its cudaHostRegister/cudaMemcpyAsync,
                // wrongly dropping the hold (and the RAM-tier source's own
                // protection, once the caller reacts to temp_pin_count
                // hitting 0) while the worker is about to read/write it.
                if (!slot.upload_ready && slot.copy_dispatched && slot.upload_event) {
                    cudaSetDevice(slot.gpu_id);
                    if (cudaEventQuery(slot.upload_event) == cudaSuccess) {
                        record_active_layer_upload_event_timing(slot, false);
                        slot.upload_ready = true;
                        if (slot.temp_pin_count > 0) slot.temp_pin_count--;
                        perf().n_active_layer_stale_hold_released.fetch_add(1, std::memory_order_relaxed);
                    }
                }
                if (slot.temp_pin_count > 0) continue;
            }
            // This slot's layer just finished (execution has moved on to
            // current_layer_idx) -- promote it into the real VRAM LRU cache
            // if it proved hot enough while resident (find_active_expert
            // already promotes+frees on an inline hit crossing the
            // threshold; this is the end-of-layer catch for anything that
            // hit fewer times than that but at least once), otherwise this
            // is exactly "swap out the one not used": just free it.
            // Gated on !in_prefill -- see that field's own comment
            // (gpu_topology.hpp) for why prefill NEVER promotes from the
            // active layer into the main cache, regardless of hit_count.
            const bool repeat_hit = slot.hit_count >= active_layer_promote_threshold;
            // A speculative fill that no dispatch ever touched is only a
            // guess; every other decode slot is a real miss that was used.
            const bool promote_miss =
                decode_promote_all_misses && slot.upload_ready &&
                (!slot.pilot_speculative_fill || slot.prewarm_touched);
            if (!in_prefill && (repeat_hit || promote_miss)) {
                // The active pool is sized by its largest slab, while the
                // compact main cache has per-layer slab sizes. Promote only
                // the bytes actually uploaded for this identity; using the
                // active slot capacity can make evict_lru reject a valid
                // smaller layer-region slot.
                const size_t payload_bytes =
                    slot.last_upload_bytes > 0
                        ? std::min(slot.last_upload_bytes, slot.size)
                        : slot.size;
                if (repeat_hit) {
                    promote_active_to_main(slot.layer_idx, slot.expert_id,
                                           slot.device_ptr, payload_bytes, on_promoted,
                                           slot.gpu_id);
                } else {
                    // COPY-style, same GPU: keep the RAM copy, no host bounce.
                    promote_active_to_main(slot.layer_idx, slot.expert_id,
                                           slot.device_ptr, payload_bytes, nullptr,
                                           slot.gpu_id, /*same_gpu_victim=*/true);
                }
            }
            // Waste accounting -- the main "swap out the one not used" path,
            // expected to catch the bulk of any real waste this feature
            // produces (a slot that reached end-of-layer with prewarm_
            // touched still false, i.e. no real dispatch EVER looked for
            // this exact expert here before its layer window closed). Must
            // run before the identity-clear below, same as every other
            // reclaim site.
            record_active_layer_prewarm_reclaim(slot);
            // Clear slot regardless of whether promotion happened above --
            // promote_active_to_main only copies the bytes elsewhere, it
            // never touches this slot's own identity.
            slot.layer_idx = -1;
            slot.expert_id = -1;
            slot.last_used_clock = 0;
            slot.hit_count = 0;
            slot.upload_ready = true;
            slot.temp_pin_count = 0;
            slot.prewarm_skip_next_promote = false;
            slots_freed++;
        }
    }

    if (slots_freed > 0) {
        LOG_DEBUG("[GPU] Active layer cleanup: freed %d slots (current layer=%d)\n",
                  slots_freed, current_layer_idx);
        // Track three-tier metrics
        perf().n_active_layer_cleanup.fetch_add(slots_freed, std::memory_order_relaxed);
    }
}

ExpertLocation* GpuTopology::find_active_expert(int layer, int expert_id,
                                                 const std::function<void(int, int)>& on_promoted) {
    if (active_layer_pool == nullptr) return nullptr;
    GPU_TOPO_TIMED_LOCK(lock);
    PERF_SCOPE(t_find_active_expert);

    for (auto& slot : active_layer_slots) {
        if (slot.layer_idx == layer && slot.expert_id == expert_id) {
            // 2026-08-26: poll-before-trust, mirrors GpuTopology::
            // finish_expert_hit's own already-proven pattern for the main
            // VRAM cache exactly -- upload_ready only ever starts false for
            // an async-issued slot (upload_expert_active_async), and this
            // is what makes it safe for a caller to skip the old
            // unconditional sync_active_layer_batch wait: if the DMA
            // genuinely hasn't landed yet, treat this as a MISS (matching
            // finish_expert_hit's own choice) rather than blocking here --
            // the in-flight copy finishes on its own regardless, and a
            // fresh fetch elsewhere is safe (temp_pin_count still protects
            // this slot from being reused for something else meanwhile).
            if (!slot.upload_ready) {
                // 2026-08-28: MUST check this before trusting cudaEventQuery
                // at all -- see ExpertLocation::copy_dispatched's own
                // comment. A worker-deferred slot that hasn't actually been
                // dispatched yet has upload_event either null or (if reused
                // from a prior promotion) recording something OLD -- either
                // way cudaEventQuery would be meaningless or wrong here, so
                // this must short-circuit to "miss" without ever calling it.
                if (!slot.copy_dispatched) {
                    perf().n_active_layer_poll_not_ready.fetch_add(1, std::memory_order_relaxed);
                    return nullptr;
                }
                if (slot.gpu_id >= 0 && (size_t)slot.gpu_id < devices.size()) cudaSetDevice(slot.gpu_id);
                if (slot.upload_event && cudaEventQuery(slot.upload_event) == cudaSuccess) {
                    record_active_layer_upload_event_timing(slot, false);
                    slot.upload_ready = true;
                    // Release the async-issue's own hold now that its DMA is
                    // confirmed complete -- exactly what sync_active_layer_
                    // batch would have done for this slot, just triggered by
                    // the first real lookup instead of an unconditional
                    // batch wait. Guarded by the !upload_ready check above,
                    // so this can never double-release (a LATER hold_for_
                    // compute() increment, added only after this function
                    // already returns a valid slot, is a completely
                    // independent +1 this branch never touches).
                    if (slot.temp_pin_count > 0) slot.temp_pin_count--;
                } else {
                    perf().n_active_layer_poll_not_ready.fetch_add(1, std::memory_order_relaxed);
                    if (slot.pilot_speculative_fill)
                        perf().n_active_layer_poll_not_ready_speculative.fetch_add(1, std::memory_order_relaxed);
                    else if (slot.was_async_prewarmed)
                        perf().n_active_layer_poll_not_ready_decode_prewarm.fetch_add(1, std::memory_order_relaxed);
                    return nullptr;  // still in flight -- miss, don't block
                }
            }
            slot.last_used_clock = ++cache_clock;
            perf().n_active_layer_hits.fetch_add(1, std::memory_order_relaxed);
            // A real dispatch found and used this slot's content -- marks it
            // as NOT wasted regardless of what happens below (hit_count
            // increment, promotion, or neither). See ExpertLocation::
            // prewarm_touched's own comment for why this must NOT be
            // conflated with hit_count (the prewarm_skip_next_promote
            // branch just below intentionally never increments hit_count
            // on a prewarmed slot's first, INTENDED, successful touch).
            slot.prewarm_touched = true;

            // SUPERSEDED (2026-08-12), then REVIVED (2026-08-14): this used
            // to special-case a pre-warmed slot's first real touch (skip the
            // hit_count increment, matching the reactive path's "first touch
            // is a miss" semantics) to prevent premature promotion, then was
            // dropped in favor of the coarser !in_prefill gate below once
            // prefill's own promotion churn was fully explained by that gate
            // alone -- at the time, upload_expert_active_async (the only
            // setter of prewarm_skip_next_promote) was prefill-only, so the
            // flag was correctly assumed dead outside prefill.
            //
            // That assumption broke once EngineConfig::decode_async_prewarm
            // was added: it calls the SAME upload_expert_active_async to
            // pre-warm the exact expert dispatch_expert_k is about to
            // request THIS token, on the decode path where in_prefill is
            // always false. That guarantees the very next call into this
            // branch is a qualifying "hit" on every single pre-warm, not
            // just on a genuine repeat visit -- crossing threshold=1 and
            // forcing promote_active_to_main unconditionally, defeating the
            // active layer's whole "prove it's genuinely reused, not just
            // today's routed pick" purpose. Confirmed as the real cause of a
            // 2026-08-14 real-hardware regression (decode tok/s 0.7->0.52,
            // vram_demote 14.84s / find_active 15.30s of a 90s request, vs.
            // near-zero for both when decode_async_prewarm is off) -- not a
            // memory leak as first suspected, pure promotion/eviction churn.
            //
            // Fix: skip counting this one qualifying touch when the flag is
            // set, exactly the semantics of the original branch this
            // replaces, but now scoped ONLY to prewarm-flagged slots so
            // prefill's own (already in_prefill-gated, thus unaffected
            // either way) behavior doesn't change.
            if (slot.prewarm_skip_next_promote) {
                slot.prewarm_skip_next_promote = false;
                return &slot;
            }
            if (slot.hit_count < UINT32_MAX) slot.hit_count++;
            if (!in_prefill && slot.hit_count >= active_layer_promote_threshold) {
                // promote_active_to_main only READS this slot (a D2D copy
                // out), which is always safe regardless of temp_pin_count --
                // any number of concurrent readers (kernels + this copy) is
                // fine. Only the FOLLOWING identity-clear (making the slot
                // eligible for a completely different expert's H2D upload,
                // a WRITE) is unsafe while a hold is outstanding, so that
                // part alone is gated. If skipped, the slot stays correctly
                // active-layer-resident under its EXISTING identity (still
                // safe to serve this same expert) and a later hit or
                // cleanup call will retry the free once the hold clears.
                // Deliberately NOT forwarding on_promoted into this call --
                // promote_active_to_main would fire it as soon as its OWN
                // copy succeeds, before the temp_pin_count check just below
                // decides whether this slot can actually be cleared and
                // `promoted` actually returned. If that check fails (a hold
                // is outstanding right now), this function falls through to
                // returning `&slot` (still active-layer!) a few lines down --
                // firing the callback here unconditionally would tell the
                // caller "loc is now main-cache" for a `loc` that's actually
                // still the active-layer slot. Fired manually below, only in
                // the branch that actually commits to returning `promoted`.
                // Active-layer slots use the largest supported slab as their
                // capacity, but compact main-cache regions are sized to the
                // layer's real expert payload. Passing slot.size here made
                // evict_lru reject every smaller layer as "slot too small",
                // so confirmed active hits were silently discarded instead
                // of becoming durable cache entries. last_upload_bytes is
                // the exact payload copied into this active slot.
                const size_t payload_bytes =
                    slot.last_upload_bytes > 0
                        ? std::min(slot.last_upload_bytes, slot.size)
                        : slot.size;
                ExpertLocation* promoted = promote_active_to_main(
                    layer, expert_id, slot.device_ptr, payload_bytes, nullptr,
                    slot.gpu_id);
                if (promoted && slot.temp_pin_count == 0) {
                    // Always "used" here by construction (prewarm_touched
                    // was just set true above, unconditionally, before this
                    // branch could even be reached) -- called for complete,
                    // consistent accounting across every reclaim site, not
                    // because this specific path could ever be a waste.
                    record_active_layer_prewarm_reclaim(slot);
                    slot.layer_idx = -1;
                    slot.expert_id = -1;
                    slot.last_used_clock = 0;
                    slot.hit_count = 0;
                    slot.upload_ready = true;
                    if (on_promoted) on_promoted(layer, expert_id);
                    return promoted;
                }
                // Main cache unavailable/copy failed, or this slot is still
                // held -- fall through and keep serving this token's
                // compute from the active slot.
            }
            return &slot;
        }
    }
    return nullptr;
}

ExpertLocation* GpuTopology::find_active_expert_wait(
    int layer, int expert_id,
    const std::function<void(int, int)>& on_promoted) {
    auto forward_promotion = [&](int promoted_layer, int promoted_expert) {
        if (on_promoted) on_promoted(promoted_layer, promoted_expert);
    };

    // Fast path: a completed active-layer hit does not add a synchronization.
    if (ExpertLocation *ready = find_active_expert(layer, expert_id,
                                                    forward_promotion))
        return ready;

    cudaEvent_t event = nullptr;
    int gpu_id = -1;
    {
        GPU_TOPO_TIMED_LOCK(lock);
        for (const auto &slot : active_layer_slots) {
            if (slot.layer_idx != layer || slot.expert_id != expert_id)
                continue;
            // A worker-claimed slot has no valid event until its copy is
            // actually dispatched. Never wait on an old/recycled event.
            if (slot.upload_ready || !slot.copy_dispatched || !slot.upload_event)
                return nullptr;
            event = slot.upload_event;
            gpu_id = slot.gpu_id;
            break;
        }
    }
    if (!event) return nullptr;

    if (gpu_id >= 0) cudaSetDevice(gpu_id);
    if (cudaEventSynchronize(event) != cudaSuccess) return nullptr;

    // Re-enter through the normal lookup so readiness accounting, hit
    // counters, and any threshold promotion remain centralized there.
    return find_active_expert(layer, expert_id, forward_promotion);
}

ExpertLocation* GpuTopology::promote_active_to_main(int layer, int expert_id,
                                                     const uint8_t* device_src, size_t bytes,
                                                     const std::function<void(int, int)>& on_promoted,
                                                     int src_gpu_id,
                                                     bool same_gpu_victim) {
    if (expert_vram_cache_cap == 0 || !device_src || bytes == 0) return nullptr;
    GPU_TOPO_TIMED_LOCK(lock);

    ExpertLocation* existing = find_expert(layer, expert_id);
    if (existing) {
        // Already durably main-cache-resident via some OTHER path (e.g. a
        // cold-batch/decode-miss upload raced ahead of this promotion) --
        // still fire on_promoted: the RAM tier still needs to know a
        // durable VRAM copy exists, regardless of which path put it there.
        if (on_promoted) on_promoted(layer, expert_id);
        return existing;
    }

    int old_layer = -1, old_expert = -1;
    ExpertLocation* slot = evict_lru(bytes, &old_layer, &old_expert,
                                     same_gpu_victim && src_gpu_id >= 0 ? src_gpu_id : -1,
                                     layer);
    if (!slot || !slot->device_ptr) return nullptr;

    // Same VRAM->RAM salvage as upload_expert() -- must run before the
    // device-to-device copy below overwrites this slot's outgoing data.
    if (on_vram_evict && old_layer != -1 && old_expert >= 0) {
        on_vram_evict(old_layer, old_expert, slot->device_ptr, slot->size, slot->gpu_id);
    }

    // Previously completely untimed -- see t_active_layer_promote_copy's own
    // comment in perf_timer.hpp. Also note on_vram_evict above (when the
    // victim slot needed eviction) may ALSO have just run a synchronous D2H
    // cudaMemcpy inside demote_from_vram, itself now timed separately into
    // t_vram_demote_copy -- both can fire in the same call, both under this
    // same held cache_mutex.
    size_t copy_bytes = std::min(bytes, slot->size);
    // 2026-08-28: cross-GPU case -- src_gpu_id defaults to -1 (assume same
    // GPU as the destination, the historical single-active-layer-pool
    // behavior, still correct for every legacy call). With multi-GPU
    // active-layer pools, the promotion source and evict_lru's own victim
    // pick CAN now be on different GPUs -- a plain same-device cudaMemcpy
    // would silently fail or corrupt (cudaMemcpyDeviceToDevice with no
    // explicit peer path is only valid within one device's own context).
    bool cross_gpu = (src_gpu_id >= 0 && src_gpu_id != slot->gpu_id);
    cudaError_t err = cudaSuccess;
    PERF_START(active_layer_promote_copy);
    if (!cross_gpu) {
        cudaSetDevice(slot->gpu_id);
        err = cudaMemcpy(slot->device_ptr, device_src, copy_bytes, cudaMemcpyDeviceToDevice);
    } else {
        // Try genuine peer-to-peer first (one call, no host round-trip,
        // works for any size) -- same "check support, don't assume"
        // discipline as the existing cross_device_transfer helper.
        int can_peer = 0;
        cudaDeviceCanAccessPeer(&can_peer, slot->gpu_id, src_gpu_id);
        if (can_peer) {
            err = cudaMemcpyPeer(slot->device_ptr, slot->gpu_id, device_src, src_gpu_id, copy_bytes);
        } else {
            // No P2P between these two GPUs (common for mixed-generation
            // consumer cards with no NVLink, e.g. Ampere+Blackwell) --
            // fall back to a host-mediated D2H-then-H2D bounce. Fresh,
            // exactly-sized cudaMallocHost per call rather than reusing
            // the existing h_xfer_pinned pool (only 256KB, far too small
            // for an expert-slab-sized transfer -- would silently
            // truncate via std::min if reused naively). This path is
            // expected to be RARE (most promotions stay same-GPU, since
            // most active-layer content lives on whichever GPU has the
            // larger routing weight) -- a per-call allocation cost here
            // is an acceptable, deliberate trade for correctness/
            // simplicity on a cold path, not a hot one.
            uint8_t* host_bounce = nullptr;
            if (!MemoryGuard::try_reserve_host_bytes(copy_bytes)) {
                LOG_STREAM_WARN() << "[MEM-GUARD] Cross-GPU active-layer promotion: "
                                  << "host bounce refused (" << copy_bytes / 1e6
                                  << " MB); promotion skipped\n";
                PERF_END(active_layer_promote_copy, t_active_layer_promote_copy);
                return nullptr;
            }
            cudaError_t alloc_err = cudaMallocHost((void**)&host_bounce, copy_bytes);
            if (alloc_err != cudaSuccess || !host_bounce) {
                MemoryGuard::release_host_bytes(copy_bytes);
                LOG_STREAM_ERROR() << "[GPU] Cross-GPU active-layer promotion: host bounce buffer "
                                   << "allocation failed (" << copy_bytes / 1e6 << " MB): "
                                   << cudaGetErrorString(alloc_err) << "\n";
                PERF_END(active_layer_promote_copy, t_active_layer_promote_copy);
                return nullptr;
            }
            cudaSetDevice(src_gpu_id);
            err = cudaMemcpy(host_bounce, device_src, copy_bytes, cudaMemcpyDeviceToHost);
            if (err == cudaSuccess) {
                cudaSetDevice(slot->gpu_id);
                err = cudaMemcpy(slot->device_ptr, host_bounce, copy_bytes, cudaMemcpyHostToDevice);
            }
            cudaFreeHost(host_bounce);
            MemoryGuard::release_host_bytes(copy_bytes);
        }
    }
    PERF_END(active_layer_promote_copy, t_active_layer_promote_copy);
    if (err != cudaSuccess) {
        LOG_STREAM_ERROR() << "[GPU] Active-layer promotion D2D copy failed: "
                           << cudaGetErrorString(err) << "\n";
        return nullptr;
    }

    perf().n_active_layer_promoted.fetch_add(1, std::memory_order_relaxed);

    slot->layer_idx = layer;
    slot->expert_id = expert_id;
    slot->last_used_clock = ++cache_clock;
    vram_swap_stats[swap_key(layer, expert_id)].swap_in++;
    expert_index[swap_key(layer, expert_id)] = (size_t)(slot - expert_vram_cache.data());
    if (on_promoted) on_promoted(layer, expert_id);
    return slot;
}
