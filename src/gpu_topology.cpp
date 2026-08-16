// gpu_topology.cpp - Active layer area implementation for three-tier VRAM caching
#include "gpu_topology.hpp"
#include "perf_timer.hpp"
#include "logger.hpp"
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
        slot.upload_event = nullptr;
        slot.prewarm_skip_next_promote = false;
        active_layer_slots.push_back(slot);
    }

    LOG_STREAM_INFO() << "[GPU] Active layer area allocated: "
                      << max_experts_per_layer << " experts ("
                      << total_bytes / 1e9 << " GB) on GPU " << target_gpu << "\n";
    return true;
}

ExpertLocation* GpuTopology::upload_expert_active(int layer, int expert_id,
                                                 const uint8_t* host_data, size_t bytes) {
    if (active_layer_pool == nullptr || active_layer_capacity == 0) {
        return nullptr;  // Active layer area not allocated
    }

    if (!host_data || bytes == 0) {
        LOG_STREAM_ERROR() << "[GPU] Invalid upload_expert_active parameters\n";
        return nullptr;
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
        if (slot.temp_pin_count > 0) continue;
        if (slot.layer_idx == -1) {
            victim = &slot;  // Prefer empty slot
            break;
        }
        if (!victim || slot.eviction_score() < victim->eviction_score()) {
            victim = &slot;
        }
    }

    if (!victim) {
        LOG_STREAM_WARN() << "[GPU] Active layer area full - cannot upload expert\n";
        return nullptr;
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
    size_t copy_bytes = std::min(bytes, victim->size);
    cudaSetDevice(victim->gpu_id);
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
        // cudaStreamQuery can leave a spurious error in the per-thread
        // last-error slot on some driver versions when it returns
        // cudaErrorNotReady -- clear it so it's never misattributed to the
        // real cudaMemcpy a few lines below (same precaution
        // demote_from_vram's own copy of this check already takes).
        cudaGetLastError();
    }
    PERF_START(active_layer_upload);
    cudaError_t err = cudaMemcpy(victim->device_ptr, host_data, copy_bytes,
                                cudaMemcpyHostToDevice);
    PERF_END(active_layer_upload, t_active_layer_upload);
    if (err != cudaSuccess) {
        LOG_STREAM_ERROR() << "[GPU] Failed to upload expert to active layer: "
                           << cudaGetErrorString(err) << "\n";
        return nullptr;
    }

    // Waste accounting for whatever this victim slot held BEFORE this call
    // overwrites it -- see record_active_layer_prewarm_reclaim's own
    // comment. This function itself never sets was_async_prewarmed (a
    // reactive/confirmed-need upload, not a speculative one), so no
    // corresponding set call after -- only the reclaim of the OUTGOING
    // content matters here.
    record_active_layer_prewarm_reclaim(*victim);

    victim->layer_idx = layer;
    victim->expert_id = expert_id;
    victim->last_used_clock = ++cache_clock;
    victim->hit_count = 0;  // fresh identity in this slot -- find_active_expert accumulates from here
    victim->upload_ready = true;
    // Explicit, not just "already false from a fresh/reused slot" --
    // this IS the reactive cold-miss path (a real find_active_expert MISS
    // just happened, or this is prefill's own legacy synchronous
    // pre-warm branch), never proactive -- its own first future hit
    // should count toward promotion normally, unlike upload_expert_active_
    // async's batch pre-warm. See prewarm_skip_next_promote's own comment.
    victim->prewarm_skip_next_promote = false;

    LOG_DEBUG("[GPU] Active layer upload: layer=%d expert=%d (%.2f MB)\n",
              layer, expert_id, copy_bytes / 1e6);

    // Track three-tier metrics
    perf().n_active_layer_uploads.fetch_add(1, std::memory_order_relaxed);

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
                                                          const uint8_t* host_data, size_t bytes) {
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
        if (slot.temp_pin_count > 0) continue;  // in flight (this batch or a prior one) -- never re-target
        if (slot.layer_idx == -1) {
            victim = &slot;
            break;
        }
        if (!victim || slot.eviction_score() < victim->eviction_score()) {
            victim = &slot;
        }
    }

    if (!victim) {
        LOG_STREAM_WARN() << "[GPU] Active layer area full - cannot async-upload expert\n";
        return nullptr;
    }

    size_t copy_bytes = std::min(bytes, victim->size);
    cudaSetDevice(victim->gpu_id);
    auto& dev = devices[victim->gpu_id];

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
    PERF_START(active_layer_upload_async);
    cudaError_t err = cudaMemcpyAsync(victim->device_ptr, host_data, copy_bytes,
                                       cudaMemcpyHostToDevice, dev.copy_stream);
    PERF_END(active_layer_upload_async, t_active_layer_upload_async);
    if (err != cudaSuccess) {
        LOG_STREAM_ERROR() << "[GPU] Failed to async-upload expert to active layer: "
                           << cudaGetErrorString(err) << "\n";
        return nullptr;
    }

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
    victim->was_async_prewarmed = true;
    // The whole point of this flag -- see its own comment in
    // gpu_topology.hpp for the real-hardware regression this closes.
    victim->prewarm_skip_next_promote = true;

    perf().n_active_layer_uploads_async_issued.fetch_add(1, std::memory_order_relaxed);
    return victim;
}

void GpuTopology::sync_active_layer_batch(const std::vector<ExpertLocation*>& victims) {
    if (victims.empty()) return;
    // The active layer area lives on exactly one GPU by construction
    // (allocate_active_layer_area allocates a single pool on one target
    // GPU) -- every victim in a batch necessarily shares that same
    // gpu_id, so one stream sync covers the whole batch.
    int gpu_id = victims[0]->gpu_id;
    cudaSetDevice(gpu_id);
    PERF_START(active_layer_batch_sync);
    cudaStreamSynchronize(devices[gpu_id].copy_stream);
    PERF_END(active_layer_batch_sync, t_active_layer_batch_sync);

    GPU_TOPO_TIMED_LOCK(lock);
    for (auto* v : victims) {
        v->upload_ready = true;
        if (v->temp_pin_count > 0) v->temp_pin_count--;
    }
    perf().n_active_layer_uploads_async_synced.fetch_add((uint64_t)victims.size(), std::memory_order_relaxed);
}

void GpuTopology::cleanup_active_layer_area(int current_layer_idx,
                                             const std::function<void(int, int)>& on_promoted) {
    if (active_layer_pool == nullptr) {
        return;  // Not allocated, nothing to cleanup
    }

    GPU_TOPO_TIMED_LOCK(lock);

    int slots_freed = 0;
    for (auto& slot : active_layer_slots) {
        if (slot.layer_idx != current_layer_idx && slot.layer_idx != -1) {
            // Never touch a slot a GPU kernel may still be reading (same
            // discipline as evict_lru/upload_expert_active's victim picker
            // above) -- skip it this round; a LATER cleanup call (once
            // expert_ffn_gpu_finish's release_hold drops this to 0) will
            // catch it. Leaves this slot's stale data resident one round
            // longer than ideal, never unsafe.
            if (slot.temp_pin_count > 0) continue;
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
            if (!in_prefill && slot.hit_count >= active_layer_promote_threshold) {
                promote_active_to_main(slot.layer_idx, slot.expert_id,
                                       slot.device_ptr, slot.size, on_promoted);
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
                ExpertLocation* promoted = promote_active_to_main(layer, expert_id,
                                                                   slot.device_ptr, slot.size);
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

ExpertLocation* GpuTopology::promote_active_to_main(int layer, int expert_id,
                                                     const uint8_t* device_src, size_t bytes,
                                                     const std::function<void(int, int)>& on_promoted) {
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
    ExpertLocation* slot = evict_lru(bytes, &old_layer, &old_expert);
    if (!slot || !slot->device_ptr) return nullptr;

    // Same VRAM->RAM salvage as upload_expert() -- must run before the
    // device-to-device copy below overwrites this slot's outgoing data.
    if (on_vram_evict && old_layer != -1) {
        on_vram_evict(old_layer, old_expert, slot->device_ptr, slot->size, slot->gpu_id);
    }

    // Previously completely untimed -- see t_active_layer_promote_copy's own
    // comment in perf_timer.hpp. Also note on_vram_evict above (when the
    // victim slot needed eviction) may ALSO have just run a synchronous D2H
    // cudaMemcpy inside demote_from_vram, itself now timed separately into
    // t_vram_demote_copy -- both can fire in the same call, both under this
    // same held cache_mutex.
    size_t copy_bytes = std::min(bytes, slot->size);
    cudaSetDevice(slot->gpu_id);
    PERF_START(active_layer_promote_copy);
    cudaError_t err = cudaMemcpy(slot->device_ptr, device_src, copy_bytes, cudaMemcpyDeviceToDevice);
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